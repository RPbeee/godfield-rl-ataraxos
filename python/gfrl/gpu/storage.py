"""Schema-aware numeric rollout storage; lossless relative to existing fp16 storage.

Current tokens use 89 numeric slots in total. Each history token needs at most
43 (a decision with all deltas), instead of reserving F slots for every token.
Only the learner boundary expands completed decisions back to dense [N,T,F].
"""
import torch
import triton as tr
import triton.language as tl
from .. import _core


@tr.jit
def _pack(Num,Types,Positions,Slots,Fields,Out,N,T:tl.constexpr,F:tl.constexpr,W:tl.constexpr,B:tl.constexpr=256):
    at=tl.program_id(0).to(tl.int64)*B+tl.arange(0,B);row=at//W;col=at%W
    token=tl.load(Positions+col);slot=tl.load(Slots+col)
    typ=tl.load(Types+row*T+token,row<N,0).to(tl.int32)
    feature=tl.load(Fields+typ*43+slot)
    value=tl.load(Num+(row*T+token)*F+feature,(row<N)&(feature>=0),0)
    tl.store(Out+at,value,row<N)


@tr.jit
def _expand(Packed,Types,Bases,Inverse,Out,N,T:tl.constexpr,F:tl.constexpr,W:tl.constexpr,B:tl.constexpr=256):
    at=tl.program_id(0).to(tl.int64)*B+tl.arange(0,B);row=at//(T*F);token=at//F%T;feature=at%F
    typ=tl.load(Types+row*T+token,row<N,0).to(tl.int32)
    slot=tl.load(Inverse+typ*F+feature)
    base=tl.load(Bases+token)
    value=tl.load(Packed+row*W+base+slot,(row<N)&(slot>=0),0)
    tl.store(Out+at,value,row<N)


class NumericStorage:
    def __init__(self,device,history_len):
        schema=_core.obs_schema(history_len)
        nums,types=schema['num'],schema['types']
        names=sorted(nums,key=nums.get)
        mapping={k:[] for k in types}
        mapping.update(cls=['gf','tiebreak_gf','tiebreak_hidden','decisions','apocalypse','first_seat','hand_count'],
            player=names[:17],hand=['disguised','used'],pending=names[26:36]+['card_overflow'],offer=['price'],
            decision=['faces_hidden','has_allocation','allocation_hp','allocation_mp','allocation_cp','age','card_overflow']+names[45:],
            removal=['age','card_overflow'],sale=['age','price'],purchase=['age','price'],
            danger=['age','power_log1p','mortar','card_overflow'],devil=['age','card_overflow'],grant=['age','card_overflow'])
        for kind in ('reveal','fate','phenomenon','guardian_change','redraw'):mapping[kind]=['age']
        if schema['F']!=81 or schema['positions']['history_start']!=55 or len(types)!=19 or set().union(*map(set,mapping.values()))!=set(nums):
            raise ValueError('unsupported numeric rollout schema')
        fields=[[-1]*43 for _ in types];inverse=[[-1]*schema['F'] for _ in types]
        for kind,columns in mapping.items():
            for slot,name in enumerate(columns):
                feature=nums[name];fields[types[kind]][slot]=feature;inverse[types[kind]][feature]=slot
        counts=[7,17,17]+[2]*18+[11,1]+[0]*32+[43]*history_len
        bases=[];positions=[];slots=[];offset=0
        for token,count in enumerate(counts):
            bases.append(offset);positions += [token]*count;slots += list(range(count));offset+=count
        self.width=offset;self.tokens=len(counts);self.features=schema['F']
        self.positions=torch.tensor(positions,dtype=torch.int32,device=device)
        self.slots=torch.tensor(slots,dtype=torch.int32,device=device)
        self.bases=torch.tensor(bases,dtype=torch.int32,device=device)
        self.fields=torch.tensor(fields,dtype=torch.int32,device=device)
        self.inverse=torch.tensor(inverse,dtype=torch.int32,device=device)

    def pack(self,numeric,types):
        n,t,f=numeric.shape
        if (t,f)!=(self.tokens,self.features):raise ValueError('numeric storage shape mismatch')
        out=torch.empty((n,self.width),dtype=torch.float16,device=numeric.device)
        _pack[(tr.cdiv(n*self.width,256),)](numeric,types,self.positions,self.slots,self.fields,out,n,t,f,self.width)
        return out

    def expand(self,packed,types):
        n=len(packed);t,f=self.tokens,self.features
        out=torch.empty((n,t,f),dtype=torch.float16,device=packed.device)
        _expand[(tr.cdiv(n*t*f,256),)](packed,types,self.bases,self.inverse,out,n,t,f,self.width)
        return out
