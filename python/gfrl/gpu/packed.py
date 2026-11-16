"""Device-counted packed inference. No token counts are read by the host.

Persistent Triton GEMMs read their active row count from a device scalar. Flash
attention reads device sequence offsets. Capacity stays fixed for CUDA Graphs;
unused capacity does not execute the encoder's matrix multiplications.
"""
import os
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.nn import functional as F


@tr.jit
def _scatter_index(Valid, Prefix, Map, CAP:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=tl.load(Valid+i,i<CAP,0)
    dest=tl.load(Prefix+i,i<CAP,0)-1
    tl.store(Map+dest,i,(i<CAP)&valid)


@tr.jit
def _head_index(Valid,Prefix,HeadPrefix,Map,N:tl.constexpr,T:tl.constexpr,BLOCK:tl.constexpr=256):
    row=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    orig=(row//21)*T+row%21
    valid=tl.load(Valid+orig,row<N*21,0)
    dest=tl.load(HeadPrefix+row,row<N*21,1)-1
    source=tl.load(Prefix+orig,row<N*21,1)-1
    tl.store(Map+dest,source,(row<N*21)&valid)


@tr.jit
def _linear(X,W,B,Y,Map,Count,K:tl.constexpr,N:tl.constexpr,GATHER:tl.constexpr,
            RELU:tl.constexpr,SM:tl.constexpr,BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32,TRANSPOSE:tl.constexpr=False,GELU:tl.constexpr=False,FP32:tl.constexpr=False):
    active=tl.load(Count)
    tiles_n=tl.cdiv(N,BN)
    tiles_m=tl.cdiv(active,BM)
    for tile in range(tl.program_id(0),tiles_m*tiles_n,SM):
        m=(tile//tiles_n)*BM+tl.arange(0,BM)
        n=(tile%tiles_n)*BN+tl.arange(0,BN)
        rows=tl.load(Map+m,m<active,0) if GATHER else m
        kk=tl.arange(0,BK)
        accum=tl.full((BM,BN),0,tl.float32)
        for start in range(tl.cdiv(K,BK)):
            k=start*BK+kk
            a=tl.load(X+rows[:,None]*K+k[None,:],(m[:,None]<active)&(k[None,:]<K),0)
            if not FP32:a=a.to(tl.bfloat16)
            if TRANSPOSE:
                w=tl.load(W+k[:,None]*N+n[None,:],(n[None,:]<N)&(k[:,None]<K),0)
            else:
                w=tl.load(W+n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),0)
            if FP32:accum=tl.dot(a,w,accum,input_precision="tf32x3")
            else:accum=tl.dot(a,w,accum)
        bias=tl.load(B+n,n<N,0).to(tl.float32)
        out=accum+bias[None,:]
        if RELU:out=tl.maximum(out,0)
        if GELU:
            out=out.to(tl.bfloat16).to(tl.float32)
            out=0.5*out*(1.+libdevice.erf(out*0.7071067811865476))
        tl.store(Y+m[:,None]*N+n[None,:],out,(m[:,None]<active)&(n[None,:]<N))


@tr.jit
def _embed(ObsType,ObsCards,ObsAux,ObsCat,Map,Count,Numeric,Types,Roles,Cards,Aux,Cats,Out,
           D:tl.constexpr,T:tl.constexpr,C:tl.constexpr,OFFSETS:tl.constexpr,SM:tl.constexpr,R:tl.constexpr=8):
    active=tl.load(Count)
    d=tl.arange(0,tr.next_power_of_2(D))
    for block in range(tl.program_id(0),tl.cdiv(active,R),SM):
        row=block*R+tl.arange(0,R)
        orig=tl.load(Map+row,row<active,0)
        pos=orig%T
        role=tl.full((R,),8,tl.int32)
        role=tl.where(pos==0,0,role);role=tl.where(pos==1,1,role);role=tl.where(pos==2,2,role)
        role=tl.where((pos>=3)&(pos<21),3,role);role=tl.where(pos==21,4,role);role=tl.where(pos==22,5,role)
        role=tl.where((pos>=23)&(pos<39),6,role);role=tl.where((pos>=39)&(pos<55),7,role)
        typ=tl.load(ObsType+orig,row<active,0).to(tl.int32)
        x=tl.load(Types+typ[:,None]*D+d[None,:],d[None,:]<D,0)+tl.load(Roles+role[:,None]*D+d[None,:],d[None,:]<D,0)
        for j in tl.static_range(8):
            card=tl.load(ObsCards+orig*8+j,row<active,0).to(tl.int32)
            x=x+tl.load(Cards+card[:,None]*D+d[None,:],d[None,:]<D,0)
        aux=tl.load(ObsAux+orig,row<active,0).to(tl.int32)
        x=x+tl.load(Aux+aux[:,None]*D+d[None,:],d[None,:]<D,0).to(tl.float32)
        for j in tl.static_range(C):
            cat=tl.load(ObsCat+orig*C+j,row<active,0).to(tl.int32)
            x=x+tl.load(Cats+(OFFSETS[j]+cat[:,None])*D+d[None,:],d[None,:]<D,0)
        numeric=tl.load(Numeric+row[:,None]*D+d[None,:],(row[:,None]<active)&(d[None,:]<D),0).to(tl.float32)
        tl.store(Out+row[:,None]*D+d[None,:],x+numeric,(row[:,None]<active)&(d[None,:]<D))


@tr.jit
def _norm(X,Residual,Weight,Bias,Out,Sum,Count,D:tl.constexpr,EPS:tl.constexpr,
          ADD:tl.constexpr,SM:tl.constexpr,R:tl.constexpr=8,Map=None,GATHER_RESIDUAL:tl.constexpr=False):
    active=tl.load(Count);d=tl.arange(0,tr.next_power_of_2(D))
    w=tl.load(Weight+d,d<D,0);b=tl.load(Bias+d,d<D,0)
    for block in range(tl.program_id(0),tl.cdiv(active,R),SM):
        row=block*R+tl.arange(0,R)
        x=tl.load(X+row[:,None]*D+d[None,:],(row[:,None]<active)&(d[None,:]<D),0).to(tl.float32)
        if ADD:
            source=tl.load(Map+row,row<active,0) if GATHER_RESIDUAL else row
            x=x+tl.load(Residual+source[:,None]*D+d[None,:],(row[:,None]<active)&(d[None,:]<D),0)
            tl.store(Sum+row[:,None]*D+d[None,:],x,(row[:,None]<active)&(d[None,:]<D))
        mean=tl.sum(x,1)/D
        v=tl.where(d[None,:]<D,x-mean[:,None],0.)
        var=tl.sum(v*v,1)/D
        y=v*tl.rsqrt(var[:,None]+EPS)
        tl.store(Out+row[:,None]*D+d[None,:],y*w[None,:]+b[None,:],(row[:,None]<active)&(d[None,:]<D))


@tr.jit
def _residual(X,Y,Out,Count,D:tl.constexpr,SM:tl.constexpr,BLOCK:tl.constexpr=1024):
    size=tl.load(Count)*D
    for block in range(tl.program_id(0),tl.cdiv(size,BLOCK),SM):
        at=block*BLOCK+tl.arange(0,BLOCK)
        x=tl.load(X+at,at<size,0).to(tl.float32);y=tl.load(Y+at,at<size,0)
        tl.store(Out+at,x+y,at<size)


@tr.jit
def _unpack(X,Valid,Prefix,Weight,Bias,Out,Residual,N:tl.constexpr,T:tl.constexpr,D:tl.constexpr,EPS:tl.constexpr,R:tl.constexpr=8,ADD:tl.constexpr=False,HEAD_COMPACT:tl.constexpr=False,FULL:tl.constexpr=False):
    row=tl.program_id(0)*R+tl.arange(0,R);d=tl.arange(0,tr.next_power_of_2(D))
    width=T if FULL else 21
    orig=row if FULL else (row//21)*T+row%21
    valid=tl.load(Valid+orig,row<N*width,0)
    packed=tl.load(Prefix+(row if HEAD_COMPACT else orig),row<N*width,1)-1
    x=tl.load(X+packed[:,None]*D+d[None,:],((row<N*width)&valid)[:,None]&(d[None,:]<D),0).to(tl.float32)
    if ADD:
        x+=tl.load(Residual+packed[:,None]*D+d[None,:],((row<N*width)&valid)[:,None]&(d[None,:]<D),0)
    mean=tl.sum(x,1)/D;v=tl.where(d[None,:]<D,x-mean[:,None],0.);var=tl.sum(v*v,1)/D
    y=v*tl.rsqrt(var[:,None]+EPS)*tl.load(Weight+d,d<D,0)[None,:]+tl.load(Bias+d,d<D,0)[None,:]
    tl.store(Out+row[:,None]*D+d[None,:],tl.where(valid[:,None],y,0),(row[:,None]<N*width)&(d[None,:]<D))

@tr.jit
def _attention(Qkv,Cu,Out,H:tl.constexpr,D:tl.constexpr,HeadCu=None,HEAD_ONLY:tl.constexpr=False,Q:tl.constexpr=32,K:tl.constexpr=32,FP32:tl.constexpr=False):
    sequence=tl.program_id(0)//H;head=tl.program_id(0)%H
    start=tl.load(Cu+sequence);end=tl.load(Cu+sequence+1);length=end-start
    if HEAD_ONLY:
        out_start=tl.load(HeadCu+sequence)
        query_length=tl.load(HeadCu+sequence+1)-out_start
    else:
        out_start=start;query_length=length
    d=tl.arange(0,tr.next_power_of_2(D));r=tl.arange(0,Q);c=tl.arange(0,K)
    for qb in range(tl.cdiv(query_length,Q)):
        rows=qb*Q+r
        q=tl.load(Qkv+(start+rows[:,None])*3*H*D+head*D+d[None,:],(rows[:,None]<length)&(d[None,:]<D),0)
        maximum=tl.full((Q,),-float('inf'),tl.float32)
        denominator=tl.full((Q,),0,tl.float32)
        accum=tl.full((Q,tr.next_power_of_2(D)),0,tl.float32)
        for kb in range(tl.cdiv(length,K)):
            cols=kb*K+c
            k=tl.load(Qkv+(start+cols[None,:])*3*H*D+H*D+head*D+d[:,None],(cols[None,:]<length)&(d[:,None]<D),0)
            score=tl.dot(q,k,input_precision="tf32x3" if FP32 else "ieee")*(D**-0.5)
            score=tl.where(cols[None,:]<length,score,-float('inf'))
            new_max=tl.maximum(maximum,tl.max(score,1))
            prob=tl.exp(score-new_max[:,None]);scale=tl.exp(maximum-new_max)
            denominator=denominator*scale+tl.sum(prob,1)
            v=tl.load(Qkv+(start+cols[:,None])*3*H*D+2*H*D+head*D+d[None,:],(cols[:,None]<length)&(d[None,:]<D),0)
            accum=accum*scale[:,None]+tl.dot(prob.to(tl.float32) if FP32 else prob.to(tl.bfloat16),v,input_precision="tf32x3" if FP32 else "ieee")
            maximum=new_max
        tl.store(Out+(out_start+rows[:,None])*H*D+head*D+d[None,:],accum/denominator[:,None],(rows[:,None]<query_length)&(d[None,:]<D))


class PackedEncoder:
    # Model widths/head widths need not be powers of two. Only Triton lanes
    # are padded; masks and LayerNorm denominators retain the real width.
    def __init__(self, model, env, weights=None, *, dtype=torch.bfloat16):
        self.model,self.env=model,env
        self.dtype=dtype
        self.fp32=dtype==torch.float32
        if dtype not in (torch.float32,torch.bfloat16):raise ValueError("packed dtype must be fp32 or bf16")
        self.sm=torch.cuda.get_device_properties(env.device).multi_processor_count*2
        self.element_sm=self.sm*8
        self.short_attention=(env.obs_type.shape[1]<=256 and 16<=model.config.d//model.config.heads<=128)
        self.d=model.config.d
        self.fused_residual_norm = True
        self.compact_final = True
        self.full_output = False
        self.attention_tiles = dict(Q=64,K=32,num_warps=4)
        self.tma = not self.fp32 and torch.cuda.get_device_capability(env.device)[0]>=9
        if self.tma:
            try:
                from .tma import linear
            except ImportError:
                self.tma=False
        if weights is None:
            self.linear={}
            self.refresh()
        else:
            self.linear=weights.linear
            self.cards,self.aux,self.cats=weights.cards,weights.aux,weights.cats
            self.offsets=weights.offsets

    @torch.no_grad()
    def refresh(self):
        """Refresh cached bf16 weights after an optimizer update, before replay."""
        model=self.model
        modules=[model.num_projection]
        for layer in model.encoder.layers:
            modules += [layer.self_attn.out_proj,layer.linear1,layer.linear2]
        for module in modules:
            key=id(module)
            if key not in self.linear:
                self.linear[key]=(module.weight.detach().to(self.dtype).contiguous(),
                                  module.bias.detach().to(self.dtype).contiguous())
            else:
                w,b=self.linear[key];w.copy_(module.weight);b.copy_(module.bias)
        for layer in model.encoder.layers:
            key=id(layer.self_attn)
            if key not in self.linear:
                self.linear[key]=(layer.self_attn.in_proj_weight.detach().to(self.dtype).contiguous(),
                                  layer.self_attn.in_proj_bias.detach().to(self.dtype).contiguous())
            else:
                w,b=self.linear[key];w.copy_(layer.self_attn.in_proj_weight);b.copy_(layer.self_attn.in_proj_bias)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not self.fp32):
            cards=model.card_embedding.weight+model.card_projection(model.card_features)
            aux=model.aux_projection(cards)
        cats=torch.cat([e.weight.detach() for e in model.category_embeddings])
        if not hasattr(self,'cards'):
            self.cards,self.aux,self.cats=cards.detach().contiguous(),aux.detach().contiguous(),cats.contiguous()
        else:
            self.cards.copy_(cards);self.aux.copy_(aux);self.cats.copy_(cats)
        offsets=[];at=0
        for e in model.category_embeddings:offsets.append(at);at+=e.num_embeddings
        self.offsets=tuple(offsets)

    def _linear(self,x,module,count,mapping,*,gather=False,relu=False,dense_tiles=False):
        w,b=self.linear[id(module)]
        y=torch.empty((self.capacity,w.shape[0]),dtype=self.dtype,device=x.device)
        if self.tma and not gather and 128<=w.shape[1]<=256 and w.shape[0]%8==0:
            from .tma import linear
            return linear(x,w,b,y,count,self.sm//2,relu=relu)
        tiles = dict(BM=128,BN=128,BK=64,num_warps=8) if not gather or dense_tiles else {}
        _linear[(self.sm,)](x,w,b,y,mapping,count,w.shape[1],w.shape[0],gather,relu,self.sm,FP32=self.fp32,**tiles)
        return y

    @torch.inference_mode()
    def __call__(self, active=None):
        env,model=self.env,self.model;n,t=env.obs_type.shape;d=self.d
        obs_valid=env.obs_valid if active is None else env.obs_valid & active[:,None]
        self.capacity=cap=n*t
        valid=obs_valid.flatten()
        prefix=torch.cumsum(valid.to(torch.int32),0,dtype=torch.int32)
        mapping=torch.empty(cap,device=env.device,dtype=torch.int32)
        _scatter_index[(tr.cdiv(cap,256),)](valid,prefix,mapping,cap,256)
        lengths=obs_valid.sum(1,dtype=torch.int32)
        cu=torch.cat((torch.zeros(1,device=env.device,dtype=torch.int32),lengths.cumsum(0,dtype=torch.int32)))
        count=cu[-1:]
        if self.compact_final:
            head_prefix=obs_valid[:,:21].reshape(-1).to(torch.int32).cumsum(0,dtype=torch.int32)
            head_count=head_prefix[-1:]
            head_cu=torch.cat((torch.zeros(1,device=env.device,dtype=torch.int32),head_prefix.reshape(n,21)[:,-1]))
            head_mapping=torch.empty(n*21,device=env.device,dtype=torch.int32)
            _head_index[(tr.cdiv(n*21,256),)](valid,prefix,head_prefix,head_mapping,n,t)
        numeric=self._linear(env.obs_num,model.num_projection,count,mapping,gather=True)
        x=torch.empty((cap,d),dtype=torch.float32,device=env.device)
        _embed[(self.element_sm,)](env.obs_type,env.obs_cards,env.obs_aux_card,env.obs_cat,mapping,count,numeric,
                           model.type_embedding.weight,model.role_embedding.weight,self.cards,self.aux,self.cats,x,
                           d,t,len(self.offsets),self.offsets,self.element_sm,enable_fp_fusion=False)
        h=model.config.heads
        norm=torch.empty((cap,d),dtype=self.dtype,device=env.device)
        layers=model.encoder.layers
        for index,layer in enumerate(layers):
            if not self.fused_residual_norm or index == 0:
                _norm[(self.element_sm,)](x,x,layer.norm1.weight,layer.norm1.bias,norm,x,count,d,layer.norm1.eps,False,self.element_sm)
            qkv=self._linear(norm,layer.self_attn,count,mapping).view(cap,3,h,d//h)
            q,k,v=qkv.unbind(1)
            final_compact=self.compact_final and index+1==len(layers)
            if self.short_attention:
                attended=torch.empty((n*21 if final_compact else cap,h,d//h),device=env.device,dtype=self.dtype)
                _attention[(n*h,)](qkv,cu,attended,h,d//h,HeadCu=head_cu if final_compact else None,HEAD_ONLY=final_compact,FP32=self.fp32,**self.attention_tiles)
            else:
                attended=torch.ops.aten._flash_attention_forward(q,k,v,cu,cu,t,t,0.,False,False)[0]
            if final_compact:
                self.capacity=n*21
                projected=self._linear(attended.reshape(-1,d),layer.self_attn.out_proj,head_count,head_mapping,gather=not self.short_attention,dense_tiles=True)
                summed=torch.empty((n*21,d),dtype=torch.float32,device=env.device)
                _norm[(self.element_sm,)](projected,x,layer.norm2.weight,layer.norm2.bias,norm,summed,head_count,d,layer.norm2.eps,True,self.element_sm,Map=head_mapping,GATHER_RESIDUAL=True)
                count=head_count
            else:
                projected=self._linear(attended.reshape(cap,d),layer.self_attn.out_proj,count,mapping)
                summed=torch.empty_like(x)
                _norm[(self.element_sm,)](projected,x,layer.norm2.weight,layer.norm2.bias,norm,summed,count,d,layer.norm2.eps,True,self.element_sm)
            ff=self._linear(norm,layer.linear1,count,mapping,relu=True)
            out=self._linear(ff,layer.linear2,count,mapping)
            if self.fused_residual_norm:
                if index+1<len(layers):
                    next_norm=layers[index+1].norm1
                    _norm[(self.element_sm,)](out,summed,next_norm.weight,next_norm.bias,norm,x,count,d,next_norm.eps,True,self.element_sm)
            else:
                _residual[(self.element_sm,)](out,summed,x,count,d,self.element_sm)
        output_width=t if self.full_output else 21
        head=torch.empty((n,output_width,d),device=env.device,dtype=torch.float32)
        final=model.encoder.norm
        late_add=self.fused_residual_norm
        _unpack[(tr.cdiv(n*output_width,8),)](out if late_add else x,valid,head_prefix if self.compact_final else prefix,final.weight,final.bias,head,summed,n,t,d,final.eps,ADD=late_add,HEAD_COMPACT=self.compact_final,FULL=self.full_output)
        return head


class ChunkedPackedEncoder:
    """Bound scratch storage while all rows/counts remain on the GPU."""
    def __init__(self, model, env, chunk=None):
        from types import SimpleNamespace
        from ..model import OBS_FIELDS
        chunk=int(os.environ.get('GFRL_GPU_ENCODER_CHUNK','256')) if chunk is None else chunk
        if not isinstance(chunk,int) or chunk<1:
            raise ValueError('encoder chunk must be a positive integer')
        self.chunk=chunk
        self.encoders=[]
        weights=None
        for start in range(0,env.num_envs,chunk):
            view=SimpleNamespace(**{k:getattr(env,k)[start:start+chunk] for k in OBS_FIELDS},device=env.device)
            encoder=PackedEncoder(model,view,weights=weights)
            self.encoders.append(encoder)
            weights=weights or encoder

    def refresh(self):
        self.encoders[0].refresh()

    def __call__(self, active=None):
        encoded=[encoder(None if active is None else active[i*self.chunk:(i+1)*self.chunk])
                 for i,encoder in enumerate(self.encoders)]
        return encoded[0] if len(encoded)==1 else torch.cat(encoded,dim=0)
