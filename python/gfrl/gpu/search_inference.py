"""Search inference with packed CUDA graphs and incremental AR belief decoding.

Production uses bf16 and the GPU self-play encoder/heads. FP32 is retained for
independent CPU-SearchEnv verification; the game engine is precision-independent.
"""
import os
import torch


class IncrementalBelief:
    """The AR decoder's last causal token, caching each layer's keys/values."""
    def __init__(self, belief, memory, padding, counts, shared=None):
        from ..belief import Belief
        self.temperature = getattr(belief, 'temperature', 1.)
        self.model = belief if isinstance(belief, Belief) else belief.model
        if memory.device.type=='cuda' and torch.is_autocast_enabled('cuda'):
            pad=(-memory.shape[1])%8
            if pad:
                memory=torch.nn.functional.pad(memory,(0,0,0,pad))
                padding=torch.nn.functional.pad(padding,(0,pad),value=True)
        self.counts, self.padding = counts, padding
        self.position = 0
        self.self_kv = []
        self.cross_kv = []
        self.cache_rows=None if shared is None else shared[2]
        self.mask = memory.new_zeros((len(memory),1,1,memory.shape[1])).masked_fill(padding[:,None,None,:],-torch.inf)
        for layer in self.model.decoder.layers:
            attention = layer.multihead_attn
            d = attention.embed_dim
            if shared is None:
                kv = torch.nn.functional.linear(memory, attention.in_proj_weight[d:], attention.in_proj_bias[d:])
                k, v = kv.chunk(2,-1)
                self.cross_kv.append((self.heads(k,attention),self.heads(v,attention)))
            else:
                self.cross_kv.append(shared[0][len(self.cross_kv)])
                kv=self.cross_kv[-1][0]
            self.self_kv.append((kv.new_empty((len(counts),attention.num_heads,18,d//attention.num_heads)),
                                 kv.new_empty((len(counts),attention.num_heads,18,d//attention.num_heads))))
        if shared is not None:self.mask=shared[1]

    @staticmethod
    def heads(x, attention):
        return x.reshape(len(x),x.shape[1],attention.num_heads,attention.head_dim).transpose(1,2)

    @staticmethod
    def output(x, attention):
        return torch.nn.functional.linear(x.transpose(1,2).reshape(len(x),1,attention.embed_dim),
                                          attention.out_proj.weight,attention.out_proj.bias)

    def step(self, previous, rows=None, position=None):
        from torch.nn import functional as F
        m=self.model;t=self.position if position is None else position
        counts=self.counts if rows is None else self.counts[rows]
        embedding=m.decoder_positions.weight[t] if position is None else F.embedding(position.reshape(1),m.decoder_positions.weight)
        x=(m.decoder_cards(previous.long())+embedding+m.decoder_counts(counts))[:,None]
        prefix_mask=None if position is None else x.new_zeros((1,1,1,self.self_kv[0][0].shape[2])).masked_fill(torch.arange(self.self_kv[0][0].shape[2],device=x.device)>position,-torch.inf)
        for i,layer in enumerate(m.decoder.layers):
            attention=layer.self_attn
            q,k,v=F.linear(layer.norm1(x),attention.in_proj_weight,attention.in_proj_bias).chunk(3,-1)
            q,k,v=(self.heads(z,attention) for z in (q,k,v))
            keys,values=self.self_kv[i]
            if position is not None:
                keys.index_copy_(2,position.reshape(1),k);values.index_copy_(2,position.reshape(1),v)
                prefix_k,prefix_v=keys,values
            elif rows is None:
                keys[:,:,t:t+1]=k;values[:,:,t:t+1]=v
                prefix_k,prefix_v=keys[:,:,:t+1],values[:,:,:t+1]
            else:
                keys[rows,:,t]=k[:,:,0];values[rows,:,t]=v[:,:,0]
                prefix_k,prefix_v=keys[rows,:,:t+1],values[rows,:,:t+1]
            if position is not None and q.device.type=='cuda' and q.dtype==torch.bfloat16 and os.environ.get('GFRL_GPU_BELIEF_SELF','1')=='1':
                from .belief_attention import cross_attention
                y=cross_attention(q,prefix_k,prefix_v,prefix_mask.expand(len(q),-1,-1,-1),torch.arange(len(q),device=q.device))
            else:y=F.scaled_dot_product_attention(q,prefix_k,prefix_v,attn_mask=prefix_mask)
            x=x+self.output(y,attention)
            attention=layer.multihead_attn;d=attention.embed_dim
            q=F.linear(layer.norm2(x),attention.in_proj_weight[:d],attention.in_proj_bias[:d])
            keys,values=self.cross_kv[i]
            query=self.heads(q,attention)
            if query.device.type=='cuda' and query.dtype==torch.bfloat16 and (self.cache_rows is not None or os.environ.get('GFRL_GPU_BELIEF_CROSS','0')=='1'):
                from .belief_attention import cross_attention
                if self.cache_rows is None:selected=torch.arange(len(query),device=query.device) if rows is None else rows
                else:selected=self.cache_rows if rows is None else self.cache_rows[rows]
                y=cross_attention(query,keys,values,self.mask,selected)
            else:
                if rows is None:y=F.scaled_dot_product_attention(query,keys,values,attn_mask=self.mask)
                else:y=F.scaled_dot_product_attention(query,keys[rows],values[rows],attn_mask=self.mask[rows])
            x=x+self.output(y,attention)
            x=x+layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        x=m.decoder.norm(x) if m.decoder.norm is not None else x
        logits=m.card_head(x[:,0]).float()
        codes=torch.arange(1,297,device=x.device)
        lp=F.log_softmax(logits.masked_fill(codes<previous[:,None],-torch.inf),-1)
        if position is None:self.position+=1
        return (lp/self.temperature).log_softmax(-1) if self.temperature!=1 else lp


class PackedSearchEncoder:
    """Device-counted packed encoder with bounded shape-bucket CUDA graphs.

    BF16 uses the training kernels; optional FP32 uses TF32x3 accumulation.
    Policy/value returns 21 tokens; belief memory retains every token.
    """
    def __init__(self,model,dtype=torch.float32,full=False):
        self.model=model
        self.dtype=dtype
        self.full=full
        self.graphs={}
        self.pool=torch.cuda.graph_pool_handle()
        self.weights=None

    @torch.inference_mode()
    def encode(self,obs):
        from types import SimpleNamespace
        from ..model import OBS_FIELDS
        from .packed import PackedEncoder
        n,t=obs['obs_type'].shape
        outputs=[]
        chunk=int(os.environ.get('GFRL_GPU_SEARCH_ENCODER_CHUNK','256'))
        if chunk not in (128,256,512,1024):raise ValueError('encoder chunk must be 128/256/512/1024')
        for lo in range(0,n,chunk):
            count=min(chunk,n-lo)
            capacity=next(x for x in (32,64,128,256,512,1024) if x>=count)
            key=(capacity,t)
            if key not in self.graphs:
                buffers={k:torch.zeros((capacity,*obs[k].shape[1:]),device=obs[k].device,dtype=obs[k].dtype) for k in OBS_FIELDS}
                env=SimpleNamespace(**buffers,device=obs['obs_type'].device)
                encoder=PackedEncoder(self.model,env,weights=self.weights,dtype=self.dtype)
                encoder.full_output=self.full
                encoder.compact_final=not self.full
                self.weights=self.weights or encoder
                # One valid token makes the warmup exercise each matrix kernel.
                env.obs_valid[:,0]=True;env.obs_type[:,0]=1
                stream=torch.cuda.Stream(device=env.device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):encoder()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=self.pool,stream=stream):output=encoder()
                torch.cuda.current_stream().wait_stream(stream)
                self.graphs[key]=(graph,buffers,output,encoder)
            graph,buffers,output,_=self.graphs[key]
            for k in OBS_FIELDS:buffers[k][:count].copy_(obs[k][lo:lo+count])
            buffers['obs_valid'][count:]=False
            graph.replay()
            outputs.append(output[:count].clone())
        return outputs[0] if len(outputs)==1 else torch.cat(outputs)


class Bf16HeadGraphs:
    """Exactly the SparseHeads kernels used by GPU self-play collection."""
    def __init__(self,model):
        self.model=model
        self.graphs={}
        self.pool=torch.cuda.graph_pool_handle()
        self.value_weight=model.value_head.weight.detach().bfloat16().contiguous()
        self.value_bias=model.value_head.bias.detach().bfloat16().contiguous()

    def values(self,encoded):
        return torch.nn.functional.linear(encoded[:,0].bfloat16(),self.value_weight,self.value_bias).float()

    @torch.inference_mode()
    def __call__(self,encoded,prefix):
        from types import SimpleNamespace
        from .heads import SparseHeads
        n=len(encoded);cap=next(x for x in (32,64,128,256,512,1024,2048,4096,8192,16384,32768) if x>=n)
        if cap not in self.graphs:
            features=torch.zeros((cap,21,self.model.config.d),device=encoded.device)
            buffers={k:torch.zeros((cap,*v.shape[1:]),device=v.device,dtype=v.dtype) for k,v in prefix.items()}
            active=torch.zeros(cap,device=encoded.device,dtype=torch.bool);active[0]=True
            env=SimpleNamespace(**buffers,num_envs=cap,device=encoded.device)
            heads=SparseHeads(self.model,env)
            stream=torch.cuda.Stream(device=encoded.device);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):heads(features,active)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool,stream=stream):lp=heads(features,active)
            torch.cuda.current_stream().wait_stream(stream)
            self.graphs[cap]=graph,features,buffers,active,lp,heads
        graph,features,buffers,active,lp,_=self.graphs[cap]
        features[:n].copy_(encoded)
        for k,v in prefix.items():buffers[k][:n].copy_(v)
        active[:n]=True;active[n:]=False
        graph.replay()
        return lp[:n].clone()


class Bf16BeliefGraph:
    """One captured AR step with a device position and bounded shared root K/V.

    Full 18-token self-attention uses a dynamic prefix mask; future slots are
    never read semantically. FP32 verification keeps the original prefix path.
    """
    def __init__(self,belief,world_capacity,root_capacity,tokens):
        self.model=getattr(belief,'model',belief)
        m=self.model;device=m.decoder_positions.weight.device
        self.capacity=world_capacity;self.roots=root_capacity;self.tokens=tokens
        self.counts=torch.zeros(world_capacity,device=device,dtype=torch.long)
        self.owners=torch.zeros_like(self.counts);self.previous=torch.zeros_like(self.counts)
        self.index=torch.zeros((),device=device,dtype=torch.long)
        self.padding=torch.ones((root_capacity,tokens),device=device,dtype=torch.bool)
        self.mask=torch.full((root_capacity,1,1,tokens),-torch.inf,device=device)
        self.mask[:,:,:,0]=0
        self.kvs=[]
        for layer in m.decoder.layers:
            a=layer.multihead_attn
            shape=(root_capacity,a.num_heads,tokens,a.head_dim)
            self.kvs.append(tuple(torch.zeros(shape,device=device,dtype=torch.bfloat16) for _ in range(2)))
        memory=torch.empty((root_capacity,tokens,m.config.d),device=device)
        self.decoder=IncrementalBelief(belief,memory,self.padding,self.counts,shared=(self.kvs,self.mask,self.owners))
        # A 32-wide aligned mask enables efficient SDPA instead of its
        # unfused fallback; positions 18..31 are permanently masked.
        self.decoder.self_kv=[tuple(torch.zeros((world_capacity,k.shape[1],32,k.shape[3]),device=device,dtype=k.dtype) for _ in range(2))
                              for k,_ in self.decoder.self_kv]
        stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream())
        # Capture casts as graph operations, avoiding temporary autocast-cache
        # pointers whose lifetime ends when the sampling context exits.
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            with torch.cuda.stream(stream):
                for _ in range(2):self.decoder.step(self.previous,position=self.index)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.lp=self.decoder.step(self.previous,position=self.index)
        torch.cuda.current_stream().wait_stream(stream)

    def begin(self,counts,shared):
        kvs,mask,owners=shared
        self.n=len(counts);self.position=0
        self.counts.zero_();self.counts[:self.n].copy_(counts)
        self.owners.zero_();self.owners[:self.n].copy_(owners)
        self.mask.fill_(-torch.inf)
        nr,_,_,nt=mask.shape;self.mask[:nr,:,:,:nt].copy_(mask)
        for target,source in zip(self.kvs,kvs):
            for dst,src in zip(target,source):
                dst.zero_();dst[:len(src),:,:src.shape[2]].copy_(src)
        return self

    def step(self,previous):
        self.previous.zero_();self.previous[:self.n].copy_(previous)
        self.index.fill_(self.position);self.graph.replay();self.position+=1
        return self.lp[:self.n]
