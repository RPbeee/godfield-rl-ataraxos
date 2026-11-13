"""Paged GPU rollout columns avoid copying multi-GiB buffers when they grow."""
from math import prod

import torch
import triton as tr
import triton.language as tl

from ..rollout import ColumnBuffer


@tr.jit
def _gather(Pages,Indices,Out,N,C:tl.constexpr,P:tl.constexpr,B:tl.constexpr=256):
    at=tl.program_id(0).to(tl.int64)*B+tl.arange(0,B);row=at//C;col=at%C
    source=tl.load(Indices+row,row<N,0)
    pointer=tl.load(Pages+source//P).to(tl.pointer_type(Out.dtype.element_ty))
    value=tl.load(pointer+source%P*C+col,row<N,0)
    tl.store(Out+at,value,row<N)


@tr.jit
def _scatter(Pages,Values,START,N,C:tl.constexpr,P:tl.constexpr,B:tl.constexpr=256):
    at=tl.program_id(0).to(tl.int64)*B+tl.arange(0,B);row=at//C;col=at%C
    dest=START+row
    pointer=tl.load(Pages+dest//P,row<N,0).to(tl.pointer_type(Values.dtype.element_ty))
    value=tl.load(Values+at,row<N,0)
    tl.store(pointer+dest%P*C+col,value,row<N)


class PagedColumn:
    def __init__(self,value,page_size):
        self.page_size=page_size;self.shape=value.shape[1:];self.dtype=value.dtype;self.device=value.device
        self.elements=prod(self.shape);self.pages=[];self.size=0
        self.table=torch.empty(1024,dtype=torch.int64,device=self.device)

    def append(self,value,start):
        end=start+len(value);needed=tr.cdiv(end,self.page_size)
        if needed>len(self.table):
            table=torch.empty(max(needed,len(self.table)*2),dtype=torch.int64,device=self.device)
            table[:len(self.pages)].copy_(self.table[:len(self.pages)]);self.table=table
        while len(self.pages)<needed:
            page=torch.empty((self.page_size,*self.shape),dtype=self.dtype,device=self.device)
            self.table[len(self.pages)]=page.data_ptr();self.pages.append(page)
        done=0
        while done<len(value):
            page,slot=divmod(start+done,self.page_size)
            count=min(len(value)-done,self.page_size-slot)
            self.pages[page][slot:slot+count].copy_(value[done:done+count])
            done+=count
        self.size=end

    def gather(self,indices):
        n=len(indices)
        out=torch.empty((n,*self.shape),dtype=self.dtype,device=self.device)
        if n:_gather[(tr.cdiv(n*self.elements,256),)](self.table,indices,out,n,self.elements,self.page_size)
        return out

    def compact(self,indices,chunk=16384):
        # Sorted retained indices satisfy indices[j]>=j. Forward chunks cannot
        # overwrite future sources; the temporary protects overlap within one.
        for start in range(0,len(indices),chunk):
            retained=self.gather(indices[start:start+chunk])
            n=len(retained)
            _scatter[(tr.cdiv(n*self.elements,256),)](self.table,retained,start,n,self.elements,self.page_size)
        self.size=len(indices)
        del self.pages[tr.cdiv(self.size,self.page_size):]


class GpuColumnBuffer:
    def __init__(self,device,capacity,paged_keys,page_size):
        self.metadata=ColumnBuffer(device,capacity)
        self.paged_keys=set(paged_keys);self.paged={};self.page_size=page_size

    @property
    def size(self):return self.metadata.size

    @size.setter
    def size(self,value):self.metadata.size=value

    @property
    def data(self):return {**self.metadata.data,**self.paged}

    def append(self,columns):
        start=self.size
        self.metadata.append({k:v for k,v in columns.items() if k not in self.paged_keys})
        for key,value in columns.items():
            if key in self.paged_keys:
                if key not in self.paged:self.paged[key]=PagedColumn(value,self.page_size)
                self.paged[key].append(value,start)
