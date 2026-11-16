"""Bounded parallel gzip, readable by the unchanged format-v1 shard loader."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip

import torch


class ParallelGzipWriter:
    """Concatenate independent gzip members, preserving the exact byte stream.

    Workers compress bounded chunks; the calling thread writes them in order.
    gzip.GzipFile (used by load_shard) transparently reads all members.
    """
    def __init__(self, output, workers, chunk_bytes=16*1024*1024):
        if workers<1 or chunk_bytes<1:raise ValueError('positive workers/chunk size required')
        self.output=output
        self.workers=workers
        self.chunk_bytes=chunk_bytes
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.pending=deque()
        self.buffer=bytearray()
        self.position=0

    def _submit(self):
        if self.buffer:
            chunk=bytes(self.buffer);self.buffer.clear()
            self.pending.append(self.pool.submit(gzip.compress,chunk,compresslevel=1,mtime=0))
            if len(self.pending)>=2*self.workers:
                self.output.write(self.pending.popleft().result())

    def write(self, value):
        data=memoryview(value).cast('B')
        self.position+=len(data)
        for lo in range(0,len(data),self.chunk_bytes):
            part=data[lo:lo+self.chunk_bytes]
            room=self.chunk_bytes-len(self.buffer)
            self.buffer.extend(part[:room])
            if len(self.buffer)==self.chunk_bytes:self._submit()
            self.buffer.extend(part[room:])
        return len(data)

    def tell(self):return self.position

    def flush(self):
        self._submit()
        while self.pending:self.output.write(self.pending.popleft().result())
        self.output.flush()

    def close(self):
        try:self.flush()
        finally:self.pool.shutdown(wait=True)


def save_shard(path, data, workers=8):
    """Only CUDA generation opts in; the existing CPU writer is unchanged."""
    path=Path(path);temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as output:
        writer=ParallelGzipWriter(output,workers)
        try:torch.save(data,writer)
        finally:writer.close()
    temporary.replace(path)
