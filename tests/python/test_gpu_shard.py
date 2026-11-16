"""Parallel compression is byte-lossless and keeps the existing loader."""
import gzip
from io import BytesIO

import numpy as np
import pytest
import torch

from gfrl.distill import load_shard
from gfrl.model import OBS_FIELDS
from gfrl.actor import PREFIX_FIELDS
from gfrl.gpu.shard import ParallelGzipWriter,save_shard


@pytest.mark.parametrize('workers',[1,3])
def test_parallel_gzip_boundaries_and_repeated_flush(workers):
    rng=np.random.default_rng(71)
    parts=[rng.bytes(n) for n in (0,3,64,13,192,5,1025)]
    output=BytesIO();writer=ParallelGzipWriter(output,workers,chunk_bytes=64)
    for i,part in enumerate(parts):
        assert writer.write(part)==len(part)
        if i%2:writer.flush()
    assert writer.tell()==sum(map(len,parts))
    writer.close()
    assert gzip.decompress(output.getvalue())==b''.join(parts)


def test_parallel_shard_unchanged_torch_loader(tmp_path):
    data=dict(format_version=1,metadata={'seed':71},obs={k:torch.randn(128,119,81) if k=='obs_num' else torch.arange(12) for k in OBS_FIELDS},prefix={k:torch.arange(12) for k in PREFIX_FIELDS},
              integer=torch.arange(10000,dtype=torch.int16),probabilities=torch.rand(20,dtype=torch.float64))
    path=tmp_path/'data.pt.gz';save_shard(path,data,workers=3)
    loaded=load_shard(path)
    assert loaded['format_version']==1 and loaded['metadata']==data['metadata']
    torch.testing.assert_close(loaded['obs']['obs_num'],data['obs']['obs_num'],atol=0,rtol=0)
    for k in ('integer','probabilities'):assert torch.equal(loaded[k],data[k])


def test_in_memory_generation_summary_matches_reloaded_shards(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    scripts=Path(__file__).resolve().parents[2]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('summary_generator',scripts/'make_distill_data.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path.pop(0)
    metadata=dict(games=2,seat0_score=1,search={'counts':{'searched':2},'seconds':{'total':1.}},
                  world_load_counts={'base:ok':32},seat_counts={'0':1,'1':1},
                  provenance={'search_config':{'worlds':16}})
    data=dict(format_version=1,metadata=metadata,obs={k:torch.arange(2) for k in OBS_FIELDS},
              prefix={k:torch.arange(2) for k in PREFIX_FIELDS},game=torch.tensor([0,1]),
              phase=torch.tensor([0,1]),probabilities=torch.tensor([.25,.75,1.],dtype=torch.float64),
              log_pi=torch.tensor([.8,.2,1.],dtype=torch.float64).log(),candidate_offsets=torch.tensor([0,2,3]))
    paths=[tmp_path/f'{i}.pt.gz' for i in range(2)]
    for path in paths:save_shard(path,data,workers=2)
    parts=[module.shard_summary_part(data) for _ in paths]
    cached=module.summarize_shards(paths,20.,parts)
    assert cached==module.summarize_shards(paths,20.)
    assert (cached['games'],cached['examples'],cached['response_examples'])==(4,4,2)
    assert cached['policy_disagreement']==.5 and cached['mean_candidates']==1.5
    assert cached['searched_positions_per_second']==.2
