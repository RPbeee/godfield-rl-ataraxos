#!/usr/bin/env python3
"""Compare public distillation traces from CPU/CUDA searches with common RNG."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from gfrl.distill import load_shard


def compare(a,b):
    for group in ('obs','prefix'):
        for k in a[group]:
            assert torch.equal(a[group][k],b[group][k]),(group,k)
    for k in ('candidate_offsets','stage_offsets','choice','valid_worlds','game','seat','decision','phase','outcome'):
        assert torch.equal(a[k],b[k]),k
    errors={}
    for k in ('q','probabilities','log_pi','log_rho'):
        torch.testing.assert_close(a[k],b[k],atol=3e-5,rtol=2e-5,equal_nan=True)
        errors[k]=float((a[k]-b[k]).nan_to_num().abs().max())
    return dict(roots=len(a['game']),response_roots=int(a['phase'].sum()),candidates=len(a['q']),max_abs_error=errors,
                discrete_and_public_trace='bit exact')


if __name__=='__main__':
    print(json.dumps(compare(*(load_shard(p) for p in sys.argv[1:3])),indent=2))
