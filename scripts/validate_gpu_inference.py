#!/usr/bin/env python3
"""Measure packed collection inference against original bf16 and FP32 models."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch
from gfrl.actor import PREFIX_FIELDS
from gfrl.gpu.packed import ChunkedPackedEncoder
from gfrl.gpu.heads import SparseHeads
from gfrl.model import Model, ModelConfig, OBS_FIELDS
from gfrl.rollout import RolloutBatch
from gfrl.runtime import configure_precision, encoder_fastpath_disabled
from validate_learner import error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--limit', type=int, default=512)
    args = p.parse_args()
    torch.set_num_threads(4)
    configure_precision('cuda', False)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Model(ModelConfig(**state['model_config'])).cuda().eval()
    model.load_state_dict(state['model'])
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    batch = RolloutBatch({k:v.cuda() for k,v in data['columns'].items()},
                         {k:v.cuda() for k,v in data['stages'].items()})
    batch = batch.take(torch.arange(min(args.limit, len(batch)), device='cuda'))
    del data, state
    obs = {k:batch.columns[k] for k in OBS_FIELDS}
    stages, owners = batch.stage_rows(torch.arange(len(batch), device='cuda'))
    prefix = {k:batch.stages[k][stages] for k in PREFIX_FIELDS}
    packed_env = SimpleNamespace(**obs, num_envs=len(batch), device=torch.device('cuda'))
    head_env = SimpleNamespace(**prefix, num_envs=len(owners), device=torch.device('cuda'))
    encoder = ChunkedPackedEncoder(model, packed_env)
    heads = SparseHeads(model, head_env)
    results = {}
    with torch.inference_mode(), encoder_fastpath_disabled():
        for name in ('fp32', 'bf16', 'packed'):
            if name == 'packed':
                features = encoder()
                lp = heads(features[owners].contiguous())
                value = heads.values(features)
            else:
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=name == 'bf16'):
                    features = model.encode(obs, pack=False)[:, :21].contiguous()
                    lp, _ = model.heads(features[owners], **prefix)
                    value = model.value_head(features[:, 0]).float()
            results[name] = dict(encoded=features[obs['obs_valid'][:, :21]].cpu(), log_probs=lp[prefix['mask']].cpu(),
                                 value=value.cpu(), choices=lp.argmax(-1).cpu())
    comparisons = {}
    for name, ref in [('bf16','fp32'), ('packed','fp32'), ('packed','bf16')]:
        a, b = results[name], results[ref]
        comparisons[name+'_vs_'+ref] = {k:error(a[k], b[k]) for k in ('encoded','log_probs','value')}
        comparisons[name+'_vs_'+ref]['argmax_disagreements'] = int((a['choices'] != b['choices']).sum())
    print(json.dumps(dict(model_config=vars(model.config), decisions=len(batch), stages=len(owners),
                          valid_log_probs=int(prefix['mask'].sum()), tf32=False,
                          comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
