#!/usr/bin/env python3
"""Measure forward, gradient, Adam and EMA error against checkpoint FP32 PPO."""
import argparse
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl.learner import Learner, LearnerConfig
from gfrl.model import Model, ModelConfig
from gfrl.rollout import RolloutBatch
from gfrl.targets import compute_targets
from gfrl.runtime import configure_precision


def error(a, b):
    a, b = a.double(), b.double()
    diff = a-b
    return dict(max_abs=float(diff.abs().max()), rms=float(diff.square().mean().sqrt()),
                relative_l2=float(diff.norm()/b.norm().clamp_min(1e-30)))


def vector(model, grad=False):
    return torch.cat([(p.grad if grad and p.grad is not None else
                       torch.zeros_like(p) if grad else p).detach().flatten()
                       for p in model.parameters()]).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--limit', type=int, default=1024)
    p.add_argument('--backend', default='packed')
    p.add_argument('--dtype', default='bf16')
    p.add_argument('--minibatch-size', type=int, default=4096)
    p.add_argument('--microbatch-size', type=int, default=4096)
    p.add_argument('--forward-limit', type=int, default=256)
    p.add_argument('--steps', type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(4)
    configure_precision('cuda', False)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    data = torch.load(args.data, weights_only=False)
    batch = RolloutBatch({k:v.cuda() for k,v in data['columns'].items()},
                         {k:v.cuda() for k,v in data['stages'].items()})
    batch = batch.take(torch.arange(min(args.limit, len(batch)), device='cuda'))
    del data
    cfg_dict = {f.name:state['config'][f.name] for f in fields(LearnerConfig)
                if f.name in state['config']}
    cfg_dict.pop('learner_backend', None)
    cfg = LearnerConfig(**cfg_dict)
    xi, advantage = compute_targets(batch, cfg.lambda_v, cfg.lambda_a, cfg.gamma)
    from gfrl.gpu.targets import compute_targets as gpu_targets
    target_errors = []
    for gamma in (1., .995, .5):
        expected = compute_targets(batch, cfg.lambda_v, cfg.lambda_a, gamma)
        actual = gpu_targets(batch, cfg.lambda_v, cfg.lambda_a, gamma)
        target_errors.append(dict(gamma=gamma, target=error(actual[0].cpu(), torch.from_numpy(expected[0])),
                                  advantage=error(actual[1].cpu(), torch.from_numpy(expected[1]))))
    cfg_dict.update(minibatch_size=args.minibatch_size, microbatch_size=args.microbatch_size)
    results = {}
    for name, dtype, backend in [('fp32', 'fp32', 'eager'),
                                  ('bf16', 'bf16', 'eager'),
                                  (args.backend, args.dtype, args.backend)]:
        model = Model(ModelConfig(**state['model_config']))
        model.load_state_dict(state['model'])
        learner = Learner(model, LearnerConfig(**cfg_dict, learner_backend=backend), 'cuda', dtype)
        learner.optimizer.load_state_dict(deepcopy(state['optimizer']))
        if backend != 'eager':
            for group in learner.optimizer.param_groups:
                group.update(fused=True, foreach=False)
        learner.ema.load_state_dict(state['ema'])
        learner.optimizer_steps = state['optimizer_steps']
        model.eval()
        rows = torch.arange(min(args.forward_limit, len(batch)), device='cuda')
        inputs = learner._prepare(batch, rows)
        terms, values = learner.forward(inputs)
        from gfrl.learner import schedules
        _, alpha = schedules(learner.config, state['iteration']+1)
        target = torch.as_tensor(xi[:len(rows)], device='cuda')
        adv = torch.as_tensor(advantage[:len(rows)], device='cuda')
        ratio = terms['ratio']
        policy = -torch.minimum(ratio*adv, ratio.clamp(1-cfg.clip, 1+cfg.clip)*adv)
        loss = (policy+learner.config.kl_coef*terms['kl']+alpha*terms['magnet']).mean() - (
            target*values.log_softmax(-1)).sum(-1).mean()
        loss.backward()
        result = dict(loss=float(loss.detach()), value=values.detach().cpu(),
                      terms={k:v.detach().cpu() for k,v in terms.items()}, gradient=vector(model, True))
        del terms, values, loss, inputs
        model.zero_grad(set_to_none=True)
        initial = vector(model)
        updates, emas, metrics = [], [], []
        for step in range(args.steps):
            np.random.seed(123+step)
            metrics.append(learner.update(batch, state['iteration']+1+step))
            updates.append(vector(model)-initial)
            emas.append(vector(learner.ema))
        result.update(updates=updates, ema=emas, metrics=metrics)
        results[name] = result
        del learner, model
        torch.cuda.empty_cache()
    report = dict(decisions=len(batch), forward_decisions=len(rows), update_calls=args.steps,
                  optimizer_steps=args.steps*((len(batch)+args.minibatch_size-1)//args.minibatch_size),
                  minibatch_size=args.minibatch_size, microbatch_size=args.microbatch_size,
                  backend=args.backend, dtype=args.dtype, tf32=False,
                  targets=target_errors, comparisons={})
    for name in results:
        if name == 'fp32':
            continue
        for ref in ['fp32'] + (['bf16'] if name != 'bf16' else []):
            a, b = results[name], results[ref]
            report['comparisons'][name+'_vs_'+ref] = dict(
                loss_abs=abs(a['loss']-b['loss']), value=error(a['value'], b['value']),
                terms={k:error(a['terms'][k], b['terms'][k]) for k in b['terms']},
                gradient=error(a['gradient'], b['gradient']),
                adam_updates=[error(x,y) for x,y in zip(a['updates'], b['updates'])],
                ema=[error(x,y) for x,y in zip(a['ema'], b['ema'])],
                losses=[{k:dict(actual=x[k], reference=y[k], abs_error=abs(x[k]-y[k]))
                         for k in ['policy_loss','kl_loss','magnet_loss','value_loss']}
                        for x,y in zip(a['metrics'], b['metrics'])])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
