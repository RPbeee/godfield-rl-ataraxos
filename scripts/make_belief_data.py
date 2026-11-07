#!/usr/bin/env python3
"""Collect fixed-policy belief validation positions or label-only fitting data."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch
from gfrl.belief_data import make_belief_data
from gfrl.runtime import configure_precision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--pool-fraction', type=float, default=.2)
    p.add_argument('--opponent', choices=('self', 'greedy'), default='self')
    p.add_argument('--positions', type=int, default=20000)
    p.add_argument('--sample-prob', type=float, default=.1)
    p.add_argument('--labels-only', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda', 'mps'))
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    args = vars(p.parse_args())
    out, tf32 = args.pop('out'), args.pop('tf32')
    torch.set_num_threads(args['threads'])
    configure_precision(args['device'], tf32)
    data = make_belief_data(**args)
    data['config']['tf32'] = tf32
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(f'{out}: {data["config"]["positions"]} positions, {out.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
