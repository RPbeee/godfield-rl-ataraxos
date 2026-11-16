#!/usr/bin/env python3
"""Compare measured generation summaries; do not add inclusive timers together."""
import json
from pathlib import Path
import sys


def summary(directory):
    d=json.loads((directory/'summary.json').read_text())
    seconds=d['search_seconds']
    phases={k:seconds.get(k,0.) for k in ('sample','candidate','rollout','leaf','records_and_selection',
             'world_workspace','world_load','real_policy','real_env_step','rollout_encoder','rollout_heads')}
    phases['search_control_and_recording']=seconds.get('total',0.)-sum(seconds.get(k,0.) for k in ('sample','candidate','rollout','leaf'))
    return dict(searched_positions_per_second=d['searched_positions_per_second'],games_per_hour=d['games_per_hour'],
                games=d['games'],examples=d['examples'],searched=d['new_searched_positions'],elapsed_seconds=d['elapsed_seconds'],
                serialize_seconds=d['serialize_seconds'],seconds=phases,search_counts=d['search_counts'],
                timer_note='world_workspace/load are INCLUDED in candidate/rollout/control; records_and_selection is INCLUDED in search_control_and_recording',
                provenance=d['provenance'])


def main():
    run=Path(sys.argv[1]);cpu=summary(run/'cpu');gpu=summary(run/'cuda')
    speedup=gpu['searched_positions_per_second']/cpu['searched_positions_per_second']
    print(json.dumps(dict(cpu=cpu,cuda=gpu,speedup=speedup,speedup_vs_historical_35=gpu['searched_positions_per_second']/35,target_met=gpu['searched_positions_per_second']>=350),indent=2))


if __name__=='__main__':main()
