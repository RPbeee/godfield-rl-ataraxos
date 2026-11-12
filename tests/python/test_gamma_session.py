"""Session interruption/restart without spending a cluster time budget."""
import json
from pathlib import Path

import pytest
import torch


def test_session_continues_partial_run_and_skips_complete(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    import gamma_session as session
    train = session.train
    source = train.run(train.parse_args([
        '--d', '32', '--depth', '1', '--num-envs', '8', '--threads', '1',
        '--games-per-iter', '2', '--iterations', '1', '--eval-every', '0',
        '--run-dir', str(tmp_path/'source')]))
    for key, value in dict(INIT=str(source), OUT_DIR=str(tmp_path/'experiment'),
                           GAMMAS='1 0.98', DEVICE='cpu', THREADS='1', NUM_ENVS='8',
                           GAMES_PER_ITER='2', ITERATIONS='2', HOURS_EACH='1', PAIRS='1').items():
        monkeypatch.setenv(key, value)
    calls = []
    interrupted = False

    def invoke(command, log):
        nonlocal interrupted
        if command[0] == 'scripts/train.py':
            args = train.parse_args(command[1:])
            calls.append((args.resume, args.init_from, args.iterations))
            if not interrupted:
                args.iterations = 1
                train.run(args)
                interrupted = True
                raise RuntimeError('simulated session interruption')
            train.run(args)
        else:
            Path(command[command.index('--out')+1]).write_text('{"results": []}')

    monkeypatch.setattr(session, 'invoke', invoke)
    with pytest.raises(RuntimeError, match='simulated'):
        session.main()
    out = tmp_path/'experiment'
    assert not (out/'g1/complete.json').exists()
    session.main()
    assert calls[0] == (None, str(source.resolve()), 2)
    assert calls[1] == ('auto', None, 1)
    assert calls[2] == (None, str(source.resolve()), 2)
    for name, gamma in (('g1', 1), ('g0.98', .98)):
        state = session.load(out/name/'checkpoint.pt')
        assert state['iteration'] == 3
        assert state['config']['gamma'] == gamma
        assert state['config']['games_per_iter'] == 2
        assert state['elapsed_s'] > 0
        assert (out/name/'complete.json').exists()
    assert (out/'summary.md').exists()
    assert len(list(out.glob('eval-*.json'))) == 3
    checkpoint = out/'g1/checkpoint.pt'
    modified = checkpoint.stat().st_mtime_ns
    session.main()
    assert len(calls) == 3
    assert checkpoint.stat().st_mtime_ns == modified
    manifest = json.loads((out/'g1/session.json').read_text())
    assert manifest['config']['seed'] == session.load(out/'g0.98/checkpoint.pt')['config']['seed']
