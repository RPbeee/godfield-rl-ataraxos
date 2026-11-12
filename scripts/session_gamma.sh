#!/usr/bin/env bash
# Resumable gamma experiment; all budgets/runtime settings accept env overrides.
# INIT, GAMMAS, HOURS_EACH, OUT_DIR, ITERATIONS, PAIRS, DEVICE, EVAL_NUM_ENVS.
# Training overrides use uppercase argument names, e.g. NUM_ENVS=8,
# GAMES_PER_ITER=8 THREADS=2 MICROBATCH_SIZE=32 for a CPU smoke.
# Completion is at the time limit (including the trainer's iteration headroom)
# or the cumulative ITERATIONS cap. Re-run with unchanged settings to continue.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
exec "$PYTHON" scripts/gamma_session.py
