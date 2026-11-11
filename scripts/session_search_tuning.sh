#!/usr/bin/env bash
# One sequential TSUBAME session. No preexisting output directory is required.
# Examples: BEST_ALPHA=0.02 BEST_BETA=0.3 ./scripts/session_search_tuning.sh
# CPU smoke: DEVICE=cpu TF32=0 PAIRS=1 FINAL_PAIRS=1 NUM_ENVS=2 THREADS=1
# WORLDS=1 DEPTH=0 CANDIDATES=1 CANDIDATE_SAMPLES=1 MAX_INFERENCE_BATCH=16
# CHECKPOINT=... BELIEF=... FIT=... OUT_DIR=measurements/session-smoke ...
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
OUT_DIR=${OUT_DIR:-measurements/session-search-tuning}
CHECKPOINT=${CHECKPOINT:-runs/belief/policy-539.pt}
BELIEF=${BELIEF:-runs/belief/train-m/checkpoint.pt}
FIT=${FIT:-runs/belief/fit.pt}
PAIRS=${PAIRS:-500}
FINAL_PAIRS=${FINAL_PAIRS:-1000}
SEED=${SEED:-73100}
mkdir -p "$OUT_DIR"
cat <<'ESTIMATE' | tee "$OUT_DIR/session.log"
TSUBAME estimate (16 worlds / depth 3, 512 envs, ~0.024 s/decision):
400 games took 6–8 min; each 500-pair setting (1000 games) ~15–20 min.
Step 1: 9 settings + shared baseline: ~2.3–3.1 hours.
Step 2: 6 belief settings + shared baseline: ~1.5–2.1 hours, plus NLL inference.
Step 3: best vs policy, 1000 pairs: ~30–40 min + shared baseline.
Total ~4.3–5.9 hours before diagnostics/overhead; allow ~8–9 hours (<10h).
Diagnostic overhead depends on AR size and searched fraction; monitor progress.
Each step uses fresh paired deals; settings within a step share deals and baseline.
ESTIMATE
common=(--checkpoint "$CHECKPOINT" --opponent policy --device "${DEVICE:-cuda}"
  --threads "${THREADS:-16}" --num-envs "${NUM_ENVS:-512}"
  --candidates "${CANDIDATES:-16}" --candidate-samples "${CANDIDATE_SAMPLES:-64}"
  --max-inference-batch "${MAX_INFERENCE_BATCH:-8192}"
  --search-cache-device auto --search-cache-mb "${SEARCH_CACHE_MB:-8192}"
  --skip-forced-inference --skip-single-candidate --search-response
  --worlds "${WORLDS:-16}" --depth "${DEPTH:-3}" --belief "$BELIEF" --fit "$FIT"
  --progress-interval "${PROGRESS_INTERVAL:-600}")
if [[ ${TF32:-1} == 1 ]]; then common+=(--tf32); else common+=(--no-tf32); fi
run_step() {
  local name=$1; shift
  "$PYTHON" scripts/sweep_search.py "${common[@]}" --resume --out-dir "$OUT_DIR/$name" "$@" 2>&1 | tee -a "$OUT_DIR/$name.log" "$OUT_DIR/session.log"
}
grid=()
for a in 0.002 0.02; do
  for b in 0.02 0.1 0.3 1.0; do
    grid+=("alpha=$a,beta=$b,leaf_view=actor")
  done
done
grid+=(alpha=0.002,beta=0.1,leaf_view=searcher)
run_step step1 --pairs "$PAIRS" --seed "$SEED" --grid "${grid[@]}"
# Substitute step-1 winner using BEST_ALPHA/BEST_BETA/BEST_LEAF_VIEW.
run_step step2 --pairs "$PAIRS" --seed "$((SEED+1))" \
  --alpha "${BEST_ALPHA:-0.002}" --beta "${BEST_BETA:-0.1}" --leaf-view "${BEST_LEAF_VIEW:-actor}" \
  --diagnostic-true-hand-nll --diagnostic-belief "$BELIEF" --grid \
  belief_temperature=0.7,belief_mix=0 \
  belief_temperature=1.5,belief_mix=0 \
  belief_temperature=2.0,belief_mix=0 \
  belief_temperature=1.0,belief_mix=0.5 \
  belief_temperature=1.0,belief_mix=0,belief=b1 \
  belief_temperature=1.0,belief_mix=0
# Existing searched-opponent evaluators cannot compare arbitrary search configs.
# Confirm the best observed setting vs policy on fresh 1000 paired seeds.
best_grid=$("$PYTHON" - "$OUT_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
reports = []
for step in ('step1', 'step2'):
    summary = json.loads((root / step / 'summary.json').read_text())
    for row in summary['results']:
        if row['kind'] == 'search':
            reports.append(json.loads((root / step / row['report']).read_text()))
best = max(reports, key=lambda r: r['win_rate'])
c = best['config']
keys = ('alpha', 'beta', 'leaf_view', 'belief_temperature', 'belief_mix')
values = {k: c[k] for k in keys if k in c}
if c['belief'] in ('b1', 'uniform'):
    values['belief'] = c['belief']
print(','.join(f'{k}={v}' for k, v in values.items()))
PY
)
echo "Final confirmation grid: ${FINAL_GRID:-$best_grid}" | tee -a "$OUT_DIR/session.log"
run_step step3 --pairs "$FINAL_PAIRS" --seed "$((SEED+2))" --grid "${FINAL_GRID:-$best_grid}"
