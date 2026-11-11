#!/usr/bin/env bash
# One resumable TSUBAME sweep; override budgets via env for a CPU smoke.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
OUT_DIR=${OUT_DIR:-measurements/session-hybrid-belief}
CHECKPOINT=${CHECKPOINT:-runs/belief/policy-539.pt}
BELIEF=${BELIEF:-runs/belief/train-m/checkpoint.pt}
FIT=${FIT:-runs/belief/fit.pt}
mkdir -p "$OUT_DIR"
cat <<'ESTIMATE' | tee -a "$OUT_DIR/session.log"
TSUBAME H100 estimate at 512 envs: ~10 min / 1000 games / setting.
Four settings at 1000 pairs (2000 games) plus shared baseline: ~1.5 h.
True-hand diagnostic inference adds overhead; monitor progress.
All settings share paired deals and a no-search policy baseline. Resume enabled.
ESTIMATE
common=(--checkpoint "$CHECKPOINT" --opponent policy --device "${DEVICE:-cuda}"
  --threads "${THREADS:-16}" --num-envs "${NUM_ENVS:-512}"
  --candidates "${CANDIDATES:-16}" --candidate-samples "${CANDIDATE_SAMPLES:-64}"
  --max-inference-batch "${MAX_INFERENCE_BATCH:-8192}"
  --search-cache-device auto --search-cache-mb "${SEARCH_CACHE_MB:-8192}"
  --skip-forced-inference --skip-single-candidate --search-response
  --worlds "${WORLDS:-16}" --depth "${DEPTH:-3}" --belief "$BELIEF" --fit "$FIT"
  --pairs "${PAIRS:-1000}" --seed "${SEED:-74100}"
  --alpha 0.002 --beta 0.1 --leaf-view actor
  --diagnostic-true-hand-nll --diagnostic-belief "$BELIEF"
  --progress-interval "${PROGRESS_INTERVAL:-600}")
if [[ ${TF32:-1} == 1 ]]; then common+=(--tf32); else common+=(--no-tf32); fi
"$PYTHON" scripts/sweep_search.py "${common[@]}" --resume --out-dir "$OUT_DIR/step1" --grid \
  belief_hybrid=1,belief_count_temperature=1 \
  belief_hybrid=1,belief_count_temperature=1.5 \
  belief_hybrid=0,belief_count_temperature=1 \
  belief_hybrid=0,belief_count_temperature=1,belief=b1 \
  2>&1 | tee -a "$OUT_DIR/step1.log" "$OUT_DIR/session.log"
