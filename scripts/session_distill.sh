#!/usr/bin/env bash
# Resumable stages; all storage destinations and compute budgets are overridable.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
OUT_DIR=${OUT_DIR:-measurements/session-distill}
CHECKPOINT=${CHECKPOINT:-runs/play/m-2166.pt}
BELIEF=${BELIEF:-runs/belief/train-m/checkpoint.pt}
DATA_DIR=${DATA_DIR:-${TMPDIR:-/tmp}/gfrl-distill}
DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-16}
NUM_ENVS=${NUM_ENVS:-512}
PAIRS=${PAIRS:-1000}
mkdir -p "$OUT_DIR"
runtime=(--device "$DEVICE" --threads "$THREADS" --num-envs "$NUM_ENVS")
if [[ ${TF32:-1} == 1 ]]; then runtime+=(--tf32); else runtime+=(--no-tf32); fi
search=(--belief "$BELIEF" --worlds "${WORLDS:-16}" --depth "${DEPTH:-3}"
  --candidates "${CANDIDATES:-16}" --candidate-samples "${CANDIDATE_SAMPLES:-64}"
  --alpha "${ALPHA:-0.002}" --beta "${BETA:-0.1}" --leaf-view "${LEAF_VIEW:-actor}"
  --select sample --search-response --skip-forced-inference --skip-single-candidate
  --max-inference-batch "${MAX_INFERENCE_BATCH:-8192}"
  --search-cache-device "${SEARCH_CACHE_DEVICE:-auto}" --search-cache-mb "${SEARCH_CACHE_MB:-8192}")
if [[ -n ${FIT:-} ]]; then search+=(--fit "$FIT"); fi
if [[ -n ${BELIEF_HYBRID:-} ]]; then search+=(--belief-hybrid "$BELIEF_HYBRID"); fi
if [[ -n ${BELIEF_TEMPERATURE:-} ]]; then search+=(--belief-temperature "$BELIEF_TEMPERATURE"); fi
if [[ -n ${BELIEF_COUNT_TEMPERATURE:-} ]]; then search+=(--belief-count-temperature "$BELIEF_COUNT_TEMPERATURE"); fi
if [[ -n ${BELIEF_MIX:-} ]]; then search+=(--belief-mix "$BELIEF_MIX"); fi
if [[ ! -f "$OUT_DIR/data.complete" ]]; then
  "$PYTHON" scripts/make_distill_data.py --checkpoint "$CHECKPOINT" "${runtime[@]}" "${search[@]}" \
    --out-dir "$DATA_DIR" --resume --seed "${DATA_SEED:-900000}" \
    --games "${DATA_GAMES:-1000000}" --games-per-shard "${GAMES_PER_SHARD:-512}" \
    --max-hours "${DATA_HOURS:-2.5}" --progress-interval "${PROGRESS_INTERVAL:-600}" \
    2>&1 | tee -a "$OUT_DIR/data.log"
  touch "$OUT_DIR/data.complete"
fi
if [[ ! -f "$OUT_DIR/train/complete.json" ]]; then
  "$PYTHON" scripts/train_distill.py --resume --checkpoint "$CHECKPOINT" --data-dir "$DATA_DIR" \
    --run-dir "$OUT_DIR/train" --device "$DEVICE" --threads "$THREADS" \
    --max-hours "${TRAIN_HOURS:-0.75}" --epochs "${EPOCHS:-2}" \
    --minibatch "${MINIBATCH:-1024}" --microbatch "${MICROBATCH:-8}" \
    --dtype "${DTYPE:-fp32}" --validation-fraction "${VAL_FRACTION:-0.03}" \
    --lr "${LR:-0.00001}" --kl-coef "${KL_COEF:-0.1}" --value-coef "${VALUE_COEF:-0.5}" \
    --warmup-steps "${WARMUP_STEPS:-10}" --log-every "${LOG_EVERY:-10}" \
    "$([[ ${TF32:-1} == 1 ]] && echo --tf32 || echo --no-tf32)" \
    2>&1 | tee -a "$OUT_DIR/train.log"
fi
DISTILLED="$OUT_DIR/train/best.pt"
if [[ ! -f "$OUT_DIR/policy.json" ]]; then
  "$PYTHON" scripts/eval_policies.py --a "$DISTILLED" --b "$CHECKPOINT" "${runtime[@]}" \
    --pairs "$PAIRS" --eval-seed "${EVAL_SEED:-74100}" --out "$OUT_DIR/policy.json" \
    2>&1 | tee -a "$OUT_DIR/policy.log"
fi
# Evaluation b always uses the untransformed AR belief (same checkpoint).
ar_search=()
for ((i=0; i<${#search[@]}; i++)); do
  case "${search[i]}" in --belief-hybrid|--belief-temperature|--belief-count-temperature|--belief-mix) ((i+=1));; *) ar_search+=("${search[i]}");; esac
done
if [[ ! -f "$OUT_DIR/distilled-search.json" ]]; then
  "$PYTHON" scripts/eval_search.py --checkpoint "$DISTILLED" --opponent policy \
    --opponent-checkpoint "$CHECKPOINT" "${runtime[@]}" "${ar_search[@]}" \
    --pairs "$PAIRS" --seed "${EVAL_SEED:-74100}" --out "$OUT_DIR/distilled-search.json" \
    2>&1 | tee -a "$OUT_DIR/distilled-search.log"
fi
if [[ ${SKIP_REFERENCE:-0} != 1 && ! -f "$OUT_DIR/reference-search.json" ]]; then
  "$PYTHON" scripts/eval_search.py --checkpoint "$CHECKPOINT" --opponent policy \
    "${runtime[@]}" "${ar_search[@]}" --pairs "$PAIRS" --seed "${EVAL_SEED:-74100}" \
    --out "$OUT_DIR/reference-search.json" 2>&1 | tee -a "$OUT_DIR/reference-search.log"
fi
"$PYTHON" scripts/distill_summary.py --out-dir "$OUT_DIR" --data-dir "$DATA_DIR"
