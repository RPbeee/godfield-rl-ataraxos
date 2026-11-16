#!/usr/bin/env bash
# Run INSIDE the existing compute shell. Each measurement has a 29-minute limit.
set -euo pipefail
cd "$(dirname "$0")/.."
case "$(hostname -s)" in login*) echo 'Run in the existing compute job, not a login node' >&2; exit 1;; esac
THREADS=${THREADS:-16}
if [[ ! "$THREADS" =~ ^[0-9]+$ ]] || (( THREADS < 1 || THREADS > 24 )); then
  echo 'THREADS must be in 1..24 on this allocation' >&2; exit 1
fi
module load gcc/14.2.0 cuda/13.1.1
export PATH="$PWD/.venv/bin:$PATH"
SCRATCH=${SCRATCH:-/tmp/$USER-gsearch}
mkdir -p "$SCRATCH"/{triton,inductor,engine,cache,tmp,pycache,cuda-cache,torch,extensions}
export TMPDIR="$SCRATCH/tmp" TRITON_CACHE_DIR="$SCRATCH/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" GFRL_GPU_CACHE="$SCRATCH/engine"
export XDG_CACHE_HOME="$SCRATCH/cache" PYTHONPYCACHEPREFIX="$SCRATCH/pycache"
export CUDA_CACHE_PATH="$SCRATCH/cuda-cache" TORCH_HOME="$SCRATCH/torch"
export TORCH_EXTENSIONS_DIR="$SCRATCH/extensions"
export OMP_NUM_THREADS=${THREADS:-16}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export GFRL_GPU_VERBOSE=1
CHECKPOINT=${CHECKPOINT:-runs/play/m-2166.pt}
BELIEF=${BELIEF:-runs/belief-m2166/train/checkpoint.pt}
export GFRL_GPU_SEARCH_CHECKPOINT="$CHECKPOINT" GFRL_GPU_SEARCH_BELIEF="$BELIEF"
RUN="$SCRATCH/measure-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_properties(0))' > "$RUN/device.txt"
# CUDA tests are mandatory; a Mac/native pass is insufficient.
timeout 1740 python -m pytest tests/python/test_gpu_search.py -q -k cuda > "$RUN/exact-cuda.log" 2>&1
common=(--checkpoint "$CHECKPOINT" --belief "$BELIEF" --device cuda --threads "${THREADS:-16}"
  --worlds 16 --depth 3 --candidates 16 --candidate-samples 64
  --alpha .002 --beta .1 --leaf-view actor --search-response --select sample
  --skip-forced-inference --skip-single-candidate --max-inference-batch 8192
  --search-cache-device device --search-cache-mb 8192
  --max-hours 0 --progress-interval 30 --tf32)
for backend in cpu cuda; do
  if [[ "$backend" == cpu ]]; then
    count=${CPU_GAMES:-128}; lanes=${CPU_NUM_ENVS:-128}; roots=${CPU_ROOT_BATCH_SIZE:-32}
  else
    count=${GAMES:-8192}; lanes=${NUM_ENVS:-8192}; roots=${ROOT_BATCH_SIZE:-128}
  fi
  batching=(--num-envs "$lanes" --root-batch-size "$roots")
  # Separate warm-up and measured outputs. Never mix a warm-up into throughput.
  timeout 1740 python scripts/make_distill_data.py "${common[@]}" "${batching[@]}" --env-backend "$backend" --search-dtype "${SEARCH_DTYPE:-bf16}" \
    --games 4 --games-per-shard 4 --seed 910000 --out-dir "$RUN/$backend-warmup" \
    > "$RUN/$backend-warmup.log" 2>&1
  timeout 1740 python scripts/make_distill_data.py "${common[@]}" "${batching[@]}" --env-backend "$backend" --search-dtype "${SEARCH_DTYPE:-bf16}" \
    --games "$count" --games-per-shard "$count" --seed 910001 --profile \
    --out-dir "$RUN/$backend" > "$RUN/$backend.log" 2>&1
 done
# The real existing loader and learner, using CUDA-generated format v1 shards.
timeout 1740 python scripts/train_distill.py --checkpoint "$CHECKPOINT" --data-dir "$RUN/cuda-warmup" \
  --run-dir "$RUN/train-smoke" --device cuda --threads "${THREADS:-16}" --epochs 1 \
  --minibatch 32 --microbatch 4 --max-hours .002 --validation-fraction .5 \
  --log-every 1 --tf32 > "$RUN/train-smoke.log" 2>&1
python scripts/summarize_gpu_search.py "$RUN" > "$RUN/comparison.json"
cat "$RUN/comparison.json"
printf 'Artifacts: %s\n' "$RUN"
