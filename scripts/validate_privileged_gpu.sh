#!/usr/bin/env bash
# Run only in the existing compute shell, after UPLIFT10_DONE.
set -euo pipefail
module load gcc/14.2.0 cuda/13.1.1
TASK=/tmp/$USER-pcritic
SOURCE="$HOME/godfield-rl-ataraxos"
mkdir -p "$TASK/src" "$TASK/logs"
export TMPDIR="$TASK/tmp"
mkdir -p "$TMPDIR"
AUDIT="$SOURCE/measurements/pcritic_20261117/gpu"
mkdir -p "$AUDIT"
copy_logs() {
  rsync -a "$TASK/logs/" "$AUDIT/"
  for name in br-before br-after; do
    if test -f "$TASK/$name.json"; then cp "$TASK/$name.json" "$AUDIT/"; fi
  done
}
trap copy_logs EXIT
export PATH="$SOURCE/.venv/bin:$PATH" OMP_NUM_THREADS=4 MAX_JOBS=8
export TORCHINDUCTOR_COMPILE_THREADS=8
export TRITON_CACHE_DIR="$TASK/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/inductor"
export GFRL_GPU_CACHE="$TASK/gpu"
rsync -a --exclude=.venv --exclude='build*' --exclude=runs --exclude=measurements \
  --exclude='*.so' --exclude=.git --exclude=.claude --exclude=.codex --exclude=.agents \
  --exclude=.aws --exclude='node_modules' --exclude='__pycache__' --exclude=.pytest_cache \
  "$SOURCE/" "$TASK/src/"
ln -sfn "$SOURCE/runs" "$TASK/src/runs"
cd "$TASK/src"
date > "$TASK/logs/machine.txt"
hostname >> "$TASK/logs/machine.txt"
nvidia-smi >> "$TASK/logs/machine.txt"
python -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_properties(0))' >> "$TASK/logs/machine.txt"
cmake -S . -B "$TASK/build" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python)" > "$TASK/logs/build.log" 2>&1
cmake --build "$TASK/build" --target _core gfrl_gpu_oracle -j 8 >> "$TASK/logs/build.log" 2>&1
export GFRL_GPU_CPU_ORACLE="$TASK/build/gfrl_gpu_oracle"
echo PCRITIC_BUILD_DONE
copy_logs
timeout 1800 python -m pytest -q tests/python/test_privileged_critic.py tests/python/test_pool_epsilon.py \
  tests/python/test_fast_learner.py tests/python/test_learner_cuda.py tests/python/test_pool.py \
  > "$TASK/logs/gpu-tests.log" 2>&1
echo PCRITIC_TESTS_DONE
copy_logs
timeout 1800 python -m pytest -q tests/python/test_gpu.py > "$TASK/logs/gpu-env-tests.log" 2>&1
echo PCRITIC_ENV_TESTS_DONE
copy_logs
timeout 1800 python -m pytest -q tests/python/test_gpu_search.py > "$TASK/logs/gpu-search-tests.log" 2>&1
echo PCRITIC_SEARCH_TESTS_DONE
copy_logs
C=runs/hp-lr1e5/ckpt-4500.pt
timeout 1800 python scripts/bench_learner.py --checkpoint "$C" --collect \
  --data "$TASK/rollout.pt" --num-envs 1024 --games 1024 --collect-warmup 4 --pool-fraction .2 \
  > "$TASK/logs/collect.log" 2>&1
for critic in shared privileged; do
  timeout 1800 python scripts/bench_learner.py --checkpoint "$C" --data "$TASK/rollout.pt" \
    --backend packed-grouped --critic "$critic" --limit 16384 \
    --minibatch-size 4096 --microbatch-size 4096 --warmup 3 --iterations 10 --no-tf32 \
    > "$TASK/logs/learner-$critic.log" 2>&1
done
echo PCRITIC_BENCH_DONE
copy_logs
COMMON=(--device cuda --env-backend cuda --dtype bf16 --learner-backend packed-grouped \
  --compile off --no-tf32 --num-envs 1024 --threads 4 --games-per-iter 1024 \
  --minibatch-size 4096 --microbatch-size 4096 --lr 1e-5 --gamma 1 \
  --eval-every 0 --max-hours .45 --pool-every 1000000 --pool-size 1)
timeout 1800 python scripts/train.py --init-from "$C" --run-dir "$TASK/shared" \
  "${COMMON[@]}" --pool-fraction .2 --critic shared --iterations 10 \
  > "$TASK/logs/train-shared.log" 2>&1
timeout 1800 python scripts/train.py --init-from "$C" --run-dir "$TASK/privileged" \
  "${COMMON[@]}" --pool-fraction .2 --critic privileged --critic-lr 1e-4 --iterations 12 \
  > "$TASK/logs/train-privileged.log" 2>&1
timeout 1800 python scripts/train.py --resume auto --run-dir "$TASK/privileged" \
  "${COMMON[@]}" --iterations 2 > "$TASK/logs/resume-privileged.log" 2>&1
echo PCRITIC_SELFPLAY_DONE
copy_logs
python - <<'PY'
import torch
state = torch.load('runs/hp-lr1e5/ckpt-4500.pt', map_location='cpu', weights_only=False)
state['pool'] = [dict(iteration=state['iteration'], weights=state['ema'])]
torch.save(state, f"/tmp/{__import__('os').environ['USER']}-pcritic/br-init.pt")
PY
timeout 1800 python scripts/train.py --init-from "$TASK/br-init.pt" --run-dir "$TASK/br" \
  "${COMMON[@]}" --pool-fraction 1 --pool-epsilon .15 --critic privileged \
  --critic-lr 1e-4 --iterations 10 > "$TASK/logs/train-br.log" 2>&1
for phase in before after; do
  A="$TASK/br-init.pt"
  if test "$phase" = after; then A="$TASK/br/checkpoint.pt"; fi
  timeout 1800 python scripts/eval_policies.py --a "$A" --b "$TASK/br-init.pt@pool:4500" \
    --pool-epsilon .15 --pairs 64 --num-envs 64 --threads 4 --device cuda --eval-seed 917 \
    --out "$TASK/br-$phase.json" > "$TASK/logs/eval-br-$phase.log" 2>&1
done
copy_logs
echo PCRITIC_ALL_DONE
date
