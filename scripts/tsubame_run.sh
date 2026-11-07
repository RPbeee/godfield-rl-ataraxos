#!/usr/bin/env bash
# Run a Python entry point with TSUBAME's allocator and runtime settings.
set -euo pipefail

export MALLOC_MMAP_THRESHOLD_=17179869184
export MALLOC_TRIM_THRESHOLD_=68719476736
export MALLOC_TOP_PAD_=1073741824
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if type module >/dev/null 2>&1; then
    module load gcc/14.2.0
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repo_root/.venv/bin/python" "$@"
