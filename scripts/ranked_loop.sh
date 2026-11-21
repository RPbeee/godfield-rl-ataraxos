#!/bin/sh
# Usage: scripts/ranked_loop.sh <games> <budget> <playwright-core path> [extra ranked_bot.py args]
# Starts one ranked game per process (budget = total slots the ledger may ever reserve)
# and stops at the first non-zero exit. Prints only the numeric JSON line of each run.
# Every new process meets the 生誕 screen again (the site shows it on each page load), so
# a loop needs --reenter in the extra args; without it the 2nd game stops before any click.
set -u
games="$1"; budget="$2"; pw="$3"; shift 3
cd "$(dirname "$0")/.." || exit 2
i=0
while [ "$i" -lt "$games" ]; do
  .venv/bin/python scripts/ranked_bot.py --budget "$budget" --playwright "$pw" "$@" || exit $?
  i=$((i + 1))
done
