#!/usr/bin/env bash
# Offline, resumable TSUBAME session. See session_belief_retrain.md for budgets.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-.venv/bin/python}" scripts/session_belief_retrain.py
