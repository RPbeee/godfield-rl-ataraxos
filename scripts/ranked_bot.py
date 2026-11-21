#!/usr/bin/env python3
"""Entry point for the capped official ranked (真剣タイマン) run. See scripts/ranked_bot.md."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from gfrl.official_ranked import main
if __name__ == '__main__':
    raise SystemExit(main())
