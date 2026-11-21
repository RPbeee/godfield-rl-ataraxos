#!/usr/bin/env python3
"""Resumable campaign driver for the capped official ranked run. See scripts/ranked_drive.md."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from gfrl.ranked_drive import main
if __name__ == '__main__':
    raise SystemExit(main())
