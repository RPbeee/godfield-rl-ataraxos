#!/usr/bin/env python3
"""Offline official receive-record advisor (no network)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from gfrl.official_advisor import main
if __name__ == '__main__':
    raise SystemExit(main())
