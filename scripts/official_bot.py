#!/usr/bin/env python3
"""Entry point for the explicit opt-in official training UI bot."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from gfrl.official_bot import main
if __name__ == '__main__':
    raise SystemExit(main())
