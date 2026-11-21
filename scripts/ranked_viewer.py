#!/usr/bin/env python3
"""Read-only live spectator and stats dashboard for official ranked campaigns."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from gfrl.ranked_viewer import main
if __name__ == '__main__':
    raise SystemExit(main())
