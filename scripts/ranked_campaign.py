#!/usr/bin/env python3
"""Read-only anonymous campaign report of the ranked ledger. See scripts/ranked_campaign.md."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from gfrl.ranked_campaign import main
if __name__ == '__main__':
    raise SystemExit(main())
