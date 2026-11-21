#!/usr/bin/env python3
"""Fetch official God Field item/guardian static webp icons into local gitignored cache.

Downloads static public card artwork into ``runs/viewer-assets/`` (gitignored) so the
local read-only spectator dashboard can display official item images without hotlinking.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CARDS_PATH = ROOT / 'third_party/goodfield/data/official-v2/cards.json'
DEFAULT_OUT = ROOT / 'runs/viewer-assets'
BASE_URL = 'https://godfield.net/images'
GUARDIANS = ('mars', 'mercury', 'jupiter', 'saturn', 'uranus', 'pluto', 'neptune', 'venus', 'earth', 'moon')
CURSES = ('cold', 'fever', 'hell', 'heaven', 'fog', 'dream', 'flash', 'darkcloud')


def _fetch_one(rel_path: str, out_dir: Path, *, force: bool = False):
    dest = out_dir / rel_path
    if not force and dest.is_file() and dest.stat().st_size > 0:
        return rel_path, 'skipped'
    url = f'{BASE_URL}/{rel_path}'
    req = urllib.request.Request(url, headers={'User-Agent': 'godfield-rl-viewer-cache/1.0'})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                if resp.status == 200 and data:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix('.tmp')
                    tmp.write_bytes(data)
                    tmp.replace(dest)
                    return rel_path, 'downloaded'
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return rel_path, 'not_found'
            if attempt == 1:
                return rel_path, f'http_{exc.code}'
        except Exception as exc:
            if attempt == 1:
                return rel_path, f'err_{type(exc).__name__}'
    return rel_path, 'failed'


def main(argv=None):
    p = argparse.ArgumentParser(description='Download official God Field static webp icons into runs/viewer-assets')
    p.add_argument('--cards', type=Path, default=CARDS_PATH)
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--force', action='store_true')
    args = p.parse_args(argv)

    cards = json.loads(args.cards.read_text(encoding='utf-8'))
    targets = ['items/fake.webp']
    for c in cards:
        cat, name = c.get('category', ''), c.get('imageName', '')
        if re.fullmatch('[a-z-]+', cat) and re.fullmatch('[a-z0-9-]+', name):
            targets.append(f'items/{cat}/{name}.webp')
    optional_targets = [f'guardians/{g}.webp' for g in GUARDIANS] + [f'curses/{c}.webp' for c in CURSES]

    counts = dict(downloaded=0, skipped=0, not_found=0, failed=0)
    failed_items = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as ex:
        futs = {ex.submit(_fetch_one, rel, args.out_dir, force=args.force): (rel, True) for rel in targets}
        for rel in optional_targets:
            futs[ex.submit(_fetch_one, rel, args.out_dir, force=args.force)] = (rel, False)
        for fut in as_completed(futs):
            rel, required = futs[fut]
            _, status = fut.result()
            if status in ('downloaded', 'skipped', 'not_found'):
                counts[status] += 1
            else:
                counts['failed'] += 1
                if required:
                    failed_items.append((rel, status))
            if required and status == 'not_found':
                failed_items.append((rel, status))

    out = dict(
        out_dir=str(args.out_dir),
        required_total=len(targets),
        optional_total=len(optional_targets),
        **counts,
        failed_required=failed_items[:20],
    )
    print(json.dumps(out, sort_keys=True), flush=True)
    return 0 if not failed_items else 1


if __name__ == '__main__':
    raise SystemExit(main())
