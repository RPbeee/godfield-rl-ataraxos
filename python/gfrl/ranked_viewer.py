"""Read-only live spectator & campaign dashboard server for official ranked runs.

Reads ``ledger.jsonl``, ``anon_turns.jsonl`` and ``screen-state.json`` without locks or
writes, so it never interferes with a running campaign. Everything served is strictly
anonymous (numeric stats, public card catalog names/IDs/images and fixed enums).
"""
from __future__ import annotations

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse

from .ranked_campaign import DEFAULT_SUPPLEMENT, DEFAULT_SUPPLEMENT_GAME, ROOT, report_from_path
from .ranked_ledger import HARD_LIMIT, MAX_CAMPAIGN_LIMIT
from .ranked_trace import GUARDIANS, validate_trace_row

CARDS_PATH = ROOT / 'third_party/goodfield/data/official-v2/cards.json'
DEFAULT_STATE_DIR = ROOT / 'runs/ranked1000'
PREV_STATE_DIR = ROOT / 'runs/ranked100'
DEFAULT_ASSETS_DIR = ROOT / 'runs/viewer-assets'
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / 'viewer_static'
STATIC_FILES = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/index.html': ('index.html', 'text/html; charset=utf-8'),
    '/static/styles.css': ('styles.css', 'text/css; charset=utf-8'),
    '/static/app.js': ('app.js', 'application/javascript; charset=utf-8'),
}
CURSE_NAMES = ('cold', 'fever', 'hell', 'heaven', 'fog', 'dream', 'flash', 'darkcloud')


def load_catalog(path=CARDS_PATH):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    out = {}
    for c in raw:
        cat = c.get('category', '')
        img_name = c.get('imageName', '')
        valid_img = bool(re.fullmatch('[a-z-]+', cat) and re.fullmatch('[a-z0-9-]+', img_name))
        out[str(c['id'])] = dict(
            id=c['id'],
            name=c.get('name', f"#{c['id']}"),
            category=cat,
            image_name=img_name,
            image_url=f'/assets/items/{cat}/{img_name}.webp' if valid_img else '',
            element=c.get('element') or '',
            atk=c.get('atk', 0),
            plus=bool(c.get('isPlusAtk', False)),
            defense=c.get('def', 0),
            cost=c.get('cost', 0),
            price=c.get('price', 0),
            hit_rate=c.get('hitRate', 0),
            ability=c.get('ability') or '',
            ability_value=c.get('abilityValue', 0),
            curse=c.get('curse') or '',
            guardian=c.get('guardian') or '',
        )
    return out


def read_trace_rows(path):
    """Read complete, validated anonymous trace lines without locking."""
    path = Path(path)
    if not path.is_file():
        return []
    data = path.read_bytes()
    head, _, tail = data.rpartition(b'\n')
    consumed = data[:len(head) + 1] if len(data) > len(tail) else b''
    rows = []
    for line in consumed.split(b'\n'):
        if not line.strip():
            continue
        try:
            rows.append(validate_trace_row(json.loads(line)))
        except (ValueError, RecursionError):
            continue
    return rows


def read_screen_state(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(obj, dict):
            return dict(
                stage=obj.get('stage'),
                lobby=bool(obj.get('lobby', False)),
                in_game=bool(obj.get('in_game', False)),
                waited_s=obj.get('waited_s'),
                steps_done=obj.get('steps_done') if isinstance(obj.get('steps_done'), list) else [],
                mtime=int(path.stat().st_mtime),
            )
    except (OSError, ValueError):
        pass
    return None


def compute_breakdowns(report):
    counted = [g for g in report['games'] if g['counted']]
    completed = [g for g in counted if g['klass'] == 'completed']
    stopped = [g for g in counted if g['klass'] not in ('completed', 'in_progress')]
    buckets = [
        (0, 1500, '<1500'),
        (1500, 1600, '1500-1599'),
        (1600, 1700, '1600-1699'),
        (1700, 1800, '1700-1799'),
        (1800, 1900, '1800-1899'),
        (1900, 9999, '1900+'),
    ]
    by_opp_rating = []
    for lo, hi, label in buckets:
        gs = [g for g in completed if g.get('rating') and g['rating']['opp_before']['value'] is not None
              and lo <= g['rating']['opp_before']['value'] < hi]
        w = sum(g['outcome'] == 'win' for g in gs)
        l = sum(g['outcome'] == 'loss' for g in gs)
        d = sum(g['outcome'] == 'draw' for g in gs)
        by_opp_rating.append(dict(label=label, n=len(gs), wins=w, losses=l, draws=d,
                                  rate=round(w / len(gs), 4) if gs else None))
    by_mover = {}
    for m in ('self', 'opponent'):
        gs = [g for g in completed if g['first_mover'] == m]
        w = sum(g['outcome'] == 'win' for g in gs)
        l = sum(g['outcome'] == 'loss' for g in gs)
        d = sum(g['outcome'] == 'draw' for g in gs)
        by_mover[m] = dict(n=len(gs), wins=w, losses=l, draws=d,
                           rate=round(w / len(gs), 4) if gs else None)
    stop_reasons = dict(Counter(g['stop_reason'] or 'unknown' for g in stopped))
    decs = [g['decisions'] for g in completed if g['decisions'] is not None]
    durs = [g['duration_s'] for g in completed if g['duration_s'] is not None]
    return dict(
        by_opp_rating=by_opp_rating,
        by_first_mover=by_mover,
        stop_reasons=stop_reasons,
        mean_decisions=round(sum(decs) / len(decs), 1) if decs else None,
        mean_duration_s=round(sum(durs) / len(durs), 1) if durs else None,
    )


class ViewerStore:
    """Thread-safe, mtime/size-cached read-only view of campaign ledger, traces and static assets."""
    def __init__(self, state_dir, *, hard_limit=MAX_CAMPAIGN_LIMIT, prev_dir=None, prev_hard_limit=HARD_LIMIT,
                 catalog_path=CARDS_PATH, assets_dir=DEFAULT_ASSETS_DIR, static_dir=DEFAULT_STATIC_DIR):
        self.state_dir = Path(state_dir).absolute()
        self.hard_limit = hard_limit
        self.prev_dir = Path(prev_dir).absolute() if prev_dir else None
        self.prev_hard_limit = prev_hard_limit
        self.catalog = load_catalog(catalog_path)
        self.assets_dir = Path(assets_dir).absolute() if assets_dir else None
        self.static_dir = Path(static_dir).absolute()
        self._asset_allowlist = self._build_asset_allowlist()
        self._lock = threading.Lock()
        self._ledger_cache = (None, None)
        self._prev_cache = (None, None)
        self._trace_cache = (None, {}, None)

    def _build_asset_allowlist(self):
        if self.assets_dir is None:
            return {}
        allowed = {'items/fake.webp': self.assets_dir / 'items/fake.webp'}
        for c in self.catalog.values():
            url = c.get('image_url') or ''
            if url.startswith('/assets/'):
                rel = url[len('/assets/'):]
                allowed[rel] = self.assets_dir / rel
        for g in GUARDIANS:
            if g != 'none':
                allowed[f'guardians/{g}.webp'] = self.assets_dir / f'guardians/{g}.webp'
        for cur in CURSE_NAMES:
            allowed[f'curses/{cur}.webp'] = self.assets_dir / f'curses/{cur}.webp'
        return allowed

    def asset_bytes(self, rel_path: str) -> bytes | None:
        p = self._asset_allowlist.get(rel_path)
        if p is None or not p.is_file():
            return None
        try:
            return p.read_bytes()
        except OSError:
            return None

    def static_bytes(self, filename: str) -> bytes | None:
        p = self.static_dir / filename
        if not p.is_file():
            return None
        try:
            return p.read_bytes()
        except OSError:
            return None

    def asset_inventory(self):
        items_cached = sum(1 for k, p in self._asset_allowlist.items() if k.startswith('items/') and p.is_file())
        guardians = sorted(k[len('guardians/'):-5] for k, p in self._asset_allowlist.items()
                           if k.startswith('guardians/') and p.is_file())
        curses = sorted(k[len('curses/'):-5] for k, p in self._asset_allowlist.items()
                        if k.startswith('curses/') and p.is_file())
        return dict(items_cached=items_cached, guardians=guardians, curses=curses)

    @staticmethod
    def _sig(path):
        try:
            st = path.stat()
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None

    def _get_report(self, directory, hard_limit, cache_attr, *, expect_preliminary):
        ledger_path = directory / 'ledger.jsonl'
        sig = self._sig(ledger_path)
        cached_sig, cached_val = getattr(self, cache_attr)
        if sig == cached_sig and cached_val is not None:
            return cached_val
        if sig is None:
            val = None
        else:
            sup = DEFAULT_SUPPLEMENT if expect_preliminary and DEFAULT_SUPPLEMENT.is_file() else None
            val = report_from_path(ledger_path, sup, DEFAULT_SUPPLEMENT_GAME,
                                   hard_limit=hard_limit, expect_preliminary=expect_preliminary)
        setattr(self, cache_attr, (sig, val))
        return val

    def _get_traces(self):
        trace_path = self.state_dir / 'anon_turns.jsonl'
        sig = self._sig(trace_path)
        cached_sig, by_game, latest_game = self._trace_cache
        if sig == cached_sig:
            return by_game, latest_game
        rows = read_trace_rows(trace_path)
        grouped = {}
        latest = None
        for r in rows:
            g = r['game']
            grouped.setdefault(g, []).append(r)
            latest = g
        self._trace_cache = (sig, grouped, latest)
        return grouped, latest

    def summary(self):
        with self._lock:
            is_default_100 = self.state_dir == PREV_STATE_DIR.absolute()
            report = self._get_report(self.state_dir, self.hard_limit, '_ledger_cache',
                                      expect_preliminary=is_default_100)
            prev = None
            if self.prev_dir is not None and self.prev_dir != self.state_dir:
                prev = self._get_report(self.prev_dir, self.prev_hard_limit, '_prev_cache',
                                        expect_preliminary=self.prev_dir == PREV_STATE_DIR.absolute())
            by_game, latest_trace_game = self._get_traces()
            traced_games = sorted(by_game.keys())

        def slim_games(rep, has_trace_set):
            if rep is None:
                return []
            out = []
            for g in rep['games']:
                if not g['counted']:
                    continue
                r = g.get('rating') or {}
                out.append(dict(
                    game=g['game'],
                    klass=g['klass'],
                    outcome=g['outcome'],
                    stop_reason=g['stop_reason'],
                    first_mover=g['first_mover'],
                    decisions=g['decisions'],
                    updates=g['updates'],
                    duration_s=g['duration_s'],
                    gf_final=g['gf_final'],
                    started_at=g['started_at'],
                    ended_at=g['ended_at'],
                    self_before=r.get('self_before', {}).get('value'),
                    self_after=r.get('self_after', {}).get('value'),
                    self_delta=r.get('self_delta'),
                    opp_before=r.get('opp_before', {}).get('value'),
                    opp_after=r.get('opp_after', {}).get('value'),
                    opp_delta=r.get('opp_delta'),
                    has_trace=g['game'] in has_trace_set,
                ))
            return out

        current = None
        if report is not None:
            current = dict(
                name=self.state_dir.name,
                cap=report['cap'],
                counts=report['counts'],
                win_rate=report['win_rate'],
                rating_self=report['rating']['self'],
                rating_opponents=report['rating']['opponents'],
                series=report['rating']['series'],
                mismatches=len(report['rating']['mismatches']),
                warnings=report['warnings'],
                breakdowns=compute_breakdowns(report),
                games=slim_games(report, set(traced_games)),
            )
        previous = None
        if prev is not None:
            previous = dict(
                name=self.prev_dir.name,
                cap=prev['cap'],
                counts=prev['counts'],
                win_rate=prev['win_rate'],
                rating_self=prev['rating']['self'],
                rating_opponents=prev['rating']['opponents'],
                series=prev['rating']['series'],
                breakdowns=compute_breakdowns(prev),
            )
        return dict(
            server_time=int(time.time()),
            current=current,
            previous=previous,
            traced_games=traced_games,
            latest_trace_game=latest_trace_game,
            screen_state=read_screen_state(self.state_dir / 'screen-state.json'),
            assets=self.asset_inventory(),
        )

    def live(self):
        with self._lock:
            is_default_100 = self.state_dir == PREV_STATE_DIR.absolute()
            report = self._get_report(self.state_dir, self.hard_limit, '_ledger_cache',
                                      expect_preliminary=is_default_100)
            by_game, latest_trace_game = self._get_traces()
            screen = read_screen_state(self.state_dir / 'screen-state.json')
            active_meta = None
            target_game = latest_trace_game
            if report is not None:
                counted = [g for g in report['games'] if g['counted']]
                if counted:
                    last = counted[-1]
                    active_meta = last
                    if last['game'] in by_game:
                        target_game = last['game']
            rows = list(by_game.get(target_game, ())) if target_game is not None else []
        return self._format_game_payload(target_game, rows, active_meta, screen=screen, is_live=True)

    def game_replay(self, game_num):
        with self._lock:
            is_default_100 = self.state_dir == PREV_STATE_DIR.absolute()
            report = self._get_report(self.state_dir, self.hard_limit, '_ledger_cache',
                                      expect_preliminary=is_default_100)
            by_game, _ = self._get_traces()
            rows = list(by_game.get(game_num, ()))
            meta = None
            if report is not None:
                meta = next((g for g in report['games'] if g['counted'] and g['game'] == game_num), None)
        if not rows and meta is None:
            return None
        return self._format_game_payload(game_num, rows, meta, screen=None, is_live=False)

    @staticmethod
    def _format_game_payload(game_num, rows, meta, *, screen, is_live):
        meta_slim = None
        if meta is not None:
            r = meta.get('rating') or {}
            meta_slim = dict(
                game=meta['game'],
                klass=meta['klass'],
                outcome=meta['outcome'],
                stop_reason=meta['stop_reason'],
                first_mover=meta['first_mover'],
                decisions=meta['decisions'],
                updates=meta['updates'],
                duration_s=meta['duration_s'],
                started_at=meta['started_at'],
                ended_at=meta['ended_at'],
                self_before=r.get('self_before', {}).get('value'),
                self_after=r.get('self_after', {}).get('value'),
                self_delta=r.get('self_delta'),
                opp_before=r.get('opp_before', {}).get('value'),
                opp_after=r.get('opp_after', {}).get('value'),
                opp_delta=r.get('opp_delta'),
            )
        return dict(
            server_time=int(time.time()),
            game=game_num,
            is_live=is_live,
            meta=meta_slim,
            screen_state=screen,
            rows=rows,
        )


def dispatch_request(store: ViewerStore, method: str, raw_path: str, *, catalog_bytes: bytes | None = None):
    """Pure read-only router returning ``(status_code, content_type, body_bytes, cache_control)``."""
    def _json_resp(code, obj):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        return code, 'application/json; charset=utf-8', body, 'no-store'

    if method != 'GET':
        return _json_resp(405, dict(error='read_only'))
    path = urlparse(raw_path).path
    if path in STATIC_FILES:
        fname, ctype = STATIC_FILES[path]
        body = store.static_bytes(fname)
        if body is None:
            return _json_resp(404, dict(error='static_not_found'))
        return 200, ctype, body, 'no-store'
    if path.startswith('/assets/'):
        rel = path[len('/assets/'):]
        body = store.asset_bytes(rel)
        if body is None:
            return _json_resp(404, dict(error='asset_not_found'))
        return 200, 'image/webp', body, 'public, max-age=86400'
    if path == '/api/catalog':
        body = (catalog_bytes if catalog_bytes is not None
                else json.dumps(store.catalog, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        return 200, 'application/json; charset=utf-8', body, 'public, max-age=3600'
    if path == '/api/summary':
        return _json_resp(200, store.summary())
    if path == '/api/live':
        return _json_resp(200, store.live())
    if path.startswith('/api/game/'):
        tail = path[len('/api/game/'):]
        if tail.isdigit():
            payload = store.game_replay(int(tail))
            if payload is not None:
                return _json_resp(200, payload)
        return _json_resp(404, dict(error='game_not_found'))
    return _json_resp(404, dict(error='not_found'))


def make_handler(store: ViewerStore):
    catalog_bytes = json.dumps(store.catalog, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return  # keep server output silent (never log IPs/URLs/headers)

        def _handle(self, method):
            code, content_type, body, cache_ctrl = dispatch_request(
                store, method, self.path, catalog_bytes=catalog_bytes,
            )
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', cache_ctrl)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._handle('GET')

        def do_POST(self):
            self._handle('POST')

        do_PUT = do_DELETE = do_PATCH = do_POST

    return Handler


def build_parser():
    p = argparse.ArgumentParser(description='Read-only live spectator & stats dashboard for ranked runs')
    p.add_argument('--state-dir', type=Path, default=DEFAULT_STATE_DIR)
    p.add_argument('--hard-limit', type=int, default=MAX_CAMPAIGN_LIMIT)
    p.add_argument('--prev-dir', type=Path, default=PREV_STATE_DIR)
    p.add_argument('--no-prev', action='store_true')
    p.add_argument('--assets-dir', type=Path, default=DEFAULT_ASSETS_DIR)
    p.add_argument('--host', default='0.0.0.0', help='Bind host (0.0.0.0 for LAN/Tailscale, 127.0.0.1 for local only)')
    p.add_argument('--port', type=int, default=8765)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    prev = None if args.no_prev else args.prev_dir
    store = ViewerStore(args.state_dir, hard_limit=args.hard_limit, prev_dir=prev, assets_dir=args.assets_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(json.dumps(dict(status='listening', host=args.host, port=server.server_address[1],
                          state_dir=args.state_dir.name,
                          assets_cached=store.asset_inventory()['items_cached']), sort_keys=True), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
