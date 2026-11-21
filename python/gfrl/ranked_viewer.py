"""Read-only live spectator & campaign dashboard server for official ranked runs.

Reads ``ledger.jsonl``, ``anon_turns.jsonl`` and ``screen-state.json`` without locks or
writes, so it never interferes with a running campaign. Everything served is strictly
anonymous (numeric stats, public card catalog names/IDs and fixed enums).
"""
from __future__ import annotations

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

from .ranked_campaign import DEFAULT_SUPPLEMENT, DEFAULT_SUPPLEMENT_GAME, ROOT, report_from_path
from .ranked_ledger import HARD_LIMIT, MAX_CAMPAIGN_LIMIT
from .ranked_trace import validate_trace_row

CARDS_PATH = ROOT / 'third_party/goodfield/data/official-v2/cards.json'
DEFAULT_STATE_DIR = ROOT / 'runs/ranked1000'
PREV_STATE_DIR = ROOT / 'runs/ranked100'


def load_catalog(path=CARDS_PATH):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    out = {}
    for c in raw:
        out[str(c['id'])] = dict(
            id=c['id'],
            name=c.get('name', f"#{c['id']}"),
            category=c.get('category', ''),
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
    """Thread-safe, mtime/size-cached read-only view of campaign ledger and trace files."""
    def __init__(self, state_dir, *, hard_limit=MAX_CAMPAIGN_LIMIT, prev_dir=None, prev_hard_limit=HARD_LIMIT,
                 catalog_path=CARDS_PATH):
        self.state_dir = Path(state_dir).absolute()
        self.hard_limit = hard_limit
        self.prev_dir = Path(prev_dir).absolute() if prev_dir else None
        self.prev_hard_limit = prev_hard_limit
        self.catalog = load_catalog(catalog_path)
        self._lock = threading.Lock()
        self._ledger_cache = (None, None)
        self._prev_cache = (None, None)
        self._trace_cache = (None, {}, None)

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


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>God Field RL — Live Spectator & Stats</title>
<style>
  :root {
    --bg: #0f131a; --panel: #171d27; --panel2: #1e2634; --border: #2c3648;
    --text: #e6edf7; --muted: #8c9bb3; --accent: #58a6ff; --win: #3fb950;
    --loss: #f85149; --draw: #d29922; --hp: #f05a5a; --mp: #4fa3f7; --cp: #e6b84c;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Noto Sans JP", sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.45; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
           gap: 12px; padding: 12px 20px; background: var(--panel); border-bottom: 1px solid var(--border);
           position: sticky; top: 0; z-index: 10; }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 16px; }
  .badge { font-size: 12px; font-weight: 700; padding: 3px 9px; border-radius: 999px; letter-spacing: .03em; }
  .badge.live { background: rgba(63,185,80,.18); color: #56d364; border: 1px solid rgba(63,185,80,.45); }
  .badge.wait { background: rgba(210,153,34,.18); color: #e3b341; border: 1px solid rgba(210,153,34,.45); }
  .badge.done { background: rgba(88,166,255,.18); color: #79c0ff; border: 1px solid rgba(88,166,255,.45); }
  .badge.replay { background: rgba(188,140,255,.2); color: #d2a8ff; border: 1px solid rgba(188,140,255,.5); }
  .kpi-strip { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; }
  .kpi { display: flex; flex-direction: column; }
  .kpi .lbl { font-size: 11px; color: var(--muted); }
  .kpi .val { font-size: 17px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .tabs { display: flex; gap: 6px; align-items: center; }
  button { background: var(--panel2); color: var(--text); border: 1px solid var(--border);
           border-radius: 8px; padding: 6px 12px; cursor: pointer; font-size: 13px; font-weight: 600; }
  button:hover { border-color: var(--accent); }
  button.active { background: var(--accent); color: #0b1118; border-color: var(--accent); }
  main { max-width: 1320px; margin: 0 auto; padding: 16px; }
  .grid-live { display: grid; grid-template-columns: 1.55fr 1fr; gap: 16px; }
  @media (max-width: 960px) { .grid-live { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
  .card h3 { margin: 0 0 10px; font-size: 14px; color: var(--muted); font-weight: 600;
             display: flex; justify-content: space-between; align-items: center; }
  .seat { background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
  .seat.acting { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(88,166,255,.35); }
  .seat-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 8px; flex-wrap: wrap; }
  .seat-title { font-weight: 700; font-size: 15px; display: flex; align-items: center; gap: 8px; }
  .bars { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 8px; }
  .bar-box { background: #121720; border-radius: 8px; padding: 6px 8px; }
  .bar-top { display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; margin-bottom: 4px; }
  .bar-track { height: 7px; background: #262f40; border-radius: 999px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 999px; transition: width .2s ease; }
  .chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
  .chip { font-size: 11px; padding: 2px 7px; border-radius: 6px; background: #252f42; color: var(--text); border: 1px solid #34415a; }
  .chip.curse { background: rgba(248,81,73,.16); color: #ff8e88; border-color: rgba(248,81,73,.4); }
  .chip.guard { background: rgba(210,153,34,.18); color: #f2cc60; border-color: rgba(210,153,34,.45); }
  .chip.used { background: rgba(140,155,179,.12); color: #b7c5da; }
  .arena-mid { margin: 12px 0; padding: 12px; border-radius: 10px; background: #121720;
               border: 1px dashed var(--border); min-height: 76px; display: flex; flex-direction: column;
               justify-content: center; gap: 8px; }
  .hand-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(126px, 1fr)); gap: 7px; margin-top: 10px; }
  .item-card { background: #131923; border: 1px solid #2d384c; border-radius: 8px; padding: 7px 8px;
               font-size: 12px; display: flex; flex-direction: column; gap: 3px; position: relative; }
  .item-card.cat-weapons { border-left: 3px solid #f06560; }
  .item-card.cat-armor { border-left: 3px solid #58a6ff; }
  .item-card.cat-miracles { border-left: 3px solid #bc8cff; }
  .item-card.cat-sundries { border-left: 3px solid #3fb950; }
  .item-card.cat-trade { border-left: 3px solid #e6b84c; }
  .item-card.cat-devils { border-left: 3px solid #d946ef; }
  .item-card.used { opacity: .65; border-style: dashed; }
  .item-name { font-weight: 700; font-size: 12.5px; }
  .item-meta { font-size: 11px; color: var(--muted); }
  .pill { display: inline-block; font-size: 10px; padding: 1px 5px; border-radius: 4px; font-weight: 700; margin-left: 4px; }
  .pill.dream { background: rgba(188,140,255,.22); color: #d2a8ff; }
  .pill.used { background: rgba(140,155,179,.22); color: #c5d1e3; }
  .pill.win { background: rgba(63,185,80,.2); color: #56d364; }
  .pill.loss { background: rgba(248,81,73,.2); color: #ff7b72; }
  .pill.draw { background: rgba(210,153,34,.2); color: #e3b341; }
  .pill.unk { background: rgba(140,155,179,.2); color: #a5b3c9; }
  .log-list { max-height: 360px; overflow-y: auto; display: flex; flex-direction: column; gap: 5px; font-size: 12.5px; }
  .log-row { padding: 6px 9px; border-radius: 7px; background: var(--panel2); border-left: 3px solid var(--border); }
  .log-row.self { border-left-color: var(--accent); }
  .log-row.opp { border-left-color: #f0883e; }
  .log-row.ai { border-left-color: #bc8cff; background: rgba(188,140,255,.08); }
  .replay-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 12px;
                padding: 8px 12px; background: var(--panel2); border-radius: 10px; border: 1px solid var(--border); }
  .replay-bar input[type=range] { flex: 1; min-width: 160px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
  th, td { padding: 7px 9px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 12px; }
  tr.clickable:hover { background: var(--panel2); cursor: pointer; }
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
  @media (max-width: 900px) { .stats-grid { grid-template-columns: 1fr; } }
  .svg-wrap { width: 100%; overflow: hidden; }
  svg { width: 100%; height: auto; display: block; }
  .hidden { display: none !important; }
  .muted { color: var(--muted); }
  .pos { color: var(--win); font-weight: 700; }
  .neg { color: var(--loss); font-weight: 700; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>God Field RL</span>
    <span id="statusBadge" class="badge wait">読込中</span>
    <span id="campaignName" class="muted" style="font-size:12px;"></span>
  </div>
  <div class="kpi-strip">
    <div class="kpi"><span class="lbl">進捗</span><span class="val" id="kpiProgress">-</span></div>
    <div class="kpi"><span class="lbl">現在レート</span><span class="val" id="kpiRating">-</span></div>
    <div class="kpi"><span class="lbl">最高レート</span><span class="val" id="kpiMaxRating">-</span></div>
    <div class="kpi"><span class="lbl">完走勝率</span><span class="val" id="kpiWinRate">-</span></div>
    <div class="kpi"><span class="lbl">相手平均</span><span class="val" id="kpiOppMean">-</span></div>
  </div>
  <div class="tabs">
    <button id="tabLive" class="active" onclick="switchTab('live')">リアルタイム観戦</button>
    <button id="tabStats" onclick="switchTab('stats')">戦績・レート推移</button>
    <button id="btnAuto" class="active" onclick="toggleAuto()">自動更新: ON</button>
  </div>
</header>

<main>
  <!-- LIVE / REPLAY VIEW -->
  <section id="viewLive">
    <div id="replayBanner" class="replay-bar hidden">
      <span id="replayLabel" class="badge replay">リプレイ</span>
      <button onclick="stepReplay(-1)">◀ 1手戻る</button>
      <input id="replaySlider" type="range" min="0" max="0" value="0" oninput="onSlider(this.value)">
      <button onclick="stepReplay(1)">1手進む ▶</button>
      <span id="replayPos" class="muted" style="font-size:12px;">0 / 0</span>
      <button onclick="exitReplay()" style="border-color:var(--win);color:#56d364;">● ライブに戻る</button>
    </div>

    <div class="grid-live">
      <div class="card">
        <h3>
          <span id="boardTitle">対局盤面</span>
          <span id="boardMeta" class="muted" style="font-weight:400;font-size:12px;"></span>
        </h3>

        <!-- Opponent Seat -->
        <div id="seatOpp" class="seat">
          <div class="seat-head">
            <div class="seat-title"><span>相手（人間）</span><span id="oppRating" class="muted" style="font-size:12px;"></span></div>
            <div id="oppBadges" class="chips"></div>
          </div>
          <div class="bars" id="oppBars"></div>
          <div id="oppUsed" class="chips"></div>
        </div>

        <!-- Center Arena -->
        <div class="arena-mid" id="arenaMid">
          <div class="muted" style="text-align:center;">対局データを待っています…</div>
        </div>

        <!-- Self (AI) Seat -->
        <div id="seatSelf" class="seat">
          <div class="seat-head">
            <div class="seat-title"><span>AI（m-xit1 greedy）</span><span id="selfRating" class="muted" style="font-size:12px;"></span></div>
            <div id="selfBadges" class="chips"></div>
          </div>
          <div class="bars" id="selfBars"></div>
          <div id="selfUsed" class="chips"></div>
          <div style="margin-top:10px;font-size:12px;color:var(--muted);display:flex;justify-content:space-between;">
            <span>AI の手札（公開絵柄）</span><span id="handCount">0 枚</span>
          </div>
          <div class="hand-grid" id="handGrid"></div>
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:16px;">
        <div class="card">
          <h3><span>AI 勝率予測（m-xit1 value）</span><span id="valNow" style="color:var(--text);">—</span></h3>
          <div class="svg-wrap" id="valueChart"></div>
          <div id="lastDecisionBox" style="margin-top:10px;font-size:12.5px;color:var(--muted);"></div>
        </div>
        <div class="card" style="flex:1;">
          <h3><span>ターン経過ログ</span><span id="logCount" class="muted" style="font-weight:400;"></span></h3>
          <div class="log-list" id="logList"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- STATS VIEW -->
  <section id="viewStats" class="hidden">
    <div class="card">
      <h3>
        <span>レート推移グラフ</span>
        <label style="font-size:12px;cursor:pointer;">
          <input type="checkbox" id="chkIncludePrev" checked onchange="renderStats()"> 初回100局（ranked100）も連結表示
        </label>
      </h3>
      <div class="svg-wrap" id="ratingChart"></div>
    </div>

    <div class="stats-grid">
      <div class="card">
        <h3><span>相手レート帯別 完走成績</span><span id="statsTotals" class="muted" style="font-weight:400;"></span></h3>
        <table>
          <thead><tr><th>相手開始レート</th><th>局数</th><th>勝</th><th>敗</th><th>分</th><th>勝率</th></tr></thead>
          <tbody id="tblOppRating"></tbody>
        </table>
      </div>
      <div class="card">
        <h3><span>先手/後手・停止理由の内訳</span></h3>
        <table>
          <thead><tr><th>区分</th><th>局数</th><th>勝</th><th>敗</th><th>分</th><th>勝率</th></tr></thead>
          <tbody id="tblMover"></tbody>
        </table>
        <div style="margin-top:14px;">
          <div class="muted" style="font-size:12px;margin-bottom:6px;">途中停止・決着不明の内訳</div>
          <div id="stopChips" class="chips"></div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <h3><span>対局履歴（クリックで棋譜リプレイ）</span><span id="historyHint" class="muted" style="font-weight:400;"></span></h3>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr><th>局</th><th>結果</th><th>自分レート</th><th>相手レート</th><th>先手</th><th>判断数</th><th>所要</th><th>備考</th></tr></thead>
          <tbody id="tblGames"></tbody>
        </table>
      </div>
    </div>
  </section>
</main>

<script>
const GUARDIANS = {none:'なし',mars:'火星神',mercury:'水星神',jupiter:'木星神',saturn:'土星神',uranus:'天王神',pluto:'冥王神',neptune:'海王神',venus:'金星神',earth:'地球神',moon:'月神'};
const CURSES = {cold:'風邪',fever:'熱病',hell:'地獄病',heaven:'天国病',fog:'霧',dream:'夢',flash:'閃光',darkcloud:'暗雲'};
const ELEMS = {fire:'火',water:'水',wood:'木',stone:'土',light:'光',darkness:'闇'};
const KINDS = {play:'使用',defend:'防御',discard:'捨てる',pray:'祈る',exchange:'両替',sell:'売る',buy:'買う',accept:'買う(承諾)',decline:'買わない',sacrifice:'ささげる'};
const PHASES = {0:'主行動',1:'防御応答',2:'終局',3:'購入応答',4:'配布'};

let catalog = {};
let summaryData = null;
let liveData = null;
let replayData = null;
let replayIdx = 0;
let currentTab = 'live';
let autoRefresh = true;

function cardInfo(id) { return catalog[String(id)] || {id, name: '#' + id, category: '', element: '', atk: 0, defense: 0, cost: 0, price: 0}; }
function cardName(id) { return cardInfo(id).name; }
function fmtDelta(d) { if (d === null || d === undefined) return ''; return d > 0 ? `<span class="pos">+${d}</span>` : d < 0 ? `<span class="neg">${d}</span>` : `<span class="muted">±0</span>`; }
function fmtPct(r) { return r === null || r === undefined ? '—' : (r * 100).toFixed(1) + '%'; }
function fmtDur(s) { if (s === null || s === undefined) return '—'; const m = Math.floor(s/60); return m ? `${m}分${s%60}秒` : `${s}秒`; }
function outcomePill(g) {
  if (g.klass === 'in_progress') return '<span class="pill" style="background:rgba(88,166,255,.2);color:#79c0ff;">対局中</span>';
  if (g.outcome === 'win') return '<span class="pill win">勝ち</span>';
  if (g.outcome === 'loss') return '<span class="pill loss">負け</span>';
  if (g.outcome === 'draw') return '<span class="pill draw">引分</span>';
  return '<span class="pill unk">停止</span>';
}

function switchTab(t) {
  currentTab = t;
  document.getElementById('viewLive').classList.toggle('hidden', t !== 'live');
  document.getElementById('viewStats').classList.toggle('hidden', t !== 'stats');
  document.getElementById('tabLive').classList.toggle('active', t === 'live');
  document.getElementById('tabStats').classList.toggle('active', t === 'stats');
  if (t === 'stats') renderStats();
}
function toggleAuto() {
  autoRefresh = !autoRefresh;
  const b = document.getElementById('btnAuto');
  b.textContent = '自動更新: ' + (autoRefresh ? 'ON' : 'OFF');
  b.classList.toggle('active', autoRefresh);
}

function renderBars(p) {
  if (!p || p.hp === null) {
    return '<div class="bar-box" style="grid-column:1/-1;text-align:center;color:var(--muted);">霧でステータス非公開</div>';
  }
  const specs = [['HP', p.hp, 'var(--hp)'], ['MP', p.mp, 'var(--mp)'], ['¥', p.cp, 'var(--cp)']];
  return specs.map(([lbl, val, col]) => {
    const pct = Math.max(0, Math.min(100, (val / 99) * 100));
    return `<div class="bar-box"><div class="bar-top"><span>${lbl}</span><span>${val}</span></div><div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${col};"></div></div></div>`;
  }).join('');
}

function renderSeatBadges(p) {
  if (!p) return '';
  const out = [];
  if (p.dead) out.push('<span class="chip curse">昇天</span>');
  if (p.fogged) out.push('<span class="chip">霧隠れ</span>');
  if (p.confused) out.push('<span class="chip curse">混乱</span>');
  if (p.guardian && p.guardian !== 'none') out.push(`<span class="chip guard">守護神: ${GUARDIANS[p.guardian] || p.guardian}</span>`);
  for (const c of (p.curses || [])) out.push(`<span class="chip curse">${CURSES[c] || c}</span>`);
  return out.join('');
}

function renderUsed(p) {
  if (!p || !p.used || !p.used.length) return '';
  return '<span class="muted" style="font-size:11px;align-self:center;">使用済奇跡:</span>' +
    p.used.map(id => `<span class="chip used">${cardName(id)}</span>`).join('');
}

function renderHand(hand) {
  document.getElementById('handCount').textContent = (hand ? hand.length : 0) + ' 枚';
  if (!hand || !hand.length) return '<div class="muted" style="font-size:12px;">手札なし</div>';
  return hand.map(([id, dis, used]) => {
    const c = cardInfo(id);
    const meta = [];
    if (c.element) meta.push(ELEMS[c.element] || c.element);
    if (c.atk) meta.push(`攻${c.plus ? '+' : ''}${c.atk}`);
    if (c.defense) meta.push(`防${c.defense}`);
    if (c.cost) meta.push(`MP${c.cost}`);
    if (c.price) meta.push(`¥${c.price}`);
    return `<div class="item-card cat-${c.category} ${used ? 'used' : ''}">
      <div class="item-name">${c.name}${dis ? '<span class="pill dream">夢</span>' : ''}${used ? '<span class="pill used">済</span>' : ''}</div>
      <div class="item-meta">${meta.join(' / ') || c.category}</div>
    </div>`;
  }).join('');
}

function describeEvent(e) {
  const who = v => v === true ? 'AI' : v === false ? '相手' : '';
  const a = e.action;
  if (a === 'startGame') return '対局開始';
  if (a === 'endGame') return '対局終了';
  if (a === 'advanceGF') return `G.F.進行 → ${who(e.player_self)}のターン`;
  if (a === 'useAttackItems') return `攻撃/使用: ${(e.faces||[]).map(cardName).join(' + ') || '祈る'}`;
  if (a === 'useDefenseItems') return (e.faces && e.faces.length) ? `防御: ${e.faces.map(cardName).join(' + ')}` : '防御なし（受ける）';
  if (a === 'dealDamage') return `${who(e.player_self)}に ${e.damage} ダメージ`;
  if (a === 'dealDarkDamage') return `闇ダメージ（HP 0）`;
  if (a === 'boostHP') return `${who(e.player_self)} HP +${e.hp}`;
  if (a === 'boostMP') return `${who(e.player_self)} MP +${e.mp}`;
  if (a === 'boostCP') return `CP +${e.cp}`;
  if (a === 'takeCP') return `CP ${e.cp} 奪取`;
  if (a === 'exchange') return `両替 → HP${e.hp} / MP${e.mp} / ¥${e.cp}`;
  if (a === 'buy') return `購入提示: ${cardName(e.model)}`;
  if (a === 'setBought') return e.bought ? `購入成立 (${cardName(e.model)})` : `購入拒否 (${cardName(e.model)})`;
  if (a === 'sell') return `${who(e.seller_self)}→${who(e.buyer_self)} 売却${e.face ? ': ' + cardName(e.face) : ''}`;
  if (a === 'reflect') return '反射！';
  if (a === 'bounce') return '乱反射！';
  if (a === 'block') return '防御成功';
  if (a === 'safe') return '回避/無効';
  if (a === 'miss') return `ミス (${who(e.player_self)})`;
  if (a === 'hit') return `命中 (${who(e.player_self)})`;
  if (a === 'addCurse') return `状態付与: ${CURSES[e.curse] || e.curse}`;
  if (a === 'removeCurses') return `治療: ${(e.curses||[]).map(c=>CURSES[c]||c).join(',')}`;
  if (a === 'setGuardian') return `守護神召喚: ${GUARDIANS[e.guardian] || e.guardian}`;
  if (a === 'attackByGuardian') return `守護神攻撃 (${GUARDIANS[e.attack?.guardian]||''}): ${(e.attack?.models||[]).map(cardName).join('+')}`;
  if (a === 'removeGuardian') return `${who(e.player_self)}の守護神離脱`;
  if (a === 'revive') return `${who(e.player_self)} 復活 (${cardName(e.model)})`;
  if (a === 'die') return `${who(e.player_self)} 昇天`;
  if (a === 'pray') return '祈る';
  if (a === 'redraw') return '手札引き直し';
  if (a === 'removeItems' || a === 'removeUsedMiracles') return `除去: ${(e.models||[]).map(cardName).join(', ')}`;
  if (a === 'gift') return `補充: ${(e.faces||[]).map(cardName).join(', ')}`;
  return a;
}

function renderValueChart(decisions) {
  if (!decisions.length) {
    document.getElementById('valNow').textContent = '—';
    document.getElementById('valueChart').innerHTML = '<div class="muted" style="font-size:12px;padding:24px 0;text-align:center;">まだ決定データがありません</div>';
    return;
  }
  const last = decisions[decisions.length - 1];
  const winP = last.value_probs ? (last.value_probs[0] * 100) : ((last.value + 1) * 50);
  document.getElementById('valNow').innerHTML = `勝率予測 <b>${winP.toFixed(1)}%</b> <span class="muted">(value ${last.value >= 0 ? '+' : ''}${last.value.toFixed(3)})</span>`;

  const W = 440, H = 135, pad = {l: 34, r: 10, t: 10, b: 22};
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const pts = decisions.map((d, i) => {
    const x = pad.l + (decisions.length === 1 ? iw / 2 : (i / (decisions.length - 1)) * iw);
    const p = d.value_probs ? d.value_probs[0] : (d.value + 1) / 2;
    const y = pad.t + (1 - p) * ih;
    return [x, y, p];
  });
  const poly = pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ');
  const midY = (pad.t + ih / 2).toFixed(1);
  const lastPt = pts[pts.length - 1];
  document.getElementById('valueChart').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}">
      <line x1="${pad.l}" y1="${pad.t}" x2="${W-pad.r}" y2="${pad.t}" stroke="#263042" stroke-width="1"/>
      <line x1="${pad.l}" y1="${midY}" x2="${W-pad.r}" y2="${midY}" stroke="#394863" stroke-dasharray="4 3" stroke-width="1"/>
      <line x1="${pad.l}" y1="${pad.t+ih}" x2="${W-pad.r}" y2="${pad.t+ih}" stroke="#263042" stroke-width="1"/>
      <text x="4" y="${pad.t+9}" fill="#8c9bb3" font-size="10">100%</text>
      <text x="9" y="${Number(midY)+3}" fill="#8c9bb3" font-size="10">50%</text>
      <text x="14" y="${pad.t+ih}" fill="#8c9bb3" font-size="10">0%</text>
      <polyline fill="none" stroke="#58a6ff" stroke-width="2.2" points="${poly}"/>
      <circle cx="${lastPt[0].toFixed(1)}" cy="${lastPt[1].toFixed(1)}" r="4" fill="#56d364"/>
      <text x="${W-pad.r}" y="${H-4}" text-anchor="end" fill="#8c9bb3" font-size="10">決定 #${decisions.length}</text>
    </svg>`;
}

function renderBoardFromPayload(data, limitIdx = null) {
  const rows = data && data.rows ? data.rows : [];
  const slice = limitIdx === null ? rows : rows.slice(0, limitIdx + 1);
  const updates = slice.filter(r => r.ev === 'update');
  const decisions = slice.filter(r => r.ev === 'decision');
  const latestUp = updates.length ? updates[updates.length - 1] : null;
  const latestDec = decisions.length ? decisions[decisions.length - 1] : null;
  const meta = data ? data.meta : null;

  const gameLabel = data && data.game ? `Game #${data.game}` : '待機中';
  document.getElementById('boardTitle').textContent = gameLabel + (latestUp ? ` — G.F. ${latestUp.gf} (${PHASES[latestUp.phase] || latestUp.phase})` : '');
  const metaParts = [];
  if (meta) {
    if (meta.klass === 'in_progress') metaParts.push('対局中');
    else if (meta.outcome) metaParts.push('結果: ' + ({win:'勝ち',loss:'負け',draw:'引分',unknown:'停止'}[meta.outcome] || meta.outcome));
    if (meta.stop_reason) metaParts.push(meta.stop_reason);
  }
  if (latestUp) metaParts.push(`update ${latestUp.update}`);
  document.getElementById('boardMeta').textContent = metaParts.join(' / ');

  document.getElementById('selfRating').innerHTML = meta && meta.self_before ? `レート ${meta.self_before}${meta.self_after ? ' → ' + meta.self_after : ''} ${fmtDelta(meta.self_delta)}` : '';
  document.getElementById('oppRating').innerHTML = meta && meta.opp_before ? `レート ${meta.opp_before}${meta.opp_after ? ' → ' + meta.opp_after : ''} ${fmtDelta(meta.opp_delta)}` : '';

  if (latestUp) {
    document.getElementById('seatOpp').classList.toggle('acting', latestUp.actor_self === false && !latestUp.is_over);
    document.getElementById('seatSelf').classList.toggle('acting', latestUp.actor_self === true && !latestUp.is_over);
    document.getElementById('oppBars').innerHTML = renderBars(latestUp.opp);
    document.getElementById('oppBadges').innerHTML = renderSeatBadges(latestUp.opp);
    document.getElementById('oppUsed').innerHTML = renderUsed(latestUp.opp);
    document.getElementById('selfBars').innerHTML = renderBars(latestUp.self);
    document.getElementById('selfBadges').innerHTML = renderSeatBadges(latestUp.self);
    document.getElementById('selfUsed').innerHTML = renderUsed(latestUp.self);
    document.getElementById('handGrid').innerHTML = renderHand(latestUp.hand);

    const mid = [];
    if (latestUp.is_over) {
      const res = latestUp.self.dead === latestUp.opp.dead ? '引き分け' : latestUp.self.dead ? 'AI 敗北' : 'AI 勝利！';
      mid.push(`<div style="text-align:center;font-size:18px;font-weight:800;">🏁 終局 — ${res}</div>`);
    } else if (latestUp.pending) {
      const p = latestUp.pending;
      const src = p.source_self === true ? 'AI' : p.source_self === false ? '相手' : '環境';
      const dst = p.target_self ? 'AI' : '相手';
      const cards = (p.models || []).map(cardName).join(' + ');
      const extra = [];
      if (p.guardian && p.guardian !== 'none') extra.push(GUARDIANS[p.guardian]);
      if (p.reflections) extra.push(`反射×${p.reflections}`);
      if (p.bounces) extra.push(`乱反射×${p.bounces}`);
      if (p.base_override !== null && p.base_override !== undefined) extra.push(`基本値 ${p.base_override}`);
      if (p.hit_rate) extra.push(`命中 ${p.hit_rate}%`);
      mid.push(`<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
        <div><b>⚔️ 保留攻撃</b>: ${src} → ${dst} — <span style="color:#ff9b85;font-weight:700;">${cards}</span></div>
        <div class="muted" style="font-size:12px;">${extra.join(' / ')}</div>
      </div>`);
    } else if (latestUp.purchase) {
      const q = latestUp.purchase;
      mid.push(`<div><b>💰 購入交渉</b>: ${q.buyer_self ? 'AI' : '相手'} が <b>${cardName(q.model)}</b> を買うか判断中</div>`);
    } else {
      mid.push(`<div class="muted" style="text-align:center;">${latestUp.actor_self ? 'AI の思考・操作中…' : '相手の行動待ち…'}</div>`);
    }
    if (latestUp.events && latestUp.events.length) {
      mid.push(`<div style="font-size:12px;color:#b8c7dc;">直近イベント: ${latestUp.events.map(describeEvent).join(' → ')}</div>`);
    }
    document.getElementById('arenaMid').innerHTML = mid.join('');
  } else {
    document.getElementById('arenaMid').innerHTML = '<div class="muted" style="text-align:center;">次の対局の開始・マッチングを待っています…</div>';
  }

  renderValueChart(decisions);
  if (latestDec) {
    const a = latestDec.action;
    const cards = (a.models || []).map(cardName).join(' + ');
    const probs = (latestDec.stage_probs || []).map(p => (p * 100).toFixed(1) + '%').join(' / ');
    document.getElementById('lastDecisionBox').innerHTML =
      `直前の AI 選択 (#${latestDec.decision_idx}): <b style="color:var(--text);">${KINDS[a.kind] || a.kind}${cards ? ' [' + cards + ']' : ''}</b>` +
      (a.allocation ? ` (HP${a.allocation.hp}/MP${a.allocation.mp}/¥${a.allocation.cp})` : '') +
      (probs ? `<br>確信度: ${probs}` : '');
  } else {
    document.getElementById('lastDecisionBox').textContent = '';
  }

  // Timeline
  const logItems = [];
  for (const r of slice.slice(-50)) {
    if (r.ev === 'decision') {
      const a = r.action;
      const cards = (a.models || []).map(cardName).join(' + ');
      const wp = r.value_probs ? (r.value_probs[0] * 100).toFixed(0) + '%' : '';
      logItems.push(`<div class="log-row ai"><b>AI決定 #${r.decision_idx}</b> (GF${r.gf}): ${KINDS[a.kind]||a.kind}${cards ? ' ['+cards+']' : ''} ${wp ? `<span class="muted">勝率${wp}</span>` : ''}</div>`);
    } else if (r.ev === 'update' && r.events && r.events.length) {
      const cls = r.owner_self ? 'self' : 'opp';
      logItems.push(`<div class="log-row ${cls}"><span class="muted">GF${r.gf} u${r.update}</span> ${r.events.map(describeEvent).join(' → ')}</div>`);
    }
  }
  document.getElementById('logCount').textContent = `${updates.length} 更新 / ${decisions.length} 決定`;
  document.getElementById('logList').innerHTML = logItems.reverse().join('') || '<div class="muted">ログなし</div>';
}

function renderHeader() {
  if (!summaryData || !summaryData.current) return;
  const c = summaryData.current;
  document.getElementById('campaignName').textContent = c.name;
  document.getElementById('kpiProgress').textContent = `${c.cap.counted} / ${c.cap.hard_limit}`;
  const lastR = c.rating_self.last_after ?? c.rating_self.first_before;
  document.getElementById('kpiRating').textContent = lastR ?? '—';
  document.getElementById('kpiMaxRating').textContent = c.rating_self.max_after ?? '—';
  const wc = c.win_rate.completed;
  document.getElementById('kpiWinRate').textContent = wc.n ? `${(wc.rate*100).toFixed(1)}% (${c.counts.wins}勝${c.counts.losses}敗${c.counts.draws ? c.counts.draws+'分' : ''})` : '—';
  document.getElementById('kpiOppMean').textContent = c.rating_opponents.mean_before ? Math.round(c.rating_opponents.mean_before) : '—';

  const badge = document.getElementById('statusBadge');
  if (replayData) {
    badge.className = 'badge replay'; badge.textContent = `REPLAY Game #${replayData.game}`;
  } else if (c.counts.in_progress > 0) {
    badge.className = 'badge live'; badge.textContent = '● LIVE 対局中';
  } else if (c.cap.remaining === 0) {
    badge.className = 'badge done'; badge.textContent = '完了';
  } else {
    badge.className = 'badge wait'; badge.textContent = 'マッチング/待機中';
  }
}

function renderStats() {
  if (!summaryData || !summaryData.current) return;
  const c = summaryData.current;
  const incPrev = document.getElementById('chkIncludePrev').checked && summaryData.previous;
  const ptsRaw = [];
  if (incPrev && summaryData.previous.series.length) {
    ptsRaw.push({label: '開始', v: summaryData.previous.rating_self.first_before});
    for (const s of summaryData.previous.series) if (s.self_after !== null) ptsRaw.push({label: '旧#'+s.game, v: s.self_after, prev: true});
  } else if (c.rating_self.first_before !== null) {
    ptsRaw.push({label: '開始', v: c.rating_self.first_before});
  }
  const splitIdx = ptsRaw.length - 1;
  for (const s of c.series) if (s.self_after !== null) ptsRaw.push({label: '#'+s.game, v: s.self_after, prev: false});

  if (ptsRaw.length >= 2) {
    const W = 900, H = 220, pad = {l: 48, r: 16, t: 16, b: 26};
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
    const vals = ptsRaw.map(p => p.v);
    const minV = Math.floor((Math.min(...vals) - 20) / 50) * 50;
    const maxV = Math.ceil((Math.max(...vals) + 20) / 50) * 50;
    const sx = i => pad.l + (i / (ptsRaw.length - 1)) * iw;
    const sy = v => pad.t + ((maxV - v) / Math.max(1, maxV - minV)) * ih;
    let grids = '';
    for (let v = minV; v <= maxV; v += 50) {
      const y = sy(v).toFixed(1);
      grids += `<line x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}" stroke="${v%100===0?'#313e54':'#202838'}" stroke-width="1"/>`;
      if (v % 100 === 0 || (maxV - minV) <= 250) grids += `<text x="6" y="${Number(y)+4}" fill="#8c9bb3" font-size="11">${v}</text>`;
    }
    const allPts = ptsRaw.map((p, i) => `${sx(i).toFixed(1)},${sy(p.v).toFixed(1)}`).join(' ');
    const splitLine = incPrev && splitIdx > 0 ? `<line x1="${sx(splitIdx).toFixed(1)}" y1="${pad.t}" x2="${sx(splitIdx).toFixed(1)}" y2="${pad.t+ih}" stroke="#e3b341" stroke-dasharray="4 3"/><text x="${sx(splitIdx)+5}" y="${pad.t+13}" fill="#e3b341" font-size="11">ここから ranked1000</text>` : '';
    const last = ptsRaw[ptsRaw.length - 1];
    document.getElementById('ratingChart').innerHTML = `
      <svg viewBox="0 0 ${W} ${H}">
        ${grids}${splitLine}
        <polyline fill="none" stroke="#58a6ff" stroke-width="2.4" points="${allPts}"/>
        <circle cx="${sx(ptsRaw.length-1).toFixed(1)}" cy="${sy(last.v).toFixed(1)}" r="4.5" fill="#56d364"/>
        <text x="${W-pad.r}" y="${H-6}" text-anchor="end" fill="#8c9bb3" font-size="11">全 ${ptsRaw.length - 1} 局 / 現在 ${last.v}</text>
      </svg>`;
  }

  const b = c.breakdowns;
  document.getElementById('statsTotals').textContent = `完走 ${c.counts.completed} / ${c.counts.counted} 局`;
  document.getElementById('tblOppRating').innerHTML = b.by_opp_rating.filter(x => x.n > 0).map(x =>
    `<tr><td>${x.label}</td><td>${x.n}</td><td>${x.wins}</td><td>${x.losses}</td><td>${x.draws}</td><td><b>${fmtPct(x.rate)}</b></td></tr>`
  ).join('') || '<tr><td colspan="6" class="muted">データなし</td></tr>';

  document.getElementById('tblMover').innerHTML = [['self','先手（AI）'],['opponent','後手（相手先手）']].map(([k, lbl]) => {
    const x = b.by_first_mover[k];
    return `<tr><td>${lbl}</td><td>${x.n}</td><td>${x.wins}</td><td>${x.losses}</td><td>${x.draws}</td><td><b>${fmtPct(x.rate)}</b></td></tr>`;
  }).join('');

  const stops = Object.entries(b.stop_reasons);
  document.getElementById('stopChips').innerHTML = stops.length
    ? stops.map(([k, v]) => `<span class="chip curse">${k}: ${v}件</span>`).join('')
    : '<span class="chip" style="color:#56d364;">停止なし (0件)</span>';

  document.getElementById('historyHint').textContent = `全 ${c.games.length} 局（新しい順）`;
  document.getElementById('tblGames').innerHTML = [...c.games].reverse().map(g => {
    const cls = g.has_trace ? 'clickable' : '';
    const click = g.has_trace ? `onclick="openReplay(${g.game})"` : '';
    const note = g.stop_reason ? g.stop_reason : (g.has_trace ? '棋譜あり' : '');
    return `<tr class="${cls}" ${click}>
      <td>#${g.game}</td><td>${outcomePill(g)}</td>
      <td>${g.self_before ?? '—'} → ${g.self_after ?? '—'} ${fmtDelta(g.self_delta)}</td>
      <td>${g.opp_before ?? '—'} → ${g.opp_after ?? '—'} ${fmtDelta(g.opp_delta)}</td>
      <td>${g.first_mover === 'self' ? '先手' : g.first_mover === 'opponent' ? '後手' : '—'}</td>
      <td>${g.decisions ?? '—'}</td><td>${fmtDur(g.duration_s)}</td>
      <td class="muted">${note}</td>
    </tr>`;
  }).join('');
}

async function openReplay(gameNum) {
  const res = await fetch('/api/game/' + gameNum);
  if (!res.ok) return;
  replayData = await res.json();
  replayIdx = Math.max(0, (replayData.rows ? replayData.rows.length : 1) - 1);
  const slider = document.getElementById('replaySlider');
  slider.max = Math.max(0, (replayData.rows ? replayData.rows.length : 1) - 1);
  slider.value = replayIdx;
  document.getElementById('replayBanner').classList.remove('hidden');
  document.getElementById('replayLabel').textContent = `REPLAY Game #${gameNum}`;
  updateReplayView();
  switchTab('live');
}

function updateReplayView() {
  if (!replayData) return;
  const n = replayData.rows ? replayData.rows.length : 0;
  document.getElementById('replaySlider').value = replayIdx;
  document.getElementById('replayPos').textContent = `${n ? replayIdx + 1 : 0} / ${n}`;
  renderHeader();
  renderBoardFromPayload(replayData, replayIdx);
}
function onSlider(v) { replayIdx = Number(v); updateReplayView(); }
function stepReplay(delta) {
  if (!replayData) return;
  const max = Math.max(0, (replayData.rows ? replayData.rows.length : 1) - 1);
  replayIdx = Math.max(0, Math.min(max, replayIdx + delta));
  updateReplayView();
}
function exitReplay() {
  replayData = null;
  document.getElementById('replayBanner').classList.add('hidden');
  renderHeader();
  if (liveData) renderBoardFromPayload(liveData);
}

async function poll() {
  try {
    const [sRes, lRes] = await Promise.all([fetch('/api/summary'), fetch('/api/live')]);
    if (sRes.ok) summaryData = await sRes.json();
    if (lRes.ok) liveData = await lRes.json();
    renderHeader();
    if (!replayData) renderBoardFromPayload(liveData);
    if (currentTab === 'stats') renderStats();
  } catch (e) {}
}

async function boot() {
  try { catalog = await (await fetch('/api/catalog')).json(); } catch (e) {}
  await poll();
  setInterval(() => { if (autoRefresh) poll(); }, 1500);
}
boot();
</script>
</body>
</html>
"""


def dispatch_request(store: ViewerStore, method: str, raw_path: str, *,
                     catalog_bytes: bytes | None = None, html_bytes: bytes | None = None):
    """Pure read-only router returning ``(status_code, content_type, body_bytes)``."""
    def _json_resp(code, obj):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        return code, 'application/json; charset=utf-8', body

    if method != 'GET':
        return _json_resp(405, dict(error='read_only'))
    path = urlparse(raw_path).path
    if path in ('/', '/index.html'):
        body = html_bytes if html_bytes is not None else HTML_PAGE.encode('utf-8')
        return 200, 'text/html; charset=utf-8', body
    if path == '/api/catalog':
        body = (catalog_bytes if catalog_bytes is not None
                else json.dumps(store.catalog, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        return 200, 'application/json; charset=utf-8', body
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
    html_bytes = HTML_PAGE.encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return  # keep server output silent (never log IPs/URLs/headers)

        def _handle(self, method):
            code, content_type, body = dispatch_request(
                store, method, self.path, catalog_bytes=catalog_bytes, html_bytes=html_bytes,
            )
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
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
    p.add_argument('--host', default='0.0.0.0', help='Bind host (0.0.0.0 for LAN/Tailscale, 127.0.0.1 for local only)')
    p.add_argument('--port', type=int, default=8765)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    prev = None if args.no_prev else args.prev_dir
    store = ViewerStore(args.state_dir, hard_limit=args.hard_limit, prev_dir=prev)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(json.dumps(dict(status='listening', host=args.host, port=server.server_address[1],
                          state_dir=args.state_dir.name), sort_keys=True), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
