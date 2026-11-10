"""Fail-closed visible UI adapter for the observed 1440x1000 training layout.

Only public state and probe-visible geometry enter this module. Names are used
transiently to locate the own pane and never returned in plans or diagnostics.
"""
from collections import Counter
import json
import re
import time

from .official import ROOT


class UIUnsupported(Exception):
    """Fixed, non-sensitive stop code."""


def require(condition, code):
    if not condition:
        raise UIUnsupported(code)


def item_paths():
    cards = json.loads((ROOT / 'third_party/goodfield/data/official-v2/cards.json').read_text())
    return {c['id']: f"/images/items/{c['category']}/{c['imageName']}.webp" for c in cards
            if re.fullmatch('[a-z-]+', c['category']) and re.fullmatch('[a-z0-9-]+', c['imageName'])}


PATHS = item_paths()


def point(x, y):
    return dict(kind='clickPoint', x=x, y=y)


def hand_images(state):
    return sorted([r for r in state['images'] if r['path'].startswith('/images/items/')
                   and r['w'] == r['h'] == 80 and 190 <= r['x'] <= 846
                   and r['y'] in (493, 595)], key=lambda r: (r['y'], r['x']))


def pane_x(state, name):
    # Top names, below the header and above the first selected card. The body
    # also contains header/status names; those must never identify the pane.
    bands = [r for r in state.get('texts', []) if r['text'] == name
             and 55 <= r['y'] <= 95 and 195 <= r['x'] < 840]
    sides = {350 if r['x'] + r['w']/2 < 520 else 690 for r in bands}
    require(len(bands) == len(sides) == 1, 'own_name_band_not_unique')
    return sides.pop()


def verify_target(state, public, action, name, x):
    """Two-player target shown by the top name bands after card selection.

    A foreign band across from the own band is the opponent at the arrow's
    destination. A lone own band can mean no target or self. Do not click a
    target to repair a mismatch: unfamiliar layouts must be recorded first.
    """
    bands = [r for r in state.get('texts', []) if 55 <= r['y'] <= 95
             and 195 <= r['x'] < 840
             and min(abs(r['x'] + r['w']/2 - c) for c in (350, 690)) <= 140]
    own = [r for r in bands if r['text'] == name]
    foreign = [r for r in bands if r['text'] != name]
    require(len(own) == 1 and len(foreign) <= 1, 'target_name_bands_ambiguous')
    if foreign:
        r = foreign[0]
        side = 350 if r['x'] + r['w']/2 < 520 else 690
        require(side == 1040-x, 'target_name_bands_ambiguous')
        other = next(p['id'] for p in public['players'] if p['id'] != public['viewer'])
        require(action['target'] == other, 'visible_target_mismatch')
    else:
        require(action['target'] in (None, public['viewer']), 'visible_target_mismatch')


def selected(state, x):
    return Counter(r['path'] for r in state['images'] if r['path'].startswith('/images/items/')
                   and r['w'] == r['h'] == 80 and r['x'] == x-145 and 103 <= r['y'] <= 403)


def preview(state):
    matches = re.findall(r'HP\s*(\d+)\s*MP\s*(\d+)\s*[¥￥]\s*(\d+)', state['body'])
    require('+10' in state['body'] and len(matches) == 3, 'exchange_preview_missing')
    return dict(zip(('hp', 'mp', 'cp'), map(int, matches[-1])))


def verify_resources(state, public):
    matches = re.findall(r'HP\s*(\d+)\s*MP\s*(\d+)\s*[¥￥]\s*(\d+)', state['body'])
    actual = sorted(tuple(map(int, m)) for m in matches[:2])
    expected = sorted(tuple(p['status'][k] for k in ('hp', 'mp', 'cp')) for p in public['players'])
    require(actual == expected, 'visible_resources_mismatch')


def verify_scene(state, public):
    require(re.findall(r'G\.F\.\s*(\d+)\b', state['body']) == [str(public['gf'])],
            'visible_gf_changed')
    # Only the witnessed opening banner is unambiguous. Damage/reflection
    # phrases also occur in card descriptions (including the top attack pane).
    # Other transitions must settle through GF, panes, hand and resources.
    body = re.sub(r'\s+', '', state['body'])
    require('今始まる' not in body,
            'visible_animation')


def ready(state, public, name):
    """Visible prerequisites for converting a fresh received decision."""
    verify_scene(state, public)
    x = pane_x(state, name)
    require(not selected(state, x), 'pane_already_selected')
    expected = Counter(PATHS.get(h['face']) for h in public['hand'])
    require(Counter(r['path'] for r in hand_images(state)) == expected, 'visible_hand_mismatch')
    verify_resources(state, public)
    if public['phase'] == 1:
        expected = Counter(PATHS.get(m) for m in public['pending']['models'])
        require(selected(state, 1040-x) == expected, 'visible_attack_mismatch')


def wait_visible(probe, state, check, *, timeout=30, refresh=None):
    """Re-read visible state without retrying any mutation; retain stop code."""
    deadline = time.monotonic() + timeout
    while True:
        if refresh is not None:
            refresh()
        try:
            check(state)
            return state
        except UIUnsupported:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
        time.sleep(min(.5, remaining))
        state = probe.send(dict(kind='state'))['state']


def allocation_steps(start, end):
    require(set(end) == {'hp', 'mp', 'cp'} and all(type(v) is int and 0 <= v <= 99 for v in end.values())
            and sum(start.values()) == sum(end.values()), 'invalid_allocation')
    # Free HP first so no intermediate click exceeds the available total.
    current = dict(start); steps = []
    for sign in (-1, 1):
        for key, x in [('mp', 620), ('cp', 760)]:
            delta = end[key] - current[key]
            if delta * sign <= 0:
                continue
            tens, ones = divmod(abs(delta), 10)
            for magnitude, count in ((10, tens), (1, ones)):
                for _ in range(count):
                    current[key] += sign * magnitude; current['hp'] -= sign * magnitude
                    require(all(0 <= v <= 99 for v in current.values()), 'allocation_intermediate_out_of_range')
                    y = {10: 123, 1: 183, -1: 303, -10: 363}[sign*magnitude]
                    steps.append((point(x, y), dict(current)))
    require(current == end, 'allocation_mismatch')
    return steps


def prepare(state, public, action, name):
    """Validate whole semantic command before the first UI mutation."""
    require((action['expected_update'], action['expected_gf'], action['viewer']) ==
            (public['update'], public['gf'], public['viewer']), 'stale_advice')
    require(public['actor'] == public['viewer'] and len(public['players']) == 2
            and not public['is_over'], 'not_own_decision')
    require(not public['unsupported'] and all(p['status'] is not None and
            not p.get('fogged', False) and 'fog' not in p['status']['curses']
            and not p['status']['confused'] for p in public['players']),
            'unobserved_screen_effect')
    kind = action['kind']
    require(kind in ('play', 'defend', 'exchange', 'sell', 'buy', 'pray'), 'unobserved_ui_kind')
    require((kind == 'defend') == (public['phase'] == 1), 'ui_phase_mismatch')
    other = next(p['id'] for p in public['players'] if p['id'] != public['viewer'])
    require(action['target'] in (None, other, public['viewer']), 'unknown_target')
    if kind == 'exchange':
        require(action['target'] in (None, public['viewer']), 'exchange_target_mismatch')
    elif kind == 'defend':
        require(action['target'] is None, 'unobserved_response_target')
    elif kind == 'pray':
        require(action['target'] in (None, public['viewer']), 'prayer_target_mismatch')
    if kind in ('sell', 'buy', 'pray'):
        require(len(action['cards']) == {'sell': 2, 'buy': 1, 'pray': 0}[kind], 'invalid_kind_cards')
    if kind in ('sell', 'buy'):
        require(action['cards'][0]['model'] == {'sell': 4, 'buy': 5}[kind], 'invalid_trade_card')
    require(len(action['cards']) <= 4, 'selected_pane_capacity_unverified')
    verify_resources(state, public)
    expected_ops = [dict(op='kind', kind=kind)]
    expected_ops += [dict(op='select_card', **c) for c in action['cards']]
    expected_ops += [dict(op='finish_cards'), dict(op='target', player_id=action['target'])]
    if action['allocation']:
        expected_ops.append(dict(op='allocate', **action['allocation']))
    expected_ops.append(dict(op='confirm'))
    require(action['operations'] == expected_ops, 'unknown_operations')
    x = pane_x(state, name)
    require(not selected(state, x), 'pane_already_selected')
    own = {h['id']: h for h in public['hand']}
    ids = [c['item_id'] for c in action['cards']]
    require(len(ids) == len(set(ids)), 'duplicate_selected_id')
    available = hand_images(state); clicks = []; expected = Counter(); stages = []
    for card in action['cards']:
        h = own.get(card['item_id'])
        require(h is not None and h['face'] == card['model'], 'card_not_in_public_hand')
        require(not h['used'] or PATHS.get(h['face'], '').startswith('/images/items/miracles/'),
                'card_not_in_public_hand')
        require(not any(i['disguised'] for i in public['hand'] if i['face'] == h['face']), 'ambiguous_disguised_face')
        path = PATHS.get(card['model'])
        matches = [r for r in available if r['path'] == path]
        require(path and matches, 'card_image_missing')
        copies = [i for i in public['hand'] if i['face'] == h['face']]
        all_matches = [r for r in hand_images(state) if r['path'] == path]
        require(len(all_matches) == len(copies), 'visible_hand_mismatch')
        if {i['used'] for i in copies} == {False, True}:
            # auto6 and live1 after reuse: the single used miracle occupies
            # the bottom-right slot; unused overflow starts at bottom-left.
            # Do not extrapolate this to multiple used miracles or a layout
            # with missing copies. The glow is not a separate probe image.
            require(sum(i['used'] for i in public['hand']) == 1, 'ambiguous_used_card_position')
            used = [r for r in all_matches if (r['x'], r['y']) == (846, 595)]
            require(len(used) == 1, 'ambiguous_used_card_position')
            matches = [r for r in matches if (r == used[0]) == h['used']]
            require(matches, 'ambiguous_used_card_position')
        r = matches[0]; available.remove(r)
        clicks.append(point(r['x']+40, r['y']+40))
        expected[path] += 1; stages.append(dict(expected))
    allocation = []
    if kind == 'exchange':
        require([c['model'] for c in action['cards']] == [3] and x == 350, 'unobserved_exchange_layout')
        start = next(p['status'] for p in public['players'] if p['id'] == public['viewer'])
        allocation = allocation_steps({k: start[k] for k in ('hp', 'mp', 'cp')}, action['allocation'])
    else:
        require(action['allocation'] is None, 'unexpected_allocation')
        require(action['cards'] or kind in ('defend', 'pray'), 'unobserved_prayer')
    confirm = point(x, 250 if kind == 'exchange' else 193)
    return dict(cards=clicks, stages=stages, allocation=allocation, confirm=confirm, pane=x,
                commands=clicks+[c for c, _ in allocation]+[confirm])


def verify(state, public, name, plan, expected):
    require(pane_x(state, name) == plan['pane'], 'own_pane_changed')
    require(selected(state, plan['pane']) == Counter(expected), 'selected_cards_mismatch')
    verify_resources(state, public)
    verify_scene(state, public)


def execute(probe, state, public, action, name, *, commands=None, settle_timeout=30, refresh=None):
    """Send one click at a time and verify its resulting visible state.

    Caller owns capture/update acceptance and saving a screenshot on failure.
    commands is populated before dispatch, including a failed attempted click.
    """
    plan = prepare(state, public, action, name)
    commands = commands if commands is not None else []
    expected = {}
    def settled(state, expected, values=None, initial=False, final=False):
        def check(s):
            verify(s, public, name, plan, expected)
            if final and action['kind'] in ('play', 'sell', 'buy', 'pray'):
                verify_target(s, public, action, name, plan['pane'])
            if values is not None:
                require(preview(s) == values,
                        'exchange_initial_mismatch' if initial else 'exchange_value_mismatch')
        return wait_visible(probe, state, check, timeout=settle_timeout, refresh=refresh)

    for command, expected in zip(plan['cards'], plan['stages']):
        if refresh is not None:
            refresh()
        commands.append(command); state = probe.send(command)['state']
        state = settled(state, expected)
    if plan['allocation']:
        start = next(p['status'] for p in public['players'] if p['id'] == public['viewer'])
        state = settled(state, expected, {k: start[k] for k in ('hp', 'mp', 'cp')}, initial=True)
        for command, values in plan['allocation']:
            if refresh is not None:
                refresh()
            commands.append(command); state = probe.send(command)['state']
            state = settled(state, expected, values)
    state = settled(state, expected, final=True)
    if action['kind'] == 'exchange':
        require(preview(state) == action['allocation'], 'exchange_final_mismatch')
    commands.append(plan['confirm']); probe.send(plan['confirm'])
    return plan
