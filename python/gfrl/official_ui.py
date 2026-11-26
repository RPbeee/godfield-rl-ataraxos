"""Fail-closed visible UI adapter for the observed 1440x1000 training layout.

Only public state and probe-visible geometry enter this module. Names are used
transiently to locate the own pane and never returned in plans or diagnostics.
"""
from collections import Counter
from copy import deepcopy
import json
import re
import time

from .official import ROOT, OBSERVED_UI_KINDS
from . import MatchConfig, _core


class UIUnsupported(Exception):
    """Fixed stop code with separately constructed public diagnostics."""
    def __init__(self, code, detail=None):
        super().__init__(code)
        self.detail = detail or {}


def require(condition, code, detail=None):
    if not condition:
        raise UIUnsupported(code, detail)


def item_paths():
    cards = json.loads((ROOT / 'third_party/goodfield/data/official-v2/cards.json').read_text())
    return {c['id']: f"/images/items/{c['category']}/{c['imageName']}.webp" for c in cards
            if re.fullmatch('[a-z-]+', c['category']) and re.fullmatch('[a-z0-9-]+', c['imageName'])}


PATHS = item_paths()
FAKE_PATH = '/images/items/fake.webp'
PUBLIC_PATHS = frozenset(PATHS.values())
REFLECT_MODELS = frozenset(c['id'] for c in json.loads(
    (ROOT / 'third_party/goodfield/data/official-v2/cards.json').read_text())
    if c.get('ability') in ('reflectWeapon', 'reflectMiracle'))


def band_texts(state):
    # Top name/target bands are semantic, including for the pixel arrow probe.
    # Include the central pane's text below them when checking animation settling.
    return [r for r in state.get('texts', []) if 48 <= r['y'] < 480
            and 195 <= r['x'] < 840]


def transient_band(row):
    # Never classify a top name/target band (or a card description containing
    # these words) as animation. Only standalone reflection/redirect notices.
    return (95 < row['y'] < 480 and 195 <= row['x'] < 840
            and row['text'] in ('反射した', '反射された', '跳ね返した', '跳ね返された'))


def selection_checks(state, before, public, name, plan, expected):
    try:
        verify(state, public, name, plan, expected)
        verified = True
    except UIUnsupported:
        verified = False
    pairs = dict(hand_images=(hand_images(before), hand_images(state)),
                 own_selected_images=(selected_images(before, plan['pane']), selected_images(state, plan['pane'])),
                 other_selected_images=(selected_images(before, 1040-plan['pane']), selected_images(state, 1040-plan['pane'])),
                 band_texts=(band_texts(before), band_texts(state)))
    checks = dict(verify=verified, **{k: a == b for k, (a, b) in pairs.items()},
                  targetArrow=state.get('targetArrow') == before.get('targetArrow'))
    checks['hand_images'] &= hand_fake_overlays(before) == hand_fake_overlays(state)
    counts = {k: dict(before=len(a), after=len(b)) for k, (a, b) in pairs.items()}
    return dict(checks=checks, counts=counts)


def wait_response_stable(probe, state, public, name, plan, expected, *, timeout, refresh=None):
    # wait_visible supplies the bounded interval and checks receives before each
    # comparison. The supplied state is the first read; require one fresh read.
    previous = deepcopy(state)
    first = True
    def check(current):
        nonlocal previous, first
        detail = selection_checks(current, previous, public, name, plan, expected)
        fresh = not first
        first = False
        previous = deepcopy(current)
        require(fresh and all(detail['checks'].values()), 'response_ui_unstable',
                dict(selection_check=detail))
    return wait_visible(probe, state, check, timeout=timeout, refresh=refresh)



def card_detail(expected, actual, clicks=()):
    def counts(values):
        safe = Counter()
        for path, count in values.items():
            safe[path if path in PUBLIC_PATHS else '?'] += count
        return dict(sorted(safe.items()))
    seen = Counter(); attempted = []
    for click in clicks:
        path = click['path']; seen[path] += 1
        attempted.append(dict(path=path, x=click['x'], y=click['y'],
                              appeared=actual[path] >= seen[path]))
    return dict(expected=counts(expected), actual=counts(actual), clicks=attempted)


def band_detail(bands):
    return dict(count=len(bands), x=sorted(r['x'] for r in bands))


def point(x, y):
    return dict(kind='clickPoint', x=x, y=y)


def is_card_face(image):
    # Dream's fake.webp is an overlay on a face, including in selected panes.
    # Keep unknown item paths: they must still fail the exact card comparison.
    return (image['path'].startswith('/images/items/')
            and image['path'] != FAKE_PATH)


def in_hand_region(image):
    x, y, w, h = (image[k] for k in ('x', 'y', 'w', 'h'))
    return w == h == 80 and 180 <= x and x+w <= 930 and 480 <= y and y+h <= 700


def hand_images(state):
    return sorted([r for r in state['images'] if is_card_face(r) and in_hand_region(r)],
                  key=lambda r: (r['y'], r['x']))


def hand_fake_overlays(state):
    return sorted([r for r in state['images'] if r['path'] == FAKE_PATH and in_hand_region(r)],
                  key=lambda r: (r['y'], r['x']))


def pane_x(state, name):
    # Top names, below the header (y+h <= 40) and above the first selected card.
    # Noto CJK on Linux renders at y=53; Hiragino on macOS at y>=55.
    bands = [r for r in state.get('texts', []) if r['text'] == name
             and 48 <= r['y'] <= 95 and 195 <= r['x'] < 840]
    sides = {350 if r['x'] + r['w']/2 < 520 else 690 for r in bands}
    require(len(bands) == len(sides) == 1, 'own_name_band_not_unique', band_detail(bands))
    return sides.pop()


def own_fog(public):
    own = next(p for p in public['players'] if p['id'] == public['viewer'])
    return bool(own['status'] and 'fog' in own['status']['curses'] and not own.get('dead', False))


def verify_target(state, public, action, name, x):
    """Two-player target shown by the top name bands after card selection.

    A destination band across from the acting pane identifies the target.
    With target None, either a lone own band or an opponent destination is valid;
    the server chooses the target and both displays use the same confirmation.
    A self destination must still stop.
    """
    if own_fog(public) and public['phase'] == 0 and action['target'] not in (None, public['viewer']):
        return  # Fog chooses a uniform enemy at launch; no visible arrow is required.
    target = action['target']
    bands = [r for r in state.get('texts', []) if 48 <= r['y'] <= 95
             and 195 <= r['x'] < 840
             and min(abs(r['x'] + r['w']/2 - c) for c in (350, 690)) <= 140]
    expected = 'self' if target == public['viewer'] else 'none' if target is None else 'opponent'
    detail = dict(expected_target=expected, bands=[dict(side='left' if r['x']+r['w']/2 < 520 else 'right',
                  owner='self' if r['text'] == name else 'opponent', x=r['x'], y=r['y'], w=r['w'], h=r['h'])
                  for r in bands])
    detail.update(band_detail(bands))
    acting = [r for r in bands if (350 if r['x']+r['w']/2 < 520 else 690) == x]
    destination = [r for r in bands if r not in acting]
    require(len(acting) == 1 and acting[0]['text'] == name and len(destination) <= 1,
            'target_name_bands_ambiguous', detail)
    if destination:
        actual = public['viewer'] if destination[0]['text'] == name else next(
            p['id'] for p in public['players'] if p['id'] != public['viewer'])
        require(target == actual or (target is None and len(public['players']) == 2
                                     and actual != public['viewer']), 'visible_target_mismatch', detail)
    else:
        require(target in (None, public['viewer']), 'visible_target_mismatch', detail)


def target_point(state, public, action, name):
    own_y = own_status_row(state, name)
    y = own_y if action['target'] == public['viewer'] else 395-own_y
    rows = [r for r in status_names(state) if y <= r['y'] < y+40]
    require(len(rows) == 1 and (rows[0]['text'] == name) == (action['target'] == public['viewer']),
            'target_switch_failed', dict(stage='status_band', target_row=y))
    r = rows[0]
    return point(r['x']+r['w']/2, r['y']+r['h']/2)


def verify_switched_target(state, public, action, name, x):
    verify_target(state, public, action, name, x)
    bands = [r for r in state.get('texts', []) if 48 <= r['y'] <= 95 and 195 <= r['x'] < 840]
    require(len(bands) == 2 and state.get('targetArrow') == ('right' if x == 350 else 'left'),
            'target_switch_failed', dict(stage='arrow_and_bands', **band_detail(bands)))


def selected_images(state, x):
    return [r for r in state['images'] if is_card_face(r)
            and r['w'] == r['h'] == 80 and x-145 <= r['x']
            and r['x']+r['w'] <= x+145 and 0 <= r['y'] < 480]


def selected(state, x):
    return Counter(r['path'] for r in selected_images(state, x))


def confirmation_point(state, x, preferred):
    images = selected_images(state, x)
    # Preserve the witnessed point when clear; otherwise search the pane's
    # inner edges. No assumption about selected-card rows or spacing is needed.
    candidates = [preferred] + [point(cx, y) for cx in (x-140, x+140)
                                 for y in range(103, 480)]
    for p in candidates:
        if not any(r['x'] <= p['x'] <= r['x']+r['w'] and
                   r['y'] <= p['y'] <= r['y']+r['h'] for r in images):
            return p
    raise UIUnsupported('confirmation_point_unavailable')


def preview(state, public=None):
    matches = re.findall(r'HP\s*(\d+)\s*MP\s*(\d+)\s*[¥￥]\s*(\d+)', state['body'])
    groups = 2 if public is not None and own_fog(public) else 3
    require('+10' in state['body'] and len(matches) == groups, 'exchange_preview_missing',
            dict(expected=dict(resource_groups=groups, plus10=True),
                 actual=dict(resources=[list(map(int, m)) for m in matches], plus10='+10' in state['body'])))
    return dict(zip(('hp', 'mp', 'cp'), map(int, matches[-1])))


def status_names(state):
    return [r for r in state.get('texts', []) if 860 <= r['x'] < 1025
            and r['w'] > 0 and r['h'] > 0
            and any(y <= r['y'] and r['y']+r['h'] <= y+40 for y in (131, 264))]


def own_status_row(state, name):
    rows = [r for r in status_names(state) if r['text'] == name]
    require(len(rows) == 1, 'own_status_band_not_unique', band_detail(rows))
    return 131 if rows[0]['y'] < 171 else 264


def verify_resources(state, public, name=None):
    matches = re.findall(r'HP\s*(\d+)\s*MP\s*(\d+)\s*[¥￥]\s*(\d+)', state['body'])
    actual = sorted(tuple(map(int, m)) for m in matches[:2])
    if own_fog(public):
        if name is None:
            # Direct diagnostic callers can identify the sole visible status
            # name in fog; execution always supplies the known own name.
            rows = status_names(state)
            require(len(rows) == 1, 'own_status_band_not_unique', band_detail(rows))
            name = rows[0]['text']
        y = 264 if own_status_row(state, name) == 131 else 131
        require(any(abs(r['x']-860)<=1 and abs(r['y']-y)<=1 and
                    abs(r['w']-340)<=1 and abs(r['h']-40)<=1 for r in state.get('fogBands', [])),
                'fog_status_band_missing', dict(expected_y=y))
        require(not any(860 <= r['x'] < 1200 and y <= r['y'] < y+40
                        for r in state.get('texts', [])), 'fog_status_band_not_empty')
        require(len(matches) == 1 or ('+10' in state['body'] and len(matches) == 2),
                'visible_resources_mismatch', dict(resource_groups=len(matches)))
        actual = sorted(tuple(map(int, m)) for m in matches[:1])
    expected = sorted(tuple(p['status'][k] for k in ('hp', 'mp', 'cp'))
                      for p in public['players'] if p['status'] is not None)
    require(actual == expected, 'visible_resources_mismatch', dict(expected=expected, actual=actual))


def verify_scene(state, public):
    require(re.findall(r'G\.F\.\s*(\d+)\b', state['body']) == [str(public['gf'])],
            'visible_gf_changed')
    # Only the witnessed opening banner is unambiguous. Damage/reflection
    # phrases also occur in card descriptions (including the top attack pane).
    # Other transitions must settle through GF, panes, hand and resources.
    body = re.sub(r'\s+', '', state['body'])
    require('今始まる' not in body,
            'visible_animation')


def purchase_button(state, public, name, kind):
    require(public['phase'] == 3 and pane_x(state, name) == 350, 'purchase_layout_mismatch')
    require(public.get('purchase') and public['purchase']['buyer'] == public['viewer'],
            'purchase_buyer_mismatch')
    buttons = {}
    for text, top in [('買う', 100), ('買わない', 300)]:
        rows = [r for r in state.get('texts', []) if r['text'] == text
                and 205 <= r['x'] and r['x']+r['w'] <= 495
                and top <= r['y'] and r['y']+r['h'] <= top+85]
        require(len(rows) == 1, 'purchase_buttons_mismatch')
        r = rows[0]
        require(r['w'] > 0 and r['h'] >= 40 and abs(r['x']+r['w']/2-350) <= 5,
                'purchase_buttons_mismatch')
        buttons[text] = point(r['x']+r['w']/2, r['y']+r['h']/2)
    offer = [r['path'] for r in state['images'] if 540 <= r['x'] and r['x']+r['w'] <= 840
             and 198 <= r['y'] and r['y']+r['h'] <= 288 and r['w'] == r['h'] == 80]
    require(offer == [PATHS.get(public['purchase']['model'])], 'purchase_offer_mismatch')
    return buttons['買う' if kind == 'accept' else '買わない']


def verify_attack(state, public, x):
    expected = Counter(PATHS.get(m) for m in public['pending']['models'])
    require(selected(state, 1040-x) == expected, 'visible_attack_mismatch',
            card_detail(expected, selected(state, 1040-x)))


def ready(state, public, name):
    """Visible prerequisites for converting a fresh received decision."""
    verify_scene(state, public)
    try:
        x = pane_x(state, name)
    except UIUnsupported as error:
        if (str(error) == 'own_name_band_not_unique' and error.detail.get('count') == 0
                and own_fog(public) and public['phase'] == 1):
            raise UIUnsupported('fog_response_ui_unobserved',
                                dict(phase='response', own_name_bands=0)) from None
        raise
    require(not selected(state, x), 'pane_already_selected')
    verify_command_control(state, x, public['phase'])
    expected = Counter(PATHS.get(h['face']) for h in public['hand'])
    actual = Counter(r['path'] for r in hand_images(state))
    require(actual == expected, 'visible_hand_mismatch', card_detail(expected, actual))
    verify_resources(state, public, name)
    if public['phase'] == 1:
        verify_attack(state, public, x)


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
    require(set(start) == set(end) == {'hp', 'mp', 'cp'}
            and all(type(v) is int and 0 <= v <= 99 for v in (*start.values(), *end.values()))
            and sum(start.values()) == sum(end.values()), 'invalid_allocation')
    # HP is the transfer buffer. Interleave withdrawals and deposits when it
    # reaches either bound; draining both resources first can overflow HP99.
    current = dict(start); steps = []
    while current != end:
        progressed = False
        for sign in (-1, 1):
            for key, x in [('mp', 620), ('cp', 760)]:
                delta = (end[key] - current[key]) * sign
                capacity = current['hp'] if sign > 0 else 99-current['hp']
                if delta <= 0 or capacity <= 0:
                    continue
                magnitude = 10 if min(delta, capacity) >= 10 else 1
                current[key] += sign * magnitude; current['hp'] -= sign * magnitude
                y = {10: 123, 1: 183, -1: 303, -10: 363}[sign*magnitude]
                steps.append((point(x, y), dict(current)))
                progressed = True
        require(progressed, 'allocation_intermediate_out_of_range')
    require(current == end, 'allocation_mismatch', dict(expected=end, actual=current))
    return steps


def verify_command_control(state, x, phase):
    # New live probes report the visible, hit-testable command overlay. Old
    # offline snapshots lack this capability and keep their original checks.
    if phase != 3 and 'commandPanes' in state:
        require(state['commandPanes'] == [x], 'command_control_unavailable')


def used_positions(state, public):
    """Validate the complete used suffix before distinguishing mixed copies.

    Pinned client fH.hr sorts used faces by model, then fH.fD right-aligns them
    in the 18-slot grid. Repeated used faces still lack an item-ID witness.
    """
    used = sorted(h['face'] for h in public['hand'] if h['used'])
    images = hand_images(state); result = []
    for slot, face in zip(range(18-len(used), 18), used):
        matches = [r for r in images if (r['x'], r['y']) ==
                   (190+82*(slot % 9), 493 if slot < 9 else 595)]
        require(len(matches) == 1 and matches[0]['path'] == PATHS.get(face),
                'ambiguous_used_card_position')
        result.append(matches[0])
    return result


def disguised_positions(overlays, faces):
    """Match overlays one-to-one and reject misaligned overlays over these slots."""
    positions = set()
    for overlay in overlays:
        nearby = [(r['x'], r['y']) for r in faces
                  if abs(overlay['x'] - r['x']) < 80 and abs(overlay['y'] - r['y']) < 80]
        if not nearby:
            continue
        x, y = nearby[0]
        require(len(nearby) == 1 and abs(overlay['x'] - x) <= 1
                and abs(overlay['y'] - y) <= 1 and (x, y) not in positions,
                'disguised_overlay_mismatch')
        positions.add((x, y))
    return positions


def prepare(state, public, action, name, *, allow_dream=False, dream_last_resort=False):
    """Validate whole semantic command before the first UI mutation."""
    require((action['expected_update'], action['expected_gf'], action['viewer']) ==
            (public['update'], public['gf'], public['viewer']), 'stale_advice')
    require(public['actor'] == public['viewer'] and len(public['players']) == 2
            and not public['is_over'], 'not_own_decision')
    require(not public['unsupported'] and all(
            (p['status'] is not None and not (p['id'] == public['viewer']
                and public['phase'] == 0 and p['status']['confused'])) or
            (own_fog(public) and p['id'] != public['viewer'] and p.get('fogged', False))
            for p in public['players']),
            'unobserved_screen_effect')
    kind = action['kind']
    require(kind in OBSERVED_UI_KINDS, 'unobserved_ui_kind', dict(kind=kind))
    require((kind in ('accept', 'decline')) == (public['phase'] == 3), 'ui_phase_mismatch')
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
    require(len(action['cards']) <= MatchConfig().hand_limit, 'selected_cards_over_hand_limit')
    verify_resources(state, public, name)
    if own_fog(public) and public['phase'] == 1:
        x = pane_x(state, name)
        verify_attack(state, public, x)
    expected_ops = [dict(op='kind', kind=kind)]
    expected_ops += [dict(op='select_card', **c) for c in action['cards']]
    expected_ops += [dict(op='finish_cards'), dict(op='target', player_id=action['target'])]
    if action['allocation']:
        expected_ops.append(dict(op='allocate', **action['allocation']))
    expected_ops.append(dict(op='confirm'))
    require(action['operations'] == expected_ops, 'unknown_operations')
    x = pane_x(state, name)
    require(not selected(state, x), 'pane_already_selected')
    verify_command_control(state, x, public['phase'])
    if kind in ('accept', 'decline'):
        require(not action['cards'] and action['target'] is None and action['allocation'] is None,
                'invalid_purchase_action')
        confirm = purchase_button(state, public, name, kind)
        return dict(cards=[], stages=[], attempted=[], card_paths=[], allocation=[],
                    confirm=confirm, pane=x, commands=[confirm])
    own = {h['id']: h for h in public['hand']}
    ids = [c['item_id'] for c in action['cards']]
    require(len(ids) == len(set(ids)), 'duplicate_selected_id')
    images = hand_images(state); available = list(images)
    overlays = hand_fake_overlays(state)
    clicks = []; expected = Counter(); stages = []
    for card in action['cards']:
        h = own.get(card['item_id'])
        require(h is not None and h['face'] == card['model'], 'card_not_in_public_hand')
        require(not h['used'] or PATHS.get(h['face'], '').startswith('/images/items/miracles/'),
                'card_not_in_public_hand')
        require(not h['disguised'] or allow_dream or dream_last_resort,
                'ambiguous_disguised_face')
        path = PATHS.get(card['model'])
        matches = [r for r in available if r['path'] == path]
        require(path and matches, 'card_image_missing')
        copies = [i for i in public['hand'] if i['face'] == h['face']]
        all_matches = [r for r in images if r['path'] == path]
        require(len(all_matches) == len(copies), 'visible_hand_mismatch',
                card_detail(Counter({path: len(copies)}), Counter({path: len(all_matches)})))
        used = []
        if {i['used'] for i in copies} == {False, True}:
            used = [r for r in used_positions(state, public) if r['path'] == path]
            require(len(used) == 1, 'ambiguous_used_card_position')
            matches = [r for r in matches if (r == used[0]) == h['used']]
            require(matches, 'ambiguous_used_card_position')
        disguised = disguised_positions(overlays, all_matches)
        disguised_count = sum(i['disguised'] for i in copies)
        # An entirely disguised face needs no overlay to identify its copies.
        # Mixed faces need an overlay witness for every disguised copy.
        require(len(disguised) == disguised_count or (not disguised and disguised_count == len(copies)),
                'disguised_overlay_mismatch',
                dict(expected=disguised_count, actual=len(disguised)))
        if 0 < disguised_count < len(copies):
            if used:
                used_is_disguised = (used[0]['x'], used[0]['y']) in disguised
                require(used_is_disguised == any(i['used'] and i['disguised'] for i in copies),
                        'disguised_overlay_mismatch')
            matches = [r for r in matches if ((r['x'], r['y']) in disguised) == h['disguised']]
            require(matches, 'disguised_overlay_mismatch')
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
    return dict(cards=clicks, stages=stages, attempted=[], card_paths=[PATHS[c['model']] for c in action['cards']],
                allocation=allocation, confirm=confirm, pane=x,
                commands=clicks+[c for c, _ in allocation]+[confirm])


def verify(state, public, name, plan, expected):
    verify_command_control(state, plan['pane'], public['phase'])
    if own_fog(public) and public['phase'] == 1:
        verify_attack(state, public, plan['pane'])
    if plan.get('target_switched'):
        bands = [r for r in state.get('texts', []) if r['text'] == name
                 and 48 <= r['y'] <= 95 and 195 <= r['x'] < 840
                 and (350 if r['x']+r['w']/2 < 520 else 690) == plan['pane']]
        require(len(bands) == 1, 'own_pane_changed')
    else:
        require(pane_x(state, name) == plan['pane'], 'own_pane_changed')
    require(selected(state, plan['pane']) == Counter(expected), 'selected_cards_mismatch',
            card_detail(Counter(expected), selected(state, plan['pane']), plan.get('attempted', [])))
    verify_resources(state, public, name)
    verify_scene(state, public)


def execute(probe, state, public, action, name, *, commands=None, settle_timeout=30, refresh=None, allow_dream=False, dream_last_resort=False, previous_reflect=False):
    """Send one click at a time and verify its resulting visible state.

    Caller owns capture/update acceptance and saving a screenshot on failure.
    commands is populated before dispatch, including a failed attempted click.
    """
    plan = prepare(state, public, action, name, allow_dream=allow_dream, dream_last_resort=dream_last_resort)
    commands = commands if commands is not None else []
    # Counters (e.g. ring counter-attacks) also arrive right after an animation.
    pending = public.get('pending') or {}
    redirected = public['phase'] == 1 and (previous_reflect or bool(pending.get('counter'))
        or 'redirect_requires_history' in public.get('_policy_only', ())
        or any(pending.get(k, 0) > 0 for k in ('reflections', 'bounces')))
    if redirected or (public['phase'] == 1 and 'commandPanes' in state):
        state = wait_response_stable(probe, state, public, name, plan, {},
                                     timeout=settle_timeout, refresh=refresh)
        # Geometry may have settled into a different hand arrangement.
        ready(state, public, name)
        plan = prepare(state, public, action, name, allow_dream=allow_dream, dream_last_resort=dream_last_resort)
    expected = {}
    def settled(state, expected, values=None, initial=False, final=False):
        def check(s):
            verify(s, public, name, plan, expected)
            if action['kind'] in ('accept', 'decline'):
                purchase_button(s, public, name, action['kind'])
            if final and action['kind'] in ('play', 'sell', 'buy', 'pray'):
                verify_target(s, public, action, name, plan['pane'])
            if values is not None:
                actual = preview(s, public)
                require(actual == values,
                        'exchange_initial_mismatch' if initial else 'exchange_value_mismatch',
                        dict(expected=values, actual=actual))
        return wait_visible(probe, state, check, timeout=settle_timeout, refresh=refresh)

    for index, (command, expected) in enumerate(zip(plan['cards'], plan['stages'])):
        if refresh is not None:
            refresh()
        before = deepcopy(state)
        previous = plan['stages'][index-1] if index else {}
        attempt = dict(path=plan['card_paths'][index], x=command['x'], y=command['y'])
        plan['attempted'].append(attempt)
        commands.append(command); state = probe.send(command)['state']
        try:
            state = settled(state, expected)
        except UIUnsupported as error:
            if str(error) != 'selected_cards_mismatch' or settle_timeout <= 0:
                raise
            state = probe.send(dict(kind='state'))['state']
            detail = selection_checks(state, before, public, name, plan, previous)
            if not all(detail['checks'].values()):
                checks = detail['checks']
                only_bands = all(v for k, v in checks.items() if k != 'band_texts')
                semantic_before = [r for r in band_texts(before) if not transient_band(r)]
                semantic_after = [r for r in band_texts(state) if not transient_band(r)]
                if not (public['phase'] == 1 and only_bands and semantic_before == semantic_after):
                    error.detail['selection_check'] = detail
                    raise error
                state = wait_response_stable(probe, state, public, name, plan, previous,
                                             timeout=settle_timeout, refresh=refresh)
                # Stability alone cannot authorize a different decision.
                detail = selection_checks(state, before, public, name, plan, previous)
                if (not all(v for k, v in detail['checks'].items() if k != 'band_texts')
                        or [r for r in band_texts(state) if not transient_band(r)] != semantic_before):
                    error.detail['selection_check'] = detail
                    raise error
            if refresh is not None:
                refresh()
            plan.setdefault('selection_retries', []).append(index)
            plan['attempted'].append(dict(attempt))
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
    state = settled(state, expected)
    if action['kind'] in ('play', 'sell', 'buy', 'pray'):
        try:
            verify_target(state, public, action, name, plan['pane'])
        except UIUnsupported as error:
            if str(error) != 'visible_target_mismatch' or action['target'] is None:
                raise
            try:
                command = target_point(state, public, action, name)
                if refresh is not None:
                    refresh()
                plan['target_switched'] = True
                plan['commands'].insert(len(plan['cards']), command)
                commands.append(command)
                state = probe.send(command)['state']
                def switched(s):
                    verify(s, public, name, plan, expected)
                    verify_switched_target(s, public, action, name, plan['pane'])
                state = wait_visible(probe, state, switched, timeout=settle_timeout, refresh=refresh)
            except UIUnsupported as failed:
                raise UIUnsupported('target_switch_failed', dict(error.detail, **failed.detail)) from None
    state = settled(state, expected, final=True)
    if action['kind'] == 'exchange':
        actual = preview(state, public)
        require(actual == action['allocation'], 'exchange_final_mismatch',
                dict(expected=action['allocation'], actual=actual))
    plan['confirmation_state'] = state
    if action['kind'] not in ('accept', 'decline'):
        plan['confirm'] = confirmation_point(state, plan['pane'], plan['confirm'])
        plan['commands'][-1] = plan['confirm']
    commands.append(plan['confirm']); probe.send(plan['confirm'])
    return plan


def same_selection(state, before, public, name, plan, expected):
    """A missing click is retryable only while the decision remains unchanged."""
    return all(selection_checks(state, before, public, name, plan, expected)['checks'].values())


def same_confirmation(state, public, action, name, plan):
    """Retry only a positively identical visible confirmation, never selection."""
    try:
        verify(state, public, name, plan, plan['stages'][-1] if plan['stages'] else {})
        if action['kind'] in ('accept', 'decline'):
            purchase_button(state, public, name, action['kind'])
        elif action['kind'] in ('play', 'sell', 'buy', 'pray'):
            verify_target(state, public, action, name, plan['pane'])
            if plan.get('target_switched'):
                verify_switched_target(state, public, action, name, plan['pane'])
        if action['kind'] == 'exchange':
            require(preview(state, public) == action['allocation'], 'exchange_final_mismatch')
        # Tooltips, sound meters and other incidental UI can change after a
        # click. Compare the decision geometry, not the entire probe snapshot.
        before = plan['confirmation_state']
        return (hand_images(state) == hand_images(before)
                and selected(state, 1040-plan['pane']) == selected(before, 1040-plan['pane']))
    except UIUnsupported:
        return False


def submission_pending(state, public, name, plan):
    """Visible submit acknowledgement permits only more read-only waiting.

    The client removes its command overlay and clears selected cards before
    awaiting the server. This is never evidence of authoritative acceptance.
    """
    before = plan['confirmation_state']
    if before.get('commandPanes') != [plan['pane']] or state.get('commandPanes') != []:
        return False
    try:
        verify_scene(state, public)
        return pane_x(state, name) == plan['pane'] and not selected(state, plan['pane'])
    except UIUnsupported:
        return False


def allow_button(state, public, action, name, plan):
    """Only an empty response can safely use the explicit Allow control."""
    require(public['phase'] == 1 and action['kind'] == 'defend' and not action['cards'],
            'allow_button_not_empty_response')
    ready(state, public, name)
    require(plan['pane'] == pane_x(state, name) == 690, 'allow_button_scene_changed')
    rows = [r for r in state.get('texts', []) if r['text'] == '許す'
            and 560 <= r['x'] and r['x']+r['w'] <= 820
            and 403 <= r['y'] and r['y']+r['h'] <= 443 and r['w'] > 0 and r['h'] > 0]
    require(len(rows) == 1, 'allow_button_missing')
    return point(690, 423)
