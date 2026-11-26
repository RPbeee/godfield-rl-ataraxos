"""Offline synthetic scenes using only auto52's public image paths/rectangles.

No receipt, name, screenshot, or redacted state is a fixture. The observed
orange icon/card is Dream, not Flash; Flash is tested independently below.
"""
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from gfrl.official import OfficialDecision
from gfrl.official_ui import (FAKE_PATH as FAKE, PATHS, UIUnsupported, card_detail, execute,
                              hand_fake_overlays, hand_images, point, prepare, ready, selected,
                              verify, verify_attack)
from test_official_bot import action, results, visible


def image(path, x, y, w=80, h=80):
    return dict(path=path, x=x, y=y, w=w, h=h)


def dream_scene(flash=False):
    p = deepcopy(results()[2]['public'])
    p['gf'] = 36
    own = next(r for r in p['players'] if r['id'] == p['viewer'])
    own['status']['curses'] = ['fog', 'dream'] + (['flash'] if flash else [])
    other = next(r for r in p['players'] if r['id'] != p['viewer'])
    other.update(status=None, fogged=True)
    # All twelve observed hand faces and positions, with synthetic item IDs.
    faces = [('weapons/saver-rod',190,493,True),
             ('weapons/bouncing-sword',272,493,True),
             ('weapons/unknown-feather',354,493,True),
             ('weapons/reflection-sword',436,493,True),
             ('armor/ogre-s-helm',518,493,True),
             ('armor/steel-armor',600,493,False),
             ('armor/angel-shield',682,493,False),
             ('sundries/smile-shell',764,493,True),
             ('sundries/sun-amulet',846,493,False),
             ('miracles/waterfall',190,595,False),
             ('miracles/dream',272,595,False),
             ('miracles/fog',846,595,False)]
    models = {path: model for model, path in PATHS.items()}
    p['hand'] = []; images = []
    for i, (face, x, y, disguised) in enumerate(faces):
        path = f'/images/items/{face}.webp'
        p['hand'].append(dict(id=100+i, face=models[path], used=i == 11,
                              disguised=disguised))
        images.append(image(path, x, y))
        if disguised:
            images.append(image(FAKE, x, y))
    images += [image('/images/screens/room.webp',180,38,1080,660),
               image('/images/screens/fog.webp',180,38,1080,660),
               image('/images/curses/small/fog.webp',1083,288,30,16),
               image('/images/curses/small/dream.webp',1141,288,30,16),
               image('/images/items/trade/discard.webp',846,405),
               image(PATHS[199],955,463)]
    if flash:
        images.append(image('/images/curses/small/flash.webp',1112,288,30,16))
    s = dict(body=f"G.F.36\nHP {own['status']['hp']} MP {own['status']['mp']} ¥ {own['status']['cp']}",
             images=images, inputs=[], targetArrow=None,
             texts=[dict(text='SyntheticName',x=320,y=60,w=60,h=20),
                    dict(text='SyntheticName',x=870,y=270,w=90,h=20)],
             fogBands=[dict(x=860,y=131,w=340,h=40)])
    return p, s, action(p,'play',[199],target=p['viewer'])


@pytest.mark.parametrize('flash', [False, True])
def test_dream_selection_overlay_executes_once(flash):
    p, s, a = dream_scene(flash)
    ready(s,p,'SyntheticName')
    assert len(hand_images(s)) == len(p['hand']) == 12
    with pytest.raises(UIUnsupported, match='ambiguous_disguised_face'):
        prepare(s,p,a,'SyntheticName')
    commands = []
    class Probe:
        def send(self, command):
            if command == point(804,533):
                s['images'] += [image(PATHS[199],205,103),image(FAKE,205,103)]
            return dict(state=deepcopy(s))
    plan = execute(Probe(),s,p,a,'SyntheticName',commands=commands,
                   dream_last_resort=True,settle_timeout=0)
    assert commands == [point(804,533),point(350,193)]
    assert selected(s,350) == Counter({PATHS[199]:1})
    assert plan['attempted'] == [dict(path=PATHS[199],x=804,y=533)]


@pytest.mark.parametrize('bad', ['wrong','extra','duplicate','missing','unknown','overlay_only'])
def test_real_selection_mismatch_still_stops(bad):
    p, s, a = dream_scene()
    plan = prepare(s,p,a,'SyntheticName',dream_last_resort=True)
    paths = {'wrong':[PATHS[200]], 'extra':[PATHS[199],PATHS[200]],
             'duplicate':[PATHS[199],PATHS[199]], 'missing':[],
             'unknown':[PATHS[199],'/images/items/unknown.webp'], 'overlay_only':[]}[bad]
    s['images'] += [image(path,205+82*i,103) for i,path in enumerate(paths)]
    s['images'].append(image(FAKE,205,103))
    with pytest.raises(UIUnsupported,match='selected_cards_mismatch'):
        verify(s,p,'SyntheticName',plan,{PATHS[199]:1})
    if bad == 'unknown':
        assert card_detail({},selected(s,350))['actual']['?'] == 1


@pytest.mark.parametrize('pane', [350,690])
def test_selected_faces_use_item_paths_size_and_pane_geometry(pane):
    x = pane-145
    s = dict(images=[image(PATHS[199],x,103),image(FAKE,x,103),
                     image('/images/curses/flash.webp',x,103),
                     image('/images/screens/fog.webp',x,103),
                     image(PATHS[200],x-1,103),image(PATHS[200],pane+66,103),
                     image(PATHS[200],x,493),image(PATHS[200],x,103,30,16),
                     image(PATHS[200],1040-pane-145,103)])
    assert selected(s,pane) == Counter({PATHS[199]:1})


def test_fog_attack_overlay_keeps_exact_face_check():
    p, s, _ = dream_scene()
    p['pending'] = dict(models=[199])
    s['images'] += [image(PATHS[199],545,103),image(FAKE,545,103)]
    verify_attack(s,p,350)
    s['images'].append(image(PATHS[200],627,103))
    with pytest.raises(UIUnsupported,match='visible_attack_mismatch'):
        verify_attack(s,p,350)


@pytest.mark.parametrize('bad', ['missing','extra','unknown'])
def test_hand_overlay_does_not_hide_real_mismatches(bad):
    p, s, _ = dream_scene()
    if bad == 'missing':
        s['images'].remove(next(r for r in s['images'] if r['path']==PATHS[199] and r['x']==764))
    else:
        s['images'].append(image(PATHS[199] if bad=='extra' else '/images/items/unknown.webp',354,595))
    with pytest.raises(UIUnsupported,match='visible_hand_mismatch'):
        ready(s,p,'SyntheticName')


def test_hand_fake_overlays_filter_and_sort_visible_rectangles():
    first = image(FAKE, 180, 480)
    second = image(FAKE, 850, 620)
    state = dict(images=[second, image(FAKE, 179, 480), image(FAKE, 851, 620),
                         image(FAKE, 180, 479), image(FAKE, 850, 621),
                         image(FAKE, 190, 493, 79, 80), image(PATHS[6], 190, 493),
                         image(FAKE, 205, 103), first])
    assert hand_fake_overlays(state) == [first, second]


@pytest.mark.parametrize('bad', ['missing', 'extra', 'stacked', 'misaligned', 'unmatched'])
@pytest.mark.parametrize('select_disguised', [False, True])
def test_mixed_face_bad_overlays_fail_before_click(bad, select_disguised):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=18, face=6, used=False, disguised=select_disguised),
                 dict(id=99, face=6, used=False, disguised=not select_disguised)]
    s = visible(p)
    overlay, = hand_fake_overlays(s)
    if bad == 'missing':
        s['images'].remove(overlay)
    elif bad == 'extra':
        s['images'].append(image(FAKE, 764 if select_disguised else 846, 595))
    elif bad == 'stacked':
        s['images'].append(dict(overlay))
    elif bad == 'misaligned':
        overlay['y'] += 2
    else:
        s['images'].append(dict(overlay, y=overlay['y'] + 2))
    ready(s, p, 'SyntheticName')
    class NoClicks:
        def send(self, command):
            pytest.fail(f'unexpected UI command: {command}')
    with pytest.raises(UIUnsupported, match='^disguised_overlay_mismatch$'):
        execute(NoClicks(), s, p, action(p, 'play', [6], target=2), 'SyntheticName',
                allow_dream=select_disguised, settle_timeout=0)


@pytest.mark.parametrize('disguised', [False, True])
@pytest.mark.parametrize('covered', [0, 1, 2])
def test_uniform_face_overlay_coverage(disguised, covered):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=i, face=6, used=False, disguised=disguised) for i in (18, 99)]
    s = visible(p)
    s['images'] = hand_images(s)
    s['images'] += [dict(r, path=FAKE) for r in s['images'][:covered]]
    a = action(p, 'play', [6], target=2)
    if covered == 0 or (disguised and covered == 2):
        assert prepare(s, p, a, 'SyntheticName', allow_dream=disguised)['cards'] == [point(804, 635)]
    else:
        with pytest.raises(UIUnsupported, match='^disguised_overlay_mismatch$'):
            prepare(s, p, a, 'SyntheticName', allow_dream=disguised)


@pytest.mark.parametrize('used_disguised', [False, True])
@pytest.mark.parametrize('select_used', [False, True])
def test_mixed_face_overlays_and_used_suffix_agree(used_disguised, select_used):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=18, face=218, used=False, disguised=not used_disguised),
                 dict(id=99, face=218, used=True, disguised=used_disguised)]
    s = visible(p)
    a = action(p, 'play', [218], target=2)
    if select_used:
        a['cards'][0]['item_id'] = a['operations'][1]['item_id'] = 99
    overlay, = hand_fake_overlays(s)
    overlay['x'] += 1
    overlay['y'] -= 1
    assert prepare(s, p, a, 'SyntheticName', allow_dream=True)['cards'] == [
        point(886 if select_used else 230, 635)]
    # The total is right but the overlay contradicts the identified used slot.
    overlay['x'] = 190 if used_disguised else 846
    with pytest.raises(UIUnsupported, match='^disguised_overlay_mismatch$'):
        prepare(s, p, a, 'SyntheticName', allow_dream=True)


@pytest.mark.parametrize('flash', [False,True])
def test_flash_response_limits_second_card_including_free(flash):
    # goodfield/docs/official-execution-subset.md: Flash is checked before FREE.
    p = deepcopy(results()[1]['public'])
    own = next(r for r in p['players'] if r['id']==p['viewer'])
    own['status']['curses'] = ['flash'] if flash else []
    p['pending'].update(models=[6],power=1,element=None,category='weapons')
    p['hand'] = [dict(id=100+i,face=m,used=False,disguised=False)
                 for i,m in enumerate((233,204))]  # Wall followed by FREE doll
    env = OfficialDecision(p)
    assert env.mask[0,0]
    env.step(np.array([0],np.int32))
    if flash:
        assert env.complete
        assert env.semantic_action()['cards'] == [dict(item_id=100,model=233)]
    else:
        assert not env.complete and env.mask[0,1]


@pytest.mark.parametrize('flash', [False,True])
def test_flash_does_not_limit_main_turn_to_one_card(flash):
    p = deepcopy(results()[2]['public'])
    own = next(r for r in p['players'] if r['id']==p['viewer'])
    own['status']['curses'] = ['flash'] if flash else []
    p['hand'] = [dict(id=100+i,face=m,used=False,disguised=False)
                 for i,m in enumerate((6,7))]
    env = OfficialDecision(p)
    env.step(np.array([0],np.int32))
    assert not env.complete and env.mask[0,1]
