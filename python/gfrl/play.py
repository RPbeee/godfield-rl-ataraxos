"""Seat-scoped terminal presentation and play loop; no privileged policy buffers."""
import argparse
from contextlib import ExitStack
import json
import re
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from . import VecEnv, _core
from .actor import Actor, synchronize
from .evaluation import load_policy_model, _policy_rng
from .search import SearchAgent, SearchConfig, load_search_belief

KINDS = ('使う', '防御する（カードなしなら受ける）', '捨てる', '祈る', '両替する',
         '売る', '買う', '購入を承諾する', '購入を断る', 'ささげる')
PHASES = dict(main='手番', response='応答', purchase='購入', grant='配布', finished='終局')
CATEGORIES = dict(trade='取引', weapons='武器', armor='防具', sundries='道具',
                  miracles='奇跡', devils='悪魔', guardians='守護神', phenomena='現象')
ELEMENTS = dict(fire='火', water='水', wood='木', stone='土', light='光', darkness='闇')
CURSES = dict(none='なし', cold='風邪', fever='熱病', hell='地獄病', heaven='天国病',
              fog='霧', dream='夢', flash='閃光', darkcloud='暗雲')
GUARDIANS = dict(none='なし', mars='火星神', mercury='水星神', jupiter='木星神',
                 saturn='土星神', uranus='天王神', pluto='冥王神', neptune='海王神',
                 venus='金星神', earth='地球神', moon='月神')
# Catalog ability identifiers remain available for unusual effects.
ABILITIES = dict(absorb_hp='HP吸収', bounce_weapon='武器を乱反射', bounce_miracle='奇跡を乱反射',
                 reflect_anything='反射', reflect_weapon='武器を反射', reflect_miracle='奇跡を反射',
                 boost_hp='HP回復', boost_mp='MP回復', boost_cp='金を増やす',
                 block_weapon='武器を防ぐ', block_miracle='奇跡を防ぐ', exchange='両替',
                 attack_twice='2回攻撃', double_atk='攻撃倍増', remove_all_curses='全状態治療')
GUARDIANS[''] = 'なし'
ABILITIES.update(dict(
    add_curse='状態付与', add_curse_on_damage='ダメージ時に状態付与', add_item='カード追加',
    atk_by2x_mp='MPの2倍の攻撃', attack_dyingly='瀕死時攻撃', attack_every_enemy='全敵攻撃',
    attack_somebody='ランダム攻撃', attract_danger='危険を引き寄せる',
    boost_cp_of_everybody='全員の金を増やす', boost_cp_to_enemy='敵の金を増やす',
    boost_hp_or_deal_damage='HP回復またはダメージ', boost_mp_and_add_curse='MP回復と状態付与',
    boost_something='資源回復', buy='買う', call_phenomenon='現象発生', category_weapons='武器化',
    collect_cp_of_everybody='全員から集金', confuse_everybody='全員混乱',
    counter2x_atk='2倍反撃', counter_atk='反撃', counter_boost2x_mp='反撃でMP回復',
    counter_curse='状態で反撃', counter_take_cp='反撃で集金', cut_cost='消費MP削減',
    danger='危険', deal_damage='ダメージ', deal_same_damage='同量ダメージ', discard='捨てる',
    filter_atk_element='属性防御', remove_items='手札除去', remove_mild_curses='軽い状態治療',
    remove_something='除去', remove_used_miracles='使用済み奇跡除去', revive='復活',
    sacrifice='ささげる', self_curse='自分に状態付与', self_curse_and_redraw='状態付与と引き直し',
    sell='売る', set_curse_of_everybody='全員に状態付与', set_element='属性変更',
    set_guardian='守護神変更', set_guardian_of_everybody='全員の守護神変更',
    set_hp_of_everybody='全員のHP変更', shuffle_items_of_everybody='全員の手札交換', take_cp='集金'))


def ability_key(name):
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2',
                  re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)).lower()



class Presenter:
    def __init__(self, seat, catalog=None):
        self.seat = seat
        self.catalog = _core.play_catalog() if catalog is None else catalog

    def who(self, seat):
        return '環境' if seat is None else ('あなた' if seat == self.seat else 'モデル')

    def name(self, face):
        return self.catalog[face]['name']

    def card(self, face):
        c = self.catalog[face]
        parts = [CATEGORIES[c['category']], ELEMENTS.get(c['element'], '無属性')]
        if c['atk']:
            parts.append(f"攻撃 {'+' if c['plus'] else ''}{c['atk']}")
        if c['defense']:
            parts.append(f"防御 {c['defense']}")
        if c['cost']:
            parts.append(f"MP {c['cost']}")
        if c['hit_rate']:
            parts.append(f"命中 {c['hit_rate']}%")
        if c['ability']:
            parts.append(ABILITIES.get(ability_key(c['ability']), c['ability']) +
                         (f" {c['ability_value']}" if c['ability_value'] else ''))
        parts.append(f"価格 {c['price']}")
        return f"{c['name']}（{'・'.join(parts)}）"

    def board(self, view, *, hand=None, selected=(), reveal_hand=None):
        lines = [f"── GF {view['gf']} / 決定 {view['decisions']} / "
                 f"{self.who(view['actor'])}の{PHASES[view['phase']]} ──"]
        for p in view['players']:
            s = p['status']
            if s is None:
                text = '霧で状態は見えません'
            else:
                curses = [CURSES.get(c, c) for c in s['curses']]
                if s['confused']:
                    curses.append('混乱')
                text = (f"HP {s['hp']} / MP {s['mp']} / CP(金) {s['cp']} / "
                        f"状態: {'・'.join(curses) or 'なし'} / "
                        f"守護神: {GUARDIANS.get(s['guardian'], s['guardian'])}")
            lines.append(f"{self.who(p['id'])}: {text}" + (' / 死亡' if p['dead'] else ''))
            lines.append('  使用済み: ' + ('、'.join(map(self.name, p['used'])) or 'なし'))
        lines.append('あなたの手札:')
        for i, h in enumerate(view['hand'] if hand is None else hand):
            flags = (' [使用済み]' if h['used'] else '') + (' [選択中]' if i in selected else '')
            lines.append(f"  [{i + 1}] {self.card(h['face'])}{flags}")
        if selected:
            cards = hand if hand is not None else view['hand']
            lines.append('選択中: ' + ' + '.join(self.name(cards[i]['face']) for i in selected))
        if view['pending']:
            p = view['pending']
            cards = ' + '.join(map(self.name, p['models'])) or '効果'
            lines.append(f"処理中: {self.who(p['source'])} → {self.who(p['target'])}: {cards} / "
                         f"威力 {p['power']} / {CATEGORIES[p['category']]}・{ELEMENTS.get(p['element'], '無属性')} / "
                         f"反射 {p['reflections']}・乱反射 {p['bounces']}")
        if view['purchase']:
            p = view['purchase']
            lines.append(f"購入提示: {self.name(p['model'])} / {p['price']} 金 / "
                         f"売手 {self.who(p['seller'])} → 買手 {self.who(p['buyer'])}")
        if reveal_hand is not None:
            lines.append('[REVEAL] モデルの手札: ' + '、'.join(self.name(h['face']) for h in reveal_hand))
        return '\n'.join(lines)

    def options(self, env, hand):
        """One label per true mask entry; returned IDs are engine choices."""
        stage = int(env.stage[0])
        options = []
        for choice in np.flatnonzero(env.mask[0]):
            choice = int(choice)
            if stage == 0:
                label = KINDS[choice]
            elif stage == 1:
                label = 'これで決定（追加カードなし）' if choice == 18 else f"[手札 {choice + 1}] " + self.card(hand[choice]['face']) + 'を選ぶ'
            elif stage == 2:
                label = ('対象を指定しない', '自分を対象にする', '相手を対象にする')[choice]
            else:
                resource = 'HP' if stage == 3 else 'MP'
                label = f'{resource} を {min(5 * choice, 99)} に配分する'
                if stage == 4:
                    hp = min(5 * int(env.chosen_alloc_hp[0]), 99)
                    total = sum(env.play_view(0, self.seat)['players'][self.seat]['status'][k]
                                for k in ('hp', 'mp', 'cp'))
                    label += f'（HP {hp}・CP {total - hp - min(5 * choice, 99)}）'
            options.append((choice, label))
        return options

    def event(self, event):
        d, before, after = event['decision'], event['before'], event['after']
        kind = KINDS[d['kind']] if d['kind'] is not None else '自動処理'
        faces = '（カード非公開）' if d['faces_hidden'] else ' + '.join(map(self.name, d['faces']))
        text = f"{self.who(d['actor'])}: {faces} {kind}"
        if d['target'] is not None:
            text += f" → {self.who(d['target'])}"
        if d['allocation']:
            text += f" / 配分 {d['allocation']}"
        effects = []
        for a, b in zip(before['players'], after['players']):
            sa, sb = a['status'], b['status']
            if sa is None or sb is None:
                if sa != sb:
                    effects.append(f"{self.who(b['id'])}: 状態の可視性が変化")
                continue
            for k, label in (('hp', 'HP'), ('mp', 'MP'), ('cp', 'CP(金)')):
                delta = sb[k] - sa[k]
                if delta:
                    effects.append(f"{self.who(b['id'])} {label} {delta:+} → {sb[k]}")
            if sa['curses'] != sb['curses']:
                effects.append(f"{self.who(b['id'])} 状態: " + ('・'.join(CURSES.get(c, c) for c in sb['curses']) or 'なし'))
            if sa['guardian'] != sb['guardian']:
                effects.append(f"{self.who(b['id'])} 守護神: {GUARDIANS.get(sb['guardian'], sb['guardian'])}")
        for key in ('removal', 'sale', 'purchase_result'):
            value = after.get(key)
            if value is None or value == before.get(key):
                continue
            if key == 'removal':
                effects.append(f"{self.who(value['target'])} 除去: " +
                               ('、'.join(map(self.name, value['models'])) or 'なし'))
            elif key == 'sale':
                effects.append(f"売却: {self.name(value['model'])} / {value['price']} 金 → {self.who(value['target'])}")
            else:
                reason = ('購入成立', '購入辞退', '商品なし', '金不足', '打切り', '自分のカードを確認')[value['reason']]
                effects.append(reason + (': ' + self.name(value['model']) if value['model'] else ''))
            if value.get('overflow_model'):
                effects.append('溢れたカード: ' + self.name(value['overflow_model']))
        if d['outcome_model']:
            effects.append('公開結果: ' + self.name(d['outcome_model']))
        p = after['pending']
        if p and p != before['pending']:
            effects.append(f"処理中の威力 {p['power']} → {self.who(p['target'])} "
                           f"(反射 {p['reflections']}・乱反射 {p['bounces']})")
        for r in after['reveals']:
            if r not in before['reveals']:
                effects.append(f"公開: {self.who(r['owner'])} {self.name(r['face'])} → {self.name(r['model'])}")
        return text + '\n  結果: ' + (' / '.join(effects) or '可視の資源変化なし')


def read_choice(options, show_board, *, input_fn=input, output=print):
    while True:
        try:
            value = input_fn('番号（h: ヘルプ / s: 盤面 / q: 終了）> ').strip().lower()
        except EOFError:
            raise QuitGame from None
        if value == 'q':
            raise QuitGame
        if value == 'h':
            output('表示された番号で選択。カードは1枚ずつ追加し「これで決定」で確定します。s: 盤面、q: 終了。')
        elif value == 's':
            output(show_board())
        elif value.isdecimal() and 1 <= int(value) <= len(options):
            return options[int(value) - 1][0]
        else:
            output(f'1〜{len(options)} の番号を入力してください。')


class QuitGame(Exception):
    pass


def parser():
    p = argparse.ArgumentParser(description='学習済み EMA と God Field を対局する。探索既定: M64・D3・β0.1、候補16、actor葉、応答探索あり。')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--seat', choices=('0', '1', 'random'), default='random')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--games', type=int, default=1)
    p.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    p.add_argument('--greedy', action='store_true', help='方策は最大確率、探索は argmax（fallbackも最大確率）')
    p.add_argument('--human-policy', choices=('interactive', 'random'), default='interactive')
    p.add_argument('--log', type=Path)
    p.add_argument('--no-color', action='store_true')
    p.add_argument('--reveal', action='store_true', help='デバッグ用: 相手の手札も表示')
    p.add_argument('--search', action='store_true')
    p.add_argument('--belief', help='学習済み belief checkpoint / uniform / b1。oracle は対局では禁止')
    p.add_argument('--fit')
    for name, default in (('worlds', 64), ('candidates', 16), ('candidate-samples', 64), ('depth', 3),
                          ('threads', 1), ('max-inference-batch', 128), ('search-cache-mb', 256), ('root-batch-size', 0)):
        p.add_argument('--' + name, type=int, default=default)
    for name, default in (('search-response', True), ('skip-forced-inference', False), ('skip-single-candidate', False)):
        p.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=default)
    p.add_argument('--leaf-view', choices=('actor', 'searcher'), default='actor')
    p.add_argument('--search-cache-device', choices=('auto', 'host', 'device'), default='auto')
    p.add_argument('--alpha', type=float, default=.002)
    p.add_argument('--beta', type=float, default=.1)
    p.add_argument('--select', choices=('sample', 'argmax'), default='sample')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    p.epilog = ('探索時間は機種・手札・合法候補で変化します。CPU 1スレッド実測: Mモデル+学習済みbelief、4局43決定で中央値0.47秒、最大3.05秒。'
                'MPS は実行環境で利用不可のため未計測。測定条件は README を参照。遅い場合は --worlds 16 --candidates 2。')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.games <= 0 or args.threads <= 0 or not 0 <= args.seed < 2**63:
        p.error('games/threads は正、seed は 0〜2^63-1 が必要です')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        p.error('この環境では MPS を利用できません。--device cpu を指定してください')
    if args.search and not args.belief:
        p.error('--search には --belief が必要です')
    if args.belief == 'oracle':
        p.error('oracle は相手の隠れた手札を使うため対局では利用できません')
    torch.set_num_threads(args.threads)
    from .runtime import configure_precision
    configure_precision(args.device, args.tf32)
    try:
        config_keys = ('worlds', 'candidates', 'candidate_samples', 'depth', 'alpha', 'beta', 'select',
                       'search_response', 'leaf_view', 'max_inference_batch', 'skip_forced_inference',
                       'skip_single_candidate', 'search_cache_device', 'search_cache_mb', 'root_batch_size')
        config = SearchConfig(**{k: ('argmax' if k == 'select' and args.greedy else getattr(args, k)) for k in config_keys})
        model = load_policy_model(args.checkpoint, args.device)
        idle = VecEnv(1, args.threads, eval_pairs=0)
        if args.search:
            agent = SearchAgent(model, load_search_belief(args.belief, args.fit, args.device), args.device,
                                config=config, seed=args.seed, num_threads=args.threads, profile=args.profile)
            agent.actor.sample = not args.greedy
        else:
            agent = Actor(model, idle, args.device, sample=not args.greedy)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    rng = np.random.default_rng(args.seed)
    color = not args.no_color and sys.stdout.isatty()

    def output(text):
        print(('\033[36m' + text + '\033[0m') if color else text, flush=True)

    with ExitStack() as stack:
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
        log = stack.enter_context(args.log.open('a', encoding='utf-8')) if args.log else None
        stack.enter_context(_policy_rng(args.seed, (agent,)))

        def record(kind, **data):
            if log:
                log.write(json.dumps(dict(type=kind, game=game + 1, **data), ensure_ascii=False) + '\n')
                log.flush()

        try:
            for game in range(args.games):
                seat = int(rng.integers(2)) if args.seat == 'random' else int(args.seat)
                presenter = Presenter(seat)
                # Use the evaluator's deal/chance streams and default experimental
                # rules. Only the first game of this pair is played; next game uses
                # a fresh evaluation seed, independent of the human's seat.
                env = VecEnv(1, args.threads, eval_pairs=1, opponent='external',
                             eval_seed=args.seed + game, track_projection=args.search)
                env.enable_play_trace(0, seat)
                output(f'第 {game + 1} 局 / あなたの席: {seat}' + (' / REVEAL 有効' if args.reveal else ''))
                record('start', seat=seat, seed=args.seed + game, reveal=args.reveal)
                turn_seconds = 0.
                while not env.eval_results():
                    view = env.play_view(0, seat)
                    human = int(env.actor_seat[0]) == seat
                    if human:
                        hand = env.play_hand(0, seat)
                        selected = list(map(int, env.chosen_cards[0, :int(env.chosen_count[0])]))
                        def screen():
                            reveal = env.play_hand(0, 1 - seat) if args.reveal else None
                            return presenter.board(view, hand=hand, selected=selected, reveal_hand=reveal)
                        board = screen()
                        output(board)
                        options = presenter.options(env, hand)
                        output('\n'.join(f'{i}) {label}' for i, (_, label) in enumerate(options, 1)))
                        choice = (options[int(rng.integers(len(options)))][0] if args.human_policy == 'random'
                                  else read_choice(options, screen, output=output))
                        output(f'選択: {next(label for c, label in options if c == choice)}')
                        record('choice', board=board, options=options, choice=choice)
                    else:
                        if int(env.substep[0]) == 0:
                            output('モデルが思考中…')
                            turn_seconds = 0.
                        start = perf_counter()
                        choice = int(agent(env, np.array([0], dtype=np.int32))[0])
                        synchronize(torch.device(args.device))
                        turn_seconds += perf_counter() - start
                    env.step(np.array([choice], dtype=np.int32))
                    timed = False
                    for event in env.take_play_trace(0):
                        text = presenter.event(event)
                        seconds = (turn_seconds if not human and not timed
                                   and event['decision']['actor'] == 1 - seat
                                   and event['decision']['source'] == 0 else None)
                        if seconds is not None:
                            text += f' / 思考 {seconds:.3f} 秒'
                            timed = True
                        output(text)
                        record('decision', **event, text=text, board=presenter.board(event['after']),
                               model_seconds=seconds)
                result = env.eval_results()[0]
                score = result['result'] if seat == result['agent_seat'] else -result['result']
                outcome = '引き分け（規定手数で打切り）' if result['truncated'] else {1: 'あなたの勝ち', -1: 'モデルの勝ち', 0: '引き分け'}[score]
                output(f'終局: {outcome} / {result["decisions"]} 決定')
                record('result', outcome=outcome, score=score, truncated=result['truncated'], decisions=result['decisions'])
        except (QuitGame, KeyboardInterrupt):
            output('対局を終了しました。')
            record('quit')
    return 0
