/* God Field 観戦ダッシュボード — 閲覧専用フロントエンド
 * API: /api/catalog /api/summary /api/live /api/game/<N>
 * 外部ネットワーク依存なし。受け取った文字列は必ずエスケープして描画する。
 */
'use strict';
(function () {
  /* ===================================================================
   * 0. 定数・辞書
   * =================================================================== */
  const POLL_LIVE_MS = 2000;
  const POLL_SUMMARY_MS = 6000;
  const STEP_MS = { update: 820, decision: 640 };
  const LIVE_SPEEDS = [1, 1.5, 2];
  const REPLAY_SPEEDS = [0.5, 1, 2];
  const GOAL_GAMES = 1000;

  const ELEMENTS = {
    fire: { ja: '火', cls: 'fire' }, water: { ja: '水', cls: 'water' }, wood: { ja: '木', cls: 'wood' },
    stone: { ja: '土', cls: 'stone' }, light: { ja: '光', cls: 'light' }, darkness: { ja: '闇', cls: 'dark' },
  };
  const CATEGORIES = {
    weapons: { ja: '武器', ic: '⚔️' }, armor: { ja: '防具', ic: '🛡️' }, miracles: { ja: '奇跡', ic: '✨' },
    sundries: { ja: '雑貨', ic: '🎒' }, trade: { ja: '取引', ic: '💱' }, devils: { ja: '悪魔', ic: '😈' },
    guardians: { ja: '守護神技', ic: '🌟' }, phenomena: { ja: '現象', ic: '🌀' },
  };
  const GUARDIAN_JA = {
    mercury: '水星', venus: '金星', earth: '地球', mars: '火星', jupiter: '木星', saturn: '土星',
    uranus: '天王星', neptune: '海王星', pluto: '冥王星', moon: '月',
  };
  const CURSE_JA = {
    cold: '風邪', fever: '熱病', hell: '地獄病', heaven: '天国病', fog: '霧', dream: '夢', flash: '閃光', darkcloud: '暗雲',
  };
  const CURSE_IC = { cold: '🥶', fever: '🤒', hell: '🔥', heaven: '😇', fog: '🌫️', dream: '🌙', flash: '⚡', darkcloud: '☁️' };
  const KIND_JA = {
    play: '出す', defend: '防御', discard: '捨てる', pray: '祈る', exchange: '両替', sell: '売る',
    buy: '買う', accept: '受諾', decline: '拒否', sacrifice: '生贄',
  };
  const PHASE_JA = { 0: '主行動', 1: '防御応答', 2: '終局', 3: '購入応答', 4: '配布' };
  const ABILITY_JA = {
    discard: '捨てる', sacrifice: '生贄', exchange: '両替', sell: '売却', buy: '購入', attackTwice: '2回攻撃',
    bounceWeapon: '武器を跳ね返す', bounceMiracle: '奇跡を跳ね返す', reflectWeapon: '武器を反射',
    reflectMiracle: '奇跡を反射', reflectAnything: '全反射', blockWeapon: '武器を防ぐ', blockMiracle: '奇跡を防ぐ',
    absorbHP: 'HP吸収', addCurseOnDamage: 'ダメージ時に呪い', addCurse: '呪い付与', cutCost: 'コスト軽減',
    dealSameDamage: '同ダメージ', setElement: '属性変更', danger: '危険', attackDyingly: '瀕死攻撃',
    selfCurse: '自分に呪い', selfCurseAndRedraw: '呪い＋引き直し', counterAtk: '反撃', counterCurse: '反撃呪い',
    counterTakeCP: '¥奪取反撃', filterAtkElement: '属性フィルタ', boostHP: 'HP回復', boostMP: 'MP回復',
    boostMPAndAddCurse: 'MP回復＋呪い', removeMildCurses: '軽い呪い解除', removeAllCurses: '呪い全解除',
    setGuardian: '守護神召喚', boostHPOrDealDamage: 'HP回復/ダメージ', removeItems: '手札破壊',
    removeUsedMiracles: '使用済み奇跡破壊', callPhenomenon: '現象召喚', revive: '復活', attractDanger: '危険を招く',
    doubleAtk: '攻撃2倍', attackEveryEnemy: '全体攻撃', dealDamage: 'ダメージ', removeSomething: '破壊',
    boostSomething: '強化', categoryWeapons: '武器化', boostCP: '¥増加', boostCPOfEverybody: '全員¥増加',
    boostCPToEnemy: '相手¥増加', takeCP: '¥奪取', addItem: '札追加', setCurseOfEverybody: '全員に呪い',
    confuseEverybody: '全員混乱', setHPOfEverybody: '全員HP設定', attackSomebody: '無差別攻撃',
    collectCPOfEverybody: '全員から¥徴収', shuffleItemsOfEverybody: '手札シャッフル', setGuardianOfEverybody: '全員に守護神',
  };
  const STOP_JA = {
    unsupported_receive: '未対応の受信', conflicting_update: '更新の矛盾', visible_gf_changed: '画面GFの変化',
    process_restart: 'プロセス再起動', acceptance_timeout: '受理タイムアウト', visible_state_timeout: '画面状態タイムアウト',
    selected_cards_mismatch: '選択カード不一致', unknown: '不明',
  };
  const NAME = { self: 'AI', opp: '相手' };
  const FULLNAME = { self: 'AI (m-xit1)', opp: '相手 (人間)' };
  const other = (s) => (s === 'self' ? 'opp' : 'self');

  /* ===================================================================
   * 1. ユーティリティ
   * =================================================================== */
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g, (c) => ESC[c]);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const isNum = (v) => typeof v === 'number' && isFinite(v);
  const sgn = (v) => (v > 0 ? '+' + v : String(v));
  const fmtPct = (r, d) => (isNum(r) ? (r * 100).toFixed(d == null ? 1 : d) + '%' : '—');
  const fmtInt = (v) => (isNum(v) ? String(Math.round(v)) : '—');
  const fmtDur = (s) => {
    if (!isNum(s)) return '—';
    const m = Math.floor(s / 60);
    return m > 0 ? m + '分' + String(Math.round(s % 60)).padStart(2, '0') + '秒' : Math.round(s) + '秒';
  };
  const sideOf = (b) => (b === true ? 'self' : b === false ? 'opp' : null);
  const reduceMotion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

  async function getJSON(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error('http_' + res.status);
    return res.json();
  }

  /* ===================================================================
   * 2. 状態
   * =================================================================== */
  const newView = () => ({
    game: null, rows: [], cursor: -1, meta: null, serverTime: 0, isLive: false,
    scale: { hp: 50, mp: 20, cp: 30 }, loaded: false,
  });
  const S = {
    catalog: {}, catalogReady: false,
    summary: null, assets: { guardians: null, curses: null },
    tab: 'live', mode: 'live',
    live: newView(), replay: newView(),
    auto: true, offline: false,
    speed: { live: 1, replay: 1 },
    playing: false,
    liveTimer: null, playTimer: null, pollTimer: null, summaryTimer: null,
    logKey: '', logUpTo: -1,
    statsSig: '', stitch: true, filter: 'all', shown: 50,
    chart: null, lastStepKind: '',
  };
  const view = () => (S.mode === 'live' ? S.live : S.replay);

  /* ===================================================================
   * 3. カード・画像ヘルパー
   * =================================================================== */
  const card = (id) => S.catalog[String(id)] || null;
  const cname = (id) => { const c = card(id); return c ? c.name : '#' + id; };
  const elemCls = (c) => (c && ELEMENTS[c.element] ? 'el-' + ELEMENTS[c.element].cls : 'el-none');
  const catInfo = (c) => (c && CATEGORIES[c.category]) || { ja: 'カード', ic: '🃏' };

  function imgTag(src, cls, alt, fb) {
    return '<img class="' + esc(cls) + '" src="' + esc(src) + '" alt="' + esc(alt) + '" decoding="async" draggable="false" data-fb="' + esc(fb) + '">';
  }
  function fbSpan(cls, fb) { return '<span class="img-fb ' + esc(cls) + '" aria-hidden="true">' + esc(fb) + '</span>'; }
  function cardIcon(id, cls) {
    const c = card(id);
    const fb = c ? catInfo(c).ic : '🃏';
    if (c && c.image_url) return imgTag(c.image_url, cls || 'ico', c.name, fb);
    return fbSpan(cls || 'ico', fb);
  }
  const fakeIcon = (cls) => imgTag('/assets/items/fake.webp', cls || 'ico', '偽装カード', '🌙');
  function guardianIcon(g, cls) {
    if (!g || g === 'none') return '';
    const have = S.assets.guardians;
    if (have && have.indexOf(g) < 0) return fbSpan(cls || 'gico', '🌟');
    return imgTag('/assets/guardians/' + encodeURIComponent(g) + '.webp', cls || 'gico', GUARDIAN_JA[g] || g, '🌟');
  }
  function curseIcon(c, cls) {
    const have = S.assets.curses;
    if (have && have.indexOf(c) < 0) return fbSpan(cls || 'cico', CURSE_IC[c] || '☠️');
    return imgTag('/assets/curses/' + encodeURIComponent(c) + '.webp', cls || 'cico', CURSE_JA[c] || c, CURSE_IC[c] || '☠️');
  }
  // 画像の読み込み失敗は壊れたアイコンを出さずに代替表示へ差し替える。
  document.addEventListener('error', (ev) => {
    const t = ev.target;
    if (!t || t.tagName !== 'IMG' || !t.dataset) return;
    const sp = document.createElement('span');
    sp.className = 'img-fb ' + t.className;
    sp.textContent = t.dataset.fb || '🃏';
    sp.setAttribute('aria-hidden', 'true');
    if (t.parentNode) t.parentNode.replaceChild(sp, t);
  }, true);

  function statChips(c) {
    if (!c) return '';
    const o = [];
    if (c.atk > 0) o.push('<i class="chip atk">攻' + (c.plus ? '+' : '') + c.atk + '</i>');
    if (c.defense > 0) o.push('<i class="chip def">防' + c.defense + '</i>');
    if (c.cost > 0) o.push('<i class="chip mp">MP' + c.cost + '</i>');
    if (c.price > 0) o.push('<i class="chip cp">¥' + c.price + '</i>');
    if (c.hit_rate > 0) o.push('<i class="chip hit">命中' + c.hit_rate + '%</i>');
    return o.join('');
  }
  function abilityText(c) {
    if (!c || !c.ability) return '';
    const base = ABILITY_JA[c.ability] || c.ability;
    return base + (c.ability_value ? ' ' + c.ability_value : '');
  }
  function cardTipHtml(id, dream) {
    const c = card(id);
    if (!c) return '<b>カード #' + esc(id) + '</b>';
    const ci = catInfo(c);
    const el = ELEMENTS[c.element];
    const lines = [];
    lines.push('<div class="tip-head">' + cardIcon(id, 'tip-ico') + '<div><b>' + esc(c.name) + '</b><small>' +
      esc(ci.ja) + (el ? ' ・ ' + esc(el.ja) + '属性' : '') + '</small></div></div>');
    const st = statChips(c);
    if (st) lines.push('<div class="tip-chips">' + st + '</div>');
    const ab = abilityText(c);
    if (ab) lines.push('<div class="tip-row">能力: ' + esc(ab) + '</div>');
    if (c.curse) lines.push('<div class="tip-row">呪い: ' + esc(CURSE_JA[c.curse] || c.curse) + '</div>');
    if (c.guardian) lines.push('<div class="tip-row">守護神: ' + esc(GUARDIAN_JA[c.guardian] || c.guardian) + '</div>');
    if (dream) lines.push('<div class="tip-row dream">🌙 夢（偽装）カード: 表向きの絵柄です</div>');
    return lines.join('');
  }

  /* ===================================================================
   * 4. 解析: 行データ → 1ステップの説明（バナー・ステージ・ポップアップ）
   * =================================================================== */
  function updateAt(rows, idx) {
    for (let i = Math.min(idx, rows.length - 1); i >= 0; i--) if (rows[i].ev === 'update') return i;
    return -1;
  }
  function applyReveals(faces, reveals) {
    const rv = (reveals || []).slice();
    return (faces || []).map((f) => {
      const k = rv.findIndex((r) => r[0] === f);
      if (k < 0) return f;
      const m = rv[k][1];
      rv.splice(k, 1);
      return m;
    });
  }
  const namesOf = (ids) => ids.map((i) => '「' + cname(i) + '」').join('・');
  const sumAtk = (ids) => ids.reduce((a, i) => { const c = card(i); return a + (c && c.atk > 0 ? c.atk : 0); }, 0);

  function classifyPlay(faces) {
    const cs = faces.map(card).filter(Boolean);
    if (!cs.length) return 'unknown';
    const has = (a) => cs.some((c) => c.ability === a);
    if (has('exchange')) return 'exchange';
    if (has('sell')) return 'sell';
    if (has('buy')) return 'buy';
    if (has('sacrifice')) return 'sacrifice';
    if (has('discard')) return 'discard';
    if (cs.some((c) => c.category === 'weapons')) return 'attack';
    if (cs.some((c) => c.category === 'miracles')) return 'miracle';
    if (cs.some((c) => c.category === 'devils')) return 'devil';
    if (cs.some((c) => c.category === 'sundries')) return 'sundry';
    return 'item';
  }

  function diffSeats(p, r) {
    const d = { self: {}, opp: {} };
    if (!p) return d;
    ['self', 'opp'].forEach((s) => ['hp', 'mp', 'cp'].forEach((k) => {
      const a = p[s][k], b = r[s][k];
      if (a != null && b != null && a !== b) d[s][k] = b - a;
    }));
    return d;
  }
  function newHandIdx(cur, prev) {
    const cnt = new Map();
    if (prev) prev.hand.forEach((h) => { const k = h[0] + ':' + h[1]; cnt.set(k, (cnt.get(k) || 0) + 1); });
    const out = new Set();
    cur.hand.forEach((h, i) => {
      const k = h[0] + ':' + h[1];
      const c = cnt.get(k) || 0;
      if (c > 0) cnt.set(k, c - 1); else out.add(i);
    });
    return out;
  }
  function finalResult(V, r) {
    const o = V.meta && V.meta.outcome;
    if (o === 'win' || o === 'loss' || o === 'draw') return o;
    if (r.opp.dead && !r.self.dead) return 'win';
    if (r.self.dead && !r.opp.dead) return 'loss';
    if (r.self.hp != null && r.opp.hp != null && r.self.hp !== r.opp.hp && !r.self.dead && !r.opp.dead) {
      return r.self.hp > r.opp.hp ? 'win' : 'loss';
    }
    return 'draw';
  }

  function analyzeUpdate(V, idx) {
    const rows = V.rows, r = rows[idx];
    const pi = updateAt(rows, idx - 1);
    const p = pi >= 0 ? rows[pi] : null;
    const ev = r.events || [];
    const A = (n) => ev.filter((e) => e.action === n);
    const has = (n) => ev.some((e) => e.action === n);
    const who = p ? sideOf(p.actor_self) : null;
    const diff = diffSeats(p, r);

    let tgt = null;
    const stp = A('setTargetPlayer')[0];
    if (stp) tgt = sideOf(stp.player_self);
    else if (p && p.pending) tgt = p.pending.target_self ? 'self' : 'opp';
    else if (r.pending) tgt = r.pending.target_self ? 'self' : 'opp';

    const beats = [];
    const zones = { self: [], opp: [] };
    const popups = [];
    const stage = { zones, center: null, arrow: null, fx: null, result: null, tone: 'idle' };
    const addBeat = (b) => { beats.push(b); return b; };
    const marks = new Set(['miss', 'block', 'safe', 'reflect', 'bounce', 'danger', 'hit', 'counterAttack'].filter(has));
    const dmgEvents = A('dealDamage').concat(A('dealDarkDamage'));
    const dmgTotal = dmgEvents.reduce((a, e) => a + (isNum(e.damage) && e.damage > 0 ? e.damage : 0), 0);

    // --- 攻撃/使用 ---
    let newAttack = null;
    let delay = 0;
    A('useAttackItems').forEach((e) => {
      const faces = applyReveals(e.faces || [], e.reveals);
      const kind = classifyPlay(faces);
      const nm = namesOf(faces) || '札';
      const side = who;
      const sName = side ? NAME[side] : '誰か';
      const role = kind === 'attack' ? 'attack' : 'use';
      faces.forEach((f, i) => zones[side || 'opp'].push({ id: f, role, delay: delay + i * 110 }));
      delay += 250;
      if (kind === 'attack') {
        const power = r.pending && r.pending.power > 0 && !newAttack ? r.pending.power : sumAtk(faces);
        newAttack = { src: side, faces, power };
        const wait = r.pending && r.phase === 1 && r.actor_self != null ? '　— ' + NAME[sideOf(r.actor_self)] + 'の応答待ち' : '';
        addBeat({ pri: 70, ic: '⚔️', tone: side === 'opp' ? 'danger' : 'attack', cards: faces,
          text: sName + 'の攻撃 → ' + (tgt ? NAME[tgt] : '?') + '（' + nm + ' 攻' + power + '）' + wait });
      } else if (kind === 'miracle') {
        addBeat({ pri: 70, ic: '✨', tone: 'miracle', cards: faces, text: sName + 'が奇跡発動（' + nm + '）' });
      } else if (kind === 'exchange') {
        addBeat({ pri: 60, ic: '💱', tone: 'trade', cards: faces, text: sName + 'が両替を実行' });
      } else if (kind === 'sell') {
        addBeat({ pri: 60, ic: '🤝', tone: 'trade', cards: faces, text: sName + 'が「売る」を使用' });
      } else if (kind === 'buy') {
        addBeat({ pri: 60, ic: '💰', tone: 'trade', cards: faces, text: sName + 'が「買う」を使用' });
      } else if (kind === 'devil') {
        addBeat({ pri: 65, ic: '😈', tone: 'devil', cards: faces, text: sName + 'が悪魔カード（' + nm + '）' });
      } else if (kind === 'discard') {
        addBeat({ pri: 55, ic: '🗑️', tone: 'sundry', cards: faces, text: sName + 'が札を捨てた' });
      } else if (kind === 'sacrifice') {
        addBeat({ pri: 55, ic: '🕯️', tone: 'devil', cards: faces, text: sName + 'が生贄を捧げた' });
      } else {
        addBeat({ pri: 60, ic: '🎒', tone: 'sundry', cards: faces, text: sName + 'が' + nm + 'を使用' });
      }
    });

    // --- 守護神・連続攻撃 ---
    ['attackByGuardian', 'nextAttack'].forEach((act) => A(act).forEach((e) => {
      const a = e.attack || {};
      const src = sideOf(a.source_self) || who, tg = sideOf(a.target_self) || tgt;
      const g = a.guardian && a.guardian !== 'none' ? a.guardian : null;
      const models = a.models || [];
      const label = act === 'nextAttack' ? '連続攻撃' : '守護神「' + (GUARDIAN_JA[g] || '？') + '」が攻撃';
      newAttack = newAttack || { src, faces: models, power: a.atk };
      if (src) {
        if (g) zones[src].push({ token: 'guardian', guardian: g, delay });
        models.forEach((m, i) => zones[src].push({ id: m, role: 'attack', delay: delay + 100 + i * 110 }));
        delay += 250;
      }
      addBeat({ pri: 72, ic: g ? '🌟' : '⚔️', tone: src === 'opp' ? 'danger' : 'attack', cards: models, guardian: g,
        text: (src ? NAME[src] + 'の' : '') + label + ' → ' + (tg ? NAME[tg] : '?') + (isNum(a.atk) ? '（攻' + a.atk + '）' : '') });
    }));

    // --- 防御 ---
    const defs = A('useDefenseItems');
    let carried = null;
    if (!newAttack && p && p.pending && (defs.length || dmgEvents.length || marks.size)) {
      carried = { src: p.pending.source_self === true ? 'self' : p.pending.source_self === false ? 'opp' : null,
        faces: p.pending.models || [], power: p.pending.power, guardian: p.pending.guardian };
    } else if (!newAttack && r.pending && !defs.length) {
      carried = { src: r.pending.source_self === true ? 'self' : r.pending.source_self === false ? 'opp' : null,
        faces: r.pending.models || [], power: r.pending.power, guardian: r.pending.guardian };
    }
    if (carried && carried.src) {
      if (carried.guardian && carried.guardian !== 'none') zones[carried.src].push({ token: 'guardian', guardian: carried.guardian, carried: true });
      carried.faces.forEach((m) => zones[carried.src].push({ id: m, role: 'attack', carried: true }));
    }
    let dmgFolded = false;
    defs.forEach((e) => {
      const faces = applyReveals(e.faces || [], e.reveals);
      const side = who || 'self';
      const outcome = [];
      if (marks.has('reflect')) outcome.push('反射！');
      else if (marks.has('bounce')) outcome.push('跳ね返した！');
      else if (marks.has('block') || marks.has('safe')) outcome.push('防ぎきった');
      else if (marks.has('miss')) outcome.push('外れた');
      if (dmgTotal > 0) { outcome.push(dmgTotal + ' ダメージ'); dmgFolded = true; }
      const tail = outcome.length ? ' → ' + outcome.join('・') : '';
      const dd = delay + 420;
      if (faces.length) {
        faces.forEach((f, i) => zones[side].push({ id: f, role: 'defense', delay: dd + i * 110 }));
        addBeat({ pri: 80, ic: '🛡️', tone: 'defend', cards: faces, text: NAME[side] + 'が防御（' + namesOf(faces) + '）' + tail });
      } else {
        zones[side].push({ token: 'pass', delay: dd });
        addBeat({ pri: 78, ic: '🙅', tone: 'pass', cards: [], text: NAME[side] + 'は防御せず' + tail });
      }
      if (marks.has('reflect')) stage.fx = 'reflect';
      else if (marks.has('bounce')) stage.fx = 'bounce';
      else if (marks.has('block') || marks.has('safe') || (faces.length && !dmgTotal)) stage.fx = 'shield';
    });

    // --- ダメージ・回復・資源 ---
    if (dmgTotal > 0 && !dmgFolded) {
      let hit = ['self', 'opp'].find((s) => diff[s].hp < 0) || tgt;
      addBeat({ pri: 62, ic: '💥', tone: 'danger', cards: [], text: (hit ? NAME[hit] : '?') + 'に ' + dmgTotal + ' ダメージ' });
    }
    if (dmgTotal > 0 && !stage.fx) stage.fx = 'impact';
    if (dmgTotal > 0 && stage.fx === 'shield') stage.fx = 'impact';
    A('boostHP').forEach((e) => {
      const s = ['self', 'opp'].find((x) => diff[x].hp > 0) || who;
      addBeat({ pri: 50, ic: '💚', tone: 'heal', cards: [], text: (s ? NAME[s] : '?') + 'の HP が回復' + (isNum(e.hp) ? '（+' + e.hp + '）' : '') });
    });
    A('boostMP').forEach((e) => {
      const s = ['self', 'opp'].find((x) => diff[x].mp > 0) || who;
      addBeat({ pri: 48, ic: '🔷', tone: 'mp', cards: [], text: (s ? NAME[s] : '?') + 'の MP が増加' + (isNum(e.mp) ? '（+' + e.mp + '）' : '') });
    });
    A('boostCP').concat(A('takeCP'), A('boostCPOfEverybody'), A('collectCPOfEverybody')).forEach((e) => {
      addBeat({ pri: 48, ic: '💰', tone: 'cp', cards: [], text: e.action === 'takeCP' ? '¥を奪った' + (isNum(e.cp) ? '（' + e.cp + '）' : '') : '¥が変動' + (isNum(e.cp) ? '（' + sgn(e.cp) + '）' : '') });
    });
    A('exchange').forEach((e) => {
      const s = sideOf(e.player_self) || who;
      const al = { hp: e.hp || 0, mp: e.mp || 0, cp: e.cp || 0 };
      if (s) zones[s].push({ token: 'alloc', alloc: al, delay });
      addBeat({ pri: 58, ic: '💱', tone: 'trade', cards: [], text: (s ? NAME[s] : '?') + 'の配分 HP' + al.hp + ' / MP' + al.mp + ' / ¥' + al.cp });
    });

    // --- 購入 ---
    if (r.purchase) {
      const seller = r.purchase.seller_self ? 'self' : 'opp';
      const buyer = r.purchase.buyer_self ? 'self' : 'opp';
      stage.center = { type: 'offer', id: r.purchase.model, seller, buyer };
    }
    A('buy').forEach((e) => {
      const m = e.model;
      const c = card(m);
      const seller = r.purchase ? (r.purchase.seller_self ? 'self' : 'opp') : null;
      const buyer = r.purchase ? (r.purchase.buyer_self ? 'self' : 'opp') : null;
      if (!stage.center && m) stage.center = { type: 'offer', id: m, seller, buyer };
      addBeat({ pri: 66, ic: '💰', tone: 'trade', cards: m ? [m] : [],
        text: (seller ? NAME[seller] + 'が' : '') + (m ? '「' + cname(m) + '」' : '商品') + 'を購入提示' + (c && c.price ? '（¥' + c.price + '）' : '') + (buyer ? '　— ' + NAME[buyer] + 'の判断待ち' : '') });
    });
    A('setBought').forEach((e) => {
      const m = e.model;
      addBeat({ pri: 66, ic: e.bought ? '✅' : '❌', tone: e.bought ? 'heal' : 'pass', cards: m ? [m] : [],
        text: e.bought ? '購入成立' + (m ? '（' + namesOf([m]) + '）' : '') : '購入は見送り' });
      if (m) stage.center = { type: 'offer', id: m, done: !!e.bought };
    });
    A('canNotBuy').forEach((e) => {
      addBeat({ pri: 62, ic: '🚫', tone: 'pass', cards: e.model ? [e.model] : [], text: '購入できない（資金不足など）' });
    });
    A('sell').forEach((e) => {
      const b = sideOf(e.buyer_self), s = sideOf(e.seller_self);
      const faces = e.faces && e.faces.length ? e.faces : (e.face ? [e.face] : []);
      addBeat({ pri: 64, ic: '🤝', tone: 'trade', cards: faces,
        text: (s ? NAME[s] : '?') + 'が' + (faces.length ? namesOf(faces) : '商品') + 'を売却 → ' + (b ? NAME[b] : '?') + 'が受け取り' });
    });

    // --- 悪魔・現象・その他 ---
    A('useDevilItem').forEach((e) => {
      if (e.model) stage.center = { type: 'devil', id: e.model };
      addBeat({ pri: 68, ic: '😈', tone: 'devil', cards: e.model ? [e.model] : [], text: '悪魔「' + cname(e.model) + '」が降臨' });
    });
    A('phenomenon').forEach((e) => {
      if (e.model) stage.center = { type: 'phenomenon', id: e.model };
      addBeat({ pri: 68, ic: '🌀', tone: 'miracle', cards: e.model ? [e.model] : [], text: '現象「' + cname(e.model) + '」が発生' });
    });
    A('pray').forEach(() => addBeat({ pri: 55, ic: '🙏', tone: 'miracle', cards: [], text: (who ? NAME[who] : '誰か') + 'が祈りを捧げた' }));
    A('setGuardian').forEach((e) => {
      const s = sideOf(e.player_self) || who;
      const g = e.guardian;
      if (s && g && g !== 'none') popups.push({ side: s, text: '守護神 ' + (GUARDIAN_JA[g] || g), tone: 'guard', guardian: g });
      addBeat({ pri: 64, ic: '🌠', tone: 'guard', cards: [], guardian: g && g !== 'none' ? g : null,
        text: (s ? NAME[s] : '?') + 'が守護神「' + (GUARDIAN_JA[g] || '？') + '」を召喚' });
    });
    A('removeGuardian').forEach((e) => {
      const s = sideOf(e.player_self) || who;
      addBeat({ pri: 52, ic: '💨', tone: 'pass', cards: [], text: (s ? NAME[s] : '?') + 'の守護神が去った' });
    });
    A('addCurse').forEach((e) => {
      const s = sideOf(e.player_self) || tgt || who;
      if (s) popups.push({ side: s, text: '呪い: ' + (CURSE_JA[e.curse] || e.curse), tone: 'curse' });
      addBeat({ pri: 56, ic: '☠️', tone: 'curse', cards: [], text: (s ? NAME[s] : '?') + 'に呪い「' + (CURSE_JA[e.curse] || e.curse) + '」' });
    });
    A('removeCurses').forEach((e) => {
      const s = sideOf(e.player_self) || who;
      if (s) popups.push({ side: s, text: '呪い解除', tone: 'heal' });
      addBeat({ pri: 52, ic: '✨', tone: 'heal', cards: [], text: (s ? NAME[s] : '?') + 'の呪いが解けた' });
    });
    A('setCurseOfEverybody').forEach((e) => addBeat({ pri: 54, ic: '☠️', tone: 'curse', cards: [], text: '全員に呪い「' + (CURSE_JA[e.curse] || e.curse) + '」' }));
    A('upgradeDisease').concat(A('upgradeHeaven'), A('disease')).forEach(() => addBeat({ pri: 40, ic: '🦠', tone: 'curse', cards: [], text: '病が進行' }));
    ['removeItems', 'removeUsedMiracles', 'removeSomething'].forEach((act) => A(act).forEach((e) => {
      const ms = e.models || e.faces || [];
      addBeat({ pri: 58, ic: '🗑️', tone: 'devil', cards: ms, text: (act === 'removeUsedMiracles' ? '使用済み奇跡' : '手札') + 'が破壊された' + (ms.length ? '（' + namesOf(ms) + '）' : '') });
    }));
    if (has('redraw') || has('replaceItems')) addBeat({ pri: 52, ic: '🔄', tone: 'sundry', cards: [], text: '手札の引き直し' });
    A('revive').forEach((e) => {
      const s = sideOf(e.player_self) || who;
      if (s) popups.push({ side: s, text: '復活!', tone: 'heal' });
      addBeat({ pri: 85, ic: '💫', tone: 'heal', cards: e.model ? [e.model] : [], text: (s ? NAME[s] : '?') + 'が復活' });
    });
    A('selfCurse').forEach(() => addBeat({ pri: 40, ic: '☠️', tone: 'curse', cards: [], text: '自分に呪い' }));

    // --- 死亡・終局 ---
    ['self', 'opp'].forEach((s) => {
      if (r[s].dead && !(p && p[s].dead)) {
        popups.push({ side: s, text: '昇天!', tone: 'die' });
        addBeat({ pri: 90, ic: '☠️', tone: s === 'self' ? 'loss' : 'win', cards: [], text: NAME[s] + 'が昇天' });
      }
    });
    let over = null;
    if (r.is_over || has('endGame')) {
      const res = finalResult(V, r);
      over = res;
      const t = res === 'win' ? 'AI の勝利！' : res === 'loss' ? 'AI の敗北…' : '引き分け';
      addBeat({ pri: 100, ic: res === 'win' ? '🏆' : res === 'loss' ? '💀' : '🤝', tone: res, cards: [], text: '対局終了 — ' + t });
      stage.center = { type: 'over', result: res };
    }

    // --- 配札・手番 ---
    if (has('startGame')) addBeat({ pri: 30, ic: '🎴', tone: 'idle', cards: [], text: '対局開始 — 手札が配られた' });
    const gifts = A('gift').filter((e) => e.player_self === true && e.faces && e.faces.length);
    if (gifts.length && !has('startGame')) {
      const gf = [].concat.apply([], gifts.map((e) => e.faces));
      addBeat({ pri: 20, ic: '🎁', tone: 'gift', cards: gf, text: 'AI が札を獲得（' + namesOf(gf) + '）' });
    }
    const adv = A('advanceGF')[0];
    if (adv) {
      const s = sideOf(adv.player_self);
      addBeat({ pri: 10, ic: '⏩', tone: 'idle', cards: [], text: (s ? NAME[s] : '?') + 'の手番へ（GF ' + r.gf + '）' });
    }
    if (!beats.length) addBeat({ pri: 5, ic: '⏳', tone: 'idle', cards: [], text: '状態が更新された（GF ' + r.gf + '）' });

    // --- 矢印（攻撃の向き） ---
    const atk = newAttack || carried;
    if (atk && atk.src) {
      const to = tgt || other(atk.src);
      stage.arrow = { from: atk.src, to, power: isNum(atk.power) && atk.power > 0 ? atk.power : sumAtk(atk.faces || []) };
    }

    // --- ポップアップ（差分） ---
    ['self', 'opp'].forEach((s) => {
      const d = diff[s];
      if (isNum(d.hp)) popups.push({ side: s, text: sgn(d.hp) + ' HP', tone: d.hp < 0 ? 'dmg' : 'heal' });
      if (isNum(d.mp)) popups.push({ side: s, text: sgn(d.mp) + ' MP', tone: 'mp' });
      if (isNum(d.cp)) popups.push({ side: s, text: sgn(d.cp) + ' ¥', tone: d.cp > 0 ? 'cp' : 'cpneg' });
    });
    if (dmgTotal > 0 && !popups.some((x) => x.tone === 'dmg')) {
      const s = tgt || (who ? other(who) : null);
      if (s) popups.push({ side: s, text: '-' + dmgTotal + ' HP', tone: 'dmg' });
    }
    const defender = who || tgt;
    if (marks.has('miss')) popups.push({ side: tgt || defender, text: 'MISS', tone: 'miss' });
    if (marks.has('block') || marks.has('safe')) popups.push({ side: defender, text: '防御成功!', tone: 'guard' });
    if (marks.has('reflect')) popups.push({ side: defender, text: '反射!', tone: 'reflect' });
    if (marks.has('bounce')) popups.push({ side: defender, text: '跳ね返し!', tone: 'reflect' });
    if (marks.has('counterAttack')) popups.push({ side: defender, text: '反撃!', tone: 'dmg' });
    const seenPop = new Set();
    const pops = popups.filter((x) => {
      if (!x.side) return false;
      const k = x.side + '|' + x.text;
      if (seenPop.has(k)) return false;
      seenPop.add(k);
      return true;
    });

    const primary = beats.slice().sort((a, b) => b.pri - a.pri)[0];
    const details = beats.filter((b) => b !== primary && b.pri >= 40).slice(0, 3);
    stage.tone = primary.tone;
    return {
      type: 'update', idx, row: r, st: r, stIdx: idx, prev: p, who, tgt, beats, primary, details, stage,
      popups: pops, newHand: newHandIdx(r, p), chosen: new Set(), over,
      logCards: [].concat.apply([], beats.filter((b) => b.pri >= 45).map((b) => b.cards || [])).slice(0, 5),
      logText: beats.filter((b) => b.pri >= 45).map((b) => b.text).slice(0, 3).join(' ／ ') || primary.text,
    };
  }

  function describeAction(a) {
    const k = KIND_JA[a.kind] || a.kind;
    const names = namesOf(a.models || []);
    let t = k + (names ? ' ' + names : '');
    if (a.allocation) t += '（HP' + a.allocation.hp + ' / MP' + a.allocation.mp + ' / ¥' + a.allocation.cp + '）';
    if (a.target_self === true) t += ' → 自分';
    else if (a.target_self === false) t += ' → 相手';
    return t;
  }

  function analyzeDecision(V, idx) {
    const rows = V.rows, d = rows[idx];
    const si = updateAt(rows, idx);
    const st = si >= 0 ? rows[si] : null;
    const chosen = new Set();
    if (st) {
      const used = new Set();
      (d.action.models || []).forEach((m) => {
        let k = st.hand.findIndex((h, i) => h[0] === m && !h[2] && !used.has(i));
        if (k < 0 && d.dream_played) k = st.hand.findIndex((h, i) => h[1] === 1 && !h[2] && !used.has(i));
        if (k >= 0) { used.add(k); chosen.add(k); }
      });
    }
    const probs = d.value_probs;
    const conf = isNum(d.action_logp) ? Math.exp(d.action_logp) : null;
    const desc = describeAction(d.action);
    const beat = { pri: 60, ic: '🤔', tone: 'ai', cards: d.action.models || [], text: 'AI の選択: ' + desc };
    const zones = { self: [], opp: [] };
    const stage = { zones, center: null, arrow: null, fx: null, result: null, tone: 'ai' };
    if (st) {
      if (st.pending) {
        const src = sideOf(st.pending.source_self);
        if (src) {
          if (st.pending.guardian && st.pending.guardian !== 'none') zones[src].push({ token: 'guardian', guardian: st.pending.guardian, carried: true });
          (st.pending.models || []).forEach((m) => zones[src].push({ id: m, role: 'attack', carried: true }));
          const to = st.pending.target_self ? 'self' : 'opp';
          stage.arrow = { from: src, to, power: st.pending.power > 0 ? st.pending.power : sumAtk(st.pending.models || []) };
        }
      }
      if (st.purchase) stage.center = { type: 'offer', id: st.purchase.model, seller: st.purchase.seller_self ? 'self' : 'opp', buyer: st.purchase.buyer_self ? 'self' : 'opp' };
    }
    (d.action.models || []).forEach((m, i) => zones.self.push({ id: m, role: 'ghost', ghost: true, delay: i * 90 }));
    if (d.action.allocation) zones.self.push({ token: 'alloc', alloc: d.action.allocation, ghost: true });
    if (!d.action.models.length && !d.action.allocation) zones.self.push({ token: 'kind', text: KIND_JA[d.action.kind] || d.action.kind, ghost: true });
    return {
      type: 'decision', idx, row: d, st, stIdx: si, prev: null, who: 'self', tgt: null,
      beats: [beat], primary: beat, details: [], stage, popups: [], newHand: new Set(), chosen,
      pick: { desc, winProb: probs ? probs[0] : null, conf, probs },
      logCards: d.action.models || [], logText: 'AI の選択: ' + desc,
    };
  }

  function analyze(V, idx) {
    const r = V.rows[idx];
    if (!r) return null;
    return r.ev === 'decision' ? analyzeDecision(V, idx) : analyzeUpdate(V, idx);
  }

  /* ===================================================================
   * 5. 描画: 席（HP/MP/¥）
   * =================================================================== */
  const GAUGES = [['hp', 'HP'], ['mp', 'MP'], ['cp', '¥']];
  function seatSkeleton(side) {
    const self = side === 'self';
    const stack = self ? '' :
      '<div class="facedown" title="相手の手札は非公開です"><div class="fd-stack">' + fakeIcon('fd-card') + fakeIcon('fd-card') + fakeIcon('fd-card') +
      '</div><small>手札 非公開</small></div>';
    return '<div class="seat-fx"></div>' +
      '<div class="seat-head">' +
      '<div class="avatar">' + (self ? '🤖' : '🧑') + '</div>' +
      '<div class="seat-id"><div class="seat-name">' + esc(FULLNAME[side]) + ' <span class="turn-tag" data-r="turn"></span></div>' +
      '<div class="seat-sub" data-r="sub"></div></div>' + stack + '</div>' +
      '<div class="gauges">' + GAUGES.map(([k, l]) =>
        '<div class="gauge g-' + k + '" data-k="' + k + '"><span class="g-label">' + l + '</span>' +
        '<div class="g-bar"><div class="g-ghost"></div><div class="g-fill"></div></div><b class="g-val">—</b></div>').join('') +
      '</div>' +
      '<div class="seat-status" data-r="status"></div>' +
      '<div class="seat-used" data-r="used"></div>' +
      '<div class="pops" data-r="pops"></div>';
  }
  function buildSeats() {
    ['self', 'opp'].forEach((s) => { $('#seat-' + s).innerHTML = seatSkeleton(s); });
  }

  const tweens = new WeakMap();
  function tweenNumber(el, to, animate) {
    const from = isNum(+el.dataset.v) && el.dataset.v !== '' ? +el.dataset.v : to;
    el.dataset.v = to;
    const old = tweens.get(el);
    if (old) cancelAnimationFrame(old);
    if (!animate || from === to || reduceMotion.matches) { el.textContent = to; return; }
    const t0 = performance.now(), dur = 600;
    const tick = (now) => {
      const k = clamp((now - t0) / dur, 0, 1);
      const e = 1 - Math.pow(1 - k, 3);
      el.textContent = Math.round(from + (to - from) * e);
      if (k < 1) tweens.set(el, requestAnimationFrame(tick));
    };
    tweens.set(el, requestAnimationFrame(tick));
  }

  function updateSeat(side, pl, a, V, animate) {
    const root = $('#seat-' + side);
    if (!root) return;
    root.classList.toggle('is-dead', !!(pl && pl.dead));
    root.classList.toggle('is-fogged', !!(pl && pl.fogged));
    GAUGES.forEach(([k]) => {
      const g = $('.g-' + k, root);
      const val = pl ? pl[k] : null;
      const fill = $('.g-fill', g), ghost = $('.g-ghost', g), num = $('.g-val', g);
      if (val == null) {
        g.classList.add('fogged');
        num.textContent = '？';
        delete num.dataset.v;
        return;
      }
      g.classList.remove('fogged');
      const w = clamp(val / V.scale[k] * 100, 0, 100) + '%';
      fill.style.width = w;
      ghost.style.width = w;
      g.classList.toggle('low', k === 'hp' && val <= V.scale.hp * 0.25);
      tweenNumber(num, val, animate);
    });
    // 手番タグ
    const st = a.st;
    const turn = $('[data-r="turn"]', root);
    let tag = '';
    if (st && !st.is_over && st.actor_self != null && sideOf(st.actor_self) === side) {
      tag = st.phase === 1 ? '応答中' : st.phase === 3 ? '購入判断' : '手番';
    }
    turn.textContent = tag;
    turn.classList.toggle('on', !!tag);
    root.classList.toggle('is-turn', !!tag);
    // 守護神
    const sub = $('[data-r="sub"]', root);
    const g = pl && pl.guardian && pl.guardian !== 'none' ? pl.guardian : null;
    sub.innerHTML = g ? '<span class="guardian">' + guardianIcon(g, 'gico') + '守護神 ' + esc(GUARDIAN_JA[g] || g) + '</span>' :
      '<span class="muted">守護神なし</span>';
    // 状態異常
    const chips = [];
    if (pl) {
      (pl.curses || []).forEach((c) => chips.push('<span class="chip-s curse" title="呪い: ' + esc(CURSE_JA[c] || c) + '">' + curseIcon(c, 'cico') + esc(CURSE_JA[c] || c) + '</span>'));
      if (pl.confused) chips.push('<span class="chip-s confused" title="混乱">💫 混乱</span>');
      if (pl.fogged) chips.push('<span class="chip-s fog" title="霧のため数値が見えません">' + curseIcon('fog', 'cico') + '霧 (数値不明)</span>');
      if (pl.dead) chips.push('<span class="chip-s dead">☠️ 昇天</span>');
    }
    $('[data-r="status"]', root).innerHTML = chips.join('');
    // 使用済み奇跡
    const used = (pl && pl.used) || [];
    $('[data-r="used"]', root).innerHTML = used.length ?
      '<span class="used-label">使用済み奇跡</span>' + used.map((id) => '<span class="mini" data-cid="' + id + '">' + cardIcon(id, 'mini-ico') + '</span>').join('') : '';
  }
  function renderSeats(a, V, animate) {
    const st = a.st;
    updateSeat('self', st ? st.self : null, a, V, animate);
    updateSeat('opp', st ? st.opp : null, a, V, animate);
  }
  function retrigger(el, cls, ms) {
    el.classList.remove(cls);
    void el.offsetWidth;
    el.classList.add(cls);
    setTimeout(() => el.classList.remove(cls), ms);
  }
  function fireFx(a) {
    if (!a.popups || !a.popups.length) return;
    const hasCards = a.stage.zones.self.length + a.stage.zones.opp.length > 0;
    const base = hasCards ? 700 : 150;
    const perSide = { self: 0, opp: 0 };
    a.popups.forEach((p) => {
      const root = $('#seat-' + p.side);
      if (!root) return;
      const n = perSide[p.side]++;
      const d = base + n * 170;
      const el = document.createElement('div');
      el.className = 'pop pop-' + p.tone;
      el.style.left = clamp(18 + n * 20, 10, 80) + '%';
      el.style.animationDelay = d + 'ms';
      el.innerHTML = (p.guardian ? guardianIcon(p.guardian, 'pop-ico') : '') + '<span>' + esc(p.text) + '</span>';
      $('[data-r="pops"]', root).appendChild(el);
      setTimeout(() => el.remove(), d + 1700);
      const fx = { dmg: 'fx-hit', heal: 'fx-heal', mp: 'fx-mp', cp: 'fx-cp', guard: 'fx-guard', die: 'fx-die', reflect: 'fx-guard', curse: 'fx-curse' }[p.tone];
      if (fx) setTimeout(() => retrigger(root, fx, 1000), d);
    });
  }

  /* ===================================================================
   * 6. 描画: 手札
   * =================================================================== */
  function renderHand(a, animate) {
    const st = a.st;
    const el = $('#hand');
    if (!st) { el.innerHTML = '<div class="empty">対局データを待っています…</div>'; $('#handCount').textContent = ''; return; }
    $('#handCount').textContent = st.hand.length + ' / 18 枚';
    el.innerHTML = st.hand.map((h, i) => {
      const [face, dream, used] = h;
      const c = card(face);
      const cls = ['hcard', elemCls(c)];
      if (used) cls.push('used');
      if (a.chosen.has(i)) cls.push('chosen');
      if (animate && a.newHand.has(i)) cls.push('new');
      return '<div class="' + cls.join(' ') + '" data-cid="' + face + '" data-dream="' + dream + '" style="--i:' + i + '">' +
        '<div class="hc-img">' + cardIcon(face, 'hc-ico') + (dream ? '<span class="dream-badge">夢</span>' + fakeIcon('fake-ind') : '') +
        (used ? '<span class="used-badge">済</span>' : '') + '</div>' +
        '<div class="hc-name">' + esc(cname(face)) + '</div>' +
        '<div class="hc-chips">' + statChips(c) + '</div>' +
        (a.chosen.has(i) ? '<span class="pick-badge">AI選択</span>' : '') + '</div>';
    }).join('') || '<div class="empty">手札なし</div>';
  }

  /* ===================================================================
   * 7. 描画: ステージ（中央の対戦エリア）
   * =================================================================== */
  function stageItemHtml(it, side) {
    const fly = it.carried ? 'settled' : side === 'self' ? 'fly-bottom' : 'fly-top';
    const dly = ' style="animation-delay:' + (it.delay || 0) + 'ms"';
    if (it.token === 'pass') return '<div class="stoken pass ' + fly + '"' + dly + '><b>🙅</b><span>防御なし</span></div>';
    if (it.token === 'guardian') {
      return '<div class="stoken guardian ' + fly + '"' + dly + '>' + guardianIcon(it.guardian, 'gbig') +
        '<span>守護神 ' + esc(GUARDIAN_JA[it.guardian] || it.guardian) + '</span></div>';
    }
    if (it.token === 'alloc') {
      const al = it.alloc;
      return '<div class="stoken alloc ' + fly + (it.ghost ? ' ghost' : '') + '"' + dly + '><b>💱 配分</b><span>HP ' + al.hp + ' ・ MP ' + al.mp + ' ・ ¥' + al.cp + '</span></div>';
    }
    if (it.token === 'kind') return '<div class="stoken kind ghost ' + fly + '"' + dly + '><b>🤔</b><span>' + esc(it.text) + '</span></div>';
    const c = card(it.id);
    const badges = [];
    if (it.role === 'attack' && c && c.atk > 0) badges.push('<i class="sb atk">攻' + (c.plus ? '+' : '') + c.atk + '</i>');
    else if (it.role === 'defense' && c && c.defense > 0) badges.push('<i class="sb def">防' + c.defense + '</i>');
    else if (c) {
      const ab = abilityText(c);
      if (ab) badges.push('<i class="sb use">' + esc(ab) + '</i>');
      else if (c.atk > 0) badges.push('<i class="sb atk">攻' + c.atk + '</i>');
    }
    return '<div class="scard ' + elemCls(c) + ' role-' + esc(it.role) + (it.ghost ? ' ghost' : '') + ' ' + fly + '"' + dly +
      ' data-cid="' + it.id + '"><div class="sc-img">' + cardIcon(it.id, 'sc-ico') + '</div><div class="sc-name">' + esc(cname(it.id)) +
      '</div><div class="sc-badges">' + badges.join('') + '</div></div>';
  }

  function centerHtml(a) {
    const s = a.stage;
    const parts = [];
    if (s.center && s.center.type === 'offer') {
      const c = card(s.center.id);
      const roles = s.center.seller ? '<small>' + NAME[s.center.seller] + ' → ' + NAME[s.center.buyer] + '</small>' : '';
      parts.push('<div class="offer' + (s.center.done ? ' done' : '') + ' ' + elemCls(c) + '" data-cid="' + s.center.id + '"><div class="of-img">' + cardIcon(s.center.id, 'sc-ico') +
        '</div><div class="of-info"><b>' + esc(cname(s.center.id)) + '</b>' + (c && c.price ? '<i class="sb cp">¥' + c.price + '</i>' : '') + roles + '</div></div>');
    } else if (s.center && (s.center.type === 'devil' || s.center.type === 'phenomenon')) {
      const c = card(s.center.id);
      parts.push('<div class="offer ' + s.center.type + ' ' + elemCls(c) + '" data-cid="' + s.center.id + '"><div class="of-img">' + cardIcon(s.center.id, 'sc-ico') +
        '</div><div class="of-info"><b>' + esc(cname(s.center.id)) + '</b><small>' + (s.center.type === 'devil' ? '😈 悪魔カード' : '🌀 現象') + '</small></div></div>');
    } else if (s.center && s.center.type === 'over') {
      const r = s.center.result;
      parts.push('<div class="over over-' + r + '">' + (r === 'win' ? '🏆 VICTORY' : r === 'loss' ? '💀 DEFEAT' : '🤝 DRAW') + '</div>');
    }
    if (s.arrow) {
      const dir = s.arrow.from === s.arrow.to ? 'self' : s.arrow.to === 'opp' ? 'up' : 'down';
      parts.push('<div class="arrow dir-' + dir + '" title="攻撃の向き"><span>' + (dir === 'self' ? '↻' : '➤') + '</span><span>' + (dir === 'self' ? '' : '➤') +
        '</span>' + (s.arrow.power > 0 ? '<em>攻' + s.arrow.power + '</em>' : '') + '</div>');
    }
    if (s.fx === 'shield') parts.push('<div class="clash-ic shield">🛡️</div>');
    else if (s.fx === 'impact') parts.push('<div class="clash-ic impact">💥</div>');
    else if (s.fx === 'reflect' || s.fx === 'bounce') parts.push('<div class="clash-ic reflect">↩️</div>');
    return parts.join('');
  }

  function renderStage(a, V, animate) {
    const el = $('#stage');
    const s = a.stage;
    el.className = 'stage tone-' + s.tone + (animate ? '' : ' static') + (s.fx ? ' fx-' + s.fx : '');
    const prim = a.primary;
    const gIcon = prim.guardian ? guardianIcon(prim.guardian, 'b-gico') : '';
    const detail = a.details.length ? '<div class="b-detail">' + a.details.map((d) => '<span>' + esc(d.ic) + ' ' + esc(d.text) + '</span>').join('') + '</div>' : '';
    let pick = '';
    if (a.type === 'decision') {
      const p = a.pick;
      pick = '<div class="pickpill"><span class="pp-tag">🧠 AIの選択</span><b>' + esc(p.desc) + '</b>' +
        (isNum(p.winProb) ? '<span class="pp-stat">勝率 <b>' + fmtPct(p.winProb) + '</b></span>' : '') +
        (isNum(p.conf) ? '<span class="pp-stat">確信度 <b>' + fmtPct(p.conf, 0) + '</b></span>' : '') + '</div>';
    }
    const top = s.zones.opp.map((it) => stageItemHtml(it, 'opp')).join('');
    const bot = s.zones.self.map((it) => stageItemHtml(it, 'self')).join('');
    el.innerHTML =
      '<div class="banner tone-' + esc(prim.tone) + '"><span class="b-ic">' + esc(prim.ic) + '</span>' + gIcon + '<span class="b-tx">' + esc(prim.text) + '</span></div>' +
      detail + pick +
      '<div class="arena-core">' +
      '<div class="zone zone-opp"><span class="zone-label">相手 (人間)</span><div class="zone-items">' + (top || '<span class="zone-empty">—</span>') + '</div></div>' +
      '<div class="clash">' + centerHtml(a) + '</div>' +
      '<div class="zone zone-self"><span class="zone-label">AI (m-xit1)</span><div class="zone-items">' + (bot || '<span class="zone-empty">—</span>') + '</div></div>' +
      '</div>' +
      '<div class="stage-foot"><span>GF ' + (a.st ? a.st.gf : '—') + (a.st && a.st.tiebreak_gf ? ' / ' + a.st.tiebreak_gf : '') + '</span><span>' +
      (a.st ? esc(PHASE_JA[a.st.phase] || '') : '') + '</span><span>更新 #' + (a.st ? a.st.update : '—') + '</span></div>';
  }
  function renderEmpty(V) {
    $('#stage').className = 'stage tone-idle static';
    $('#stage').innerHTML = '<div class="banner tone-idle"><span class="b-ic">⏳</span><span class="b-tx">' +
      (V.game == null ? '対局データを待っています…' : '対局 #' + V.game + ' の記録を読み込み中…') + '</span></div>' +
      '<div class="arena-core"><div class="zone zone-opp"><div class="zone-items"><span class="zone-empty">—</span></div></div><div class="clash"></div><div class="zone zone-self"><div class="zone-items"><span class="zone-empty">—</span></div></div></div>';
    $('#hand').innerHTML = '<div class="empty">対局データを待っています…</div>';
    $('#handCount').textContent = '';
    ['self', 'opp'].forEach((s) => updateSeat(s, null, { st: null }, V, false));
  }

  /* ===================================================================
   * 8. 描画: 実況ログ・対局情報
   * =================================================================== */
  function logEntryHtml(V, i) {
    const r = V.rows[i];
    const a = analyze(V, i);
    if (!a) return '';
    const side = a.type === 'decision' ? 'self' : (a.who || 'sys');
    const thumbs = (a.logCards || []).map((id) => '<span class="lthumb" data-cid="' + id + '">' + cardIcon(id, 'lico') + '</span>').join('');
    const win = a.type === 'decision' && a.pick && isNum(a.pick.winProb) ? '<span class="lwin">' + fmtPct(a.pick.winProb, 0) + '</span>' : '';
    const sideLabel = side === 'sys' ? '' : '<span class="lw lw-' + side + '">' + NAME[side] + '</span>';
    return '<li class="li li-' + a.type + ' tone-' + esc(a.primary.tone) + '" data-idx="' + i + '"><span class="lgf">GF' + r.gf + '</span>' + sideLabel +
      '<span class="lic">' + esc(a.primary.ic) + '</span><span class="ltx">' + esc(a.logText) + '</span>' + thumbs + win + '</li>';
  }
  function renderLog(V, idx) {
    const key = S.mode + ':' + V.game;
    const ol = $('#log');
    const replay = S.mode === 'replay';
    if (replay) {
      // 棋譜モードは全行を一度だけ作り、現在位置より後ろを薄く表示する（スクラブが軽い）。
      if (S.logKey !== key) {
        const out = [];
        for (let i = 0; i < V.rows.length; i++) out.push(logEntryHtml(V, i));
        ol.innerHTML = out.join('');
      }
      $$('.li', ol).forEach((li, i) => { li.classList.toggle('future', i > idx); li.classList.toggle('cur', i === idx); });
    } else {
      if (S.logKey !== key || idx !== S.logUpTo + 1) {
        const out = [];
        for (let i = 0; i <= idx; i++) out.push(logEntryHtml(V, i));
        ol.innerHTML = out.join('');
      } else {
        ol.insertAdjacentHTML('beforeend', logEntryHtml(V, idx));
      }
      $$('.li.cur', ol).forEach((e) => e.classList.remove('cur'));
      if (ol.lastElementChild) ol.lastElementChild.classList.add('cur');
    }
    S.logKey = key;
    S.logUpTo = idx;
    const curEl = ol.querySelector('.li.cur');
    if (curEl) {
      const top = curEl.offsetTop; // .log は position:relative
      if (replay) ol.scrollTop = Math.max(0, top - ol.clientHeight / 2);
      else ol.scrollTop = ol.scrollHeight;
    }
    $('#logCount').textContent = idx >= 0 ? (idx + 1) + ' / ' + V.rows.length + ' 行' : '';
  }

  function resultOfMeta(m) {
    if (!m) return { k: 'none', t: '—' };
    if (m.klass === 'in_progress') return { k: 'live', t: '進行中' };
    if (m.outcome === 'win') return { k: 'win', t: '勝ち' };
    if (m.outcome === 'loss') return { k: 'loss', t: '負け' };
    if (m.outcome === 'draw') return { k: 'draw', t: '引き分け' };
    return { k: 'stop', t: '停止' };
  }
  function renderGameInfo(V, a) {
    const m = V.meta;
    const rs = resultOfMeta(m);
    const rows = [];
    rows.push(['対局', V.game != null ? '#' + V.game : '—']);
    rows.push(['状態', '<span class="res res-' + rs.k + '">' + esc(rs.t) + '</span>']);
    if (m) {
      rows.push(['先攻', m.first_mover === 'self' ? 'AI' : m.first_mover === 'opponent' ? '相手' : '—']);
      rows.push(['相手レート', fmtInt(m.opp_before)]);
      rows.push(['AI レート', fmtInt(m.self_before) + (isNum(m.self_after) ? ' → ' + m.self_after + (isNum(m.self_delta) ? ' (' + sgn(m.self_delta) + ')' : '') : '')]);
      rows.push(['判断数', m.decisions != null ? m.decisions : '—']);
      rows.push(['所要時間', fmtDur(m.duration_s)]);
      if (m.klass !== 'completed' && m.klass !== 'in_progress' && m.stop_reason) rows.push(['停止理由', STOP_JA[m.stop_reason] || m.stop_reason]);
    }
    if (a && a.st) {
      rows.push(['現在', 'GF ' + a.st.gf + (a.st.tiebreak_gf ? ' / ' + a.st.tiebreak_gf : '') + ' ・ ' + esc(PHASE_JA[a.st.phase] || '')]);
    }
    $('#gameInfo').innerHTML = '<dl>' + rows.map((r) => '<div><dt>' + esc(r[0]) + '</dt><dd>' + (typeof r[1] === 'string' && r[1].indexOf('<span') === 0 ? r[1] : esc(r[1])) + '</dd></div>').join('') + '</dl>';
  }

  /* ===================================================================
   * 9. チャート共通 + 勝率チャート
   * =================================================================== */
  function pathFrom(pts) { return pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' '); }

  function bindHover(wrap, W, getPts, tipFn, onClick) {
    const svg = $('svg', wrap), tip = $('.chart-tip', wrap);
    if (!svg || !tip) return;
    const cross = $('.cross', svg), dot = $('.hdot', svg);
    const nearest = (ev) => {
      const pts = getPts();
      if (!pts.length) return null;
      const r = svg.getBoundingClientRect();
      const x = (ev.clientX - r.left) / r.width * W;
      let best = 0, bd = Infinity;
      pts.forEach((p, i) => { const d = Math.abs(p.x - x); if (d < bd) { bd = d; best = i; } });
      return pts[best];
    };
    svg.onpointermove = (ev) => {
      const p = nearest(ev);
      if (!p) return;
      cross.setAttribute('x1', p.x); cross.setAttribute('x2', p.x);
      dot.setAttribute('cx', p.x); dot.setAttribute('cy', p.y);
      cross.style.display = dot.style.display = '';
      tip.innerHTML = tipFn(p);
      tip.hidden = false;
      const r = svg.getBoundingClientRect();
      const px = p.x / W * r.width;
      tip.style.left = clamp(px, 70, r.width - 70) + 'px';
    };
    svg.onpointerleave = () => { tip.hidden = true; cross.style.display = dot.style.display = 'none'; };
    svg.onclick = onClick ? (ev) => { const p = nearest(ev); if (p) onClick(p); } : null;
  }

  function renderWinChart(V, idx) {
    const wrap = $('#winChart');
    const decs = [];
    V.rows.forEach((r, i) => {
      if (r.ev === 'decision' && r.value_probs && (S.mode === 'replay' || i <= idx)) decs.push({ r, i });
    });
    if (!decs.length) { wrap.innerHTML = '<div class="empty">AI の判断が記録されるとここに勝率が表示されます</div>'; return; }
    const W = 340, H = 150, m = { l: 30, r: 10, t: 10, b: 20 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const n = decs.length;
    const X = (k) => m.l + (n === 1 ? iw / 2 : k / (n - 1) * iw);
    const Y = (p) => m.t + (1 - p) * ih;
    const mid = Y(0.5);
    const pts = decs.map((d, k) => ({ x: X(k), y: Y(d.r.value_probs[0]), p: d.r.value_probs[0], d, k }));
    const line = pathFrom(pts.map((p) => [p.x, p.y]));
    const area = line + ' L' + pts[n - 1].x.toFixed(1) + ' ' + mid.toFixed(1) + ' L' + pts[0].x.toFixed(1) + ' ' + mid.toFixed(1) + ' Z';
    let curPt = null;
    pts.forEach((p) => { if (p.d.i <= idx) curPt = p; });
    const last = pts[n - 1];
    const color = (p) => (p >= 0.5 ? 'var(--win)' : 'var(--loss)');
    wrap.innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" class="svg-chart" role="img" aria-label="AI勝率の推移">' +
      '<defs><clipPath id="wcA"><rect x="0" y="0" width="' + W + '" height="' + mid.toFixed(1) + '"/></clipPath>' +
      '<clipPath id="wcB"><rect x="0" y="' + mid.toFixed(1) + '" width="' + W + '" height="' + H + '"/></clipPath></defs>' +
      [0, 0.25, 0.5, 0.75, 1].map((v) => '<line class="grid" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(v).toFixed(1) + '" y2="' + Y(v).toFixed(1) + '"/>' +
        '<text class="ax" x="' + (m.l - 5) + '" y="' + (Y(v) + 3).toFixed(1) + '" text-anchor="end">' + Math.round(v * 100) + '</text>').join('') +
      '<path d="' + area + '" class="area-up" clip-path="url(#wcA)"/><path d="' + area + '" class="area-dn" clip-path="url(#wcB)"/>' +
      '<line class="base" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + mid.toFixed(1) + '" y2="' + mid.toFixed(1) + '"/>' +
      '<path d="' + line + '" class="wline"/>' +
      (curPt ? '<line class="curline" x1="' + curPt.x.toFixed(1) + '" x2="' + curPt.x.toFixed(1) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/>' +
        '<circle class="curdot" cx="' + curPt.x.toFixed(1) + '" cy="' + curPt.y.toFixed(1) + '" r="4.5" style="fill:' + color(curPt.p) + '"/>' : '') +
      '<line class="cross" y1="' + m.t + '" y2="' + (H - m.b) + '" style="display:none"/><circle class="hdot" r="3.5" style="display:none"/>' +
      '<text class="ax" x="' + m.l + '" y="' + (H - 5) + '">1手目</text><text class="ax" x="' + (W - m.r) + '" y="' + (H - 5) + '" text-anchor="end">' + n + '手目</text>' +
      '</svg><div class="chart-tip" hidden></div>' +
      '<div class="win-now"><span>現在の勝率予測</span><b style="color:' + color((curPt || last).p) + '">' + fmtPct((curPt || last).p) + '</b></div>';
    bindHover(wrap, W, () => pts, (p) => {
      const r = p.d.r;
      return '<b>判断 #' + r.decision_idx + '</b> ・ GF ' + r.gf + '<br>' + esc(describeAction(r.action)) +
        '<br>勝率 <b>' + fmtPct(p.p) + '</b>' + (r.value_probs ? '　負 ' + fmtPct(r.value_probs[1], 0) : '');
    }, S.mode === 'replay' ? (p) => { seek(p.d.i, false); } : null);
  }

  /* ===================================================================
   * 10. 搬送ボタン（トランスポート）
   * =================================================================== */
  function buildTransport() {
    const el = $('#transport');
    if (S.mode === 'live') {
      el.className = 'transport t-live';
      el.innerHTML =
        '<div class="t-group"><span class="t-title">ライブ観戦</span><span id="tBehind" class="behind" hidden></span></div>' +
        '<div class="t-group"><span class="t-label">再生速度</span><div class="seg" id="tSpeed">' +
        LIVE_SPEEDS.map((s) => '<button data-act="speed" data-v="' + s + '">' + s + 'x</button>').join('') + '</div>' +
        '<button class="btn btn-gold" data-act="latest" id="tLatest" hidden>⏭ 最新へ</button></div>';
    } else {
      el.className = 'transport t-replay';
      el.innerHTML =
        '<div class="t-group t-ctrl">' +
        '<button class="btn" data-act="first" title="最初へ">⏮</button>' +
        '<button class="btn" data-act="prev" title="1つ戻る (←)">◀</button>' +
        '<button class="btn btn-play" data-act="play" id="tPlay">▶ 再生</button>' +
        '<button class="btn" data-act="next" title="1つ進む (→)">▶</button>' +
        '<button class="btn" data-act="last" title="最後へ">⏭</button></div>' +
        '<input id="tScrub" class="scrub" type="range" min="0" max="0" value="0" step="1" aria-label="再生位置">' +
        '<span id="tPos" class="t-pos">0 / 0</span>' +
        '<div class="t-group"><div class="seg" id="tSpeed">' +
        REPLAY_SPEEDS.map((s) => '<button data-act="speed" data-v="' + s + '">' + s + 'x</button>').join('') + '</div>' +
        '<button class="btn btn-live" data-act="golive">● ライブに戻る</button></div>';
      const sc = $('#tScrub');
      sc.addEventListener('input', () => { stopPlay(); seek(+sc.value, false); });
    }
  }
  function updateTransport(V) {
    $$('#tSpeed button').forEach((b) => b.classList.toggle('on', +b.dataset.v === S.speed[S.mode]));
    if (S.mode === 'live') {
      const behind = V.rows.length - 1 - V.cursor;
      const bh = $('#tBehind'), lt = $('#tLatest');
      if (bh) { bh.hidden = behind <= 0; bh.textContent = '再生待ち あと ' + behind + ' 手'; }
      if (lt) lt.hidden = behind <= 1;
    } else {
      const sc = $('#tScrub');
      if (sc) { sc.max = Math.max(0, V.rows.length - 1); sc.value = Math.max(0, V.cursor); }
      const pos = $('#tPos');
      if (pos) pos.textContent = (V.cursor + 1) + ' / ' + V.rows.length;
      const pl = $('#tPlay');
      if (pl) pl.textContent = S.playing ? '⏸ 一時停止' : '▶ 再生';
    }
  }

  /* ===================================================================
   * 11. 再生コア
   * =================================================================== */
  function recomputeScale(V) {
    const sc = { hp: 40, mp: 20, cp: 30 };
    V.rows.forEach((r) => {
      if (r.ev !== 'update') return;
      ['self', 'opp'].forEach((s) => ['hp', 'mp', 'cp'].forEach((k) => { if (isNum(r[s][k])) sc[k] = Math.max(sc[k], r[s][k]); }));
    });
    V.scale = { hp: Math.ceil(sc.hp / 10) * 10, mp: Math.ceil(sc.mp / 10) * 10, cp: Math.ceil(sc.cp / 10) * 10 };
  }

  // 1ステップ表示。animate=true のときだけカード飛翔・ポップアップを出す。
  function showRow(V, idx, animate) {
    idx = clamp(idx, -1, V.rows.length - 1);
    V.cursor = idx;
    if (V !== view() || S.tab !== 'live') return;
    if (idx < 0) { renderEmpty(V); renderLog(V, -1); renderWinChart(V, -1); renderGameInfo(V, null); updateTransport(V); updateHeader(); return; }
    const a = analyze(V, idx);
    S.lastStepKind = a.type;
    renderSeats(a, V, animate);
    renderHand(a, animate);
    renderStage(a, V, animate);
    if (animate) fireFx(a);
    renderLog(V, idx);
    renderWinChart(V, idx);
    renderGameInfo(V, a);
    updateTransport(V);
    updateHeader();
  }
  function seek(idx, animate) {
    const V = view();
    showRow(V, idx, animate);
  }

  /* --- ライブ再生キュー --- */
  function kickLive() {
    if (S.liveTimer || S.mode !== 'live') return;
    liveTick();
  }
  function liveTick() {
    S.liveTimer = null;
    const V = S.live;
    if (S.mode !== 'live') return;
    const behind = V.rows.length - 1 - V.cursor;
    if (behind <= 0) { updateTransport(V); return; }
    const next = V.cursor + 1;
    const kind = V.rows[next].ev;
    const animate = S.tab === 'live' && behind <= 12;
    showRow(V, next, animate);
    let wait = (kind === 'decision' ? STEP_MS.decision : STEP_MS.update) / S.speed.live;
    if (behind > 12) wait = 80;
    else if (behind > 6) wait *= 0.35;
    S.liveTimer = setTimeout(liveTick, wait);
  }
  function stopLiveTimer() { if (S.liveTimer) { clearTimeout(S.liveTimer); S.liveTimer = null; } }
  function jumpLatest() {
    const V = S.live;
    stopLiveTimer();
    showRow(V, V.rows.length - 1, false);
  }

  /* --- リプレイ再生 --- */
  function startPlay() {
    const V = S.replay;
    if (!V.rows.length) return;
    if (V.cursor >= V.rows.length - 1) showRow(V, -1, false);
    S.playing = true;
    const tick = () => {
      S.playTimer = null;
      if (!S.playing || S.mode !== 'replay') return;
      if (V.cursor >= V.rows.length - 1) { stopPlay(); return; }
      const next = V.cursor + 1;
      showRow(V, next, true);
      if (next >= V.rows.length - 1) { stopPlay(); return; }
      const ev = V.rows[next].ev;
      S.playTimer = setTimeout(tick, (ev === 'decision' ? STEP_MS.decision : STEP_MS.update) / S.speed.replay);
    };
    updateTransport(V);
    tick();
  }
  function stopPlay() {
    S.playing = false;
    if (S.playTimer) { clearTimeout(S.playTimer); S.playTimer = null; }
    if (S.mode === 'replay') updateTransport(S.replay);
  }
  function stepReplay(d) {
    const V = S.replay;
    stopPlay();
    const next = clamp(V.cursor + d, 0, V.rows.length - 1);
    showRow(V, next, d > 0);
  }

  /* --- モード切替 --- */
  function enterLive() {
    stopPlay();
    S.mode = 'live';
    S.logKey = '';
    buildTransport();
    setHash('');
    $('#gamePicker').value = 'live';
    const V = S.live;
    if (S.tab === 'live') showRow(V, V.rows.length - 1, false);
    kickLive();
    updateHeader();
  }
  async function openReplay(n, autoplay) {
    stopPlay();
    stopLiveTimer();
    try {
      const d = await getJSON('/api/game/' + encodeURIComponent(n));
      const V = newView();
      V.game = d.game; V.rows = d.rows || []; V.meta = d.meta; V.loaded = true;
      recomputeScale(V);
      S.replay = V;
      S.mode = 'replay';
      S.logKey = '';
      setTab('live', true);
      buildTransport();
      setHash('replay/' + n);
      $('#gamePicker').value = String(n);
      showRow(V, 0, false);
      if (autoplay) startPlay();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (e) {
      toast('第' + n + '局の棋譜は取得できませんでした');
      enterLive();
    }
  }

  /* ===================================================================
   * 12. ライブ取得
   * =================================================================== */
  function applyLive(d) {
    const V = S.live;
    const rows = d.rows || [];
    const first = !V.loaded;
    const gameChanged = !first && d.game !== V.game;
    V.serverTime = d.server_time;
    V.isLive = !!d.is_live;
    V.meta = d.meta;
    if (first || gameChanged) {
      V.game = d.game;
      V.rows = rows;
      V.loaded = true;
      recomputeScale(V);
      S.logKey = '';
      // 新しい対局に切り替わったときは冒頭から再生する（行が少ない場合）。
      V.cursor = !first && rows.length <= 8 ? -1 : rows.length - 1;
      if (S.mode === 'live') showRow(V, V.cursor, false);
    } else if (rows.length < V.rows.length) {
      V.rows = rows;
      recomputeScale(V);
      S.logKey = '';
      V.cursor = rows.length - 1;
      if (S.mode === 'live') showRow(V, V.cursor, false);
    } else {
      const grew = rows.length > V.rows.length;
      V.rows = rows;
      if (grew) recomputeScale(V);
    }
    if (S.mode === 'live' && S.tab !== 'live') { V.cursor = V.rows.length - 1; }
    if (S.mode === 'live') { updateTransport(V); kickLive(); }
    updateHeader();
  }
  async function pollLive() {
    try {
      applyLive(await getJSON('/api/live'));
      setOffline(false);
    } catch (e) { setOffline(true); }
  }
  async function pollSummary() {
    try {
      S.summary = await getJSON('/api/summary');
      if (S.summary.assets) { S.assets.guardians = S.summary.assets.guardians || []; S.assets.curses = S.summary.assets.curses || []; }
      updatePicker();
      updateHeader();
      if (S.tab === 'stats') renderStats(false);
      setOffline(false);
    } catch (e) { setOffline(true); }
  }
  function setOffline(v) {
    if (S.offline === v) return;
    S.offline = v;
    updateHeader();
  }
  function schedulePolling() {
    clearInterval(S.pollTimer);
    clearInterval(S.summaryTimer);
    if (!S.auto) return;
    S.pollTimer = setInterval(pollLive, POLL_LIVE_MS);
    S.summaryTimer = setInterval(pollSummary, POLL_SUMMARY_MS);
  }

  /* ===================================================================
   * 13. ヘッダー（状態ピル・KPI・棋譜ピッカー）
   * =================================================================== */
  function isLiveNow() {
    const V = S.live;
    if (V.meta && V.meta.klass === 'in_progress') return true;
    if (!V.meta && V.rows.length) {
      const last = V.rows[V.rows.length - 1];
      return !last.is_over && V.serverTime - last.t < 180;
    }
    return false;
  }
  function updateHeader() {
    const pill = $('#statusPill');
    let cls = 'idle', txt = '待機中';
    if (S.offline) { cls = 'off'; txt = '接続エラー（再試行中）'; }
    else if (S.mode === 'replay') { cls = 'replay'; txt = 'REPLAY #' + S.replay.game; }
    else if (isLiveNow()) { cls = 'live'; txt = '● LIVE 対局中'; }
    else if (S.live.game != null) { txt = '待機中（前局 #' + S.live.game + ' 終了）'; }
    pill.className = 'pill pill-' + cls;
    pill.innerHTML = '<i class="dot"></i><span>' + esc(txt) + '</span>';
    document.title = 'God Field AI 観戦 — ' + (cls === 'live' ? 'LIVE' : cls === 'replay' ? 'REPLAY' : '待機中');
    renderKpis();
    const lv = $('#tab-live');
    if (lv) lv.classList.toggle('is-replay', S.mode === 'replay');
  }
  function kpi(label, value, sub, cls) {
    return '<div class="kpi ' + (cls || '') + '"><span class="kpi-l">' + esc(label) + '</span><b class="kpi-v">' + value + '</b><span class="kpi-s">' + (sub || '') + '</span></div>';
  }
  function renderKpis() {
    const el = $('#kpis');
    const sm = S.summary;
    const cur = sm && sm.current, prev = sm && sm.previous;
    if (!sm) { el.innerHTML = kpi('進捗', '—') + kpi('現在レート', '—') + kpi('最高レート', '—') + kpi('完走勝率', '—') + kpi('相手平均レート', '—'); return; }
    const cap = cur ? cur.cap : null;
    const done = cap && isNum(cap.counted) ? cap.counted : 0, lim = cap && isNum(cap.hard_limit) ? cap.hard_limit : GOAL_GAMES;
    const rs = (cur && cur.rating_self && isNum(cur.rating_self.last_after) ? cur.rating_self : prev && prev.rating_self) || {};
    const net = rs.net_change;
    const wr = cur && cur.win_rate && cur.win_rate.completed;
    const pwr = prev && prev.win_rate && prev.win_rate.completed;
    const opp = cur && cur.rating_opponents && isNum(cur.rating_opponents.mean_before) ? cur.rating_opponents : prev && prev.rating_opponents;
    const counts = cur && cur.counts;
    const pct = lim ? clamp(done / lim * 100, 0, 100) : 0;
    el.innerHTML =
      kpi('進捗', done + '<small> / ' + lim + '</small>', '<span class="mini-bar"><i style="width:' + pct.toFixed(1) + '%"></i></span>' +
        (counts ? esc('完走 ' + counts.completed + ' ・ 勝 ' + counts.wins + ' 敗 ' + counts.losses + ' 分 ' + counts.draws) : '')) +
      kpi('現在レート', fmtInt(rs.last_after) + (isNum(net) ? ' <em class="delta ' + (net >= 0 ? 'up' : 'down') + '">' + (net >= 0 ? '▲' : '▼') + Math.abs(net) + '</em>' : ''), '開始 ' + fmtInt(rs.first_before), 'k-rate') +
      kpi('最高レート', fmtInt(rs.max_after), '最低 ' + fmtInt(rs.min_after)) +
      kpi('完走勝率', fmtPct(wr && wr.rate), (wr && isNum(wr.n) ? wr.n + ' 局' : '') + (pwr && isNum(pwr.rate) ? ' ・ 旧100局 ' + fmtPct(pwr.rate) : '')) +
      kpi('相手平均レート', opp ? fmtInt(opp.mean_before) : '—', '対戦相手の開始時');
  }
  function updatePicker() {
    const sel = $('#gamePicker');
    const sm = S.summary;
    const traced = sm && sm.traced_games ? sm.traced_games.slice().sort((a, b) => b - a) : [];
    const resMap = new Map();
    if (sm && sm.current) sm.current.games.forEach((g) => resMap.set(g.game, g));
    const want = S.mode === 'replay' ? String(S.replay.game) : 'live';
    const sig = traced.join(',');
    if (sel.dataset.sig !== sig) {
      sel.dataset.sig = sig;
      sel.innerHTML = '<option value="live">ライブ（最新）</option>' + traced.map((n) => {
        const g = resMap.get(n);
        const r = g ? resultOfMeta(g).t : '';
        return '<option value="' + n + '">第' + n + '局' + (r ? ' ・ ' + esc(r) : '') + '</option>';
      }).join('');
    }
    sel.value = want;
  }

  /* ===================================================================
   * 14. 戦績タブ
   * =================================================================== */
  function buildRatingPoints() {
    const sm = S.summary;
    const cur = sm && sm.current, prev = sm && sm.previous;
    const pts = [];
    const add = (series, seg, label) => (series || []).forEach((s) => {
      if (isNum(s.self_after)) pts.push({ rating: s.self_after, game: s.game, seg, label, outcome: s.outcome, delta: s.self_delta });
    });
    if (S.stitch && prev) add(prev.series, 0, 'ranked100');
    const boundary = pts.length;
    add(cur && cur.series, S.stitch && prev ? 1 : 0, 'ranked1000');
    let start = null;
    const firstSrc = S.stitch && prev ? (prev.series || [])[0] : (cur && cur.series || [])[0];
    if (firstSrc && isNum(firstSrc.self_before)) start = firstSrc.self_before;
    return { pts, boundary: S.stitch && prev && boundary > 0 ? boundary : -1, start, hasPrev: !!prev };
  }
  function ratingChartHtml() {
    const { pts, boundary, start } = buildRatingPoints();
    if (!pts.length) return '<div class="empty">レートの記録がまだありません</div>';
    const all = (start != null ? [{ rating: start, label: '開始', game: 0, start: true }] : []).concat(pts);
    const W = 900, H = 320, m = { l: 52, r: 18, t: 16, b: 30 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const ys = all.map((p) => p.rating);
    let lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
    const pad = Math.max(10, (hi - lo) * 0.12);
    lo = Math.floor((lo - pad) / 10) * 10; hi = Math.ceil((hi + pad) / 10) * 10;
    const n = all.length;
    const X = (k) => m.l + (n === 1 ? iw / 2 : k / (n - 1) * iw);
    const Y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * ih;
    const P = all.map((p, k) => Object.assign({ x: X(k), y: Y(p.rating), k }, p));
    const line = pathFrom(P.map((p) => [p.x, p.y]));
    const area = line + ' L' + P[n - 1].x.toFixed(1) + ' ' + (H - m.b) + ' L' + P[0].x.toFixed(1) + ' ' + (H - m.b) + ' Z';
    const ticks = 5, grid = [];
    for (let i = 0; i <= ticks; i++) {
      const v = lo + (hi - lo) * i / ticks;
      grid.push('<line class="grid" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(v).toFixed(1) + '" y2="' + Y(v).toFixed(1) + '"/><text class="ax" x="' + (m.l - 7) + '" y="' + (Y(v) + 4).toFixed(1) + '" text-anchor="end">' + Math.round(v) + '</text>');
    }
    const showDots = n <= 260;
    const dots = showDots ? P.filter((p) => !p.start).map((p) => '<circle class="dot-' + esc(p.outcome || 'x') + '" cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" r="2.4"/>').join('') : '';
    let divider = '';
    const bi = boundary + (start != null ? 1 : 0);
    if (boundary > 0 && bi > 0 && bi < P.length) {
      const dx = (P[bi - 1].x + P[bi].x) / 2;
      divider = '<line class="divider" x1="' + dx.toFixed(1) + '" x2="' + dx.toFixed(1) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/>' +
        '<text class="ax" x="' + (dx - 6).toFixed(1) + '" y="' + (m.t + 11) + '" text-anchor="end">← ranked100</text><text class="ax" x="' + (dx + 6).toFixed(1) + '" y="' + (m.t + 11) + '">ranked1000 →</text>';
    }
    const xl = (k) => '<text class="ax" x="' + X(k).toFixed(1) + '" y="' + (H - 9) + '" text-anchor="middle">' + k + '</text>';
    const xticks = [0, Math.round((n - 1) / 2), n - 1].filter((v, i, a) => a.indexOf(v) === i && v >= 0).map(xl).join('');
    S.ratingPts = P;
    return '<div class="chart-wrap rating-chart" id="ratingWrap"><svg viewBox="0 0 ' + W + ' ' + H + '" class="svg-chart" role="img" aria-label="レート推移">' +
      '<defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--gold)" stop-opacity=".32"/><stop offset="1" stop-color="var(--gold)" stop-opacity="0"/></linearGradient></defs>' +
      grid.join('') + '<path d="' + area + '" fill="url(#rg)"/>' + divider + '<path d="' + line + '" class="rline"/>' + dots + xticks +
      '<line class="cross" y1="' + m.t + '" y2="' + (H - m.b) + '" style="display:none"/><circle class="hdot" r="4.5" style="display:none"/></svg>' +
      '<div class="chart-tip" hidden></div></div>';
  }

  function barRow(label, n, w, l, d, rate) {
    const pct = isNum(rate) ? rate * 100 : 0;
    return '<tr><th>' + esc(label) + '</th><td class="num">' + n + '</td><td class="num">' + w + ' / ' + l + (d ? ' / ' + d : '') + '</td>' +
      '<td class="bar-cell"><div class="wbar"><i style="width:' + pct.toFixed(1) + '%"></i><span>' + (n ? fmtPct(rate) : '—') + '</span></div></td></tr>';
  }
  function histRows() {
    const cur = S.summary && S.summary.current;
    if (!cur) return [];
    let g = cur.games.slice().sort((a, b) => b.game - a.game);
    const f = S.filter;
    g = g.filter((x) => {
      const r = resultOfMeta(x).k;
      if (f === 'win') return r === 'win';
      if (f === 'loss') return r === 'loss';
      if (f === 'draw') return r === 'draw';
      if (f === 'stop') return r === 'stop';
      if (f === 'trace') return !!x.has_trace;
      return true;
    });
    return g;
  }
  function historyHtml() {
    const g = histRows();
    const filters = [['all', 'すべて'], ['win', '勝'], ['loss', '敗'], ['draw', '分'], ['stop', '停止'], ['trace', '棋譜あり']];
    const head = '<div class="filters" role="group" aria-label="絞り込み">' + filters.map(([k, l]) =>
      '<button class="chipbtn' + (S.filter === k ? ' on' : '') + '" data-filter="' + k + '">' + l + '</button>').join('') + '<span class="muted">' + g.length + ' 局</span></div>';
    const body = g.slice(0, S.shown).map((x) => {
      const r = resultOfMeta(x);
      const d = x.self_delta;
      return '<tr class="r-' + r.k + '"><td class="num">#' + x.game + '</td><td><span class="res res-' + r.k + '">' + esc(r.t) + '</span></td>' +
        '<td>' + (x.first_mover === 'self' ? 'AI先攻' : x.first_mover === 'opponent' ? '相手先攻' : '—') + '</td>' +
        '<td class="num">' + fmtInt(x.opp_before) + '</td>' +
        '<td class="num">' + fmtInt(x.self_after) + (isNum(d) ? ' <em class="delta ' + (d >= 0 ? 'up' : 'down') + '">' + sgn(d) + '</em>' : '') + '</td>' +
        '<td class="num">' + (x.decisions != null ? x.decisions : '—') + '</td><td class="num">' + fmtDur(x.duration_s) + '</td>' +
        '<td class="num">' + (x.gf_final != null ? x.gf_final : '—') + '</td>' +
        '<td>' + (x.has_trace ? '<button class="btn btn-sm" data-replay="' + x.game + '">▶ 再生</button>' : (r.k === 'stop' && x.stop_reason ? '<small class="muted">' + esc(STOP_JA[x.stop_reason] || x.stop_reason) + '</small>' : '<span class="muted">—</span>')) + '</td></tr>';
    }).join('');
    const more = g.length > S.shown ? '<div class="more"><button class="btn" data-more="1">さらに表示（残り ' + (g.length - S.shown) + ' 局）</button></div>' : '';
    return head + '<div class="table-scroll"><table class="tbl hist"><thead><tr><th>局</th><th>結果</th><th>先攻</th><th>相手R</th><th>AI R</th><th>判断数</th><th>時間</th><th>GF</th><th>棋譜</th></tr></thead><tbody>' +
      (body || '<tr><td colspan="9" class="empty">該当する対局はありません</td></tr>') + '</tbody></table></div>' + more;
  }
  function statsSig() {
    const sm = S.summary;
    if (!sm || !sm.current) return 'none';
    const c = sm.current;
    return [c.counts.counted, c.counts.completed, c.counts.in_progress, (c.series[c.series.length - 1] || {}).game, sm.traced_games.length, S.stitch, !!sm.previous].join('|');
  }
  function renderStats(force) {
    const root = $('#statsRoot');
    const sig = statsSig();
    if (!force && sig === S.statsSig && root.children.length) return;
    S.statsSig = sig;
    const sm = S.summary;
    if (!sm || !sm.current) { root.innerHTML = '<div class="panel"><div class="empty">戦績データを読み込み中、または台帳がまだありません。</div></div>'; return; }
    const cur = sm.current;
    const bd = cur.breakdowns || {};
    const buckets = (bd.by_opp_rating || []).map((b) => barRow(b.label, b.n, b.wins, b.losses, b.draws, b.rate)).join('');
    const fm = bd.by_first_mover || {};
    const fmCard = (k, l) => {
      const x = fm[k] || { n: 0, wins: 0, losses: 0, draws: 0, rate: null };
      return '<div class="split"><b>' + l + '</b><div class="wbar big"><i style="width:' + (isNum(x.rate) ? x.rate * 100 : 0).toFixed(1) + '%"></i><span>' + (x.n ? fmtPct(x.rate) : '—') + '</span></div>' +
        '<small>' + x.n + ' 局 ・ ' + x.wins + ' 勝 ' + x.losses + ' 敗' + (x.draws ? ' ' + x.draws + ' 分' : '') + '</small></div>';
    };
    const stops = Object.keys(bd.stop_reasons || {}).map((k) => [k, bd.stop_reasons[k]]).sort((a, b) => b[1] - a[1]);
    const maxStop = stops.length ? stops[0][1] : 1;
    const stopHtml = stops.length ? stops.map(([k, v]) => '<li><span>' + esc(STOP_JA[k] || (k.indexOf('ur_') === 0 ? '未対応受信 (' + k + ')' : k)) + '</span><div class="sbar"><i style="width:' + (v / maxStop * 100).toFixed(0) + '%"></i></div><b>' + v + '</b></li>').join('') :
      '<li class="muted">停止した対局はありません</li>';
    const hasPrev = !!sm.previous;
    root.innerHTML =
      '<div class="stats-grid">' +
      '<div class="panel span2"><div class="panel-head"><h2>レート推移</h2>' +
      (hasPrev ? '<label class="switch sm"><input id="stitchToggle" type="checkbox"' + (S.stitch ? ' checked' : '') + '><span class="switch-track"><i></i></span><span class="switch-label">ranked100 と連結</span></label>' : '') +
      '</div>' + ratingChartHtml() + '</div>' +
      '<div class="panel"><div class="panel-head"><h2>相手レート帯別の勝率</h2><span class="muted">完走局のみ</span></div>' +
      '<table class="tbl"><thead><tr><th>相手R</th><th>局</th><th>勝/敗/分</th><th>勝率</th></tr></thead><tbody>' + buckets + '</tbody></table></div>' +
      '<div class="panel"><div class="panel-head"><h2>先手・後手の内訳</h2></div><div class="splits">' + fmCard('self', 'AI が先攻') + fmCard('opponent', '相手が先攻') + '</div>' +
      '<div class="panel-head sub"><h3>平均</h3></div><p class="muted">判断数 ' + (bd.mean_decisions != null ? bd.mean_decisions : '—') + ' ・ 所要 ' + fmtDur(bd.mean_duration_s) + ' / 局</p>' +
      '<div class="panel-head sub"><h3>停止理由</h3></div><ul class="stops">' + stopHtml + '</ul></div>' +
      '<div class="panel span2"><div class="panel-head"><h2>対局履歴</h2><span class="muted">「▶ 再生」で棋譜をアニメーション再生</span></div><div id="histWrap">' + historyHtml() + '</div></div>' +
      '</div>';
    const rw = $('#ratingWrap');
    if (rw) {
      bindHover(rw, 900, () => S.ratingPts || [], (p) => p.start ? '<b>開始</b><br>レート ' + p.rating :
        '<b>' + (S.stitch && S.summary.previous ? esc(p.label) + ' ' : '') + '第' + p.game + '局</b><br>レート <b>' + p.rating + '</b>' +
        (isNum(p.delta) ? ' (' + sgn(p.delta) + ')' : '') + '<br>' + ({ win: '勝ち', loss: '負け', draw: '引き分け' }[p.outcome] || '結果不明'));
    }
  }
  function rerenderHistory() {
    const w = $('#histWrap');
    if (w) w.innerHTML = historyHtml();
  }

  /* ===================================================================
   * 15. タブ・トースト・ツールチップ・キーボード
   * =================================================================== */
  function setTab(t, silent) {
    S.tab = t;
    $$('.tab').forEach((b) => { const on = b.dataset.tab === t; b.classList.toggle('is-active', on); b.setAttribute('aria-selected', on ? 'true' : 'false'); });
    $('#tab-live').hidden = t !== 'live';
    $('#tab-stats').hidden = t !== 'stats';
    if (t === 'stats') { stopPlay(); renderStats(true); if (!silent) setHash('stats'); }
    else if (!silent) {
      if (S.mode === 'live') { S.live.cursor = S.live.rows.length - 1; S.logKey = ''; showRow(S.live, S.live.cursor, false); kickLive(); }
      else showRow(S.replay, S.replay.cursor, false);
      setHash(S.mode === 'replay' ? 'replay/' + S.replay.game : '');
    }
  }
  function setHash(h) {
    const want = h ? '#' + h : '';
    if (location.hash !== want) history.replaceState(null, '', location.pathname + location.search + want);
  }
  let toastTimer = null;
  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 3500);
  }
  const tip = $('#tip');
  function showTip(target) {
    let html = '';
    if (target.dataset.cid) html = cardTipHtml(+target.dataset.cid, target.dataset.dream === '1');
    else return hideTip();
    tip.innerHTML = html;
    tip.hidden = false;
    const r = target.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = r.left + r.width / 2 - tw / 2;
    let y = r.top - th - 10;
    if (y < 8) y = r.bottom + 10;
    x = clamp(x, 8, window.innerWidth - tw - 8);
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  function hideTip() { tip.hidden = true; }
  document.addEventListener('mouseover', (e) => {
    const t = e.target.closest ? e.target.closest('[data-cid]') : null;
    if (t) showTip(t); else hideTip();
  });
  document.addEventListener('scroll', hideTip, true);

  function bindUi() {
    $$('.tab').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));
    $('#autoToggle').addEventListener('change', (e) => {
      S.auto = e.target.checked;
      schedulePolling();
      if (S.auto) { pollLive(); pollSummary(); }
    });
    $('#gamePicker').addEventListener('change', (e) => {
      const v = e.target.value;
      if (v === 'live') { setTab('live', true); enterLive(); } else openReplay(+v, false);
    });
    $('#transport').addEventListener('click', (e) => {
      const b = e.target.closest('[data-act]');
      if (!b) return;
      const act = b.dataset.act;
      if (act === 'speed') { S.speed[S.mode] = +b.dataset.v; updateTransport(view()); }
      else if (act === 'latest') jumpLatest();
      else if (act === 'golive') { setTab('live', true); enterLive(); }
      else if (act === 'play') { if (S.playing) stopPlay(); else startPlay(); }
      else if (act === 'prev') stepReplay(-1);
      else if (act === 'next') stepReplay(1);
      else if (act === 'first') { stopPlay(); showRow(S.replay, 0, false); }
      else if (act === 'last') { stopPlay(); showRow(S.replay, S.replay.rows.length - 1, false); }
    });
    $('#log').addEventListener('click', (e) => {
      const li = e.target.closest('.li');
      if (li && S.mode === 'replay') { stopPlay(); showRow(S.replay, +li.dataset.idx, false); }
    });
    $('#statsRoot').addEventListener('click', (e) => {
      const rp = e.target.closest('[data-replay]');
      if (rp) { openReplay(+rp.dataset.replay, true); return; }
      const f = e.target.closest('[data-filter]');
      if (f) { S.filter = f.dataset.filter; S.shown = 50; rerenderHistory(); return; }
      if (e.target.closest('[data-more]')) { S.shown += 100; rerenderHistory(); }
    });
    $('#statsRoot').addEventListener('change', (e) => {
      if (e.target.id === 'stitchToggle') { S.stitch = e.target.checked; renderStats(true); }
    });
    document.addEventListener('keydown', (e) => {
      if (S.mode !== 'replay' || S.tab !== 'live') return;
      const tg = e.target;
      const tag = tg && tg.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === ' ' || e.code === 'Space') {
        if (tag === 'BUTTON') return;
        e.preventDefault();
        if (S.playing) stopPlay(); else startPlay();
      } else if (e.key === 'ArrowLeft') { e.preventDefault(); stepReplay(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); stepReplay(1); }
    });
    window.addEventListener('hashchange', routeFromHash);
  }
  function routeFromHash() {
    const h = location.hash.replace(/^#\/?/, '');
    const m = /^replay\/(\d+)$/.exec(h);
    if (m) {
      if (!(S.mode === 'replay' && S.replay.game === +m[1])) openReplay(+m[1], false);
    } else if (h === 'stats') setTab('stats', true);
    else if (S.mode === 'replay' || S.tab !== 'live') { setTab('live', true); enterLive(); }
  }

  /* ===================================================================
   * 16. 起動
   * =================================================================== */
  async function init() {
    buildSeats();
    buildTransport();
    bindUi();
    renderEmpty(S.live);
    updateHeader();
    try {
      S.catalog = await getJSON('/api/catalog');
      S.catalogReady = true;
    } catch (e) {
      toast('カード辞書を取得できませんでした（カード名は #ID 表示になります）');
    }
    await Promise.all([pollSummary(), pollLive()]);
    schedulePolling();
    routeFromHash();
  }
  init();
})();
