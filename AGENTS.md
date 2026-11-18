# AGENTS.md — 作業の約束・手順・経緯

このリポジトリで作業するエージェント（Claude / Codex）向けの記録。設計の全体は `DESIGN.md`、
各ツールの使い方は `docs/` と `scripts/*.md` にある。ここには **守る約束・実際の作業手順・
実験の経緯と結果・生データの置き場所** をまとめる。新しい実験をしたら、この文書の
「経緯」と「結果の置き場所」に追記すること。

---

## 1. 目的

God Field の 1 対 1 で人間に勝つ。Ataraxos 流の自己対戦 RL（`DESIGN.md`）と、
公式サイトを操作する bot（`scripts/official_bot.md`）からなる。
**人間個別への対策はしない**（人間は無数にいるので、一般的な強さで勝つ）。

## 2. 守る約束（ユーザーの指示）

- 公式サイトでは **パスワード付きの非公開部屋か CPU 練習だけ**で打つ。公開対戦はしない。
- bot は **相手の非公開の手札を絶対に使わない**。oracle（真の手札）はオフラインの診断だけに使い、
  `official_*`・bot・蒸留には決してつながない。
- 受信した生データ・認証情報・名前や識別子・パスワード・スクリーンショットは **commit しない**。
  名前やパスワードをログに残さない。ユーザーのメールアドレスをどこにも送らない。
- 実装は `codex exec` で Codex が行い、Claude が指示書・レビュー・テストを担当する（§4）。
- commit はしてよい。**push はしない**。
- 大きな設計の分岐（API の形、損失の変更、bot の動作変更など）は、実装前にユーザーに相談する。
- ユーザーへの報告・説明は **平易な日本語** で書く。
- `third_party/goodfield` は変更しない（goodfield の変更が要る場合は相談）。
- TSUBAME のジョブは `exit` で終える（`qdel` は使えない）。1 ユーザー 1 ジョブ。ポイントは気にしなくてよく、
  24 時間ジョブも可。

## 3. 環境と手順

### 3.1 手元（Mac M4）

- Python は `.venv/bin/python`（3.12）。素の `python3` には numpy がない。
- ネイティブ部分のビルドとテスト:
  ```sh
  cmake --build build && ctest --test-dir build
  .venv/bin/python -m pytest tests/python -q
  ```
  `tests/python/test_rollout_storage.py::test_storage_training_equivalence[mps]` は
  2026-11-18 時点で変更前から失敗している（MPS の数値差）。無関係な失敗として扱う。
- Codex が `build/` を作り直している間に同じ `build/` の `.so` を読む評価を走らせない。
  並行して評価するときは別の worktree（例: `../godfield-rl-ataraxos-srch`）を
  `git checkout --detach <commit>` してそこでビルド・実行する。
- macOS には `timeout` コマンドがない。

### 3.2 TSUBAME 4.0

接続先・アカウント名・鍵の場所は、リポジトリに入れないローカルファイル `AGENTS.local.md`
（`.gitignore` 済み）に書いてある。アカウントは 2 つ（以下 A・B）あり、それぞれ 1 ジョブずつ持てる。

- ログインノードの tmux セッション `gfrl` の中で `iqrsh` の対話ジョブ（node_o, H100 MIG 半分,
  24 コア）を持つ。計算ノードには直接 ssh できないので、コマンドは
  `tmux send-keys -t gfrl "<コマンド>" Enter` で送る。
- ジョブの取り直し: `HRT=24:00:00 bash ~/reroll.sh 4`（空いた GPU を引くまで最大 4 回）。
  状態確認: `/apps/t4/rhel9/uge/latest/bin/lx-amd64/iqstat`。
- 実行スクリプトの先頭は必ず次の形（素の python は /bin/python 3.9 で GLIBCXX エラーになる）:
  ```sh
  cd ~/godfield-rl-ataraxos; module load gcc/14.2.0 cuda/13.1.1 2>/dev/null || true
  export PATH="$PWD/.venv/bin:$PATH"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cmake --build build > runs/<name>-build.log 2>&1 || { echo BUILD_FAIL; exit 1; }
  ```
- コードの同期（TSUBAME 側は git 管理ではない）:
  `git archive HEAD | ssh <A or B> 'cd ~/godfield-rl-ataraxos && tar xf -'`
- 起動は **リポジトリに cd してから** リダイレクトする:
  `tmux send-keys -t gfrl "cd ~/godfield-rl-ataraxos && bash runs/x.sh > runs/x.out 2>&1" Enter`
  （2026-11-18、cd を忘れて `runs/...` へのリダイレクトが失敗し、5 時間気づかなかった。）
- 終了は手元から `until ssh ... 'grep -q DONE ...'; do sleep 300; done` をバックグラウンドで待つ。
  スクリプトは最後に `echo XXX_DONE`、失敗時に `XXX_FAIL` を出す。
- ネイティブコードの中で止まった Python は Ctrl-C を受け付けない。`C-\`（SIGQUIT）は時間をおいて効いた。
- `runs/` は git 管理外。結果は手元の `measurements/` に取ってくる（§6）。

### 3.3 公式 bot の練習対局（CPU 練習）

```sh
.venv/bin/python scripts/official_bot.py --mode training --checkpoint runs/play/m-xit1.pt --greedy \
  --search --belief runs/belief-m2166/train/checkpoint.pt --fit runs/belief-m2166/fit.pt \
  --worlds 16 --depth 3 --candidates 16 --candidate-samples 64 --alpha 0.002 --beta 0.1 \
  --search-response --device mps --threads 8 --games 1 \
  --playwright <playwright-core のパス> --out-dir <scratchpad>/<run>/srch-NN
```

1 局ずつ 40 局回すループは `measurements/practice_20261119/offs6-loop.sh`。各局の
`moves.jsonl` には受信データが入るので **リポジトリに入れない**（集計結果だけ残す）。
集計は `measurements/analysis_scripts_20261119/offs_tally.py <dir>`。

## 4. Codex への依頼の仕方

```sh
codex exec -C ~/godfield-rl-ataraxos -s workspace-write \
  -c service_tier='"default"' -c model_reasoning_effort='"high"' \
  -o <scratchpad>/<task>-last.md - < <scratchpad>/spec-<task>.md
```

- fast は使わない（`service_tier="default"`）。
- 標準入力を必ずファイルか `< /dev/null` にする。端末のままだと
  "Reading additional input from stdin" で止まる（2 時間止まった例あり）。
- 初回の作成は「目的と完了条件（一致・テスト）」だけを渡し、設計は任せる。
- sandbox では `.git` が読み取り専用で Codex は commit できない。Claude が差分を読み、
  テストを実行して commit する。
- Codex のネットワーク利用は TSUBAME への ssh/rsync だけ。
- 過去の指示書は `measurements/analysis_scripts_20261119/spec-*.md`。

## 5. 経緯と結果（2026-11-12 〜 11-19）

11-12 までの P0〜P5（基盤・自己対戦・belief・探索・比較・蒸留 1 回目）は `DESIGN.md` §11 を参照。
勝率は特記のない限り「同じ配牌の組で席を入れ替えた評価」で、[ ] は 95% 区間。

### 5.1 学習まわり（11-13 〜 11-17）

| 実験 | 内容 | 結果 |
|---|---|---|
| GPU 化 | CUDA 自己対戦エンジン（CPU と bit 一致）、bf16 packed PPO、過去プール、L サイズ | 大量の学習が回せるようになった |
| 長期学習 hp-* | ml-M から lr 1e-4 / 3e-5 / 1e-5、pool32、γ=0.99、big4x、R-NaD 風 magnet（a05/a20） | どれも元とほぼ互角（48〜53%）。主モデルは `hp-lr1e5/ckpt-4500`（m-2166 に 52〜53%） |
| γ<1 | γ=0.99 | 伸びず（メモ: future-discount） |
| 別系統 ml-L | 初期化なしの L サイズ | 同じ段階で ml-M とほぼ同じ強さ |
| 探索の上乗せ（4500） | belief＋探索 vs 素の方策、世界数・深さ・選び方の掃引 | argmax・w16d3 で 52.7% [51.1–54.2]（4000 局）。世界・深さを増やしても伸びない（w64d3 51.9%、w32d6 50.8%）。真の手札（oracle）でも 55% 程度 |
| CRN ロールアウト | 1 世界あたり複数ロールアウト・共通乱数 | 51.5 / 50.5 / 50.9 / 51.9%（いずれも 52.7% 以下） |
| 探索の蒸留 ExIt | 4500 を探索の argmax に近づける（`runs/xit`） | 1 回目 対 4500 51.3% [50.2–52.4]（8000 局）→ `m-xit1`、2 回目も 51.2%。K=1 の対照は 50.2% |
| belief の作り直し | 4500 用 belief、長期学習（5 倍）、温度・混合 | 上乗せは増えず（t1.0 52.0%、t1.25 49.3%、t1.5 50.1%） |
| privileged critic / BR | 4500 を固定した最適応答の学習（共有 vs 特権 critic） | ノイズ入り相手には 79% まで伸びる（正の対照）が、素の 4500 には 49〜50% で伸びない |
| exploiter | 4500 を固定して lr 1e-4 / 1e-5 で倒す学習 | 48.2〜50.8%。弱い ml-M@500 相手でも 62.4% → 61〜60.6% で伸びず |

### 5.2 運の雑音の測定（11-17 〜 11-18, `scripts/luck_diagnostic.py`）

真の状態の複製から、候補手ごとに「運×自分の方策の乱数」の格子で多数回打ち切り、
1 手だけ変えたときの勝率差を測る（oracle 診断、bot には使わない）。

| 測定 | 局面数 | 結果 |
|---|---|---|
| 4500 自己対戦（方策の上位候補） | 493 | 候補間の真の差 0.0015 vs 1 局の雑音 0.68（SNR 0.002）。別データで選んだ最善手の得 0.014（±1 尺度、勝率約 0.7%）。雑音の内訳: 運 23%、相互作用 76%、方策 ≈0。CRN で分散 0.83 倍まで。GAE と真の有利の相関 0.26 |
| 低確率の手（π≤1%） | 225 | 平均で −0.25（悪い）。より良い手が見つかる局面 0.4% |
| 弱い相手（ml-M@500, 勝率 62.5%） | 252 | 最善手の得 0.0004 [−0.013, 0.015]。低確率の手 −0.27。良い手のある局面 0.8% |

**結論**: 自己対戦でも弱い相手でも、1 手単位の改善余地はほぼない。弱い相手への 62.5% は
運で決まる天井に近い。学習方法の工夫（ノイズ除去・CRN・AIVAT 等）は主な打ち手にならない。

### 5.3 総当たり（11-18, 17 モデル × 136 組 × 2000 局）

`runs/rr.sh`、集計 `rr_sum.py`。強さは一列に並び（Bradley–Terry の残差 rms 0.94 ≒ 偶然、
有意な三すくみ 0 件）、上位（m-xit1 56.1、rnad 55.7/55.6、lr3e5c 55.4、4500 55.1 の全体平均）は
直接対決 48〜52%。別系統の ml-L も同じ強さに収束。ml-L-nomagnet は壊れている（4.7%）。
→ 相手を混ぜる集団学習は効果が見込めない。bot のモデル m-xit1 が最上位。

### 5.4 bot の縛りの損失（11-18, `eval_policies.py --a-restrict`）

公式 bot は UI 操作を確認していない手を避けている。同じ縛りをシミュレーターで片側だけにかけて
自分自身と打たせた（m-xit1、各 1 万局、TSUBAME B `runs/restrict2/`）。

| 縛り | 勝率 |
|---|---|
| 全部（bot と同じ） | 46.5% [45.5–47.5] |
| 夢（偽装）のカードを使わない | 46.9% [46.0–47.9] |
| 捨てるを使わない | 49.9% |
| 攻撃系を自分に使わない | 50.2% |
| 捨てる＋自分対象 | 49.6% |

4500 に全部の縛り: 46.7%（1 万局）。→ **損は夢のカードの縛りだけで約 3 ポイント**。
途中、縛りの判定が珍しい局面で指数的に遅くなり止まる不具合があり、25716f9 で上限つきの探索に直した
（公式 bot の同じ経路も対象）。

### 5.5 夢のカードの解禁（11-18〜19）

`--dream allow`（既定、282fe6d）: 手札の中で絵柄が一意な偽装カードは普通に使う。
同じ絵柄が他にもある場合は、画面上でどれをクリックするか区別できないので従来どおり最後の手段だけ。
CPU 練習 40 局（offs6）: 夢のカードを使った手 47 回すべてサーバーが受理、停止なし。
完了 33 局で 31 勝。停止 7 局は 22:40〜22:47 に連続した受理タイムアウト（夢のカードとは無関係、
サイト側の一時的な遅延とみられる）。

### 5.6 公式 bot の安定化（11-17）

0698264: 動的命中率の死の弓、confusionCount、遅い画面への対応。CPU 練習の停止は
offs4 40 局中 8 → offs5 0。

### 5.7 いまの見立てと次の候補（11-19）

- シミュレーター内の強さはほぼ頭打ち（1 手の改善余地なし、総当たりで推移的、探索の上乗せ +3 程度）。
- 実戦で取りこぼしていた最大のもの（夢のカードの縛り、約 3 ポイント）は解消した。
- 残る実戦の差: 探索を使えず方策だけで打つ局面（霧の中 約 4%、応答前の経緯が組めない 約 6%）。
  探索の上乗せから見て、全部拾っても 1 ポイント未満。
- 次の一手の候補: 非公開部屋での人間との対局で負け方の傾向を見る（20〜30 局）。

## 6. 結果の置き場所

| 場所 | 内容 | git |
|---|---|---|
| `DESIGN.md` | 設計と P0〜P5 の結果 | 管理 |
| `docs/*.md`, `scripts/*.md` | 各ツールの仕様・使い方 | 管理 |
| `measurements/tsubame_20261119/<アカウント>/runs/` | TSUBAME の `runs/` から取った結果 JSON・ログ・実行スクリプト（チェックポイントと生パネルは除く） | 管理外 |
| `measurements/practice_20261119/` | 練習対局の勝敗集計・ループ（受信データは含まない） | 管理外 |
| `measurements/analysis_scripts_20261119/` | 集計スクリプトと Codex への指示書 | 管理外 |
| TSUBAME `~/godfield-rl-ataraxos/runs/` | チェックポイント（`hp-lr1e5/ckpt-4500.pt` など）と生データ | — |
| 手元 `runs/play/` | bot 用モデル `m-xit1.pt`、`m-2166.pt`、`m-lr1e5-4500.pt`、belief | 管理外 |

`measurements/tsubame_20261119/` のフォルダ名はアカウント名（A・B との対応は `AGENTS.local.md`）。
