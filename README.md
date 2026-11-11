# gfrl

God Field 1v1 用の C++20 VecEnv と nanobind 拡張。現段階は逐次選択と対局実行のみ。

```sh
export PATH="$PWD/.venv/bin:$PATH"  # venv 内の cmake / ninja を使う
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build build
ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q

cmake -S . -B build-sanitize -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGFRL_SANITIZE=ON -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build build-sanitize
ctest --test-dir build-sanitize --output-on-failure

build/gfrl_bench
.venv/bin/python tests/python/bench_vecenv.py
```

Sanitizer は core・エンジン・C++ テストを計装し、Python 拡張はビルドしない。
Release の拡張は `python/gfrl/` に出力する。単体スクリプトでは
`PYTHONPATH=python .venv/bin/python script.py` として import する。

```python
import numpy as np
from gfrl import VecEnv

env = VecEnv(256, 4, 42, canonical=True)
mask = env.mask                       # bool[256,21]、ゼロコピー
choices = mask.argmax(axis=1).astype(np.int32)
env.step(choices)                      # 全 env の合法性検証後、GIL を外して並列実行
print(env.stats())
```

`VecEnv(num_envs, num_threads, base_seed, canonical=True, config=MatchConfig())`
は2人個人戦・既定 MatchConfig のみを受け付ける。非既定 config は ValueError。
席は players 配列順、ID は 0/1。先手は常に席0（エンジンの reset が先頭を選ぶ）。
各 env と試合番号から独立に seed を導出し、終局または512決定で次の試合へ自動 reset。

`step` の choices は C contiguous な int32[N]。不正な長さ・選択は ValueError、
不正な型は TypeError。C++ の入力は `std::span<const int32_t>`。
`gfrl::Selection` は不変の Match 上の選択カーソルで、`choose` は Command を実行しない。
`random_choices(mask, stage, rng)` は各行の合法な選択肢から一様に選ぶ。

公開配列はすべて読み取り専用の numpy view で、env の寿命を保持し、次の step で更新される。
保存する場合は `.copy()` を使う。step 実行中に別スレッドから配列を読まないこと。

| 配列 | dtype / shape | 内容 |
| --- | --- | --- |
| stage | int8[N] | kind=0/card=1/target=2/alloc_hp=3/alloc_mp=4 |
| actor_seat | int8[N] | 現在の手番の席0/1 |
| mask | bool[N,21] | kind=0..9、card=0..17とSTOP=18、target=なし0/自分1/相手2、alloc=0..20。未使用列はfalse |
| chosen_kind | int8[N] | 未選択は−1 |
| chosen_cards | int8[N,18] | 決定開始時の手札添字、選択順。残り−1 |
| chosen_count | int8[N] | 選択済み枚数 |
| substep | int16[N] | 決定内で既に消費した方策stage数（最初は0） |
| decision_done | bool[N] | 直前のstepで1つ以上の決定を実行 |
| episode_done | bool[N] | 直前のstepで試合が終了、既にreset済み |
| outcome | int8[N,2] | 終了した試合の席別+1/−1/0。終了なしなら0 |
| truncated | bool[N] | 終了した試合が打ち切り |
| episode_id | uint64[N] | 現在の試合番号（envごと、0から） |

`stats()` は累積の `episodes`, `decisions`, `policy_stages`, `automatic_decisions`,
`truncations` と検証用の `confusion_decisions`, `grant_decisions` を返す。
`policy_stages` は消費した方策stage数。`decisions` はchanceを含む実行数で、
方策stageが0だった決定を `automatic_decisions` に数える。生成時の自動処理も累積に含む。
1つのstepで実行した複数の強制決定も全数加算する。

alloc は HP/MP を `min(5*b,99)`、CP を残差として扱う。
全total=0..297・全HP bucketについてMP bucketが存在することをテストし、
実際の選択マスクも全ての生存可能totalで照合する。
残量Rが0..198なら、R≤99ではMP=0、R≥99ではMP=99が必ず可能なので、
HP選択後にMPの離散化による行き止まりはない。

ベンチマークは N=256/1024/4096 × threads=1/4/8 の各組合せを300step実行。
時間にはランダム方策のサンプリングも含み、生成時間は含まない。
平均stage数・自動割合・打ち切り率は計測区間の累積差分から算出する。

既定configではgrant phaseは発生しない（補充はエンジン内部で解決）。
C++テストでは専用fixtureでgrantと混乱の自動解決、強制決定、打ち切りを検証する。
fixtureのみ非既定configのMatchを内部注入し、公開APIのconfig制限は維持している。

TSUBAME の対話ノードでは、リポジトリ直下から次のラッパーで実行する。
引数はそのまま venv の Python に渡されるので、ベンチマークなどにも使える。

```sh
scripts/tsubame_run.sh scripts/train.py --device cuda --threads 4
scripts/tsubame_run.sh scripts/bench_training.py --help
```

ラッパーは `MALLOC_MMAP_THRESHOLD_=17179869184`、
`MALLOC_TRIM_THRESHOLD_=68719476736`、`MALLOC_TOP_PAD_=1073741824` を設定し、
`module` があれば libstdc++ 用に `gcc/14.2.0` をロードする。
`OMP_NUM_THREADS` は未設定なら4（指定済みならその値）を使う。

## 探索予算と勝率・推論時間

P4 の固定予算比較は `scripts/eval_p4.py` で一括実行する。A は探索なし、B は
学習済み AR belief、C は `--fit` のデータから推定した B1。同じ paired deals で
B vs A、C vs A、B vs C を評価する。`--with-oracle` の場合だけ、真の相手手札を使う
評価専用の D vs A を追加する。B・C・D の探索設定は共通で、両側の探索割合と時間、
勝率・Wilson 95%・ペア bootstrap 95% を `summary.json` / `summary.md` に保存する。
`--resume` は完成した組を再利用し、引数や入力ファイルの変更は拒否する。
`--fit` には評価局とは別に収集した belief fitting データを指定する。

```sh
scripts/tsubame_run.sh scripts/eval_p4.py \
  --checkpoint runs/play/m-2166.pt --belief runs/play/belief-ar-m.pt \
  --fit runs/belief-data/fit.pt --device cuda --threads 4 \
  --pairs 500 --num-envs 32 --seed 0 --worlds 64 --depth 3 --beta 0.02 \
  --search-response --candidates 16 --candidate-samples 64 \
  --max-inference-batch 128 --out-dir measurements/p4-m --resume
```

`sweep_search.py` は `eval_search.py` の評価処理と共通引数を使い、同じ seed の
paired deals で、探索あり対探索なしの同一モデルを比較する。
`--worlds` と `--depth` は直積、`--grid 16:1 32:2 64:5` は明示の組だけを評価する。
モデル規模は checkpoint の設定から読み込む。TSUBAME で M の checkpoint を使う例:

```sh
scripts/tsubame_run.sh scripts/sweep_search.py \
  --checkpoint runs/play/m-2166.pt --belief runs/play/belief-ar-m.pt \
  --device cuda --threads 4 --num-envs 32 --pairs 500 --seed 0 \
  --worlds 16 32 64 128 --depth 1 2 3 5 \
  --search-response --leaf-view actor --beta 0.02 \
  --candidates 16 --candidate-samples 64 --max-inference-batch 128 \
  --out-dir measurements/search-budget-m --resume
```

設定ごとの `worlds-N_depth-D.json` は既存の評価 JSON に `decision_timing` と
再開検証用の `sweep_inputs` を追加した形。`baseline.json` は探索なしの基準を一度測る。
`summary.json` と `summary.md` に勝率、Wilson / ペア bootstrap 95% 区間、局数、
探索割合、main / 応答の秒数、全決定あたりの秒数をまとめ、設定が終わるたび保存する。
`--resume` は完成済み JSON を再利用し、未完の設定を最初から実行する。
引数や入力ファイルのサイズ・更新時刻が変わっていれば混在を防ぐためエラーになる。
別の出力先を使うか、`--resume` を外して再測定する。

全決定の時間は評価対象側の全 Actor 呼び出しの実時間を CUDA / MPS 同期付きで合計し、
`substep == 0` の決定数で割る。探索しない決定と全 action substep を含み、
相手の推論・環境の step・モデル読込・レポート出力を除く。warm-up は行わず、
初回のキャッシュ構築も含める。バッチで償却した秒数なので単一局の応答遅延とは異なる。
探索内の main / 応答の時間は既存と同じ定義を使い、全決定の時間には追加の呼び出し境界の
同期コストが入る。探索なしの基準も同じ測り方を使う。`--profile` は測定負荷を増やす。
`oracle` は真の相手手札を使う診断専用。

## ターミナルで対局

```sh
.venv/bin/python scripts/play.py --checkpoint runs/play/m-2166.pt --seat random --seed 0
.venv/bin/python scripts/play.py --checkpoint runs/play/m-2166.pt --search --belief runs/play/belief-ar-m.pt --device cpu
.venv/bin/python scripts/play.py --checkpoint runs/play/m-2166.pt --greedy --games 3 --log runs/play/matches.jsonl --no-color
```

チェックポイントの `ema` と `model_config` は評価と同じローダーで読み込む。
席は `--seat 0|1|random`。番号で合法な選択肢を選び、カードを1枚ずつ追加して
「これで決定」で確定する。`h` はヘルプ、`s` は盤面、`q` / EOF / Ctrl-C は終了。
合法な選択肢が1つの段階、混乱、配布は評価と同じ VecEnv が自動処理する。
各局は評価用 seed `--seed + 局番号 - 1` の最初の対局を使う。既定の規則・chance
処理は変更しない。512決定での打切りは引き分け。

表示・JSONL は人の席の `Match::view` / `last_decision` の公開情報から作る。
霧で見えない状態は非表示とし、夢のカードは見かけの名前・性能を使う。
選択時の手札番号は VecEnv のスロットと対応するが、物理 token や乱数は出力しない。
`--reveal` はデバッグ用で、相手の手札も表示する。oracle belief は対局では禁止。
JSONL は追記式で、開始・各入力・公開決定の前後の盤面・可視の結果・勝敗・中断を記録する。

探索は `eval_search.py` と同じ `SearchAgent` / belief ローダーを使う。
既定は **worlds=64 (M64)、depth=3 (D3)、beta=0.1、alpha=0.002、candidates=16、
candidate-samples=64、leaf-view=actor、search-response 有効**。
推論の最大バッチは128、threads=1、キャッシュは auto / 256 MiB。
`--worlds`、`--depth`、`--candidates`、`--candidate-samples`、`--alpha`、`--beta`、
`--select`、`--leaf-view`、`--max-inference-batch`、`--search-cache-device`、
`--search-cache-mb`、`--root-batch-size`、`--skip-forced-inference`、
`--skip-single-candidate`、`--search-response`、`--fit`、`--profile`、`--tf32` は
評価スクリプトと同じ引数名。Boolean 設定は `--no-search-response` などで無効化できる。
既定はサンプリング、`--greedy` は方策・探索選択・探索の fallback を最大確率にする。
探索中の候補生成とロールアウトは通常のサンプリングを使う。

2026-11-10、Apple Silicon (arm64) のこの実行環境で、CPU 1スレッド、
`runs/play/m-2166.pt` + `runs/play/belief-ar-m.pt`、上記の既定、seed=0、random席、
人側ランダム入力4局を実測した。モデルの43決定（探索対象外の方策 fallback を含む）の
推論時間は **中央値0.47秒、平均0.53秒、範囲0.002〜3.05秒**。
モデルの最初の推論からカード・対象等の全 substep を合計し、表示・人の待ち時間を除く。
M64・D3 で候補上限4と16を試し、16でも実用範囲だったため上限16を採用した。
手札と候補数で時間は変わる。遅い場合は `--worlds 16 --candidates 2` で減らせる。
**MPS はこの実行環境で `torch.backends.mps.is_available() == False` のため未計測**。
対応する Mac では `--device mps` を指定できる。

自動対局の例:

```sh
.venv/bin/python scripts/play.py --checkpoint runs/m4-s-ppo-v1/checkpoint.pt --human-policy random --games 30 --no-color
.venv/bin/python scripts/play.py --checkpoint runs/m4-s-ppo-v1/checkpoint.pt --search --belief uniform --human-policy random --games 12 --no-color
```

C++ の追加は対局表示用の読み取り API と、任意で有効にする席別の記録に限る。
記録は `record_decision` の公開投影を保持するので、自動処理や自動リセット前の終局も
欠けない。通常の学習・収集・評価では記録を有効にせず、行動・RNG・観測は変更しない。
