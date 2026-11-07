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
