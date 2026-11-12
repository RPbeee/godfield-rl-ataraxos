# Value floor 測定

```sh
.venv/bin/python scripts/value_floor.py --device mps --threads 8 \
  --out measurements/value-floor-mps.json
```

既定値は400ゲーム、各ゲーム1 root、各variant K=64、root batch=64。
実ゲームは既定で全ゲーム並列、`--num-envs` 指定時はその数を上限にする。
MPSの推論batchは1024、token cache予算は4096 MB（unified memoryを想定）。
CPU/CUDAは推論batch 128、cache 256 MB。
`--inference-batch` / `--cache-mb` で変更でき、cacheが予算内ならdeviceに保持する。
継続は分割せず同じSearchEnvで走らせるため、world chunk設定は設けていない。
checkpoint / belief は `runs/play/m-2166.pt` / `runs/play/belief-ar-m.pt`。

## オンライン抽出と凍結

* ゲームごとに外部decision index `[0,100)` から一様・非復元抽出する。
  `--roots-per-game` の既定値は1、上限100。抽出用NumPy RNGは実方策と独立。
  indexは0始まりで、両seatの外部decisionを数え、autoregressive substepや
  自動decisionは数えない。対象より先に終局した場合はスキップする。
* 対象indexでsubstep 0・`determinize_support==0`・Oracle assignment検証・
  検証用`load_worlds`が成功した行だけを凍結する。失敗時は別indexに置換しない。
  従って観測rootの分布はゲームの生存とloadabilityに条件づき、全decisionの
  一様抽出ではない。JSONの `root_selection` に目標・実際のindex・欠落数と
  `decisions_from_start` / `decisions_to_end` のbucket分布を記録する。
* 凍結行には `VecEnv.step` で `-1` を渡し、実ゲームの `Actor.act(rows=...)`
  から除外する。C++はstate、RNG、全buffer、flags、outcome、statsを更新しない。
  `-1` は生きているdecisionのsubstep 0でのみ有効。他の有効行の挙動は従来通り。
* 凍結root数が `--root-batch` 以上、または凍結していないlive行がなくなった時、
  全凍結rootの `state`（OracleBelief）と `info`（指定belief）のK worldsを
  **同じSearchEnv**へロードし、真の終局または `--max-decisions`（既定10000）
  まで一括実行する。root数は閾値を超えることも、最後に下回ることもある。
* 継続のTorch/CUDA/MPS RNGは保存・復元し、belief/world seedも別NumPy RNGを使う。
  実ゲームengineのchance RNGには触れない。継続後に凍結を解除し、実ゲームの
  最初のactionに付随するlogitsから `v=P(win)-P(loss)` を取る。
  stateが凍結中に不変なので、これは同じrootのvalueであり、余分な方策抽選もない。
* ゲームは1回だけ実行し、最終結果からroot actor視点のyと残りdecision数を付ける。
  残り数はroot自身を含む（最後なら1）。凍結により実方策batchの乱数割当は変わるため、
  旧スケジュールとのサンプル軌跡一致は主張しない。方策とengineの遷移則は同じ。

## 出力と統計

JSONの `roots`、`games`、`summary`、`continuation_totals`、`sampler_statistics`、
`timing` および表の形式は維持する。`continuation_batches` には実際のroot/world数・
decision数・cacheのdeviceを追加した。
列挙パス廃止に伴い `timing.enumeration_seconds=0`、`replay_verified=false`。
`real_trajectory_sha256` は凍結sentinelを含む実行action列のhash。
互換キー `eligible_roots_by_game` はオンラインで実際に検証・選択されたroot数。
`continuation_seconds.state/info` は各variantのworld sampling時間、`rollout` は
両variant共通のロード・推論・集計時間。速度の分母はこの合計。
rootが0の場合も有効なJSONを出力する。

`p_hat`はroot actor視点の `P(win)+0.5 P(draw)`。
実ゲーム打ち切り、どちらかのvariantの有効継続数が2未満の場合は集計から除外。
継続打ち切りをdrawとして数えず、不正load・抽選枯渇も別計上する。
有効継続数n、draw数d、平均score pについてraw Brierから

```
[p*(1-p) - 0.25*d/n] / (n-1) = sample_variance(score) / n
```

を引く。負の補正後Brierも保持する。
`reducible_part_upper_bound = Brier(value) - Brier(state, corrected)`。
`mean_abs_v_state` に有限K補正は行わない。
全体と残りdecision bucketごとの表、およびゲーム単位bootstrap 95%区間を出す。
複数root時も同じゲームをclusterとして再抽出する。

既存world loaderはopponent historyをprojectionから、own Dreamの隠れた実体を
public priorから再構成する。この復元則に条件づく診断であり、元の完全状態と双方の
履歴を固定した厳密な下限は保証しない。infoにはbelief誤差も含まれる。
打ち切り・復元失敗にはcensoring biasがあり、方策samplingもluckに含まれる。

## CPUでの検証と計測

```sh
cmake --build build
ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q
.venv/bin/python scripts/bench_value_floor.py
```

benchmarkは初期decision 0の異なるゲームから8/32/64 rootsを取り、各variant K=32で
終局まで実行する。checkpoint/beliefは本スクリプトと同じ、8 CPU threads、
推論batch128。モデルloadと実ゲーム進行を除き、継続sampling・load・推論・集計を
含むwall timeで測る。オンライン抽出分布とは異なる初期rootの制御されたpilotで、
各batch 1回の測定。JSONは `measurements/value-floor-cpu-root-batches.json`。
MPSはこのsandboxで使用不可。本番400ゲームの設定は実行しない。

検証結果（2026-11-12）:

* C++再ビルド成功。全ctest **18/18成功**（7.92秒）。
* 全pytest **1284 passed, 2 skipped**（151.26秒）。既存のread-only NumPy→Torch
  変換warningが1件。
* C++で凍結・解除後のsnapshot、buffer、statsと非凍結referenceを比較。
  Pythonでも別行を進めながら凍結行の全公開観測・ラベル・flagsをreferenceと比較。
* 小さなCPUモデルで、同じ凍結scheduleにおける継続あり/なしのaction hashと
  実ゲーム結果を比較（RNG分離）。greedy方策では凍結scheduleを変えたreferenceとも
  結果・長さが一致。複数root・両variantのworld数、終局数・Kの内訳を検証。

変更ファイル:

* `cpp/gfrl_core/vec_env.cpp` / `vec_env.hpp`: substep 0の行単位freeze。
* `CMakeLists.txt` / `tests/cpp/test_core.cpp`: freezeのC++テスト登録とreference比較。
* `tests/python/test_vecenv.py`: 混在行・flags/stats・不正freezeの検証。
* `scripts/value_floor.py`: オンライン抽出、複数root・両variantの一括継続、推論/cache設定。
* `tests/python/test_value_floor.py`: 複数ゲームの凍結、RNG分離、sampling・空rootケース。
* `scripts/bench_value_floor.py`: CPU K=32の8/32/64 root pilot。
* `scripts/value_floor.md`: この説明と計測結果。

CPU実測（各variant K=32、8 threads、推論batch 128）:

| root batch | worlds（両variant合計） | decisions | 秒 | decisions/s |
|---:|---:|---:|---:|---:|
| 8 | 512 | 21,987 | 43.97 | 500.0 |
| 32 | 2048 | 98,485 | 206.64 | 476.6 |
| 64 | 4096 | 200,201 | 424.34 | 471.8 |

8/32では全worldsが終局。64では4096中4095が終局、state側に打ち切り1件。
全サイズで不正load・抽選枯渇は0。打ち切りはdrawと別計上する。
CPUではbatch拡大による加速は見られなかった。この変更はMPSの小batch呼び出し遅延を
減らす構成変更であり、MPSの速度は未計測。上表は各サイズ1回で、ゲーム数・根の集合も
サイズにより異なるため、CPUでの小さな差を精密な比較と解釈しない。

オンラインCLI smoke:

```sh
.venv/bin/python scripts/value_floor.py --games 8 --roots-per-game 1 \
  --root-batch 4 --k 4 --threads 4 --device cpu --bootstrap 100 \
  --out measurements/value-floor-online-cpu-smoke.json
```

8ゲームから2 rootsを選択、16継続すべて終局。打ち切り・不正load・抽選枯渇0。
残り6目標はスキップ。実際の開始indexは1と75、残りdecision数は22と13。
開始からの分布は `[0,5):1, [60,inf):1`、終局までの分布は
`[10,20):1, [20,30):1`。全体3.52秒、継続2.21秒、607 decisions。
JSONのworld内訳・root value/outcomeの範囲・残りdecisionの対応も確認した。
この2 rootsの統計は動作確認のみで、luckの割合の結論には使わない。

DESIGN.md・third_party/goodfieldは変更せず、依存追加・ネットワーク利用・commitなし。
