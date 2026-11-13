# GPU PPO learner

モデル M/L 向けに、CUDA の learner 実行経路を追加した。
**180,000 decisions/s は未達**。同じ16,384 decisionの比較では、既存bf16が10,538/s、
`packed-grouped` が53,876/s（10回合算、約5.11倍）だった。

## 学習則と実装

`Learner.update` のdecision単位PPO clip、masked KL、magnet KL、value cross entropy、
filtering、learning rate/alpha schedule、gradient clipping、Adam、EMAの式を共有する。
NumPyのshuffleとminibatch単位のAdam更新を維持する。microbatchは勾配を加算し、
その途中でAdam/EMAを更新しない。モデルのパラメータ名・形状も変えていない。

高速化した処理は次の通り。

- 有効tokenだけをpackし、全key/valueを使う双方向attentionを実行する。
- 最終層のattention以降は、headsが使う先頭21スロットのqueryだけを計算する。
  未使用queryは他のqueryの出力に影響せず、その勾配は元の計算でも0である。
- stageごとに全観測を複製せず、headsが使うhand/CLSだけをgatherする。
- embeddingの和と語彙ごとの勾配集計を融合する。勾配集計の入力はbf16、出力はfp32。
- 小さいhead幅32のattentionで複数headをまとめ、Q/K/Vの勾配を交互配置に直接保存する。
- encoder、embedding、headsをtorch.compileし、AdamとEMAを複数tensorまとめて実行する。
- targetsをGPU上で計算し、細かい同期をCUDA eventによる計時へ置き換える。

GPU targetsはCPUと同じtrajectory順序・fp32の演算順を使い、FMAを無効にする。
gamma=1、0.995、0.5でtargetsとadvantagesの全要素がbit一致した。

`packed-grouped` はencoderの残差とLayerNormの入出力もbf16にする。
マスター重み、Adamのmoment、EMA、損失・log probabilityはfp32を維持する。
数値精度と演算の加算順の差は以下に測定している。

## 実行

TSUBAME H100 MIG 4g.47gb、64 SM、torch 2.15.0+cu132、Triton 3.9で検証した。
高速経路は明示的に選ぶ。CPU/MPSおよび既定の`eager`動作は従来通り。

```sh
module load gcc/14.2.0 cuda/13.1.1
export OMP_NUM_THREADS=4 MAX_JOBS=8 TORCHINDUCTOR_COMPILE_THREADS=8
export TRITON_CACHE_DIR=/tmp/$USER-learner/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER-learner/inductor

# 既存checkpointを読み、出力は新規ディレクトリへ置く。
.venv/bin/python scripts/train.py --device cuda --env-backend cuda \
  --dtype bf16 --pool-fraction .2 --rollout-storage device --compile off \
  --resume runs/play/m-2166.pt --run-dir /tmp/$USER-learner/new-training \
  --num-envs 4096 --threads 4 --learner-backend packed-grouped \
  --microbatch-size 4096
```

`packed-grouped` はhead幅32/48/64、偶数head数、bf16専用。M（幅256/head幅32）と
L（幅384/head幅48）の両方に対応する。head幅48のattention tileは64までzero-padし、
実幅48だけを読み書きする。scale、モデル形状、パラメータ名、PPO/Adam/EMAの式は変えない。
`packed` / `packed-eager` はPyTorch varlen attentionを使う比較経路。
`packed-reduce`、`packed-short`、`packed-reduce-short`、`packed-bf16`は個別最適化の比較用。
`packed-fp8`は動的E4M3量子化の実験経路で、今回の測定ではbf16より遅かった。
初回compileは数十秒かかり、token数やmicrobatchの形状によって追加compileが起きる。

## 再現用ベンチマーク

checkpoint SHA-256:
`c0cae10b4ffa79992fae7ad84d41b5bd817d9c779f616adb0e2cdd049ce7fd91`。
ベンチマークは重みだけでなく、保存済みoptimizer、EMA、step数、学習設定も読み込む。

```sh
# 実ゲームを20回warmupし、終局した軌跡をNVMeに保存する。
.venv/bin/python scripts/bench_learner.py --checkpoint runs/play/m-2166.pt \
  --data /tmp/$USER-learner/data.pt --collect --num-envs 1024 \
  --games 2048 --collect-warmup 20

# 全101,255 decisionを使用。limitを指定すると比較用の先頭部分だけを使う。
.venv/bin/python scripts/bench_learner.py --checkpoint runs/play/m-2166.pt \
  --data /tmp/$USER-learner/data.pt --backend packed-grouped \
  --microbatch-size 4096 --warmup 2 --iterations 10 --profile

.venv/bin/python scripts/validate_learner.py --checkpoint runs/play/m-2166.pt \
  --data /tmp/$USER-learner/data.pt --backend packed-grouped --limit 8192 \
  --minibatch-size 4096 --microbatch-size 4096 --forward-limit 512 --steps 3
```

速度は**外部decision数 / learner.update全体の壁時計時間**。
forward、backward、targets、minibatch準備、gradient clipping、Adam、EMA、metricsと同期を含む。
self-play、保存ファイルの読み込み、compile/warmupは含まない。
stage数をdecision数として数えない。16,384 decisionには42,542 stage、平均49.83有効tokenがある。

## 精度

TF32を無効にした元のFP32実装を参照とし、同じcheckpoint、データ、shuffleで比較した。
forward/gradientは512 decision、更新は8,192 decision、minibatch/microbatchとも4096、
3回のupdate呼び出し（計6 Adam step）。Adam差は初期重みからの累積更新ベクトルに対する差。

| 指標 | 元のbf16対FP32 | packed-grouped対FP32 |
|---|---:|---:|
| forward損失の絶対差 | 8.06e-5 | 5.64e-5 |
| value logits 相対L2 | 0.190% | 0.216% |
| gradient 相対L2 | 3.22% | 3.85% |
| 6 step後のAdam累積更新 相対L2 | 9.37% | 9.99% |
| 同更新の最大絶対差 | 6.62e-4 | 7.36e-4 |
| 6 step後のEMA重み 最大絶対差 | 2.10e-6 | 2.34e-6 |

高速経路と元のbf16の直接比較では、勾配の相対L2は3.66%、Adam累積更新は12.24%。
bit一致する学習ではない。短期間の数値差を測定したもので、長期の学習結果の同等性は未検証。
全出力・各損失・各時点の更新差は測定JSONを参照。

embedding単体で当初約2.02%あった勾配誤差は、torch 2.15/cu132の`mm(out_dtype=float32)`が
非整列のreduction次元の末尾を落とすケースによるものだった。16行にpaddingすることで修正し、
同テストで0.167%（bf16丸めの範囲）まで下がった。

## 測定記録

[measurements/learner_continue_20261113](../measurements/learner_continue_20261113/) に実行ログと精度JSONを保存した。
既存bf16と最速bf16の同条件再測定は`gfrl-learner-final-checks.txt`、
microbatch比較とbf16精度は`gfrl-learner-final-sweep.txt` / `precision-bf16.json`。
FP8は`gfrl-learner-fp8-fast-v3.txt`、融合FFNは`gfrl-learner-ffn-v1.txt`、
attention逆伝播の分割は`gfrl-learner-split-v1.txt`。
いずれも最速bf16を上回らず、FFN融合・逆伝播分割・cuDNNの試作は最終コードへ採用しなかった。

FP8（`precision-fp8.json`）は対FP32で損失差4.75e-4、勾配相対L2 37.79%、
6 step後の累積Adam更新差25.50%だった。速度も37,603/sで、採用したbf16より遅い。
したがって通常の学習には`packed-grouped`を選ぶ。

精度ログの旧版は`optimizer_steps=3`にupdate呼び出し数を記録していた。
実際は8192/4096×3=6 Adam step。精度JSONには補正理由を記録し、
現在の検証スクリプトは`update_calls`と実際の`optimizer_steps`を別々に出力する。

## 全rolloutと統合確認

終局済み101,255 decision / 262,958 stage、平均53.6035有効tokenの全rolloutを
10回更新した結果は**50,413 decisions/s**（1,012,550 decision / 20.0851秒）。
計測中のDynamo compile/recompile/guard failureは全て0、peak allocatedは10.15 GiB。
目標180,000/sの28.0%であり、約3.57倍の追加高速化が必要。
forwardが約0.69秒、backwardが約1.25秒を占め、targets・Adam・EMAの最適化だけでは残差を埋められない。

全rolloutのデータSHA-256:
`747cadf40c40b9ff31313fe11d4675abd5fff7af28b2da2d9821d06fba5855a7`。
`gfrl-learner-full-and-train.txt` に全測定と`train.py`による2回のPPO更新・checkpoint保存を記録した。
統合確認の小さいrolloutはcompileを含むため、速度の根拠には使用しない。
既存bindingのnanobind reference-count警告は終了時に出ているが、更新と保存は成功した。

Mac Pythonテストは1295件合格（実行時12件skip）、GPU上の関連テストは11件合格。
CPUの構造比較はembedding/headsの出力と勾配、全key/valueを使う最終query省略の出力と勾配を検証する。
CUDAではattentionのforward/backward、paddingを含むembedding勾配、3種類のgammaのtargets、FP8を検証する。

実験はhomeのhome配下のstagingをstagingに使い、計算ノードの
`/tmp/$USER-learner/repo`で実行した。大きいdata・cache・生成checkpointはNVMeへ置いた。
既存`runs/`、`third_party/goodfield`、`official_*`は変更していない。commitも作成していない。

最終整理後の再測定（`gfrl-learner-final-full.txt`）は5回、506,275 decision / 10.3090秒、
**49,110 decisions/s**。計測中compile/recompile/guard failureはいずれも0。
約49,000〜50,400/sが全rolloutでの実測範囲であり、目標には達していない。
最終版ではtrajectory数をTritonの実行時引数にし、小さいrolloutごとにtargetsを再compileする問題も除いた。
その後のGPU上の関連11テストも合格した（`gfrl-learner-final-tests.txt`）。

## 2026-11-14 pool / L の再検証

TSUBAME H100 MIG 4g.47gb（64 SM）で、保存pool 8モデル、`pool_fraction=.2`、
既定history長64、bf16、1 stream、pooled arena 256 KiB/環境相当を検証した。
Mは既存`m-2166.pt`から2更新したcheckpoint、Lは`--size L`から8更新したcheckpoint。
Lは動作検証用の初期学習モデルであり、Mとの棋力や学習成熟度の比較ではない。
M/Lのモデル形状とパラメータ数（5,528,379 / 15,822,779）は変更していない。

同じサイズ内のlearner比較は、同一checkpoint・同一の終局済み16,384 decisionを使用した。
minibatch/microbatchは4096、warmup 2回の後に5回を合算、TF32無効。
平均有効token数はM 44.58 / L 35.14、stage数は41,621 / 40,712。
速度にはtargets、forward/backward、勾配clip、Adam、EMA、metrics、同期を含む。
計測区間のcompile/recompile/guard failureはいずれも0だった。

| モデル | 既存bf16 eager | packed-grouped | 速度比 |
|---|---:|---:|---:|
| M | 10,593 decision/s | 59,430 decision/s | 5.61倍 |
| L | 7,543 decision/s | 35,489 decision/s | 4.70倍 |

Lもcollectionとlearnerの両方でpacked経路を使用する。推論の幅384の行列積には
既存のpersistent GEMMを使う。M向けTMAの幅上限は256であり、LをTMA使用と誤記しないよう
collectionベンチマークではTMAの利用可否と実際の幅条件を分けて出力する。

FP32との比較はforward/gradientが512 decision、更新が8,192 decision、3 update呼び出し
（計6 Adam step）。全方式で同じshuffle・microbatch・初期optimizer/EMA状態を使った。
Adam差は初期重みからの累積更新ベクトルに対する差である。

| packed-grouped 対 FP32 | M | L |
|---|---:|---:|
| forward損失の絶対差 | 3.12e-5 | 3.05e-5 |
| value logits 相対L2 | 0.197% | 0.403% |
| gradient 相対L2 | 3.97% | 1.46% |
| 6 Adam stepの累積更新 相対L2 | 7.09% | 6.46% |
| 同更新の最大絶対差 | 2.39e-3 | 4.66e-4 |
| EMA重みの最大絶対差 | 8.13e-6 | 3.81e-4 |

元のbf16対FP32の勾配差はM 3.03% / L 1.12%、Adam累積更新差は4.74% / 4.81%。
高速経路と元のbf16の直接比較では、勾配差が3.51% / 0.684%、Adam差が7.72% / 6.28%。
LのEMAは初期学習中のため、保存されたoptimizer step数に応じたEMA decayもMと異なる。
各モデル内で同じEMA則を比較している。長期間の棋力・収束の同等性は未検証。
M/Lともgamma=1、0.995、0.5のtargetsとadvantagesはCPU計算と全要素がbit一致した。

記録は [`measurements/pool_l_20261114`](../measurements/pool_l_20261114/)。
`gfrl-pool-precision-{M,L}.txt` が詳細な数値誤差、`gfrl-pool-learner-*.txt` が各更新の速度、
`gfrl-pool-train-{M,L}.txt` が実際のtrain.pyによる更新・pool保存の記録である。
初回compileやpool構成変更時のgraph captureを含む小さい統合学習の速度は、上表には使用していない。


継続収集は4,096環境で、128ゲーム要求のwarmupを20回、2,048ゲーム要求の計測を5回実行した。
固定した保存pool 8モデルを使用する。checkpoint読み込みと初回compile/captureは計測から除く。
実学習でpool構成が変わる際の再capture時間はtrain.pyの`collect_s`に含まれる。

| モデル | 学習対象decision/s | 終局済み返却decision/s | 相手席も含む外部decision/s | 最後の平均有効token |
|---|---:|---:|---:|---:|
| M | 47,416 | 45,564 | 52,622 | 51.98 |
| L | 34,664 | 34,827 | 38,057 | 38.04 |

Lの独立した直前測定も34,680 decision/sだった。Lの返却数が今回のサンプル数を少し上回るのは、
warmup中に記録した未完了ゲームのdecisionが終局して返却されるため。
GPU peak allocatedはM 5.81 GiB / L 5.42 GiB。M/Lでpolicyと観測長が異なるため、
この表をモデル幅だけの速度比較として扱わない。collectionとlearnerを別々に計測しており、
両方を合わせた訓練全体のthroughputではない。

収集側の推論精度も`validate_gpu_inference.py`で、同じ保存rolloutの512 decisionから比較した。
Mは1,271 stage、Lは1,332 stage。無効なhand tokenとmask外log probabilityを誤差集計から除いた。

| packed推論 対 FP32 | M | L |
|---|---:|---:|
| encoder出力 相対L2 | 0.229% | 0.255% |
| 合法手log probability 相対L2 | 0.197% | 0.155% |
| 同log probability 最大絶対差 | 0.210 | 0.0570 |
| value logits 相対L2 | 0.159% | 0.231% |
| argmax選択の相違 | 0 / 1,271 | 1 / 1,332 |

元のbf16との直接比較では、合法手log probabilityの相対L2差はM 0.188% / L 0.136%。
推論・サンプリング結果のbit一致は保証しない。エンジンの厳密一致条件は同じseedと明示選択列である。

最終GPU関連テストは122件合格。pool fraction 0/.2/1、空poolからの追加、eviction、
空poolへの変更、両席、未完了ゲーム、1/2 streams、現行重みrefreshを検証した。
従来CPUの抽選をそのまま書いた独立参照ともNumPy RNG stateまで一致した。
エンジンの全23出力と統計はmonolithic/split/compact、pooled/unpooled、
複数seed/history/canonical設定で完全一致。別途、seed42の2,000 stepで32,016状態を完全比較した。
Macの関連60テストも合格（全体実行は最終CLIテスト追加前に1,298合格・64 skip）。
終了時には既存のnanobind reference-count警告が出るが、失敗や未保存はない。

Lの8更新とMの2更新は`--pool-fraction .2 --learner-backend packed-grouped`で完了し、
pool・optimizer・EMAを保存した。再現用checkpointと各16,384 decisionのrolloutを
`measurements/pool_l_20261114/artifacts/`へ保存し、SHA-256を`manifest.json`に記録した。
bench/validateスクリプトの`--checkpoint`と`--data`にこれらを指定すれば再検証できる。
元の大きいrolloutのSHAは`gfrl-pool-sha-{M,L}.txt`、保存subsetのSHAはmanifestに分けて記録している。
`summary.json`に測定の主要JSONを集約した。

```sh
ART=measurements/pool_l_20261114/artifacts
.venv/bin/python scripts/bench_learner.py --checkpoint "$ART/checkpoint-L.pt" \
  --data "$ART/rollout-L-16384.pt" --backend packed-grouped \
  --microbatch-size 4096 --warmup 2 --iterations 5 --no-tf32
.venv/bin/python scripts/validate_learner.py --checkpoint "$ART/checkpoint-L.pt" \
  --data "$ART/rollout-L-16384.pt" --backend packed-grouped --limit 8192 \
  --minibatch-size 4096 --microbatch-size 4096 --forward-limit 512 --steps 3
```

`gfrl-pool-collection-L-4096-final.txt`が表示項目整理後の最終収集測定。
以前の同名`...L-4096.txt`と1024環境のログの`tma_linear=true`はTMAの利用可能性だけを
示していた。Lで実際にTMA GEMMを実行した意味ではなく、最終版では両者を分離している。
ソースは専用staging、build/cache/大きい出力は計算ノードのNVMeで扱った。
既存runs、third_party/goodfield、official_*は変更せず、commitも作成していない。

再現用4ファイルのSHA-256とモデル/pool/rolloutの形状をMacで確認済み。
対話ジョブをexitで終了し、iqstatが空であることを確認した。homeの転送用大容量コピーも削除した。
