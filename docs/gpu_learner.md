# GPU PPO learner

モデル M (`runs/play/m-2166.pt`) 向けに、CUDA の learner 実行経路を追加した。
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
  --dtype bf16 --pool-fraction 0 --rollout-storage device --compile off \
  --resume runs/play/m-2166.pt --run-dir /tmp/$USER-learner/new-training \
  --num-envs 4096 --threads 4 --learner-backend packed-grouped \
  --microbatch-size 4096
```

`packed-grouped` はhead幅32、偶数head数、bf16専用（現在のMは該当）。
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
