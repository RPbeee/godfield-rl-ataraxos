# GPU search self-play

2026-11-16、TSUBAME r3n11のH100 MIG **3g.47gb / 60 SM**で検証・計測。
モデルM（幅256・6層）の `runs/play/m-2166.pt` と、M向け再学習AR belief
`runs/belief-m2166/train/checkpoint.pt` を使用した。
**8,192局・336,265 searched positionsを952.48秒、353.04 positions/sで生成した。**
旧約35/s比10.09倍。保存・集計込みの実測であり、異なるseedや長期運転での下限保証ではない。
この2026-11-16のGPU実装ではCPU SearchAgent/SearchEnv、`third_party/goodfield`、`official_*`は変更していない。
2026-11-17のSearchAgent向け複数rollout/CRNは末尾を参照。

## 設計

`gpu/source.py`の`search=True`は、CPUエンジンからdeterminization、応答replay、
Dreamの自分手札サンプリング、公開履歴のopposite投影を生成する。
`search_source.py`と`cpp/gfrl_gpu/search.cuh`が外部決定の深さ、真の終局、
assignmentの拒否理由、葉観測、有限paired-gameの補充を追加する。
通常のGPU PPOモードの生成・推論の既定動作は維持する。

`GpuSearchAgent`は公開OBS_FIELDSだけをbelief/policyに渡す。ARサンプル、
determinized Match、履歴、合法prefix、rollout、葉観測、q̂と探索分布の計算はGPU上。
公開履歴の再構築はsearcher側のprojected historyだけを使う。
`label_*`と相手の実手札は推論のallowlistに含めない。oracleは拒否する。
記録時はdeterminized rootと元rootの全公開観測の一致も確認する。
CPUにはカーネル起動、動的batch/counterの読み出し、既存NumPy最終選択乱数列、
公開蒸留recordの転送と保存が残る。CPUでゲームを進める処理はない。

高速化は以下の組合せ。

- 本番はbf16。学習self-playと同じPackedEncoder/TMA・SparseHeadsを、shape bucketの
  CUDA Graphで実行する。21個のpolicy/value tokenだけをcacheする。
- AR beliefはroot memoryとcross-attention K/Vを世界間で共有し、self K/Vも逐次cacheする。
  decoderの1 stepをGraph化し、自己・cross attentionには1-query専用kernelを使う。
  **少数行の再サンプルにもGraphを使う**。0枚ならdecoderを実行しない。
- 同じ外部決定のrootをまとめ、完了したlaneは次の決定まで凍結する。
- world用arenaを再利用する。determinizationはworldにつき1回、各candidateのrolloutは
  公開履歴・chance RNGを含めてそのworldから複製する。
- 候補生成は独立したselection cursorで合法prefixだけを進める。完成commandの実行は
  実際のcandidate/world rolloutで行う。last-duplicate-winsと`(-log_pi, path)`順はCPUと同じ。
- rollout内のActionBatch anchorは以降のdeterminizationに使わない部分を省く。
  合法手・ゲーム結果・履歴への影響がないことを差分検証する。
- 採用したprefixだけをGPUでまとめてCPUへ転送する。`search_records.py`が既存format v1の
  root/candidate/stage配列を直接組み立て、多数のPython recordオブジェクトを避ける。
  gzipは8 workerの順序付きmember圧縮。既存loaderで読み戻したTensorは完全一致する。
  集計は保存前のTensorから統計だけを保持し、巨大shardの再読込を避ける。

`--search-dtype fp32`はCPU比較用の通常Torch推論。ゲームエンジンは本番と共通で、
q̂/π_searchはatol=3e-5、rtol=2e-5で検証する。本番bf16にはFP32一致を要求しない。
`--cuda-record-format legacy`で従来のPython record構築を選べる。
Baseline beliefは既存NumPyサンプラ、AR/independent beliefはGPUサンプラ。
mixture/hybrid belief、公式サイトのpublic-root import/exclusionsは今回のCUDA API対象外。

## 最終生成コマンド

compute shellでgcc/cuda moduleとscratch cache設定を用意してから実行する。
`bench_gpu_search.sh`は下記cache環境を自動設定し、login nodeでの実行を拒否する。

```bash
module load gcc/14.2.0 cuda/13.1.1
export PATH="$PWD/.venv/bin:$PATH"
export TMPDIR=/tmp/$USER-gsearch/tmp
export GFRL_GPU_CACHE=/tmp/$USER-gsearch/engine
export TRITON_CACHE_DIR=/tmp/$USER-gsearch/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER-gsearch/inductor
export XDG_CACHE_HOME=/tmp/$USER-gsearch/cache
export PYTHONPYCACHEPREFIX=/tmp/$USER-gsearch/pycache
export CUDA_CACHE_PATH=/tmp/$USER-gsearch/cuda-cache
export TORCH_HOME=/tmp/$USER-gsearch/torch
export TORCH_EXTENSIONS_DIR=/tmp/$USER-gsearch/extensions
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 最終実測と同じ予算・並列数・seed。--profileの同期コストも含む。
timeout 1740 python scripts/make_distill_data.py \
  --checkpoint runs/play/m-2166.pt --belief runs/belief-m2166/train/checkpoint.pt \
  --device cuda --env-backend cuda --search-dtype bf16 --threads 16 \
  --num-envs 8192 --games 8192 --games-per-shard 8192 --max-hours 0 \
  --worlds 16 --depth 3 --candidates 16 --candidate-samples 64 \
  --alpha .002 --beta .1 --leaf-view actor --search-response \
  --skip-forced-inference --skip-single-candidate \
  --root-batch-size 128 --max-inference-batch 8192 --tf32 --profile \
  --seed 910051 --out-dir /tmp/$USER-gsearch/data-final
```

bf16のGraph・K/V cache・packed recordはCUDA生成の既定。
root batch未指定時は40 GiB以上のdeviceで128、それ以外で32を選ぶ。
47 GB sliceでroot512は後半にOOMしたため採用しない。8192 envs/root128は最終runを完走した。
モデル読込は計時外、初回NVRTC/Tritonコンパイルは事前にwarm-up済み。
Graph作成・公開record転送・shard化・圧縮・集計は最終952.48秒に含む。
`--max-hours 0`は既存仕様どおり1つのin-flight shardを完了する。

```bash
ENV_BACKEND=cuda SEARCH_DTYPE=bf16 \
  BELIEF=runs/belief-m2166/train/checkpoint.pt \
  DATA_DIR=/tmp/$USER-gsearch/data OUT_DIR=/tmp/$USER-gsearch/session \
  scripts/session_distill.sh
```

セッションのGPU生成既定は`DATA_NUM_ENVS=8192`、`DATA_ROOT_BATCH_SIZE=128`、
`GAMES_PER_SHARD=8192`。CPU評価の`NUM_ENVS`既定512とは分けている。
明示的な`NUM_ENVS`/`ROOT_BATCH_SIZE`は生成にも引き継ぎ、`DATA_*`で個別に上書きできる。
生成の`SEARCH_DTYPE`と学習の`DTYPE`は別々。後段の評価は既存CPU経路。
CUDA provenanceにはbackend、dtype、record方式、探索設定、性能flag、checkpoint/belief MD5を記録する。
CPUの既存resume provenanceは維持する。batch/dtypeを変えると乱数消費順・対局分布も変わる。

## 実機での一致・リーク検証

TSUBAMEの探索・記録テストは**90 passed / 2 skipped**（41.57秒）。2件はCPU/native backendでのbf16専用検査をskip。
Mac全pytestは**1350 passed / 118 skipped / 1 deselected**（95.38秒）。指定の既知MPS失敗だけをdeselect。
CTestは**18/18 passed**。既存GPU training packed encoderの追加実機検査も**8 passed**。

- 各stepの観測、prefix、legal mask、履歴、terminal/truncation/outcomeを`assert_array_equal`で検査。
  history_len=0/1/64、多数のroot/world/seed、深さcutoffと真の終局、両席の葉観測を含む。
- Dream main/response、chance/danger、無効assignment、refused root、hidden-status/Fog、reload、
  lane補充、共有worldの複製、独立candidate cursorをCPU SearchEnvと直接比較。
- 相手の未使用実手札を消しても結果が変わらないcounterfactual検査。
  main/response、FP32、モデルMのbf16、AR Graph経路を含む。
- モデルMでTF32 off/onのFP32 q̂/π_search比較。decoder Graphは1/7/509行でも通常decoderと比較。
- packed recordとlegacy recordの全値、ragged offsets、選択結果、既存learnerのlossが一致する。
- 完全対局の追加比較: **5,555 roots、response 1,458、20,996候補**。
  公開tensor・prefix・選択・game/seat/decision/phase/outcomeは一致。
  q̂/log_pi/log_rho最大差**0**、π_search最大差**6.66e-16**。
  [final-packed-equality.json](../measurements/gpu_search_20261115/final-packed-equality.json)

## bf16とFP32の近さ

3,000 roots（response 768）。共通のFP32 belief worlds、候補集合、初期rollout RNGで比較。
bf16 prior・rollout policy・leaf valueを使用し、belief再サンプルの分散は除外する。
rollout policyの数値差による展開の違いは含むため、単純なleaf丸め誤差だけの測定ではない。

| 指標 | 実測 |
|---|---:|
| root内候補平均の平均 \|Δq̂\| | 0.0082513 |
| q̂ argmax一致 | 90.17% |
| π_search argmax一致 | 97.47% |
| KL(FP32 ‖ bf16) | 0.0043084 |
| KL(bf16 ‖ FP32) | 0.0043553 |
| 平均 \|Δπ\| | 0.0092668 |
| 全候補中の最大 \|Δq̂\| | 0.20927 |

[precision-latest.json](../measurements/gpu_search_20261115/precision-latest.json)。
これは候補集合内の比較であり、belief分布全体・保持されなかったpolicy mass・勝率改善の保証ではない。

## スループットと時間内訳

| 経路 | games | searched positions | 秒 | positions/s | games/h |
|---|---:|---:|---:|---:|---:|
| 依頼文の旧CPU | 7,168 | 約289k | 約2.3h | 約35 | 約3,117 |
| CPU再測定（FP32、128 envs/root32） | 128 | 5,555 | 182.55 | 30.43 | 2,524 |
| 最終GPU（bf16、8192 envs/root128） | 8,192 | 336,265 | 952.48 | **353.04** | **30,963** |

旧35/s比**10.09倍**、今回CPU再測定比**11.60倍**。探索予算・重みは同じ、batchと推論精度は異なる。
CPU値は従来の生成・保存計時で、最後のshard再読込集計を除く。最終GPU値は集計も含む。
同一batch・FP32の比較は上記の完全対局traceで行っている。
GPU最終runのmain 247,748 / response 88,517、fallback・invalid world load・invalid candidate world・
実対局の行動不一致はいずれも0。rejected belief sample率は0.318%。

| 区間 | CPU再測定 秒 | 最終GPU 秒 |
|---|---:|---:|
| belief sampling | 35.40 | 58.65 |
| candidate generation | 46.38 | 44.16 |
| rollout | 80.44 | 601.06 |
| leaf observation + value | 14.72 | 162.41 |
| search control・base world準備・記録 | 1.81 | 64.66 |
| 実対局policy / env step | 2.74 / 0.13 | 4.08 / 2.75 |
| shard化・保存 | 0.81 | 9.17 |
| 最終GPUの集計 | — | 2.75 |

処理件数が異なるため、区間秒はそのまま速度比にはしない。
GPU rolloutの内数はencoder **354.81秒**、heads **32.24秒**。
world workspace **3.52秒** / load・clone **69.05秒**は各探索区間の内数、
records/selection **15.44秒**はcontrolの内数。これらを重複して足してはいけない。
残る最大の時間はrollout（全体63.1%）、特にencoder（全体37.3%）。
leafの内部を観測生成とNNへ分けた計測、Nsight解析は行っていない。

改善過程ではFP32約95/s→bf16 217〜246/s→最終353/s。
decoder単体18 stepは通常86.77 ms→SDPA Graph 24.50 ms→1-query Graph 11.93 ms。
初めは512未満の再サンプルを通常実行していたため、それもGraph化した。
候補cursor/world共有、32 KiB stack設定、arena再利用、record配列化も採用。
encoder chunkは128/256/512/1024を比較して256を維持した。TMA範囲拡大は差が小さく不採用。
大きいroot batchのOOMと、packed record移行時のログ表示の0除算は修正・再測定済み。
失敗runや途中の瞬間速度は達成値に含めていない。

## generate → train_distill

最新packed形式のCUDA生成32局: **1,490 roots（main 1,097 / response 393）**。
変更していない既存`train_distill.py`が**1175 train / 315 validation**のpaired-deal分割で
CUDA学習を1 epoch、**37 optimizer updates**まで完了した。`best.pt`・`last.pt`・`complete.json`を出力。
全出力はcompute nodeの`/tmp/$USER-gsearch/smoke-final-confirm`と
`train-final-confirm`。小さいsmokeであり、蒸留後の勝率評価・長時間のexpert iterationは未実施。

再現benchmarkはcompute shellで`bash scripts/bench_gpu_search.sh`。
各測定・テスト・学習は29分timeout、CPU/生成並列数・ゲーム数は環境変数で上書きできる。
最終調整runは`fast-resample-final`、実機ログ/小さいJSONは
[measurements/gpu_search_20261115](../measurements/gpu_search_20261115/)に回収した。
割当jobは起動したまま。コミットは作成していない。

## 変更ファイル

- 生成エンジン: `cpp/gfrl_gpu/{compat.cuh,search.cuh}`、
  `python/gfrl/gpu/{__init__.py,source.py,search_source.py,search_env.py}`。
- GPU探索・推論・記録: `python/gfrl/gpu/{search.py,search_inference.py,belief_attention.py,packed.py,search_records.py,shard.py}`。
- 生成・実行・計測: `scripts/{make_distill_data.py,session_distill.sh,bench_gpu_search.sh,summarize_gpu_search.py,compare_search_precision.py,compare_search_shards.py}`。
- 検証: `tests/python/{test_gpu_search.py,test_gpu_shard.py,helpers/gpu_search_native.py}`、`tests/cpp/gpu_search_fixture.cuh`。
- 文書: 本書と`scripts/distill.md`。2026-11-16のGPU実装では`python/gfrl/{search.py,search_belief.py,distill.py}`、既存learnerは変更なし。

## 2026-11-17: SearchAgentの世界あたり複数rolloutと共通乱数

`scripts/eval_search.py`と公式bot/advisorの共通`SearchAgent`に
`--rollouts-per-world R`（正整数、既定1）を追加した。
beliefは従来どおりK世界をサンプルし、候補生成後に各世界をR本へ展開する。
候補生成の64サンプル、重複除去、上位16候補はRで変わらない。
各候補のqは有効な全(k,r)の葉value/終局結果の平均。
`--depth`は従来どおりroot後の外部決定数なので、浅いdepthと大きいRを組み合わせられる。

- R>1ではchanceとrollout方策の両方を候補間で同期する。
  chanceは同じ(k,r)の全候補に同じエンジンseedを渡す。
  r=0は従来の世界seed、残りのrは別seedを使い、相手手札assignmentは同じ世界内で共有する。
- 方策はseedを別の定数でdomain分離し、SplitMix64のcounterから一様乱数を作る。
  候補ごとのrollout方策サブステップで1個ずつ消費し、その候補の分布の逆CDFでサンプルする。
  強制するroot手は消費しない。別候補の経路長、終了、推論chunkによって乱数列はずれない。
  分岐後の分布やchance事象が違えば、同じ乱数列から異なる結果になる。
- R=1既定では従来のTorch multinomialと乱数消費順を維持する。
  `--rollout-crn`を付けるとR=1でも方策CRNを有効にできる。R>1ではこのフラグなしでも有効。
  最終選択の`--select`、正則化alpha/beta、葉view、未対応rootのfallbackは従来どおり。
- 全候補×世界×rolloutを1つのSearchEnvに載せ、推論もまとめる。
  `--max-inference-batch`でNNのchunk、`--root-batch-size`で同時root数を制限できる。
  メモリとrollout計算量はおおむねRに比例する。beliefと候補生成をR回繰り返さない。
- `valid_worlds`は有効な異なる世界数のまま。R>1のreportには`rollouts_per_world`と
  候補ごとの`valid_rollouts`を追加する。既存counterの`rollout_worlds`は候補×世界×Rのjob数。
  `report_values=True`では`rollout_values`/`rollout_world_indices`も返し、
  `world_values`は世界ごとのrollout平均、`q_se`は世界平均の標準誤差にする。
  同じ世界のR本を独立したbelief世界とみなして標準誤差を過小評価しない。

公開OBS_FIELDS以外を推論に追加していない。相手の真の非公開手札は探索根・公式botに渡さない。
oracleは従来どおりoffline診断だけで、公式bot/advisorは拒否する。
`third_party/goodfield`は無変更。

この機能はCPUエンジンの`SearchAgent/SearchEnv`用。
`eval_search.py --device cuda`はこのエンジンとCUDA推論の組合せなので利用できる。
別実装の`GpuSearchAgent`（`--env-backend cuda`の蒸留生成）は未対応で、
R>1/方策CRNを明示的に拒否する。設定を無視してR=1で動くことを防ぐ。

公式botは既存の起動引数に、例えば以下を追加する（CPUで実行可能）。

```bash
--search --worlds 16 --depth 1 --rollouts-per-world 4 --greedy
```

### CPU計測

macOS arm64、Torch 2.15.0、CPU 1 thread、policy
`runs/play/m-lr1e5-4500.pt`（幅256・6層）、AR belief
`runs/belief-m2166/train/checkpoint.pt`、fit `runs/belief-m2166/fit.pt`。
単一の対応main rootの`search()`を計時し、belief・候補生成・rollout・葉value・最終選択を含む。
モデルロードと実対局rootの方策推論は除外。各設定1回warmup後、同じ3初期局面
（seed 741–743）で測り、中央値を取った。上限16候補、64候補サンプル、NN chunk=128。
実際の重複除去後の候補数は各設定とも[1,2,2]。

| 予算 | R=1（従来） | R=4（CRN） | R=8（CRN） |
|---|---:|---:|---:|
| w16d3 | 99.663 ms | 288.551 ms（2.90倍） | 549.018 ms（5.51倍） |
| w16d1 | 57.160 ms | 154.514 ms（2.70倍） | 282.286 ms（4.94倍） |

この小規模な初期局面計測ではR倍未満の遅延だった。16候補が残る中盤やCUDAの速度は未測定。
生データは[search_rollouts_cpu_20261117.json](../measurements/search_rollouts_cpu_20261117.json)。
再現コマンド:

```bash
.venv/bin/python scripts/bench_rollout_crn.py \
  --out measurements/search_rollouts_cpu_20261117.json
```

検証結果: Release build成功、`ctest --test-dir build`は18/18 passed。
指定のpytestコマンド（storage training equivalenceのMPS 1件をdeselect）は
**1434 passed / 122 skipped / 1 deselected**。既存の凍結fixtureは変更せず通過した。
追加検証は候補間chance seed/方策乱数列の一致、非同期row実行と推論chunkによる不変性、
逆CDFとmask、Rによらない候補集合、(k,r)平均、公式main/responseの実手札非依存性。
R=4・探索/相手argmaxのCPU CLIスモークも2局完走し、world load・候補適用・
実対局行動の不一致は0だった。GPUハードウェアの検証は行っていない。

### TSUBAMEでの勝率比較（未実行）

compute shellで以下を実行する。各設定2000 pairs（4000局）、同じseed・checkpoint・対戦相手。
`--select argmax`は探索の最終選択、追加の`--opponent-select argmax`は探索なしの対戦相手の
各方策サブステップをargmaxにする。相手選択の既定は従来どおりsample。
rollout中の方策は引き続きsamplingする。

```bash
run_case() {
  local search_depth=$1 search_rollouts=$2 search_tag=$3
  shift 3
  .venv/bin/python scripts/eval_search.py \
    --checkpoint runs/hp-lr1e5/ckpt-4500.pt \
    --belief runs/belief-m2166/train/checkpoint.pt \
    --fit runs/belief-m2166/fit.pt \
    --opponent policy --opponent-select argmax --select argmax --device cuda \
    --pairs 2000 --seed 741 --num-envs 32 --threads 4 \
    --worlds 16 --depth "$search_depth" --rollouts-per-world "$search_rollouts" \
    --candidates 16 --candidate-samples 64 --alpha .002 --beta .02 \
    --root-batch-size 1 --max-inference-batch 512 "$@" \
    --out "measurements/search-rollouts/w16d${search_depth}r${search_rollouts}-${search_tag}.json"
}
run_case 3 1 legacy
run_case 3 1 crn --rollout-crn
run_case 3 4 crn
run_case 3 8 crn
run_case 2 4 crn
run_case 1 2 crn
run_case 1 4 crn
run_case 1 8 crn
```

まずd3R1の従来/CRNで同期だけの影響を分離する。
d3R1対d1R2、d3R4対d1R8はrootを含む外部決定予算K×R×(depth+1)が等しい。
葉推論・root候補生成・経路長も時間に影響するため、実際の秒/searchも併記して比較する。
CUDA推論の精度・勝率・時間はClaudeの実機計測で確認する。
