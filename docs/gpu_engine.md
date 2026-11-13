# GPU self-play backend

CUDA専用の学習用環境を `python/gfrl/gpu` と `cpp/gfrl_gpu` に追加した。
CPUの既定動作は変更していない。`third_party/goodfield` と `official_*` は読み取り専用の参照元であり、変更していない。

## 対応範囲

- 既定のMatchConfig、2席とも外部policy、通常の学習用VecEnvに対応する。
- 状態遷移、RNG、合法手カーソル、観測、両席の履歴、belief用正解ラベル、終局、自動リセットをCUDAで処理する。カーソルの選択、完了commandの実行、観測の生成を別カーネルに分ける。大きいbatchでは完了commandをGPU上でまとめて実行する。
- `GpuActor` は観測を各decisionで1回だけencodeする。最長22段の選択をCUDA Graphに捕捉し、先に選択が終わった行はGPU上で凍結する。
- `GpuCollector` は軌跡と未完了ゲームをGPUに保持し、既存のPPO learnerにRolloutBatchを渡す。
- CUDA backendは `pool_fraction` の0〜1、M/Lのモデル、`rollout_storage=device`、`compile=off` に対応する。評価bot、search、public-root制限、state importはCPU backendに残る。
- ホストは初期化、カーネル起動、4 decision roundごとの終局数による収集停止判定、learner境界のragged compaction、統計出力を担当する。ゲーム状態・観測・履歴・選択をCPUへ往復させない。PPOのtarget計算などlearner内部は今回の移植対象外。

## 実装

`source.py` が既存エンジンをCUDA翻訳単位へ組み立てる。標準コンテナを環境ごとのdevice arenaへ置換し、CPU例外をdevice invariantへ置換する。CPU側のimport検証・rollback・search経路はGPU学習経路から除く。合法手探索は18スロットの固定配列を使う。履歴はringに保持し、観測への協調コピー時に直接読み出してageを計算する。中間Token配列へ履歴全体をコピーしない。beliefラベルのゼロ初期化も観測カーネル内で協調して行う。

NVRTCでPTXを作り、CUDA driver linkerでCUBINを生成してからロードする。直接PTXをmoduleへ渡すと、この大きな翻訳単位では非常に長いJIT時間が発生した。キャッシュは既定でジョブ内の一時領域に置く。初回コンパイルは数分かかり、計測のwarmupに含める。CPUエンジンの数値処理には `--fmad=false` を使う。

モデルM/Lのbf16推論には、GPU上で有効tokenをpackするencoderと、選択中の行だけを計算するpolicy headsを使う。既定の処理単位は256環境。Hopper以降ではTMAで行列演算の入力・出力を転送し、有効行数はdevice scalarから読む。長いFFNの射影には通常のpersistent GEMMを使う。attentionはquery幅64、key幅32。最終層ではpolicy headsが必要とする先頭21スロットだけをattention以後に計算する。FFN後の残差加算と次層のLayerNormをまとめる。

重みは各PPO更新後にrefreshする。これは既存bf16モデルとの許容誤差による数値検証であり、CPUとCUDAのモデル推論や乱数サンプリングのbit一致は保証しない。エンジンの一致条件は**同じseedと明示的な同じ合法選択列**である。今回のTMA・attention幅・処理単位の速度比較では、モデルMの既存GPU encoderとの最大差は0だった。FP8は使用していない。

## 過去モデルとの対戦

ゲーム開始時に確率 `pool_fraction`（既定0.2）でpool対戦を選び、保存poolのentryと
相手席をそれぞれ一様に抽選する。抽選処理はCPU Collectorと共通の`pool.assign_opponents`で、
同じnew-game行集合に対するNumPy RNGの消費順も従来のCPU実装と一致する。
空poolでは両席とも現行モデルを使い、乱数を消費しない。
一度決めた相手席・iteration・保存重みは終局まで保持する。更新境界やpool evictionをまたいでも
再抽選しない。終局時に割当てを解除し、自動リセット後の次ゲームで抽選する。

相手席のdecision/stageはlearnerへ渡さない。現行モデルの両席対戦では両席を学習に使う。
終局結果は元の席・episodeに対応付け、打ち切りはdrawとして記録する。
未完了ゲームの観測、old log probability、valueをそのまま次のcollectへ持ち越す。
CUDA Graphのwarmup/captureは全環境を凍結するため、pool構成変更に伴う再captureでも
ゲームや未完了軌跡を進めない。現行重みだけをPPO更新後にrefreshする。

policyごとに担当行の有効tokenだけをpackし、同じdecision内の全stageを同じpolicyで選ぶ。
pool有効時は各roundの終局フラグだけをホストへ読み、CPUと共通の抽選を行う。
ゲーム状態・観測・選択はGPUに置く。GPUのroundごとに完了行を凍結する実行順序は
CPU Collectorのstageごとの実行順序とは異なるため、完全な訓練runのサンプル列のbit一致は
保証しない。同じseed・明示選択列に対するエンジン出力の完全一致は引き続き検証する。

Lのモデル形状は変更しない（幅384、8層、8 heads、head幅48、FFN1536）。
Tritonのtileだけを幅512/head幅64までzero-padし、読み書きとLayerNormのreductionを
実幅でmaskする。attentionのscaleも実head幅48から計算する。

## 実行例

検証済み環境はTSUBAME 4.0のH100 MIG 4g.47gb（64 SM）、torch 2.15.0+cu132、Triton 3.9、CUDA 13.1.1。MacではGPU経路は実行できない。CUDA toolkitのnvcc、NVRTC、CCCL headersが必要。

```sh
module load gcc/14.2.0 cuda/13.1.1
export OMP_NUM_THREADS=4

# 独立したCPU VecEnv oracleが必要。既存のビルドがあれば再利用可能。
cmake -S . -B /tmp/$USER-gpu/cpu-build -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$HOME/godfield-rl-ataraxos/.venv/bin/python"
cmake --build /tmp/$USER-gpu/cpu-build --target _core gfrl_gpu_oracle -j 8
export GFRL_GPU_CPU_ORACLE=/tmp/$USER-gpu/cpu-build/gfrl_gpu_oracle

PYTHONPATH=python .venv/bin/python -m pytest tests/python/test_gpu.py -q
.venv/bin/python scripts/validate_gpu.py --num-envs 16 --steps 2000 --seed 42

# engineのみ。policy stage数と外部decision数を区別して表示する。
.venv/bin/python scripts/bench_gpu.py --num-envs 32768 --steps 100 --warmup 20

# モデル推論とengine。軌跡保存は含まない。
.venv/bin/python scripts/bench_gpu.py --checkpoint runs/play/m-2166.pt \
  --num-envs 8192 --steps 30 --warmup 10

# 軌跡保存、未完了ゲーム保持、完了ゲームの圧縮まで含む。
.venv/bin/python scripts/bench_gpu_collection.py --checkpoint runs/play/m-2166.pt \
  --num-envs 32768 --games 4096 --warmup-games 128 --warmup-iterations 20 \
  --iterations 10 --pooled --arena-bytes 262144

.venv/bin/python scripts/train.py --device cuda --env-backend cuda \
  --dtype bf16 --pool-fraction .2 --rollout-storage device --compile off \
  --init-from runs/play/m-2166.pt --run-dir /tmp/$USER-gpu/train-new \
  --num-envs 4096 --threads 4
```

実際のTSUBAME実験は元repoのrunsを変更しないよう、別のsource stagingとcompute nodeの`/tmp`で行った。上記例の出力先にも、既存の実験を上書きしない新しいディレクトリを指定する。

`GFRL_GPU_ENCODER_CHUNK`でencoderの処理単位を変更できる。`GFRL_GPU_SPLIT_STEP=0`は比較用の単一stepカーネルへ戻す。`GFRL_GPU_COMPACT_STEP=0/1`は完了commandの整列を切り替える。既定はsplit有効、整列は16,384環境以上で有効。いずれもゲームのseed・選択列の意味は変えない。

大きいbatchには十分な空きVRAMが必要。`arena-bytes=262144`は今回の長時間測定で使用した設定で、既定値は余裕を持たせた512 KiB以上。pooled arenaでは環境ごとの固定上限ではなく全環境の共通pool容量になる。

## 一致検証と制約

`validate_gpu.py` は各policy stepの全23出力（mask、全観測、履歴、labels、選択prefix、終局・reset flags等）を `np.array_equal` で比較し、累積統計も比較する。選択列は独立したNumPy RNGで一度生成し、CPUとGPUの両方へ渡す。freezeも含める。seed 0、42、uint64最大値、history長0/1/64、canonical両設定をテストする。

サンプリングによる差分検証は、全ての到達可能状態の形式的証明ではない。state importや非既定設定を使う単体局面のGPU再現は未対応。512 decisionの境界はtest専用のカウンター設定で強制的に到達させ、CPU VecEnvの独立したバイナリ出力と全23配列・統計を比較する。打ち切り、draw outcome、episode id、自動リセット後の新しい観測まで検証する。fixtureは学習用CUDA翻訳単位には含めない。CPU oracle実行ファイルはtest時に一時領域へ自動buildするか、`GFRL_GPU_CPU_ORACLE`で指定できる。

無効なCUDA選択は非同期device trapで失敗する。CPUの例外・rollback APIとの一致は対象外であり、渡す選択は公開maskで合法である必要がある。

## 性能の判定

positions/sは外部policyによる**完了したdecision**数。内部の選択stageや自動decisionをpositionとして水増ししない。collectorでは今回サンプルしたdecision数と、終局済みでlearnerへ返せるdecision数を分けて報告する。初回コンパイル・warmupは速度計測から除く。

初期状態は有効tokenが少なく、短いwarmupだけでは持続収集の速度を過大評価する。大きいbatchではwarmup収集を複数回に分け、戻ったbatchを毎回解放しながら履歴と未完了ゲームを蓄積する。`--warmup-iterations`と`--iterations`を使い、最後の有効token数と各回の速度も記録する。

エンジン単体の30万positions/s超えと、モデルMを含む収集での30万positions/s達成は別である。**モデルMを含む収集の30万positions/sは未達**。

## 2026-11-13 最終測定

H100 MIG 4g.47gb、64 SM、32,768環境、history_len=64、seed=42、bf16、pooled arena 256 KiB/環境相当。checkpointのSHA-256は `c0cae10b4ffa79992fae7ad84d41b5bd817d9c779f616adb0e2cdd049ce7fd91`（MacとTSUBAMEで一致）。

| 測定 | 外部decision/s |
|---|---:|
| エンジン＋ランダム合法選択 | 1,300,462 |
| M＋エンジン、軌跡保存なし、1 stream | 158,937 |
| M＋持続収集、1 stream | 140,347 |
| そのうち完了ゲームから学習へ返却 | 133,786 |

モデル込み測定は70 decision roundのwarmup後に100 round。持続収集は20回のwarmup収集後、各4,096ゲームを要求する10回を合算した。計測区間で2,621,440 decisionをサンプルし、2,498,878 decisionを返却した。有効token数は最後に平均52.77、PyTorch peak allocatedは32.96 GiB。1 streamの独立した再測定は140,384 decision/sだった。2・4 streamsはそれぞれ134,126・133,191 decision/sで、今回の設定では1 streamが速かった。

エンコーダ単体の固定観測測定は208,074 decision相当/sで、モデル込み30万/sへの主要な制約として残っている。FFN融合、LayerNorm＋行列演算融合、link-time optimization、別の行列tile設定も比較した。速度の改善を確認した変更のみ既定経路へ採用した。

検証はGPUテスト36件、Macの関連CPUテスト65件に合格。長い独立差分replayは232,056 policy-stage状態を比較した。512回目の打ち切り・リセットも独立CPU oracleとの4状態比較を通過した。MからのPPO更新は1・2 streamsで各2回成功し、更新後の重みrefreshと未完了ゲームの持ち越しを実行した。

測定値と実行ログは [`measurements/gpu_continue`](../measurements/gpu_continue/) に保存している。主な証跡は `gfrl-gpu-port-continue-final-engine.json`、`gfrl-gpu-port-continue-final-collection.json`、`gfrl-gpu-port-continue-full-suite.txt`、`gfrl-gpu-port-continue-tests-tma.txt`、`gfrl-gpu-port-continue-history-v2.txt`、`gfrl-gpu-port-continue-ppo-metrics-s{1,2}.jsonl`。

## 2026-11-14 pool / L 対応

`train.py` のCUDA経路で `--pool-fraction .2` を使用できる。CPUと同様、resume時は
保存された実験のpool設定を引き継ぐ。Lを新規に開始する例は次の通り。
既存checkpointを指定した場合は、そのcheckpointのモデル形状を使用する。

```sh
.venv/bin/python scripts/train.py --device cuda --env-backend cuda \
  --learner-backend packed-grouped --dtype bf16 --size L \
  --pool-fraction .2 --rollout-storage device --compile off \
  --num-envs 4096 --threads 4 --minibatch-size 4096 --microbatch-size 4096 \
  --gpu-pooled-arena --gpu-arena-bytes 262144 \
  --run-dir /tmp/$USER-gpu/train-L-new
```

空poolから開始し、通常は `--pool-every 10 --pool-size 8` で保存モデルを追加する。
20%は**ゲームをpool対戦にする確率**であり、そのゲームの片席だけが相手モデルになる。
相手席のdecisionを `sampled_decisions` や学習用batchに含めない。
`bench_gpu_collection.py --pool-fraction .2` はcheckpointの保存poolと現行学習重みを使い、
全外部decision、学習対象decision、終局済みの返却decisionを別々に出力する。
`positions_per_s` は学習対象decision/sである。

測定・精度・検証の詳細は [gpu_learner.md](gpu_learner.md) の同日記録を参照。
