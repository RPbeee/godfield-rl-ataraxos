# m-2166 AR belief warm start / TSUBAME

ネットワーク・追加依存は不要。リポジトリルートで実行する。

```bash
PYTHON=.venv/bin/python DEVICE=cuda bash scripts/session_belief_retrain.sh
```

既定の出力は `runs/belief-m2166`。`OUT`（または `OUT_DIR`）で変更できる。
旧 belief の encoder / decoder 全重みから開始し、optimizer と step は初期化する。
自己対戦を生成する policy は m-2166。初期化元のパスと MD5 は学習の
`config.json` と checkpoint に保存する。再開時に初期化元ファイルは不要。
新規学習は既存 run dir を拒否する。

学習・fit・self 検証は D5 と同じ latest/pool = 80:20 (`--pool-fraction 0.2`)。
当初のデータ生成例の 0 より、タスク後段の整合性指定を優先する。
greedy 検証も設定値は 0.2 と記録するが、対戦相手は常に greedy で pool は使わない。
診断は既存スクリプトの pool=0 の自己対戦、seed=0 を新旧両方に用い、同じ局面を比較する。

学習は lr=1e-4、warmup=200、replay ratio=2、buffer=200000、
warmup positions=50000、batch=1024、microbatch=128、eval/save=500、
log=50、検証 batch=64、presence samples=32、512 envs、4 threads、TF32。
presence sampling は一度に最大 32 samples に分割する。

総ステップは次で計算する（明示的な `STEPS` が優先）。

```text
floor(((MAX_HOURS - SCHEDULE_OVERHEAD_HOURS) * 3600 * COLLECT_POSITIONS_PER_S
       - WARMUP_POSITIONS) * REPLAY_RATIO / BATCH_SIZE)
```

既定では rate=290、MAX_HOURS=6.5、overhead=0 → **13156 steps**。
290 × 3600 × 2 / 1024 は **2039 steps/h** で、依頼文の 3300 steps/h と
一致しない。3300 steps/h には約469 positions/s が必要。
この式は収集律速の上限であり、更新・step 0 および定期検証の時間を含まない。
実機のログから `SCHEDULE_OVERHEAD_HOURS` を指定して cosine 終端を調整する。
`MAX_HOURS` は初期化・検証を含む学習呼び出し全体の上限（実行中の一回の検証は中断しない）。
再開は指定した総ステップ数を保ち、replay を再収集する。

データは一時ファイル完成後に公開する。完成した段階はスキップし、学習 checkpoint が
あれば `--resume` する。正常に時間上限または総ステップに達すると `train.done` を記録し
評価へ進む。学習をさらに延長する場合は別のセッションを使用する。
セッション設定・入力ファイル情報が変わった場合は混在を避けるため別 OUT を要求する。
各段階のログ、`timings.json`、JSON/Markdown 評価、全表を含む `summary.md` が残る。
中断された対戦評価はその対戦全体をやり直す。

探索は worlds=16、depth=3、candidates=16、candidate samples=64、alpha=.002、
beta=.1、leaf-view=actor、response search、forced inference / single candidate の省略。
新旧それぞれ対 policy、さらに新対旧の直接対戦を同じ seed=74100、1000 pairs で行う。
`eval_search.py --opponent-belief PATH` は対戦相手にも同じ探索設定を適用する。
勝率・Wilson 区間・pair bootstrap 区間・探索統計を各レポートに記録する。

CPU smoke:

```bash
OUT=measurements/session-belief-retrain-smoke DEVICE=cpu NUM_ENVS=8 THREADS=2 \
FIT_POSITIONS=200 VALIDATION_POSITIONS=200 DIAGNOSTIC_POSITIONS=200 \
DIAGNOSTIC_SAMPLES=2 BOOTSTRAP=10 STEPS=5 MAX_HOURS=0.25 \
BUFFER_SIZE=256 WARMUP_POSITIONS=32 BATCH_SIZE=8 MICROBATCH=4 WARMUP_STEPS=2 \
EVAL_EVERY=5 SAVE_EVERY=1 LOG_EVERY=1 EVAL_BATCH_SIZE=8 \
PRESENCE_SAMPLES=2 PRESENCE_SAMPLE_BATCH_SIZE=2 PAIRS=2 \
WORLDS=1 DEPTH=0 CANDIDATES=2 CANDIDATE_SAMPLES=2 \
MAX_INFERENCE_BATCH=128 SEARCH_CACHE_MB=64 bash scripts/session_belief_retrain.sh
```

`CHECKPOINT`、`BELIEF` で入力を変更できる。予算・バッチ・探索設定の環境変数は上記の
ほか `LR`、`REPLAY_RATIO`、`COLLECT_POSITIONS_PER_S`、`SCHEDULE_OVERHEAD_HOURS`、
`TF32`、`PROGRESS_INTERVAL`。CPU smoke の探索予算は動作確認用で性能判断には使わない。

TSUBAME H100 MIG / 12h ジョブの暫定見積もり:

| 段階 | 見積もり | 根拠・限界 |
|---|---:|---|
| fit 1M | 約2.4h | sample .25 の290/sを .1 に比例換算すると116/s。labels-only により短縮する可能性あり |
| self / greedy 検証 各20k | 合計約6min | 同じ116/sの目安。greedy側は policy 推論が減る |
| 学習 | 6.5h | step 0 と定期検証を含む。単一検証の時間分は上限超過しうる |
| 新旧 likelihood 評価 | 実測要 | 2 checkpoints × 40k positions × 32 presence samples。生成速度からは推定不可 |
| 新旧診断 各3k | 実測要 | AR/hybrid/B1 各64 samples、拒否再試行と bootstrap を含む |
| 探索3対戦 各2000 games | 約80min | 既存 session の10min/1000 games/探索側より、20+20+40min。MIG・対戦長に依存 |

既知部分だけで約10.3h。残り約1.7hに likelihood・診断が収まるかは MIG 上で実測が必要。
収集の .1/.25 比例換算が成立しない場合や検証が重い場合、12h を超える。
分割ジョブで同じコマンドを再実行できる。これらは実機測定ではなく計画用の概算。

2026-11-13 ローカル CPU smoke 実測（上記コマンド）:

| 段階 | 秒 |
|---|---:|
| fit 200 | 9.48 |
| self / greedy 各200 | 9.31 / 7.39 |
| warm start 5 steps、step 0/5 検証込み | 18.16 |
| 新旧 likelihood | 15.45 |
| 新 / 旧診断 各200、2 samples | 24.40 / 21.60 |
| 新 vs policy / 旧 vs policy / 新 vs 旧 各2 pairs | 5.22 / 7.27 / 5.33 |
| 合計 | 123.62 |

同じコマンドの再実行で完成済み全段階を再利用することを確認した。
H100 MIG の likelihood・診断時間は CPU smoke だけからは確定できない。
サイズ・samples 数を比例拡大した CPU 上限目安はそれぞれ約6.9h・6.1h。
仮に10〜30倍の高速化なら likelihood 約14〜41min、診断合計約12〜37min。
この**未検証の高速化仮定**で全体約10.7〜11.6h。12h枠には余裕が少なく、
定期検証が重い場合や MIG の混雑時は次ジョブで再開する。

検証結果: full pytest **1293 passed, 2 skipped** / 165.71s、ctest **18/18** / 7.98s。
変更後の belief / sweep / search tuning 対象37件、および最後のCLI調整後の
sweep / search tuning 23件も成功。既存の非 writable NumPy 配列に関する警告1件あり。
ネットワーク利用・依存追加・commit・指定禁止ファイルの編集は行っていない。
