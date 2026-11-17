# 運の雑音・手の差・GAE の offline 診断

`scripts/luck_diagnostic.py`。bot・探索・learner から呼ばれない独立 CLI。
両者の **online `model`** を固定し、学習 collector と同じカテゴリカル sampling を使う。
`--weights ema` で評価用 EMA に変更可能。CUDA の `--dtype auto` は checkpoint の
学習 dtype、CPU は fp32。CUDA の TF32 設定も checkpoint から引き継ぐ。
相手 pool、探索、privileged critic は使わず、shared value head を読む。

## 実行

この worktree だけでビルドする。メインの `.venv` と checkpoint は読むだけ。

```sh
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build
```

Mac の小規模検証（本番の結論には使わない）:

```sh
.venv/bin/python scripts/luck_diagnostic.py \
  --checkpoint /Users/yuto/godfield-rl-ataraxos/runs/play/m-xit1.pt \
  --device cpu --threads 1 --games 8 --num-envs 8 \
  --roots-per-game 2 --root-selection reservoir \
  --chance-repeats 4 --policy-repeats 4 \
  --max-candidates 8 --candidate-nodes 256 --prefix-batch 8 \
  --world-batch 128 --inference-batch 64 --bootstrap 100 \
  --out measurements/luck-reservoir-cpu.json
```

TSUBAME では CUDA が見える既存環境で、最初に wide pilot:

```sh
python scripts/luck_diagnostic.py \
  --checkpoint runs/hp-lr1e5/ckpt-4500.pt --device cuda --threads 32 \
  --games 16 --num-envs 16 --roots-per-game 2 --root-selection reservoir \
  --mass .99 --min-candidates 2 --max-candidates 32 --candidate-nodes 2048 \
  --wide-candidates 8 --wide-method prefix-uniform --wide-max-pi .01 --wide-draws 256 \
  --chance-repeats 16 --policy-repeats 16 \
  --world-batch 2048 --inference-batch 512 --cache-mb 1024 \
  --bootstrap 200 --out measurements/luck-wide-pilot.json
```

本番の推奨開始点（最大 256 根、追加最大 8 手、16×16。以下以外は既定値）:

```sh
python scripts/luck_diagnostic.py \
  --checkpoint runs/hp-lr1e5/ckpt-4500.pt --device cuda --threads 32 \
  --games 128 --num-envs 32 --roots-per-game 2 --root-selection reservoir \
  --mass .99 --min-candidates 2 --max-candidates 32 --candidate-nodes 2048 \
  --wide-candidates 8 --wide-method prefix-uniform --wide-max-pi .01 --wide-draws 256 \
  --chance-repeats 16 --policy-repeats 16 \
  --world-batch 2048 --inference-batch 512 --cache-mb 1024 --bootstrap 2000 \
  --out measurements/luck-wide-production.json
```

各根 `(従来候補数 K + tail 1 + 実際の追加数 N)×16×16` 継続。
external decision が 2 回未満のゲームでは根が減る。K は実着手を含み、
最大候補数には実着手追加で +1 があり得る。低 π の合法手が少なければ N は減る。
**CUDA は推論だけ、ゲーム進行は C++ CPU workers**。GPU engine の oracle importer
を新設する代わりに、CPU の完全状態をそのまま利用する。40GB MIG 向けに推論を
512 行、world を 2048 行に分割し、encoding cache は 1GB 上限かつ空き GPU
メモリの 1/4 以下。収まらない cache は host に置く。共有 GPU で不足する場合は
`--inference-batch 256 --world-batch 1024 --cache-mb 512`。

時間見積りは前回の **493 根（要求 512）、16×16、約 85 分、16 threads で
約 1700 continuation decisions/s** という実測を基準にする。32 threads が比例して
速くなるとは仮定しない。同じ残り長・速度で K が 2 / 3 / 5 なら、追加 8 手・256 根は
`85分 × (256/493) × (K+9)/(K+1)` = **約 162 / 132 / 103 分**。
したがって **約 2–3 時間を狙う設定**。これは K の単純例であり、厳密な見積りには
根の残り長で重み付けした arm 数が必要。追加手がゲーム長を変えること、候補生成と
bootstrap の時間、共有 GPU の混雑で超過し得る。新機能の CUDA 計時は手元では未実測。
pilot は同じ 16×16 と 32 threads なので、本番はその継続時間の約 8 倍が目安。
pilot の diagnostic_seconds が 20 分超なら 160 分超となるため、
本番を `--games 96`（最大 192 根）に下げて約 6 倍にする。比較時は
`timing.continuation_decisions`、速度、N の不足、根の phase/progress 分布も確認する。
CI が未解決なら根数か種数を増やし、精度不足をゼロと解釈しない。
特に chance main が大きいなら chance 種、policy main が大きいなら policy 種を
増やす。交互作用優勢なら両方を増やす。GPU・TSUBAME にはここから接続していない。

## 抽出・完全状態・候補

既定の `--root-selection reservoir` は、各ゲームの全 external decision を流しながら
標準の reservoir sampling で最大 roots-per-game 個の異なる時点を保持する。
両 seat を含め、終局後に各ゲーム内で一様な根が残る。ゲームを等重みにしてその中の
決定点を抽出する分布であり、長いゲームを長さに比例して重くする分布ではない。
比較用の `--root-selection window` では、開始前に `[0,target-range)` から異なる
index を一様に抽出し、未到達を置換せず別計上する。この場合だけ生存・窓内に条件付く。
`selection` と各根の開始 index・残り長・progress を保存する。
実ゲームの方策 RNG は独立。実着手の完全な外部 choice 列とその時点の V を保存する。

`SearchEnv.diagnostic_copy_exact` は Match、選択 cursor、**双方の実履歴**をコピーする。
既存 oracle loader の determinization・履歴 projection・own Dream 再抽選を使わず、
通常の探索で未対応の根も対象にする。コピー先は別 SearchEnv に限定。
チャンスの将来だけを再抽選する際、Match の RNG と grant/confusion の外部 RNG を
変更し、応答 launch・過去の命中・redirect anchor は保持する。
応答 snapshot の replay proof は過去 RNG と結び付くため、`restore()` で再ロードせず
診断内で非 const Match の RNG フィールドだけを明示的に差し替える。この介入後の
snapshot は履歴 replay 用ではない。ゲームの step は同じ既存 engine を使う。
`third_party/goodfield` は変更しない。

候補は prefix 確率を上界とする best-first 列挙。各完全パスの正確な joint π を計算し、
上位から mass（既定 .95）、最低 2 手、最大 32 手を採る。実着手は必ず追加。
`candidate_nodes` が上限。`top_certified` は未展開 prefix の上界との比較、
`coverage_met` と `candidate_mass` は実際のカバー率。予算不足を隠さず、
coverage 達成/未達、多候補の集計も別に出す。最大 Q はこの候補集合内の最善で、
全合法手の最善を保証しない。

全方策平均は `Σ_S π(a)Q(a) + tail_mass×Q(tail)`。
候補外の π に条件付けた追加 1 arm を、各 prefix の残存確率から rejection なしで
サンプリングする。よって候補集合に再正規化した平均ではない。
tail ≤ 1e-7 では桁落ち回避のため通常 π サンプルと既知候補の CRN control variate を使う。
raw NPZ の arm `K` は tail（微小 tail では通常 π）であり、JSON の `q[-1]`
と covariance の最後の arm は上式の **層化された全方策平均推定量**。両者を混同しない。
wide 無効時は raw の最後が tail。wide 有効時はその後に追加手を並べる。

## 低 π の追加手（wide、明示的に有効化）

`--wide-candidates 0` が既定。新オプション無指定時は従来の候補、tail、乱数、
評価バッチ、統計を維持する。有効時も `candidate_paths/pi/mass`、`tail_mass`、
`actual_index`、`q`、`summary` は従来集合だけを表す。追加手を全方策平均の層化に
組み込まない。追加手自身が tail サンプルで選ばれることは正しい挙動。
追加は `roots[i].wide` と `wide_summary` に保存し、schema_version は 2（無効時は 1）。
raw NPZ は `[従来 K 手, tail, 追加 N 手]`。C/P seed はすべての arm で同じ。

| オプション | 既定 | 意味 |
|---|---:|---|
| `--wide-candidates` | 0 | 根あたりの追加手数上限。推奨 8 |
| `--wide-method` | prefix-uniform | prefix-uniform / tempered |
| `--wide-temperature` | 2 | tempered の温度。1 より大きいと平坦化 |
| `--wide-max-pi` | .01 | 採用する完全パスの joint π 上限（以下を採用） |
| `--wide-draws` | 1024 | 根あたりの proposal 試行上限。推奨開始値 256 |

`prefix-uniform` は各 prefix の**合法分岐**を一様に選んで external decision が
完了するまで進む。**完全な合法手集合からの一様分布ではない**。分岐数・深さの
違いが proposal 確率に影響する。`tempered` も prefix ごとに
`π(choice|prefix)^(1/temperature)` を合法分岐内で正規化する方式で、完全パスの
`π(a)^(1/T)` 分布とは限らない。いずれも元の joint π は温度をかけずに計算し、
retained prefix の確率は元の列挙と共有する。π が underflow する合法手も uniform
proposal から到達でき、`log_pi` で bucket と上限を判定する。

従来候補との重複、追加手同士の重複、π 上限超過を棄却し、採用順を保存する。
候補生成専用の根別 RNG を使うので collection、chance、continuation RNG を消費しない。
試行予算内で手数を満たせないと `shortfall`、`draws`、`rejected` を報告する。
shortfall は合法手を尽くしたという証明ではない。各根の最初の m 手を使うため、
1・2・4・8…手の結果は独立した実験ではなく入れ子の曲線になる。

### wide 統計の定義

`wide_summary` は `overall/main/response/early/middle/late` と phase×stage の
各 group に `estimate`、`metric_roots`、`bootstrap_root_95`、
`bootstrap_deal_pair_95` とそれぞれの valid_fraction を保存する。
root bootstrap は group 内の根を独立に再抽出する参考区間、deal-pair bootstrap は
同一 deal の両ゲーム・全根をまとめて再抽出する主区間。どちらも root 加重。
bootstrap 数は `--bootstrap`、wide の種は `seed+10`。有効 replicate <97.5% は null。
根ごとの各手・各 fold には crossed MC 分散による normal 95% 区間も保存する。

* `distribution/{bucket}/mean_advantage`: 追加手の `Q(a)-Q(full π)`。
  各根内で追加手を等重みにし、根間でも等重みにする。π bucket は
  `<1e-6`、`[1e-6,1e-4)`、`[1e-4,1e-3)`、`[1e-3,.01)`、`≥.01` と all。
  該当手がある根だけで集計し、`metric_roots` が分母。
  `actions` は該当根あたり平均手数。`second_moment_mc_corrected` は
  `mean(Â²-Ω_A)`、`mean_mc_variance` は平均 Ω_A。Ω_A は方策平均との
  **共分散を含む差の MC 分散**。
  `observed_cdf_at_{-.5,-.1,0,.1,.5}` は noisy な Â の経験分布。
  これは真の Q 分布の deconvolution ではなく、0 超の比率も真の改善率ではない。
  全合法手分布の推定でもない。指定 proposal で得た追加手集合の診断であり、
  importance weighting による他分布への補正は行わない。
* `discovery/{m}/heldout_gap`: 元の全候補（実着手を含む）で train Q 最大の手と、
  最初の `min(m,N)` 追加手で train Q 最大の手を選び、独立 test 象限で
  `Q(wide選択)-Q(従来選択)` を評価。C/P を両方二分して対角象限を入れ替える
  2-fold cross-fit の平均。各 fold は `wide.discovery[m]` に保存する。
  これは**選択手どうし**の比較で、oracle max 同士の gap の不偏推定ではない。
* `train_positive_fraction` は train gap >0 の fold 比率を根平均した値。
  `promoted_gap` は train gap >0 の時だけ wide を採用した場合の held-out 改善量
  （不採用は 0）。集計の `promoted_gap / train_positive_fraction` は、
  train で採用された root/fold における平均 held-out gap。test の勝敗で条件付けない。
  `heldout_win_fraction` は test gap >0 の fold 比率の根平均。選択の楽観は避けるが、
  有限 test 雑音による符号判定の誤りは残る。**真に良い手がある根の割合とは異なる**。
* `evidence_fraction` は train で選んだ追加手が、test で**元の K 手すべて**を
  上回る片側 lower bound を持つ fold が一つ以上ある根の割合。
  各比較は `Δ_j - Φ⁻¹(1-.05/(2K)) sqrt(Ω_Δj) > 0` を要求する。
  K 比較と 2 folds に Bonferroni 補正した、根・m ごとの近似 5% 検定。
  分散が負ならその fold は証拠に数えず `evidence_unresolved_fraction` に計上。
  複数 m・group 間は補正していない。normal 近似依存であり、特に 4×4 の CPU
  検証では精度の主張に使わない。この割合と区間は「証拠を得た率」の統計であって、
  真の存在率の信頼区間ではない。未検出を「存在しない」と解釈しない。
* N=0 の根も `examined`、train/win/evidence 率、promoted gap の分母に残し、0 とする。
  heldout gap と分布・V 統計は追加手がある根だけ。`has_extra`、`shortfall`、
  `available_actions` と `metric_roots` を併読する。m 別の `examined` は実際の平均手数。
* `value/*` は追加手内の Q と、手の直後の V（根 actor 視点、chance 平均）の
  等重み・根内中心化分散／共分散を、joint crossed MC 共分散で補正した値。
  根間平均の比から `within_root_correlation/r2` を出す。相関の符号も確認する。
  m 別 `value_gain_vs_wide_mean` は train の post V 最大で選び、test の Q を
  同じ m 手の等重み Q 平均と比較する。`value_selected_advantage` はその選択手と
  full π の比較。V の順位によって改善を選べるかを held-out で直接測る。

wide panel のどこかに失敗・打ち切りがあれば根の wide 統計を除外する。
従来 panel が完全なら従来統計は維持する。実ゲーム打ち切りは双方の集計から除外。
失敗・shortfall を黙って置換せず、観測できた条件付き集団であることを明示する。

## 統計の定義と出力

JSON に root 統計・ゲーム情報・group 別 point estimate と deal-pair bootstrap 95% 区間。
同一 deal の 2 ゲームと全根を cluster として再抽出し、root 加重で集計する。
`overall/main/response`、実ゲームの進行率三分割 `early/middle/late` とその交差、
開始/残り decision 数の bucket がある。進行率分類は終局後に定義する。
raw seed・outcome・GAE・post V は同名 `.samples/root-xxxxx.npz` に保存。
JSON は根ごとに atomic 更新し、例外・中断時も終了済み根の集計を残す。
`final=false` は未完了。再開用の状態 archive は保存しない。

* **Q**: 真の完全状態から、根だけ候補パスを強制し、以後両者が sampling して真の
  終局まで進めた結果 `Y∈{-1,0,+1}` の平均。draw は 0。根 actor の視点。
  chance C 種と policy P 種の全交差を使う。各候補は同じ C/P の seed を共有し、
  continuation の policy counter は根の強制パスから独立。枝の長さや batch 境界は
  別候補の乱数列をずらさない。root Q の `q_mc_se/q_normal_95` は下記 MC 分散に
  よる近似で、多重選択の補正ではない。
* **交差分散**: `Σ_I` は二元 ANOVA の residual cross-product / `(C-1)(P-1)`、
  `Σ_C = Cov_C(mean_P Y) - Σ_I/P`、`Σ_P = Cov_P(mean_C Y) - Σ_I/C`。
  一局の共分散 `Σ=Σ_C+Σ_P+Σ_I`、Q 平均の MC 共分散
  `Ω=Σ_C/C+Σ_P/P+Σ_I/(CP)`。CP 局を独立と扱わない。
* **信号**: 候補内重み `w=π/Σ_Sπ`、`M=diag(w)-wwᵀ` として
  `signal=q̂ᵀMq̂-tr(MΩ)`。候補間の方策加重 Q 分散で、有限標本補正済み。
  `unweighted_signal` は w を等分にした値。`signal_raw` は補正前。
  **雑音** `outcome_noise=Σ_a w_a Σ_aa`。**SNR** は平均信号 / 平均雑音という分散比。
* **運の内訳**: `chance_main/policy_main/interaction` は上記成分の候補加重対角平均。
  chance を固定した残り分散は `policy_main+interaction`、policy 固定は
  `chance_main+interaction`。両条件付き分散は重複するので単純な二分割はできない。
  `chance_shapley/policy_shapley` は交互作用を半分ずつ配る明示的な便宜的配分。
* **CRN**: 各候補対の独立分散 `Σ_aa+Σ_bb` と共通乱数分散
  `Σ_aa+Σ_bb-2Σ_ab` を `w_a w_b` で加重平均し、その比を出す。
  `crn_pair_variance_ratio` は一局対、`crn_grid_variance_ratio` は Σ を Ω に置換した
  C×P 推定平均の比。候補間相関と共分散行列も保存する。比 >1 もそのまま保持。
* **最善候補と方策平均**: 同じサンプルで最大 Q を選んだ値は
  `best_gap_optimistic` と明記。主評価は C と P を**両方**二分割し、片方の独立象限で
  最善候補を選び、もう片方で評価して入れ替える cross-fit。
  `heldout_best_gap` は、この選択法による手の評価であって oracle max Q の不偏推定ではない。
  Δ² からも MC 分散を引く。正の gap が検出できた場合だけ `4×分散/Δ²` を出す。
  独立比較は best と通常 π の **各 arm n 局**、CRN は各候補＋tail を共有 seed で
  評価する **n 組**で、1 組に K+1 継続を要する（CRN の n を総局数と読まない）。
  根ごとの値は `heldout_best`。集計の必要数は平均分散 / 平均補正 Δ² に基づく
  **RMS gap の代表値**で、根ごとの n の平均ではない。負/未検出の gap は null。
* **GAE**: 実ゲームで記録した各 seat の系列に `sequence_targets` をそのまま適用。
  λ_v=.8、λ_a=.5、γ=1 が既定。λ_v は value target 用で GAE 自体を変えない。
  continuation でも同じ GAE を online の等価再帰で計算する。
  真の実着手 advantage は `A=Q(actual)-Q(full π)`。
  `true_advantage_variance=Var_root(Â)-mean Ω_A`、相関の分母も MC 補正する。
  元の GAE と MC が独立なので共分散の補正は不要。
  `Var(GAE)=Var(A)+Var(GAE-A)+2Cov(A,GAE-A)` の全項と bias を出す。
  GAE の bootstrap bias がある場合に交差項を捨てた二分割を強制しない。
  continuation の conditional GAE 平均/分散から conditional bias の MSE・分散、
  total variance の別推定も出す。
* **V と Q**: 根 V と全方策 Q/実着手 Q、根の手が解決した直後の次 actor の V
  （根 actor 視点へ符号変換、終局なら結果）、次の自分の decision の V と実着手 Q。
  post V は chance を平均した値。各回帰は MC 共分散も補正した correlation² の R² と
  calibration MSE を出す。`post_value_within_root_q_r2` は根平均を除去した候補差の
  説明率で、根間の価値差による見かけの説明率と区別する。

補正後の負の分散・範囲外相関/R² は精度不足の情報なのでクリップしない。
分母非正・未識別の比は null。bootstrap でも識別不能な replicate を都合よく捨てず、
`bootstrap_valid_fraction` を併記する。必要局数の上限 null は無限/未解決、
その他の統計で有効率 <97.5% の区間は null。打ち切りやロード失敗は world 別に
別計上し、一つでも失敗のある従来 crossed grid の**根全体を従来統計から除外**する。
実ゲーム打ち切りも除外。打ち切りを draw としない。失敗がある測定は条件付き集団の
診断となり、censoring bias がある。ゼロ失敗または失敗要因の追加調査なしに一般化しない。

区間だけを増やして再集計（raw 継続の再実行はしない）:

```sh
python scripts/luck_diagnostic.py --summarize measurements/luck-production.json \
  --bootstrap 5000 --out measurements/luck-production-ci.json
```

wide report も同じ `--summarize` で両 summary を再計算する（新たな候補抽出や
cross-fit 選択は行わず、保存済みの根統計から bootstrap を作り直す）。

運補正への判断材料は chance main と「policy 固定時の残り分散」、CRN 学習への判断材料は
候補対の分散比・信号・GAE との相関。交互作用が大きいだけでは AIVAT の実現可能な
除去率を特定できない。また、seed を同じにしただけの CRN は異なる枝の同じ事象を
整列させる保証がなく、減少率は実測値として読む。

## 検証記録（2026-11-17）

ビルド成功、CTest **19/19**、指定 MPS case を除く pytest **1487 passed,
122 skipped, 1 deselected**（最終全体実行207.49秒）。既存 read-only NumPy→Torch warning 1 件。
追加テストは完全 snapshot・両履歴・main/response/途中 cursor・元状態不変、
同 seed の再現、crossed ANOVA の全符号組合せでの不偏性、CRN と信号補正、
conditional tail、learner GAE との等価性、推論/world 分割不変性、censoring 除外。
CUDA はこの Mac では未検証。

根の確率は retained prefix ごとに固定・記録する。CUDA bf16 の GEMM の batch サイズに
よる丸め差が、非常に小さい tail の残存質量を壊すことを避ける。continuation の
丸めまで異なるハードウェア・batch 間で bitwise 一致する保証はない。

既定の reservoir による CPU 確認は `m-xit1.pt` online、8 games/16 roots、4×4 seeds、
候補最大8。メインに `runs/hp-lr1e5/ckpt-4500.pt` は無かった。所要 **108.14秒**、
26,352 decisions、**247.6 decisions/s**、800 requested / 800 completed、load failure・
continuation 打ち切り・実ゲーム打ち切りは全て0。有効根16（main11 / response5、
early7 / middle4 / late5）、平均候補質量 **.998958**。
補正信号 **−.0000658**、雑音 **.761359**、SNR **−.0000864**（負値を保持）。
一局対 CRN 分散比 **.72284** [95%: .50434, 1.02282]、grid 平均 CRN 比 **.79669**
[.39489, 1.22153]。GAE と A の補正相関 **−.09830** だが区間は識別不能で null、
必要局数も未解決で null。小規模 bootstrap の負の信号区間を物理的な負の分散と解釈しない。
**別 checkpoint・少数の根/seed の動作確認であり、停滞原因や進路の結論には使わない。**
数値と区間は `measurements/luck-reservoir-cpu.json`、raw は隣の `.samples/`。
比較用 window モードも確認済み（`luck-cpu-smoke.json`、864 中863終局、打ち切り1を
別計上し、その根を除外）。CPU 計時は他のテストと並走した参考値。
コミット・ネットワーク利用・TSUBAME 接続・メインへの書き込みは行わない。

## wide 拡張の検証記録（2026-11-17）

同じ worktree 内でビルド成功、CTest **19/19**、指定 MPS case を除く全 pytest
**1555 passed, 122 skipped, 1 deselected**（204.23 秒）。既存の read-only
NumPy→Torch warning 1 件。診断テストは 20 case。
追加あり／なしで従来 arm の outcome/GAE/post V/next-own V/status/tail index を
array 完全一致、全従来 metrics と q を辞書／list 完全一致で確認した。
合法パス replay、重複除外、π 上限、shortfall、両 proposal、確率 underflow でも
合法手に到達すること、選択・評価象限の分離、MC 共分散、bootstrap の cluster 単位、
追加手だけ失敗した場合の従来統計維持、CLI の不正値をテストしている。
HEAD 版との直接比較でも、wide 無効の 4 根について collection・候補・全 raw 配列・
全根指標・summary が完全一致した（`measurements/luck-wide-legacy-equivalence.log`）。
wide 無効時の config にも新フィールドを加えず、schema_version=1 を保つ。
実 checkpoint の CPU report を `--summarize` し直し、両 summary の完全一致も確認した。

JSON は `_json_default` を全 `json.dumps` に適用し、NumPy bool/int/float/ndarray を
含む初期保存・途中保存・最終保存・`--summarize` とその stdout を確認。
CUDA device_name 分岐を mock して CPU から保存経路を通した。
CUDA 推論と bf16 の実機検証はこの macOS では実施していない。

checkpoint を使った小規模 CPU 確認:

```sh
.venv/bin/python scripts/luck_diagnostic.py \
  --checkpoint runs/play/m-xit1.pt --device cpu --threads 1 \
  --games 4 --num-envs 4 --roots-per-game 2 --root-selection reservoir \
  --chance-repeats 4 --policy-repeats 4 \
  --max-candidates 8 --candidate-nodes 256 --prefix-batch 8 \
  --wide-candidates 4 --wide-draws 256 \
  --world-batch 128 --inference-batch 64 --bootstrap 100 \
  --out measurements/luck-wide-validation.json
```

8 根、追加 30 手（根平均 3.75、平均 shortfall .25）、880/880 継続が終局、
load failure・打ち切り 0。約 72.5 秒、19,033 continuation decisions、約 265 decisions/s。
以下は deal-pair bootstrap 95% 区間（わずか 2 deal-pairs）:

| 指標 | 点推定 | 区間 |
|---|---:|---:|
| 追加手の根平均 `Q−Qπ` | −0.2561 | [−0.5064, −0.0058] |
| 最大 4 手の cross-fit 選択 gap | −0.2813 | [−0.6875, 0.1250] |
| held-out gap >0 の fold 率 | 0.125 | [0, 0.25] |
| 元候補すべてを上回る evidence 率 | 0 | [0, 0] |
| post V 選択 Q − 追加手平均 Q | 0.0573 | [−0.0365, 0.1510] |

補正した根内 V/Q 相関は 0.6415、区間は識別不足で null。
**これは m-xit1 の動作確認だけで、ckpt-4500 や低 π 探索の有効性の結論には使わない。**
特に 0 の evidence 区間は「存在しない」証明ではない。全例ゼロの通常 bootstrap は
未検出の母集団率を推定できず、小さな crossed grid の検出力も低い。
JSON、stdout log、raw NPZ は `measurements/luck-wide-validation*` に保存した。
