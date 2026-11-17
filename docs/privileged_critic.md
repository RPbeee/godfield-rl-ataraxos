# Privileged critic と noisy pool

`--critic shared` と `--pool-epsilon 0` が既定。従来の方策・観測・actor の
入力形式、PPO/targets の式、乱数呼び出しを維持する。`shared` では critic を
生成しない。変更前の learner を凍結した参照テストで、CPU の損失・勾配更新・
Adam・EMA が bit 一致することを確認する。追加の `advantage_std` は診断のみ。

## 学習

`--critic privileged` は方策とパラメータを共有しない独立 Transformer を作る。
公開観測を同じ形の encoder で処理し、CLS の値 logits に、以下の残差 MLP を足す。

```
h = GELU(Linear297→d(真の相手未使用手札の多重集合))
critic_logits = Linear_d→3(CLS) + MLP([CLS, h, unused_count / 18])
MLP = Linear_(2d+1)→d → GELU → Linear_d→3
```

手札の順序は入力しない。真のカード ID を使うため Dream の見かけにも依存しない。
hidden hand は値ヘッドで融合し、追加の手札トークンによる attention コストを避ける。
未来の RNG、霧で隠れた HP 等、相手だけの履歴は追加していない。

**既存の `label_opp_hand` / `label_opp_hand_count` は使用済みカードも含む**。
belief/search の既存契約を変えず、CPU/CUDA VecEnv に
`label_opp_unused_hand[297]` / `label_opp_unused_hand_count` を追加した。
`item.used == false` の実モデルだけを数える。collector は privileged のときだけ
これを訓練 rollout に保存する。`OBS_FIELDS` と policy の state_dict は変更しない。

完了した rollout の保存観測・ラベルを、更新開始時点の固定 critic で再評価する。
この確率を既存の trajectory ごとの λ-return/GAE に渡す。γ、λ、terminal/truncated、
pool 席の除外、filtering の扱いは従来と同じ。収集中・持ち越し中に critic の
古い値を固定する必要がなく、actor/CUDA Graph に critic の forward を入れない。
targets を全体で固定してから、方策と critic をそれぞれ 1 epoch 更新する。

方策の CE 値損失は、**公開 encoder 出力を detach** して元の値ヘッドだけに流す。
方策 encoder は PPO/KL/magnet だけで更新し、値損失との勾配干渉を避ける。
公開値ヘッドは同じ λ-return を教師として学び、評価・探索の既存 value API を維持する。
方策/EMA/pool は引き続き公開情報だけを使う。

critic は独立 Adam・独立 norm clip=1 を使う。`--critic-lr` は固定学習率（既定 1e-4）。
`--critic-d 0 --critic-depth 0` は方策の幅・深さ・FF を継承する。
幅を変えると FF=4d、attention heads は方策と同じ。packed-grouped では方策同様に
head 数が偶数かつ head dimension が 32/48/64 である必要がある。
`eager`（CPU を含む）と `packed-*` の両方で利用できる。
packed critic は既存 embedding/attention/autograd kernel を再利用し、最終層は
CLS の query 1 個だけを計算する。bf16 の残差・FP32 の重み/Adam/loss を維持する。

同じ形の旧 checkpoint から `--init-from` すると、online 方策の encoder と値ヘッドを
コピーする。hidden correction の最終層は 0 初期化し、最初の予測は既存の公開値に一致する。
異なる形を指定した場合は critic を新規初期化する。既存 critic と形が一致する
init-from は critic/optimizer/step も引き継ぐ。warm-up 専用期間は設けていない。

## Checkpoint と推論境界

通常の `checkpoint.pt` は従来の policy/EMA/optimizer/pool と設定を持ち、
critic の重み・Adam tensor を**含まない**。相対名 `critic_checkpoint.file` だけを持つ。
訓練専用の `checkpoint.critic-<iteration>-<id>.pt` が critic/optimizer/config/step を持つ。
伴走ファイルを書き終えてから主ファイルを atomic replace するため、途中で止まっても
主ファイルが未完成の critic を指さない。同じ主ファイルへの次回保存後に、直前の
自分が書いた伴走ファイルを片付ける。別名の snapshot が参照するものは削除しない。

**学習 checkpoint を移動・バックアップするときは主ファイルと参照先の 2 ファイルをコピーする。**
`gfrl.training_checkpoint.load_training_checkpoint` と train だけが伴走ファイルを読む。
`--resume auto` は保存した critic 設定・重み・Adam・step と RNG を復元する。
進行中の環境は従来の設計どおり保存せず、resume は新しい deal stream から始まる。
同じ batch/RNG からの learner 継続は CPU で bit 一致をテストする。

`load_policy_model`、evaluation/search、official bot/advisor、distillation は
通常の policy state_dict だけをロードする。伴走ファイルがなくても推論可能。
critic がない旧 checkpoint と、旧 pool selector (`@pool:N`) もそのまま利用できる。
official_* のコードは変更していない。

## 診断指標

- `critic_loss`: privileged critic の CE（固定 λ-return に対する全 decision 平均）。
- `value_loss`: 公開値ヘッドの CE。同じ target を使う。
- `critic_explained_variance`, `shared_value_explained_variance`: 同じ decision の
  終局結果 (-1/0/1) に対する W−L 値の EV。critic は更新前、shared は収集時の予測。
  自分自身の bootstrap target を基準にすると有利になるため、共通の終局結果で比較する。
  結果の分散が 0 の batch は EV=0 と定義する。
- `critic_brier`, `shared_value_brier`, `critic_accuracy`, `shared_value_accuracy`:
  同じ終局 W/L/D one-hot に対する指標。引き分け・打ち切りは draw。
- `advantage_std`: filtering/更新前の GAE の population std。正規化は追加しない。
- `learn_targets_s`: critic の事前評価と targets 計算を含む。
  `learn_critic_s`: critic の訓練時間。`learn_s` / `learn_decisions_per_s` は両者を含む。

## Noisy frozen opponent

`--pool-epsilon E`（0≤E≤1）は、pool/frozen 側の各合法選択 stage を
`(1-E)π + E Uniform(legal)` からサンプルする。これは各 stage で確率 E だけ
一様手に置き換えることと同じ分布。合法な完成行動全体の一様分布ではない。
ゲーム/席/モデルの routing は従来どおり固定し、online 側には適用しない。
CUDA Graph、1/複数 streams、pool の入れ替え・未完了ゲームにも同じ設定を使う。
E=0 では混合演算・追加の乱数消費を行わない。

評価も `scripts/eval_policies.py --pool-epsilon .15` で **B 側だけ**に同じ分布を使う。
A は通常の policy。出力 JSON に epsilon を記録する。

## 実行例

TSUBAME の計算ノードで、既存の評価が終了した後に実行する。キャッシュ/試験出力は
`/tmp/$USER-pcritic/` に限定する。以下は smoke 用の 30 反復までのコマンド。

```sh
cd ~/godfield-rl-ataraxos
module load gcc/14.2.0 cuda/13.1.1
export PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/tmp/$USER-pcritic/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER-pcritic/inductor
export GFRL_GPU_CACHE=/tmp/$USER-pcritic/gpu

# (a) 4500 から privileged critic 自己対戦
python scripts/train.py --init-from runs/hp-lr1e5/ckpt-4500.pt \
  --run-dir /tmp/$USER-pcritic/selfplay --device cuda --env-backend cuda \
  --dtype bf16 --learner-backend packed-grouped --compile off --no-tf32 \
  --num-envs 1024 --threads 4 --games-per-iter 1024 \
  --minibatch-size 4096 --microbatch-size 4096 --pool-fraction .2 \
  --critic privileged --critic-lr 1e-4 --lr 1e-5 --gamma 1 \
  --eval-every 0 --iterations 30 --max-hours .5

# (b) 相手を ckpt-4500 の EMA 1 個に固定する初期 checkpoint を作る
python - <<'PY'
import os
from pathlib import Path
import torch
destination = Path(f"/tmp/{os.environ['USER']}-pcritic/br-control-init.pt")
destination.parent.mkdir(parents=True, exist_ok=True)
state = torch.load('runs/hp-lr1e5/ckpt-4500.pt', map_location='cpu', weights_only=False)
state['pool'] = [dict(iteration=state['iteration'], weights=state['ema'])]
torch.save(state, destination)
PY
python scripts/train.py --init-from /tmp/$USER-pcritic/br-control-init.pt \
  --run-dir /tmp/$USER-pcritic/br-control --device cuda --env-backend cuda \
  --dtype bf16 --learner-backend packed-grouped --compile off --no-tf32 \
  --num-envs 1024 --threads 4 --games-per-iter 1024 \
  --minibatch-size 4096 --microbatch-size 4096 \
  --pool-fraction 1 --pool-every 1000000 --pool-size 1 --pool-epsilon .15 \
  --critic privileged --critic-lr 1e-4 --lr 1e-5 --gamma 1 \
  --eval-every 0 --iterations 30 --max-hours .5

# BR の前後を、同じ noisy frozen 相手で測る（critic 伴走ファイルは読まない）
python scripts/eval_policies.py --a /tmp/$USER-pcritic/br-control-init.pt \
  --b /tmp/$USER-pcritic/br-control-init.pt@pool:4500 --pool-epsilon .15 \
  --pairs 100 --num-envs 64 --threads 4 --device cuda --eval-seed 917 \
  --out /tmp/$USER-pcritic/br-control-before.json
python scripts/eval_policies.py --a /tmp/$USER-pcritic/br-control/checkpoint.pt \
  --b /tmp/$USER-pcritic/br-control-init.pt@pool:4500 --pool-epsilon .15 \
  --pairs 100 --num-envs 64 --threads 4 --device cuda --eval-seed 917 \
  --out /tmp/$USER-pcritic/br-control-after.json
```

短い smoke は実装・数値の健全性の確認であり、plateau 解消や BR の棋力改善の証明にはならない。
実測の throughput と smoke 指標は
[2026-11-17 検証記録](../measurements/pcritic_20261117/README.md) に記録した。
