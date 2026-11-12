# Search-policy distillation (expert iteration step 1)

```bash
# TSUBAME: use node-local scratch; no packages or downloads are needed.
PYTHON=.venv/bin/python DEVICE=cuda THREADS=16 NUM_ENVS=512 \
  DATA_DIR="${TMPDIR:-/tmp}/gfrl-distill" OUT_DIR=measurements/distill-1 \
  CHECKPOINT=runs/play/m-2166.pt BELIEF=runs/belief/train-m/checkpoint.pt \
  scripts/session_distill.sh
```

Defaults: 2.5 hours of both-seat search self-play, 0.75 hours of training,
1000 paired evaluation deals at seed 74100. Search uses 16 worlds, depth 3,
16 candidates from 64 samples, alpha .002, beta .1, actor leaf view, response
search and sample selection. Session enables forced/singleton inference skips.
The standalone generator follows `eval_search.py` option names (including AR,
B1 `--fit`, mixture/hybrid/count-temperature, cache, batching and precision).
Its skip options default off, as in `eval_search.py`.

CPU smoke with the local checkpoints:

```bash
DEVICE=cpu THREADS=1 NUM_ENVS=4 \
  BELIEF=runs/play/belief-ar-m.pt DATA_GAMES=8 GAMES_PER_SHARD=8 \
  DATA_HOURS=.002 TRAIN_HOURS=.01 WORLDS=2 DEPTH=1 \
  CANDIDATES=3 CANDIDATE_SAMPLES=6 PAIRS=4 \
  VAL_FRACTION=.8 MINIBATCH=32 MICROBATCH=4 \
  DATA_DIR=/tmp/gfrl-distill-smoke-data OUT_DIR=/tmp/gfrl-distill-smoke \
  scripts/session_distill.sh
```

The smoke's larger holdout threshold ensures this particular tiny seed batch has
both train and validation games. Production defaults to .03. An empty split is
an error: generate more games instead of validating on training games.
`VAL_FRACTION` is a hash threshold, not an exact quota for tiny data.

Useful overrides: `DATA_HOURS`, `TRAIN_HOURS`, `DATA_GAMES`, `GAMES_PER_SHARD`
(default 512), `WORLDS`, `DEPTH`, `CANDIDATES`, `CANDIDATE_SAMPLES`,
`MAX_INFERENCE_BATCH`, `SEARCH_CACHE_DEVICE`, `SEARCH_CACHE_MB`, `DTYPE=bf16`,
`TF32=0`, `MICROBATCH`, `MINIBATCH`, `EPOCHS`, `LR`, `KL_COEF`, `VALUE_COEF`.
`FIT`, `BELIEF_HYBRID`, `BELIEF_COUNT_TEMPERATURE`, `BELIEF_TEMPERATURE`,
`BELIEF_MIX` configure generation; evaluations use the untransformed AR `BELIEF`.
Use an AR checkpoint for the session's evaluation steps. `SKIP_REFERENCE=1`
skips the original+search evaluation. `PYTHON` is overridable.

## Data and reproducibility

Generation completes finite paired-game batches and atomically writes one
`shard-NNNNNN.pt.gz` per batch. `--max-hours` stops launching batches after the
budget; the in-flight batch finishes and is flushed (even when the budget is
zero). Only complete games are saved. Memory is bounded by the selected
`--games-per-shard` and `--num-envs`, not the total generation duration. Larger
batches improve parallelism but retain more observations until games finish.
Progress and summary report examples/s, compressed bytes/example, entropy,
main/response counts, candidate disagreement, seat-0 results and search/load
failure counts. Seat-0 win rate counts draws as half wins.

Default deal namespace: shard k uses `eval_seed=900000+k`, with paired games
numbered from zero inside the shard. Evaluation uses a separate namespace such
as 74100. `--resume` checks checkpoint MD5, belief/fit MD5, search configuration,
seed continuity and runtime settings, and continues after completed shards.
Use a new output directory for a new configuration. Custom seeds must remain
disjoint from evaluation seeds. Game identity is `(shard seed, episode_id)`;
paired games share a validation split because they share the initial deal.

Format v1 contains explicit public `OBS_FIELDS`, stage `PREFIX_FIELDS`, choice,
root→candidate and candidate→stage offsets, float64 search probabilities,
log_pi/log_rho/q, valid-world counts, game/seat/phase/decision and uint8 outcome
one-hots ordered **win, loss, draw**. Singleton-skipped q is NaN. Public float
observations stay exact; integer tensors use the smallest exact byte/signed
dtype, and gzip compresses padding/repeated prefixes. `gfrl.distill.load_shard`
and `prepare_batch` read this format. Metadata includes model path+MD5, belief
configuration, search configuration, format version and seed range.

`SearchAgent(record=True).records` holds only successfully searched roots.
Prefixes are copied before each actual candidate action; only retained paths
are serialized. Candidate-world observations must equal the root observation.
Belief sampling uses each root's acting-player observation, including when one
agent handles both seats. Oracle recording/generation is forbidden. No
`label_*` buffer or actual hidden opponent hand enters a model input or target.
To bound memory in another caller, drain/clear `agent.records` as games finish.

## Training and outputs

Standalone:

```bash
.venv/bin/python scripts/train_distill.py \
  --checkpoint runs/play/m-2166.pt --data-dir /tmp/gfrl-distill \
  --run-dir measurements/distill-1/train --device cuda --dtype bf16
```

Training starts from checkpoint EMA. The frozen reference is that initial
model. Each microbatch encodes each root once, then indexes its encoding for
candidate stages. The loss is `policy_coef * policy + kl_coef * anchor_kl + value_coef * value_ce`,
averaged over roots. `policy` is target-weighted path NLL, and `anchor_kl` is
the target-weighted sum of masked stage KL(theta || reference).
`--policy-coef` defaults to 1; setting it to 0 removes the policy imitation loss
while retaining the KL anchor on recorded prefixes and training the shared
encoder and value head.
`--value-coef 0` freezes value-head weights. Adam defaults to lr 1e-5 with ten
warmup updates, gradient clipping 1, minibatch 1024 roots and two epochs.
`--value-target-mix w` (default 0, range [0,1]) trains against the soft target
`(1-w)*outcome + w*[(1+v)/2, (1-v)/2, 0]`, ordered win/loss/draw. Here `v` is
`sum(probabilities*q)` from the root actor's view, clipped to [-1,1]. If any
candidate q is nonfinite (including singleton-skipped roots), that root uses
its outcome unchanged. At w=0 the loss and gradients match the original
outcome-only objective exactly.
Microbatches (default eight roots) bound activation memory without changing the
minibatch loss weighting. CPU, CUDA and MPS use the same code path.

`metrics.jsonl` includes initial, periodic and epoch-end train/validation
metrics. `policy` is path cross entropy; `candidate_kl` is
KL(renormalized policy paths || search), and `search_to_policy_kl` is its reverse.
Search probabilities are floored at 1e-30 for finite KL diagnostics only.
`value_ce` is CE against the **training target**, so its meaning changes with
`--value-target-mix`. Always-computed `outcome_ce` is CE against the real game
outcome; `outcome_brier` is scalar MSE of P(win)-P(loss) against win=1/loss=-1/draw=0
(not the three-class Brier score). `search_value_mse` is scalar MSE against the
unclipped search expectation, aggregated only over roots whose candidate q
values are all finite. It is null if none qualify. These metrics are also in
the step-0 validation of the initial frozen reference. Validation subsets,
including their candidates and prefixes, are copied to CPU memory once per
shard during split counting; periodic validation never reopens gzip shards.
Compact integer columns retain their types in the CPU cache; floating-point
probabilities/q are converted to float32 before device transfer (including MPS).
`anchor_kl` is the unscaled target-weighted stage KL. q expectations use only
roots with finite q for every retained candidate; `q_roots` reports that count.
Validation evaluates the exported weights, including optional `--ema-decay`.

`best.pt` minimizes validation policy loss; `last.pt` saves the latest update.
Both have train.py-compatible `model`, `ema`, `model_config`, `iteration`,
`pool`, `config` and optimizer fields. By default `ema` equals trained weights.
The session evaluates `best.pt`. `--max-hours` stops at a minibatch boundary,
then performs final validation/checkpoint writes; this overhead is outside the
training deadline. `--resume` restores last weights/optimizer/RNG and repeats
an interrupted epoch; it is not an exact mid-epoch cursor resume.

The session skips completed stages on rerun. An interrupted generation resumes
at its last completed shard; interrupted training uses `--resume`. A time-bound
successful stage counts as complete. To intentionally extend data or training,
run the corresponding CLI directly and use a fresh session directory for new
evaluations. Cached evaluation JSONs are reused; change `OUT_DIR` for new flags.
`summary.md` contains data details, initial/final/best validation metrics, three
paired-deal win rates and Wilson intervals, plus both-seat vs single-seat
fallback/rejection statistics. No fixed rejection-rate tolerance is enforced:
these rates depend on the model, belief and deal distribution.

Candidate-set agreement/KL/q do not measure policy mass outside retained paths.
Self-play changes state coverage, and tiny smoke win rates cannot establish an
improvement. Use the default held-out evaluation deals and adequate pairs for
an expert-iteration comparison.
