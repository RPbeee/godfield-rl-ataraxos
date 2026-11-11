# One-session search tuning

Run `scripts/session_search_tuning.sh` from any directory. Outputs and logs default
to `measurements/session-search-tuning`, created automatically. No existing output
run is needed; the three input checkpoints/data default to the TSUBAME paths in
the script. No installation or network access is used.

The script runs sequentially with `--resume`:

1. Nine settings, 500 pairs each: alpha {0.002, 0.02} × beta
   {0.02, 0.1, 0.3, 1.0}, actor leaves; plus 0.002/0.1/searcher.
2. Six settings, 500 pairs each: AR T={0.7, 1.5, 2.0}, AR/B1 mix P=0.5,
   B1, AR T=1. All include offline true-hand diagnostics. Set `BEST_ALPHA`,
   `BEST_BETA`, and `BEST_LEAF_VIEW` to substitute a step-1 setting (defaults
   0.002, 0.1, actor).
3. Select the highest observed win rate across steps 1–2 and confirm vs policy
   using 1000 fresh pairs. The existing P4 searched-opponent interface compares
   named agent types with one common search config; it cannot compare arbitrary
   search settings. `FINAL_GRID` overrides automatic selection. Selection from
   noisy estimates is exploratory; use the fresh confirmation interval.

The seeds are `SEED`, `SEED+1`, `SEED+2` (default SEED=73100).
Settings within each step share paired deals and one timed no-search baseline.
At 6–8 min / 400 games, estimate step 1 at 2.3–3.1 h, step 2 at 1.5–2.1 h
plus diagnostic inference, step 3 at 30–40 min. Allow 8–9 h total including
headroom. This is an extrapolation, not a measured diagnostic runtime guarantee.
Progress logs include wall time and ETA.

## Options

`sweep_search.py` accepts Cartesian lists for `--worlds`, `--depth`, `--alpha`,
`--beta`, `--leaf-view`, `--belief-temperature`, `--belief-mix`:

```sh
.venv/bin/python scripts/sweep_search.py --checkpoint POLICY --belief AR --fit FIT \
  --worlds 16 --depth 3 --alpha 0.002 0.02 --beta 0.02 0.1 0.3 1.0 \
  --leaf-view actor searcher --out-dir measurements/tuning --resume
```

Sparse settings use `--grid alpha=0.002,beta=0.3,worlds=16,depth=3` (multiple
space-separated entries). Keys accept underscores or hyphens; `belief=b1` is
also supported. Omitted keys use the common single CLI value; ambiguous omitted
list-valued keys are rejected. Legacy `--grid 16:3 32:5` still works.
All varying keys appear in filenames and JSON/Markdown summaries. Legacy
invocations keep their report fields, filenames, and output format.
Resume checks configuration and input file size/mtime and rejects mismatches.

`eval_search.py`, `eval_p4.py`, and official bot/advisor share
`--belief-temperature T` (positive; default 1) and `--belief-mix P` (0..1;
default 0). Variants require an AR checkpoint; mixing also requires `--fit`.
Count logits and every sorted AR conditional are tempered by 1/T. T=1/P=0
returns the original model, without additional RNG draws. Mixing chooses a
source once per world, retains it for rejection retries, and uses the existing
search seed stream. B1 and oracle arms of P4 remain their respective controls;
only its AR arm receives transforms.

## Privileged diagnostics

`--diagnostic-true-hand-nll` is available only to offline eval scripts and requires
an AR checkpoint and `--fit`. When searching with B1, use `--diagnostic-belief AR`
to supply the AR diagnostic model. JSON/Markdown section
`diagnostic_true_hand_nll` reports log likelihood, NLL, and hand-size accuracy
for untempered AR, AR at the selected T, B1, and hybrid on searched decisions, with means
and 95% normal intervals over decisions. These intervals are descriptive and do
not adjust for within-game correlation. NLL uses the existing belief metrics and
model likelihood implementations; B1 retains their 1e-12 card-probability floor.

The target is the true unused hand, obtained by `unused_assignment` from engine
labels. Existing belief models are trained on retained hands **including public
used cards**. Thus this diagnostic scores the unused multiset directly under
that model distribution; it is not the probability of the world sampler's
rejection-normalized/pushforward distribution. This distinction matters at
response roots. Diagnostic inference never changes actions or sampling RNG.
The offline subclass lives in `scripts/search_diagnostics.py`; official parsers
reject the flag, and their shared search constructor asserts against a
programmatically injected diagnostic option before loading models.

## CPU smoke

Supply tiny policy/AR checkpoints and fitting labels, then run:

```sh
DEVICE=cpu TF32=0 PAIRS=1 FINAL_PAIRS=1 NUM_ENVS=2 THREADS=1 \
WORLDS=1 DEPTH=0 CANDIDATES=1 CANDIDATE_SAMPLES=1 MAX_INFERENCE_BATCH=16 \
CHECKPOINT=/path/policy.pt BELIEF=/path/ar.pt FIT=/path/fit.pt \
OUT_DIR=measurements/session-smoke scripts/session_search_tuning.sh
```

Other overrides: `PYTHON`, `SEARCH_CACHE_MB`, `PROGRESS_INTERVAL`. Rerunning the
same command skips completed matching results without modifying their reports.


## Hybrid belief (no training)

Use `--belief AR_CHECKPOINT --fit FIT --belief-hybrid` in offline evaluation or
in the official bot/advisor. `--belief-hybrid 0` disables it (the default).
Each proposal samples the unused count from the AR count head at T=1, then draws
that many cards from B1's multinomial composition. Public used cards are added
to the assignment before the existing world-loading conversion subtracts them.
Capacity and legality rejection still apply, so accepted worlds may differ from
the raw proposal distribution. No privileged labels enter this sampler.

`--belief-count-temperature 1.5` tempers only the hybrid count logits; composition
is unchanged. Hybrid requires an AR checkpoint and `--fit` and cannot combine
with nondefault `--belief-temperature` or `--belief-mix`. Sweep grid keys are
`belief_hybrid=0|1` and `belief_count_temperature=T`. Absent/off options retain
legacy sampling and RNG behavior.

The offline `hybrid` diagnostic reports AR count log probability plus B1's
conditional multinomial composition log probability (including the multinomial
coefficient), and AR count argmax accuracy. It uses the selected count temperature
(default 1), including on control runs, and B1's existing card-probability floor.

`scripts/session_hybrid_belief.sh` runs one resumable step: hybrid T=1, hybrid
count T=1.5, AR T=1, and B1, at 1000 pairs each, 16 worlds, depth 3, alpha .002,
beta .1, actor leaves, and true-hand diagnostics. It uses the same `PYTHON`,
`OUT_DIR`, checkpoint, device, and budget environment overrides shown above.
For a tiny CPU run use the CPU smoke command with this script (omit FINAL_PAIRS).
