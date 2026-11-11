# Search worlds (P3 / D6)

`SearchEnv` is a thin public subclass backed by VecEnv's search mode. It shares
selection, automatic resolution, encoding and persistent workers with VecEnv;
there is no second rollout engine. The mode changes lifecycle (load instead of
reset, persistent terminal results, depth limits), while the existing training
and evaluation paths retain their observations, labels and decisions. Token
normalization uses goodfield’s optional `determinized(..., normalize_tokens=true)`
so response replay anchors remain valid. Leaf buffers are allocated on first `observe`,
so training environments do not allocate a second observation batch.

## Python interface

```python
from gfrl import VecEnv, SearchEnv
# src is a live training/evaluation VecEnv, paused at policy decisions.
src = VecEnv(N, num_threads, track_projection=True)
reason = src.determinize_support(rows)  # int32 [R]
worlds = SearchEnv(R, num_threads, history_len=64, max_decisions=512)
status = worlds.load_worlds(src, rows, assignments, seeds)
worlds.step(choices)                    # same interface/buffers as Actor
leaf = worlds.observe(rows, seat)       # dict containing exactly OBS_FIELDS
```

`rows`/`choices` are contiguous int32 arrays, `assignments` is int32 `[R,297]`,
`seeds` is uint64 `[R]`. Duplicate source rows are allowed. Destination size is
fixed at construction. Source/destination history length and canonical setting
must match. Loading from the destination itself is rejected.
`VecEnv` defaults to `track_projection=False`: no opposite views are computed
or recorded. Search roots must opt in with `track_projection=True`; loading
from a source without tracking raises an error. `SearchEnv` does not track new
projections, but still copies the root searcher's retained history and its
projection when loading. `seat` is either
an absolute seat integer (0 or 1) or an int32 array parallel to `rows`.

Support is the integer value of `gf::DeterminizeSupport` (0 = ok), or -1 when
idle or inside a decision (`substep != 0`). World status is 0 = ok,
1 = invalid assignment, 2 = unsupported root. Invalid rows have `stage=-1`, an
empty mask and persistent `episode_done=true`; their choices are ignored.
Malformed shapes/indices are batch errors, checked before modifying rows.

`src.unused_assignment(row, assignment)` exposes the same C++ conversion used
by loading. Counts must be nonnegative, padding code 0 absent, total at most
hand_limit, and every supplied model in the configured gift catalog. For each
public opponent used model, subtract one copy, clamping at zero. Retained used
cards plus the resulting unused cards must also fit hand_limit. At response roots,
the launch inventory must fit too: count anchor used items and all public
launch/response selected items once, including weapons already consumed in the
current state. Newly used miracles are subtracted from the current label and
retained by replay; they must not be appended a second time. Codes are
inverted by iterating `gf::catalog()` and applying `card_code`, not by assuming
CardId equals card code. Conversion does not read unused opponent items.

### Oracle diagnostic (evaluation only)

`load_search_belief('oracle')`, `eval_search.py --belief oracle`, and
`search_noise.py --belief oracle` deliberately cheat to diagnose belief error.
They copy the root actor's privileged `label_opp_hand` (297 counts, including
public used cards) to every sampled world. Labels and `label_opp_hand_count`
are already populated by `encode_observation` for both training and evaluation;
no C++ runtime or policy input change is required. The usual `unused_assignment`
validation and `load_worlds` conversion subtract public used cards to reconstruct
the true unused hand. Invalid assignments are rejected once per requested world
and counted in `invalid_oracle_assignments` as well as `invalid_samples` and
`exhausted_worlds`; they are never replaced by another belief. The expected
invalid oracle count is zero. Chance seeds still come independently from the
search RNG for every world; only the hand is privileged. Opponent history,
candidate generation, rollouts, leaf evaluation and the search formula retain
their usual behavior. JSON reports include `belief: "oracle"` and
`diagnostic_only: true`; these results are not fair-play evaluation results.

The C++ search test checks labels, converted unused counts and loaded physical
unused hands against snapshots across thousands of training/evaluation roots.
These are test-only C++ changes; production C++ is unchanged.

CPU paired smoke (S policy, same seed and search budgets for both beliefs):

```sh
.venv/bin/python scripts/eval_search.py \
  --checkpoint runs/m4-s-ppo-v1/checkpoint.pt --belief oracle \
  --pairs 32 --num-envs 8 --threads 1 --seed 20261109 \
  --worlds 64 --depth 3 --beta 0.1 --device cpu \
  --out runs/oracle-diagnostic-smoke/oracle.json
.venv/bin/python scripts/eval_search.py \
  --checkpoint runs/m4-s-ppo-v1/checkpoint.pt \
  --belief b1 --fit runs/belief-smoke/fit.pt \
  --pairs 32 --num-envs 8 --threads 1 --seed 20261109 \
  --worlds 64 --depth 3 --beta 0.1 --device cpu \
  --out runs/oracle-diagnostic-smoke/b1.json
```

The match is produced by `Match::determinized(..., normalize_tokens=true)`.
Normalization happens after discarding hidden unused items and before appending
sampled items or replaying the response journal. The first fresh token is the
maximum of 1, every retained token + 1, Dream/redraw accounting bounds, and any
retained tick revival token + 1. In a response this includes the searcher's
launch inventory and public selected opponent items; their IDs still occur in
commands and must survive. Sorted sampled unused items receive consecutive IDs.
Replaying the launch and reflection journal creates both current and anchor
inventories and allocator copies consistently, then `restore` validates them.
The optional engine argument defaults false, preserving the engine's existing
API behavior. Doing this inside `determinized` avoids post-hoc editing of a
replay proof and also avoids allocation from a discarded source `next_token`.
Own slots and used objects are unchanged. Chance RNG is SplitMix64 of
`seed ^ 0xa4093822299f31d0`; Match RNG and the response anchor/redirect RNG use
`seed` directly for deterministic launches. For public chance response launches,
goodfield conditions a fresh launch stream on the visible hit/sign and consumes
its launch draws; anchor/current/redirect RNG can therefore differ from `seed`.
No source RNG or row index enters either stream.

### Opt-in response roots

`SearchConfig.search_response=False` (CLI default) keeps main-only search.
`--search-response` enables supported ordinary responses to the opponent's
`play`; goodfield reasons 11–13 are `response_special`, `response_random`, and
`response_history`. Eligibility reads only the public CLS phase and support.
Disabled responses keep the old `skip_not_main` precedence and count, including
responses which the extended engine now rejects under a more specific reason.
The additional `skip_response_disabled` counts those same disabled responses;
it is a subset of `skip_not_main`, not another disjoint skip category. With the
flag enabled, `skip_not_main` decreases because responses are searched or get
the engine's specific rejection reason instead.

Candidates still come from the current decision's policy, using complete paths
of externally selected choices. A defense can span multiple substeps; completing
all of them counts as **one** decision, exactly as a main action does. Worlds
run to `depth + 1` completed external decisions including the root. Automatic
work is not counted. Rollouts, actor-relative leaves and the search distribution
are shared by both kinds. Belief sees the current response observation; its
training label includes all current physical opposing items, including used
miracles, and excludes consumed weapons. `unused_assignment` converts this to
the current unused hand, while replay restores public selected cards only in
the anchor. Oracle follows precisely the same conversion and validation.

`searched_main` / `searched_response` count root searches (also for direct
`search` diagnostics). Under normal agent evaluation their sum is `searched`.
`seconds_per_search_by_root` reports wall seconds per root, including sampling,
candidates, rollouts and leaves. Mixed batches run main then response groups,
so each timing is measured independently; grouping changes RNG order only when
response search is enabled. Within a kind, root batching keeps its usual rules.
These are amortized batched costs, not serial single-root latency.

Tests cover all six OBS_FIELDS at varied main and response roots, oracle labels
and physical unused hands, anchor capacity, full snapshot equality across hidden
source hands/allocators, and equal-size oracle vs sampled hands differing only
in unused models. Engine tests validate normalized snapshots by `restore` and
check seeded current/anchor RNG. The frozen `search_exact_before.json` remains
unchanged and checks bit-exact main-only candidates, q, probabilities, choices
and game results.

`decisions` counts only completed decisions involving an external policy stage,
including the root decision; automatic/chance decisions do not count.
`decision_done` is a one-step pulse for those decisions. At max_decisions,
automatic work finishes up to the next policy boundary, then the row stops with
`episode_done=true`, `truncated=true`, `result=0`. An earlier engine terminal
keeps its actual truncation flag and `result` is win/loss/draw (+1/-1/0) relative
to `searcher_seat`. `outcome` retains the usual absolute-seat pair. Stopped rows
never reset; repeated steps leave results, depth and terminal flags unchanged.

Actor can bind directly to SearchEnv. After loading/reloading, all live rows
start at substep 0, so Actor's observation cache is refreshed normally.

Policy and result arrays are persistent read-only NumPy views owned by the env.
`load_worlds` returns its status buffer without copying. `observe` uses separate
persistent storage and returns views of the first R rows in request order
(R <= num_envs), with no labels or action prefixes. The next observe overwrites
these views; step/load never overwrite them. It works for either seat, at
intermediate stages and at stopped leaves, without changing the policy buffers,
selection or stats. Query results from support/conversion own their C++ storage.

## Reconstructing opponent history

Only the searcher's retained history is copied. The true opponent history is
never used. At each normal history append we also derive an optional opposite
Token for each source-visible event, using that source's before/after views
and **event-time visibility metadata** (owner_only / hidden_viewers). No hidden
opponent hand, missing status value or opponent-only event is inserted.
These optional tokens are metadata on the same source ring entries, so eviction
also evicts their projections. Loading builds the opponent ring from those
entries only; private events that were dropped cannot resurrect older entries.

The field rules are:

- Swap relative seats 1 and 2 (keep 0 absent) in Cat Actor, Target, Origin and
  SourceSeat, for every history token type.
- Decision's two 18-field Num Delta blocks exchange seats. Recompute fogged
  deltas and status visibility for the destination viewer from the source
  panes, then encode deltas only when both source-visible endpoints remain
  visible. A missing pane stays missing; when the other player's Fog is itself
  unknown, conservatively hide the affected pane. Dead/game-over visibility
  follows the public view rules.
- Decision faces remain absent when hidden from the source. Also clear cards
  and CountOverflow, and set FacesHidden when the event-time destination mask
  hides them. Do not infer faces from real inventories.
- Reveal passes only if visible to both viewers: owner-only reveals are omitted.
  GuardianChange passes the intersection of both event-time visibility masks.
  Opponent-only events that the source never saw remain absent.
- All other category fields, cards/aux card codes, numeric event fields,
  allocation, outcome, and the absolute event decision number are preserved.
  Age is recomputed normally from the leaf's public decisions and history_len.
  CLS/Player/Hand/Pending/Offer/Used tokens are freshly encoded from that world's
  selected viewer; they are not history tokens and are never seat-swapped.

With fully public history and sufficient ring capacity this equals the true
opponent history. Missing private facts and differences caused by ring eviction
are the intended D6 approximation. Tests cover public two-seat discards, actual Dream-cure private reveals, all
companion types, private reveals/guardian changes, Fog, retained-ring eviction,
full-snapshot leak equality (including future transitions), and root observations
at thousands of self-play boundaries.

## Reproduction and measurements

`build/gfrl_search_tests` includes deterministic Random-vs-Random self-play:
32 training rows, 4 workers, base seed 20261108, policy seed 101, 2,000 stage
steps. It counts each fresh policy decision, inspecting the opponent's used
cards (the part included in the belief label). In 26,924 positions, 2,160 had
used cards: **8.02258%**, mean **0.0836428** cards per position, or **1.04259**
conditional on having used cards. It also loads/compares 2,358 supported worlds
from this trajectory. These frequencies describe this random policy, not πθ.

```sh
.venv/bin/python scripts/smoke_search.py
```

The smoke loads `runs/m4-s-ppo-v1/checkpoint.pt`, fits B1 using
`runs/belief-smoke/fit.pt`, gathers supported evaluation roots after 20 policy
stage steps, samples 16 assignments per root and uses both seats' πθ on CPU to
10 policy decisions. Invalid B1 samples are explicitly rejected and resampled
using the C++ conversion (B1 smoothing also assigns mass to non-gift cards).
The printed JSON includes per-world leaf values, actual terminal results,
load throughput and end-to-end policy rollout throughput. Sampling and model
loading are excluded from load timing; rollout includes Actor inference and
step, while leaf evaluation is timed separately.

One CPU thread, seed 20261108, 4 roots / 64 worlds: about **122,945 worlds/s**
loading (20 repeats), **305 completed world rollouts/s**, **2,771 policy
decisions/s**. Rollout took 0.210 s for 581 decisions: 9 true terminals and 55
depth cutoffs; all 64 leaf values were finite (mean 0.030914, range
[-0.712145, 0.883088]). Leaf observation/encoder/value evaluation took 0.0266 s.
Timings vary by host load; the script prints fresh measurements.

### Optional projection tracking benchmark (P3 2/4)

`tests/python/bench_projection.py` compares separately loaded native modules
in isolated subprocesses. It rotates baseline/false/true ordering across five
repeats and reports median decisions/s. Uniform legal random action sampling
and `step` are timed together; creation and 100 warmup stages are excluded.
Each measured run uses 256 environments, 2,000 stage steps, environment seed 42,
policy seed 123, default history_len=64. All runs have identical stats:
253,017 decisions (including 37,503 automatic), 512,000 policy stages, and
9,039 episodes. Trajectory-counter equality is asserted by the script.

Release / AppleClang 21, local CPU measurements (2026-11-08):

| Workers | Before projection (HEAD `7b53337`) | track_projection=false | track_projection=true |
| --- | ---: | ---: | ---: |
| 1 | 119,040 | 116,278 | 106,273 |
| 4 | 362,159 | 353,054 | 326,390 |

Disabling projections improves throughput by 9.4% / 8.2% over tracking.
Relative to the pre-projection baseline it is 2.3% / 2.5% slower in these runs;
these are measured rates, not a claim of exact performance equality. Timing
varies with host load. No projection views are allocated or projection events
recorded when tracking is disabled.

Reproduce without changing/stashing the working tree (the goodfield source is
shared read-only and its build outputs remain under the temporary directory):

```sh
baseline=$(mktemp -d /tmp/gfrl-projection-head.XXXXXX)
git archive HEAD | tar -x -C "$baseline"
rmdir "$baseline/third_party/goodfield"
ln -s "$PWD/third_party/goodfield" "$baseline/third_party/goodfield"
.venv/bin/cmake -S "$baseline" -B "$baseline/build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build "$baseline/build" --target _core -j 4
.venv/bin/cmake --build build -j 4
.venv/bin/python tests/python/bench_projection.py \
  --baseline-core "$baseline"/python/gfrl/_core*.so
.venv/bin/python tests/python/bench_projection.py --threads 4 \
  --baseline-core "$baseline"/python/gfrl/_core*.so
```

Here HEAD is the pre-projection commit above; if HEAD advances, archive
`7b53337` explicitly instead. The search smoke also retains the recorded
64-world trajectory: 581 decisions, 9 terminals, 55 depth cutoffs, and leaf
value mean 0.0309136733 with range [-0.7121445537, 0.8830882907].

### Leaf actor and bounded inference

`next_actor_seats(rows)` returns an owning int32 array of absolute seats at the
next decision boundary, including depth-stopped rows; true terminals and invalid
worlds return -1. This read-only query is needed because `stop()` clears the
published `actor_seat` and selection. It reads the retained engine actor without
publishing observations, advancing work, or accessing hidden opponent state.

Search defaults to `SearchConfig(leaf_view='actor')`: nonterminal leaves use
`observe(rows, next_actor_seats(rows))` and negate opponent values to recover the
searcher's win-minus-loss value. `leaf_view='searcher'` preserves the previous
leaf evaluation for comparisons. Terminal results remain searcher-relative.
The counters `depth_leaves_searcher` and `depth_leaves_opponent` partition
`depth_leaves` by the boundary actor, independent of the selected leaf view.

At either boundary actor, `observe` produces exactly the OBS_FIELDS buffers
that a live SearchEnv publishes for Actor at substep 0. Forced singleton choices
before the first external stage do not modify the match observation. Tests
compare all fields with a longer-lived world at the same boundary, covering
both own turns and opponent responses immediately after a root attack at D0.
The opponent's retained pre-root history still uses the existing projected
history approximation described above: missing private events or eviction can
make it differ from the real opponent's training trajectory. This change does
not reconstruct those unavailable facts.

`max_inference_batch=0` (CLI `--max-inference-batch`) retains unsplit inference.
A positive limit chunks search belief encoding/decoding, candidate and rollout
policy encoding/heads, and leaf value inference. The search-only Actor keeps
its decision token cache in host memory, transferring at most that many rows
for each head call; ordinary Actor is unchanged. Probabilities are collected
before the original whole-batch sampling calls to preserve RNG ordering.
CPU tests cover baseline, autoregressive and independent beliefs and compare
candidates, values and selected paths with unsplit inference. Floating point
roundoff can still depend on batch size/device. Start with 128 rows on a 10 GB
GPU and reduce to 64 if other jobs leave less available memory; GPU memory and
throughput require measurement on the target machine.

### Response integration validation (2026-11-09)

Release parent CTest: 17/17; standalone goodfield CTest: 165/165.
Python: 170 passed, 1 skipped (existing NumPy read-only warning).
An isolated pre-response engine build also reproduced the frozen exact trace
and all original integer counters; `search_counts_before.json` retains those
counters alongside the unchanged action/value fixture.

CPU paired smoke used `runs/m4-s-ppo-v1/checkpoint.pt` (S, d128, four layers)
and `runs/belief-b2-smoke/autoregressive/checkpoint.pt`, policy opponent,
seed 20261111, 3 pairs, four environments, one thread, M64/D3, 16 candidates,
64 candidate samples, alpha .002, beta .1, actor leaves, max inference batch
8192, cache auto/8192 MiB, skip-forced-inference and skip-single-candidate.
Only `--search-response` differs between the two runs.

| Response roots | Games | Main searches | Response searches | Main s/root | Response s/root | Mismatches | Invalid loads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| disabled | 6 | 84 | 0 | 1.8173 | — | 0 | 0 |
| enabled | 6 | 64 | 26 | 1.6181 | 1.5304 | 0 | 0 |

Total elapsed: 153.61 s disabled, 144.10 s enabled. These runs executed
concurrently on the same CPU host, so timings include contention and are not
a speed comparison. AR invalid samples were rejected/resampled (16,195 / 13,930);
no invalid world reached rollout. Reports are in
`runs/response-search-smoke/{disabled,enabled}.json` with exact CLI arguments,
phase timings and counters. The small game sample is a completion check only.

### Public chance response roots (2026-11-11)

`--search-response` now also admits ordinary percentage hits, two-player Fog
single-target launches, and 202 HP-gamble launches. No new option or reason code
is needed: `_SUPPORT` already maps the engine's reasons 11–13. The support and
world-loading paths delegate to the extended engine. Pending percentage attacks
publicly imply a hit; the Fog target is public and unique; 202's signed amount
is public in the pending observation. The engine conditions a fresh RNG on
these facts and replays the journal with full observation/decision/inventory/
legal-kind equality checks. Danger (including Wide), bounce and resolved-body
histories remain outside this subset. See goodfield's `docs/determinization.md`
for the information audit and fixed-policy coverage counts.

The unchanged main-only exact fixtures pass. For an additional direct engine
comparison, the pre-change `official_match.cpp` was compiled into
`build/baseline/libgodfield.dylib` using this worktree's other engine objects.
Running `measurements/compare_roots.py` with that library selected through
`DYLD_LIBRARY_PATH`, and again with the current engine, gives exact JSON equality
(choices, candidate values, probabilities and integer counters) for all six
main-only fixtures, and for enabled main/ordinary-response fixtures. Only the
four new enabled fixtures (percentage, Fog, 202, Fog+202) change support 12→0
and begin searching. Artifacts: `measurements/roots-{before,after}.json` and
`measurements/comparison.json`.

Release validation: parent CTest 17/17, goodfield CTest 166/166, Python 262
passed / 1 skipped. This worktree uses a local `.venv` symlink to the supplied
read/execute-only venv so the existing TSUBAME-wrapper test resolves its expected
interpreter path. No dependencies were installed.

CPU paired evaluation used the supplied M policy `m-2166.pt` and AR belief
`belief-ar-m.pt`, both read only, against the same policy: 20 pairs (40 games),
seed 0, 8 environments, 4 threads, 8 worlds, depth 3, 8 candidates / 32 samples,
alpha .002, beta .1, actor leaves, response search enabled. Exact arguments and
raw reports are in `measurements/chance-{before,after}.json`.

| Counter | Before | After |
| --- | ---: | ---: |
| Searcher decisions | 1,025 | 1,084 |
| Searched | 709 (69.1707%) | 917 (84.5941%) |
| Searched main | 535 | 648 |
| Searched response | 174 | 269 |
| skip_not_main (purchase etc.) | 41 | 36 |
| skip_not_actor | 29 | 20 |
| skip_hidden_status | 31 | 21 |
| skip_hidden_own_model | 185 | 76 |
| skip_response_random | 28 | 9 |
| skip_response_special | 0 | 4 |
| skip_response_history | 2 | 1 |
| Invalid loads / real mismatches | 0 / 0 | 0 / 0 |

The enabled new searches change subsequent game trajectories, so other main
and skip counts also change. This aggregate increase is a smoke measurement,
not a causal estimate of coverage or strength. The identical-trajectory 1,000
game engine comparison isolates eligibility: response support 5,204/7,339 →
5,843/7,339 (70.9088% → 79.6158%), response_random 784 → 71, with all main
counts unchanged. The direct root comparison above separately proves that
previously supported root outputs and disabled-response outputs stay exact.
No strength conclusion is drawn from the 20 pairs.


### Own Dream worlds (2026-11-11)

Search now opts into `determinize_support(viewer, true)` and the new overload
`determinized(viewer, opponent_models, own_disguises, seed, normalize_tokens)`.
The old support call and old determinized overload retain their behavior,
including rejecting nonidentity own disguises. No training, collection, transition,
observation or snapshot format changes were made.

`own_disguises` supplies one real model for **every** disguised slot in
`view(viewer).board.hand` order (the public hand is sorted; the physical inventory
is not). Slots are located by token in the main snapshot or response launch
anchor. Faces, fake-model presence, used flags, tokens and physical order remain
fixed. Replay regenerates both the current response and its proof anchor.
Eligibility and sampling never distinguish identity from nonidentity disguises;
even a restore-valid identity disguise must be resampled in the new path.
Unmasked viewer cards and publicly played/revealed cards remain known.

The parent samples each own slot independently, using only its public face and
the configured gift catalog. For face f, the prior is
`P(actual=a | disguised, face=f) ∝ giftRate(a) * match_dream_compatible(a,f)`.
Goodfield's natural Dream draw chooses a different face uniformly from the
compatible class after its mask coin. All members of a class have the same
number of alternative faces, so that likelihood factor cancels. In particular,
the displayed face itself has zero probability under a natural masked gift.
The engine's broader restore-only compatibility rules (identity and the 19→27
fixture alias) remain valid inputs to its assignment API, but are not extra
natural-gift prior mass. An empty natural prior rejects the sample.

**Design approximation:** the belief model predicts only the opponent hand.
Own disguised reals therefore use this gift prior, without conditioning on
private source truth or estimating history-dependent card survival. This is
an approximation to the full posterior, not an oracle. Opponent sampling is
unchanged. A separate seed-derived stream (`seed ^ 0x36d8a971c2ef504b`, mixed)
with exact bounded draws selects own models; the existing world/chance seeds
and the Python search RNG schedule stay unchanged for old roots.

Each own-assigned result passes full restore validation and exact typed viewer
view, last-decision and legal-kind equality guards. Ordered actor inventories
are compared after replacing reals with faces. `SelectionTree` otherwise reads
faces, disguise/used bits, resources and the public response context. Its only
true-model tests are base call-phenomenon eligibility and mortar presence for
exact 94+232. Those own-card predicates are checked invariant; root/one-card
prefix nodes and every reachable 94+232 node are compared in both canonical
modes. This proves the entire mask tree agrees without enumerating factorially
many discard/defense permutations. Any mismatch rejects the sample, including
a sampled enemy mortar changing the known 94+232 exception in a Dream root.
The existing exception behavior on roots without own disguises is unchanged.

Tests exhaust all 992 natural Dream compatibility pairs, compare full prefix
trees for small hands in both modes, and reject missing/extra/incompatible
assignments. Parent C++ tests compare complete snapshots (including anchors),
all observation fields, masks and future transitions over 2,048 seeds per main/
response fixture (future transitions on the first 256), with multiple disguises and changed source actuals/RNGs.
Python tests additionally compare both seats' leaf observations and complete
search outputs under changed true reals, including identity source disguises,
with response search enabled and disabled. No privileged label is used for
own sampling.

Fixed pre-change fixtures in `tests/python/fixtures/search_supported_before.json`
cover main and ordinary/percentage/Fog/202/Fog+202 response roots with both
flag settings. All 12 output records remain exactly equal: candidates, q,
probabilities, choices and integer counters. The earlier exact main-only
fixture, including completed game outcomes, also remains unchanged.

#### Fixed-policy coverage before and after

Both runs use default goodfield config, 1,000 games, game seeds 0..999,
policy RNG `game+101`, chance RNG `game+203`, and `sample_action` / the same
existing automatic chance helpers. Sampling worlds never changes the played
match or those streams. These counts include every active-player engine
boundary, including forced choices; they are not the external-policy-only
counts from M-policy evaluation. First-reason precedence is exclusive. Each
percentage below uses its own main/response total as denominator (both seats).

```sh
build-goodfield/godfield_official_determinized_tests --measure 1000
build-goodfield/godfield_official_determinized_tests --measure-own 1000
```

The first command retains the legacy API; the second uses the new own-assignment
API on exactly the same fixed-policy games. Raw reports are
`measurements/dream-engine-{before,after}.txt`.

| First reason | Main before (22,577) | Main after (22,577) | Response before (7,339) | Response after (7,339) |
| --- | ---: | ---: | ---: | ---: |
| ok | 21,153 (93.6927%) | 21,858 (96.8153%) | 5,843 (79.6158%) | 6,032 (82.1910%) |
| hidden_own_model | 705 (3.1226%) | 0 (0.0000%) | 189 (2.5753%) | 0 (0.0000%) |
| hidden_status | 719 (3.1847%) | 719 (3.1847%) | 183 (2.4935%) | 183 (2.4935%) |
| not_actor | 0 (0.0000%) | 0 (0.0000%) | 372 (5.0688%) | 372 (5.0688%) |
| response_special | 0 (0.0000%) | 0 (0.0000%) | 643 (8.7614%) | 643 (8.7614%) |
| response_random | 0 (0.0000%) | 0 (0.0000%) | 71 (0.9674%) | 71 (0.9674%) |
| response_history | 0 (0.0000%) | 0 (0.0000%) | 38 (0.5178%) | 38 (0.5178%) |
| Every other reason, each | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |

Other reasons are invalid_viewer, unsupported_players, not_main, pending_work,
reserved_gifts, replay_proof and hidden_confusion. There are also 361 purchase
boundaries (not_main, not newly supported) and no grant boundaries under default
local gifts. With `--search-response` absent, all 7,339 response boundaries
remain outside frontend search (`skip_not_main`, including response_disabled).
With it enabled, engine response eligibility above applies. Actual policy
searches still require a nonautomatic decision and sufficient valid samples.

| Rejection detail (first reason) | Main before → after | Response before → after |
| --- | ---: | ---: |
| Dream: own nonidentity disguised card(s) | 705 → 0 | 189 → 0 |
| Fog on viewer: enemy status hidden | 719 → 719 | 183 → 183 |
| Response to viewer's own launch | 0 | 372 → 372 |
| Special: buy launch | 0 | 385 → 385 |
| Special: sell launch | 0 | 257 → 257 |
| Special: tick already processed | 0 | 1 → 1 |
| Random: Danger | 0 | 24 → 24 |
| Random: bounce | 0 | 47 → 47 |
| History: previous resolved body | 0 | 38 → 38 |

Thus eligible main roots increase by 705 and responses by 189, totaling 894.
No old rejection category is redistributed except hidden_own_model→ok; all phase
totals match. Main sample attempts reject 0/10,000 after the change. Response
samples reject 312/6,032 (5.1724%), entirely launch physical capacity; there are
zero view/decision/mask mismatch rejections. Before: main 0/10,000, response
298/5,843 (5.1001%), also entirely capacity. Assignment RNG=778 is separate from
play; adding newly eligible roots changes its subsequent samples, so these
rejection totals are not a paired comparison of assignments. Timings were run
alongside tests and are not used for performance conclusions.

#### Remaining exclusions and implementation estimates

These are engineering estimates including fixtures/leak tests, not measured
runtime budgets. They remain excluded in this change:

- **Viewer Fog (719 main, 183 responses):** the opponent HP/MP/CP, displayed
  diseases and curses, guardian and confusion status are hidden. A safe world
  needs a joint posterior over them, not copied source values or independently
  uniform fields. Confusion counters, disease/death/revival bookkeeping and
  response initial-state/derived pending fields must remain restore-consistent;
  retained public history constrains the samples too. Add a status-assignment
  contract, choose/train an estimator and rebuild response anchors/history.
  Estimate 5–10 engineering days for an explicit prior approximation with
  invariance/replay tests; a calibrated learned posterior is additional work.
- **Buy (385) / sell (257):** these are before trade resolution, so the launch
  itself need not resample a commodity already observed at these roots. Buy's
  later random commodity must come from the sampled seller inventory and new
  RNG; eventual escrow/price/return/overflow and purchase proofs cannot retain
  source objects. Sell must preserve both public launch tokens (including the
  commodity) while deriving transfer/gifts/capacity from the rebuilt inventory.
  Existing `replay_launch` can dispatch both, but eligibility, public fields,
  reflection contexts and current-vs-anchor used/consumed/escrow accounting need
  a trade-specific audit and leak fixtures. Estimate 2–4 days for both (buy
  alone 1–2, sell 1–2); not treated as a safe one-line whitelist expansion.
- **Tick already processed (1 special), resolved-body histories (38):** replay
  can involve disease damage, revival consumption, death/counter queues, guardian
  opportunities, gifts/redraws and potentially private results. Replace those
  hidden outcomes and reconstruct all dependent anchors rather than retaining
  them. Estimate 3–5 days for a narrowly defined tick/body subset, larger for
  the full general journal.
- **Danger/bounce (71):** Danger depends on hidden mortar presence; bounce
  requires fresh redirect target draws conditioned on public routing. Neither
  was included. Response to one's own launch (372) is a separate owner/actor
  contract expansion, also unchanged.


Final validation: Release cmake builds succeeded for parent and standalone
goodfield; parent CTest **17/17**, standalone goodfield CTest **166/166**,
`pytest -q` **279 passed / 1 skipped** (including MPS equivalence). After adding
extra sorted-slot, mask-rejection and prior-frequency fixtures, the rebuilt
parent CTest 17/17 and goodfield official_determinized test passed again.
The pre-change 12 root output records compare exactly equal after the final
build. The legacy `--measure 1000` also reproduces every pre-change count and
detail, including sample capacity rejection totals, under the changed engine.
Logs and raw comparison outputs are under `measurements/dream-*`.
