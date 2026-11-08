# Search worlds (P3 / D6)

`SearchEnv` is a thin public subclass backed by VecEnv's search mode. It shares
selection, automatic resolution, encoding and persistent workers with VecEnv;
there is no second rollout engine. The mode changes lifecycle (load instead of
reset, persistent terminal results, depth limits), while the existing training
and evaluation paths retain their observations, labels and decisions. No
changes to goodfield are needed. Leaf buffers are allocated on first `observe`,
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
cards plus the resulting unused cards must also fit hand_limit. Codes are
inverted by iterating `gf::catalog()` and applying `card_code`, not by assuming
CardId equals card code. Conversion does not read unused opponent items.

The match is produced by `Match::determinized`. Fresh opponent tokens and
`next_token` are then normalized using retained own/used tokens and the engine's
accounting bounds (Dream/redraw counters and a retained tick revival token),
followed by `restore` validation. This prevents a source already created with
`determinized` from carrying its discarded unused-token allocator into the new
world. Own slots and used objects are unchanged. Chance RNG is SplitMix64 of
`seed ^ 0xa4093822299f31d0`; Match RNG uses `seed` directly. No source RNG or row
index enters either stream; equal seeds intentionally reproduce equal streams.

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
