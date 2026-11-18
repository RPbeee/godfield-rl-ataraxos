# Measuring official UI restrictions

Run the same evaluation with and without `--a-restrict official`, keeping the
seed, batching, checkpoints, and device fixed:

```sh
.venv/bin/python scripts/eval_policies.py \
  --a runs/play/m-2166.pt --b runs/play/m-2166.pt \
  --a-restrict official --pairs 8 --num-envs 8 --threads 4 \
  --compile off --out /tmp/official-restricted.json
```

`--a-restrict` also accepts a comma list of `kind`, `dream`, and `self_target`,
for example `--a-restrict kind,self_target`. `dream` includes both disguised
items and ordinary items sharing a disguised face. Only active categories are
counted and applied; relaxation restores active Dream exclusions first, then
any remaining active restrictions. JSON records the normalized category list
in `a_restrict_categories` at the top level and in each opponent result.

Only A is restricted, including when A occupies the second seat of a paired
deal. The native sequential masks remove unobserved kinds, disguised items,
ordinary items sharing a disguised face, and enemy-default self targets for
kind indices 0, 5, and 6 (`play`, `sell`, `buy`). Restrictions use the viewer's
public hand and displayed faces. Each policy head samples from its masked,
renormalized distribution; no rejected-action sampling is performed.

The masks retain only prefixes with a surviving completion and preserve the
original canonical selection tree. If the combined restrictions leave no
completion, Dream and ambiguous items are restored together. Kind and self-target
restrictions remain; if they still leave no completion, the simulator restores
the original mask. This final fallback keeps simulations progressing. The
existing 94+232 prohibition remains part of the original mask.

Each opponent's JSON result contains `a_restrictions`:

| Field | Meaning |
| --- | --- |
| `decisions` | A's ordinary decisions, including forced selections; excludes automatic confusion/grant events |
| `restricted_decisions` | Decisions with at least one legal completion blocked before relaxation |
| `kind`, `dream`, `ambiguous`, `self_target` | Decisions where that category alone removes a legal completion from the original tree |
| `dream_last_resort` | Decisions that restore excluded Dream/ambiguous items after exhausting the combined mask |
| `unrestricted_last_resort` | Decisions that restore the original mask after hard restrictions still exhaust it |

Categories can overlap and are counted before last-resort relaxation. An item
being present in the hand does not count unless it participates in a legal
completion. Counts aggregate across all games against that opponent and reset
for the next opponent.

`--a-restrict none` is the default and preserves the original execution path,
RNG use, and output schema. The option changes neither paired-deal generation
nor evaluation seeds.

The diagnostic counts use exact short witnesses, without enumerating complete
commands or exchange allocations. They inspect at most the root, one-card,
and two-card prefixes for each kind. Under the current append rules, a face
can be selected directly, after a base, or after one enabler (a miracle before
FREE, FREE after a miracle, or a defense filter before a reaction). Removing
other material reduces cost and cannot introduce a canonical ordering barrier.
Every equivalent slot is in the same Dream/ambiguous category, so testing the
first canonical copy suffices. Explicit self targets need at most a base plus
FREE or a sale commodity; additional attack material only removes targets.

Completion pruning uses the engine's `SelectionTree::reachable` invariant:
only an immediate FREE or a sale commodity can repair a nonempty incomplete
prefix. A root therefore needs at most two further cards. This preserves item
order and works for arbitrary advisor exclusions, including noncanonical slots.
A multiset memo would be unsafe because FREE, reactions and barriers depend on
order. Both searches have polynomial bounds in hand size, with no timeout or
approximate fallback.

`gfrl_official_restrictions` retains the recursive `dac9be8` implementation
only in test sources. It compares plans and sequential masks on simulated-game
positions and randomized Dream scenes, with both canonical settings and arbitrary
advisor exclusions. Its stress phase times thousands of random/greedy self-play
decisions, full hands, Dream, ordered defense, and allocation stages; every plan,
Selection construction and substep must take less than 100 ms. The test prints
observed maxima. CTest also imposes an overall timeout so a regression cannot
hang the test runner indefinitely.
