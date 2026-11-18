# Measuring official UI restrictions

Run the same evaluation with and without `--a-restrict official`, keeping the
seed, batching, checkpoints, and device fixed:

```sh
.venv/bin/python scripts/eval_policies.py \
  --a runs/play/m-2166.pt --b runs/play/m-2166.pt \
  --a-restrict official --pairs 8 --num-envs 8 --threads 4 \
  --compile off --out /tmp/official-restricted.json
```

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
