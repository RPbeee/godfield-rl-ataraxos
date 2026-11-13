"""The CPU and CUDA collectors share the game-start opponent lottery."""


def assign_opponents(rng, new_games, pool, fraction, seats, policies):
    """Mutate only new games; return the entries actually selected.

    Keep the CPU Collector's draw order, including zero-length draws and no
    random draws at all for an empty pool. Pool identities are iterations,
    never list positions (the saved pool may be evicted during a game).
    """
    selected_entries = {}
    if len(new_games) and pool:
        selected = new_games[rng.random(len(new_games)) < fraction]
        entries = rng.integers(len(pool), size=len(selected))
        seats[selected] = rng.integers(2, size=len(selected))
        for j, entry in enumerate(pool):
            rows = selected[entries == j]
            if len(rows):
                key = entry['iteration']
                policies[rows] = key
                selected_entries[key] = entry
    return selected_entries
