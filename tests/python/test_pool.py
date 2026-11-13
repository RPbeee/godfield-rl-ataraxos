"""The shared pool lottery preserves the original CPU RNG stream exactly."""
import numpy as np
import pytest

from gfrl.pool import assign_opponents


@pytest.mark.parametrize('fraction', [0., .2, 1.])
def test_pool_lottery_matches_original_cpu_draws(fraction):
    actual_rng = np.random.default_rng(42)
    reference_rng = np.random.default_rng(42)
    seats = np.full(31, -1, dtype=np.int8)
    policies = np.full(31, -1, dtype=np.int64)
    expected_seats, expected_policies = seats.copy(), policies.copy()
    pools = [[], [dict(iteration=7)], [dict(iteration=k) for k in (9, 11, 13)], []]
    for tick in range(40):
        pool = pools[tick % len(pools)]
        new_games = np.arange(tick % 4, 31, 4)
        seats[new_games] = policies[new_games] = -1
        expected_seats[new_games] = expected_policies[new_games] = -1
        used = set()
        if pool:
            selected = new_games[reference_rng.random(len(new_games)) < fraction]
            entries = reference_rng.integers(len(pool), size=len(selected))
            expected_seats[selected] = reference_rng.integers(2, size=len(selected))
            for j, entry in enumerate(pool):
                rows = selected[entries == j]
                if len(rows):
                    expected_policies[rows] = entry['iteration']
                    used.add(entry['iteration'])
        entries = assign_opponents(actual_rng, new_games, pool, fraction, seats, policies)
        assert set(entries) == used
        np.testing.assert_array_equal(seats, expected_seats)
        np.testing.assert_array_equal(policies, expected_policies)
        assert actual_rng.bit_generator.state == reference_rng.bit_generator.state
