"""Explicit random streams for new, reproducible Monte Carlo calculations."""

import numpy as np

MC_RANDOM_SEED = 42
RandomState = int | np.random.Generator | None


def generator(random_state: RandomState = MC_RANDOM_SEED) -> np.random.Generator:
    """Accept an existing stream so successive samples do not restart it."""
    return np.random.default_rng(random_state)
