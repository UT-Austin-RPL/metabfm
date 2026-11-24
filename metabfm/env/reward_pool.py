"""Deterministic ColoredGrid cell permutations and reward grids with train/test
splitting. Pools are cached as ``.npy`` under ``$METABFM_DIR/reward_pool_cache``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np

from metabfm import config as C

_CACHE_DIR = os.path.join(C.PROJECT_DIR, "reward_pool_cache")


def _cache_path(kind: str, grid_size: int, count: int, seed: int) -> str:
    return os.path.join(_CACHE_DIR, f"{kind}_g{grid_size}_n{count}_s{seed}.npy")


def generate_cell_perms(grid_size: int, count: int, seed: int = 0) -> List[np.ndarray]:
    """Generate *count* unique cell permutations deterministically from *seed*."""
    cache = _cache_path("cells", grid_size, count, seed)
    if os.path.exists(cache):
        arr = np.load(cache)
        return [arr[i] for i in range(len(arr))]
    rng = np.random.default_rng(seed)
    perms: List[np.ndarray] = []
    seen: set[tuple] = set()
    while len(perms) < count:
        perm = rng.permutation(grid_size * grid_size)
        key = tuple(perm.tolist())
        if key in seen:
            continue
        seen.add(key)
        perms.append(perm.astype(np.int64))
    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.save(cache, np.stack(perms))
    return perms


def _sample_random_reward_grid(rng: np.random.Generator, grid_size: int) -> np.ndarray:
    grid = rng.normal(0.0, 3.0, size=(grid_size, grid_size)).astype(np.float32)
    return np.clip(grid, -10.0, 10.0)


def generate_reward_grids(
    grid_size: int, count: int, seed: int = 0
) -> List[np.ndarray]:
    """Generate *count* unique reward grids deterministically from *seed*.

    Each cell is drawn from N(0, 3) and clipped to [-10, 10]. Sampling is
    sequential, so the first *N* grids are the same for any ``count >= N``.
    """
    cache = _cache_path("rewards", grid_size, count, seed)
    if os.path.exists(cache):
        arr = np.load(cache)
        return [arr[i] for i in range(len(arr))]
    rng = np.random.default_rng(seed)
    grids: List[np.ndarray] = []
    seen: set[bytes] = set()
    while len(grids) < count:
        grid = _sample_random_reward_grid(rng, grid_size)
        key = grid.tobytes()
        if key in seen:
            continue
        seen.add(key)
        grids.append(grid)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.save(cache, np.stack(grids))
    return grids


@dataclass
class Split:
    """Train / test partition."""

    train: List[np.ndarray]
    test: List[np.ndarray]


_MAX_TRAIN = 10_000


def make_cell_perm_split(
    grid_size: int,
    n_train: int,
    n_test: int = 10_000,
    seed: int = 0,
) -> Split:
    """Create a train/test split of cell permutations with no overlap.

    Same layout as :func:`make_reward_split`: a pool of ``_MAX_TRAIN + n_test``
    permutations is generated; training takes a prefix of the first
    ``_MAX_TRAIN`` and the test set is always the final ``n_test``.
    """
    assert n_train <= _MAX_TRAIN, f"n_train={n_train} exceeds {_MAX_TRAIN}"
    pool = generate_cell_perms(grid_size, _MAX_TRAIN + n_test, seed=seed)
    return Split(
        train=pool[:n_train],
        test=pool[_MAX_TRAIN : _MAX_TRAIN + n_test],
    )


def make_reward_split(
    grid_size: int,
    n_train: int,
    n_test: int = 10_000,
    seed: int = 0,
) -> Split:
    """Create a train/test split of reward grids with no overlap.

    A pool of ``_MAX_TRAIN + n_test`` grids is generated. Training takes a
    prefix of the first ``_MAX_TRAIN``; the test set is always the final
    ``n_test``.  For ``n_train=0`` the train list is empty.
    """
    assert n_train <= _MAX_TRAIN, f"n_train={n_train} exceeds {_MAX_TRAIN}"
    pool = generate_reward_grids(grid_size, _MAX_TRAIN + n_test, seed=seed)
    return Split(
        train=pool[:n_train],
        test=pool[_MAX_TRAIN : _MAX_TRAIN + n_test],
    )
