
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COORD_MAX = 100.0

_ALGO_STREAM = {"sa": 1, "ga": 2}


@dataclass(frozen=True)
class Instance:

    n: int
    run: int
    coords: np.ndarray
    dist: np.ndarray


def instance_seed(master_seed: int, n: int, run: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([master_seed, n, run])


def algo_seed(master_seed: int, n: int, run: int, algo: str) -> np.random.SeedSequence:
    if algo not in _ALGO_STREAM:
        raise ValueError(f"algoritmo desconhecido: {algo!r}")
    return np.random.SeedSequence([master_seed, n, run, _ALGO_STREAM[algo]])


def seed_label(seq: np.random.SeedSequence) -> int:
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def random_coords(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.0, COORD_MAX, size=(n, 2)).astype(np.float64)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    return np.ascontiguousarray(dist, dtype=np.float64)


def make_instance(n: int, run: int, master_seed: int) -> Instance:
    rng = np.random.default_rng(instance_seed(master_seed, n, run))
    coords = random_coords(rng, n)
    return Instance(n=n, run=run, coords=coords, dist=distance_matrix(coords))
