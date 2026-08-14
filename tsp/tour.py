
from __future__ import annotations

import numpy as np


def random_tour(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.permutation(n).astype(np.int64)


def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> np.ndarray:
    n = dist.shape[0]
    unvisited = np.ones(n, dtype=bool)
    tour = np.empty(n, dtype=np.int64)
    current = int(start)
    tour[0] = current
    unvisited[current] = False
    for k in range(1, n):
        candidates = np.where(unvisited, dist[current], np.inf)
        current = int(np.argmin(candidates))
        tour[k] = current
        unvisited[current] = False
    return tour


def tour_cost(tour: np.ndarray, dist: np.ndarray) -> float:
    return float(dist[tour, np.roll(tour, -1)].sum())


def is_valid_tour(tour: np.ndarray, n: int) -> bool:
    if tour.shape != (n,):
        return False
    return bool(np.array_equal(np.sort(tour), np.arange(n)))
