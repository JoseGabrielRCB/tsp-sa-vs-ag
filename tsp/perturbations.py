
from __future__ import annotations

from typing import NamedTuple

import numpy as np

MOVE_INVERSION = 0
MOVE_TRANSLATION = 1
MOVE_SWAP = 2

MIN_N = 4


class Move(NamedTuple):

    kind: int
    i: int
    j: int


def propose_inversion(rng: np.random.Generator, n: int) -> Move:
    while True:
        i = int(rng.integers(n))
        j = int(rng.integers(n))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if i == 0 and j == n - 1:
            continue
        return Move(MOVE_INVERSION, i, j)


def delta_inversion(tour: np.ndarray, dist: np.ndarray, i: int, j: int) -> float:
    n = tour.shape[0]
    prev = tour[i - 1]
    first = tour[i]
    last = tour[j]
    nxt = tour[(j + 1) % n]
    return float(
        dist[prev, last] + dist[first, nxt] - dist[prev, first] - dist[last, nxt]
    )


def apply_inversion(tour: np.ndarray, i: int, j: int) -> None:
    tour[i : j + 1] = tour[i : j + 1][::-1].copy()


def propose_translation(rng: np.random.Generator, n: int) -> Move:
    i = int(rng.integers(n))
    while True:
        j = int(rng.integers(n - 1))
        if j != i:
            return Move(MOVE_TRANSLATION, i, j)


def delta_translation(tour: np.ndarray, dist: np.ndarray, i: int, j: int) -> float:
    n = tour.shape[0]
    m = n - 1
    city = tour[i]
    prev = tour[i - 1]
    nxt = tour[(i + 1) % n]
    removal = dist[prev, nxt] - dist[prev, city] - dist[city, nxt]

    left_idx = (j - 1) % m
    left = tour[left_idx if left_idx < i else left_idx + 1]
    right = tour[j if j < i else j + 1]
    insertion = dist[left, city] + dist[city, right] - dist[left, right]
    return float(removal + insertion)


def apply_translation(tour: np.ndarray, i: int, j: int) -> None:
    city = tour[i]
    if j < i:
        tour[j + 1 : i + 1] = tour[j:i].copy()
    else:
        tour[i:j] = tour[i + 1 : j + 1].copy()
    tour[j] = city


def propose_swap(rng: np.random.Generator, n: int) -> Move:
    i = int(rng.integers(n))
    while True:
        j = int(rng.integers(n))
        if j != i:
            break
    if i > j:
        i, j = j, i
    return Move(MOVE_SWAP, i, j)


def delta_swap(tour: np.ndarray, dist: np.ndarray, i: int, j: int) -> float:
    n = tour.shape[0]
    ci = tour[i]
    cj = tour[j]

    if j == i + 1:
        prev = tour[i - 1]
        nxt = tour[(j + 1) % n]
        return float(dist[prev, cj] + dist[ci, nxt] - dist[prev, ci] - dist[cj, nxt])

    if i == 0 and j == n - 1:
        prev = tour[j - 1]
        nxt = tour[i + 1]
        return float(dist[prev, ci] + dist[cj, nxt] - dist[prev, cj] - dist[ci, nxt])

    prev_i = tour[i - 1]
    next_i = tour[i + 1]
    prev_j = tour[j - 1]
    next_j = tour[(j + 1) % n]
    return float(
        dist[prev_i, cj]
        + dist[cj, next_i]
        + dist[prev_j, ci]
        + dist[ci, next_j]
        - dist[prev_i, ci]
        - dist[ci, next_i]
        - dist[prev_j, cj]
        - dist[cj, next_j]
    )


def apply_swap(tour: np.ndarray, i: int, j: int) -> None:
    tour[i], tour[j] = tour[j], tour[i]


_PROPOSERS = (propose_inversion, propose_translation, propose_swap)
_DELTAS = (delta_inversion, delta_translation, delta_swap)
_APPLIERS = (apply_inversion, apply_translation, apply_swap)


def propose_move(
    rng: np.random.Generator, n: int, weights: tuple[float, float, float]
) -> Move:
    if n < MIN_N:
        raise ValueError(f"n = {n} é pequeno demais para os operadores (mínimo {MIN_N})")
    u = rng.random() * (weights[0] + weights[1] + weights[2])
    if u < weights[0]:
        kind = MOVE_INVERSION
    elif u < weights[0] + weights[1]:
        kind = MOVE_TRANSLATION
    else:
        kind = MOVE_SWAP
    return _PROPOSERS[kind](rng, n)


def move_delta(tour: np.ndarray, dist: np.ndarray, move: Move) -> float:
    return _DELTAS[move.kind](tour, dist, move.i, move.j)


def apply_move(tour: np.ndarray, move: Move) -> None:
    _APPLIERS[move.kind](tour, move.i, move.j)
