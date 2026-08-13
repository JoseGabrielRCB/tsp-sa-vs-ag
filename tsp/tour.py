"""Representação da rota e cálculo de custo.

Uma rota é uma permutação dos índices `0..n-1` armazenada em um `np.ndarray` de
inteiros. O ciclo é fechado: a aresta entre a última e a primeira cidade também conta.
Este módulo é compartilhado pelo SA e pelo AG.
"""

from __future__ import annotations

import numpy as np


def random_tour(rng: np.random.Generator, n: int) -> np.ndarray:
    """Permutação aleatória de `0..n-1`."""
    return rng.permutation(n).astype(np.int64)


def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> np.ndarray:
    """Rota construída pela heurística do vizinho mais próximo a partir de `start`."""
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
    """Custo total do ciclo fechado, somando as `n` arestas (incluindo o retorno)."""
    return float(dist[tour, np.roll(tour, -1)].sum())


def is_valid_tour(tour: np.ndarray, n: int) -> bool:
    """True se `tour` é uma permutação de `0..n-1` (sem repetição nem cidade faltando)."""
    if tour.shape != (n,):
        return False
    return bool(np.array_equal(np.sort(tour), np.arange(n)))
