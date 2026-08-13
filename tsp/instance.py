"""Geração determinística de instâncias do TSP e matriz de distâncias.

As instâncias são reproduzíveis a partir de `(master_seed, n, run)`: qualquer execução
futura, em qualquer máquina, gera exatamente as mesmas coordenadas. É isso que permite
comparar SA e AG sobre as mesmas instâncias e replotar tudo a partir do CSV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COORD_MAX = 100.0

_ALGO_STREAM = {"sa": 1, "ga": 2}


@dataclass(frozen=True)
class Instance:
    """Uma instância do TSP euclidiano 2D com a matriz de distâncias pré-computada."""

    n: int
    run: int
    coords: np.ndarray
    dist: np.ndarray


def instance_seed(master_seed: int, n: int, run: int) -> np.random.SeedSequence:
    """Semente da instância — a mesma para os dois algoritmos."""
    return np.random.SeedSequence([master_seed, n, run])


def algo_seed(master_seed: int, n: int, run: int, algo: str) -> np.random.SeedSequence:
    """Semente do fluxo aleatório de um algoritmo, independente da semente da instância.

    Separar os fluxos garante que trocar o algoritmo não altera a instância sorteada.
    """
    if algo not in _ALGO_STREAM:
        raise ValueError(f"algoritmo desconhecido: {algo!r}")
    return np.random.SeedSequence([master_seed, n, run, _ALGO_STREAM[algo]])


def seed_label(seq: np.random.SeedSequence) -> int:
    """Inteiro estável derivado de uma SeedSequence, para registrar no CSV."""
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def random_coords(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sorteia `n` coordenadas uniformes em [0, COORD_MAX]^2."""
    return rng.uniform(0.0, COORD_MAX, size=(n, 2)).astype(np.float64)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Matriz de distâncias euclidianas (n, n), float64, simétrica e com diagonal nula.

    Pré-computada uma única vez por instância: nenhum `sqrt` é calculado dentro do
    laço quente dos algoritmos.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    return np.ascontiguousarray(dist, dtype=np.float64)


def make_instance(n: int, run: int, master_seed: int) -> Instance:
    """Constrói a instância determinística associada a `(master_seed, n, run)`."""
    rng = np.random.default_rng(instance_seed(master_seed, n, run))
    coords = random_coords(rng, n)
    return Instance(n=n, run=run, coords=coords, dist=distance_matrix(coords))
