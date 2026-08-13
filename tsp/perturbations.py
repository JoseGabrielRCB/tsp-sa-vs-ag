"""Operadores de perturbação sobre rotas, com delta de custo O(1).

Três movimentos, usados tanto pelo SA (vizinhança) quanto pelo AG (mutação) — é o que
torna a comparação entre os dois algoritmos justa: a diferença fica só na estratégia de
busca, não no operador de vizinhança.

* **inversão** (2-opt): inverte o segmento `tour[i..j]`; altera 2 arestas.
* **translação** (or-opt): remove a cidade da posição `i` e a reinsere na posição `j`
  do array reduzido; altera 3 arestas.
* **troca** (swap): troca as cidades das posições `i` e `j`; altera até 4 arestas.

Cada movimento expõe três funções: `propose_*` (sorteio), `delta_*` (variação de custo
em tempo constante, olhando só as arestas afetadas) e `apply_*` (aplicação in-place).
Como a rota é um ciclo fechado, todos os deltas tratam o wrap-around da aresta que liga
a última cidade à primeira — inclusive os casos de adjacência entre as posições `0` e
`n-1`, que é onde este tipo de código costuma errar.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

MOVE_INVERSION = 0
MOVE_TRANSLATION = 1
MOVE_SWAP = 2

MIN_N = 4


class Move(NamedTuple):
    """Um movimento proposto: tipo do operador e suas duas posições."""

    kind: int
    i: int
    j: int


def propose_inversion(rng: np.random.Generator, n: int) -> Move:
    """Sorteia `i < j` para inverter `tour[i..j]`.

    O par `(0, n-1)` é rejeitado: inverter a rota inteira devolve o mesmo ciclo e o
    delta correspondente seria degenerado.
    """
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
    """Variação de custo da inversão de `tour[i..j]` (duas arestas trocadas)."""
    n = tour.shape[0]
    prev = tour[i - 1]
    first = tour[i]
    last = tour[j]
    nxt = tour[(j + 1) % n]
    return float(
        dist[prev, last] + dist[first, nxt] - dist[prev, first] - dist[last, nxt]
    )


def apply_inversion(tour: np.ndarray, i: int, j: int) -> None:
    """Inverte `tour[i..j]` in-place."""
    tour[i : j + 1] = tour[i : j + 1][::-1].copy()


def propose_translation(rng: np.random.Generator, n: int) -> Move:
    """Sorteia a posição `i` da cidade a mover e a posição `j` de reinserção.

    `j` é uma posição no array *reduzido* (a rota sem a cidade removida), logo vive em
    `0..n-2`. `j == i` é descartado por ser a identidade.
    """
    i = int(rng.integers(n))
    while True:
        j = int(rng.integers(n - 1))
        if j != i:
            return Move(MOVE_TRANSLATION, i, j)


def delta_translation(tour: np.ndarray, dist: np.ndarray, i: int, j: int) -> float:
    """Variação de custo de mover `tour[i]` para a posição `j` do array reduzido.

    Duas parcelas: a remoção fecha a lacuna deixada pela cidade (uma aresta criada,
    duas removidas) e a inserção abre a aresta de destino (uma removida, duas criadas).
    """
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
    """Move `tour[i]` para a posição `j` do array reduzido, in-place."""
    city = tour[i]
    if j < i:
        tour[j + 1 : i + 1] = tour[j:i].copy()
    else:
        tour[i:j] = tour[i + 1 : j + 1].copy()
    tour[j] = city


def propose_swap(rng: np.random.Generator, n: int) -> Move:
    """Sorteia duas posições distintas `i < j` para trocar."""
    i = int(rng.integers(n))
    while True:
        j = int(rng.integers(n))
        if j != i:
            break
    if i > j:
        i, j = j, i
    return Move(MOVE_SWAP, i, j)


def delta_swap(tour: np.ndarray, dist: np.ndarray, i: int, j: int) -> float:
    """Variação de custo da troca de `tour[i]` com `tour[j]` (`i < j`).

    Três casos: posições adjacentes, posições adjacentes *através* do fechamento do
    ciclo (`i == 0` e `j == n-1`) e o caso geral com quatro arestas afetadas.
    """
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
    """Troca `tour[i]` com `tour[j]` in-place."""
    tour[i], tour[j] = tour[j], tour[i]


_PROPOSERS = (propose_inversion, propose_translation, propose_swap)
_DELTAS = (delta_inversion, delta_translation, delta_swap)
_APPLIERS = (apply_inversion, apply_translation, apply_swap)


def propose_move(
    rng: np.random.Generator, n: int, weights: tuple[float, float, float]
) -> Move:
    """Sorteia um dos três operadores segundo `weights` e propõe um movimento dele."""
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
    """Variação de custo do movimento, em tempo constante."""
    return _DELTAS[move.kind](tour, dist, move.i, move.j)


def apply_move(tour: np.ndarray, move: Move) -> None:
    """Aplica o movimento in-place."""
    _APPLIERS[move.kind](tour, move.i, move.j)
