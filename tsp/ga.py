
from __future__ import annotations

from typing import Iterator

import numpy as np

from tsp.config import (
    EvalBudget,
    GAConfig,
    HistoryRecorder,
    RunResult,
    Snapshot,
    drain,
)
from tsp.instance import Instance
from tsp.perturbations import apply_move, propose_move
from tsp.tour import random_tour, tour_cost


def _corte_duplo(rng: np.random.Generator, n: int) -> tuple[int, int]:
    a = int(rng.integers(n))
    b = int(rng.integers(n))
    return (a, b) if a <= b else (b, a)


def ox_crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n = p1.shape[0]
    a, b = _corte_duplo(rng, n)
    filho = np.empty(n, dtype=np.int64)
    filho[a : b + 1] = p1[a : b + 1]

    no_segmento = np.zeros(n, dtype=bool)
    no_segmento[p1[a : b + 1]] = True

    candidatas = np.roll(p2, -(b + 1) % n)
    restantes = candidatas[~no_segmento[candidatas]]

    posicoes = np.concatenate([np.arange(b + 1, n), np.arange(0, a)])
    filho[posicoes] = restantes
    return filho


def pmx_crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n = p1.shape[0]
    a, b = _corte_duplo(rng, n)
    filho = np.full(n, -1, dtype=np.int64)
    filho[a : b + 1] = p1[a : b + 1]

    no_segmento = np.zeros(n, dtype=bool)
    no_segmento[p1[a : b + 1]] = True

    posicao_em_p2 = np.empty(n, dtype=np.int64)
    posicao_em_p2[p2] = np.arange(n, dtype=np.int64)

    for k in range(a, b + 1):
        cidade = p2[k]
        if no_segmento[cidade]:
            continue
        destino = k
        while a <= destino <= b:
            destino = int(posicao_em_p2[p1[destino]])
        filho[destino] = cidade

    faltando = filho == -1
    filho[faltando] = p2[faltando]
    return filho


CROSSOVERS = {"ox": ox_crossover, "pmx": pmx_crossover}


def tournament_select(costs: np.ndarray, rng: np.random.Generator, k: int) -> int:
    candidatos = rng.integers(costs.shape[0], size=k)
    return int(candidatos[int(np.argmin(costs[candidatos]))])


def iter_ga(
    inst: Instance,
    rng: np.random.Generator,
    cfg: GAConfig,
    budget: EvalBudget,
    history_points: int,
    snapshots: bool = False,
) -> Iterator[Snapshot]:
    n, dist = inst.n, inst.dist
    recorder = HistoryRecorder(budget.total, history_points)
    cruzar = CROSSOVERS[cfg.crossover]
    elite = min(cfg.elite, cfg.pop_size - 1)

    pop = np.empty((cfg.pop_size, n), dtype=np.int64)
    custos = np.empty(cfg.pop_size, dtype=np.float64)
    for idx in range(cfg.pop_size):
        pop[idx] = random_tour(rng, n)
        custos[idx] = tour_cost(pop[idx], dist)
    budget.spend(cfg.pop_size)

    melhor_idx = int(np.argmin(custos))
    melhor_custo = float(custos[melhor_idx])
    melhor_rota = pop[melhor_idx].copy()
    melhor_em_aval = budget.used
    melhor_em_passo = 0

    geracoes = 0
    geracoes_sem_melhora = 0
    motivo = "orcamento"
    recorder.record(
        budget.used,
        0,
        melhor_custo,
        mean_cost=float(custos.mean()),
        std_cost=float(custos.std()),
    )

    if snapshots:
        yield Snapshot(
            algo="ga",
            tour=pop[melhor_idx].copy(),
            best_tour=melhor_rota.copy(),
            cost=melhor_custo,
            best_cost=melhor_custo,
            evals=budget.used,
            step=0,
            mean_cost=float(custos.mean()),
            std_cost=float(custos.std()),
        )

    while True:
        if budget.exhausted:
            motivo = "orcamento"
            break
        if geracoes_sem_melhora >= cfg.max_gens_no_improve:
            motivo = "estagnacao"
            break

        ordem = np.argsort(custos, kind="stable")
        nova_pop = np.empty_like(pop)
        novos_custos = np.empty_like(custos)

        nova_pop[:elite] = pop[ordem[:elite]]
        novos_custos[:elite] = custos[ordem[:elite]]

        geracoes += 1
        melhorou = False
        for slot in range(elite, cfg.pop_size):
            if budget.exhausted:
                sobra = ordem[: cfg.pop_size - slot]
                nova_pop[slot:] = pop[sobra]
                novos_custos[slot:] = custos[sobra]
                break

            pai1 = pop[tournament_select(custos, rng, cfg.tournament_k)]
            pai2 = pop[tournament_select(custos, rng, cfg.tournament_k)]
            if rng.random() < cfg.crossover_rate:
                filho = cruzar(pai1, pai2, rng)
            else:
                filho = pai1.copy()

            if rng.random() < cfg.mutation_rate:
                apply_move(filho, propose_move(rng, n, cfg.move_weights))

            custo = tour_cost(filho, dist)
            budget.spend()
            nova_pop[slot] = filho
            novos_custos[slot] = custo

            if custo < melhor_custo:
                melhor_custo = custo
                melhor_rota = filho.copy()
                melhor_em_aval = budget.used
                melhor_em_passo = geracoes
                melhorou = True

        pop, custos = nova_pop, novos_custos
        geracoes_sem_melhora = 0 if melhorou else geracoes_sem_melhora + 1
        recorder.maybe_record(
            budget.used,
            geracoes,
            melhor_custo,
            mean_cost=float(custos.mean()),
            std_cost=float(custos.std()),
        )

        if snapshots:
            melhor_da_geracao = int(np.argmin(custos))
            yield Snapshot(
                algo="ga",
                tour=pop[melhor_da_geracao].copy(),
                best_tour=melhor_rota.copy(),
                cost=float(custos[melhor_da_geracao]),
                best_cost=float(melhor_custo),
                evals=budget.used,
                step=geracoes,
                mean_cost=float(custos.mean()),
                std_cost=float(custos.std()),
            )

    recorder.record(
        budget.used,
        geracoes,
        melhor_custo,
        mean_cost=float(custos.mean()),
        std_cost=float(custos.std()),
    )

    return RunResult(
        algo="ga",
        n=n,
        run=inst.run,
        best_cost=float(melhor_custo),
        best_tour=melhor_rota,
        steps=geracoes,
        evals_used=budget.used,
        evals_budget=budget.total,
        best_at_eval=melhor_em_aval,
        best_at_step=melhor_em_passo,
        stop_reason=motivo,
        history=recorder.rows,
    )


def run_ga(
    inst: Instance,
    rng: np.random.Generator,
    cfg: GAConfig,
    budget: EvalBudget,
    history_points: int,
) -> RunResult:
    return drain(iter_ga(inst, rng, cfg, budget, history_points))
