
from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from tsp.config import (
    EvalBudget,
    HistoryRecorder,
    RunResult,
    SAConfig,
    Snapshot,
    drain,
)
from tsp.instance import Instance
from tsp.perturbations import apply_move, move_delta, propose_move
from tsp.tour import nearest_neighbor_tour, random_tour, tour_cost


def calibrate_t0(
    dist: np.ndarray,
    rng: np.random.Generator,
    cfg: SAConfig,
    budget: EvalBudget | None = None,
) -> float:
    n = dist.shape[0]
    tour = random_tour(rng, n)
    pioras: list[float] = []
    for _ in range(cfg.calib_samples):
        if budget is not None:
            if budget.exhausted:
                break
            budget.spend()
        mv = propose_move(rng, n, cfg.move_weights)
        delta = move_delta(tour, dist, mv)
        if delta > 0.0:
            pioras.append(delta)
        apply_move(tour, mv)

    if not pioras:
        media = float(dist[np.triu_indices(n, k=1)].mean())
    else:
        media = float(np.mean(pioras))
    return -media / math.log(cfg.target_accept_rate)


def _initial_tour(inst: Instance, rng: np.random.Generator, cfg: SAConfig) -> np.ndarray:
    if cfg.init_tour == "random":
        return random_tour(rng, inst.n)
    if cfg.init_tour == "nearest":
        return nearest_neighbor_tour(inst.dist, start=int(rng.integers(inst.n)))
    raise ValueError(f"init_tour inválido: {cfg.init_tour!r}")


def iter_sa(
    inst: Instance,
    rng: np.random.Generator,
    cfg: SAConfig,
    budget: EvalBudget,
    history_points: int,
    snapshots: bool = False,
) -> Iterator[Snapshot]:
    n, dist = inst.n, inst.dist
    recorder = HistoryRecorder(budget.total, history_points)

    t0 = calibrate_t0(dist, rng, cfg, budget)
    t_min = t0 * cfg.t_min_ratio
    n_por_nivel = max(1, int(round(cfg.l_factor * n)))

    alpha = cfg.alpha
    if cfg.auto_alpha and budget.remaining > n_por_nivel:
        niveis = budget.remaining / n_por_nivel
        alpha = float(cfg.t_min_ratio ** (1.0 / niveis))

    tour = _initial_tour(inst, rng, cfg)
    custo = tour_cost(tour, dist)
    melhor_custo = custo
    melhor_rota = tour.copy()
    melhor_em_aval = budget.used
    melhor_em_passo = 0

    temperatura = t0
    passos = 0
    niveis_sem_melhora = 0
    motivo = "orcamento"
    recorder.record(
        budget.used,
        0,
        melhor_custo,
        current_cost=custo,
        temperature=temperatura,
        accept_rate=float("nan"),
    )

    if snapshots:
        yield Snapshot(
            algo="sa",
            tour=tour.copy(),
            best_tour=melhor_rota.copy(),
            cost=custo,
            best_cost=melhor_custo,
            evals=budget.used,
            step=0,
            temperature=temperatura,
            accept_rate=None,
        )

    while True:
        if budget.exhausted:
            motivo = "orcamento"
            break
        if temperatura < t_min:
            motivo = "temperatura_minima"
            break
        if niveis_sem_melhora >= cfg.max_levels_no_improve:
            motivo = "estagnacao"
            break

        aceitos = 0
        tentativas = 0
        melhorou = False
        for _ in range(n_por_nivel):
            if budget.exhausted:
                break
            mv = propose_move(rng, n, cfg.move_weights)
            delta = move_delta(tour, dist, mv)
            budget.spend()
            passos += 1
            tentativas += 1

            if delta <= 0.0 or rng.random() < math.exp(-delta / temperatura):
                apply_move(tour, mv)
                custo += delta
                aceitos += 1
                if custo < melhor_custo:
                    melhor_custo = custo
                    melhor_rota = tour.copy()
                    melhor_em_aval = budget.used
                    melhor_em_passo = passos
                    melhorou = True

            recorder.maybe_record(
                budget.used,
                passos,
                melhor_custo,
                current_cost=custo,
                temperature=temperatura,
                accept_rate=aceitos / tentativas,
            )

        custo = tour_cost(tour, dist)

        taxa = aceitos / tentativas if tentativas else 0.0
        congelado = taxa <= cfg.freeze_accept_rate
        niveis_sem_melhora = 0 if (melhorou or not congelado) else niveis_sem_melhora + 1
        temperatura *= alpha

        if snapshots:
            yield Snapshot(
                algo="sa",
                tour=tour.copy(),
                best_tour=melhor_rota.copy(),
                cost=custo,
                best_cost=melhor_custo,
                evals=budget.used,
                step=passos,
                temperature=temperatura,
                accept_rate=taxa,
            )

    melhor_custo = tour_cost(melhor_rota, dist)
    recorder.record(
        budget.used,
        passos,
        melhor_custo,
        current_cost=custo,
        temperature=temperatura,
        accept_rate=float("nan"),
    )

    return RunResult(
        algo="sa",
        n=n,
        run=inst.run,
        best_cost=melhor_custo,
        best_tour=melhor_rota,
        steps=passos,
        evals_used=budget.used,
        evals_budget=budget.total,
        best_at_eval=melhor_em_aval,
        best_at_step=melhor_em_passo,
        stop_reason=motivo,
        history=recorder.rows,
    )


def run_sa(
    inst: Instance,
    rng: np.random.Generator,
    cfg: SAConfig,
    budget: EvalBudget,
    history_points: int,
) -> RunResult:
    return drain(iter_sa(inst, rng, cfg, budget, history_points))
