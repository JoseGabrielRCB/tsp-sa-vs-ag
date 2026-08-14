
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


SA_COLOR = "#D55E00"
GA_COLOR = "#0072B2"
ALGO_COLORS = {"sa": SA_COLOR, "ga": GA_COLOR}
ALGO_LABELS = {"sa": "Recozimento Simulado", "ga": "Algoritmo Genético"}
ALGO_SIGLAS = {"sa": "SA", "ga": "AG"}
DPI = 150

MOVE_KINDS = ("inversion", "translation", "swap")


@dataclass(frozen=True)
class SAConfig:

    alpha: float = 0.995
    l_factor: float = 1.0
    t_min_ratio: float = 1e-5
    target_accept_rate: float = 0.8
    calib_samples: int = 1000
    max_levels_no_improve: int = 200
    freeze_accept_rate: float = 0.02
    init_tour: str = "random"
    move_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    auto_alpha: bool = False

    def cooling_levels(self) -> float:
        return float(np.log(self.t_min_ratio) / np.log(self.alpha))


@dataclass(frozen=True)
class GAConfig:

    pop_size: int = 100
    tournament_k: int = 3
    crossover: str = "ox"
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    elite: int = 2
    max_gens_no_improve: int = 200
    move_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)


@dataclass(frozen=True)
class ExperimentConfig:

    sizes: tuple[int, ...] = (20, 30, 50, 75, 100)
    n_runs: int = 10
    evals_per_city: int = 2000
    master_seed: int = 20260805
    history_points: int = 400
    out_dir: str = "results"
    sa: SAConfig = field(default_factory=SAConfig)
    ga: GAConfig = field(default_factory=GAConfig)

    def budget(self, n: int) -> int:
        return self.evals_per_city * n


@dataclass
class EvalBudget:

    total: int
    used: int = 0

    def spend(self, k: int = 1) -> None:
        self.used += k

    @property
    def exhausted(self) -> bool:
        return self.used >= self.total

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)


@dataclass
class HistoryRecorder:

    budget: int
    n_points: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    _interval: int = field(init=False)
    _next_at: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._interval = max(1, self.budget // max(1, self.n_points))

    def maybe_record(self, evals: int, step: int, best_cost: float, **extra: float) -> None:
        if evals >= self._next_at:
            self.record(evals, step, best_cost, **extra)

    def record(self, evals: int, step: int, best_cost: float, **extra: float) -> None:
        row: dict[str, Any] = {"evals": evals, "step": step, "best_cost": best_cost}
        row.update(extra)
        self.rows.append(row)
        self._next_at = evals + self._interval


@dataclass
class RunResult:

    algo: str
    n: int
    run: int
    best_cost: float
    best_tour: np.ndarray
    steps: int
    evals_used: int
    evals_budget: int
    best_at_eval: int
    best_at_step: int
    stop_reason: str
    history: list[dict[str, Any]]
    time_s: float = 0.0


@dataclass(frozen=True)
class Snapshot:

    algo: str
    tour: np.ndarray
    best_tour: np.ndarray
    cost: float
    best_cost: float
    evals: int
    step: int
    temperature: float | None = None
    accept_rate: float | None = None
    mean_cost: float | None = None
    std_cost: float | None = None


def drain(gerador: Iterator[Snapshot]) -> RunResult:
    while True:
        try:
            next(gerador)
        except StopIteration as fim:
            return fim.value
