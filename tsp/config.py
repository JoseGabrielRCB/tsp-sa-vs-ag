"""Configuração central do experimento.

Todos os hiperparâmetros dos dois algoritmos vivem aqui, em dataclasses imutáveis,
com o motivo de cada default documentado. Nenhum valor mágico deve aparecer espalhado
pelos outros módulos.

Este módulo também abriga os tipos compartilhados (`EvalBudget`, `HistoryRecorder`,
`RunResult`) usados por `sa.py`, `ga.py` e `runner.py`. Eles ficam aqui — e não em
`runner.py` — para evitar import circular: o runner importa os algoritmos, então os
algoritmos não podem importar o runner.
"""

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
    """Hiperparâmetros do Recozimento Simulado."""

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
        """Número de níveis de temperatura até T_min: ln(t_min_ratio) / ln(alpha)."""
        return float(np.log(self.t_min_ratio) / np.log(self.alpha))


@dataclass(frozen=True)
class GAConfig:
    """Hiperparâmetros do Algoritmo Genético."""

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
    """Protocolo de comparação: tamanhos, repetições, orçamento e saídas."""

    sizes: tuple[int, ...] = (20, 30, 50, 75, 100)
    n_runs: int = 10
    evals_per_city: int = 2000
    master_seed: int = 20260805
    history_points: int = 400
    out_dir: str = "results"
    sa: SAConfig = field(default_factory=SAConfig)
    ga: GAConfig = field(default_factory=GAConfig)

    def budget(self, n: int) -> int:
        """Orçamento de avaliações da função objetivo para uma instância de `n` cidades."""
        return self.evals_per_city * n


@dataclass
class EvalBudget:
    """Contador de avaliações da função objetivo, compartilhado por SA e AG.

    A paridade da comparação é definida aqui: os dois algoritmos param quando
    `used >= total`. Uma avaliação é um cálculo do custo de uma solução candidata —
    para o SA, um delta O(1); para o AG, o custo completo de um filho.
    """

    total: int
    used: int = 0

    def spend(self, k: int = 1) -> None:
        """Contabiliza `k` avaliações."""
        self.used += k

    @property
    def exhausted(self) -> bool:
        """True quando o orçamento acabou."""
        return self.used >= self.total

    @property
    def remaining(self) -> int:
        """Avaliações ainda disponíveis (nunca negativo)."""
        return max(0, self.total - self.used)


@dataclass
class HistoryRecorder:
    """Amostra o histórico de convergência em uma grade fixa de avaliações.

    Registrar a cada iteração estouraria a memória (200k iterações x 100 execuções);
    registrar em uma grade comum a SA e AG mantém o CSV pequeno e, principalmente,
    torna as curvas diretamente comparáveis no eixo de avaliações.
    """

    budget: int
    n_points: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    _interval: int = field(init=False)
    _next_at: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._interval = max(1, self.budget // max(1, self.n_points))

    def maybe_record(self, evals: int, step: int, best_cost: float, **extra: float) -> None:
        """Registra uma linha se `evals` cruzou o próximo ponto da grade."""
        if evals >= self._next_at:
            self.record(evals, step, best_cost, **extra)

    def record(self, evals: int, step: int, best_cost: float, **extra: float) -> None:
        """Registra uma linha incondicionalmente (usado também no ponto final)."""
        row: dict[str, Any] = {"evals": evals, "step": step, "best_cost": best_cost}
        row.update(extra)
        self.rows.append(row)
        self._next_at = evals + self._interval


@dataclass
class RunResult:
    """Resultado completo de uma execução de um algoritmo em uma instância."""

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
    """Foto do estado de um algoritmo em um instante, para a visualização ao vivo.

    Emitida por `iter_sa`/`iter_ga` em pontos naturais do algoritmo (um nível de
    temperatura, uma geração). As rotas vêm copiadas: quem consome pode desenhá-las
    sem risco de o algoritmo alterá-las por baixo.

    Campos específicos ficam em `None` no algoritmo que não os produz: `temperature` e
    `accept_rate` só existem no SA; `mean_cost` e `std_cost`, só no AG.
    """

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
    """Consome um gerador de algoritmo até o fim e devolve o `RunResult` que ele retorna.

    Os algoritmos são geradores para que a visualização ao vivo possa acompanhá-los
    passo a passo; o caminho normal (headless) só quer o resultado final.
    """
    while True:
        try:
            next(gerador)
        except StopIteration as fim:
            return fim.value
