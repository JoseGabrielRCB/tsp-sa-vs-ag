
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tsp.config import ALGO_LABELS, EvalBudget, ExperimentConfig, RunResult
from tsp.ga import run_ga
from tsp.instance import algo_seed, make_instance, seed_label
from tsp.sa import run_sa

ALGOS = ("sa", "ga")

RUNS_FIELDS = (
    "algo",
    "n",
    "run",
    "seed",
    "best_cost",
    "time_s",
    "steps",
    "evals_used",
    "evals_budget",
    "best_at_eval",
    "best_at_step",
    "stop_reason",
    "best_tour",
)

HISTORY_FIELDS = (
    "algo",
    "n",
    "run",
    "evals",
    "step",
    "best_cost",
    "current_cost",
    "temperature",
    "accept_rate",
    "mean_cost",
    "std_cost",
)


def run_single(algo: str, n: int, run: int, exp: ExperimentConfig) -> RunResult:
    inst = make_instance(n, run, exp.master_seed)
    semente = algo_seed(exp.master_seed, n, run, algo)
    rng = np.random.default_rng(semente)
    budget = EvalBudget(exp.budget(n))

    inicio = time.perf_counter()
    if algo == "sa":
        resultado = run_sa(inst, rng, exp.sa, budget, exp.history_points)
    elif algo == "ga":
        resultado = run_ga(inst, rng, exp.ga, budget, exp.history_points)
    else:
        raise ValueError(f"algoritmo desconhecido: {algo!r}")
    resultado.time_s = time.perf_counter() - inicio
    return resultado


def run_experiment(
    exp: ExperimentConfig,
    algos: Iterable[str] = ALGOS,
    verbose: bool = True,
) -> list[RunResult]:
    algos = tuple(algos)
    resultados: list[RunResult] = []
    total = len(exp.sizes) * exp.n_runs * len(algos)
    feitos = 0
    for n in exp.sizes:
        for run in range(exp.n_runs):
            for algo in algos:
                resultado = run_single(algo, n, run, exp)
                resultados.append(resultado)
                feitos += 1
                if verbose:
                    print(
                        f"[{feitos:>4}/{total}] n={n:>3} run={run:>2} {algo:<2} "
                        f"custo={resultado.best_cost:9.2f} "
                        f"t={resultado.time_s:6.2f}s "
                        f"avals={resultado.evals_used}/{resultado.evals_budget} "
                        f"({resultado.stop_reason})",
                        flush=True,
                    )
    return resultados


def _run_row(res: RunResult, exp: ExperimentConfig) -> dict[str, Any]:
    return {
        "algo": res.algo,
        "n": res.n,
        "run": res.run,
        "seed": seed_label(algo_seed(exp.master_seed, res.n, res.run, res.algo)),
        "best_cost": f"{res.best_cost:.6f}",
        "time_s": f"{res.time_s:.6f}",
        "steps": res.steps,
        "evals_used": res.evals_used,
        "evals_budget": res.evals_budget,
        "best_at_eval": res.best_at_eval,
        "best_at_step": res.best_at_step,
        "stop_reason": res.stop_reason,
        "best_tour": "-".join(str(c) for c in res.best_tour.tolist()),
    }


def _history_rows(res: RunResult) -> list[dict[str, Any]]:
    linhas = []
    for linha in res.history:
        registro: dict[str, Any] = {
            "algo": res.algo,
            "n": res.n,
            "run": res.run,
            "evals": linha["evals"],
            "step": linha["step"],
            "best_cost": f"{linha['best_cost']:.6f}",
        }
        for campo in ("current_cost", "temperature", "accept_rate", "mean_cost", "std_cost"):
            valor = linha.get(campo)
            registro[campo] = "" if valor is None else f"{valor:.6f}"
        linhas.append(registro)
    return linhas


def _merge(
    antigas: list[dict[str, str]], novas: list[dict[str, Any]], chave: tuple[str, ...]
) -> list[dict[str, Any]]:
    recalculadas = {tuple(str(linha[c]) for c in chave) for linha in novas}
    preservadas = [
        linha for linha in antigas if tuple(str(linha[c]) for c in chave) not in recalculadas
    ]
    return preservadas + novas


def _write_csv(caminho: Path, campos: tuple[str, ...], linhas: list[dict[str, Any]]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


def _read_csv(caminho: Path) -> list[dict[str, str]]:
    if not caminho.exists():
        return []
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def raw_dir(exp: ExperimentConfig) -> Path:
    return Path(exp.out_dir) / "raw"


def save_results(
    resultados: list[RunResult], exp: ExperimentConfig, fresh: bool = False
) -> tuple[Path, Path]:
    runs_path = raw_dir(exp) / "runs.csv"
    hist_path = raw_dir(exp) / "history.csv"

    novas_runs = [_run_row(r, exp) for r in resultados]
    novas_hist = [linha for r in resultados for linha in _history_rows(r)]

    if fresh:
        linhas_runs, linhas_hist = novas_runs, novas_hist
    else:
        chave = ("algo", "n", "run")
        linhas_runs = _merge(_read_csv(runs_path), novas_runs, chave)
        linhas_hist = _merge(_read_csv(hist_path), novas_hist, chave)

    _write_csv(runs_path, RUNS_FIELDS, linhas_runs)
    _write_csv(hist_path, HISTORY_FIELDS, linhas_hist)
    return runs_path, hist_path


def _to_float(valor: str) -> float:
    return float("nan") if valor == "" else float(valor)


def load_runs(exp: ExperimentConfig) -> list[dict[str, Any]]:
    linhas = []
    for bruta in _read_csv(raw_dir(exp) / "runs.csv"):
        linhas.append(
            {
                "algo": bruta["algo"],
                "n": int(bruta["n"]),
                "run": int(bruta["run"]),
                "seed": int(bruta["seed"]),
                "best_cost": float(bruta["best_cost"]),
                "time_s": float(bruta["time_s"]),
                "steps": int(bruta["steps"]),
                "evals_used": int(bruta["evals_used"]),
                "evals_budget": int(bruta["evals_budget"]),
                "best_at_eval": int(bruta["best_at_eval"]),
                "best_at_step": int(bruta["best_at_step"]),
                "stop_reason": bruta["stop_reason"],
                "best_tour": np.array(
                    [int(c) for c in bruta["best_tour"].split("-")], dtype=np.int64
                ),
            }
        )
    return linhas


def load_history(exp: ExperimentConfig) -> list[dict[str, Any]]:
    linhas = []
    for bruta in _read_csv(raw_dir(exp) / "history.csv"):
        registro: dict[str, Any] = {
            "algo": bruta["algo"],
            "n": int(bruta["n"]),
            "run": int(bruta["run"]),
            "evals": int(bruta["evals"]),
            "step": int(bruta["step"]),
        }
        for campo in (
            "best_cost",
            "current_cost",
            "temperature",
            "accept_rate",
            "mean_cost",
            "std_cost",
        ):
            registro[campo] = _to_float(bruta[campo])
        linhas.append(registro)
    return linhas


def summary_table(linhas: list[dict[str, Any]]) -> str:
    tamanhos = sorted({linha["n"] for linha in linhas})
    cabecalho = (
        f"{'n':>4}  {'algoritmo':<22} {'distância (média ± dp)':>26} "
        f"{'tempo (s)':>18} {'passos':>18} {'avaliações':>12}"
    )
    saida = [cabecalho, "-" * len(cabecalho)]

    for n in tamanhos:
        for algo in ALGOS:
            grupo = [x for x in linhas if x["n"] == n and x["algo"] == algo]
            if not grupo:
                continue
            custo = np.array([x["best_cost"] for x in grupo])
            tempo = np.array([x["time_s"] for x in grupo])
            passos = np.array([x["steps"] for x in grupo], dtype=float)
            avals = np.array([x["evals_used"] for x in grupo], dtype=float)
            saida.append(
                f"{n:>4}  {ALGO_LABELS[algo]:<22} "
                f"{custo.mean():>13.2f} ± {custo.std():<10.2f} "
                f"{tempo.mean():>10.2f} ± {tempo.std():<5.2f} "
                f"{passos.mean():>10.0f} ± {passos.std():<5.0f} "
                f"{avals.mean():>11.0f}"
            )
        saida.append("")
    return "\n".join(saida)
