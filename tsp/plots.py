"""Geração dos gráficos estáticos do trabalho, sempre a partir dos CSVs.

Este módulo **não escolhe o backend do matplotlib**. Quem escolhe é o `main.py`, que
começa em `Agg` (só salva PNG, sem janela); o menu troca para `TkAgg` enquanto a
visualização ao vivo está aberta. Aqui só chamamos `savefig`, que funciona nos dois.

Nenhuma função aqui recebe estado em memória dos algoritmos: tudo vem de
`results/raw/runs.csv` e `results/raw/history.csv`. Assim os PNGs podem ser regerados
sem repetir o experimento (opção 3 do menu). As coordenadas das instâncias, que
não estão no CSV, são reconstruídas por `make_instance` — são determinísticas a partir
de `(master_seed, n, run)`.

Convenções aplicadas em todos os gráficos: uma cor fixa por algoritmo (laranja para o
SA, azul para o AG, par validado para daltonismo), marcador e traço também distintos
(a identidade nunca depende só da cor), título, eixos rotulados, legenda, grade
discreta e `dpi=150`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from tsp.config import ALGO_COLORS, ALGO_LABELS, DPI, ExperimentConfig
from tsp.instance import make_instance
from tsp.runner import ALGOS, load_history, load_runs

TINTA = "#1a1a1a"
TINTA_SUAVE = "#5a5a5a"
GRADE = "#dededa"
SUPERFICIE = "#fcfcfb"
MARCADORES = {"sa": "o", "ga": "s"}
TRACOS = {"sa": "-", "ga": "--"}
MARCADORES_TAMANHO = ("o", "s", "^", "D", "v")


def figures_dir(exp: ExperimentConfig) -> Path:
    """Diretório dos PNGs entregues."""
    caminho = Path(exp.out_dir) / "figures"
    caminho.mkdir(parents=True, exist_ok=True)
    return caminho


def _novo_grafico(figsize=(8, 5), nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SUPERFICIE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        estilizar_eixo(ax)
    return fig, axes


def estilizar_eixo(ax) -> None:
    """Grade e eixos recessivos, texto em tinta neutra.

    Compartilhada com a visualização ao vivo (`tsp/live.py`), para que a janela em
    tempo real e os PNGs entregues tenham exatamente a mesma aparência.
    """
    ax.set_facecolor(SUPERFICIE)
    ax.grid(True, color=GRADE, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TINTA_SUAVE, labelsize=9)
    ax.title.set_color(TINTA)
    ax.xaxis.label.set_color(TINTA_SUAVE)
    ax.yaxis.label.set_color(TINTA_SUAVE)


def _salvar(fig, caminho: Path, apertar: bool = True) -> Path:
    """Salva o PNG. `apertar=False` para figuras que reservam margem manualmente
    (legendas fora da área de plotagem, que o `tight_layout` recortaria)."""
    if apertar:
        fig.tight_layout()
    fig.savefig(caminho, dpi=DPI, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _tamanhos(runs: list[dict[str, Any]]) -> list[int]:
    return sorted({r["n"] for r in runs})


def _filtrar(linhas: list[dict[str, Any]], **criterios: Any) -> list[dict[str, Any]]:
    return [x for x in linhas if all(x[k] == v for k, v in criterios.items())]


def _media_desvio(
    runs: list[dict[str, Any]], algo: str, tamanhos: Sequence[int], campo: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Média e desvio de `campo` por tamanho; devolve também os tamanhos com dados."""
    ns, medias, desvios = [], [], []
    for n in tamanhos:
        valores = [float(r[campo]) for r in _filtrar(runs, algo=algo, n=n)]
        if valores:
            ns.append(n)
            medias.append(float(np.mean(valores)))
            desvios.append(float(np.std(valores)))
    return np.array(ns), np.array(medias), np.array(desvios)


def _curva_media(
    hist: list[dict[str, Any]], algo: str, n: int, budget: int, pontos: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curva média de convergência em uma grade comum de avaliações.

    Cada repetição é reamostrada na mesma grade (`np.interp` mantém o último valor
    depois que a execução para, que é o comportamento correto: o melhor custo não muda
    mais). Só então as repetições são agregadas.
    """
    grade = np.linspace(0, budget, pontos)
    curvas = []
    for run in sorted({h["run"] for h in _filtrar(hist, algo=algo, n=n)}):
        linhas = sorted(_filtrar(hist, algo=algo, n=n, run=run), key=lambda x: x["evals"])
        x = np.array([linha["evals"] for linha in linhas], dtype=float)
        y = np.array([linha["best_cost"] for linha in linhas], dtype=float)
        curvas.append(np.interp(grade, x, y))
    matriz = np.vstack(curvas)
    return grade, matriz.mean(axis=0), matriz.std(axis=0)


def plot_distancia_media(runs, exp) -> Path:
    """1. Distância final média x nº de cidades, com banda de ±1 desvio padrão."""
    fig, ax = _novo_grafico()
    for algo in ALGOS:
        ns, media, desvio = _media_desvio(runs, algo, _tamanhos(runs), "best_cost")
        if not len(ns):
            continue
        cor = ALGO_COLORS[algo]
        ax.plot(
            ns, media, TRACOS[algo], color=cor, linewidth=2,
            marker=MARCADORES[algo], markersize=8, label=ALGO_LABELS[algo],
        )
        ax.fill_between(ns, media - desvio, media + desvio, color=cor, alpha=0.15, linewidth=0)
    ax.set_title("Distância final média por tamanho de instância\n"
                 "(banda = ±1 desvio padrão sobre as repetições)", fontsize=12)
    ax.set_xlabel("Número de cidades (n)")
    ax.set_ylabel("Distância do melhor tour encontrado")
    ax.set_xticks(_tamanhos(runs))
    ax.legend(frameon=False, labelcolor=TINTA)
    return _salvar(fig, figures_dir(exp) / "fig01_distancia_media.png")


def plot_tempo_medio(runs, exp) -> Path:
    """2. Tempo de execução médio x nº de cidades, com banda de ±1 desvio padrão."""
    fig, ax = _novo_grafico()
    for algo in ALGOS:
        ns, media, desvio = _media_desvio(runs, algo, _tamanhos(runs), "time_s")
        if not len(ns):
            continue
        cor = ALGO_COLORS[algo]
        ax.plot(
            ns, media, TRACOS[algo], color=cor, linewidth=2,
            marker=MARCADORES[algo], markersize=8, label=ALGO_LABELS[algo],
        )
        ax.fill_between(ns, media - desvio, media + desvio, color=cor, alpha=0.15, linewidth=0)
    ax.set_title("Tempo de execução médio por tamanho de instância\n"
                 "(mesmo orçamento de avaliações da função objetivo)", fontsize=12)
    ax.set_xlabel("Número de cidades (n)")
    ax.set_ylabel("Tempo de execução (s)")
    ax.set_xticks(_tamanhos(runs))
    ax.legend(frameon=False, labelcolor=TINTA)
    return _salvar(fig, figures_dir(exp) / "fig02_tempo_medio.png")


def plot_passos(runs, exp) -> Path:
    """3. Passos até a parada x nº de cidades.

    Iterações (SA) e gerações (AG) são unidades diferentes, então cada uma fica em seu
    próprio painel — sobrepor as duas em um eixo só daria uma comparação falsa.
    """
    fig, axes = _novo_grafico(figsize=(11, 4.5), ncols=2)
    unidades = {"sa": "Iterações até a parada", "ga": "Gerações até a parada"}
    for ax, algo in zip(axes, ALGOS):
        ns, media, desvio = _media_desvio(runs, algo, _tamanhos(runs), "steps")
        if not len(ns):
            continue
        cor = ALGO_COLORS[algo]
        ax.plot(ns, media, TRACOS[algo], color=cor, linewidth=2,
                marker=MARCADORES[algo], markersize=8)
        ax.fill_between(ns, media - desvio, media + desvio, color=cor, alpha=0.15, linewidth=0)
        ax.set_title(ALGO_LABELS[algo], fontsize=11, color=cor)
        ax.set_xlabel("Número de cidades (n)")
        ax.set_ylabel(unidades[algo])
        ax.set_xticks(_tamanhos(runs))
    fig.suptitle("Passos até o critério de parada (banda = ±1 desvio padrão)",
                 fontsize=12, color=TINTA)
    return _salvar(fig, figures_dir(exp) / "fig03_passos.png")


def plot_convergencia(runs, hist, exp) -> list[Path]:
    """4. Melhor custo x avaliações da função objetivo, SA e AG sobrepostos.

    O eixo x é a contagem de avaliações — e não iterações — porque é essa a moeda em
    que o orçamento dos dois algoritmos foi igualado.
    """
    caminhos = []
    for n in _tamanhos(runs):
        budget = max(r["evals_budget"] for r in _filtrar(runs, n=n))
        fig, ax = _novo_grafico()
        for algo in ALGOS:
            if not _filtrar(hist, algo=algo, n=n):
                continue
            grade, media, desvio = _curva_media(hist, algo, n, budget)
            cor = ALGO_COLORS[algo]
            ax.plot(grade, media, TRACOS[algo], color=cor, linewidth=2, label=ALGO_LABELS[algo])
            ax.fill_between(grade, media - desvio, media + desvio,
                            color=cor, alpha=0.15, linewidth=0)
        orcamento = f"{budget:,}".replace(",", ".")
        ax.set_title(f"Convergência em n = {n} cidades\n"
                     f"(média de {exp.n_runs} repetições, orçamento de {orcamento} avaliações)",
                     fontsize=12)
        ax.set_xlabel("Avaliações da função objetivo")
        ax.set_ylabel("Melhor distância encontrada até o momento")
        ax.legend(frameon=False, labelcolor=TINTA)
        caminhos.append(_salvar(fig, figures_dir(exp) / f"fig04_convergencia_n{n}.png"))
    return caminhos


def plot_boxplot(runs, exp) -> Path:
    """5. Distribuição das distâncias finais, SA e AG lado a lado por tamanho."""
    tamanhos = _tamanhos(runs)
    fig, ax = _novo_grafico(figsize=(9, 5))
    largura = 0.34
    for deslocamento, algo in zip((-largura / 2 - 0.02, largura / 2 + 0.02), ALGOS):
        dados = [[r["best_cost"] for r in _filtrar(runs, algo=algo, n=n)] for n in tamanhos]
        posicoes = np.arange(len(tamanhos)) + deslocamento
        cor = ALGO_COLORS[algo]
        caixa = ax.boxplot(
            dados, positions=posicoes, widths=largura, patch_artist=True,
            medianprops=dict(color=TINTA, linewidth=1.5),
            flierprops=dict(marker=MARCADORES[algo], markersize=5,
                            markerfacecolor=cor, markeredgecolor=cor, alpha=0.7),
        )
        for peca in caixa["boxes"]:
            peca.set(facecolor=cor, alpha=0.35, edgecolor=cor, linewidth=1.5)
        for chave in ("whiskers", "caps"):
            for peca in caixa[chave]:
                peca.set(color=cor, linewidth=1.2)
        ax.plot([], [], color=cor, linewidth=6, alpha=0.5, label=ALGO_LABELS[algo])

    ax.set_xticks(np.arange(len(tamanhos)))
    ax.set_xticklabels([str(n) for n in tamanhos])
    ax.set_title("Distribuição das distâncias finais por tamanho de instância\n"
                 f"({exp.n_runs} repetições independentes por algoritmo)", fontsize=12)
    ax.set_xlabel("Número de cidades (n)")
    ax.set_ylabel("Distância do melhor tour encontrado")
    ax.legend(frameon=False, labelcolor=TINTA)
    return _salvar(fig, figures_dir(exp) / "fig05_boxplot_distancias.png")


def _melhor_por_run(runs, n: int) -> int:
    """Repetição em que o melhor tour geral (entre os dois algoritmos) foi encontrado."""
    do_tamanho = _filtrar(runs, n=n)
    return min(do_tamanho, key=lambda r: r["best_cost"])["run"]


def plot_melhores_rotas(runs, exp) -> list[Path]:
    """6. Melhor rota de cada algoritmo, lado a lado, **sobre a mesma instância**.

    Usar a mesma repetição nos dois painéis é essencial: comparar rotas desenhadas
    sobre nuvens de pontos diferentes não diria nada sobre os algoritmos.
    """
    caminhos = []
    for n in _tamanhos(runs):
        run = _melhor_por_run(runs, n)
        inst = make_instance(n, run, exp.master_seed)
        fig, axes = _novo_grafico(figsize=(11, 5.5), ncols=2)
        for ax, algo in zip(axes, ALGOS):
            selecao = _filtrar(runs, algo=algo, n=n, run=run)
            if not selecao:
                ax.set_visible(False)
                continue
            registro = selecao[0]
            tour = registro["best_tour"]
            ciclo = np.append(tour, tour[0])
            cor = ALGO_COLORS[algo]
            ax.plot(inst.coords[ciclo, 0], inst.coords[ciclo, 1],
                    "-", color=cor, linewidth=1.4, alpha=0.9, zorder=1)
            ax.scatter(inst.coords[:, 0], inst.coords[:, 1], s=28, color=TINTA,
                       zorder=2, edgecolors=SUPERFICIE, linewidths=0.8)
            ax.scatter(*inst.coords[tour[0]], s=130, marker="*", color=cor,
                       zorder=3, edgecolors=SUPERFICIE, linewidths=0.8)
            ax.set_title(f"{ALGO_LABELS[algo]} — distância {registro['best_cost']:.2f}",
                         fontsize=11)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle(f"Melhor rota encontrada — n = {n} cidades (instância run={run})\n"
                     "pontos = cidades, estrela = cidade inicial do ciclo",
                     fontsize=12, color=TINTA, y=1.04)
        caminhos.append(_salvar(fig, figures_dir(exp) / f"fig06_melhor_rota_n{n}.png"))
    return caminhos


def plot_gap_relativo(runs, exp) -> Path:
    """7. Gap médio (%) em relação à melhor solução conhecida entre os dois algoritmos.

    A referência é calculada **por instância**: para cada `(n, run)` toma-se o menor
    custo obtido pelos dois algoritmos e mede-se quanto cada um ficou acima dele.
    """
    tamanhos = _tamanhos(runs)
    gaps: dict[str, list[float]] = {algo: [] for algo in ALGOS}
    erros: dict[str, list[float]] = {algo: [] for algo in ALGOS}

    for n in tamanhos:
        por_algo: dict[str, list[float]] = {algo: [] for algo in ALGOS}
        for run in sorted({r["run"] for r in _filtrar(runs, n=n)}):
            do_run = _filtrar(runs, n=n, run=run)
            referencia = min(r["best_cost"] for r in do_run)
            for r in do_run:
                por_algo[r["algo"]].append((r["best_cost"] / referencia - 1.0) * 100.0)
        for algo in ALGOS:
            valores = por_algo[algo] or [np.nan]
            gaps[algo].append(float(np.mean(valores)))
            erros[algo].append(float(np.std(valores)))

    fig, ax = _novo_grafico(figsize=(9, 5))
    x = np.arange(len(tamanhos))
    largura = 0.36
    for deslocamento, algo in zip((-largura / 2 - 0.01, largura / 2 + 0.01), ALGOS):
        cor = ALGO_COLORS[algo]
        barras = ax.bar(x + deslocamento, gaps[algo], largura, yerr=erros[algo],
                        color=cor, alpha=0.85, label=ALGO_LABELS[algo],
                        error_kw=dict(ecolor=TINTA_SUAVE, capsize=3, linewidth=1))
        ax.bar_label(barras, fmt="%.1f%%", padding=3, fontsize=8, color=TINTA_SUAVE)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in tamanhos])
    ax.set_xlim(-0.5, len(tamanhos) - 0.5)
    ax.set_title("Gap relativo médio em relação à melhor solução conhecida\n"
                 "(referência: o menor custo entre os dois algoritmos, por instância)",
                 fontsize=12)
    ax.set_xlabel("Número de cidades (n)")
    ax.set_ylabel("Gap médio (%)")
    ax.axhline(0, color=GRADE, linewidth=1)
    ax.legend(frameon=False, labelcolor=TINTA)
    return _salvar(fig, figures_dir(exp) / "fig07_gap_relativo.png")


def plot_tradeoff(runs, exp) -> Path:
    """8. Distância final x tempo, um ponto por execução (cor = algoritmo, marcador = n)."""
    tamanhos = _tamanhos(runs)
    fig, ax = _novo_grafico(figsize=(11, 5.5))
    fig.subplots_adjust(right=0.84, top=0.80)
    for algo in ALGOS:
        cor = ALGO_COLORS[algo]
        for idx, n in enumerate(tamanhos):
            grupo = _filtrar(runs, algo=algo, n=n)
            if not grupo:
                continue
            ax.scatter(
                [r["time_s"] for r in grupo], [r["best_cost"] for r in grupo],
                s=52, color=cor, alpha=0.75,
                marker=MARCADORES_TAMANHO[idx % len(MARCADORES_TAMANHO)],
                edgecolors=SUPERFICIE, linewidths=0.6,
            )
    legenda_algo = [
        plt.Line2D([], [], color=ALGO_COLORS[a], marker="o", linestyle="",
                   markersize=8, label=ALGO_LABELS[a])
        for a in ALGOS
    ]
    legenda_n = [
        plt.Line2D([], [], color=TINTA_SUAVE,
                   marker=MARCADORES_TAMANHO[i % len(MARCADORES_TAMANHO)],
                   linestyle="", markersize=7, label=f"n = {n}")
        for i, n in enumerate(tamanhos)
    ]
    primeira = ax.legend(handles=legenda_algo, frameon=False, labelcolor=TINTA,
                         loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2)
    ax.add_artist(primeira)
    segunda = ax.legend(handles=legenda_n, frameon=False, labelcolor=TINTA,
                        loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        fontsize=9, title="tamanho")
    segunda.get_title().set_color(TINTA_SUAVE)

    ax.set_xscale("log")
    ax.set_title("Trade-off entre qualidade e tempo\n"
                 "(um ponto por execução; canto inferior esquerdo = melhor)",
                 fontsize=12, pad=34)
    ax.set_xlabel("Tempo de execução (s, escala logarítmica)")
    ax.set_ylabel("Distância do melhor tour encontrado")
    return _salvar(fig, figures_dir(exp) / "fig08_tradeoff_qualidade_tempo.png", apertar=False)


def plot_diagnostico_sa(hist, exp) -> Path:
    """9. Temperatura e taxa de aceitação do SA ao longo das iterações.

    Duas grandezas de escalas incomparáveis, então dois painéis empilhados em vez de
    um gráfico de eixo duplo.
    """
    tamanhos = sorted({h["n"] for h in hist if h["algo"] == "sa"})
    n = tamanhos[-1]
    linhas = _filtrar(hist, algo="sa", n=n)
    fig, axes = _novo_grafico(figsize=(9, 6.5), nrows=2, sharex=True)
    cor = ALGO_COLORS["sa"]

    for run in sorted({linha["run"] for linha in linhas}):
        serie = sorted(_filtrar(linhas, run=run), key=lambda x: x["step"])
        passos = [x["step"] for x in serie]
        axes[0].plot(passos, [x["temperature"] for x in serie],
                     color=cor, linewidth=1, alpha=0.45)
        axes[1].plot(passos, [x["accept_rate"] for x in serie],
                     color=cor, linewidth=1, alpha=0.45)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Temperatura (escala log)")
    axes[0].set_title(f"Diagnóstico do Recozimento Simulado — n = {n} cidades\n"
                      f"(uma linha por repetição, {exp.n_runs} no total)", fontsize=12)
    axes[1].set_ylabel("Taxa de aceitação no nível")
    axes[1].set_xlabel("Iteração")
    axes[1].set_ylim(0, 1.02)
    axes[1].axhline(exp.sa.target_accept_rate, color=TINTA_SUAVE, linestyle=":", linewidth=1.2)
    axes[1].annotate(f"alvo da calibração de T0 = {exp.sa.target_accept_rate:.2f}",
                     xy=(0.02, exp.sa.target_accept_rate), xycoords=("axes fraction", "data"),
                     xytext=(0, 4), textcoords="offset points",
                     va="bottom", color=TINTA_SUAVE, fontsize=8)
    axes[1].axhline(exp.sa.freeze_accept_rate, color=TINTA_SUAVE, linestyle=":", linewidth=1.2)
    axes[1].annotate(f"limiar de congelamento = {exp.sa.freeze_accept_rate:.2f}",
                     xy=(0.02, exp.sa.freeze_accept_rate), xycoords=("axes fraction", "data"),
                     xytext=(0, 4), textcoords="offset points",
                     va="bottom", color=TINTA_SUAVE, fontsize=8)
    return _salvar(fig, figures_dir(exp) / "fig09_diagnostico_sa.png")


def plot_diagnostico_ga(hist, exp) -> Path:
    """10. Melhor custo x custo médio da população do AG, ao longo das gerações.

    A distância entre as duas curvas é a leitura de diversidade: quando o custo médio
    encosta no melhor, a população convergiu (prematuramente ou não).
    """
    tamanhos = sorted({h["n"] for h in hist if h["algo"] == "ga"})
    n = tamanhos[-1]
    linhas = _filtrar(hist, algo="ga", n=n)
    geracoes = sorted({linha["step"] for linha in linhas})

    def media_por_geracao(campo: str) -> np.ndarray:
        return np.array([
            float(np.mean([x[campo] for x in linhas if x["step"] == g])) for g in geracoes
        ])

    melhor = media_por_geracao("best_cost")
    medio = media_por_geracao("mean_cost")
    desvio = media_por_geracao("std_cost")

    fig, ax = _novo_grafico(figsize=(9, 5.5))
    cor = ALGO_COLORS["ga"]
    ax.fill_between(geracoes, medio - desvio, medio + desvio, color=TINTA_SUAVE,
                    alpha=0.15, linewidth=0, label="±1 desvio padrão da população")
    ax.plot(geracoes, medio, "--", color=TINTA_SUAVE, linewidth=1.8,
            label="custo médio da população")
    ax.plot(geracoes, melhor, "-", color=cor, linewidth=2, label="melhor custo global")

    ax.set_title(f"Diagnóstico do Algoritmo Genético — n = {n} cidades\n"
                 f"(média de {exp.n_runs} repetições; curvas juntas = perda de diversidade)",
                 fontsize=12)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Distância")
    ax.legend(frameon=False, labelcolor=TINTA)
    return _salvar(fig, figures_dir(exp) / "fig10_diagnostico_ga.png")


FIGURAS = {
    "distancia": ("distância final média x n",
                  lambda runs, hist, exp: [plot_distancia_media(runs, exp)]),
    "tempo": ("tempo de execução médio x n",
              lambda runs, hist, exp: [plot_tempo_medio(runs, exp)]),
    "passos": ("iterações/gerações até a parada x n",
               lambda runs, hist, exp: [plot_passos(runs, exp)]),
    "rotas": ("melhor rota encontrada, um arquivo por tamanho",
              lambda runs, hist, exp: plot_melhores_rotas(runs, exp)),
    "convergencia": ("melhor custo x avaliações, um arquivo por tamanho",
                     lambda runs, hist, exp: plot_convergencia(runs, hist, exp)),
    "boxplot": ("distribuição das distâncias finais",
                lambda runs, hist, exp: [plot_boxplot(runs, exp)]),
    "gap": ("gap relativo (%) por tamanho",
            lambda runs, hist, exp: [plot_gap_relativo(runs, exp)]),
    "tradeoff": ("distância final x tempo, um ponto por execução",
                 lambda runs, hist, exp: [plot_tradeoff(runs, exp)]),
    "diag-sa": ("temperatura e taxa de aceitação do SA",
                lambda runs, hist, exp: ([plot_diagnostico_sa(hist, exp)]
                                         if any(h["algo"] == "sa" for h in hist) else [])),
    "diag-ga": ("melhor custo x custo médio da população do AG",
                lambda runs, hist, exp: ([plot_diagnostico_ga(hist, exp)]
                                         if any(h["algo"] == "ga" for h in hist) else [])),
}

FIGURAS_PADRAO = ("distancia", "tempo", "passos", "rotas")


def gerar_todos(
    exp: ExperimentConfig,
    figuras: Sequence[str] = FIGURAS_PADRAO,
    verbose: bool = True,
) -> list[Path]:
    """Gera os PNGs selecionados a partir dos CSVs em `results/raw/`."""
    desconhecidas = [nome for nome in figuras if nome not in FIGURAS]
    if desconhecidas:
        disponiveis = ", ".join(FIGURAS)
        raise SystemExit(
            f"Gráfico desconhecido: {', '.join(desconhecidas)}. Disponíveis: {disponiveis}"
        )

    runs = load_runs(exp)
    hist = load_history(exp)
    if not runs:
        raise SystemExit(
            "Nenhum resultado em results/raw/runs.csv — rode o experimento antes de plotar."
        )

    caminhos: list[Path] = []
    for nome in figuras:
        _, funcao = FIGURAS[nome]
        caminhos += funcao(runs, hist, exp)

    if verbose:
        for caminho in caminhos:
            print(f"  {caminho}")
    return caminhos
