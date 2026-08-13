"""Visualização em tempo real: SA e AG disputando o mesmo orçamento de avaliações.

Abre uma janela matplotlib com seis painéis que se atualizam enquanto os algoritmos
rodam. Os dois avançam **até a mesma marca de avaliações a cada quadro** — é o protocolo
de orçamento equivalente do trabalho, mostrado em vez de descrito.

O módulo tem duas partes bem separadas:

* `EstadoAoVivo` — o motor, sem nenhuma dependência de interface gráfica. Guarda os dois
  geradores, avança ambos até um alvo comum de avaliações e acumula o histórico. É essa
  separação que permite testá-lo sem abrir janela nenhuma (`tests/test_live.py`).
* `PainelAoVivo` e `rodar_ao_vivo` — a figura, os controles e o laço de quadros.

Nada aqui grava CSV: o ritmo é artificial, os tempos não seriam comparáveis com os do
experimento e não podem contaminar `results/raw/`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from tsp.config import (
    ALGO_COLORS,
    ALGO_LABELS,
    EvalBudget,
    GAConfig,
    RunResult,
    SAConfig,
    Snapshot,
)
from tsp.ga import iter_ga
from tsp.instance import Instance, algo_seed
from tsp.plots import GRADE, SUPERFICIE, TINTA, TINTA_SUAVE, estilizar_eixo
from tsp.sa import iter_sa

MAX_N_COM_ROTULOS = 40

SEGUNDOS_A_VELOCIDADE_1 = 120.0

LOG2_VEL_MIN, LOG2_VEL_MAX = -2.0, 6.0

MAX_PONTOS_NA_CURVA = 500


def _colunas(historico: list[tuple]) -> list[np.ndarray]:
    """Converte o histórico em colunas numpy, decimando **antes** de converter.

    A decimação vem primeiro de propósito: converter listas de tuplas do Python para
    numpy custa caro por elemento, e o histórico só cresce. Amostrando primeiro, o custo
    por quadro fica constante em vez de crescer junto com a execução.
    """
    total = len(historico)
    if total <= MAX_PONTOS_NA_CURVA:
        linhas = historico
    else:
        indices = np.linspace(0, total - 1, MAX_PONTOS_NA_CURVA).astype(np.int64)
        linhas = [historico[i] for i in indices]
    return [np.asarray(coluna, dtype=float) for coluna in zip(*linhas)]


class _Blit:
    """Redesenha só os artistas que mudam, sobre um fundo cacheado.

    Sem isso a janela roda a ~2 quadros por segundo: um render completo desta figura
    (seis painéis) custa cerca de 450 ms, e o matplotlib redesenha tudo — grades, eixos,
    rótulos, legendas — a cada quadro. Aqui o fundo estático é fotografado uma vez e
    reaproveitado; por quadro só as linhas e os textos são desenhados por cima.

    Os artistas dinâmicos são criados com `animated=True`, o que faz o matplotlib
    ignorá-los no desenho normal — é isso que impede que eles fiquem impressos no fundo.
    Cada desenho completo (redimensionar a janela, mexer no slider) dispara
    `draw_event` e o fundo é refotografado sozinho.
    """

    def __init__(self, canvas, figura, artistas: list[Any]) -> None:
        self.canvas = canvas
        self.figura = figura
        self.artistas = artistas
        self._fundo = None
        canvas.mpl_connect("draw_event", self._ao_desenhar)

    def _ao_desenhar(self, _evento) -> None:
        self._fundo = self.canvas.copy_from_bbox(self.figura.bbox)
        self._desenhar_artistas()

    def _desenhar_artistas(self) -> None:
        for artista in self.artistas:
            self.figura.draw_artist(artista)

    def atualizar(self) -> None:
        """Restaura o fundo, redesenha os artistas dinâmicos e manda para a tela."""
        if self._fundo is None:
            self.canvas.draw()
            return
        self.canvas.restore_region(self._fundo)
        self._desenhar_artistas()
        self.canvas.blit(self.figura.bbox)


@dataclass
class EstadoAoVivo:
    """Conduz SA e AG em paralelo, sincronizados pela contagem de avaliações.

    Cada algoritmo tem seu próprio `EvalBudget` com o mesmo total, exatamente como no
    experimento. A cada chamada de `avancar`, ambos são puxados até que tenham gasto o
    mesmo número de avaliações — é o que mantém o eixo x da convergência honesto.
    """

    inst: Instance
    sa_cfg: SAConfig
    ga_cfg: GAConfig
    orcamento: int
    master_seed: int
    run: int
    history_points: int = 400

    snap_sa: Snapshot | None = field(init=False, default=None)
    snap_ga: Snapshot | None = field(init=False, default=None)
    snap_inicial: Snapshot | None = field(init=False, default=None)
    resultado_sa: RunResult | None = field(init=False, default=None)
    resultado_ga: RunResult | None = field(init=False, default=None)
    historico_sa: list[tuple[int, int, float, float]] = field(init=False, default_factory=list)
    historico_ga: list[tuple[int, int, float, float, float]] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        self.reiniciar()

    def reiniciar(self) -> None:
        """Volta ao ponto de partida, com as mesmas sementes — a corrida se repete igual."""
        n, run = self.inst.n, self.run
        self.budget_sa = EvalBudget(self.orcamento)
        self.budget_ga = EvalBudget(self.orcamento)
        self._ger_sa: Iterator[Snapshot] = iter_sa(
            self.inst,
            np.random.default_rng(algo_seed(self.master_seed, n, run, "sa")),
            self.sa_cfg,
            self.budget_sa,
            self.history_points,
            snapshots=True,
        )
        self._ger_ga: Iterator[Snapshot] = iter_ga(
            self.inst,
            np.random.default_rng(algo_seed(self.master_seed, n, run, "ga")),
            self.ga_cfg,
            self.budget_ga,
            self.history_points,
            snapshots=True,
        )
        self.snap_sa = self.snap_ga = self.snap_inicial = None
        self.resultado_sa = self.resultado_ga = None
        self.historico_sa.clear()
        self.historico_ga.clear()
        self._alvo = 0.0

    @property
    def terminou(self) -> bool:
        """True quando os dois algoritmos chegaram ao fim."""
        return self.resultado_sa is not None and self.resultado_ga is not None

    @property
    def progresso(self) -> float:
        """Fração do orçamento já consumida (a maior das duas execuções)."""
        gasto = max(self.budget_sa.used, self.budget_ga.used)
        return min(1.0, gasto / self.orcamento)

    def avancar(self, avaliacoes: float) -> None:
        """Avança os dois algoritmos até a próxima marca comum de avaliações."""
        self._alvo = min(self._alvo + avaliacoes, float(self.orcamento))
        self._avancar_algoritmo("sa")
        self._avancar_algoritmo("ga")

    def _avancar_algoritmo(self, algo: str) -> None:
        sa = algo == "sa"
        if (self.resultado_sa if sa else self.resultado_ga) is not None:
            return
        gerador = self._ger_sa if sa else self._ger_ga
        budget = self.budget_sa if sa else self.budget_ga

        while (
            budget.used < self._alvo
            or (self.snap_sa if sa else self.snap_ga) is None
            or budget.exhausted
        ):
            try:
                snapshot = next(gerador)
            except StopIteration as fim:
                if sa:
                    self.resultado_sa = fim.value
                else:
                    self.resultado_ga = fim.value
                return
            self._registrar(snapshot)

    def _registrar(self, snapshot: Snapshot) -> None:
        if snapshot.algo == "sa":
            self.snap_sa = snapshot
            if snapshot.step == 0:
                self.snap_inicial = snapshot
            self.historico_sa.append(
                (snapshot.evals, snapshot.step, snapshot.best_cost,
                 float("nan") if snapshot.accept_rate is None else snapshot.accept_rate)
            )
        else:
            self.snap_ga = snapshot
            self.historico_ga.append(
                (snapshot.evals, snapshot.step, snapshot.best_cost,
                 snapshot.mean_cost or 0.0, snapshot.std_cost or 0.0)
            )


class PainelAoVivo:
    """A figura de seis painéis, os controles e a lógica de redesenho."""

    def __init__(self, estado: EstadoAoVivo, fps: int, velocidade: float) -> None:
        self.estado = estado
        self.fps = fps
        self.pausado = False
        self.fechado = False
        self._rota_inicial_desenhada = False
        self._limites_prontos = False

        n = estado.inst.n
        self.fig = plt.figure(figsize=(15, 8.6), facecolor=SUPERFICIE)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(
                "Otimização de Rota (TSP): Recozimento Simulado vs Algoritmo Genético"
            )
        grade = self.fig.add_gridspec(
            2, 3, top=0.83, bottom=0.17, left=0.05, right=0.98, hspace=0.52, wspace=0.22
        )
        self.ax_inicial = self.fig.add_subplot(grade[0, 0])
        self.ax_sa = self.fig.add_subplot(grade[0, 1])
        self.ax_ga = self.fig.add_subplot(grade[0, 2])
        self.ax_conv = self.fig.add_subplot(grade[1, 0])
        self.ax_aceit = self.fig.add_subplot(grade[1, 1])
        self.ax_div = self.fig.add_subplot(grade[1, 2])
        for ax in (self.ax_inicial, self.ax_sa, self.ax_ga, self.ax_conv,
                   self.ax_aceit, self.ax_div):
            estilizar_eixo(ax)

        self._montar_paineis_de_rota(n)
        self._montar_paineis_de_curva()
        self._montar_controles(velocidade)

        orcamento = f"{estado.orcamento:,}".replace(",", ".")
        self.fig.suptitle(
            f"Otimização de Rota (TSP) — n = {n} cidades (run {estado.run}), "
            f"orçamento de {orcamento} avaliações para cada algoritmo",
            fontsize=12.5, color=TINTA, y=0.965,
        )
        self.titulo = self.fig.text(0.5, 0.921, "", fontsize=10.5, color=TINTA_SUAVE,
                                    ha="center", va="center", animated=True)

        self.fig.canvas.mpl_connect("key_press_event", self._ao_teclar)
        self.fig.canvas.mpl_connect("close_event", self._ao_fechar)
        self.blit = _Blit(
            self.fig.canvas,
            self.fig,
            [*self.linhas_rota.values(), self.linha_conv_sa, self.linha_conv_ga,
             self.linha_aceit, self.linha_div_melhor, self.linha_div_media,
             self.titulo, *self.leituras.values()],
        )


    def _montar_paineis_de_rota(self, n: int) -> None:
        coords = self.estado.inst.coords
        self.linhas_rota: dict[str, Any] = {}
        for ax, chave, cor in (
            (self.ax_inicial, "inicial", TINTA_SUAVE),
            (self.ax_sa, "sa", ALGO_COLORS["sa"]),
            (self.ax_ga, "ga", ALGO_COLORS["ga"]),
        ):
            (linha,) = ax.plot([], [], "-", color=cor, linewidth=1.3, alpha=0.9,
                               zorder=1, animated=True)
            self.linhas_rota[chave] = linha
            ax.scatter(coords[:, 0], coords[:, 1], s=26, color=TINTA, zorder=2,
                       edgecolors=SUPERFICIE, linewidths=0.7)
            if n <= MAX_N_COM_ROTULOS:
                for indice, (x, y) in enumerate(coords):
                    ax.annotate(str(indice), (x, y), fontsize=6, color=TINTA_SUAVE,
                                xytext=(3, 3), textcoords="offset points")
            ax.set_xlim(-6, 106)
            ax.set_ylim(-6, 106)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=8)
        self.ax_inicial.set_title("Rota inicial", fontsize=10.5, pad=24)
        self.ax_sa.set_title(ALGO_LABELS["sa"], fontsize=10.5, pad=24,
                             color=ALGO_COLORS["sa"])
        self.ax_ga.set_title(ALGO_LABELS["ga"], fontsize=10.5, pad=24,
                             color=ALGO_COLORS["ga"])
        self.leituras = {
            chave: ax.text(0.5, 1.015, "", transform=ax.transAxes, ha="center",
                           va="bottom", fontsize=9, color=cor, animated=True,
                           clip_on=False)
            for ax, chave, cor in (
                (self.ax_inicial, "inicial", TINTA_SUAVE),
                (self.ax_sa, "sa", ALGO_COLORS["sa"]),
                (self.ax_ga, "ga", ALGO_COLORS["ga"]),
            )
        }

    def _montar_paineis_de_curva(self) -> None:
        cor_sa, cor_ga = ALGO_COLORS["sa"], ALGO_COLORS["ga"]

        (self.linha_conv_sa,) = self.ax_conv.plot([], [], "-", color=cor_sa, linewidth=2,
                                                  label=ALGO_LABELS["sa"], animated=True)
        (self.linha_conv_ga,) = self.ax_conv.plot([], [], "--", color=cor_ga, linewidth=2,
                                                  label=ALGO_LABELS["ga"], animated=True)
        self.ax_conv.set_xlim(0, self.estado.orcamento)
        self.ax_conv.set_title("Convergência (orçamento compartilhado)", fontsize=10)
        self.ax_conv.set_xlabel("Avaliações da função objetivo", fontsize=9)
        self.ax_conv.set_ylabel("Melhor distância", fontsize=9)
        self.ax_conv.legend(frameon=False, labelcolor=TINTA, fontsize=8)

        (self.linha_aceit,) = self.ax_aceit.plot([], [], "-", color=cor_sa, linewidth=1.4,
                                                 animated=True)
        self.ax_aceit.axhline(self.estado.sa_cfg.freeze_accept_rate, color=TINTA_SUAVE,
                              linestyle=":", linewidth=1.2)
        self.ax_aceit.text(
            0.97, 0.94,
            f"···  limiar de congelamento = {self.estado.sa_cfg.freeze_accept_rate:.2f}",
            transform=self.ax_aceit.transAxes, ha="right", va="top",
            color=TINTA_SUAVE, fontsize=8,
        )
        self.ax_aceit.set_ylim(0, 1.02)
        self.ax_aceit.set_title("SA: taxa de aceitação", fontsize=10, color=cor_sa)
        self.ax_aceit.set_xlabel("Iteração", fontsize=9)
        self.ax_aceit.set_ylabel("Aceitos / tentados no nível", fontsize=9)

        (self.linha_div_melhor,) = self.ax_div.plot([], [], "-", color=cor_ga, linewidth=2,
                                                    label="melhor custo global", animated=True)
        (self.linha_div_media,) = self.ax_div.plot([], [], "--", color=TINTA_SUAVE,
                                                   linewidth=1.6, animated=True,
                                                   label="custo médio da população")
        self.ax_div.set_title("AG: diversidade da população", fontsize=10, color=cor_ga)
        self.ax_div.set_xlabel("Geração", fontsize=9)
        self.ax_div.set_ylabel("Distância", fontsize=9)
        self.ax_div.legend(frameon=False, labelcolor=TINTA, fontsize=8)

    def _montar_controles(self, velocidade: float) -> None:
        eixo_slider = self.fig.add_axes([0.08, 0.055, 0.42, 0.03], facecolor="#eeeeea")
        self.slider = Slider(
            eixo_slider, "velocidade", LOG2_VEL_MIN, LOG2_VEL_MAX,
            valinit=float(np.log2(velocidade)), valstep=0.25, color=TINTA_SUAVE,
        )
        self.slider.label.set_color(TINTA_SUAVE)
        self.slider.label.set_fontsize(9)
        self.slider.on_changed(lambda _: self._atualizar_texto_do_slider())
        self._atualizar_texto_do_slider()

        self.botao_pausa = Button(self.fig.add_axes([0.56, 0.05, 0.10, 0.042]), "Pausar")
        self.botao_pausa.on_clicked(lambda _: self.alternar_pausa())
        self.botao_reiniciar = Button(self.fig.add_axes([0.68, 0.05, 0.10, 0.042]), "Reiniciar")
        self.botao_reiniciar.on_clicked(lambda _: self.reiniciar())
        for botao in (self.botao_pausa, self.botao_reiniciar):
            botao.label.set_fontsize(9)

        self.fig.text(
            0.985, 0.071, "espaço: pausa    +/−: velocidade",
            fontsize=8.5, color=TINTA_SUAVE, va="center", ha="right",
        )
        self.fig.text(
            0.985, 0.045, "r: reinicia    q: fecha",
            fontsize=8.5, color=TINTA_SUAVE, va="center", ha="right",
        )


    @property
    def velocidade(self) -> float:
        """Multiplicador de velocidade lido do slider (escala log2)."""
        return float(2.0 ** self.slider.val)

    def avaliacoes_por_quadro(self) -> float:
        """Quantas avaliações avançar neste quadro, dado o slider.

        A 1x, a execução inteira leva `SEGUNDOS_A_VELOCIDADE_1`; o slider multiplica.
        Velocidades abaixo de 1x funcionam naturalmente porque o alvo é um float: os
        geradores simplesmente não são puxados em todo quadro.
        """
        base = self.estado.orcamento / (self.fps * SEGUNDOS_A_VELOCIDADE_1)
        return base * self.velocidade

    def alternar_pausa(self) -> None:
        """Pausa ou retoma a animação."""
        self.pausado = not self.pausado
        self.botao_pausa.label.set_text("Continuar" if self.pausado else "Pausar")

    def reiniciar(self) -> None:
        """Recomeça a corrida do zero, com as mesmas sementes."""
        self.estado.reiniciar()
        self._rota_inicial_desenhada = False
        for linha in (self.linha_conv_sa, self.linha_conv_ga, self.linha_aceit,
                      self.linha_div_melhor, self.linha_div_media,
                      self.linhas_rota["sa"], self.linhas_rota["ga"]):
            linha.set_data([], [])

    def _atualizar_texto_do_slider(self) -> None:
        self.slider.valtext.set_text(f"{self.velocidade:.2f}x")
        self.slider.valtext.set_color(TINTA)
        self.slider.valtext.set_fontsize(9)

    def _ao_teclar(self, evento) -> None:
        if evento.key == " ":
            self.alternar_pausa()
        elif evento.key in ("+", "right", "up"):
            self.slider.set_val(min(LOG2_VEL_MAX, self.slider.val + 0.5))
        elif evento.key in ("-", "left", "down"):
            self.slider.set_val(max(LOG2_VEL_MIN, self.slider.val - 0.5))
        elif evento.key == "r":
            self.reiniciar()
        elif evento.key == "q":
            plt.close(self.fig)

    def _ao_fechar(self, _evento) -> None:
        self.fechado = True


    def _garantir_limites(self) -> None:
        """Fixa os limites dos eixos uma única vez, assim que o estado inicial existe.

        Limites fixos servem a dois propósitos: a animação não fica pulando de escala a
        cada quadro, e — o que importa para o desempenho — o fundo cacheado do blitting
        permanece válido, já que eixos e rótulos nunca mudam depois disso.

        Todos os tetos são cotas superiores exatas, calculadas do orçamento e da
        configuração; se um algoritmo parar antes, a linha simplesmente termina no meio
        do eixo, o que já mostra a parada antecipada.
        """
        estado = self.estado
        if self._limites_prontos or estado.snap_inicial is None or not estado.historico_ga:
            return

        teto = max(estado.snap_inicial.cost, estado.historico_ga[0][3]) * 1.05
        self.ax_conv.set_ylim(0, teto)
        self.ax_div.set_ylim(0, teto)
        self.ax_aceit.set_xlim(0, max(1, estado.orcamento - estado.sa_cfg.calib_samples))
        filhos_por_geracao = max(1, estado.ga_cfg.pop_size - estado.ga_cfg.elite)
        self.ax_div.set_xlim(0, estado.orcamento / filhos_por_geracao)
        self._limites_prontos = True
        self.fig.canvas.draw()

    def atualizar(self) -> None:
        """Atualiza os artistas dinâmicos e manda o quadro para a tela."""
        estado = self.estado
        coords = estado.inst.coords
        self._garantir_limites()

        if not self._rota_inicial_desenhada and estado.snap_inicial is not None:
            self._desenhar_rota("inicial", estado.snap_inicial.tour, coords)
            self.leituras["inicial"].set_text(
                f"Distância: {estado.snap_inicial.cost:.2f}"
            )
            self._rota_inicial_desenhada = True

        if estado.snap_sa is not None:
            snap = estado.snap_sa
            self._desenhar_rota("sa", snap.best_tour, coords)
            temperatura = "—" if snap.temperature is None else f"{snap.temperature:.2f}"
            self.leituras["sa"].set_text(
                f"T = {temperatura}  |  atual {snap.cost:.2f}  |  melhor {snap.best_cost:.2f}"
            )
        if estado.snap_ga is not None:
            snap = estado.snap_ga
            self._desenhar_rota("ga", snap.best_tour, coords)
            media = "—" if snap.mean_cost is None else f"{snap.mean_cost:.2f}"
            self.leituras["ga"].set_text(
                f"geração {snap.step}  |  média {media}  |  melhor {snap.best_cost:.2f}"
            )

        if estado.historico_sa:
            avals, passos, melhores, taxas = _colunas(estado.historico_sa)
            self.linha_conv_sa.set_data(avals, melhores)
            self.linha_aceit.set_data(passos, taxas)
        if estado.historico_ga:
            avals, geracoes, melhores, medias, _ = _colunas(estado.historico_ga)
            self.linha_conv_ga.set_data(avals, melhores)
            self.linha_div_melhor.set_data(geracoes, melhores)
            self.linha_div_media.set_data(geracoes, medias)

        self.titulo.set_text(self._texto_do_titulo())
        self.blit.atualizar()

    def _desenhar_rota(self, chave: str, tour: np.ndarray, coords: np.ndarray) -> None:
        ciclo = np.append(tour, tour[0])
        self.linhas_rota[chave].set_data(coords[ciclo, 0], coords[ciclo, 1])

    def _texto_do_titulo(self) -> str:
        """Só a parte que muda — o resto está no `suptitle`, dentro do fundo cacheado."""
        if self.estado.terminou:
            return "CONCLUÍDO"
        if self.pausado:
            return "PAUSADO"
        return f"{self.estado.progresso * 100:.0f}% do orçamento consumido"


def rodar_ao_vivo(
    inst: Instance,
    sa_cfg: SAConfig,
    ga_cfg: GAConfig,
    orcamento: int,
    master_seed: int,
    run: int,
    fps: int = 20,
    velocidade: float = 4.0,
) -> tuple[RunResult | None, RunResult | None]:
    """Abre a janela e conduz a corrida até o fim (ou até o usuário fechar).

    Devolve os dois `RunResult` — `None` se a janela for fechada antes do término.
    """
    estado = EstadoAoVivo(
        inst=inst, sa_cfg=sa_cfg, ga_cfg=ga_cfg, orcamento=orcamento,
        master_seed=master_seed, run=run,
    )
    painel = PainelAoVivo(estado, fps=fps, velocidade=velocidade)
    intervalo = 1.0 / fps
    plt.show(block=False)
    painel.fig.canvas.draw()

    while not painel.fechado:
        inicio = time.perf_counter()
        if not painel.pausado and not estado.terminou:
            estado.avancar(painel.avaliacoes_por_quadro())
        painel.atualizar()
        painel.fig.canvas.flush_events()
        folga = intervalo - (time.perf_counter() - inicio)
        if folga > 0:
            time.sleep(folga)

    return estado.resultado_sa, estado.resultado_ga
