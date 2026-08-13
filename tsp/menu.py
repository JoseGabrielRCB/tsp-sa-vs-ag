"""Menu interativo de terminal — a única forma de rodar o projeto.

Não há argumentos de linha de comando: `python main.py` abre este menu e tudo é
escolhido por aqui. As preferências valem para a sessão; sair e entrar de novo volta
aos valores padrão do protocolo.

A escolha do backend do matplotlib acontece em tempo de execução: o programa começa em
`Agg` (só salva PNG, sem janela) e troca para `TkAgg` apenas quando a visualização ao
vivo é aberta, voltando para `Agg` em seguida. Por isso `tsp.live` é importado dentro da
função, e não no topo do módulo.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib

from tsp.config import ALGO_LABELS, ALGO_SIGLAS, ExperimentConfig
from tsp.instance import make_instance
from tsp.perturbations import MIN_N
from tsp.plots import FIGURAS, FIGURAS_PADRAO, gerar_todos
from tsp.runner import ALGOS, load_runs, run_experiment, save_results, summary_table

LARGURA = 74
REGUA = "=" * LARGURA
TRACO = "-" * LARGURA

_SEG_POR_AVAL_SA = 1.1e-5
_SEG_POR_AVAL_GA_BASE = 3.5e-5
_SEG_POR_AVAL_GA_POR_CIDADE = 0.05e-5


@dataclass
class Preferencias:
    """Tudo que o menu deixa configurar, com os defaults do protocolo do trabalho."""

    sizes: tuple[int, ...] = (20, 30, 50, 75, 100)
    n_runs: int = 10
    evals_per_city: int = 2000
    master_seed: int = 20260805
    algos: tuple[str, ...] = ALGOS
    crossover: str = "ox"
    figuras: tuple[str, ...] = FIGURAS_PADRAO
    out_dir: str = "results"
    fresh: bool = False
    live_n: int = 30
    live_run: int = 0
    live_fps: int = 20
    live_speed: float = 4.0

    def validar(self) -> None:
        """Confere os valores e explica o que corrigir se algum estiver fora.

        Existe porque as constantes ficam no topo do `main.py` para serem editadas à
        mão: um erro ali deve virar uma mensagem clara, não um traceback no meio do
        experimento.
        """
        if not self.sizes:
            raise ValueError("TAMANHOS está vazio — informe ao menos um tamanho")
        pequenos = [n for n in self.sizes if n < MIN_N]
        if pequenos:
            raise ValueError(
                f"TAMANHOS tem valores abaixo de {MIN_N} cidades: {pequenos} — "
                "os operadores de perturbação precisam de pelo menos 4"
            )
        if self.n_runs < 1:
            raise ValueError("REPETICOES precisa ser pelo menos 1")
        if self.evals_per_city < 1:
            raise ValueError("AVALIACOES_POR_CIDADE precisa ser pelo menos 1")
        desconhecidos = [a for a in self.algos if a not in ALGOS]
        if desconhecidos or not self.algos:
            raise ValueError(
                f"ALGORITMOS deve ser uma combinação de {ALGOS}; recebi {self.algos}"
            )
        if self.crossover not in ("ox", "pmx"):
            raise ValueError(f"CRUZAMENTO deve ser 'ox' ou 'pmx'; recebi {self.crossover!r}")
        invalidos = [g for g in self.figuras if g not in FIGURAS]
        if invalidos or not self.figuras:
            raise ValueError(
                f"GRAFICOS tem nomes desconhecidos: {invalidos or 'lista vazia'} — "
                f"disponíveis: {', '.join(FIGURAS)}"
            )
        if self.live_n < MIN_N:
            raise ValueError(f"AO_VIVO_N precisa ter pelo menos {MIN_N} cidades")
        if self.live_run < 0:
            raise ValueError("AO_VIVO_REPETICAO não pode ser negativa")
        if self.live_fps < 1:
            raise ValueError("AO_VIVO_FPS precisa ser pelo menos 1")
        if self.live_speed <= 0:
            raise ValueError("AO_VIVO_VELOCIDADE precisa ser maior que zero")

    def experiment_config(self) -> ExperimentConfig:
        """Traduz as preferências para a configuração usada pelos algoritmos."""
        padrao = ExperimentConfig()
        return replace(
            padrao,
            sizes=self.sizes,
            n_runs=self.n_runs,
            evals_per_city=self.evals_per_city,
            master_seed=self.master_seed,
            out_dir=self.out_dir,
            ga=replace(padrao.ga, crossover=self.crossover),
        )

    @property
    def total_execucoes(self) -> int:
        """Quantas execuções o experimento vai disparar."""
        return len(self.sizes) * self.n_runs * len(self.algos)

    def duracao_estimada(self) -> float:
        """Estimativa grosseira, em segundos, do experimento completo."""
        total = 0.0
        for n in self.sizes:
            avaliacoes = self.evals_per_city * n * self.n_runs
            if "sa" in self.algos:
                total += avaliacoes * _SEG_POR_AVAL_SA
            if "ga" in self.algos:
                total += avaliacoes * (
                    _SEG_POR_AVAL_GA_BASE + _SEG_POR_AVAL_GA_POR_CIDADE * n
                )
        return total


def interpretar_tamanhos(texto: str) -> tuple[int, ...]:
    """Lê "20,30,50" como tamanhos de instância, validando o mínimo de 4 cidades."""
    valores = []
    for pedaco in texto.replace(";", ",").split(","):
        pedaco = pedaco.strip()
        if not pedaco:
            continue
        n = int(pedaco)
        if n < 4:
            raise ValueError("cada tamanho precisa ter pelo menos 4 cidades")
        valores.append(n)
    if not valores:
        raise ValueError("informe ao menos um tamanho")
    return tuple(sorted(set(valores)))


def interpretar_figuras(texto: str, disponiveis: tuple[str, ...]) -> tuple[str, ...]:
    """Lê "1,3,5" (posições) ou "todos" como seleção de gráficos."""
    texto = texto.strip().lower()
    if texto in ("todos", "todas", "tudo"):
        return tuple(disponiveis)
    escolhidas = []
    for pedaco in texto.replace(";", ",").split(","):
        pedaco = pedaco.strip()
        if not pedaco:
            continue
        indice = int(pedaco)
        if not 1 <= indice <= len(disponiveis):
            raise ValueError(f"número fora da lista: {indice}")
        nome = disponiveis[indice - 1]
        if nome not in escolhidas:
            escolhidas.append(nome)
    if not escolhidas:
        raise ValueError("selecione ao menos um gráfico")
    return tuple(escolhidas)


def interpretar_crossover(texto: str) -> str:
    """Valida o nome do operador de cruzamento do AG."""
    valor = texto.strip().lower()
    if valor not in ("ox", "pmx"):
        raise ValueError("use 'ox' ou 'pmx'")
    return valor


def formatar_duracao(segundos: float) -> str:
    """Duração legível: segundos abaixo de um minuto, minutos acima."""
    if segundos < 60:
        return f"{segundos:.0f} s"
    minutos = segundos / 60
    if minutos < 60:
        return f"{minutos:.0f} min"
    return f"{minutos / 60:.1f} h"


class _Saida(Exception):
    """Sinaliza que o usuário encerrou a entrada (Ctrl+C ou fim do stdin)."""


class _SemTerminal(_Saida):
    """Não há teclado para ler: o processo foi iniciado sem entrada interativa."""


def _tem_teclado() -> bool:
    """True quando dá para ler opções do teclado."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _ler(pergunta: str) -> str:
    """Lê uma linha, tratando Ctrl+C e fim de entrada como pedido de saída."""
    try:
        return input(pergunta).strip()
    except KeyboardInterrupt:
        print()
        raise _Saida
    except EOFError:
        print()
        if _tem_teclado():
            raise _Saida
        raise _SemTerminal
    except RuntimeError:
        raise _SemTerminal


def _perguntar(rotulo: str, atual, converter, dica: str = "") -> object:
    """Pergunta um valor, repetindo até ser válido. Enter vazio mantém o atual."""
    sufixo = f" ({dica})" if dica else ""
    while True:
        resposta = _ler(f"\n  {rotulo}{sufixo}\n  Enter mantém [{atual}] > ")
        if not resposta:
            return atual
        try:
            return converter(resposta)
        except (ValueError, TypeError) as erro:
            print(f"  !! Entrada inválida: {erro}\n")


def _pausar() -> None:
    _ler("\n  Pressione Enter para voltar ao menu... ")


def _titulo(texto: str) -> None:
    print(f"\n{REGUA}\n  {texto}\n{REGUA}")


def _linha_opcao(numero: str, rotulo: str, valor: str = "") -> None:
    if valor:
        pontos = "." * max(3, 44 - len(rotulo))
        print(f"  {numero:>2}  {rotulo} {pontos} {valor}")
    else:
        print(f"  {numero:>2}  {rotulo}")


def _resumo_configuracao(prefs: Preferencias) -> None:
    tamanhos = ", ".join(str(n) for n in prefs.sizes)
    algos = " e ".join(ALGO_SIGLAS[a] for a in prefs.algos)
    print(f"  tamanhos: {tamanhos}   |   repetições: {prefs.n_runs}   |   "
          f"orçamento: {prefs.evals_per_city} x n avaliações")
    print(f"  algoritmos: {algos}   |   cruzamento: {prefs.crossover.upper()}   |   "
          f"semente: {prefs.master_seed}")
    print(f"  saída: {prefs.out_dir}/   |   gráficos: {', '.join(prefs.figuras)}")
    print(f"  {prefs.total_execucoes} execuções, ~{formatar_duracao(prefs.duracao_estimada())} "
          f"estimados")


def _tela_experimento(prefs: Preferencias) -> None:
    _titulo("Rodar o experimento")
    _resumo_configuracao(prefs)
    print(TRACO)
    if _ler("  Confirmar? [S/n] > ").lower() in ("n", "nao", "não"):
        return

    exp = prefs.experiment_config()
    print()
    inicio = time.perf_counter()
    resultados = run_experiment(exp, algos=prefs.algos)
    decorrido = time.perf_counter() - inicio

    runs_csv, hist_csv = save_results(resultados, exp, fresh=prefs.fresh)
    print(f"\n  Concluído em {formatar_duracao(decorrido)}.")
    print(f"  Resultados brutos: {runs_csv}\n                     {hist_csv}")

    print("\n  Gráficos gerados:")
    gerar_todos(exp, figuras=prefs.figuras)

    print(f"\n{TRACO}\n  RESUMO (média ± desvio padrão sobre as repetições)\n{TRACO}")
    print(summary_table(load_runs(exp)))
    _pausar()


def _tela_ao_vivo(prefs: Preferencias) -> None:
    _titulo("Visualização ao vivo")
    exp = prefs.experiment_config()
    orcamento = exp.budget(prefs.live_n)
    print(f"  Instância: n = {prefs.live_n} cidades (repetição {prefs.live_run})")
    print(f"  Orçamento: {orcamento} avaliações para cada algoritmo")
    print(f"  Janela: {prefs.live_fps} quadros por segundo, velocidade inicial "
          f"{prefs.live_speed:g}x")
    print(TRACO)
    print("  Controles na janela: slider de velocidade, botões Pausar e Reiniciar.")
    print("  Pelo teclado: espaço pausa, +/- ajusta a velocidade, r reinicia, q fecha.")
    print("\n  Feche a janela para voltar ao menu.")

    matplotlib.use("TkAgg")
    try:
        from tsp.live import rodar_ao_vivo

        resultado_sa, resultado_ga = rodar_ao_vivo(
            inst=make_instance(prefs.live_n, prefs.live_run, exp.master_seed),
            sa_cfg=exp.sa, ga_cfg=exp.ga, orcamento=orcamento,
            master_seed=exp.master_seed, run=prefs.live_run,
            fps=prefs.live_fps, velocidade=prefs.live_speed,
        )
    finally:
        matplotlib.use("Agg")

    print()
    if resultado_sa is None or resultado_ga is None:
        print("  Janela fechada antes do término da execução.")
    else:
        for resultado in (resultado_sa, resultado_ga):
            print(f"  {ALGO_LABELS[resultado.algo]:<22} distância {resultado.best_cost:9.2f}"
                  f" | passos {resultado.steps:>7}"
                  f" | avaliações {resultado.evals_used}/{resultado.evals_budget}"
                  f" | parada por {resultado.stop_reason}")
    _pausar()


def _tela_replot(prefs: Preferencias) -> None:
    _titulo("Regerar os gráficos a partir do CSV")
    exp = prefs.experiment_config()
    if not load_runs(exp):
        print(f"  Nenhum resultado em {prefs.out_dir}/raw/runs.csv.")
        print("  Rode o experimento antes de gerar os gráficos.")
        _pausar()
        return
    print(f"  Gráficos: {', '.join(prefs.figuras)}\n")
    gerar_todos(exp, figuras=prefs.figuras)
    _pausar()


def _tela_resumo(prefs: Preferencias) -> None:
    _titulo("Resumo dos resultados salvos")
    exp = prefs.experiment_config()
    linhas = load_runs(exp)
    if not linhas:
        print(f"  Nenhum resultado em {prefs.out_dir}/raw/runs.csv.")
    else:
        print(f"  {len(linhas)} execuções em {Path(prefs.out_dir) / 'raw' / 'runs.csv'}\n")
        print(summary_table(linhas))
    _pausar()


def _tela_figuras(prefs: Preferencias) -> Preferencias:
    disponiveis = tuple(FIGURAS)
    _titulo("Gráficos a gerar")
    for indice, nome in enumerate(disponiveis, start=1):
        marca = "x" if nome in prefs.figuras else " "
        print(f"  [{marca}] {indice:>2}  {nome:<14} {FIGURAS[nome][0]}")
    print(TRACO)
    escolha = _perguntar(
        "Números separados por vírgula, ou 'todos'",
        ", ".join(prefs.figuras),
        lambda t: interpretar_figuras(t, disponiveis),
    )
    return replace(prefs, figuras=escolha if isinstance(escolha, tuple) else prefs.figuras)


def _tela_algoritmos(prefs: Preferencias) -> Preferencias:
    _titulo("Algoritmos a executar")
    opcoes = {"1": ALGOS, "2": ("sa",), "3": ("ga",)}
    _linha_opcao("1", "Os dois (comparação completa)")
    _linha_opcao("2", "Apenas o Recozimento Simulado")
    _linha_opcao("3", "Apenas o Algoritmo Genético")
    print(TRACO)
    escolha = _ler("  Escolha > ")
    return replace(prefs, algos=opcoes.get(escolha, prefs.algos))


def _tela_ao_vivo_config(prefs: Preferencias) -> Preferencias:
    _titulo("Parâmetros da visualização ao vivo")
    n = _perguntar("Tamanho da instância", prefs.live_n, int, "mínimo 4")
    run = _perguntar("Repetição (escolhe qual instância)", prefs.live_run, int)
    fps = _perguntar("Quadros por segundo", prefs.live_fps, int, "medido: ~45 ms/quadro")
    velocidade = _perguntar("Velocidade inicial", prefs.live_speed, float,
                            "1x leva ~2 min; ajustável na janela")
    return replace(prefs, live_n=max(4, int(n)), live_run=max(0, int(run)),
                   live_fps=max(1, int(fps)), live_speed=max(0.25, float(velocidade)))


def _tela_configuracoes(prefs: Preferencias, iniciais: Preferencias) -> Preferencias:
    """Edita as preferências. `iniciais` é o estado escrito no `main.py`, para onde a
    opção "Restaurar" volta."""
    while True:
        _titulo("Configurações")
        _linha_opcao("1", "Tamanhos das instâncias", ", ".join(str(n) for n in prefs.sizes))
        _linha_opcao("2", "Repetições por tamanho", str(prefs.n_runs))
        _linha_opcao("3", "Avaliações por cidade", str(prefs.evals_per_city))
        _linha_opcao("4", "Semente mestra", str(prefs.master_seed))
        _linha_opcao("5", "Algoritmos", " e ".join(ALGO_SIGLAS[a] for a in prefs.algos))
        _linha_opcao("6", "Cruzamento do AG", prefs.crossover.upper())
        _linha_opcao("7", "Gráficos a gerar", f"{len(prefs.figuras)} selecionados")
        _linha_opcao("8", "Diretório de saída", prefs.out_dir)
        _linha_opcao("9", "Ao salvar o CSV",
                     "sobrescrever" if prefs.fresh else "mesclar com o existente")
        _linha_opcao("10", "Visualização ao vivo",
                     f"n={prefs.live_n}, {prefs.live_fps} fps, {prefs.live_speed:g}x")
        _linha_opcao("11", "Restaurar os valores definidos no main.py")
        _linha_opcao("0", "Voltar")
        print(TRACO)

        escolha = _ler("  Escolha > ")
        if escolha in ("0", ""):
            return prefs
        if escolha == "1":
            prefs = replace(prefs, sizes=_perguntar(
                "Tamanhos separados por vírgula", ", ".join(str(n) for n in prefs.sizes),
                interpretar_tamanhos))
        elif escolha == "2":
            prefs = replace(prefs, n_runs=max(1, int(_perguntar(
                "Repetições por tamanho e algoritmo", prefs.n_runs, int))))
        elif escolha == "3":
            prefs = replace(prefs, evals_per_city=max(100, int(_perguntar(
                "Avaliações por cidade", prefs.evals_per_city, int,
                "define o orçamento: n x este valor"))))
        elif escolha == "4":
            prefs = replace(prefs, master_seed=int(_perguntar(
                "Semente mestra", prefs.master_seed, int,
                "muda as instâncias; ligue 'sobrescrever' na opção 9")))
        elif escolha == "5":
            prefs = _tela_algoritmos(prefs)
        elif escolha == "6":
            prefs = replace(prefs, crossover=_perguntar(
                "Cruzamento do AG", prefs.crossover, interpretar_crossover, "ox ou pmx"))
        elif escolha == "7":
            prefs = _tela_figuras(prefs)
        elif escolha == "8":
            prefs = replace(prefs, out_dir=_perguntar(
                "Diretório de saída", prefs.out_dir, str))
        elif escolha == "9":
            prefs = replace(prefs, fresh=not prefs.fresh)
        elif escolha == "10":
            prefs = _tela_ao_vivo_config(prefs)
        elif escolha == "11":
            prefs = iniciais
            print("\n  Valores do main.py restaurados.")
        else:
            print("\n  !! Opção inválida.")


def executar(iniciais: Preferencias | None = None) -> int:
    """Abre o menu e devolve o código de saída do programa.

    `iniciais` vem das constantes no topo do `main.py`. É também para onde a opção
    "Restaurar" volta — o que está escrito naquele arquivo é a referência da sessão.
    """
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass

    iniciais = iniciais or Preferencias()
    try:
        iniciais.validar()
    except ValueError as erro:
        print(f"\n{REGUA}\n  Configuração inválida no topo do main.py\n{REGUA}")
        print(f"  {erro}\n")
        return 1

    prefs = iniciais
    try:
        while True:
            _titulo("TSP: Recozimento Simulado vs Algoritmo Genético")
            _resumo_configuracao(prefs)
            print(TRACO)
            _linha_opcao("1", "Rodar o experimento")
            _linha_opcao("2", "Visualização ao vivo (abre uma janela)")
            _linha_opcao("3", "Regerar os gráficos a partir do CSV")
            _linha_opcao("4", "Ver o resumo dos resultados salvos")
            _linha_opcao("5", "Configurações")
            _linha_opcao("0", "Sair")
            print(TRACO)

            escolha = _ler("  Escolha > ")
            if escolha == "0":
                print("\n  Até mais.\n")
                return 0
            if escolha == "1":
                _tela_experimento(prefs)
            elif escolha == "2":
                _tela_ao_vivo(prefs)
            elif escolha == "3":
                _tela_replot(prefs)
            elif escolha == "4":
                _tela_resumo(prefs)
            elif escolha == "5":
                prefs = _tela_configuracoes(prefs, iniciais)
            else:
                print("\n  !! Opção inválida.")
    except _SemTerminal:
        print(f"\n{REGUA}\n  Sem entrada de teclado\n{REGUA}")
        print("  O menu lê as opções digitadas, e este processo não tem terminal.")
        print("  Isso acontece ao rodar pelo botão de 'Run' de um editor cujo painel de")
        print("  saída não aceita digitação, ou com pythonw.exe.")
        print("\n  Abra um terminal (PowerShell, CMD ou o terminal integrado do editor) na")
        print("  pasta do projeto e rode:\n")
        print(r"      .venv\Scripts\python.exe main.py")
        print()
        return 1
    except _Saida:
        print("\n  Encerrado.\n")
        return 0
