
from __future__ import annotations

import sys

try:
    import matplotlib

    matplotlib.use("Agg")

    from tsp.menu import Preferencias, executar
except ModuleNotFoundError as ausente:
    print(f"Falta a biblioteca '{ausente.name}' neste Python ({sys.executable}).")
    print("O projeto roda dentro do ambiente virtual da pasta. Use:\n")
    print(r"    .venv\Scripts\python.exe main.py")
    print("\nSe a pasta .venv não existir, crie-a antes:\n")
    print(r"    python -m venv .venv")
    print(r"    .venv\Scripts\python.exe -m pip install -r requirements.txt")
    raise SystemExit(1)


TAMANHOS = (20, 30, 50, 75, 100)

REPETICOES = 10

AVALIACOES_POR_CIDADE = 2000

ALGORITMOS = ("sa", "ga")

CRUZAMENTO = "ox"

SEMENTE = 20260805

DIRETORIO_SAIDA = "results"

GRAFICOS = ("distancia", "tempo", "passos", "rotas")

SOBRESCREVER_CSV = False


AO_VIVO_N = 30

AO_VIVO_REPETICAO = 0

AO_VIVO_FPS = 20

AO_VIVO_VELOCIDADE = 4.0



def preferencias_iniciais() -> Preferencias:
    return Preferencias(
        sizes=tuple(TAMANHOS),
        n_runs=REPETICOES,
        evals_per_city=AVALIACOES_POR_CIDADE,
        master_seed=SEMENTE,
        algos=tuple(ALGORITMOS),
        crossover=CRUZAMENTO,
        figuras=tuple(GRAFICOS),
        out_dir=DIRETORIO_SAIDA,
        fresh=SOBRESCREVER_CSV,
        live_n=AO_VIVO_N,
        live_run=AO_VIVO_REPETICAO,
        live_fps=AO_VIVO_FPS,
        live_speed=AO_VIVO_VELOCIDADE,
    )


def main() -> int:
    if len(sys.argv) > 1:
        print("Este programa não usa argumentos de linha de comando "
              f"(recebi: {' '.join(sys.argv[1:])}).")
        print("Edite as constantes no topo de main.py ou use o menu.\n")
    return executar(preferencias_iniciais())


if __name__ == "__main__":
    raise SystemExit(main())
