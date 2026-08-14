# TSP: Recozimento Simulado vs. Algoritmo Genético

Comparação experimental entre duas metaheurísticas no Problema do Caixeiro Viajante
simétrico euclidiano 2D. Ambas implementadas do zero — só `numpy` e `matplotlib`.

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Como rodar

```powershell
.venv\Scripts\python.exe main.py
```

Não há argumentos de linha de comando: tudo é escolhido pelo menu.

O arquivo precisa ser passado *para* o Python. Digitar só `.\main.py` no PowerShell não
executa nada — o Windows entrega o arquivo ao programa associado à extensão `.py`, que
costuma ser um editor. O menu também precisa de um terminal de verdade (PowerShell, CMD
ou o terminal integrado do editor); pelo botão "Run" de alguns editores não há como
digitar as opções.

## O menu

```
==========================================================================
  TSP: Recozimento Simulado vs Algoritmo Genético
==========================================================================
  tamanhos: 20, 30, 50, 75, 100   |   repetições: 10   |   orçamento: 2000 x n avaliações
  algoritmos: SA e AG   |   cruzamento: OX   |   semente: 20260805
  saída: results/   |   gráficos: distancia, tempo, passos, rotas
  100 execuções, ~7 min estimados
--------------------------------------------------------------------------
   1  Rodar o experimento
   2  Visualização ao vivo (abre uma janela)
   3  Regerar os gráficos a partir do CSV
   4  Ver o resumo dos resultados salvos
   5  Configurações
   0  Sair
--------------------------------------------------------------------------
```

O cabeçalho mostra sempre o que vai rodar e uma estimativa de duração antes de você
confirmar.

| Opção | O que faz |
|---|---|
| **1** | Executa todas as combinações (tamanho × repetição × algoritmo), grava os CSVs, gera os gráficos e imprime a tabela resumo. Pede confirmação antes. |
| **2** | Abre a janela com SA e AG rodando ao mesmo tempo. Não grava nada. |
| **3** | Regera os gráficos a partir do CSV, sem repetir o experimento — leva segundos. |
| **4** | Mostra a tabela de média ± desvio das execuções já salvas. |
| **5** | Ajusta os parâmetros só para esta sessão. |

### Configurações (opção 5)

Enter vazio mantém o valor atual. A opção **11** volta exatamente para o que está escrito
no `main.py`.

| | |
|---|---|
| 1 | Tamanhos das instâncias (mínimo 4 cidades) |
| 2 | Repetições por tamanho e algoritmo |
| 3 | Avaliações por cidade (orçamento = n × este valor) |
| 4 | Semente mestra — trocar gera instâncias novas |
| 5 | Algoritmos: os dois, só o SA ou só o AG |
| 6 | Cruzamento do AG: `ox` ou `pmx` |
| 7 | Quais gráficos gerar |
| 8 | Diretório de saída |
| 9 | Sobrescrever o CSV ou mesclar com o existente |
| 10 | Parâmetros da janela ao vivo |
| 11 | Restaurar os valores do `main.py` |

Os hiperparâmetros dos algoritmos (alpha, taxa de aceitação alvo, tamanho da população,
torneio, elitismo, taxa de mutação) ficam em `tsp/config.py`.

### Visualização ao vivo (opção 2)

Seis painéis que se atualizam enquanto os dois algoritmos rodam ao mesmo tempo,
disputando o mesmo relógio de avaliações.

Controles na janela: slider de velocidade (0,25× a 64×) e botões Pausar e Reiniciar.
Pelo teclado: `espaço` pausa, `+`/`−` ajusta a velocidade, `r` reinicia, `q` fecha.
Fechar a janela devolve ao menu.

## Constantes no topo do main.py

Editáveis antes de rodar; são os valores com que o menu abre. Um valor fora do lugar vira
uma mensagem apontando a constante, não um traceback no meio do experimento.

```python
TAMANHOS = (20, 30, 50, 75, 100)   # cidades por instância (mínimo 4)
REPETICOES = 10                    # repetições por tamanho e algoritmo
AVALIACOES_POR_CIDADE = 2000       # orçamento por execução = este valor x n
ALGORITMOS = ("sa", "ga")          # ou só um deles
CRUZAMENTO = "ox"                  # "ox" ou "pmx"
SEMENTE = 20260805                 # muda as instâncias
DIRETORIO_SAIDA = "results"
GRAFICOS = ("distancia", "tempo", "passos", "rotas")
SOBRESCREVER_CSV = False           # False mescla com o CSV existente

AO_VIVO_N = 30                     # instância mostrada na janela
AO_VIVO_REPETICAO = 0
AO_VIVO_FPS = 20
AO_VIVO_VELOCIDADE = 4.0
```

## Gráficos

Nomes aceitos em `GRAFICOS` e na opção 7. Os quatro primeiros são o default.

| Nome | Conteúdo |
|---|---|
| `distancia` | distância final média × n, com banda de ±1 desvio |
| `tempo` | tempo de execução médio × n |
| `passos` | iterações (SA) e gerações (AG) até a parada |
| `rotas` | melhor rota de cada algoritmo, mesma instância |
| `convergencia` | melhor custo × avaliações, SA e AG sobrepostos |
| `boxplot` | variabilidade das distâncias finais |
| `gap` | gap % em relação à melhor solução conhecida |
| `tradeoff` | distância × tempo, um ponto por execução |
| `diag-sa` | temperatura e taxa de aceitação do SA |
| `diag-ga` | melhor custo × custo médio da população |

## Saídas

```
results/raw/       runs.csv (uma linha por execução) e history.csv (convergência)
results/figures/   os PNGs
```

Os gráficos saem sempre do CSV, nunca de estado em memória — por isso a opção 3 replota
sem re-executar nada. A pasta `results/` não é versionada: é gerada ao rodar.
