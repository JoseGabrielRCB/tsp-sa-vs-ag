# TSP: Recozimento Simulado vs. Algoritmo Genético

Comparação experimental entre duas metaheurísticas no Problema do Caixeiro Viajante
simétrico euclidiano 2D, para instâncias de 20, 30, 50, 75 e 100 cidades.
Ambos os algoritmos são implementados do zero — só `numpy` e `matplotlib`.

## Como rodar

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

.venv\Scripts\python.exe main.py
```

O `main.py` precisa ser passado *para* o Python. Digitar só `.\main.py` no PowerShell não
executa nada: o Windows entrega o arquivo ao programa associado à extensão `.py`, que
costuma ser um editor.

**Não há argumentos de linha de comando.** Os valores iniciais são constantes no topo do
`main.py` — edite lá antes de rodar:

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

Um valor fora do lugar vira uma mensagem apontando a constante, antes de qualquer coisa
rodar — não um traceback no meio do experimento.

`python main.py` então abre um menu no terminal, já com esses valores:

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

O cabeçalho mostra sempre o que vai rodar, com uma estimativa de duração antes de você
confirmar. Em **Configurações** dá para ajustar as mesmas coisas durante a execução, sem
mexer no arquivo; Enter vazio mantém o valor atual, e a opção 11 volta exatamente para o
que está escrito no `main.py`. As alterações feitas pelo menu valem só para a sessão.

Os hiperparâmetros mais finos dos algoritmos (alpha, L, taxa de aceitação alvo, tamanho
da população, torneio, elitismo, taxa de mutação, pesos das perturbações) ficam em
`tsp/config.py`, cada um com o motivo do default documentado.

Saídas: CSVs brutos em `results/raw/`, PNGs em `results/figures/`.

## Visualização ao vivo

A opção **2** do menu abre uma janela com seis painéis que se atualizam enquanto os dois
algoritmos rodam **ao mesmo tempo, disputando o mesmo relógio de avaliações** — o
protocolo de orçamento equivalente mostrado em vez de descrito.

| | | |
|---|---|---|
| Rota inicial | SA ao vivo (T, custo atual, melhor) | AG ao vivo (geração, média, melhor) |
| Convergência SA × AG (eixo x = avaliações) | SA: taxa de aceitação + limiar de congelamento | AG: melhor × custo médio da população |

Controles: slider de velocidade (0,25× a 64×), botões Pausar e Reiniciar; pelo teclado,
`espaço` pausa, `+`/`−` ajusta a velocidade, `r` reinicia e `q` fecha. Fechar a janela
devolve ao menu. Em Configurações → opção 10 ajustam-se tamanho da instância (default
30), repetição, quadros por segundo (20) e velocidade inicial (4×; a 1× a execução leva
cerca de 2 minutos).

Duas coisas que o modo ao vivo **não** faz, de propósito: não grava CSV (o ritmo é
artificial, então os tempos não são comparáveis com os do experimento e não podem
contaminar `results/raw/`) e não altera os algoritmos. `run_sa`/`run_ga` são cascas
finas sobre os geradores `iter_sa`/`iter_ga`; sem snapshots nenhum `yield` acontece e o
laço é exatamente o mesmo, com o mesmo consumo de números aleatórios. O experimento
completo foi re-executado após o refactor para conferir: as 100 execuções e as 30.005
linhas de histórico saíram idênticas.

Detalhe de desempenho, caso a janela precise de manutenção: um render completo desta
figura custa ~450 ms, o que daria 2 quadros por segundo. A janela usa **blitting** —
fundo estático fotografado uma vez, só os artistas dinâmicos redesenhados por cima — o
que exige limites de eixo fixos (`_garantir_limites`) e textos animados separados dos
estáticos. Depois disso o quadro custa ~45 ms. Os textos são a parte cara (~0,2 ms por
caractere): por isso o título fixo é `suptitle` comum e só o status é animado.

## O que torna a comparação honesta

**Orçamento de avaliações equivalente.** Para cada instância os dois algoritmos
recebem exatamente `2000 x n` avaliações da função objetivo, contadas por um mesmo
`EvalBudget` (`tsp/config.py`):

| | o que conta como uma avaliação | distribuição |
|---|---|---|
| SA | um movimento candidato (delta O(1)), inclusive os da calibração de T0 | iterações |
| AG | o custo completo de um indivíduo | `pop_size` inicial + 1 por filho ≈ `pop_size x gerações` |

A paridade é em *avaliações*, não em tempo de parede — a avaliação do SA é O(1) e a do
AG é O(n). Por isso o tempo também é medido, e há um gráfico específico de trade-off
qualidade x tempo.

**Mesmas instâncias e mesmas seeds.** Instância e fluxos aleatórios derivam de
`SeedSequence([master_seed, n, run])`, então os dois algoritmos resolvem exatamente os
mesmos problemas e qualquer execução futura reproduz tudo bit a bit. Nenhum
`random.seed()` global: todo sorteio passa por um `numpy.random.Generator` recebido por
parâmetro.

**Mesma vizinhança.** Inversão (2-opt), translação (or-opt) e troca vivem em
`tsp/perturbations.py` e são usadas tanto pelo SA quanto pela mutação do AG. A diferença
entre os algoritmos fica sendo só a estratégia de busca, não o operador.

## Decisões de projeto que valem comentário no relatório

**Calibração automática de T0.** Amostram-se 1000 movimentos aleatórios e coletam-se as
pioras `delta+`. Igualando `exp(-mean(delta+)/T)` à taxa de aceitação alvo `chi_0 = 0.8`
e isolando T:

```
T0 = -mean(delta+) / ln(chi_0)
```

Nada de chutar um número fixo. A taxa de aceitação observada no início fica um pouco
acima de 0.8 porque a maioria das pioras é menor que a média — visível no gráfico de
diagnóstico do SA.

**Parada por estagnação exige congelamento.** O SA só conta um nível como estagnado se a
taxa de aceitação daquele nível caiu abaixo de `freeze_accept_rate = 0.02`. Sem essa
condição o contador dispara ainda na fase quente — quando o SA aceita quase tudo, o
melhor global fica parado por construção — e a busca era abortada antes do resfriamento
agir. Medido em n=20/30: 2 a 3 execuções em 10 terminavam cedo demais e a distância
média piorava 17-21%. Com o limiar, o resultado é idêntico ao de gastar o orçamento
inteiro, gastando 15-26% menos avaliações em n grande.

**Deltas O(1) com wrap-around.** Cada perturbação calcula só as arestas afetadas. O
custo total só é recalculado do zero na revalidação (uma vez por nível de temperatura) e
no valor final reportado. A igualdade `delta == custo(nova) - custo(antiga)` foi
verificada por enumeração exaustiva de todo movimento legal em n <= 9, incluindo os casos
em que o movimento toca a aresta que fecha o ciclo — que é onde este tipo de código
costuma errar.

## Estrutura

```
tsp/
  config.py         dataclasses de hiperparâmetros + tipos compartilhados
  instance.py       instâncias determinísticas + matriz de distâncias
  tour.py           representação da rota e custo
  perturbations.py  inversão, translação, troca (proposta / delta O(1) / aplicação)
  sa.py             Recozimento Simulado
  ga.py             Algoritmo Genético (OX e PMX, torneio, elitismo)
  runner.py         protocolo, cronometragem e CSVs
  plots.py          gráficos estáticos, lidos do CSV
  live.py           visualização em tempo real (motor + painel + controles)
  menu.py           menu de terminal
main.py             constantes editáveis + ponto de entrada
results/raw/        runs.csv (uma linha por execução) e history.csv (convergência)
results/figures/    os PNGs
```

`results/` não está versionada: é gerada ao rodar o experimento pelo menu.

Os gráficos são sempre gerados a partir do CSV, nunca de estado em memória — dá para
replotar sem re-executar o experimento. O backend do matplotlib começa em `Agg`
(escolhido pelo `main.py` antes de importar `tsp`) e o menu troca para `TkAgg` só
enquanto a janela ao vivo está aberta.

## Figuras

Quais gráficos são gerados se escolhe em Configurações → "Gráficos a gerar" (opção 7).
O default são os quatro da entrega; lá dá para marcar qualquer subconjunto ou "todos".

**Gerados por default:**

| Nome no menu | Arquivo | Conteúdo |
|---|---|---|
| `distancia` | `fig01_distancia_media.png` | distância final média x n, banda de ±1 desvio |
| `tempo` | `fig02_tempo_medio.png` | tempo de execução médio x n |
| `passos` | `fig03_passos.png` | iterações (SA) e gerações (AG) até a parada, em painéis separados |
| `rotas` | `fig06_melhor_rota_n{n}.png` | melhor rota de cada algoritmo, mesma instância (5 arquivos) |

**Disponíveis sob demanda:**

| Nome no menu | Arquivo | Conteúdo |
|---|---|---|
| `convergencia` | `fig04_convergencia_n{n}.png` | melhor custo x avaliações, SA e AG sobrepostos (5 arquivos) |
| `boxplot` | `fig05_boxplot_distancias.png` | variabilidade das distâncias finais |
| `gap` | `fig07_gap_relativo.png` | gap % em relação à melhor solução conhecida, por instância |
| `tradeoff` | `fig08_tradeoff_qualidade_tempo.png` | distância x tempo, um ponto por execução |
| `diag-sa` | `fig09_diagnostico_sa.png` | temperatura e taxa de aceitação ao longo das iterações |
| `diag-ga` | `fig10_diagnostico_ga.png` | melhor custo x custo médio da população (diversidade) |

Como os gráficos saem do CSV, trocar a seleção não exige re-executar nada: marque os
que quiser na opção 7 e use a opção 3 do menu — leva alguns segundos.

Paleta fixa em todos: laranja `#D55E00` para o SA, azul `#0072B2` para o AG — par
verificado para daltonismo, com marcador e traço também distintos para que a identidade
não dependa só da cor.
