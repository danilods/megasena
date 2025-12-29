# Mega-Sena Prediction System

Sistema completo em Python para analise e previsao de resultados da Mega-Sena usando metodos estatisticos (ARIMA), Facebook Prophet e Deep Learning (LSTM).

## Overview

Este projeto implementa um pipeline de previsao multi-modelo que:

1. **Carrega e valida** dados historicos da Mega-Sena (2.954+ sorteios)
2. **Engenharia de features** a partir dos numeros (261 features derivadas)
3. **Analise estatistica** (Chi-square, entropia, frequencia)
4. **Treina multiplos modelos**:
   - ARIMA para previsoes agregadas (soma, pares/impares)
   - Prophet para analise de tendencias
   - LSTM para padroes sequenciais
5. **Gera previsoes** com metodos ensemble
6. **Cria combinacoes otimizadas** usando sistemas de wheeling

## Estrutura do Projeto

```
megasena/
├── config/
│   └── settings.py              # Configuracoes globais
├── src/
│   ├── data/
│   │   ├── loader.py            # Carregamento e validacao
│   │   └── preprocessor.py      # Preprocessamento para ML
│   ├── features/
│   │   └── engineer.py          # Engenharia de features (261)
│   ├── models/
│   │   ├── arima_model.py       # ARIMA para soma/agregados
│   │   ├── prophet_model.py     # Prophet para tendencias
│   │   └── lstm_model.py        # Rede neural LSTM
│   ├── evaluation/
│   │   ├── metrics.py           # Hit rate, Top-K accuracy
│   │   └── wheeling.py          # Otimizacao de apostas
│   ├── utils/
│   │   ├── statistics.py        # Ferramentas estatisticas
│   │   └── visualization.py     # Graficos e visualizacoes
│   └── pipeline.py              # Orquestrador principal
├── results/                     # Resultados gerados
│   ├── ANALISE_MEGA_SENA_2500.md
│   ├── estrategia_2500.txt
│   ├── jogos_otimos.txt
│   └── predictions.csv
├── tests/                       # Testes unitarios
├── examples/                    # Exemplos de uso
├── main.py                      # CLI entry point
├── pyproject.toml               # Configuracao do projeto
├── uv.lock                      # Lock file (reproducibilidade)
├── requirements.txt             # Dependencias (legacy)
└── Mega-Sena.csv               # Dados historicos
```

## Instalacao

### Com uv (Recomendado)

[uv](https://github.com/astral-sh/uv) e um gerenciador de pacotes Python ultra-rapido.

```bash
# Instalar uv (se necessario)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Criar ambiente e instalar dependencias
uv venv
uv sync
```

### Com pip (Legacy)

```bash
pip install -r requirements.txt
```

## Uso

### Com uv

```bash
# Pipeline completo
uv run python main.py

# Previsao rapida (somente frequencia)
uv run python main.py --quick

# Analise estatistica
uv run python main.py --analyze

# Treinar modelos
uv run python main.py --train

# Gerar previsoes
uv run python main.py --predict

# Executar sem LSTM (mais rapido)
uv run python main.py --no-lstm

# Customizar quantidade
uv run python main.py --top-k 15 --max-bets 30
```

### Opcoes Disponiveis

```bash
uv run python main.py --help

Options:
  --analyze, -a         Somente analise estatistica
  --train, -t           Treinar todos os modelos
  --predict, -p         Gerar previsoes
  --quick, -q           Previsao rapida por frequencia
  --top-k TOP_K         Numero de previsoes (default: 20)
  --max-bets MAX_BETS   Maximo de combinacoes (default: 20)
  --no-arima            Pular modelo ARIMA
  --no-prophet          Pular modelo Prophet
  --no-lstm             Pular modelo LSTM
  --quiet               Suprimir mensagens
  --output OUTPUT       Arquivo de saida (default: predictions.csv)
```

## Features Engenheiradas (261 total)

| Categoria | Features | Descricao |
|-----------|----------|-----------|
| Agregadas | 6 | sum, mean, std, min, max, range |
| Paridade | 3 | even_count, odd_count, parity_ratio |
| Consecutivos | 2 | consecutive_pairs, has_consecutive |
| Quadrantes | 4 | q1_count a q4_count |
| Gap Analysis | 60 | Sorteios desde ultima aparicao |
| Frequencia Rolling | 180+ | Frequencia por janela temporal |

## Modelos

| Modelo | Proposito | Previsoes |
|--------|-----------|-----------|
| ARIMA | Series temporais | Faixas de soma, pares/impares |
| Prophet | Tendencias | Padroes de longo prazo |
| LSTM | Sequencias | Probabilidades por numero |

## Resultados Gerados

Apos execucao, os seguintes arquivos sao gerados em `results/`:

| Arquivo | Descricao |
|---------|-----------|
| `ANALISE_MEGA_SENA_2500.md` | Documento completo de analise |
| `estrategia_2500.txt` | Jogos otimizados para orcamento |
| `jogos_otimos.txt` | Jogos filtrados por parametros ideais |
| `jogos_top8.txt` | 28 combinacoes (8 numeros) |
| `jogos_top10.txt` | 210 combinacoes (10 numeros) |
| `jogos_top12.txt` | 924 combinacoes (12 numeros) |
| `predictions.csv` | Probabilidades por numero |
| `analysis_data.json` | Dados brutos da analise |

## Exemplo de Saida

```
MEGA-SENA - ANALISE ESTATISTICA
============================================================

Numeros QUENTES (ultimos 100 sorteios):
  15: 17x | 54: 14x | 04: 14x | 38: 14x | 27: 14x

Numeros ATRASADOS:
  43: 49 sorteios | 16: 33 sorteios | 41: 32 sorteios

Faixa ideal de soma: 132 a 234 (80% dos sorteios)
Distribuicao ideal: 2-4 pares (79% dos sorteios)

Top 10 previsoes: [15, 54, 04, 38, 27, 09, 37, 40, 51, 43]
```

## Metricas de Avaliacao

- **Hit Rate**: Numeros previstos vs sorteados
- **Top-K Accuracy**: Acertos dentro do top K
- **Precision@K / Recall@K**: Metricas de information retrieval

## Testes

```bash
# Executar todos os testes
uv run pytest

# Com cobertura
uv run pytest --cov=src
```

## Tecnologias

- **Python 3.10+**
- **Pandas / NumPy** - Manipulacao de dados
- **Scikit-learn** - Machine Learning
- **TensorFlow/Keras** - Deep Learning (LSTM)
- **Statsmodels / PMDarima** - ARIMA
- **Prophet** - Facebook Prophet
- **Matplotlib / Seaborn / Plotly** - Visualizacao

## Aviso Legal

Este sistema e apenas para fins educacionais e de pesquisa. Resultados de loteria sao fundamentalmente aleatorios e nenhum sistema de previsao pode garantir ganhos. A base matematica (Lei dos Grandes Numeros, testes Chi-quadrado) confirma que, ao longo do tempo, todos os numeros devem aparecer com probabilidade igual.

Esta ferramenta ajuda a:
1. **Auditar aleatoriedade** do mecanismo da loteria
2. **Filtrar combinacoes improvaveis** baseado em padroes estatisticos
3. **Otimizar estrategias** usando sistemas de wheeling

**Jogue com responsabilidade.**

## Licenca

MIT License
