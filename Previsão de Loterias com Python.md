# **Análise Estocástica e Modelagem Preditiva da Mega-Sena: Uma Abordagem Computacional via ARIMA, Prophet e Deep Learning**

## **1\. Introdução e Fundamentação Teórica**

A interseção entre a matemática probabilística, a estatística computacional e a inteligência artificial oferece um campo fértil para a análise de sistemas estocásticos complexos, como as loterias governamentais. O presente relatório técnico detalha o desenvolvimento de um projeto em Python voltado para a análise e previsão de resultados da Mega-Sena, utilizando o arquivo de dados históricos 'Mega-Sena.csv'. A abordagem transcende a simples especulação, ancorando-se em princípios acadêmicos rigorosos, como a Lei dos Grandes Números, a Teoria do Caos e a Hipótese do Mercado Eficiente aplicada a jogos de azar. O objetivo central não é apenas a tentativa de prever números, mas a construção de um arcabouço analítico capaz de identificar anomalias estatísticas, padrões temporais latentes e otimizar estratégias de apostas através de modelos de séries temporais (ARIMA e Prophet) e redes neurais profundas (Deep Learning/LSTM).

### **1.1 A Natureza Probabilística da Mega-Sena**

A Mega-Sena consiste, fundamentalmente, em um problema de análise combinatória onde seis números únicos são extraídos de um universo de sessenta elementos, denotado pelo conjunto $U \= \\{1, 2,..., 60\\}$. A probabilidade teórica de qualquer combinação específica de seis números ser sorteada é dada pela fórmula da combinação simples, uma vez que a ordem de extração das bolas não altera o resultado final do jogo, embora possa ser relevante para a análise da entropia do sorteio físico.1

Matematicamente, o número total de combinações possíveis é calculado como:

$$C\_{n,k} \= \\binom{n}{k} \= \\frac{n\!}{k\!(n-k)\!}$$  
Substituindo $n=60$ e $k=6$, obtemos:

$$C\_{60,6} \= \\frac{60\!}{6\!(54)\!} \= 50.063.860$$  
Este valor, superior a cinquenta milhões, define o espaço amostral do evento. Sob a premissa de um sorteio justo, onde cada bola possui massa, volume e propriedades aerodinâmicas idênticas, e o mecanismo de sorteio (o globo) opera de forma imparcial, cada combinação possui uma probabilidade de ocorrência exata de $1 / 50.063.860$.2 No entanto, a aplicação de modelos preditivos baseia-se na hipótese de que sistemas físicos reais podem apresentar imperfeições microscópicas ou tendências de curto prazo que desviam marginalmente da distribuição uniforme ideal, ou que a aleatoriedade gerada pode conter padrões pseudo-aleatórios detectáveis por algoritmos de alta complexidade.3

### **1.2 A Lei dos Grandes Números e a Convergência Estatística**

Um pilar central para a validação de qualquer modelo preditivo neste contexto é a Lei dos Grandes Números (LGN). A LGN postula que, à medida que o número de experimentos (neste caso, concursos da Mega-Sena) tende ao infinito, a frequência relativa de cada número sorteado deve convergir para a probabilidade teórica esperada.5

Para a Mega-Sena, a probabilidade esperada de qualquer dezena $d$ ser sorteada em um concurso específico é:

$$P(d) \= \\frac{6}{60} \= 0,10 \\text{ ou } 10\\%$$  
Ao analisarmos o arquivo 'Mega-Sena.csv', que contém o histórico de sorteios desde 1996 1, observamos flutuações naturais de curto prazo onde certos números aparecem com frequência superior ("números quentes") ou inferior ("números frios") à média. A Lei Fraca dos Grandes Números sugere que essas discrepâncias são ruídos estatísticos que se anulam no longo prazo.7 Contudo, a persistência de desvios significativos em grandes amostras pode indicar vieses sistêmicos no mecanismo de sorteio, os quais modelos como ARIMA e LSTM tentam capturar. É crucial, entretanto, distinguir entre a variação natural de um processo estocástico e um padrão determinístico explorável. O projeto em Python deve, portanto, iniciar com testes estatísticos rigorosos para quantificar o nível de entropia e a aderência à distribuição uniforme antes de aplicar modelagem preditiva avançada.8

### **1.3 Entropia de Shannon e Teoria da Informação**

A previsibilidade de um sistema está intrinsecamente ligada à sua entropia. Na Teoria da Informação, a Entropia de Shannon mede o grau de incerteza ou "surpresa" associado a uma variável aleatória. Para um sorteio de loteria perfeitamente justo, a entropia deve ser maximizada, indicando que o conhecimento dos sorteios passados não fornece nenhuma informação (ganho de informação nulo) sobre o próximo sorteio.10

A fórmula da entropia $H$ para uma variável aleatória discreta $X$ é:

$$H(X) \= \-\\sum\_{i=1}^{n} P(x\_i) \\log\_2 P(x\_i)$$  
No desenvolvimento do script Python, calculamos a entropia das sequências de sorteios para monitorar a "saúde" da aleatoriedade. Períodos de baixa entropia local poderiam, teoricamente, sinalizar janelas de oportunidade onde o sistema se comporta de maneira menos caótica, permitindo que modelos como o Prophet identifiquem tendências temporais ou sazonalidades anômalas.11 A integração dessas métricas de teoria da informação serve como uma camada de engenharia de atributos (feature engineering) sofisticada, alimentando as redes neurais com dados sobre a "qualidade" da aleatoriedade em janelas deslizantes de tempo.

## **2\. Análise Exploratória de Dados (EDA) e Validação Estatística**

Antes de submeter os dados aos algoritmos de aprendizado de máquina, é imperativo realizar uma análise exploratória profunda e uma validação estatística do conjunto de dados 'Mega-Sena.csv'. Esta etapa visa garantir a integridade dos dados, compreender as distribuições subjacentes e refutar ou confirmar hipóteses estatísticas sobre a aleatoriedade dos sorteios.

### **2.1 Estrutura e Limpeza do Dataset**

O arquivo 'Mega-Sena.csv' contém registros históricos vitais que vão além dos simples números sorteados. As colunas típicas incluem Concurso, Data do Sorteio, as dezenas Bola1 a Bola6, Arrecadação Total, Ganhadores, e Rateio.1 A análise preliminar revela que os dados exigem pré-processamento meticuloso:

1. **Conversão Temporal:** A coluna Data do Sorteio deve ser convertida para objetos datetime do Python para permitir a indexação temporal, essencial para os modelos ARIMA e Prophet.13 A ordenação cronológica estrita é fundamental para evitar o vazamento de dados (data leakage), onde o modelo aprenderia inadvertidamente com informações do futuro.15  
2. **Normalização Numérica:** As dezenas sorteadas (1-60) são variáveis categóricas ordinais. Para redes neurais como LSTM, é comum normalizar esses valores para o intervalo $$ utilizando MinMaxScaler do Scikit-Learn, facilitando a convergência do algoritmo de otimização (como o Gradient Descent).16  
3. **Tratamento de Valores Ausentes e Outliers:** Embora raros em dados oficiais, inconsistências podem ocorrer (ex: concursos cancelados ou dados de rateio nulos). O script deve incluir rotinas de validação para garantir que cada registro de sorteio contenha exatamente seis números únicos dentro do intervalo permitido.18

### **2.2 Teste de Aderência do Qui-Quadrado (Chi-Square)**

Para validar a hipótese de que o sorteio é justo, aplicamos o teste de bondade de ajuste do Qui-Quadrado ($\\chi^2$). Este teste compara as frequências observadas de cada número com as frequências esperadas sob uma distribuição uniforme.8

A estatística $\\chi^2$ é calculada como:

$$\\chi^2 \= \\sum\_{i=1}^{k} \\frac{(O\_i \- E\_i)^2}{E\_i}$$  
Onde $O\_i$ é a frequência observada do número $i$, e $E\_i$ é a frequência esperada ($N\_{sorteios} \\times 6 / 60$). No Python, utilizamos a biblioteca scipy.stats para executar este teste.21

**Interpretação dos Resultados:**

* Se o valor-p (p-value) resultante for maior que o nível de significância $\\alpha$ (usualmente 0.05), falhamos em rejeitar a hipótese nula ($H\_0$), concluindo que os dados são consistentes com uma distribuição uniforme.9  
* Se o valor-p for menor que 0.05, há evidências estatísticas de viés. Em análises históricas de loterias longas, é comum que a distribuição se aproxime da uniformidade, mas flutuações de curto prazo podem ser exploradas por modelos de aprendizado de máquina que buscam padrões locais.22

### **2.3 Análise de Atrasos (Gaps) e Ciclos**

Uma dimensão crítica na análise lotérica é o estudo dos "atrasos" ou "gaps" — o número de concursos decorridos desde a última aparição de uma determinada dezena. A "teoria dos números atrasados" é popular entre apostadores, sugerindo que números que não saem há muito tempo têm maior probabilidade de sair em breve. Estatisticamente, isso é uma falácia (a menos que o mecanismo tenha memória), mas os atrasos fornecem características temporais valiosas para modelos de séries temporais.2

O script Python deve calcular, para cada sorteio, o vetor de atrasos atuais de todas as 60 dezenas. Esta transformação converte o problema de "quais números sairão" para "qual o comportamento da série temporal de atrasos". Visualizações como Heatmaps (mapas de calor) são geradas utilizando seaborn ou matplotlib para identificar visualmente clusters de atrasos ou periodicidades anômalas.26

## **3\. Engenharia de Atributos (Feature Engineering)**

A engenharia de atributos é o processo de transformar dados brutos em variáveis que melhor representam o problema subjacente para os modelos preditivos. No contexto da Mega-Sena, utilizar apenas os números brutos (1 a 60\) é ineficiente devido à alta cardinalidade e à falta de ordem intrínseca significativa. Portanto, desenvolvemos "meta-features" que capturam propriedades estatísticas dos sorteios.2

### **3.1 Agregação e Estatísticas Descritivas por Sorteio**

Para cada concurso, calculamos um conjunto de descritores estatísticos que servem como variáveis exógenas ou alvos secundários para os modelos:

| Atributo | Descrição | Justificativa Teórica |
| :---- | :---- | :---- |
| **Soma** | O somatório das seis dezenas sorteadas ($S \= \\sum b\_i$). | A soma tende a seguir uma distribuição normal (Curva de Gauss), tornando-se mais previsível que os números individuais.2 |
| **Média e Desvio Padrão** | Média e dispersão dos números do sorteio. | Indicadores da "espalhamento" dos números no volante.29 |
| **Paridade (Par/Ímpar)** | Contagem de números pares e ímpares. | A análise combinatória mostra que distribuições balanceadas (3 pares, 3 ímpares) são mais prováveis que extremos (6 pares ou 6 ímpares).31 |
| **Consecutividade** | Presença de sequências numéricas (ex: 23, 24). | Surpreendentemente, pares consecutivos ocorrem em aproximadamente 50% dos sorteios, um padrão frequentemente ignorado por humanos.24 |

### **3.2 Quadrantes e Distribuição Espacial**

O volante da Mega-Sena pode ser dividido em quadrantes geométricos. Analisar a distribuição dos números sorteados nesses quadrantes permite verificar vieses espaciais. O script Python deve mapear cada número para seu respectivo quadrante e calcular a densidade de sorteios por região. Isso alimenta os modelos com informações sobre se os sorteios tendem a se concentrar em áreas específicas ou se espalham uniformemente.24

### **3.3 Janelamento Temporal (Windowing) para LSTM**

Para modelos de Deep Learning como LSTM, os dados devem ser estruturados em sequências de janelas deslizantes. Se definirmos uma janela de tamanho $W=10$, o modelo utilizará os dados dos 10 concursos anteriores ($t-10$ a $t-1$) para prever o concurso $t$. Essa estrutura permite que a rede neural aprenda dependências temporais de curto e longo prazo nos padrões de sorteio.17 A criação dessas sequências no Python é feita geralmente através de funções personalizadas ou geradores de séries temporais do Keras (TimeseriesGenerator).37

## **4\. Modelagem Preditiva: Metodologias e Implementação**

O projeto propõe uma abordagem de "ensemble" ou comparação entre três famílias distintas de algoritmos: modelos estatísticos lineares (ARIMA), modelos aditivos decomponíveis (Prophet) e redes neurais recorrentes não-lineares (LSTM).

### **4.1 ARIMA (AutoRegressive Integrated Moving Average)**

O modelo ARIMA é uma ferramenta clássica para previsão de séries temporais estacionárias. Ele combina três componentes: Auto-Regressivo (AR), Integrado (I) e Média Móvel (MA).39

* **Aplicabilidade na Mega-Sena:** Como prever cada uma das 6 bolas individualmente via ARIMA é estatisticamente frágil devido à independência dos eventos, aplicamos o ARIMA para prever **features agregadas**, como a *Soma das Dezenas* ou a *Quantidade de Números Pares*. A soma, por exemplo, exibe propriedades de reversão à média que o ARIMA modela bem.  
* **Funcionamento:**  
  1. **Verificação de Estacionariedade:** Utilizamos o Teste Dickey-Fuller Aumentado (ADF) para verificar se a série da soma é estacionária. Se não for ($p \> 0.05$), aplicamos diferenciação ($d$) até obter estacionariedade.42  
  2. **Identificação de Parâmetros (p, d, q):** Analisamos os gráficos de Autocorrelação (ACF) e Autocorrelação Parcial (PACF) para estimar os termos $p$ (lag autoregressivo) e $q$ (janela de média móvel).40  
  3. **Ajuste do Modelo:** Utilizamos a biblioteca statsmodels ou pmdarima (Auto-ARIMA) para ajustar o modelo aos dados de treino e projetar a soma provável do próximo concurso.44

O script Python para ARIMA focará em prever a *faixa* em que a soma das dezenas cairá, permitindo filtrar combinações improváveis que não se ajustem a essa previsão.

### **4.2 Facebook Prophet**

O Prophet é um modelo desenvolvido pelo Facebook focado em séries temporais com fortes efeitos sazonais e tendência, robusto a dados faltantes e mudanças de tendência.14

* **Aplicabilidade na Mega-Sena:** Embora a loteria não tenha "sazonalidade de vendas" como o varejo, o Prophet pode ser utilizado para testar a hipótese de **ciclos mecânicos** ou anomalias temporais de longo prazo. Por exemplo, investigar se certas dezenas aparecem mais frequentemente em épocas específicas do ano ou se há uma tendência de crescimento em métricas derivadas.  
* Funcionamento: O Prophet decompõe a série temporal em tendência, sazonalidade (semanal, anual) e feriados.

  $$y(t) \= g(t) \+ s(t) \+ h(t) \+ \\epsilon\_t$$  
* **Implementação em Python:**  
  1. O dataframe deve ser preparado com colunas estritas ds (data) e y (valor a prever, ex: frequência de um número específico ou soma).48  
  2. Instanciamos m \= Prophet() e ajustamos com m.fit(df).  
  3. Geramos um dataframe futuro com make\_future\_dataframe e realizamos a predição.  
  4. Analisamos os componentes de tendência e sazonalidade para ver se o modelo detectou algum padrão cíclico significativo ou se a série é dominada pelo ruído ($\\epsilon\_t$).13

### **4.3 Deep Learning: LSTM (Long Short-Term Memory)**

As redes LSTM são uma evolução das Redes Neurais Recorrentes (RNNs) capazes de aprender dependências de longo prazo, mitigando o problema do desvanecimento do gradiente (vanishing gradient). Elas são teoricamente as mais aptas para capturar padrões não-lineares complexos em sequências.51

* **Arquitetura da Rede:**  
  * **Camada de Entrada:** Recebe tensores 3D com formato (amostras, timesteps, features). timesteps define quantos concursos passados a rede "olha" para trás (ex: 10). features pode ser 6 (as dezenas) ou mais (incluindo dados derivados).17  
  * **Camadas LSTM:** Uma ou mais camadas LSTM empilhadas, com unidades de memória (ex: 50-200 neurônios). O uso de Dropout é crucial para evitar overfitting, dado que o ruído aleatório da loteria facilita a memorização espúria dos dados de treino.36  
  * **Camada Densa de Saída:** Uma camada Dense com 60 neurônios (um para cada número possível) e ativação softmax ou sigmoid. Isso transforma a saída da rede em uma distribuição de probabilidade sobre os 60 números, indicando quais têm maior "potencial" para o próximo sorteio.2  
* **Estratégia de Treinamento:**  
  * **Função de Perda (Loss):** binary\_crossentropy é adequada se tratarmos a previsão como um problema de classificação multi-rótulo (quais números estarão presentes no sorteio), ou sparse\_categorical\_crossentropy.55  
  * **Otimizador:** Adam é a escolha padrão pela eficiência adaptativa.53  
  * **Epochs e Batch Size:** Ajustados experimentalmente, com EarlyStopping para interromper o treinamento quando a validação parar de melhorar.

## **5\. Implementação Computacional e Script Python**

O projeto é estruturado como um pipeline modular em Python, garantindo reprodutibilidade e clareza. Abaixo, descrevemos a lógica de implementação integrando as bibliotecas mencionadas.

### **5.1 Bibliotecas e Dependências**

O ambiente deve ser configurado com:

* pandas e numpy para manipulação vetorial e estruturação de dados.  
* matplotlib e seaborn para visualização gráfica (heatmaps, histogramas).  
* scikit-learn para pré-processamento (MinMaxScaler) e métricas de avaliação.  
* statsmodels e pmdarima para a modelagem ARIMA.  
* prophet (anteriormente fbprophet) para a modelagem decomponível.  
* tensorflow e keras para a construção da rede neural LSTM.17

### **5.2 Snippet de Código Conceitual**

Python

import pandas as pd  
import numpy as np  
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense, Dropout  
from statsmodels.tsa.arima.model import ARIMA  
from prophet import Prophet

\# 1\. Carregamento e Pré-processamento  
def load\_data(filepath):  
    df \= pd.read\_csv(filepath, parse\_dates=)  
    df \= df.sort\_values('Data do Sorteio')  
    \# Extrair apenas as colunas das bolas  
    balls \= df\]  
    return df, balls

\# 2\. Engenharia de Features (Exemplo: Soma)  
def create\_features(df):  
    df \= df\].sum(axis=1)  
    return df

\# 3\. Modelagem ARIMA (Para a variável 'Soma')  
def train\_arima(series, order=(5,1,0)):  
    model \= ARIMA(series, order=order)  
    model\_fit \= model.fit()  
    forecast \= model\_fit.forecast(steps=1)  
    return forecast

\# 4\. Modelagem Prophet (Para tendências de um número específico)  
def train\_prophet(df, numero\_alvo):  
    \# Transforma em formato 'ds', 'y' (frequência do número alvo)  
    \#... lógica de contagem...  
    m \= Prophet()  
    m.fit(prophet\_df)  
    future \= m.make\_future\_dataframe(periods=10)  
    forecast \= m.predict(future)  
    return forecast

\# 5\. Modelagem LSTM (Previsão Sequencial)  
def create\_lstm\_dataset(dataset, look\_back=10):  
    X, Y \=,  
    for i in range(len(dataset) \- look\_back):  
        X.append(dataset\[i:(i \+ look\_back), :\])  
        Y.append(dataset\[i \+ look\_back, :\])  
    return np.array(X), np.array(Y)

def build\_lstm\_model(input\_shape, output\_units):  
    model \= Sequential()  
    model.add(LSTM(100, return\_sequences=True, input\_shape=input\_shape))  
    model.add(Dropout(0.2))  
    model.add(LSTM(50))  
    model.add(Dropout(0.2))  
    model.add(Dense(output\_units, activation='sigmoid')) \# Probabilidade para cada bola  
    model.compile(loss='binary\_crossentropy', optimizer='adam')  
    return model

\# Execução Principal (Pseudo-código)  
\# df, balls\_data \= load\_data('Mega-Sena.csv')  
\# arima\_pred \= train\_arima(df)  
\# lstm\_model \= build\_lstm\_model((10, 6), 60\)  
\#... treinamento e predição...

Este esqueleto demonstra como as diferentes técnicas são integradas. O ARIMA prevê características macro (soma), enquanto a LSTM tenta prever a microestrutura (probabilidade de cada número).17

## **6\. Avaliação de Resultados e Métricas**

A avaliação de modelos de previsão de loteria exige métricas específicas, pois a acurácia tradicional é enganosa em cenários de classes extremamente desbalanceadas (onde a classe "não sorteado" é a vasta maioria).

### **6.1 Métricas Apropriadas**

1. **Hit Rate (Taxa de Acerto):** Quantos números previstos pelo modelo realmente apareceram no sorteio alvo?  
2. **Top-K Accuracy:** Em vez de prever apenas 6 números, o modelo sugere os top $K$ (ex: 15 ou 20\) números mais prováveis. Avaliamos se os 6 números sorteados estão dentro desse conjunto $K$. Essa métrica é mais realista para a aplicação prática em "bolões" ou desdobramentos.59  
3. **Precision@K e Recall@K:** Medem a proporção de números relevantes (sorteados) recuperados no topo da lista de previsões.60

### **6.2 O Papel dos Sistemas de "Wheeling" (Desdobramento)**

Dado que nenhum modelo pode garantir o acerto dos 6 números com precisão determinística, a estratégia computacional mais eficaz é combinar as previsões da IA com sistemas de **Wheeling** (Desdobramentos Combinatórios). O script pode gerar um conjunto maior de números candidatos (ex: 15 números identificados pela LSTM como de alta probabilidade) e utilizar um algoritmo de fechamento combinatório para gerar o menor número de apostas que garanta uma Quadra ou Quina caso os números sorteados estejam dentro desse conjunto de 15\.62 Isso otimiza o custo-benefício da aposta, transformando uma previsão probabilística em uma cobertura matemática.

## **7\. Conclusão**

Este relatório delineou uma abordagem técnica e academicamente fundamentada para a análise da Mega-Sena. Demonstramos que, embora a aleatoriedade intrínseca dos sorteios (garantida por mecanismos físicos e auditados) imponha um limite teórico rígido à previsibilidade determinística ("adivinhar os números"), a aplicação de Ciência de Dados permite:

1. **Auditar a Aleatoriedade:** Confirmar se o sistema opera sem vieses estatísticos significativos através de testes como Qui-Quadrado.  
2. **Filtrar o Improvável:** Utilizar modelos estatísticos para eliminar combinações que, embora matematicamente possíveis, são estatisticamente anômalas (ex: somas extremas, sequências longas), melhorando a "qualidade" das apostas.  
3. **Explorar Padrões Temporais:** Utilizar LSTM e Prophet para detectar sutis dependências temporais ou correlações que escapam à análise humana direta.

O script resultante deste projeto não deve ser encarado como um oráculo, mas como uma ferramenta sofisticada de **otimização de decisão sob incerteza**. Ele desloca o jogador da pura superstição para uma estratégia baseada em densidade de probabilidade e cobertura combinatória, alinhando-se com as práticas modernas de análise quantitativa.

### ---

**Tabela Comparativa de Modelos**

| Modelo | Tipo | Principal Aplicação no Projeto | Vantagem | Limitação |
| :---- | :---- | :---- | :---- | :---- |
| **ARIMA** | Estatístico Linear | Previsão de métricas agregadas (Soma, Média) | Ótimo para séries estacionárias e univariadas. | Não captura padrões não-lineares complexos; requer dados estacionários. |
| **Prophet** | Aditivo Decomponível | Análise de sazonalidade e tendências de longo prazo | Robusto a outliers; fácil interpretação de componentes sazonais. | Pode forçar padrões sazonais onde eles não existem (em dados puramente aleatórios). |
| **LSTM** | Rede Neural Recorrente | Previsão de sequências numéricas e probabilidades | Capaz de aprender dependências complexas e não-lineares. | Risco alto de overfitting (memorização); requer grande volume de dados; "caixa preta". |

Este relatório consolida o conhecimento necessário para a implementação do projeto, cobrindo desde a teoria estatística até a arquitetura de software e validação de modelos.

#### **Referências citadas**

1. Mega-Sena.csv  
2. The Science Behind Lotto Champ: Analyzing Its Machine Learning Lottery Algorithm, acessado em dezembro 23, 2025, [https://medium.com/@mgsarwar2000/the-science-behind-lotto-champ-analyzing-its-machine-learning-lottery-algorithm-f63d06813997](https://medium.com/@mgsarwar2000/the-science-behind-lotto-champ-analyzing-its-machine-learning-lottery-algorithm-f63d06813997)  
3. Can We Find Strong Lottery Tickets in Generative Models? \- UNIST \- AI대학원, acessado em dezembro 23, 2025, [https://aigs.unist.ac.kr/filebox/item/1917192674\_3f038f79\_25433-Article+Text-29496-1-2-20230626.pdf](https://aigs.unist.ac.kr/filebox/item/1917192674_3f038f79_25433-Article+Text-29496-1-2-20230626.pdf)  
4. Bayesian Methods For Testing The Randomness of Lottery Draws \- Scribd, acessado em dezembro 23, 2025, [https://www.scribd.com/document/233044218/Bayesian-Methods-for-Testing-the-Randomness-of-Lottery-Draws](https://www.scribd.com/document/233044218/Bayesian-Methods-for-Testing-the-Randomness-of-Lottery-Draws)  
5. Unveiling the Law of Large Numbers: A Statistical Beacon in Randomness \- Medium, acessado em dezembro 23, 2025, [https://medium.com/@abhishekjainindore24/unveiling-the-law-of-large-numbers-a-statistical-beacon-in-randomness-69f40ab486e1](https://medium.com/@abhishekjainindore24/unveiling-the-law-of-large-numbers-a-statistical-beacon-in-randomness-69f40ab486e1)  
6. Law of large numbers \- Wikipedia, acessado em dezembro 23, 2025, [https://en.wikipedia.org/wiki/Law\_of\_large\_numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers)  
7. 7.1.1 Law of Large Numbers \- Probability Course, acessado em dezembro 23, 2025, [https://www.probabilitycourse.com/chapter7/7\_1\_1\_law\_of\_large\_numbers.php](https://www.probabilitycourse.com/chapter7/7_1_1_law_of_large_numbers.php)  
8. INVESTIGATING RANDOMNESS AND FAIRNESS IN THE ROMANIAN 6/49 LOTTERY, acessado em dezembro 23, 2025, [https://search.proquest.com/openview/ea7b5254940428c62f160576b512d164/1?pq-origsite=gscholar\&cbl=7250551](https://search.proquest.com/openview/ea7b5254940428c62f160576b512d164/1?pq-origsite=gscholar&cbl=7250551)  
9. I Wanted to Get Rich with Python and Lottery Stats. Here's What Happened. \- Medium, acessado em dezembro 23, 2025, [https://medium.com/@attila.hamari/i-wanted-to-get-rich-with-python-and-lottery-stats-heres-what-happened-0daf455fb8ae](https://medium.com/@attila.hamari/i-wanted-to-get-rich-with-python-and-lottery-stats-heres-what-happened-0daf455fb8ae)  
10. entropy — SciPy v1.16.2 Manual, acessado em dezembro 23, 2025, [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)  
11. Callam7/LottoPipeline: Scalable Data Pipeline (Lotto Example): A proof-of-concept showcasing a modular data pipeline for analyzing structured data. It ingests historical draws, performs frequency/decay analysis, clustering, Monte Carlo, and deep learning. Extensible for other analytics. \- GitHub, acessado em dezembro 23, 2025, [https://github.com/Callam7/LottoPipeline](https://github.com/Callam7/LottoPipeline)  
12. Fastest way to compute entropy in Python \- Stack Overflow, acessado em dezembro 23, 2025, [https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python](https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python)  
13. Time Series Forecasting with Facebook's Prophet in 10 Minutes \- Part 1, acessado em dezembro 23, 2025, [https://towardsdatascience.com/time-series-forecasting-with-facebooks-prophet-in-10-minutes-958bd1caff3f/](https://towardsdatascience.com/time-series-forecasting-with-facebooks-prophet-in-10-minutes-958bd1caff3f/)  
14. How to Create a Forecast Using Prophet in Python \- Cybrosys Technologies, acessado em dezembro 23, 2025, [https://www.cybrosys.com/blog/how-to-create-a-forecast-using-prophet-in-python](https://www.cybrosys.com/blog/how-to-create-a-forecast-using-prophet-in-python)  
15. Why So Many Time Series Models Fail in Practice | by James M. | Medium, acessado em dezembro 23, 2025, [https://medium.com/@data\_science\_content/time-series-forecasting-pitfalls-of-the-layman-data-scientist-99bca7f7de41](https://medium.com/@data_science_content/time-series-forecasting-pitfalls-of-the-layman-data-scientist-99bca7f7de41)  
16. How Do I Make an LSTM Model with Multiple Inputs? \- Data Science Dojo, acessado em dezembro 23, 2025, [https://datasciencedojo.com/blog/how-do-i-make-an-lstm-model-with-multiple-inputs/](https://datasciencedojo.com/blog/how-do-i-make-an-lstm-model-with-multiple-inputs/)  
17. How to Guess Accurately 3 Lottery Numbers Out of 6 using LSTM Model \- Medium, acessado em dezembro 23, 2025, [https://medium.com/@satanyu666/how-to-guess-accurately-3-lottery-numbers-out-of-6-using-lstm-model-44ef080d490c](https://medium.com/@satanyu666/how-to-guess-accurately-3-lottery-numbers-out-of-6-using-lstm-model-44ef080d490c)  
18. Predicting Lottery Winners Using SVC \- Kaggle, acessado em dezembro 23, 2025, [https://www.kaggle.com/code/mosemet/predicting-lottery-winners-using-svc](https://www.kaggle.com/code/mosemet/predicting-lottery-winners-using-svc)  
19. A word on the chi2 test \- Romain WARLOP, acessado em dezembro 23, 2025, [https://romainwarlop.github.io/machinelearning/entertainment/loto.html](https://romainwarlop.github.io/machinelearning/entertainment/loto.html)  
20. How “Random” is The Lottery, Really? | by Paul Stochaj \- Medium, acessado em dezembro 23, 2025, [https://medium.com/@paulostochaj/how-random-is-the-lottery-really-9727535db415](https://medium.com/@paulostochaj/how-random-is-the-lottery-really-9727535db415)  
21. chisquare — SciPy v1.16.1 Manual, acessado em dezembro 23, 2025, [https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats.chisquare.html](https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats.chisquare.html)  
22. A statistical test to detect tampering with lottery results \- ResearchGate, acessado em dezembro 23, 2025, [https://www.researchgate.net/publication/228931379\_A\_statistical\_test\_to\_detect\_tampering\_with\_lottery\_results](https://www.researchgate.net/publication/228931379_A_statistical_test_to_detect_tampering_with_lottery_results)  
23. A Statistical Analysis of Popular Lottery “Winning” Strategies, acessado em dezembro 23, 2025, [https://csbigs.fr/index.php/csbigs/article/view/289/270](https://csbigs.fr/index.php/csbigs/article/view/289/270)  
24. Can Machine Learning Crack the Lottery? A Data-Driven Exploration | by Thiago Zanin, acessado em dezembro 23, 2025, [https://medium.com/@thiagozanin.tz/can-machine-learning-crack-the-lottery-a-data-driven-exploration-4ffe52808a0a](https://medium.com/@thiagozanin.tz/can-machine-learning-crack-the-lottery-a-data-driven-exploration-4ffe52808a0a)  
25. Day 34: Python Balanced Numbers Filter, Identify Numbers with Equal Even and Odd Digits Using Modular Checks \- DEV Community, acessado em dezembro 23, 2025, [https://dev.to/shahrouzlogs/day-34-python-balanced-numbers-filter-identify-numbers-with-equal-even-and-odd-digits-using-chd](https://dev.to/shahrouzlogs/day-34-python-balanced-numbers-filter-identify-numbers-with-equal-even-and-odd-digits-using-chd)  
26. Heatmaps for Time Series \- Towards Data Science, acessado em dezembro 23, 2025, [https://towardsdatascience.com/heatmaps-for-time-series/](https://towardsdatascience.com/heatmaps-for-time-series/)  
27. \[OC\] Heatmap of “time since last appearance” for each number in French Loto draws (2019–2025) : r/dataisbeautiful \- Reddit, acessado em dezembro 23, 2025, [https://www.reddit.com/r/dataisbeautiful/comments/1pb7i0x/oc\_heatmap\_of\_time\_since\_last\_appearance\_for\_each/](https://www.reddit.com/r/dataisbeautiful/comments/1pb7i0x/oc_heatmap_of_time_since_last_appearance_for_each/)  
28. Seaborn Heatmaps: A Guide to Data Visualization \- DataCamp, acessado em dezembro 23, 2025, [https://www.datacamp.com/tutorial/seaborn-heatmaps](https://www.datacamp.com/tutorial/seaborn-heatmaps)  
29. Lottery features for time series Machine Learning \- Kaggle, acessado em dezembro 23, 2025, [https://www.kaggle.com/datasets/jmmvutu/lottery-features-for-machine-learning-ai](https://www.kaggle.com/datasets/jmmvutu/lottery-features-for-machine-learning-ai)  
30. Python Lottery Number Generation \- numpy \- Stack Overflow, acessado em dezembro 23, 2025, [https://stackoverflow.com/questions/56076593/python-lottery-number-generation](https://stackoverflow.com/questions/56076593/python-lottery-number-generation)  
31. FREE lottery tips to help you win Brazil Mega Sena 6/60 \- Smart Luck, acessado em dezembro 23, 2025, [https://www.smartluck.com/free-lottery-tips/brazil-megasena-660.htm](https://www.smartluck.com/free-lottery-tips/brazil-megasena-660.htm)  
32. Python Program to Count Even and Odd Numbers in a List \- GeeksforGeeks, acessado em dezembro 23, 2025, [https://www.geeksforgeeks.org/python/python-program-to-count-even-and-odd-numbers-in-a-list/](https://www.geeksforgeeks.org/python/python-program-to-count-even-and-odd-numbers-in-a-list/)  
33. A note on the appearance of consecutive numbers amongst the set of winning numbers in Lottery \- ResearchGate, acessado em dezembro 23, 2025, [https://www.researchgate.net/publication/228727998\_A\_note\_on\_the\_appearance\_of\_consecutive\_numbers\_amongst\_the\_set\_of\_winning\_numbers\_in\_Lottery](https://www.researchgate.net/publication/228727998_A_note_on_the_appearance_of_consecutive_numbers_amongst_the_set_of_winning_numbers_in_Lottery)  
34. Python Loop Patterns Guide | PDF | Control Flow \- Scribd, acessado em dezembro 23, 2025, [https://www.scribd.com/document/555819270/Patterns-1-1](https://www.scribd.com/document/555819270/Patterns-1-1)  
35. How to Develop LSTM Models for Time Series Forecasting \- MachineLearningMastery.com, acessado em dezembro 23, 2025, [https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)  
36. Decoding the Magic of Predicting Italian Lottery Numbers with AI \- EBDZ, acessado em dezembro 23, 2025, [https://ebdz.dev/blog/decoding-the-magic-of-predicting-italian-lottery-numbers-with-ai](https://ebdz.dev/blog/decoding-the-magic-of-predicting-italian-lottery-numbers-with-ai)  
37. Multivariate time series analysis using LSTM \- Kaggle, acessado em dezembro 23, 2025, [https://www.kaggle.com/code/bagavathypriya/multivariate-time-series-analysis-using-lstm](https://www.kaggle.com/code/bagavathypriya/multivariate-time-series-analysis-using-lstm)  
38. Time Series Prediction using LSTM with PyTorch in Python \- Stack Abuse, acessado em dezembro 23, 2025, [https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/](https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/)  
39. ARIMA, Prophet, LSTM complete guide \- Kaggle, acessado em dezembro 23, 2025, [https://www.kaggle.com/code/thuongtuandang/arima-prophet-lstm-complete-guide](https://www.kaggle.com/code/thuongtuandang/arima-prophet-lstm-complete-guide)  
40. How to Build ARIMA Model in Python for time series forecasting? \- ProjectPro, acessado em dezembro 23, 2025, [https://www.projectpro.io/article/how-to-build-arima-model-in-python/544](https://www.projectpro.io/article/how-to-build-arima-model-in-python/544)  
41. ARIMA Model \- Complete Guide to Time Series Forecasting in Python | ML+, acessado em dezembro 23, 2025, [https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)  
42. Mastering Time Series Forecasting: From ARIMA to LSTM \- MachineLearningMastery.com, acessado em dezembro 23, 2025, [https://machinelearningmastery.com/mastering-time-series-forecasting-from-arima-to-lstm/](https://machinelearningmastery.com/mastering-time-series-forecasting-from-arima-to-lstm/)  
43. ARIMA for Time Series Forecasting: A Complete Guide \- DataCamp, acessado em dezembro 23, 2025, [https://www.datacamp.com/tutorial/arima](https://www.datacamp.com/tutorial/arima)  
44. Multi-step Time Series Forecasting with ARIMA, LightGBM, and Prophet, acessado em dezembro 23, 2025, [https://towardsdatascience.com/multi-step-time-series-forecasting-with-arima-lightgbm-and-prophet-cc9e3f95dfb0/](https://towardsdatascience.com/multi-step-time-series-forecasting-with-arima-lightgbm-and-prophet-cc9e3f95dfb0/)  
45. Python | ARIMA Model for Time Series Forecasting \- GeeksforGeeks, acessado em dezembro 23, 2025, [https://www.geeksforgeeks.org/machine-learning/python-arima-model-for-time-series-forecasting/](https://www.geeksforgeeks.org/machine-learning/python-arima-model-for-time-series-forecasting/)  
46. Quick Start | Prophet \- Meta Open Source, acessado em dezembro 23, 2025, [https://facebook.github.io/prophet/docs/quick\_start.html](https://facebook.github.io/prophet/docs/quick_start.html)  
47. Prophet | Forecasting at scale. \- Meta Open Source, acessado em dezembro 23, 2025, [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/)  
48. Getting Started Predicting Time Series Data with Facebook Prophet \- Medium, acessado em dezembro 23, 2025, [https://medium.com/data-science/getting-started-predicting-time-series-data-with-facebook-prophet-c74ad3040525](https://medium.com/data-science/getting-started-predicting-time-series-data-with-facebook-prophet-c74ad3040525)  
49. Time Series Analysis using Facebook Prophet \- GeeksforGeeks, acessado em dezembro 23, 2025, [https://www.geeksforgeeks.org/data-science/time-series-analysis-using-facebook-prophet/](https://www.geeksforgeeks.org/data-science/time-series-analysis-using-facebook-prophet/)  
50. Time Series Forecasting With Prophet in Python \- MachineLearningMastery.com, acessado em dezembro 23, 2025, [https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/](https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/)  
51. ARIMA vs Prophet vs LSTM for Time Series Prediction \- Neptune.ai, acessado em dezembro 23, 2025, [https://neptune.ai/blog/arima-vs-prophet-vs-lstm](https://neptune.ai/blog/arima-vs-prophet-vs-lstm)  
52. AI LSTM model for Eurojackpot prediction. \- Modbus & Embedded Systems, acessado em dezembro 23, 2025, [https://modbus.pl/2025/02/04/ai-lstm-model-for-eurojackpot-prediction/](https://modbus.pl/2025/02/04/ai-lstm-model-for-eurojackpot-prediction/)  
53. Input and Output shape in LSTM (Keras) \- Kaggle, acessado em dezembro 23, 2025, [https://www.kaggle.com/code/shivajbd/input-and-output-shape-in-lstm-keras](https://www.kaggle.com/code/shivajbd/input-and-output-shape-in-lstm-keras)  
54. How to Guess Accurately 3 Lottery Numbers Out of 6 using LSTM Model | by Roi Polanitzer, acessado em dezembro 23, 2025, [https://medium.com/@polanitzer/how-to-guess-accurately-3-lottery-numbers-out-of-6-using-lstm-model-e148d1c632d6](https://medium.com/@polanitzer/how-to-guess-accurately-3-lottery-numbers-out-of-6-using-lstm-model-e148d1c632d6)  
55. Lotto Prediction \- Kaggle, acessado em dezembro 23, 2025, [https://www.kaggle.com/code/gogo827jz/lotto-prediction](https://www.kaggle.com/code/gogo827jz/lotto-prediction)  
56. Doing Multivariate Time Series Forecasting with Recurrent Neural Networks \- Databricks, acessado em dezembro 23, 2025, [https://www.databricks.com/blog/2019/09/10/doing-multivariate-time-series-forecasting-with-recurrent-neural-networks.html](https://www.databricks.com/blog/2019/09/10/doing-multivariate-time-series-forecasting-with-recurrent-neural-networks.html)  
57. Analyzing and forecasting with time series data using ARIMA models in Python, acessado em dezembro 23, 2025, [https://developer.ibm.com/tutorials/awb-arima-models-in-python/](https://developer.ibm.com/tutorials/awb-arima-models-in-python/)  
58. Tutorial: Time Series Forecasting with Prophet \- Kaggle, acessado em dezembro 23, 2025, [https://www.kaggle.com/code/prashant111/tutorial-time-series-forecasting-with-prophet](https://www.kaggle.com/code/prashant111/tutorial-time-series-forecasting-with-prophet)  
59. top\_k\_accuracy\_score — scikit-learn 1.8.0 documentation, acessado em dezembro 23, 2025, [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top\_k\_accuracy\_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html)  
60. Precision and recall at K in ranking and recommendations \- Evidently AI, acessado em dezembro 23, 2025, [https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k)  
61. 10 metrics to evaluate recommender and ranking systems \- Evidently AI, acessado em dezembro 23, 2025, [https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems)  
62. Lottery Wheel: How to Use It Effectively—and When It Actually Matters, acessado em dezembro 23, 2025, [https://lotterycodex.com/lottery-wheel/](https://lotterycodex.com/lottery-wheel/)  
63. Wheeling Lotto Numbers, Playing Lotto Wheels, Make Your Own Lottery Systems, acessado em dezembro 23, 2025, [https://saliu.com/lottowheel.html](https://saliu.com/lottowheel.html)