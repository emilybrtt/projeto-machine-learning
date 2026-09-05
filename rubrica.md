Objetivo: **Desenvolver um pipeline completo de Machine Learning escolhendo um dos datasets abaixo**

Defina seu grupo: `dupla` - `até 2 pessoas`

Escolha: **APENAS UM** dos datasets abaixo para trabalhar nas 2 etapas do projeto: EDA e <Classificação | Regressão>:

* **Classificação**  
    https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset  
    https://www.kaggle.com/datasets/anaghakp/adult-income-census  

* **Regressão**  
    https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data  
    https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho  
  
- Cada etapa do projeto tem seu `deadline`;
- A entrega deve ser feita via Form disponibilizado, com um arquivo `.ipynb` ou com um link para um `Colab`;
- O material entregue deve ter, além dos requisitos básicos de cada etapa:
    - Identificação dos membros do grupo;
    - Título do projeto;
    - Data da entrega;
    - Referências bibliográficas utilizadas (se houver);
    - Qualquer outra informação que o grupo julgar relevante para a avaliação do projeto.

    Na etapa de EDA, o grupo deve realizar uma análise exploratória abrangente para entender a estrutura, qualidade, distribuições, relações entre variáveis (numéricas e categóricas) e desafios, como missing values e vieses, preparando para a modelagem nas fases subsequentes.

Tarefas básicas de EDA:

1. Carregamento e inspeção inicial dos dados:

    - Explicar cada feature do dataset.
    - Verificar o número de instâncias e features.
    - Identificar tipos de dados (numéricos, categóricos).
    - Detectar valores ausentes e inconsistências.
    - Verificar se há desbalanceamento de classes.
    - Realizar a separação dos dados em conjuntos de treino e teste, quando necessário.

2. Análise univariada:

    - Calcular estatísticas descritivas (média, mediana, desvio padrão) de todas as variáveis numéricas.
    - Criar histogramas, boxplots e/ou violinos para variáveis numéricas (máximo 3).
    - Para variáveis categóricas: calcular frequências e criar gráficos de barras (máximo 3).

3. Análise bivariada e multivariada:

    - Analisar correlações entre variáveis numéricas usando matrizes de correlação e scatter plots (máximo 3).
    - Analisar relações entre variáveis categóricas e o target 'income' usando gráficos de barras ou tabelas de contingência (máximo 3).
    - Analisar relações entre variáveis numéricas e categóricas usando boxplots conjuntos (máximo 3).

4. Pré-processamento para modelagem:

    - Escolher e justificar estratégia para lidar com valores ausentes (remoção, imputação).
    - Escolher e justificar estratégia para lidar com outliers (remoção, transformação).
    - Escolher e justificar estratégia para encoding de variáveis categóricas (one-hot encoding, label encoding).
    - Escolher e justificar estratégia para normalização ou padronização de variáveis numéricas (Min-Max Scaling, Standardization, etc.).
    - Nas variáveis numéricas, aplicar uma técnica de redução de dimensionalidade, como PCA, para visualizar a estrutura dos dados e identificar possíveis agrupamentos ou padrões.
    - Analisar os resultados da redução de dimensionalidade para obter insights sobre a separabilidade das classes e a importância das features.
    - Construir um pipeline de pré-processamento (utilizando Pipeline e ColumnTransformer do sklearn) que inclua as etapas acima, garantindo que seja aplicável para as fases de modelagem subsequentes.

5. Documentação e apresentação:

    - Todos os requisitos do projeto explicitados em cada ponto acima, assim como todas as figuras produzidas e decisões tomadas, devem ser justificados de maneira objetiva.
    - As visualizações produzidas devem ser claras e informativas com títulos, rótulos e legendas adequados.
    - Preparar um relatório que resuma os principais achados e as estratégias propostas para a modelagem.
    - O código deve ser bem organizado, com comentários explicativos para cada etapa.
    - O grupo deve justificar todas as escolhas de pré-processamento com base nos achados da EDA, garantindo que as estratégias propostas sejam adequadas para os desafios identificados no dataset.
    - Python obrigatório, utilizando as bibliotecas  Pandas, NumPy, Matplotlib, Seaborn e Scikit-learn.
### Critérios de avaliação:

| Conceito | Descrição |
| --- | --- |
| `I` | Não entrega da APS. |
| `D` | Entrega com contúdo incompleto, análise superficial, falta de justificativas para as escolhas de pré-processamento, ou visualizações inadequadas. |
| `C` | Entrega completa, mas com algumas falhas na análise, justificativas ou visualizações. |
| `B` | Entrega completa, com análise detalhada, justificativas claras para as escolhas de pré-processamento, e visualizações adequadas. |
| `A` | Entrega completa, com análise aprofundada, justificativas muito claras e bem fundamentadas para as escolhas de pré-processamento, visualizações excelentes, e um relatório muito bem estruturado e informativo. |

Para entregas de documentação aberta (eg: GitPages) e bem documentados, o grupo agrega o conceito `+`.