# Tech Challenge — Fase 1

Projeto desenvolvido para o Tech Challenge da Pós-Tech, com o objetivo de construir uma solução inicial de Machine Learning para apoiar a classificação de tumores mamários como benignos ou malignos.

## Problema

O projeto utiliza o dataset **Breast Cancer Wisconsin (Diagnostic)**. A base contém características numéricas extraídas de imagens digitalizadas de massas mamárias, como raio, perímetro, área, concavidade e pontos côncavos dos núcleos celulares.

A variável alvo é `diagnosis`:

- `B`: benigno, codificado como `0`;
- `M`: maligno, codificado como `1`.

O modelo será desenvolvido como uma ferramenta de apoio à análise. A decisão final sobre um diagnóstico deve permanecer sob responsabilidade de profissionais de saúde.

## Estado atual

O notebook contém:

- carregamento e inspeção da base;
- análise das dimensões, tipos e estatísticas descritivas;
- verificação de valores ausentes e registros duplicados;
- remoção das colunas `id` e `Unnamed: 32`;
- codificação da variável alvo;
- análise da distribuição das classes;
- histogramas e boxplots das características numéricas;
- comparação de características entre diagnósticos;
- análise de correlação;
- visualização das características mais correlacionadas com o diagnóstico.

As próximas etapas incluem a divisão dos dados em treino, validação e teste, criação do pipeline de pré-processamento, treinamento de pelo menos dois modelos, avaliação das métricas e interpretação dos resultados.

## Estrutura do projeto

```text
.
├── app/
│   └── app.py              # interface web (Streamlit)
├── database/
│   └── data.csv
├── desafio-visao-computacional/   # imagem Docker da pneumonia
│   ├── app/app.py
│   ├── src/train.py
│   ├── models/                  # CNN treinada
│   ├── exemplos/                # raios-X de exemplo
│   ├── Dockerfile
│   └── Tech_Challenge_Fase1_Visao_Computacional.ipynb
├── src/
│   └── train.py            # treino do modelo final
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requisitos/
│   ├── POSTECH - Tech Challenge - Fase 1.md
│   └── POSTECH - Tech Challenge - Fase 1.pdf
├── checklist.md
├── tech-challenge.ipynb
├── Relatorio_Tecnico_Tech_Challenge_Fase1_v1.docx
└── README.md
```

## Executar com Docker (recomendado para avaliação)

O projeto é entregue em **duas imagens independentes**, uma para cada desafio. As duas já vêm com todas as dependências e com o modelo treinado. Só é preciso ter o [Docker](https://www.docker.com/products/docker-desktop/) instalado.

| Desafio | Imagem | Endereço |
|---|---|---|
| Classificação de tumores mamários (Regressão Logística) | `ghcr.io/douglas-mcosta/tech-challenge-tumores` | http://localhost:8501 |
| Detecção de pneumonia em raio-X (CNN) | `ghcr.io/douglas-mcosta/tech-challenge-pneumonia` | http://localhost:8502 |

### Tumores mamários

```bash
docker run --rm -p 8501:8501 ghcr.io/douglas-mcosta/tech-challenge-tumores:latest
```

Abra **http://localhost:8501**. A interface permite:

- classificar registros do conjunto de teste e comparar com o diagnóstico real;
- informar manualmente as 30 características e obter a probabilidade de maligno;
- enviar um CSV no formato de `database/data.csv` e baixar as previsões;
- ver as métricas de teste (Accuracy, Recall, F1, falsos negativos) da Regressão Logística e do KNN.

O modelo é treinado durante o `docker build` (`src/train.py`, com a mesma divisão e `random_state` do notebook).

### Pneumonia em raio-X

```bash
docker run --rm -p 8502:8501 ghcr.io/douglas-mcosta/tech-challenge-pneumonia:latest
```

Abra **http://localhost:8502** (a primeira abertura leva uns 15 segundos para carregar o TensorFlow). A interface permite:

- classificar imagens de exemplo do conjunto de teste (com acertos e erros do modelo) e comparar com o diagnóstico real;
- enviar um raio-X de tórax (JPEG/PNG) e obter a probabilidade de pneumonia;
- ver as métricas de teste (acurácia, recall, F1, AUC-ROC e matriz de confusão).

A CNN foi treinada uma vez com `desafio-visao-computacional/src/train.py` (mesma arquitetura e sementes do notebook) e o modelo salvo em `desafio-visao-computacional/models/` vai dentro da imagem. O dataset de 2 GB não é necessário para usar o app.

### Notebooks no JupyterLab

Qualquer uma das imagens abre o notebook correspondente no JupyterLab, sem senha:

```bash
docker run --rm -p 8888:8888 ghcr.io/douglas-mcosta/tech-challenge-tumores:latest notebook
```

Acesse **http://localhost:8888/lab/tree/tech-challenge.ipynb**. No caso da pneumonia, troque a imagem por `tech-challenge-pneumonia` e acesse `Tech_Challenge_Fase1_Visao_Computacional.ipynb`. Esse notebook baixa o dataset do Kaggle, então precisa de internet, e o treino na CPU é demorado.

### Construir as imagens localmente

A partir do código-fonte, sem baixar as imagens publicadas:

```bash
docker compose up --build
```

Os dois apps sobem juntos: tumores em http://localhost:8501 e pneumonia em http://localhost:8502.

Para retreinar a CNN (precisa de internet, leva de 10 a 30 minutos na CPU):

```bash
docker run --rm -v "$(pwd)/desafio-visao-computacional:/app" tech-challenge-pneumonia:latest train
```

### Publicação das imagens

O workflow `.github/workflows/docker-publish.yml` publica as duas imagens no GitHub Container Registry a cada push na branch principal (amd64 e arm64, funciona em Windows, Linux e Mac com Apple Silicon). Após a primeira execução, deixe cada pacote **público** em *GitHub → Packages → (pacote) → Package settings → Change visibility*.

## Requisitos (execução sem Docker)

- Python 3.9 ou superior;
- Jupyter Notebook ou uma IDE compatível, como o Visual Studio Code.

As bibliotecas utilizadas podem ser instaladas com:

```bash
python -m pip install numpy pandas matplotlib seaborn missingno scikit-learn jupyter
```

## Como executar

Clone o repositório:

```bash
git clone https://github.com/douglas-mcosta/fiap-tech-challenge-fase01.git
cd fiap-tech-challenge-fase01
```

Crie e ative um ambiente virtual:

```bash
python -m venv env
source env/bin/activate
```

No Windows, use:

```powershell
env\Scripts\activate
```

Instale as dependências e abra o notebook:

```bash
python -m pip install numpy pandas matplotlib seaborn missingno scikit-learn jupyter
jupyter notebook tech-challenge.ipynb
```

Execute as células em ordem para reproduzir a análise.

## Dataset

O dataset utilizado está disponível no Kaggle:

[Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

A cópia utilizada pelo notebook está em `database/data.csv`.

## Observações da análise exploratória

A base possui 569 registros: 357 benignos e 212 malignos. Não foram encontrados valores ausentes após a remoção da coluna completamente vazia, nem registros duplicados.

Características relacionadas ao tamanho e à irregularidade dos núcleos apresentaram associação positiva relevante com a classe maligna. Entre elas estão `concave points_worst`, `perimeter_worst`, `concave points_mean` e `radius_worst`. Também existem correlações elevadas entre raio, perímetro e área, indicando que algumas características carregam informações semelhantes.

Essas relações são exploratórias: correlação não representa causalidade e uma característica isolada não determina o diagnóstico.
