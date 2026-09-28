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
├── database/
│   └── data.csv
├── requisitos/
│   ├── POSTECH - Tech Challenge - Fase 1.md
│   └── POSTECH - Tech Challenge - Fase 1.pdf
├── checklist.md
├── tech-challenge.ipynb
└── README.md
```

## Requisitos

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
