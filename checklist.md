# Checklist Tech Challenge - Fase 1 (Pós-Graduação)

Documento consolidado de acompanhamento e guia de desenvolvimento para o **Tech Challenge - Fase 1** (Sistema Inteligente de Suporte ao Diagnóstico Médico)[cite: 1].

---

## 📌 Visão Geral do Projeto
- **Objetivo:** Construir uma solução base em IA/Machine Learning para processamento e classificação de dados médicos tabulares (diagnóstico)[cite: 1].
- **Peso:** 90% da nota das disciplinas da Fase 1[cite: 1].
- **Formato:** Projeto prático em Python com entregáveis de código, documentação e vídeo[cite: 1].

---

## 📋 Checklist de Desenvolvimento

### 1. Definição do Problema e Escolha dos Dados
- [ ] **Escolha do Dataset Tabular:**
  - Ex: *Breast Cancer Wisconsin* (Diagnóstico de câncer de mama) ou *Pima Indians Diabetes*[cite: 1].
- [ ] **Definição da Variável Alvo (Target):** Identificar a coluna contendo o diagnóstico (ex: Benigno vs Maligno, Positivo vs Negativo)[cite: 1].

### 2. Análise Exploratória de Dados (EDA)
- [ ] **Estatística Descritiva:** Calcular média, mediana, desvio padrão, quartis e intervalos[cite: 1].
- [ ] **Visualização de Distribuições:**
  - Histogramas e KDEs para verificar normalidade e assimetria[cite: 1].
  - Boxplots para identificação visual de outliers[cite: 1].
- [ ] **Análise de Correlação:**
  - Matriz de correlação (Pearson/Spearman) e Mapa de Calor (Heatmap)[cite: 1].
  - Identificação de multicolinearidade entre variáveis preditoras[cite: 1].

### 3. Pré-processamento e Construção de Pipelines
- [ ] **Limpeza de Dados:** Tratar duplicatas e registros inconsistentes[cite: 1].
- [ ] **Tratamento de Nulos:** Imputação de dados ausentes (Média, Mediana, KNNImputer) ou remoção consciente[cite: 1].
- [ ] **Encoding de Variáveis Categóricas:** Aplicar *One-Hot Encoding* ou *Ordinal Encoding*, se aplicável[cite: 1].
- [ ] **Normalização / Padronização:**
  - Escalonamento numérico (*StandardScaler* ou *MinMaxScaler*)[cite: 1].
- [ ] **Redução de Dimensionalidade (Opcional/Se necessário):**
  - Aplicação de PCA ou técnicas de Seleção de Atributos (*SelectKBest*, *RFE*).
- [ ] **Encapsulamento em Pipeline:**
  - Agrupar todas as etapas de transformação usando `sklearn.pipeline.Pipeline` para evitar *data leakage*[cite: 1].

### 4. Divisão e Treinamento de Modelos
- [ ] **Divisão Amostral Rígida:**
  - Separar os dados em **Treino**, **Validação** e **Teste** (usando `stratify=y` se a classe for desbalanceada)[cite: 1].
- [ ] **Treinamento Multi-modelo:**
  - Treinar no mínimo 2 algoritmos de classificação (ex: *Regressão Logística*, *Decision Tree*, *Random Forest*, *KNN*, *XGBoost*)[cite: 1].
  - Ajuste de hiperparâmetros (GridSearchCV / RandomizedSearchCV).

### 5. Avaliação, Explicabilidade e Análise Crítica
- [ ] **Métricas de Performance:**
  - Calcular **Accuracy**, **Recall** (Sensibilidade) e **F1-Score** no conjunto de teste[cite: 1].
  - Matriz de Confusão para visualização dos erros (Falsos Positivos vs Falsos Negativos).
- [ ] **Justificativa Contextual:**
  - Discutir o impacto do **Recall** no cenário médico (minimizar Falsos Negativos em diagnósticos críticos)[cite: 1].
- [ ] **Interpretabilidade do Modelo:**
  - Obter a importância das variáveis (*Feature Importance*)[cite: 1].
  - Aplicar **SHAP** (*SHapley Additive exPlanations*) para explicar individualmente as decisões do modelo[cite: 1].
- [ ] **Discussão Crítica:**
  - Avaliar viabilidade de uso na prática clínica como ferramenta de apoio (sistema de triagem/segunda opinião com palavra final do médico)[cite: 1].

---

## 🎨 Desafio Extra (Pontuação Adicional)
- [ ] **Visão Computacional (CNN):**
  - Implementar classificação baseada em imagem médica (ex: detecção de pneumonia via raio-X de tórax usando PyTorch/TensorFlow)[cite: 1].

---

## 📦 Entregáveis da Fase 1

### 1. Repositório Git
- [ ] Código-fonte estruturado e comentado (`.py` e/ou `.ipynb`)[cite: 1].
- [ ] `README.md` detalhado com instruções claras de instalação e execução[cite: 1].
- [ ] `Dockerfile` funcional para execução contêinerizada da aplicação/ambiente[cite: 1].
- [ ] Link do dataset ou instruções/scripts para download[cite: 1].

### 2. Relatório Técnico em PDF
- [ ] Estratégias e justificativas de pré-processamento[cite: 1].
- [ ] Arquitetura dos modelos e motivação das escolhas[cite: 1].
- [ ] Apresentação gráfica dos resultados (tabelas, gráficos, matriz de confusão, SHAP plots)[cite: 1].
- [ ] Conclusões e análise crítica de negócios/medicina[cite: 1].

### 3. Vídeo de Demonstração
- [ ] Gravação de até 15 minutos (YouTube/Vimeo, público ou não listado)[cite: 1].
- [ ] Explicação do fluxo de dados, pipeline de ML e demonstração do sistema em execução[cite: 1].