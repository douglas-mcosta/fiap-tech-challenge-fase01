# 🎙️ ROTEIRO DE FALA PARA GRAVAÇÃO DO VÍDEO
## Tech Challenge — Fase 1 | FIAP Pós-Tech
**Sistema Inteligente de Suporte ao Diagnóstico de Câncer de Mama**

- **Duração Estimada:** 10 a 13 minutos (Teto máximo do edital: 15 minutos)
- **Dinâmica do Vídeo:**
  1. **Início (Slides HTML):** Abertura, grupo e situação problema (00:00 - 02:00)
  2. **Meio (Jupyter Notebook):** Demonstração do código e dados em execução (02:00 - 10:00)
  3. **Fim (Slides HTML):** Conclusão clínica, próximos passos e agradecimento (10:00 - 12:00)

---

# PARTE 1: ABERTURA NOS SLIDES INICIAIS (PPT / HTML)

### 📌 SLIDE 1: Apresentação do Grupo e Projeto
- **O que mostrar na tela:** Slide 1 do arquivo `apresentacao_slides.html` (Capa com título e nomes dos integrantes).
- **Tempo sugerido:** `00:00 - 00:45`

> **FALA DO APRESENTADOR:**
> *"Olá a todos, professores e avaliadores! Sejam muito bem-vindos à apresentação do nosso Tech Challenge da Fase 1 da Pós-Tech da FIAP em IA para Devs.*
>
> *O nosso projeto é o desenvolvimento de um **Sistema Inteligente de Suporte ao Diagnóstico Médico para Classificação de Tumores Mamários**. Os integrantes do nosso grupo são: Davi de Aquino Barbosa, Douglas Martins de Medeiros Costa, Everton de Souza, Elton dos Santos e Luiz Ricardo Marquesete de Barros.*"

---

### 📌 SLIDE 2: A Situação Problema do Hospital Universitário
- **O que mostrar na tela:** Slide 2 do arquivo `apresentacao_slides.html` (Desafio Clínico, Gargalo de Laudos e Triagem).
- **Tempo sugerido:** `00:45 - 01:45`

> **FALA DO APRESENTADOR:**
> *"O nosso desafio se passa no cenário real de um grande hospital universitário que lida com um volume imenso e crescente de exames diários. O gargalo na fila de laudos anatomopatológicos e biópsias mamárias pode atrasar o início do tratamento oncológico de casos graves.*
>
> *O nosso objetivo foi construir uma solução em Machine Learning baseada no dataset de referência **Breast Cancer Wisconsin**, capaz de analisar características celulares e classificar o diagnóstico em Benigno ou Maligno.*
>
> *Um ponto fundamental que guiou todo o nosso trabalho é a postura ética e clínica: o nosso modelo não substitui o médico patologista; ele atua como uma ferramenta inteligente de triagem e suporte para priorizar casos urgentes, onde o foco principal é maximizar o Recall para evitar falsos negativos."*

---

### 📌 SLIDE 3: Transição para o Código e Pipeline
- **O que mostrar na tela:** Slide 3 do arquivo `apresentacao_slides.html` (Card de transição com seta para o Notebook).
- **Tempo sugerido:** `01:45 - 02:05`

> **FALA DO APRESENTADOR:**
> *"Vamos agora alternar a tela para o nosso Jupyter Notebook (`apresentacao_tech_challenge.ipynb`), onde demonstraremos todo o pipeline de dados, a limpeza, os testes estatísticos e o treinamento dos modelos em tempo real."*

---

# PARTE 2: DEMONSTRAÇÃO PRÁTICA NO NOTEBOOK (`apresentacao_tech_challenge.ipynb`)

### 📌 SEÇÃO 1 DO NOTEBOOK: Contexto e Objetivo
- **O que mostrar na tela:** Seção 1 do notebook.
- **Tempo sugerido:** `02:05 - 02:40`

> **FALA DO APRESENTADOR:**
> *"Aqui na Seção 1, temos a formalização do problema. Estamos trabalhando com 569 amostras de biópsias aspirativas, com 30 medidas morfométricas contínuas. A variável alvo é o diagnóstico: Benigno representado por zero e Maligno representado por um."*

---

### 📌 SEÇÃO 2 DO NOTEBOOK: Carregamento, Inspeção e Limpeza dos Dados
- **O que mostrar na tela:** Células 2 até 21 (instalação, `pd.read_csv`, `missingno`, `drop` de colunas e `LabelEncoder`).
- **Tempo sugerido:** `02:40 - 04:00`

> **FALA DO APRESENTADOR:**
> *"Na Seção 2, importamos as bibliotecas necessárias — como Scikit-Learn, Pandas e Seaborn — e carregamos os dados brutos.*
>
> *Durante a inspeção inicial, tomamos duas decisões importantes de limpeza de dados:*
> *1. Removemos a coluna 'id', pois ela é apenas um código administrativo do paciente e geraria ruído desnecessário.*
> *2. Removemos a coluna 'Unnamed: 32', que foi criada por um delimitador extra no CSV e continha 100% de valores nulos, como comprovamos visualmente pelo gráfico do missingno.*
>
> *Após isso, auditamos a base e confirmamos: temos zero linhas duplicadas e zero valores ausentes residuais nas 30 características. Por fim, codificamos a classe alvo: 0 para benigno e 1 para maligno."*

---

### 📌 SEÇÃO 3 DO NOTEBOOK: Análise Exploratória dos Dados (EDA) e Outliers
- **O que mostrar na tela:** Células 24 até 37 (gráfico de barras das classes, boxplots, histogramas e mapa de calor de correlação).
- **Tempo sugerido:** `04:00 - 05:30`

> **FALA DO APRESENTADOR:**
> *"Na Seção 3, partimos para a Análise Exploratória. Observamos que 62,7% dos casos são benignos e 37,3% são malignos. É uma proporção bem distribuída que nos permite trabalhar com segurança.*
>
> *Ao olhar os boxplots, notamos valores extremos, os chamados outliers, principalmente nas grandezas de raio e área. Uma decisão metodológica fundamental do grupo foi **manter esses outliers**. Por quê? Porque na medicina oncológica, células cancerígenas sofrem mutações severas e crescem desordenadamente. Esses valores extremos são a assinatura biológica da doença, não um erro de medição!*
>
> *Na matriz de correlação, identificamos associações lineares muito fortes com a malignidade, acima de 0,78, especialmente em 'concave points_worst' e 'perimeter_worst'. Ou seja: núcleos maiores e com bordas mais irregulares estão fortemente ligados a tumores malignos."*

---

### 📌 SEÇÃO 4 E 5 DO NOTEBOOK: Preparação e Divisão Rigorosa dos Dados
- **O que mostrar na tela:** Células 39 até 50 (separação de X e y, gráficos demonstrando o `stratify` e a partição definitiva 60/20/20).
- **Tempo sugerido:** `05:30 - 06:45`

> **FALA DO APRESENTADOR:**
> *"Aqui nas Seções 4 e 5 demonstramos nosso rigor metodológico na separação das amostras.*
>
> *Primeiro, provamos empiricamente a importância do parâmetro 'stratify'. Sem ele, a proporção de casos graves entre treino e teste poderia oscilar. Com a estratificação, mantemos a taxa exata de cerca de 37% de malignos em todos os grupos.*
>
> *Além disso, nós dividimos a base em **três conjuntos independentes**: 60% para Treino (341 amostras), 20% para Validação (114 amostras) e 20% para Teste Final Cego (114 amostras). O conjunto de teste permaneceu completamente isolado e nunca foi tocado durante os experimentos de ajuste!"*

---

### 📌 SEÇÃO 6 DO NOTEBOOK: Construção dos Pipelines de Classificação
- **O que mostrar na tela:** Células 53 e 55 (definição dos objetos `Pipeline` com `StandardScaler`, `LogisticRegression` e `KNeighborsClassifier`).
- **Tempo sugerido:** `06:45 - 07:30`

> **FALA DO APRESENTADOR:**
> *"Na Seção 6, implementamos dois modelos: a Regressão Logística e o K-Nearest Neighbors (KNN).*
>
> *Como as nossas 30 medidas têm grandezas muito diferentes — algumas na casa dos milhares e outras menores que um —, padronizamos os dados com o 'StandardScaler'. E o ponto alto de engenharia de software foi encapsular tudo dentro de um 'Pipeline' do Scikit-Learn.*
>
> *Isso garante que a média e o desvio padrão sejam calculados **estritamente sobre o conjunto de treino**, eliminando 100% de qualquer vazamento de dados ('data leakage') para a validação e teste."*

---

### 📌 SEÇÃO 7 DO NOTEBOOK: Validação, Escolha de K e Teste de Dimensionalidade
- **O que mostrar na tela:** Células 57 até 66 (loop testando K de 3 a 11, e a tabela comparando 30 características vs 10 características).
- **Tempo sugerido:** `07:30 - 08:30`

> **FALA DO APRESENTADOR:**
> *"Na Seção 7, usamos o split de validação para tomar duas decisões técnicas:*
> *1. Para o KNN, testamos vizinhos de 3 a 11. O valor de K=5 empatou no topo com 97,37% de acurácia e o melhor recall, sendo adotado como padrão.*
>
> *2. Realizamos um experimento: testamos treinar a Regressão Logística usando apenas as 10 características mais correlacionadas. A acurácia geral até empatou, mas o modelo com 30 características gerou apenas **1 falso negativo**, enquanto o modelo com 10 características gerou **2 falsos negativos**.*
>
> *Em diagnósticos oncológicos, um falso negativo significa uma paciente com câncer voltando para casa sem o tratamento urgente. Portanto, a decisão médica e técnica correta foi manter todas as 30 características!"*

---

### 📌 SEÇÃO 8, 9 E 10 DO NOTEBOOK: Teste Final Cego, Métricas e Matrizes de Confusão
- **O que mostrar na tela:** Células 69 até 78 (treinamento final, predições no teste, tabela consolidada e gráficos de `ConfusionMatrixDisplay`).
- **Tempo sugerido:** `08:30 - 09:30`

> **FALA DO APRESENTADOR:**
> *"Chegamos à prova definitiva nas Seções 8, 9 e 10: a avaliação no conjunto de Teste Cego com 114 pacientes que os modelos nunca tinham visto.*
>
> *A Regressão Logística foi amplamente superior ao KNN:*
> *- Teve maior Acurácia: **97,37%** contra 94,74%.*
> *- Teve maior Recall para casos malignos: **95,24%** contra 88,10%.*
> *- E o mais importante de tudo: olhando as matrizes de confusão, a Regressão Logística cometeu apenas **2 falsos negativos**, enquanto o KNN cometeu **5 falsos negativos**.*
>
> *Em um hospital universitário, deixar passar mais que o dobro de lesões malignas seria inaceitável. A Regressão Logística provou ser muito mais segura e consistente para salvar vidas."*

---

### 📌 SEÇÃO 11 DO NOTEBOOK: Interpretação das Características (Feature Importance)
- **O que mostrar na tela:** Células 81 e 82 (extração dos coeficientes e gráfico de barras horizontais).
- **Tempo sugerido:** `09:30 - 10:15`

> **FALA DO APRESENTADOR:**
> *"Na Seção 11, abrimos a 'caixa-preta' do modelo analisando os coeficientes da Regressão Logística. Como padronizamos os dados, os coeficientes são diretamente comparáveis.*
>
> *A variável que mais pesou no modelo foi 'texture_worst', com coeficiente positivo de +1,35, seguida por pontos côncavos e medidas de raio.*
>
> *Isso bate perfeitamente com a literatura patológica: carcinomas mamários invasivos apresentam cromatina celular muito heterogênea e deformidades severas de contorno. Ou seja, o modelo aprendeu características reais da patologia, oferecendo explicabilidade total para a equipe médica."*

---

# PARTE 3: FECHAMENTO NOS SLIDES FINAIS (PPT / HTML)

### 📌 SLIDE 4: Conclusões Clínicas e Viabilidade Prática
- **O que mostrar na tela:** Slide 4 do arquivo `apresentacao_slides.html` (Viabilidade Hospitalar, Triagem Inteligente e Governança Ética).
- **Tempo sugerido:** `10:15 - 11:15`

> **FALA DO APRESENTADOR:**
> *"Retornando agora aos nossos slides para a conclusão do trabalho: respondemos à pergunta central do desafio — **O modelo pode ser utilizado na prática hospitalar?***
>
> *A resposta é **sim, como ferramenta de triagem inteligente e segunda opinião automatizada**.*
> *O fluxo sugerido é simples: exames onde o modelo detecta alta probabilidade de malignidade são automaticamente posicionados no topo da fila de laudos do patologista, reduzindo o tempo de espera do paciente.*
>
> *E reforçamos a governança ética: a palavra final é inegociavelmente do médico especialista. A inteligência artificial serve para potencializar e dar segurança à decisão humana."*

---

### 📌 SLIDE 5: Próximos Passos (CNN) e Agradecimentos
- **O que mostrar na tela:** Slide 5 do arquivo `apresentacao_slides.html` (Desafio Extra de Visão Computacional, Resumo dos Entregáveis e Agradecimentos).
- **Tempo sugerido:** `11:15 - 12:00`

> **FALA DO APRESENTADOR:**
> *"Como próximos passos e desafio extra do Tech Challenge, pretendemos evoluir a arquitetura para uma abordagem multimodal, incorporando Redes Neurais Convolucionais (CNNs) para analisar imagens diretas de mamografias e lâminas histopatológicas em conjunto com os dados tabulares.*
>
> *Todos os entregáveis — o repositório Git, o relatório técnico detalhado e este vídeo — foram desenvolvidos com rigor técnico, reprodutibilidade e foco na ética médica.*
>
> *Muito obrigado pela atenção de todos os professores e avaliadores, e estamos à disposição para esclarecimentos!"*
