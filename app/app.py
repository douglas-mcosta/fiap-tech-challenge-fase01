"""Interface web para testar o modelo de Regressão Logística treinado em src/train.py."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "models"
NOMES_CLASSES = {0: "Benigno", 1: "Maligno"}

st.set_page_config(page_title="Tech Challenge Fase 1", page_icon="🩺", layout="wide")


@st.cache_resource
def carregar_modelo():
    return joblib.load(PASTA_MODELOS / "modelo_regressao_logistica.joblib")


@st.cache_data
def carregar_artefatos():
    metricas = json.loads((PASTA_MODELOS / "metricas.json").read_text(encoding="utf-8"))
    teste = pd.read_csv(PASTA_MODELOS / "conjunto_teste.csv", index_col=0)
    return metricas, teste


modelo = carregar_modelo()
metricas, teste = carregar_artefatos()
caracteristicas = list(modelo.feature_names_in_)

st.title("🩺 Classificação de tumores mamários")
st.caption(
    "Tech Challenge — Fase 1 · Breast Cancer Wisconsin (Diagnostic) · "
    "Regressão Logística com StandardScaler. Ferramenta de apoio: a decisão final "
    "é sempre de um profissional de saúde."
)

aba_teste, aba_manual, aba_lote, aba_metricas = st.tabs(
    ["Amostra do teste", "Entrada manual", "Arquivo CSV", "Métricas"]
)

with aba_teste:
    st.write("Escolha um registro do conjunto de teste (não usado no treino) e compare com o diagnóstico real.")
    indice = st.selectbox(
        "Registro",
        teste.index,
        format_func=lambda i: f"#{i} — real: {NOMES_CLASSES[teste.loc[i, 'diagnosis']]}",
    )
    registro = teste.loc[[indice], caracteristicas]
    probabilidade = modelo.predict_proba(registro)[0, 1]
    previsto = int(probabilidade >= 0.5)
    real = int(teste.loc[indice, "diagnosis"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Previsão do modelo", NOMES_CLASSES[previsto])
    c2.metric("Probabilidade de maligno", f"{probabilidade:.1%}")
    c3.metric("Diagnóstico real", NOMES_CLASSES[real], "acertou" if previsto == real else "errou",
              delta_color="normal" if previsto == real else "inverse")
    st.dataframe(registro.T.rename(columns={indice: "valor"}), use_container_width=True)

with aba_manual:
    st.write("Ajuste as 30 características. Os valores iniciais são as medianas da base de teste.")
    medianas = teste[caracteristicas].median()
    valores = {}
    colunas = st.columns(3)
    for i, nome in enumerate(caracteristicas):
        valores[nome] = colunas[i % 3].number_input(
            nome, value=float(medianas[nome]), format="%.5f", key=f"manual_{nome}"
        )
    if st.button("Classificar", type="primary"):
        entrada = pd.DataFrame([valores])[caracteristicas]
        probabilidade = modelo.predict_proba(entrada)[0, 1]
        rotulo = NOMES_CLASSES[int(probabilidade >= 0.5)]
        (st.error if rotulo == "Maligno" else st.success)(
            f"**{rotulo}** — probabilidade de maligno: {probabilidade:.1%}"
        )

with aba_lote:
    st.write(
        "Envie um CSV com as 30 colunas de características (mesmo formato de `database/data.csv`; "
        "colunas extras como `id` e `diagnosis` são ignoradas)."
    )
    arquivo = st.file_uploader("CSV", type="csv")
    if arquivo is not None:
        dados = pd.read_csv(arquivo)
        faltando = [c for c in caracteristicas if c not in dados.columns]
        if faltando:
            st.error(f"Colunas ausentes: {', '.join(faltando)}")
        else:
            resultado = dados.copy()
            resultado["prob_maligno"] = modelo.predict_proba(dados[caracteristicas])[:, 1]
            resultado["previsao"] = (resultado["prob_maligno"] >= 0.5).map({True: "M", False: "B"})
            st.dataframe(resultado, use_container_width=True)
            st.download_button("Baixar resultado", resultado.to_csv(index=False), "previsoes.csv")

with aba_metricas:
    st.write("Desempenho no conjunto de teste (114 registros), reproduzindo o notebook.")
    tabela = pd.DataFrame({
        nome: {
            "Accuracy": m["accuracy"],
            "Recall (maligno)": m["recall"],
            "F1-score": m["f1"],
            "Falsos negativos": m["matriz_confusao"]["fn"],
            "Falsos positivos": m["matriz_confusao"]["fp"],
        }
        for nome, m in metricas.items()
    }).T
    st.dataframe(
        tabela.style.format({"Accuracy": "{:.2%}", "Recall (maligno)": "{:.2%}", "F1-score": "{:.2%}",
                             "Falsos negativos": "{:.0f}", "Falsos positivos": "{:.0f}"}),
        use_container_width=True,
    )

    coeficientes = pd.Series(modelo.named_steps["modelo"].coef_[0], index=caracteristicas)
    top10 = coeficientes.reindex(coeficientes.abs().sort_values(ascending=False).index).head(10)
    st.subheader("10 características com maior peso na Regressão Logística")
    st.bar_chart(top10)
