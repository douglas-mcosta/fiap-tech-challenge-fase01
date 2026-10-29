"""Interface web para testar a CNN de detecção de pneumonia treinada em src/train.py."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "models"
PASTA_EXEMPLOS = RAIZ / "exemplos"
tamanho_da_imagem_em_pixels = 128
nomes_das_classes = ["NORMAL", "PNEUMONIA"]

st.set_page_config(page_title="Tech Challenge Fase 1 · Pneumonia", page_icon="🫁", layout="wide")


@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model(PASTA_MODELOS / "modelo_cnn_pneumonia.keras")


@st.cache_data
def carregar_metricas():
    return json.loads((PASTA_MODELOS / "metricas.json").read_text(encoding="utf-8"))


def preparar_imagem(conteudo):
    """Mesmo pré-processamento do treino: cinza, 128x128, valores entre 0 e 1."""
    imagem_colorida_bgr = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    imagem_em_escala_de_cinza = cv2.cvtColor(imagem_colorida_bgr, cv2.COLOR_BGR2GRAY)
    imagem_redimensionada = cv2.resize(imagem_em_escala_de_cinza,
                                       (tamanho_da_imagem_em_pixels, tamanho_da_imagem_em_pixels),
                                       interpolation=cv2.INTER_AREA)
    entrada = imagem_redimensionada.astype("float32") / 255
    return imagem_em_escala_de_cinza, entrada.reshape(1, tamanho_da_imagem_em_pixels, tamanho_da_imagem_em_pixels, 1)


def mostrar_previsao(conteudo, rotulo_real=None):
    imagem_original, entrada = preparar_imagem(conteudo)
    probabilidade = float(modelo.predict(entrada, verbose=0)[0, 1])
    previsto = nomes_das_classes[int(probabilidade >= 0.5)]

    c1, c2 = st.columns([1, 1])
    c1.image(imagem_original, caption="Imagem analisada", width=420, clamp=True)
    with c2:
        (st.error if previsto == "PNEUMONIA" else st.success)(f"**Previsão: {previsto}**")
        st.metric("Probabilidade de pneumonia", f"{probabilidade:.1%}")
        st.progress(probabilidade)
        if rotulo_real is not None:
            st.metric("Diagnóstico real", rotulo_real, "acertou" if previsto == rotulo_real else "errou",
                      delta_color="normal" if previsto == rotulo_real else "inverse")


modelo = carregar_modelo()
metricas = carregar_metricas()

st.title("🫁 Detecção de pneumonia em raio-X de tórax")
st.caption(
    "Tech Challenge — Fase 1 · Desafio de visão computacional · Chest X-Ray Images (Pneumonia) · "
    "CNN treinada com TensorFlow/Keras. Ferramenta de apoio: a decisão final é sempre de um profissional de saúde."
)

aba_exemplos, aba_envio, aba_metricas = st.tabs(["Exemplos do teste", "Enviar imagem", "Métricas"])

with aba_exemplos:
    st.write(
        "Imagens do conjunto de teste (não usadas no treino). O nome indica o diagnóstico real. "
        "Há acertos e erros do modelo em cada classe: o principal erro é classificar raio-X normais como pneumonia."
    )
    exemplos = sorted(PASTA_EXEMPLOS.glob("*.jpeg"))
    escolhido = st.selectbox("Imagem", exemplos, format_func=lambda caminho: caminho.name)
    mostrar_previsao(escolhido.read_bytes(), rotulo_real=escolhido.name.split("_")[0])

with aba_envio:
    arquivo = st.file_uploader("Raio-X de tórax (JPEG ou PNG)", type=["jpeg", "jpg", "png"])
    if arquivo is not None:
        mostrar_previsao(arquivo.getvalue())

with aba_metricas:
    matriz = metricas["matriz_de_confusao"]
    st.write(f"Desempenho no conjunto de teste ({metricas['imagens']['teste']} imagens).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acurácia", f"{metricas['accuracy']:.1%}")
    c2.metric("Recall (pneumonia)", f"{metricas['PNEUMONIA']['recall']:.1%}")
    c3.metric("F1 (pneumonia)", f"{metricas['PNEUMONIA']['f1-score']:.1%}")
    c4.metric("AUC-ROC", f"{metricas['auc_roc']:.3f}")

    st.subheader("Matriz de confusão")
    st.table(pd.DataFrame(
        [[matriz["verdadeiros_negativos"], matriz["falsos_positivos"]],
         [matriz["falsos_negativos"], matriz["verdadeiros_positivos"]]],
        index=["Real: NORMAL", "Real: PNEUMONIA"],
        columns=["Previsto: NORMAL", "Previsto: PNEUMONIA"],
    ))
    st.caption(
        f"Falsos negativos (pneumonia não detectada): {matriz['falsos_negativos']} · "
        f"Falsos positivos: {matriz['falsos_positivos']} · "
        f"Melhor época: {metricas['melhor_epoca_pela_perda_de_validacao']} de {metricas['epocas_executadas']} · "
        f"Parâmetros: {metricas['total_de_parametros']:,}".replace(",", ".")
    )
