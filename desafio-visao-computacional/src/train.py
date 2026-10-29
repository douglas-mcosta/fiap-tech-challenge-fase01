"""Treina a CNN de detecção de pneumonia (mesma arquitetura e sementes de
Tech_Challenge_Fase1_Visao_Computacional.ipynb) e salva em models/:

- modelo_cnn_pneumonia.keras: modelo treinado;
- metricas.json: métricas no conjunto de teste;
- ../exemplos/: algumas imagens do teste para experimentar no app.

Baixa o dataset do Kaggle (cerca de 2 GB), então precisa de internet.
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import kagglehub
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "models"
PASTA_EXEMPLOS = RAIZ / "exemplos"

semente_para_reproduzir_os_resultados = 42
tamanho_da_imagem_em_pixels = 128
numero_de_epocas = 35
nomes_das_classes = ["NORMAL", "PNEUMONIA"]  # 0 = NORMAL, 1 = PNEUMONIA
exemplos_por_classe = 5


def fixar_sementes():
    random.seed(semente_para_reproduzir_os_resultados)
    np.random.seed(semente_para_reproduzir_os_resultados)
    tf.keras.utils.set_random_seed(semente_para_reproduzir_os_resultados)
    tf.config.experimental.enable_op_determinism()


def carregar_imagem_em_escala_de_cinza_redimensionada_e_normalizada(caminho_da_imagem):
    imagem_colorida_bgr = cv2.imread(str(caminho_da_imagem))
    imagem_em_escala_de_cinza = cv2.cvtColor(imagem_colorida_bgr, cv2.COLOR_BGR2GRAY)
    imagem_redimensionada = cv2.resize(imagem_em_escala_de_cinza,
                                       (tamanho_da_imagem_em_pixels, tamanho_da_imagem_em_pixels),
                                       interpolation=cv2.INTER_AREA)
    return imagem_redimensionada.astype("float32") / 255


def carregar_todas_as_imagens_de_uma_divisao(pasta_com_treino_e_teste, nome_da_divisao):
    lista_de_imagens, lista_de_classes, lista_de_caminhos = [], [], []
    for numero_da_classe, nome_da_classe in enumerate(nomes_das_classes):
        for caminho_da_imagem in sorted((pasta_com_treino_e_teste / nome_da_divisao / nome_da_classe).glob("*.jpeg")):
            lista_de_imagens.append(carregar_imagem_em_escala_de_cinza_redimensionada_e_normalizada(caminho_da_imagem))
            lista_de_classes.append(numero_da_classe)
            lista_de_caminhos.append(caminho_da_imagem)
    imagens = np.array(lista_de_imagens).reshape(-1, tamanho_da_imagem_em_pixels, tamanho_da_imagem_em_pixels, 1)
    return imagens, np.array(lista_de_classes), lista_de_caminhos


def criar_modelo():
    modelo = models.Sequential([
        layers.Input(shape=(tamanho_da_imagem_em_pixels, tamanho_da_imagem_em_pixels, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def salvar_exemplos(caminhos_de_teste, classes_de_teste, classes_previstas):
    """Copia algumas imagens do teste, reduzidas para no máximo 512 px.

    Em cada classe sorteia 3 acertos e 2 erros do modelo (quando houver), para o
    app mostrar os dois comportamentos e não depender da sorte do sorteio.
    """
    shutil.rmtree(PASTA_EXEMPLOS, ignore_errors=True)
    PASTA_EXEMPLOS.mkdir()
    sorteador = np.random.default_rng(semente_para_reproduzir_os_resultados)
    for numero_da_classe, nome_da_classe in enumerate(nomes_das_classes):
        da_classe = classes_de_teste == numero_da_classe
        acertos = np.flatnonzero(da_classe & (classes_previstas == numero_da_classe))
        erros = np.flatnonzero(da_classe & (classes_previstas != numero_da_classe))
        quantidade_de_erros = min(2, len(erros))
        escolhidos = np.concatenate([
            sorteador.choice(acertos, exemplos_por_classe - quantidade_de_erros, replace=False),
            sorteador.choice(erros, quantidade_de_erros, replace=False),
        ])
        for posicao in escolhidos:
            imagem = cv2.imread(str(caminhos_de_teste[posicao]))
            escala = 512 / max(imagem.shape[:2])
            if escala < 1:
                imagem = cv2.resize(imagem, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(PASTA_EXEMPLOS / f"{nome_da_classe}_{caminhos_de_teste[posicao].name}"), imagem)


def main():
    fixar_sementes()

    pasta_do_dataset_baixado = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    pasta_com_treino_e_teste = [
        pasta.parent for pasta in Path(pasta_do_dataset_baixado).rglob("train")
        if pasta.is_dir() and "__MACOSX" not in str(pasta)
    ][0]
    print("Pasta do dataset:", pasta_com_treino_e_teste)

    imagens_do_treino_original, classes_do_treino_original, _ = carregar_todas_as_imagens_de_uma_divisao(
        pasta_com_treino_e_teste, "train")
    imagens_de_teste, classes_de_teste, caminhos_de_teste = carregar_todas_as_imagens_de_uma_divisao(
        pasta_com_treino_e_teste, "test")

    imagens_de_treino, imagens_de_validacao, classes_de_treino, classes_de_validacao = train_test_split(
        imagens_do_treino_original, classes_do_treino_original,
        test_size=0.2, stratify=classes_do_treino_original,
        random_state=semente_para_reproduzir_os_resultados)

    peso_de_cada_classe_no_treino = {
        numero_da_classe: len(classes_de_treino) / (2 * np.sum(classes_de_treino == numero_da_classe))
        for numero_da_classe in (0, 1)
    }

    modelo = criar_modelo()
    parada_antecipada = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)
    historico = modelo.fit(
        imagens_de_treino, classes_de_treino,
        epochs=numero_de_epocas,
        validation_data=(imagens_de_validacao, classes_de_validacao),
        class_weight=peso_de_cada_classe_no_treino,
        callbacks=[parada_antecipada],
        verbose=2)

    probabilidades = modelo.predict(imagens_de_teste, verbose=0)
    classes_previstas = np.argmax(probabilidades, axis=1)
    vn, fp, fn, vp = confusion_matrix(classes_de_teste, classes_previstas, labels=[0, 1]).ravel()

    metricas = classification_report(classes_de_teste, classes_previstas,
                                     target_names=nomes_das_classes, output_dict=True, zero_division=0)
    metricas["auc_roc"] = float(roc_auc_score(classes_de_teste, probabilidades[:, 1]))
    metricas["matriz_de_confusao"] = {"verdadeiros_negativos": int(vn), "falsos_positivos": int(fp),
                                      "falsos_negativos": int(fn), "verdadeiros_positivos": int(vp)}
    metricas["imagens"] = {"treino": len(classes_de_treino), "validacao": len(classes_de_validacao),
                           "teste": len(classes_de_teste)}
    metricas["melhor_epoca_pela_perda_de_validacao"] = int(np.argmin(historico.history["val_loss"])) + 1
    metricas["epocas_executadas"] = len(historico.history["val_loss"])
    metricas["total_de_parametros"] = int(modelo.count_params())

    PASTA_MODELOS.mkdir(exist_ok=True)
    modelo.save(PASTA_MODELOS / "modelo_cnn_pneumonia.keras")
    (PASTA_MODELOS / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    salvar_exemplos(caminhos_de_teste, classes_de_teste, classes_previstas)

    print(classification_report(classes_de_teste, classes_previstas, target_names=nomes_das_classes, digits=3))
    print("AUC-ROC:", round(metricas["auc_roc"], 3), "|", metricas["matriz_de_confusao"])
    print(f"Modelo salvo em {PASTA_MODELOS}")


if __name__ == "__main__":
    main()
