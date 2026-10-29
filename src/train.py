"""Treina o modelo final do notebook (Regressão Logística, 30 características)
e salva o pipeline, as métricas e o conjunto de teste em models/.

Reproduz exatamente a divisão e o pipeline de tech-challenge.ipynb.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DADOS = RAIZ / "database" / "data.csv"
PASTA_MODELOS = RAIZ / "models"


def carregar_dados():
    dados = pd.read_csv(CAMINHO_DADOS)
    dados = dados.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    # B = 0 (benigno), M = 1 (maligno), igual ao LabelEncoder do notebook
    dados["diagnosis"] = dados["diagnosis"].map({"B": 0, "M": 1})
    x = dados.drop(columns="diagnosis")
    y = dados["diagnosis"]
    return x, y


def avaliar(modelo, x_test, y_test):
    previsto = modelo.predict(x_test)
    vn, fp, fn, vp = confusion_matrix(y_test, previsto, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_test, previsto),
        "recall": recall_score(y_test, previsto, pos_label=1),
        "f1": f1_score(y_test, previsto, pos_label=1),
        "matriz_confusao": {"vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)},
    }


def main():
    x, y = carregar_dados()

    # Divisão final: aproximadamente 60% treino, 20% validação e 20% teste.
    x_desenvolvimento, x_test, y_desenvolvimento, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_desenvolvimento, y_desenvolvimento,
        test_size=0.25, stratify=y_desenvolvimento, random_state=42
    )

    modelo_regressao = Pipeline([
        ("padronizacao", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    modelo_knn = Pipeline([
        ("padronizacao", StandardScaler()),
        ("modelo", KNeighborsClassifier(n_neighbors=5)),
    ])

    modelo_regressao.fit(x_train, y_train)
    modelo_knn.fit(x_train, y_train)

    metricas = {
        "Regressão Logística": avaliar(modelo_regressao, x_test, y_test),
        "KNN (K=5)": avaliar(modelo_knn, x_test, y_test),
    }

    PASTA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(modelo_regressao, PASTA_MODELOS / "modelo_regressao_logistica.joblib")
    (PASTA_MODELOS / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    x_test.assign(diagnosis=y_test).to_csv(PASTA_MODELOS / "conjunto_teste.csv")

    for nome, m in metricas.items():
        print(
            f"{nome}: Accuracy={m['accuracy']:.4f} | "
            f"Recall={m['recall']:.4f} | F1={m['f1']:.4f} | {m['matriz_confusao']}"
        )
    print(f"Modelo salvo em {PASTA_MODELOS}")


if __name__ == "__main__":
    main()
