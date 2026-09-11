"""
Melatih model klasifikasi untuk memprediksi keterlambatan proyek IT.

Model: RandomForestClassifier (dipilih karena kompatibel langsung dengan
shap.TreeExplainer sehingga perhitungan nilai SHAP cepat dan eksak).
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_generator import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    save_dataset,
)

DATA_PATH = "data/proyek_it.csv"
MODEL_PATH = "models/model_keterlambatan.pkl"
METRICS_PATH = "models/metrics.json"


def build_pipeline() -> Pipeline:
    numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    return numeric_columns + cat_names


def main() -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        save_dataset(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "transformed_feature_names": get_transformed_feature_names(pipeline),
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Simpan juga data uji (fitur asli) agar dashboard bisa menampilkan
    # contoh proyek nyata beserta hasil prediksinya.
    X_test_out = X_test.copy()
    X_test_out[TARGET_COLUMN] = y_test.values
    X_test_out["prediksi_probabilitas"] = y_proba
    X_test_out.to_csv("data/data_uji.csv", index=False)

    print("Model berhasil dilatih dan disimpan di", MODEL_PATH)
    print(json.dumps({k: v for k, v in metrics.items() if k not in
                       ("feature_columns", "transformed_feature_names")}, indent=2))


if __name__ == "__main__":
    main()
