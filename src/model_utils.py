"""
Utilitas pelatihan model yang dipakai bersama oleh semua profil dataset
(sintetis maupun Asana). Dipisah dari `train_model.py` /
`train_model_asana.py` agar logika pipeline & evaluasi tidak diduplikasi.
"""

from __future__ import annotations

import json
import os

import joblib
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


def build_pipeline(
    feature_columns: list[str],
    categorical_columns: list[str],
    n_estimators: int = 300,
    max_depth: int = 8,
    min_samples_leaf: int = 4,
) -> Pipeline:
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_save(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    model_path: str,
    metrics_path: str,
    data_test_out_path: str | None = None,
    **pipeline_kwargs,
) -> dict:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(feature_columns, categorical_columns, **pipeline_kwargs)
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
        "feature_columns": feature_columns,
    }

    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if data_test_out_path:
        X_test_out = X_test.copy()
        X_test_out[target_column] = y_test.values
        X_test_out["prediksi_probabilitas"] = y_proba
        X_test_out.to_csv(data_test_out_path, index=False)

    return metrics
