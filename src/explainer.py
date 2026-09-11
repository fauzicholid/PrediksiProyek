"""
Modul penjelas (explainer) berbasis SHAP untuk model prediksi keterlambatan
proyek IT.

Modul ini membungkus shap.TreeExplainer agar dapat dipakai langsung di atas
sklearn Pipeline (ColumnTransformer + RandomForestClassifier), dan menangani
perbedaan bentuk output antar versi pustaka `shap` (array 2D vs 3D, dengan
atau tanpa dimensi kelas).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.data_generator import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


@dataclass
class ShapResult:
    """Hasil perhitungan SHAP untuk kelas positif (proyek terlambat)."""

    values: np.ndarray  # shape (n_samples, n_features_transformed)
    base_value: float
    feature_names: list[str]
    transformed_data: pd.DataFrame  # nilai fitur hasil transformasi (untuk plotting)


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=float)


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    return numeric_columns + cat_names


def build_tree_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    model = pipeline.named_steps["model"]
    return shap.TreeExplainer(model)


def _extract_positive_class(values: np.ndarray) -> np.ndarray:
    """Ambil kontribusi SHAP untuk kelas positif (indeks 1 = 'terlambat').

    Menangani beberapa kemungkinan bentuk array yang dikembalikan oleh
    versi-versi `shap` yang berbeda:
      - (n_samples, n_features)               -> sudah untuk 1 output, pakai apa adanya
      - (n_samples, n_features, n_classes)     -> ambil kelas terakhir (indeks 1)
      - (n_classes, n_samples, n_features)     -> ambil kelas terakhir (indeks 1)
    """
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        # Kasus umum shap>=0.42: (n_samples, n_features, n_classes)
        if values.shape[-1] in (2,):
            return values[..., 1]
        # Kasus lama: (n_classes, n_samples, n_features)
        if values.shape[0] in (2,):
            return values[1]
    raise ValueError(f"Bentuk shap values tidak dikenali: {values.shape}")


def _extract_positive_base_value(base_values) -> float:
    base_arr = np.asarray(base_values)
    if base_arr.ndim == 0:
        return float(base_arr)
    if base_arr.ndim == 1:
        # Bisa berupa (n_classes,) atau (n_samples,)
        if base_arr.shape[0] == 2:
            return float(base_arr[1])
        return float(np.mean(base_arr))
    if base_arr.ndim == 2:
        # (n_samples, n_classes)
        return float(np.mean(base_arr[:, -1]))
    return float(np.mean(base_arr))


def compute_shap_values(pipeline: Pipeline, X: pd.DataFrame) -> ShapResult:
    """Hitung SHAP values untuk kelas 'terlambat' pada data X (fitur asli)."""
    X_transformed = transform_features(pipeline, X)
    feature_names = get_feature_names(pipeline)

    explainer = build_tree_explainer(pipeline)
    raw = explainer(X_transformed)

    values = _extract_positive_class(np.asarray(raw.values))
    base_value = _extract_positive_base_value(raw.base_values)

    transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    return ShapResult(
        values=values,
        base_value=base_value,
        feature_names=feature_names,
        transformed_data=transformed_df,
    )


def readable_feature_label(name: str) -> str:
    """Ubah nama fitur teknis (hasil one-hot) menjadi label yang mudah dibaca."""
    labels = {
        "ukuran_tim": "Ukuran Tim",
        "durasi_rencana_hari": "Durasi Rencana (hari)",
        "anggaran_juta": "Anggaran (juta Rp)",
        "kompleksitas": "Kompleksitas Proyek",
        "pengalaman_tim_tahun": "Pengalaman Tim (tahun)",
        "perubahan_requirement": "Jumlah Perubahan Requirement",
        "keterlibatan_klien": "Keterlibatan Klien",
        "risiko_teknologi": "Risiko Teknologi",
        "jumlah_stakeholder": "Jumlah Stakeholder",
        "turnover_tim_persen": "Turnover Tim (%)",
        "ketersediaan_sumber_daya": "Ketersediaan Sumber Daya",
    }
    if name in labels:
        return labels[name]
    if name.startswith("metodologi_"):
        return f"Metodologi = {name.split('metodologi_', 1)[1]}"
    return name
