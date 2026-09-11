"""
Modul penjelas (explainer) berbasis SHAP untuk model prediksi keterlambatan
proyek IT.

Modul ini membungkus shap.TreeExplainer agar dapat dipakai langsung di atas
sklearn Pipeline (ColumnTransformer + RandomForestClassifier), dan menangani
perbedaan bentuk output antar versi pustaka `shap` (array 2D vs 3D, dengan
atau tanpa dimensi kelas).

Modul ini sengaja dibuat **tidak terikat pada satu dataset**: nama kolom
numerik & kategorikal diambil langsung dari `ColumnTransformer` yang sudah
di-fit di dalam pipeline (bukan di-import dari modul dataset tertentu),
sehingga bisa dipakai untuk profil dataset manapun (data sintetis, data
Asana, atau dataset lain di masa depan) selama pipeline-nya mengikuti pola
`Pipeline([("preprocessor", ColumnTransformer([("num", "passthrough", ...),
("cat", OneHotEncoder(), ...)])), ("model", ...)])`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


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
    """Ambil nama fitur hasil transformasi langsung dari ColumnTransformer
    yang sudah di-fit, tanpa bergantung pada konstanta dataset eksternal."""
    preprocessor = pipeline.named_steps["preprocessor"]
    names: list[str] = []
    for trans_name, transformer, columns in preprocessor.transformers_:
        if trans_name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(columns))
        else:
            names.extend(columns)
    return names


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


def readable_feature_label(name: str, label_map: dict[str, str] | None = None) -> str:
    """Ubah nama fitur teknis (hasil one-hot, mis. `tim_Backend Engineering`)
    menjadi label yang mudah dibaca, memakai `label_map` khusus dataset bila
    tersedia (lihat `FEATURE_LABELS` pada masing-masing modul dataset)."""
    label_map = label_map or {}
    if name in label_map:
        return label_map[name]
    # Nama kolom one-hot berbentuk "<kolom_asli>_<nilai_kategori>" - cocokkan
    # nama kolom asli terpanjang yang jadi prefix (agar kolom multi-kata
    # seperti "status_proyek" tidak salah kepotong di underscore pertama).
    for base in sorted(label_map, key=len, reverse=True):
        if name.startswith(base + "_"):
            suffix = name[len(base) + 1 :]
            return f"{label_map[base]} = {suffix}"
    return name.replace("_", " ").capitalize()
