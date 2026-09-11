"""
Dashboard Model Prediksi Keterlambatan Proyek IT dengan Pendekatan
Explainable AI (XAI) Menggunakan SHAP.

Tujuan dashboard ini adalah mendukung transparansi keputusan manajer
proyek: selain memberikan prediksi probabilitas keterlambatan, dashboard
menjelaskan *mengapa* model memberikan prediksi tersebut melalui nilai
SHAP (SHapley Additive exPlanations), baik secara lokal (per proyek/task)
maupun global (pola umum pada seluruh data).

Dashboard mendukung lebih dari satu **sumber data** (lihat
`src/profiles.py`) - saat ini data sintetis (demo) dan data task Asana
(simulasi enterprise) - dipilih lewat sidebar. Seluruh halaman di bawah
ditulis generik terhadap profil yang aktif, bukan hardcode ke satu
skema fitur.
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.charts import (
    category_share_pie,
    confusion_matrix_heatmap,
    feature_distribution,
    global_importance_bar,
    local_shap_bar,
    probability_bar,
)
from src.explainer import compute_shap_values, readable_feature_label
from src.profiles import PROFILES, DatasetProfile, FieldSpec

st.set_page_config(
    page_title="Prediksi Keterlambatan Proyek IT (XAI - SHAP)",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Pemuatan aset (model, data, metrik) - di-cache per profil agar responsif
# ---------------------------------------------------------------------------
def _ensure_trained(profile: DatasetProfile) -> None:
    if os.path.exists(profile.model_path) and os.path.exists(profile.data_path):
        return
    if profile.key == "sintetis":
        from src.train_model import main as train_main
    else:
        from src.train_model_asana import main as train_main
    train_main()


@st.cache_resource(show_spinner="Menyiapkan model & data...")
def load_assets(profile_key: str):
    profile = PROFILES[profile_key]
    _ensure_trained(profile)

    pipeline = joblib.load(profile.model_path)
    df = pd.read_csv(profile.data_path)
    with open(profile.metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    return pipeline, df, metrics


@st.cache_resource(show_spinner="Menghitung nilai SHAP global...")
def compute_global_shap(_pipeline, profile_key: str, df: pd.DataFrame, sample_size: int = 400):
    profile = PROFILES[profile_key]
    sample = df.sample(n=min(sample_size, len(df)), random_state=7)
    X_sample = sample[profile.feature_columns]
    result = compute_shap_values(_pipeline, X_sample)
    return result, X_sample.reset_index(drop=True)


RISK_LABELS = {
    "low": ("Risiko Rendah", "🟢"),
    "medium": ("Risiko Sedang", "🟡"),
    "high": ("Risiko Tinggi", "🔴"),
}


def risk_category(probability: float) -> tuple[str, str]:
    if probability < 0.35:
        return RISK_LABELS["low"]
    if probability < 0.65:
        return RISK_LABELS["medium"]
    return RISK_LABELS["high"]


def render_field(field_spec: FieldSpec):
    """Render satu widget input Streamlit sesuai FieldSpec, kembalikan nilainya."""
    if field_spec.kind == "slider_int":
        return st.slider(field_spec.label, int(field_spec.min), int(field_spec.max), int(field_spec.default))
    if field_spec.kind == "slider_float":
        return st.slider(
            field_spec.label, float(field_spec.min), float(field_spec.max),
            float(field_spec.default), step=float(field_spec.step or 1.0),
        )
    if field_spec.kind == "number":
        return st.number_input(
            field_spec.label, min_value=float(field_spec.min),
            value=float(field_spec.default), step=float(field_spec.step or 1.0),
        )
    if field_spec.kind == "select":
        default_idx = field_spec.options.index(field_spec.default) if field_spec.default in field_spec.options else 0
        return st.selectbox(field_spec.label, field_spec.options, index=default_idx)
    if field_spec.kind == "checkbox":
        return int(st.checkbox(field_spec.label, value=bool(field_spec.default)))
    raise ValueError(f"Jenis field tidak dikenal: {field_spec.kind}")


# ---------------------------------------------------------------------------
# Sidebar: pilih sumber data & navigasi
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Navigasi Dashboard")

profile_key = st.sidebar.selectbox(
    "🗂️ Sumber Data",
    list(PROFILES.keys()),
    format_func=lambda k: PROFILES[k].display_name,
)
profile = PROFILES[profile_key]

page = st.sidebar.radio(
    "Pilih halaman",
    [
        "🏠 Beranda",
        f"🔮 Prediksi {profile.unit_label.capitalize()} Baru",
        "🧠 Analisis Global (XAI)",
        "📈 Evaluasi Model",
        "🔍 Eksplorasi Data",
        "📁 Prediksi Batch (CSV)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(profile.caveat)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Dashboard ini dibangun untuk mendukung **transparansi keputusan** "
    "manajer proyek IT melalui pendekatan *Explainable AI* (SHAP) di atas "
    "model klasifikasi Random Forest."
)

pipeline, dataset, metrics = load_assets(profile_key)
shap_global, shap_sample_X = compute_global_shap(pipeline, profile_key, dataset)
readable_labels = [readable_feature_label(name, profile.feature_labels) for name in shap_global.feature_names]


# ---------------------------------------------------------------------------
# Halaman: Beranda
# ---------------------------------------------------------------------------
if page == "🏠 Beranda":
    st.title("📊 Model Prediksi Keterlambatan Proyek IT")
    st.subheader("Pendekatan Explainable AI (XAI) Menggunakan SHAP")

    st.markdown(
        """
        Dashboard ini membantu **manajer proyek IT** memperkirakan risiko
        keterlambatan, sekaligus memahami **faktor-faktor apa saja yang
        mendorong prediksi tersebut** — bukan sekadar angka probabilitas
        dari "kotak hitam". Transparansi ini dicapai dengan menghitung
        nilai **SHAP (SHapley Additive exPlanations)** dari model Random
        Forest yang dilatih pada data historis.
        """
    )

    st.info(f"**Sumber data aktif:** {profile.display_name}\n\n{profile.description}")
    st.warning(profile.caveat)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Jumlah Data {profile.unit_label.capitalize()}", f"{len(dataset):,}")
    col2.metric("Akurasi Model", f"{metrics['accuracy'] * 100:.1f}%")
    col3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col4.metric(
        f"Proporsi {profile.unit_label.capitalize()} Terlambat",
        f"{dataset[profile.target_column].mean() * 100:.1f}%",
    )

    st.markdown("---")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Bagaimana cara memakai dashboard ini?")
        st.markdown(
            f"""
            1. **🔮 Prediksi {profile.unit_label.capitalize()} Baru** — masukkan
               karakteristik untuk melihat probabilitas keterlambatan beserta
               penjelasan SHAP.
            2. **🧠 Analisis Global (XAI)** — lihat pola umum: fitur apa yang
               paling memengaruhi keterlambatan pada seluruh dataset.
            3. **📈 Evaluasi Model** — tinjau performa model (akurasi,
               precision, recall, confusion matrix) sebelum dipakai sebagai
               dasar keputusan.
            4. **🔍 Eksplorasi Data** — telusuri karakteristik data historis
               yang menjadi dasar pelatihan model.
            5. **📁 Prediksi Batch (CSV)** — unggah beberapa data sekaligus
               untuk diprediksi secara massal.

            Ganti **Sumber Data** di sidebar untuk membandingkan bagaimana
            model & penjelasan SHAP berubah antar dataset.
            """
        )
    with right:
        st.markdown("#### Mengapa Explainable AI penting?")
        st.info(
            "Model *black-box* yang akurat tapi tidak bisa dijelaskan sulit "
            "dipercaya untuk pengambilan keputusan manajerial. Dengan SHAP, "
            "setiap prediksi dapat diuraikan menjadi kontribusi tiap faktor, "
            "sehingga manajer proyek dapat memvalidasi apakah alasan model "
            "masuk akal dan menentukan tindakan mitigasi yang tepat sasaran — "
            "termasuk saat jawabannya adalah faktor yang diduga penting "
            "ternyata tidak berpengaruh (lihat sumber data Asana)."
        )


# ---------------------------------------------------------------------------
# Halaman: Prediksi Baru
# ---------------------------------------------------------------------------
elif page.startswith("🔮 Prediksi"):
    st.title(f"🔮 Prediksi Keterlambatan {profile.unit_label.capitalize()} Baru")
    st.caption(f"Masukkan karakteristik {profile.unit_label} untuk memprediksi risiko keterlambatan.")

    with st.form("form_prediksi"):
        cols = st.columns(3)
        values = {}
        for i, field_spec in enumerate(profile.fields):
            with cols[i % 3]:
                values[field_spec.name] = render_field(field_spec)

        submitted = st.form_submit_button("Prediksi Sekarang", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([values])

        probability = float(pipeline.predict_proba(input_df[profile.feature_columns])[0, 1])
        label, emoji = risk_category(probability)

        st.markdown("### Hasil Prediksi")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Probabilitas Terlambat", f"{probability * 100:.1f}%")
            st.markdown(f"### {emoji} {label}")
        with r2:
            st.plotly_chart(probability_bar(probability), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧠 Penjelasan SHAP (Mengapa model memberi prediksi ini?)")

        local_result = compute_shap_values(pipeline, input_df[profile.feature_columns])
        local_labels = [readable_feature_label(name, profile.feature_labels) for name in local_result.feature_names]
        local_feature_values = list(local_result.transformed_data.iloc[0])

        st.plotly_chart(
            local_shap_bar(local_labels, local_result.values[0], local_feature_values),
            use_container_width=True,
        )

        # Narasi otomatis top-3 faktor
        order = np.argsort(-np.abs(local_result.values[0]))[:3]
        narasi = []
        for idx in order:
            arah = "**meningkatkan**" if local_result.values[0][idx] > 0 else "**menurunkan**"
            narasi.append(f"- {local_labels[idx]} {arah} risiko keterlambatan {profile.unit_label} ini.")
        st.markdown("**Ringkasan faktor paling berpengaruh:**\n" + "\n".join(narasi))

        with st.expander("Lihat SHAP Waterfall Plot (format standar SHAP)"):
            explanation = shap.Explanation(
                values=local_result.values[0],
                base_values=local_result.base_value,
                data=local_result.transformed_data.iloc[0].values,
                feature_names=local_labels,
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(fig, clear_figure=True)


# ---------------------------------------------------------------------------
# Halaman: Analisis Global (XAI)
# ---------------------------------------------------------------------------
elif page == "🧠 Analisis Global (XAI)":
    st.title(f"🧠 Analisis Global: Faktor Utama Keterlambatan {profile.unit_label.capitalize()}")
    st.caption(
        f"Dihitung dari nilai SHAP pada {len(shap_sample_X)} sampel {profile.unit_label} "
        "untuk melihat pola keseluruhan (bukan hanya satu data)."
    )

    mean_abs_shap = np.abs(shap_global.values).mean(axis=0)
    st.plotly_chart(
        global_importance_bar(readable_labels, mean_abs_shap),
        use_container_width=True,
    )
    st.caption(
        "Semakin besar rata-rata |nilai SHAP| suatu fitur, semakin besar "
        "pengaruh fitur tersebut terhadap prediksi keterlambatan secara umum."
    )

    with st.expander("Lihat SHAP Summary Plot / Beeswarm (format standar SHAP)"):
        explanation = shap.Explanation(
            values=shap_global.values,
            base_values=np.full(len(shap_sample_X), shap_global.base_value),
            data=shap_global.transformed_data.values,
            feature_names=readable_labels,
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.plots.beeswarm(explanation, max_display=14, show=False)
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Setiap titik mewakili satu data. Warna merah = nilai fitur "
            "tinggi, biru = nilai fitur rendah. Posisi ke kanan berarti "
            "fitur tersebut mendorong probabilitas keterlambatan naik."
        )


# ---------------------------------------------------------------------------
# Halaman: Evaluasi Model
# ---------------------------------------------------------------------------
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Performa Model")
    st.caption(
        f"Model dievaluasi pada {metrics['n_test']} data uji yang tidak "
        f"dipakai saat pelatihan ({metrics['n_train']} data latih)."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Akurasi", f"{metrics['accuracy'] * 100:.1f}%")
    c2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    c3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    c4.metric("F1-Score", f"{metrics['f1_score'] * 100:.1f}%")
    c5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.plotly_chart(
        confusion_matrix_heatmap(metrics["confusion_matrix"], ["Tepat Waktu", "Terlambat"]),
        use_container_width=True,
    )

    st.markdown(
        """
        **Cara membaca confusion matrix:** baris menunjukkan kondisi aktual,
        kolom menunjukkan hasil prediksi model. Sel diagonal (kiri-atas dan
        kanan-bawah) adalah prediksi yang benar.
        """
    )

    st.warning(profile.caveat)


# ---------------------------------------------------------------------------
# Halaman: Eksplorasi Data
# ---------------------------------------------------------------------------
elif page == "🔍 Eksplorasi Data":
    st.title("🔍 Eksplorasi Data Historis")

    st.dataframe(dataset.head(20), use_container_width=True)

    numeric_features = [c for c in profile.feature_columns if c not in profile.categorical_columns]
    col_a, col_b = st.columns([2, 1])
    with col_a:
        selected_feature = st.selectbox("Pilih fitur numerik untuk dilihat distribusinya", numeric_features)
        st.plotly_chart(
            feature_distribution(dataset, selected_feature, profile.target_column),
            use_container_width=True,
        )
    with col_b:
        if profile.categorical_columns:
            cat_choice = st.selectbox("Pilih fitur kategorikal", profile.categorical_columns)
            st.plotly_chart(category_share_pie(dataset, cat_choice), use_container_width=True)

    st.markdown("#### Ringkasan Statistik")
    st.dataframe(dataset[numeric_features].describe().T, use_container_width=True)


# ---------------------------------------------------------------------------
# Halaman: Prediksi Batch (CSV)
# ---------------------------------------------------------------------------
elif page == "📁 Prediksi Batch (CSV)":
    st.title(f"📁 Prediksi Batch untuk Beberapa {profile.unit_label.capitalize()} Sekaligus")
    st.markdown(
        "Unggah file CSV berisi kolom berikut: `"
        + "`, `".join(profile.feature_columns)
        + "`"
    )

    template_csv = dataset[profile.feature_columns].head(5).to_csv(index=False)
    st.download_button(
        "⬇️ Unduh Contoh Template CSV",
        data=template_csv,
        file_name=f"template_prediksi_{profile.key}.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"], key=f"upload_{profile.key}")
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in profile.feature_columns if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan pada file: {missing_cols}")
            else:
                X_batch = batch_df[profile.feature_columns]
                probs = pipeline.predict_proba(X_batch)[:, 1]
                batch_result = batch_df.copy()
                batch_result["probabilitas_terlambat"] = probs.round(3)
                batch_result["kategori_risiko"] = [risk_category(p)[0] for p in probs]

                st.success(f"Berhasil memprediksi {len(batch_result)} {profile.unit_label}.")
                st.dataframe(batch_result, use_container_width=True)

                st.download_button(
                    "⬇️ Unduh Hasil Prediksi (CSV)",
                    data=batch_result.to_csv(index=False),
                    file_name="hasil_prediksi_batch.csv",
                    mime="text/csv",
                )

                st.markdown("#### Ringkasan Kepentingan Fitur untuk Batch Ini")
                batch_shap = compute_shap_values(pipeline, X_batch)
                batch_labels = [readable_feature_label(name, profile.feature_labels) for name in batch_shap.feature_names]
                mean_abs = np.abs(batch_shap.values).mean(axis=0)
                st.plotly_chart(
                    global_importance_bar(batch_labels, mean_abs),
                    use_container_width=True,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal memproses file: {exc}")
