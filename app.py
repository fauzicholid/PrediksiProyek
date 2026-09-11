"""
Dashboard Model Prediksi Keterlambatan Proyek IT dengan Pendekatan
Explainable AI (XAI) Menggunakan SHAP.

Tujuan dashboard ini adalah mendukung transparansi keputusan manajer
proyek: selain memberikan prediksi probabilitas keterlambatan, dashboard
menjelaskan *mengapa* model memberikan prediksi tersebut melalui nilai
SHAP (SHapley Additive exPlanations), baik secara lokal (per proyek)
maupun global (pola umum pada seluruh data).
"""

from __future__ import annotations

import json

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
from src.data_generator import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    METODOLOGI_OPTIONS,
    TARGET_COLUMN,
)
from src.explainer import compute_shap_values, readable_feature_label
from src.train_model import DATA_PATH, METRICS_PATH, MODEL_PATH, main as train_main

st.set_page_config(
    page_title="Prediksi Keterlambatan Proyek IT (XAI - SHAP)",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Pemuatan aset (model, data, metrik) - di-cache agar dashboard responsif
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Menyiapkan model & data...")
def load_assets():
    import os

    if not (os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH)):
        train_main()

    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    return pipeline, df, metrics


@st.cache_resource(show_spinner="Menghitung nilai SHAP global...")
def compute_global_shap(_pipeline, df: pd.DataFrame, sample_size: int = 400):
    sample = df.sample(n=min(sample_size, len(df)), random_state=7)
    X_sample = sample[FEATURE_COLUMNS]
    result = compute_shap_values(_pipeline, X_sample)
    return result, X_sample.reset_index(drop=True)


pipeline, dataset, metrics = load_assets()
shap_global, shap_sample_X = compute_global_shap(pipeline, dataset)
readable_labels = [readable_feature_label(name) for name in shap_global.feature_names]

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


# ---------------------------------------------------------------------------
# Sidebar navigasi
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Navigasi Dashboard")
page = st.sidebar.radio(
    "Pilih halaman",
    [
        "🏠 Beranda",
        "🔮 Prediksi Proyek Baru",
        "🧠 Analisis Global (XAI)",
        "📈 Evaluasi Model",
        "🔍 Eksplorasi Data",
        "📁 Prediksi Batch (CSV)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dashboard ini dibangun untuk mendukung **transparansi keputusan** "
    "manajer proyek IT melalui pendekatan *Explainable AI* (SHAP) di atas "
    "model klasifikasi Random Forest."
)


# ---------------------------------------------------------------------------
# Halaman: Beranda
# ---------------------------------------------------------------------------
if page == "🏠 Beranda":
    st.title("📊 Model Prediksi Keterlambatan Proyek IT")
    st.subheader("Pendekatan Explainable AI (XAI) Menggunakan SHAP")

    st.markdown(
        """
        Dashboard ini membantu **manajer proyek IT** memperkirakan risiko
        keterlambatan sebuah proyek, sekaligus memahami **faktor-faktor apa
        saja yang mendorong prediksi tersebut** — bukan sekadar angka
        probabilitas dari "kotak hitam". Transparansi ini dicapai dengan
        menghitung nilai **SHAP (SHapley Additive exPlanations)** dari model
        Random Forest yang dilatih pada data historis proyek.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Data Proyek", f"{len(dataset):,}")
    col2.metric("Akurasi Model", f"{metrics['accuracy'] * 100:.1f}%")
    col3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col4.metric(
        "Proporsi Proyek Terlambat",
        f"{dataset[TARGET_COLUMN].mean() * 100:.1f}%",
    )

    st.markdown("---")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Bagaimana cara memakai dashboard ini?")
        st.markdown(
            """
            1. **🔮 Prediksi Proyek Baru** — masukkan karakteristik proyek
               (ukuran tim, anggaran, kompleksitas, dsb.) untuk melihat
               probabilitas keterlambatan beserta penjelasan SHAP per proyek.
            2. **🧠 Analisis Global (XAI)** — lihat pola umum: fitur apa yang
               paling memengaruhi keterlambatan proyek pada seluruh dataset.
            3. **📈 Evaluasi Model** — tinjau performa model (akurasi,
               precision, recall, confusion matrix) sebelum dipakai sebagai
               dasar keputusan.
            4. **🔍 Eksplorasi Data** — telusuri karakteristik data historis
               yang menjadi dasar pelatihan model.
            5. **📁 Prediksi Batch (CSV)** — unggah beberapa proyek sekaligus
               untuk diprediksi secara massal.
            """
        )
    with right:
        st.markdown("#### Mengapa Explainable AI penting?")
        st.info(
            "Model *black-box* yang akurat tapi tidak bisa dijelaskan sulit "
            "dipercaya untuk pengambilan keputusan manajerial. Dengan SHAP, "
            "setiap prediksi dapat diuraikan menjadi kontribusi tiap faktor, "
            "sehingga manajer proyek dapat memvalidasi apakah alasan model "
            "masuk akal dan menentukan tindakan mitigasi yang tepat sasaran."
        )


# ---------------------------------------------------------------------------
# Halaman: Prediksi Proyek Baru
# ---------------------------------------------------------------------------
elif page == "🔮 Prediksi Proyek Baru":
    st.title("🔮 Prediksi Keterlambatan Proyek Baru")
    st.caption("Masukkan karakteristik proyek untuk memprediksi risiko keterlambatan.")

    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
            ukuran_tim = st.slider("Ukuran Tim (orang)", 3, 40, 10)
            durasi_rencana_hari = st.slider("Durasi Rencana (hari)", 30, 365, 120)
            anggaran_juta = st.number_input("Anggaran (juta Rp)", min_value=50.0, value=500.0, step=10.0)
            kompleksitas = st.slider("Kompleksitas Proyek (1=Rendah, 5=Sangat Tinggi)", 1, 5, 3)
        with c2:
            pengalaman_tim_tahun = st.slider("Rata-rata Pengalaman Tim (tahun)", 0.0, 15.0, 3.0, step=0.5)
            perubahan_requirement = st.slider("Jumlah Perubahan Requirement", 0, 20, 4)
            keterlibatan_klien = st.slider("Keterlibatan Klien (1=Rendah, 5=Sangat Aktif)", 1, 5, 3)
            risiko_teknologi = st.slider("Risiko Teknologi (1=Rendah, 5=Sangat Tinggi)", 1, 5, 3)
        with c3:
            jumlah_stakeholder = st.slider("Jumlah Stakeholder", 2, 20, 6)
            turnover_tim_persen = st.slider("Turnover Tim (%)", 0.0, 60.0, 10.0, step=1.0)
            ketersediaan_sumber_daya = st.slider("Ketersediaan Sumber Daya (1=Rendah, 5=Sangat Baik)", 1, 5, 3)
            metodologi = st.selectbox("Metodologi Pengembangan", METODOLOGI_OPTIONS, index=1)

        submitted = st.form_submit_button("Prediksi Sekarang", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "ukuran_tim": ukuran_tim,
                    "durasi_rencana_hari": durasi_rencana_hari,
                    "anggaran_juta": anggaran_juta,
                    "kompleksitas": kompleksitas,
                    "pengalaman_tim_tahun": pengalaman_tim_tahun,
                    "perubahan_requirement": perubahan_requirement,
                    "keterlibatan_klien": keterlibatan_klien,
                    "risiko_teknologi": risiko_teknologi,
                    "jumlah_stakeholder": jumlah_stakeholder,
                    "turnover_tim_persen": turnover_tim_persen,
                    "ketersediaan_sumber_daya": ketersediaan_sumber_daya,
                    "metodologi": metodologi,
                }
            ]
        )

        probability = float(pipeline.predict_proba(input_df[FEATURE_COLUMNS])[0, 1])
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

        local_result = compute_shap_values(pipeline, input_df[FEATURE_COLUMNS])
        local_labels = [readable_feature_label(name) for name in local_result.feature_names]
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
            narasi.append(f"- {local_labels[idx]} {arah} risiko keterlambatan proyek ini.")
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
    st.title("🧠 Analisis Global: Faktor Utama Keterlambatan Proyek")
    st.caption(
        f"Dihitung dari nilai SHAP pada {len(shap_sample_X)} sampel proyek "
        "untuk melihat pola keseluruhan (bukan hanya satu proyek)."
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
            "Setiap titik mewakili satu proyek. Warna merah = nilai fitur "
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
        **Cara membaca confusion matrix:** baris menunjukkan kondisi aktual
        proyek, kolom menunjukkan hasil prediksi model. Sel diagonal
        (kiri-atas dan kanan-bawah) adalah prediksi yang benar.
        """
    )

    st.warning(
        "⚠️ Model ini dilatih pada **data sintetis** yang disusun berdasarkan "
        "pola umum manajemen proyek untuk keperluan demonstrasi. Sebelum "
        "dipakai pada pengambilan keputusan nyata, latih ulang model "
        "menggunakan data historis proyek IT organisasi Anda sendiri "
        "(lihat `src/train_model.py`)."
    )


# ---------------------------------------------------------------------------
# Halaman: Eksplorasi Data
# ---------------------------------------------------------------------------
elif page == "🔍 Eksplorasi Data":
    st.title("🔍 Eksplorasi Data Historis Proyek")

    st.dataframe(dataset.head(20), use_container_width=True)

    numeric_features = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    col_a, col_b = st.columns([2, 1])
    with col_a:
        selected_feature = st.selectbox("Pilih fitur numerik untuk dilihat distribusinya", numeric_features)
        st.plotly_chart(
            feature_distribution(dataset, selected_feature, TARGET_COLUMN),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(category_share_pie(dataset, "metodologi"), use_container_width=True)

    st.markdown("#### Ringkasan Statistik")
    st.dataframe(dataset[numeric_features].describe().T, use_container_width=True)


# ---------------------------------------------------------------------------
# Halaman: Prediksi Batch (CSV)
# ---------------------------------------------------------------------------
elif page == "📁 Prediksi Batch (CSV)":
    st.title("📁 Prediksi Batch untuk Beberapa Proyek Sekaligus")
    st.markdown(
        "Unggah file CSV berisi kolom berikut: `"
        + "`, `".join(FEATURE_COLUMNS)
        + "`"
    )

    template_csv = dataset[FEATURE_COLUMNS].head(5).to_csv(index=False)
    st.download_button(
        "⬇️ Unduh Contoh Template CSV",
        data=template_csv,
        file_name="template_prediksi_proyek.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan pada file: {missing_cols}")
            else:
                X_batch = batch_df[FEATURE_COLUMNS]
                probs = pipeline.predict_proba(X_batch)[:, 1]
                batch_result = batch_df.copy()
                batch_result["probabilitas_terlambat"] = probs.round(3)
                batch_result["kategori_risiko"] = [risk_category(p)[0] for p in probs]

                st.success(f"Berhasil memprediksi {len(batch_result)} proyek.")
                st.dataframe(batch_result, use_container_width=True)

                st.download_button(
                    "⬇️ Unduh Hasil Prediksi (CSV)",
                    data=batch_result.to_csv(index=False),
                    file_name="hasil_prediksi_batch.csv",
                    mime="text/csv",
                )

                st.markdown("#### Ringkasan Kepentingan Fitur untuk Batch Ini")
                batch_shap = compute_shap_values(pipeline, X_batch)
                batch_labels = [readable_feature_label(name) for name in batch_shap.feature_names]
                mean_abs = np.abs(batch_shap.values).mean(axis=0)
                st.plotly_chart(
                    global_importance_bar(batch_labels, mean_abs),
                    use_container_width=True,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal memproses file: {exc}")
