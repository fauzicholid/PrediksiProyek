# Prediksi Keterlambatan Proyek IT — Dashboard XAI (SHAP)

Dashboard interaktif untuk memprediksi **risiko keterlambatan proyek IT**
menggunakan model machine learning (Random Forest), dilengkapi pendekatan
**Explainable AI (XAI)** berbasis **SHAP (SHapley Additive exPlanations)**
agar setiap prediksi dapat dijelaskan secara transparan kepada manajer
proyek — bukan sekadar keluaran "kotak hitam".

## Fitur Dashboard

1. **🏠 Beranda** — ringkasan model dan cara pakai dashboard.
2. **🔮 Prediksi Proyek Baru** — form input karakteristik proyek, hasil
   probabilitas keterlambatan, kategori risiko, serta **penjelasan SHAP
   lokal** (bar chart kontribusi fitur + SHAP waterfall plot standar).
3. **🧠 Analisis Global (XAI)** — kepentingan fitur secara global
   (rata-rata |SHAP value|) dan SHAP summary/beeswarm plot untuk melihat
   pola keseluruhan dataset.
4. **📈 Evaluasi Model** — akurasi, precision, recall, F1-score, ROC-AUC,
   dan confusion matrix pada data uji.
5. **🔍 Eksplorasi Data** — distribusi fitur historis proyek.
6. **📁 Prediksi Batch (CSV)** — unggah banyak proyek sekaligus untuk
   diprediksi dan dijelaskan secara massal.

## Struktur Proyek

```
PrediksiProyek/
├── app.py                     # Dashboard Streamlit (entry point)
├── requirements.txt
├── src/
│   ├── data_generator.py      # Pembangkit dataset sintetis proyek IT
│   ├── train_model.py         # Pipeline pelatihan model + evaluasi
│   ├── explainer.py           # Wrapper perhitungan SHAP (lokal & global)
│   └── charts.py              # Chart Plotly dashboard (palet konsisten)
├── data/
│   ├── proyek_it.csv          # Dataset historis (sintetis)
│   └── data_uji.csv           # Subset data uji + prediksi
└── models/
    ├── model_keterlambatan.pkl
    └── metrics.json
```

## Menjalankan Dashboard

```bash
pip install -r requirements.txt

# (opsional) latih ulang model — otomatis dijalankan saat dashboard
# pertama kali dibuka jika model belum ada
python -m src.train_model

streamlit run app.py
```

Dashboard akan terbuka di `http://localhost:8501`.

## Tentang Data & Model

Dataset yang disertakan bersifat **sintetis** (dibangkitkan oleh
`src/data_generator.py`) dengan hubungan antar-fitur yang disusun
menyerupai pola umum manajemen proyek IT (kompleksitas tinggi, requirement
sering berubah, pengalaman tim rendah, dsb. meningkatkan risiko
keterlambatan). Data ini dibuat untuk **mendemonstrasikan alur XAI secara
end-to-end**.

Untuk penggunaan pada organisasi nyata, ganti `data/proyek_it.csv` dengan
data historis proyek Anda (kolom sesuai `FEATURE_COLUMNS` pada
`src/data_generator.py` dan `TARGET_COLUMN` = `terlambat`), lalu jalankan
`python -m src.train_model` untuk melatih ulang model sebelum digunakan
dalam pengambilan keputusan.

## Mengapa Random Forest + SHAP?

- **Random Forest** dipilih karena performa yang solid pada data tabular
  dengan fitur campuran (numerik & kategorikal) serta kompatibel langsung
  dengan `shap.TreeExplainer`, sehingga nilai SHAP dapat dihitung secara
  **eksak** (bukan aproksimasi) dan cepat.
- **SHAP** memberikan kontribusi tiap fitur terhadap prediksi berdasarkan
  teori permainan kooperatif (Shapley value), sehingga penjelasan bersifat
  konsisten secara matematis dan dapat diagregasi dari level lokal
  (satu proyek) ke level global (seluruh dataset) — cocok untuk mendukung
  transparansi keputusan manajerial.
