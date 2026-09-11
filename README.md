# Prediksi Keterlambatan Proyek IT — Dashboard XAI (SHAP)

Dashboard interaktif untuk memprediksi **risiko keterlambatan proyek IT**
menggunakan model machine learning (Random Forest), dilengkapi pendekatan
**Explainable AI (XAI)** berbasis **SHAP (SHapley Additive exPlanations)**
agar setiap prediksi dapat dijelaskan secara transparan kepada manajer
proyek — bukan sekadar keluaran "kotak hitam".

## Sumber Data (bisa dipilih di sidebar)

Dashboard mendukung lebih dari satu **profil dataset**, dipilih lewat
dropdown "🗂️ Sumber Data" di sidebar — seluruh halaman (form prediksi,
SHAP, evaluasi, eksplorasi, batch) otomatis menyesuaikan skema fitur yang
aktif:

1. **Data Sintetis (Demo)** — 1.500 proyek IT dibangkitkan dengan formula
   risiko buatan (lihat `src/data_generator.py`), dirancang agar sinyalnya
   bersih untuk demonstrasi alur XAI end-to-end.
2. **Data Asana (Simulasi Enterprise)** — 16.639 task nyata (sudah selesai)
   dari simulasi workspace Asana perusahaan SaaS B2B (~6.000 pengguna,
   ~1.200 proyek; lihat `data/asana_raw/SOURCE_README.md`). Label
   keterlambatan dihitung dari selisih tanggal selesai vs tenggat asli
   pada data, bukan dari skor risiko buatan. **Temuan penting**: analisis
   SHAP menunjukkan sinyal keterlambatan pada data ini hampir seluruhnya
   berasal dari fitur `durasi_rencana_hari` — faktor kontekstual lain
   (tim, peran, tag, komentar, beban kerja) hampir tidak berkontribusi.
   Ini adalah contoh nyata mengapa XAI penting: jangan asumsikan faktor
   yang "kelihatannya penting" pasti berpengaruh tanpa verifikasi SHAP.

Kedua profil tetap **data sintetis/simulasi** (bukan data proyek IT
dunia nyata) — lihat peringatan di sidebar/beranda dashboard untuk detail.

## Fitur Dashboard

1. **🏠 Beranda** — ringkasan model, deskripsi & peringatan sumber data aktif.
2. **🔮 Prediksi Baru** — form input (dibangun dinamis sesuai profil data
   aktif), hasil probabilitas keterlambatan, kategori risiko, serta
   **penjelasan SHAP lokal** (bar chart kontribusi fitur + SHAP waterfall
   plot standar).
3. **🧠 Analisis Global (XAI)** — kepentingan fitur secara global
   (rata-rata |SHAP value|) dan SHAP summary/beeswarm plot untuk melihat
   pola keseluruhan dataset.
4. **📈 Evaluasi Model** — akurasi, precision, recall, F1-score, ROC-AUC,
   dan confusion matrix pada data uji.
5. **🔍 Eksplorasi Data** — distribusi fitur historis.
6. **📁 Prediksi Batch (CSV)** — unggah banyak data sekaligus untuk
   diprediksi dan dijelaskan secara massal.

## Struktur Proyek

```
PrediksiProyek/
├── app.py                     # Dashboard Streamlit (entry point, generik per-profil)
├── requirements.txt
├── src/
│   ├── profiles.py            # Registry profil dataset (fitur, label, skema form)
│   ├── data_generator.py      # Pembangkit dataset sintetis proyek IT
│   ├── asana_data.py          # Rekayasa fitur dari data mentah Asana
│   ├── model_utils.py         # Pipeline & training generik (dipakai kedua profil)
│   ├── train_model.py         # Training model profil sintetis
│   ├── train_model_asana.py   # Training model profil Asana
│   ├── explainer.py           # Wrapper perhitungan SHAP (lokal & global, dataset-agnostic)
│   └── charts.py              # Chart Plotly dashboard (palet konsisten)
├── data/
│   ├── proyek_it.csv          # Dataset historis (sintetis)
│   ├── data_uji.csv           # Subset data uji + prediksi (sintetis)
│   ├── asana_tasks.csv        # Fitur task Asana hasil rekayasa (siap latih)
│   ├── data_uji_asana.csv     # Subset data uji + prediksi (Asana)
│   └── asana_raw/             # CSV mentah dataset simulasi Asana + README sumber
└── models/
    ├── model_keterlambatan.pkl / metrics.json           # profil sintetis
    └── model_keterlambatan_asana.pkl / metrics_asana.json # profil Asana
```

## Menjalankan Dashboard

```bash
pip install -r requirements.txt

# (opsional) latih ulang model — otomatis dijalankan saat dashboard
# pertama kali dibuka jika model belum ada
python -m src.train_model          # profil sintetis
python -m src.train_model_asana    # profil Asana

streamlit run app.py
```

Dashboard akan terbuka di `http://localhost:8501`. Pilih sumber data di
sidebar untuk berpindah antar profil.

## Tentang Data & Model

Kedua dataset yang disertakan bersifat **sintetis/simulasi** (bukan data
proyek IT dunia nyata):

- `src/data_generator.py` membangkitkan data proyek dengan formula risiko
  buatan — dirancang agar hubungan antar-fitur masuk akal & mudah
  dijelaskan.
- `src/asana_data.py` merekayasa fitur task-level dari dump CSV simulasi
  workspace Asana (`data/asana_raw/`, dibuat dengan bantuan LLM untuk
  riset agent/RL) — label keterlambatan dihitung dari tanggal selesai vs
  tenggat asli pada data, tapi tetap simulasi, bukan histori proyek nyata.

Untuk penggunaan pada organisasi nyata, tambahkan profil baru di
`src/profiles.py` yang mengarah ke data historis proyek/task organisasi
Anda (lihat pola `DatasetProfile` yang sudah ada), lalu latih modelnya
sebelum digunakan dalam pengambilan keputusan.

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
