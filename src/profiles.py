"""
Registry "profil dataset" untuk dashboard: membungkus semua yang
membedakan satu sumber data dari sumber data lain (kolom fitur, label,
lokasi model, dan skema form input) di satu tempat, sehingga `app.py`
bisa merender setiap halaman secara generik tanpa hardcode ke satu
dataset tertentu.

Menambah sumber data baru di masa depan = menambah satu `DatasetProfile`
baru di sini, tanpa menyentuh `app.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src import asana_data, data_generator


@dataclass
class FieldSpec:
    """Deskripsi satu input pada form 'Prediksi Proyek/Task Baru'."""

    name: str
    label: str
    kind: str  # "slider_int" | "slider_float" | "number" | "select" | "checkbox"
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: float | int | bool | str | None = None
    options: list[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    key: str
    display_name: str
    unit_label: str  # "proyek" atau "task"
    description: str
    caveat: str  # peringatan/catatan penting tentang sifat data
    feature_columns: list[str]
    categorical_columns: list[str]
    target_column: str
    feature_labels: dict[str, str]
    data_path: str
    model_path: str
    metrics_path: str
    fields: list[FieldSpec]
    id_columns: list[str] = field(default_factory=list)  # kolom non-fitur untuk ditampilkan di tabel


SYNTHETIC_PROFILE = DatasetProfile(
    key="sintetis",
    display_name="Data Sintetis (Demo)",
    unit_label="proyek",
    description=(
        "Data sintetis 1.500 proyek IT yang dibangkitkan dengan formula risiko "
        "buatan (lihat `src/data_generator.py`) - dirancang agar hubungan "
        "antar-fitur masuk akal dan mudah dijelaskan. Cocok untuk demonstrasi "
        "alur XAI end-to-end dengan sinyal yang bersih."
    ),
    caveat=(
        "⚠️ Data ini **sintetis** (dibangkitkan dengan rumus, bukan dari proyek "
        "nyata) - performa & pola SHAP di sini menggambarkan cara kerja "
        "pipeline, bukan bukti empiris tentang proyek IT sungguhan."
    ),
    feature_columns=data_generator.FEATURE_COLUMNS,
    categorical_columns=data_generator.CATEGORICAL_COLUMNS,
    target_column=data_generator.TARGET_COLUMN,
    feature_labels=data_generator.FEATURE_LABELS,
    data_path="data/proyek_it.csv",
    model_path="models/model_keterlambatan.pkl",
    metrics_path="models/metrics.json",
    fields=[
        FieldSpec("ukuran_tim", "Ukuran Tim (orang)", "slider_int", min=3, max=40, default=10),
        FieldSpec("durasi_rencana_hari", "Durasi Rencana (hari)", "slider_int", min=30, max=365, default=120),
        FieldSpec("anggaran_juta", "Anggaran (juta Rp)", "number", min=50.0, default=500.0, step=10.0),
        FieldSpec("kompleksitas", "Kompleksitas Proyek (1=Rendah, 5=Sangat Tinggi)", "slider_int", min=1, max=5, default=3),
        FieldSpec("pengalaman_tim_tahun", "Rata-rata Pengalaman Tim (tahun)", "slider_float", min=0.0, max=15.0, default=3.0, step=0.5),
        FieldSpec("perubahan_requirement", "Jumlah Perubahan Requirement", "slider_int", min=0, max=20, default=4),
        FieldSpec("keterlibatan_klien", "Keterlibatan Klien (1=Rendah, 5=Sangat Aktif)", "slider_int", min=1, max=5, default=3),
        FieldSpec("risiko_teknologi", "Risiko Teknologi (1=Rendah, 5=Sangat Tinggi)", "slider_int", min=1, max=5, default=3),
        FieldSpec("jumlah_stakeholder", "Jumlah Stakeholder", "slider_int", min=2, max=20, default=6),
        FieldSpec("turnover_tim_persen", "Turnover Tim (%)", "slider_float", min=0.0, max=60.0, default=10.0, step=1.0),
        FieldSpec("ketersediaan_sumber_daya", "Ketersediaan Sumber Daya (1=Rendah, 5=Sangat Baik)", "slider_int", min=1, max=5, default=3),
        FieldSpec("metodologi", "Metodologi Pengembangan", "select", options=data_generator.METODOLOGI_OPTIONS, default="Agile"),
    ],
)


ASANA_PROFILE = DatasetProfile(
    key="asana",
    display_name="Data Asana (Simulasi Enterprise)",
    unit_label="task",
    description=(
        "16.639 task nyata (sudah selesai) dari simulasi workspace Asana "
        "perusahaan SaaS B2B (~6.000 pengguna, ~1.200 proyek) - lihat "
        "`data/asana_raw/SOURCE_README.md`. Label keterlambatan dihitung dari "
        "selisih tanggal selesai vs tenggat asli pada data, bukan dari skor "
        "risiko buatan."
    ),
    caveat=(
        "⚠️ Data ini tetap **simulasi** (dibuat dengan bantuan LLM untuk riset "
        "agent/RL), bukan data proyek nyata. Analisis SHAP pada data ini "
        "menunjukkan sinyal keterlambatan **hampir seluruhnya berasal dari "
        "durasi rencana** (sisa waktu antara task dibuat & tenggat) - fitur "
        "kontekstual lain (tim, peran, tag, komentar, beban kerja) hampir "
        "tidak berkontribusi. Ini justru poin penting dari XAI: jangan asumsikan "
        "faktor \"kelihatannya penting\" pasti berpengaruh - selalu verifikasi "
        "lewat SHAP, jangan diasumsikan."
    ),
    feature_columns=asana_data.FEATURE_COLUMNS,
    categorical_columns=asana_data.CATEGORICAL_COLUMNS,
    target_column=asana_data.TARGET_COLUMN,
    feature_labels=asana_data.FEATURE_LABELS,
    data_path=asana_data.OUTPUT_PATH,
    model_path="models/model_keterlambatan_asana.pkl",
    metrics_path="models/metrics_asana.json",
    id_columns=["nama_task"],
    fields=[
        FieldSpec("durasi_rencana_hari", "Durasi Rencana (hari, boleh negatif = tenggat sudah lewat saat dibuat)", "slider_float", min=-30.0, max=90.0, default=14.0, step=1.0),
        FieldSpec("panjang_deskripsi_kata", "Panjang Deskripsi (jumlah kata)", "slider_int", min=0, max=40, default=15),
        FieldSpec("jumlah_tag", "Jumlah Tag pada Task", "slider_int", min=0, max=6, default=1),
        FieldSpec("prioritas_tinggi", "Ditandai Prioritas Tinggi?", "checkbox", default=False),
        FieldSpec("adalah_bug", "Task Jenis Bug?", "checkbox", default=False),
        FieldSpec("sedang_blocked", "Sedang Ditandai Blocked?", "checkbox", default=False),
        FieldSpec("jumlah_komentar", "Jumlah Komentar pada Task", "slider_int", min=0, max=8, default=1),
        FieldSpec("beban_kerja_assignee", "Beban Kerja Assignee (task lain yg aktif bersamaan)", "slider_int", min=0, max=10, default=1),
        FieldSpec("bulan_dibuat", "Bulan Dibuat (1-12)", "slider_int", min=1, max=12, default=6),
        FieldSpec("tim", "Tim", "select", options=[
            "Backend Engineering", "Frontend Engineering", "DevOps", "Product Management",
            "Data Science", "Mobile Development", "Platform Engineering", "Security",
            "QA Engineering", "Infrastructure", "Machine Learning", "Customer Success",
        ] + [f"Team {i}" for i in range(20)], default="Backend Engineering"),
        FieldSpec("peran_assignee", "Peran Assignee", "select",
                   options=["Engineer", "Designer", "Marketer", "Ops", "Product Manager", "Spesialis Lainnya"],
                   default="Engineer"),
        FieldSpec("status_proyek", "Status Proyek", "select",
                   options=["active", "completed", "on_hold"], default="active"),
    ],
)


PROFILES: dict[str, DatasetProfile] = {
    SYNTHETIC_PROFILE.key: SYNTHETIC_PROFILE,
    ASANA_PROFILE.key: ASANA_PROFILE,
}
