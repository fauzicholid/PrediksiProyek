"""
Rekayasa fitur dari dataset simulasi workspace Asana (lihat
`data/asana_raw/SOURCE_README.md`) menjadi tabel task-level siap latih
untuk model prediksi keterlambatan.

Berbeda dari `data_generator.py` (data sintetis yang dibangkitkan dengan
formula matematis), dataset ini berasal dari simulasi workspace
perusahaan SaaS B2B (~6.000 pengguna, ~1.200 proyek, ~30.000 task) dengan
relasi antar-tabel yang realistis (tim, assignee, tag, komentar). Label
keterlambatan dihitung dari selisih `completed_at` vs `due_date`
sungguhan pada data, bukan dibangkitkan dari skor risiko buatan.

Catatan penting: dataset ini tetap merupakan **simulasi** (LLM-assisted),
bukan data proyek IT dunia nyata - cocok untuk demonstrasi & pengujian
pipeline dengan skala dan struktur relasional yang jauh lebih realistis
dari data sintetis murni, namun pola statistiknya tidak harus
merepresentasikan dinamika proyek IT sungguhan.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "asana_raw")
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "asana_tasks.csv"
)

FEATURE_COLUMNS = [
    "durasi_rencana_hari",
    "panjang_deskripsi_kata",
    "jumlah_tag",
    "prioritas_tinggi",
    "adalah_bug",
    "sedang_blocked",
    "jumlah_komentar",
    "beban_kerja_assignee",
    "bulan_dibuat",
    "tim",
    "peran_assignee",
    "status_proyek",
]
# Catatan: kolom `parent_task_id` pada data mentah selalu kosong (tidak
# ada relasi subtask yang benar-benar dipakai pada dataset simulasi ini),
# sehingga fitur "adalah_subtask" konstan/tidak informatif dan sengaja
# tidak disertakan.
CATEGORICAL_COLUMNS = ["tim", "peran_assignee", "status_proyek"]
TARGET_COLUMN = "terlambat"

FEATURE_LABELS = {
    "durasi_rencana_hari": "Durasi Rencana (hari)",
    "panjang_deskripsi_kata": "Panjang Deskripsi (kata)",
    "jumlah_tag": "Jumlah Tag",
    "prioritas_tinggi": "Prioritas Tinggi",
    "adalah_bug": "Task Bug",
    "sedang_blocked": "Sedang Blocked",
    "jumlah_komentar": "Jumlah Komentar",
    "beban_kerja_assignee": "Beban Kerja Assignee (task lain yg aktif)",
    "bulan_dibuat": "Bulan Dibuat",
    "tim": "Tim",
    "peran_assignee": "Peran Assignee",
    "status_proyek": "Status Proyek",
}

MAIN_ROLES = {"Marketer", "Engineer", "Designer", "Ops", "Product Manager"}


def _bucket_role(role: str) -> str:
    return role if role in MAIN_ROLES else "Spesialis Lainnya"


def _compute_assignee_workload(tasks: pd.DataFrame) -> pd.Series:
    """Jumlah task lain milik assignee yang sama & masih terbuka pada
    saat task ini dibuat - proksi beban kerja (workload) assignee."""
    workload = pd.Series(0, index=tasks.index, dtype=int)
    for assignee_id, group in tasks.groupby("assignee_id"):
        if pd.isna(assignee_id):
            continue
        g = group.sort_values("created_at")
        created = g["created_at"].values
        completed = g["completed_at"].values  # NaT jika belum selesai
        n = len(g)
        counts = np.zeros(n, dtype=int)
        for i in range(n):
            t = created[i]
            open_before = (created < t) & ((pd.isna(completed)) | (completed > t))
            counts[i] = open_before.sum()
        workload.loc[g.index] = counts
    return workload


def build_features() -> pd.DataFrame:
    tasks = pd.read_csv(os.path.join(RAW_DIR, "tasks.csv"))
    for col in ("created_at", "due_date", "completed_at"):
        # format="mixed": kolom ini mencampur "YYYY-MM-DD" & ISO datetime penuh.
        tasks[col] = pd.to_datetime(tasks[col], errors="coerce", format="mixed")
    projects = pd.read_csv(os.path.join(RAW_DIR, "projects.csv"))
    teams = pd.read_csv(os.path.join(RAW_DIR, "teams.csv"))
    users = pd.read_csv(os.path.join(RAW_DIR, "users.csv"))
    tags = pd.read_csv(os.path.join(RAW_DIR, "tags.csv"))
    task_tags = pd.read_csv(os.path.join(RAW_DIR, "task_tags.csv"))
    comments = pd.read_csv(os.path.join(RAW_DIR, "comments.csv"))

    # --- Beban kerja assignee (dihitung sebelum filter, pakai seluruh riwayat task) ---
    tasks["beban_kerja_assignee"] = _compute_assignee_workload(tasks)

    # --- Hanya task yang sudah selesai & punya due_date -> ada ground truth ---
    df = tasks[(tasks["completed"] == 1) & (tasks["due_date"].notna())].copy()
    df[TARGET_COLUMN] = (df["completed_at"] > df["due_date"]).astype(int)
    df["estimasi_keterlambatan_hari"] = (
        (df["completed_at"] - df["due_date"]).dt.total_seconds() / 86400
    ).clip(lower=0).round(1)

    # --- Fitur dasar task ---
    df["durasi_rencana_hari"] = (df["due_date"] - df["created_at"]).dt.total_seconds() / 86400
    df["panjang_deskripsi_kata"] = df["description"].fillna("").str.split().str.len()
    df["bulan_dibuat"] = df["created_at"].dt.month

    # --- Tag (dedup berdasarkan nama, karena ada id duplikat untuk nama sama) ---
    tag_name_map = tags.set_index("id")["name"]
    tt = task_tags.copy()
    tt["tag_name"] = tt["tag_id"].map(tag_name_map)
    tag_counts = tt.groupby("task_id").size().rename("jumlah_tag")
    has_high_priority = tt[tt["tag_name"] == "High Priority"].groupby("task_id").size() > 0
    has_bug = tt[tt["tag_name"] == "Bug"].groupby("task_id").size() > 0
    has_blocked = tt[tt["tag_name"] == "Blocked"].groupby("task_id").size() > 0

    df = df.set_index("id")
    df["jumlah_tag"] = tag_counts.reindex(df.index).fillna(0).astype(int)
    df["prioritas_tinggi"] = has_high_priority.reindex(df.index).fillna(False).astype(int)
    df["adalah_bug"] = has_bug.reindex(df.index).fillna(False).astype(int)
    df["sedang_blocked"] = has_blocked.reindex(df.index).fillna(False).astype(int)

    # --- Komentar ---
    comment_counts = comments.groupby("task_id").size().rename("jumlah_komentar")
    df["jumlah_komentar"] = comment_counts.reindex(df.index).fillna(0).astype(int)

    # --- Konteks organisasi: tim & status proyek (via project), peran assignee ---
    proj_team = projects.set_index("id")[["team_id", "status"]]
    df = df.join(proj_team, on="project_id")
    team_name_map = teams.set_index("id")["name"]
    df["tim"] = df["team_id"].map(team_name_map).fillna("Tidak diketahui")
    df["status_proyek"] = df["status"].fillna("tidak diketahui")

    role_map = users.set_index("id")["role"]
    df["peran_assignee"] = df["assignee_id"].map(role_map).fillna("Tidak ditugaskan")
    df["peran_assignee"] = df["peran_assignee"].apply(_bucket_role)

    df = df.reset_index()

    final_cols = ["id", "name"] + FEATURE_COLUMNS + [TARGET_COLUMN, "estimasi_keterlambatan_hari"]
    result = df[final_cols].rename(columns={"name": "nama_task"})
    result = result.dropna(subset=FEATURE_COLUMNS)
    return result


def main():
    df = build_features()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Fitur tersimpan: {OUTPUT_PATH} ({len(df)} task)")
    print(df[TARGET_COLUMN].value_counts(normalize=True))


if __name__ == "__main__":
    main()
