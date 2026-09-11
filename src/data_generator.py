"""
Generator dataset sintetis proyek IT untuk pemodelan prediksi keterlambatan.

Fitur dirancang menyerupai atribut manajemen proyek IT pada umumnya
(ukuran tim, durasi rencana, anggaran, kompleksitas, dsb). Label
`terlambat` dibangkitkan dari kombinasi fitur-fitur tersebut ditambah
noise acak, sehingga model yang dilatih di atasnya memiliki pola yang
masuk akal untuk dijelaskan dengan SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ukuran_tim",
    "durasi_rencana_hari",
    "anggaran_juta",
    "kompleksitas",
    "pengalaman_tim_tahun",
    "perubahan_requirement",
    "keterlibatan_klien",
    "risiko_teknologi",
    "jumlah_stakeholder",
    "turnover_tim_persen",
    "ketersediaan_sumber_daya",
    "metodologi",
]

CATEGORICAL_COLUMNS = ["metodologi"]
METODOLOGI_OPTIONS = ["Waterfall", "Agile", "Hybrid"]

TARGET_COLUMN = "terlambat"


def generate_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Bangkitkan dataset sintetis proyek IT beserta label keterlambatan."""
    rng = np.random.default_rng(random_state)

    ukuran_tim = rng.integers(3, 41, size=n_samples)
    durasi_rencana_hari = rng.integers(30, 366, size=n_samples)
    anggaran_juta = rng.gamma(shape=4.0, scale=150, size=n_samples).round(1) + 50
    kompleksitas = rng.integers(1, 6, size=n_samples)
    pengalaman_tim_tahun = rng.gamma(shape=2.0, scale=2.2, size=n_samples).round(1)
    perubahan_requirement = rng.poisson(lam=4, size=n_samples)
    keterlibatan_klien = rng.integers(1, 6, size=n_samples)
    risiko_teknologi = rng.integers(1, 6, size=n_samples)
    jumlah_stakeholder = rng.integers(2, 21, size=n_samples)
    turnover_tim_persen = np.clip(rng.normal(loc=12, scale=8, size=n_samples), 0, 60).round(1)
    ketersediaan_sumber_daya = rng.integers(1, 6, size=n_samples)
    metodologi = rng.choice(METODOLOGI_OPTIONS, size=n_samples, p=[0.35, 0.45, 0.20])

    df = pd.DataFrame(
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
    )

    # Anggaran relatif terhadap kebutuhan proyek (semakin kecil semakin berisiko).
    kebutuhan_anggaran = (
        50 + kompleksitas * 60 + ukuran_tim * 8 + durasi_rencana_hari * 0.5
    )
    rasio_anggaran = anggaran_juta / kebutuhan_anggaran

    metodologi_risk = pd.Series(metodologi).map(
        {"Waterfall": 0.9, "Hybrid": 0.35, "Agile": -0.4}
    ).to_numpy()

    # Skor risiko: kombinasi linear fitur yang relevan secara domain.
    risk_score = (
        0.55 * kompleksitas
        + 0.45 * risiko_teknologi
        + 0.35 * perubahan_requirement
        + 0.30 * turnover_tim_persen / 10
        + 0.25 * jumlah_stakeholder / 5
        - 0.55 * pengalaman_tim_tahun
        - 0.45 * keterlibatan_klien
        - 0.40 * ketersediaan_sumber_daya
        - 1.20 * np.clip(rasio_anggaran - 1, -1, 1)
        + metodologi_risk
        + 0.02 * (ukuran_tim - 15)
        + 0.01 * (durasi_rencana_hari - 150) / 10
    )

    noise = rng.normal(loc=0, scale=1.0, size=n_samples)
    logit = 2.1 * (risk_score - risk_score.mean()) / risk_score.std() + noise
    probabilitas_terlambat = 1 / (1 + np.exp(-logit))
    terlambat = (rng.random(n_samples) < probabilitas_terlambat).astype(int)

    # Estimasi lama keterlambatan (hari) - hanya bermakna bila proyek terlambat.
    keterlambatan_hari = np.where(
        terlambat == 1,
        np.clip(
            (risk_score - np.median(risk_score)) * 8
            + rng.normal(loc=15, scale=10, size=n_samples),
            1,
            None,
        ).round(0),
        0,
    )

    df["probabilitas_risiko_sintetis"] = probabilitas_terlambat.round(3)
    df["estimasi_keterlambatan_hari"] = keterlambatan_hari.astype(int)
    df[TARGET_COLUMN] = terlambat

    return df


def save_dataset(path: str, n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    df = generate_dataset(n_samples=n_samples, random_state=random_state)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    output_path = "data/proyek_it.csv"
    dataset = save_dataset(output_path)
    print(f"Dataset tersimpan di {output_path} ({len(dataset)} baris)")
    print(dataset[TARGET_COLUMN].value_counts(normalize=True))
