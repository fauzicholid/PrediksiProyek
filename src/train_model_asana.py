"""
Melatih model klasifikasi keterlambatan task menggunakan dataset Asana
(task-level, lihat `src/asana_data.py`).
"""

from __future__ import annotations

import json
import os

import pandas as pd

from src.asana_data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, OUTPUT_PATH, TARGET_COLUMN, build_features
from src.model_utils import train_and_save

DATA_PATH = OUTPUT_PATH
MODEL_PATH = "models/model_keterlambatan_asana.pkl"
METRICS_PATH = "models/metrics_asana.json"
DATA_TEST_OUT_PATH = "data/data_uji_asana.csv"


def main() -> None:
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        build_features().to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)
    metrics = train_and_save(
        df,
        feature_columns=FEATURE_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        target_column=TARGET_COLUMN,
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        data_test_out_path=DATA_TEST_OUT_PATH,
    )

    print("Model berhasil dilatih dan disimpan di", MODEL_PATH)
    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_columns"}, indent=2))


if __name__ == "__main__":
    main()
