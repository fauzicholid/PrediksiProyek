"""
Perbandingan Random Forest (regresi berbasis fitur) vs ARIMA (time series)
untuk memprediksi Effort proyek software pada dataset Desharnais.

Dataset Desharnais berisi 81 proyek software dari perusahaan Kanada
(1982-1988) dengan atribut ukuran & kompleksitas proyek (Transactions,
Entities, function points, dsb.) serta target Effort (person-hours).
Dataset ini BUKAN time series asli (81 proyek independen, bukan
pengamatan berurutan dari satu entitas) - satu-satunya penanda waktu
adalah `YearEnd`. Untuk keperluan perbandingan, proyek diurutkan
berdasarkan YearEnd (lalu id sebagai pemecah seri) sehingga Effort dapat
diperlakukan sebagai deret waktu semu (pseudo time series) dan dibagi
menjadi train/test secara kronologis - baik untuk model Random Forest
maupun ARIMA, agar perbandingan adil (keduanya diuji pada proyek "masa
depan" yang sama).

Keluaran:
  - outputs/metrics.json                 -> metrik perbandingan
  - outputs/predictions.csv              -> prediksi vs aktual (data uji)
  - outputs/actual_vs_predicted.png      -> grafik garis
  - outputs/metrics_comparison.png       -> bar chart metrik
  - outputs/rf_feature_importance.png    -> feature importance RF
"""

from __future__ import annotations

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "desharnais.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FEATURE_COLUMNS = [
    "TeamExp",
    "ManagerExp",
    "Length",
    "Transactions",
    "Entities",
    "PointsNonAdjust",
    "Adjustment",
    "PointsAjust",
    "Language",
]
TARGET_COLUMN = "Effort"
TEST_SIZE = 16  # ~20% dari 77 proyek valid, dipakai sebagai holdout kronologis


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # -1 pada TeamExp / ManagerExp menandakan data hilang (4 proyek pada
    # dataset asli) - dibuang mengikuti praktik umum pada literatur.
    missing_mask = (df["TeamExp"] == -1) | (df["ManagerExp"] == -1)
    df = df[~missing_mask].copy()
    # Urutkan kronologis: YearEnd lalu id sebagai pemecah seri.
    df = df.sort_values(["YearEnd", "id"]).reset_index(drop=True)
    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAPE (%)": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def run_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = pd.get_dummies(train_df[FEATURE_COLUMNS], columns=["Language"])
    X_test = pd.get_dummies(test_df[FEATURE_COLUMNS], columns=["Language"])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    model = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
    return y_pred, compute_metrics(y_test, y_pred), importances


def best_arima_order(series: pd.Series, max_p=3, max_d=2, max_q=3):
    best_aic = np.inf
    best_order = (0, 1, 0)
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    fit = ARIMA(series, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order, best_aic


def run_arima(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_series = train_df[TARGET_COLUMN].reset_index(drop=True)
    n_test = len(test_df)

    order, aic = best_arima_order(train_series)
    fitted = ARIMA(train_series, order=order).fit()

    forecast = fitted.forecast(steps=n_test)
    y_pred = np.asarray(forecast)
    y_test = test_df[TARGET_COLUMN].values

    return y_pred, compute_metrics(y_test, y_pred), order, aic


def plot_actual_vs_predicted(test_df, y_pred_rf, y_pred_arima, path):
    x = np.arange(len(test_df))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, test_df[TARGET_COLUMN].values, "o-", color="#0b0b0b", label="Aktual", linewidth=2)
    ax.plot(x, y_pred_rf, "s--", color="#2a78d6", label="Random Forest", linewidth=2)
    ax.plot(x, y_pred_arima, "^--", color="#e34948", label="ARIMA", linewidth=2)
    ax.set_xlabel("Proyek pada data uji (urutan kronologis)")
    ax.set_ylabel("Effort (person-hours)")
    ax.set_title("Aktual vs Prediksi Effort - Random Forest vs ARIMA")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_comparison(metrics_rf, metrics_arima, path):
    # MAE/RMSE (satuan person-hours) dan MAPE (%) dipisah ke dua panel
    # karena skalanya jauh berbeda - menyatukan dalam satu sumbu akan
    # membuat salah satu batang tidak terbaca.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    width = 0.35

    error_labels = ["MAE", "RMSE"]
    rf_errors = [metrics_rf[m] for m in error_labels]
    arima_errors = [metrics_arima[m] for m in error_labels]
    x1 = np.arange(len(error_labels))
    ax1.bar(x1 - width / 2, rf_errors, width, label="Random Forest", color="#2a78d6")
    ax1.bar(x1 + width / 2, arima_errors, width, label="ARIMA", color="#e34948")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(error_labels)
    ax1.set_ylabel("Person-hours (lebih rendah lebih baik)")
    ax1.set_title("MAE & RMSE")
    ax1.legend(loc="upper left")
    for i, v in enumerate(rf_errors):
        ax1.text(i - width / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(arima_errors):
        ax1.text(i + width / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

    mape_labels = ["MAPE (%)"]
    rf_mape = [metrics_rf["MAPE (%)"]]
    arima_mape = [metrics_arima["MAPE (%)"]]
    x2 = np.arange(len(mape_labels))
    ax2.bar(x2 - width / 2, rf_mape, width, label="Random Forest", color="#2a78d6")
    ax2.bar(x2 + width / 2, arima_mape, width, label="ARIMA", color="#e34948")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(mape_labels)
    ax2.set_ylabel("% (lebih rendah lebih baik)")
    ax2.set_title("MAPE")
    ax2.legend()
    for i, v in enumerate(rf_mape):
        ax2.text(i - width / 2, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(arima_mape):
        ax2.text(i + width / 2, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Perbandingan Metrik Error: Random Forest vs ARIMA")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot(kind="barh", ax=ax, color="#2a78d6")
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title("Fitur Paling Berpengaruh terhadap Prediksi Effort")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_clean_data()
    print(f"Data bersih: {len(df)} proyek (setelah membuang data hilang)")

    train_df = df.iloc[: -TEST_SIZE].reset_index(drop=True)
    test_df = df.iloc[-TEST_SIZE:].reset_index(drop=True)
    print(f"Train: {len(train_df)} proyek | Test (holdout kronologis): {len(test_df)} proyek")

    y_pred_rf, metrics_rf, importances = run_random_forest(train_df, test_df)
    print("Random Forest:", metrics_rf)

    y_pred_arima, metrics_arima, order, aic = run_arima(train_df, test_df)
    print(f"ARIMA order={order} (AIC={aic:.1f}):", metrics_arima)

    plot_actual_vs_predicted(
        test_df, y_pred_rf, y_pred_arima, os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    )
    plot_metrics_comparison(metrics_rf, metrics_arima, os.path.join(OUTPUT_DIR, "metrics_comparison.png"))
    plot_feature_importance(importances, os.path.join(OUTPUT_DIR, "rf_feature_importance.png"))

    predictions_out = test_df[["id", "YearEnd", TARGET_COLUMN]].copy()
    predictions_out["prediksi_random_forest"] = np.round(y_pred_rf, 1)
    predictions_out["prediksi_arima"] = np.round(y_pred_arima, 1)
    predictions_out.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

    results = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "random_forest": metrics_rf,
        "arima": {**metrics_arima, "order": list(order), "aic": round(aic, 2)},
        "feature_importance_random_forest": importances.sort_values(ascending=False).round(4).to_dict(),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSelesai. Hasil tersimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
