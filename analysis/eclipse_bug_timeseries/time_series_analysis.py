"""
Analisis time series & perbandingan SARIMA vs Random Forest (fitur lag)
pada deret bulanan jumlah bug baru proyek Eclipse Platform (2006-2011).

Berbeda dari studi sebelumnya pada dataset Desharnais (81 proyek software
independen, tanpa struktur temporal asli), dataset ini adalah **time
series sungguhan**: satu proyek yang sama, diamati berturut-turut setiap
bulan selama >5 tahun - kandidat yang jauh lebih tepat untuk pendekatan
time series.

Alur:
  1. Eksplorasi: plot deret, uji stasioneritas (ADF), ACF/PACF, dekomposisi
     tren/musiman.
  2. Split kronologis: 52 bulan train, 12 bulan test (holdout masa depan).
  3. SARIMA otomatis (grid search AIC) sebagai model time series "native".
  4. Random Forest dengan fitur lag (lag_1, lag_2, lag_3, lag_12, bulan,
     indeks tren) sebagai model berbasis fitur, diramalkan secara
     rekursif (multi-step) - pembanding yang sama seperti pada studi
     Desharnais.
  5. Bandingkan MAE, RMSE, MAPE, R² pada 12 bulan data uji.
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MONTHLY_PATH = os.path.join(BASE_DIR, "data", "eclipse_platform_bugs_monthly.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TEST_SIZE = 12
SEASONAL_PERIOD = 12

BLUE = "#2a78d6"   # Random Forest
RED = "#e34948"    # SARIMA (konsisten dengan warna ARIMA pada studi Desharnais)
INK = "#191a17"


def load_series() -> pd.Series:
    df = pd.read_csv(MONTHLY_PATH, parse_dates=["bulan"])
    df = df.set_index("bulan").asfreq("MS")
    return df["jumlah_bug"]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAPE (%)": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


# --------------------------------------------------------------------
# 1. Eksplorasi & uji stasioneritas
# --------------------------------------------------------------------
def plot_series_overview(series: pd.Series, path: str):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    train, test = series.iloc[:-TEST_SIZE], series.iloc[-TEST_SIZE:]
    ax.plot(train.index, train.values, color=INK, label="Train")
    ax.plot(test.index, test.values, color=INK, linestyle="--", marker="o", markersize=3, label="Test (holdout)")
    ax.axvline(test.index[0], color="#8b887f", linestyle=":", linewidth=1)
    ax.set_title("Jumlah Bug Baru per Bulan - Eclipse Platform (2006-2011)")
    ax.set_ylabel("Jumlah bug baru")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def adf_report(series: pd.Series) -> dict:
    def _run(s, label):
        stat, pvalue, *_ = adfuller(s.dropna())
        return {"label": label, "adf_stat": round(stat, 3), "p_value": round(pvalue, 4), "stationary": pvalue < 0.05}

    results = [
        _run(series, "level (belum di-diff)"),
        _run(series.diff(), "differencing orde-1"),
    ]
    return results


def plot_acf_pacf(series: pd.Series, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(series.diff().dropna(), ax=axes[0], lags=24, color=BLUE)
    axes[0].set_title("ACF (setelah differencing orde-1)")
    plot_pacf(series.diff().dropna(), ax=axes[1], lags=24, color=BLUE, method="ywm")
    axes[1].set_title("PACF (setelah differencing orde-1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_decomposition(series: pd.Series, path: str):
    result = seasonal_decompose(series, model="additive", period=SEASONAL_PERIOD)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    result.observed.plot(ax=axes[0], color=INK)
    axes[0].set_ylabel("Observed")
    result.trend.plot(ax=axes[1], color=BLUE)
    axes[1].set_ylabel("Trend")
    result.seasonal.plot(ax=axes[2], color=RED)
    axes[2].set_ylabel("Seasonal")
    result.resid.plot(ax=axes[3], color="#8b887f", marker="o", markersize=2, linestyle="none")
    axes[3].axhline(0, color=INK, linewidth=0.8)
    axes[3].set_ylabel("Residual")
    fig.suptitle("Dekomposisi Deret: Trend + Seasonal + Residual")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------
# 2. SARIMA
# --------------------------------------------------------------------
def best_sarima_order(train: pd.Series):
    best_aic = np.inf
    best_spec = None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                for P in range(2):
                    for D in range(2):
                        for Q in range(2):
                            if p == q == P == Q == 0:
                                continue
                            try:
                                model = SARIMAX(
                                    train,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, SEASONAL_PERIOD),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                fit = model.fit(disp=False)
                                if fit.aic < best_aic:
                                    best_aic = fit.aic
                                    best_spec = (p, d, q, P, D, Q)
                            except Exception:
                                continue
    return best_spec, best_aic


def run_sarima(train: pd.Series, test: pd.Series):
    spec, aic = best_sarima_order(train)
    p, d, q, P, D, Q = spec
    model = SARIMAX(
        train,
        order=(p, d, q),
        seasonal_order=(P, D, Q, SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=len(test))
    y_pred = np.clip(forecast.values, 0, None)
    return y_pred, compute_metrics(test.values, y_pred), spec, aic


# --------------------------------------------------------------------
# 3. Random Forest dengan fitur lag (forecasting rekursif)
# --------------------------------------------------------------------
def make_lag_features(series: pd.Series, lags=(1, 2, 3, 12)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    df["bulan_ke"] = np.arange(len(series))
    df["bulan_kalender"] = series.index.month
    return df


def run_random_forest_lag(series: pd.Series, lags=(1, 2, 3, 12)):
    feat_df = make_lag_features(series, lags).dropna()
    train_feat = feat_df.iloc[: -TEST_SIZE]
    X_train = train_feat.drop(columns=["y"])
    y_train = train_feat["y"]

    model = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    history = list(series.iloc[: -TEST_SIZE].values)
    preds = []
    for step in range(TEST_SIZE):
        idx = len(history)
        row = {f"lag_{lag}": history[idx - lag] for lag in lags}
        row["bulan_ke"] = idx
        row["bulan_kalender"] = series.index[idx].month
        X_next = pd.DataFrame([row])[X_train.columns]
        pred = max(0.0, float(model.predict(X_next)[0]))
        preds.append(pred)
        history.append(pred)

    y_test = series.iloc[-TEST_SIZE:].values
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
    return np.array(preds), compute_metrics(y_test, preds), importances


# --------------------------------------------------------------------
# 4. Plot hasil
# --------------------------------------------------------------------
def plot_forecast(test_index, y_test, y_sarima, y_rf, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(test_index))
    ax.plot(x, y_test, "o-", color=INK, label="Aktual", linewidth=2)
    ax.plot(x, y_rf, "s--", color=BLUE, label="Random Forest (lag features)", linewidth=2)
    ax.plot(x, y_sarima, "^--", color=RED, label="SARIMA", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in test_index], rotation=45, ha="right")
    ax.set_ylabel("Jumlah bug baru")
    ax.set_title("Aktual vs Forecast - 12 Bulan Terakhir (Eclipse Platform)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_comparison(metrics_rf, metrics_sarima, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    width = 0.35

    error_labels = ["MAE", "RMSE"]
    rf_errors = [metrics_rf[m] for m in error_labels]
    sarima_errors = [metrics_sarima[m] for m in error_labels]
    x1 = np.arange(len(error_labels))
    ax1.bar(x1 - width / 2, rf_errors, width, label="Random Forest", color=BLUE)
    ax1.bar(x1 + width / 2, sarima_errors, width, label="SARIMA", color=RED)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(error_labels)
    ax1.set_ylabel("Jumlah bug (lebih rendah lebih baik)")
    ax1.set_title("MAE & RMSE")
    ax1.legend(loc="upper left")
    for i, v in enumerate(rf_errors):
        ax1.text(i - width / 2, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(sarima_errors):
        ax1.text(i + width / 2, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=9)

    x2 = np.arange(1)
    ax2.bar(x2 - width / 2, [metrics_rf["MAPE (%)"]], width, label="Random Forest", color=BLUE)
    ax2.bar(x2 + width / 2, [metrics_sarima["MAPE (%)"]], width, label="SARIMA", color=RED)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["MAPE (%)"])
    ax2.set_ylabel("% (lebih rendah lebih baik)")
    ax2.set_title("MAPE")
    ax2.legend()
    for i, v in enumerate([metrics_rf["MAPE (%)"]]):
        ax2.text(i - width / 2, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate([metrics_sarima["MAPE (%)"]]):
        ax2.text(i + width / 2, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Perbandingan Metrik Error: Random Forest vs SARIMA")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    series = load_series()
    print(f"Deret waktu: {len(series)} bulan ({series.index.min().date()} s.d. {series.index.max().date()})")

    train, test = series.iloc[:-TEST_SIZE], series.iloc[-TEST_SIZE:]
    print(f"Train: {len(train)} bulan | Test: {len(test)} bulan")

    # --- Eksplorasi ---
    plot_series_overview(series, os.path.join(OUTPUT_DIR, "series_overview.png"))
    adf_results = adf_report(series)
    for r in adf_results:
        print(f"ADF [{r['label']}]: stat={r['adf_stat']}, p={r['p_value']}, stasioner={r['stationary']}")
    plot_acf_pacf(series, os.path.join(OUTPUT_DIR, "acf_pacf.png"))
    plot_decomposition(series, os.path.join(OUTPUT_DIR, "decomposition.png"))

    # --- Modeling ---
    y_pred_sarima, metrics_sarima, spec, aic = run_sarima(train, test)
    print(f"SARIMA order={spec[:3]} seasonal={spec[3:]}x{SEASONAL_PERIOD} (AIC={aic:.1f}):", metrics_sarima)

    y_pred_rf, metrics_rf, importances = run_random_forest_lag(series)
    print("Random Forest (lag features):", metrics_rf)

    # --- Visualisasi hasil ---
    plot_forecast(test.index, test.values, y_pred_sarima, y_pred_rf, os.path.join(OUTPUT_DIR, "forecast_comparison.png"))
    plot_metrics_comparison(metrics_rf, metrics_sarima, os.path.join(OUTPUT_DIR, "metrics_comparison.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax, color=BLUE)
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title("Fitur Paling Berpengaruh (Random Forest, lag features)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rf_feature_importance.png"), dpi=150)
    plt.close(fig)

    predictions_out = pd.DataFrame(
        {
            "bulan": test.index.strftime("%Y-%m"),
            "aktual": test.values,
            "prediksi_random_forest": np.round(y_pred_rf, 1),
            "prediksi_sarima": np.round(y_pred_sarima, 1),
        }
    )
    predictions_out.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

    results = {
        "n_bulan_total": len(series),
        "n_train": len(train),
        "n_test": len(test),
        "adf_test": adf_results,
        "random_forest_lag": metrics_rf,
        "sarima": {
            **metrics_sarima,
            "order": list(spec[:3]),
            "seasonal_order": list(spec[3:]) + [SEASONAL_PERIOD],
            "aic": round(aic, 2),
        },
        "feature_importance_random_forest": importances.sort_values(ascending=False).round(4).to_dict(),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSelesai. Hasil tersimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
