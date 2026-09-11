"""
Ekstraksi data time series dari dataset bug tracking Eclipse Platform
(Ohira et al., "The Mozilla and Eclipse Defect Tracking Dataset", MSR 2013).

Sumber: https://github.com/ansymo/msr2013-bug_dataset (data/eclipse.tar.gz)
File `reports.xml` berisi 24.775 laporan bug proyek Eclipse Platform
(2006-01 s.d. 2011-05) dengan timestamp pembukaan (`opening_time`, Unix
epoch). Berbeda dari dataset Desharnais, ini adalah **time series asli**:
setiap bug punya waktu kemunculan yang berurutan dari satu proyek yang
sama, sehingga pola tren/musiman/autokorelasi benar-benar bisa muncul.

Skrip ini mengagregasi timestamp mentah menjadi deret waktu **jumlah bug
baru per bulan**, yang menjadi input untuk analisis SARIMA vs Random
Forest pada `time_series_analysis.py`.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "data", "eclipse_platform_reports.xml")
DAILY_OUT = os.path.join(BASE_DIR, "data", "eclipse_platform_bugs_daily.csv")
MONTHLY_OUT = os.path.join(BASE_DIR, "data", "eclipse_platform_bugs_monthly.csv")


def parse_opening_times(xml_path: str) -> list[datetime]:
    text = open(xml_path, encoding="utf-8").read()
    timestamps = re.findall(r"<opening_time>(\d+)</opening_time>", text)
    return [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in timestamps]


def main():
    times = parse_opening_times(XML_PATH)
    print(f"Total laporan bug: {len(times)}")
    print(f"Rentang: {min(times).date()} s.d. {max(times).date()}")

    df = pd.DataFrame({"opened_at": times})
    df["date"] = df["opened_at"].dt.date

    daily = df.groupby("date").size().rename("jumlah_bug").reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    # Isi tanggal yang tidak punya bug baru dengan 0 (deret harian lengkap)
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_range, fill_value=0)
    daily.index.name = "date"
    daily.reset_index().to_csv(DAILY_OUT, index=False)

    monthly = df.set_index("opened_at").resample("MS").size().rename("jumlah_bug")
    monthly.index.name = "bulan"
    # Bulan terakhir dalam data mentah tidak lengkap (data berhenti di
    # tengah bulan), sehingga hitungannya akan tampak anjlok secara
    # artifisial - dibuang agar tidak mendistorsi evaluasi model.
    last_day_of_data = df["opened_at"].max()
    last_month_start = last_day_of_data.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if last_day_of_data.day < 28:
        monthly = monthly[monthly.index < last_month_start]
    monthly.reset_index().to_csv(MONTHLY_OUT, index=False)

    print(f"Deret harian disimpan: {DAILY_OUT} ({len(daily)} hari)")
    print(f"Deret bulanan disimpan: {MONTHLY_OUT} ({len(monthly)} bulan)")


if __name__ == "__main__":
    main()
