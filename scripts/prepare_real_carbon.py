"""
scripts/prepare_real_carbon.py

Converts the public carbon-intensity traces published with
"Let's Wait Awhile" (Wiesner et al., Middleware 2021;
https://github.com/dos-group/lets-wait-awhile, data/*_ci.csv) into one
hourly CSV used by environment/real_carbon.py.

Source traces (gCO2/kWh, computed by the dataset authors from ENTSO-E and
CAISO generation data with IPCC life-cycle emission factors):
    ger_ci.csv (Germany, 15-min), fr_ci.csv (France, 30-min),
    gb_ci.csv (Great Britain, 30-min), cal_ci.csv (California, 5-min)

Output: data/real_carbon/hourly_2020.csv with columns
    timestamp, DE, FR, GB, CAL      (hourly means, 2020-01-01 .. 2020-12-31)

Usage:
    git clone --depth 1 https://github.com/dos-group/lets-wait-awhile
    python scripts/prepare_real_carbon.py --src lets-wait-awhile/data
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

FILES = {"DE": "ger_ci.csv", "FR": "fr_ci.csv", "GB": "gb_ci.csv", "CAL": "cal_ci.csv"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to lets-wait-awhile/data")
    ap.add_argument("--out", default="data/real_carbon/hourly_2020.csv")
    args = ap.parse_args()

    series = {}
    for code, fname in FILES.items():
        df = pd.read_csv(os.path.join(args.src, fname), parse_dates=["Time"]).set_index("Time")
        series[code] = df["Carbon Intensity"].resample("1h").mean()
    hourly = pd.DataFrame(series).loc["2020-01-01":"2020-12-31 23:00"]
    hourly = hourly.interpolate(limit=3).dropna()
    hourly.index.name = "timestamp"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    hourly.round(2).to_csv(args.out)
    print(f"wrote {args.out}: {len(hourly)} hours")
    print(hourly.describe().round(1))


if __name__ == "__main__":
    main()
