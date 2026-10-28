"""
environment/real_carbon.py

Real carbon-intensity traces for the simulator, as an alternative to the
synthetic generator in carbon_data.py. Produces rows with exactly the same
schema, so CarbonDataProvider and the environment are unchanged.

Data: hourly 2020 traces for Germany (DE), France (FR), Great Britain (GB)
and California (CAL), derived from the public dataset of Wiesner et al.,
"Let's Wait Awhile" (Middleware 2021). See scripts/prepare_real_carbon.py.

Region mappings (the simulator's region profiles - capacity and price - are
unchanged; only the carbon trace of each region is replaced):
    real_eu    : A=DE (dirtiest), B=FR (cleanest), C=GB (intermediate)
                 -> same carbon ORDER as the synthetic profiles (A > C > B)
    real_mixed : A=DE, B=GB, C=CAL  -> smaller spatial gaps, larger daily swings

Scenario windows (time-based train/test split, so test traces come from a
period the agent never saw):
    split="train": windows starting in Jan-Aug 2020
    split="test" : windows starting in Sep-Dec 2020
The scenario seed picks the start hour deterministically within the split.

Forecasts: no forecasts are published with the data, so the t+k forecast is
the true future value plus Gaussian noise with standard deviation
forecast_error_frac_per_step * k * (region's mean intensity), clipped at 0.
"""

from __future__ import annotations

import csv
import os
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "real_carbon", "hourly_2020.csv")

MAPPINGS: Dict[str, Dict[str, str]] = {
    "real_eu": {"A": "DE", "B": "FR", "C": "GB"},
    "real_mixed": {"A": "DE", "B": "GB", "C": "CAL"},
}

TRAIN_FRACTION = 2.0 / 3.0  # first two thirds of the year (Jan-Aug) for training windows


@lru_cache(maxsize=4)
def load_hourly_traces(path: str = DEFAULT_PATH) -> Dict[str, List[float]]:
    """Return {"DE": [...], "FR": [...], ...} hourly values for 2020."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it with scripts/prepare_real_carbon.py "
            "from the lets-wait-awhile dataset."
        )
    columns: Dict[str, List[float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key != "timestamp":
                    columns.setdefault(key, []).append(float(value))
    return columns


def window_start(seed: int, n_hours_total: int, length: int, split: str) -> int:
    """Deterministic start hour for a scenario seed within a split."""
    boundary = int(n_hours_total * TRAIN_FRACTION)
    if split == "train":
        lo, hi = 0, boundary - length
    elif split == "test":
        lo, hi = boundary, n_hours_total - length
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if hi <= lo:
        raise ValueError("window too long for the requested split")
    return random.Random(seed).randint(lo, hi)


def generate_real_carbon_series(
    region_ids: List[str],
    n_timesteps: int,
    horizon: int = 3,
    mapping: str = "real_eu",
    split: str = "train",
    seed: Optional[int] = None,
    forecast_error_frac_per_step: float = 0.05,
    path: str = DEFAULT_PATH,
) -> List[Dict[str, Any]]:
    """Real-trace rows in the same schema as carbon_data.generate_carbon_series."""
    if mapping not in MAPPINGS:
        raise ValueError(f"unknown mapping {mapping!r}; choose from {list(MAPPINGS)}")
    traces = load_hourly_traces(path)
    region_to_trace = MAPPINGS[mapping]
    n_total = len(next(iter(traces.values())))
    start = window_start(seed or 0, n_total, n_timesteps + horizon, split)
    rng = random.Random((seed or 0) * 7919 + 17)

    rows: List[Dict[str, Any]] = []
    for region_id in region_ids:
        if region_id not in region_to_trace:
            raise ValueError(f"mapping {mapping!r} has no trace for region {region_id!r}")
        full = traces[region_to_trace[region_id]]
        mean_level = sum(full) / len(full)
        actual = full[start:start + n_timesteps + horizon]
        for t in range(n_timesteps):
            row: Dict[str, Any] = {
                "timestamp": float(t),
                "region_id": region_id,
                "carbon_intensity_current": round(actual[t], 2),
            }
            for k in range(1, horizon + 1):
                noisy = actual[t + k] + rng.gauss(0.0, forecast_error_frac_per_step * k * mean_level)
                row[f"carbon_intensity_t+{k}"] = round(max(0.0, noisy), 2)
            rows.append(row)
    return rows
