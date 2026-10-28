"""
environment/switch_carbon.py

Controlled carbon traces for testing when carbon observations help.

Three intensity levels (low 120, mid 280, high 420 gCO2/kWh, each with AR(1)
noise of standard deviation about 20) are assigned to the three regions.
At the start the assignment is fixed (B low, C mid, A high, the same order as
the real-EU scenario). In every hour, with probability `switch_prob`, the low
level moves to a different, randomly chosen region and the other two regions
take the mid and high levels in random order. Grid levels and variance are
therefore the same for every switch_prob; only how often the cleanest region
changes differs (0 % of hours for switch_prob = 0, about 15/30/50 % for
0.15/0.30/0.50).

Forecasts: true future value plus Gaussian noise with standard deviation
0.05 * k * 273 (the same error model as environment/real_carbon.py).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

LEVELS = (120.0, 280.0, 420.0)  # low, mid, high
NOISE_SD = 20.0
AR = 0.8
FORECAST_ERR = 0.05


def generate_switch_series(region_ids: List[str], n_timesteps: int, horizon: int = 3, switch_prob: float = 0.0,
                           seed: Optional[int] = None) -> List[Dict[str, Any]]:
    rng = random.Random((seed or 0) * 104729 + int(switch_prob * 1000))
    n = n_timesteps + horizon
    order = {"B": 0, "C": 1, "A": 2}
    assign = [order.get(r, i) for i, r in enumerate(region_ids)]  # level index per region
    noise = [0.0] * len(region_ids)
    actual = [[0.0] * n for _ in region_ids]
    for t in range(n):
        if t > 0 and rng.random() < switch_prob:
            cur_low = assign.index(0)
            new_low = rng.choice([i for i in range(len(region_ids)) if i != cur_low])
            others = [i for i in range(len(region_ids)) if i != new_low]
            rng.shuffle(others)
            assign[new_low] = 0
            for lvl, i in enumerate(others, start=1):
                assign[i] = lvl
        for i in range(len(region_ids)):
            noise[i] = AR * noise[i] + rng.gauss(0.0, NOISE_SD * (1 - AR ** 2) ** 0.5)
            actual[i][t] = max(10.0, LEVELS[assign[i]] + noise[i])
    mean = sum(LEVELS) / len(LEVELS)
    rows: List[Dict[str, Any]] = []
    for i, rid in enumerate(region_ids):
        for t in range(n_timesteps):
            row: Dict[str, Any] = {"timestamp": float(t), "region_id": rid,
                                   "carbon_intensity_current": round(actual[i][t], 2)}
            for k in range(1, horizon + 1):
                row[f"carbon_intensity_t+{k}"] = round(max(0.0, actual[i][t + k] + rng.gauss(0.0, FORECAST_ERR * k * mean)), 2)
            rows.append(row)
    return rows


def switch_fraction(rows: List[Dict[str, Any]]) -> float:
    """Fraction of hours in which the cleanest region differs from the previous hour."""
    by_t: Dict[float, Dict[str, float]] = {}
    for r in rows:
        by_t.setdefault(r["timestamp"], {})[r["region_id"]] = r["carbon_intensity_current"]
    ts = sorted(by_t)
    clean = [min(by_t[t], key=by_t[t].get) for t in ts]
    return sum(1 for a, b in zip(clean, clean[1:]) if a != b) / max(1, len(clean) - 1)
