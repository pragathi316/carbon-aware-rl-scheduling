"""
environment/carbon_data.py

Generates synthetic per-region carbon-intensity time series — current
value plus an n-step-ahead forecast — and provides a lookup class the
Gymnasium environment (Phase 4) will query at each timestep.

Per the roadmap: "The first prototype can use synthetic time-series
data if real data integration is not yet ready." This module produces
that synthetic data, but is structured so swapping in a real feed later
(e.g. the UK Carbon Intensity API) only means changing how the "actual"
series is populated — the CSV schema and CarbonDataProvider interface
stay the same.

Required CSV fields (per roadmap Phase 3):
    timestamp, region_id, carbon_intensity_current,
    carbon_intensity_t+1, carbon_intensity_t+2, ..., carbon_intensity_t+n

Why prediction matters: a DELAY action is only useful if the scheduler
can see that a cleaner period is coming before the job's deadline —
that's what the t+1..t+n columns give it.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Volatility presets — control how much carbon intensity swings around its
# baseline. "stable" vs "volatile" map directly to the roadmap's required
# sensitivity study (Section 8: "Carbon volatility sensitivity").
#
# amplitude_frac:   diurnal swing size, as a fraction of the region's base
#                   carbon intensity (a day/night cleaner-grid cycle).
# noise_sigma_frac: per-step random-walk noise, as a fraction of base.
# ar_phi:           AR(1) persistence of the noise (higher = smoother,
#                   slower-changing noise; lower = jumpier).
# forecast_error_frac_per_step: how much extra Gaussian error is added to
#                   a forecast per step of horizon distance — models the
#                   realistic fact that forecasts get less accurate the
#                   further ahead they look.
VOLATILITY_PRESETS: Dict[str, Dict[str, float]] = {
    "stable": {
        "amplitude_frac": 0.10,
        "noise_sigma_frac": 0.02,
        "ar_phi": 0.85,
        "forecast_error_frac_per_step": 0.02,
    },
    "medium": {
        "amplitude_frac": 0.25,
        "noise_sigma_frac": 0.05,
        "ar_phi": 0.70,
        "forecast_error_frac_per_step": 0.05,
    },
    "volatile": {
        "amplitude_frac": 0.45,
        "noise_sigma_frac": 0.12,
        "ar_phi": 0.50,
        "forecast_error_frac_per_step": 0.10,
    },
}

# Default baseline carbon intensity per region, matching region.py's
# create_default_regions() defaults so the two modules compose cleanly.
DEFAULT_BASE_INTENSITY: Dict[str, float] = {
    "A": 600.0,
    "B": 250.0,
    "C": 400.0,
    "D": 120.0,
    "E": 550.0,
}

MIN_CARBON_INTENSITY = 10.0  # gCO2/kWh floor — intensity can't go to 0 or negative.


def _generate_actual_series(
    n_steps: int,
    base_intensity: float,
    preset: Dict[str, float],
    rng: random.Random,
    period_steps: float = 24.0,
) -> List[float]:
    """Generate one region's 'ground truth' carbon-intensity series.

    Combines a diurnal sinusoid (cleaner grid at certain hours) with
    AR(1) noise, clipped to stay non-negative.
    """
    amplitude = base_intensity * preset["amplitude_frac"]
    sigma = base_intensity * preset["noise_sigma_frac"]
    phi = preset["ar_phi"]
    phase = rng.uniform(0, 2 * math.pi)

    series: List[float] = []
    noise = 0.0
    for t in range(n_steps):
        diurnal = amplitude * math.sin(2 * math.pi * t / period_steps + phase)
        noise = phi * noise + rng.gauss(0, sigma)
        value = base_intensity + diurnal + noise
        series.append(max(MIN_CARBON_INTENSITY, value))
    return series


def generate_carbon_series(
    region_ids: List[str],
    n_timesteps: int,
    horizon: int = 3,
    volatility: str = "medium",
    seed: Optional[int] = None,
    base_intensity: Optional[Dict[str, float]] = None,
    timestep_duration: float = 1.0,
) -> List[Dict[str, Any]]:
    """Generate synthetic carbon-intensity rows for all regions and timesteps.

    Args:
        region_ids: Region identifiers to generate data for, e.g. ["A","B","C"].
        n_timesteps: Number of timesteps of data to produce (rows per region).
        horizon: How many steps ahead to forecast (t+1 .. t+horizon).
        volatility: One of "stable", "medium", "volatile".
        seed: Random seed for reproducibility.
        base_intensity: Optional {region_id: baseline gCO2/kWh}. Falls back to
            DEFAULT_BASE_INTENSITY for any region_id not given.
        timestep_duration: Time units per simulation step (for the timestamp
            column); defaults to 1.0 so timestamp == step index.

    Returns:
        A flat list of row dicts, one per (region, timestep), each with keys:
        timestamp, region_id, carbon_intensity_current,
        carbon_intensity_t+1, ..., carbon_intensity_t+<horizon>.
    """
    if n_timesteps <= 0:
        raise ValueError("n_timesteps must be > 0")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if volatility not in VOLATILITY_PRESETS:
        raise ValueError(
            f"Unknown volatility '{volatility}'. Choose from {list(VOLATILITY_PRESETS)}."
        )

    rng = random.Random(seed)
    preset = VOLATILITY_PRESETS[volatility]
    base_intensity = {**DEFAULT_BASE_INTENSITY, **(base_intensity or {})}

    rows: List[Dict[str, Any]] = []

    for region_id in region_ids:
        base = base_intensity.get(region_id, 400.0)

        # Generate extra steps beyond n_timesteps so t+horizon lookups near
        # the end of the series still have real "actual" values to draw on.
        actual = _generate_actual_series(
            n_steps=n_timesteps + horizon,
            base_intensity=base,
            preset=preset,
            rng=rng,
        )

        for t in range(n_timesteps):
            row: Dict[str, Any] = {
                "timestamp": round(t * timestep_duration, 3),
                "region_id": region_id,
                "carbon_intensity_current": round(actual[t], 2),
            }
            for k in range(1, horizon + 1):
                # Forecast = true future value + growing Gaussian error,
                # simulating imperfect but directionally-useful predictions.
                error_sigma = base * preset["forecast_error_frac_per_step"] * k
                forecast = actual[t + k] + rng.gauss(0, error_sigma)
                row[f"carbon_intensity_t+{k}"] = round(max(MIN_CARBON_INTENSITY, forecast), 2)
            rows.append(row)

    return rows


def _fieldnames(horizon: int) -> List[str]:
    return (
        ["timestamp", "region_id", "carbon_intensity_current"]
        + [f"carbon_intensity_t+{k}" for k in range(1, horizon + 1)]
    )


def save_carbon_data_to_csv(rows: List[Dict[str, Any]], filepath: str, horizon: int) -> None:
    """Write generated carbon-data rows to CSV with the required schema."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(horizon))
        writer.writeheader()
        writer.writerows(rows)


def load_carbon_data_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """Read carbon-data rows back from CSV, coercing numeric fields to float."""
    rows: List[Dict[str, Any]] = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row: Dict[str, Any] = {"region_id": raw_row["region_id"]}
            for key, value in raw_row.items():
                if key != "region_id":
                    row[key] = float(value)
            rows.append(row)
    return rows


class CarbonDataProvider:
    """Fast lookup of current + predicted carbon intensity, by region and timestep.

    This is what cloud_env.py (Phase 4) will query inside get_state() and
    calculate_carbon() — it hides whether the underlying data came from the
    synthetic generator above or, later, a real feed.
    """

    def __init__(self, rows: List[Dict[str, Any]]):
        self._horizon = 0
        self._by_region_and_time: Dict[str, Dict[float, Dict[str, Any]]] = {}
        for row in rows:
            region_id = row["region_id"]
            timestamp = row["timestamp"]
            self._by_region_and_time.setdefault(region_id, {})[timestamp] = row
            self._horizon = max(
                self._horizon,
                sum(1 for k in row if k.startswith("carbon_intensity_t+")),
            )

    @classmethod
    def from_csv(cls, filepath: str) -> "CarbonDataProvider":
        return cls(load_carbon_data_from_csv(filepath))

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def region_ids(self) -> List[str]:
        return list(self._by_region_and_time.keys())

    def get_current(self, region_id: str, timestamp: float) -> float:
        """Current carbon intensity for a region at a given timestamp."""
        return self._row(region_id, timestamp)["carbon_intensity_current"]

    def get_predicted(self, region_id: str, timestamp: float, steps_ahead: int) -> float:
        """Forecasted carbon intensity `steps_ahead` steps beyond `timestamp`."""
        if not (1 <= steps_ahead <= self._horizon):
            raise ValueError(f"steps_ahead must be in [1, {self._horizon}]")
        return self._row(region_id, timestamp)[f"carbon_intensity_t+{steps_ahead}"]

    def get_forecast_vector(self, region_id: str, timestamp: float) -> List[float]:
        """[current, t+1, t+2, ..., t+horizon] — convenient for state vectors."""
        row = self._row(region_id, timestamp)
        vector = [row["carbon_intensity_current"]]
        vector.extend(row[f"carbon_intensity_t+{k}"] for k in range(1, self._horizon + 1))
        return vector

    def _row(self, region_id: str, timestamp: float) -> Dict[str, Any]:
        try:
            return self._by_region_and_time[region_id][timestamp]
        except KeyError as exc:
            raise KeyError(
                f"No carbon data for region '{region_id}' at timestamp {timestamp}"
            ) from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic carbon-intensity data for the "
        "carbon-aware scheduler."
    )
    parser.add_argument(
        "--regions", type=str, default="A,B,C", help="Comma-separated region ids."
    )
    parser.add_argument("--n_timesteps", type=int, default=200, help="Timesteps to generate.")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon (t+1..t+n).")
    parser.add_argument(
        "--volatility",
        type=str,
        default="medium",
        choices=list(VOLATILITY_PRESETS.keys()),
        help="Carbon volatility preset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output", type=str, default="data/carbon_data.csv", help="Output CSV path."
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    region_ids = [r.strip() for r in args.regions.split(",") if r.strip()]

    rows = generate_carbon_series(
        region_ids=region_ids,
        n_timesteps=args.n_timesteps,
        horizon=args.horizon,
        volatility=args.volatility,
        seed=args.seed,
    )
    save_carbon_data_to_csv(rows, args.output, horizon=args.horizon)

    print(
        f"Generated {args.n_timesteps} timesteps x {len(region_ids)} regions "
        f"({args.volatility} volatility, horizon={args.horizon}, seed={args.seed})."
    )
    print(f"Saved to: {args.output}")

    provider = CarbonDataProvider(rows)
    print("\nSample lookups at timestamp=0:")
    for region_id in region_ids:
        vector = provider.get_forecast_vector(region_id, 0.0)
        print(f"  Region {region_id}: current={vector[0]:.1f}, forecast(t+1..)={vector[1:]}")


if __name__ == "__main__":
    main()
