"""
rl/sensitivity.py

Sensitivity experiments: how do the schedulers behave when the carbon
volatility and the workload intensity differ from the training setting
(medium / medium)?

IMPORTANT: this is a ZERO-SHOT robustness test. The PPO models were trained
on medium volatility / medium workload (rl.generalization) and are only
EVALUATED here under the other conditions; the heuristics are run on the same
scenarios. It does not retrain per condition.

Grid: volatility in {stable, medium, volatile} x intensity in {light, medium, heavy},
evaluated on held-out scenario seeds.

Usage:
    python -m rl.sensitivity --dir results/generalization --variants M2 M3 M3.5 --n_test 10
Output: <dir>/sensitivity.csv and <dir>/sensitivity_table.txt
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List

import numpy as np

from rl.generalization import (
    METRIC_KEYS, VARIANTS, model_files, variant_scenario, run_policy_episode, run_heuristic_episode, _heuristic_fns,
)

VOLATILITIES = ["stable", "medium", "volatile"]
INTENSITIES = ["light", "medium", "heavy"]


def main() -> None:
    from stable_baselines3 import PPO

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/generalization")
    ap.add_argument("--variants", nargs="+", default=["M2", "M3", "M3.5"], choices=list(VARIANTS))
    ap.add_argument("--n_test", type=int, default=10)
    args = ap.parse_args()

    test_seeds = list(range(100, 100 + args.n_test))
    heuristics = _heuristic_fns()
    rows: List[Dict[str, Any]] = []

    for vol in VOLATILITIES:
        for inten in INTENSITIES:
            overrides = {"carbon_volatility": vol, "workload_intensity": inten}
            print(f"condition: volatility={vol}, intensity={inten}", flush=True)

            base = variant_scenario("M3", overrides)
            for name, fn in heuristics.items():
                ms = [run_heuristic_episode(base, s, fn) for s in test_seeds]
                rows.append(dict(volatility=vol, intensity=inten, model=name, ppo_seed="",
                                 **{k: float(np.mean([m[k] for m in ms])) for k in METRIC_KEYS}))

            for variant in args.variants:
                for path in model_files(args.dir, variant):
                    seed = os.path.basename(path).split("seed")[-1].replace(".zip", "")
                    model = PPO.load(path)
                    scen = variant_scenario(variant, overrides)
                    ms = [run_policy_episode(model, scen, s) for s in test_seeds]
                    rows.append(dict(volatility=vol, intensity=inten, model=variant, ppo_seed=seed,
                                     **{k: float(np.mean([m[k] for m in ms])) for k in METRIC_KEYS}))

    csv_path = os.path.join(args.dir, "sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volatility", "intensity", "model", "ppo_seed"] + METRIC_KEYS)
        w.writeheader()
        w.writerows(rows)

    lines = ["Zero-shot sensitivity (models trained at medium/medium). Carbon in g, mean over scenarios;",
             "learned models: mean over training seeds. 'rel M1' = carbon relative to M1 under the same condition.\n"]
    models = list(heuristics) + args.variants
    lines.append(f"{'volatility':10s} {'intensity':9s} " + " ".join(f"{m[:18]:>18s}" for m in models))
    for vol in VOLATILITIES:
        for inten in INTENSITIES:
            m1 = [r for r in rows if r["volatility"] == vol and r["intensity"] == inten and r["model"] == "M1 Deadline+Carbon"][0]["total_carbon_grams"]
            cells = []
            for m in models:
                sel = [r["total_carbon_grams"] for r in rows if r["volatility"] == vol and r["intensity"] == inten and r["model"] == m]
                cells.append(f"{np.mean(sel):9.0f} ({np.mean(sel) / m1:4.2f}x)" if sel else f"{'-':>18s}")
            lines.append(f"{vol:10s} {inten:9s} " + " ".join(f"{c:>18s}" for c in cells))
    lines.append("\nSLA violation rate (%), mean:")
    lines.append(f"{'volatility':10s} {'intensity':9s} " + " ".join(f"{m[:18]:>18s}" for m in models))
    for vol in VOLATILITIES:
        for inten in INTENSITIES:
            cells = []
            for m in models:
                sel = [r["sla_violation_rate"] * 100 for r in rows if r["volatility"] == vol and r["intensity"] == inten and r["model"] == m]
                cells.append(f"{np.mean(sel):18.1f}" if sel else f"{'-':>18s}")
            lines.append(f"{vol:10s} {inten:9s} " + " ".join(cells))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.dir, "sensitivity_table.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
