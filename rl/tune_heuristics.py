"""
rl/tune_heuristics.py

Tunes the thresholds of the deadline-aware heuristics (M1, Wait-for-Clean) and
of the new Forecast-Optimal heuristic on the SAME validation scenarios (seeds
50-59) that were used to tune the RL hyperparameters, selecting by mean
evaluation reward. Every grid point is also evaluated on the 30 test scenarios,
which gives the threshold sweeps for the carbon-vs-SLA Pareto plot; the test
results are never used for selection.

Usage:  python -m rl.tune_heuristics --scenario S2 --workers 2
Output: results/benchmark/heuristic_tuning/<S>.json
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from rl.benchmark import TEST_SEEDS, VALIDATION_SEEDS, heuristic_fns, run_episode, scenario

KEEP = ["total_carbon_grams", "total_cost", "sla_violation_rate", "avg_delay", "total_reward", "share_action_3",
        "energy_share_A", "energy_share_B", "energy_share_C", "sum_c", "sum_k", "sum_w", "sum_v", "shaping"]


def _eval(task):
    from baselines.tunable import FAMILIES
    scen, family, params, seeds = task
    if family == "fixed":
        fn = heuristic_fns()[params["name"]]
    else:
        fn = FAMILIES[family][0](**params)
    sc = scenario(scen)
    out = {}
    for s in seeds:
        m = run_episode(sc, s, lambda o, e: int(fn(e.unwrapped)))
        out[str(s)] = {k: m[k] for k in KEEP}
    return out


def main():
    from baselines.tunable import FAMILIES
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/benchmark/heuristic_tuning")
    a = ap.parse_args()
    entries = [("fixed", {"name": n}) for n in heuristic_fns()]
    for fam, (_, grid) in FAMILIES.items():
        entries += [(fam, p) for p in grid]
    tasks = []
    for fam, p in entries:
        tasks.append((a.scenario, fam, p, VALIDATION_SEEDS))
        tasks.append((a.scenario, fam, p, TEST_SEEDS))
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(_eval, tasks, chunksize=4))
    res = []
    for i, (fam, p) in enumerate(entries):
        res.append({"family": fam, "params": p, "validation": results[2 * i], "test": results[2 * i + 1]})
    best = {}
    for fam in FAMILIES:
        cand = [r for r in res if r["family"] == fam]
        score = lambda r: np.mean([v["total_reward"] for v in r["validation"].values()])
        b = max(cand, key=score)
        best[fam] = {"params": b["params"], "validation_reward": score(b),
                     "test_reward": float(np.mean([v["total_reward"] for v in b["test"].values()]))}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"scenario": a.scenario, "entries": res, "best": best}, open(os.path.join(a.out, f"{a.scenario}.json"), "w"))
    for fam, b in best.items():
        print(a.scenario, fam, b)


if __name__ == "__main__":
    main()
