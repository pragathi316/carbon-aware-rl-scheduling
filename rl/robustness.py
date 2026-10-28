"""
rl/robustness.py

Robustness checks of the main benchmark (evaluation only, no retraining).
Trained S2/S3 agents and the heuristics are re-evaluated on the same 30
held-out scenarios under modified conditions:

  * energy model "calibrated" (0.25 kW per GPU, 50 % CPU idle overhead,
    PUE 1.58) instead of the simple linear model used for training;
  * deadline slack factor tight U[1.2, 2.5] and loose U[2.0, 6.0] instead of
    U[1.5, 4.0] (the trace has no deadlines, so their tightness is an assumption).

Usage:  python -m rl.robustness --scenario S2 --algos MPPO --out results/benchmark/robustness
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from rl.benchmark import ALGOS, LoadedAgent, TEST_SEEDS, heuristic_fns, run_episode, scenario

CONDITIONS = {
    "baseline": {},
    "energy_calibrated": {"energy_model": "calibrated"},
    "slack_tight": {"trace_slack": (1.2, 2.5)},
    "slack_loose": {"trace_slack": (2.0, 6.0)},
}
KEYS = ["total_carbon_grams", "total_cost", "sla_violation_rate", "avg_delay", "total_reward"]


def sc_for(scen, carbon_aware, overrides):
    s = scenario(scen, carbon_aware=carbon_aware)
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def main():
    import torch
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--algos", nargs="+", default=["MPPO"])
    ap.add_argument("--dir", default="results/benchmark")
    ap.add_argument("--out", default="results/benchmark/robustness")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for cond, ov in CONDITIONS.items():
        res[cond] = {}
        for name, fn in heuristic_fns().items():
            sc = sc_for(a.scenario, True, ov)
            ms = [run_episode(sc, s, lambda o, e, fn=fn: int(fn(e.unwrapped))) for s in TEST_SEEDS]
            res[cond][name] = {"per_scenario": {k: [m[k] for m in ms] for k in KEYS}}
        for algo in a.algos:
            for shield in (False, True):
                label = algo + (" +shield" if shield else "")
                per_seed = []
                for zp in sorted(glob.glob(os.path.join(a.dir, a.scenario, "models", f"{algo}_seed*.zip"))):
                    agent = LoadedAgent(algo, zp[:-4])
                    sc = sc_for(a.scenario, not ALGOS[algo]["blind"], ov)
                    ms = [run_episode(sc, s, agent.act, shield=shield) for s in TEST_SEEDS]
                    per_seed.append({k: [m[k] for m in ms] for k in KEYS})
                res[cond][label] = {"per_seed": per_seed}
        print("done", cond, flush=True)
    path = os.path.join(a.out, f"{a.scenario}.json")
    json.dump(res, open(path, "w"))

    from scipy import stats
    lines = [f"Robustness ({a.scenario}): mean over 30 test scenarios (learned: mean over seeds).",
             "Paired test vs Least Carbon: Wilcoxon over scenarios of (agent seed-mean - LC)."]
    for cond in CONDITIONS:
        lines.append(f"\n=== {cond} ===")
        lc = res[cond]["Least Carbon"]["per_scenario"]
        for label, d in res[cond].items():
            if "per_scenario" in d:
                v = {k: np.array(d["per_scenario"][k]) for k in KEYS}
            else:
                v = {k: np.mean([np.array(ps[k]) for ps in d["per_seed"]], axis=0) for k in KEYS}
            extra = ""
            if label != "Least Carbon":
                for k in ("total_reward", "total_carbon_grams"):
                    diff = v[k] - np.array(lc[k])
                    p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                    extra += f"  d_{k.split('_')[1]}={diff.mean():+.3f} (p={p:.3f})"
            lines.append(f"{label:20s} carbon={v['total_carbon_grams'].mean():9.0f} SLA={100*v['sla_violation_rate'].mean():5.2f}% "
                         f"delay={v['avg_delay'].mean():5.2f} reward={v['total_reward'].mean():8.3f}{extra}")
    text = "\n".join(lines)
    open(os.path.join(a.out, f"{a.scenario}_report.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
