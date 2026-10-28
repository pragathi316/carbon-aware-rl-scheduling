"""
rl/tune_benchmark.py

Hyperparameter and training-reward-weight sweep for the revised benchmark.
Each algorithm gets its own small grid. Agents are trained on scenario S2's
training pool (seeds 0-19) for 100k steps with 2 seeds and scored on
VALIDATION scenarios 50-59 (disjoint from the training and test scenarios)
by the mean fixed-weight reward. The best configuration per algorithm is
written to <out>/best_configs.json and used for every scenario.

    python -m rl.tune_benchmark --out results/benchmark/tuning
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor

import numpy as np

GRIDS = {
    "MaskablePPO": {"learning_rate": [3e-4, 1e-4], "ent_coef": [0.0, 0.01], "train_alpha": [0.4, 1.2]},
    "A2C": {"learning_rate": [7e-4, 2e-4], "ent_coef": [0.0, 0.01]},
    "DQN": {"learning_rate": [1e-4, 5e-4], "exploration_fraction": [0.1, 0.3]},
}
ALGO_FOR = {"MaskablePPO": "MPPO", "A2C": "A2C", "DQN": "DQN"}


def _job(args):
    cls, cfg, seed, steps, out, scen = args
    import torch
    torch.set_num_threads(1)
    from rl.benchmark import train, LoadedAgent, evaluate_agent, VALIDATION_SEEDS
    algo = ALGO_FOR[cls]
    name = "_".join(f"{k}={v}" for k, v in cfg.items())
    path = os.path.join(out, "models", f"{cls}_{name}_seed{seed}")
    model, venv = train(algo, scen, steps, seed, path, dict(cfg))
    res = evaluate_agent(LoadedAgent(algo, path, model, venv), scen, VALIDATION_SEEDS, False)
    ms = list(res.values())
    return dict(algo=cls, config=cfg, seed=seed, reward=float(np.mean([m["total_reward"] for m in ms])),
                carbon=float(np.mean([m["total_carbon_grams"] for m in ms])),
                sla=float(np.mean([m["sla_violation_rate"] for m in ms])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/benchmark/tuning")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for cls, grid in GRIDS.items():
        keys = list(grid)
        for vals in itertools.product(*[grid[k] for k in keys]):
            for s in range(a.seeds):
                jobs.append((cls, dict(zip(keys, vals)), 2000 + s, a.steps, a.out, a.scenario))
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(_job, jobs))
    json.dump(rows, open(os.path.join(a.out, "sweep.json"), "w"), indent=1)
    best, lines = {}, []
    for cls in GRIDS:
        groups = {}
        for r in rows:
            if r["algo"] == cls:
                groups.setdefault(json.dumps(r["config"], sort_keys=True), []).append(r)
        ranked = sorted(groups.items(), key=lambda kv: -statistics.mean(x["reward"] for x in kv[1]))
        for k, rs in ranked:
            lines.append(f"{cls:12s} {k:70s} reward={statistics.mean(x['reward'] for x in rs):8.3f} "
                         f"carbon={statistics.mean(x['carbon'] for x in rs):8.0f} sla={100*statistics.mean(x['sla'] for x in rs):5.2f}%")
        best[cls] = json.loads(ranked[0][0])
    # PPO and the carbon-blind MaskablePPO reuse the MaskablePPO selection (same algorithm family)
    best["PPO"] = dict(best["MaskablePPO"])
    out = {"MPPO": best["MaskablePPO"], "MPPO-blind": best["MaskablePPO"], "PPO": best["PPO"],
           "A2C": best["A2C"], "DQN": best["DQN"]}
    json.dump(out, open(os.path.join(a.out, "best_configs.json"), "w"), indent=1)
    text = "\n".join(lines)
    open(os.path.join(a.out, "sweep_table.txt"), "w").write(text + "\n")
    print(text)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
