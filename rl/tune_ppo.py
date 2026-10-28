"""
rl/tune_ppo.py

Small PPO hyperparameter sweep, run on VALIDATION scenarios that are disjoint
from both the training pool (seeds 0-19) and the test scenarios (100-129):
validation seeds are 50-59. The selected configuration is written to
<out>/best_config.json and is then used unchanged for every variant in
rl/experiment.py, so the test scenarios play no role in tuning.

Selection rule (fixed before running): highest mean validation reward
(fixed weights, delta = 0.6) over the tuning seeds.

Usage:
    python -m rl.tune_ppo                         # 8 configs x 2 seeds x 100k steps
    python -m rl.tune_ppo --steps 4096 --seeds 1  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CONFIGS = {
    "default":      {},
    "lr_1e-4":      {"learning_rate": 1e-4},
    "lr_1e-3":      {"learning_rate": 1e-3},
    "ent_0":        {"ent_coef": 0.0},
    "ent_0.05":     {"ent_coef": 0.05},
    "net_128x128":  {"net_arch": [128, 128]},
    "batch_2048":   {"n_steps": 2048, "batch_size": 128},
    "gamma_0.95":   {"gamma": 0.95},
}
VALIDATION_SEEDS = list(range(50, 60))


def _job(args):
    name, overrides, seed, steps, out, source = args
    import torch
    torch.set_num_threads(1)
    from rl.experiment import train, scenario_for
    from rl.generalization import run_policy_episode
    model, _ = train("M3", source, steps, seed, list(range(20)),
                     os.path.join(out, "models", f"{name}_seed{seed}"), overrides)
    scen = scenario_for("M3", source)
    ms = [run_policy_episode(model, scen, s) for s in VALIDATION_SEEDS]
    return dict(config=name, seed=seed,
                reward=float(np.mean([m["total_reward"] for m in ms])),
                carbon=float(np.mean([m["total_carbon_grams"] for m in ms])),
                sla=float(np.mean([m["sla_violation_rate"] for m in ms])),
                invalid=float(np.mean([m["invalid_actions"] for m in ms])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--carbon_source", default="synthetic")
    ap.add_argument("--out", default="results/tuning")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(n, o, 2000 + k, args.steps, args.out, args.carbon_source)
            for n, o in CONFIGS.items() for k in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(_job, jobs))
    with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for name in CONFIGS:
        sel = [r for r in rows if r["config"] == name]
        summary.append((name, statistics.mean(r["reward"] for r in sel), statistics.mean(r["carbon"] for r in sel),
                        100 * statistics.mean(r["sla"] for r in sel), statistics.mean(r["invalid"] for r in sel)))
    summary.sort(key=lambda t: -t[1])
    lines = [f"{'config':14s} {'val reward':>11s} {'val carbon':>11s} {'SLA %':>7s} {'invalid/ep':>10s}"]
    lines += [f"{n:14s} {r:>11.3f} {c:>11.0f} {s:>7.2f} {i:>10.1f}" for n, r, c, s, i in summary]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "sweep_table.txt"), "w") as f:
        f.write(text + "\n")
    best = summary[0][0]
    with open(os.path.join(args.out, "best_config.json"), "w") as f:
        json.dump({"name": best, "overrides": CONFIGS[best], "selection": "max mean validation reward",
                   "validation_seeds": VALIDATION_SEEDS, "steps": args.steps, "seeds": args.seeds}, f, indent=2)
    print(f"best: {best} {CONFIGS[best]}")


if __name__ == "__main__":
    main()
