"""
rl/agent_components.py

Re-evaluates trained agents on the validation (50-59) and test (100-129)
scenarios, with and without the shield, and stores the reward components
(sum of normalized carbon, cost, waiting time, violations and the shaping
term) so the results can be re-scored under other reward weightings.

Usage:  python -m rl.agent_components --scenario S2 --algos MPPO MPPO-blind --workers 2
Output: results/benchmark/agent_components/<S>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

from rl.benchmark import ALGOS, TEST_SEEDS, VALIDATION_SEEDS, LoadedAgent, run_episode, scenario
from rl.tune_heuristics import KEEP


def _eval(task):
    import torch
    torch.set_num_threads(1)
    scen, algo, path, models_tag = task
    agent = LoadedAgent(algo, path)
    sc = scenario(scen, carbon_aware=not ALGOS[algo]["blind"])
    if models_tag:
        from rl.revision_train import apply_variant
        sc = apply_variant(sc, models_tag)
    out = {}
    for shield in (False, True):
        for split, seeds in (("validation", VALIDATION_SEEDS), ("test", TEST_SEEDS)):
            out[f"{split}{'_shielded' if shield else ''}"] = {
                str(s): {k: m[k] for k in KEEP}
                for s in seeds for m in [run_episode(sc, s, agent.act, shield=shield)]}
    return algo, os.path.basename(path), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--algos", nargs="+", default=["MPPO", "MPPO-blind"])
    ap.add_argument("--dir", default="results/benchmark")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/benchmark/agent_components")
    a = ap.parse_args()
    tasks = []
    for algo in a.algos:
        for zp in sorted(glob.glob(os.path.join(a.dir, a.scenario, "models", f"{algo}_seed*.zip"))):
            tasks.append((a.scenario, algo, zp[:-4], None))
    res = {}
    with ProcessPoolExecutor(a.workers) as pool:
        for algo, name, out in pool.map(_eval, tasks):
            res.setdefault(algo, {})[name] = out
            print("done", name, flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"{a.scenario}.json"), "w"))


if __name__ == "__main__":
    main()
