"""
rl/experiment.py

Main experiment runner (v2). Extends rl/generalization.py with:
  * real carbon-intensity traces (--carbon_source real_eu | real_mixed),
    with a time-based train/test split (train: Jan-Aug 2020, test: Sep-Dec 2020);
  * tuned PPO hyperparameters (--ppo_config, produced by rl/tune_ppo.py);
  * the constrained variant M3-Lag (Lagrangian SLA penalty, rl/constrained.py);
  * evaluation with and without the SLA shield (rl/shield.py);
  * behaviour statistics (action shares, energy share per region, invalid
    actions, shield overrides) for every scheduler.

Variants
    M2       carbon-blind PPO
    M3       carbon-aware PPO
    M3.5     carbon-aware PPO + rule-based context features
    M3-Lag   carbon-aware PPO trained with the adaptive (Lagrangian) SLA penalty
    M3-a5    carbon-aware PPO trained with a 5x larger carbon weight (reward-scale test);
             evaluation still reports the standard fixed-weight reward
    M3-obs   carbon-aware PPO with an improved observation: job GPU demand and free
             GPUs per region added (25 features), all features divided by fixed scales
    M3-scale / M3-gpu   only one of the two M3-obs changes (attribution)
    M2-obs   carbon-blind PPO with the improved observation (13 features)

Usage
    python -m rl.experiment --carbon_source synthetic --seeds 10 --steps 200000 \
        --ppo_config results/tuning/best_config.json --out results/v2/synthetic
    python -m rl.experiment --carbon_source real_eu --seeds 10 --steps 200000 \
        --ppo_config results/tuning/best_config.json --out results/v2/real_eu
    python -m rl.experiment --steps 4096 --seeds 1 --n_test 3 --out results/v2/smoke   # quick check

Outputs (under --out): raw/*.json, models/*.zip, per_seed_summary.csv,
summary_table.txt, behaviour_table.txt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from environment.reward import RewardWeights
from rl.config import RunConfig
from rl.env_factory import build_env
from rl.generalization import RandomizedScenarioEnv, run_policy_episode, variant_scenario, VARIANTS as BASE_VARIANTS

VARIANTS = {
    "M2": dict(base="M2", constrained=False),
    "M3": dict(base="M3", constrained=False),
    "M3.5": dict(base="M3.5", constrained=False),
    "M3-Lag": dict(base="M3", constrained=True),
    # reward-scale test: carbon weight x5 in the TRAINING reward only (alpha 0.4 -> 2.0)
    "M3-a5": dict(base="M3", constrained=False, train_alpha=2.0),
    # improved observation: GPU demand/availability features + fixed input scaling
    "M3-obs": dict(base="M3", constrained=False, scenario_overrides={"include_gpu_state": True, "scale_obs": True}),
    # attribution of the M3-obs gain, and carbon visibility re-tested with the improved observation
    "M3-scale": dict(base="M3", constrained=False, scenario_overrides={"scale_obs": True}),
    "M3-gpu": dict(base="M3", constrained=False, scenario_overrides={"include_gpu_state": True}),
    "M2-obs": dict(base="M2", constrained=False, scenario_overrides={"include_gpu_state": True, "scale_obs": True}),
}
HEURISTICS = ["M0 FIFO", "Least Carbon", "M1 Deadline+Carbon"]
METRICS = ["total_carbon_grams", "total_cost", "avg_delay", "sla_violation_rate", "total_reward"]
BEHAVIOUR = ["share_action_3", "energy_share_A", "energy_share_B", "energy_share_C",
             "invalid_actions", "shield_overrides"]


def scenario_for(variant_or_none: Optional[str], carbon_source: str):
    base = VARIANTS[variant_or_none]["base"] if variant_or_none else "M3"
    overrides = {"carbon_source": carbon_source}
    if variant_or_none:
        overrides.update(VARIANTS[variant_or_none].get("scenario_overrides", {}))
    return variant_scenario(base, overrides)


def ppo_kwargs(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = RunConfig().ppo
    kw = dict(n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
              learning_rate=cfg.learning_rate, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
              clip_range=cfg.clip_range, ent_coef=cfg.ent_coef, net_arch=list(cfg.net_arch))
    kw.update(overrides or {})
    net_arch = list(kw.pop("net_arch"))
    kw["policy_kwargs"] = dict(net_arch=net_arch)
    return kw


def train(variant: str, carbon_source: str, steps: int, ppo_seed: int, train_pool: List[int],
          save_path: str, ppo_overrides: Optional[Dict[str, Any]] = None):
    """Train one agent. Returns (model, lambda_history or None)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.constrained import LagrangianSLACallback

    scenario = scenario_for(variant, carbon_source)
    weights = RewardWeights()  # shared with the env; the Lagrangian callback adapts weights.delta
    if "train_alpha" in VARIANTS[variant]:
        weights.alpha = VARIANTS[variant]["train_alpha"]
    vec_env = DummyVecEnv([lambda: RandomizedScenarioEnv(scenario, train_pool, rng_seed=ppo_seed,
                                                         reward_weights=weights)])
    model = PPO("MlpPolicy", vec_env, seed=ppo_seed, verbose=0, **ppo_kwargs(ppo_overrides))
    callback = LagrangianSLACallback(weights) if VARIANTS[variant]["constrained"] else None
    model.learn(total_timesteps=steps, callback=callback)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path if save_path.endswith(".zip") else save_path + ".zip")
    return model, (callback.history if callback else None)


def _heuristic_fns() -> Dict[str, Callable]:
    from baselines.fifo import pick_action as fifo
    from baselines.least_carbon import pick_action as lc
    from baselines.deadline_carbon import pick_action as dc
    return {"M0 FIFO": fifo, "Least Carbon": lc, "M1 Deadline+Carbon": dc}


def run_heuristic(scenario, seed: int, fn: Callable) -> Dict[str, float]:
    env = build_env(scenario, seed=seed)
    env.reset(seed=seed)
    done = False
    total = 0.0
    counts = [0, 0, 0, 0]
    info: Dict[str, Any] = {}
    while not done:
        a = int(fn(env.unwrapped))
        counts[a] += 1
        _, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    m = dict(info["episode_metrics"])
    m["total_reward"] = total
    n = max(1, sum(counts))
    m.update({f"share_action_{k}": counts[k] / n for k in range(4)})
    m["invalid_actions"] = 0
    m["shield_overrides"] = 0
    completed = env.unwrapped.completed_jobs
    energy = sum(j["energy_kwh"] for j in completed) or 1.0
    for rid in ("A", "B", "C"):
        m[f"energy_share_{rid}"] = sum(j["energy_kwh"] for j in completed if j["region_id"] == rid) / energy
    return m


def _worker(task: Dict[str, Any]) -> str:
    import torch
    torch.set_num_threads(1)

    variant, seed, out = task["variant"], task["ppo_seed"], task["out"]
    raw_path = os.path.join(out, "raw", f"{variant}_seed{seed}.json")
    if os.path.exists(raw_path):
        return f"skip {variant} seed {seed}"
    model, lam = train(variant, task["carbon_source"], task["steps"], seed, task["train_pool"],
                       os.path.join(out, "models", f"{variant}_seed{seed}"), task["ppo_overrides"])
    scen = scenario_for(variant, task["carbon_source"])
    result = {
        "variant": variant, "ppo_seed": seed, "steps": task["steps"], "carbon_source": task["carbon_source"],
        "ppo_overrides": task["ppo_overrides"], "lambda_history": lam,
        "heldout": {str(s): run_policy_episode(model, scen, s) for s in task["test_seeds"]},
        "heldout_shielded": {str(s): run_policy_episode(model, scen, s, shield=True) for s in task["test_seeds"]},
        "in_distribution": {str(s): run_policy_episode(model, scen, s) for s in task["indist_seeds"]},
    }
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(result, f)
    return f"done {variant} seed {seed}"


def run_heuristics(out: str, carbon_source: str, test_seeds: List[int], indist_seeds: List[int]) -> None:
    path = os.path.join(out, "raw", "heuristics.json")
    if os.path.exists(path):
        return
    scen = scenario_for(None, carbon_source)
    res = {name: {"heldout": {str(s): run_heuristic(scen, s, fn) for s in test_seeds},
                  "in_distribution": {str(s): run_heuristic(scen, s, fn) for s in indist_seeds}}
           for name, fn in _heuristic_fns().items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f)


def _mean(per_scenario: Dict[str, Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    return {k: float(np.mean([m[k] for m in per_scenario.values()])) for k in keys}


def aggregate(out: str) -> None:
    raw = os.path.join(out, "raw")
    rows: List[Dict[str, Any]] = []
    with open(os.path.join(raw, "heuristics.json")) as f:
        heur = json.load(f)
    for name, d in heur.items():
        for split in ("heldout", "in_distribution"):
            rows.append({"model": name, "split": split, "ppo_seed": "", **_mean(d[split], METRICS + BEHAVIOUR)})
    for fname in sorted(os.listdir(raw)):
        if not fname.endswith(".json") or fname == "heuristics.json":
            continue
        with open(os.path.join(raw, fname)) as f:
            d = json.load(f)
        for split in ("heldout", "heldout_shielded", "in_distribution"):
            label = d["variant"] + (" +shield" if split == "heldout_shielded" else "")
            rows.append({"model": label, "split": "heldout" if split == "heldout_shielded" else split,
                         "ppo_seed": d["ppo_seed"], **_mean(d[split], METRICS + BEHAVIOUR)})

    with open(os.path.join(out, "per_seed_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "ppo_seed"] + METRICS + BEHAVIOUR)
        w.writeheader()
        w.writerows(rows)

    order = HEURISTICS + [v for v in VARIANTS] + [v + " +shield" for v in VARIANTS]

    def fmt(sel, key, scale=1.0, nd=0):
        vals = [r[key] * scale for r in sel]
        if len(vals) == 1:
            return f"{vals[0]:.{nd}f}"
        return f"{statistics.mean(vals):.{nd}f}+/-{statistics.stdev(vals):.{nd}f}"

    lines = []
    for split in ("heldout", "in_distribution"):
        lines.append(f"\n=== {split} (mean over scenarios; +/- std over training seeds) ===")
        lines.append(f"{'Model':20s} {'n':>2s} {'Carbon (g)':>15s} {'Cost':>11s} {'AvgDelay':>11s} {'SLA %':>10s} {'Reward':>15s}")
        for m in order:
            sel = [r for r in rows if r["model"] == m and r["split"] == split]
            if sel:
                lines.append(f"{m:20s} {len(sel):>2d} {fmt(sel,'total_carbon_grams'):>15s} {fmt(sel,'total_cost',nd=2):>11s} "
                             f"{fmt(sel,'avg_delay',nd=2):>11s} {fmt(sel,'sla_violation_rate',100,2):>10s} {fmt(sel,'total_reward',nd=3):>15s}")
    lines.append("\n=== behaviour on held-out scenarios (means) ===")
    lines.append(f"{'Model':20s} {'delay share':>11s} {'energy A':>9s} {'energy B':>9s} {'energy C':>9s} {'invalid/ep':>10s} {'overrides/ep':>12s}")
    for m in order:
        sel = [r for r in rows if r["model"] == m and r["split"] == "heldout"]
        if sel:
            g = lambda k: statistics.mean(r[k] for r in sel)
            lines.append(f"{m:20s} {g('share_action_3'):>11.3f} {g('energy_share_A'):>9.3f} {g('energy_share_B'):>9.3f} "
                         f"{g('energy_share_C'):>9.3f} {g('invalid_actions'):>10.2f} {g('shield_overrides'):>12.2f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out, "summary_table.txt"), "w") as f:
        f.write(text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carbon_source", default="synthetic", choices=["synthetic", "real_eu", "real_mixed"])
    ap.add_argument("--variants", nargs="+", default=["M3", "M3-Lag", "M2", "M3.5"], choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first_seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--ppo_config", default=None, help="JSON with PPO overrides (from rl.tune_ppo)")
    ap.add_argument("--train_pool", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/v2/synthetic")
    args = ap.parse_args()

    overrides = None
    if args.ppo_config:
        with open(args.ppo_config) as f:
            overrides = json.load(f)["overrides"]
    train_pool = list(range(args.train_pool))
    test_seeds = list(range(100, 100 + args.n_test))
    indist = train_pool[:10]
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    run_heuristics(args.out, args.carbon_source, test_seeds, indist)

    tasks = [dict(variant=v, ppo_seed=args.first_seed + k, steps=args.steps, train_pool=train_pool,
                  test_seeds=test_seeds, indist_seeds=indist, out=args.out,
                  carbon_source=args.carbon_source, ppo_overrides=overrides)
             for k in range(args.seeds) for v in args.variants]
    print(f"{len(tasks)} runs x {args.steps} steps ({args.carbon_source}), workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_worker, t) for t in tasks]):
            print(fut.result(), flush=True)
    aggregate(args.out)


if __name__ == "__main__":
    main()
