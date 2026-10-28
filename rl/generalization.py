"""
rl/generalization.py

Longer training + held-out scenario evaluation (roadmap items: "close the gap
to M1", "test on different scenario seeds", "ablation study").

What it does
  * Trains PPO variants for many more steps than the 50k prototype
    (default 200k) on a POOL of randomized training scenarios: every episode
    draws a fresh workload + carbon trace from `--train_pool` seeds.
  * Evaluates every trained model on
        - HELD-OUT scenarios (seeds never used for training), and
        - a sample of the training-pool scenarios (in-distribution),
    and evaluates the heuristics (FIFO, Least Carbon, Deadline+Carbon) on
    exactly the same scenarios.
  * Saves one JSON per (variant, training seed), so an interrupted run can be
    restarted and will skip finished work.

Variants
    M2         carbon-blind agent
    M3         carbon-aware agent
    M3.5       carbon-aware + rule-based context features
    M3.5-rand  ablation CONTROL: same 25-dim input as M3.5, but the 4 context
               features are random noise (no information)

Usage
    python -m rl.generalization                       # defaults: 200k steps, 5 seeds, M2 M3 M3.5
    python -m rl.generalization --variants M3 M3.5 M3.5-rand --seeds 5 --steps 200000 --workers 2
    python -m rl.generalization --steps 20000 --seeds 2 --n_test 5 --out results/generalization_smoke   # quick check

Outputs (under --out)
    raw/<variant>_seed<k>.json       per-run metrics on every evaluation scenario
    raw/heuristics.json              heuristic metrics on the same scenarios
    models/<variant>_seed<k>.zip     trained models
    per_seed_summary.csv             one row per (variant, training seed): mean over test scenarios
    summary_table.txt                mean +/- std across training seeds
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

import gymnasium as gym
import numpy as np

from rl.config import RunConfig, ScenarioConfig
from rl.env_factory import build_env

VARIANTS: Dict[str, Dict[str, Any]] = {
    "M2": dict(carbon_aware_state=False, use_llm_context=False, llm_context_backend="rule_based"),
    "M3": dict(carbon_aware_state=True, use_llm_context=False, llm_context_backend="rule_based"),
    "M3.5": dict(carbon_aware_state=True, use_llm_context=True, llm_context_backend="rule_based"),
    "M3.5-rand": dict(carbon_aware_state=True, use_llm_context=True, llm_context_backend="random"),
}

HEURISTIC_NAMES = ["M0 FIFO", "Least Carbon", "M1 Deadline+Carbon"]


def variant_scenario(variant: str, overrides: Optional[Dict[str, Any]] = None) -> ScenarioConfig:
    scenario = ScenarioConfig()
    for key, value in VARIANTS[variant].items():
        setattr(scenario, key, value)
    for key, value in (overrides or {}).items():
        setattr(scenario, key, value)
    return scenario


class RandomizedScenarioEnv(gym.Env):
    """Each reset() draws a new scenario seed from a pool, rebuilding the
    workload and the carbon trace. Used for training so the agent does not
    just memorize one scenario."""

    metadata = {"render_modes": []}

    def __init__(self, scenario: ScenarioConfig, seed_pool: List[int], rng_seed: int = 0,
                 reward_weights=None):
        super().__init__()
        self.scenario = scenario
        self.seed_pool = list(seed_pool)
        self.reward_weights = reward_weights  # shared object; the Lagrangian callback mutates .delta
        self._rng = np.random.default_rng(rng_seed)
        self._inner = build_env(scenario, seed=self.seed_pool[0], reward_weights=reward_weights)
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        scenario_seed = int(self._rng.choice(self.seed_pool))
        self._inner = build_env(self.scenario, seed=scenario_seed, reward_weights=self.reward_weights)
        return self._inner.reset(seed=scenario_seed)

    def step(self, action):
        return self._inner.step(action)

    def action_masks(self):
        return self._inner.unwrapped.action_masks()


def _heuristic_fns() -> Dict[str, Callable]:
    from baselines.fifo import pick_action as fifo
    from baselines.least_carbon import pick_action as lc
    from baselines.deadline_carbon import pick_action as dc

    return {"M0 FIFO": fifo, "Least Carbon": lc, "M1 Deadline+Carbon": dc}


def run_heuristic_episode(scenario: ScenarioConfig, scenario_seed: int, fn: Callable) -> Dict[str, float]:
    env = build_env(scenario, seed=scenario_seed)
    env.reset(seed=scenario_seed)
    terminated = truncated = False
    total_reward = 0.0
    info: Dict[str, Any] = {}
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(fn(env.unwrapped))
        total_reward += reward
    metrics = dict(info["episode_metrics"])
    metrics["total_reward"] = total_reward
    return metrics


def run_policy_episode(model, scenario: ScenarioConfig, scenario_seed: int,
                       shield: bool = False) -> Dict[str, float]:
    """Deterministic episode of a trained policy. With shield=True the SLA
    shield (rl/shield.py) may override risky actions. Besides the episode
    metrics, records behaviour statistics: share of each action, invalid
    actions, shield overrides, and the share of energy placed in each region."""
    from rl.shield import shield_action

    env = build_env(scenario, seed=scenario_seed)
    obs, _ = env.reset(seed=scenario_seed)
    terminated = truncated = False
    total_reward = 0.0
    info: Dict[str, Any] = {}
    counts = [0, 0, 0, 0]
    invalid = overrides = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        if shield:
            action, overridden = shield_action(env.unwrapped, action)
            overrides += int(overridden)
        counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        invalid += int(bool(info.get("invalid")))
        total_reward += reward
    metrics = dict(info["episode_metrics"])
    metrics["total_reward"] = total_reward
    n = max(1, sum(counts))
    metrics.update({f"share_action_{k}": counts[k] / n for k in range(4)})
    metrics["n_decisions"] = sum(counts)
    metrics["invalid_actions"] = invalid
    metrics["shield_overrides"] = overrides
    completed = env.unwrapped.completed_jobs
    energy = sum(j["energy_kwh"] for j in completed) or 1.0
    for rid in ("A", "B", "C"):
        metrics[f"energy_share_{rid}"] = sum(j["energy_kwh"] for j in completed if j["region_id"] == rid) / energy
    return metrics


def train_variant(
    variant: str,
    total_timesteps: int,
    ppo_seed: int,
    train_pool: List[int],
    save_path: str,
    overrides: Optional[Dict[str, Any]] = None,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    scenario = variant_scenario(variant, overrides)
    ppo_cfg = RunConfig().ppo
    vec_env = DummyVecEnv([lambda: RandomizedScenarioEnv(scenario, train_pool, rng_seed=ppo_seed)])
    model = PPO(
        "MlpPolicy", vec_env,
        n_steps=ppo_cfg.n_steps, batch_size=ppo_cfg.batch_size, n_epochs=ppo_cfg.n_epochs,
        learning_rate=ppo_cfg.learning_rate, gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range, ent_coef=ppo_cfg.ent_coef, seed=ppo_seed, verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # SB3 only appends ".zip" when the name has no dot, and "M3.5_seed1000" contains one.
    model.save(save_path if save_path.endswith(".zip") else save_path + ".zip")
    return model


def model_files(out_dir: str, variant: str) -> List[str]:
    """Trained model files for a variant (handles names saved with or without .zip)."""
    import glob

    found = sorted(glob.glob(os.path.join(out_dir, "models", f"{variant}_seed*")))
    return found


def _worker(task: Dict[str, Any]) -> str:
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO

    variant, ppo_seed, out = task["variant"], task["ppo_seed"], task["out"]
    raw_path = os.path.join(out, "raw", f"{variant}_seed{ppo_seed}.json")
    if os.path.exists(raw_path):
        return f"skip {variant} seed {ppo_seed} (already done)"

    model_path = os.path.join(out, "models", f"{variant}_seed{ppo_seed}")
    model = train_variant(variant, task["steps"], ppo_seed, task["train_pool"], model_path)

    scenario = variant_scenario(variant)
    result = {
        "variant": variant, "ppo_seed": ppo_seed, "steps": task["steps"],
        "heldout": {str(s): run_policy_episode(model, scenario, s) for s in task["test_seeds"]},
        "in_distribution": {str(s): run_policy_episode(model, scenario, s) for s in task["indist_seeds"]},
    }
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(result, f)
    return f"done {variant} seed {ppo_seed}"


def run_heuristics(out: str, test_seeds: List[int], indist_seeds: List[int]) -> None:
    path = os.path.join(out, "raw", "heuristics.json")
    if os.path.exists(path):
        return
    scenario = ScenarioConfig()
    fns = _heuristic_fns()
    result = {
        name: {
            "heldout": {str(s): run_heuristic_episode(scenario, s, fn) for s in test_seeds},
            "in_distribution": {str(s): run_heuristic_episode(scenario, s, fn) for s in indist_seeds},
        }
        for name, fn in fns.items()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f)


METRIC_KEYS = ["total_carbon_grams", "total_cost", "avg_delay", "sla_violation_rate", "total_reward"]


def _mean_over_scenarios(per_scenario: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {k: float(np.mean([m[k] for m in per_scenario.values()])) for k in METRIC_KEYS}


def aggregate(out: str) -> None:
    raw_dir = os.path.join(out, "raw")
    rows: List[Dict[str, Any]] = []
    with open(os.path.join(raw_dir, "heuristics.json")) as f:
        heur = json.load(f)
    for name, d in heur.items():
        for split in ("heldout", "in_distribution"):
            rows.append({"model": name, "split": split, "ppo_seed": "", **_mean_over_scenarios(d[split])})
    for fname in sorted(os.listdir(raw_dir)):
        if fname == "heuristics.json" or not fname.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, fname)) as f:
            d = json.load(f)
        for split in ("heldout", "in_distribution"):
            rows.append({"model": d["variant"], "split": split, "ppo_seed": d["ppo_seed"],
                         **_mean_over_scenarios(d[split])})

    csv_path = os.path.join(out, "per_seed_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "ppo_seed"] + METRIC_KEYS)
        w.writeheader()
        w.writerows(rows)

    lines: List[str] = []
    for split in ("heldout", "in_distribution"):
        lines.append(f"\n=== {split.upper()} scenarios (mean over scenarios; +/- std across training seeds) ===")
        lines.append(f"{'Model':22s} {'n':>2s} {'Carbon (g)':>16s} {'Cost':>12s} {'AvgDelay':>12s} {'SLA %':>10s} {'Reward':>16s}")
        order = HEURISTIC_NAMES + [v for v in VARIANTS]
        for model in order:
            sel = [r for r in rows if r["model"] == model and r["split"] == split]
            if not sel:
                continue

            def ms(key, scale=1.0, nd=0):
                vals = [r[key] * scale for r in sel]
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                return f"{statistics.mean(vals):.{nd}f}" + (f"+/-{sd:.{nd}f}" if len(vals) > 1 else "")

            lines.append(
                f"{model:22s} {len(sel):>2d} {ms('total_carbon_grams'):>16s} {ms('total_cost', nd=2):>12s} "
                f"{ms('avg_delay', nd=2):>12s} {ms('sla_violation_rate', 100, 1):>10s} {ms('total_reward', nd=3):>16s}"
            )
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out, "summary_table.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=5, help="training seeds per variant")
    ap.add_argument("--variants", nargs="+", default=["M3", "M3.5", "M2"], choices=list(VARIANTS))
    ap.add_argument("--train_pool", type=int, default=20, help="number of training scenario seeds (0..N-1)")
    ap.add_argument("--n_test", type=int, default=30, help="number of held-out scenario seeds (from 100)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/generalization")
    args = ap.parse_args()

    train_pool = list(range(args.train_pool))
    test_seeds = list(range(100, 100 + args.n_test))
    indist_seeds = train_pool[:10]
    assert not set(test_seeds) & set(train_pool), "test seeds must be disjoint from training seeds"

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    run_heuristics(args.out, test_seeds, indist_seeds)

    tasks = [
        dict(variant=v, ppo_seed=1000 + k, steps=args.steps, train_pool=train_pool,
             test_seeds=test_seeds, indist_seeds=indist_seeds, out=args.out)
        for k in range(args.seeds) for v in args.variants
    ]
    print(f"{len(tasks)} training runs x {args.steps} steps, {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)
    aggregate(args.out)


if __name__ == "__main__":
    main()
