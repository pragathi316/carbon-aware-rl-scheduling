"""
rl/evaluate_rl.py

Phase 8 of the roadmap: "Evaluate Carbon-Aware RL Before Adding the LLM."
Compares the four required models on the SAME scenario:

    M0 -> FIFO
    M1 -> Deadline + Carbon heuristic
    M2 -> RL without carbon-aware features (state hides carbon intensity)
    M3 -> RL + Carbon Awareness (state includes carbon intensity/forecast)

Reports total carbon, energy, cost, average delay, makespan, SLA
violation rate, and average reward for each — the exact metric set
Section 7 of the roadmap requires.

Two modes:
  1. Quick mode (default): evaluates the single already-trained
     ppo_carbon_blind.zip / ppo_carbon_aware.zip models. Fast, but is
     ONE run per model — not yet the "multiple seeds, report mean and
     variability" Section 8 asks for.
  2. --train_seeds N: trains N fresh PPO models per RL variant (M2, M3),
     each with a different training seed but the SAME evaluation
     scenario, then reports mean +/- std across seeds. This is slower
     (each seed = one full training run) but is the statistically
     honest version for anything you'd put in the paper.

Usage:
    python -m rl.evaluate_rl                    # quick mode
    python -m rl.evaluate_rl --train_seeds 3     # proper multi-seed mode
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from stable_baselines3 import PPO

from rl.config import RunConfig
from rl.env_factory import build_env
from rl.train_ppo import train
from baselines.fifo import pick_action as fifo_action
from baselines.deadline_carbon import pick_action as dc_action


@dataclass
class ModelResult:
    label: str
    metrics: List[Dict[str, Any]]  # one dict per evaluation run (usually 1, or N for RL)

    def mean_std(self, key: str) -> Dict[str, float]:
        values = [m[key] for m in self.metrics]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return {"mean": mean, "std": std}


def _run_heuristic(run_config: RunConfig, policy_fn: Callable) -> Dict[str, Any]:
    env = build_env(run_config.scenario)
    obs, info = env.reset(seed=run_config.scenario.seed)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = policy_fn(env.unwrapped)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    metrics = dict(info["episode_metrics"])
    metrics["total_reward"] = total_reward
    return metrics


def _run_ppo(
    run_config: RunConfig,
    model_path: str,
    carbon_aware_state: bool,
    use_llm_context: bool = False,
) -> Dict[str, Any]:
    scenario = run_config.scenario
    scenario.carbon_aware_state = carbon_aware_state
    scenario.use_llm_context = use_llm_context
    env = build_env(scenario)
    model = PPO.load(model_path)
    obs, info = env.reset(seed=scenario.seed)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
    metrics = dict(info["episode_metrics"])
    metrics["total_reward"] = total_reward
    return metrics


def _train_and_evaluate_seeds(
    base_run_config: RunConfig,
    carbon_aware_state: bool,
    n_seeds: int,
    label: str,
    use_llm_context: bool = False,
) -> ModelResult:
    """Train n_seeds fresh models (different PPO seed each, same scenario) and
    evaluate each deterministically on that same scenario."""
    results = []
    for i in range(n_seeds):
        ppo_seed = 1000 + i  # distinct from the scenario's own seed
        run_config = copy.deepcopy(base_run_config)
        run_config.scenario.carbon_aware_state = carbon_aware_state
        run_config.scenario.use_llm_context = use_llm_context
        run_config.ppo.seed = ppo_seed
        tag = "llm" if use_llm_context else ("aware" if carbon_aware_state else "blind")
        run_config.ppo.model_save_path = f"results/trained_models/ppo_{tag}_seed{ppo_seed}"

        print(f"\n--- Training {label}, seed {i + 1}/{n_seeds} (ppo_seed={ppo_seed}) ---")
        train(run_config)

        eval_config = copy.deepcopy(base_run_config)
        metrics = _run_ppo(
            eval_config, f"{run_config.ppo.model_save_path}.zip",
            carbon_aware_state, use_llm_context,
        )
        results.append(metrics)

    return ModelResult(label=label, metrics=results)


def evaluate_all(run_config: RunConfig, train_seeds: Optional[int] = None) -> List[ModelResult]:
    results: List[ModelResult] = []

    print("Evaluating M0: FIFO...")
    results.append(ModelResult("M0: FIFO", [_run_heuristic(run_config, fifo_action)]))

    print("Evaluating M1: Deadline+Carbon...")
    results.append(ModelResult("M1: Deadline+Carbon", [_run_heuristic(run_config, dc_action)]))

    if train_seeds:
        results.append(
            _train_and_evaluate_seeds(run_config, False, train_seeds, "M2: RL (carbon-blind)")
        )
        results.append(
            _train_and_evaluate_seeds(run_config, True, train_seeds, "M3: RL (carbon-aware)")
        )
        results.append(
            _train_and_evaluate_seeds(
                run_config, True, train_seeds, "M3.5: RL+Carbon+LLM", use_llm_context=True
            )
        )
    else:
        print("Evaluating M2: RL (carbon-blind), using existing saved model...")
        blind_path = "results/trained_models/ppo_carbon_blind.zip"
        if os.path.exists(blind_path):
            results.append(
                ModelResult("M2: RL (carbon-blind)", [_run_ppo(run_config, blind_path, False)])
            )
        else:
            print(f"  Skipped — no model found at {blind_path}. "
                  f"Train it first with: python -m rl.train_ppo --carbon_blind")

        print("Evaluating M3: RL (carbon-aware), using existing saved model...")
        aware_path = "results/trained_models/ppo_carbon_aware.zip"
        if os.path.exists(aware_path):
            results.append(
                ModelResult("M3: RL (carbon-aware)", [_run_ppo(run_config, aware_path, True)])
            )
        else:
            print(f"  Skipped — no model found at {aware_path}. "
                  f"Train it first with: python -m rl.train_ppo")

        print("Evaluating M3.5: RL + Carbon + LLM Context, using existing saved model...")
        llm_path = "results/trained_models/ppo_carbon_llm.zip"
        if os.path.exists(llm_path):
            results.append(
                ModelResult(
                    "M3.5: RL+Carbon+LLM",
                    [_run_ppo(run_config, llm_path, True, use_llm_context=True)],
                )
            )
        else:
            print(f"  Skipped — no model found at {llm_path}. "
                  f"Train it first with: python -m rl.train_ppo --llm_context")

    return results


def print_comparison_table(results: List[ModelResult]) -> None:
    header = (
        f"{'Model':24s} {'Carbon (g)':>14s} {'Cost':>8s} {'AvgDelay':>10s} "
        f"{'Makespan':>10s} {'SLA%':>7s} {'Reward':>10s}"
    )
    print("\n" + header)
    print("-" * len(header))

    for result in results:
        n = len(result.metrics)
        carbon = result.mean_std("total_carbon_grams")
        cost = result.mean_std("total_cost")
        delay = result.mean_std("avg_delay")
        sla = result.mean_std("sla_violation_rate")
        reward = result.mean_std("total_reward")
        makespan_vals = [m.get("makespan", 0.0) for m in result.metrics]
        makespan_mean = statistics.mean(makespan_vals) if makespan_vals else 0.0

        carbon_str = f"{carbon['mean']:.0f}" + (f"+/-{carbon['std']:.0f}" if n > 1 else "")
        cost_str = f"{cost['mean']:.2f}" + (f"+/-{cost['std']:.2f}" if n > 1 else "")
        delay_str = f"{delay['mean']:.2f}" + (f"+/-{delay['std']:.2f}" if n > 1 else "")
        sla_str = f"{sla['mean']:.0%}"
        reward_str = f"{reward['mean']:.3f}" + (f"+/-{reward['std']:.3f}" if n > 1 else "")

        print(
            f"{result.label:24s} {carbon_str:>14s} {cost_str:>8s} {delay_str:>10s} "
            f"{makespan_mean:>10.1f} {sla_str:>7s} {reward_str:>10s}"
        )


def save_comparison_csv(results: List[ModelResult], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "run_index", "total_carbon_grams", "total_cost",
                          "avg_delay", "makespan", "sla_violation_rate", "total_reward"])
        for result in results:
            for i, m in enumerate(result.metrics):
                writer.writerow([
                    result.label, i, m["total_carbon_grams"], m["total_cost"],
                    m["avg_delay"], m.get("makespan", 0.0), m["sla_violation_rate"],
                    m["total_reward"],
                ])
    print(f"\nSaved per-run results to {filepath}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 8: compare FIFO, Deadline+Carbon, RL-blind, RL-carbon-aware."
    )
    parser.add_argument(
        "--train_seeds", type=int, default=None,
        help="If set, trains this many fresh seeds per RL variant instead of using "
        "existing saved models, and reports mean +/- std (Section 8's requirement). "
        "Each seed is a full training run, so this is slow.",
    )
    parser.add_argument(
        "--output", type=str, default="results/metrics/phase8_comparison.csv",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_config = RunConfig()  # same default scenario used throughout

    results = evaluate_all(run_config, train_seeds=args.train_seeds)
    print_comparison_table(results)
    save_comparison_csv(results, args.output)


if __name__ == "__main__":
    main()
