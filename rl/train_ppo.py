"""
rl/train_ppo.py

Trains a PPO agent on the carbon-aware cloud scheduling environment
(Phase 7 of the roadmap). Logs episode reward AND the domain metrics
(carbon, cost, delay, SLA violations) that actually matter for the
paper — not just the scalar reward SB3 tracks by default — and saves
periodic checkpoints plus the exact config used, for reproducibility.

NOTE on training budget: none was specified, so PPOConfig defaults to a
small ~50k-timestep run (a few minutes on CPU) meant to validate that
the pipeline learns at all before committing to a longer run for real
experimental results. Bump `PPOConfig.total_timesteps` (or pass
--total_timesteps) once you're ready to run something you'd report
numbers from.

Usage:
    python -m rl.train_ppo
    python -m rl.train_ppo --total_timesteps 200000 --n_jobs 100 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from rl.config import RunConfig
from rl.env_factory import build_env


class EpisodeMetricsLoggingCallback(BaseCallback):
    """Pulls `episode_metrics` out of `info` at episode end and logs them
    to SB3's logger (console + tensorboard, if configured)."""

    def __init__(self, print_every: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if metrics is None:
                continue
            self.episode_count += 1
            for key, value in metrics.items():
                self.logger.record(f"episode/{key}", value)

            if self.episode_count % self.print_every == 0:
                print(
                    f"[episode {self.episode_count}] "
                    f"completed={metrics['n_completed']}/{metrics['n_jobs']} "
                    f"carbon={metrics['total_carbon_grams']:.0f}g "
                    f"cost={metrics['total_cost']:.2f} "
                    f"avg_delay={metrics['avg_delay']:.2f} "
                    f"sla_violations={metrics['sla_violation_rate']:.0%}"
                )
        return True


def train(run_config: RunConfig) -> PPO:
    save_dir = os.path.dirname(run_config.ppo.model_save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(run_config.ppo.log_dir, exist_ok=True)

    vec_env = DummyVecEnv([lambda: build_env(run_config.scenario)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=run_config.ppo.n_steps,
        batch_size=run_config.ppo.batch_size,
        n_epochs=run_config.ppo.n_epochs,
        learning_rate=run_config.ppo.learning_rate,
        gamma=run_config.ppo.gamma,
        gae_lambda=run_config.ppo.gae_lambda,
        clip_range=run_config.ppo.clip_range,
        ent_coef=run_config.ppo.ent_coef,
        seed=run_config.ppo.seed,
        tensorboard_log=run_config.ppo.log_dir,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=run_config.ppo.checkpoint_freq,
        save_path=save_dir,
        name_prefix="ppo_checkpoint",
    )
    metrics_callback = EpisodeMetricsLoggingCallback(print_every=5)

    # Save the exact config used, next to the model — required for
    # reproducibility (Section 8: "record the exact weights/settings").
    config_path = f"{run_config.ppo.model_save_path}_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config.to_dict(), f, indent=2)
    print(f"Saved run config to {config_path}")

    model.learn(
        total_timesteps=run_config.ppo.total_timesteps,
        callback=[checkpoint_callback, metrics_callback],
    )

    model.save(run_config.ppo.model_save_path)
    print(f"\nSaved final model to {run_config.ppo.model_save_path}.zip")
    return model


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the carbon-aware cloud scheduler.")
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--intensity", type=str, default=None, choices=["light", "medium", "heavy"])
    parser.add_argument("--volatility", type=str, default=None, choices=["stable", "medium", "volatile"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--carbon_blind",
        action="store_true",
        help="Train the M2 ablation: hide carbon intensity/forecast from the agent's state "
        "(the environment's physics and reward still account for real carbon emissions).",
    )
    parser.add_argument(
        "--llm_context",
        action="store_true",
        help="Train M3.5: append the LLM/context-engine's structured features "
        "(carbon_trend, deadline_flexibility, delay_opportunity, carbon_risk) to the "
        "state, on top of carbon-awareness. Uses the rule-based engine by default "
        "(fast enough for training); pass --llm_context_backend anthropic for the "
        "real LLM engine (slow — evaluation-scale only).",
    )
    parser.add_argument(
        "--llm_context_backend", type=str, default="rule_based",
        choices=["rule_based", "anthropic"],
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_config = RunConfig()

    if args.total_timesteps is not None:
        run_config.ppo.total_timesteps = args.total_timesteps
    if args.n_jobs is not None:
        run_config.scenario.n_jobs = args.n_jobs
    if args.intensity is not None:
        run_config.scenario.workload_intensity = args.intensity
    if args.volatility is not None:
        run_config.scenario.carbon_volatility = args.volatility
    if args.seed is not None:
        run_config.scenario.seed = args.seed
        run_config.ppo.seed = args.seed

    if args.carbon_blind:
        run_config.scenario.carbon_aware_state = False
        run_config.ppo.model_save_path = "results/trained_models/ppo_carbon_blind"
    elif args.llm_context:
        run_config.scenario.use_llm_context = True
        run_config.scenario.llm_context_backend = args.llm_context_backend
        run_config.ppo.model_save_path = "results/trained_models/ppo_carbon_llm"
    else:
        run_config.ppo.model_save_path = "results/trained_models/ppo_carbon_aware"

    train(run_config)


if __name__ == "__main__":
    main()
