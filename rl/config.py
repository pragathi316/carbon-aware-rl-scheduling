"""
rl/config.py

Centralized configuration for RL training runs. Keeping this in one
dataclass (rather than scattered argparse defaults) makes it trivial to
log the exact config alongside results — the roadmap repeatedly stresses
recording exact settings for reproducibility (Phase 5's reward weights,
Section 8's "multiple seeds... report mean and variability").
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any


@dataclass
class ScenarioConfig:
    """Defines the workload/region/carbon scenario the agent trains on."""

    n_jobs: int = 60
    workload_intensity: str = "medium"      # "light" | "medium" | "heavy"
    carbon_volatility: str = "medium"       # "stable" | "medium" | "volatile"
    carbon_horizon: int = 3                 # how many forecast steps (t+1..t+n)
    n_carbon_timesteps: int = 400           # must comfortably exceed max job deadline
    n_regions: int = 3
    seed: int = 42
    carbon_aware_state: bool = True  # False = M2 ablation (RL blind to carbon in its state)
    use_llm_context: bool = False    # True = M3.5 (RL + Carbon + LLM/context-engine features)
    llm_context_backend: str = "rule_based"  # "rule_based" | "anthropic" | "random" (ablation control) — only used if use_llm_context
    carbon_source: str = "synthetic"  # "synthetic" | "real_eu" | "real_mixed" (environment/real_carbon.py)
    count_invalid_as_delay: bool = True  # False = pre-fix behaviour (invalid placements were uncapped waits)
    include_gpu_state: bool = False   # True = add job GPU demand + free GPUs per region to the state
    scale_obs: bool = False           # True = divide state features by fixed scales (inputs roughly in [0, 1])
    energy_model: str = "simple"          # "simple" | "calibrated" (see ENERGY_MODELS in rl/env_factory.py)
    region_profile: str = "default"       # "default" | "gpu_cluster" (environment/region.py)
    workload_source: str = "synthetic"   # "synthetic" | "pai_trace" (environment/trace_workload.py)
    trace_interarrival: float = 2.0       # pai_trace only: mean inter-arrival time in steps
    trace_slack: tuple = (1.5, 4.0)       # pai_trace only: deadline slack factor range
    time_features: bool = False       # True = append hour-of-day (sin, cos) of the carbon trace to the state
    switch_prob: float = 0.0          # carbon_source="switch" only: per-hour probability that the cleanest region changes
    real_split: str = "auto"          # real data only: "train" | "test" | "auto" (seed >= 100 -> test window)


@dataclass
class PPOConfig:
    """PPO hyperparameters (stable-baselines3 defaults, tuned down for a
    prototype-scale run). NOTE: no training budget was specified, so this
    defaults to a small run (~50k timesteps, a few minutes on CPU) meant to
    validate that the pipeline learns at all — scale total_timesteps up
    for real experimental results.
    """

    total_timesteps: int = 50_000
    n_steps: int = 512            # rollout length per env before each update
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01        # small entropy bonus — helps avoid collapsing to "always run in region X"
    seed: int = 42
    net_arch: tuple = (64, 64)    # hidden layers of the actor and critic MLPs (SB3 default)

    checkpoint_freq: int = 10_000
    model_save_path: str = "results/trained_models/ppo_carbon_scheduler"
    log_dir: str = "results/metrics/ppo_tensorboard"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    """Bundles scenario + PPO config so a full run's settings are one object."""

    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"scenario": asdict(self.scenario), "ppo": self.ppo.to_dict()}


if __name__ == "__main__":
    import json

    config = RunConfig()
    print(json.dumps(config.to_dict(), indent=2))
