"""
rl/env_factory.py

Builds a CloudSchedulingEnv from a ScenarioConfig, and wraps it so
episode-level metrics (carbon, cost, delay, SLA violations) — not just
reward — are visible to SB3's callbacks and logs. Shared by
train_ppo.py, train_dqn.py, and evaluate_rl.py so they all train/evaluate
on identically-specified scenarios.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from environment.cloud_env import CloudSchedulingEnv
from environment.region import create_default_regions
from environment.workload_generator import generate_jobs
from environment.carbon_data import generate_carbon_series, CarbonDataProvider
from llm_layer.context_builder import (
    RuleBasedContextEngine, AnthropicLLMContextEngine, RandomContextEngine,
)
from environment.real_carbon import generate_real_carbon_series
from environment.reward import RewardWeights
from rl.config import ScenarioConfig

TEST_SEED_START = 100

# "calibrated": 0.25 kW per GPU (data-center GPUs of the V100/P100 class draw
# 250-300 W at full load), 50 % idle-power overhead on CPU energy (servers draw
# a large fraction of peak power when idle; Barroso et al., The Datacenter as
# a Computer, 2018), and a facility PUE of 1.58 (Uptime Institute 2023 global
# average). The simple model is the default used for training.
ENERGY_MODELS = {
    "simple": {},
    "calibrated": dict(gpu_kw=0.25, cpu_idle_overhead=0.5, pue=1.58),
}  # real data, split='auto': scenario seeds >= this use the held-out (Sep-Dec) window


FEATURE_SCALES = [
    ("job_cpu_requirement", 10.0), ("job_execution_time", 20.0), ("remaining_deadline", 50.0),
    ("queue_length", 10.0), ("available_cpu", 100.0), ("carbon", 1000.0), ("context_", 1.0),
    ("cheapest_electricity_cost", 0.2), ("mean_region_utilization", 1.0),
    ("job_gpu_requirement", 4.0), ("available_gpu", 8.0),
]


class ScaledObservation(gym.ObservationWrapper):
    """Divides every state feature by a fixed, hand-chosen scale so that the
    policy network sees inputs of order 1 instead of raw values that range
    from 0-1 (utilization) to ~600 (carbon intensity). Fixed scales (rather
    than running statistics) keep saved models self-contained."""

    def __init__(self, env):
        super().__init__(env)
        names = env.unwrapped.get_feature_names()
        scales = []
        for n in names:
            scale = next((v for key, v in FEATURE_SCALES if key in n), 1.0)
            scales.append(scale)
        self._scales = np.array(scales, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf, shape=env.observation_space.shape,
                                                dtype=np.float32)

    def observation(self, obs):
        return (obs / self._scales).astype(np.float32)


class HourOfDayObservation(gym.ObservationWrapper):
    """Appends sin/cos of the hour of day of the carbon trace (UTC) at the
    current simulation time, so that a carbon-blind agent can still learn
    daily patterns such as California's solar cycle."""

    def __init__(self, env, start_hour: int):
        super().__init__(env)
        self.start_hour = int(start_hour)
        lo = np.concatenate([env.observation_space.low, [-1.0, -1.0]]).astype(np.float32)
        hi = np.concatenate([env.observation_space.high, [1.0, 1.0]]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=lo, high=hi, dtype=np.float32)

    def observation(self, obs):
        h = (self.start_hour + int(np.floor(self.env.unwrapped.current_time))) % 24
        ang = 2.0 * np.pi * h / 24.0
        return np.concatenate([obs, [np.sin(ang), np.cos(ang)]]).astype(np.float32)


class EpisodeMetricsWrapper(gym.Wrapper):
    """Adds an `episode_metrics` dict to `info` on the terminal step of each
    episode, mirroring what SB3's own Monitor wrapper does for reward/length
    but for the domain metrics this project actually cares about.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            completed = self.env.unwrapped.completed_jobs
            n_completed = len(completed)
            n_jobs = len(self.env.unwrapped._initial_jobs)
            sla_violations = sum(1 for j in completed if j["sla_violated"])

            info["episode_metrics"] = {
                "n_jobs": n_jobs,
                "n_completed": n_completed,
                "completion_rate": n_completed / n_jobs if n_jobs else 0.0,
                "total_carbon_grams": sum(j["carbon_grams"] for j in completed),
                "total_energy_kwh": sum(j["energy_kwh"] for j in completed),
                "total_cost": sum(j["cost"] for j in completed),
                "makespan": max((j["finish_time"] for j in completed), default=0.0),
                "avg_delay": (
                    sum(j["delay"] for j in completed) / n_completed if n_completed else 0.0
                ),
                "sla_violations": sla_violations,
                "sla_violation_rate": sla_violations / n_completed if n_completed else 0.0,
            }
        return obs, reward, terminated, truncated, info


def build_env(
    scenario: ScenarioConfig,
    seed: Optional[int] = None,
    reward_weights: Optional[RewardWeights] = None,
) -> CloudSchedulingEnv:
    """Build one (wrapped) CloudSchedulingEnv instance from a ScenarioConfig.

    Args:
        scenario: Workload/region/carbon settings.
        seed: Overrides scenario.seed if given — useful for generating
            distinct-but-comparable envs across multiple training seeds
            (Section 8's "multiple random seeds and repeated runs").
    """
    effective_seed = seed if seed is not None else scenario.seed

    if scenario.workload_source == "pai_trace":
        from environment.trace_workload import generate_trace_jobs
        jobs = generate_trace_jobs(
            n_jobs=scenario.n_jobs, seed=effective_seed,
            split="test" if effective_seed >= TEST_SEED_START else "train",
            mean_interarrival=scenario.trace_interarrival, slack_range=tuple(scenario.trace_slack),
        )
    else:
        jobs = generate_jobs(
            n_jobs=scenario.n_jobs,
            intensity=scenario.workload_intensity,
            seed=effective_seed,
        )
    if scenario.region_profile == "gpu_cluster":
        from environment.region import create_gpu_cluster_regions
        regions = create_gpu_cluster_regions()
    else:
        regions = create_default_regions(scenario.n_regions)
    start_hour = 0
    if scenario.carbon_source == "switch":
        from environment.switch_carbon import generate_switch_series
        carbon_rows = generate_switch_series(
            region_ids=[r.region_id for r in regions], n_timesteps=scenario.n_carbon_timesteps,
            horizon=scenario.carbon_horizon, switch_prob=scenario.switch_prob, seed=effective_seed,
        )
    elif scenario.carbon_source == "synthetic":
        carbon_rows = generate_carbon_series(
            region_ids=[r.region_id for r in regions],
            n_timesteps=scenario.n_carbon_timesteps,
            horizon=scenario.carbon_horizon,
            volatility=scenario.carbon_volatility,
            seed=effective_seed,
        )
    else:
        split = scenario.real_split
        if split == "auto":
            split = "test" if effective_seed >= TEST_SEED_START else "train"
        from environment.real_carbon import load_hourly_traces, window_start
        n_total = len(next(iter(load_hourly_traces().values())))
        start_hour = window_start(effective_seed or 0, n_total, scenario.n_carbon_timesteps + scenario.carbon_horizon, split)
        carbon_rows = generate_real_carbon_series(
            region_ids=[r.region_id for r in regions],
            n_timesteps=scenario.n_carbon_timesteps,
            horizon=scenario.carbon_horizon,
            mapping=scenario.carbon_source,
            split=split,
            seed=effective_seed,
        )
    carbon_provider = CarbonDataProvider(carbon_rows)

    context_engine = None
    if scenario.use_llm_context:
        if scenario.llm_context_backend == "anthropic":
            context_engine = AnthropicLLMContextEngine()  # falls back to rule-based on any failure
        elif scenario.llm_context_backend == "random":
            context_engine = RandomContextEngine(seed=effective_seed)  # ablation control
        else:
            context_engine = RuleBasedContextEngine()

    env = CloudSchedulingEnv(
        jobs=jobs,
        regions=regions,
        carbon_provider=carbon_provider,
        carbon_aware_state=scenario.carbon_aware_state,
        context_engine=context_engine,
        count_invalid_as_delay=scenario.count_invalid_as_delay,
        include_gpu_state=scenario.include_gpu_state,
        energy_params=ENERGY_MODELS[scenario.energy_model],
        **({"reward_weights": reward_weights} if reward_weights is not None else {}),
    )
    if scenario.scale_obs:
        env = ScaledObservation(env)
    if scenario.time_features:
        env = HourOfDayObservation(env, start_hour)
    return EpisodeMetricsWrapper(env)
