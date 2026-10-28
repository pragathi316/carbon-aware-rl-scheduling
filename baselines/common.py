"""
baselines/common.py

Not one of the roadmap's named files, but a small shared helper so
fifo.py, least_carbon.py, and deadline_carbon.py don't each duplicate
the same episode loop and metric calculations. Each baseline still only
needs to define its own `pick_action(env) -> int` policy function.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from environment.cloud_env import CloudSchedulingEnv

PolicyFn = Callable[[CloudSchedulingEnv], int]


def run_episode(
    env: CloudSchedulingEnv,
    policy_fn: PolicyFn,
    seed: Optional[int] = None,
    render: bool = False,
) -> Dict[str, Any]:
    """Run one full episode with the given policy and return summary metrics.

    These are exactly the metrics the roadmap's Phase 8 comparison table
    needs: total carbon, energy, cost, average delay, makespan, SLA
    violation rate, and average reward.
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            env.render()

    completed = env.completed_jobs
    n_completed = len(completed)
    n_jobs = len(env._initial_jobs)
    sla_violations = sum(1 for j in completed if j["sla_violated"])

    return {
        "n_jobs": n_jobs,
        "n_completed": n_completed,
        "completion_rate": n_completed / n_jobs if n_jobs else 0.0,
        "total_reward": total_reward,
        "avg_reward": total_reward / n_completed if n_completed else 0.0,
        "total_carbon_grams": sum(j["carbon_grams"] for j in completed),
        "total_energy_kwh": sum(j["energy_kwh"] for j in completed),
        "total_cost": sum(j["cost"] for j in completed),
        "avg_delay": sum(j["delay"] for j in completed) / n_completed if n_completed else 0.0,
        "makespan": max((j["finish_time"] for j in completed), default=0.0),
        "sla_violations": sla_violations,
        "sla_violation_rate": sla_violations / n_completed if n_completed else 0.0,
        "terminated": terminated,
        "truncated": truncated,
    }


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(f"  Jobs completed:     {summary['n_completed']}/{summary['n_jobs']} "
          f"({summary['completion_rate']:.0%})")
    print(f"  Total carbon:       {summary['total_carbon_grams']:.0f} gCO2")
    print(f"  Total energy:       {summary['total_energy_kwh']:.2f} kWh")
    print(f"  Total cost:         {summary['total_cost']:.2f}")
    print(f"  Avg delay:          {summary['avg_delay']:.2f}")
    print(f"  Makespan:           {summary['makespan']:.1f}")
    print(f"  SLA violations:     {summary['sla_violations']} "
          f"({summary['sla_violation_rate']:.0%})")
    print(f"  Total reward:       {summary['total_reward']:.3f}")


def build_demo_env(n_jobs: int = 30, seed: int = 1) -> CloudSchedulingEnv:
    """Build a small demo environment, shared across baseline __main__ blocks
    so FIFO / Least Carbon / Deadline+Carbon can be run on identical inputs
    and compared directly."""
    from environment.workload_generator import generate_jobs
    from environment.region import create_default_regions
    from environment.carbon_data import generate_carbon_series, CarbonDataProvider

    jobs = generate_jobs(n_jobs=n_jobs, intensity="medium", seed=seed)
    regions = create_default_regions(3)
    carbon_rows = generate_carbon_series(
        region_ids=[r.region_id for r in regions],
        n_timesteps=300,
        horizon=2,
        volatility="medium",
        seed=seed,
    )
    carbon_provider = CarbonDataProvider(carbon_rows)
    return CloudSchedulingEnv(jobs=jobs, regions=regions, carbon_provider=carbon_provider)
