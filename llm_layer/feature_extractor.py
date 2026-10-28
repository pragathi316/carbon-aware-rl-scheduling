"""
llm_layer/feature_extractor.py

Pulls the raw numeric facts about the CURRENT decision out of the
environment — exactly the roadmap's example input:

    Current carbon = high
    Predicted carbon in 30 minutes = low
    Remaining deadline = large
    Queue = low
    Resources = sufficient

This is deliberately just extraction, no interpretation — turning those
numbers into judgments ("high"/"low"/flexibility scores) is
context_builder.py's job (Phase 9's second file).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.cloud_env import CloudSchedulingEnv


@dataclass
class RawContext:
    """Everything the context layer needs about the current decision,
    already pulled out of the environment — no environment object is
    needed downstream of this point."""

    job_id: int
    current_time: float
    cpu_requirement: float
    execution_time: float
    remaining_deadline: float
    queue_length: int
    best_region_id: str
    current_carbon: float
    predicted_carbon: List[float]      # [t+1, t+2, ..., t+horizon] for best_region_id
    resource_sufficiency: float        # fraction of regions that can fit the job right now


def extract_raw_context(env: "CloudSchedulingEnv") -> RawContext:
    """Extract a RawContext from the environment's CURRENT decision point.

    Args:
        env: A CloudSchedulingEnv (unwrapped) with env.current_job set —
            i.e. call this between reset()/step() calls, same timing as
            when you'd call env's own _get_state().
    """
    job = env.current_job
    if job is None:
        raise ValueError("env.current_job is None — call this between reset()/step() calls")

    feasible = [
        (idx, region) for idx, region in enumerate(env.regions)
        if region.can_fit(job.cpu_requirement, job.gpu_requirement)
    ]
    # If nothing is feasible right now, still report the globally cleanest
    # region so the context layer has something concrete to reason about —
    # the RL agent will see the infeasibility itself via region_cpu in its
    # own state; this layer's job is carbon/deadline reasoning, not capacity.
    candidates = feasible if feasible else list(enumerate(env.regions))
    best_idx, best_region = min(candidates, key=lambda pair: pair[1].carbon_intensity)

    predicted_carbon: List[float] = []
    if env.carbon_provider is not None:
        try:
            predicted_carbon = env.carbon_provider.get_forecast_vector(
                best_region.region_id, env.current_time
            )[1:]
        except KeyError:
            predicted_carbon = []

    resource_sufficiency = len(feasible) / len(env.regions) if env.regions else 0.0

    return RawContext(
        job_id=job.job_id,
        current_time=env.current_time,
        cpu_requirement=job.cpu_requirement,
        execution_time=job.execution_time,
        remaining_deadline=max(0.0, job.deadline - env.current_time),
        queue_length=len(env.queue),
        best_region_id=best_region.region_id,
        current_carbon=best_region.carbon_intensity,
        predicted_carbon=predicted_carbon,
        resource_sufficiency=resource_sufficiency,
    )


if __name__ == "__main__":
    from environment.cloud_env import CloudSchedulingEnv
    from environment.region import create_default_regions
    from environment.workload_generator import generate_jobs
    from environment.carbon_data import generate_carbon_series, CarbonDataProvider

    jobs = generate_jobs(n_jobs=10, intensity="medium", seed=1)
    regions = create_default_regions(3)
    rows = generate_carbon_series(
        [r.region_id for r in regions], n_timesteps=200, horizon=3, seed=1
    )
    provider = CarbonDataProvider(rows)
    env = CloudSchedulingEnv(jobs=jobs, regions=regions, carbon_provider=provider)
    env.reset(seed=1)

    raw = extract_raw_context(env)
    print(raw)
