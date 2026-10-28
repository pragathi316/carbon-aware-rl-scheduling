"""
baselines/deadline_carbon.py

Deadline + Carbon baseline: the strongest non-RL heuristic (Phase 6 of
the roadmap). Decision rule:

  1. Find the currently-feasible region with the lowest carbon intensity.
  2. If the job's remaining slack is too tight to safely wait one more
     step, run now in that region — protecting the SLA always wins.
  3. If there's slack to spare, only delay if the carbon forecast says a
     cleaner period is coming (using the carbon provider's t+1
     prediction). Otherwise, running now in the cleanest feasible region
     is already as good as waiting, so just run.

This is deliberately the only baseline that uses BOTH the deadline and
the carbon forecast — it's meant to be a strong reference that RL has to
beat, not a strawman.
"""

from __future__ import annotations

from environment.cloud_env import CloudSchedulingEnv

# Fraction of the job's own execution_time used as a safety buffer before
# its deadline. Below this much slack, we stop gambling on cleaner power
# and just run the job now.
SAFETY_MARGIN_FRACTION = 0.1

# Minimum relative drop in forecast carbon intensity required before we'll
# delay to chase it. Without this, noisy forecasts make the policy delay
# on essentially a coin flip — a delay costs a small shaping penalty and a
# real timestep, so it must buy a real (not noise-sized) improvement.
MIN_FORECAST_IMPROVEMENT_FRACTION = 0.10


def pick_action(env: CloudSchedulingEnv) -> int:
    job = env.current_job
    assert job is not None

    feasible = [
        idx for idx, region in enumerate(env.regions)
        if region.can_fit(job.cpu_requirement, job.gpu_requirement)
    ]
    if not feasible:
        return 3  # nothing fits right now — forced wait

    best_idx = min(feasible, key=lambda idx: env.regions[idx].carbon_intensity)
    best_region = env.regions[best_idx]

    remaining_slack = job.deadline - (env.current_time + job.execution_time)
    safety_margin = job.execution_time * SAFETY_MARGIN_FRACTION
    can_afford_to_wait = remaining_slack > safety_margin

    if not can_afford_to_wait:
        return best_idx  # deadline is close — run now, no gambling

    if env.carbon_provider is not None:
        try:
            forecast_next = env.carbon_provider.get_predicted(
                best_region.region_id, env.current_time, 1
            )
        except (KeyError, ValueError):
            forecast_next = best_region.carbon_intensity

        improvement = (best_region.carbon_intensity - forecast_next) / best_region.carbon_intensity
        if improvement > MIN_FORECAST_IMPROVEMENT_FRACTION:
            return 3  # cleaner power is forecast soon and the deadline allows waiting

    return best_idx


if __name__ == "__main__":
    from baselines.common import run_episode, print_summary, build_demo_env

    env = build_demo_env(n_jobs=30, seed=1)
    summary = run_episode(env, pick_action, seed=1)
    print_summary("Deadline + Carbon baseline", summary)
