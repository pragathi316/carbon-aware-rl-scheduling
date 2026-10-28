"""
baselines/least_carbon.py

Least Carbon baseline: among regions that can currently fit the job,
always choose the one with the lowest carbon intensity right now. Never
delays deliberately, and ignores deadlines entirely — this isolates the
pure carbon-only trade-off (Phase 6 of the roadmap) with no SLA
awareness at all, which is exactly what makes it a useful reference:
comparing it against Deadline+Carbon shows what SLA-awareness costs or
saves.
"""

from __future__ import annotations

from environment.cloud_env import CloudSchedulingEnv


def pick_action(env: CloudSchedulingEnv) -> int:
    """Choose the lowest-carbon region that can fit the job right now.

    Returns DELAY (3) only if no region has capacity at all — this
    policy does not use delay as a carbon-avoidance strategy, only FIFO
    and this baseline share that "forced wait" fallback; Deadline+Carbon
    is the one that delays on purpose.
    """
    job = env.current_job
    assert job is not None

    feasible = [
        idx for idx, region in enumerate(env.regions)
        if region.can_fit(job.cpu_requirement, job.gpu_requirement)
    ]
    if not feasible:
        return 3  # nothing fits right now — forced wait

    return min(feasible, key=lambda idx: env.regions[idx].carbon_intensity)


if __name__ == "__main__":
    from baselines.common import run_episode, print_summary, build_demo_env

    env = build_demo_env(n_jobs=30, seed=1)
    summary = run_episode(env, pick_action, seed=1)
    print_summary("Least Carbon baseline", summary)
