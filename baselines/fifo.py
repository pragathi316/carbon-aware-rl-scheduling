"""
baselines/fifo.py

FIFO baseline: run each job as soon as any region has capacity, in
arrival order. No carbon or cost awareness at all — this is the
traditional-scheduling reference point every other model is compared
against (Phase 6 of the roadmap).

The environment already presents jobs in arrival order (its queue is
FIFO), so this policy just needs to pick the first feasible region and
never deliberately delay.
"""

from __future__ import annotations

from environment.cloud_env import CloudSchedulingEnv


def pick_action(env: CloudSchedulingEnv) -> int:
    """Run in the first region (in order) that currently has capacity.

    Only returns DELAY (3) if no region can fit the job right now — that's
    a forced wait due to resource contention, not a deliberate choice to
    hold out for cleaner power (this policy never does that).
    """
    job = env.current_job
    assert job is not None

    for idx, region in enumerate(env.regions):
        if region.can_fit(job.cpu_requirement, job.gpu_requirement):
            return idx
    return 3  # nothing fits right now — forced wait


if __name__ == "__main__":
    from baselines.common import run_episode, print_summary, build_demo_env

    env = build_demo_env(n_jobs=30, seed=1)
    summary = run_episode(env, pick_action, seed=1)
    print_summary("FIFO baseline", summary)
