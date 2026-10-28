"""
baselines/wait_clean.py

Wait-for-Clean heuristic (stronger temporal baseline added in the revision).

If the currently cleanest region cannot fit the job and the job's deadline
still allows waiting one more step (remaining slack after waiting > 10 % of
its execution time), delay; otherwise run in the lowest-carbon feasible
region. Unlike M1 it waits for CAPACITY in the clean region, not for a
forecast drop.
"""

from __future__ import annotations

SAFETY_MARGIN_FRACTION = 0.1


def pick_action(env) -> int:
    job = env.current_job
    feasible = [i for i, r in enumerate(env.regions) if r.can_fit(job.cpu_requirement, job.gpu_requirement)]
    cleanest = min(range(len(env.regions)), key=lambda i: env.regions[i].carbon_intensity)
    slack_after_wait = job.deadline - (env.current_time + 1.0 + job.execution_time)
    if cleanest not in feasible and slack_after_wait > SAFETY_MARGIN_FRACTION * job.execution_time:
        return 3
    if not feasible:
        return 3
    return min(feasible, key=lambda i: env.regions[i].carbon_intensity)
