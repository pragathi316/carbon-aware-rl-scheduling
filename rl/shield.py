"""
rl/shield.py

SLA shield: a run-time safeguard that overrides an RL policy's action when it
would put a deadline at risk, falling back to the Least Carbon heuristic.

Override rules (checked at every decision):
  1. The policy chooses DELAY, but after waiting one more step the job's
     remaining slack would be at most 10 % of its execution time (the same
     safety margin the M1 heuristic uses)  -> run now in the lowest-carbon
     feasible region.
  2. The policy chooses a region that cannot fit the job (an invalid action
     that only wastes a step) -> lowest-carbon feasible region.

The shield never changes a valid, deadline-safe action, so whatever carbon
benefit the policy finds is kept; it only removes the risky tail.
"""

from __future__ import annotations

from typing import Tuple

from baselines.least_carbon import pick_action as least_carbon_action

DELAY = 3
SAFETY_MARGIN_FRACTION = 0.1


def shield_action(env, action: int) -> Tuple[int, bool]:
    """Return (possibly overridden action, whether an override happened)."""
    job = env.current_job
    if job is None:
        return action, False
    if action == DELAY:
        slack_after_wait = job.deadline - (env.current_time + 1.0 + job.execution_time)
        if slack_after_wait <= SAFETY_MARGIN_FRACTION * job.execution_time:
            fallback = least_carbon_action(env)
            return fallback, fallback != action
        return action, False
    region = env.regions[action]
    if not region.can_fit(job.cpu_requirement, job.gpu_requirement):
        fallback = least_carbon_action(env)
        return fallback, fallback != action
    return action, False
