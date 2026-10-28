"""
baselines/tunable.py

Parameterized versions of the deadline-aware heuristics, used to tune their
thresholds on the same validation scenarios as the RL hyperparameters
(rl/tune_heuristics.py).

  make_m1(margin, threshold)       M1 Deadline+Carbon. margin = safety margin as a
                                   fraction of execution time; threshold = minimum
                                   relative drop of the t+1 forecast before delaying.
                                   make_m1(0.1, 0.1) reproduces baselines/deadline_carbon.py.
  make_wfc(margin, tolerance)      Wait-for-Clean. Waits while the cleanest region is
                                   full and its intensity is more than `tolerance`
                                   (relative) below the best feasible region.
                                   make_wfc(0.1, 0.0) reproduces baselines/wait_clean.py.
  make_fo(horizon, margin, thr)    Forecast-Optimal (new). Over all regions that can
                                   host the job and all start offsets h = 0..horizon
                                   that keep the deadline safe and stay within the
                                   simulator's delay limit, pick the (region, start)
                                   pair with the lowest forecast intensity. Runs now
                                   unless waiting is forecast to be more than `thr`
                                   (relative) cleaner than the best feasible region now.

Forecasts for h <= 3 are the simulator's forecasts (what the agents observe).
For h > 3 the heuristic uses the true future value plus Gaussian noise with the
same error model as environment/real_carbon.py (std = 0.05 * h * region mean),
so the longer horizons give the heuristic more information than the agents have.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict

DELAY = 3
MAX_DELAY_ATTEMPTS = 5
FORECAST_ERR = 0.05


def _feasible(env, job):
    return [i for i, r in enumerate(env.regions) if r.can_fit(job.cpu_requirement, job.gpu_requirement)]


def make_m1(margin: float = 0.1, threshold: float = 0.1) -> Callable:
    def pick(env) -> int:
        job = env.current_job
        feas = _feasible(env, job)
        if not feas:
            return DELAY
        best = min(feas, key=lambda i: env.regions[i].carbon_intensity)
        reg = env.regions[best]
        slack = job.deadline - (env.current_time + job.execution_time)
        if slack <= job.execution_time * margin:
            return best
        if env.carbon_provider is not None:
            try:
                f1 = env.carbon_provider.get_predicted(reg.region_id, env.current_time, 1)
            except (KeyError, ValueError):
                f1 = reg.carbon_intensity
            if reg.carbon_intensity > 0 and (reg.carbon_intensity - f1) / reg.carbon_intensity > threshold:
                return DELAY
        return best
    return pick


def make_wfc(margin: float = 0.1, tolerance: float = 0.0) -> Callable:
    def pick(env) -> int:
        job = env.current_job
        feas = _feasible(env, job)
        cleanest = min(range(len(env.regions)), key=lambda i: env.regions[i].carbon_intensity)
        slack_after_wait = job.deadline - (env.current_time + 1.0 + job.execution_time)
        if cleanest not in feas and slack_after_wait > margin * job.execution_time:
            if not feas:
                return DELAY
            best_now = min(env.regions[i].carbon_intensity for i in feas)
            if best_now <= 0 or env.regions[cleanest].carbon_intensity < best_now * (1.0 - tolerance):
                return DELAY
        if not feas:
            return DELAY
        return min(feas, key=lambda i: env.regions[i].carbon_intensity)
    return pick


class _Forecaster:
    """Forecast of region intensity h steps ahead, cached per (region, time, h)."""

    def __init__(self, env):
        self.p = env.carbon_provider
        self.cache: Dict = {}
        self.means = {}
        for rid in self.p.region_ids:
            vals = [row["carbon_intensity_current"] for row in self.p._by_region_and_time[rid].values()]
            self.means[rid] = sum(vals) / len(vals)

    def __call__(self, rid: str, t: float, h: int, current: float) -> float:
        tb = float(math.floor(t))
        key = (rid, tb, h)
        if key in self.cache:
            return self.cache[key]
        val = current
        try:
            if h <= self.p.horizon:
                val = self.p.get_predicted(rid, tb, h)
            else:
                true = self.p.get_current(rid, tb + h)
                rng = random.Random(hash(key) & 0xFFFFFFFF)
                val = max(0.0, true + rng.gauss(0.0, FORECAST_ERR * h * self.means[rid]))
        except (KeyError, ValueError):
            pass
        self.cache[key] = val
        return val


def make_fo(horizon: int = 3, margin: float = 0.1, threshold: float = 0.0) -> Callable:
    state = {"env": None, "fc": None}

    def pick(env) -> int:
        if state["env"] is not env or state["fc"] is None:
            state["env"], state["fc"] = env, _Forecaster(env)
        fc = state["fc"]
        job = env.current_job
        feas = _feasible(env, job)
        used = env.delay_attempts.get(job.job_id, 0)
        max_h = min(horizon, MAX_DELAY_ATTEMPTS - used)
        best_now_idx = min(feas, key=lambda i: env.regions[i].carbon_intensity) if feas else None
        best_now = env.regions[best_now_idx].carbon_intensity if feas else math.inf
        hosts = [i for i, r in enumerate(env.regions)
                 if r.total_cpu >= job.cpu_requirement and r.total_gpu >= job.gpu_requirement]
        best_later = math.inf
        for h in range(1, max_h + 1):
            slack_after = job.deadline - (env.current_time + h + job.execution_time)
            if slack_after <= margin * job.execution_time:
                break
            for i in hosts:
                r = env.regions[i]
                best_later = min(best_later, fc(r.region_id, env.current_time, h, r.carbon_intensity))
        if best_now_idx is None:
            return DELAY
        if best_later < best_now * (1.0 - threshold):
            return DELAY
        return best_now_idx
    return pick


M1_GRID = [dict(margin=m, threshold=t) for m in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0) for t in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)]
WFC_GRID = [dict(margin=m, tolerance=t) for m in (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0) for t in (0.0, 0.1, 0.2, 0.4)]
FO_GRID = [dict(horizon=h, margin=m, threshold=t) for h in (1, 2, 3, 5) for m in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0)
           for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)]
FAMILIES = {"M1": (make_m1, M1_GRID), "WfC": (make_wfc, WFC_GRID), "FO": (make_fo, FO_GRID)}
