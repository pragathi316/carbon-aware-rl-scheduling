"""
environment/trace_workload.py

Trace-driven workload from the Alibaba PAI GPU-cluster trace (Weng et al.,
NSDI 2022), preprocessed by scripts/prepare_pai_trace.py.

For a scenario seed, a window of consecutive trace jobs is taken (training
windows from the first two thirds of the trace's 7.7 days, test windows from
the last third) and replayed with:
  * the jobs' real CPU/GPU demand and execution time;
  * arrival ORDER and relative GAPS from the trace, rescaled so that the mean
    inter-arrival time equals `mean_interarrival` simulator steps (the real
    cluster is far larger than the simulated one, so absolute rates cannot
    be kept);
  * deadlines = arrival + execution_time * U(slack_range), because the trace
    has no deadlines.
"""

from __future__ import annotations

import csv
import os
import random
from functools import lru_cache
from typing import List, Tuple

from environment.job import Job

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "workload", "pai_jobs.csv")
TRAIN_FRACTION = 2.0 / 3.0


@lru_cache(maxsize=2)
def load_trace(path: str = DEFAULT_PATH) -> List[Tuple[float, float, float, float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; build it with scripts/prepare_pai_trace.py")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["submit_time"]), float(r["cpu"]), float(r["gpu"]), float(r["exec_hours"])))
    return rows


def generate_trace_jobs(n_jobs: int, seed: int, split: str = "train", mean_interarrival: float = 2.0,
                        slack_range: Tuple[float, float] = (1.5, 4.0), path: str = DEFAULT_PATH) -> List[Job]:
    trace = load_trace(path)
    boundary = int(len(trace) * TRAIN_FRACTION)
    lo, hi = (0, boundary - n_jobs) if split == "train" else (boundary, len(trace) - n_jobs)
    rng = random.Random(seed)
    start = rng.randint(lo, hi)
    window = trace[start:start + n_jobs]
    t0 = window[0][0]
    span = window[-1][0] - t0
    scale = (span / max(1, n_jobs - 1)) / mean_interarrival if span > 0 else 1.0
    jobs = []
    for i, (submit, cpu, gpu, exec_h) in enumerate(window):
        arrival = (submit - t0) / scale if scale > 0 else i * mean_interarrival
        deadline = arrival + exec_h * rng.uniform(*slack_range)
        jobs.append(Job(job_id=i, arrival_time=round(arrival, 2), cpu_requirement=round(cpu, 2),
                        execution_time=round(exec_h, 2), deadline=round(deadline, 2), priority=1,
                        gpu_requirement=float(gpu), job_type="pai_trace"))
    return jobs
