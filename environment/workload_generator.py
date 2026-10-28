"""
environment/workload_generator.py

Generates reproducible synthetic cloud workloads (lists of Job objects)
for the carbon-aware scheduler simulation, and reads/writes them to CSV
so experiments can be replayed exactly (Phase 1 of the roadmap).

Pipeline per job, as specified in the roadmap:
    sample CPU requirement
    -> sample execution duration
    -> sample arrival time
    -> assign deadline
    -> return workload list

Usage as a library:
    from environment.workload_generator import generate_jobs, save_workload_to_csv

    jobs = generate_jobs(n_jobs=200, intensity="medium", seed=42)
    save_workload_to_csv(jobs, "data/workload_data.csv")

Usage from the command line:
    python -m environment.workload_generator --n_jobs 200 --intensity heavy --seed 42 \
        --output data/workload_data.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import fields
from typing import List, Optional, Dict, Any

from environment.job import Job


# ---------------------------------------------------------------------------
# Intensity presets
# ---------------------------------------------------------------------------
# mean_interarrival_time: average simulation-time gap between consecutive
#   job arrivals (smaller = jobs arrive more frequently = heavier load).
# cpu_range: (min, max) CPU units a job may request.
# exec_time_range: (min, max) execution duration.
# slack_factor_range: (min, max) multiplier applied to execution_time to
#   produce extra slack before the deadline (1.0 = no slack at all).
# gpu_probability: chance a given job also requests GPU units.
# gpu_range: (min, max) GPU units requested, when a job does request GPU.
INTENSITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "light": {
        "mean_interarrival_time": 5.0,
        "cpu_range": (1.0, 4.0),
        "exec_time_range": (2.0, 10.0),
        "slack_factor_range": (1.5, 4.0),
        "gpu_probability": 0.05,
        "gpu_range": (1.0, 2.0),
    },
    "medium": {
        "mean_interarrival_time": 2.0,
        "cpu_range": (2.0, 8.0),
        "exec_time_range": (5.0, 20.0),
        "slack_factor_range": (1.2, 2.5),
        "gpu_probability": 0.15,
        "gpu_range": (1.0, 4.0),
    },
    # Same as "medium" but with more deadline slack, so temporal shifting can pay off.
    "medium_flexible": {
        "mean_interarrival_time": 2.0,
        "cpu_range": (2.0, 8.0),
        "exec_time_range": (5.0, 20.0),
        "slack_factor_range": (1.5, 4.0),
        "gpu_probability": 0.15,
        "gpu_range": (1.0, 4.0),
    },
    "heavy": {
        "mean_interarrival_time": 0.5,
        "cpu_range": (4.0, 16.0),
        "exec_time_range": (10.0, 40.0),
        "slack_factor_range": (1.0, 1.5),
        "gpu_probability": 0.30,
        "gpu_range": (2.0, 8.0),
    },
}

CSV_FIELDNAMES = [f.name for f in fields(Job)]


def generate_jobs(
    n_jobs: int,
    intensity: str = "medium",
    seed: Optional[int] = None,
    priority_weights: Optional[Dict[int, float]] = None,
) -> List[Job]:
    """Generate a reproducible list of Job objects.

    Args:
        n_jobs: Number of jobs to generate.
        intensity: One of "light", "medium", "heavy" — controls arrival
            rate and resource-requirement distributions (see
            INTENSITY_PRESETS).
        seed: Random seed for reproducibility. If None, generation is
            non-deterministic.
        priority_weights: Optional mapping of priority level -> relative
            weight, e.g. {1: 0.7, 2: 0.2, 3: 0.1}. Defaults to a simple
            three-level distribution weighted toward normal priority.

    Returns:
        List of Job objects sorted by arrival_time ascending.
    """
    if n_jobs <= 0:
        raise ValueError("n_jobs must be > 0")
    if intensity not in INTENSITY_PRESETS:
        raise ValueError(
            f"Unknown intensity '{intensity}'. Choose from {list(INTENSITY_PRESETS)}."
        )

    rng = random.Random(seed)
    preset = INTENSITY_PRESETS[intensity]

    if priority_weights is None:
        priority_weights = {1: 0.7, 2: 0.2, 3: 0.1}
    priorities = list(priority_weights.keys())
    weights = list(priority_weights.values())

    jobs: List[Job] = []
    arrival_time = 0.0

    for i in range(n_jobs):
        # Sample CPU requirement.
        cpu_requirement = rng.uniform(*preset["cpu_range"])

        # Sample execution duration.
        execution_time = rng.uniform(*preset["exec_time_range"])

        # Sample arrival time via exponential inter-arrival gaps, which
        # produces a realistic bursty/poisson-like arrival process.
        if i > 0:
            interarrival = rng.expovariate(1.0 / preset["mean_interarrival_time"])
            arrival_time += interarrival

        # Assign deadline = arrival + execution_time * slack_factor.
        slack_factor = rng.uniform(*preset["slack_factor_range"])
        deadline = arrival_time + execution_time * slack_factor

        # Optional GPU requirement.
        gpu_requirement = 0.0
        if rng.random() < preset["gpu_probability"]:
            gpu_requirement = rng.uniform(*preset["gpu_range"])

        priority = rng.choices(priorities, weights=weights, k=1)[0]

        job = Job(
            job_id=i,
            arrival_time=round(arrival_time, 2),
            cpu_requirement=round(cpu_requirement, 2),
            execution_time=round(execution_time, 2),
            deadline=round(deadline, 2),
            priority=priority,
            gpu_requirement=round(gpu_requirement, 2),
            job_type=intensity,
        )
        jobs.append(job)

    return jobs


def save_workload_to_csv(jobs: List[Job], filepath: str) -> None:
    """Write a list of Jobs to CSV so the exact workload can be replayed."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for job in jobs:
            writer.writerow(job.to_dict())


def load_workload_from_csv(filepath: str) -> List[Job]:
    """Read a list of Jobs back from a CSV written by save_workload_to_csv."""
    jobs: List[Job] = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(Job.from_dict(row))
    return jobs


def summarize_workload(jobs: List[Job]) -> Dict[str, Any]:
    """Quick summary stats, useful for sanity-checking a generated workload."""
    if not jobs:
        return {"n_jobs": 0}

    cpu_vals = [j.cpu_requirement for j in jobs]
    exec_vals = [j.execution_time for j in jobs]
    slack_vals = [j.slack for j in jobs]
    gpu_jobs = sum(1 for j in jobs if j.gpu_requirement > 0)

    return {
        "n_jobs": len(jobs),
        "time_span": round(jobs[-1].arrival_time - jobs[0].arrival_time, 2),
        "avg_cpu_requirement": round(sum(cpu_vals) / len(cpu_vals), 2),
        "avg_execution_time": round(sum(exec_vals) / len(exec_vals), 2),
        "avg_slack": round(sum(slack_vals) / len(slack_vals), 2),
        "jobs_with_gpu": gpu_jobs,
        "gpu_job_fraction": round(gpu_jobs / len(jobs), 2),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic cloud workload CSV for the "
        "carbon-aware scheduler."
    )
    parser.add_argument("--n_jobs", type=int, default=200, help="Number of jobs to generate.")
    parser.add_argument(
        "--intensity",
        type=str,
        default="medium",
        choices=list(INTENSITY_PRESETS.keys()),
        help="Workload intensity preset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/workload_data.csv",
        help="Path to write the generated workload CSV.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    jobs = generate_jobs(n_jobs=args.n_jobs, intensity=args.intensity, seed=args.seed)
    save_workload_to_csv(jobs, args.output)

    summary = summarize_workload(jobs)
    print(f"Generated {summary['n_jobs']} jobs at '{args.intensity}' intensity "
          f"(seed={args.seed}).")
    print(f"Saved to: {args.output}")
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nFirst 5 jobs:")
    for job in jobs[:5]:
        print(f"  {job}")


if __name__ == "__main__":
    main()
