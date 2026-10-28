"""
environment/job.py

Defines the Job class: the basic unit of work scheduled by the
carbon-aware cloud scheduler. A Job carries everything the environment
and reward function need to know about a single workload — when it
arrived, how much compute it needs, how long it will take to run, and
by when it must finish.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class Job:
    """A single cloud workload job.

    Attributes:
        job_id: Unique identifier for the job.
        arrival_time: Simulation timestep (or seconds) at which the job
            becomes available to be scheduled.
        cpu_requirement: Number of CPU units/cores the job needs.
        execution_time: How long the job takes to run once started,
            in the same time units as arrival_time/deadline.
        deadline: Latest simulation timestep by which the job must have
            completed execution without violating its SLA.
        priority: Integer priority level (higher = more urgent/important).
            Defaults to 1 (normal priority).
        gpu_requirement: Optional GPU units the job needs. Defaults to 0
            (no GPU needed), per the roadmap's suggestion to keep GPU
            optional in early phases.
        job_type: Optional free-text label (e.g. "batch", "inference",
            "training") for later analysis/plots. Not required by the
            environment logic.

    Derived / mutable scheduling state (filled in later by the
    environment, not by the generator):
        assigned_region: Region id the job was scheduled to run in,
            or None if not yet scheduled.
        start_time: Simulation timestep the job actually started
            running, or None if not yet started.
        finish_time: Simulation timestep the job actually finished,
            or None if not yet finished.
    """

    job_id: int
    arrival_time: float
    cpu_requirement: float
    execution_time: float
    deadline: float
    priority: int = 1
    gpu_requirement: float = 0.0
    job_type: Optional[str] = None

    # Scheduling state — set by the environment during simulation.
    assigned_region: Optional[str] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    def __post_init__(self) -> None:
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be >= 0")
        if self.cpu_requirement <= 0:
            raise ValueError("cpu_requirement must be > 0")
        if self.gpu_requirement < 0:
            raise ValueError("gpu_requirement must be >= 0")
        if self.execution_time <= 0:
            raise ValueError("execution_time must be > 0")
        if self.deadline <= self.arrival_time:
            raise ValueError("deadline must be after arrival_time")

    @property
    def slack(self) -> float:
        """Time available between arrival and deadline beyond the
        minimum execution time. Larger slack = more room to delay."""
        return (self.deadline - self.arrival_time) - self.execution_time

    def is_sla_violated(self, finish_time: float) -> bool:
        """Whether finishing at `finish_time` breaks this job's deadline."""
        return finish_time > self.deadline

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict representation, e.g. for CSV export via csv.DictWriter."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Reconstruct a Job from a dict (e.g. a row read back from CSV).

        Handles the string->correct-type coercion needed when the dict
        comes from a CSV row (all values arrive as strings).
        """
        def _num(value: Any) -> Optional[float]:
            if value is None or value == "" or value == "None":
                return None
            return float(value)

        return cls(
            job_id=int(float(data["job_id"])),
            arrival_time=float(data["arrival_time"]),
            cpu_requirement=float(data["cpu_requirement"]),
            execution_time=float(data["execution_time"]),
            deadline=float(data["deadline"]),
            priority=int(float(data.get("priority", 1) or 1)),
            gpu_requirement=float(data.get("gpu_requirement", 0.0) or 0.0),
            job_type=data.get("job_type") or None,
            assigned_region=data.get("assigned_region") or None,
            start_time=_num(data.get("start_time")),
            finish_time=_num(data.get("finish_time")),
        )

    def __repr__(self) -> str:
        return (
            f"Job(id={self.job_id}, arrival={self.arrival_time:.1f}, "
            f"cpu={self.cpu_requirement:.1f}, gpu={self.gpu_requirement:.1f}, "
            f"exec_time={self.execution_time:.1f}, deadline={self.deadline:.1f}, "
            f"priority={self.priority}, slack={self.slack:.1f})"
        )


if __name__ == "__main__":
    # Minimal smoke test / usage example.
    job = Job(
        job_id=1,
        arrival_time=0.0,
        cpu_requirement=4.0,
        execution_time=10.0,
        deadline=30.0,
        priority=2,
        gpu_requirement=0.0,
        job_type="batch",
    )
    print(job)
    print("As dict:", job.to_dict())

    rebuilt = Job.from_dict(job.to_dict())
    assert rebuilt == job, "round-trip through to_dict/from_dict should be lossless"
    print("Round-trip OK.")
