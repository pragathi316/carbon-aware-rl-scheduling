"""
environment/region.py

Defines the Region class: a simulated cloud region with finite CPU/GPU
capacity, an electricity price, and a carbon intensity. Jobs are
allocated to (and released from) a Region as the scheduler runs them
(Phase 2 of the roadmap).

GPU capacity is optional per the roadmap ("Initially keep GPU optional
to reduce complexity") — regions default to gpu_capacity=0.0, meaning
they simply can't accept GPU jobs unless you configure otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class Region:
    """A simulated cloud region.

    Attributes:
        region_id: Unique identifier, e.g. "A", "B", "us-east-1".
        total_cpu: Total CPU capacity of the region.
        available_cpu: CPU currently free. Starts equal to total_cpu.
        total_gpu: Total GPU capacity of the region (0.0 = no GPU support).
        available_gpu: GPU currently free.
        electricity_price: Current electricity price (currency units per
            kWh, or an abstract unit — stay consistent across regions).
        carbon_intensity: Current carbon intensity, in gCO2/kWh.
    """

    region_id: str
    total_cpu: float
    available_cpu: float = field(default=None)  # type: ignore[assignment]
    total_gpu: float = 0.0
    available_gpu: float = field(default=None)  # type: ignore[assignment]
    electricity_price: float = 0.10
    carbon_intensity: float = 400.0

    def __post_init__(self) -> None:
        if self.total_cpu <= 0:
            raise ValueError("total_cpu must be > 0")
        if self.total_gpu < 0:
            raise ValueError("total_gpu must be >= 0")
        # available_cpu/available_gpu default to full capacity if not given.
        if self.available_cpu is None:
            self.available_cpu = self.total_cpu
        if self.available_gpu is None:
            self.available_gpu = self.total_gpu
        self._validate_bounds()

    def _validate_bounds(self) -> None:
        if not (0.0 <= self.available_cpu <= self.total_cpu):
            raise ValueError(
                f"available_cpu ({self.available_cpu}) out of bounds "
                f"[0, {self.total_cpu}] for region {self.region_id}"
            )
        if not (0.0 <= self.available_gpu <= self.total_gpu):
            raise ValueError(
                f"available_gpu ({self.available_gpu}) out of bounds "
                f"[0, {self.total_gpu}] for region {self.region_id}"
            )

    # -- Capacity / utilization -------------------------------------------------

    @property
    def cpu_utilization(self) -> float:
        """Fraction of CPU capacity currently in use, in [0, 1]."""
        return 1.0 - (self.available_cpu / self.total_cpu)

    @property
    def gpu_utilization(self) -> float:
        """Fraction of GPU capacity currently in use, in [0, 1]. 0 if no GPU."""
        if self.total_gpu == 0:
            return 0.0
        return 1.0 - (self.available_gpu / self.total_gpu)

    def can_fit(self, cpu_requirement: float, gpu_requirement: float = 0.0) -> bool:
        """Whether this region currently has enough free capacity for a job."""
        if cpu_requirement > self.available_cpu:
            return False
        if gpu_requirement > 0 and gpu_requirement > self.available_gpu:
            return False
        return True

    def allocate(self, cpu_requirement: float, gpu_requirement: float = 0.0) -> bool:
        """Reserve capacity for a job. Returns False (no-op) if it doesn't fit."""
        if not self.can_fit(cpu_requirement, gpu_requirement):
            return False
        self.available_cpu -= cpu_requirement
        self.available_gpu -= gpu_requirement
        return True

    def release(self, cpu_requirement: float, gpu_requirement: float = 0.0) -> None:
        """Free capacity previously reserved by allocate(), e.g. on job completion."""
        self.available_cpu = min(self.total_cpu, self.available_cpu + cpu_requirement)
        self.available_gpu = min(self.total_gpu, self.available_gpu + gpu_requirement)

    def reset(self) -> None:
        """Return the region to full available capacity (e.g. at episode start)."""
        self.available_cpu = self.total_cpu
        self.available_gpu = self.total_gpu

    # -- Time-varying signals -----------------------------------------------

    def update_carbon_intensity(self, value: float) -> None:
        if value < 0:
            raise ValueError("carbon_intensity must be >= 0")
        self.carbon_intensity = value

    def update_electricity_price(self, value: float) -> None:
        if value < 0:
            raise ValueError("electricity_price must be >= 0")
        self.electricity_price = value

    # -- Serialization / introspection ---------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "total_cpu": self.total_cpu,
            "available_cpu": self.available_cpu,
            "total_gpu": self.total_gpu,
            "available_gpu": self.available_gpu,
            "electricity_price": self.electricity_price,
            "carbon_intensity": self.carbon_intensity,
            "cpu_utilization": round(self.cpu_utilization, 3),
            "gpu_utilization": round(self.gpu_utilization, 3),
        }

    def __repr__(self) -> str:
        return (
            f"Region({self.region_id}: cpu={self.available_cpu:.1f}/{self.total_cpu:.1f} "
            f"({self.cpu_utilization:.0%} used), "
            f"gpu={self.available_gpu:.1f}/{self.total_gpu:.1f}, "
            f"price={self.electricity_price:.3f}, "
            f"carbon={self.carbon_intensity:.0f} gCO2/kWh)"
        )


def create_default_regions(n: int = 3) -> List[Region]:
    """Create 3-5 example regions with distinct cost/carbon profiles.

    The profiles are deliberately non-uniform (some cheap-but-dirty, some
    clean-but-expensive) so a scheduler actually has a meaningful trade-off
    to learn, rather than one region being strictly better on every axis.

    Args:
        n: Number of regions to create, 3 to 5 inclusive.
    """
    if not (3 <= n <= 5):
        raise ValueError("n must be between 3 and 5, per the roadmap's spec")

    profiles = [
        # region_id, total_cpu, total_gpu, electricity_price, carbon_intensity
        ("A", 100.0, 8.0, 0.12, 600.0),  # cheap-ish, dirty grid
        ("B", 80.0, 4.0, 0.18, 250.0),   # pricier, cleaner grid
        ("C", 120.0, 0.0, 0.15, 400.0),  # large CPU-only, mid carbon
        ("D", 60.0, 2.0, 0.22, 120.0),   # small, expensive, very clean
        ("E", 90.0, 6.0, 0.10, 550.0),   # cheap, dirty, GPU-heavy
    ]

    regions = []
    for region_id, total_cpu, total_gpu, price, carbon in profiles[:n]:
        regions.append(
            Region(
                region_id=region_id,
                total_cpu=total_cpu,
                total_gpu=total_gpu,
                electricity_price=price,
                carbon_intensity=carbon,
            )
        )
    return regions


if __name__ == "__main__":
    regions = create_default_regions(3)
    print("Initial region states:")
    for r in regions:
        print(f"  {r}")

    # Simulate allocating a job to Region A.
    region_a = regions[0]
    ok = region_a.allocate(cpu_requirement=20.0, gpu_requirement=2.0)
    print(f"\nAllocated 20 CPU + 2 GPU to Region A: success={ok}")
    print(f"  {region_a}")

    # Try to over-allocate — should fail cleanly.
    ok = region_a.allocate(cpu_requirement=1000.0)
    print(f"\nAttempted to over-allocate Region A: success={ok} (expected False)")

    # Release the job's resources back.
    region_a.release(cpu_requirement=20.0, gpu_requirement=2.0)
    print(f"\nAfter releasing: {region_a}")

    # Simulate a carbon-intensity update, e.g. from a new simulation timestep.
    region_a.update_carbon_intensity(700.0)
    print(f"\nAfter carbon update: {region_a}")

    print("\nAs dicts (e.g. for logging/state construction):")
    for r in regions:
        print(f"  {r.to_dict()}")


# Region profiles for the trace-driven (GPU-heavy) scenarios: every region has
# GPUs, so GPU jobs can choose between all three; the clean region (B) is the
# smallest, which makes "wait for the clean region" a real trade-off.
GPU_CLUSTER_PROFILES = {
    # region_id: (total_cpu, total_gpu, electricity_price, nominal carbon)
    "A": (160.0, 16.0, 0.12, 600.0),
    "B": (96.0, 8.0, 0.18, 250.0),
    "C": (128.0, 12.0, 0.15, 400.0),
}


def create_gpu_cluster_regions() -> List[Region]:
    return [Region(region_id=rid, total_cpu=c, total_gpu=g, electricity_price=p, carbon_intensity=ci)
            for rid, (c, g, p, ci) in GPU_CLUSTER_PROFILES.items()]
