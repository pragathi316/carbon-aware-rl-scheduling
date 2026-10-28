"""
environment/cloud_env.py

The core Gymnasium environment for the carbon-aware cloud scheduler
(Phase 4 of the roadmap). One decision = one job: at each step, the
agent sees the job at the front of the queue plus the current state of
all regions and carbon forecasts, and chooses to run it in Region A/B/C
or delay it.

Action space (Discrete(3 + 1), fixed at 3 regions for the MVP per
Section 11 of the roadmap):
    0 -> Run in Region A
    1 -> Run in Region B
    2 -> Run in Region C
    3 -> Delay

State vector (matches the roadmap's Phase 4 spec exactly, with the
single "predicted_carbon_features" placeholder expanded into one
column per region per forecast step, since a flat vector needs a fixed
number of scalars):
    [ current_job_cpu_requirement,
      current_job_execution_time,
      remaining_deadline,
      queue_length,
      region_0_available_cpu, region_1_available_cpu, region_2_available_cpu,
      region_0_current_carbon, region_1_current_carbon, region_2_current_carbon,
      <predicted carbon, per region, per horizon step>,
      electricity_cost (of the current job's cheapest feasible region),
      region_utilization (mean across regions) ]

This module deliberately does NOT touch the LLM context layer (Phase 9)
or XAI (Phase 11) — per the roadmap's incremental-development principle,
those are added only after this environment and RL on top of it are
validated (Phase 8).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.job import Job
from environment.region import Region, create_default_regions
from environment.carbon_data import CarbonDataProvider
from environment.reward import calculate_reward, RewardWeights, NormalizationRanges
from llm_layer.context_builder import BaseContextEngine, CONTEXT_VECTOR_SIZE
from llm_layer.feature_extractor import extract_raw_context


# Energy model constants — deliberately simple and clearly-labeled for the
# MVP (Section 11). Swap these for a measured/literature-backed model later
# without touching the rest of the environment.
CPU_POWER_KW_PER_UNIT = 0.01  # kW consumed per CPU unit, while running.
GPU_POWER_KW_PER_UNIT = 0.05  # kW consumed per GPU unit, while running.
TIMESTEP_HOURS = 1.0          # wall-clock hours represented by 1 simulation timestep.

MAX_DELAY_ATTEMPTS = 5  # after this many DELAY actions, force the job to run.


class CloudSchedulingEnv(gym.Env):
    """Gymnasium environment for carbon-, cost-, and SLA-aware job scheduling."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        jobs: List[Job],
        regions: Optional[List[Region]] = None,
        carbon_provider: Optional[CarbonDataProvider] = None,
        reward_weights: RewardWeights = RewardWeights(),
        reward_ranges: NormalizationRanges = NormalizationRanges(),
        max_time: Optional[float] = None,
        render_mode: Optional[str] = None,
        carbon_aware_state: bool = True,
        context_engine: Optional[BaseContextEngine] = None,
        count_invalid_as_delay: bool = True,
        include_gpu_state: bool = False,
        energy_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        if not jobs:
            raise ValueError("jobs must be a non-empty list")
        if regions is not None and len(regions) != 3:
            raise ValueError(
                "This MVP environment fixes the action space to 3 regions + delay. "
                "Pass exactly 3 regions, or omit `regions` to use the defaults."
            )

        self._initial_jobs = jobs
        self._initial_regions = regions if regions is not None else create_default_regions(3)
        self.carbon_provider = carbon_provider
        self.reward_weights = reward_weights
        self.reward_ranges = reward_ranges
        self.render_mode = render_mode

        # Ablation switch for Phase 8's M2 vs M3 comparison: when False, the
        # agent's OBSERVATION excludes current + predicted carbon intensity
        # entirely (region carbon columns and the forecast vector are
        # dropped from the state). The environment's physics and reward
        # still account for real carbon emissions either way — this isolates
        # whether SEEING carbon information helps the agent reduce it, which
        # is exactly the M2 vs M3 question the roadmap's Phase 8 asks.
        self.carbon_aware_state = carbon_aware_state

        # Phase 10: optional LLM/context-engine layer. When set, its 4
        # structured features (carbon_trend, deadline_flexibility,
        # delay_opportunity, carbon_risk — see llm_layer/context_builder.py)
        # are appended to the state. None means "no context layer" (M0-M3
        # from Phase 8); set this to compare M3 against M3.5 (RL+Carbon+LLM).
        self.context_engine = context_engine
        # An invalid placement (region without capacity) defers the job by one
        # step, i.e. it acts as a wait. If it does not count towards
        # MAX_DELAY_ATTEMPTS, an agent can wait indefinitely for a preferred
        # region by repeatedly choosing it while it is full. Counting it closes
        # that loophole. False reproduces the behaviour of the earlier
        # experiments (Protocols 1 and 2 in the paper).
        self.count_invalid_as_delay = count_invalid_as_delay
        # The original state has no GPU information, so an agent cannot tell
        # that a GPU job will not fit in a region without GPUs. True appends
        # the job's GPU demand and each region's free GPUs (1 + n_regions).
        self.include_gpu_state = include_gpu_state
        # Energy model parameters (defaults = the simple linear model). The
        # "calibrated" variant (rl/config.py) adds idle-power overhead, a
        # higher per-GPU power and a facility PUE.
        ep = dict(cpu_kw=CPU_POWER_KW_PER_UNIT, gpu_kw=GPU_POWER_KW_PER_UNIT, cpu_idle_overhead=0.0, pue=1.0)
        ep.update(energy_params or {})
        self.energy_params = ep

        self.horizon = carbon_provider.horizon if carbon_provider is not None else 0

        # Fallback max_time: last job's deadline plus headroom, unless the
        # caller (or the carbon data's timestep range) constrains it further.
        self.max_time = max_time if max_time is not None else max(j.deadline for j in jobs) + 10

        carbon_state_size = (3 + 3 * self.horizon) if self.carbon_aware_state else 0
        context_state_size = CONTEXT_VECTOR_SIZE if self.context_engine is not None else 0
        n_state = 4 + 3 + carbon_state_size + context_state_size + 1 + 1
        if include_gpu_state:
            n_state += 1 + len(self._initial_regions)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(n_state,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # 0,1,2 = regions; 3 = delay

        # Mutable simulation state — (re)initialized in reset().
        self.current_time: float = 0.0
        self.regions: List[Region] = []
        self.pending_jobs: List[Job] = []
        self.queue: List[Job] = []
        self.deferred_jobs: List[Tuple[float, Job]] = []  # (eligible_time, job)
        self.delay_attempts: Dict[int, int] = {}
        self.reservations: List[Tuple[float, Region, float, float]] = []  # (release_time, region, cpu, gpu)
        self.completed_jobs: List[Dict[str, Any]] = []
        self.current_job: Optional[Job] = None

    # -- Gym API ---------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_time = 0.0
        self.regions = [
            Region(
                region_id=r.region_id,
                total_cpu=r.total_cpu,
                total_gpu=r.total_gpu,
                electricity_price=r.electricity_price,
                carbon_intensity=r.carbon_intensity,
            )
            for r in self._initial_regions
        ]
        self.pending_jobs = sorted(self._initial_jobs, key=lambda j: j.arrival_time)
        self.queue = []
        self.deferred_jobs = []
        self.delay_attempts = {}
        self.reservations = []
        self.completed_jobs = []
        self.current_job = None

        self._advance_to_next_decision()
        return self._get_state(), {"current_time": self.current_time}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_job is None:
            raise RuntimeError("step() called with no current job — call reset() first "
                               "or check the `terminated`/`truncated` flags.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; must be in {{0,1,2,3}}")

        job = self.current_job
        info: Dict[str, Any] = {"job_id": job.job_id, "action": int(action)}

        if action == 3:
            reward, info = self._handle_delay(job, info)
        else:
            reward, info = self._handle_run(job, action, info)

        terminated = not self.pending_jobs and not self.queue and not self.deferred_jobs
        truncated = self.current_time >= self.max_time

        if not terminated and not truncated:
            self._advance_to_next_decision()
            if self.current_job is None:
                # Ran out of jobs while advancing (e.g. all forced through).
                terminated = True

        next_state = self._get_state() if not (terminated or truncated) else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        info["current_time"] = self.current_time
        return next_state, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        queued = len(self.queue)
        print(
            f"t={self.current_time:.1f} | queue={queued} | "
            f"pending={len(self.pending_jobs)} | completed={len(self.completed_jobs)} | "
            + " | ".join(f"{r.region_id}:{r.cpu_utilization:.0%}" for r in self.regions)
        )

    # -- Simulation mechanics ----------------------------------------------

    def _advance_to_next_decision(self) -> None:
        """Advance time until a job is available to decide on, or the sim ends."""
        while True:
            self._release_finished_reservations()
            self._admit_arrived_jobs()
            self._admit_deferred_jobs()
            self._sync_carbon_intensities()

            if self.queue:
                self.current_job = self.queue.pop(0)
                return

            if not self.pending_jobs and not self.reservations and not self.deferred_jobs:
                self.current_job = None
                return

            if self.current_time >= self.max_time:
                self.current_job = None
                return

            self.current_time += 1.0

    def _release_finished_reservations(self) -> None:
        still_running = []
        for release_time, region, cpu, gpu in self.reservations:
            if release_time <= self.current_time:
                region.release(cpu, gpu)
            else:
                still_running.append((release_time, region, cpu, gpu))
        self.reservations = still_running

    def _admit_arrived_jobs(self) -> None:
        still_pending = []
        for job in self.pending_jobs:
            if job.arrival_time <= self.current_time:
                self.queue.append(job)
            else:
                still_pending.append(job)
        self.pending_jobs = still_pending

    def _admit_deferred_jobs(self) -> None:
        """Move jobs delayed earlier back into the queue, but only once
        simulation time has actually reached their eligible_time — this is
        what makes DELAY a real wait rather than an immediate re-pick."""
        still_deferred = []
        for eligible_time, job in self.deferred_jobs:
            if eligible_time <= self.current_time:
                self.queue.append(job)
            else:
                still_deferred.append((eligible_time, job))
        self.deferred_jobs = still_deferred

    def _sync_carbon_intensities(self) -> None:
        """Pull each region's carbon_intensity from the carbon provider so
        state, energy/carbon calculations, and the provider's own current
        value all agree at this timestep."""
        if self.carbon_provider is None:
            return
        for region in self.regions:
            try:
                region.update_carbon_intensity(
                    self.carbon_provider.get_current(region.region_id, self.current_time)
                )
            except KeyError:
                # No carbon data at this exact timestamp (e.g. ran past the
                # generated series) — keep the region's last known value.
                pass

    def _handle_delay(self, job: Job, info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        attempts = self.delay_attempts.get(job.job_id, 0) + 1
        self.delay_attempts[job.job_id] = attempts

        if attempts > MAX_DELAY_ATTEMPTS:
            # Force it through in whatever region is currently cheapest on
            # carbon and can fit it, so the episode can't stall forever.
            forced_region_idx = self._best_feasible_region_index(job)
            if forced_region_idx is None:
                # Nothing fits right now — defer one real timestep rather
                # than looping on the same instant.
                self.deferred_jobs.append((self.current_time + 1.0, job))
                return -0.05, {**info, "delay_attempts": attempts, "forced": False}
            return self._handle_run(job, forced_region_idx, {**info, "forced": True})

        # Defer to the NEXT timestep — not back into the current queue —
        # so a delayed job is only reconsidered once real time has passed
        # and carbon/region state has actually had a chance to change.
        self.deferred_jobs.append((self.current_time + 1.0, job))
        # Small shaping penalty so "always delay" isn't free, without
        # dominating the SLA/carbon/cost terms.
        return -0.01, {**info, "delay_attempts": attempts}

    def _handle_run(
        self, job: Job, region_idx: int, info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        region = self.regions[region_idx]

        if not region.allocate(job.cpu_requirement, job.gpu_requirement):
            # Invalid choice — region doesn't have capacity right now, and
            # capacity only frees up when a running job's reservation ends
            # at some future time, so retrying at this same instant would
            # just fail again. Defer to the next timestep instead.
            if self.count_invalid_as_delay and not info.get("forced"):
                attempts = self.delay_attempts.get(job.job_id, 0) + 1
                self.delay_attempts[job.job_id] = attempts
                if attempts > MAX_DELAY_ATTEMPTS:
                    forced_idx = self._best_feasible_region_index(job)
                    if forced_idx is not None:
                        reward, run_info = self._handle_run(job, forced_idx, {**info, "forced": True})
                        return reward - 0.1, {**run_info, "invalid": True}
            self.deferred_jobs.append((self.current_time + 1.0, job))
            return -0.1, {**info, "invalid": True, "region_id": region.region_id}

        start_time = self.current_time
        finish_time = start_time + job.execution_time
        job.assigned_region = region.region_id
        job.start_time = start_time
        job.finish_time = finish_time

        self.reservations.append((finish_time, region, job.cpu_requirement, job.gpu_requirement))

        energy_kwh = self._calculate_energy(job)
        carbon_grams = self._calculate_carbon(energy_kwh, region.carbon_intensity)
        cost = energy_kwh * region.electricity_price
        delay = max(0.0, start_time - job.arrival_time)
        sla_violated = job.is_sla_violated(finish_time)

        result = calculate_reward(
            carbon_grams=carbon_grams,
            cost=cost,
            delay=delay,
            sla_violated=sla_violated,
            weights=self.reward_weights,
            ranges=self.reward_ranges,
        )

        self.completed_jobs.append(
            {
                "job_id": job.job_id,
                "region_id": region.region_id,
                "start_time": start_time,
                "finish_time": finish_time,
                "energy_kwh": energy_kwh,
                "carbon_grams": carbon_grams,
                "cost": cost,
                "delay": delay,
                "sla_violated": sla_violated,
                "reward": result["reward"],
            }
        )

        info.update(
            {
                "region_id": region.region_id,
                "energy_kwh": energy_kwh,
                "carbon_grams": carbon_grams,
                "cost": cost,
                "delay": delay,
                "sla_violated": sla_violated,
                "reward_breakdown": result["breakdown"],
            }
        )
        return result["reward"], info

    def _best_feasible_region_index(self, job: Job) -> Optional[int]:
        feasible = [
            (idx, r.carbon_intensity)
            for idx, r in enumerate(self.regions)
            if r.can_fit(job.cpu_requirement, job.gpu_requirement)
        ]
        if not feasible:
            return None
        return min(feasible, key=lambda pair: pair[1])[0]

    # -- Energy / carbon models ----------------------------------------------

    def _calculate_energy(self, job: Job) -> float:
        """Estimate energy used by executing the job, in kWh."""
        exec_hours = job.execution_time * TIMESTEP_HOURS
        ep = self.energy_params
        cpu_energy = job.cpu_requirement * ep["cpu_kw"] * (1.0 + ep["cpu_idle_overhead"]) * exec_hours
        gpu_energy = job.gpu_requirement * ep["gpu_kw"] * exec_hours
        return (cpu_energy + gpu_energy) * ep["pue"]

    def _calculate_carbon(self, energy_kwh: float, carbon_intensity: float) -> float:
        """Convert energy usage (kWh) to carbon emissions (gCO2)."""
        return energy_kwh * carbon_intensity

    # -- State construction ---------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Valid actions for the current job (used by MaskablePPO): a region is
        valid only if it can fit the job right now; delay is always valid."""
        job = self.current_job
        if job is None:
            return np.ones(4, dtype=bool)
        mask = [r.can_fit(job.cpu_requirement, job.gpu_requirement) for r in self.regions]
        return np.array(mask + [True], dtype=bool)

    def get_feature_names(self) -> List[str]:
        """Names for every column of the state vector, in the EXACT order
        _get_state() builds them — the single source of truth the XAI layer
        (Phase 11) uses to map SHAP values back to meaningful names. Update
        this alongside _get_state() if that method's column order changes.
        """
        names = ["job_cpu_requirement", "job_execution_time", "remaining_deadline", "queue_length"]
        names += [f"region_{r.region_id}_available_cpu" for r in self._initial_regions]

        if self.carbon_aware_state:
            names += [f"region_{r.region_id}_carbon_intensity" for r in self._initial_regions]
            if self.carbon_provider is not None:
                for r in self._initial_regions:
                    names += [
                        f"region_{r.region_id}_carbon_forecast_t+{k}"
                        for k in range(1, self.horizon + 1)
                    ]

        if self.context_engine is not None:
            names += ["context_carbon_trend", "context_deadline_flexibility",
                      "context_delay_opportunity", "context_carbon_risk"]

        names += ["cheapest_electricity_cost", "mean_region_utilization"]
        if self.include_gpu_state:
            names += ["job_gpu_requirement"] + [f"region_{r.region_id}_available_gpu" for r in self._initial_regions]
        return names

    def _get_state(self) -> np.ndarray:
        job = self.current_job
        assert job is not None

        remaining_deadline = max(0.0, job.deadline - self.current_time)
        queue_length = float(len(self.queue))

        region_cpu = [r.available_cpu for r in self.regions]

        region_carbon: List[float] = []
        predicted_features: List[float] = []
        if self.carbon_aware_state:
            region_carbon = [r.carbon_intensity for r in self.regions]
            if self.carbon_provider is not None:
                for r in self.regions:
                    predicted_features.extend(
                        self.carbon_provider.get_forecast_vector(r.region_id, self.current_time)[1:]
                    )

        cheapest_electricity_cost = min(r.electricity_price for r in self.regions)
        mean_utilization = float(np.mean([r.cpu_utilization for r in self.regions]))

        context_vector: List[float] = []
        if self.context_engine is not None:
            raw_context = extract_raw_context(self)
            context_vector = self.context_engine.build_context(raw_context).to_vector()

        state = (
            [job.cpu_requirement, job.execution_time, remaining_deadline, queue_length]
            + region_cpu
            + region_carbon
            + predicted_features
            + context_vector
            + [cheapest_electricity_cost, mean_utilization]
        )
        if self.include_gpu_state:
            state += [job.gpu_requirement] + [r.available_gpu for r in self.regions]
        return np.array(state, dtype=np.float32)


if __name__ == "__main__":
    from environment.workload_generator import generate_jobs
    from environment.carbon_data import generate_carbon_series

    jobs = generate_jobs(n_jobs=30, intensity="medium", seed=1)
    regions = create_default_regions(3)
    carbon_rows = generate_carbon_series(
        region_ids=[r.region_id for r in regions],
        n_timesteps=200,
        horizon=2,
        volatility="medium",
        seed=1,
    )
    carbon_provider = CarbonDataProvider(carbon_rows)

    env = CloudSchedulingEnv(jobs=jobs, regions=regions, carbon_provider=carbon_provider)
    obs, info = env.reset(seed=1)
    print(f"Observation shape: {obs.shape} (expected {env.observation_space.shape})")

    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    rng = np.random.default_rng(1)

    while not (terminated or truncated):
        action = rng.integers(0, 4)  # random policy, just to prove the loop runs
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if steps <= 5:
            print(f"step={steps} action={action} reward={reward:.3f} "
                  f"region={info.get('region_id')} sla_violated={info.get('sla_violated')}")

    print(f"\nEpisode finished after {steps} steps "
          f"(terminated={terminated}, truncated={truncated}).")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Jobs completed: {len(env.completed_jobs)} / {len(jobs)}")
    if env.completed_jobs:
        sla_violations = sum(1 for j in env.completed_jobs if j["sla_violated"])
        total_carbon = sum(j["carbon_grams"] for j in env.completed_jobs)
        print(f"SLA violations: {sla_violations} ({sla_violations/len(env.completed_jobs):.0%})")
        print(f"Total carbon: {total_carbon:.0f} gCO2")
