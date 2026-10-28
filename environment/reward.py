"""
environment/reward.py

Multi-objective reward function combining carbon emissions, electricity
cost, job delay, and SLA violations (Phase 5 of the roadmap).

    reward = - alpha * normalized_carbon
             - beta  * normalized_cost
             - gamma * normalized_delay
             - delta * SLA_violation

Design choices called out explicitly in the roadmap, preserved here:
  - All terms are normalized to comparable scales so no term dominates
    purely because of unit/magnitude differences (e.g. grams of CO2 vs
    dollars vs timesteps).
  - The SLA penalty is deliberately the largest lever (delta) by default,
    because a weakly-penalized delay term lets the agent learn to delay
    everything to chase cleaner electricity, which Section 5 of the
    roadmap explicitly warns against.
  - Every reward call returns a full breakdown (not just the scalar) so
    experiments can log and later ablate which term drove a decision —
    required for the XAI layer (Phase 11) and the ablation study
    (Section 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RewardWeights:
    """Weights for each term of the multi-objective reward.

    Record the exact weights used in every experiment (per the roadmap) —
    e.g. by logging `RewardWeights.__dict__` alongside your results.
    """

    alpha: float = 0.4  # carbon weight
    beta: float = 0.2   # cost weight
    gamma: float = 0.2  # delay weight
    delta: float = 0.6  # SLA violation weight — kept strong deliberately


@dataclass
class NormalizationRanges:
    """Expected max values used to scale raw metrics into [0, 1] before
    weighting. These are rough operating ranges, not hard limits — actual
    values are clipped to [0, 1] after division, so occasional outliers
    beyond the max just saturate at 1.0 rather than blowing up the reward.

    Tune these to your actual workload/region config: run a baseline
    (e.g. FIFO) first, look at the resulting carbon/cost/delay
    distributions, and set max_* around the 90th-95th percentile.
    """

    max_carbon_grams: float = 5000.0
    max_cost: float = 50.0
    max_delay: float = 100.0


def _normalize(value: float, max_value: float) -> float:
    """Scale a non-negative value into [0, 1], clipping outliers at 1.0."""
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def calculate_reward(
    carbon_grams: float,
    cost: float,
    delay: float,
    sla_violated: bool,
    weights: RewardWeights = RewardWeights(),
    ranges: NormalizationRanges = NormalizationRanges(),
) -> Dict[str, Any]:
    """Compute the multi-objective reward and a full breakdown.

    Args:
        carbon_grams: Carbon emitted by this decision, in gCO2.
        cost: Electricity cost incurred by this decision.
        delay: Time the job waited beyond its arrival time before running.
        sla_violated: Whether the job's deadline was missed.
        weights: Per-term weights (alpha/beta/gamma/delta).
        ranges: Normalization ranges for scaling raw metrics to [0, 1].

    Returns:
        Dict with keys: "reward" (the scalar to give the RL agent) and
        "breakdown" (per-term normalized values and weighted contributions,
        useful for logging, ablation, and later feeding the XAI layer).
    """
    norm_carbon = _normalize(carbon_grams, ranges.max_carbon_grams)
    norm_cost = _normalize(cost, ranges.max_cost)
    norm_delay = _normalize(delay, ranges.max_delay)
    sla_term = 1.0 if sla_violated else 0.0

    carbon_penalty = weights.alpha * norm_carbon
    cost_penalty = weights.beta * norm_cost
    delay_penalty = weights.gamma * norm_delay
    sla_penalty = weights.delta * sla_term

    reward = -(carbon_penalty + cost_penalty + delay_penalty + sla_penalty)

    return {
        "reward": reward,
        "breakdown": {
            "normalized_carbon": norm_carbon,
            "normalized_cost": norm_cost,
            "normalized_delay": norm_delay,
            "sla_violated": sla_term,
            "carbon_penalty": carbon_penalty,
            "cost_penalty": cost_penalty,
            "delay_penalty": delay_penalty,
            "sla_penalty": sla_penalty,
            "raw_carbon_grams": carbon_grams,
            "raw_cost": cost,
            "raw_delay": delay,
        },
    }


if __name__ == "__main__":
    weights = RewardWeights()
    ranges = NormalizationRanges()

    print("Case 1: clean, cheap, on-time job")
    result = calculate_reward(
        carbon_grams=300.0, cost=1.5, delay=0.0, sla_violated=False,
        weights=weights, ranges=ranges,
    )
    print(f"  reward={result['reward']:.4f}")
    print(f"  breakdown={result['breakdown']}")

    print("\nCase 2: dirty, expensive job that also missed its deadline")
    result = calculate_reward(
        carbon_grams=4800.0, cost=45.0, delay=80.0, sla_violated=True,
        weights=weights, ranges=ranges,
    )
    print(f"  reward={result['reward']:.4f}")
    print(f"  breakdown={result['breakdown']}")

    assert result["reward"] < 0, "reward should be negative for a bad outcome"
    print("\nSanity checks passed.")
