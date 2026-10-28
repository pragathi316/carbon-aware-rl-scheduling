"""
xai/feature_attribution.py

Turns raw SHAP values (one number per state feature) into a ranked,
categorized list — the "ranked important features" step of Phase 11's
pipeline:

    feature attribution -> ranked important features -> human-readable explanation

Categorization lets explanation_generator.py phrase a feature naturally
("Region B's carbon intensity is low") without a giant per-feature-name
lookup table — it works for any number of regions or forecast horizon
steps, since those are pattern-matched rather than hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from xai.shap_explainer import ExplanationResult


@dataclass
class FeatureContribution:
    name: str
    category: str
    shap_value: float
    raw_value: float

    @property
    def supports_decision(self) -> bool:
        """True if this feature pushed TOWARD the chosen action (positive
        SHAP value for that action's probability), False if it pushed
        against it."""
        return self.shap_value > 0


def categorize_feature(name: str) -> str:
    """Maps a feature name (from env.get_feature_names()) to a semantic
    category, by pattern rather than an exhaustive per-name table — this
    keeps working regardless of how many regions or forecast steps exist.
    """
    if name == "remaining_deadline":
        return "deadline"
    if name == "queue_length":
        return "queue"
    if name.startswith("job_"):
        return "job"
    if "carbon_forecast" in name:
        return "region_carbon_forecast"
    if "carbon_intensity" in name:
        return "region_carbon_intensity"
    if "available_cpu" in name:
        return "region_cpu"
    if "available_gpu" in name:
        return "region_gpu"
    if name.startswith("context_"):
        return "context"
    if name in ("cheapest_electricity_cost", "mean_region_utilization"):
        return "system"
    return "other"


def rank_features(result: ExplanationResult, top_k: int = 3) -> List[FeatureContribution]:
    """Rank features by the magnitude of their SHAP contribution to the
    chosen action's probability, largest impact first."""
    contributions = [
        FeatureContribution(
            name=name,
            category=categorize_feature(name),
            shap_value=float(shap_val),
            raw_value=float(raw_val),
        )
        for name, shap_val, raw_val in zip(
            result.feature_names, result.shap_values, result.raw_feature_values
        )
    ]
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
    return contributions[:top_k]
