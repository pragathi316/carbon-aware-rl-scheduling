"""
llm_layer/context_builder.py

Turns a RawContext (raw numbers) into a ContextFeatures object — a FIXED
schema of structured contextual features, exactly as the roadmap
requires: "force the context layer to produce a fixed schema. Avoid
free-form text being directly fed into the RL policy without
reproducible preprocessing."

Two engines share one interface (BaseContextEngine):

  RuleBasedContextEngine
      Deterministic formulas, no API calls. This is what actually trains
      the RL policy (Phase 10) — an LLM API call at every one of PPO's
      tens of thousands of training steps is not feasible in latency or
      cost. This engine IS the "context engine" the roadmap explicitly
      allows as an alternative to a live LLM ("LLM or context engine").

  AnthropicLLMContextEngine
      A real LLM call (via the `anthropic` package), prompted to return
      ONLY the fixed JSON schema below, with validation and automatic
      fallback to the rule-based engine on any parse/range/network
      failure. Intended for the Phase 10 ablation (RL+Carbon vs
      RL+Carbon+LLM) at evaluation scale, or for generating a cached
      dataset of (raw -> llm context) pairs — not for driving every
      single training step.

Fixed schema (matches the roadmap's example exactly):
    {
      "carbon_trend": "decreasing" | "stable" | "increasing",
      "deadline_flexibility": float in [0, 1],
      "delay_opportunity": float in [0, 1],
      "carbon_risk": float in [0, 1],
    }
"""

from __future__ import annotations

import json
import os
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from llm_layer.feature_extractor import RawContext

VALID_TRENDS = ("decreasing", "stable", "increasing")
CONTEXT_VECTOR_SIZE = 4  # length of ContextFeatures.to_vector()

# Trend classification threshold: relative change in forecast carbon vs
# current carbon, below/above which we call it "decreasing"/"increasing"
# rather than "stable". Kept as a constant so it's tunable per Section 8's
# sensitivity-study spirit.
TREND_THRESHOLD_FRACTION = 0.05

# Normalization reference for carbon_risk: carbon intensity at or above
# this value maps to risk 1.0. Matches the top of this project's synthetic
# carbon ranges (see environment/carbon_data.py's DEFAULT_BASE_INTENSITY).
CARBON_RISK_REFERENCE = 800.0


@dataclass
class ContextFeatures:
    """Fixed-schema structured output of the context layer."""

    carbon_trend: str              # one of VALID_TRENDS
    deadline_flexibility: float    # 0 (no slack) .. 1 (lots of slack)
    delay_opportunity: float       # 0 (don't wait) .. 1 (strong case to wait)
    carbon_risk: float             # 0 (clean) .. 1 (dirty)

    def __post_init__(self) -> None:
        if self.carbon_trend not in VALID_TRENDS:
            raise ValueError(f"carbon_trend must be one of {VALID_TRENDS}, got {self.carbon_trend!r}")
        for name in ("deadline_flexibility", "delay_opportunity", "carbon_risk"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def to_vector(self) -> List[float]:
        """Numeric encoding for appending to the RL state (Phase 10)."""
        trend_encoding = {"decreasing": -1.0, "stable": 0.0, "increasing": 1.0}
        return [
            trend_encoding[self.carbon_trend],
            self.deadline_flexibility,
            self.delay_opportunity,
            self.carbon_risk,
        ]

    def to_dict(self) -> dict:
        return {
            "carbon_trend": self.carbon_trend,
            "deadline_flexibility": self.deadline_flexibility,
            "delay_opportunity": self.delay_opportunity,
            "carbon_risk": self.carbon_risk,
        }


class BaseContextEngine(ABC):
    @abstractmethod
    def build_context(self, raw: RawContext) -> ContextFeatures:
        ...


class RuleBasedContextEngine(BaseContextEngine):
    """Deterministic stand-in for an LLM — same fixed schema, no API calls.
    Used for actual RL training; see module docstring for why.
    """

    def build_context(self, raw: RawContext) -> ContextFeatures:
        carbon_trend = self._classify_trend(raw)
        deadline_flexibility = self._deadline_flexibility(raw)
        delay_opportunity = self._delay_opportunity(carbon_trend, deadline_flexibility)
        carbon_risk = self._carbon_risk(raw)

        return ContextFeatures(
            carbon_trend=carbon_trend,
            deadline_flexibility=deadline_flexibility,
            delay_opportunity=delay_opportunity,
            carbon_risk=carbon_risk,
        )

    @staticmethod
    def _classify_trend(raw: RawContext) -> str:
        if not raw.predicted_carbon or raw.current_carbon <= 0:
            return "stable"
        avg_future = statistics.mean(raw.predicted_carbon)
        relative_change = (avg_future - raw.current_carbon) / raw.current_carbon
        if relative_change <= -TREND_THRESHOLD_FRACTION:
            return "decreasing"
        if relative_change >= TREND_THRESHOLD_FRACTION:
            return "increasing"
        return "stable"

    @staticmethod
    def _deadline_flexibility(raw: RawContext) -> float:
        if raw.execution_time <= 0:
            return 0.0
        slack = raw.remaining_deadline - raw.execution_time
        # 3x execution_time of slack or more counts as "fully flexible" (1.0).
        flexibility = slack / (raw.execution_time * 3.0)
        return max(0.0, min(1.0, flexibility))

    @staticmethod
    def _delay_opportunity(carbon_trend: str, deadline_flexibility: float) -> float:
        trend_bonus = {"decreasing": 1.0, "stable": 0.4, "increasing": 0.0}[carbon_trend]
        return max(0.0, min(1.0, deadline_flexibility * trend_bonus))

    @staticmethod
    def _carbon_risk(raw: RawContext) -> float:
        return max(0.0, min(1.0, raw.current_carbon / CARBON_RISK_REFERENCE))


class AnthropicLLMContextEngine(BaseContextEngine):
    """Real LLM-backed context engine using the Anthropic API.

    Requires the `anthropic` package and an ANTHROPIC_API_KEY environment
    variable. On ANY failure (missing package, missing key, network error,
    malformed/out-of-schema response), falls back to a RuleBasedContextEngine
    instance and logs a one-line warning — the RL training loop should
    never crash because of an LLM hiccup.

    NOTE: not intended to drive every PPO training step (cost/latency) —
    see this module's docstring. Use for Phase 10's evaluation-time
    RL+Carbon+LLM comparison, or to build a cached dataset offline.
    """

    def __init__(
        self,
        model: str = os.environ.get("CONTEXT_LLM_MODEL", "claude-sonnet-4-5"),
        fallback: Optional[BaseContextEngine] = None,
        max_tokens: int = 200,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.fallback = fallback or RuleBasedContextEngine()
        self._client = None
        self._init_error: Optional[str] = None

        try:
            import anthropic  # noqa: F401 — import check only
        except ImportError:
            self._init_error = (
                "the `anthropic` package is not installed (pip install anthropic)"
            )
            return

        if not os.environ.get("ANTHROPIC_API_KEY"):
            self._init_error = "ANTHROPIC_API_KEY environment variable is not set"
            return

        import anthropic
        self._client = anthropic.Anthropic()

    def build_context(self, raw: RawContext) -> ContextFeatures:
        if self._client is None:
            print(f"[AnthropicLLMContextEngine] Falling back to rule-based engine: "
                  f"{self._init_error}")
            return self.fallback.build_context(raw)

        prompt = self._build_prompt(raw)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return self._parse_response(text)
        except Exception as exc:  # noqa: BLE001 — any failure falls back safely
            print(f"[AnthropicLLMContextEngine] LLM call failed ({exc!r}), "
                  f"falling back to rule-based engine.")
            return self.fallback.build_context(raw)

    @staticmethod
    def _build_prompt(raw: RawContext) -> str:
        forecast_str = ", ".join(f"{v:.1f}" for v in raw.predicted_carbon) or "unavailable"
        return (
            "You are a contextual feature extractor for a cloud job scheduler. "
            "Given the facts below, respond with ONLY a JSON object matching this "
            "exact schema, with no other text:\n"
            '{"carbon_trend": "decreasing" | "stable" | "increasing", '
            '"deadline_flexibility": <float 0-1>, '
            '"delay_opportunity": <float 0-1>, '
            '"carbon_risk": <float 0-1>}\n\n'
            f"Current carbon intensity (region {raw.best_region_id}): {raw.current_carbon:.1f} gCO2/kWh\n"
            f"Predicted carbon intensity, next steps: {forecast_str} gCO2/kWh\n"
            f"Job execution time: {raw.execution_time:.1f}\n"
            f"Remaining time until deadline: {raw.remaining_deadline:.1f}\n"
            f"Queue length: {raw.queue_length}\n"
            f"Fraction of regions with sufficient resources right now: {raw.resource_sufficiency:.2f}\n"
        )

    @staticmethod
    def _parse_response(text: str) -> ContextFeatures:
        text = text.strip()
        # Defensive: strip markdown code fences if the model adds them
        # despite instructions not to.
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        return ContextFeatures(
            carbon_trend=data["carbon_trend"],
            deadline_flexibility=float(data["deadline_flexibility"]),
            delay_opportunity=float(data["delay_opportunity"]),
            carbon_risk=float(data["carbon_risk"]),
        )



class RandomContextEngine(BaseContextEngine):
    """Ablation CONTROL: returns schema-valid but information-free context.

    Each call draws a random trend and three uniform values, independent of
    the real state. An agent trained with this engine has the same input
    size as M3.5 but no useful extra information, so comparing M3.5 against
    it separates "the context features help" from "a bigger input changes
    training"."""

    def __init__(self, seed: int = 0):
        import random
        self._rng = random.Random(seed)

    def build_context(self, raw: RawContext) -> ContextFeatures:
        return ContextFeatures(
            carbon_trend=self._rng.choice(list(VALID_TRENDS)),
            deadline_flexibility=self._rng.random(),
            delay_opportunity=self._rng.random(),
            carbon_risk=self._rng.random(),
        )


if __name__ == "__main__":
    from llm_layer.feature_extractor import extract_raw_context
    from environment.cloud_env import CloudSchedulingEnv
    from environment.region import create_default_regions
    from environment.workload_generator import generate_jobs
    from environment.carbon_data import generate_carbon_series, CarbonDataProvider

    jobs = generate_jobs(n_jobs=10, intensity="medium", seed=1)
    regions = create_default_regions(3)
    rows = generate_carbon_series(
        [r.region_id for r in regions], n_timesteps=200, horizon=3, seed=1
    )
    provider = CarbonDataProvider(rows)
    env = CloudSchedulingEnv(jobs=jobs, regions=regions, carbon_provider=provider)
    env.reset(seed=1)

    engine = RuleBasedContextEngine()

    print("=== Rule-based context engine, across several decisions ===")
    terminated = truncated = False
    step = 0
    while not (terminated or truncated) and step < 6:
        raw = extract_raw_context(env)
        context = engine.build_context(raw)
        print(f"\nstep={step} job={raw.job_id} carbon={raw.current_carbon:.0f} "
              f"forecast={[round(v, 0) for v in raw.predicted_carbon]}")
        print(f"  -> {context.to_dict()}")
        print(f"  -> vector: {context.to_vector()}")
        obs, reward, terminated, truncated, info = env.step(0)  # arbitrary action, just to advance
        step += 1

    print("\n=== Anthropic-backed engine (expected to fall back, no API key set here) ===")
    llm_engine = AnthropicLLMContextEngine()
    raw = RawContext(
        job_id=99, current_time=0.0, cpu_requirement=4.0, execution_time=10.0,
        remaining_deadline=40.0, queue_length=2, best_region_id="B",
        current_carbon=600.0, predicted_carbon=[520.0, 480.0, 450.0],
        resource_sufficiency=1.0,
    )
    context = llm_engine.build_context(raw)
    print(f"  -> {context.to_dict()}")
