"""
rl/llm_eval.py

Evaluates the REAL LLM context engine (AnthropicLLMContextEngine) against the
rule-based engine, using an already-trained M3.5 policy.

Why this script exists
    Every M3.5 result so far used the rule-based engine. Nothing about an LLM
    helping can be claimed until the real engine has been run. This script
    needs a real API key; without one the engine silently falls back to the
    rule-based features, so the script REFUSES to run unless the key is set
    (use --allow_fallback only to test the plumbing).

What it measures (per held-out scenario)
    1. Outcome of the policy when its context features come from
         (a) the rule-based engine, (b) the LLM engine.
    2. Agreement between the two engines on the same states (mean absolute
       difference per feature, and trend-label agreement).
    3. How many LLM calls fell back to the rule-based engine (if many, the
       'LLM' result is mostly the rule-based result).

IMPORTANT caveat: the policy was TRAINED on rule-based features. Feeding it
LLM features is a distribution shift, so a worse result does not prove the
LLM is useless, and a better one is not proof it helps in general. The clean
test would also TRAIN with LLM features (expensive: one API call per decision).

Setup (PowerShell):
    pip install anthropic
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    $env:CONTEXT_LLM_MODEL = "<model id you have access to>"   # optional
    python -m rl.llm_eval --model results/generalization/models/M3.5_seed1000.zip --n_scenarios 3

Cost: roughly (60-100 decisions) x n_scenarios API calls.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import PPO

from llm_layer.context_builder import (
    AnthropicLLMContextEngine, RuleBasedContextEngine, RawContext, ContextFeatures,
)
from rl.generalization import variant_scenario
from rl.env_factory import build_env


class RecordingEngine(RuleBasedContextEngine):
    """Rule-based engine that records every (raw context -> features) pair."""

    def __init__(self):
        self.calls = 0

    def build_context(self, raw: RawContext) -> ContextFeatures:
        self.calls += 1
        return super().build_context(raw)


def run_episode(model, scenario, seed: int, engine) -> Dict[str, Any]:
    env = build_env(scenario, seed=seed)
    env.unwrapped.context_engine = engine
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    info: Dict[str, Any] = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        total_reward += reward
        done = term or trunc
    m = dict(info["episode_metrics"])
    m["total_reward"] = total_reward
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained M3.5 model (.zip)")
    ap.add_argument("--n_scenarios", type=int, default=3)
    ap.add_argument("--first_seed", type=int, default=100)
    ap.add_argument("--allow_fallback", action="store_true", help="plumbing test only; NOT a valid LLM result")
    ap.add_argument("--out", default="results/llm_eval.json")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and not args.allow_fallback:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Set it, or pass --allow_fallback to test the "
                         "plumbing (results would then be rule-based, not LLM).")

    model = PPO.load(args.model)
    scenario = variant_scenario("M3.5")
    seeds = list(range(args.first_seed, args.first_seed + args.n_scenarios))

    results: List[Dict[str, Any]] = []
    for s in seeds:
        rule = RecordingEngine()
        rule_metrics = run_episode(model, scenario, s, rule)

        llm = AnthropicLLMContextEngine()
        fallbacks = {"n": 0, "total": 0}
        original_fallback = llm.fallback.build_context
        diffs: List[np.ndarray] = []
        trend_match: List[float] = []
        reference = RuleBasedContextEngine()

        def counting_fallback(raw, _orig=original_fallback):
            fallbacks["n"] += 1
            return _orig(raw)

        llm.fallback.build_context = counting_fallback  # type: ignore[method-assign]
        original_build = llm.build_context

        def recording_build(raw, _orig=original_build):
            fallbacks["total"] += 1
            out = _orig(raw)
            ref = reference.build_context(raw)
            diffs.append(np.abs(np.array(out.to_vector()[1:]) - np.array(ref.to_vector()[1:])))
            trend_match.append(float(out.carbon_trend == ref.carbon_trend))
            return out

        llm.build_context = recording_build  # type: ignore[method-assign]
        llm_metrics = run_episode(model, scenario, s, llm)

        entry = dict(
            scenario_seed=s, rule_based=rule_metrics, llm=llm_metrics,
            llm_calls=fallbacks["total"], llm_fallbacks=fallbacks["n"],
            mean_abs_diff_flex_delay_risk=np.mean(diffs, axis=0).tolist() if diffs else None,
            trend_agreement=float(np.mean(trend_match)) if trend_match else None,
        )
        results.append(entry)
        print(f"scenario {s}: carbon rule={rule_metrics['total_carbon_grams']:.0f} llm={llm_metrics['total_carbon_grams']:.0f} | "
              f"reward rule={rule_metrics['total_reward']:.3f} llm={llm_metrics['total_reward']:.3f} | "
              f"LLM calls={fallbacks['total']} fallbacks={fallbacks['n']} trend agreement={entry['trend_agreement']}")

    total_calls = sum(r["llm_calls"] for r in results)
    total_fb = sum(r["llm_fallbacks"] for r in results)
    print(f"\nTotal LLM calls: {total_calls}, fell back to rule-based: {total_fb}")
    if total_calls and total_fb / total_calls > 0.05:
        print("WARNING: >5% of calls fell back to the rule-based engine; do not describe these as LLM results.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
