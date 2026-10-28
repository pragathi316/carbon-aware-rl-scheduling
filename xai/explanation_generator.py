"""
xai/explanation_generator.py

The final step of Phase 11's pipeline: ranked important features -> a
human-readable explanation, in exactly the format the roadmap specifies:

    Action: DELAY
    Top factors:
    1. Current carbon intensity is high
    2. Carbon forecast is decreasing
    3. Deadline has sufficient flexibility

    Explanation:
    The workload was delayed because current carbon intensity was high,
    cleaner electricity was predicted soon, and the job deadline allowed
    waiting.
"""

from __future__ import annotations

from typing import List

from xai.shap_explainer import ExplanationResult, ACTION_NAMES
from xai.feature_attribution import FeatureContribution, rank_features

# Thresholds purely for turning a raw carbon number into a word like
# "high"/"low" in the explanation text — not used anywhere in scheduling
# logic itself. Roughly centered on this project's default region carbon
# range (see environment/region.py's create_default_regions()).
CARBON_LOW_THRESHOLD = 300.0
CARBON_HIGH_THRESHOLD = 500.0


def _carbon_qualifier(value: float) -> str:
    if value >= CARBON_HIGH_THRESHOLD:
        return "high"
    if value <= CARBON_LOW_THRESHOLD:
        return "low"
    return "moderate"


def _phrase_for_feature(contribution: FeatureContribution) -> str:
    """One short factual clause describing this feature's value —
    deliberately just the fact, not the direction it pushed the decision;
    the Explanation paragraph is where causality gets stated."""
    name, category, raw = contribution.name, contribution.category, contribution.raw_value

    if category == "deadline":
        return f"the job's remaining time before its deadline is {raw:.1f}"

    if category == "queue":
        return f"there are {int(raw)} job(s) waiting in the queue"

    if category == "job":
        label = "CPU requirement" if "cpu" in name else ("GPU requirement" if "gpu" in name else "execution time")
        return f"the job's {label} is {raw:.1f}"

    if category == "region_carbon_intensity":
        region_id = name.split("_")[1]
        return f"Region {region_id}'s current carbon intensity is {_carbon_qualifier(raw)} ({raw:.0f} gCO2/kWh)"

    if category == "region_carbon_forecast":
        parts = name.split("_")
        region_id = parts[1]
        step = name.split("t+")[-1]
        return (
            f"Region {region_id}'s forecasted carbon intensity {step} step(s) ahead "
            f"is {_carbon_qualifier(raw)} ({raw:.0f} gCO2/kWh)"
        )

    if category == "region_cpu":
        region_id = name.split("_")[1]
        return f"Region {region_id} currently has {raw:.1f} CPU units available"

    if category == "region_gpu":
        region_id = name.split("_")[1]
        return f"Region {region_id} currently has {raw:.0f} GPU unit(s) available"

    if category == "context":
        label = name.replace("context_", "").replace("_", " ")
        return f"the context layer's {label} signal is {raw:.2f}"

    if category == "system":
        label = name.replace("_", " ")
        return f"{label} is {raw:.2f}"

    return f"{name.replace('_', ' ')} is {raw:.2f}"


def _slack_clause(result: ExplanationResult) -> str:
    """Remaining slack = time left before the deadline minus execution time.
    Shown for every delay decision, because slack decides whether waiting is
    affordable, even when it is not among the top-ranked features."""
    names = result.feature_names
    try:
        remaining = result.raw_feature_values[names.index("remaining_deadline")]
        exec_time = result.raw_feature_values[names.index("job_execution_time")]
    except ValueError:
        return ""
    return f"the job's remaining slack (time to deadline minus execution time) is {remaining - exec_time:.1f} steps"


def generate_explanation(result: ExplanationResult, top_k: int = 3) -> str:
    """Builds the full formatted explanation text for one decision.

    Wording is deliberately associational ("most associated with"): SHAP
    describes how the policy's output depends on its inputs, not causes in
    the environment."""
    contributions = rank_features(result, top_k=top_k)
    action_name = ACTION_NAMES[result.action]

    lines = [f"Action: {action_name}", "Top factors (by SHAP attribution):"]
    for i, contribution in enumerate(contributions, start=1):
        phrase = _phrase_for_feature(contribution)
        lines.append(f"{i}. {phrase[0].upper()}{phrase[1:]}")

    clauses = [_phrase_for_feature(c) for c in contributions]
    if len(clauses) == 1:
        joined = clauses[0]
    elif len(clauses) == 2:
        joined = f"{clauses[0]} and {clauses[1]}"
    else:
        joined = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"

    lines.append("")
    lines.append("Explanation:")
    lines.append(f"The policy's choice ({action_name[0].lower()}{action_name[1:]}) was most strongly associated "
                 f"with these inputs: {joined}.")
    if result.action == 3:
        slack = _slack_clause(result)
        if slack:
            lines.append(f"Context for this delay: {slack}.")
    return "\n".join(lines)


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from environment.cloud_env import CloudSchedulingEnv
    from environment.region import create_default_regions
    from environment.workload_generator import generate_jobs
    from environment.carbon_data import generate_carbon_series, CarbonDataProvider
    from xai.shap_explainer import PolicyExplainer, collect_background_states

    jobs = generate_jobs(n_jobs=30, intensity="medium", seed=1)
    regions = create_default_regions(3)
    rows = generate_carbon_series(
        [r.region_id for r in regions], n_timesteps=300, horizon=3, seed=1
    )
    provider = CarbonDataProvider(rows)
    env = CloudSchedulingEnv(jobs=jobs, regions=regions, carbon_provider=provider)

    model = PPO.load("results/trained_models/ppo_carbon_aware.zip")
    feature_names = env.get_feature_names()

    print("Collecting background states...")
    background = collect_background_states(model, env, n_states=40, seed=1)
    explainer = PolicyExplainer(model, background, feature_names, nsamples=100)

    obs, _ = env.reset(seed=1)
    terminated = truncated = False
    step = 0
    while not (terminated or truncated) and step < 3:
        result = explainer.explain(obs)
        print(f"\n{'=' * 60}\nDecision #{step + 1}\n{'=' * 60}")
        print(generate_explanation(result, top_k=3))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        step += 1
