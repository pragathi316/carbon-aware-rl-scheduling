"""
xai/shap_explainer.py

Explains an individual scheduling decision made by a trained PPO policy,
using SHAP (KernelExplainer) to attribute the policy's confidence in the
CHOSEN action back to individual state features (Phase 11 of the roadmap):

    State -> trained policy -> selected action
          -> feature attribution
          -> ranked important features

SHAP treats the policy as a black box: it only needs a function
state -> action probabilities, so this works unchanged regardless of
which state configuration was used (M2/M3/M3.5's differing feature sets),
as long as feature_names (from env.get_feature_names()) matches.

Design notes:
  - KernelExplainer is model-agnostic but needs many perturbed samples per
    explanation, so this is meant for explaining SPECIFIC decisions after
    training/evaluation — not for use inside the training loop.
  - We attribute to the probability of the action the policy actually
    picked (its argmax), which directly answers "why did the policy lean
    toward this action" — the question Phase 11's explanations are for.
  - A background dataset (a sample of realistic states) is required by
    KernelExplainer to represent "what's a typical state" — we sample this
    by rolling out the SAME environment with the SAME policy for a few
    episodes, which is diverse enough for KernelExplainer while realistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np
import shap
import torch

if TYPE_CHECKING:
    from stable_baselines3 import PPO
    from environment.cloud_env import CloudSchedulingEnv

ACTION_NAMES = {0: "Run in Region A", 1: "Run in Region B", 2: "Run in Region C", 3: "Delay"}


@dataclass
class ExplanationResult:
    """Raw output of explaining one decision — explanation_generator.py
    turns this into the human-readable text."""

    state: np.ndarray
    feature_names: List[str]
    action: int
    action_probs: np.ndarray                 # policy's probability for each action
    shap_values: np.ndarray                  # attribution to the CHOSEN action's probability, one per feature
    raw_feature_values: List[float]          # same order as feature_names, for phrasing


def _action_probability_fn(model: "PPO"):
    """Builds a state -> action_probabilities function from an SB3 PPO
    model, for SHAP to query as a black box."""

    def predict_proba(states: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(states, dtype=torch.float32, device=model.device)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()
        return probs

    return predict_proba


def collect_background_states(
    model: "PPO", env: "CloudSchedulingEnv", n_states: int = 50, seed: Optional[int] = None
) -> np.ndarray:
    """Roll out the policy on its own environment to collect a sample of
    realistic states, used as SHAP's background distribution ("what does a
    typical state look like"). Runs multiple episodes if one isn't enough.
    """
    states: List[np.ndarray] = []
    episode_seed = seed if seed is not None else 0

    while len(states) < n_states:
        obs, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        while not (terminated or truncated) and len(states) < n_states:
            states.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
        episode_seed += 1

    return np.array(states[:n_states])


class PolicyExplainer:
    """Explains individual decisions of a trained PPO policy via SHAP."""

    def __init__(
        self,
        model: "PPO",
        background_states: np.ndarray,
        feature_names: List[str],
        nsamples: int = 100,
    ):
        if background_states.shape[1] != len(feature_names):
            raise ValueError(
                f"background_states has {background_states.shape[1]} columns but "
                f"{len(feature_names)} feature_names were given — these must match "
                f"(use env.get_feature_names() from the SAME env configuration)."
            )
        self.feature_names = feature_names
        self.nsamples = nsamples
        self._predict_proba = _action_probability_fn(model)
        self._explainer = shap.KernelExplainer(self._predict_proba, background_states)

    def explain(self, state: np.ndarray, display_values: Optional[np.ndarray] = None) -> ExplanationResult:
        """Explain a single decision: which action the policy picked, and
        which features pushed it toward that action.

        display_values: the UNSCALED feature values to use in the text, for
        policies trained on scaled observations (see rl/env_factory.py's
        ScaledObservation; unscaled = state * env._scales). Defaults to state."""
        state_2d = state.reshape(1, -1)
        action_probs = self._predict_proba(state_2d)[0]
        action = int(np.argmax(action_probs))

        shap_values = self._explainer.shap_values(state_2d, nsamples=self.nsamples, silent=True)
        # shap>=0.45 returns shape (n_samples, n_features, n_outputs);
        # older versions return a list of (n_samples, n_features) arrays,
        # one per output. Handle both so this doesn't break on a version bump.
        if isinstance(shap_values, list):
            action_shap = np.asarray(shap_values[action])[0]
        else:
            shap_values = np.asarray(shap_values)
            action_shap = shap_values[0, :, action]

        return ExplanationResult(
            state=state,
            feature_names=self.feature_names,
            action=action,
            action_probs=action_probs,
            shap_values=action_shap,
            raw_feature_values=list(state if display_values is None else display_values),
        )


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from environment.cloud_env import CloudSchedulingEnv
    from environment.region import create_default_regions
    from environment.workload_generator import generate_jobs
    from environment.carbon_data import generate_carbon_series, CarbonDataProvider

    jobs = generate_jobs(n_jobs=30, intensity="medium", seed=1)
    regions = create_default_regions(3)
    rows = generate_carbon_series(
        [r.region_id for r in regions], n_timesteps=300, horizon=3, seed=1
    )
    provider = CarbonDataProvider(rows)
    env = CloudSchedulingEnv(jobs=jobs, regions=regions, carbon_provider=provider)

    model = PPO.load("results/trained_models/ppo_carbon_aware.zip")
    feature_names = env.get_feature_names()

    print("Collecting background states from policy rollouts...")
    background = collect_background_states(model, env, n_states=40, seed=1)

    explainer = PolicyExplainer(model, background, feature_names, nsamples=100)

    obs, _ = env.reset(seed=1)
    result = explainer.explain(obs)

    print(f"\nAction chosen: {ACTION_NAMES[result.action]}")
    print(f"Action probabilities: {dict(zip(ACTION_NAMES.values(), result.action_probs.round(3)))}")
    ranking = sorted(
        zip(result.feature_names, result.shap_values, result.raw_feature_values),
        key=lambda t: abs(t[1]), reverse=True,
    )
    print("\nTop contributing features:")
    for name, shap_val, raw_val in ranking[:5]:
        print(f"  {name:35s} shap={shap_val:+.4f}  raw_value={raw_val:.2f}")
