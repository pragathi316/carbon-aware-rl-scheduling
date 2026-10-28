"""
rl/constrained.py

Constrained ("Lagrangian") PPO for deadlines.

Instead of a fixed SLA weight delta, the deadline-violation penalty is a
Lagrange multiplier lambda that is adapted during training by dual ascent so
that the violation rate is driven towards a target budget epsilon:

    reward_j = -(alpha*C_j + beta*K_j + gamma*W_j) - lambda * V_j
    after every PPO rollout:
        rate    = violations / placements in that rollout
        lambda <- clip(lambda + eta * (rate - epsilon), 0, lambda_max)

This is the standard primal-dual approach to constrained MDPs (Altman 1999),
applied with PPO as the primal learner. Only the TRAINING reward changes;
evaluation always reports the fixed-weight reward (delta = 0.6) so results
stay comparable with the other variants.

Defaults (fixed a priori, not tuned): epsilon = 0.005 (0.5 % of jobs),
eta = 5.0, lambda_0 = 0.6 (= the fixed delta), lambda_max = 10.
"""

from __future__ import annotations

from typing import List

from stable_baselines3.common.callbacks import BaseCallback


class LagrangianSLACallback(BaseCallback):
    def __init__(self, reward_weights, epsilon: float = 0.005, eta: float = 5.0,
                 lambda_max: float = 10.0, verbose: int = 0):
        super().__init__(verbose)
        self.weights = reward_weights          # shared RewardWeights object used by the training env
        self.epsilon = epsilon
        self.eta = eta
        self.lambda_max = lambda_max
        self._violations = 0
        self._placements = 0
        self.history: List[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "sla_violated" in info:
                self._placements += 1
                self._violations += int(bool(info["sla_violated"]))
        return True

    def _on_rollout_end(self) -> None:
        if self._placements == 0:
            return
        rate = self._violations / self._placements
        new_lambda = self.weights.delta + self.eta * (rate - self.epsilon)
        self.weights.delta = float(min(self.lambda_max, max(0.0, new_lambda)))
        self.history.append({"timesteps": int(self.num_timesteps), "violation_rate": rate,
                             "lambda": self.weights.delta})
        self.logger.record("constraint/violation_rate", rate)
        self.logger.record("constraint/lambda", self.weights.delta)
        self._violations = 0
        self._placements = 0
