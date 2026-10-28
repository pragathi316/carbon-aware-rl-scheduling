"""
rl/benchmark.py

Revised benchmark (Protocol 4). Compared with rl/experiment.py:
  * several RL algorithms: MaskablePPO (action masking, sb3-contrib), PPO,
    A2C and DQN (Stable-Baselines3), plus a carbon-blind MaskablePPO;
  * VecNormalize (running observation and reward normalization); the
    statistics are saved next to each model and reused at evaluation;
  * the observation includes GPU demand and free GPUs per region;
  * tuned hyperparameters and training-reward carbon weight per algorithm
    (rl/tune_benchmark.py), evaluated with the FIXED standard reward;
  * three scenarios:
      S1  synthetic carbon + synthetic workload, default regions (as before)
      S2  real-EU carbon (DE/FR/GB) + Alibaba PAI trace workload, GPU-cluster regions
      S3  real-mixed carbon (DE/GB/California) + Alibaba PAI trace, GPU-cluster regions
    real data and trace use a time split (train windows early, test windows late);
  * evaluation on 30 held-out scenarios with and without the SLA shield, with
    behaviour statistics, and heuristics FIFO, Least Carbon, M1, Wait-for-Clean.

Usage:
    python -m rl.benchmark --scenario S2 --algos MPPO PPO A2C DQN MPPO-blind --seeds 10 \
        --config results/benchmark/tuning/best_configs.json --out results/benchmark
    python -m rl.benchmark --scenario S2 --algos MPPO --seeds 1 --steps 4096 --n_test 2 --out /tmp/x  # smoke
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from environment.reward import RewardWeights
from rl.config import RunConfig, ScenarioConfig
from rl.env_factory import build_env
from rl.generalization import RandomizedScenarioEnv

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "S1": dict(),
    "S2": dict(carbon_source="real_eu", workload_source="pai_trace", region_profile="gpu_cluster", trace_interarrival=1.0),
    "S3": dict(carbon_source="real_mixed", workload_source="pai_trace", region_profile="gpu_cluster", trace_interarrival=1.0),
}
ALGOS = {"MPPO": dict(cls="MaskablePPO", blind=False), "MPPO-blind": dict(cls="MaskablePPO", blind=True),
         "PPO": dict(cls="PPO", blind=False), "A2C": dict(cls="A2C", blind=False), "DQN": dict(cls="DQN", blind=False)}
HEURISTICS = ["M0 FIFO", "Least Carbon", "M1 Deadline+Carbon", "Wait-for-Clean"]
TRAIN_POOL = list(range(20))
VALIDATION_SEEDS = list(range(50, 60))
TEST_SEEDS = list(range(100, 130))

DEFAULT_HP = {
    "MaskablePPO": dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, ent_coef=0.01),
    "PPO": dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01),
    "A2C": dict(learning_rate=7e-4, n_steps=5, gamma=0.99, ent_coef=0.0),
    "DQN": dict(learning_rate=1e-4, buffer_size=50_000, learning_starts=1000, batch_size=64, gamma=0.99,
                exploration_fraction=0.1, exploration_final_eps=0.05, target_update_interval=1000, train_freq=4),
}


def scenario(name: str, carbon_aware: bool = True) -> ScenarioConfig:
    s = ScenarioConfig()
    s.include_gpu_state = True
    s.carbon_aware_state = carbon_aware
    for k, v in SCENARIOS[name].items():
        setattr(s, k, v)
    return s


def _algo_class(name: str):
    if name == "MaskablePPO":
        from sb3_contrib import MaskablePPO
        return MaskablePPO
    import stable_baselines3 as sb3
    return getattr(sb3, name)


def train(algo: str, scen: str, steps: int, seed: int, save_path: str, cfg: Optional[Dict[str, Any]] = None):
    """Train one agent; saves <save_path>.zip and <save_path>_vecnorm.pkl."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    cfg = dict(cfg or {})
    alpha = cfg.pop("train_alpha", 0.4)
    spec = ALGOS[algo]
    cls_name = spec["cls"]
    hp = dict(DEFAULT_HP[cls_name])
    hp.update(cfg)
    net_arch = hp.pop("net_arch", [64, 64])
    weights = RewardWeights(alpha=alpha)
    sc = scenario(scen, carbon_aware=not spec["blind"])
    venv = DummyVecEnv([lambda: RandomizedScenarioEnv(sc, TRAIN_POOL, rng_seed=seed, reward_weights=weights)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=hp.get("gamma", 0.99))
    model = _algo_class(cls_name)("MlpPolicy", venv, seed=seed, verbose=0, policy_kwargs=dict(net_arch=list(net_arch)), **hp)
    model.learn(total_timesteps=steps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path + ".zip")
    venv.save(save_path + "_vecnorm.pkl")
    return model, venv


class LoadedAgent:
    def __init__(self, algo: str, path: str, model=None, venv=None):
        import pickle
        self.algo = algo
        self.masked = ALGOS[algo]["cls"] == "MaskablePPO"
        if model is None:
            model = _algo_class(ALGOS[algo]["cls"]).load(path + ".zip")
            with open(path + "_vecnorm.pkl", "rb") as f:
                venv = pickle.load(f)
        self.model, self.vn = model, venv

    def normalize(self, obs):
        return self.vn.normalize_obs(obs)

    def act(self, obs, env) -> int:
        o = self.normalize(obs)
        if self.masked:
            a, _ = self.model.predict(o, deterministic=True, action_masks=env.unwrapped.action_masks())
        else:
            a, _ = self.model.predict(o, deterministic=True)
        return int(a)


def _behaviour(env, counts, invalid, overrides, info, total_reward) -> Dict[str, float]:
    m = dict(info["episode_metrics"])
    m["total_reward"] = total_reward
    n = max(1, sum(counts))
    m.update({f"share_action_{k}": counts[k] / n for k in range(4)})
    m["invalid_actions"] = invalid
    m["shield_overrides"] = overrides
    comp = env.unwrapped.completed_jobs
    e = sum(j["energy_kwh"] for j in comp) or 1.0
    for rid in ("A", "B", "C"):
        m[f"energy_share_{rid}"] = sum(j["energy_kwh"] for j in comp if j["region_id"] == rid) / e
    # Reward components, so that results can be re-scored under other weight
    # vectors without re-running: R = -(a*sum_c + b*sum_k + g*sum_w + d*sum_v) + shaping.
    from environment.reward import NormalizationRanges, RewardWeights
    rng_, w0 = NormalizationRanges(), RewardWeights()
    clip = lambda x, mx: max(0.0, min(1.0, x / mx))
    m["sum_c"] = sum(clip(j["carbon_grams"], rng_.max_carbon_grams) for j in comp)
    m["sum_k"] = sum(clip(j["cost"], rng_.max_cost) for j in comp)
    m["sum_w"] = sum(clip(j["delay"], rng_.max_delay) for j in comp)
    m["sum_v"] = float(sum(1 for j in comp if j["sla_violated"]))
    m["shaping"] = total_reward + (w0.alpha * m["sum_c"] + w0.beta * m["sum_k"] + w0.gamma * m["sum_w"]
                                   + w0.delta * m["sum_v"])
    return m


def run_episode(sc: ScenarioConfig, seed: int, choose: Callable, shield: bool = False) -> Dict[str, float]:
    from rl.shield import shield_action
    env = build_env(sc, seed=seed)
    obs, _ = env.reset(seed=seed)
    done, total, counts, invalid, overrides, info = False, 0.0, [0, 0, 0, 0], 0, 0, {}
    while not done:
        a = choose(obs, env)
        if shield:
            a, o = shield_action(env.unwrapped, a)
            overrides += int(o)
        counts[a] += 1
        obs, r, term, trunc, info = env.step(a)
        invalid += int(bool(info.get("invalid")))
        total += r
        done = term or trunc
    return _behaviour(env, counts, invalid, overrides, info, total)


def heuristic_fns() -> Dict[str, Callable]:
    from baselines.fifo import pick_action as fifo
    from baselines.least_carbon import pick_action as lc
    from baselines.deadline_carbon import pick_action as dc
    from baselines.wait_clean import pick_action as wc
    return {"M0 FIFO": fifo, "Least Carbon": lc, "M1 Deadline+Carbon": dc, "Wait-for-Clean": wc}


def run_heuristics(scen: str, seeds: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    sc = scenario(scen)
    return {n: {str(s): run_episode(sc, s, lambda o, e, fn=fn: int(fn(e.unwrapped))) for s in seeds}
            for n, fn in heuristic_fns().items()}


def evaluate_agent(agent: LoadedAgent, scen: str, seeds: List[int], shield: bool) -> Dict[str, Dict[str, float]]:
    sc = scenario(scen, carbon_aware=not ALGOS[agent.algo]["blind"])
    return {str(s): run_episode(sc, s, agent.act, shield=shield) for s in seeds}


def _worker(task):
    import torch
    torch.set_num_threads(1)
    algo, scen, seed, out = task["algo"], task["scenario"], task["seed"], task["out"]
    raw = os.path.join(out, scen, "raw", f"{algo}_seed{seed}.json")
    if os.path.exists(raw):
        return f"skip {scen} {algo} {seed}"
    path = os.path.join(out, scen, "models", f"{algo}_seed{seed}")
    model, venv = train(algo, scen, task["steps"], seed, path, task["cfg"])
    agent = LoadedAgent(algo, path, model, venv)
    res = {"algo": algo, "scenario": scen, "seed": seed, "steps": task["steps"], "cfg": task["cfg"],
           "heldout": evaluate_agent(agent, scen, task["test_seeds"], False),
           "heldout_shielded": evaluate_agent(agent, scen, task["test_seeds"], True)}
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    with open(raw, "w") as f:
        json.dump(res, f)
    return f"done {scen} {algo} {seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="S2", choices=list(SCENARIOS))
    ap.add_argument("--algos", nargs="+", default=["MPPO", "PPO", "A2C", "DQN", "MPPO-blind"], choices=list(ALGOS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--n_test", type=int, default=30)
    ap.add_argument("--config", default=None, help="JSON {algo: config} from rl.tune_benchmark")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/benchmark")
    a = ap.parse_args()
    cfgs = json.load(open(a.config)) if a.config else {}
    test_seeds = TEST_SEEDS[:a.n_test]
    hpath = os.path.join(a.out, a.scenario, "raw", "heuristics.json")
    if not os.path.exists(hpath):
        os.makedirs(os.path.dirname(hpath), exist_ok=True)
        json.dump(run_heuristics(a.scenario, test_seeds), open(hpath, "w"))
    tasks = [dict(algo=al, scenario=a.scenario, seed=1000 + k, steps=a.steps, out=a.out, test_seeds=test_seeds,
                  cfg=cfgs.get(al, cfgs.get(ALGOS[al]["cls"], {})))
             for k in range(a.seeds) for al in a.algos]
    print(f"{len(tasks)} runs ({a.scenario}), {a.steps} steps", flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        for fut in as_completed([pool.submit(_worker, t) for t in tasks]):
            print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
