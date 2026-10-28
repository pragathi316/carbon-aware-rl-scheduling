"""
rl/revision_train.py

Training jobs requested in the third review round (meant for a multi-core
machine). Each job trains agents, evaluates them on the 30 test scenarios with
and without the shield (with reward components), and writes one JSON per run
to results/revision/<job>/<scenario>/raw/. Existing JSONs are skipped, so an
interrupted job can simply be restarted.

  --job hod     S3, MaskablePPO-blind + hour-of-day (sin/cos) features, 10 seeds, 200k steps.
  --job long    S2 and S3, MaskablePPO for 1M steps, 3 seeds, network 2x64 and 2x128, with
                learning curves (validation reward every 50k steps, with and without shield).
  --job switch  Controlled carbon traces where the cleanest region changes in 0/15/30/50 % of
                hours (environment/switch_carbon.py), Alibaba trace, GPU-cluster regions;
                MaskablePPO and MaskablePPO-blind, 5 seeds, 200k steps, plus the heuristics.

Usage:  python -m rl.revision_train --job hod --workers 10
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from rl.benchmark import (ALGOS, TEST_SEEDS, VALIDATION_SEEDS, LoadedAgent, heuristic_fns, run_episode, scenario,
                          train)

SWITCH_PROBS = (0.0, 0.15, 0.30, 0.50)
CURVE_EVERY = 50_000
KEEP = ["total_carbon_grams", "total_cost", "sla_violation_rate", "avg_delay", "total_reward", "share_action_3",
        "energy_share_A", "energy_share_B", "energy_share_C", "sum_c", "sum_k", "sum_w", "sum_v", "shaping"]


def apply_variant(sc, variant: str):
    """Scenario modifications for a variant tag (also used when re-loading models)."""
    if variant == "hod":
        sc.time_features = True
    elif variant.startswith("switch_p"):
        sc.carbon_source = "switch"
        sc.switch_prob = float(variant[len("switch_p"):])
    return sc


def base_scenario(scen: str, algo: str, variant: str):
    sc = scenario(scen if scen in ("S1", "S2", "S3") else "S2", carbon_aware=not ALGOS[algo]["blind"])
    return apply_variant(sc, variant)


def _eval(agent, sc, seeds, shield):
    return {str(s): {k: m[k] for k in KEEP} for s in seeds for m in [run_episode(sc, s, agent.act, shield=shield)]}


def _train_with_curve(algo, sc, steps, seed, path, cfg, curve, every=CURVE_EVERY):
    """Like rl.benchmark.train but for an arbitrary ScenarioConfig, optionally logging a validation curve."""
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environment.reward import RewardWeights
    from rl.benchmark import DEFAULT_HP, TRAIN_POOL, _algo_class
    from rl.generalization import RandomizedScenarioEnv
    cfg = dict(cfg or {})
    alpha = cfg.pop("train_alpha", 0.4)
    cls_name = ALGOS[algo]["cls"]
    hp = dict(DEFAULT_HP[cls_name]); hp.update(cfg)
    net_arch = hp.pop("net_arch", [64, 64])
    venv = DummyVecEnv([lambda: RandomizedScenarioEnv(sc, TRAIN_POOL, rng_seed=seed, reward_weights=RewardWeights(alpha=alpha))])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=hp.get("gamma", 0.99))
    model = _algo_class(cls_name)("MlpPolicy", venv, seed=seed, verbose=0, policy_kwargs=dict(net_arch=list(net_arch)), **hp)
    points = []

    class Curve(BaseCallback):
        def _on_step(self):
            if curve and self.num_timesteps % every == 0:
                ag = LoadedAgent(algo, path, self.model, venv)
                row = {"steps": int(self.num_timesteps)}
                for shield in (False, True):
                    ms = [run_episode(sc, s, ag.act, shield=shield) for s in VALIDATION_SEEDS]
                    tag = "_shielded" if shield else ""
                    row["reward" + tag] = float(np.mean([m["total_reward"] for m in ms]))
                    row["carbon" + tag] = float(np.mean([m["total_carbon_grams"] for m in ms]))
                    row["sla" + tag] = float(np.mean([m["sla_violation_rate"] for m in ms]))
                points.append(row)
            return True

    model.learn(total_timesteps=steps, callback=Curve())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path + ".zip")
    venv.save(path + "_vecnorm.pkl")
    return model, venv, points


def _worker(t):
    import torch
    torch.set_num_threads(1)
    raw = os.path.join(t["out"], t["job"], t["scen"], "raw", f"{t['tag']}_seed{t['seed']}.json")
    if os.path.exists(raw):
        return f"skip {raw}"
    sc = base_scenario(t["scen"], t["algo"], t["variant"])
    path = os.path.join(t["out"], t["job"], t["scen"], "models", f"{t['tag']}_seed{t['seed']}")
    model, venv, curve = _train_with_curve(t["algo"], sc, t["steps"], t["seed"], path, t["cfg"], t["curve"],
                                           t.get("curve_every", CURVE_EVERY))
    ag = LoadedAgent(t["algo"], path, model, venv)
    res = dict(job=t["job"], scenario=t["scen"], algo=t["algo"], variant=t["variant"], tag=t["tag"], seed=t["seed"],
               steps=t["steps"], cfg=t["cfg"], curve=curve,
               heldout=_eval(ag, sc, TEST_SEEDS, False), heldout_shielded=_eval(ag, sc, TEST_SEEDS, True),
               validation=_eval(ag, sc, VALIDATION_SEEDS, False),
               validation_shielded=_eval(ag, sc, VALIDATION_SEEDS, True))
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    json.dump(res, open(raw, "w"))
    return f"done {raw}"


def _heuristics(job, scen, variant, out):
    path = os.path.join(out, job, scen, "raw", "heuristics.json")
    if os.path.exists(path):
        return
    from baselines.tunable import make_fo
    fns = dict(heuristic_fns()); fns["Forecast-Optimal"] = make_fo(3, 0.1, 0.05)
    sc = apply_variant(scenario("S2"), variant)
    res = {n: {str(s): {k: m[k] for k in KEEP} for s in TEST_SEEDS
               for m in [run_episode(sc, s, lambda o, e, fn=fn: int(fn(e.unwrapped)))]} for n, fn in fns.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"))


def tasks_for(job, cfgs, out):
    T = []
    mp = cfgs.get("MPPO", {}); mb = cfgs.get("MPPO-blind", mp)
    if job == "hod":
        for k in range(10):
            T.append(dict(job=job, scen="S3", algo="MPPO-blind", variant="hod", tag="MPPO-blind-hod", seed=1000 + k,
                          steps=200_000, cfg=dict(mb), curve=False))
    elif job == "long":
        for scen in ("S2", "S3"):
            for arch in (64, 128):
                for k in range(3):
                    c = dict(mp); c["net_arch"] = [arch, arch]
                    T.append(dict(job=job, scen=scen, algo="MPPO", variant="", tag=f"MPPO-1M-{arch}", seed=1000 + k,
                                  steps=1_000_000, cfg=c, curve=True))
    elif job == "switch":
        for p in SWITCH_PROBS:
            for algo, c in (("MPPO", mp), ("MPPO-blind", mb)):
                for k in range(5):
                    T.append(dict(job=job, scen=f"p{p:.2f}", algo=algo, variant=f"switch_p{p}", tag=algo,
                                  seed=1000 + k, steps=200_000, cfg=dict(c), curve=False))
    for t in T:
        t["out"] = out
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, choices=["hod", "long", "switch"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="results/benchmark/tuning/best_configs.json")
    ap.add_argument("--out", default="results/revision")
    ap.add_argument("--steps", type=int, default=None, help="override steps (smoke test)")
    ap.add_argument("--curve_every", type=int, default=CURVE_EVERY)
    ap.add_argument("--limit", type=int, default=None, help="only the first N runs (smoke test)")
    a = ap.parse_args()
    cfgs = json.load(open(a.config)) if os.path.exists(a.config) else {}
    T = tasks_for(a.job, cfgs, a.out)
    if a.limit:
        T = T[:a.limit]
    for t in T:
        t["curve_every"] = a.curve_every
    if a.steps:
        for t in T:
            t["steps"] = a.steps
    if a.job == "switch":
        for p in SWITCH_PROBS:
            _heuristics(a.job, f"p{p:.2f}", f"switch_p{p}", a.out)
        print("heuristics done", flush=True)
    print(f"{len(T)} runs for job {a.job}", flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        for fut in as_completed([pool.submit(_worker, t) for t in T]):
            print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
