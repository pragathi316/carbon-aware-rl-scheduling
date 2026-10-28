"""
xai/xai_eval.py

Extended explanation evaluation (revision). For a trained agent:

  1. Deletion test   : replace the top-k features by the background mean and
                       measure the drop in probability of the chosen action
                       (vs. k random features and the bottom-k features).
  2. Insertion test  : start from the background mean and insert the top-k
                       features of the real state; measure how much of the
                       chosen action's probability is recovered (vs. random-k).
  3. Second method   : Integrated Gradients (Sundararajan et al., 2017), with
                       the background mean as baseline; agreement with SHAP is
                       measured by the Spearman correlation of |attributions|
                       and the top-3 overlap.
  4. Confidence      : mean probability of the chosen action and policy
                       entropy, to explain why deletion effects differ
                       between agents (a near-deterministic policy barely
                       reacts to removing a few features).

Works for agents from rl/benchmark.py (MaskablePPO/PPO/A2C with VecNormalize;
inputs are explained in the NORMALIZED space the network sees, with the
action mask of the explained state held fixed) and for older PPO models
(rl/generalization.py, rl/experiment.py) via --legacy_variant.

Usage:
    python -m xai.xai_eval --bench results/benchmark/S2/models/MPPO_seed1000 --algo MPPO --scenario S2
    python -m xai.xai_eval --legacy results/generalization/models/M3_seed1000.zip --legacy_variant M3
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import shap
import torch
from scipy import stats

from xai.feature_attribution import categorize_feature


class PolicyFn:
    """state(s) in network input space -> probabilities over the 4 actions."""

    def __init__(self, model, mask=None):
        self.model = model
        self.mask = mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        pol = self.model.policy
        dist = pol.get_distribution(x)
        logits = dist.distribution.logits
        if self.mask is not None:
            logits = logits.masked_fill(~torch.as_tensor(self.mask, dtype=torch.bool), -1e8)
        return logits

    def proba_t(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=-1)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.proba_t(torch.as_tensor(states, dtype=torch.float32)).numpy()


def integrated_gradients(fn: PolicyFn, x: np.ndarray, baseline: np.ndarray, action: int, steps: int = 64) -> np.ndarray:
    alphas = torch.linspace(0, 1, steps).unsqueeze(1)
    xb = torch.as_tensor(baseline, dtype=torch.float32)
    xx = torch.as_tensor(x, dtype=torch.float32)
    path = (xb + alphas * (xx - xb)).requires_grad_(True)
    out = fn.proba_t(path)[:, action].sum()
    grads = torch.autograd.grad(out, path)[0]
    return ((xx - xb) * grads.mean(dim=0)).detach().numpy()


def collect_states(choose, env_builder, seeds, n, rng):
    states, raws, masks = [], [], []
    for s in seeds:
        env = env_builder(s)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            x, raw = choose.transform(obs), obs.copy()
            states.append(x)
            raws.append(raw)
            masks.append(env.unwrapped.action_masks())
            a = choose(obs, env)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
    idx = rng.choice(len(states), size=min(n, len(states)), replace=False)
    return np.array(states)[idx], np.array(raws)[idx], np.array(masks)[idx], np.array(states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", help="benchmark model path prefix (without .zip)")
    ap.add_argument("--algo", default="MPPO")
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--legacy", help="older PPO .zip model")
    ap.add_argument("--legacy_variant", default="M3")
    ap.add_argument("--n_states", type=int, default=40)
    ap.add_argument("--nsamples", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    if a.bench:
        from rl.benchmark import LoadedAgent, scenario, ALGOS
        from rl.env_factory import build_env
        agent = LoadedAgent(a.algo, a.bench)
        model = agent.model
        sc = scenario(a.scenario, carbon_aware=not ALGOS[a.algo]["blind"])

        class Chooser:
            def transform(self, obs): return agent.normalize(obs)
            def __call__(self, obs, env): return agent.act(obs, env)
        choose = Chooser()
        builder = lambda s: build_env(sc, seed=s)
        use_mask = ALGOS[a.algo]["cls"] == "MaskablePPO"
    else:
        from stable_baselines3 import PPO
        from rl.generalization import variant_scenario
        from rl.env_factory import build_env
        model = PPO.load(a.legacy)
        sc = variant_scenario(a.legacy_variant, {"count_invalid_as_delay": False})

        class Chooser:
            def transform(self, obs): return obs
            def __call__(self, obs, env): return int(model.predict(obs, deterministic=True)[0])
        choose = Chooser()
        builder = lambda s: build_env(sc, seed=s)
        use_mask = False

    names = builder(100).unwrapped.get_feature_names()
    states, raws, masks, pool = collect_states(choose, builder, [100, 101, 102, 103], a.n_states, rng)
    _, _, _, bg_pool_a = collect_states(choose, builder, [200, 201], 40, rng)
    background = bg_pool_a[rng.choice(len(bg_pool_a), size=min(40, len(bg_pool_a)), replace=False)]
    bg_mean = background.mean(axis=0)

    ks = [1, 3, 5]
    res = {k: {"del_top": [], "del_rand": [], "del_bottom": [], "ins_top": [], "ins_rand": []} for k in ks}
    conf, ent, sp_ig, ov_ig, cats = [], [], [], [], Counter()
    top3_mass, sum_abs, grad_norm = [], [], []
    for x, raw, m in zip(states, raws, masks):
        fn = PolicyFn(model, m if use_mask else None)
        p = fn(x[None])[0]
        act = int(np.argmax(p))
        conf.append(p[act])
        ent.append(float(-(p * np.log(p + 1e-12)).sum()))
        explainer = shap.KernelExplainer(lambda z: fn(z)[:, act], background)
        sv = np.asarray(explainer.shap_values(x[None], nsamples=a.nsamples, silent=True)).reshape(-1)
        ig = integrated_gradients(fn, x, bg_mean, act)
        order = np.argsort(-np.abs(sv))
        tot = np.abs(sv).sum()
        sum_abs.append(float(tot))
        top3_mass.append(float(np.abs(sv)[order[:3]].sum() / tot) if tot > 0 else 0.0)
        xt = torch.tensor(x[None], dtype=torch.float32, requires_grad=True)
        g = torch.autograd.grad(fn.proba_t(xt)[0, act], xt)[0]
        grad_norm.append(float(g.norm()))
        for i in order[:3]:
            cats[categorize_feature(names[i])] += 1
        if np.abs(sv).std() > 0 and np.abs(ig).std() > 0:
            sp_ig.append(stats.spearmanr(np.abs(sv), np.abs(ig)).correlation)
        ov_ig.append(len(set(order[:3]) & set(np.argsort(-np.abs(ig))[:3])) / 3)
        p_bg = fn(bg_mean[None])[0][act]
        for k in ks:
            def deleted(idx):
                z = x.copy(); z[idx] = bg_mean[idx]; return fn(z[None])[0][act]

            def inserted(idx):
                z = bg_mean.copy(); z[idx] = x[idx]; return fn(z[None])[0][act]
            rand = [rng.choice(len(x), k, replace=False) for _ in range(20)]
            res[k]["del_top"].append(p[act] - deleted(order[:k]))
            res[k]["del_bottom"].append(p[act] - deleted(order[-k:]))
            res[k]["del_rand"].append(np.mean([p[act] - deleted(r) for r in rand]))
            res[k]["ins_top"].append(inserted(order[:k]) - p_bg)
            res[k]["ins_rand"].append(np.mean([inserted(r) - p_bg for r in rand]))

    out = {"model": a.bench or a.legacy, "algo": a.algo if a.bench else a.legacy_variant, "n_states": len(states),
           "mean_confidence": float(np.mean(conf)), "mean_entropy": float(np.mean(ent)),
           "ig_vs_shap_spearman": float(np.mean(sp_ig)) if sp_ig else None, "ig_vs_shap_top3_overlap": float(np.mean(ov_ig)),
           "top3_categories": dict(cats), "shap_top3_mass": float(np.mean(top3_mass)),
           "shap_sum_abs": float(np.mean(sum_abs)), "grad_norm": float(np.mean(grad_norm)), "tests": {}}
    for k in ks:
        r = {kk: np.array(v) for kk, v in res[k].items()}
        out["tests"][k] = {kk: float(v.mean()) for kk, v in r.items()}
        out["tests"][k]["p_del_top_gt_rand"] = float(stats.wilcoxon(r["del_top"], r["del_rand"], alternative="greater").pvalue) if np.any(r["del_top"] != r["del_rand"]) else 1.0
        out["tests"][k]["p_ins_top_gt_rand"] = float(stats.wilcoxon(r["ins_top"], r["ins_rand"], alternative="greater").pvalue) if np.any(r["ins_top"] != r["ins_rand"]) else 1.0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
