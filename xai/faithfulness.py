"""
xai/faithfulness.py

Quantitative evaluation of the SHAP explanations (roadmap Phase 11 follow-up).

Tests
  1. DELETION faithfulness. For each sampled decision, take the top-k features
     by |SHAP| and replace them with their background mean. If the explanation
     is faithful, the probability of the originally chosen action should drop
     MORE than when k randomly chosen features are removed. We also report how
     often the chosen action changes. (Control: bottom-k features by |SHAP|.)
  2. STABILITY. Re-explain the same states with a different SHAP sample seed and
     background set; report the mean top-3 overlap and Spearman rank correlation
     of |SHAP| between the two explanations.
  3. Category summary: which feature categories the top-3 factors fall into.

Caveat: deletion by mean-replacement can create out-of-distribution states, so
a large drop shows the policy depends on those features, not that the
explanation is "causal" in the environment.

Usage:
    python -m xai.faithfulness --model results/generalization/models/M3_seed1000.zip --variant M3 --n_states 40
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
from scipy import stats
from stable_baselines3 import PPO

from rl.generalization import variant_scenario
from rl.env_factory import build_env
from xai.shap_explainer import PolicyExplainer, collect_background_states, _action_probability_fn
from xai.feature_attribution import categorize_feature


def sample_states(model, scenario, seeds, n_states, rng):
    states = []
    for s in seeds:
        env = build_env(scenario, seed=s)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            states.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(int(action))
            done = term or trunc
    states = np.array(states)
    idx = rng.choice(len(states), size=min(n_states, len(states)), replace=False)
    return states[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="M3",
                    help="M2, M3, M3.5, M3.5-rand (rl.generalization) or any rl.experiment variant, e.g. M3-obs")
    ap.add_argument("--carbon_source", default="synthetic")
    ap.add_argument("--n_states", type=int, default=40)
    ap.add_argument("--nsamples", type=int, default=100)
    ap.add_argument("--out", default="results/xai_faithfulness.json")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    model = PPO.load(args.model)
    from rl.experiment import VARIANTS as EXP_VARIANTS, scenario_for
    if args.variant in EXP_VARIANTS:
        scenario = scenario_for(args.variant, args.carbon_source)
    else:
        scenario = variant_scenario(args.variant, {"carbon_source": args.carbon_source})
    proba = _action_probability_fn(model)

    env = build_env(scenario, seed=100)
    names = env.unwrapped.get_feature_names()

    bg_a = collect_background_states(model, env, n_states=40, seed=200)
    bg_b = collect_background_states(model, env, n_states=40, seed=300)
    bg_mean = bg_a.mean(axis=0)

    states = sample_states(model, scenario, seeds=[100, 101, 102, 103], n_states=args.n_states, rng=rng)
    expl_a = PolicyExplainer(model, bg_a, names, nsamples=args.nsamples)
    expl_b = PolicyExplainer(model, bg_b, names, nsamples=args.nsamples)

    ks = [1, 3, 5]
    drops = {k: {"shap_top": [], "random": [], "shap_bottom": []} for k in ks}
    flips = {k: {"shap_top": [], "random": [], "shap_bottom": []} for k in ks}
    overlaps, spearmans = [], []
    categories = Counter()

    for state in states:
        ra = expl_a.explain(state)
        rb = expl_b.explain(state)
        a_abs, b_abs = np.abs(ra.shap_values), np.abs(rb.shap_values)
        order = np.argsort(-a_abs)

        overlaps.append(len(set(order[:3]) & set(np.argsort(-b_abs)[:3])) / 3.0)
        if a_abs.std() > 0 and b_abs.std() > 0:
            spearmans.append(stats.spearmanr(a_abs, b_abs).correlation)
        for i in order[:3]:
            categories[categorize_feature(names[i])] += 1

        base_p = proba(state.reshape(1, -1))[0]
        action = int(np.argmax(base_p))
        for k in ks:
            sets = {
                "shap_top": [order[:k]],
                "shap_bottom": [order[-k:]],
                "random": [rng.choice(len(names), k, replace=False) for _ in range(20)],
            }
            for kind, idx_sets in sets.items():
                ds, fl = [], []
                for idx in idx_sets:
                    perturbed = state.copy()
                    perturbed[idx] = bg_mean[idx]
                    p = proba(perturbed.reshape(1, -1))[0]
                    ds.append(base_p[action] - p[action])
                    fl.append(float(np.argmax(p) != action))
                drops[k][kind].append(float(np.mean(ds)))
                flips[k][kind].append(float(np.mean(fl)))

    result = {"model": args.model, "variant": args.variant, "n_states": len(states),
              "deletion": {}, "stability": {}, "top3_categories": dict(categories)}
    print(f"\nDeletion test on {len(states)} decisions ({args.variant})")
    print(f"{'k':>2s} {'drop SHAP-top':>14s} {'drop random':>12s} {'drop SHAP-bottom':>17s} | {'flip top':>9s} {'flip rand':>10s} {'flip bottom':>12s} | paired p (top>random)")
    for k in ks:
        t, r, b = (np.array(drops[k][x]) for x in ("shap_top", "random", "shap_bottom"))
        p = stats.wilcoxon(t, r, alternative="greater").pvalue if np.any(t != r) else 1.0
        ft, fr, fb = (np.mean(flips[k][x]) for x in ("shap_top", "random", "shap_bottom"))
        print(f"{k:>2d} {t.mean():>14.3f} {r.mean():>12.3f} {b.mean():>17.3f} | {ft:>9.2f} {fr:>10.2f} {fb:>12.2f} | {p:.4f}")
        result["deletion"][k] = dict(drop_top=float(t.mean()), drop_random=float(r.mean()), drop_bottom=float(b.mean()),
                                     flip_top=float(ft), flip_random=float(fr), flip_bottom=float(fb), p_top_gt_random=float(p))
    result["stability"] = dict(top3_overlap=float(np.mean(overlaps)), spearman=float(np.mean(spearmans)) if spearmans else None)
    print(f"\nStability across SHAP sample/background seeds: top-3 overlap = {np.mean(overlaps):.2f}, mean Spearman = {np.mean(spearmans):.2f}")
    print("Top-3 factor categories:", dict(categories))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
