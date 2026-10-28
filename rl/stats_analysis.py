"""
rl/stats_analysis.py

Statistical testing for multi-seed experiments (rl.generalization and
rl.experiment). Input: <dir>/per_seed_summary.csv (one row per model x
training seed; each value is the mean over the evaluation scenarios).
Output: <dir>/stats_report_<split>.txt

Reported per metric (carbon, reward, SLA violation rate):
  * independent comparisons between variants: mean difference, 95 % bootstrap
    CI, Welch t-test, Mann-Whitney U, Cohen's d, Holm-adjusted p;
  * PAIRED comparisons "X +shield" vs "X" (same trained agents): paired t-test
    and Wilcoxon signed-rank;
  * every learned model vs the M1 heuristic (one-sample t-test of seed means
    against M1's value on the same scenarios; M1 is deterministic).

The unit of replication is the training seed, so power is limited:
"not significant" means "not detected". Read the CIs.

Usage:  python -m rl.stats_analysis --dir results/v2/synthetic --split heldout
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

METRICS = ["total_carbon_grams", "total_reward", "sla_violation_rate"]
PAIRS: List[Tuple[str, str]] = [
    ("M3", "M2"), ("M3.5", "M3"), ("M3.5", "M3.5-rand"), ("M3-Lag", "M3"), ("M3-Lag", "M2"),
    ("M3 +shield", "M2 +shield"), ("M3-Lag +shield", "M3 +shield"),
    ("M3-obs", "M3"), ("M3-scale", "M3"), ("M3-gpu", "M3"), ("M3-obs", "M2-obs"),
    ("M3-obs +shield", "M2-obs +shield"), ("M3-a5", "M3"),
]
HEURISTIC = "M1 Deadline+Carbon"


def holm(p: List[float]) -> List[float]:
    order = np.argsort(p)
    out = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out.tolist()


def bootstrap_ci(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def load(path: str, split: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{model: {ppo_seed: {metric: value}}}"""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                out.setdefault(row["model"], {})[row["ppo_seed"]] = {m: float(row[m]) for m in METRICS}
    return out


def arr(data, model, metric):
    return np.array([v[metric] for _, v in sorted(data[model].items())])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/v2/synthetic")
    ap.add_argument("--split", default="heldout", choices=["heldout", "in_distribution"])
    args = ap.parse_args()
    data = load(os.path.join(args.dir, "per_seed_summary.csv"), args.split)
    learned = [m for m in data if m not in ("M0 FIFO", "Least Carbon", HEURISTIC)]
    lines = [f"Statistical report: {args.dir} ({args.split}). Unit = training seed.\n"]

    for metric in METRICS:
        lines.append(f"=== {metric} ===")
        rows = []
        for a, b in PAIRS:
            if a in data and b in data:
                x, y = arr(data, a, metric), arr(data, b, metric)
                lo, hi = bootstrap_ci(x, y)
                rows.append((f"{a} vs {b}", len(x), len(y), x.mean() - y.mean(), lo, hi,
                             stats.ttest_ind(x, y, equal_var=False).pvalue,
                             stats.mannwhitneyu(x, y, alternative="two-sided").pvalue, cohens_d(x, y)))
        if rows:
            adj = holm([r[6] for r in rows])
            lines.append(f"{'independent comparison':30s} {'nA':>3s} {'nB':>3s} {'diff':>10s} {'95% CI':>24s} {'Welch p':>8s} {'MWU p':>7s} {'Holm p':>7s} {'d':>6s}")
            for r, h in zip(rows, adj):
                lines.append(f"{r[0]:30s} {r[1]:>3d} {r[2]:>3d} {r[3]:>10.4f} [{r[4]:>10.4f},{r[5]:>10.4f}] {r[6]:>8.4f} {r[7]:>7.4f} {h:>7.4f} {r[8]:>6.2f}")

        paired = []
        for m in learned:
            if m.endswith(" +shield") and m[:-8] in data:
                base = m[:-8]
                seeds = sorted(set(data[m]) & set(data[base]))
                x = np.array([data[m][s][metric] for s in seeds])
                y = np.array([data[base][s][metric] for s in seeds])
                if len(seeds) > 1 and np.any(x != y):
                    paired.append((f"{m} vs {base}", len(seeds), (x - y).mean(),
                                   stats.ttest_rel(x, y).pvalue, stats.wilcoxon(x, y).pvalue))
        if paired:
            lines.append(f"\n{'paired (same agents)':30s} {'n':>3s} {'mean diff':>12s} {'paired t p':>11s} {'Wilcoxon p':>11s}")
            for r in paired:
                lines.append(f"{r[0]:30s} {r[1]:>3d} {r[2]:>12.4f} {r[3]:>11.4f} {r[4]:>11.4f}")

        if HEURISTIC in data:
            ref = next(iter(data[HEURISTIC].values()))[metric]
            lines.append(f"\nvs {HEURISTIC} (= {ref:.4f}), one-sample t-test of seed means:")
            for m in sorted(learned):
                x = arr(data, m, metric)
                if len(x) > 1:
                    pct = f"({(x.mean() / ref - 1) * 100:+.1f}%)" if metric == "total_carbon_grams" else ""
                    better = int(np.sum(x < ref)) if metric != "total_reward" else int(np.sum(x > ref))
                    lines.append(f"  {m:18s} mean={x.mean():11.4f} std={x.std(ddof=1):9.4f} diff={x.mean() - ref:+11.4f} {pct:9s} "
                                 f"p={stats.ttest_1samp(x, ref).pvalue:.4f}  seeds better than M1: {better}/{len(x)}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.dir, f"stats_report_{args.split}.txt"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
