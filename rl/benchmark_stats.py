"""
rl/benchmark_stats.py

Statistics for the revised benchmark (rl/benchmark.py), following the
recommendations of Agarwal et al. (NeurIPS 2021, "rliable"; procedures re-implemented here):

  * scores are normalized per test scenario as (x - FIFO) / mean(LC - FIFO),
    so FIFO = 0 and Least Carbon averages 1 (M1 can score below FIFO on reward), separately for reward and for carbon
    reduction (a common denominator avoids division by near-zero gaps);
  * aggregate metrics: IQM, mean and median over all (seed x scenario) runs,
    with stratified-bootstrap 95 % confidence intervals;
  * performance profiles (fraction of runs whose normalized score exceeds tau);
  * probability of improvement P(X > Y) between algorithms;
  * comparison against a heuristic: for every algorithm, the per-scenario
    difference (mean over seeds minus the heuristic on the SAME scenario) is
    tested with a two-sided Wilcoxon signed-rank test over the 30 scenarios,
    with a bootstrap CI; all p-values are Holm-adjusted together.

Usage:  python -m rl.benchmark_stats --dir results/benchmark --scenario S2
Writes: <dir>/<scenario>/stats_report.txt, summary.csv, profiles.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import numpy as np
from scipy import stats

ORDER = ["MPPO", "PPO", "A2C", "DQN", "MPPO-blind"]


def holm(p: List[float]) -> List[float]:
    order = np.argsort(p)
    out = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        out[i] = min(1.0, run)
    return out.tolist()


def iqm(x: np.ndarray) -> float:
    return float(stats.trim_mean(x.reshape(-1), 0.25))


def aggregates(x: np.ndarray) -> np.ndarray:
    return np.array([iqm(x), float(x.mean()), float(np.median(x))])


def stratified_bootstrap(x: np.ndarray, reps: int, seed: int = 0):
    """rliable-style stratified bootstrap: within every scenario (column),
    resample the runs (rows) with replacement. Returns point estimates and
    95 % percentile CIs of (IQM, mean, median). Implemented directly because
    the rliable package is incompatible with current versions of its
    `arch` dependency; the procedure follows Agarwal et al. (2021)."""
    rng = np.random.default_rng(seed)
    n, m = x.shape
    boots = np.empty((reps, 3))
    for b in range(reps):
        idx = rng.integers(0, n, size=(n, m))
        boots[b] = aggregates(np.take_along_axis(x, idx, axis=0))
    return aggregates(x), np.percentile(boots, [2.5, 97.5], axis=0)


def prob_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) averaged over scenarios (Mann-Whitney statistic per scenario)."""
    vals = []
    for j in range(x.shape[1]):
        a, b = x[:, j][:, None], y[:, j][None, :]
        vals.append(float(((a > b) + 0.5 * (a == b)).mean()))
    return float(np.mean(vals))


def load(d: str):
    heur = json.load(open(os.path.join(d, "raw", "heuristics.json")))
    agents: Dict[str, Dict[str, Dict[int, Dict[str, dict]]]] = {}
    for f in glob.glob(os.path.join(d, "raw", "*_seed*.json")):
        r = json.load(open(f))
        for split, label in (("heldout", r["algo"]), ("heldout_shielded", r["algo"] + " +shield")):
            agents.setdefault(label, {})[r["seed"]] = r[split]
    return heur, agents


def matrix(per_seed: Dict[int, Dict[str, dict]], scen: List[str], key: str) -> np.ndarray:
    seeds = sorted(per_seed)
    return np.array([[per_seed[s][sc][key] for sc in scen] for s in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/benchmark")
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--reps", type=int, default=5000)
    a = ap.parse_args()

    d = os.path.join(a.dir, a.scenario)
    heur, agents = load(d)
    scen = sorted(heur["M0 FIFO"], key=int)
    H = {h: {k: np.array([heur[h][s][k] for s in scen]) for k in ("total_reward", "total_carbon_grams", "sla_violation_rate", "avg_delay")}
         for h in heur}
    fifo_r, m1_r = H["M0 FIFO"]["total_reward"], H["M1 Deadline+Carbon"]["total_reward"]
    fifo_c, m1_c = H["M0 FIFO"]["total_carbon_grams"], H["M1 Deadline+Carbon"]["total_carbon_grams"]
    # One denominator per scenario set (mean gap between M1 and FIFO over the test
    # scenarios), so that a scenario where M1 and FIFO happen to tie cannot blow
    # up the normalized score. 0 = FIFO on that scenario; M1 averages 1.
    lc_r, lc_c = H["Least Carbon"]["total_reward"], H["Least Carbon"]["total_carbon_grams"]
    den_r = float(np.mean(lc_r - fifo_r))
    den_c = float(np.mean(fifo_c - lc_c))
    labels = [l for base in ORDER for l in (base, base + " +shield") if l in agents]

    norm_r, norm_c, raw = {}, {}, {}
    for l in labels:
        R = matrix(agents[l], scen, "total_reward")
        C = matrix(agents[l], scen, "total_carbon_grams")
        raw[l] = {k: matrix(agents[l], scen, k) for k in ("total_reward", "total_carbon_grams", "sla_violation_rate", "avg_delay",
                                                           "share_action_3", "energy_share_B", "invalid_actions", "shield_overrides")}
        norm_r[l] = (R - fifo_r) / den_r
        norm_c[l] = (fifo_c - C) / den_c
    # heuristics on the same normalized scale (one "seed")
    for h in ("M1 Deadline+Carbon", "Least Carbon", "Wait-for-Clean"):
        norm_r[h] = ((H[h]["total_reward"] - fifo_r) / den_r)[None]
        norm_c[h] = ((fifo_c - H[h]["total_carbon_grams"]) / den_c)[None]

    lines = [f"Benchmark statistics, scenario {a.scenario}: {len(scen)} held-out scenarios.",
             "Normalized score: (x - FIFO on the same scenario) / mean(Least Carbon - FIFO over scenarios); FIFO = 0, Least Carbon averages 1.",
             "IQM / mean / median with stratified-bootstrap 95% CIs over (seed x scenario) runs.\n"]
    summary_rows = []
    for title, N in (("reward", norm_r), ("carbon reduction", norm_c)):
        keys = [k for k in N if k in labels]
        pe, ci = {}, {}
        for k in keys:
            pe[k], ci[k] = stratified_bootstrap(N[k], a.reps)
        lines.append(f"--- normalized {title} ---")
        lines.append(f"{'agent':20s} {'IQM [95% CI]':>26s} {'mean [95% CI]':>26s} {'median':>8s}")
        for k in keys:
            lines.append(f"{k:20s} {pe[k][0]:7.3f} [{ci[k][0,0]:6.3f},{ci[k][1,0]:6.3f}]   {pe[k][1]:7.3f} [{ci[k][0,1]:6.3f},{ci[k][1,1]:6.3f}] {pe[k][2]:8.3f}")
            summary_rows.append(dict(scenario=a.scenario, agent=k, metric=title, iqm=pe[k][0], iqm_lo=ci[k][0, 0], iqm_hi=ci[k][1, 0],
                                     mean=pe[k][1], mean_lo=ci[k][0, 1], mean_hi=ci[k][1, 1], median=pe[k][2]))
        for h in ("M1 Deadline+Carbon", "Least Carbon", "Wait-for-Clean"):
            lines.append(f"{h:20s} mean {float(N[h].mean()):7.3f} (deterministic)")
        lines.append("")

    # raw means
    lines.append("--- raw held-out means (mean over seeds and scenarios; +/- std over seed means) ---")
    lines.append(f"{'agent':20s} {'carbon (g)':>16s} {'SLA %':>8s} {'delay':>7s} {'reward':>16s} {'delay share':>11s} {'E_B':>6s} {'invalid':>8s} {'overrides':>9s}")
    for h in heur:
        lines.append(f"{h:20s} {H[h]['total_carbon_grams'].mean():16.0f} {100*H[h]['sla_violation_rate'].mean():8.2f} "
                     f"{H[h]['avg_delay'].mean():7.2f} {H[h]['total_reward'].mean():16.3f}")
    for l in labels:
        r = raw[l]
        sm = lambda k: r[k].mean(axis=1)
        lines.append(f"{l:20s} {sm('total_carbon_grams').mean():9.0f}+/-{sm('total_carbon_grams').std(ddof=1):5.0f} "
                     f"{100*sm('sla_violation_rate').mean():8.2f} {sm('avg_delay').mean():7.2f} "
                     f"{sm('total_reward').mean():9.3f}+/-{sm('total_reward').std(ddof=1):5.3f} {sm('share_action_3').mean():11.3f} "
                     f"{sm('energy_share_B').mean():6.2f} {sm('invalid_actions').mean():8.2f} {sm('shield_overrides').mean():9.2f}")
    lines.append("")

    # paired per-scenario tests vs heuristics
    tests = []
    for ref in ("M1 Deadline+Carbon", "Least Carbon"):
        for l in labels:
            for key in ("total_reward", "total_carbon_grams"):
                diff = raw[l][key].mean(axis=0) - H[ref][key]   # per scenario, mean over seeds
                p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                rng = np.random.default_rng(0)
                boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
                frac = float(np.mean(diff > 0)) if key == "total_reward" else float(np.mean(diff < 0))
                tests.append([l, ref, key, diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), p, frac])
    adj = holm([t[6] for t in tests])
    lines.append("--- per-scenario paired comparison vs heuristic (agent seed-mean minus heuristic, 30 scenarios) ---")
    lines.append("Wilcoxon signed-rank (two-sided), Holm-adjusted over all rows; 'better' = share of scenarios where the agent is better.")
    lines.append(f"{'agent':20s} {'vs':20s} {'metric':18s} {'mean diff':>11s} {'95% CI':>24s} {'p':>8s} {'p Holm':>8s} {'better':>7s}")
    for t, h in zip(tests, adj):
        lines.append(f"{t[0]:20s} {t[1]:20s} {t[2]:18s} {t[3]:11.3f} [{t[4]:10.3f},{t[5]:10.3f}] {t[6]:8.4f} {h:8.4f} {t[7]:7.2f}")
    lines.append("")

    # paired per-scenario comparisons between agents (seed-means on the same 30 scenarios)
    pairs = [("MPPO", "MPPO-blind"), ("MPPO +shield", "MPPO-blind +shield"), ("MPPO", "PPO"), ("MPPO", "A2C"), ("MPPO", "DQN"),
             ("MPPO +shield", "PPO +shield"), ("MPPO +shield", "A2C +shield"), ("MPPO +shield", "DQN +shield")]
    rows2 = []
    for x, y in pairs:
        if x in raw and y in raw:
            for key in ("total_reward", "total_carbon_grams"):
                diff = raw[x][key].mean(axis=0) - raw[y][key].mean(axis=0)
                p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                rng = np.random.default_rng(1)
                b = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
                rows2.append([x, y, key, diff.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), p])
    if rows2:
        adj2 = holm([r[6] for r in rows2])
        lines.append("--- paired per-scenario comparison between agents (seed-means, 30 scenarios; Wilcoxon, Holm) ---")
        for r, h in zip(rows2, adj2):
            lines.append(f"{r[0]:20s} vs {r[1]:20s} {r[2]:18s} {r[3]:11.3f} [{r[4]:10.3f},{r[5]:10.3f}] p={r[6]:.4f} Holm={h:.4f}")
        lines.append("")

    # probability of improvement between base agents (reward)
    base = [l for l in ORDER if l in norm_r]
    lines.append("--- probability of improvement P(row > column), normalized reward ---")
    lines.append(" " * 14 + "".join(f"{c:>12s}" for c in base))
    for x in base:
        row = []
        for y in base:
            row.append("" if x == y else f"{prob_improvement(norm_r[x], norm_r[y]):.2f}")
        lines.append(f"{x:14s}" + "".join(f"{v:>12s}" for v in row))
    lines.append("")

    # performance profiles
    taus = np.linspace(-1.0, 1.5, 26)
    prof = {l: [float(np.mean(norm_r[l] > t)) for t in taus] for l in labels}
    json.dump({"taus": taus.tolist(), "profiles_reward": prof}, open(os.path.join(d, "profiles.json"), "w"))

    text = "\n".join(lines)
    open(os.path.join(d, "stats_report.txt"), "w").write(text + "\n")
    with open(os.path.join(d, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)
    print(text)


if __name__ == "__main__":
    main()
