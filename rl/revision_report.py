"""
rl/revision_report.py

Analysis for the third review round, from
  results/benchmark/heuristic_tuning/<S>.json   (rl.tune_heuristics)
  results/benchmark/agent_components/<S>.json   (rl.agent_components)
  results/benchmark/<S>/raw/*.json              (rl.benchmark)

Writes results/benchmark/revision_report.txt and revision_report.json with
  1. tuned heuristics (selected on validation) vs shielded MaskablePPO on test,
  2. validation and test scores of heuristics and agents,
  3. results under three reward weightings,
  4. carbon / SLA points for the Pareto plot (pgfplots coordinates),
  5. energy share in Region B per scheduler.

Usage: python -m rl.revision_report
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import stats

D = "results/benchmark"
WEIGHTS = {"carbon-heavy": (1.2, 0.2, 0.2, 0.6), "balanced (default)": (0.4, 0.2, 0.2, 0.6),
           "deadline-heavy": (0.4, 0.2, 0.2, 1.8)}
FIXED = {"M0 FIFO": "FIFO", "Least Carbon": "LC", "M1 Deadline+Carbon": "M1 (default)", "Wait-for-Clean": "WfC (default)"}


def score(m, w):
    a, b, g, d = w
    return -(a * m["sum_c"] + b * m["sum_k"] + g * m["sum_w"] + d * m["sum_v"]) + m["shaping"]


def per_scen(d, key, w=None):
    """dict seed->metrics  ->  array over sorted seeds"""
    ks = sorted(d, key=int)
    return np.array([score(d[k], w) if w else d[k][key] for k in ks])


def agent_matrix(comp, algo, split, key, w=None):
    """seeds x scenarios"""
    return np.array([per_scen(v[split], key, w) for v in comp[algo].values()])


def paired(a, b, rng, n=5000):
    diff = a - b
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    return float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(p)


def select(entries, fam, split, w):
    """best grid point of a family on `split` under weights w; ties -> lowest validation carbon."""
    cand = [e for e in entries if e["family"] == fam]
    key = lambda e: (round(np.mean(per_scen(e[split], None, w)), 9),
                     -np.mean(per_scen(e[split], "total_carbon_grams")))
    return max(cand, key=key)


def main():
    rng = np.random.default_rng(0)
    rep, lines = {}, []
    for S in ("S2", "S3"):
        tun = json.load(open(f"{D}/heuristic_tuning/{S}.json"))
        comp = json.load(open(f"{D}/agent_components/{S}.json"))
        E = tun["entries"]
        fixed = {e["params"]["name"]: e for e in E if e["family"] == "fixed"}
        r = rep[S] = {}
        w0 = WEIGHTS["balanced (default)"]
        mp_sh = agent_matrix(comp, "MPPO", "test_shielded", None, w0).mean(0)

        # 1 + 2: tuned heuristics, validation and test
        rows = {}
        for name, lab in FIXED.items():
            rows[lab] = fixed[name]
        for fam, lab in (("M1", "M1 (tuned)"), ("WfC", "WfC (tuned)"), ("FO", "Forecast-Optimal (tuned)")):
            rows[lab] = select(E, fam, "validation", w0)
        for fam, lab in (("M1", "M1 (best on test)"), ("WfC", "WfC (best on test)"), ("FO", "Forecast-Optimal (best on test)")):
            rows[lab] = select(E, fam, "test", w0)
        tab = {}
        for lab, e in rows.items():
            tr = per_scen(e["test"], None, w0)
            tab[lab] = dict(params=e["params"], val_reward=float(np.mean(per_scen(e["validation"], None, w0))),
                            test_reward=float(tr.mean()), test_carbon=float(np.mean(per_scen(e["test"], "total_carbon_grams"))),
                            test_sla=100 * float(np.mean(per_scen(e["test"], "sla_violation_rate"))),
                            test_delay_share=float(np.mean(per_scen(e["test"], "share_action_3"))),
                            test_energy_B=float(np.mean(per_scen(e["test"], "energy_share_B"))),
                            mppo_shield_minus=paired(mp_sh, tr, rng))
        for algo in ("MPPO", "MPPO-blind"):
            for sh in ("", "_shielded"):
                lab = algo + (" +shield" if sh else "")
                tab[lab] = dict(val_reward=float(agent_matrix(comp, algo, "validation" + sh, None, w0).mean()),
                                test_reward=float(agent_matrix(comp, algo, "test" + sh, None, w0).mean()),
                                test_carbon=float(agent_matrix(comp, algo, "test" + sh, "total_carbon_grams").mean()),
                                test_sla=100 * float(agent_matrix(comp, algo, "test" + sh, "sla_violation_rate").mean()),
                                test_delay_share=float(agent_matrix(comp, algo, "test" + sh, "share_action_3").mean()),
                                test_energy_B=float(agent_matrix(comp, algo, "test" + sh, "energy_share_B").mean()))
        r["tuned"] = tab
        lines.append(f"\n===== {S}: heuristics (default / tuned on validation / best on test) and agents =====")
        lines.append(f"{'scheduler':34s} {'val R':>8s} {'test R':>8s} {'carbon':>8s} {'SLA%':>6s} {'delay':>6s} {'E_B':>5s}  MPPO+shield - this [CI] p   params")
        for lab, t in tab.items():
            extra = ""
            if "mppo_shield_minus" in t:
                m, lo, hi, p = t["mppo_shield_minus"]
                extra = f"  {m:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}  {t['params']}"
            lines.append(f"{lab:34s} {t['val_reward']:8.3f} {t['test_reward']:8.3f} {t['test_carbon']:8.0f} {t['test_sla']:6.2f} "
                         f"{t['test_delay_share']:6.2f} {t['test_energy_B']:5.2f}{extra}")

        # 3: reweighting (heuristic grid re-selected on validation under each weighting)
        r["weights"] = {}
        lines.append(f"\n----- {S}: reward under three weightings (test; heuristics re-selected on validation per weighting) -----")
        for wn, w in WEIGHTS.items():
            row = {}
            for name, lab in FIXED.items():
                row[lab] = per_scen(fixed[name]["test"], None, w)
            for fam, lab in (("M1", "M1 tuned"), ("WfC", "WfC tuned"), ("FO", "FO tuned")):
                e = select(E, fam, "validation", w)
                row[lab] = per_scen(e["test"], None, w)
            for algo in ("MPPO", "MPPO-blind"):
                row[algo] = agent_matrix(comp, algo, "test", None, w).mean(0)
                row[algo + " +shield"] = agent_matrix(comp, algo, "test_shielded", None, w).mean(0)
            heur = [k for k in row if not k.startswith("MPPO")]
            best_h = max(heur, key=lambda k: row[k].mean())
            out = {k: float(v.mean()) for k, v in row.items()}
            out["best_heuristic"] = best_h
            out["mppo_shield_minus_best"] = paired(row["MPPO +shield"], row[best_h], rng)
            r["weights"][wn] = out
            m, lo, hi, p = out["mppo_shield_minus_best"]
            lines.append(f"{wn:20s} " + " ".join(f"{k}={v.mean():.3f}" for k, v in row.items())
                         + f" | best heuristic {best_h}; MPPO+shield - best {m:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")

        # 4: Pareto points
        pts = []
        for e in E:
            pts.append(dict(kind=e["family"] if e["family"] != "fixed" else FIXED[e["params"]["name"]],
                            carbon=float(np.mean(per_scen(e["test"], "total_carbon_grams"))),
                            sla=100 * float(np.mean(per_scen(e["test"], "sla_violation_rate")))))
        for lab in ("MPPO", "MPPO +shield", "MPPO-blind", "MPPO-blind +shield"):
            pts.append(dict(kind=lab, carbon=tab[lab]["test_carbon"], sla=tab[lab]["test_sla"]))
        for f in sorted(glob.glob(f"{D}/{S}/raw/*_seed*.json")):
            j = json.load(open(f))
            if j["algo"] in ("PPO", "A2C", "DQN"):
                for sh in ("heldout", "heldout_shielded"):
                    pts.append(dict(kind=j["algo"] + (" +shield" if "shield" in sh else "") + "#seed",
                                    carbon=float(np.mean([v["total_carbon_grams"] for v in j[sh].values()])),
                                    sla=100 * float(np.mean([v["sla_violation_rate"] for v in j[sh].values()]))))
        agg = {}
        for p_ in pts:
            if p_["kind"].endswith("#seed"):
                agg.setdefault(p_["kind"][:-5], []).append(p_)
        pts = [p_ for p_ in pts if not p_["kind"].endswith("#seed")]
        for k, v in agg.items():
            pts.append(dict(kind=k, carbon=float(np.mean([x["carbon"] for x in v])), sla=float(np.mean([x["sla"] for x in v]))))
        front = [p_ for p_ in pts if not any(q["carbon"] <= p_["carbon"] and q["sla"] <= p_["sla"] and
                                               (q["carbon"] < p_["carbon"] or q["sla"] < p_["sla"]) for q in pts)]
        r["pareto_points"], r["pareto_front"] = pts, sorted(front, key=lambda x: x["carbon"])
        lines.append(f"\n----- {S}: Pareto front (carbon, SLA%) -----")
        for p_ in r["pareto_front"]:
            lines.append(f"  {p_['kind']:22s} {p_['carbon']:8.0f} {p_['sla']:6.2f}")

        # 5: energy share in B for the fixed heuristics and all agents
        eb = {FIXED[n]: float(np.mean(per_scen(fixed[n]["test"], "energy_share_B"))) for n in FIXED}
        for f in sorted(glob.glob(f"{D}/{S}/raw/*_seed*.json")):
            j = json.load(open(f))
            for sh in ("heldout", "heldout_shielded"):
                eb.setdefault(j["algo"] + (" +shield" if "shield" in sh else ""), []) if False else None
        byalgo = {}
        for f in sorted(glob.glob(f"{D}/{S}/raw/*_seed*.json")):
            j = json.load(open(f))
            for sh in ("heldout", "heldout_shielded"):
                byalgo.setdefault(j["algo"] + (" +shield" if "shield" in sh else ""), []).append(
                    np.mean([v["energy_share_B"] for v in j[sh].values()]))
        eb.update({k: float(np.mean(v)) for k, v in byalgo.items()})
        r["energy_B"] = eb
        lines.append(f"\n----- {S}: energy share in Region B -----\n  " + ", ".join(f"{k}={v:.2f}" for k, v in eb.items()))

    json.dump(rep, open(f"{D}/revision_report.json", "w"), indent=1)
    open(f"{D}/revision_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
