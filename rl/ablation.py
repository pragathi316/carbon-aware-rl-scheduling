"""
rl/ablation.py

One-command ablation study. Runs the four agent variants that isolate each
design choice, on randomized training scenarios and held-out test scenarios,
then runs the statistical tests:

    M2         no carbon information            (value of SEEING carbon)
    M3         + current carbon and forecasts   (baseline learned agent)
    M3.5       + rule-based context features    (value of the context layer)
    M3.5-rand  + RANDOM context features        (control: same input size, no information)

Reading the result:
    M3 vs M2          -> does carbon visibility help?
    M3.5 vs M3        -> do context features help?
    M3.5 vs M3.5-rand -> is any gain from the INFORMATION in the features,
                         or just from a larger input?

Usage:
    python -m rl.ablation                       # 200k steps, 5 seeds (long)
    python -m rl.ablation --steps 50000 --seeds 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/generalization")
    args = ap.parse_args()

    subprocess.run(
        [sys.executable, "-m", "rl.generalization", "--steps", str(args.steps), "--seeds", str(args.seeds),
         "--variants", "M3", "M3.5", "M2", "M3.5-rand", "--workers", str(args.workers), "--out", args.out],
        check=True,
    )
    for split in ("heldout", "in_distribution"):
        subprocess.run([sys.executable, "-m", "rl.stats_analysis", "--dir", args.out, "--split", split], check=True)


if __name__ == "__main__":
    main()
