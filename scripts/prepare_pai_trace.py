"""
scripts/prepare_pai_trace.py

Builds data/workload/pai_jobs.csv from the public Alibaba PAI GPU-cluster
trace (cluster-trace-gpu-v2020; Weng et al., "MLaaS in the Wild", NSDI 2022;
https://github.com/alibaba/clusterdata). Input file:
    cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_no_estimate_100K.csv
    columns: job_id, num_inst, submit_time (s), num_cpu, num_gpu, gpu_type, duration (s), user

Processing (documented assumptions, needed to fit an hourly simulator):
  * keep jobs that run at least 1 hour; cap execution time at 20 hours;
  * CPU demand = num_cpu * num_inst, clipped to [1, 32] simulator CPU units;
  * GPU demand = num_gpu * num_inst, rounded up to whole units, clipped to 4;
  * keep submit_time for arrival ordering and gaps.
The trace has no deadlines; the simulator adds them from a slack factor.

Usage:
    git clone --depth 1 https://github.com/alibaba/clusterdata
    python scripts/prepare_pai_trace.py --src clusterdata/cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_no_estimate_100K.csv
"""

from __future__ import annotations

import argparse
import math
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/workload/pai_jobs.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.src)
    d = d[d.duration >= 3600].copy()
    d["exec_hours"] = (d.duration / 3600.0).clip(upper=20.0).round(3)
    d["cpu"] = (d.num_cpu * d.num_inst).clip(lower=1, upper=32).round(2)
    d["gpu"] = (d.num_gpu * d.num_inst).apply(lambda g: min(4, math.ceil(g)) if g > 0 else 0)
    d = d.sort_values("submit_time")[["submit_time", "cpu", "gpu", "exec_hours", "gpu_type"]]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(d)} jobs")
    print(d.describe().round(2))
    print("share with GPU:", round((d.gpu > 0).mean(), 3))


if __name__ == "__main__":
    main()
