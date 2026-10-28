# Carbon-Aware RL Scheduling Benchmark

Code for the paper **"Does Reinforcement Learning Beat Carbon-Aware Heuristics? An Explainable Benchmark of Cloud Job Scheduling on Real Carbon and Workload Traces"** (Divita Dubey, Pragathi, Ritu Pritika Chandrasekar — VIT Vellore).

The repository contains a three-region cloud scheduling simulator (Gymnasium), five heuristics (FIFO, Least Carbon, Deadline+Carbon, Wait-for-Clean, Forecast-Optimal), RL agents (MaskablePPO, PPO, A2C, DQN) with VecNormalize and an SLA shield, statistics (IQM, stratified bootstrap, paired tests), and SHAP / Integrated-Gradients explanation tests.

## Installation

Python 3.11 is recommended.

```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1     Linux/macOS: source venv/bin/activate
pip install numpy scipy pandas gymnasium stable-baselines3==2.9.2 sb3-contrib==2.9.2 torch shap pytest
python -m pytest -q tests
```

## Data

The data files are not redistributed here. Rebuild them from the public sources:

```bash
# Carbon intensity, 2020 (Wiesner et al., "Let's Wait Awhile", Middleware 2021)
git clone --depth 1 https://github.com/dos-group/lets-wait-awhile
python scripts/prepare_real_carbon.py --src lets-wait-awhile/data

# Alibaba PAI GPU-cluster job trace (Weng et al., NSDI 2022)
git clone --depth 1 https://github.com/alibaba/clusterdata
python scripts/prepare_pai_trace.py --src clusterdata/cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_no_estimate_100K.csv
```

This creates `data/real_carbon/hourly_2020.csv` and `data/workload/pai_jobs.csv`.

## Reproducing the results

| Step | Command |
|---|---|
| RL hyperparameter tuning | `python -m rl.tune_benchmark` |
| Main benchmark (S2, S3; 10 seeds) | `python -m rl.benchmark --scenario S2 --algos MPPO MPPO-blind PPO A2C DQN --seeds 10 --config results/benchmark/tuning/best_configs.json` |
| Statistics | `python -m rl.benchmark_stats --scenario S2` |
| Heuristic threshold tuning | `python -m rl.tune_heuristics --scenario S2` |
| Agent re-evaluation with reward components | `python -m rl.agent_components --scenario S2` |
| Tuned heuristics, reward weightings, Pareto points | `python -m rl.revision_report` |
| Energy-model and deadline sensitivity | `python -m rl.robustness --scenario S2` |
| Hour-of-day agent / 1M-step runs / controlled switching traces | `python -m rl.revision_train --job hod\|long\|switch --workers 4` |
| Analysis of those runs | `python -m rl.laptop_report` |
| Explanation faithfulness | `python -m xai.xai_eval --bench results/benchmark/S2/models/MPPO_seed1000 --algo MPPO --scenario S2 --out results/xai_v3/x.json` |
| Explanations of exploit decisions | `python -m xai.exploit_explanations` |

Replace `S2` by `S3` for the second scenario. All experiments run on CPU; `--workers` sets the number of parallel training runs (use fewer than your CPU cores).

## Repository layout

- `environment/` – simulator, regions, carbon data (real, synthetic, controlled switching), workload trace
- `baselines/` – heuristics; `baselines/tunable.py` holds the tunable versions and grids
- `rl/` – training, evaluation, tuning, statistics, shield
- `xai/` – SHAP, Integrated Gradients, deletion/insertion tests, explanation text
- `scripts/` – data preparation
- `tests/` – unit tests (`python -m pytest -q tests`)
- `CHANGES.md` – change log of the revisions

## Citation

If you use this code, please cite the paper (citation to be added after publication) and the archived release on Zenodo (DOI to be added).

## License

MIT (see `LICENSE`). The carbon and workload datasets are subject to the terms of their original sources.
