# Changes in this version

## Bug fixes / simulator
- `environment/cloud_env.py`: invalid placements (choosing a full region) now count towards
  `MAX_DELAY_ATTEMPTS` (`count_invalid_as_delay=True`, default). Before, an agent could wait
  forever for a preferred region; trained agents exploited this. `False` reproduces the old
  behaviour (Protocols 1-2). Heuristic results are unchanged.
- `environment/cloud_env.py`: optional `include_gpu_state` adds job GPU demand + free GPUs per
  region to the observation (the original state had no GPU information).

## New features
- `environment/real_carbon.py`, `scripts/prepare_real_carbon.py`, `data/real_carbon/hourly_2020.csv`:
  real 2020 hourly carbon intensity (DE, FR, GB, CAL) from the Let's Wait Awhile dataset
  (Wiesner et al., Middleware 2021). `carbon_source = real_eu | real_mixed`; train windows Jan-Aug,
  test windows Sep-Dec.
- `rl/env_factory.py`: `ScaledObservation` wrapper (`scale_obs=True`), `reward_weights` argument.
- `rl/constrained.py`: Lagrangian (dual-ascent) SLA penalty callback -> variant M3-Lag.
- `rl/shield.py`: run-time SLA shield (overrides deadline-risky delays and invalid placements).
- `rl/tune_ppo.py`: hyperparameter sweep on validation scenarios -> `results/tuning/best_config.json`.
- `rl/experiment.py`: final experiment runner (variants M2, M3, M3.5, M3-Lag, M3-a5, M3-obs,
  M3-scale, M3-gpu, M2-obs; shielded + unshielded evaluation; behaviour statistics).
- `rl/generalization.py`: behaviour statistics and shield option in `run_policy_episode`.
- `rl/stats_analysis.py`: paired shield tests, more comparisons, SLA metric.
- `xai/`: GPU feature categories/phrasing, unscaled display values for scaled observations,
  faithfulness test accepts the new variants.
- `tests/test_core.py`: 16 unit tests (`python -m pytest -q tests`).

## Commands for the final protocol
    python -m rl.tune_ppo
    python -m rl.experiment --carbon_source synthetic --variants M2 M3 M3.5 M3-Lag M3-a5 M3-obs M3-scale M3-gpu M2-obs --seeds 10 --ppo_config results/tuning/best_config.json --out results/v2/synthetic
    python -m rl.experiment --carbon_source real_eu --variants M2 M3 M3-Lag M3-obs M2-obs --seeds 10 --ppo_config results/tuning/best_config.json --out results/v2/real_eu
    python -m rl.stats_analysis --dir results/v2/synthetic
    python -m rl.stats_analysis --dir results/v2/real_eu
    python -m xai.faithfulness --model results/v2/synthetic/models/M3-obs_seed1000.zip --variant M3-obs


# Revision 2 (reviewer response)
- `environment/trace_workload.py`, `scripts/prepare_pai_trace.py`, `data/workload/pai_jobs.csv`:
  trace-driven workload from the Alibaba PAI GPU-cluster trace (NSDI 2022), temporal train/test split.
- `environment/region.py`: GPU-cluster region profile; `environment/workload_generator.py`: flexible-slack preset.
- `environment/cloud_env.py`: `action_masks()` for MaskablePPO.
- `baselines/wait_clean.py`: Wait-for-Clean heuristic (waits for capacity in the clean region).
- `rl/benchmark.py`: MaskablePPO / PPO / A2C / DQN / MaskablePPO-blind with VecNormalize, scenarios S1-S3,
  shielded and unshielded evaluation, behaviour statistics.
- `rl/tune_benchmark.py`: per-algorithm hyperparameter + training-reward-weight sweep on validation scenarios.
- `rl/benchmark_stats.py`: IQM / mean / median with stratified bootstrap CIs, performance profiles,
  probability of improvement, paired per-scenario Wilcoxon tests vs heuristics and between agents, Holm.
- `xai/xai_eval.py`: deletion + insertion tests, Integrated Gradients vs SHAP, confidence / sensitivity analysis.
- `xai/explanation_generator.py`: associational wording; deadline slack reported for every delay.
- Commands:
    python -m rl.tune_benchmark
    python -m rl.benchmark --scenario S2 --algos MPPO MPPO-blind --seeds 10 --config results/benchmark/tuning/best_configs.json
    python -m rl.benchmark --scenario S3 --algos MPPO MPPO-blind --seeds 10 --config results/benchmark/tuning/best_configs.json
    python -m rl.benchmark --scenario S2 --algos PPO A2C DQN --seeds 5 --config results/benchmark/tuning/best_configs.json
    python -m rl.benchmark_stats --scenario S2   (and S3)
    python -m xai.xai_eval --bench results/benchmark/S2/models/MPPO_seed1000 --algo MPPO --scenario S2 --out results/xai_v3/x.json
  Extra dependency: pip install sb3-contrib

## Revision 3 (faculty review)

* `baselines/tunable.py`: parameterized M1 and Wait-for-Clean, new Forecast-Optimal heuristic, tuning grids.
* `rl/tune_heuristics.py`: heuristic thresholds tuned on validation seeds 50-59 (same as RL); all grid points also scored on test for the Pareto plot.
* `rl/benchmark.py`: episode results now include reward components (sum_c, sum_k, sum_w, sum_v, shaping) for re-weighting.
* `rl/agent_components.py`: MaskablePPO(-blind) re-evaluated on validation and test with components.
* `rl/revision_report.py`: tuned-heuristic table, validation vs test, three reward weightings, Pareto points, energy share in B.
* `xai/exploit_explanations.py`: SHAP on exploit decisions of original-simulator agents (finding: 99-100 % of invalid Region-B choices were GPU-bound; explanations look like those of valid decisions).
* `environment/switch_carbon.py`, `rl/config.py` (time_features, switch_prob), `rl/env_factory.py` (HourOfDayObservation, switch carbon source), `rl/revision_train.py` (laptop jobs: hod, long, switch).
* Tests: 23 pass.

Commands:
    python -m rl.tune_heuristics --scenario S2 ; python -m rl.tune_heuristics --scenario S3
    python -m rl.agent_components --scenario S2 ; python -m rl.agent_components --scenario S3
    python -m rl.revision_report
    python -m xai.exploit_explanations
    python -m rl.revision_train --job hod|long|switch --workers 10
