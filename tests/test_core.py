"""Unit tests for the simulator, reward, real-data loader, shield, Lagrangian
callback and context engines.  Run from the project root:  python -m pytest -q"""

import types

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.cloud_env import MAX_DELAY_ATTEMPTS
from environment.real_carbon import generate_real_carbon_series, load_hourly_traces, window_start, TRAIN_FRACTION
from environment.reward import RewardWeights, calculate_reward
from llm_layer.context_builder import RandomContextEngine, RuleBasedContextEngine
from rl.config import ScenarioConfig
from rl.env_factory import build_env


def scenario(**kw):
    s = ScenarioConfig()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------- environment
@pytest.mark.parametrize("kw,dim", [
    (dict(carbon_aware_state=False), 9),
    (dict(), 21),
    (dict(use_llm_context=True), 25),
    (dict(carbon_source="real_eu"), 21),
])
def test_observation_sizes_and_gym_api(kw, dim):
    env = build_env(scenario(**kw), seed=3)
    assert env.observation_space.shape == (dim,)
    check_env(env.unwrapped, skip_render_check=True)


def test_reward_worked_examples():
    assert calculate_reward(300, 1.5, 0, False)["reward"] == pytest.approx(-0.030, abs=1e-3)
    assert calculate_reward(4800, 45, 80, True)["reward"] == pytest.approx(-1.324, abs=1e-3)


def test_delay_consumes_time():
    env = build_env(scenario(), seed=5).unwrapped
    env.reset(seed=5)
    job, t0 = env.current_job, env.current_time
    env.step(3)
    # the delayed job may only come back once time has advanced
    while env.current_job is not None and env.current_job.job_id != job.job_id:
        env.step(0)
    assert env.current_time > t0


def _run_until_gpu_job(env):
    from baselines.least_carbon import pick_action
    while env.current_job is not None and env.current_job.gpu_requirement <= 0:
        env.step(pick_action(env))
    return env.current_job


@pytest.mark.parametrize("count_invalid", [True, False])
def test_invalid_actions_count_towards_delay_cap(count_invalid):
    env = build_env(scenario(count_invalid_as_delay=count_invalid), seed=7).unwrapped
    env.reset(seed=7)
    job = _run_until_gpu_job(env)
    assert job is not None, "scenario has no GPU job"
    placed = False
    for _ in range(MAX_DELAY_ATTEMPTS + 3):
        if env.current_job is None or env.current_job.job_id != job.job_id:
            break
        env.step(2)  # region C has no GPUs -> always invalid for this job
        placed = any(j["job_id"] == job.job_id for j in env.completed_jobs)
        if placed:
            break
        while env.current_job is not None and env.current_job.job_id != job.job_id:
            from baselines.least_carbon import pick_action
            env.step(pick_action(env))
    assert placed == count_invalid


def test_heuristics_are_deterministic():
    from rl.generalization import run_heuristic_episode, _heuristic_fns
    fn = _heuristic_fns()["M1 Deadline+Carbon"]
    a = run_heuristic_episode(scenario(), 42, fn)
    b = run_heuristic_episode(scenario(), 42, fn)
    assert a["total_carbon_grams"] == b["total_carbon_grams"]
    assert a["total_carbon_grams"] == pytest.approx(18432, abs=1)


# ------------------------------------------------------------------ real data
def test_real_carbon_schema_and_determinism():
    rows = generate_real_carbon_series(["A", "B", "C"], 50, 3, "real_eu", "train", seed=4)
    assert len(rows) == 150
    assert {"timestamp", "region_id", "carbon_intensity_current", "carbon_intensity_t+3"} <= set(rows[0])
    assert rows == generate_real_carbon_series(["A", "B", "C"], 50, 3, "real_eu", "train", seed=4)
    means = {r: np.mean([x["carbon_intensity_current"] for x in rows if x["region_id"] == r]) for r in "ABC"}
    assert means["B"] < means["A"]  # France cleaner than Germany


def test_real_train_and_test_windows_do_not_overlap():
    n = len(load_hourly_traces()["DE"])
    boundary = int(n * TRAIN_FRACTION)
    for seed in range(50):
        assert window_start(seed, n, 403, "train") + 403 <= boundary
        assert window_start(seed, n, 403, "test") >= boundary


def test_auto_split_uses_test_window_for_seed_100_plus():
    env_tr = build_env(scenario(carbon_source="real_eu"), seed=5).unwrapped
    env_te = build_env(scenario(carbon_source="real_eu"), seed=105).unwrapped
    assert env_tr.carbon_provider.get_current("A", 0.0) != env_te.carbon_provider.get_current("A", 0.0)


# ---------------------------------------------------------------------- shield
def test_shield_overrides_only_risky_delay():
    from rl.shield import shield_action
    env = build_env(scenario(), seed=11).unwrapped
    env.reset(seed=11)
    job = env.current_job
    job.deadline = env.current_time + job.execution_time + 0.5   # almost no slack
    action, overridden = shield_action(env, 3)
    assert overridden and action != 3
    job.deadline = env.current_time + 3 * job.execution_time      # plenty of slack
    assert shield_action(env, 3) == (3, False)


# ------------------------------------------------------------------ Lagrangian
def _fake_callback(weights, infos):
    from rl.constrained import LagrangianSLACallback
    cb = LagrangianSLACallback(weights, epsilon=0.01, eta=5.0)
    cb.locals = {"infos": infos}
    cb.num_timesteps = 512
    cb.model = types.SimpleNamespace(logger=types.SimpleNamespace(record=lambda *a, **k: None))
    cb._on_step()
    cb._on_rollout_end()
    return cb


def test_lagrangian_increases_and_decreases_penalty():
    w = RewardWeights()
    _fake_callback(w, [{"sla_violated": True}] * 10 + [{"sla_violated": False}] * 90)
    assert w.delta == pytest.approx(0.6 + 5 * (0.10 - 0.01))
    w2 = RewardWeights()
    _fake_callback(w2, [{"sla_violated": False}] * 100)
    assert w2.delta == pytest.approx(0.6 - 0.05)
    w3 = RewardWeights(delta=0.01)
    _fake_callback(w3, [{"sla_violated": False}] * 100)
    assert w3.delta == 0.0


# ------------------------------------------------------------------- context
def test_context_engines_return_valid_schema():
    env = build_env(scenario(use_llm_context=True), seed=2).unwrapped
    env.reset(seed=2)
    from llm_layer.feature_extractor import extract_raw_context
    raw = extract_raw_context(env)
    for engine in (RuleBasedContextEngine(), RandomContextEngine(seed=1)):
        vec = engine.build_context(raw).to_vector()
        assert vec[0] in (-1.0, 0.0, 1.0) and all(0.0 <= v <= 1.0 for v in vec[1:])


# ------------------------------------------------------- improved observation
def test_gpu_state_and_scaled_observation():
    env = build_env(scenario(include_gpu_state=True, scale_obs=True), seed=3)
    obs, _ = env.reset(seed=3)
    names = env.unwrapped.get_feature_names()
    assert env.observation_space.shape == (25,) and len(names) == 25
    assert names[-4:] == ["job_gpu_requirement", "region_A_available_gpu", "region_B_available_gpu",
                          "region_C_available_gpu"]
    assert obs.max() < 5.0  # scaled inputs are of order 1
    raw = env.unwrapped._get_state()
    assert raw[names.index("region_A_available_gpu")] == 8.0


# ------------------------------------------------------------ revision tests
def test_action_masks_and_trace_workload():
    from rl.benchmark import scenario
    env = build_env(scenario("S2"), seed=3).unwrapped
    env.reset(seed=3)
    m = env.action_masks()
    assert m.dtype == bool and m.shape == (4,) and m[3]
    job = env.current_job
    for i, r in enumerate(env.regions):
        assert m[i] == r.can_fit(job.cpu_requirement, job.gpu_requirement)
    assert job.job_type == "pai_trace"


def test_trace_train_test_split_and_gpu_cluster():
    from environment.trace_workload import generate_trace_jobs
    tr = generate_trace_jobs(60, seed=5, split="train")
    te = generate_trace_jobs(60, seed=105, split="test")
    assert len(tr) == len(te) == 60
    assert all(j.deadline > j.arrival_time + j.execution_time for j in tr)
    from environment.region import create_gpu_cluster_regions
    assert all(r.total_gpu > 0 for r in create_gpu_cluster_regions())


def test_wait_for_clean_delays_only_when_clean_region_full():
    from baselines.wait_clean import pick_action
    from rl.benchmark import scenario
    env = build_env(scenario("S2"), seed=7).unwrapped
    env.reset(seed=7)
    a = pick_action(env)
    cleanest = min(range(3), key=lambda i: env.regions[i].carbon_intensity)
    if env.regions[cleanest].can_fit(env.current_job.cpu_requirement, env.current_job.gpu_requirement):
        assert a == cleanest


def test_tunable_heuristics_reproduce_defaults():
    from baselines.deadline_carbon import pick_action as m1
    from baselines.wait_clean import pick_action as wfc
    from baselines.tunable import make_m1, make_wfc
    from rl.benchmark import run_episode, scenario
    sc = scenario("S2")
    for ref, new in ((m1, make_m1(0.1, 0.1)), (wfc, make_wfc(0.1, 0.0))):
        a = run_episode(sc, 101, lambda o, e: int(ref(e.unwrapped)))
        b = run_episode(sc, 101, lambda o, e: int(new(e.unwrapped)))
        assert a["total_reward"] == b["total_reward"]


def test_reward_components_recompose_reward():
    from baselines.least_carbon import pick_action
    from rl.benchmark import run_episode, scenario
    m = run_episode(scenario("S2"), 102, lambda o, e: int(pick_action(e.unwrapped)))
    rec = -(0.4 * m["sum_c"] + 0.2 * m["sum_k"] + 0.2 * m["sum_w"] + 0.6 * m["sum_v"]) + m["shaping"]
    assert abs(rec - m["total_reward"]) < 1e-9 and m["shaping"] <= 0


def test_forecast_optimal_and_switch_traces():
    from baselines.tunable import make_fo
    from environment.switch_carbon import generate_switch_series, switch_fraction
    from rl.benchmark import run_episode, scenario
    m = run_episode(scenario("S2"), 103, lambda o, e, f=make_fo(3, 0.5, 0.2): int(f(e.unwrapped)))
    assert m["n_completed"] == m["n_jobs"]
    assert switch_fraction(generate_switch_series(["A", "B", "C"], 400, 3, 0.0, 1)) == 0.0
    assert 0.4 < switch_fraction(generate_switch_series(["A", "B", "C"], 2000, 3, 0.5, 1)) < 0.6


def test_hour_of_day_features():
    from rl.benchmark import scenario
    sc = scenario("S3", carbon_aware=False)
    base = build_env(sc, seed=4).observation_space.shape[0]
    sc.time_features = True
    env = build_env(sc, seed=4)
    obs, _ = env.reset(seed=4)
    assert obs.shape[0] == base + 2 and abs(obs[-1] ** 2 + obs[-2] ** 2 - 1) < 1e-5
