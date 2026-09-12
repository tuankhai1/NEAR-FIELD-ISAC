"""Search correctness and physical RF/SDR integration checks."""

from dataclasses import replace

import numpy as np
import pytest

from near_field_isac.channels import generate_scenario
from near_field_isac.config import SimulationConfig
from near_field_isac.fim import crb_matrix
from near_field_isac.meta_experiment import aggregate_results
from near_field_isac.metaheuristics import (
    SearchSettings,
    focusing_matrix,
    minimize_population,
    solve_hybrid_metaheuristic,
)
from near_field_isac.optimization import (
    hybrid_analog_beamformer,
    random_hybrid_combiner,
    solve_hybrid_sdr,
)


@pytest.fixture(scope="module")
def hybrid_case():
    pytest.importorskip("cvxpy")
    config = SimulationConfig(n_antennas=9, n_users=1, n_rf_chains=2, min_rate=1, seed=19)
    rng = np.random.default_rng(config.seed)
    scenario = generate_scenario(config, rng)
    combiner = random_hybrid_combiner(config, rng)
    options = dict(solver="auto", tolerance=1e-8, solver_threads=1)
    baseline = solve_hybrid_sdr(config, scenario, receive_combiner=combiner, **options)
    return config, scenario, combiner, options, baseline


@pytest.mark.parametrize("algorithm", ["pso", "de"])
def test_search_improves_shifted_sphere_reproducibly_with_exact_budget(algorithm):
    settings = SearchSettings(population=12, generations=25)
    seen = []

    def objective(x):
        seen.append(x.copy())
        return float(np.sum((x - 0.27) ** 2))

    result = minimize_population(objective, 3, algorithm=algorithm, settings=settings, seed=11)
    repeated = minimize_population(objective, 3, algorithm=algorithm, settings=settings, seed=11)
    assert result.value < 0.003
    assert result.value == repeated.value
    assert np.array_equal(result.position, repeated.position)
    assert len(seen) == 2 * result.evaluations == 2 * 12 * 26
    assert np.max(np.abs(seen)) <= 1
    assert all(
        a["best_objective"] >= b["best_objective"]
        for a, b in zip(result.history, result.history[1:])
    )


def test_algorithms_use_identical_initial_candidates_and_reject_invalid_fitness():
    settings = SearchSettings(population=5, generations=2)
    initial = []
    for algorithm in ("pso", "de"):
        seen = []

        def objective(x):
            seen.append(x.copy())
            return 2.0 if np.all(x == 0) else np.nan

        result = minimize_population(objective, 2, algorithm=algorithm, settings=settings, seed=3)
        assert result.value == 2.0
        assert np.array_equal(result.position, np.zeros(2))
        initial.append(np.asarray(seen[: settings.population]))
    assert np.array_equal(*initial)


def test_virtual_focusing_preserves_baseline_and_physical_scenario():
    config = SimulationConfig.smoke()
    scenario = generate_scenario(config)
    channels = scenario.communication_channels.copy()
    target = scenario.target_channel.copy()
    settings = SearchSettings()
    assert np.array_equal(
        focusing_matrix(config, scenario, np.zeros(6), settings),
        hybrid_analog_beamformer(config, scenario),
    )
    analog = focusing_matrix(config, scenario, np.linspace(-1, 1, 6), settings)
    assert np.allclose(np.abs(analog), 1, rtol=0, atol=1e-14)
    assert np.array_equal(channels, scenario.communication_channels)
    assert np.array_equal(target, scenario.target_channel)
    with pytest.raises(ValueError, match="lie in"):
        focusing_matrix(config, scenario, np.full(6, 1.1), settings)


@pytest.mark.parametrize("algorithm", ["pso", "de"])
def test_metaheuristic_winner_is_physical_and_no_worse_than_original_hb(algorithm, hybrid_case):
    config, scenario, combiner, options, baseline = hybrid_case
    result, details = solve_hybrid_metaheuristic(
        config,
        scenario,
        algorithm=algorithm,
        settings=SearchSettings(population=4, generations=2),
        seed=17,
        receive_combiner=combiner,
        baseline=baseline,
        **options,
    )
    physical = crb_matrix(
        config, result.waveform.covariance, scenario.target_gain, receive_combiner=combiner
    )
    assert np.isclose(np.trace(physical), details["best_objective"], rtol=1e-7)
    assert details["best_objective"] <= baseline.metadata["physical_crb_trace"]
    assert details["fitness_evaluations"] == 12
    assert details["solver_calls"] < 12  # the shared baseline is cached
    assert min(result.rates) >= config.min_rate - 1e-5
    assert np.trace(result.waveform.covariance).real <= config.transmit_power * (1 + 1e-7)
    analog = result.metadata["transmit_basis"]
    assert np.allclose(
        analog,
        focusing_matrix(config, scenario, result.metadata["search_position"], SearchSettings()),
    )
    assert np.allclose(
        analog @ result.metadata["baseband_covariance"] @ analog.conj().T,
        result.waveform.covariance,
        rtol=1e-7,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "mismatch,reason",
    [
        ("receiver", "same receive combiner"),
        ("rf", "nominal focusing"),
        ("target", "sensing scenario"),
        ("channels", "communication constraints"),
        ("rate", "communication constraints"),
        ("power", "transmit power budget"),
        ("model", "near-field sensing model"),
    ],
)
def test_search_rejects_an_unmatched_cached_baseline(hybrid_case, monkeypatch, mismatch, reason):
    config, scenario, combiner, options, baseline = hybrid_case
    options = options.copy()
    if mismatch == "receiver":
        combiner = combiner * np.exp(0.1j)
    elif mismatch == "rf":
        baseline = replace(
            baseline,
            metadata={
                **baseline.metadata,
                "transmit_basis": baseline.metadata["transmit_basis"] * 1j,
            },
        )
    elif mismatch == "target":
        scenario = replace(scenario, target_gain=2 * scenario.target_gain)
    elif mismatch == "channels":
        scenario = replace(scenario, communication_channels=scenario.communication_channels / 2)
    elif mismatch == "rate":
        options["min_rate"] = 0.0
    elif mismatch == "power":
        config = config.with_updates(transmit_power_dbm=config.transmit_power_dbm + 1)
    elif mismatch == "model":
        options["sensing_model"] = "far"

    def unexpected_solve(*args, **kwargs):
        pytest.fail("A mismatched baseline must fail before starting any candidate SDR")

    monkeypatch.setattr("near_field_isac.metaheuristics.solve_hybrid_sdr", unexpected_solve)
    with pytest.raises(ValueError, match=reason):
        solve_hybrid_metaheuristic(
            config,
            scenario,
            algorithm="de",
            settings=SearchSettings(population=4, generations=1),
            seed=17,
            receive_combiner=combiner,
            baseline=baseline,
            **options,
        )


@pytest.mark.parametrize(
    "cost,validated", [(-1.0, True), (np.inf, True), (np.nan, True), (1e-50, False)]
)
def test_search_rejects_invalid_solver_fitness(hybrid_case, monkeypatch, cost, validated):
    config, scenario, combiner, options, baseline = hybrid_case
    invalid = replace(
        baseline,
        metadata={
            **baseline.metadata,
            "physical_crb_trace": cost,
            "validation_passed": validated,
        },
    )
    monkeypatch.setattr("near_field_isac.metaheuristics.solve_hybrid_sdr", lambda *a, **k: invalid)
    result, details = solve_hybrid_metaheuristic(
        config,
        scenario,
        algorithm="de",
        settings=SearchSettings(population=4, generations=1),
        seed=17,
        receive_combiner=combiner,
        baseline=baseline,
        **options,
    )
    assert details["best_objective"] == baseline.metadata["physical_crb_trace"]
    assert details["infeasible_evaluations"] > 0
    assert np.array_equal(result.metadata["search_position"], np.zeros(4))
    assert np.array_equal(result.waveform.covariance, baseline.waveform.covariance)


@pytest.mark.parametrize(
    "updates", [{"population": 4.5}, {"generations": np.nan}, {"generations": True}]
)
def test_search_budgets_require_integers(updates):
    with pytest.raises(ValueError, match="integer"):
        SearchSettings(**updates)


def test_custom_rf_rejects_nonphysical_hardware():
    config = SimulationConfig.smoke()
    scenario = generate_scenario(config)
    with pytest.raises(ValueError, match="unit modulus"):
        solve_hybrid_sdr(config, scenario, analog_beamformer=np.full((17, 3), 2))
    with pytest.raises(ValueError, match="shape"):
        solve_hybrid_sdr(config, scenario, analog_beamformer=np.ones((17, 2)))
    with pytest.raises(ValueError, match="unit modulus"):
        solve_hybrid_sdr(config, scenario, receive_combiner=np.zeros((3, 17)))
    with pytest.raises(ValueError, match="shape"):
        solve_hybrid_sdr(config, scenario, receive_combiner=np.ones((17, 3)))


def test_aggregation_reports_mean_not_best_restart():
    rows = []
    for method in ("fully-digital-sdr", "hybrid-two-stage-sdr", "hybrid-pso-sdr", "hybrid-de-sdr"):
        for value in [1.0, 3.0]:
            rows.append(
                dict(
                    distance_m=20.0,
                    method=method,
                    range_rcrb_m=value,
                    angle_rcrb_deg=value,
                    physical_crb_trace=value,
                    wall_seconds=1,
                    fitness_evaluations=10,
                    solver_calls=9,
                    infeasible_evaluations=0,
                )
            )
    results = aggregate_results(rows)
    assert all(r["range_rcrb_m_mean"] == 2.0 for r in results)
    assert all(np.isclose(r["range_rcrb_m_std"], np.sqrt(2)) for r in results)
