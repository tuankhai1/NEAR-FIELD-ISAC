"""Component-capped hybrid search and normalized SDR objective checks."""

from dataclasses import replace

import numpy as np
import pytest

from near_field_isac.channels import generate_scenario
from near_field_isac.config import SimulationConfig
from near_field_isac.metaheuristics import focusing_matrix
from near_field_isac.optimization import random_hybrid_combiner, solve_hybrid_sdr
from near_field_isac.pareto import (
    ParetoSettings,
    improve_population,
    normalized_score,
    phase_focusing_matrix,
    search_dimension,
    solve_pareto_hybrid,
)


def test_phase_parameterization_preserves_nominal_hardware_design() -> None:
    config = SimulationConfig.smoke()
    scenario = generate_scenario(config)
    settings = ParetoSettings(population=4, generations=1, phase_modes=2, polish_evaluations=0)
    position = np.zeros(search_dimension(config, settings))
    analog = phase_focusing_matrix(config, scenario, position, settings)

    assert np.array_equal(
        analog,
        focusing_matrix(
            config,
            scenario,
            np.zeros(2 * config.n_rf_chains),
            settings.focusing_settings(),
        ),
    )
    assert np.allclose(np.abs(analog), 1, rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match="Invalid phase"):
        phase_focusing_matrix(config, scenario, position + 2, settings)


@pytest.mark.parametrize("algorithm", ["pso", "de"])
def test_balanced_search_has_paired_reproducible_budget_and_polish(algorithm: str) -> None:
    settings = ParetoSettings(
        population=4,
        generations=2,
        phase_modes=0,
        polish_evaluations=6,
        restart_patience=1,
    )
    seen = []

    def objective(position: np.ndarray) -> float:
        seen.append(position.copy())
        return float(np.sum((position - 0.2) ** 2))

    result = improve_population(objective, 3, algorithm=algorithm, settings=settings, seed=9)
    repeated = improve_population(
        lambda x: float(np.sum((x - 0.2) ** 2)),
        3,
        algorithm=algorithm,
        settings=settings,
        seed=9,
    )

    assert result["evaluations"] == settings.budget == len(seen)
    assert result["value"] == repeated["value"]
    assert np.array_equal(result["position"], repeated["position"])
    assert np.array_equal(result["initial_population"], repeated["initial_population"])
    assert np.array_equal(seen[0], np.zeros(3))
    assert all(
        earlier["best_objective"] >= later["best_objective"]
        for earlier, later in zip(result["history"], result["history"][1:])
    )
    assert result["history"][-1]["evaluations"] == settings.budget


def test_normalized_score_rejects_incomplete_or_dominated_candidates() -> None:
    reference = np.array([2.0, 4.0])
    assert normalized_score(np.diag([1.0, 2.0]), reference) == 0.5
    with pytest.raises(ValueError, match="exceeds"):
        normalized_score(np.diag([2.000001, 2.0]), reference)
    with pytest.raises(ValueError, match="two finite"):
        normalized_score(np.eye(3), reference)


@pytest.fixture(scope="module")
def pareto_case():
    pytest.importorskip("cvxpy")
    config = SimulationConfig(n_antennas=9, n_users=1, n_rf_chains=2, min_rate=1, seed=23)
    rng = np.random.default_rng(config.seed)
    scenario = generate_scenario(config, rng)
    combiner = random_hybrid_combiner(config, rng)
    options = {"solver": "auto", "tolerance": 1e-7, "solver_threads": 1}
    baseline = solve_hybrid_sdr(config, scenario, receive_combiner=combiner, **options)
    reference = np.diag(baseline.metadata["physical_crb"])
    balanced = solve_hybrid_sdr(
        config,
        scenario,
        receive_combiner=combiner,
        crb_reference=reference,
        method_name="hybrid-balanced-sdr",
        **options,
    )
    return config, scenario, combiner, options, baseline, balanced


def test_normalized_sdr_caps_each_physical_variance(pareto_case) -> None:
    _, _, _, _, baseline, balanced = pareto_case
    reference = np.diag(baseline.metadata["physical_crb"])
    ratios = np.diag(balanced.metadata["physical_crb"]) / reference

    assert np.max(ratios) <= 1 + 1e-7
    assert np.isclose(balanced.objective, np.mean(ratios), rtol=2e-5)
    assert np.isclose(balanced.metadata["normalized_crb_objective"], np.mean(ratios), rtol=2e-5)


@pytest.mark.parametrize("algorithm", ["pso", "de"])
def test_pareto_hybrid_preserves_both_baseline_crb_components(algorithm: str, pareto_case) -> None:
    config, scenario, combiner, options, baseline, balanced = pareto_case
    settings = ParetoSettings(
        population=4,
        generations=1,
        phase_modes=1,
        polish_evaluations=2,
        restart_patience=1,
    )
    result, details = solve_pareto_hybrid(
        config,
        scenario,
        algorithm=algorithm,
        settings=settings,
        seed=7,
        baseline=baseline,
        balanced_baseline=balanced,
        receive_combiner=combiner,
        **options,
    )
    reference = np.diag(baseline.metadata["physical_crb"])
    ratios = np.diag(result.metadata["physical_crb"]) / reference

    assert np.max(ratios) <= 1 + 1e-7
    assert np.isclose(details["best_objective"], np.mean(ratios), rtol=1e-7)
    assert details["fitness_evaluations"] == settings.budget
    assert details["solver_calls"] < settings.budget
    assert details["baseline_objective"] <= 1 + 1e-7
    assert np.allclose(
        result.metadata["transmit_basis"],
        phase_focusing_matrix(config, scenario, result.metadata["search_position"], settings),
        rtol=0,
        atol=1e-10,
    )


def test_pareto_search_rejects_unmatched_cached_baseline(pareto_case, monkeypatch) -> None:
    config, scenario, combiner, options, baseline, balanced = pareto_case
    mismatched = replace(
        balanced,
        metadata={
            **balanced.metadata,
            "receive_combiner": balanced.metadata["receive_combiner"] * 1j,
        },
    )
    monkeypatch.setattr(
        "near_field_isac.pareto.solve_hybrid_sdr",
        lambda *args, **kwargs: pytest.fail("candidate solver must not be called"),
    )
    with pytest.raises(ValueError, match="matching nominal"):
        solve_pareto_hybrid(
            config,
            scenario,
            algorithm="pso",
            settings=ParetoSettings(population=4, generations=1, polish_evaluations=0),
            seed=1,
            baseline=baseline,
            balanced_baseline=mismatched,
            receive_combiner=combiner,
            **options,
        )
