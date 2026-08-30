import numpy as np
import pytest

pytest.importorskip("cvxpy")

from near_field_isac.channels import generate_scenario
from near_field_isac.config import SimulationConfig
from near_field_isac.optimization import (
    _solve_options,
    _solver_candidates,
    solve_fully_digital_sdr,
    solve_hybrid_sdr,
)


def test_mosek_limits_iterations_and_threads() -> None:
    options = _solve_options("MOSEK", 1.0e-7, 20_000, 2)
    assert options["mosek_params"] == {
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 20_000,
        "MSK_IPAR_NUM_THREADS": 2,
    }


def test_auto_prefers_clarabel_for_hybrid_sdr() -> None:
    class FakeCvxpy:
        @staticmethod
        def installed_solvers() -> list[str]:
            return ["SCS", "MOSEK", "CLARABEL"]

    assert _solver_candidates(FakeCvxpy, "auto") == ["MOSEK", "CLARABEL", "SCS"]
    assert _solver_candidates(FakeCvxpy, "auto", prefer_clarabel=True) == [
        "CLARABEL",
        "MOSEK",
        "SCS",
    ]


def test_small_sdr_solution_meets_rate_and_power_constraints() -> None:
    config = SimulationConfig(
        n_antennas=9,
        n_users=1,
        n_rf_chains=2,
        min_rate=1.0,
        seed=17,
    )
    scenario = generate_scenario(config)
    result = solve_fully_digital_sdr(
        config,
        scenario,
        solver="auto",
        tolerance=1.0e-6,
        max_iterations=1_000,
    )
    assert result.status in {"optimal", "optimal_inaccurate"}
    assert np.min(result.rates) >= config.min_rate - 2.0e-3
    assert np.real(np.trace(result.waveform.covariance)) <= config.transmit_power * 1.001


def test_hybrid_solver_preserves_unit_modulus_rf_implementation() -> None:
    config = SimulationConfig(
        n_antennas=9,
        n_users=1,
        n_rf_chains=2,
        min_rate=1.0,
        seed=19,
    )
    scenario = generate_scenario(config)
    result = solve_hybrid_sdr(
        config,
        scenario,
        solver="auto",
        tolerance=1.0e-6,
        max_iterations=1_000,
    )
    analog_beamformer = result.metadata["transmit_basis"]
    baseband_covariance = result.metadata["baseband_covariance"]
    baseband_beamformers = result.metadata["baseband_communication_beamformers"]

    assert np.allclose(np.abs(analog_beamformer), 1.0)
    assert np.allclose(
        analog_beamformer @ baseband_covariance @ analog_beamformer.conj().T,
        result.waveform.covariance,
        atol=1.0e-8,
    )
    assert np.allclose(
        analog_beamformer @ baseband_beamformers,
        result.waveform.communication_beamformers,
        atol=1.0e-8,
    )
