import numpy as np
import pytest

pytest.importorskip("cvxpy")

from near_field_isac.channels import generate_scenario
from near_field_isac.config import SimulationConfig
from near_field_isac.optimization import _solve_options, solve_fully_digital_sdr


def test_mosek_solves_cvxpy_canonicalization_in_dual_form() -> None:
    options = _solve_options("MOSEK", 1.0e-7, 20_000, 2)
    assert options["mosek_params"] == {
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 20_000,
        "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
        "MSK_IPAR_NUM_THREADS": 2,
    }


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
