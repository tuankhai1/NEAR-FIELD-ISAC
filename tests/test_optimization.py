import numpy as np
import pytest

pytest.importorskip("cvxpy")

from near_field_isac.channels import generate_scenario
from near_field_isac.config import SimulationConfig
from near_field_isac.optimization import solve_fully_digital_sdr


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

