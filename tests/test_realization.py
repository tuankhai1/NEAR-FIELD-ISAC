import cvxpy as cp
import numpy as np
import pytest

from near_field_isac.channels import generate_scenario, near_field_response
from near_field_isac.config import SimulationConfig
from near_field_isac.fim import crb_covariance, crb_matrix, root_crb
from near_field_isac.optimization import solve_fully_digital_sdr, solve_hybrid_sdr
from near_field_isac.realization import (
    PAPER_REFLECTION_MAGNITUDE,
    paper_realization,
    random_realization,
)


def _scenario(config, draw, **kwargs):
    return generate_scenario(
        config,
        user_ranges=draw.user_ranges,
        user_angles=draw.user_angles,
        target_reflection=draw.target_reflection,
        **kwargs,
    )


def _focusing(config):
    a = near_field_response(config, config.target_range, config.target_angle)
    return config.transmit_power * np.outer(a.conj(), a) / config.n_antennas


def test_paper_realization_is_deterministic_and_figure_specific() -> None:
    config = SimulationConfig.paper()
    first, again, fig4 = (paper_realization(config, f) for f in (2, 2, 4))
    assert np.array_equal(first.user_ranges, again.user_ranges)
    assert np.array_equal(first.receive_combiner, again.receive_combiner)
    assert np.array_equal(first.user_ranges, fig4.user_ranges)
    assert not np.array_equal(first.receive_combiner, fig4.receive_combiner)
    assert np.allclose(np.abs(first.receive_combiner), 1)
    assert np.all(first.user_ranges <= config.rayleigh_distance)


def test_paper_realization_requires_the_paper_model() -> None:
    with pytest.raises(ValueError, match="random"):
        paper_realization(SimulationConfig.smoke(), 2)


def test_random_realization_keeps_the_seeded_scenario_sequence() -> None:
    config = SimulationConfig.smoke()
    draw = random_realization(config)
    expected = generate_scenario(config, np.random.default_rng(config.seed))
    assert np.array_equal(draw.user_ranges, expected.user_ranges)
    assert draw.target_reflection == expected.target_reflection


def test_recovered_reflection_reproduces_the_paper_zero_rate_fd_bound() -> None:
    # Digitized paper Fig. 2 at R_min = 0: 5.1985e-3 m and 3.4515e-5 deg.
    config = SimulationConfig.paper()
    scenario = _scenario(config, paper_realization(config, 2))
    assert abs(scenario.target_reflection) == PAPER_REFLECTION_MAGNITUDE
    rcrb = root_crb(crb_matrix(config, _focusing(config), scenario.target_gain))
    assert np.allclose(rcrb, (5.1985e-3, 3.4515e-5), rtol=2e-4)


def test_zero_rate_optimum_is_target_focusing_after_angle_refinement() -> None:
    config = SimulationConfig.paper()
    scenario = _scenario(config, paper_realization(config, 2))
    result = solve_fully_digital_sdr(config, scenario, min_rate=0.0, solver_threads=2)
    expected = crb_matrix(config, _focusing(config), scenario.target_gain)
    assert np.allclose(np.diag(result.metadata["physical_crb"]), np.diag(expected), rtol=1e-5)


# At 5 bit/s/Hz, CLARABEL stalls or fails depending on the platform (it fails on
# the Linux CI runner at every tolerance), and the angle refinement needs MOSEK.
@pytest.mark.skipif(
    "MOSEK" not in cp.installed_solvers(), reason="needs MOSEK for 5 bit/s/Hz accuracy"
)
def test_dedicated_crb_scales_all_four_curves_by_one_factor() -> None:
    config = SimulationConfig.paper()
    draw = paper_realization(config, 2)
    scenario = _scenario(config, draw)
    ratios = []
    for rate in (0.0, 5.0):
        full = solve_fully_digital_sdr(config, scenario, min_rate=rate, solver_threads=2)
        hybrid = solve_hybrid_sdr(
            config,
            scenario,
            min_rate=rate,
            receive_combiner=draw.receive_combiner,
            solver_threads=2,
        )
        ratios.append(
            [
                *root_crb(crb_matrix(config, crb_covariance(config, full.waveform), 1.0)),
                *root_crb(
                    crb_matrix(
                        config,
                        crb_covariance(config, hybrid.waveform),
                        1.0,
                        receive_combiner=draw.receive_combiner,
                    )
                ),
            ]
        )
    factor = np.array(ratios[1]) / np.array(ratios[0])
    assert factor.min() > 1.005
    assert np.ptp(factor) / factor.mean() < 1e-3


def test_pathloss_free_users_have_unit_element_gain() -> None:
    config = SimulationConfig.paper()
    scenario = _scenario(config, paper_realization(config, 4), user_pathloss=False)
    norms = np.linalg.norm(scenario.communication_channels, axis=0)
    assert np.allclose(norms, np.sqrt(config.n_antennas / config.noise_power))


def test_crb_signal_is_validated() -> None:
    with pytest.raises(ValueError, match="crb_signal"):
        SimulationConfig(crb_signal="echo")
