import numpy as np

from near_field_isac.channels import generate_scenario
from near_field_isac.config import SimulationConfig
from near_field_isac.fim import (
    crb_matrix,
    far_field_angle_crb,
    fisher_information_blocks,
    root_crb,
)


def test_fim_physical_scaling_and_crb_are_well_formed() -> None:
    config = SimulationConfig.smoke()
    scenario = generate_scenario(config)
    covariance = config.transmit_power / config.n_antennas * np.eye(config.n_antennas)
    physical = fisher_information_blocks(config, covariance, scenario.target_gain)
    scaled = fisher_information_blocks(
        config,
        covariance,
        scenario.target_gain,
        scale=config.optimization_scale,
    )
    expected_ratio = (
        config.coherent_block_length / config.noise_power / config.optimization_scale
    )
    assert np.allclose(physical.j11, scaled.j11 * expected_ratio)
    assert np.allclose(physical.j12, scaled.j12 * expected_ratio)
    assert np.allclose(physical.j22, scaled.j22 * expected_ratio)

    crb = crb_matrix(config, covariance, scenario.target_gain)
    range_rcrb, angle_rcrb = root_crb(crb)
    assert crb.shape == (2, 2)
    assert np.all(np.isfinite(crb))
    assert range_rcrb > 0
    assert angle_rcrb > 0


def test_far_field_angle_crb_is_finite_and_improves_with_power() -> None:
    config = SimulationConfig.smoke()
    scenario = generate_scenario(config)
    identity = np.eye(config.n_antennas, dtype=np.complex128)
    low_power = far_field_angle_crb(
        config, identity, scenario.target_gain
    )
    high_power = far_field_angle_crb(
        config, 2.0 * identity, scenario.target_gain
    )
    assert np.isfinite(low_power)
    assert low_power > 0
    assert np.isclose(high_power, low_power / 2.0)
