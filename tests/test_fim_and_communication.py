import numpy as np

from near_field_isac.channels import generate_scenario
from near_field_isac.communication import communication_rates, zf_sensing_baseline
from near_field_isac.config import SimulationConfig
from near_field_isac.fim import crb_matrix, fisher_information_blocks, root_crb


def test_zf_baseline_meets_rate_and_power_constraints() -> None:
    config = SimulationConfig.quick()
    scenario = generate_scenario(config)
    waveform = zf_sensing_baseline(config, scenario.communication_channels)
    rates = communication_rates(
        scenario.communication_channels,
        waveform.covariance,
        waveform.communication_beamformers,
    )
    assert np.min(rates) >= config.min_rate - 1.0e-9
    assert np.real(np.trace(waveform.covariance)) <= config.transmit_power * (1 + 1e-10)
    assert np.min(np.linalg.eigvalsh(waveform.sensing_covariance)) >= -1.0e-9


def test_fim_physical_scaling_and_crb_are_well_formed() -> None:
    config = SimulationConfig.quick()
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

