import numpy as np

from near_field_isac.channels import (
    far_field_response,
    near_field_response,
    response_derivatives,
)
from near_field_isac.config import SimulationConfig


def test_paper_parameters_match_section_iv() -> None:
    config = SimulationConfig.paper()
    assert config.n_antennas == 65
    assert config.n_users == 4
    assert config.n_rf_chains == 5
    assert np.isclose(config.wavelength, 3e8 / 28e9)
    assert np.isclose(config.rayleigh_distance, 46.666666666666664)
    assert np.isclose(config.transmit_power, 100.0)
    assert np.isclose(config.noise_power, 1.0e-6)


def test_quick_keeps_the_paper_system_dimensions() -> None:
    quick = SimulationConfig.quick()
    paper = SimulationConfig.paper()
    assert quick.n_antennas == paper.n_antennas == 65
    assert quick.n_users == paper.n_users == 4
    assert quick.n_rf_chains == paper.n_rf_chains == 5


def test_smoke_uses_the_small_validation_dimensions() -> None:
    config = SimulationConfig.smoke()
    assert config.n_antennas == 17
    assert config.n_users == 2
    assert config.n_rf_chains == 3


def test_response_has_unit_modulus_and_far_field_limit() -> None:
    config = SimulationConfig.smoke()
    angle = np.deg2rad(37.0)
    near = near_field_response(config, 10_000.0, angle)
    far = far_field_response(config, angle)
    assert np.allclose(np.abs(near), 1.0)
    assert np.max(np.abs(near - far)) < 3.0e-3


def test_analytic_response_derivatives_match_finite_difference() -> None:
    config = SimulationConfig.smoke()
    distance = 20.0
    angle = np.deg2rad(45.0)
    _, derivative_range, derivative_angle = response_derivatives(config, distance, angle)
    range_step = 1.0e-4
    angle_step = 1.0e-6
    numerical_range = (
        near_field_response(config, distance + range_step, angle)
        - near_field_response(config, distance - range_step, angle)
    ) / (2.0 * range_step)
    numerical_angle = (
        near_field_response(config, distance, angle + angle_step)
        - near_field_response(config, distance, angle - angle_step)
    ) / (2.0 * angle_step)
    assert np.allclose(derivative_range, numerical_range, rtol=3e-5, atol=1e-7)
    assert np.allclose(derivative_angle, numerical_angle, rtol=3e-5, atol=1e-7)
