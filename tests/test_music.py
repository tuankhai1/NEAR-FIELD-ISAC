import numpy as np

from near_field_isac.channels import far_field_response, near_field_response
from near_field_isac.config import SimulationConfig
from near_field_isac.music import music_spectrum_xy


def test_near_field_music_localizes_range_and_angle_on_grid() -> None:
    target_x = 8.0
    target_y = 8.0
    target_range = float(np.hypot(target_x, target_y))
    config = SimulationConfig.smoke(
        target_range=target_range,
        target_angle_deg=45.0,
    )
    response = near_field_response(config, config.target_range, config.target_angle)
    covariance = np.outer(response, response.conj()) + 1.0e-8 * np.eye(config.n_antennas)
    axis = np.arange(0.0, 17.0, 1.0)
    result = music_spectrum_xy(config, covariance, axis, axis, model="near")
    assert result.estimated_x == target_x
    assert result.estimated_y == target_y
    assert np.isclose(result.estimated_angle_deg, 45.0)


def test_far_field_music_is_range_ambiguous_along_target_direction() -> None:
    config = SimulationConfig.smoke(target_angle_deg=45.0)
    response = far_field_response(config, config.target_angle)
    covariance = np.outer(response, response.conj()) + 1.0e-8 * np.eye(config.n_antennas)
    axis = np.arange(0.0, 17.0, 1.0)
    result = music_spectrum_xy(config, covariance, axis, axis, model="far")
    diagonal = np.diag(result.spectrum)[1:]
    assert np.max(diagonal) - np.min(diagonal) < 1.0e-10
    assert np.allclose(diagonal, 1.0)
