"""Near-field and far-field ULA channel models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Scenario:
    """One deterministic realization of users and a sensing target."""

    communication_channels: ComplexArray
    target_channel: ComplexArray
    target_gain: complex
    target_reflection: complex
    user_ranges: FloatArray
    user_angles: FloatArray
    target_range: float
    target_angle: float


def antenna_positions(config: SimulationConfig) -> FloatArray:
    """Return ULA element positions centered at the origin."""

    half = (config.n_antennas - 1) // 2
    return np.arange(-half, half + 1, dtype=float) * config.antenna_spacing


def near_field_response(config: SimulationConfig, distance: float, angle: float) -> ComplexArray:
    """Compute the exact spherical-wave response in paper Eq. (4)."""

    if distance < 0:
        raise ValueError("distance must be non-negative")
    positions = antenna_positions(config)
    element_ranges = np.sqrt(
        np.maximum(
            distance**2
            + positions**2
            - 2.0 * distance * positions * np.cos(angle),
            0.0,
        )
    )
    relative_ranges = element_ranges - distance
    return np.exp(-1j * 2.0 * np.pi / config.wavelength * relative_ranges)


def far_field_response(config: SimulationConfig, angle: float) -> ComplexArray:
    """Compute the planar-wave response in paper Eq. (5)."""

    positions = antenna_positions(config)
    return np.exp(1j * 2.0 * np.pi / config.wavelength * positions * np.cos(angle))


def response_derivatives(
    config: SimulationConfig, distance: float, angle: float
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """Return ``a``, ``da/dr``, and ``da/dtheta`` analytically."""

    if distance <= 0:
        raise ValueError("distance must be positive when computing derivatives")
    positions = antenna_positions(config)
    element_ranges = np.sqrt(
        distance**2
        + positions**2
        - 2.0 * distance * positions * np.cos(angle)
    )
    response = near_field_response(config, distance, angle)
    wave_number = 2.0 * np.pi / config.wavelength
    derivative_range = (
        -1j
        * wave_number
        * ((distance - positions * np.cos(angle)) / element_ranges - 1.0)
        * response
    )
    derivative_angle = (
        -1j
        * wave_number
        * (distance * positions * np.sin(angle) / element_ranges)
        * response
    )
    return response, derivative_range, derivative_angle


def target_response_matrices(
    config: SimulationConfig, distance: float, angle: float
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """Return unscaled round-trip response and its two derivatives."""

    response, derivative_range, derivative_angle = response_derivatives(
        config, distance, angle
    )
    target = np.outer(response, response)
    target_range_derivative = np.outer(derivative_range, response) + np.outer(
        response, derivative_range
    )
    target_angle_derivative = np.outer(derivative_angle, response) + np.outer(
        response, derivative_angle
    )
    return target, target_range_derivative, target_angle_derivative


def generate_scenario(
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
    *,
    user_ranges: FloatArray | None = None,
    user_angles: FloatArray | None = None,
    target_reflection: complex | None = None,
) -> Scenario:
    """Generate channels with the conventions of the public MATLAB code.

    User locations are uniform over range ``[0, Rayleigh distance]`` and angle
    ``[0, pi]``.  This follows ``generate_channel.m`` exactly; note that a few
    realizations can therefore fall below the Fresnel lower bound stated in the
    paper.  A tiny range floor only prevents division by zero.
    """

    rng = np.random.default_rng(config.seed) if rng is None else rng
    if user_ranges is None:
        user_ranges = rng.random(config.n_users) * config.rayleigh_distance
    if user_angles is None:
        user_angles = rng.random(config.n_users) * np.pi
    user_ranges = np.asarray(user_ranges, dtype=float)
    user_angles = np.asarray(user_angles, dtype=float)
    if user_ranges.shape != (config.n_users,) or user_angles.shape != (config.n_users,):
        raise ValueError("user_ranges and user_angles must have shape (n_users,)")

    safe_ranges = np.maximum(user_ranges, np.finfo(float).eps)
    channels = np.empty((config.n_antennas, config.n_users), dtype=np.complex128)
    for user in range(config.n_users):
        gain = (
            np.sqrt(1.0 / config.noise_power)
            * np.sqrt(config.reference_path_gain)
            / safe_ranges[user]
            * np.exp(-1j * 2.0 * np.pi / config.wavelength * safe_ranges[user])
        )
        channels[:, user] = gain * near_field_response(
            config, safe_ranges[user], user_angles[user]
        )

    if target_reflection is None:
        target_reflection = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
    target_gain = (
        np.sqrt(config.reference_path_gain)
        / (2.0 * config.target_range)
        * np.exp(-1j * 4.0 * np.pi / config.wavelength * config.target_range)
        * target_reflection
    )
    target_response = near_field_response(config, config.target_range, config.target_angle)
    target_channel = target_gain * np.outer(target_response, target_response)

    return Scenario(
        communication_channels=channels,
        target_channel=target_channel,
        target_gain=complex(target_gain),
        target_reflection=complex(target_reflection),
        user_ranges=user_ranges,
        user_angles=user_angles,
        target_range=config.target_range,
        target_angle=config.target_angle,
    )

