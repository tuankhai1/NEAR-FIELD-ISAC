"""Communication-rate metrics and a lightweight feasible baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .channels import near_field_response
from .config import SimulationConfig

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Waveform:
    """Transmit covariance and recovered user beamformers."""

    covariance: ComplexArray
    communication_beamformers: ComplexArray
    sensing_covariance: ComplexArray
    method: str


def communication_rates(
    channels: ComplexArray,
    covariance: ComplexArray,
    beamformers: ComplexArray,
    *,
    noise_power: float = 1.0,
) -> FloatArray:
    """Calculate Eq. (11) using the paper's transpose channel convention."""

    n_users = channels.shape[1]
    rates = np.empty(n_users, dtype=float)
    for user in range(n_users):
        channel = channels[:, user]
        beamformer = beamformers[:, user]
        signal = abs(channel.T @ beamformer) ** 2
        total_received = float(np.real(channel.T @ covariance @ channel.conj()))
        interference_noise = max(total_received - signal + noise_power, np.finfo(float).eps)
        rates[user] = np.log2(1.0 + signal / interference_noise)
    return rates


def zf_sensing_baseline(
    config: SimulationConfig,
    channels: ComplexArray,
    *,
    min_rate: float | None = None,
    target_range: float | None = None,
    target_angle: float | None = None,
) -> Waveform:
    """Construct a fast feasible ZF communication + focused-sensing waveform.

    The communication directions zero-force inter-user interference.  Power is
    allocated analytically so every user meets the requested SINR after the
    dedicated sensing signal is counted as interference.  All remaining power
    is focused on the target.  This is a useful baseline, not an algorithm from
    the paper and not a replacement for the SDR optimum.
    """

    min_rate = config.min_rate if min_rate is None else min_rate
    target_range = config.target_range if target_range is None else target_range
    target_angle = config.target_angle if target_angle is None else target_angle
    if channels.shape != (config.n_antennas, config.n_users):
        raise ValueError("channels has an incompatible shape")

    raw_directions = np.linalg.pinv(channels.T)
    norms = np.linalg.norm(raw_directions, axis=0)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("communication channel matrix is rank deficient")
    directions = raw_directions / norms
    effective = channels.T @ directions
    gains = np.abs(np.diag(effective)) ** 2

    target_response = near_field_response(config, target_range, target_angle)
    sensing_direction = np.conj(target_response) / np.linalg.norm(target_response)
    sensing_leakage = np.abs(channels.T @ sensing_direction) ** 2

    sinr_target = 2.0**min_rate - 1.0
    communication_at_zero_sensing = sinr_target / gains
    leakage_slopes = sinr_target * sensing_leakage / gains
    required_without_sensing = float(np.sum(communication_at_zero_sensing))
    if required_without_sensing > config.transmit_power * (1.0 + 1.0e-10):
        raise ValueError(
            f"ZF baseline is infeasible: needs {required_without_sensing:.3g} mW "
            f"before sensing, available {config.transmit_power:.3g} mW"
        )

    sensing_power = max(
        0.0,
        (config.transmit_power - required_without_sensing)
        / (1.0 + float(np.sum(leakage_slopes))),
    )
    user_powers = communication_at_zero_sensing + leakage_slopes * sensing_power
    beamformers = directions * np.sqrt(user_powers)[None, :]
    sensing_covariance = sensing_power * np.outer(
        sensing_direction, sensing_direction.conj()
    )
    covariance = beamformers @ beamformers.conj().T + sensing_covariance
    return Waveform(
        covariance=covariance,
        communication_beamformers=beamformers,
        sensing_covariance=sensing_covariance,
        method="zf-sensing",
    )

