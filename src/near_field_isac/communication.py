"""Transmit waveform container and the communication rate of Eq. (11)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
