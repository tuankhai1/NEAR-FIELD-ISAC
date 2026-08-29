"""Fisher information and Cramer-Rao bound calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .channels import antenna_positions, far_field_response, target_response_matrices
from .config import SimulationConfig

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FisherBlocks:
    """Blocks of the FIM for ``[range, angle, Re(beta), Im(beta)]``."""

    j11: FloatArray
    j12: FloatArray
    j22: FloatArray


def _real_trace(left: ComplexArray, covariance: ComplexArray, right: ComplexArray) -> float:
    return float(np.real(np.trace(left @ covariance @ right.conj().T)))


def fisher_information_blocks(
    config: SimulationConfig,
    covariance: ComplexArray,
    target_gain: complex,
    *,
    distance: float | None = None,
    angle: float | None = None,
    receive_combiner: ComplexArray | None = None,
    scale: float | None = None,
) -> FisherBlocks:
    """Calculate Appendix-B FIM blocks.

    ``scale`` defaults to ``T / sigma_s^2`` for the physical CRB.  Supplying
    ``config.optimization_scale`` instead reproduces the numerical scaling used
    inside the authors' CVX optimization.  For a hybrid receiver the effective
    noise power is ``N * sigma_s^2`` as assumed below Eq. (15).
    """

    distance = config.target_range if distance is None else distance
    angle = config.target_angle if angle is None else angle
    target, derivative_range, derivative_angle = target_response_matrices(
        config, distance, angle
    )
    if receive_combiner is not None:
        target = receive_combiner @ target
        derivative_range = receive_combiner @ derivative_range
        derivative_angle = receive_combiner @ derivative_angle
        default_noise = config.n_antennas * config.noise_power
    else:
        default_noise = config.noise_power
    if scale is None:
        scale = config.coherent_block_length / default_noise

    rr = _real_trace(derivative_range, covariance, derivative_range)
    rt = _real_trace(derivative_range, covariance, derivative_angle)
    tt = _real_trace(derivative_angle, covariance, derivative_angle)
    j11 = 2.0 * scale * abs(target_gain) ** 2 * np.array([[rr, rt], [rt, tt]])

    range_cross = np.trace(target @ covariance @ derivative_range.conj().T)
    angle_cross = np.trace(target @ covariance @ derivative_angle.conj().T)
    j12 = 2.0 * scale * np.real(
        np.conj(target_gain)
        * np.array(
            [
                [range_cross, 1j * range_cross],
                [angle_cross, 1j * angle_cross],
            ]
        )
    )
    beta_information = 2.0 * scale * _real_trace(target, covariance, target)
    j22 = beta_information * np.eye(2)
    return FisherBlocks(j11=j11, j12=j12, j22=j22)


def crb_from_blocks(blocks: FisherBlocks, *, rcond: float = 1.0e-12) -> FloatArray:
    """Eliminate the nuisance reflection coefficient via a Schur complement."""

    j22_inverse = np.linalg.pinv(blocks.j22, rcond=rcond, hermitian=True)
    equivalent_fim = blocks.j11 - blocks.j12 @ j22_inverse @ blocks.j12.T
    equivalent_fim = 0.5 * (equivalent_fim + equivalent_fim.T)
    crb = np.linalg.pinv(equivalent_fim, rcond=rcond, hermitian=True)
    return np.real(0.5 * (crb + crb.T))


def crb_matrix(
    config: SimulationConfig,
    covariance: ComplexArray,
    target_gain: complex,
    **kwargs: object,
) -> FloatArray:
    """Convenience wrapper returning the 2x2 range/angle CRB."""

    return crb_from_blocks(
        fisher_information_blocks(config, covariance, target_gain, **kwargs)
    )


def root_crb(crb: FloatArray) -> tuple[float, float]:
    """Return range RCRB in metres and angle RCRB in degrees."""

    diagonal = np.maximum(np.diag(crb), 0.0)
    return float(np.sqrt(diagonal[0])), float(np.rad2deg(np.sqrt(diagonal[1])))


def far_field_angle_crb(
    config: SimulationConfig,
    covariance: ComplexArray,
    target_gain: complex,
    *,
    angle: float | None = None,
    receive_combiner: ComplexArray | None = None,
    scale: float | None = None,
) -> float:
    """Return the far-field angle CRB with complex target gain as nuisance.

    This is the one-dimensional far-field counterpart of Appendix B. Range is
    absent because a planar-wave steering vector carries no spatial range
    information.
    """

    angle = config.target_angle if angle is None else angle
    response = far_field_response(config, angle)
    derivative = (
        -1j
        * 2.0
        * np.pi
        / config.wavelength
        * antenna_positions(config)
        * np.sin(angle)
        * response
    )
    target = np.outer(response, response)
    target_angle_derivative = np.outer(derivative, response) + np.outer(
        response, derivative
    )
    if receive_combiner is not None:
        target = receive_combiner @ target
        target_angle_derivative = receive_combiner @ target_angle_derivative
        default_noise = config.n_antennas * config.noise_power
    else:
        default_noise = config.noise_power
    if scale is None:
        scale = config.coherent_block_length / default_noise

    angle_information = (
        2.0
        * scale
        * abs(target_gain) ** 2
        * _real_trace(target_angle_derivative, covariance, target_angle_derivative)
    )
    cross_trace = np.trace(
        target @ covariance @ target_angle_derivative.conj().T
    )
    cross = 2.0 * scale * np.real(
        np.conj(target_gain) * np.array([cross_trace, 1j * cross_trace])
    )
    nuisance_information = 2.0 * scale * _real_trace(target, covariance, target)
    nuisance = nuisance_information * np.eye(2)
    equivalent_information = angle_information - float(
        cross @ np.linalg.pinv(nuisance, hermitian=True) @ cross.T
    )
    if equivalent_information <= 0:
        return float("inf")
    return float(1.0 / equivalent_information)
