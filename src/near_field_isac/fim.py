"""Fisher information and Cramer-Rao bound calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .channels import sensing_response_matrices
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
    model: str = "near",
) -> FisherBlocks:
    """Calculate Appendix-B FIM blocks.

    ``scale`` defaults to ``T / sigma_s^2`` for the physical CRB.  Supplying
    ``config.optimization_scale`` instead reproduces the numerical scaling used
    inside the authors' CVX optimization.  For a hybrid receiver the effective
    noise power is ``N * sigma_s^2`` as assumed below Eq. (15).
    """

    distance = config.target_range if distance is None else distance
    angle = config.target_angle if angle is None else angle
    target, derivatives = sensing_response_matrices(config, distance, angle, model)
    if receive_combiner is not None:
        target = receive_combiner @ target
        derivatives = [receive_combiner @ derivative for derivative in derivatives]
        default_noise = config.n_antennas * config.noise_power
    else:
        default_noise = config.noise_power
    if scale is None:
        scale = config.coherent_block_length / default_noise

    j11 = (
        2.0
        * scale
        * abs(target_gain) ** 2
        * np.array(
            [
                [_real_trace(left, covariance, right) for right in derivatives]
                for left in derivatives
            ]
        )
    )
    cross = np.array([np.trace(target @ covariance @ d.conj().T) for d in derivatives])
    j12 = 2.0 * scale * np.real(np.conj(target_gain) * np.column_stack([cross, 1j * cross]))
    beta_information = 2.0 * scale * _real_trace(target, covariance, target)
    j22 = beta_information * np.eye(2)
    return FisherBlocks(j11=j11, j12=j12, j22=j22)


def crb_from_blocks(blocks: FisherBlocks, *, rcond: float = 1.0e-12) -> FloatArray:
    """Invert identifiable information in balanced coordinates; never hide nulls."""
    if not all(np.all(np.isfinite(b)) for b in (blocks.j11, blocks.j12, blocks.j22)):
        raise ValueError("FIM contains non-finite entries")
    if np.min(np.linalg.eigvalsh(blocks.j22)) <= 0:
        raise ValueError("Target gain is unidentifiable: non-positive nuisance information")
    equivalent_fim = blocks.j11 - blocks.j12 @ np.linalg.solve(blocks.j22, blocks.j12.T)
    equivalent_fim = 0.5 * (equivalent_fim + equivalent_fim.T)
    diagonal = np.diag(equivalent_fim)
    if np.any(diagonal <= 0):
        raise ValueError("Target parameters are unidentifiable or FIM is indefinite")
    scaler = np.diag(1 / np.sqrt(diagonal))
    balanced = scaler @ equivalent_fim @ scaler
    if np.min(np.linalg.eigvalsh(balanced)) <= rcond:
        raise ValueError("Target parameters are unidentifiable or FIM is ill-conditioned")
    crb = scaler @ np.linalg.solve(balanced, scaler)
    return np.real(0.5 * (crb + crb.T))


def crb_matrix(
    config: SimulationConfig,
    covariance: ComplexArray,
    target_gain: complex,
    **kwargs: object,
) -> FloatArray:
    """Convenience wrapper returning the 2x2 range/angle CRB."""

    return crb_from_blocks(fisher_information_blocks(config, covariance, target_gain, **kwargs))


def root_crb(crb: FloatArray) -> tuple[float, float]:
    """Return range RCRB in metres and angle RCRB in degrees."""

    diagonal = np.diag(crb)
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise ValueError("RCRB requires finite, strictly positive variances")
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

    try:
        blocks = fisher_information_blocks(
            config,
            covariance,
            target_gain,
            angle=angle,
            receive_combiner=receive_combiner,
            scale=scale,
            model="far",
        )
        return float(crb_from_blocks(blocks)[0, 0])
    except ValueError:
        return float("inf")
