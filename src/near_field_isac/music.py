"""Signal generation and two-dimensional near-field MUSIC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .channels import antenna_positions, far_field_response, near_field_response
from .config import SimulationConfig

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MusicResult:
    """Normalized MUSIC spectrum and its grid-maximum estimate."""

    spectrum: FloatArray
    x_grid: FloatArray
    y_grid: FloatArray
    estimated_x: float
    estimated_y: float
    estimated_range: float
    estimated_angle_deg: float


def complex_normal(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> ComplexArray:
    """Draw proper complex Gaussian samples with unit variance."""

    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def positive_semidefinite_part(matrix: ComplexArray, *, tolerance: float = 1.0e-6) -> ComplexArray:
    """Hermitianize a matrix and clip solver-scale negative eigenvalues."""

    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -tolerance * scale:
        raise ValueError(
            "matrix is not positive semidefinite; minimum eigenvalue is "
            f"{float(np.min(eigenvalues)):.3e}"
        )
    clipped = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * clipped[None, :]) @ eigenvectors.conj().T


def covariance_samples(
    covariance: ComplexArray,
    n_samples: int,
    rng: np.random.Generator,
) -> ComplexArray:
    """Generate zero-mean complex samples with a requested covariance."""

    covariance = positive_semidefinite_part(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    square_root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :])
    return square_root @ complex_normal(rng, (covariance.shape[0], n_samples))


def generate_transmit_samples(
    covariance: ComplexArray,
    beamformers: ComplexArray,
    n_samples: int,
    rng: np.random.Generator,
) -> ComplexArray:
    """Generate communication plus dedicated-sensing samples from Eq. (8)."""

    sensing_covariance = covariance - beamformers @ beamformers.conj().T
    sensing_samples = covariance_samples(sensing_covariance, n_samples, rng)
    symbols = complex_normal(rng, (beamformers.shape[1], n_samples))
    return beamformers @ symbols + sensing_samples


def simulate_echo(
    config: SimulationConfig,
    transmit_samples: ComplexArray,
    target_gain: complex,
    rng: np.random.Generator,
    *,
    model: str = "near",
    distance: float | None = None,
    angle: float | None = None,
    noise_samples: ComplexArray | None = None,
) -> ComplexArray:
    """Simulate the monostatic echo in Eq. (12)."""

    distance = config.target_range if distance is None else distance
    angle = config.target_angle if angle is None else angle
    if model == "near":
        response = near_field_response(config, distance, angle)
    elif model == "far":
        response = far_field_response(config, angle)
    else:
        raise ValueError("model must be 'near' or 'far'")
    channel = target_gain * np.outer(response, response)
    if noise_samples is None:
        noise_samples = np.sqrt(config.noise_power) * complex_normal(
            rng, (config.n_antennas, transmit_samples.shape[1])
        )
    return channel @ transmit_samples + noise_samples


def sample_covariance(samples: ComplexArray) -> ComplexArray:
    """Return the maximum-likelihood sample covariance."""

    return samples @ samples.conj().T / samples.shape[1]


def noise_projector(covariance: ComplexArray, n_targets: int = 1) -> ComplexArray:
    """Estimate the MUSIC noise-subspace projector from Eq. (23)."""

    if not (1 <= n_targets < covariance.shape[0]):
        raise ValueError("n_targets must be between 1 and n_antennas - 1")
    _, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.conj().T))
    noise_vectors = eigenvectors[:, : covariance.shape[0] - n_targets]
    return noise_vectors @ noise_vectors.conj().T


def signal_subspace(covariance: ComplexArray, n_targets: int = 1) -> ComplexArray:
    """Return the dominant orthonormal signal-subspace eigenvectors."""

    if not (1 <= n_targets < covariance.shape[0]):
        raise ValueError("n_targets must be between 1 and n_antennas - 1")
    _, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.conj().T))
    return eigenvectors[:, -n_targets:]


def steering_matrix_xy(
    config: SimulationConfig,
    x: FloatArray,
    y: FloatArray,
    *,
    model: str,
) -> ComplexArray:
    """Build steering vectors for flattened Cartesian grid coordinates."""

    distance = np.hypot(x, y)
    angle = np.arctan2(y, x)
    positions = antenna_positions(config)[:, None]
    wave_number = 2.0 * np.pi / config.wavelength
    if model == "near":
        element_ranges = np.sqrt(
            np.maximum(
                distance[None, :] ** 2
                + positions**2
                - 2.0 * distance[None, :] * positions * np.cos(angle)[None, :],
                0.0,
            )
        )
        phase_distance = element_ranges - distance[None, :]
        return np.exp(-1j * wave_number * phase_distance)
    if model == "far":
        return np.exp(1j * wave_number * positions * np.cos(angle)[None, :])
    raise ValueError("model must be 'near' or 'far'")


def music_spectrum_xy(
    config: SimulationConfig,
    covariance: ComplexArray,
    x_axis: FloatArray,
    y_axis: FloatArray,
    *,
    model: str = "near",
    n_targets: int = 1,
    batch_size: int = 20_000,
    receive_combiner: ComplexArray | None = None,
) -> MusicResult:
    """Evaluate normalized ``1 / p(r, theta)`` from paper Eq. (24)."""

    x_grid, y_grid = np.meshgrid(
        np.asarray(x_axis, dtype=float), np.asarray(y_axis, dtype=float)
    )
    flat_x = x_grid.ravel()
    flat_y = y_grid.ravel()
    signal_vectors = signal_subspace(covariance, n_targets=n_targets)
    denominator = np.empty(flat_x.size, dtype=float)
    for start in range(0, flat_x.size, batch_size):
        stop = min(start + batch_size, flat_x.size)
        steering = steering_matrix_xy(
            config, flat_x[start:stop], flat_y[start:stop], model=model
        )
        if receive_combiner is not None:
            steering = receive_combiner @ steering
        total_energy = np.sum(np.abs(steering) ** 2, axis=0)
        signal_energy = np.sum(
            np.abs(signal_vectors.conj().T @ steering) ** 2, axis=0
        )
        denominator[start:stop] = np.real(total_energy - signal_energy)
    floor = max(float(np.max(np.abs(denominator))) * 1.0e-15, np.finfo(float).tiny)
    pseudo_spectrum = 1.0 / np.maximum(denominator, floor)
    pseudo_spectrum /= np.max(pseudo_spectrum)
    spectrum = pseudo_spectrum.reshape(x_grid.shape)
    maximum = int(np.argmax(spectrum))
    estimated_x = float(flat_x[maximum])
    estimated_y = float(flat_y[maximum])
    return MusicResult(
        spectrum=spectrum,
        x_grid=x_grid,
        y_grid=y_grid,
        estimated_x=estimated_x,
        estimated_y=estimated_y,
        estimated_range=float(np.hypot(estimated_x, estimated_y)),
        estimated_angle_deg=float(np.rad2deg(np.arctan2(estimated_y, estimated_x))),
    )
