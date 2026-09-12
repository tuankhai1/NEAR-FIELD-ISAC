"""Conditioned SDR with exact dimension reduction and physical validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .channels import Scenario, far_field_response, near_field_response, sensing_response_matrices
from .communication import Waveform, communication_rates
from .config import SimulationConfig
from .fim import crb_from_blocks, fisher_information_blocks

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OptimizationResult:
    """Waveform plus independently recomputed solver diagnostics."""

    waveform: Waveform
    status: str
    objective: float
    rates: FloatArray
    solver: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _import_cvxpy() -> Any:
    try:
        import cvxpy as cp
    except ImportError as error:
        raise RuntimeError('Install dependencies: pip install -e ".[optimization]"') from error
    return cp


def _solver_candidates(cp: Any, requested: str, *, prefer_clarabel: bool = False) -> list[str]:
    installed = set(cp.installed_solvers())
    if requested.lower() != "auto":
        if requested.upper() not in installed:
            raise RuntimeError(f"Solver {requested} is unavailable: {sorted(installed)}")
        return [requested.upper()]
    order = ("CLARABEL", "MOSEK", "SCS") if prefer_clarabel else ("MOSEK", "CLARABEL", "SCS")
    candidates = [name for name in order if name in installed]
    if not candidates:
        raise RuntimeError("No SDP-capable solver is installed")
    return candidates


def _solve_options(
    solver: str, tolerance: float, max_iterations: int, solver_threads: int | None
) -> dict[str, Any]:
    if solver == "SCS":
        return {"eps": tolerance, "max_iters": max_iterations}
    if solver == "CLARABEL":
        options: dict[str, Any] = {
            "tol_gap_abs": tolerance,
            "tol_gap_rel": tolerance,
            "tol_feas": tolerance,
            "max_iter": max_iterations,
        }
        if solver_threads is not None:
            options["max_threads"] = solver_threads
        return options
    if solver == "MOSEK":
        params: dict[str, Any] = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": max_iterations,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tolerance,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tolerance,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tolerance,
        }
        if solver_threads is not None:
            params["MSK_IPAR_NUM_THREADS"] = solver_threads
        return {"mosek_params": params}
    return {}


def _hermitian(matrix: ComplexArray) -> ComplexArray:
    return 0.5 * (matrix + matrix.conj().T)


def orthonormal_span(columns: ComplexArray) -> ComplexArray:
    """Rank-revealing basis, safe for duplicate or differently scaled columns."""
    norms = np.linalg.norm(columns, axis=0)
    nonzero = norms > np.finfo(float).tiny
    if not np.any(nonzero):
        raise ValueError("Transmit subspace is empty")
    u, values, _ = np.linalg.svd(columns[:, nonzero] / norms[nonzero], full_matrices=False)
    return u[:, values > values[0] * 1e-12]


def exact_transmit_basis(
    config: SimulationConfig,
    scenario: Scenario,
    model: str = "near",
    *,
    include_communication: bool = True,
) -> ComplexArray:
    """Span all echo/channel transmit functionals; orthogonal power is unnecessary.

    Projecting into the combined right row spaces of G, its derivatives, and
    user channel transposes preserves every objective/constraint functional,
    while never increasing power. This is exact, not hybrid beamforming.
    """
    target, derivatives = sensing_response_matrices(
        config,
        scenario.target_range,
        scenario.target_angle,
        model,
    )
    columns = [scenario.communication_channels.conj()] if include_communication else []
    for response in [target, *derivatives]:
        _, values, vh = np.linalg.svd(response, full_matrices=False)
        columns.append(vh[values > values[0] * 1e-10].conj().T)
    return orthonormal_span(np.column_stack(columns))


def _recover_rank_one(channel: ComplexArray, lifted: ComplexArray) -> ComplexArray:
    power = float(np.real(channel.T @ lifted @ channel.conj()))
    if power <= 0:
        raise ValueError("Non-positive desired power during rank-one recovery")
    return lifted @ channel.conj() / np.sqrt(power)


def solve_sdr(
    config: SimulationConfig,
    scenario: Scenario,
    *,
    min_rate: float | None = None,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1e-11,
    max_iterations: int = 20_000,
    solver_threads: int | None = None,
    transmit_basis: ComplexArray | None = None,
    receive_combiner: ComplexArray | None = None,
    method_name: str = "fully-digital-sdr",
    sensing_model: str = "near",
    reduce_dimension: bool = True,
) -> OptimizationResult:
    """Unchanged radians-based trace-CRB objective, with a balanced inverse LMI.

    Far-field mode optimizes angle alone, retaining the supplied communication
    channels. No secondary objective or smoothing is applied.
    """
    cp = _import_cvxpy()
    rate = config.min_rate if min_rate is None else float(min_rate)
    if not np.isfinite(rate) or rate < 0 or not 0 < tolerance < 1:
        raise ValueError("Rate must be finite and non-negative; tolerance must lie in (0,1)")
    if solver_threads is not None and solver_threads < 1:
        raise ValueError("solver_threads must be positive")
    if abs(scenario.target_gain) == 0:
        raise ValueError("Cannot optimize sensing with zero target gain")
    if transmit_basis is None:
        basis = (
            exact_transmit_basis(
                config,
                scenario,
                sensing_model,
                include_communication=rate > 0,
            )
            if reduce_dimension
            else np.eye(
                config.n_antennas,
                dtype=complex,
            )
        )
    else:
        if transmit_basis.shape[0] != config.n_antennas:
            raise ValueError("Transmit basis has incompatible shape")
        basis = orthonormal_span(transmit_basis)
    dimension, power = basis.shape[1], config.transmit_power
    covariance = cp.Variable((dimension, dimension), hermitian=True)
    lifted = (
        [cp.Variable((dimension, dimension), hermitian=True) for _ in range(config.n_users)]
        if rate > 0
        else []
    )
    target, derivatives = sensing_response_matrices(
        config,
        scenario.target_range,
        scenario.target_angle,
        sensing_model,
    )
    if receive_combiner is not None:
        target = receive_combiner @ target
        derivatives = [receive_combiner @ d for d in derivatives]
    # Remove nuisance-parallel derivative components before forming the FIM.
    # This invertible nuisance reparameterization leaves the target Schur
    # complement unchanged, and avoids subtracting large nearly equal terms.
    target_energy = np.linalg.norm(target, "fro") ** 2
    derivatives = [d - np.vdot(target, d) / target_energy * target for d in derivatives]
    jacobians = [abs(scenario.target_gain) * d for d in derivatives] + [target, 1j * target]
    count, fim_scale = len(derivatives), 2 * config.optimization_scale
    reference_diagonal = np.array(
        [fim_scale * power / config.n_antennas * np.linalg.norm(d, "fro") ** 2 for d in jacobians]
    )
    if np.any(reference_diagonal <= 0):
        raise ValueError("Unidentifiable sensing parameter")
    coordinate_scale = 1 / np.sqrt(reference_diagonal)
    balanced_derivatives = [
        np.sqrt(fim_scale * power) * scale * d @ basis
        for scale, d in zip(coordinate_scale, jacobians, strict=True)
    ]
    information = cp.bmat(
        [
            [
                cp.real(cp.trace((left.conj().T @ right) @ covariance))
                for right in balanced_derivatives
            ]
            for left in balanced_derivatives
        ]
    )
    normalizer = float(max(coordinate_scale[:count] ** 2))
    selector = np.zeros((count + 2, count))
    selector[:count] = np.diag(coordinate_scale[:count] / np.sqrt(normalizer))
    epigraph = cp.Variable((count, count), symmetric=True)
    inverse_lmi = cp.bmat([[information, selector], [selector.T, epigraph]])
    constraints = [covariance >> 0, cp.real(cp.trace(covariance)) <= 1, inverse_lmi >> 0]
    if lifted:
        constraints.append(covariance - sum(lifted) >> 0)
    sinr = np.expm1(np.log(2) * rate)
    for user, beam in enumerate(lifted):
        channel = basis.T @ scenario.communication_channels[:, user]
        norm = np.linalg.norm(channel)
        if norm == 0:
            raise ValueError("User channel is zero in the transmit subspace")
        normalized = channel.conj() / norm
        signal = cp.real(cp.quad_form(normalized, beam))
        total = cp.real(cp.quad_form(normalized, covariance))
        # At high SINR, dividing the constraint by SINR amplifies a small
        # conic feasibility residual into a failed physical communication rate.
        # Scale the same inequality back up without changing its feasible set;
        # retain the original scaling below 1 bit/s/Hz.
        sinr_scale = max(1.0, sinr)
        constraints.extend(
            [
                beam >> 0,
                (sinr_scale + sinr_scale / sinr) * signal
                >= sinr_scale * total + sinr_scale / (power * norm**2),
            ]
        )
    problem = cp.Problem(cp.Minimize(cp.trace(epigraph)), constraints)
    attempts: list[dict[str, Any]] = []
    accepted: list[OptimizationResult] = []
    for candidate in _solver_candidates(cp, solver):
        try:
            objective = problem.solve(
                solver=candidate,
                verbose=verbose,
                **_solve_options(candidate, tolerance, max_iterations, solver_threads),
            )
            if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                raise ValueError(f"status={problem.status}")
            if covariance.value is None or any(b.value is None for b in lifted):
                raise ValueError("No primal solution")
            baseband = power * _hermitian(np.asarray(covariance.value))
            beams = np.zeros((dimension, config.n_users), dtype=complex)
            for user, beam in enumerate(lifted):
                channel = basis.T @ scenario.communication_channels[:, user]
                beams[:, user] = _recover_rank_one(channel, power * _hermitian(beam.value))
            physical = _hermitian(basis @ baseband @ basis.conj().T)
            physical_beams = basis @ beams
            residual = _hermitian(physical - physical_beams @ physical_beams.conj().T)
            rates = communication_rates(scenario.communication_channels, physical, physical_beams)
            blocks = fisher_information_blocks(
                config,
                physical,
                scenario.target_gain,
                distance=scenario.target_range,
                angle=scenario.target_angle,
                receive_combiner=receive_combiner,
                scale=config.optimization_scale,
                model=sensing_model,
            )
            crb = crb_from_blocks(blocks)
            actual_objective = float(np.trace(crb))
            rate_margin = float(np.min(rates) - rate)
            power_margin = float(power - np.trace(physical).real)
            residual_eigenvalue = float(np.linalg.eigvalsh(residual).min())
            covariance_eigenvalue = float(np.linalg.eigvalsh(physical).min())
            epigraph_error = (
                abs(actual_objective - float(objective) * normalizer) / actual_objective
            )
            lmi_eigenvalue = float(np.linalg.eigvalsh(inverse_lmi.value).min())
            if rate_margin < -1e-5 or power_margin < -power * 1e-7:
                raise ValueError(
                    f"Primal infeasibility: rate={rate_margin:g}, power={power_margin:g}"
                )
            if min(residual_eigenvalue, covariance_eigenvalue) < -power * 1e-7:
                raise ValueError(f"Indefinite waveform covariance: {residual_eigenvalue:g}")
            if epigraph_error > 2e-5 or lmi_eigenvalue < -1e-7:
                raise ValueError(f"Information residual: {epigraph_error:g}, {lmi_eigenvalue:g}")
            noise_factor = config.n_antennas if receive_combiner is not None else 1
            factor = (
                config.optimization_scale
                * config.noise_power
                * noise_factor
                / (config.coherent_block_length)
            )
            metadata = {
                "baseband_covariance": baseband,
                "transmit_basis": basis,
                "receive_combiner": receive_combiner,
                "sensing_model": sensing_model,
                "solver_dimension": dimension,
                "conditioned_solver_objective": float(objective),
                "minimum_rate_margin": rate_margin,
                "transmit_power_margin": power_margin,
                "minimum_sensing_covariance_eigenvalue": residual_eigenvalue,
                "minimum_covariance_eigenvalue": covariance_eigenvalue,
                "epigraph_relative_error": float(epigraph_error),
                "minimum_information_lmi_eigenvalue": lmi_eigenvalue,
                "physical_crb_trace": actual_objective * factor,
                "physical_crb": crb * factor,
                "solver_tolerance": tolerance,
                "validation_passed": True,
                "solve_time_seconds": problem.solver_stats.solve_time,
                "iterations": problem.solver_stats.num_iters,
            }
            accepted.append(
                OptimizationResult(
                    Waveform(physical, physical_beams, residual, method_name),
                    str(problem.status),
                    actual_objective,
                    rates,
                    candidate,
                    metadata,
                )
            )
            attempts.append({"solver": candidate, "status": str(problem.status), "accepted": True})
            if problem.status == cp.OPTIMAL:
                break
        except Exception as error:
            if not isinstance(error, (ValueError, cp.error.SolverError)) and not (
                error.__class__.__module__.partition(".")[0] == "mosek"
            ):
                raise
            attempts.append({"solver": candidate, "accepted": False, "reason": str(error)})
    if not accepted:
        raise RuntimeError(f"No solver produced a validated waveform: {attempts}")
    best = min(accepted, key=lambda result: result.objective)
    best.metadata["solver_attempts"] = attempts
    return best


def solve_fully_digital_sdr(
    config: SimulationConfig, scenario: Scenario, **kwargs: Any
) -> OptimizationResult:
    """Fully digital SDR, with exact dimension reduction by default."""
    return solve_sdr(config, scenario, method_name="fully-digital-sdr", **kwargs)


def hybrid_analog_beamformer(
    config: SimulationConfig, scenario: Scenario, *, sensing_model: str = "near"
) -> ComplexArray:
    """Eq. (22), retaining near-field user focusing for the far-field target reference."""
    columns = [
        near_field_response(config, float(r), float(t)).conj()
        for r, t in zip(scenario.user_ranges, scenario.user_angles, strict=True)
    ]
    target = (
        far_field_response(config, scenario.target_angle).conj()
        if sensing_model == "far"
        else (near_field_response(config, scenario.target_range, scenario.target_angle).conj())
    )
    columns.extend([target] * (config.n_rf_chains - config.n_users))
    return np.column_stack(columns)


def random_hybrid_combiner(config: SimulationConfig, rng: np.random.Generator) -> ComplexArray:
    """Paper's random unit-modulus receiver; approximate noise covariance N I."""
    return np.exp(1j * rng.uniform(0, 2 * np.pi, (config.n_rf_chains, config.n_antennas)))


def solve_hybrid_sdr(
    config: SimulationConfig,
    scenario: Scenario,
    *,
    rng: np.random.Generator | None = None,
    receive_combiner: ComplexArray | None = None,
    analog_beamformer: ComplexArray | None = None,
    method_name: str = "hybrid-two-stage-sdr",
    **kwargs: Any,
) -> OptimizationResult:
    """Optimize baseband for paper focusing or a supplied unit-modulus RF matrix."""
    rng = np.random.default_rng(config.seed + 1) if rng is None else rng
    if analog_beamformer is None:
        analog = hybrid_analog_beamformer(
            config,
            scenario,
            sensing_model=kwargs.get("sensing_model", "near"),
        )
    else:
        analog = np.asarray(analog_beamformer, dtype=complex)
        if analog.shape != (config.n_antennas, config.n_rf_chains):
            raise ValueError("Analog beamformer has incompatible shape")
        if not np.all(np.isfinite(analog)) or not np.allclose(
            np.abs(analog), 1, rtol=0, atol=1e-10
        ):
            raise ValueError("Analog beamformer must be finite and unit modulus")
    if receive_combiner is None:
        receive_combiner = random_hybrid_combiner(config, rng)
    else:
        receive_combiner = np.asarray(receive_combiner, dtype=complex)
        if receive_combiner.shape != (config.n_rf_chains, config.n_antennas):
            raise ValueError("Receive combiner has incompatible shape")
        if not np.all(np.isfinite(receive_combiner)) or not np.allclose(
            np.abs(receive_combiner), 1, rtol=0, atol=1e-10
        ):
            raise ValueError("Receive combiner must be finite and unit modulus")
    result = solve_sdr(
        config,
        scenario,
        transmit_basis=analog,
        receive_combiner=receive_combiner,
        method_name=method_name,
        **kwargs,
    )
    basis = result.metadata["transmit_basis"]
    inverse = np.linalg.pinv(analog, rcond=1e-12)
    transform = inverse @ basis
    baseband = _hermitian(transform @ result.metadata["baseband_covariance"] @ transform.conj().T)
    beams = inverse @ result.waveform.communication_beamformers
    error = float(
        np.linalg.norm(analog @ baseband @ analog.conj().T - result.waveform.covariance)
        / np.linalg.norm(result.waveform.covariance)
    )
    if error > 1e-8:
        raise RuntimeError(f"RF/baseband covariance reconstruction failed: {error:g}")
    return replace(
        result,
        metadata={
            **result.metadata,
            "baseband_covariance": baseband,
            "baseband_communication_beamformers": beams,
            "solver_basis": basis,
            "transmit_basis": analog,
            "rf_reconstruction_relative_error": error,
        },
    )
