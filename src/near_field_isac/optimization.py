"""CVXPY implementation of the paper's SDR and hybrid two-stage design."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .channels import Scenario, near_field_response, target_response_matrices
from .communication import Waveform, communication_rates
from .config import SimulationConfig
from .fim import crb_from_blocks, fisher_information_blocks

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OptimizationResult:
    """Waveform plus solver diagnostics."""

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
        raise RuntimeError(
            "CVXPY is required for SDR optimization. Install it with "
            "`python -m pip install -e \".[optimization]\"`."
        ) from error
    return cp


def _solver_candidates(cp: Any, requested: str) -> list[str]:
    installed = set(cp.installed_solvers())
    if requested.lower() != "auto":
        solver = requested.upper()
        if solver not in installed:
            raise RuntimeError(
                f"Requested solver {solver!r} is not installed. Available: {sorted(installed)}"
            )
        return [solver]
    candidates = [
        candidate for candidate in ("MOSEK", "CLARABEL", "SCS") if candidate in installed
    ]
    if candidates:
        return candidates
    raise RuntimeError(
        "No SDP-capable solver was found. Install CVXPY with SCS or configure MOSEK."
    )


def _solve_options(
    solver: str,
    tolerance: float,
    max_iterations: int,
    solver_threads: int | None,
) -> dict[str, Any]:
    if solver == "SCS":
        return {"eps": tolerance, "max_iters": max_iterations}
    if solver == "CLARABEL":
        options = {
            "tol_gap_abs": tolerance,
            "tol_feas": tolerance,
            "max_iter": max_iterations,
        }
        if solver_threads is not None:
            options["max_threads"] = solver_threads
        return options
    if solver == "MOSEK":
        mosek_params = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": max_iterations,
            # CVXPY dualizes continuous conic problems before handing them to
            # MOSEK. Solving that canonicalized form as a dual avoids a much
            # larger factorization for the paper-size fully digital SDP.
            "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
        }
        if solver_threads is not None:
            mosek_params["MSK_IPAR_NUM_THREADS"] = solver_threads
        return {"eps": tolerance, "mosek_params": mosek_params}
    return {}


def _fim_expressions(
    cp: Any,
    config: SimulationConfig,
    covariance: Any,
    target_gain: complex,
    distance: float,
    angle: float,
    receive_combiner: ComplexArray | None,
) -> tuple[Any, Any, Any]:
    target, derivative_range, derivative_angle = target_response_matrices(
        config, distance, angle
    )
    if receive_combiner is not None:
        target = receive_combiner @ target
        derivative_range = receive_combiner @ derivative_range
        derivative_angle = receive_combiner @ derivative_angle

    def trace_form(left: ComplexArray, right: ComplexArray) -> Any:
        return cp.trace(left @ covariance @ right.conj().T)

    scale = config.optimization_scale
    range_range = cp.real(trace_form(derivative_range, derivative_range))
    range_angle = cp.real(trace_form(derivative_range, derivative_angle))
    angle_angle = cp.real(trace_form(derivative_angle, derivative_angle))
    j11 = 2.0 * scale * abs(target_gain) ** 2 * cp.bmat(
        [[range_range, range_angle], [range_angle, angle_angle]]
    )

    range_cross = trace_form(target, derivative_range)
    angle_cross = trace_form(target, derivative_angle)
    j12 = 2.0 * scale * cp.bmat(
        [
            [
                cp.real(np.conj(target_gain) * range_cross),
                cp.real(np.conj(target_gain) * 1j * range_cross),
            ],
            [
                cp.real(np.conj(target_gain) * angle_cross),
                cp.real(np.conj(target_gain) * 1j * angle_cross),
            ],
        ]
    )
    beta_information = 2.0 * scale * cp.real(trace_form(target, target))
    j22 = beta_information * np.eye(2)
    return j11, j12, j22


def _hermitian(matrix: ComplexArray) -> ComplexArray:
    return 0.5 * (matrix + matrix.conj().T)


def _fim_preconditioners(
    config: SimulationConfig,
    target_gain: complex,
    distance: float,
    angle: float,
    receive_combiner: ComplexArray | None,
) -> tuple[FloatArray, FloatArray]:
    """Balance the FIM blocks using an isotropic reference covariance.

    The resulting congruence transform is exact; it changes only the numerical
    coordinates seen by the cone solver, not the feasible set or objective.
    """

    reference_covariance = (
        config.transmit_power / config.n_antennas * np.eye(config.n_antennas)
    )
    reference = fisher_information_blocks(
        config,
        reference_covariance,
        target_gain,
        distance=distance,
        angle=angle,
        receive_combiner=receive_combiner,
        scale=config.optimization_scale,
    )

    def diagonal_scaler(matrix: FloatArray) -> FloatArray:
        diagonal = np.maximum(np.abs(np.diag(matrix)), 1.0e-12)
        return np.diag(1.0 / np.sqrt(diagonal))

    return diagonal_scaler(reference.j11), diagonal_scaler(reference.j22)


def _recover_rank_one(
    channel: ComplexArray,
    lifted_beamformer: ComplexArray,
    *,
    allow_zero: bool = False,
) -> ComplexArray:
    denominator_squared = float(
        np.real(channel.T @ lifted_beamformer @ channel.conj())
    )
    if denominator_squared <= 0:
        if allow_zero:
            return np.zeros_like(channel)
        raise RuntimeError("rank-one recovery encountered a non-positive desired power")
    return lifted_beamformer @ channel.conj() / np.sqrt(denominator_squared)


def solve_sdr(
    config: SimulationConfig,
    scenario: Scenario,
    *,
    min_rate: float | None = None,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1.0e-7,
    max_iterations: int = 20_000,
    solver_threads: int | None = None,
    transmit_basis: ComplexArray | None = None,
    receive_combiner: ComplexArray | None = None,
    method_name: str = "fully-digital-sdr",
) -> OptimizationResult:
    """Solve paper problem (20), optionally in a fixed hybrid RF subspace."""

    cp = _import_cvxpy()
    if solver_threads is not None and solver_threads < 1:
        raise ValueError("solver_threads must be at least 1")
    min_rate = config.min_rate if min_rate is None else min_rate
    n_dimension = config.n_antennas if transmit_basis is None else transmit_basis.shape[1]
    if transmit_basis is None:
        transmit_basis = np.eye(config.n_antennas, dtype=np.complex128)
    if transmit_basis.shape[0] != config.n_antennas:
        raise ValueError("transmit_basis has an incompatible number of rows")

    baseband_covariance = cp.Variable((n_dimension, n_dimension), hermitian=True)
    lifted = [
        cp.Variable((n_dimension, n_dimension), hermitian=True)
        for _ in range(config.n_users)
    ]
    scaled_auxiliary = cp.Variable((2, 2), symmetric=True)
    inverse_epigraph = cp.Variable((2, 2), symmetric=True)
    physical_covariance = (
        transmit_basis @ baseband_covariance @ transmit_basis.conj().T
    )
    j11, j12, j22 = _fim_expressions(
        cp,
        config,
        physical_covariance,
        scenario.target_gain,
        scenario.target_range,
        scenario.target_angle,
        receive_combiner,
    )
    parameter_scaler, nuisance_scaler = _fim_preconditioners(
        config,
        scenario.target_gain,
        scenario.target_range,
        scenario.target_angle,
        receive_combiner,
    )
    scaled_j11 = parameter_scaler @ j11 @ parameter_scaler
    scaled_j12 = parameter_scaler @ j12 @ nuisance_scaler
    scaled_j22 = nuisance_scaler @ j22 @ nuisance_scaler
    objective_weight = parameter_scaler @ parameter_scaler
    objective_normalizer = float(np.max(np.diag(objective_weight)))
    normalized_weight = objective_weight / objective_normalizer

    constraints: list[Any] = [
        baseband_covariance >> 0,
        scaled_auxiliary >> 1.0e-9 * np.eye(2),
        inverse_epigraph >> 0,
        cp.bmat(
            [
                [scaled_j11 - scaled_auxiliary, scaled_j12],
                [scaled_j12.T, scaled_j22],
            ]
        )
        >> 0,
        cp.bmat(
            [
                [scaled_auxiliary, np.eye(2)],
                [np.eye(2), inverse_epigraph],
            ]
        )
        >> 0,
        cp.real(cp.trace(physical_covariance)) <= config.transmit_power,
        baseband_covariance - sum(lifted) >> 0,
    ]
    sinr_target = 2.0**min_rate - 1.0
    for user, lifted_user in enumerate(lifted):
        effective_channel = transmit_basis.T @ scenario.communication_channels[:, user]
        signal = cp.real(cp.quad_form(np.conj(effective_channel), lifted_user))
        total = cp.real(
            cp.quad_form(np.conj(effective_channel), baseband_covariance)
        )
        constraints.extend(
            [
                lifted_user >> 0,
                signal >= sinr_target * (total - signal + 1.0),
            ]
        )

    problem = cp.Problem(
        cp.Minimize(cp.trace(normalized_weight @ inverse_epigraph)), constraints
    )
    selected_solver = ""
    objective = None
    solver_errors: list[str] = []
    for candidate in _solver_candidates(cp, solver):
        try:
            objective = problem.solve(
                solver=candidate,
                verbose=verbose,
                **_solve_options(
                    candidate, tolerance, max_iterations, solver_threads
                ),
            )
        except cp.error.SolverError as error:
            solver_errors.append(f"{candidate}: {error}")
            continue
        except Exception as error:
            is_mosek_error = (
                candidate == "MOSEK"
                and error.__class__.__module__.partition(".")[0] == "mosek"
            )
            if not is_mosek_error:
                raise
            if "err_space" in str(getattr(error, "errno", "")).lower():
                raise RuntimeError(
                    "MOSEK ran out of memory while solving the paper-size SDP. "
                    "Close memory-heavy applications or rerun with "
                    "`--solver-threads 1`."
                ) from error
            solver_errors.append(f"{candidate}: {error}")
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            selected_solver = candidate
            break
        solver_errors.append(f"{candidate}: status {problem.status}")
    if not selected_solver:
        details = "; ".join(solver_errors)
        raise RuntimeError(f"All candidate SDP solvers failed. {details}")
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"SDR failed with status {problem.status!r}")
    if baseband_covariance.value is None or any(item.value is None for item in lifted):
        raise RuntimeError("solver returned no primal solution")

    baseband_value = _hermitian(np.asarray(baseband_covariance.value))
    recovered_baseband = np.empty(
        (n_dimension, config.n_users), dtype=np.complex128
    )
    for user, lifted_user in enumerate(lifted):
        effective_channel = transmit_basis.T @ scenario.communication_channels[:, user]
        recovered_baseband[:, user] = _recover_rank_one(
            effective_channel,
            _hermitian(np.asarray(lifted_user.value)),
            allow_zero=min_rate <= 1.0e-12,
        )
    physical_value = _hermitian(
        transmit_basis @ baseband_value @ transmit_basis.conj().T
    )
    beamformers = transmit_basis @ recovered_baseband
    sensing_covariance = _hermitian(
        physical_value - beamformers @ beamformers.conj().T
    )
    waveform = Waveform(
        covariance=physical_value,
        communication_beamformers=beamformers,
        sensing_covariance=sensing_covariance,
        method=method_name,
    )
    rates = communication_rates(
        scenario.communication_channels, physical_value, beamformers
    )
    minimum_sensing_eigenvalue = float(np.min(np.linalg.eigvalsh(sensing_covariance)))
    transmit_power_used = float(np.real(np.trace(physical_value)))
    optimized_blocks = fisher_information_blocks(
        config,
        physical_value,
        scenario.target_gain,
        distance=scenario.target_range,
        angle=scenario.target_angle,
        receive_combiner=receive_combiner,
        scale=config.optimization_scale,
    )
    unscaled_objective = float(np.trace(crb_from_blocks(optimized_blocks)))
    return OptimizationResult(
        waveform=waveform,
        status=str(problem.status),
        objective=unscaled_objective,
        rates=rates,
        solver=selected_solver,
        metadata={
            "baseband_covariance": baseband_value,
            "transmit_basis": transmit_basis,
            "receive_combiner": receive_combiner,
            "conditioned_solver_objective": float(objective),
            "minimum_rate_margin": float(np.min(rates) - min_rate),
            "transmit_power_margin": float(config.transmit_power - transmit_power_used),
            "minimum_sensing_covariance_eigenvalue": minimum_sensing_eigenvalue,
        },
    )


def solve_fully_digital_sdr(
    config: SimulationConfig,
    scenario: Scenario,
    **kwargs: Any,
) -> OptimizationResult:
    """Solve the globally optimal fully digital SDR in paper Section III-B."""

    return solve_sdr(config, scenario, method_name="fully-digital-sdr", **kwargs)


def hybrid_analog_beamformer(config: SimulationConfig, scenario: Scenario) -> ComplexArray:
    """Construct the unit-modulus RF beamformer from paper Eq. (22)."""

    columns: list[ComplexArray] = []
    for distance, angle in zip(scenario.user_ranges, scenario.user_angles, strict=True):
        columns.append(np.conj(near_field_response(config, float(distance), float(angle))))
    target_column = np.conj(
        near_field_response(config, scenario.target_range, scenario.target_angle)
    )
    while len(columns) < config.n_rf_chains:
        columns.append(target_column.copy())
    return np.column_stack(columns[: config.n_rf_chains])


def random_hybrid_combiner(
    config: SimulationConfig, rng: np.random.Generator
) -> ComplexArray:
    """Draw the random unit-modulus receive combiner assumed below Eq. (15)."""

    phases = rng.uniform(0.0, 2.0 * np.pi, (config.n_rf_chains, config.n_antennas))
    return np.exp(1j * phases)


def solve_hybrid_sdr(
    config: SimulationConfig,
    scenario: Scenario,
    *,
    rng: np.random.Generator | None = None,
    receive_combiner: ComplexArray | None = None,
    **kwargs: Any,
) -> OptimizationResult:
    """Apply paper Section III-C: RF focusing followed by baseband SDR."""

    rng = np.random.default_rng(config.seed + 1) if rng is None else rng
    analog_beamformer = hybrid_analog_beamformer(config, scenario)
    if receive_combiner is None:
        receive_combiner = random_hybrid_combiner(config, rng)
    return solve_sdr(
        config,
        scenario,
        transmit_basis=analog_beamformer,
        receive_combiner=receive_combiner,
        method_name="hybrid-two-stage-sdr",
        **kwargs,
    )
