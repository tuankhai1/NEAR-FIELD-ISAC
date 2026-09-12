"""Seeded PSO and DE searches over virtual RF focusing locations, with inner SDR.

The physical users/target, receive combiner, and objective never change during
a search. Search seeds are separate from the channel realization seed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from numbers import Integral
from time import perf_counter
from typing import Any

import numpy as np

from .channels import Scenario, near_field_response
from .communication import communication_rates
from .config import SimulationConfig
from .fim import crb_matrix
from .optimization import OptimizationResult, solve_hybrid_sdr


@dataclass(frozen=True)
class SearchSettings:
    population: int = 12
    generations: int = 20
    range_fraction: float = 0.5
    angle_radius_deg: float = 3.0
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    cognitive: float = 1.5
    social: float = 1.5
    velocity_limit: float = 0.4
    differential_weight: float = 0.7
    crossover_probability: float = 0.9

    def __post_init__(self) -> None:
        if (
            any(
                isinstance(v, bool) or not isinstance(v, Integral)
                for v in (self.population, self.generations)
            )
            or self.population < 4
            or self.generations < 1
        ):
            raise ValueError("Use integer population >= 4 and generations >= 1")
        if not 0 < self.range_fraction < 1 or not 0 < self.angle_radius_deg <= 90:
            raise ValueError("Invalid focusing search radius")
        values = np.array(
            [
                self.inertia_start,
                self.inertia_end,
                self.cognitive,
                self.social,
                self.velocity_limit,
                self.differential_weight,
                self.crossover_probability,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Search coefficients must be finite and nonnegative")
        if self.velocity_limit == 0 or not 0 < self.differential_weight <= 2:
            raise ValueError("Invalid velocity limit or differential weight")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("Crossover probability must lie in [0,1]")


@dataclass(frozen=True)
class PopulationResult:
    position: np.ndarray
    value: float
    evaluations: int
    history: list[dict[str, float | int]]


def initial_population(dimension: int, settings: SearchSettings, seed: int) -> np.ndarray:
    """Identical initial population for PSO/DE, including the paper design at zero."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    population = rng.uniform(-1, 1, (settings.population, dimension))
    population[0] = 0
    return population


def minimize_population(
    objective: Callable[[np.ndarray], float],
    dimension: int,
    *,
    algorithm: str,
    settings: SearchSettings,
    seed: int,
) -> PopulationResult:
    """Bounded synchronous PSO or DE/rand/1/bin with equal evaluation budgets.

    Coordinates are normalized to [-1,1]. Both use clipping at bounds; PSO
    additionally stops outward velocity. Nonfinite fitness is rejected. The
    first, zero-offset candidate must be feasible and remains an incumbent.
    """
    if (
        algorithm not in {"pso", "de"}
        or any(isinstance(v, bool) or not isinstance(v, Integral) for v in (dimension, seed))
        or dimension < 1
        or seed < 0
    ):
        raise ValueError("Use pso/de, positive dimension and nonnegative seed")
    position = initial_population(dimension, settings, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    evaluations = 0

    def evaluate(points: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        costs = []
        for point in points:
            value = float(objective(point.copy()))
            costs.append(value if np.isfinite(value) else np.inf)
            evaluations += 1
        return np.asarray(costs)

    costs = evaluate(position)
    if not np.isfinite(costs[0]):
        raise ValueError("The baseline at zero must be feasible")
    personal, personal_cost = position.copy(), costs.copy()
    velocity = np.zeros_like(position)
    best_index = int(np.argmin(costs))
    best, best_cost = position[best_index].copy(), float(costs[best_index])
    history = [{"generation": 0, "evaluations": evaluations, "best_objective": best_cost}]
    for generation in range(1, settings.generations + 1):
        if algorithm == "pso":
            progress = (generation - 1) / max(1, settings.generations - 1)
            inertia = settings.inertia_start + progress * (
                settings.inertia_end - settings.inertia_start
            )
            velocity = (
                inertia * velocity
                + settings.cognitive * rng.random(position.shape) * (personal - position)
                + settings.social * rng.random(position.shape) * (best - position)
            )
            velocity = np.clip(velocity, -settings.velocity_limit, settings.velocity_limit)
            proposed = position + velocity
            velocity[(proposed < -1) | (proposed > 1)] = 0
            position = np.clip(proposed, -1, 1)
            costs = evaluate(position)
            improved = costs < personal_cost
            personal[improved], personal_cost[improved] = position[improved], costs[improved]
            candidate = int(np.argmin(personal_cost))
            if personal_cost[candidate] < best_cost:
                best, best_cost = personal[candidate].copy(), float(personal_cost[candidate])
        else:
            # All donors and comparisons come from the previous generation.
            # Distinct donors exclude the target, as required by DE/rand/1/bin.
            trials = np.empty_like(position)
            for index in range(settings.population):
                donors = np.delete(np.arange(settings.population), index)
                a, b, c = rng.choice(donors, 3, replace=False)
                mutant = np.clip(
                    position[a] + settings.differential_weight * (position[b] - position[c]),
                    -1,
                    1,
                )
                cross = rng.random(dimension) < settings.crossover_probability
                cross[rng.integers(dimension)] = True
                trials[index] = np.where(cross, mutant, position[index])
            trial_cost = evaluate(trials)
            improved = trial_cost < costs
            position[improved], costs[improved] = trials[improved], trial_cost[improved]
            candidate = int(np.argmin(costs))
            if costs[candidate] < best_cost:
                best, best_cost = position[candidate].copy(), float(costs[candidate])
        history.append(
            {
                "generation": generation,
                "evaluations": evaluations,
                "best_objective": best_cost,
            }
        )
    return PopulationResult(best, best_cost, evaluations, history)


def focusing_coordinates(
    config: SimulationConfig, scenario: Scenario, position: np.ndarray, settings: SearchSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Move virtual beam foci, without moving the actual users or target."""
    extra = config.n_rf_chains - config.n_users
    ranges = np.r_[scenario.user_ranges, np.full(extra, scenario.target_range)]
    angles = np.r_[scenario.user_angles, np.full(extra, scenario.target_angle)]
    position = np.asarray(position, dtype=float)
    if position.shape != (2 * config.n_rf_chains,) or not np.all(np.isfinite(position)):
        raise ValueError("Expected two finite focusing coordinates per RF chain")
    if np.any(np.abs(position) > 1):
        raise ValueError("Focusing coordinates must lie in [-1,1]")
    ranges = ranges * (1 + settings.range_fraction * position[: config.n_rf_chains])
    offsets = position[config.n_rf_chains :]
    radius = np.deg2rad(settings.angle_radius_deg)
    # Piecewise scaling includes the nominal angle even near endfire, without
    # clipping many distinct candidates to the same physical focusing angle.
    angles = angles + offsets * np.where(
        offsets < 0, np.minimum(radius, angles), np.minimum(radius, np.pi - angles)
    )
    return ranges, angles


def focusing_matrix(
    config: SimulationConfig, scenario: Scenario, position: np.ndarray, settings: SearchSettings
) -> np.ndarray:
    ranges, angles = focusing_coordinates(config, scenario, position, settings)
    return np.column_stack(
        [
            near_field_response(config, float(r), float(theta)).conj()
            for r, theta in zip(ranges, angles, strict=True)
        ]
    )


def solve_hybrid_metaheuristic(
    config: SimulationConfig,
    scenario: Scenario,
    *,
    algorithm: str,
    settings: SearchSettings,
    seed: int,
    receive_combiner: np.ndarray,
    baseline: OptimizationResult,
    **solver_options: Any,
) -> tuple[OptimizationResult, dict[str, Any]]:
    """Search analog foci; each fitness is a validated baseband SDR solution."""
    started = perf_counter()
    dimension = 2 * config.n_rf_chains
    baseline_cost = float(baseline.metadata.get("physical_crb_trace", np.nan))
    if (
        not baseline.metadata.get("validation_passed", False)
        or not np.isfinite(baseline_cost)
        or baseline_cost <= 0
    ):
        raise ValueError("A validated positive finite baseline is required")
    if (
        solver_options.get("sensing_model", "near") != "near"
        or baseline.metadata.get("sensing_model") != "near"
    ):
        raise ValueError("The focusing search requires the near-field sensing model")
    # The cached zero candidate must describe this exact RF/receiver design.
    # Otherwise an unrelated incumbent can win and be mislabeled with zero foci.
    nominal = focusing_matrix(config, scenario, np.zeros(dimension), settings)
    if not np.array_equal(baseline.metadata.get("transmit_basis"), nominal):
        raise ValueError("Baseline must use the nominal focusing matrix")
    if receive_combiner is None or not np.array_equal(
        baseline.metadata.get("receive_combiner"), receive_combiner
    ):
        raise ValueError("Baseline and search must use the same receive combiner")
    physical = crb_matrix(
        config,
        baseline.waveform.covariance,
        scenario.target_gain,
        distance=scenario.target_range,
        angle=scenario.target_angle,
        receive_combiner=receive_combiner,
    )
    if not np.isclose(float(np.trace(physical)), baseline_cost, rtol=1e-7, atol=0):
        raise ValueError("Baseline CRB does not match the current sensing scenario")
    requested_rate = solver_options.get("min_rate")
    rate = config.min_rate if requested_rate is None else float(requested_rate)
    baseline_rate = np.min(baseline.rates) - baseline.metadata["minimum_rate_margin"]
    rates = communication_rates(
        scenario.communication_channels,
        baseline.waveform.covariance,
        baseline.waveform.communication_beamformers,
    )
    if (
        not np.isfinite(rate)
        or rate < 0
        or not np.isclose(rate, baseline_rate, rtol=0, atol=1e-10)
        or not np.all(np.isfinite(rates))
        or np.min(rates) < rate - 1e-5
        or not np.allclose(rates, baseline.rates, rtol=1e-7, atol=1e-8)
    ):
        raise ValueError("Baseline rates do not match the current communication constraints")
    used_power = float(np.trace(baseline.waveform.covariance).real)
    baseline_power = used_power + baseline.metadata["transmit_power_margin"]
    if (
        not np.isclose(baseline_power, config.transmit_power, rtol=1e-10, atol=0)
        or used_power > config.transmit_power * (1 + 1e-7)
    ):
        raise ValueError("Baseline does not match the current transmit power budget")
    best_result, best_cost = baseline, baseline_cost
    cache = {np.zeros(dimension).tobytes(): baseline_cost}
    evaluations: list[dict[str, Any]] = []
    solver_calls = 0

    def objective(position: np.ndarray) -> float:
        nonlocal best_result, best_cost, solver_calls
        key = position.tobytes()
        tick = perf_counter()
        record: dict[str, Any] = {
            "evaluation": len(evaluations) + 1,
            "position": position.tolist(),
            "cached": key in cache,
        }
        if key in cache:
            value = cache[key]
        else:
            solver_calls += 1
            try:
                result = solve_hybrid_sdr(
                    config,
                    scenario,
                    receive_combiner=receive_combiner,
                    analog_beamformer=focusing_matrix(config, scenario, position, settings),
                    method_name=f"hybrid-{algorithm}-sdr",
                    **solver_options,
                )
                value = float(result.metadata["physical_crb_trace"])
                if (
                    not result.metadata.get("validation_passed", False)
                    or not np.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("Candidate has no validated positive finite CRB")
                if value < best_cost:
                    best_result, best_cost = result, value
            except (ValueError, RuntimeError) as error:
                value = np.inf
                record["rejection_reason"] = str(error)
            cache[key] = value
        record.update(
            feasible=bool(np.isfinite(value)),
            objective=value if np.isfinite(value) else None,
            wall_seconds=perf_counter() - tick,
        )
        evaluations.append(record)
        return value

    search = minimize_population(
        objective,
        dimension,
        algorithm=algorithm,
        settings=settings,
        seed=seed,
    )
    ranges, angles = focusing_coordinates(config, scenario, search.position, settings)
    result = replace(
        best_result,
        waveform=replace(best_result.waveform, method=f"hybrid-{algorithm}-sdr"),
        metadata={
            **best_result.metadata,
            "focus_ranges_m": ranges,
            "focus_angles_deg": np.rad2deg(angles),
            "search_position": search.position,
            "search_seed": seed,
            "search_algorithm": algorithm,
        },
    )
    details = {
        "algorithm": algorithm,
        "search_seed": seed,
        "baseline_objective": baseline_cost,
        "best_objective": search.value,
        "improvement_percent": 100 * (1 - search.value / baseline_cost),
        "fitness_evaluations": search.evaluations,
        "solver_calls": solver_calls,
        "infeasible_evaluations": sum(not r["feasible"] for r in evaluations),
        "wall_seconds": perf_counter() - started,
        "history": search.history,
        "evaluations": evaluations,
    }
    return result, details
