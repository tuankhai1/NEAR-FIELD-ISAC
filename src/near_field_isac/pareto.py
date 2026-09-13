"""Balanced, component-capped hybrid search with phase modes and local refinement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from numbers import Integral
from time import perf_counter
from typing import Any

import numpy as np

from .channels import Scenario
from .communication import communication_rates
from .config import SimulationConfig
from .fim import crb_matrix
from .metaheuristics import SearchSettings, focusing_matrix
from .optimization import OptimizationResult, solve_hybrid_sdr

CAP_TOLERANCE = 1e-7


@dataclass(frozen=True)
class ParetoSettings:
    population: int = 12
    generations: int = 20
    phase_modes: int = 3
    phase_radius: float = 0.35
    polish_evaluations: int = 64
    restart_patience: int = 4
    range_fraction: float = 0.5
    angle_radius_deg: float = 3.0

    def __post_init__(self) -> None:
        for name in ("population", "generations", "phase_modes", "polish_evaluations",
                     "restart_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.population < 4 or self.generations < 1 or self.restart_patience < 1:
            raise ValueError("Use population >= 4, generations >= 1 and restart_patience >= 1")
        if not np.isfinite(self.phase_radius) or not 0 < self.phase_radius <= np.pi:
            raise ValueError("phase_radius must lie in (0, pi]")
        self.focusing_settings()

    def focusing_settings(self) -> SearchSettings:
        return SearchSettings(population=self.population, generations=self.generations,
                              range_fraction=self.range_fraction,
                              angle_radius_deg=self.angle_radius_deg)

    @property
    def budget(self) -> int:
        return self.population * (self.generations + 1) + self.polish_evaluations


def phase_basis(config: SimulationConfig, modes: int) -> np.ndarray:
    """Nonconstant cosine modes; constant column phase is redundant with baseband."""
    if isinstance(modes, bool) or not isinstance(modes, Integral) or modes < 0:
        raise ValueError("phase_modes must be a nonnegative integer")
    if modes >= config.n_antennas:
        raise ValueError("phase_modes must be smaller than the antenna count")
    elements = np.arange(config.n_antennas) + 0.5
    return np.cos(np.pi * elements[:, None] * np.arange(1, modes + 1) / config.n_antennas)


def search_dimension(config: SimulationConfig, settings: ParetoSettings) -> int:
    phase_basis(config, settings.phase_modes)
    return config.n_rf_chains * (2 + settings.phase_modes)


def phase_focusing_matrix(config: SimulationConfig, scenario: Scenario,
                          position: np.ndarray, settings: ParetoSettings) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    if (position.shape != (search_dimension(config, settings),)
            or not np.isfinite(position).all() or np.any(np.abs(position) > 1)):
        raise ValueError("Invalid phase/focusing search position")
    split = 2 * config.n_rf_chains
    analog = focusing_matrix(config, scenario, position[:split], settings.focusing_settings())
    coefficients = position[split:].reshape(config.n_rf_chains, settings.phase_modes)
    phases = settings.phase_radius * phase_basis(config, settings.phase_modes) @ coefficients.T
    return analog * np.exp(1j * phases)


def normalized_score(crb: np.ndarray, reference: np.ndarray) -> float:
    crb = np.asarray(crb, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (
        crb.shape != (2, 2)
        or reference.shape != (2,)
        or not np.isfinite(crb).all()
        or not np.isfinite(reference).all()
        or np.any(reference <= 0)
    ):
        raise ValueError("CRB and reference must contain two finite physical variances")
    ratios = np.diag(crb) / reference
    if (not np.isfinite(ratios).all() or np.any(ratios <= 0)
            or np.max(ratios) > 1 + CAP_TOLERANCE):
        raise ValueError("Candidate exceeds a baseline CRB component")
    return float(np.mean(ratios))


def improve_population(objective: Callable[[np.ndarray], float], dimension: int, *,
                       algorithm: str, settings: ParetoSettings, seed: int) -> dict[str, Any]:
    """Paired multiscale initialization, incumbent-safe restarts, equal local budgets."""
    if (algorithm not in {"pso", "de"} or isinstance(seed, bool)
            or not isinstance(seed, Integral) or seed < 0 or dimension < 1):
        raise ValueError("Use pso/de, a nonnegative integer seed and positive dimension")
    initial_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    positions = initial_rng.uniform(-1, 1, (settings.population, dimension))
    positions[1:] *= np.geomspace(0.03, 1, settings.population - 1)[:, None]
    positions[0] = 0
    initial = positions.copy()
    count = 0

    def evaluate(points):
        nonlocal count
        values = np.array([float(objective(p.copy())) for p in points])
        count += len(points)
        return np.where(np.isfinite(values), values, np.inf)

    costs = evaluate(positions)
    if not np.isfinite(costs[0]):
        raise ValueError("Zero-position baseline must be feasible")
    index = int(np.argmin(costs))
    best, value = positions[index].copy(), float(costs[index])
    personal, personal_cost = positions.copy(), costs.copy()
    velocity = rng.uniform(-0.05, 0.05, positions.shape)
    history = [dict(stage="population", iteration=0, evaluations=count, best_objective=value)]
    stalled, restarts = 0, []
    for generation in range(1, settings.generations + 1):
        if algorithm == "pso":
            inertia = 0.9 - 0.5 * (generation - 1) / max(1, settings.generations - 1)
            velocity = np.clip(
                inertia * velocity
                + 1.5 * rng.random(positions.shape) * (personal - positions)
                + 1.5 * rng.random(positions.shape) * (best - positions), -0.4, 0.4,
            )
            proposed = positions + velocity
            velocity[np.abs(proposed) > 1] = 0
            proposed = np.clip(proposed, -1, 1)
        else:
            proposed = np.empty_like(positions)
            for i in range(settings.population):
                a, b, c = rng.choice(np.delete(np.arange(settings.population), i), 3, replace=False)
                donor = np.clip(positions[a] + rng.uniform(0.5, 0.9)
                                * (positions[b] - positions[c]), -1, 1)
                cross = rng.random(dimension) < 0.9
                cross[rng.integers(dimension)] = True
                proposed[i] = np.where(cross, donor, positions[i])
        restarted = np.zeros(settings.population, dtype=bool)
        if stalled >= settings.restart_patience:
            worst = np.argsort(costs)[-max(1, settings.population // 3):]
            for j, i in enumerate(worst):
                proposed[i] = (rng.uniform(-1, 1, dimension) if j % 2 else
                               np.clip(best + rng.normal(0, 0.15, dimension), -1, 1))
                velocity[i] = rng.uniform(-0.05, 0.05, dimension)
            restarted[worst] = True
            restarts.append(dict(generation=generation, individuals=worst.tolist()))
            stalled = 0
        trials = evaluate(proposed)
        if algorithm == "pso":
            positions, costs = proposed, trials
            improved = (costs < personal_cost) | restarted
            personal[improved], personal_cost[improved] = positions[improved], costs[improved]
        else:
            improved = (trials < costs) | restarted
            positions[improved], costs[improved] = proposed[improved], trials[improved]
        index = int(np.argmin(trials))
        previous = value
        if trials[index] < value:
            best, value = proposed[index].copy(), float(trials[index])
        stalled = 0 if value < previous - 1e-10 else stalled + 1
        history.append(dict(stage="population", iteration=generation,
                            evaluations=count, best_objective=value))

    # Paired coordinate proposals use the same origin for +/- directions.
    # Refinement never discards an incumbent and is counted in the total budget.
    polish_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    step, order, improved_sweep = 0.1, polish_rng.permutation(dimension), False
    origin, coordinate = best.copy(), int(order[0])
    for evaluation in range(settings.polish_evaluations):
        pair = evaluation // 2
        if evaluation % 2 == 0:
            if pair and pair % dimension == 0:
                step = step if improved_sweep else step / 2
                improved_sweep = False
                order = polish_rng.permutation(dimension)
            origin, coordinate = best.copy(), int(order[pair % dimension])
        candidate = origin.copy()
        candidate[coordinate] = np.clip(candidate[coordinate]
                                        + (step if evaluation % 2 == 0 else -step), -1, 1)
        cost = float(evaluate([candidate])[0])
        if cost < value:
            best, value, improved_sweep = candidate, cost, True
        history.append(dict(stage="polish", iteration=evaluation + 1,
                            evaluations=count, best_objective=value))
    return dict(position=best, value=value, evaluations=count, history=history,
                initial_population=initial.tolist(), restarts=restarts)


def _validate_baseline(
    config: SimulationConfig,
    scenario: Scenario,
    result: OptimizationResult,
    *,
    nominal: np.ndarray,
    receive_combiner: np.ndarray,
    rate: float,
) -> np.ndarray:
    """Reject a cached incumbent that belongs to another physical problem."""
    metadata = result.metadata
    if (
        not metadata.get("validation_passed")
        or metadata.get("sensing_model") != "near"
        or not np.array_equal(metadata.get("transmit_basis"), nominal)
        or not np.array_equal(metadata.get("receive_combiner"), receive_combiner)
    ):
        raise ValueError("Search requires validated matching nominal HB baselines")
    covariance = result.waveform.covariance
    beams = result.waveform.communication_beamformers
    rates = communication_rates(scenario.communication_channels, covariance, beams)
    if (
        not np.all(np.isfinite(rates))
        or np.min(rates) < rate - 1e-5
        or not np.allclose(rates, result.rates, rtol=1e-7, atol=1e-8)
        or np.trace(covariance).real > config.transmit_power * (1 + 1e-7)
    ):
        raise ValueError("Baseline does not match the current communication constraints")
    actual = crb_matrix(
        config,
        covariance,
        scenario.target_gain,
        distance=scenario.target_range,
        angle=scenario.target_angle,
        receive_combiner=receive_combiner,
    )
    reported = np.asarray(metadata.get("physical_crb"), dtype=float)
    if reported.shape != (2, 2) or not np.allclose(actual, reported, rtol=1e-7, atol=0):
        raise ValueError("Baseline CRB does not match the search scenario")
    return actual


def solve_pareto_hybrid(
    config: SimulationConfig,
    scenario: Scenario,
    *,
    algorithm: str,
    settings: ParetoSettings,
    seed: int,
    baseline: OptimizationResult,
    balanced_baseline: OptimizationResult,
    receive_combiner: np.ndarray,
    **solver_options: Any,
) -> tuple[OptimizationResult, dict[str, Any]]:
    """Optimize analog hardware while keeping both physical CRBs below original HB."""
    if any(key in solver_options for key in ("crb_reference", "method_name")):
        raise ValueError("The balanced search controls its own CRB reference and method name")
    if solver_options.get("sensing_model", "near") != "near":
        raise ValueError("The balanced search requires the near-field sensing model")
    started = perf_counter()
    dimension = search_dimension(config, settings)
    nominal = phase_focusing_matrix(config, scenario, np.zeros(dimension), settings)
    rate = config.min_rate if solver_options.get("min_rate") is None else float(
        solver_options["min_rate"]
    )
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("Minimum rate must be finite and nonnegative")
    original_crb = _validate_baseline(
        config,
        scenario,
        baseline,
        nominal=nominal,
        receive_combiner=receive_combiner,
        rate=rate,
    )
    balanced_crb = _validate_baseline(
        config,
        scenario,
        balanced_baseline,
        nominal=nominal,
        receive_combiner=receive_combiner,
        rate=rate,
    )
    reference = np.diag(original_crb).copy()
    initial_cost = normalized_score(balanced_crb, reference)
    best_result, best_cost = balanced_baseline, initial_cost
    cache = {np.zeros(dimension).tobytes(): initial_cost}
    evaluations = []

    def objective(position):
        nonlocal best_result, best_cost
        tick = perf_counter()
        key = position.tobytes()
        entry = dict(
            evaluation=len(evaluations) + 1,
            position=position.tolist(),
            cached=key in cache,
        )
        if key in cache:
            cost = cache[key]
        else:
            try:
                result = solve_hybrid_sdr(
                    config, scenario, receive_combiner=receive_combiner,
                    analog_beamformer=phase_focusing_matrix(config, scenario, position, settings),
                    crb_reference=reference, method_name=f"hybrid-pareto-{algorithm}",
                    **solver_options,
                )
                if not result.metadata.get("validation_passed"):
                    raise ValueError("Candidate failed physical validation")
                cost = normalized_score(result.metadata["physical_crb"], reference)
                if cost < best_cost:
                    best_result, best_cost = result, cost
            except (ValueError, RuntimeError) as error:
                cost = np.inf
                entry["rejection_reason"] = str(error)
            cache[key] = cost
        entry.update(objective=float(cost) if np.isfinite(cost) else None,
                     feasible=bool(np.isfinite(cost)), wall_seconds=perf_counter() - tick)
        evaluations.append(entry)
        return cost

    search = improve_population(
        objective,
        dimension,
        algorithm=algorithm,
        settings=settings,
        seed=seed,
    )
    result = replace(
        best_result,
        waveform=replace(best_result.waveform, method=f"hybrid-pareto-{algorithm}"),
        metadata={
            **best_result.metadata,
            "search_position": search["position"],
            "crb_reference": reference,
            "search_seed": seed,
            "normalized_crb_objective": best_cost,
        },
    )
    detail = {k: v for k, v in search.items() if k not in {"position", "value", "evaluations"}}
    detail.update(
        algorithm=algorithm,
        search_seed=seed,
        dimension=dimension,
        original_baseline_objective=1.0,
        balanced_baseline_objective=initial_cost,
        baseline_objective=initial_cost,
        best_objective=best_cost,
        fitness_evaluations=len(evaluations),
        solver_calls=sum(not e["cached"] for e in evaluations),
        infeasible_evaluations=sum(not e["feasible"] for e in evaluations),
        evaluations=evaluations,
        wall_seconds=perf_counter() - started,
    )
    return result, detail
