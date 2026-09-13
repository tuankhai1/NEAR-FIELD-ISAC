"""Component-capped PSO/DE comparison kept separate from the paper objective."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .channels import generate_scenario
from .config import SimulationConfig
from .experiments import _save_provenance, _save_solution, _scenario_at_range
from .fim import root_crb
from .meta_experiment import distance_folder_name
from .optimization import random_hybrid_combiner, solve_fully_digital_sdr, solve_hybrid_sdr
from .pareto import ParetoSettings, solve_pareto_hybrid
from .plotting import PARETO_METHODS, plot_metaheuristic_comparison, plot_metaheuristic_convergence


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_pareto_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report means across restarts; never retain only a favorable seed."""
    aggregates = []
    for distance in sorted({row["distance_m"] for row in rows}):
        for method, *_ in PARETO_METHODS:
            part = [
                row for row in rows if row["distance_m"] == distance and row["method"] == method
            ]
            item = {"distance_m": distance, "method": method, "runs": len(part)}
            for key in (
                "range_rcrb_m",
                "angle_rcrb_deg",
                "physical_crb_trace",
                "normalized_range_variance",
                "normalized_angle_variance",
                "normalized_crb_objective",
                "wall_seconds",
                "fitness_evaluations",
                "solver_calls",
                "infeasible_evaluations",
            ):
                values = np.asarray([row[key] for row in part], dtype=float)
                item[key + "_mean"] = float(np.mean(values))
                item[key + "_std"] = float(np.std(values, ddof=1)) if len(part) > 1 else 0.0
            item["maximum_component_ratio"] = float(
                max(row["maximum_component_ratio"] for row in part)
            )
            aggregates.append(item)
    return aggregates


def _distance_point(
    config: SimulationConfig,
    distance: float,
    settings: ParetoSettings,
    search_seeds: list[int],
    output: Path,
    solver_options: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(config.seed)
    original = generate_scenario(config, rng)
    combiner = random_hybrid_combiner(config, rng)
    local_config = config.with_updates(target_range=distance)
    scenario = _scenario_at_range(local_config, original, distance, original.target_gain)
    folder = output / distance_folder_name(distance)
    folder.mkdir(exist_ok=True)
    rows, convergence = [], []

    def record(result, elapsed, reference, *, seed=None, details=None):
        physical_crb = np.asarray(result.metadata["physical_crb"], dtype=float)
        variances = np.diag(physical_crb)
        ratios = variances / reference
        range_bound, angle_bound = root_crb(physical_crb)
        method = result.waveform.method
        name = method + (f"_seed_{seed}" if seed is not None else "")
        waveform_file = _save_solution(folder, name, result)
        row = {
            "distance_m": distance,
            "method": method,
            "search_seed": seed,
            "range_rcrb_m": range_bound,
            "angle_rcrb_deg": angle_bound,
            "physical_crb_trace": result.metadata["physical_crb_trace"],
            "normalized_range_variance": float(ratios[0]),
            "normalized_angle_variance": float(ratios[1]),
            "normalized_crb_objective": float(np.mean(ratios)),
            "maximum_component_ratio": float(np.max(ratios)),
            "minimum_achieved_rate": float(np.min(result.rates)),
            "minimum_rate_margin": result.metadata["minimum_rate_margin"],
            "transmit_power_margin": result.metadata["transmit_power_margin"],
            "validation_passed": result.metadata["validation_passed"],
            "wall_seconds": elapsed,
            "fitness_evaluations": details["fitness_evaluations"] if details else 1,
            "solver_calls": details["solver_calls"] if details else 1,
            "infeasible_evaluations": details["infeasible_evaluations"] if details else 0,
            "waveform_file": (folder.relative_to(output) / waveform_file).as_posix(),
        }
        if details:
            _write_json(folder / f"{name}_search.json", details)
            for point in details["history"]:
                convergence.append(
                    {
                        "distance_m": distance,
                        "method": method,
                        "search_seed": seed,
                        **point,
                        "objective_over_original_hb": point["best_objective"],
                    }
                )
        rows.append(row)

    started = perf_counter()
    full = solve_fully_digital_sdr(local_config, scenario, **solver_options)
    # The original HB, rather than FD, defines the hardware-matched component caps.
    full_elapsed = perf_counter() - started
    started = perf_counter()
    hybrid = solve_hybrid_sdr(
        local_config,
        scenario,
        receive_combiner=combiner,
        **solver_options,
    )
    hybrid_elapsed = perf_counter() - started
    reference = np.diag(np.asarray(hybrid.metadata["physical_crb"], dtype=float)).copy()
    record(full, full_elapsed, reference)
    record(hybrid, hybrid_elapsed, reference)

    started = perf_counter()
    balanced = solve_hybrid_sdr(
        local_config,
        scenario,
        receive_combiner=combiner,
        crb_reference=reference,
        method_name="hybrid-balanced-sdr",
        **solver_options,
    )
    record(balanced, perf_counter() - started, reference)
    for seed in search_seeds:
        for algorithm in ("pso", "de"):
            result, details = solve_pareto_hybrid(
                local_config,
                scenario,
                algorithm=algorithm,
                settings=settings,
                seed=seed,
                baseline=hybrid,
                balanced_baseline=balanced,
                receive_combiner=combiner,
                **solver_options,
            )
            record(result, details["wall_seconds"], reference, seed=seed, details=details)
            print(
                f"  r={distance:g} m, capped {algorithm.upper()}, seed={seed}: "
                f"mean normalized variance {details['best_objective']:.5f}, "
                f"{details['solver_calls']} SDR calls, {details['wall_seconds']:.1f} s",
                flush=True,
            )
    _write_json(folder / "rows.json", rows)
    return rows, convergence


def reproduce_pareto(
    config: SimulationConfig,
    distances: list[float],
    *,
    output_dir: str | Path,
    settings: ParetoSettings,
    search_seeds: list[int],
    workers: int = 1,
    **solver_options: Any,
) -> dict[str, Any]:
    """Run a balanced, component-capped PSO/DE Figure-4 follow-up experiment."""
    distances = [float(distance) for distance in distances]
    if (
        not distances
        or not np.all(np.isfinite(distances))
        or min(distances) <= 0
        or len(set(distances)) != len(distances)
    ):
        raise ValueError("Distances must be finite, positive and distinct")
    if (
        not search_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in search_seeds)
        or min(search_seeds) < 0
        or len(set(search_seeds)) != len(search_seeds)
        or isinstance(workers, bool)
        or not isinstance(workers, Integral)
        or workers < 1
    ):
        raise ValueError("Use distinct nonnegative integer search seeds and integer workers >= 1")
    search_seeds = [int(seed) for seed in search_seeds]
    workers = int(workers)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    scenario = generate_scenario(config, rng)
    combiner = random_hybrid_combiner(config, rng)
    _save_provenance(output, config, scenario, combiner)
    setup = {
        "search_settings": asdict(settings),
        "search_seeds": search_seeds,
        "channel_seed": config.seed,
        "distances_m": distances,
        "workers": workers,
        "solver_options": solver_options,
        "dimension": settings.phase_modes * config.n_rf_chains + 2 * config.n_rf_chains,
        "pathloss_in_sweep": False,
        "objective": (
            "Mean(range CRB / original HB range CRB, angle CRB / original HB angle CRB)"
        ),
        "constraints": "Both physical CRB components must not exceed original HB",
        "search_space": "Virtual RF focusing plus nonconstant per-chain cosine phase modes",
        "comparison": "Same channel, target gain, receive combiner and inner SDR constraints",
        "budget": "population * (generations + 1) + polish_evaluations fitness requests",
        "initialization": "PSO/DE share a multiscale population; zero is the balanced HB solve",
        "aggregation": "Mean +/- sample standard deviation over search seeds, one channel",
    }
    _write_json(output / "settings.json", setup)
    print(
        f"Component-capped comparison: N={config.n_antennas}, {len(distances)} distances, "
        f"{len(search_seeds)} search seeds, {settings.budget} evaluations/search.",
        flush=True,
    )
    started = perf_counter()
    rows, convergence = [], []
    if workers == 1:
        for distance in distances:
            point, history = _distance_point(
                config,
                distance,
                settings,
                search_seeds,
                output,
                solver_options,
            )
            rows.extend(point)
            convergence.extend(history)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _distance_point,
                    config,
                    distance,
                    settings,
                    search_seeds,
                    output,
                    solver_options,
                )
                for distance in distances
            ]
            for future in as_completed(futures):
                point, history = future.result()
                rows.extend(point)
                convergence.extend(history)
    elapsed = perf_counter() - started
    rows.sort(key=lambda row: (row["distance_m"], row["method"], row["search_seed"] or -1))
    convergence.sort(
        key=lambda row: (
            row["distance_m"],
            row["method"],
            row["search_seed"],
            row["stage"],
            row["iteration"],
        )
    )
    aggregated = aggregate_pareto_results(rows)
    _write_csv(output / "runs.csv", rows)
    _write_csv(output / "comparison.csv", aggregated)
    _write_csv(output / "convergence.csv", convergence)
    plot_metaheuristic_comparison(
        aggregated,
        output / "figure4_pareto_comparison.png",
        minimum_rate=config.min_rate,
        methods=PARETO_METHODS,
    )
    plot_metaheuristic_convergence(
        convergence,
        output / "pareto_convergence.png",
        methods=PARETO_METHODS[3:],
        baseline_label="Original HB component cap",
        ylabel="Best mean normalized CRB",
        unit_interval=True,
    )
    summary = {
        "experiment": "pareto",
        "config": asdict(config),
        **setup,
        "total_wall_seconds": elapsed,
        "rows": rows,
        "aggregated": aggregated,
    }
    _write_json(output / "summary.json", summary)
    lines = [
        "# Component-capped PSO/DE hybrid comparison",
        "",
        "The original hybrid beamformer defines separate range and angle variance caps. "
        "A candidate is retained only if it meets both caps; the search objective is their mean "
        "normalized variance. Values at or below one in either normalized-component column meet "
        "the corresponding cap.",
        "",
        "| Distance (m) | Method | Range RCRB (m) | Angle RCRB (deg) | "
        "Mean normalized CRB | Max component ratio |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in aggregated:
        lines.append(
            f"| {row['distance_m']:g} | {row['method']} | {row['range_rcrb_m_mean']:.7g} | "
            f"{row['angle_rcrb_deg_mean']:.7g} | "
            f"{row['normalized_crb_objective_mean']:.6g} | "
            f"{row['maximum_component_ratio']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The fully digital row is a reference comparison and is not component-capped. "
            "Original HB is exactly one by construction. The balanced and capped-search rows "
            "must have maximum component ratio no greater than one, within numerical tolerance.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "experiment": "pareto",
        "summary_file": str(output / "summary.json"),
        "comparison_figure": str(output / "figure4_pareto_comparison.png"),
        "total_wall_seconds": elapsed,
        "completed_runs": len(rows),
    }
