"""Matched Figure-4 comparison: original FD/HB versus PSO+SDR and DE+SDR."""

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
from .metaheuristics import SearchSettings, solve_hybrid_metaheuristic
from .optimization import random_hybrid_combiner, solve_fully_digital_sdr, solve_hybrid_sdr
from .plotting import plot_metaheuristic_comparison, plot_metaheuristic_convergence


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distance_folder_name(distance: float) -> str:
    """Retain enough digits to keep distinct floating-point distances separate."""
    return f"distance_{str(float(distance)).removesuffix('.0')}m"


def _distance_point(
    config: SimulationConfig,
    distance: float,
    settings: SearchSettings,
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

    def record(result, elapsed, seed=None, details=None):
        range_bound, angle_bound = root_crb(result.metadata["physical_crb"])
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
                        "objective_over_original_hb": (
                            point["best_objective"] / details["baseline_objective"]
                        ),
                    }
                )
        rows.append(row)

    started = perf_counter()
    full = solve_fully_digital_sdr(local_config, scenario, **solver_options)
    record(full, perf_counter() - started)
    started = perf_counter()
    hybrid = solve_hybrid_sdr(
        local_config,
        scenario,
        receive_combiner=combiner,
        **solver_options,
    )
    record(hybrid, perf_counter() - started)
    for seed in search_seeds:
        for algorithm in ("pso", "de"):
            result, details = solve_hybrid_metaheuristic(
                local_config,
                scenario,
                algorithm=algorithm,
                settings=settings,
                seed=seed,
                receive_combiner=combiner,
                baseline=hybrid,
                **solver_options,
            )
            record(result, details["wall_seconds"], seed, details)
            print(
                f"  r={distance:g} m, {algorithm.upper()}, seed={seed}: "
                f"trace CRB reduction {details['improvement_percent']:.2f}%, "
                f"{details['solver_calls']} SDR calls, {details['wall_seconds']:.1f} s",
                flush=True,
            )
    _write_json(folder / "rows.json", rows)
    return rows, convergence


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and sample SD over search restarts; never select a lucky restart."""
    aggregated = []
    for distance in sorted({r["distance_m"] for r in rows}):
        baseline = next(
            r["physical_crb_trace"]
            for r in rows
            if r["distance_m"] == distance and r["method"] == "hybrid-two-stage-sdr"
        )
        for method in (
            "fully-digital-sdr",
            "hybrid-two-stage-sdr",
            "hybrid-pso-sdr",
            "hybrid-de-sdr",
        ):
            part = [r for r in rows if r["distance_m"] == distance and r["method"] == method]
            item = {"distance_m": distance, "method": method, "runs": len(part)}
            for key in (
                "range_rcrb_m",
                "angle_rcrb_deg",
                "physical_crb_trace",
                "wall_seconds",
                "fitness_evaluations",
                "solver_calls",
                "infeasible_evaluations",
            ):
                values = np.array([r[key] for r in part])
                item[key + "_mean"] = float(np.mean(values))
                item[key + "_std"] = float(np.std(values, ddof=1)) if len(part) > 1 else 0.0
            item["trace_reduction_vs_original_hb_percent"] = 100 * (
                1 - item["physical_crb_trace_mean"] / baseline
            )
            aggregated.append(item)
    return aggregated


def reproduce_metaheuristics(
    config: SimulationConfig,
    distances: list[float],
    *,
    output_dir: str | Path,
    settings: SearchSettings,
    search_seeds: list[int],
    workers: int = 1,
    **solver_options: Any,
) -> dict[str, Any]:
    """New experiment folder; original Figures 2--4 and their data remain intact."""
    distances = [float(r) for r in distances]
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
    original = generate_scenario(config, rng)
    combiner = random_hybrid_combiner(config, rng)
    _save_provenance(output, config, original, combiner)
    setup = {
        "search_settings": asdict(settings),
        "search_seeds": search_seeds,
        "channel_seed": config.seed,
        "distances_m": distances,
        "workers": workers,
        "solver_options": solver_options,
        "dimension": 2 * config.n_rf_chains,
        "pathloss_in_sweep": False,
        "objective": "CRB_range [m^2] + CRB_angle [rad^2]",
        "search_space": "Virtual RF focusing ranges +/-50% and angles +/-3 deg by default",
        "comparison": "Same channel, target gain, receive combiner and inner SDR constraints",
        "budget": "population * (generations + 1) fitness requests per algorithm/restart",
        "initialization": "Identical PSO/DE population for each seed; paper HB included",
        "aggregation": "Mean +/- sample standard deviation over search seeds, one channel",
        "timing": "Per-search wall time excludes shared original HB solve; includes inner SDRs",
        "references": {
            "paper": "https://arxiv.org/html/2302.01153v5#S3",
            "pso": "https://doi.org/10.1109/ICNN.1995.488968",
            "de": "https://doi.org/10.1023/A:1008202821328",
        },
    }
    _write_json(output / "settings.json", setup)
    print(
        f"Metaheuristic comparison: N={config.n_antennas}, "
        f"{len(distances)} distances, {len(search_seeds)} search seeds, "
        f"{settings.population * (settings.generations + 1)} evaluations/search.",
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
    rows.sort(key=lambda r: (r["distance_m"], r["method"], r["search_seed"] or -1))
    convergence.sort(
        key=lambda r: (r["distance_m"], r["method"], r["search_seed"], r["generation"])
    )
    aggregated = aggregate_results(rows)
    _write_csv(output / "runs.csv", rows)
    _write_csv(output / "comparison.csv", aggregated)
    _write_csv(output / "convergence.csv", convergence)
    plot_metaheuristic_comparison(
        aggregated, output / "figure4_metaheuristic_comparison.png", minimum_rate=config.min_rate,
    )
    plot_metaheuristic_convergence(convergence, output / "metaheuristic_convergence.png")
    summary = {
        "experiment": "metaheuristics",
        "config": asdict(config),
        **setup,
        "total_wall_seconds": elapsed,
        "rows": rows,
        "aggregated": aggregated,
    }
    _write_json(output / "summary.json", summary)
    lines = [
        "# PSO/DE versus original Figure 4",
        "",
        "Mean over independent search restarts on the same channel realization; "
        "shading in the figure is one sample standard deviation. Smaller CRBs are better.",
        "",
        "| Distance (m) | Method | Range RCRB (m) | Angle RCRB (deg) | "
        "Trace reduction vs HB (%) | Search/solve time (s) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in aggregated:
        lines.append(
            f"| {row['distance_m']:g} | {row['method']} | {row['range_rcrb_m_mean']:.7g} | "
            f"{row['angle_rcrb_deg_mean']:.7g} | "
            f"{row['trace_reduction_vs_original_hb_percent']:.3f} | "
            f"{row['wall_seconds_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Search runtimes exclude the shared HB initialization solve. "
            "FD/HB entries show a single solve. Timings reflect concurrent execution.",
            "",
            "All methods minimize the original mixed-unit trace; neither individual RCRB "
            "is guaranteed to improve. This comparison does not resolve the original "
            "paper's absolute-scale discrepancy and is not a channel Monte Carlo study.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "experiment": "metaheuristics",
        "summary_file": str(output / "summary.json"),
        "comparison_figure": str(output / "figure4_metaheuristic_comparison.png"),
        "total_wall_seconds": elapsed,
        "completed_runs": len(rows),
    }
