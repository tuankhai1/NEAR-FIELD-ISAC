"""Reproducible experiment drivers for paper Figures 2--4."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, MutableMapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "near-field-isac-matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from .channels import Scenario, generate_scenario, near_field_response
from .communication import Waveform, communication_rates, zf_sensing_baseline
from .config import SimulationConfig
from .fim import crb_matrix, far_field_angle_crb, root_crb
from .music import (
    complex_normal,
    generate_transmit_samples,
    music_spectrum_xy,
    sample_covariance,
    simulate_echo,
)
from .optimization import (
    OptimizationResult,
    random_hybrid_combiner,
    solve_fully_digital_sdr,
    solve_hybrid_sdr,
)
from .plotting import (
    plot_figure2 as _plot_figure2,
)
from .plotting import (
    plot_figure4 as _plot_figure4,
)
from .plotting import (
    plot_music_pair as _plot_music_pair,
)


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validated_sweep(
    values: Iterable[float], *, name: str, allow_zero: bool
) -> list[float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError(f"{name} must contain at least one value")
    if any(
        not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero)
        for value in samples
    ):
        domain = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain finite, {domain} values")
    return samples


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_solution(output: Path, name: str, result: OptimizationResult) -> str:
    arrays = {
        "covariance": result.waveform.covariance,
        "beamformers": result.waveform.communication_beamformers,
        "sensing_covariance": result.waveform.sensing_covariance,
        "rates": result.rates,
    }
    arrays.update(
        {key: value for key, value in result.metadata.items() if isinstance(value, np.ndarray)}
    )
    path = output / (name + ".npz")
    np.savez_compressed(path, **arrays)
    metadata = {
        key: value for key, value in result.metadata.items() if not isinstance(value, np.ndarray)
    }
    _save_json(
        output / (name + ".json"),
        {
            "solver": result.solver,
            "status": result.status,
            "objective": result.objective,
            **metadata,
        },
    )
    return path.name


def _save_provenance(
    output: Path, config: SimulationConfig, scenario: Scenario, receive_combiner: np.ndarray | None
) -> None:
    import hashlib
    import importlib.metadata
    import platform

    arrays = {
        "communication_channels": scenario.communication_channels,
        "target_channel": scenario.target_channel,
        "target_gain": np.array(scenario.target_gain),
        "target_reflection": np.array(scenario.target_reflection),
        "user_ranges": scenario.user_ranges,
        "user_angles": scenario.user_angles,
    }
    if receive_combiner is not None:
        arrays["receive_combiner"] = receive_combiner
    np.savez_compressed(output / "scenario.npz", **arrays)
    versions = {}
    for package in ["numpy", "scipy", "matplotlib", "cvxpy", "mosek", "clarabel"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    _save_json(
        output / "provenance.json",
        {
            "config": asdict(config),
            "versions": versions,
            "python": platform.python_version(),
            "source_sha256": source_hashes,
            "user_ranges_m": scenario.user_ranges.tolist(),
            "user_angles_deg": np.rad2deg(scenario.user_angles).tolist(),
            "target_gain": [scenario.target_gain.real, scenario.target_gain.imag],
            "target_reflection": [scenario.target_reflection.real, scenario.target_reflection.imag],
            "hybrid_crb_noise_model": "Paper approximation: N * sigma^2 * I",
            "hybrid_music_noise_model": "Exact combined antenna noise with whitening",
            "objective_units": "range variance in m^2 plus angle variance in rad^2",
        },
    )


def _solver_kwargs(
    solver: str,
    verbose: bool,
    tolerance: float,
    max_iterations: int,
    solver_threads: int | None,
) -> dict[str, Any]:
    return {
        "solver": solver,
        "verbose": verbose,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "solver_threads": solver_threads,
    }


def _waveform_for_figure3(
    config: SimulationConfig,
    scenario: Scenario,
    optimizer: str,
    solver_kwargs: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[Waveform, OptimizationResult | None]:
    if optimizer == "zf":
        return zf_sensing_baseline(config, scenario.communication_channels), None
    if optimizer == "sdr":
        result = solve_fully_digital_sdr(config, scenario, **solver_kwargs)
        return result.waveform, result
    if optimizer == "hybrid":
        result = solve_hybrid_sdr(config, scenario, rng=rng, **solver_kwargs)
        return result.waveform, result
    raise ValueError("optimizer must be 'zf', 'sdr', or 'hybrid'")


def reproduce_figure3(
    config: SimulationConfig,
    *,
    output_dir: str | Path,
    optimizer: str = "sdr",
    grid_size: int = 121,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1.0e-11,
    max_iterations: int = 20_000,
    solver_threads: int | None = None,
    precomputed_result: OptimizationResult | None = None,
) -> dict[str, Any]:
    """Reproduce the near-/far-field MUSIC comparison in paper Fig. 3."""

    if grid_size < 21:
        raise ValueError("grid_size must be at least 21")
    output = _prepare_output(output_dir)
    rng = np.random.default_rng(config.seed)
    scenario = generate_scenario(config, rng)
    solver_options = _solver_kwargs(solver, verbose, tolerance, max_iterations, solver_threads)
    if precomputed_result is not None:
        if optimizer != "sdr":
            raise ValueError("a precomputed Figure 3 result requires optimizer='sdr'")
        if precomputed_result.waveform.covariance.shape != (
            config.n_antennas,
            config.n_antennas,
        ):
            raise ValueError("precomputed Figure 3 result is incompatible with config")
        waveform = precomputed_result.waveform
        optimization_result = precomputed_result
    else:
        waveform, optimization_result = _waveform_for_figure3(
            config, scenario, optimizer, solver_options, rng
        )
    rates = communication_rates(
        scenario.communication_channels,
        waveform.covariance,
        waveform.communication_beamformers,
    )
    receive_combiner = None
    if optimization_result is not None:
        receive_combiner = optimization_result.metadata.get("receive_combiner")
    sensing_crb = crb_matrix(
        config,
        waveform.covariance,
        scenario.target_gain,
        receive_combiner=receive_combiner,
    )
    range_rcrb, angle_rcrb = root_crb(sensing_crb)

    transmit = generate_transmit_samples(
        waveform.covariance,
        waveform.communication_beamformers,
        config.coherent_block_length,
        rng,
    )
    common_noise = np.sqrt(config.noise_power) * complex_normal(
        rng, (config.n_antennas, config.coherent_block_length)
    )
    near_echo = simulate_echo(
        config,
        transmit,
        scenario.target_gain,
        rng,
        model="near",
        noise_samples=common_noise,
    )
    far_echo = simulate_echo(
        config,
        transmit,
        scenario.target_gain,
        rng,
        model="far",
        noise_samples=common_noise,
    )
    if receive_combiner is not None:
        near_echo = receive_combiner @ near_echo
        far_echo = receive_combiner @ far_echo
    axis = np.linspace(0.0, 40.0, grid_size)
    near_music = music_spectrum_xy(
        config,
        sample_covariance(near_echo),
        axis,
        axis,
        model="near",
        receive_combiner=receive_combiner,
    )
    far_music = music_spectrum_xy(
        config,
        sample_covariance(far_echo),
        axis,
        axis,
        model="far",
        receive_combiner=receive_combiner,
    )

    _save_provenance(output, config, scenario, receive_combiner)
    if optimization_result is not None:
        _save_solution(output, "nominal", optimization_result)

    _plot_music_pair(
        near_music,
        far_music,
        output / "figure3_music_spectrum.png",
        target_range=scenario.target_range,
        target_angle=scenario.target_angle,
    )
    np.savez_compressed(
        output / "figure3_music_data.npz",
        x_grid=near_music.x_grid,
        y_grid=near_music.y_grid,
        near_spectrum=near_music.spectrum,
        far_spectrum=far_music.spectrum,
        covariance=waveform.covariance,
        beamformers=waveform.communication_beamformers,
        user_ranges=scenario.user_ranges,
        user_angles=scenario.user_angles,
        rates=rates,
    )
    summary: dict[str, Any] = {
        "experiment": "figure3",
        "optimizer": optimizer,
        "preset": {
            "n_antennas": config.n_antennas,
            "n_users": config.n_users,
            "grid_size": grid_size,
            "seed": config.seed,
        },
        "target": {
            "range_m": scenario.target_range,
            "angle_deg": float(np.rad2deg(scenario.target_angle)),
        },
        "near_field_estimate": {
            "range_m": near_music.estimated_range,
            "angle_deg": near_music.estimated_angle_deg,
            "x_m": near_music.estimated_x,
            "y_m": near_music.estimated_y,
        },
        "far_field_grid_maximum": {
            "range_m": far_music.estimated_range,
            "angle_deg": far_music.estimated_angle_deg,
        },
        "communication_rates_bit_s_hz": rates.tolist(),
        "range_rcrb_m": range_rcrb,
        "angle_rcrb_deg": angle_rcrb,
    }
    if optimization_result is not None:
        summary["solver"] = {
            "name": optimization_result.solver,
            "status": optimization_result.status,
            "objective": optimization_result.objective,
        }
    _save_json(output / "figure3_summary.json", summary)
    return summary


def _curve_row(
    config: SimulationConfig,
    scenario: Scenario,
    result: OptimizationResult,
    x_name: str,
    x_value: float,
) -> dict[str, float | str]:
    receive_combiner = result.metadata.get("receive_combiner")
    crb = crb_matrix(
        config,
        result.waveform.covariance,
        scenario.target_gain,
        distance=scenario.target_range,
        angle=scenario.target_angle,
        receive_combiner=receive_combiner,
    )
    range_rcrb, angle_rcrb = root_crb(crb)
    return {
        x_name: x_value,
        "architecture": result.waveform.method,
        "range_rcrb_m": range_rcrb,
        "angle_rcrb_deg": angle_rcrb,
        "minimum_achieved_rate": float(np.min(result.rates)),
        "objective": result.objective,
        "solver": result.solver,
        "solver_status": result.status,
        "physical_crb_trace": result.metadata["physical_crb_trace"],
        "minimum_rate_margin": result.metadata["minimum_rate_margin"],
        "transmit_power_margin": result.metadata["transmit_power_margin"],
        "minimum_sensing_covariance_eigenvalue": result.metadata[
            "minimum_sensing_covariance_eigenvalue"
        ],
        "epigraph_relative_error": result.metadata["epigraph_relative_error"],
        "validation_passed": result.metadata["validation_passed"],
        "_result": result,
    }


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    for index, row in enumerate(rows):
        result = row.pop("_result", None)
        if result is not None:
            row["waveform_file"] = _save_solution(path.parent, f"point_{index:03d}", result)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _solve_figure2_point(
    config: SimulationConfig,
    scenario: Scenario,
    receive_combiner: np.ndarray,
    rate: float,
    solver_options: dict[str, Any],
) -> tuple[
    list[dict[str, float | str]],
    tuple[OptimizationResult, OptimizationResult],
]:
    full = solve_fully_digital_sdr(config, scenario, min_rate=rate, **solver_options)
    hybrid = solve_hybrid_sdr(
        config,
        scenario,
        min_rate=rate,
        receive_combiner=receive_combiner,
        **solver_options,
    )
    rows = [
        _curve_row(config, scenario, full, "minimum_rate", rate),
        _curve_row(config, scenario, hybrid, "minimum_rate", rate),
    ]
    return rows, (full, hybrid)


def reproduce_figure2(
    config: SimulationConfig,
    rates: Iterable[float],
    *,
    output_dir: str | Path,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1.0e-11,
    max_iterations: int = 20_000,
    solver_threads: int | None = None,
    workers: int = 1,
    result_cache: MutableMapping[float, tuple[OptimizationResult, OptimizationResult]]
    | None = None,
) -> dict[str, Any]:
    """Reproduce the sensing/communication tradeoff in paper Fig. 2."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    rates = _validated_sweep(rates, name="rates", allow_zero=True)
    output = _prepare_output(output_dir)
    rng = np.random.default_rng(config.seed)
    scenario = generate_scenario(config, rng)
    receive_combiner = random_hybrid_combiner(config, rng)
    _save_provenance(output, config, scenario, receive_combiner)
    solver_options = _solver_kwargs(solver, verbose, tolerance, max_iterations, solver_threads)
    rows: list[dict[str, Any]] = []
    if workers == 1:
        for index, rate in enumerate(rates, start=1):
            numeric_rate = float(rate)
            print(f"  Figure 2 [{index}/{len(rates)}]: R_min={numeric_rate:g} bit/s/Hz")
            if result_cache is not None and numeric_rate in result_cache:
                full, hybrid = result_cache[numeric_rate]
                point_rows = [
                    _curve_row(config, scenario, full, "minimum_rate", numeric_rate),
                    _curve_row(config, scenario, hybrid, "minimum_rate", numeric_rate),
                ]
            else:
                point_rows, point_results = _solve_figure2_point(
                    config,
                    scenario,
                    receive_combiner,
                    numeric_rate,
                    solver_options,
                )
                if result_cache is not None:
                    result_cache[numeric_rate] = point_results
            rows.extend(point_rows)
    else:
        print(f"  Figure 2: solving {len(rates)} rate points with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _solve_figure2_point,
                    config,
                    scenario,
                    receive_combiner,
                    float(rate),
                    solver_options,
                ): float(rate)
                for rate in rates
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                rate = futures[future]
                point_rows, point_results = future.result()
                rows.extend(point_rows)
                if result_cache is not None:
                    result_cache[rate] = point_results
                print(f"  Figure 2 [{completed}/{len(rates)} completed]: R_min={rate:g} bit/s/Hz")
    architecture_order = {"fully-digital-sdr": 0, "hybrid-two-stage-sdr": 1}
    rows.sort(
        key=lambda row: (
            float(row["minimum_rate"]),
            architecture_order[str(row["architecture"])],
        )
    )

    _save_rows(output / "figure2_rcrb_vs_rate.csv", rows)
    _plot_figure2(rows, output / "figure2_rcrb_vs_rate.png")
    summary = {
        "experiment": "figure2",
        "seed": config.seed,
        "workers": workers,
        "solver_threads": solver_threads,
        "rates": [float(value) for value in rates],
        "rows": rows,
    }
    _save_json(output / "figure2_summary.json", summary)
    return summary


def _scenario_at_range(
    config: SimulationConfig,
    original: Scenario,
    distance: float,
    fixed_target_gain: complex,
) -> Scenario:
    response = near_field_response(config, distance, original.target_angle)
    return replace(
        original,
        target_channel=fixed_target_gain * np.outer(response, response),
        target_gain=fixed_target_gain,
        target_range=distance,
    )


def _solve_figure4_point(
    config: SimulationConfig,
    base_scenario: Scenario,
    receive_combiner: np.ndarray,
    distance: float,
    fixed_target_gain: complex,
    solver_options: dict[str, Any],
) -> list[dict[str, float | str]]:
    distance_config = config.with_updates(target_range=distance)
    scenario = _scenario_at_range(distance_config, base_scenario, distance, fixed_target_gain)
    full = solve_fully_digital_sdr(distance_config, scenario, **solver_options)
    hybrid = solve_hybrid_sdr(
        distance_config,
        scenario,
        receive_combiner=receive_combiner,
        **solver_options,
    )
    return _figure4_rows_from_results(
        distance_config,
        scenario,
        receive_combiner,
        distance,
        full,
        hybrid,
    )


def _figure4_rows_from_results(
    distance_config: SimulationConfig,
    scenario: Scenario,
    receive_combiner: np.ndarray,
    distance: float,
    full: OptimizationResult,
    hybrid: OptimizationResult,
) -> list[dict[str, float | str]]:
    full_row = _curve_row(distance_config, scenario, full, "distance_m", distance)
    hybrid_row = _curve_row(distance_config, scenario, hybrid, "distance_m", distance)
    return [full_row, hybrid_row]


def reproduce_figure4(
    config: SimulationConfig,
    distances: Iterable[float],
    *,
    output_dir: str | Path,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1.0e-11,
    max_iterations: int = 20_000,
    solver_threads: int | None = None,
    workers: int = 1,
    precomputed_results: dict[float, tuple[OptimizationResult, OptimizationResult]] | None = None,
) -> dict[str, Any]:
    """Reproduce the range-dependence experiment in paper Fig. 4.

    The target gain generated at the nominal 20 m location is held fixed over
    the sweep, implementing the paper's instruction to exclude pathloss.
    """

    if workers < 1:
        raise ValueError("workers must be at least 1")
    distances = _validated_sweep(distances, name="distances", allow_zero=False)
    output = _prepare_output(output_dir)
    rng = np.random.default_rng(config.seed)
    base_scenario = generate_scenario(config, rng)
    fixed_target_gain = base_scenario.target_gain
    receive_combiner = random_hybrid_combiner(config, rng)
    _save_provenance(output, config, base_scenario, receive_combiner)
    solver_options = _solver_kwargs(solver, verbose, tolerance, max_iterations, solver_threads)
    rows: list[dict[str, Any]] = []
    if workers == 1:
        for index, distance in enumerate(distances, start=1):
            numeric_distance = float(distance)
            print(f"  Figure 4 [{index}/{len(distances)}]: range={numeric_distance:g} m")
            if precomputed_results is not None and numeric_distance in precomputed_results:
                distance_config = config.with_updates(target_range=numeric_distance)
                scenario = _scenario_at_range(
                    distance_config,
                    base_scenario,
                    numeric_distance,
                    fixed_target_gain,
                )
                full, hybrid = precomputed_results[numeric_distance]
                point_rows = _figure4_rows_from_results(
                    distance_config,
                    scenario,
                    receive_combiner,
                    numeric_distance,
                    full,
                    hybrid,
                )
            else:
                point_rows = _solve_figure4_point(
                    config,
                    base_scenario,
                    receive_combiner,
                    numeric_distance,
                    fixed_target_gain,
                    solver_options,
                )
            rows.extend(point_rows)
    else:
        print(f"  Figure 4: solving {len(distances)} distances with {workers} workers")
        pending_distances: list[float] = []
        for distance in distances:
            numeric_distance = float(distance)
            if precomputed_results is None or numeric_distance not in precomputed_results:
                pending_distances.append(numeric_distance)
                continue
            distance_config = config.with_updates(target_range=numeric_distance)
            scenario = _scenario_at_range(
                distance_config,
                base_scenario,
                numeric_distance,
                fixed_target_gain,
            )
            full, hybrid = precomputed_results[numeric_distance]
            rows.extend(
                _figure4_rows_from_results(
                    distance_config,
                    scenario,
                    receive_combiner,
                    numeric_distance,
                    full,
                    hybrid,
                )
            )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _solve_figure4_point,
                    config,
                    base_scenario,
                    receive_combiner,
                    float(distance),
                    fixed_target_gain,
                    solver_options,
                ): float(distance)
                for distance in pending_distances
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                distance = futures[future]
                rows.extend(future.result())
                print(f"  Figure 4 [{completed}/{len(distances)} completed]: range={distance:g} m")
    architecture_order = {"fully-digital-sdr": 0, "hybrid-two-stage-sdr": 1}
    rows.sort(
        key=lambda row: (
            float(row["distance_m"]),
            architecture_order[str(row["architecture"])],
        )
    )

    print("  Figure 4: optimizing independent far-field angle references")
    far_full = solve_fully_digital_sdr(
        config,
        base_scenario,
        sensing_model="far",
        **solver_options,
    )
    far_hybrid = solve_hybrid_sdr(
        config,
        base_scenario,
        sensing_model="far",
        receive_combiner=receive_combiner,
        **solver_options,
    )
    references = {}
    for result in (far_full, far_hybrid):
        value = far_field_angle_crb(
            config,
            result.waveform.covariance,
            fixed_target_gain,
            receive_combiner=result.metadata.get("receive_combiner"),
        )
        references[result.waveform.method] = float(np.rad2deg(np.sqrt(value)))
        _save_solution(output, "far_" + result.waveform.method, result)
    for row in rows:
        row["far_field_angle_rcrb_deg"] = references[row["architecture"]]
    far_field_reference = {
        "fully_digital_angle_rcrb_deg": references["fully-digital-sdr"],
        "hybrid_angle_rcrb_deg": references["hybrid-two-stage-sdr"],
    }
    _save_rows(output / "figure4_rcrb_vs_distance.csv", rows)
    _plot_figure4(rows, output / "figure4_rcrb_vs_distance.png")
    summary = {
        "experiment": "figure4",
        "seed": config.seed,
        "workers": workers,
        "solver_threads": solver_threads,
        "pathloss_in_sweep": False,
        "far_field_reference": far_field_reference,
        "far_field_reference_convention": (
            "Independent angle-only SDR with far-field target steering; fixed near-field "
            "communication channels, gain, and receive combiner. Hybrid target RF column "
            "uses far-field steering; user RF columns remain near-field. This is an "
            "explicit reconstruction convention, independent of sweep endpoints."
        ),
        "distances_m": [float(value) for value in distances],
        "rows": rows,
    }
    _save_json(output / "figure4_summary.json", summary)
    return summary
