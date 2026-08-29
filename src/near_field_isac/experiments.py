"""Reproducible experiment drivers for paper Figures 2--4."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "near-field-isac-matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .channels import Scenario, generate_scenario, near_field_response
from .communication import Waveform, communication_rates, zf_sensing_baseline
from .config import SimulationConfig
from .fim import crb_matrix, root_crb
from .music import (
    MusicResult,
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


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _solver_kwargs(
    solver: str, verbose: bool, tolerance: float, max_iterations: int
) -> dict[str, Any]:
    return {
        "solver": solver,
        "verbose": verbose,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
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


def _plot_music_pair(
    near: MusicResult,
    far: MusicResult,
    destination: Path,
) -> None:
    figure = plt.figure(figsize=(14, 5.5), constrained_layout=True)
    for index, (result, title) in enumerate(
        ((near, "Near-field ISAC"), (far, "Far-field ISAC")), start=1
    ):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        stride = max(1, result.x_grid.shape[0] // 140)
        spectrum_db = 10.0 * np.log10(np.maximum(result.spectrum, 1.0e-12))
        axis.plot_surface(
            result.x_grid[::stride, ::stride],
            result.y_grid[::stride, ::stride],
            spectrum_db[::stride, ::stride],
            cmap="jet",
            linewidth=0,
            antialiased=True,
        )
        axis.scatter(
            [result.estimated_x],
            [result.estimated_y],
            [0.0],
            color="black",
            marker="x",
            s=45,
        )
        axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="Spectrum (dB)", title=title)
        axis.set_xlim(float(np.min(result.x_grid)), float(np.max(result.x_grid)))
        axis.set_ylim(float(np.min(result.y_grid)), float(np.max(result.y_grid)))
        axis.set_zlim(-50.0, 0.5)
        axis.view_init(elev=35, azim=-125)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def reproduce_figure3(
    config: SimulationConfig,
    *,
    output_dir: str | Path,
    optimizer: str = "zf",
    grid_size: int = 121,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1.0e-5,
    max_iterations: int = 20_000,
) -> dict[str, Any]:
    """Reproduce the near-/far-field MUSIC comparison in paper Fig. 3."""

    if grid_size < 21:
        raise ValueError("grid_size must be at least 21")
    output = _prepare_output(output_dir)
    rng = np.random.default_rng(config.seed)
    scenario = generate_scenario(config, rng)
    solver_options = _solver_kwargs(solver, verbose, tolerance, max_iterations)
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

    _plot_music_pair(near_music, far_music, output / "figure3_music_spectrum.png")
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
    }


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_rcrb_curves(
    rows: list[dict[str, Any]],
    x_name: str,
    x_label: str,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    labels = {
        "fully-digital-sdr": "Fully digital",
        "hybrid-two-stage-sdr": "Hybrid",
    }
    for architecture in labels:
        selected = [row for row in rows if row["architecture"] == architecture]
        selected.sort(key=lambda row: float(row[x_name]))
        if not selected:
            continue
        x = [float(row[x_name]) for row in selected]
        axes[0].semilogy(
            x,
            [float(row["range_rcrb_m"]) for row in selected],
            marker="o",
            label=labels[architecture],
        )
        axes[1].semilogy(
            x,
            [float(row["angle_rcrb_deg"]) for row in selected],
            marker="o",
            label=labels[architecture],
        )
    axes[0].set(xlabel=x_label, ylabel="Range RCRB (m)")
    axes[1].set(xlabel=x_label, ylabel="Angle RCRB (deg)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def reproduce_figure2(
    config: SimulationConfig,
    rates: Iterable[float],
    *,
    output_dir: str | Path,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1.0e-5,
    max_iterations: int = 20_000,
) -> dict[str, Any]:
    """Reproduce the sensing/communication tradeoff in paper Fig. 2."""

    output = _prepare_output(output_dir)
    rates = list(rates)
    rng = np.random.default_rng(config.seed)
    scenario = generate_scenario(config, rng)
    receive_combiner = random_hybrid_combiner(config, rng)
    solver_options = _solver_kwargs(solver, verbose, tolerance, max_iterations)
    rows: list[dict[str, Any]] = []
    for rate in rates:
        full = solve_fully_digital_sdr(
            config, scenario, min_rate=float(rate), **solver_options
        )
        hybrid = solve_hybrid_sdr(
            config,
            scenario,
            min_rate=float(rate),
            receive_combiner=receive_combiner,
            rng=rng,
            **solver_options,
        )
        rows.append(_curve_row(config, scenario, full, "minimum_rate", float(rate)))
        rows.append(_curve_row(config, scenario, hybrid, "minimum_rate", float(rate)))

    _save_rows(output / "figure2_rcrb_vs_rate.csv", rows)
    _plot_rcrb_curves(
        rows,
        "minimum_rate",
        "Minimum communication rate (bit/s/Hz)",
        output / "figure2_rcrb_vs_rate.png",
    )
    summary = {
        "experiment": "figure2",
        "seed": config.seed,
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


def reproduce_figure4(
    config: SimulationConfig,
    distances: Iterable[float],
    *,
    output_dir: str | Path,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1.0e-5,
    max_iterations: int = 20_000,
) -> dict[str, Any]:
    """Reproduce the range-dependence experiment in paper Fig. 4.

    The target gain generated at the nominal 20 m location is held fixed over
    the sweep, implementing the paper's instruction to exclude pathloss.
    """

    output = _prepare_output(output_dir)
    distances = list(distances)
    rng = np.random.default_rng(config.seed)
    base_scenario = generate_scenario(config, rng)
    fixed_target_gain = base_scenario.target_gain
    receive_combiner = random_hybrid_combiner(config, rng)
    solver_options = _solver_kwargs(solver, verbose, tolerance, max_iterations)
    rows: list[dict[str, Any]] = []
    for distance in distances:
        distance = float(distance)
        distance_config = config.with_updates(target_range=distance)
        scenario = _scenario_at_range(
            distance_config, base_scenario, distance, fixed_target_gain
        )
        full = solve_fully_digital_sdr(distance_config, scenario, **solver_options)
        hybrid = solve_hybrid_sdr(
            distance_config,
            scenario,
            receive_combiner=receive_combiner,
            rng=rng,
            **solver_options,
        )
        rows.append(_curve_row(distance_config, scenario, full, "distance_m", distance))
        rows.append(_curve_row(distance_config, scenario, hybrid, "distance_m", distance))

    _save_rows(output / "figure4_rcrb_vs_distance.csv", rows)
    _plot_rcrb_curves(
        rows,
        "distance_m",
        "Target distance (m)",
        output / "figure4_rcrb_vs_distance.png",
    )
    summary = {
        "experiment": "figure4",
        "seed": config.seed,
        "pathloss_in_sweep": False,
        "distances_m": [float(value) for value in distances],
        "rows": rows,
    }
    _save_json(output / "figure4_summary.json", summary)
    return summary
