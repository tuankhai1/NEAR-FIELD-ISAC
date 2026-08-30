"""Reproducible experiment drivers for paper Figures 2--4."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, MutableMapping
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

from .channels import Scenario, generate_scenario, near_field_response
from .communication import Waveform, communication_rates, zf_sensing_baseline
from .config import SimulationConfig
from .fim import crb_matrix, far_field_angle_crb, root_crb
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


def _plot_music_pair(
    near: MusicResult,
    far: MusicResult,
    destination: Path,
    *,
    target_range: float,
    target_angle: float,
) -> None:
    figure = plt.figure(figsize=(11.4, 5.0))
    target_x = target_range * np.cos(target_angle)
    target_y = target_range * np.sin(target_angle)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d6ae3d",
            markeredgecolor="#d6ae3d",
            markersize=7,
            label="BS",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#e53935",
            markeredgecolor="#e53935",
            markersize=10,
            label="Actual location of target",
        ),
    ]
    axes = []
    for index, (result, caption) in enumerate(
        ((near, "(a) Near-field ISAC."), (far, "(b) Far-field ISAC.")), start=1
    ):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        axes.append(axis)
        stride = max(1, result.x_grid.shape[0] // 180)
        spectrum_db = np.clip(
            10.0 * np.log10(np.maximum(result.spectrum, 1.0e-12)), -60.0, 0.0
        )
        axis.plot_surface(
            result.x_grid[::stride, ::stride],
            result.y_grid[::stride, ::stride],
            spectrum_db[::stride, ::stride],
            cmap="jet",
            vmin=-52.0,
            vmax=0.0,
            linewidth=0,
            antialiased=True,
            shade=True,
            rcount=180,
            ccount=180,
        )
        axis.scatter(
            [0.0],
            [0.0],
            [0.0],
            color="#d6ae3d",
            edgecolor="white",
            linewidth=0.5,
            marker="o",
            s=45,
            depthshade=False,
            zorder=10,
        )
        axis.scatter(
            [target_x],
            [target_y],
            [0.0],
            color="#e53935",
            edgecolor="white",
            linewidth=0.4,
            marker="*",
            s=90,
            depthshade=False,
            zorder=11,
        )
        axis.text(
            target_x + 0.8,
            target_y + 0.8,
            2.5,
            f"({target_range:g} m, {np.rad2deg(target_angle):g}\N{DEGREE SIGN})",
            fontsize=8,
            ha="center",
        )
        axis.set_xlabel("x (m)", labelpad=5)
        axis.set_ylabel("y (m)", labelpad=5)
        axis.set_zlabel("Spectrum (dB)", labelpad=5)
        axis.set_xlim(0.0, 40.0)
        axis.set_ylim(0.0, 40.0)
        axis.set_zlim(-60.0, 4.0)
        axis.set_xticks([0, 10, 20, 30, 40])
        axis.set_yticks([0, 10, 20, 30, 40])
        axis.set_zticks([-60, -40, -20, 0])
        axis.view_init(elev=27, azim=-128)
        axis.set_box_aspect((1.0, 1.0, 0.62))
        axis.tick_params(labelsize=8, pad=0)
        axis.grid(True, alpha=0.18)
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
            pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))
        axis.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=8,
            frameon=True,
            fancybox=False,
            edgecolor="0.65",
            borderpad=0.35,
            handletextpad=0.4,
        )
        axis.text2D(
            0.5,
            -0.13,
            caption,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontfamily="serif",
        )
    figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.14, wspace=0.02)
    figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(figure)


def reproduce_figure3(
    config: SimulationConfig,
    *,
    output_dir: str | Path,
    optimizer: str = "zf",
    grid_size: int = 121,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1.0e-7,
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
    solver_options = _solver_kwargs(
        solver, verbose, tolerance, max_iterations, solver_threads
    )
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
    }


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _architecture_rows(
    rows: list[dict[str, Any]], architecture: str, x_name: str
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["architecture"] == architecture]
    return sorted(selected, key=lambda row: float(row[x_name]))


def _paper_axis_style(axis: plt.Axes) -> None:
    axis.grid(True, which="major", color="0.84", linewidth=0.65, alpha=0.75)
    axis.grid(True, which="minor", color="0.91", linewidth=0.45, alpha=0.55)
    axis.tick_params(which="both", direction="in", top=True, width=0.8)
    axis.tick_params(which="major", length=4)
    axis.tick_params(which="minor", length=2.5)
    for spine in axis.spines.values():
        spine.set_color("0.28")
        spine.set_linewidth(0.8)


def _plot_figure2(rows: list[dict[str, Any]], destination: Path) -> None:
    fd = _architecture_rows(rows, "fully-digital-sdr", "minimum_rate")
    hb = _architecture_rows(rows, "hybrid-two-stage-sdr", "minimum_rate")
    if not fd or not hb:
        raise ValueError("Figure 2 needs both fully digital and hybrid rows")

    figure, distance_axis = plt.subplots(figsize=(6.8, 4.8))
    angle_axis = distance_axis.twinx()
    blue = "#1848e8"
    red = "#ef342a"
    common = {"linewidth": 1.45, "markersize": 5.2, "markerfacecolor": "white"}

    fd_x = [float(row["minimum_rate"]) for row in fd]
    hb_x = [float(row["minimum_rate"]) for row in hb]
    fd_distance = distance_axis.semilogy(
        fd_x,
        [float(row["range_rcrb_m"]) for row in fd],
        color=blue,
        linestyle="-",
        marker="o",
        markeredgecolor=blue,
        label="FD, distance",
        **common,
    )[0]
    hb_distance = distance_axis.semilogy(
        hb_x,
        [float(row["range_rcrb_m"]) for row in hb],
        color=blue,
        linestyle=(0, (4, 3)),
        marker="o",
        markeredgecolor=blue,
        label="HB, distance",
        **common,
    )[0]
    fd_angle = angle_axis.semilogy(
        fd_x,
        [float(row["angle_rcrb_deg"]) for row in fd],
        color=red,
        linestyle="-",
        marker="s",
        markeredgecolor=red,
        label="FD, angle",
        **common,
    )[0]
    hb_angle = angle_axis.semilogy(
        hb_x,
        [float(row["angle_rcrb_deg"]) for row in hb],
        color=red,
        linestyle=(0, (4, 3)),
        marker="s",
        markeredgecolor=red,
        label="HB, angle",
        **common,
    )[0]

    all_x = sorted({*fd_x, *hb_x})
    distance_axis.set_xlabel("Minimum communication rate (bit/s/Hz)")
    distance_axis.set_ylabel("RCRB for distance (m)")
    angle_axis.set_ylabel("RCRB for angle (deg)")
    distance_values = [
        *[float(row["range_rcrb_m"]) for row in fd],
        *[float(row["range_rcrb_m"]) for row in hb],
    ]
    angle_values = [
        *[float(row["angle_rcrb_deg"]) for row in fd],
        *[float(row["angle_rcrb_deg"]) for row in hb],
    ]
    distance_log = np.log10(distance_values)
    angle_log = np.log10(angle_values)
    distance_axis.set_ylim(
        10.0 ** (float(np.min(distance_log)) - 0.05),
        10.0 ** (float(np.max(distance_log)) + 0.16),
    )
    angle_axis.set_ylim(
        10.0 ** (float(np.min(angle_log)) - 0.05),
        10.0 ** (float(np.max(angle_log)) + 0.16),
    )
    distance_axis.set_xticks(all_x)
    if len(all_x) > 7:
        distance_axis.set_xticks(all_x[::2])
    x_margin = max(0.15, 0.02 * (max(all_x) - min(all_x) or 1.0))
    distance_axis.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    _paper_axis_style(distance_axis)
    angle_axis.tick_params(which="both", direction="in", right=True, width=0.8)
    angle_axis.spines["right"].set_color("0.28")
    angle_axis.spines["right"].set_linewidth(0.8)
    distance_axis.legend(
        handles=[fd_distance, fd_angle, hb_distance, hb_angle],
        ncols=2,
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="0.55",
        fontsize=9,
        columnspacing=1.0,
        handlelength=2.6,
        borderpad=0.45,
    )

    if len(fd_x) >= 2 and len(hb_x) >= 2:
        hybrid_index = min(1, len(hb_x) - 1)
        digital_index = min(1, len(fd_x) - 1)
        x_span = max(all_x) - min(all_x) or 1.0
        hybrid_text_x = min(
            0.75, (hb_x[hybrid_index] - min(all_x)) / x_span + 0.08
        )
        digital_text_x = min(
            0.75, (fd_x[digital_index] - min(all_x)) / x_span + 0.08
        )
        distance_axis.annotate(
            "Hybrid",
            xy=(hb_x[hybrid_index], float(hb[hybrid_index]["range_rcrb_m"])),
            xycoords="data",
            xytext=(hybrid_text_x, 0.67),
            textcoords="axes fraction",
            fontsize=9,
            arrowprops={"arrowstyle": "-|>", "lw": 0.85, "color": "0.15"},
        )
        distance_axis.annotate(
            "Fully digital",
            xy=(fd_x[digital_index], float(fd[digital_index]["range_rcrb_m"])),
            xycoords="data",
            xytext=(digital_text_x, 0.23),
            textcoords="axes fraction",
            fontsize=9,
            arrowprops={"arrowstyle": "-|>", "lw": 0.85, "color": "0.15"},
        )

    figure.tight_layout(pad=0.8)
    figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(figure)


def _scientific_formatter(axis: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    axis.yaxis.set_major_formatter(formatter)


def _plot_figure4(rows: list[dict[str, Any]], destination: Path) -> None:
    fd = _architecture_rows(rows, "fully-digital-sdr", "distance_m")
    hb = _architecture_rows(rows, "hybrid-two-stage-sdr", "distance_m")
    if not fd or not hb:
        raise ValueError("Figure 4 needs both fully digital and hybrid rows")

    figure, (range_axis, fd_angle_axis) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.08},
    )
    hb_angle_axis = fd_angle_axis.twinx()
    blue = "#1848e8"
    red = "#ef342a"
    green = "#31852b"
    common = {"linewidth": 1.45, "markersize": 5.2, "markerfacecolor": "white"}
    fd_x = [float(row["distance_m"]) for row in fd]
    hb_x = [float(row["distance_m"]) for row in hb]

    fd_range = range_axis.semilogy(
        fd_x,
        [float(row["range_rcrb_m"]) for row in fd],
        color=blue,
        linestyle="-",
        marker="o",
        markeredgecolor=blue,
        label="FD",
        **common,
    )[0]
    hb_range = range_axis.semilogy(
        hb_x,
        [float(row["range_rcrb_m"]) for row in hb],
        color=blue,
        linestyle=(0, (4, 3)),
        marker="o",
        markeredgecolor=blue,
        label="HB",
        **common,
    )[0]
    range_axis.set_ylabel("RCRB (m)")
    range_axis.legend(
        handles=[fd_range, hb_range],
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="0.55",
        fontsize=9,
    )

    fd_near = fd_angle_axis.plot(
        fd_x,
        [float(row["angle_rcrb_deg"]) for row in fd],
        color=red,
        linestyle="-",
        marker="s",
        markeredgecolor=red,
        label="FD, near-field",
        **common,
    )[0]
    hb_near = hb_angle_axis.plot(
        hb_x,
        [float(row["angle_rcrb_deg"]) for row in hb],
        color=green,
        linestyle="-",
        marker=">",
        markeredgecolor=green,
        label="HB, near-field",
        **common,
    )[0]
    fd_far_value = float(fd[-1]["far_field_angle_rcrb_deg"])
    hb_far_value = float(hb[-1]["far_field_angle_rcrb_deg"])
    fd_far = fd_angle_axis.axhline(
        fd_far_value,
        color=red,
        linestyle="-.",
        linewidth=1.35,
        label="FD, far-field",
    )
    hb_far = hb_angle_axis.axhline(
        hb_far_value,
        color=green,
        linestyle=":",
        linewidth=1.5,
        label="HB, far-field",
    )

    fd_angle_axis.set_xlabel("Distance, $r$ (m)")
    fd_angle_axis.set_ylabel("RCRB, FD (deg)")
    hb_angle_axis.set_ylabel("RCRB, HB (deg)")
    all_x = sorted({*fd_x, *hb_x})
    fd_angle_axis.set_xticks(all_x)
    x_margin = max(0.15, 0.02 * (max(all_x) - min(all_x) or 1.0))
    fd_angle_axis.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    _scientific_formatter(fd_angle_axis)
    _scientific_formatter(hb_angle_axis)
    _paper_axis_style(range_axis)
    _paper_axis_style(fd_angle_axis)
    hb_angle_axis.tick_params(which="both", direction="in", right=True, width=0.8)
    hb_angle_axis.spines["right"].set_color("0.28")
    hb_angle_axis.spines["right"].set_linewidth(0.8)
    fd_angle_axis.legend(
        handles=[fd_near, hb_near, fd_far, hb_far],
        ncols=2,
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="0.55",
        fontsize=8.7,
        columnspacing=1.0,
        handlelength=2.5,
        borderpad=0.45,
    )

    figure.subplots_adjust(left=0.13, right=0.87, top=0.98, bottom=0.10)
    figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(figure)


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
    full = solve_fully_digital_sdr(
        config, scenario, min_rate=rate, **solver_options
    )
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
    tolerance: float = 1.0e-7,
    max_iterations: int = 20_000,
    solver_threads: int | None = None,
    workers: int = 1,
    result_cache: MutableMapping[
        float, tuple[OptimizationResult, OptimizationResult]
    ]
    | None = None,
) -> dict[str, Any]:
    """Reproduce the sensing/communication tradeoff in paper Fig. 2."""

    output = _prepare_output(output_dir)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    rates = list(rates)
    rng = np.random.default_rng(config.seed)
    scenario = generate_scenario(config, rng)
    receive_combiner = random_hybrid_combiner(config, rng)
    solver_options = _solver_kwargs(
        solver, verbose, tolerance, max_iterations, solver_threads
    )
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
                print(
                    f"  Figure 2 [{completed}/{len(rates)} completed]: "
                    f"R_min={rate:g} bit/s/Hz"
                )
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
    scenario = _scenario_at_range(
        distance_config, base_scenario, distance, fixed_target_gain
    )
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
    full_row = _curve_row(
        distance_config, scenario, full, "distance_m", distance
    )
    hybrid_row = _curve_row(
        distance_config, scenario, hybrid, "distance_m", distance
    )
    full_far_crb = far_field_angle_crb(
        distance_config,
        full.waveform.covariance,
        scenario.target_gain,
        angle=scenario.target_angle,
    )
    hybrid_far_crb = far_field_angle_crb(
        distance_config,
        hybrid.waveform.covariance,
        scenario.target_gain,
        angle=scenario.target_angle,
        receive_combiner=receive_combiner,
    )
    full_row["far_field_angle_rcrb_deg"] = float(
        np.rad2deg(np.sqrt(full_far_crb))
    )
    hybrid_row["far_field_angle_rcrb_deg"] = float(
        np.rad2deg(np.sqrt(hybrid_far_crb))
    )
    return [full_row, hybrid_row]


def reproduce_figure4(
    config: SimulationConfig,
    distances: Iterable[float],
    *,
    output_dir: str | Path,
    solver: str = "auto",
    verbose: bool = False,
    tolerance: float = 1.0e-7,
    max_iterations: int = 20_000,
    solver_threads: int | None = None,
    workers: int = 1,
    precomputed_results: dict[
        float, tuple[OptimizationResult, OptimizationResult]
    ]
    | None = None,
) -> dict[str, Any]:
    """Reproduce the range-dependence experiment in paper Fig. 4.

    The target gain generated at the nominal 20 m location is held fixed over
    the sweep, implementing the paper's instruction to exclude pathloss.
    """

    output = _prepare_output(output_dir)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    distances = list(distances)
    rng = np.random.default_rng(config.seed)
    base_scenario = generate_scenario(config, rng)
    fixed_target_gain = base_scenario.target_gain
    receive_combiner = random_hybrid_combiner(config, rng)
    solver_options = _solver_kwargs(
        solver, verbose, tolerance, max_iterations, solver_threads
    )
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
                print(
                    f"  Figure 4 [{completed}/{len(distances)} completed]: "
                    f"range={distance:g} m"
                )
    architecture_order = {"fully-digital-sdr": 0, "hybrid-two-stage-sdr": 1}
    rows.sort(
        key=lambda row: (
            float(row["distance_m"]),
            architecture_order[str(row["architecture"])],
        )
    )

    _save_rows(output / "figure4_rcrb_vs_distance.csv", rows)
    _plot_figure4(rows, output / "figure4_rcrb_vs_distance.png")
    far_field_reference = {
        "distance_m": max(float(value) for value in distances),
        "fully_digital_angle_rcrb_deg": float(
            _architecture_rows(rows, "fully-digital-sdr", "distance_m")[-1][
                "far_field_angle_rcrb_deg"
            ]
        ),
        "hybrid_angle_rcrb_deg": float(
            _architecture_rows(rows, "hybrid-two-stage-sdr", "distance_m")[-1][
                "far_field_angle_rcrb_deg"
            ]
        ),
    }
    summary = {
        "experiment": "figure4",
        "seed": config.seed,
        "workers": workers,
        "solver_threads": solver_threads,
        "pathloss_in_sweep": False,
        "far_field_reference": far_field_reference,
        "far_field_reference_convention": (
            "Far-field steering with the covariance optimized at the largest "
            "swept near-field distance"
        ),
        "distances_m": [float(value) for value in distances],
        "rows": rows,
    }
    _save_json(output / "figure4_summary.json", summary)
    return summary
