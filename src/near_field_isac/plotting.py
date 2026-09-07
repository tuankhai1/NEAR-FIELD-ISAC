"""Paper-style figures; styling never alters or smooths simulated values."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter, ScalarFormatter

from .music import MusicResult

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.linewidth": 0.7,
    "legend.fontsize": 9,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
}
BLUE, RED, GREEN = "#0000ff", "#ff0000", "#008000"


def _axis(ax: plt.Axes) -> None:
    ax.grid(False, which="both")
    ax.tick_params(which="both", direction="in", top=True, right=True, width=0.6)
    ax.tick_params(which="major", length=3.5)
    ax.tick_params(which="minor", length=2)


def _log_axis(ax: plt.Axes, values: list[float], *, decades: int | None = None) -> None:
    values_array = np.asarray(values)
    if np.any(~np.isfinite(values_array)) or np.any(values_array <= 0):
        raise ValueError("Logarithmic figures require finite, positive CRBs")
    lo = float(np.floor(np.log10(values_array.min())))
    hi = float(np.ceil(np.log10(values_array.max())))
    if decades is not None:
        lo = hi - max(decades, int(hi - lo))
    ax.set_yscale("log")
    ax.set_ylim(10**lo, 10**hi)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _save(fig: plt.Figure, destination: Path) -> None:
    fig.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_figure2(rows: list[dict], destination: Path) -> None:
    with plt.rc_context(STYLE):
        fig, left = plt.subplots(figsize=(5.1, 3.6))
        right = left.twinx()
        handles = []
        distances, angles = [], []
        for name, label, line in [
            ("fully-digital-sdr", "FD", "-"),
            ("hybrid-two-stage-sdr", "HB", "--"),
        ]:
            part = sorted(
                [r for r in rows if r["architecture"] == name], key=lambda r: r["minimum_rate"]
            )
            x = [r["minimum_rate"] for r in part]
            d = [r["range_rcrb_m"] for r in part]
            a = [r["angle_rcrb_deg"] for r in part]
            distances.extend(d)
            angles.extend(a)
            for axis, values, color, marker, quantity in [
                (left, d, BLUE, "o", "distance"),
                (right, a, RED, "s", "angle"),
            ]:
                handles.append(
                    axis.plot(
                        x,
                        values,
                        color=color,
                        marker=marker,
                        linestyle=line,
                        markerfacecolor="white",
                        markeredgewidth=0.7,
                        linewidth=1.0,
                        markersize=3.8,
                        label=f"{label}, {quantity}",
                    )[0]
                )
        _log_axis(left, distances, decades=2)
        _log_axis(right, angles, decades=2)
        # Separate physical units retain independent labels. Offset the right
        # logarithmic axis so distance/angle pairs are legible as in the paper.
        bounds = left.get_ylim()
        angle_scale = 0.8 * float(np.median(np.asarray(angles) / np.asarray(distances)))
        proposed = (bounds[0] * angle_scale, bounds[1] * angle_scale)
        if proposed[0] < min(angles) and proposed[1] > max(angles):
            right.set_ylim(*proposed)
        left.set(xlabel="Minimum communication rate (bit/s/Hz)", ylabel="RCRB for distance (m)")
        right.set_ylabel("RCRB for angle (deg)")
        xmin, xmax = min(r["minimum_rate"] for r in rows), max(r["minimum_rate"] for r in rows)
        left.set_xlim(xmin - 0.08, xmax + 0.08)
        left.set_xticks(np.arange(np.ceil(xmin / 2) * 2, xmax + 0.01, 2))
        _axis(left)
        _axis(right)
        left.legend(
            handles=handles,
            ncols=2,
            loc="upper left",
            frameon=True,
            fancybox=False,
            edgecolor="black",
            framealpha=1,
            borderpad=0.3,
            columnspacing=0.9,
            handletextpad=0.4,
            handlelength=2,
        )
        for name, label in [
            ("fully-digital-sdr", "Fully digital"),
            ("hybrid-two-stage-sdr", "Hybrid"),
        ]:
            part = sorted(
                [r for r in rows if r["architecture"] == name], key=lambda r: r["minimum_rate"]
            )
            row = min(part, key=lambda r: abs(r["minimum_rate"] - 1))
            left.annotate(
                label,
                xy=(row["minimum_rate"], row["range_rcrb_m"]),
                xytext=(22, 18),
                textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "->", "lw": 0.7},
            )
        fig.tight_layout(pad=0.5)
        _save(fig, destination)


def plot_figure4(rows: list[dict], destination: Path) -> None:
    with plt.rc_context(STYLE):
        fig, (top, left) = plt.subplots(
            2,
            1,
            figsize=(5.3, 4.35),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.48},
        )
        right = left.twinx()
        angle_handles = []
        ranges = []
        for name, label, line, axis, color, marker in [
            ("fully-digital-sdr", "FD", "-", left, RED, "s"),
            ("hybrid-two-stage-sdr", "HB", "--", right, GREEN, ">"),
        ]:
            part = sorted(
                [r for r in rows if r["architecture"] == name], key=lambda r: r["distance_m"]
            )
            x = [r["distance_m"] for r in part]
            d = [r["range_rcrb_m"] for r in part]
            a = [r["angle_rcrb_deg"] for r in part]
            ranges.extend(d)
            top.plot(
                x,
                d,
                color=BLUE,
                linestyle=line,
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=0.7,
                linewidth=1,
                label=label,
            )
            angle_handles.append(
                axis.plot(
                    x,
                    a,
                    color=color,
                    marker=marker,
                    markersize=3.5,
                    markerfacecolor="white",
                    markeredgewidth=0.7,
                    linewidth=1,
                    label=f"{label}, near-field",
                )[0]
            )
            reference = part[0]["far_field_angle_rcrb_deg"]
            if not np.allclose([r["far_field_angle_rcrb_deg"] for r in part], reference):
                raise ValueError("Figure 4 requires an independently computed constant reference")
            angle_handles.append(
                axis.axhline(
                    reference,
                    color=color,
                    linestyle="-." if label == "FD" else ":",
                    linewidth=1,
                    label=f"{label}, far-field",
                )
            )
            formatter = ScalarFormatter(useMathText=True, useOffset=False)
            formatter.set_powerlimits((0, 0))
            axis.yaxis.set_major_formatter(formatter)
            low, high = min(*a, reference), max(*a, reference)
            margin = max((high - low) * 0.18, high * 0.002)
            axis.set_ylim(low - margin, high + margin)
        _log_axis(top, ranges)
        top.set_ylabel("RCRB (m)")
        left.set(xlabel="Distance, $r$ (m)", ylabel="RCRB, FD (deg)")
        right.set_ylabel("RCRB, HB (deg)")
        left.set_xticks(sorted({r["distance_m"] for r in rows}))
        left.set_xlim(
            min(r["distance_m"] for r in rows) - 0.25, max(r["distance_m"] for r in rows) + 0.25
        )
        for ax in [top, left, right]:
            _axis(ax)
        top.legend(loc="upper left", fancybox=False, edgecolor="black", borderpad=0.25)
        # Keep the narrow numerical panel unobscured by placing its legend above it.
        left.legend(
            handles=angle_handles,
            ncols=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            fancybox=False,
            edgecolor="black",
            fontsize=8,
            borderpad=0.25,
            columnspacing=1,
            handlelength=2,
        )
        fig.subplots_adjust(left=0.16, right=0.83, bottom=0.12, top=0.98, hspace=0.43)
        _save(fig, destination)


def plot_music_pair(
    near: MusicResult,
    far: MusicResult,
    destination: Path,
    *,
    target_range: float,
    target_angle: float,
) -> None:
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(8.3, 3.4))
        tx, ty = target_range * np.cos(target_angle), target_range * np.sin(target_angle)
        handles = [
            Line2D([], [], color="#dca526", marker="o", linestyle="", markersize=4, label="BS"),
            Line2D(
                [],
                [],
                color=RED,
                marker="*",
                linestyle="",
                markersize=6,
                label="Actual location of target",
            ),
        ]
        for i, (result, caption) in enumerate(
            [(near, "(a) Near-field ISAC."), (far, "(b) Far-field ISAC.")], 1
        ):
            ax = fig.add_subplot(1, 2, i, projection="3d")
            db = np.maximum(10 * np.log10(np.maximum(result.spectrum, 1e-30)), -60)
            # Preserve the full numerical grid, including a potentially one-pixel peak.
            ax.plot_surface(
                result.y_grid,
                result.x_grid,
                db,
                rstride=1,
                cstride=1,
                cmap="jet",
                vmin=-50,
                vmax=0,
                linewidth=0,
                antialiased=False,
                shade=False,
                rasterized=True,
            )
            ax.scatter([0], [0], [0], color="#dca526", s=15, depthshade=False)
            ax.scatter([ty], [tx], [0], color=RED, marker="*", s=30, depthshade=False)
            ax.text(
                ty,
                tx,
                5,
                f"({target_range:g} m, {np.rad2deg(target_angle):g}°)",
                fontsize=7,
                ha="center",
            )
            ax.set(xlim=(0, 40), ylim=(0, 40), zlim=(-60, 3))
            ax.set_xlabel("y (m)", fontsize=9, labelpad=-1)
            ax.set_ylabel("x (m)", fontsize=9, labelpad=-1)
            ax.text2D(
                -0.11,
                0.52,
                "Spectrum (dB)",
                transform=ax.transAxes,
                rotation=90,
                fontsize=9,
                va="center",
            )
            ax.set_xticks([0, 10, 20, 30, 40])
            ax.set_yticks([0, 10, 20, 30, 40])
            ax.set_zticks([-60, -40, -20, 0])
            ax.view_init(elev=25, azim=52)
            ax.set_box_aspect((1, 1, 0.55))
            ax.tick_params(labelsize=7, pad=-2)
            ax.grid(False)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_visible(False)
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(0, 1.03),
                fontsize=7,
                fancybox=False,
                edgecolor="0.6",
                borderpad=0.2,
                handletextpad=0.3,
            )
            ax.text2D(0.5, -0.09, caption, transform=ax.transAxes, ha="center", fontsize=11)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.12, wspace=0.01)
        _save(fig, destination)
