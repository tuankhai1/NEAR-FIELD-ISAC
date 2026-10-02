"""Paper-style figures; styling never alters or smooths simulated values."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    NullFormatter,
    ScalarFormatter,
)

from .music import MusicResult

# As in the paper's MATLAB figures: sans-serif axes, LaTeX-like serif legends.
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.linewidth": 0.7,
    "legend.fontsize": 9,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
}
BLUE, RED, GREEN = "#0000ff", "#ff0000", "#008000"
SERIF = "serif"


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


FIGURE2_DISTANCE_LIMITS = (10**-2.526, 1e-1)
FIGURE2_ANGLE_LIMITS = (1e-5, 10**-2.82)


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
        # Axis limits of the paper's Fig. 2, used whenever the data fit them.
        for axis, values, limits in [
            (left, distances, FIGURE2_DISTANCE_LIMITS),
            (right, angles, FIGURE2_ANGLE_LIMITS),
        ]:
            if limits[0] < min(values) and max(values) < limits[1]:
                axis.set_ylim(limits[0], limits[1])
        left.set(xlabel="Minimum communication rate (bit/s/Hz)", ylabel="RCRB for distance (m)")
        right.set_ylabel("RCRB for angle (deg)")
        xmin, xmax = min(r["minimum_rate"] for r in rows), max(r["minimum_rate"] for r in rows)
        left.set_xlim(xmin - 0.08, xmax + 0.08)
        left.set_xticks(np.arange(np.ceil(xmin / 2) * 2, xmax + 0.01, 2))
        _axis(left)
        _axis(right)
        left.legend(
            # Matplotlib fills columns first: distance on the left, angle on
            # the right, with FD above HB, as in the paper.
            handles=[handles[i] for i in (0, 2, 1, 3)],
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
            prop={"family": SERIF, "size": 9},
        )
        for name, label in [
            ("fully-digital-sdr", "Fully digital"),
            ("hybrid-two-stage-sdr", "Hybrid"),
        ]:
            part = sorted(
                [r for r in rows if r["architecture"] == name], key=lambda r: r["minimum_rate"]
            )
            # As in the paper: an ellipse around the distance/angle pair near
            # the first marker, with an arrow pointing to the label.
            row = min(part, key=lambda r: abs(r["minimum_rate"] - 0.75))
            to_axes = left.transAxes.inverted()
            x = row["minimum_rate"] + 0.75 if row["minimum_rate"] == 0 else row["minimum_rate"]
            heights = [
                to_axes.transform(axis.transData.transform((x, value)))[1]
                for axis, value in [(left, row["range_rcrb_m"]), (right, row["angle_rcrb_deg"])]
            ]
            x_axes = to_axes.transform(left.transData.transform((x, row["range_rcrb_m"])))[0]
            center = (x_axes, float(np.mean(heights)))
            left.add_patch(
                Ellipse(
                    center,
                    width=0.04,
                    height=abs(heights[1] - heights[0]) + 0.08,
                    transform=left.transAxes,
                    fill=False,
                    linewidth=0.7,
                )
            )
            left.annotate(
                label,
                xy=(center[0] + 0.01, center[1] + 0.02),
                xycoords="axes fraction",
                xytext=(center[0] + 0.12, center[1] + 0.1),
                textcoords="axes fraction",
                fontsize=9,
                va="bottom",
                arrowprops={
                    "arrowstyle": "<|-",
                    "lw": 0.7,
                    "fc": "black",
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
        fig.tight_layout(pad=0.5)
        _save(fig, destination)


# Axis limits and ticks of the paper's Fig. 4, used whenever the data fit them.
FIGURE4_RANGE_LIMITS = (1e-4, 1.0)
FIGURE4_FD_ANGLE_LIMITS = (3.4445e-5, 3.49e-5, np.arange(3.45, 3.495, 0.01) * 1e-5)
FIGURE4_HB_ANGLE_LIMITS = (1.714e-4, 2.344e-4, np.array([1.8, 2.0, 2.2]) * 1e-4)


def plot_figure4(rows: list[dict], destination: Path) -> None:
    with plt.rc_context(STYLE):
        fig, (top, left) = plt.subplots(
            2,
            1,
            figsize=(5.3, 4.65),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1]},
        )
        right = left.twinx()
        angle_handles = []
        ranges = []
        for name, label, line, axis, color, marker, far_line, limits in [
            ("fully-digital-sdr", "FD", "-", left, RED, "s", "-.", FIGURE4_FD_ANGLE_LIMITS),
            ("hybrid-two-stage-sdr", "HB", "--", right, GREEN, ">", ":", FIGURE4_HB_ANGLE_LIMITS),
        ]:
            part = sorted(
                [r for r in rows if r["architecture"] == name], key=lambda r: r["distance_m"]
            )
            x = [r["distance_m"] for r in part]
            d = [r["range_rcrb_m"] for r in part]
            a = [r["angle_rcrb_deg"] for r in part]
            ranges.extend(d)
            # Paper Fig. 4(a) plots only the 5 m grid; (b) uses every distance.
            coarse = [i for i, value in enumerate(x) if value % 5 == 0]
            top.plot(
                [x[i] for i in coarse],
                [d[i] for i in coarse],
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
                    linestyle=far_line,
                    linewidth=1,
                    label=f"{label}, far-field",
                )
            )
            formatter = ScalarFormatter(useMathText=True, useOffset=False)
            formatter.set_powerlimits((0, 0))
            axis.yaxis.set_major_formatter(formatter)
            low, high = min(*a, reference), max(*a, reference)
            if limits[0] < low and high < limits[1]:
                axis.set_ylim(limits[0], limits[1])
                axis.set_yticks(limits[2])
            else:
                margin = max((high - low) * 0.18, high * 0.002)
                axis.set_ylim(low - margin, high + margin)
        _log_axis(top, ranges)
        if FIGURE4_RANGE_LIMITS[0] < min(ranges) and max(ranges) < FIGURE4_RANGE_LIMITS[1]:
            top.set_ylim(*FIGURE4_RANGE_LIMITS)
            top.set_yticks([1e-4, 1e-2])
        top.set(xlabel="Distance, $r$ (m)", ylabel="RCRB (m)")
        # Shared axes suppress upper tick labels by default; the paper labels
        # the distance axis separately on both panels.
        top.tick_params(axis="x", labelbottom=True)
        left.set(xlabel="Distance, $r$ (m)", ylabel="RCRB, FD (deg)")
        right.set_ylabel("RCRB, HB (deg)")
        xmin, xmax = min(r["distance_m"] for r in rows), max(r["distance_m"] for r in rows)
        left.set_xticks(np.arange(np.ceil(xmin / 5) * 5, xmax + 0.01, 5))
        left.set_xlim(xmin - 0.25, xmax + 0.25)
        for ax in [top, left, right]:
            _axis(ax)
        top.legend(
            loc="upper left",
            fancybox=False,
            edgecolor="black",
            borderpad=0.25,
            prop={"family": SERIF, "size": 9},
        )
        # The paper places this legend inside the upper-right corner. Curves
        # at small distances remain visible to its left.
        left.legend(
            handles=angle_handles,
            ncols=2,
            loc="upper right",
            fancybox=False,
            edgecolor="black",
            framealpha=1,
            prop={"family": SERIF, "size": 8},
            borderpad=0.25,
            columnspacing=1,
            handlelength=2,
        )
        fig.subplots_adjust(left=0.16, right=0.83, bottom=0.11, top=0.97, hspace=0.53)
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
            db = 10 * np.log10(np.maximum(result.spectrum, 1e-30))
            floor = min(-60.0, float(np.floor(db.min())))
            # Preserve the full numerical grid, including a potentially one-pixel peak.
            surface = ax.plot_surface(
                result.x_grid,
                result.y_grid,
                db,
                rstride=1,
                cstride=1,
                cmap="jet",
                vmin=floor,
                vmax=0,
                linewidth=0.1,
                antialiased=True,
                shade=False,
                rasterized=True,
            )
            # Colour each face by its highest corner: the default corner
            # average would dim one-sample-wide features (the peak and the
            # far-field ridge), which MATLAB's mesh shows at 0 dB.
            surface.set_array(
                np.maximum.reduce([db[:-1, :-1], db[1:, :-1], db[:-1, 1:], db[1:, 1:]]).ravel()
            )
            surface.set_clim(floor, 0)
            surface.set_edgecolor("face")
            ax.scatter([0], [0], [0], color="#dca526", s=15, depthshade=False)
            ax.scatter([tx], [ty], [0], color=RED, marker="*", s=30, depthshade=False)
            ax.text(
                tx,
                ty,
                5,
                f"({target_range:g} m, {np.rad2deg(target_angle):g}°)",
                fontsize=7,
                family=SERIF,
                ha="center",
            )
            ax.set(xlim=(0, 40), ylim=(0, 40), zlim=(floor, 3))
            ax.set_xlabel("x (m)", fontsize=9, labelpad=-1)
            ax.set_ylabel("y (m)", fontsize=9, labelpad=-1)
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
            # Camera matches the paper; the physical x/y data remain unmodified.
            ax.view_init(elev=25, azim=-38)
            ax.zaxis.set_ticks_position("lower")
            ax.set_box_aspect((1, 1, 0.55))
            ax.tick_params(labelsize=7, pad=-2)
            ax.grid(False)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_visible(False)
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.86),
                prop={"family": SERIF, "size": 7},
                fancybox=False,
                edgecolor="0.6",
                borderpad=0.2,
                handletextpad=0.3,
            )
            ax.text2D(
                0.5, -0.09, caption, transform=ax.transAxes, ha="center", fontsize=11, family=SERIF
            )
        fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.12, wspace=0.01)
        _save(fig, destination)
