"""Paper-style figures (MATLAB look of Fig. 2 and Fig. 3 in the letter)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

BLUE = (0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 0.5, 0.0)

MATLAB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "savefig.dpi": 300,
}


# ============================================================================ Fig. 2
def plot_fig2(R_min: np.ndarray, rcrb: dict, path_stem: str, annotate_at: float = 0.6):
    """RCRB vs minimum communication rate, two log y-axes (distance left, angle right).

    rcrb keys: 'FD_r', 'FD_th', 'HB_r', 'HB_th' (m and deg).
    """
    with plt.rc_context(MATLAB_RC):
        fig, ax1 = plt.subplots(figsize=(5.6, 4.2))
        ax2 = ax1.twinx()
        mk = dict(markersize=5.5, markerfacecolor="none", markeredgewidth=0.9, linewidth=1.0)
        l1, = ax1.semilogy(R_min, rcrb["FD_r"], "-o", color=BLUE, **mk)
        l3, = ax1.semilogy(R_min, rcrb["HB_r"], "--o", color=BLUE, **mk)
        l2, = ax2.semilogy(R_min, rcrb["FD_th"], "-s", color=RED, **mk)
        l4, = ax2.semilogy(R_min, rcrb["HB_th"], "--s", color=RED, **mk)

        # Same number of decades on both axes, shifted so that the four curves
        # interleave as in the paper.
        r_all = np.concatenate([rcrb["FD_r"], rcrb["HB_r"]])
        t_all = np.concatenate([rcrb["FD_th"], rcrb["HB_th"]])
        shift = 10.0 ** np.round(np.log10(np.median(r_all) / np.median(t_all)))
        lo = min(r_all.min(), (t_all * shift).min()) / 2.5
        hi = max(r_all.max(), (t_all * shift).max()) * 6.0       # head-room for the legend
        ax1.set_ylim(lo, hi)
        ax2.set_ylim(lo / shift, hi / shift)
        ax1.set_xlim(R_min.min(), R_min.max())
        ax1.set_xticks(np.arange(0, R_min.max() + 1e-9, 2))
        ax1.set_xlabel("Minimum communication rate (bit/s/Hz)")
        ax1.set_ylabel("RCRB for distance (m)")
        ax2.set_ylabel("RCRB for angle (deg)")
        for ax in (ax1, ax2):
            ax.tick_params(which="both", direction="in")

        ax1.legend([l1, l2, l3, l4], ["FD, distance", "FD, angle", "HB, distance", "HB, angle"],
                   loc="upper right", ncol=2, fontsize=8, handlelength=2.6, columnspacing=1.0,
                   borderpad=0.3, labelspacing=0.2)

        # Ellipses + arrows "Hybrid" / "Fully digital" as in the paper.
        fig.canvas.draw()
        to_frac = lambda ax, v: ax.transLimits.transform(ax.transScale.transform((annotate_at, v)))[1]
        xf = ax1.transLimits.transform(ax1.transScale.transform((annotate_at, lo)))[0]
        for (kr, kt, label, dy) in (("HB_r", "HB_th", "Hybrid", 0.07), ("FD_r", "FD_th", "Fully digital", 0.07)):
            yr = to_frac(ax1, np.interp(annotate_at, R_min, rcrb[kr]))
            yt = to_frac(ax2, np.interp(annotate_at, R_min, rcrb[kt]))
            yc, h = 0.5 * (yr + yt), abs(yr - yt) + 0.07
            ax1.add_patch(Ellipse((xf, yc), 0.045, h, transform=ax1.transAxes,
                                  fill=False, edgecolor="black", linewidth=0.8, zorder=5))
            ax1.annotate(label, xy=(xf + 0.012, yc + h / 2 - 0.005), xycoords="axes fraction",
                         xytext=(xf + 0.10, yc + h / 2 + dy), textcoords="axes fraction",
                         fontsize=8, ha="left", va="bottom",
                         arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, shrinkA=0, shrinkB=0,
                                         mutation_scale=8))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{path_stem}.{ext}", bbox_inches="tight")
        plt.close(fig)


# ============================================================================ Fig. 4
def _band_limits(values, f_lo: float, f_hi: float):
    """Linear axis limits that place min(values) at axes fraction f_lo and max(values) at f_hi."""
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo if hi > lo else 0.02 * abs(hi)
    unit = span / (f_hi - f_lo)
    return lo - f_lo * unit, lo + (1.0 - f_lo) * unit


def plot_fig4(r_s: np.ndarray, rcrb: dict, path_stem: str, marker_step: float = 5.0, note: str | None = None):
    """RCRB vs target distance, two stacked panels as in the paper.

    (a) top: distance RCRB (log), FD solid / HB dashed, markers at multiples of ``marker_step``;
    (b) bottom: angle RCRB, FD (left linear axis, red) and HB (right linear axis, green),
        near-field curves with markers, far-field ISAC as horizontal lines.
    rcrb keys: 'FD_r', 'HB_r' (m), 'FD_th', 'HB_th' (deg, arrays), 'FF_FD_th', 'FF_HB_th' (deg, scalars).
    ``note``: optional label written in the lower-right corner of panel (a), e.g. to mark a variant.
    """
    from matplotlib.ticker import ScalarFormatter

    every = np.flatnonzero(np.isclose(np.mod(r_s, marker_step), 0.0) | np.isclose(np.mod(r_s, marker_step), marker_step))
    mk = dict(markersize=5.0, markerfacecolor="none", markeredgewidth=0.9, linewidth=1.0)
    with plt.rc_context(MATLAB_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 5.6))

        # (a) distance
        ax1.semilogy(r_s, rcrb["FD_r"], "-o", color=BLUE, markevery=list(every), label="FD", **mk)
        ax1.semilogy(r_s, rcrb["HB_r"], "--o", color=BLUE, markevery=list(every), label="HB", **mk)
        ax1.set_ylabel("RCRB (m)")
        ax1.legend(loc="upper left", fontsize=8, handlelength=2.6, borderpad=0.3, labelspacing=0.2)
        if note:
            ax1.text(0.98, 0.04, note, transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.5)

        # (b) angle, FD left / HB right
        ax3 = ax2.twinx()
        ff_style = dict(linewidth=0.9)
        l1, = ax2.plot(r_s, rcrb["FD_th"], "-s", color=RED, **mk)
        l2 = ax2.axhline(rcrb["FF_FD_th"], linestyle="-.", color=RED, **ff_style)
        l3, = ax3.plot(r_s, rcrb["HB_th"], "->", color=GREEN, **mk)
        l4 = ax3.axhline(rcrb["FF_HB_th"], linestyle=":", color=GREEN, **ff_style)
        # FD in the lower band, HB in the upper band (as in the paper), legend on top
        ax2.set_ylim(*_band_limits(np.append(rcrb["FD_th"], rcrb["FF_FD_th"]), 0.06, 0.40))
        ax3.set_ylim(*_band_limits(np.append(rcrb["HB_th"], rcrb["FF_HB_th"]), 0.42, 0.76))
        ax2.set_ylabel("RCRB, FD (deg)")
        ax3.set_ylabel("RCRB, HB (deg)")
        for ax in (ax2, ax3):
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(fmt)
        ax3.legend([l1, l2, l3, l4], ["FD, near-field", "FD, far-field", "HB, near-field", "HB, far-field"],
                   loc="upper right", ncol=2, fontsize=7.5, handlelength=2.6, columnspacing=1.0,
                   borderpad=0.3, labelspacing=0.2)

        for ax in (ax1, ax2):
            ax.set_xlim(r_s.min(), r_s.max())
            ax.set_xticks(np.arange(np.ceil(r_s.min() / 5) * 5, r_s.max() + 1e-9, 5))
            ax.set_xlabel(r"Distance, $r$ (m)")
        for ax in (ax1, ax2, ax3):
            ax.tick_params(which="both", direction="in")
        fig.tight_layout(h_pad=1.2)
        for ext in ("png", "pdf"):
            fig.savefig(f"{path_stem}.{ext}", bbox_inches="tight")
        plt.close(fig)


# ============================================================================ Fig. 3
# main.m calls view([60, 60]) (MATLAB az = 60 -> matplotlib azim = az - 90 = -30), but the
# published images (paper Fig. 3, results/MUSIC_spectrum.jpg of the official repo) were
# exported from a slightly rotated view: measuring the projected floor edges of that image
# gives azim ~ -30 deg, elev ~ 34 deg and a z/xy box ratio ~ 0.4, which are used here.
ELEV, AZIM = 34.0, -30.0
BOX_ASPECT = (1.0, 1.0, 0.4)


def _heightfield_order(surf, azim_deg: float):
    """Draw the surface cells back-to-front by horizontal distance to the viewer.

    matplotlib sorts 3-D polygons by their mean projected depth, which fails on
    the steep MUSIC ridge (saw-tooth artefacts).  For a height field seen from
    above, ordering by horizontal depth is exact (painter's algorithm for terrain).
    Uses the private ``_zsortfunc`` hook; silently skipped if it is unavailable.
    """
    faces = getattr(surf, "_faces", None)
    if faces is None or not hasattr(surf, "_zsortfunc"):
        return
    az = np.deg2rad(azim_deg)
    key = -(faces[..., 0].mean(-1) * np.cos(az) + faces[..., 1].mean(-1) * np.sin(az))
    surf._zsortfunc = lambda pzs, axis=-1: key


def _spectrum_axes(ax, res, p, floor_db: float, title: str | None):
    Z = np.clip(res.spectrum_db, floor_db, 0.0)
    XX, YY = np.meshgrid(res.x, res.y)
    # plot_surface colours cell (i, j) with facecolors[i, j]; use the largest of its four corners so a
    # one-point peak keeps its colour, as the vertex-coloured mesh of MATLAB (official MUSIC_spectrum.jpg)
    Zc = Z.copy()
    Zc[:-1, :-1] = np.maximum.reduce([Z[:-1, :-1], Z[1:, :-1], Z[:-1, 1:], Z[1:, 1:]])
    colors = plt.get_cmap("jet")((Zc - floor_db) / -floor_db)
    surf = ax.plot_surface(XX, YY, Z, facecolors=colors, rcount=Z.shape[0], ccount=Z.shape[1],
                           linewidth=0, antialiased=False, shade=False, zorder=1)
    _heightfield_order(surf, AZIM)
    surf.set_rasterized(True)                 # keeps the PDF small (axes/text stay vector)
    xs, ys = p.r_s * np.cos(p.theta_s), p.r_s * np.sin(p.theta_s)
    bs = ax.scatter([0], [0], [0], marker="o", s=28, color=(0.93, 0.69, 0.13), edgecolors=(0.93, 0.69, 0.13),
                    depthshade=False, zorder=10, label="BS")
    tg = ax.scatter([xs], [ys], [0], marker="*", s=70, color="red", depthshade=False, zorder=11,
                    label="Actual location of target")
    ax.text(xs - 4, ys + 3, 3, f"({p.r_s:g} m, {p.theta_s_deg:g}" + r"$^\circ$)", fontsize=7, zorder=12)
    ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.set_zlim(floor_db, 0)
    ax.set_xticks([0, 10, 20, 30, 40]); ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_zticks(np.arange(floor_db, 1, 20))
    ax.set_xlabel("x (m)", labelpad=2); ax.set_ylabel("y (m)", labelpad=2)
    ax.set_zlabel("Spectrum (dB)", labelpad=2)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect(BOX_ASPECT)
    ax.zaxis.set_ticks_position("lower")      # z-axis on the left, as in MATLAB
    ax.zaxis.set_label_position("lower")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85))
        axis._axinfo["grid"].update(color=(0.88, 0.88, 0.88), linewidth=0.5)
    ax.tick_params(labelsize=7, pad=0)
    ax.legend(handles=[bs, tg], loc="upper left", fontsize=7, markerscale=0.9, borderpad=0.3,
              handletextpad=0.3, bbox_to_anchor=(0.0, 0.98))
    if title:
        ax.set_title(title, y=-0.04, fontsize=10)


def plot_fig3(res_nf, res_ff, p, path_stem: str, floor_db: float = -60.0):
    """Normalised MUSIC spectrum: (a) near-field ISAC, (b) far-field ISAC; also one file per panel."""
    with plt.rc_context(MATLAB_RC):
        fig = plt.figure(figsize=(9.6, 4.4))
        for i, (res, cap) in enumerate(((res_nf, "(a) Near-field ISAC."), (res_ff, "(b) Far-field ISAC."))):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d", computed_zorder=False)
            _spectrum_axes(ax, res, p, floor_db, cap)
        fig.subplots_adjust(left=0.0, right=0.97, bottom=0.08, top=1.0, wspace=0.05)
        for ext in ("png", "pdf"):
            fig.savefig(f"{path_stem}.{ext}", bbox_inches="tight")
        plt.close(fig)
        for res, tag in ((res_nf, "a_near_field"), (res_ff, "b_far_field")):
            fig = plt.figure(figsize=(5.2, 4.4))
            ax = fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)
            _spectrum_axes(ax, res, p, floor_db, None)
            fig.savefig(f"{path_stem}_{tag}.png", bbox_inches="tight")
            plt.close(fig)
