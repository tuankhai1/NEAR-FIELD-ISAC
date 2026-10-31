"""Overlay the pipeline curves on the paper's Fig. 2 and Fig. 4 (vector-extracted from arXiv v5).

    python scripts/compare_with_paper.py                       # inputs and outputs in results/alpha_paper/
    python scripts/compare_with_paper.py --results results     # e.g. the alpha = 1 runs

Inputs (from main.py / run_fig2.py / run_fig4.py with --formulation fast --alpha paper):
    fig2_curves.csv, fig2_curves_normalized.csv                               tr(CRB) / normalized objective
    fig4_curves.csv, fig4_curves_normalized.csv, fig4_curves_normalized_wrf_matched.csv
and the paper reference data in reference/ (see reference/metadata.json).  Missing pipeline files are skipped.

Outputs in <results>/comparison/:
    fig2_vs_paper.png, fig4_vs_paper.png   2 x 2 small multiples (FD/HB x distance/angle), one log axis each
    comparison.json                         pipeline / paper ratio at every common abscissa
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("FD_r", "FD_th", "HB_r", "HB_th")
TITLES = {"FD_r": "Fully digital, distance", "FD_th": "Fully digital, angle",
          "HB_r": "Hybrid, distance", "HB_th": "Hybrid, angle"}
YLABELS = {"FD_r": "RCRB (m)", "HB_r": "RCRB (m)", "FD_th": "RCRB (deg)", "HB_th": "RCRB (deg)"}

# Chart chrome and categorical slots 1-3 of the dataviz reference palette (light mode); the paper is drawn in ink.
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# (file suffix, legend label, colour, line style, marker)
VARIANTS = (("", "Pipeline, tr(CRB) (paper objective)", BLUE, "--", "o"),
            ("_normalized", "Pipeline, normalized objective", ORANGE, "-", "s"),
            ("_normalized_wrf_matched", "Pipeline, normalized + matched $W_{RF}$", AQUA, ":", "^"))
FIG4_QUANTITY = {("fully-digital-sdr", "range_rcrb_m"): "FD_r", ("fully-digital-sdr", "angle_rcrb_deg"): "FD_th",
                 ("hybrid-two-stage-sdr", "range_rcrb_m"): "HB_r", ("hybrid-two-stage-sdr", "angle_rcrb_deg"): "HB_th"}


def load_curves(path: Path):
    """Pipeline / paper Fig. 2 CSV: first column abscissa, then FD_r, FD_th, HB_r, HB_th."""
    with open(path) as fh:
        rows = list(csv.reader(fh))
    a = np.array(rows[1:], float)
    return {k: (a[:, 0], a[:, i + 1]) for i, k in enumerate(KEYS)}


def load_paper_fig4(path: Path):
    """reference/paper_fig4.csv (long format) -> {key: (r_s, value)}."""
    out = {k: ([], []) for k in KEYS}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            x, y = out[FIG4_QUANTITY[(row["architecture"], row["quantity"])]]
            x.append(float(row["distance_m"]))
            y.append(float(row["value"]))
    return {k: (np.array(x), np.array(y)) for k, (x, y) in out.items()}


def ratios(curves, paper):
    """pipeline / paper at the abscissae present in both (tolerance 1e-3)."""
    out = {}
    for k in KEYS:
        (x, y), (xp, yp) = curves[k], paper[k]
        pairs = [(float(xi), float(yi / yp[j])) for xi, yi in zip(x, y)
                 for j in np.flatnonzero(np.abs(xp - xi) < 1e-3)[:1]]
        out[k] = {"x": [p[0] for p in pairs], "ratio": [p[1] for p in pairs]}
    return out


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, which="both")


def small_multiples(paper, series, xlabel, title, path, skip=()):
    """2 x 2 panels, paper in ink + pipeline series; ``skip`` = {(suffix, key)} not drawn (identical curves)."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True, facecolor=SURF)
    handles = {}
    for ax, key in zip(axes.ravel(), KEYS):
        xp, yp = paper[key]
        handles["Paper (arXiv v5, extracted)"], = ax.semilogy(xp, yp, "-", color=INK, lw=1.4, zorder=5)
        for suffix, label, color, ls, mk, curves in series:
            if (suffix, key) in skip:
                continue
            x, y = curves[key]
            handles[label], = ax.semilogy(x, y, ls, color=color, lw=2, marker=mk, ms=6, mfc=SURF, mew=1.6, zorder=4)
        style(ax, TITLES[key], YLABELS[key])
    for ax in axes[1]:
        ax.set_xlabel(xlabel, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.865), fontsize=9, labelcolor=INK)
    fig.suptitle(title, y=0.985, fontsize=11, color=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results" / "alpha_paper"), help="folder with the pipeline CSVs")
    ap.add_argument("--reference", default=str(ROOT / "reference"), help="folder with the paper reference data")
    ap.add_argument("--out", default=None, help="output folder (default <results>/comparison)")
    args = ap.parse_args()
    res, ref = Path(args.results), Path(args.reference)
    out = Path(args.out) if args.out else res / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ref / "metadata.json").read_text())
    report = {"results": str(res), "reference": meta["source"]}

    for fig, paper, xlabel, title in (
            ("fig2", load_curves(ref / "paper_fig2.csv"), "Minimum rate $R_{min}$ (bit/s/Hz)",
             "Fig. 2: RCRB vs minimum communication rate"),
            ("fig4", load_paper_fig4(ref / "paper_fig4.csv"), "Target distance $r_s$ (m)",
             "Fig. 4: RCRB vs target distance ($R_{min}$ = 5 bit/s/Hz)")):
        series = [(s, label, c, ls, mk, load_curves(res / f"{fig}_curves{s}.csv"))
                  for s, label, c, ls, mk in VARIANTS if (res / f"{fig}_curves{s}.csv").exists()]
        if not series:
            print(f"{fig}: no pipeline curves in {res}, skipped")
            continue
        # the receive combiner only changes the hybrid curves: do not redraw the identical FD ones
        skip = {("_normalized_wrf_matched", k) for k in ("FD_r", "FD_th")}
        small_multiples(paper, series, xlabel, title, out / f"{fig}_vs_paper.png", skip)
        report[fig] = {label: ratios(curves, paper) for _, label, _, _, _, curves in series}
        print(f"{fig}: pipeline / paper (min-max over common points)")
        for label, rat in report[fig].items():
            print(f"  {label.replace('$', ''):45s}","  ".join(f"{k} {min(v['ratio']):.3f}-{max(v['ratio']):.3f}"
                                              for k, v in rat.items() if v["ratio"]))
    (out / "comparison.json").write_text(json.dumps(report, indent=1))
    print(f"saved to {out}")


if __name__ == "__main__":
    sys.exit(main())
