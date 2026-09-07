"""Recompute physical checks and compare with vector-digitized paper figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from near_field_isac.communication import communication_rates  # noqa: E402
from near_field_isac.config import SimulationConfig  # noqa: E402
from near_field_isac.plotting import GREEN, RED, STYLE  # noqa: E402


def rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, data):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def finite_difference_crb(config, covariance, gain, distance, combiner):
    """Independent mean-echo Jacobian with central finite differences."""
    position = np.linspace(-config.aperture / 2, config.aperture / 2, config.n_antennas)

    def channel(r, theta):
        length = np.sqrt(r * r + position * position - 2 * r * position * np.cos(theta))
        a = np.exp(-2j * np.pi / config.wavelength * (length - r))
        return np.outer(a, a)

    theta = config.target_angle
    base = channel(distance, theta)
    dr = (channel(distance + 1e-4, theta) - channel(distance - 1e-4, theta)) / 2e-4
    dt = (channel(distance, theta + 1e-7) - channel(distance, theta - 1e-7)) / 2e-7
    derivative = [gain * dr, gain * dt, base, 1j * base]
    values, vectors = np.linalg.eigh(covariance)
    x = vectors * np.sqrt(np.maximum(values, 0) * config.coherent_block_length)[None, :]
    if combiner is not None:
        derivative = [combiner @ d for d in derivative]
    jac = np.column_stack([(d @ x).ravel() for d in derivative])
    noise = config.noise_power * (config.n_antennas if combiner is not None else 1)
    information = 2 / noise * (jac.conj().T @ jac).real
    scale = np.diag(1 / np.sqrt(np.diag(information)))
    inverse = scale @ np.linalg.inv(scale @ information @ scale) @ scale
    diagonal = np.diag(inverse)[:2]
    return np.sqrt(diagonal) * np.array([1, 180 / np.pi])


def compare_plot(data, destination, normalized=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
        for i, experiment in enumerate(["figure2", "figure4"]):
            for j, quantity in enumerate(["range_rcrb_m", "angle_rcrb_deg"]):
                axis = axes[i, j]
                for arch, label, color in [
                    ("fully-digital-sdr", "FD", RED),
                    ("hybrid-two-stage-sdr", "HB", GREEN),
                ]:
                    part = sorted(
                        [
                            r
                            for r in data
                            if r["experiment"] == experiment
                            and r["quantity"] == quantity
                            and r["architecture"] == arch
                        ],
                        key=lambda r: r["x"],
                    )
                    x = [r["x"] for r in part]
                    actual = np.array([r["new_value"] for r in part])
                    paper = np.array([r["paper_value"] for r in part])
                    if normalized:
                        actual, paper = actual / actual[0], paper / paper[0]
                    axis.plot(x, actual, color=color, linewidth=1.1, label=f"New {label}")
                    axis.plot(
                        x,
                        paper,
                        color=color,
                        linestyle="--",
                        marker="o",
                        markersize=3,
                        markerfacecolor="white",
                        linewidth=0.8,
                        label=f"Paper {label}",
                    )
                axis.set_yscale("log")
                unit = "m" if j == 0 else "deg"
                axis.set_ylabel("Relative to first point" if normalized else f"RCRB ({unit})")
                axis.set_xlabel("Minimum rate (bit/s/Hz)" if i == 0 else "Distance (m)")
                axis.set_title(f"Fig. {2 if i == 0 else 4}: {'distance' if j == 0 else 'angle'}")
                axis.tick_params(direction="in", top=True, right=True)
                axis.legend(fontsize=8, ncols=2, fancybox=False)
        fig.tight_layout()
        fig.savefig(destination, dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--reference", type=Path, default=ROOT / "docs/paper_reference")
    args = parser.parse_args()
    output = args.results / "validation"
    output.mkdir(parents=True, exist_ok=True)
    physical, comparisons = [], []
    for experiment, filename, x_name in [
        ("figure2", "figure2_rcrb_vs_rate.csv", "minimum_rate"),
        ("figure4", "figure4_rcrb_vs_distance.csv", "distance_m"),
    ]:
        folder = args.results / experiment
        provenance = json.loads((folder / "provenance.json").read_text())
        for name, expected in provenance["source_sha256"].items():
            assert (
                hashlib.sha256((ROOT / "src/near_field_isac" / name).read_bytes()).hexdigest()
                == expected
            )
        config = SimulationConfig(**provenance["config"])
        scenario = np.load(folder / "scenario.npz")
        all_rows = rows(folder / filename)
        reference = rows(args.reference / (experiment + ".csv"))
        for row in all_rows:
            data = np.load(folder / row["waveform_file"])
            covariance, beams = data["covariance"], data["beamformers"]
            rate = float(row[x_name]) if experiment == "figure2" else config.min_rate
            distance = float(row[x_name]) if experiment == "figure4" else config.target_range
            combiner = (
                scenario["receive_combiner"] if row["architecture"].startswith("hybrid") else None
            )
            actual_rates = communication_rates(
                scenario["communication_channels"], covariance, beams
            )
            residual = covariance - beams @ beams.conj().T
            independent = finite_difference_crb(
                config, covariance, complex(scenario["target_gain"]), distance, combiner
            )
            saved = np.array([float(row["range_rcrb_m"]), float(row["angle_rcrb_deg"])])
            relative_error = float(np.max(np.abs(independent / saved - 1)))
            margin = float(actual_rates.min() - rate)
            used = float(np.trace(covariance).real)
            eigenvalue = float(np.linalg.eigvalsh(residual).min())
            passed = (
                relative_error < 2e-4
                and margin >= -1e-5
                and used <= config.transmit_power * (1 + 1e-7)
                and eigenvalue >= -1e-7 * config.transmit_power
            )
            physical.append(
                dict(
                    experiment=experiment,
                    architecture=row["architecture"],
                    x=float(row[x_name]),
                    rate_margin=margin,
                    power_mw=used,
                    minimum_residual_eigenvalue=eigenvalue,
                    independent_crb_relative_error=relative_error,
                    passed=passed,
                )
            )
            for quantity in ["range_rcrb_m", "angle_rcrb_deg"]:
                matches = [
                    r
                    for r in reference
                    if r["architecture"] == row["architecture"]
                    and r["quantity"] == quantity
                    and abs(float(r[x_name]) - float(row[x_name])) < 1e-4
                ]
                for match in matches:
                    current, expected = float(row[quantity]), float(match["value"])
                    comparisons.append(
                        dict(
                            experiment=experiment,
                            architecture=row["architecture"],
                            x=float(row[x_name]),
                            quantity=quantity,
                            new_value=current,
                            paper_value=expected,
                            ratio=current / expected,
                            relative_difference_percent=100 * (current / expected - 1),
                        )
                    )
        # Tightening rates must not improve the globally optimal total objective.
        if experiment == "figure2":
            for arch in {r["architecture"] for r in all_rows}:
                objectives = [
                    float(r["physical_crb_trace"]) for r in all_rows if r["architecture"] == arch
                ]
                assert np.all(np.diff(objectives) >= -1e-7 * np.max(objectives))
    write_rows(output / "physical_checks.csv", physical)
    write_rows(output / "paper_comparison.csv", comparisons)
    compare_plot(comparisons, output / "comparison_to_paper.png")
    compare_plot(comparisons, output / "normalized_comparison.png", normalized=True)
    f3 = json.loads((args.results / "figure3/figure3_summary.json").read_text())
    music = np.load(args.results / "figure3/figure3_music_data.npz")
    far_diagonal = np.diag(music["far_spectrum"])[1:]
    metadata = json.loads((args.reference / "metadata.json").read_text())
    summary = {
        "physical_points_checked": len(physical),
        "all_physical_checks_passed": all(r["passed"] for r in physical),
        "largest_independent_crb_relative_error": max(
            r["independent_crb_relative_error"] for r in physical
        ),
        "smallest_rate_margin": min(r["rate_margin"] for r in physical),
        "figure3_range_m": f3["near_field_estimate"]["range_m"],
        "figure3_angle_deg": f3["near_field_estimate"]["angle_deg"],
        "figure3_range_difference_from_reported_paper_m": f3["near_field_estimate"]["range_m"]
        - metadata["figure3_estimated_range_m"],
        "far_music_relative_spread_along_target_ray": float(
            np.ptp(far_diagonal) / np.mean(far_diagonal)
        ),
        "reference_source": metadata,
        "interpretation": (
            "Paper values are digitized plot coordinates, not author simulation data. "
            "No amplitude fitting or curve smoothing was applied."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    method_link = Path(os.path.relpath(ROOT / "docs/numerical_method.md", output)).as_posix()
    report = [
        "# Reproduction validation report",
        "",
        "The corrected pipeline is physically consistent within the stated numerical tolerances. "
        "Figure 3 reproduces the paper's reported location estimate. Figures 2 and 4 still "
        "differ in amplitude and, for some curves, shape; "
        "this is not an exact numerical reproduction.",
        "",
        f"- Curve points independently checked: {len(physical)}; "
        f"all passed: {summary['all_physical_checks_passed']}.",
        "- Largest finite-difference CRB discrepancy: "
        f"{100 * summary['largest_independent_crb_relative_error']:.6g}%.",
        f"- Smallest communication-rate margin: {summary['smallest_rate_margin']:.6g} "
        "bit/s/Hz (acceptance: -1e-5).",
        f"- MUSIC estimate: {summary['figure3_range_m']:.9f} m, "
        f"{summary['figure3_angle_deg']:.6f} degrees; paper: 19.952 m, 45 degrees.",
        "",
        "## Figure 2 at 5 bit/s/Hz",
        "",
        "| Architecture | Quantity | New | Digitized paper | New / paper |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        if row["experiment"] == "figure2" and abs(row["x"] - 5) < 1e-4:
            report.append(
                f"| {row['architecture']} | {row['quantity']} | {row['new_value']:.8g} | "
                f"{row['paper_value']:.8g} | {row['ratio']:.4f} |"
            )
    report += [
        "",
        "## Interpretation and limitations",
        "",
        "The changes balance the SDP without changing the paper's mixed-unit trace objective, "
        "use an exact reduced transmit space, validate recovered waveforms, reject singular CRBs, "
        "whiten hybrid MUSIC noise, and compute Figure 4's far-field references independently. "
        "Plot styling follows the paper; numerical values are neither rescaled "
        "nor smoothed to fit it.",
        "",
        "The exact random realization and complete author implementation of Figures 2 and 4 are "
        "unavailable in the supplied materials. The remaining disagreement is unresolved; "
        "it cannot "
        "be attributed to randomness alone from these checks. Figure 4's gain/reference convention "
        "and the hybrid CRB noise approximation are documented in "
        f"[numerical_method.md]({method_link}).",
        "",
        "The range variance dominates the mixed-unit objective. The zero-rate FD angle remains "
        "sensitive across solvers despite a nearly identical total objective. See "
        "solver_convergence.json for tighter-tolerance and second-solver checks, "
        "including rejected "
        "solutions. Passing physical checks is not proof that every separately reported variance "
        "is numerically unique or that the paper is reproduced.",
        "",
        "Paper references are digitized vector paths, not original simulation arrays. Comparisons "
        "use matching coordinates without interpolation. All experiment manifests record source "
        "hashes, configuration, software versions and the saved random scenario.",
        "",
        "![Absolute comparison](comparison_to_paper.png)",
        "",
        "![Shape comparison, explicitly normalized at the first point](normalized_comparison.png)",
        "",
        "Machine-readable evidence: physical_checks.csv, paper_comparison.csv and summary.json.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_physical_checks_passed"]:
        raise SystemExit("Physical cross-validation failed; inspect physical_checks.csv")


if __name__ == "__main__":
    main()
