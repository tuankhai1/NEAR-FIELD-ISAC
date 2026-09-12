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


def require(condition, message):
    """Validation must remain active when Python runs with optimization (-O)."""
    if not condition:
        raise ValueError(str(message))


def check_source_hashes(provenance):
    hashes = provenance["source_sha256"]
    required = {"channels.py", "communication.py", "config.py", "fim.py", "optimization.py"}
    require(required <= hashes.keys(), "Incomplete numerical source manifest")
    changed = []
    for name, expected in hashes.items():
        path = ROOT / "src/near_field_isac" / name
        require(path.name == name and path.is_file(), f"Invalid source manifest entry: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    return changed


def check_waveform(config, data, scenario, *, rate, combiner=None):
    """Check finite Hermitian physical arrays before eigvalsh can ignore corruption."""
    covariance, beams = data["covariance"], data["beamformers"]
    require(covariance.shape == (config.n_antennas, config.n_antennas), "Covariance shape")
    require(beams.shape == (config.n_antennas, config.n_users), "Beamformer shape")
    require(np.isfinite(covariance).all() and np.isfinite(beams).all(), "Nonfinite waveform")
    tolerance = config.transmit_power * 1e-7
    require(
        np.allclose(covariance, covariance.conj().T, rtol=0, atol=tolerance),
        "Covariance is not Hermitian",
    )
    require(np.linalg.eigvalsh(covariance).min() >= -tolerance, "Covariance is not PSD")
    channels = scenario["communication_channels"]
    require(channels.shape == (config.n_antennas, config.n_users), "Channel shape")
    require(np.isfinite(channels).all(), "Nonfinite channels")
    rates = communication_rates(channels, covariance, beams)
    require(np.isfinite(rates).all(), "Nonfinite communication rates")
    residual = covariance - beams @ beams.conj().T
    margin = float(rates.min() - rate)
    power = float(np.trace(covariance).real)
    eigenvalue = float(np.linalg.eigvalsh(residual).min())
    require(margin >= -1e-5, "Communication rate constraint failed")
    require(power <= config.transmit_power + tolerance, "Transmit power constraint failed")
    require(eigenvalue >= -tolerance, "Residual sensing covariance is not PSD")
    if "rates" in data:
        require(np.allclose(data["rates"], rates, rtol=1e-8, atol=1e-8), "Saved rates differ")
    if "sensing_covariance" in data:
        require(
            np.allclose(data["sensing_covariance"], residual, rtol=1e-7, atol=tolerance),
            "Saved sensing covariance differs",
        )
    if combiner is not None:
        require(combiner.shape == (config.n_rf_chains, config.n_antennas), "Combiner shape")
        require(
            np.isfinite(combiner).all() and np.allclose(np.abs(combiner), 1, atol=1e-10, rtol=0),
            "Invalid RF combiner",
        )
        analog = data["transmit_basis"]
        require(analog.shape == (config.n_antennas, config.n_rf_chains), "RF basis shape")
        require(
            np.isfinite(analog).all() and np.allclose(np.abs(analog), 1, rtol=0, atol=1e-10),
            "RF basis must have unit modulus",
        )
        require(
            np.allclose(
                analog @ data["baseband_covariance"] @ analog.conj().T,
                covariance,
                rtol=1e-7,
                atol=tolerance,
            ),
            "RF covariance reconstruction",
        )
        require(
            np.allclose(
                analog @ data["baseband_communication_beamformers"],
                beams,
                rtol=1e-7,
                atol=tolerance,
            ),
            "RF beamformer reconstruction",
        )
    return {"rate_margin": margin, "power_mw": power, "minimum_residual_eigenvalue": eigenvalue}


def check_curve_coverage(data, summary, x_name):
    axis_key = "rates" if x_name == "minimum_rate" else "distances_m"
    axis = summary[axis_key]
    require(
        axis and np.isfinite(axis).all() and len(set(axis)) == len(axis),
        "Sweep coordinates must be finite, nonempty and distinct",
    )
    expected = {
        (float(x), arch) for x in axis for arch in ("fully-digital-sdr", "hybrid-two-stage-sdr")
    }
    keys = [(float(row[x_name]), row["architecture"]) for row in data]
    require(
        len(keys) == len(set(keys)) and set(keys) == expected,
        "Curve rows do not cover the requested sweep exactly",
    )
    saved = {(float(row[x_name]), row["architecture"]): row for row in summary["rows"]}
    require(
        len(summary["rows"]) == len(expected) and saved.keys() == expected,
        "Summary rows do not cover the requested sweep exactly",
    )
    for row, key in zip(data, keys, strict=True):
        for name, value in saved[key].items():
            require(name in row and str(value) == row[name], f"CSV/summary mismatch: {key}, {name}")


def rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, data):
    require(bool(data), f"No validation rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def finite_difference_crb(config, covariance, gain, distance, combiner, *, model="near"):
    """Independent mean-echo Jacobian with central finite differences."""
    position = np.linspace(-config.aperture / 2, config.aperture / 2, config.n_antennas)

    def channel(r, theta):
        if model == "far":
            a = np.exp(2j * np.pi / config.wavelength * position * np.cos(theta))
            return np.outer(a, a)
        length = np.sqrt(r * r + position * position - 2 * r * position * np.cos(theta))
        a = np.exp(-2j * np.pi / config.wavelength * (length - r))
        return np.outer(a, a)

    theta = config.target_angle
    base = channel(distance, theta)
    dr = (channel(distance + 1e-4, theta) - channel(distance - 1e-4, theta)) / 2e-4
    dt = (channel(distance, theta + 1e-7) - channel(distance, theta - 1e-7)) / 2e-7
    derivative = [gain * dr, gain * dt, base, 1j * base]
    if model == "far":
        derivative = derivative[1:]
    else:
        require(model == "near", "Unknown CRB model")
    values, vectors = np.linalg.eigh(covariance)
    x = vectors * np.sqrt(np.maximum(values, 0) * config.coherent_block_length)[None, :]
    if combiner is not None:
        derivative = [combiner @ d for d in derivative]
    jac = np.column_stack([(d @ x).ravel() for d in derivative])
    noise = config.noise_power * (config.n_antennas if combiner is not None else 1)
    information = 2 / noise * (jac.conj().T @ jac).real
    require(
        np.isfinite(information).all() and np.all(np.diag(information) > 0),
        "Finite-difference information is unidentifiable",
    )
    scale = np.diag(1 / np.sqrt(np.diag(information)))
    balanced = scale @ information @ scale
    require(
        np.linalg.eigvalsh(balanced).min() > 0,
        "Finite-difference information is not positive definite",
    )
    inverse = scale @ np.linalg.inv(balanced) @ scale
    if model == "far":
        return np.rad2deg(np.sqrt(inverse[0, 0]))
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
                    require(
                        bool(part), f"No matching paper points for {experiment}, {arch}, {quantity}"
                    )
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


def check_figure3(folder):
    provenance = json.loads((folder / "provenance.json").read_text())
    require(not check_source_hashes(provenance), "Figure 3 source hashes changed")
    config = SimulationConfig(**provenance["config"])
    summary = json.loads((folder / "figure3_summary.json").read_text())
    with (
        np.load(folder / "scenario.npz") as scenario,
        np.load(folder / "figure3_music_data.npz") as data,
    ):
        combiner = scenario["receive_combiner"] if summary["optimizer"] == "hybrid" else None
        if summary["optimizer"] != "zf":
            with np.load(folder / "nominal.npz") as nominal:
                for key in ("covariance", "beamformers", "rates"):
                    require(np.array_equal(data[key], nominal[key]), f"Figure 3 {key} mismatch")
                physical = check_waveform(
                    config, nominal, scenario, rate=config.min_rate, combiner=combiner
                )
        else:
            physical = check_waveform(config, data, scenario, rate=config.min_rate)
        require(
            np.allclose(
                data["rates"], summary["communication_rates_bit_s_hz"], rtol=1e-8, atol=1e-8
            ),
            "Figure 3 summary rates differ",
        )
        reported = np.array([summary["range_rcrb_m"], summary["angle_rcrb_deg"]])
        require(np.isfinite(reported).all() and (reported > 0).all(), "Invalid Figure 3 RCRB")
        independent = finite_difference_crb(
            config,
            data["covariance"],
            complex(scenario["target_gain"]),
            config.target_range,
            combiner,
        )
        error = float(np.max(np.abs(independent / reported - 1)))
        require(error < 2e-4, "Figure 3 finite-difference CRB differs")
        x, y = data["x_grid"], data["y_grid"]
        grid_size = summary["preset"]["grid_size"]
        require(x.shape == y.shape == (grid_size, grid_size), "Figure 3 grid shape")
        require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite Figure 3 grid")
        for name, key in (("near", "near_field_estimate"), ("far", "far_field_grid_maximum")):
            spectrum = data[name + "_spectrum"]
            require(
                spectrum.shape == x.shape
                and np.isfinite(spectrum).all()
                and (spectrum > 0).all()
                and np.isclose(spectrum.max(), 1, atol=1e-12, rtol=0),
                "Invalid MUSIC spectrum",
            )
            index = int(np.argmax(spectrum))
            px, py = x.ravel()[index], y.ravel()[index]
            estimate = {
                "range_m": np.hypot(px, py),
                "angle_deg": np.rad2deg(np.arctan2(py, px)),
                "x_m": px,
                "y_m": py,
            }
            for quantity, value in summary[key].items():
                require(
                    np.isclose(value, estimate[quantity], rtol=1e-10, atol=1e-10),
                    f"MUSIC {name} estimate does not match saved spectrum: {quantity}",
                )
        ray = (np.hypot(x, y) > 0) & np.isclose(
            np.arctan2(y, x), config.target_angle, rtol=0, atol=1e-12
        )
        values = data["far_spectrum"][ray]
        spread = float(np.ptp(values) / np.mean(values)) if len(values) > 1 else None
        if spread is not None:
            require(spread < 1e-6, "Far-field MUSIC varies along the same target ray")
    return summary, {**physical, "independent_crb_relative_error": error, "passed": True}, spread


def check_far_references(folder, config, scenario, summary, curve_rows):
    checks = []
    for arch, key in (
        ("fully-digital-sdr", "fully_digital_angle_rcrb_deg"),
        ("hybrid-two-stage-sdr", "hybrid_angle_rcrb_deg"),
    ):
        combiner = scenario["receive_combiner"] if arch.startswith("hybrid") else None
        with np.load(folder / ("far_" + arch + ".npz")) as data:
            physical = check_waveform(
                config, data, scenario, rate=config.min_rate, combiner=combiner
            )
            bound = finite_difference_crb(
                config,
                data["covariance"],
                complex(scenario["target_gain"]),
                config.target_range,
                combiner,
                model="far",
            )
        reported = summary["far_field_reference"][key]
        require(np.isfinite(reported) and reported > 0, "Invalid far-field reference")
        error = float(abs(bound / reported - 1))
        require(error < 2e-4, f"Far-field {arch} CRB differs")
        for row in curve_rows:
            if row["architecture"] == arch:
                require(
                    np.isclose(
                        float(row["far_field_angle_rcrb_deg"]), reported, rtol=1e-10, atol=0
                    ),
                    "Far-field CSV reference differs",
                )
        checks.append(
            dict(
                experiment="figure4_far",
                architecture=arch,
                x=config.target_range,
                **physical,
                independent_crb_relative_error=error,
                passed=True,
            )
        )
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--reference", type=Path, default=ROOT / "docs/paper_reference")
    args = parser.parse_args()
    output = args.results / "validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps({"all_physical_checks_passed": False, "status": "Validation incomplete"}) + "\n"
    )
    (output / "report.md").write_text("# Validation incomplete\n\nNo successful audit yet.\n")
    physical, comparisons = [], []
    for experiment, filename, x_name in [
        ("figure2", "figure2_rcrb_vs_rate.csv", "minimum_rate"),
        ("figure4", "figure4_rcrb_vs_distance.csv", "distance_m"),
    ]:
        folder = args.results / experiment
        provenance = json.loads((folder / "provenance.json").read_text())
        changed = check_source_hashes(provenance)
        require(not changed, f"{experiment} source hashes changed: {changed}")
        config = SimulationConfig(**provenance["config"])
        scenario = np.load(folder / "scenario.npz")
        all_rows = rows(folder / filename)
        experiment_summary = json.loads((folder / (experiment + "_summary.json")).read_text())
        check_curve_coverage(all_rows, experiment_summary, x_name)
        reference = rows(args.reference / (experiment + ".csv"))
        for row in all_rows:
            data = np.load(folder / row["waveform_file"])
            covariance = data["covariance"]
            rate = float(row[x_name]) if experiment == "figure2" else config.min_rate
            distance = float(row[x_name]) if experiment == "figure4" else config.target_range
            combiner = (
                scenario["receive_combiner"] if row["architecture"].startswith("hybrid") else None
            )
            checks = check_waveform(config, data, scenario, rate=rate, combiner=combiner)
            independent = finite_difference_crb(
                config, covariance, complex(scenario["target_gain"]), distance, combiner
            )
            saved = np.array([float(row["range_rcrb_m"]), float(row["angle_rcrb_deg"])])
            require(np.isfinite(saved).all() and (saved > 0).all(), "Invalid saved RCRB")
            trace = saved[0] ** 2 + np.deg2rad(saved[1]) ** 2
            require(
                np.isclose(trace, float(row["physical_crb_trace"]), rtol=1e-8, atol=0),
                "Saved CRB trace differs",
            )
            relative_error = float(np.max(np.abs(independent / saved - 1)))
            passed = relative_error < 2e-4
            physical.append(
                dict(
                    experiment=experiment,
                    architecture=row["architecture"],
                    x=float(row[x_name]),
                    **checks,
                    independent_crb_relative_error=relative_error,
                    passed=passed,
                )
            )
            data.close()
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
                    require(np.isfinite(expected) and expected > 0, "Invalid paper reference")
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
                ordered = sorted(
                    (r for r in all_rows if r["architecture"] == arch),
                    key=lambda r: float(r[x_name]),
                )
                objectives = [float(r["physical_crb_trace"]) for r in ordered]
                require(
                    np.all(np.diff(objectives) >= -1e-7 * np.max(objectives)),
                    f"Figure 2 objective decreases as rates tighten: {arch}",
                )
        else:
            physical.extend(
                check_far_references(folder, config, scenario, experiment_summary, all_rows)
            )
        scenario.close()
    f3, f3_checks, far_spread = check_figure3(args.results / "figure3")
    physical.append(
        dict(
            experiment="figure3",
            architecture=f3["optimizer"],
            x=f3["target"]["range_m"],
            **f3_checks,
        )
    )
    write_rows(output / "physical_checks.csv", physical)
    write_rows(output / "paper_comparison.csv", comparisons)
    compare_plot(comparisons, output / "comparison_to_paper.png")
    compare_plot(comparisons, output / "normalized_comparison.png", normalized=True)
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
        "far_music_relative_spread_along_target_ray": far_spread,
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
        (
            "Saved waveforms pass the independent physical checks. "
            if summary["all_physical_checks_passed"]
            else "Independent physical checks FAILED. "
        )
        + "Paper comparisons below are diagnostic; passing physical checks does not establish "
        "an exact numerical reproduction.",
        "",
        f"- Waveforms independently checked (curves, far references, MUSIC): {len(physical)}; "
        f"all passed: {summary['all_physical_checks_passed']}.",
        "- Largest finite-difference CRB discrepancy: "
        f"{100 * summary['largest_independent_crb_relative_error']:.6g}%.",
        f"- Smallest communication-rate margin: {summary['smallest_rate_margin']:.6g} "
        "bit/s/Hz (acceptance: -1e-5).",
        f"- MUSIC estimate: {summary['figure3_range_m']:.9f} m, "
        f"{summary['figure3_angle_deg']:.6f} degrees; paper: "
        f"{metadata['figure3_estimated_range_m']} m, "
        f"{metadata['figure3_estimated_angle_deg']} degrees.",
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
