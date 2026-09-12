"""Independent waveform, RF, CRB and equal-budget audit for PSO/DE results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from validate_results import (  # noqa: E402
    check_source_hashes,
    check_waveform,
    finite_difference_crb,
    require,
)

from near_field_isac.communication import communication_rates  # noqa: E402
from near_field_isac.config import SimulationConfig  # noqa: E402
from near_field_isac.meta_experiment import aggregate_results, distance_folder_name  # noqa: E402
from near_field_isac.metaheuristics import SearchSettings, focusing_matrix  # noqa: E402


def check_csv(path, expected):
    with path.open(newline="", encoding="utf-8") as handle:
        actual = list(csv.DictReader(handle))
    require(len(actual) == len(expected), f"CSV row count differs: {path.name}")
    for row, saved in zip(actual, expected, strict=True):
        require(row.keys() == saved.keys(), f"CSV columns differ: {path.name}")
        for key, value in saved.items():
            require(
                row[key] == ("" if value is None else str(value)),
                f"CSV value differs: {path.name}, {key}",
            )


def check_artifact_coverage(output, summary):
    distances, seeds = summary["distances_m"], summary["search_seeds"]
    require(
        distances
        and np.isfinite(distances).all()
        and min(distances) > 0
        and len(set(distances)) == len(distances),
        "Invalid requested distances",
    )
    require(
        seeds
        and all(type(seed) is int and seed >= 0 for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "Invalid search seeds",
    )
    expected = {
        (distance, method, seed)
        for distance in distances
        for method, seed in [("fully-digital-sdr", None), ("hybrid-two-stage-sdr", None)]
        + [(f"hybrid-{algorithm}-sdr", seed) for algorithm in ("pso", "de") for seed in seeds]
    }
    keys = [(row["distance_m"], row["method"], row["search_seed"]) for row in summary["rows"]]
    require(
        len(keys) == len(expected) and set(keys) == expected,
        "Run rows do not cover the requested distances, methods and seeds exactly",
    )
    files = [row["waveform_file"] for row in summary["rows"]]
    require(len(set(files)) == len(files), "Waveforms reused across runs")
    setup = json.loads((output / "settings.json").read_text())
    for key, value in setup.items():
        require(summary.get(key) == value, f"Summary/settings mismatch: {key}")
    for distance in distances:
        rows = json.loads((output / distance_folder_name(distance) / "rows.json").read_text())
        saved = [row for row in summary["rows"] if row["distance_m"] == distance]

        def key(row):
            return row["method"], -1 if row["search_seed"] is None else row["search_seed"]

        require(sorted(rows, key=key) == sorted(saved, key=key), "Per-distance rows differ")
    aggregated = aggregate_results(summary["rows"])
    require(aggregated == summary["aggregated"], "Reported aggregates differ from run data")
    check_csv(output / "runs.csv", summary["rows"])
    check_csv(output / "comparison.csv", aggregated)
    for name in (
        "figure4_metaheuristic_comparison.png",
        "metaheuristic_convergence.png",
        "report.md",
    ):
        path = output / name
        require(path.is_file() and path.stat().st_size > 0, f"Missing output: {name}")


def check_search_detail(detail, row, summary, trace, baseline):
    """Check each logged evaluation and generation, not only total request counts."""
    settings = summary["search_settings"]
    population, generations = settings["population"], settings["generations"]
    count = population * (generations + 1)
    require(
        detail["algorithm"] == row["method"].split("-")[1]
        and detail["search_seed"] == row["search_seed"],
        "Search identity differs",
    )
    require(
        detail["fitness_evaluations"] == len(detail["evaluations"]) == count,
        "Search evaluation budget differs",
    )
    require(len(detail["history"]) == generations + 1, "Incomplete search history")
    require(
        np.isclose(detail["baseline_objective"], baseline, rtol=1e-8, atol=0),
        "Search baseline differs",
    )
    dimension = summary["dimension"]
    seen = {np.zeros(dimension).tobytes(): baseline}
    costs = []
    for index, item in enumerate(detail["evaluations"], start=1):
        position = np.asarray(item["position"], dtype=float)
        require(item["evaluation"] == index, "Search evaluation numbering differs")
        require(
            position.shape == (dimension,)
            and np.isfinite(position).all()
            and (np.abs(position) <= 1).all(),
            "Search position outside bounds",
        )
        key = position.tobytes()
        require(
            type(item["cached"]) is bool and item["cached"] == (key in seen),
            "Search cache accounting differs",
        )
        objective = item["objective"]
        feasible = objective is not None and np.isfinite(objective) and objective > 0
        require(
            type(item["feasible"]) is bool and item["feasible"] == feasible,
            "Search feasibility differs",
        )
        require(objective is None or feasible, "Invalid search objective")
        if item["cached"]:
            previous = seen[key]
            require(objective == previous, "Cached objective differs")
        seen[key] = objective
        costs.append(objective if feasible else np.inf)
    require(
        np.array_equal(detail["evaluations"][0]["position"], np.zeros(dimension)),
        "Search does not include baseline initialization",
    )
    for generation, point in enumerate(detail["history"]):
        evaluations = population * (generation + 1)
        require(
            point["generation"] == generation and point["evaluations"] == evaluations,
            "Search history coverage differs",
        )
        require(
            np.isclose(point["best_objective"], min(costs[:evaluations]), rtol=1e-8, atol=0),
            "Search history does not match evaluated objectives",
        )
    for key, value in (
        ("fitness_evaluations", count),
        ("solver_calls", sum(not item["cached"] for item in detail["evaluations"])),
        ("infeasible_evaluations", sum(not item["feasible"] for item in detail["evaluations"])),
    ):
        require(detail[key] == row[key] == value, f"Search accounting differs: {key}")
    require(
        np.isclose(detail["best_objective"], trace, rtol=1e-8, atol=0)
        and np.isclose(min(costs), trace, rtol=1e-8, atol=0),
        "Search winner differs",
    )


def validate(output: Path, original: Path | None = None) -> dict:
    (output / "validation.json").write_text(
        json.dumps({"passed": False, "status": "Validation started; no successful audit yet"})
        + "\n",
        encoding="utf-8",
    )
    summary = json.loads((output / "summary.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    require(summary["config"] == provenance["config"], "Summary/provenance config mismatch")
    config = SimulationConfig(**summary["config"])
    settings = SearchSettings(**summary["search_settings"])
    require(summary["dimension"] == 2 * config.n_rf_chains, "Search dimension mismatch")
    require(
        {"metaheuristics.py", "meta_experiment.py"} <= provenance["source_sha256"].keys(),
        "Incomplete search source manifest",
    )
    changed_sources = check_source_hashes(provenance)
    require(not set(changed_sources) - {"plotting.py"}, "Numerical source hashes changed")
    if changed_sources:
        rendering = json.loads((output / "rendering_provenance.json").read_text())
        require(
            rendering["plotting_source_sha256"]
            == hashlib.sha256((ROOT / "src/near_field_isac/plotting.py").read_bytes()).hexdigest(),
            "Rendering source hash differs",
        )
        for group in ("input_sha256", "output_sha256"):
            require(bool(rendering[group]), f"Empty rendering manifest: {group}")
            for name, sha in rendering[group].items():
                require(
                    hashlib.sha256((output / name).read_bytes()).hexdigest() == sha,
                    f"Rendering {group} differs: {name}",
                )
    check_artifact_coverage(output, summary)
    errors, rate_margins, power_margins = [], [], []
    checks, convergence = [], []
    with np.load(output / "scenario.npz") as scenario:
        for row in summary["rows"]:
            is_hybrid = row["method"].startswith("hybrid")
            w = scenario["receive_combiner"] if is_hybrid else None
            path = output / row["waveform_file"]
            with np.load(path) as data:
                check_waveform(config, data, scenario, rate=config.min_rate, combiner=w)
                covariance, beams = data["covariance"], data["beamformers"]
                rates = communication_rates(scenario["communication_channels"], covariance, beams)
                rate_margin = float(np.min(rates) - config.min_rate)
                power_margin = float(config.transmit_power - np.trace(covariance).real)
                metadata = json.loads(path.with_suffix(".json").read_text())
                require(
                    row["validation_passed"] is True and metadata["validation_passed"] is True,
                    "Saved solver validation failed",
                )
                for key, value in (
                    ("minimum_achieved_rate", float(np.min(rates))),
                    ("minimum_rate_margin", rate_margin),
                    ("transmit_power_margin", power_margin),
                ):
                    require(
                        np.isclose(row[key], value, rtol=1e-8, atol=1e-8),
                        f"Saved run metric differs: {key}",
                    )
                require(rate_margin >= -1e-05, path)
                require(power_margin >= -config.transmit_power * 1e-07, path)
                require(
                    np.linalg.eigvalsh(covariance).min() >= -config.transmit_power * 1e-07, path
                )
                residual = covariance - beams @ beams.conj().T
                require(np.linalg.eigvalsh(residual).min() >= -config.transmit_power * 1e-07, path)
                reported = np.array([row["range_rcrb_m"], row["angle_rcrb_deg"]])
                require(np.isfinite(reported).all() and (reported > 0).all(), "Invalid saved RCRB")
                numerical = finite_difference_crb(
                    config, covariance, complex(scenario["target_gain"]), row["distance_m"], w
                )
                error = float(np.max(np.abs(numerical / reported - 1)))
                require(error < 2e-05, (path, error))
                trace = reported[0] ** 2 + np.deg2rad(reported[1]) ** 2
                require(np.isclose(trace, row["physical_crb_trace"], rtol=1e-08, atol=0), path)
                require(
                    np.isclose(trace, metadata["physical_crb_trace"], rtol=1e-8, atol=0),
                    "Saved solution metadata trace differs",
                )
                if row["search_seed"] is not None:
                    baseline = next(
                        r["physical_crb_trace"]
                        for r in summary["rows"]
                        if r["distance_m"] == row["distance_m"]
                        and r["method"] == "hybrid-two-stage-sdr"
                    )
                    require(trace <= baseline * (1 + 1e-08), path)
                    detail = json.loads(path.with_name(path.stem + "_search.json").read_text())
                    check_search_detail(detail, row, summary, trace, baseline)
                    local_config = config.with_updates(target_range=row["distance_m"])
                    focus_scenario = SimpleNamespace(
                        user_ranges=scenario["user_ranges"],
                        user_angles=scenario["user_angles"],
                        target_range=row["distance_m"],
                        target_angle=config.target_angle,
                    )
                    require(
                        np.allclose(
                            data["transmit_basis"],
                            focusing_matrix(
                                local_config, focus_scenario, data["search_position"], settings
                            ),
                            rtol=0,
                            atol=1e-10,
                        ),
                        "Saved winner RF focus differs",
                    )
                    require(
                        any(
                            np.array_equal(item["position"], data["search_position"])
                            and item["feasible"]
                            and np.isclose(item["objective"], trace, rtol=1e-8, atol=0)
                            for item in detail["evaluations"]
                        ),
                        "Saved winner position was not evaluated",
                    )
                    for point in detail["history"]:
                        convergence.append(
                            {
                                "distance_m": row["distance_m"],
                                "method": row["method"],
                                "search_seed": row["search_seed"],
                                **point,
                                "objective_over_original_hb": point["best_objective"]
                                / detail["baseline_objective"],
                            }
                        )
                errors.append(error)
                rate_margins.append(rate_margin)
                power_margins.append(power_margin)
                checks.append(
                    {
                        "waveform_file": row["waveform_file"],
                        "passed": True,
                        "finite_difference_relative_error": error,
                    }
                )
        for distance in summary["distances_m"]:
            folder = output / distance_folder_name(distance)
            for seed in summary["search_seeds"]:
                initial = []
                for method in ("pso", "de"):
                    detail = json.loads(
                        (folder / f"hybrid-{method}-sdr_seed_{seed}_search.json").read_text()
                    )
                    initial.append(
                        [
                            e["position"]
                            for e in detail["evaluations"][
                                : summary["search_settings"]["population"]
                            ]
                        ]
                    )
                require(initial[0] == initial[1], (distance, seed))
        original_error = None
        if original is not None:
            old = json.loads((original / "figure4_summary.json").read_text())
            with np.load(original / "scenario.npz") as old_scenario:
                for key in scenario.files:
                    require(np.array_equal(scenario[key], old_scenario[key]), key)
            differences = []
            for row in summary["rows"]:
                if row["search_seed"] is None:
                    previous = next(
                        r
                        for r in old["rows"]
                        if r["architecture"] == row["method"]
                        and r["distance_m"] == row["distance_m"]
                    )
                    differences.extend(
                        abs(row[k] / previous[k] - 1) for k in ("range_rcrb_m", "angle_rcrb_deg")
                    )
            original_error = max(differences)
            require(original_error < 0.0001, original_error)
    convergence.sort(
        key=lambda row: (row["distance_m"], row["method"], row["search_seed"], row["generation"])
    )
    check_csv(output / "convergence.csv", convergence)
    report = {
        "passed": True,
        "numerical_source_hashes_match": True,
        "presentation_sources_changed_since_solve": changed_sources,
        "waveforms_checked": len(checks),
        "max_finite_difference_relative_error_percent": 100 * max(errors),
        "minimum_rate_margin": min(rate_margins),
        "minimum_power_margin": min(power_margins),
        "original_figure4_max_relative_difference_percent": (
            100 * original_error if original_error is not None else None
        ),
        "checks": checks,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "checks"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/metaheuristics")
    parser.add_argument("--original-figure4", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.results, args.original_figure4), indent=2))
