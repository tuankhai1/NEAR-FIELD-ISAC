"""Command-line interface for reproducibility experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import SimulationConfig
from .experiments import reproduce_figure2, reproduce_figure3, reproduce_figure4

SMOKE_RATES = [0.0, 2.0, 4.0]
QUICK_RATES = [0.0, 5.0, 8.0, 9.0, 10.0]
PAPER_RATES = [float(value) for value in range(11)]
SMOKE_DISTANCES = [5.0, 20.0, 40.0]
QUICK_DISTANCES = [5.0, 10.0, 20.0, 30.0, 40.0]
PAPER_DISTANCES = [float(value) for value in range(5, 41, 5)]
GRID_SIZES = {"smoke": 121, "quick": 281, "paper": 500}


def _default_rates(preset: str) -> list[float]:
    return {
        "smoke": SMOKE_RATES,
        "quick": QUICK_RATES,
        "paper": PAPER_RATES,
    }[preset].copy()


def _default_distances(preset: str) -> list[float]:
    return {
        "smoke": SMOKE_DISTANCES,
        "quick": QUICK_DISTANCES,
        "paper": PAPER_DISTANCES,
    }[preset].copy()


def _add_common_arguments(
    parser: argparse.ArgumentParser, *, default_preset: str = "quick"
) -> None:
    parser.add_argument(
        "--preset",
        choices=("smoke", "quick", "paper"),
        default=default_preset,
        help=(
            "smoke uses a tiny model; quick and paper both use the published "
            "65-antenna model with reduced/full sampling"
        ),
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--solver", default="auto", help="auto, MOSEK, CLARABEL, or SCS")
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument(
        "--solver-threads",
        type=int,
        default=None,
        help="threads inside CLARABEL/MOSEK; omit to use the solver default",
    )
    parser.add_argument("--verbose", action="store_true", help="show solver logs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nf-isac",
        description="Reproduce experiments from Wang, Mu, and Liu (2023).",
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    all_experiments = subparsers.add_parser(
        "all", help="run Figures 2--4 as one complete pipeline"
    )
    _add_common_arguments(all_experiments, default_preset="paper")
    all_experiments.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="MUSIC points per Cartesian axis; defaults to 500 for paper",
    )
    all_experiments.add_argument("--rates", type=float, nargs="+", default=None)
    all_experiments.add_argument("--distances", type=float, nargs="+", default=None)
    all_experiments.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel processes for independent Fig. 2/4 sweep points",
    )

    figure3 = subparsers.add_parser("figure3", help="near-/far-field MUSIC spectrum")
    _add_common_arguments(figure3)
    figure3.add_argument("--optimizer", choices=("zf", "sdr", "hybrid"), default="zf")
    figure3.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="points per Cartesian axis; paper code uses 500",
    )

    figure2 = subparsers.add_parser("figure2", help="RCRB versus minimum rate")
    _add_common_arguments(figure2)
    figure2.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=None,
        help="rate values; defaults depend on the selected preset",
    )
    figure2.add_argument("--workers", type=int, default=1)

    figure4 = subparsers.add_parser("figure4", help="RCRB versus target distance")
    _add_common_arguments(figure4)
    figure4.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=None,
        help="distance values; defaults depend on the selected preset",
    )
    figure4.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_factory = {
        "smoke": SimulationConfig.smoke,
        "quick": SimulationConfig.quick,
        "paper": SimulationConfig.paper,
    }[args.preset]
    config = config_factory(seed=args.seed)
    common = {
        "output_dir": args.output / args.experiment,
        "solver": args.solver,
        "verbose": args.verbose,
        "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "solver_threads": args.solver_threads,
    }
    if args.experiment == "all":
        rates = args.rates or _default_rates(args.preset)
        distances = args.distances or _default_distances(args.preset)
        grid_size = args.grid_size or GRID_SIZES[args.preset]
        shared = {
            "solver": args.solver,
            "verbose": args.verbose,
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
            "solver_threads": args.solver_threads,
        }
        print(
            f"Full pipeline: preset={args.preset}, solver={args.solver}, "
            f"grid={grid_size}x{grid_size}, workers={args.workers}."
        )
        result_cache = {}
        print("[1/3] Reproducing Figure 2: RCRB versus minimum rate...")
        figure2_summary = reproduce_figure2(
            config,
            rates,
            output_dir=args.output / "figure2",
            workers=args.workers,
            result_cache=result_cache,
            **shared,
        )
        nominal_results = result_cache.get(float(config.min_rate))
        print("[2/3] Reproducing Figure 3: near-/far-field MUSIC...")
        figure3_summary = reproduce_figure3(
            config,
            output_dir=args.output / "figure3",
            optimizer="sdr",
            grid_size=grid_size,
            precomputed_result=(
                nominal_results[0] if nominal_results is not None else None
            ),
            **shared,
        )
        print("[3/3] Reproducing Figure 4: RCRB versus target distance...")
        figure4_summary = reproduce_figure4(
            config,
            distances,
            output_dir=args.output / "figure4",
            workers=args.workers,
            precomputed_results=(
                {float(config.target_range): nominal_results}
                if nominal_results is not None
                else None
            ),
            **shared,
        )
        details = {
            "experiment": "all",
            "preset": args.preset,
            "workers": args.workers,
            "solver_threads": args.solver_threads,
            "figure2": figure2_summary,
            "figure3": figure3_summary,
            "figure4": figure4_summary,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        summary_path = args.output / "all_summary.json"
        summary_path.write_text(
            json.dumps(details, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary = {
            "experiment": "all",
            "preset": args.preset,
            "completed": ["figure2", "figure3", "figure4"],
            "summary_file": str(summary_path),
        }
    elif args.experiment == "figure3":
        grid_size = args.grid_size or GRID_SIZES[args.preset]
        summary = reproduce_figure3(
            config,
            optimizer=args.optimizer,
            grid_size=grid_size,
            **common,
        )
    elif args.experiment == "figure2":
        rates = args.rates or _default_rates(args.preset)
        summary = reproduce_figure2(config, rates, workers=args.workers, **common)
    else:
        distances = args.distances or _default_distances(args.preset)
        summary = reproduce_figure4(config, distances, workers=args.workers, **common)
    print(json.dumps(summary, indent=2, sort_keys=True))
