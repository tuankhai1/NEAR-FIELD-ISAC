"""Command-line interface for reproducibility experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import SimulationConfig
from .experiments import reproduce_figure2, reproduce_figure3, reproduce_figure4


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        choices=("quick", "paper"),
        default="quick",
        help="quick uses 17 antennas; paper uses the published 65-antenna setup",
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--solver", default="auto", help="auto, MOSEK, CLARABEL, or SCS")
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument("--verbose", action="store_true", help="show solver logs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nf-isac",
        description="Reproduce experiments from Wang, Mu, and Liu (2023).",
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

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
        help="rate values; defaults to 0,2,4 for quick and 0..10 for paper",
    )

    figure4 = subparsers.add_parser("figure4", help="RCRB versus target distance")
    _add_common_arguments(figure4)
    figure4.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=None,
        help="distance values; defaults to 5,20,40 for quick and 5:5:40 for paper",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_factory = SimulationConfig.quick if args.preset == "quick" else SimulationConfig.paper
    config = config_factory(seed=args.seed)
    common = {
        "output_dir": args.output / args.experiment,
        "solver": args.solver,
        "verbose": args.verbose,
        "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
    }
    if args.experiment == "figure3":
        grid_size = args.grid_size or (121 if args.preset == "quick" else 500)
        summary = reproduce_figure3(
            config,
            optimizer=args.optimizer,
            grid_size=grid_size,
            **common,
        )
    elif args.experiment == "figure2":
        rates = args.rates or ([0.0, 2.0, 4.0] if args.preset == "quick" else list(range(11)))
        summary = reproduce_figure2(config, rates, **common)
    else:
        distances = args.distances or (
            [5.0, 20.0, 40.0]
            if args.preset == "quick"
            else [float(value) for value in range(5, 41, 5)]
        )
        summary = reproduce_figure4(config, distances, **common)
    print(json.dumps(summary, indent=2, sort_keys=True))

