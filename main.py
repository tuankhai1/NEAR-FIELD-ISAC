"""Simple entry point for the near-field ISAC reproduction baseline.

Running ``python main.py`` executes the complete Fig. 2--4 pipeline with the
full paper preset. Pass a subcommand to run an individual experiment.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main() -> None:
    source_directory = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source_directory))
    cli_main = importlib.import_module("near_field_isac.cli").main

    arguments = sys.argv[1:]
    if not arguments:
        arguments = ["all"]
        print("No arguments supplied; running Figures 2--4 with the full paper preset.")
    try:
        cli_main(arguments)
    except RuntimeError as error:
        raise SystemExit(f"Experiment stopped: {error}") from None


if __name__ == "__main__":
    main()
