"""Measure objective and component stability against a second solver/tolerance."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from near_field_isac.channels import generate_scenario  # noqa: E402
from near_field_isac.config import SimulationConfig  # noqa: E402
from near_field_isac.fim import root_crb  # noqa: E402
from near_field_isac.optimization import (  # noqa: E402
    random_hybrid_combiner,
    solve_fully_digital_sdr,
    solve_hybrid_sdr,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    provenance = json.loads((args.results / "figure2/provenance.json").read_text())
    config = SimulationConfig(**provenance["config"])
    rng = np.random.default_rng(config.seed)
    scenario = generate_scenario(config, rng)
    combiner = random_hybrid_combiner(config, rng)
    cases = []
    for architecture, solve in [("FD", solve_fully_digital_sdr), ("HB", solve_hybrid_sdr)]:
        for rate in [0.0, 1.0, 5.0, 10.7]:
            for solver, tolerance in [("MOSEK", 1e-11), ("MOSEK", 1e-12), ("CLARABEL", 1e-11)]:
                row = {
                    "architecture": architecture,
                    "rate": rate,
                    "solver": solver,
                    "tolerance": tolerance,
                }
                try:
                    result = solve(
                        config,
                        scenario,
                        min_rate=rate,
                        solver=solver,
                        tolerance=tolerance,
                        solver_threads=2,
                        **({"receive_combiner": combiner} if architecture == "HB" else {}),
                    )
                    rcrb = root_crb(result.metadata["physical_crb"])
                    row.update(
                        status=result.status,
                        objective=result.objective,
                        range_rcrb_m=rcrb[0],
                        angle_rcrb_deg=rcrb[1],
                        rate_margin=result.metadata["minimum_rate_margin"],
                        epigraph_relative_error=result.metadata["epigraph_relative_error"],
                    )
                except (RuntimeError, ValueError) as error:
                    row.update(status="rejected", reason=str(error))
                cases.append(row)
                print(row, flush=True)
    output = args.results / "validation"
    output.mkdir(exist_ok=True)
    (output / "solver_convergence.json").write_text(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
