import numpy as np
import pytest

from near_field_isac.config import SimulationConfig
from near_field_isac.pareto import ParetoSettings
from near_field_isac.pareto_experiment import aggregate_pareto_results, reproduce_pareto
from near_field_isac.plotting import PARETO_METHODS


def test_component_capped_aggregation_reports_restart_mean_and_worst_component() -> None:
    rows = []
    for method, *_ in PARETO_METHODS:
        for value in (0.5, 0.75):
            rows.append(
                {
                    "distance_m": 20.0,
                    "method": method,
                    "range_rcrb_m": value,
                    "angle_rcrb_deg": value,
                    "physical_crb_trace": value,
                    "normalized_range_variance": value,
                    "normalized_angle_variance": value,
                    "normalized_crb_objective": value,
                    "maximum_component_ratio": value,
                    "wall_seconds": value,
                    "fitness_evaluations": 10,
                    "solver_calls": 9,
                    "infeasible_evaluations": 0,
                }
            )

    aggregate = aggregate_pareto_results(rows)

    assert len(aggregate) == len(PARETO_METHODS)
    assert all(row["runs"] == 2 for row in aggregate)
    assert all(np.isclose(row["normalized_crb_objective_mean"], 0.625) for row in aggregate)
    assert all(np.isclose(row["maximum_component_ratio"], 0.75) for row in aggregate)


@pytest.mark.parametrize("seeds, workers", [([1.5], 1), ([1], True)])
def test_invalid_component_capped_inputs_fail_before_writing(tmp_path, seeds, workers) -> None:
    output = tmp_path / "pareto"
    with pytest.raises(ValueError, match="integer"):
        reproduce_pareto(
            SimulationConfig.smoke(),
            [20.0],
            output_dir=output,
            settings=ParetoSettings(population=4, generations=1, polish_evaluations=0),
            search_seeds=seeds,
            workers=workers,
        )
    assert not output.exists()
