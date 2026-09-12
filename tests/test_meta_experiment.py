import numpy as np
import pytest

from near_field_isac.config import SimulationConfig
from near_field_isac.meta_experiment import distance_folder_name, reproduce_metaheuristics
from near_field_isac.metaheuristics import SearchSettings


def test_distinct_distances_do_not_overwrite_each_others_waveforms():
    distances = [20.0, np.nextafter(20.0, 21.0), 20.00001, 20.00002]
    assert len({distance_folder_name(distance) for distance in distances}) == len(distances)
    assert distance_folder_name(20.0) == "distance_20m"


@pytest.mark.parametrize("seeds, workers", [([1.5], 1), ([True], 1), ([1], 1.5), ([1], True)])
def test_invalid_search_run_inputs_fail_before_writing_output(tmp_path, seeds, workers):
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="integer"):
        reproduce_metaheuristics(
            SimulationConfig.smoke(), [20.0], output_dir=output,
            settings=SearchSettings(population=4, generations=1),
            search_seeds=seeds, workers=workers,
        )
    assert not output.exists()
