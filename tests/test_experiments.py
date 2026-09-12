"""Experiment input failures must precede solver work and output creation."""

import numpy as np
import pytest

import near_field_isac.experiments as experiments
from near_field_isac.config import SimulationConfig


@pytest.mark.parametrize(
    ("driver", "samples"),
    [
        (experiments.reproduce_figure2, []),
        (experiments.reproduce_figure2, [0.0, -1.0]),
        (experiments.reproduce_figure2, [0.0, np.nan]),
        (experiments.reproduce_figure2, [0.0, np.inf]),
        (experiments.reproduce_figure4, []),
        (experiments.reproduce_figure4, [20.0, 0.0]),
        (experiments.reproduce_figure4, [20.0, -1.0]),
        (experiments.reproduce_figure4, [20.0, np.nan]),
        (experiments.reproduce_figure4, [20.0, np.inf]),
    ],
)
def test_invalid_sweeps_do_not_solve_or_create_artifacts(driver, samples, monkeypatch, tmp_path):
    def unexpected_solver(*args, **kwargs):
        pytest.fail("an invalid sweep must fail before its first solver call")

    monkeypatch.setattr(experiments, "solve_fully_digital_sdr", unexpected_solver)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="rates|distances"):
        driver(SimulationConfig.smoke(), iter(samples), output_dir=output)
    assert not output.exists()
