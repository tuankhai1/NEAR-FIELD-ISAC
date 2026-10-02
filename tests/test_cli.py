import pytest

import near_field_isac.cli as cli
from near_field_isac.cli import build_parser


def test_all_command_defaults_to_full_paper_preset() -> None:
    arguments = build_parser().parse_args(["all"])
    assert arguments.experiment == "all"
    assert arguments.preset == "paper"
    assert arguments.grid_size is None
    assert arguments.rates is None
    assert arguments.distances is None
    assert arguments.solver_threads == 4


def test_all_command_orchestrates_all_three_figures(monkeypatch, tmp_path) -> None:
    calls = []
    nominal_results = (object(), object())

    def fake_figure2(config, rates, **kwargs):
        calls.append(("figure2", config.n_antennas, list(rates), kwargs["output_dir"]))
        kwargs["result_cache"][5.0] = nominal_results
        return {"experiment": "figure2"}

    def fake_figure3(config, **kwargs):
        calls.append(
            (
                "figure3",
                config.n_antennas,
                kwargs["grid_size"],
                kwargs["output_dir"],
                kwargs["precomputed_result"],
            )
        )
        return {"experiment": "figure3"}

    def fake_figure4(config, distances, **kwargs):
        calls.append(
            (
                "figure4",
                config.n_antennas,
                list(distances),
                kwargs["output_dir"],
                kwargs.get("precomputed_results"),
                kwargs["realization"],
            )
        )
        return {"experiment": "figure4"}

    monkeypatch.setattr(cli, "reproduce_figure2", fake_figure2)
    monkeypatch.setattr(cli, "reproduce_figure3", fake_figure3)
    monkeypatch.setattr(cli, "reproduce_figure4", fake_figure4)
    cli.main(["all", "--preset", "quick", "--output", str(tmp_path)])

    assert [call[0] for call in calls] == ["figure2", "figure3", "figure4"]
    assert all(call[1] == 65 for call in calls)
    assert calls[0][2] == [0.0, 5.0, 8.0, 9.0, 10.0]
    assert calls[1][2] == 281
    assert calls[1][4] is nominal_results[0]
    # Fig. 4 uses pathloss-free users and its own combiner, so it is solved anew.
    assert calls[2][4] is None
    assert calls[2][5] == "paper"
    assert (tmp_path / "all_summary.json").is_file()


def test_smoke_preset_keeps_the_tiny_validation_model() -> None:
    arguments = build_parser().parse_args(["figure3", "--preset", "smoke"])
    assert arguments.preset == "smoke"
    assert arguments.realization is None


def test_paper_distances_follow_the_published_markers() -> None:
    assert cli.PAPER_DISTANCES == [5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 35, 40]


@pytest.mark.parametrize("arguments", [
    ["all", "--grid-size", "0"],
    ["figure3", "--grid-size", "1"],
    ["all", "--workers", "0"],
    ["all", "--solver-threads", "0"],
    ["all", "--rates", "nan"],
    ["figure2", "--rates", "-1"],
    ["figure4", "--distances", "inf"],
    ["all", "--tolerance", "nan"],
    ["all", "--max-iterations", "0"],
])
def test_invalid_cli_inputs_fail_before_running_experiments(arguments):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)
    assert error.value.code == 2
