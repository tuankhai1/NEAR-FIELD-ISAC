import near_field_isac.cli as cli
from near_field_isac.cli import build_parser


def test_all_command_defaults_to_full_paper_preset() -> None:
    arguments = build_parser().parse_args(["all"])
    assert arguments.experiment == "all"
    assert arguments.preset == "paper"
    assert arguments.grid_size is None
    assert arguments.rates is None
    assert arguments.distances is None


def test_all_command_orchestrates_all_three_figures(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_figure2(config, rates, **kwargs):
        calls.append(("figure2", config.n_antennas, list(rates), kwargs["output_dir"]))
        return {"experiment": "figure2"}

    def fake_figure3(config, **kwargs):
        calls.append(("figure3", config.n_antennas, kwargs["grid_size"], kwargs["output_dir"]))
        return {"experiment": "figure3"}

    def fake_figure4(config, distances, **kwargs):
        calls.append(("figure4", config.n_antennas, list(distances), kwargs["output_dir"]))
        return {"experiment": "figure4"}

    monkeypatch.setattr(cli, "reproduce_figure2", fake_figure2)
    monkeypatch.setattr(cli, "reproduce_figure3", fake_figure3)
    monkeypatch.setattr(cli, "reproduce_figure4", fake_figure4)
    cli.main(["all", "--preset", "quick", "--output", str(tmp_path)])

    assert [call[0] for call in calls] == ["figure2", "figure3", "figure4"]
    assert all(call[1] == 17 for call in calls)
    assert calls[1][2] == 121
    assert (tmp_path / "all_summary.json").is_file()
