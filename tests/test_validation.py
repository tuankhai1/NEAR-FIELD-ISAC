"""Regression checks for incomplete, stale and corrupt experiment artifacts."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from near_field_isac.config import SimulationConfig
from near_field_isac.fim import far_field_angle_crb

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import validate_results as figures  # noqa: E402


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_validation_is_active_with_python_optimization():
    code = f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); "
    code += "from validate_results import require; require(False, 'audit failed')"
    result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "ValueError: audit failed" in result.stderr


@pytest.fixture
def waveform():
    config = SimulationConfig(n_antennas=9, n_users=1, n_rf_chains=2, min_rate=0)
    data = {
        "covariance": np.eye(9, dtype=complex),
        "beamformers": np.zeros((9, 1), complex),
        "rates": np.zeros(1),
        "sensing_covariance": np.eye(9),
    }
    scenario = {"communication_channels": np.ones((9, 1), complex)}
    return config, data, scenario


def test_waveform_audit_accepts_feasible_arrays(waveform):
    config, data, scenario = waveform
    assert figures.check_waveform(config, data, scenario, rate=0)["rate_margin"] == 0


@pytest.mark.parametrize(
    "damage, message",
    [
        ("upper_triangle", "Hermitian"),
        ("nan", "Nonfinite"),
        ("rates", "Saved rates"),
        ("residual", "sensing covariance"),
        ("power", "power constraint"),
    ],
)
def test_waveform_audit_rejects_corrupt_arrays(waveform, damage, message):
    config, data, scenario = waveform
    if damage == "upper_triangle":
        data["covariance"][0, 1] = 1j  # eigvalsh alone silently ignores this corruption.
    elif damage == "nan":
        data["beamformers"][0, 0] = np.nan
    elif damage == "rates":
        data["rates"] += 1
    elif damage == "residual":
        data["sensing_covariance"] *= 2
    else:
        data["covariance"] *= config.transmit_power
    with pytest.raises(ValueError, match=message):
        figures.check_waveform(config, data, scenario, rate=0)


def test_far_reference_finite_difference_matches_analytic_crb(waveform):
    config, data, _ = waveform
    actual = figures.finite_difference_crb(
        config, data["covariance"], 0.001j, config.target_range, None, model="far"
    )
    expected = np.rad2deg(np.sqrt(far_field_angle_crb(config, data["covariance"], 0.001j)))
    assert np.isclose(actual, expected, rtol=1e-7, atol=0)


def test_finite_difference_rejects_unidentifiable_information(waveform):
    config, data, _ = waveform
    with pytest.raises(ValueError, match="unidentifiable"):
        figures.finite_difference_crb(config, data["covariance"], 0j, 20, None)


@pytest.fixture
def curve():
    rows = [
        {"minimum_rate": rate, "architecture": architecture, "range_rcrb_m": 0.1}
        for rate in (0.0, 5.0)
        for architecture in ("fully-digital-sdr", "hybrid-two-stage-sdr")
    ]
    summary = {"rates": [0.0, 5.0], "rows": rows}
    return [{key: str(value) for key, value in row.items()} for row in rows], summary


def test_curve_audit_accepts_complete_sweep(curve):
    figures.check_curve_coverage(*curve, "minimum_rate")


@pytest.mark.parametrize("damage", ["missing", "duplicate", "summary", "csv"])
def test_curve_audit_rejects_missing_or_inconsistent_points(curve, damage):
    rows, summary = curve
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[-1] = rows[0]
    elif damage == "summary":
        summary["rows"].pop()
    else:
        rows[0]["range_rcrb_m"] = "99"
    with pytest.raises(ValueError, match="cover|mismatch"):
        figures.check_curve_coverage(rows, summary, "minimum_rate")


def test_source_audit_rejects_empty_manifest():
    with pytest.raises(ValueError, match="Incomplete"):
        figures.check_source_hashes({"source_sha256": {}})


def test_figure3_audit_rejects_summary_peak_that_disagrees_with_spectrum(tmp_path, waveform):
    config, data, scenario = waveform
    source_dir = SCRIPTS.parent / "src/near_field_isac"
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_dir.glob("*.py")
    }
    write_json(tmp_path / "provenance.json", {"config": asdict(config), "source_sha256": hashes})
    scenario["target_gain"] = 0.001j
    np.savez(tmp_path / "scenario.npz", **scenario)
    x, y = np.meshgrid([1.0, 2.0], [1.0, 2.0])
    near = np.array([[1.0, 0.5], [0.5, 0.5]])
    bound = figures.finite_difference_crb(config, data["covariance"], 0.001j, 20, None)
    np.savez(tmp_path / "nominal.npz", **data)
    np.savez(
        tmp_path / "figure3_music_data.npz",
        **data,
        x_grid=x,
        y_grid=y,
        near_spectrum=near,
        far_spectrum=np.ones((2, 2)),
    )
    summary = {
        "preset": {"grid_size": 2},
        "range_rcrb_m": bound[0],
        "angle_rcrb_deg": bound[1],
        "communication_rates_bit_s_hz": [0.0],
        "near_field_estimate": {"range_m": np.sqrt(2), "angle_deg": 45.0},
        "far_field_grid_maximum": {"range_m": np.sqrt(2), "angle_deg": 45.0},
    }
    write_json(tmp_path / "figure3_summary.json", summary)
    figures.check_figure3(tmp_path)
    summary["near_field_estimate"]["range_m"] = 20
    write_json(tmp_path / "figure3_summary.json", summary)
    with pytest.raises(ValueError, match="does not match saved spectrum"):
        figures.check_figure3(tmp_path)
