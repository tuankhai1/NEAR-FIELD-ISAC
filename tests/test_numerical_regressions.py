"""Independent physical checks for the numerical audit's failure modes."""

from dataclasses import replace

import numpy as np
import pytest

from near_field_isac.channels import (
    generate_scenario,
    near_field_response,
    target_response_matrices,
)
from near_field_isac.config import SimulationConfig
from near_field_isac.fim import (
    FisherBlocks,
    crb_from_blocks,
    crb_matrix,
    far_field_angle_crb,
    fisher_information_blocks,
    root_crb,
)
from near_field_isac.music import music_spectrum_xy
from near_field_isac.optimization import (
    exact_transmit_basis,
    hybrid_analog_beamformer,
    random_hybrid_combiner,
    solve_fully_digital_sdr,
    solve_hybrid_sdr,
)


@pytest.mark.parametrize("zero_gain", [False, True])
def test_no_information_cannot_report_perfect_sensing(zero_gain):
    config = SimulationConfig.smoke()
    scenario = generate_scenario(config)
    covariance = np.eye(config.n_antennas) if zero_gain else np.zeros((17, 17))
    with pytest.raises(ValueError, match="unidentifiable"):
        crb_matrix(config, covariance, 0j if zero_gain else scenario.target_gain)


def test_invalid_variances_and_indefinite_information_are_not_clipped():
    with pytest.raises(ValueError):
        root_crb(np.diag([-1.0, 1.0]))
    with pytest.raises(ValueError):
        crb_from_blocks(
            FisherBlocks(np.array([[1.0, 2.0], [2.0, 1.0]]), np.zeros((2, 2)), np.eye(2))
        )


def test_balancing_preserves_identifiable_parameters_with_different_units():
    blocks = FisherBlocks(np.diag([1e-8, 1e12]), np.zeros((2, 2)), np.eye(2))
    assert np.allclose(crb_from_blocks(blocks), np.diag([1e8, 1e-12]))


def test_fim_matches_independent_vectorized_echo_jacobian():
    c = SimulationConfig.smoke()
    s = generate_scenario(c)
    rng = np.random.default_rng(731)
    x = rng.normal(size=(c.n_antennas, 128)) + 1j * rng.normal(size=(c.n_antennas, 128))
    covariance = x @ x.conj().T / 128
    g, dr, dt = target_response_matrices(c, s.target_range, s.target_angle)
    for w in [None, random_hybrid_combiner(c, rng)]:
        derivatives = [s.target_gain * dr, s.target_gain * dt, g, 1j * g]
        if w is not None:
            derivatives = [w @ d for d in derivatives]
        jac = np.column_stack([(d @ x).ravel() for d in derivatives])
        noise = c.noise_power * (c.n_antennas if w is not None else 1)
        direct = 2 / noise * np.real(jac.conj().T @ jac)
        blocks = fisher_information_blocks(c, covariance, s.target_gain, receive_combiner=w)
        result = np.block([[blocks.j11, blocks.j12], [blocks.j12.T, blocks.j22]])
        assert np.allclose(result, direct, rtol=1e-10, atol=1e-5)
        assert np.allclose(crb_from_blocks(blocks), np.linalg.inv(direct)[:2, :2], rtol=1e-8)


def test_exact_subspace_preserves_all_waveform_functionals():
    c = SimulationConfig.smoke()
    s = generate_scenario(c)
    q = exact_transmit_basis(c, s)
    rng = np.random.default_rng(813)
    x = rng.normal(size=(17, 17)) + 1j * rng.normal(size=(17, 17))
    original = x @ x.conj().T
    projected = q @ (q.conj().T @ original @ q) @ q.conj().T
    assert np.trace(projected).real <= np.trace(original).real + 1e-10
    assert np.allclose(
        crb_matrix(c, original, s.target_gain), crb_matrix(c, projected, s.target_gain), rtol=1e-9
    )
    h = s.communication_channels
    assert np.allclose(h.T @ original @ h.conj(), h.T @ projected @ h.conj())


def test_extra_rf_chains_recover_a_realizable_waveform():
    c = SimulationConfig.smoke(n_rf_chains=4)
    s = generate_scenario(c)
    r = solve_hybrid_sdr(c, s, solver="auto", solver_threads=2)
    rf = hybrid_analog_beamformer(c, s)
    assert np.linalg.matrix_rank(rf) == 3
    assert r.metadata["solver_dimension"] == 3
    assert r.metadata["rf_reconstruction_relative_error"] < 1e-9
    assert np.allclose(
        rf @ r.metadata["baseband_covariance"] @ rf.conj().T,
        r.waveform.covariance,
        rtol=1e-9,
        atol=1e-9,
    )


def test_hybrid_music_whitens_actual_colored_noise():
    c = SimulationConfig.smoke(target_range=np.sqrt(128), target_angle_deg=45)
    rng = np.random.default_rng(34)
    w = random_hybrid_combiner(c, rng)
    a = w @ near_field_response(c, c.target_range, c.target_angle)
    covariance = np.outer(a, a.conj()) + 20 * w @ w.conj().T
    result = music_spectrum_xy(
        c, covariance, np.arange(1.0, 17.0), np.arange(1.0, 17.0), receive_combiner=w
    )
    assert result.estimated_x == result.estimated_y == 8


def test_far_field_reference_does_not_depend_on_nominal_target_range():
    c = SimulationConfig.smoke()
    s = generate_scenario(c)
    first = solve_fully_digital_sdr(c, s, sensing_model="far", solver="auto", solver_threads=2)
    other = solve_fully_digital_sdr(
        c.with_updates(target_range=40),
        replace(s, target_range=40),
        sensing_model="far",
        solver="auto",
        solver_threads=2,
    )
    values = [far_field_angle_crb(c, r.waveform.covariance, s.target_gain) for r in [first, other]]
    assert np.isclose(*values, rtol=1e-8)
    assert first.metadata["physical_crb"].shape == (1, 1)


def test_full_and_reduced_sdr_agree_in_small_model():
    c = SimulationConfig(n_antennas=9, n_users=1, n_rf_chains=2, seed=19, min_rate=5)
    s = generate_scenario(c)
    small = solve_fully_digital_sdr(c, s, solver="auto", solver_threads=2)
    full = solve_fully_digital_sdr(c, s, solver="auto", solver_threads=2, reduce_dimension=False)
    assert np.isclose(small.objective, full.objective, rtol=2e-5)
    for r in [small, full]:
        assert r.metadata["validation_passed"]
        assert r.metadata["minimum_rate_margin"] >= -1e-5
