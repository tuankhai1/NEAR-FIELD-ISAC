"""Consistency checks of the model against the paper's equations.

Run:  python -m pytest -q tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfisac.channel import (Scenario, array_response, array_response_derivatives,  # noqa: E402
                            array_response_far, build_scenario, element_distances, random_scenario)
from nfisac.crb import far_field_model, near_field_model  # noqa: E402
from nfisac.metrics import achievable_rates  # noqa: E402
from nfisac.music import music_spectrum, simulate_echo  # noqa: E402
from nfisac.optimization import (analog_beamformer, design_fully_digital, design_hybrid,  # noqa: E402
                                 random_analog_combiner, solve_crb_sdr)
from nfisac.params import SystemParams  # noqa: E402

P = SystemParams()
SMALL = SystemParams(N=17, N_RF=3, K=2)


def test_parameters_match_paper():
    assert P.lam == pytest.approx(1.0714e-2, rel=1e-3)          # "lambda = 1.07 cm"
    assert P.rayleigh_distance == pytest.approx(46.67, abs=0.1)  # paper: 46.73 m with lambda rounded
    assert P.d == pytest.approx(P.D / 64)
    assert P.P_max == pytest.approx(100.0) and P.noise == pytest.approx(1e-6)


def test_distance_formula_eq1():
    r, th = 7.3, 0.8
    pos = np.stack([P.antenna_positions, np.zeros(P.N)], 1)
    pt = np.array([r * np.cos(th), r * np.sin(th)])
    assert np.allclose(element_distances(P, r, th), np.linalg.norm(pt - pos, axis=1))


def test_far_field_limit_eq5():
    # for r >> D the near-field response tends to the far-field one (Eq. (5))
    th = 1.1
    err = np.abs(np.angle(array_response(P, 1e5, th) / array_response_far(P, th)))
    assert err.max() < 1e-2


def test_response_derivatives():
    r, th, h = 12.0, 0.9, 1e-6
    a, a_r, a_t = array_response_derivatives(P, r, th)
    num_r = (array_response(P, r + h, th) - array_response(P, r - h, th)) / (2 * h)
    num_t = (array_response(P, r, th + h) - array_response(P, r, th - h)) / (2 * h)
    assert np.allclose(a_r, num_r, rtol=1e-5, atol=1e-6)
    assert np.allclose(a_t, num_t, rtol=1e-5, atol=1e-4)


def test_fim_matches_numerical_fisher():
    """Eqs. (25)-(29) versus J = 2/sigma^2 Re{(d mu/d xi)^H (d mu/d xi)} computed numerically."""
    p = SMALL
    rng = np.random.default_rng(0)
    T = p.T
    X = (rng.standard_normal((p.N, T)) + 1j * rng.standard_normal((p.N, T))) / np.sqrt(2)
    Rx = X @ X.conj().T / T
    beta = 0.3 - 0.7j

    def mu(xi):
        r, th, br, bi = xi
        a = array_response(p, r, th)
        return ((br + 1j * bi) * np.outer(a, a) @ X).ravel()

    xi0 = np.array([p.r_s, p.theta_s, beta.real, beta.imag])
    steps = np.array([1e-6, 1e-7, 1e-6, 1e-6])
    D = np.stack([(mu(xi0 + np.eye(4)[i] * steps[i]) - mu(xi0 - np.eye(4)[i] * steps[i])) / (2 * steps[i])
                  for i in range(4)], 1)
    J_num = 2.0 / p.noise * np.real(D.conj().T @ D)
    J_model = near_field_model(p, beta).fim(Rx)
    assert np.allclose(J_model, J_num, rtol=1e-3, atol=1e-6 * np.abs(J_num).max())   # finite-difference error


def _small_scenario(seed=3):
    return random_scenario(SMALL, np.random.default_rng(seed))


def test_subspace_reduction_is_exact():
    scn = _small_scenario()
    fim = near_field_model(SMALL, scn.beta_s)
    full = solve_crb_sdr(scn.H, fim, SMALL.P_max, SMALL.noise, 3.0, reduce=False)
    red = solve_crb_sdr(scn.H, fim, SMALL.P_max, SMALL.noise, 3.0, reduce=True)
    assert red.info["subspace_dim"] < SMALL.N
    assert np.trace(fim.crb(red.R)) == pytest.approx(np.trace(fim.crb(full.R)), rel=1e-4)


@pytest.mark.parametrize("R_min", [0.0, 3.0])
def test_paper_formulation_gives_same_optimum(R_min):
    """The literal problem (20) (full N x N, authors' scaling, R_BB for the hybrid) and the
    reformulated one reach the same tr(CRB)."""
    rng = np.random.default_rng(3)
    scn = random_scenario(SMALL, rng)
    W = random_analog_combiner(SMALL, rng)
    for design in (lambda f: design_fully_digital(scn, R_min, formulation=f),
                   lambda f: design_hybrid(scn, R_min, W, formulation=f)):
        fast, paper = design("fast"), design("paper")
        assert np.trace(paper.crb) == pytest.approx(np.trace(fast.crb), rel=1e-4)
        assert paper.rates.min() >= R_min - 1e-3


def test_paper_backends_agree():
    """MOSEK Fusion (real 2n x 2n embedding) and the plain CVXPY transcription of (20) agree."""
    from nfisac.optimization import _paper_cvxpy, _paper_fusion
    scn = _small_scenario()
    fim = near_field_model(SMALL, scn.beta_s)
    G = np.conj(scn.H) / np.sqrt(SMALL.noise)
    args = (fim, 1e2 * SMALL.noise / SMALL.T, G, 1.0 + 1.0 / (2.0 ** 3 - 1.0), np.eye(SMALL.N), SMALL.P_max,
            None, "MOSEK", False, {})
    _, R_cvx, _, obj_cvx = _paper_cvxpy(*args)
    _, R_fus, _, obj_fus = _paper_fusion(*args)
    assert obj_fus == pytest.approx(obj_cvx, rel=1e-5)
    assert np.trace(fim.crb(R_fus)) == pytest.approx(np.trace(fim.crb(R_cvx)), rel=1e-5)


@pytest.mark.parametrize("R_min", [0.0, 2.0, 6.0])
def test_fully_digital_solution_is_feasible(R_min):
    scn = random_scenario(P, np.random.default_rng(7))
    d = design_fully_digital(scn, R_min)
    assert np.real(np.trace(d.Rx)) <= P.P_max * (1 + 1e-6)                    # (16c)
    assert d.rates.min() >= R_min - 1e-3                                      # (16b)
    Rs = d.Rx - d.P @ d.P.conj().T                                            # (16d) with rank-one p_k*
    assert np.linalg.eigvalsh(Rs).min() >= -1e-6 * P.P_max
    assert np.allclose(achievable_rates(scn.H, d.P, d.Rx, P.noise), d.rates)


def test_crb_increases_with_rate_requirement():
    scn = random_scenario(P, np.random.default_rng(7))
    tr = [np.trace(design_fully_digital(scn, R).crb) for R in (0.0, 4.0, 8.0)]
    assert tr[0] <= tr[1] * (1 + 1e-5) and tr[1] <= tr[2] * (1 + 1e-5)


def test_hybrid_structure():
    rng = np.random.default_rng(11)
    scn = random_scenario(P, rng)
    W = random_analog_combiner(P, rng)
    d = design_hybrid(scn, 4.0, W)
    P_RF = d.extras["P_RF"]
    assert P_RF.shape == (P.N, P.N_RF)
    assert np.allclose(np.abs(P_RF), 1.0) and np.allclose(np.abs(W), 1.0)     # unit modulus
    assert np.allclose(P_RF @ d.extras["R_BB"] @ P_RF.conj().T, d.Rx, atol=1e-8 * P.P_max)
    assert np.allclose(P_RF @ d.extras["P_BB"], d.P, atol=1e-8)
    assert np.real(np.trace(d.Rx)) <= P.P_max * (1 + 1e-6)
    assert d.rates.min() >= 4.0 - 1e-3
    assert np.allclose(analog_beamformer(scn)[:, -1], np.conj(array_response(P, P.r_s, P.theta_s)))


def test_music_locates_target():
    rng = np.random.default_rng(5)
    scn = random_scenario(P, rng)
    d = design_fully_digital(scn, 5.0)
    Y = simulate_echo(scn, d.Rx, d.P, rng)
    x = np.arange(12.0, 16.01, 0.04)
    res = music_spectrum(scn, Y, x, x)
    # MUSIC only uses the sample covariance (std of r_hat ~ 0.25 m here) and is not expected to
    # reach the CRB, which assumes a known transmit waveform; the paper reports r_hat = 19.952 m.
    assert res.r_hat == pytest.approx(P.r_s, abs=0.6)
    assert np.rad2deg(res.theta_hat) == pytest.approx(P.theta_s_deg, abs=0.3)


def test_near_field_angle_crb_tends_to_far_field():
    """Fig. 4: with beta_s fixed, the near-field angle CRB (r_s unknown) decreases to the far-field
    angle-only CRB as r_s grows (the r-theta coupling vanishes)."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((P.N, 8)) + 1j * rng.standard_normal((P.N, 8))
    R = X @ X.conj().T
    beta = 1e-3
    crb_th = [near_field_model(P.with_(r_s=r), beta).crb(R)[1, 1] for r in (5.0, 20.0, 1e4)]
    crb_ff = far_field_model(P, beta).crb(R)[0, 0]
    assert crb_th[0] > crb_th[1] > crb_th[2]
    assert crb_th[2] == pytest.approx(crb_ff, rel=1e-3)


def test_hybrid_far_field_benchmark():
    """Fig. 4 benchmark: near-field users, far-field target; analog beams (22) follow each model."""
    rng = np.random.default_rng(3)
    scn = random_scenario(SMALL, rng)
    W = random_analog_combiner(SMALL, rng)
    ff = Scenario(SMALL, scn.r_users, scn.theta_users, scn.H, scn.beta_s, far_field=True, far_field_users=False)
    P_RF = analog_beamformer(ff)
    assert np.allclose(P_RF[:, 0], np.conj(array_response(SMALL, scn.r_users[0], scn.theta_users[0])))
    assert np.allclose(P_RF[:, -1], np.conj(array_response_far(SMALL, SMALL.theta_s)))
    fast, paper = (design_hybrid(ff, 3.0, W, formulation=f) for f in ("fast", "paper"))
    assert paper.crb.shape == (1, 1)
    assert paper.crb[0, 0] == pytest.approx(fast.crb[0, 0], rel=1e-4)
    assert paper.rates.min() >= 3.0 - 1e-3


def test_matched_analog_combiner():
    """Fig. 4 variant: unit modulus, conjugate symmetry about the centre after removing the target phase,
    (1/N) W W^H ~ I, and a feasible hybrid design with the same CRB for both formulations."""
    from nfisac.optimization import conj_symmetric_phases, matched_analog_combiner
    rng = np.random.default_rng(4)
    U = conj_symmetric_phases(P, rng)
    a = array_response(P, P.r_s, P.theta_s)
    W = matched_analog_combiner(a, U)
    assert W.shape == (P.N_RF, P.N) and np.allclose(np.abs(W), 1.0)
    V = np.conj(W) / a[None, :]                                  # = U
    assert np.allclose(V[:, ::-1], np.conj(V)) and np.allclose(V[:, P.N_tilde], 1.0)
    assert np.abs(W @ W.conj().T / P.N - np.eye(P.N_RF)).max() < 0.5
    scn = random_scenario(SMALL, np.random.default_rng(3))
    Ws = matched_analog_combiner(scn.a_s, conj_symmetric_phases(SMALL, rng))
    fast, paper = (design_hybrid(scn, 3.0, Ws, formulation=f) for f in ("fast", "paper"))
    assert np.trace(paper.crb) == pytest.approx(np.trace(fast.crb), rel=1e-4) and paper.rates.min() >= 3.0 - 1e-3


def test_fixed_alpha_option():
    """--alpha: 'paper' -> PAPER_ALPHA, nothing drawn (same realisations as alpha = 1), own results folder."""
    from nfisac.channel import PAPER_ALPHA, alpha_value, draw_reflection
    from scripts.run_fig2 import ROOT, build_parser, output_and_reflection
    assert alpha_value("paper") == PAPER_ALPHA and alpha_value("2.5") == 2.5
    r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
    assert draw_reflection(r1, alpha_value("paper")) == PAPER_ALPHA and draw_reflection(r2, "unit") == 1.0
    assert r1.random() == r2.random()                                    # no random number consumed
    out, refl = output_and_reflection(build_parser().parse_args(["--alpha", "paper", "--out", str(ROOT / "results")]))
    assert refl == PAPER_ALPHA and out == ROOT / "results"               # explicit --out wins
