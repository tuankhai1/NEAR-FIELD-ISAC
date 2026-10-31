"""CRB minimisation under per-user rate constraints, Sec. III of the paper.

Fully digital (Sec. III-B):  problem (16) -> (17) via the auxiliary matrix U and
the Schur complement -> SDR problem (20) with P_k = p_k p_k^H -> rank-one
reconstruction (21), which is globally optimal for (16).

Hybrid (Sec. III-C): two-stage heuristic.  Stage 1 fixes the analog
beamformer by Eq. (22); stage 2 solves the same SDR in the baseband with the
effective channels P_RF^T h_k, the power constraint tr(P_RF R_BB P_RF^H) <= P_max
and the hybrid CRB(R_HB, W_RF G, N sigma_s^2).

``solve_crb_sdr`` is written once for an arbitrary transmit dimension n
(n = N for fully digital, n = N_RF for the digital stage of the hybrid array).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from .channel import Scenario, array_response, array_response_far
from .crb import EchoFisherModel, far_field_model, near_field_model, target_model
from .metrics import achievable_rates


class InfeasibleError(RuntimeError):
    pass


@dataclass
class SDRSolution:
    status: str
    R: np.ndarray            # optimal transmit covariance (true units, mW)
    F: list                  # relaxed P_k (true units)
    p: np.ndarray            # (n, K) rank-one beamformers, Eq. (21)
    solve_time: float
    info: dict = field(default_factory=dict)


def _herm(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + X.conj().T)


def solve_crb_sdr(H_eff: np.ndarray, fim: EchoFisherModel, P_max: float, noise_c: float,
                  R_min: float, solver: str = "MOSEK", crb_weights=None, verbose: bool = False,
                  reduce: bool = True, solver_opts: dict | None = None) -> SDRSolution:
    """Solve problem (20) and reconstruct the rank-one solution (21).

        min_{P_k >= 0, R >= 0, U >= 0}  tr(U^{-1})
        s.t.  [[J11(R) - U, J12(R)], [J12(R)^T, J22(R)]] >= 0          (17b)
              (2^Rmin - 1)^{-1} h^T P_k h^* + h^T P_k h^* >= h^T R h^* + sigma^2   (18)
              tr(R) <= P_max                                            (16c)
              R - sum_k P_k >= 0                                        (19)

    Constraint (18) is implemented in the multiplied-out form used by the
    official SDR_fully_digital.m,  h^T P_k h^* >= (2^Rmin - 1)(h^T R h^* - h^T P_k h^* + sigma^2),
    which is identical to (18) for R_min > 0 and stays valid for R_min = 0.

    Numerical scaling only (the optimum is unchanged): R = P_max * R_n,
    channels are normalised by sqrt(P_max) / sigma_k and each rate constraint is
    divided by 2^Rmin (both sides are then O(1) at the optimum), and the LMI
    is congruence-scaled by S = diag(s_1..s_P, s_b, s_b) with s_i^{-2} the
    corresponding FIM diagonal at the isotropic covariance, U~ = D U D / P_max.
    Then tr(U^{-1}) = tr(D U~^{-1} D) / P_max = matrix_frac(D, U~) / P_max.

    ``crb_weights``: None (default) = the paper's objective tr(CRB).  An array w
    minimises sum_i w_i [CRB]_ii; "normalized" uses w_i = 1 / [CRB_iso]_ii, the CRB
    of isotropic transmission with full power (unit-free objective; suggestion in
    the report, not part of the paper).

    ``reduce`` (exact): every constraint and every FIM entry depends on R and P_k
    only through x^H R y with x, y in
        V = span{h_k^*, rows of G~, dG~/dr, dG~/dtheta}^*   (dim <= K + 3).
    Replacing (R, P_k) by their compressions B B^H R B B^H onto V keeps all
    these values, keeps every PSD constraint and does not increase tr(R), so
    the optimal value is unchanged.  Solving in R = B R~ B^H (7 x 7 instead of
    65 x 65 for the paper's setting) turns an ~90 s SDP into a ~0.1 s one.
    """
    n, K = H_eff.shape
    if isinstance(crb_weights, str):
        if crb_weights != "normalized":
            raise ValueError(f"unknown crb_weights {crb_weights!r}")
        crb_weights = 1.0 / np.diag(fim.crb(np.eye(n) * P_max / n))
    if reduce:
        B = _orthonormal_basis(np.hstack([H_eff.conj(), fim.transmit_directions()]))
        if B.shape[1] < n:
            sol = solve_crb_sdr(B.T @ H_eff, fim.restricted(B), P_max, noise_c, R_min, solver,
                                crb_weights, verbose, reduce=False, solver_opts=solver_opts)
            lift = lambda X: B @ X @ B.conj().T
            sol.R, sol.F, sol.p = lift(sol.R), [lift(X) for X in sol.F], B @ sol.p
            sol.info["subspace_dim"] = B.shape[1]
            return sol
    P = fim.n_par

    # --- scaling factors from the isotropic covariance with unit power ----------
    J11_0, _, J22_0 = fim.blocks(np.eye(n) / n)
    s = 1.0 / np.sqrt(np.diag(J11_0))
    sb = 1.0 / np.sqrt(J22_0[0, 0])
    w = np.ones(P) if crb_weights is None else np.asarray(crb_weights, float)
    Dn = np.diag(np.sqrt(w) * s / np.max(np.sqrt(w) * s))

    # --- variables ------------------------------------------------------------
    Rn = cp.Variable((n, n), hermitian=True)
    F = [cp.Variable((n, n), hermitian=True) for _ in range(K)]
    Ut = cp.Variable((P, P), symmetric=True)
    M = cp.Variable((P + 2, P + 2), symmetric=True)

    cons = [Rn >> 0, Rn - sum(F) >> 0, M >> 0]                    # R >= 0, (19), (17b)
    cons += [Fk >> 0 for Fk in F]
    cons += [cp.real(cp.trace(Rn)) <= 1.0]                         # (16c)

    # rate constraints (18)
    tau = 2.0 ** R_min - 1.0
    for k in range(K):
        g = np.conj(H_eff[:, k]) * np.sqrt(P_max / noise_c)     # SNR-normalised channel
        q = lambda X, g=g: cp.real(g.conj() @ X @ g)
        cons += [q(F[k]) >= tau / (1.0 + tau) * (q(Rn) + 1.0)]

    # CRB LMI (17b), congruence-scaled
    J11, J12, J22 = fim.blocks(Rn, cvx=True)
    for i in range(P):
        for j in range(i, P):
            cons += [M[i, j] == s[i] * s[j] * J11[i][j] - Ut[i, j]]
        for b in range(2):
            cons += [M[i, P + b] == s[i] * sb * J12[i][b]]
    cons += [M[P, P] == sb ** 2 * J22[0][0], M[P + 1, P + 1] == sb ** 2 * J22[1][1], M[P, P + 1] == 0]

    prob = cp.Problem(cp.Minimize(cp.matrix_frac(Dn, Ut)), cons)   # (20a): tr(U^{-1}) up to a constant
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver, verbose=verbose, **(solver_opts or {}))
    except cp.error.SolverError as exc:
        raise InfeasibleError(f"solver error: {exc}") from exc
    t1 = time.perf_counter() - t0
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise InfeasibleError(f"problem status: {prob.status}")

    R = P_max * _herm(Rn.value)
    Fv = [P_max * _herm(Fk.value) for Fk in F]
    p = _rank_one(H_eff, Fv, P_max)
    info = {"objective": prob.value, "solver": solver, "rank_Pk": [_numerical_rank(X) for X in Fv]}
    return SDRSolution(prob.status, R, Fv, p, t1, info)


PAPER_SCALES = (1e2, 1e3, 1e1, 1e4, 1e0, 1e5, 1e-1, 1e6, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)   # 1e2 = main.m first
POWER_TOL = 1e-7      # relative, full-power check of the paper formulation
RATE_TOL = 1e-3       # bit/s/Hz, same tolerance as the Fig. 2 driver


def solve_crb_sdr_paper(H_eff: np.ndarray, fim: EchoFisherModel, P_max: float, noise_c: float,
                        R_min: float, solver: str = "MOSEK", Q_power: np.ndarray | None = None,
                        scales=PAPER_SCALES, crb_weights=None,
                        verbose: bool = False, solver_opts: dict | None = None) -> SDRSolution:
    """Problem (20) written literally as in the paper and SDR_fully_digital.m.

        min_{P_k >= 0, R >= 0, U >= 0}  tr(U^{-1})                                  (20a)
        s.t.  [[J11(R) - U, J12(R)], [J12(R)^T, J22(R)]] >= 0                       (17b)
              gamma_k h_k^T P_k h_k^* >= h_k^T R h_k^* + sigma_k^2,
              gamma_k = 1 + 1 / (2^Rmin - 1)                                        (18)
              tr(Q R) <= P_max                                                      (16c)
              R - sum_k P_k >= 0                                                    (19)

    None of the reformulations of ``solve_crb_sdr``: full n x n variables (no subspace
    reduction), the LMI and tr(U^{-1}) as written, constraint (18) with gamma_k.
    Q = I for the fully digital array; Q = P_RF^H P_RF for the digital stage of the
    hybrid array (R = R_BB, power tr(P_RF R_BB P_RF^H) <= P_max).
    The only conditioning is the authors' own (generate_channel.m, FIM.m, main.m):
    channels divided by sigma_k and FIM multiplied by scale * sigma_s^2 / T, scale = 1e2.
    main.m adds "if the optimization fails, try to adjust this scale factor"; this is done
    automatically by trying ``scales`` in order (the optimum does not depend on scale).  A run
    counts as failed when the solver gives no solution, or when the solution misses one of two
    necessary conditions of the optimum: full power tr(Q R) = P_max (any feasible point scaled up
    to full power stays feasible and divides the CRB) and rates >= R_min of the rank-one
    beamformers (21).  Both catch MOSEK stopping early when the scaled objective is << 1.
    For R_min = 0, gamma_k = inf and (18) reduces to h_k^T P_k h_k^* >= 0, implied by P_k >= 0.
    ``crb_weights`` as in ``solve_crb_sdr`` (None = tr(CRB) of the paper).
    """
    n, K = H_eff.shape
    Q = np.eye(n) if Q_power is None else Q_power
    if isinstance(crb_weights, str):
        if crb_weights != "normalized":
            raise ValueError(f"unknown crb_weights {crb_weights!r}")
        # isotropic transmission with full power over the available directions (range(P_RF) for the hybrid)
        crb_weights = 1.0 / np.diag(fim.crb(np.linalg.inv(Q) * P_max / n))
    w = None if crb_weights is None else np.asarray(crb_weights, float)
    gamma = 1.0 + 1.0 / (2.0 ** R_min - 1.0) if R_min > 0 else None
    G = np.conj(H_eff) / np.sqrt(noise_c)                         # h_k / sigma_k as in generate_channel.m
    backend = _paper_fusion if solver.upper() == "MOSEK" else _paper_cvxpy
    t0 = time.perf_counter()
    errors = []
    for scale in scales:
        f = scale * fim.noise / fim.T                             # FIM.m: scale * 2|beta|^2 Re tr(...)
        status, R, Fv, obj = backend(fim, f, G, gamma, Q, P_max, w, solver, verbose, solver_opts or {})
        if R is None:
            errors.append(f"scale={scale:g}: {status}")
            continue
        p = _rank_one(H_eff, Fv, P_max)
        power = np.real(np.trace(Q @ R))
        rate_gap = achievable_rates(H_eff, p, R, noise_c).min() - R_min
        if power >= (1.0 - POWER_TOL) * P_max and rate_gap >= -RATE_TOL:
            break
        errors.append(f"scale={scale:g}: inaccurate (power {power / P_max:.6f} P_max, rate - R_min {rate_gap:+.1e})")
    else:
        raise InfeasibleError("; ".join(errors))
    t1 = time.perf_counter() - t0

    info = {"objective": obj, "solver": solver, "scale": scale, "failed_scales": errors,
            "rank_Pk": [_numerical_rank(X) for X in Fv]}
    return SDRSolution(status, R, Fv, p, t1, info)


def _paper_cvxpy(fim, f, G, gamma, Q, P_max, w, solver, verbose, solver_opts):
    """Problem (20) transcribed in CVXPY.  Readable reference; too slow for n = 65
    (CVXPY dualises it into ~5 n^2 rows: > 30 min and ~6 GB per solve)."""
    n, K = G.shape
    P = fim.n_par
    R = cp.Variable((n, n), hermitian=True)
    F = [cp.Variable((n, n), hermitian=True) for _ in range(K)]
    U = cp.Variable((P, P), symmetric=True)

    J11, J12, J22 = fim.blocks(R, cvx=True)
    cell = lambda x: cp.reshape(x if isinstance(x, cp.Expression) else cp.Constant(x), (1, 1), order="F")
    rows = [[f * J11[i][j] - U[i, j] for j in range(P)] + [f * J12[i][b] for b in range(2)] for i in range(P)]
    rows += [[f * J12[i][b] for i in range(P)] + [f * J22[b][c] for c in range(2)] for b in range(2)]
    M = cp.bmat([[cell(x) for x in row] for row in rows])

    cons = [M >> 0, U >> 0, R >> 0, R - sum(F) >> 0]               # (17b), (19)
    cons += [Fk >> 0 for Fk in F]
    cons += [cp.real(cp.trace(Q @ R)) <= P_max]                   # (16c)
    if gamma is not None:
        for k in range(K):
            q = lambda X, g=G[:, k]: cp.real(g.conj() @ X @ g)
            cons += [gamma * q(F[k]) >= q(R) + 1.0]              # (18)

    obj = cp.tr_inv(U) if w is None else cp.matrix_frac(np.diag(np.sqrt(w)), U)   # (20a) / sum_i w_i [U^-1]_ii
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=solver, verbose=verbose, **solver_opts)
    except cp.error.SolverError:
        return "solver error", None, None, None
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return prob.status, None, None, None
    return prob.status, _herm(R.value), [_herm(Fk.value) for Fk in F], prob.value


def _real_embed(D: np.ndarray) -> np.ndarray:
    """W with <W, Z> = Re sum(D * X) for X = (Z11 + Z22)/2 + j (Z21 - Z12)/2."""
    re, im = 0.5 * D.real, 0.5 * D.imag
    return np.block([[re, im], [-im, re]])


def _paper_fusion(fim, f, G, gamma, Q, P_max, w, solver, verbose, solver_opts):
    """Same problem (20) passed to MOSEK Fusion directly (what CVX does in SDR_fully_digital.m).

    Each Hermitian n x n unknown X is a real PSD 2n x 2n matrix Z with
    X = (Z11 + Z22)/2 + j (Z21 - Z12)/2: Z >= 0 gives X >= 0, and every X >= 0 is reached
    by Z = [[Re X, -Im X], [Im X, Re X]], so the feasible set is unchanged.  The unknowns
    are P_k and R_s = R - sum_k P_k (Eq. (9)); (19) is then R_s >= 0 and R >= 0 is implied.
    tr(U^{-1}) = min tr(V) s.t. [[V, I], [I, U]] >= 0.  MOSEK then sees only the ~20 scalar
    constraints of (20) instead of ~5 n^2 rows.
    """
    import mosek.fusion as mf

    n, K = G.shape
    P = fim.n_par
    D11, D12, D22 = fim.block_coefficients()
    with mf.Model() as M:
        Z = [M.variable(mf.Domain.inPSDCone(2 * n)) for _ in range(K + 1)]     # Z[0]: R_s, Z[k+1]: P_k
        form = lambda D, idx: mf.Expr.add([mf.Expr.dot(mf.Matrix.dense(_real_embed(D)), Z[i]) for i in idx])
        on_R = range(K + 1)

        Y = M.variable(mf.Domain.inPSDCone(2 * P))                # [[V, I], [I, U]]
        M.constraint(Y.slice([P, 0], [2 * P, P]), mf.Domain.equalsTo(mf.Matrix.eye(P)))
        L = M.variable(mf.Domain.inPSDCone(P + 2))                # LMI (17b)
        for i in range(P):
            for j in range(i, P):
                M.constraint(mf.Expr.sub(mf.Expr.add(L.index(i, j), Y.index(P + i, P + j)),
                                         form(f * D11[(i, j)], on_R)), mf.Domain.equalsTo(0.0))
            for b in range(2):
                M.constraint(mf.Expr.sub(L.index(i, P + b), form(f * D12[i][b], on_R)), mf.Domain.equalsTo(0.0))
        for b in range(2):
            M.constraint(mf.Expr.sub(L.index(P + b, P + b), form(f * D22, on_R)), mf.Domain.equalsTo(0.0))
        M.constraint(L.index(P, P + 1), mf.Domain.equalsTo(0.0))

        M.constraint(form(Q.T, on_R), mf.Domain.lessThan(P_max))                       # (16c)
        if gamma is not None:
            for k in range(K):
                D = np.outer(G[:, k].conj(), G[:, k])
                M.constraint(mf.Expr.sub(form(gamma * D, [k + 1]), form(D, on_R)),
                             mf.Domain.greaterThan(1.0))                                 # (18)

        wt = np.ones(P) if w is None else w
        M.objective(mf.ObjectiveSense.Minimize,
                    mf.Expr.add([mf.Expr.mul(float(wt[i]), Y.index(i, i)) for i in range(P)]))   # (20a)
        threads = solver_opts.get("mosek_params", {}).get("MSK_IPAR_NUM_THREADS")
        if threads:
            M.setSolverParam("numThreads", int(threads))
        if verbose:
            import sys
            M.setLogHandler(sys.stdout)
        try:
            M.solve()
            sol_status = M.getPrimalSolutionStatus()
        except mf.FusionException as exc:
            return f"fusion error: {exc}", None, None, None
        if sol_status != mf.SolutionStatus.Optimal:
            return f"{M.getProblemStatus()}/{sol_status}", None, None, None
        X = []
        for Zi in Z:
            Zv = np.asarray(Zi.level()).reshape(2 * n, 2 * n)
            X.append(_herm(0.5 * (Zv[:n, :n] + Zv[n:, n:]) + 0.5j * (Zv[n:, :n] - Zv[:n, n:])))
        return "optimal", sum(X), X[1:], M.primalObjValue()


def _rank_one(H_eff: np.ndarray, Fv: list, P_max: float) -> np.ndarray:
    """Eq. (21): p_k = (h_k^T P_k h_k^*)^{-1/2} P_k h_k^*  (P_k = 0, only when R_min = 0 -> p_k = 0)."""
    n, K = H_eff.shape
    p = np.zeros((n, K), complex)
    for k in range(K):
        g = np.conj(H_eff[:, k])
        den = np.real(g.conj() @ Fv[k] @ g)
        if den > 0 and np.real(np.trace(Fv[k])) > 1e-9 * P_max:
            p[:, k] = Fv[k] @ g / np.sqrt(den)
    return p


def _orthonormal_basis(V: np.ndarray, rel: float = 1e-10) -> np.ndarray:
    nrm = np.linalg.norm(V, axis=0)
    V = V[:, nrm > 0] / nrm[nrm > 0]                  # drop all-zero columns
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    return U[:, s > rel * s[0]]


def _numerical_rank(X: np.ndarray, rel: float = 1e-6) -> int:
    ev = np.linalg.eigvalsh(X)
    return int(np.sum(ev > rel * max(ev.max(), 1e-300)))


# =========================================================================== designs
@dataclass
class DesignResult:
    arch: str
    R_min: float
    Rx: np.ndarray          # N x N covariance radiated by the array
    P: np.ndarray           # N x K beamformers on the array
    crb: np.ndarray         # CRB matrix (true units: m^2, rad^2)
    rates: np.ndarray       # achieved rates (bit/s/Hz)
    sol: SDRSolution
    extras: dict = field(default_factory=dict)

    @property
    def rcrb_r(self) -> float:
        return float(np.sqrt(self.crb[0, 0]))

    @property
    def rcrb_theta_deg(self) -> float:
        return float(np.rad2deg(np.sqrt(self.crb[-1, -1])))


FORMULATIONS = ("fast", "paper")


def _solver(formulation: str):
    """"fast": solve_crb_sdr (exact reformulation); "paper": solve_crb_sdr_paper (literal)."""
    if formulation not in FORMULATIONS:
        raise ValueError(f"unknown formulation {formulation!r}")
    return solve_crb_sdr if formulation == "fast" else solve_crb_sdr_paper


def design_fully_digital(scn: Scenario, R_min: float, solver: str = "MOSEK", formulation: str = "paper",
                         **kw) -> DesignResult:
    p = scn.p
    fim = near_field_model(p, scn.beta_s)
    sol = _solver(formulation)(scn.H, fim, p.P_max, p.noise, R_min, solver=solver, **kw)
    crb = fim.crb(sol.R)
    rates = achievable_rates(scn.H, sol.p, sol.R, p.noise)
    return DesignResult("FD", R_min, sol.R, sol.p, crb, rates, sol)


def analog_beamformer(scn: Scenario) -> np.ndarray:
    """Eq. (22): p_RF,l = a^*(r_l, theta_l) for l <= K, a^*(r_s, theta_s) for K < l <= N_RF
    (a_far^* for the users / the target when the scenario models them in the far field)."""
    p = scn.p
    if scn.far_field_users:
        cols = [np.conj(array_response_far(p, scn.theta_users[l])) for l in range(p.K)]
    else:
        cols = [np.conj(array_response(p, scn.r_users[l], scn.theta_users[l])) for l in range(p.K)]
    cols += [np.conj(scn.a_s)] * (p.N_RF - p.K)
    return np.stack(cols, axis=1)


def random_analog_combiner(p, rng: np.random.Generator) -> np.ndarray:
    """W_RF in C^{N_RF x N} with entries drawn uniformly from the unit circle (Sec. II-C, ref. [15])."""
    return np.exp(1j * 2.0 * np.pi * rng.random((p.N_RF, p.N)))


def conj_symmetric_phases(p, rng: np.random.Generator) -> np.ndarray:
    """U in C^{N_RF x N}, unit modulus, U[:, -n] = conj(U[:, n]) about the array centre, U[:, 0] = 1;
    the phases for n > 0 are uniform on the unit circle."""
    ph = 2.0 * np.pi * rng.random((p.N_RF, p.N_tilde))
    return np.exp(1j * np.hstack([-ph[:, ::-1], np.zeros((p.N_RF, 1)), ph]))


def matched_analog_combiner(a_s: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Target-matched random combiner (variant, NOT the paper's [15]): rows (a_s .* u_i)^H.

    Unit modulus and (1/N) W W^H ~ I as for the random combiner, but the rows are conjugate-symmetric
    about the array centre once the target phase is removed, so the r / theta derivatives of the echo
    stay (nearly) orthogonal after W_RF, as for the fully digital receiver, and the r-theta coupling
    vanishes as r_s grows.  Uses the target location, like the analog beamformer (22).
    """
    return np.conj(a_s[None, :] * U)


def design_hybrid(scn: Scenario, R_min: float, W_RF: np.ndarray, solver: str = "MOSEK",
                  formulation: str = "paper", **kw) -> DesignResult:
    """Two-stage hybrid design of Sec. III-C.

    Stage 2 optimises (p_BB,k, R_BB) for the fixed P_RF:
        rates  with h_BB,k = P_RF^T h_k,   power tr(P_RF R_BB P_RF^H) <= P_max,
        CRB(P_RF R_BB P_RF^H, W_RF G, N sigma_s^2).
    "paper": solved directly in R_BB, as written.
    "fast": solved in the equivalent whitened variable R' = S V^H R_BB V S (from the SVD
    P_RF = U_r S V^H), for which R_HB = U_r R' U_r^H and the power constraint is
    tr(R') <= P_max -- i.e. exactly the fully digital problem restricted to range(P_RF).
    The change of variables is one-to-one, so the optimum is unchanged; it only removes
    the bad conditioning of tr(P_RF^H P_RF R_BB) for users very close to the array.
    A far-field scenario gives the hybrid far-field ISAC benchmark of Fig. 4 (angle-only CRB).
    """
    p = scn.p
    model = lambda **kw: target_model(p, scn.beta_s, scn.far_field, noise=p.N * p.noise, **kw)
    P_RF = analog_beamformer(scn)                       # stage 1, Eq. (22)
    if formulation == "paper":
        fim_bb = model(W_RF=W_RF, P_RF=P_RF)            # effective noise N sigma_s^2
        sol = solve_crb_sdr_paper(P_RF.T @ scn.H, fim_bb, p.P_max, p.noise, R_min, solver=solver,
                                  Q_power=P_RF.conj().T @ P_RF, **kw)                    # stage 2
        R_bb, P_bb = sol.R, sol.p
        R_hb, P_hb = P_RF @ R_bb @ P_RF.conj().T, P_RF @ P_bb
    else:
        U, sv, _ = np.linalg.svd(P_RF, full_matrices=False)
        Ur = U[:, sv > 1e-10 * sv[0]]                   # orthonormal basis of range(P_RF)
        fim_w = model(W_RF=W_RF, P_RF=Ur)
        sol = _solver(formulation)(Ur.T @ scn.H, fim_w, p.P_max, p.noise, R_min, solver=solver, **kw)
        R_hb = Ur @ sol.R @ Ur.conj().T                 # = P_RF R_BB P_RF^H
        P_hb = Ur @ sol.p                               # = P_RF p_BB,k
        pinv = np.linalg.pinv(P_RF)
        R_bb, P_bb = pinv @ R_hb @ pinv.conj().T, pinv @ P_hb
    fim_hb = model(W_RF=W_RF)
    crb = fim_hb.crb(R_hb)                              # CRB(R_HB, W_RF G, N sigma_s^2)
    rates = achievable_rates(scn.H, P_hb, R_hb, p.noise)
    return DesignResult("HB", R_min, R_hb, P_hb, crb, rates, sol,
                        {"P_RF": P_RF, "W_RF": W_RF, "R_BB": R_bb, "P_BB": P_bb})


def design_far_field(scn: Scenario, R_min: float, solver: str = "MOSEK", formulation: str = "paper",
                     **kw) -> DesignResult:
    """Far-field ISAC benchmark of Fig. 3(b): far-field channels (Eq. (5)) and
    the angle-only CRB of ref. [7], same constraints and same SDR algorithm."""
    if not scn.far_field:
        raise ValueError("design_far_field expects a far-field scenario")
    p = scn.p
    fim = far_field_model(p, scn.beta_s)
    sol = _solver(formulation)(scn.H, fim, p.P_max, p.noise, R_min, solver=solver, **kw)
    crb = fim.crb(sol.R)
    rates = achievable_rates(scn.H, sol.p, sol.R, p.noise)
    return DesignResult("FF", R_min, sol.R, sol.p, crb, rates, sol)
