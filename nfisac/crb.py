"""Fisher information matrix and CRB of the echo model, Eq. (13) and Appendix B.

Echo model (Eqs. (12), (15)):  y_s[t] = beta_s * G_eff x[t] + z[t],  z ~ CN(0, sigma^2 I)
with  G_eff = W_RF * G_tilde * P_RF   (W_RF = I for fully digital, P_RF = I when
the covariance is given directly on the antennas), G_tilde = a a^T.

Unknowns xi = [eta_1, ..., eta_P, Re(beta_s), Im(beta_s)] where eta = (r_s, theta_s)
for near-field sensing and eta = (theta_s,) for the far-field benchmark. With
R_x ~= (1/T) sum_t x[t] x[t]^H the FIM entries are (Eqs. (26)-(29)):

    J_{l p}       = 2|beta|^2 T / sigma^2 * Re tr(G_p R G_l^H)
    J_{l beta_r}  = 2 T / sigma^2 * Re{ beta^*  tr(G R G_l^H) }
    J_{l beta_i}  = 2 T / sigma^2 * Re{ j beta^* tr(G R G_l^H) }
    J_{beta beta} = 2 T / sigma^2 * tr(G R G^H) * I_2

All entries are affine in R, which is what makes the LMI reformulation (17b)
possible.  The same class therefore evaluates the FIM for a numerical R and
builds it as a CVXPY expression for an optimisation variable R.
"""

from __future__ import annotations

import numpy as np

from .channel import array_response_derivatives, array_response_far_derivative
from .params import SystemParams


def _trace_coeff(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return C such that tr(A R B^H) = sum(C * R) for any R."""
    return (B.conj().T @ A).T


class EchoFisherModel:
    def __init__(self, G0: np.ndarray, dG: list[np.ndarray], beta: complex, T: int, noise: float):
        self.G0, self.dG, self.T, self.noise = G0, dG, T, noise
        self.n_par = len(dG)
        self.beta = complex(beta)
        self.c = 2.0 * T / noise
        self.C_pp = {(i, j): _trace_coeff(dG[i], dG[j]) for i in range(self.n_par) for j in range(i, self.n_par)}
        self.C_0p = [_trace_coeff(G0, dG[i]) for i in range(self.n_par)]
        self.C_00 = _trace_coeff(G0, G0)

    def transmit_directions(self) -> np.ndarray:
        """Columns spanning every x with which R enters the FIM as x^H R y.

        tr(A R B^H) only involves the rows of A and B, i.e. range(A^H), range(B^H).
        """
        return np.hstack([self.G0.conj().T] + [X.conj().T for X in self.dG])

    def restricted(self, B: np.ndarray) -> "EchoFisherModel":
        """Model for R = B R~ B^H, i.e. G -> G B (used for the exact subspace reduction)."""
        return EchoFisherModel(self.G0 @ B, [X @ B for X in self.dG], self.beta, self.T, self.noise)

    # ------------------------------------------------------------------ blocks
    def blocks(self, R, cvx: bool = False):
        """J11 (P x P), J12 (P x 2), J22 (2 x 2) for covariance R.

        With ``cvx=True`` R is a CVXPY variable and nested lists of affine
        CVXPY expressions are returned instead of arrays.  All traces are then
        built as one product A vec(R), which keeps CVXPY's compile time small
        for full N x N variables.
        """
        coefs = list(self.C_pp.values()) + self.C_0p + [self.C_00]
        if cvx:
            import cvxpy as cp

            A = np.stack([C.ravel(order="F") for C in coefs])
            v = A @ cp.vec(R, order="F")
            traces = [v[i] for i in range(len(coefs))]
            re = cp.real
        else:
            R = np.asarray(R)
            traces = [np.sum(C * R) for C in coefs]
            re = np.real
        t_pp = dict(zip(self.C_pp, traces))
        t_0p = traces[len(self.C_pp):-1]
        b2 = abs(self.beta) ** 2
        bc = np.conj(self.beta)
        P = self.n_par
        J11 = [[None] * P for _ in range(P)]
        for (i, j), t in t_pp.items():
            J11[i][j] = J11[j][i] = self.c * b2 * re(t)
        J12 = []
        for i in range(P):
            t = t_0p[i]
            J12.append([self.c * re(bc * t), self.c * re(1j * bc * t)])
        g = self.c * re(traces[-1])
        J22 = [[g, 0.0], [0.0, g]]
        if cvx:
            return J11, J12, J22
        return np.array(J11, float), np.array(J12, float), np.array(J22, float)

    def block_coefficients(self):
        """Complex D with J_entry(R) = Re sum(D * R) for every FIM entry (Eqs. (26)-(29)):
        J11 as {(i, j): D} (i <= j), J12 as [[D_i_br, D_i_bi], ...], J22 (both diagonal entries)."""
        b2, bc = abs(self.beta) ** 2, np.conj(self.beta)
        J11 = {k: self.c * b2 * C for k, C in self.C_pp.items()}
        J12 = [[self.c * bc * C, self.c * 1j * bc * C] for C in self.C_0p]
        return J11, J12, self.c * self.C_00

    def fim(self, R) -> np.ndarray:
        J11, J12, J22 = self.blocks(R)
        return np.block([[J11, J12], [J12.T, J22]])

    def crb(self, R) -> np.ndarray:
        """Eq. (13): CRB = (J11 - J12 J22^{-1} J12^T)^{-1}."""
        J11, J12, J22 = self.blocks(R)
        return np.linalg.inv(J11 - J12 @ np.linalg.solve(J22, J12.T))


# ---------------------------------------------------------------------- builders
def near_field_model(p: SystemParams, beta_s: complex, noise: float | None = None,
                     W_RF: np.ndarray | None = None, P_RF: np.ndarray | None = None) -> EchoFisherModel:
    """FIM model for joint (r_s, theta_s) estimation.

    Fully digital: W_RF = P_RF = None, noise = sigma_s^2.
    Hybrid (Sec. II-C): G_eff = W_RF G_tilde, effective noise N sigma_s^2; pass
    P_RF as well to express the FIM directly in the baseband covariance R_BB
    (R_HB = P_RF R_BB P_RF^H), which is what the digital-stage optimisation uses.

    dG~/dr = a_r a^T + a a_r^T (both the transmit and the receive leg depend on r),
    as in Eqs. (26)-(29) and the official FIM.m after commit 77929f8 (2024).
    """
    a, a_r, a_t = array_response_derivatives(p, p.r_s, p.theta_s)
    G0 = np.outer(a, a)
    dG = [np.outer(a_r, a) + np.outer(a, a_r), np.outer(a_t, a) + np.outer(a, a_t)]
    if W_RF is not None:
        G0, dG = W_RF @ G0, [W_RF @ X for X in dG]
    if P_RF is not None:
        G0, dG = G0 @ P_RF, [X @ P_RF for X in dG]
    return EchoFisherModel(G0, dG, beta_s, p.T, p.noise if noise is None else noise)


def far_field_model(p: SystemParams, beta_s: complex, noise: float | None = None,
                    W_RF: np.ndarray | None = None, P_RF: np.ndarray | None = None) -> EchoFisherModel:
    """FIM model of the far-field ISAC benchmark (angle only, as in ref. [7]).
    W_RF, P_RF and noise as in ``near_field_model`` (hybrid far-field benchmark of Fig. 4)."""
    a, a_t = array_response_far_derivative(p, p.theta_s)
    G0 = np.outer(a, a)
    dG = [np.outer(a_t, a) + np.outer(a, a_t)]
    if W_RF is not None:
        G0, dG = W_RF @ G0, [W_RF @ X for X in dG]
    if P_RF is not None:
        G0, dG = G0 @ P_RF, [X @ P_RF for X in dG]
    return EchoFisherModel(G0, dG, beta_s, p.T, p.noise if noise is None else noise)


def target_model(p: SystemParams, beta_s: complex, far_field: bool = False, **kw) -> EchoFisherModel:
    """``far_field_model`` for a far-field scenario, ``near_field_model`` otherwise."""
    return (far_field_model if far_field else near_field_model)(p, beta_s, **kw)
