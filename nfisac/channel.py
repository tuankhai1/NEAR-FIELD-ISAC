"""Near-field (and far-field) channel model, Sec. II-A of the paper, Eqs. (1)-(7).

Coordinate system (Fig. 1): origin at the ULA centre, array along the x-axis,
element n at s_n = [n d, 0]^T, a point at r = [r cos(theta), r sin(theta)]^T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import SystemParams


# --------------------------------------------------------------------------- geometry
def element_distances(p: SystemParams, r, theta) -> np.ndarray:
    """Eq. (1): r_n(r, theta) = sqrt(r^2 + n^2 d^2 - 2 r n d cos(theta)).

    ``r`` and ``theta`` may be scalars or broadcastable arrays; the antenna
    index is appended as the last axis.
    """
    r = np.asarray(r, dtype=float)[..., None]
    theta = np.asarray(theta, dtype=float)[..., None]
    delta = p.antenna_positions
    return np.sqrt(r ** 2 + delta ** 2 - 2.0 * r * delta * np.cos(theta))


def array_response(p: SystemParams, r, theta) -> np.ndarray:
    """Eq. (4): near-field array response, [a(r,theta)]_n = exp(-j 2pi/lambda (r_n - r))."""
    rr = np.asarray(r, dtype=float)[..., None]
    return np.exp(-1j * 2.0 * np.pi / p.lam * (element_distances(p, r, theta) - rr))


def array_response_far(p: SystemParams, theta) -> np.ndarray:
    """Eq. (5): far-field array response, [a_far(theta)]_n = exp(-j 2pi/lambda (-n d cos(theta)))."""
    theta = np.asarray(theta, dtype=float)[..., None]
    return np.exp(-1j * 2.0 * np.pi / p.lam * (-p.antenna_positions * np.cos(theta)))


def array_response_derivatives(p: SystemParams, r: float, theta: float):
    """a(r,theta) and its partial derivatives w.r.t. r and theta (needed for the FIM).

    d r_n / d r     = (r - n d cos(theta)) / r_n
    d r_n / d theta = r n d sin(theta) / r_n
    => d a_n / d x  = -j 2pi/lambda * d(r_n - r)/dx * a_n     (same as FIM.m)
    """
    delta = p.antenna_positions
    rn = element_distances(p, r, theta)
    a = array_response(p, r, theta)
    k = 2.0 * np.pi / p.lam
    a_r = -1j * k * ((r - delta * np.cos(theta)) / rn - 1.0) * a
    a_t = -1j * k * (r * delta * np.sin(theta) / rn) * a
    return a, a_r, a_t


def array_response_far_derivative(p: SystemParams, theta: float):
    """a_far(theta) and d a_far / d theta (far-field ISAC benchmark, angle only)."""
    a = array_response_far(p, theta)
    a_t = 1j * 2.0 * np.pi / p.lam * (-p.antenna_positions * np.sin(theta)) * a
    return a, a_t


# --------------------------------------------------------------------------- gains
def link_gain(p: SystemParams, r) -> np.ndarray:
    """Complex gain of the central link, beta = beta_tilde e^{-j 2pi r / lambda},
    beta_tilde = sqrt(rho_0) / r  (Eq. (2))."""
    r = np.asarray(r, dtype=float)
    return np.sqrt(p.rho0) / r * np.exp(-1j * 2.0 * np.pi / p.lam * r)


def target_gain(p: SystemParams, reflection: complex = 1.0) -> complex:
    """Round-trip complex gain beta_s of the target.

    The paper only states that beta_s is the complex channel gain of the
    target.  ``generate_channel.m`` uses free-space loss over the round-trip
    length 2 r_s times a reflection coefficient alpha ~ CN(0,1):
        beta_s = sqrt(rho_0) / (2 r_s) * exp(-j 2pi 2 r_s / lambda) * alpha.
    """
    r2 = 2.0 * p.r_s
    return complex(np.sqrt(p.rho0) / r2 * np.exp(-1j * 2.0 * np.pi / p.lam * r2) * reflection)


# |alpha| that brings the fully digital RCRB of the correct FIM to the level of the paper's Fig. 2 and
# Fig. 4 (R_min = 0, r_s = 20 m: 5.1985e-3 m and 3.4515e-5 deg, vector-extracted from arXiv v5).
# Not reported by the authors (generate_channel.m draws alpha ~ CN(0,1) without a seed): a value
# calibrated on the figures, with P(|alpha| >= 3.5985) ~ 2e-6 for alpha ~ CN(0,1).
PAPER_ALPHA = 3.5985


def alpha_value(alpha) -> float:
    """--alpha argument: "paper" -> PAPER_ALPHA, otherwise a number."""
    return PAPER_ALPHA if str(alpha).lower() == "paper" else float(alpha)


def draw_reflection(rng: np.random.Generator, mode) -> complex:
    """Reflection coefficient alpha of the target.

    ``"rayleigh"``: alpha ~ CN(0,1) exactly as in generate_channel.m.
    ``"unit"``    : alpha = 1, i.e. |alpha|^2 = E|alpha|^2.  Recommended for the
                    averaged curves of Fig. 2: the optimal waveform does not depend
                    on beta_s and the CRB simply scales with 1/|beta_s|^2, so a
                    random alpha would only add a heavy-tailed scalar factor.
    a number      : fixed alpha (--alpha); like "unit" nothing is drawn, so users and W_RF
                    are the same as with alpha = 1 and every CRB is divided by |alpha|^2.
    """
    if isinstance(mode, (int, float)):
        return complex(mode)
    if mode == "unit":
        return 1.0 + 0.0j
    if mode == "rayleigh":
        return complex((rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0))
    raise ValueError(f"unknown reflection mode {mode!r}")


# --------------------------------------------------------------------------- scenario
@dataclass
class Scenario:
    """One channel realisation: K users + one target."""

    p: SystemParams
    r_users: np.ndarray        # (K,)
    theta_users: np.ndarray    # (K,)
    H: np.ndarray              # (N, K), column k = h_k, Eq. (6)
    beta_s: complex            # round-trip gain of the target
    far_field: bool = False    # True -> target (sensing) responses use the far-field model (Eq. (5))
    far_field_users: bool | None = None   # model of the users' channels in H; None -> same as far_field

    def __post_init__(self):
        if self.far_field_users is None:
            self.far_field_users = self.far_field

    @property
    def a_s(self) -> np.ndarray:
        if self.far_field:
            return array_response_far(self.p, self.p.theta_s)
        return array_response(self.p, self.p.r_s, self.p.theta_s)

    @property
    def G(self) -> np.ndarray:
        """Eq. (7): round-trip channel G = beta_s a(r_s,theta_s) a^T(r_s,theta_s)."""
        a = self.a_s
        return self.beta_s * np.outer(a, a)


def draw_users(p: SystemParams, rng: np.random.Generator, r_low: float | None = None):
    """Random user locations inside the near-field region.

    Distance ~ U[r_low, 2D^2/lambda] with r_low = 1.2 D by default (the Fresnel
    region in which the paper's uniform-amplitude channel model is valid);
    angle ~ U(0, pi) as in generate_channel.m.
    """
    r_low = p.fresnel_min if r_low is None else r_low
    r = rng.uniform(r_low, p.rayleigh_distance, size=p.K)
    theta = rng.uniform(0.0, np.pi, size=p.K)
    return r, theta


def build_scenario(p: SystemParams, r_users, theta_users, reflection: complex = 1.0,
                   far_field: bool = False) -> Scenario:
    r_users = np.asarray(r_users, dtype=float)
    theta_users = np.asarray(theta_users, dtype=float)
    if far_field:
        A = array_response_far(p, theta_users)            # (K, N)
    else:
        A = array_response(p, r_users, theta_users)       # (K, N)
    H = (link_gain(p, r_users)[:, None] * A).T             # (N, K), h_k = beta_k a(r_k, theta_k)
    return Scenario(p, r_users, theta_users, H, target_gain(p, reflection), far_field)


def random_scenario(p: SystemParams, rng: np.random.Generator, reflection_mode: str = "unit",
                    far_field: bool = False) -> Scenario:
    r, th = draw_users(p, rng)
    return build_scenario(p, r, th, draw_reflection(rng, reflection_mode), far_field)
