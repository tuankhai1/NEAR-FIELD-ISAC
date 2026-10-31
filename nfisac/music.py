"""Echo simulation and MUSIC estimation, Appendix A of the paper (MUSIC_estimation.m)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import Scenario, array_response, array_response_far


def _psd_sqrt(R: np.ndarray) -> np.ndarray:
    ev, V = np.linalg.eigh(0.5 * (R + R.conj().T))
    return (V * np.sqrt(np.clip(ev, 0.0, None))) @ V.conj().T


def _cn(rng: np.random.Generator, shape) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def simulate_echo(scn: Scenario, Rx: np.ndarray, P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Y_s = [y_s[1], ..., y_s[T]] with y_s[t] = G x[t] + z_s[t]  (Eqs. (8), (12)).

    x[t] = sum_k p_k c_k[t] + s[t]: c_k[t] ~ CN(0,1) i.i.d. and the dedicated
    sensing signal s[t] ~ CN(0, R_s), R_s = R_x - sum_k p_k p_k^H, exactly as
    generated in MUSIC_estimation.m.
    """
    p = scn.p
    Rs = Rx - P @ P.conj().T
    s = _psd_sqrt(Rs) @ _cn(rng, (p.N, p.T))
    c = _cn(rng, (p.K, p.T))
    X = P @ c + s
    Z = np.sqrt(p.noise) * _cn(rng, (p.N, p.T))
    return scn.G @ X + Z


@dataclass
class MusicResult:
    x: np.ndarray            # grid x-coordinates (along the array) [m]
    y: np.ndarray            # grid y-coordinates [m]
    spectrum: np.ndarray     # normalised 1/p(r,theta), shape (len(y), len(x))
    r_hat: float
    theta_hat: float         # [rad]

    @property
    def spectrum_db(self) -> np.ndarray:
        return 10.0 * np.log10(self.spectrum)


def music_spectrum(scn: Scenario, Y: np.ndarray, x: np.ndarray, y: np.ndarray,
                   n_targets: int = 1, far_field: bool | None = None, chunk: int = 20000) -> MusicResult:
    """MUSIC pseudo-spectrum over the Cartesian grid (x, y), Eqs. (23)-(24).

    R_bar = Y Y^H / T = E_s D_s E_s^H + E_n D_n E_n^H,
    p(r, theta) = a^H(r,theta) E_n E_n^H a(r,theta),   spectrum = 1 / p, normalised to max = 1.
    Near-field ISAC scans a(r, theta) (Eq. (4)); the far-field benchmark scans a_far(theta) (Eq. (5)).
    """
    p = scn.p
    far = scn.far_field if far_field is None else far_field
    R_bar = Y @ Y.conj().T / Y.shape[1]
    ev, E = np.linalg.eigh(R_bar)                    # ascending eigenvalues
    En = E[:, : p.N - n_targets]                     # noise subspace (N - M smallest)

    XX, YY = np.meshgrid(x, y)                       # array along x (Fig. 1)
    r_all = np.hypot(XX, YY).ravel()
    th_all = np.arctan2(YY, XX).ravel()
    proj = np.empty(r_all.size)
    for i in range(0, r_all.size, chunk):
        sl = slice(i, i + chunk)
        A = array_response_far(p, th_all[sl]) if far else array_response(p, r_all[sl], th_all[sl])
        proj[sl] = np.sum(np.abs(A.conj() @ En) ** 2, axis=1)   # ||E_n^H a||^2
    spec = 1.0 / proj
    spec /= spec.max()
    idx = int(np.argmax(spec))
    return MusicResult(x, y, spec.reshape(XX.shape), float(r_all[idx]), float(th_all[idx]))
