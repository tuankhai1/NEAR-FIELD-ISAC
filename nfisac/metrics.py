"""Communication metric, Eq. (11)."""

from __future__ import annotations

import numpy as np


def achievable_rates(H: np.ndarray, P: np.ndarray, Rx: np.ndarray, noise: float) -> np.ndarray:
    """R(p_k, R_x) = log2(1 + |h_k^T p_k|^2 / (h_k^T R_x h_k^* - |h_k^T p_k|^2 + sigma_k^2)).

    H: (N, K) channels h_k, P: (N, K) beamformers p_k, Rx: (N, N) transmit covariance.
    """
    sig = np.abs(np.einsum("nk,nk->k", H, P)) ** 2
    tot = np.real(np.einsum("nk,nm,mk->k", H, Rx, H.conj()))
    sinr = sig / np.maximum(tot - sig + noise, 1e-300)
    return np.log2(1.0 + sinr)
