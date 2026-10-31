"""System parameters of the near-field ISAC letter (Wang, Mu, Liu, IEEE CL 2023).

Every default below is taken from Section IV ("Numerical Results") of the paper
and, where the paper is silent, from ``para_init.m`` of the authors' official
MATLAB code (github.com/zhaolin820/near-field-integrated-sensing-and-communications).

Units: lengths in metres, angles in radians, powers in mW (so that the dBm
values of the paper convert directly), rates in bit/s/Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def dbm_to_mw(p_dbm: float) -> float:
    return 10.0 ** (p_dbm / 10.0)


@dataclass(frozen=True)
class SystemParams:
    # --- array / hardware (Sec. II, Sec. IV) -------------------------------
    N: int = 65                 # number of BS antennas, N = 2*N_tilde + 1
    N_RF: int = 5               # RF chains of the hybrid array, K+1 <= N_RF << N
    D: float = 0.5              # antenna aperture D = (N-1) d  [m]
    fc: float = 28e9            # carrier frequency [Hz]
    c: float = 3e8              # speed of light [m/s]

    # --- users / target (Sec. IV) -------------------------------------------
    K: int = 4                  # single-antenna communication users
    r_s: float = 20.0           # target distance [m]
    theta_s_deg: float = 45.0   # target angle [deg], measured from the array axis

    # --- signal / power (Sec. IV, para_init.m) -------------------------------
    T: int = 128                # coherent time block length (para_init.m)
    P_max_dBm: float = 20.0     # maximum transmit power
    noise_dBm: float = -60.0    # noise power, sigma_k^2 = sigma_s^2
    R_min: float = 5.0          # minimum rate per user [bit/s/Hz]

    # ---------------------------------------------------------------- derived
    @property
    def lam(self) -> float:
        """Wavelength lambda = c / fc (1.07 cm at 28 GHz)."""
        return self.c / self.fc

    @property
    def d(self) -> float:
        """Antenna spacing d = D / (N - 1)."""
        return self.D / (self.N - 1)

    @property
    def N_tilde(self) -> int:
        return (self.N - 1) // 2

    @property
    def rayleigh_distance(self) -> float:
        """Rayleigh distance 2 D^2 / lambda (Sec. II)."""
        return 2.0 * self.D ** 2 / self.lam

    @property
    def fresnel_min(self) -> float:
        """Lower edge 1.2 D of the Fresnel region used for the channel model (Sec. II-A)."""
        return 1.2 * self.D

    @property
    def rho0(self) -> float:
        """Reference path loss at 1 m, rho_0 = lambda / (4 pi), as written in the paper
        and implemented in para_init.m."""
        return self.lam / (4.0 * np.pi)

    @property
    def theta_s(self) -> float:
        return np.deg2rad(self.theta_s_deg)

    @property
    def P_max(self) -> float:
        return dbm_to_mw(self.P_max_dBm)

    @property
    def noise(self) -> float:
        return dbm_to_mw(self.noise_dBm)

    @property
    def antenna_positions(self) -> np.ndarray:
        """x-coordinates n d of the ULA elements, n = -N_tilde, ..., N_tilde."""
        n = np.arange(-self.N_tilde, self.N_tilde + 1)
        return n * self.d

    def with_(self, **kwargs) -> "SystemParams":
        return replace(self, **kwargs)

    def __post_init__(self):
        if self.N % 2 != 1:
            raise ValueError("N must be odd (N = 2*N_tilde + 1).")
        if not (self.K + 1 <= self.N_RF <= self.N):
            raise ValueError("The paper assumes K + 1 <= N_RF << N.")
