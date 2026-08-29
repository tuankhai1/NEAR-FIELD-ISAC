"""Simulation parameters and paper/quick presets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi


def dbm_to_mw(value_dbm: float) -> float:
    """Convert dBm to mW, matching the convention used by the author code."""

    return 10.0 ** (value_dbm / 10.0)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the narrowband monostatic near-field ISAC model.

    The defaults reproduce the parameters listed in Section IV of the paper and
    in the authors' MATLAB ``para_init.m``.  Communication channels are divided
    by the receiver noise standard deviation, so the optimization uses unit
    communication-noise power.  Sensing calculations retain the physical noise
    power in mW.
    """

    n_antennas: int = 65
    n_rf_chains: int = 5
    coherent_block_length: int = 128
    transmit_power_dbm: float = 20.0
    n_users: int = 4
    noise_power_dbm: float = -60.0
    min_rate: float = 5.0
    speed_of_light: float = 3.0e8
    carrier_frequency: float = 28.0e9
    aperture: float = 0.5
    target_range: float = 20.0
    target_angle_deg: float = 45.0
    optimization_scale: float = 1.0e2
    seed: int = 2023

    def __post_init__(self) -> None:
        if self.n_antennas < 3 or self.n_antennas % 2 != 1:
            raise ValueError("n_antennas must be an odd integer >= 3")
        if not (1 <= self.n_users < self.n_antennas):
            raise ValueError("n_users must satisfy 1 <= n_users < n_antennas")
        if self.n_rf_chains < self.n_users + 1:
            raise ValueError("n_rf_chains must be at least n_users + 1")
        if self.aperture <= 0 or self.carrier_frequency <= 0:
            raise ValueError("aperture and carrier_frequency must be positive")
        if self.target_range <= 0:
            raise ValueError("target_range must be positive")
        if self.coherent_block_length < 1:
            raise ValueError("coherent_block_length must be positive")

    @classmethod
    def paper(cls, **updates: object) -> SimulationConfig:
        """Return the full paper preset (65 antennas, 4 users, 5 RF chains)."""

        return replace(cls(), **updates)

    @classmethod
    def quick(cls, **updates: object) -> SimulationConfig:
        """Return a small deterministic preset for tests and local smoke runs."""

        config = cls(
            n_antennas=17,
            n_users=2,
            n_rf_chains=3,
            coherent_block_length=128,
            aperture=0.5,
            seed=2023,
        )
        return replace(config, **updates)

    def with_updates(self, **updates: object) -> SimulationConfig:
        """Return an immutable copy with selected fields changed."""

        return replace(self, **updates)

    @property
    def wavelength(self) -> float:
        return self.speed_of_light / self.carrier_frequency

    @property
    def antenna_spacing(self) -> float:
        return self.aperture / (self.n_antennas - 1)

    @property
    def rayleigh_distance(self) -> float:
        return 2.0 * self.aperture**2 / self.wavelength

    @property
    def fresnel_lower_bound(self) -> float:
        return 1.2 * self.aperture

    @property
    def transmit_power(self) -> float:
        return dbm_to_mw(self.transmit_power_dbm)

    @property
    def noise_power(self) -> float:
        return dbm_to_mw(self.noise_power_dbm)

    @property
    def target_angle(self) -> float:
        return self.target_angle_deg * pi / 180.0

    @property
    def reference_path_gain(self) -> float:
        """Return rho_0 from the public MATLAB implementation.

        The author code sets ``rho_0 = lambda / (4*pi)`` and then uses
        ``sqrt(rho_0)/r`` as the one-way amplitude.  This is retained for a
        faithful port even though alternative Friis conventions are common.
        """

        return self.wavelength / (4.0 * pi)

