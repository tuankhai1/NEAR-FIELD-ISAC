"""Random draws the paper uses but does not report, recovered from its figures.

The paper's Figs. 2--4 come from single realizations of three random
quantities that are not reported: the four user locations, the target
reflection coefficient, and the random unit-modulus hybrid receive combiner.
The public MATLAB code sets no RNG seed.  The values below were recovered
from the digitized curves:

* Reflection magnitude.  With zero rate requirement the optimal fully digital
  waveform is the beam focused on the target, so the RCRBs scale exactly as
  ``1 / |beta_s|``.  One scalar matches FD range *and* angle in Figs. 2 and 4,
  and the RCRB range/angle ratio (150.61 m/deg) then agrees with the paper to
  five digits.  The resulting echo SNR is also consistent with the ~-70 dB
  floor of the paper's MUSIC spectrum.  The phase does not affect any result.
* Users.  Every Fig. 2 curve rises by the same factor with ``R_min``, so the
  user geometry only sets the communication power cost.  The seed below was
  selected so both FD and HB curves follow the paper's rise up to 10.7 bit/s/Hz.
* Combiners.  At zero rate cost the optimal hybrid waveform is also pure
  target focusing, so the HB/FD gap depends only on the random combiner.
  Fig. 2 and Fig. 4 evidently used different draws (their HB values at
  20 m and 5 bit/s/Hz differ by 38 %), hence two seeds.

Candidate draws use the same distributions as the model (uniform users in
``[0, Rayleigh distance] x [0, pi]``; i.i.d. uniform combiner phases).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .optimization import random_hybrid_combiner

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

# Matches the paper's FD RCRB of 5.1985e-3 m at 20 m (Figs. 2 and 4).
PAPER_REFLECTION_MAGNITUDE = 3.59852
PAPER_USER_SEED = 1522
PAPER_FIGURE2_COMBINER_SEED = 1429895
PAPER_FIGURE4_COMBINER_SEED = 6067369


@dataclass(frozen=True)
class Realization:
    """User locations, target reflection and hybrid receive combiner."""

    user_ranges: FloatArray
    user_angles: FloatArray
    target_reflection: complex
    receive_combiner: ComplexArray
    source: str


def draw_users(config: SimulationConfig, seed: int) -> tuple[FloatArray, FloatArray]:
    """Uniform user ranges over ``[0, Rayleigh]`` and angles over ``[0, pi]``."""

    rng = np.random.default_rng(seed)
    ranges = rng.random(config.n_users) * config.rayleigh_distance
    angles = rng.random(config.n_users) * np.pi
    return ranges, angles


def draw_combiner(config: SimulationConfig, seed: int) -> ComplexArray:
    """Random unit-modulus receive combiner, phases i.i.d. uniform."""

    return random_hybrid_combiner(config, np.random.default_rng(seed))


def _require_paper_dimensions(config: SimulationConfig) -> None:
    paper = SimulationConfig.paper()
    fields = (
        "n_antennas",
        "n_rf_chains",
        "n_users",
        "carrier_frequency",
        "aperture",
        "target_range",
        "target_angle_deg",
    )
    mismatched = [name for name in fields if getattr(config, name) != getattr(paper, name)]
    if mismatched:
        raise ValueError(
            "The recovered paper realization requires the paper system model; "
            f"changed: {', '.join(mismatched)}. Use realization='random'."
        )


def paper_realization(config: SimulationConfig, figure: int) -> Realization:
    """Recovered realization for paper Fig. 2/3 (``figure=2``) or Fig. 4."""

    if figure not in (2, 3, 4):
        raise ValueError("figure must be 2, 3 or 4")
    _require_paper_dimensions(config)
    ranges, angles = draw_users(config, PAPER_USER_SEED)
    combiner_seed = PAPER_FIGURE4_COMBINER_SEED if figure == 4 else PAPER_FIGURE2_COMBINER_SEED
    return Realization(
        user_ranges=ranges,
        user_angles=angles,
        target_reflection=complex(PAPER_REFLECTION_MAGNITUDE),
        receive_combiner=draw_combiner(config, combiner_seed),
        source=(
            f"paper (users seed {PAPER_USER_SEED}, combiner seed {combiner_seed}, "
            f"|reflection| {PAPER_REFLECTION_MAGNITUDE})"
        ),
    )


def random_realization(config: SimulationConfig, seed: int | None = None) -> Realization:
    """Independent draw: CN(0, 1) reflection, as in ``generate_channel.m``."""

    rng = np.random.default_rng(config.seed if seed is None else seed)
    ranges = rng.random(config.n_users) * config.rayleigh_distance
    angles = rng.random(config.n_users) * np.pi
    reflection = complex((rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0))
    combiner = random_hybrid_combiner(config, rng)
    return Realization(ranges, angles, reflection, combiner, f"random (seed {config.seed})")


def realization_for(config: SimulationConfig, figure: int, kind: str) -> Realization:
    if kind == "paper":
        return paper_realization(config, figure)
    if kind == "random":
        return random_realization(config)
    raise ValueError("realization must be 'paper' or 'random'")
