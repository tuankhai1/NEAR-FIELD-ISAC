# Near-Field ISAC — Python Reproduction Baseline

A tested Python baseline for reproducing the main numerical results of:

> Z. Wang, X. Mu, and Y. Liu, “Near-Field Integrated Sensing and Communications,” *IEEE Communications Letters*, vol. 27, no. 8, pp. 2048–2052, Aug. 2023. [DOI](https://doi.org/10.1109/LCOMM.2023.3280132) · [arXiv](https://arxiv.org/abs/2302.01153) · [authors' MATLAB code](https://github.com/zhaolin820/near-field-integrated-sensing-and-communications)

The repository implements the spherical-wave channel, joint range/angle Fisher information matrix (FIM), Cramér–Rao bound (CRB), two-dimensional MUSIC, the paper's fully digital semidefinite relaxation (SDR), and its two-stage hybrid design. A fast zero-forcing (ZF) sensing baseline is included as a non-paper comparison.

The detailed Vietnamese paper analysis is in [docs/paper_analysis_vi.md](docs/paper_analysis_vi.md).

## Implemented experiments

| Paper result | What this repository generates | Entry point |
|---|---|---|
| Fig. 2 | Range/angle RCRB versus minimum user rate for fully digital and hybrid arrays | `nf-isac figure2` |
| Fig. 3 | Near-field localized MUSIC peak versus far-field range ambiguity | `nf-isac figure3` |
| Fig. 4 | Range/angle RCRB versus target distance with target pathloss held fixed | `nf-isac figure4` |

Two presets are available:

- `quick`: 17 antennas and 2 users. Use it for tests, debugging, and solver checks. It is not a numerical reproduction of the paper.
- `paper`: 65 antennas, 4 users, 5 RF chains, 28 GHz, 0.5 m aperture, 128 snapshots, 20 dBm transmit power, and −60 dBm noise power.

## Installation

Python 3.10 or newer is required. Python 3.11–3.13 is the most conservative choice for scientific-computing environments.

```bash
git clone <your-repository-url>
cd near-field-isac-reproduction
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the package, CVXPY, and development tools:

```bash
python -m pip install -e ".[optimization,dev]"
```

If only the analytical ZF/MUSIC baseline is needed, CVXPY can be omitted:

```bash
python -m pip install -e .
```

### Solver choice

`--solver auto` tries MOSEK, then CLARABEL, then SCS. CLARABEL is a good open-source choice for `quick`. The paper-size fully digital problem contains five large complex PSD variables; MOSEK is strongly recommended for Fig. 2 and Fig. 4, matching the public MATLAB code. CLARABEL/SCS can be substantially slower at `N=65`, and SCS may return `optimal_inaccurate`.

## Quick start

Run the complete MUSIC pipeline without a convex solver:

```bash
nf-isac figure3 --preset quick --optimizer zf --grid-size 121
```

Check the paper's SDR on the small preset:

```bash
nf-isac figure3 --preset quick --optimizer sdr --solver CLARABEL --grid-size 121
```

Generate short versions of the CRB sweeps:

```bash
nf-isac figure2 --preset quick --solver CLARABEL --rates 0 2 5
nf-isac figure4 --preset quick --solver CLARABEL --distances 5 20 40
```

Equivalent wrapper scripts are available under `scripts/`.

## Paper-parameter runs

The following commands use the parameters and grid sizes reported by the paper/author code:

```bash
nf-isac figure3 --preset paper --optimizer sdr --solver MOSEK --grid-size 500
nf-isac figure2 --preset paper --solver MOSEK --rates 0 1 2 3 4 5 6 7 8 9 10
nf-isac figure4 --preset paper --solver MOSEK --distances 5 10 15 20 25 30 35 40
```

Generated plots, CSV/NPZ data, and JSON summaries are written to `results/<figure>/`. Generated artifacts are ignored by Git; `results/.gitkeep` preserves the directory.

## Expected behavior

- Near-field MUSIC should peak around the true target at `(r, theta) = (20 m, 45°)` because spherical curvature contains both range and angle information.
- Far-field MUSIC should form a ridge along `theta = 45°`; its grid maximum has no unique range interpretation.
- Increasing the minimum communication rate consumes waveform degrees of freedom and generally worsens sensing CRBs.
- The range RCRB grows rapidly as the target moves farther away because the spherical response approaches a planar wave.
- The two-stage hybrid design should be below the fully digital sensing performance bound because its RF stage is constrained to unit-modulus focusing vectors.

## Codebase map

```text
.
├── docs/
│   └── paper_analysis_vi.md       # paper derivation, critique, and reproduction notes
├── scripts/
│   ├── reproduce_figure2.py
│   ├── reproduce_figure3.py
│   └── reproduce_figure4.py
├── src/near_field_isac/
│   ├── channels.py                # Eq. (1)–(7): ULA, spherical/planar channels
│   ├── communication.py           # Eq. (11), feasible ZF+sensing baseline
│   ├── config.py                  # paper/quick presets and unit conventions
│   ├── experiments.py             # deterministic Fig. 2–4 pipelines and plots
│   ├── fim.py                     # Eq. (13), Appendix-B FIM and CRB
│   ├── music.py                   # Eq. (23)–(24), signal simulation and 2D MUSIC
│   ├── optimization.py            # Eq. (17)–(22), FD-SDR and hybrid SDR
│   └── cli.py                     # command-line interface
└── tests/                         # analytical, numerical, MUSIC, and small-SDP tests
```

### Data flow

1. `SimulationConfig` fixes units and parameters.
2. `generate_scenario` samples user positions and target reflection with a deterministic NumPy RNG.
3. An optimizer returns total covariance `Rx`, user beamformers `f`, and residual sensing covariance `Rs`.
4. `fim.py` computes the physical CRB; `music.py` separately generates finite-snapshot echoes and estimates the target.
5. `experiments.py` saves plots plus machine-readable inputs/results for later comparisons.

## Optimization methods

### Paper methods

- **Fully digital SDR:** lifts `f_k f_k^H` to PSD matrices, solves problem (20), then performs the exact rank-one recovery in Eq. (21).
- **Hybrid two-stage SDR:** constructs the unit-modulus RF beamformer from user/target focusing vectors in Eq. (22), draws the paper's random unit-modulus receive combiner, and optimizes the lower-dimensional baseband covariance.

The CVXPY formulation uses an exact congruence preconditioner and a Schur-complement epigraph for `trace(U^{-1})`. These improve open-source solver conditioning without changing the mathematical objective or feasible set.

### Additional baseline

- **ZF + focused sensing:** zero-forces communication users, analytically allocates the minimum communication power including sensing leakage, and puts all remaining power into a target-focused rank-one sensing covariance. This method is fast and feasible but is not globally optimal.

Good next extensions are WMMSE/SCA waveform optimization, Riemannian optimization of hybrid phase shifters, robust CRB design under target-location uncertainty, and first-order/ADMM solvers for larger arrays. They are intentionally not labeled as reproduced paper results.

## Reproducibility notes

Exact pixel-level reproduction is not guaranteed for several reasons:

1. The paper states that user positions are random but does not publish their realization, random seed, or Monte Carlo averaging count. The public MATLAB script also does not set a seed. This repository defaults to `seed=2023` and records it in every JSON result.
2. The public author repository implements the fully digital Fig. 3 path only; the Fig. 2/4 sweep loops and hybrid optimization are reconstructed from the paper equations.
3. `rho_0 = lambda/(4*pi)`, normalized communication noise, and dBm-to-mW conversion deliberately follow the public MATLAB implementation.
4. For Fig. 4, the complex target gain generated at the nominal 20 m location is held fixed across the distance sweep. This is the explicit implementation of “without factoring in pathloss”; the paper does not provide a more detailed convention.
5. The paper describes a Cartesian step of 0.08 m while the author code uses `linspace(0, 40, 500)`, whose actual step is approximately 0.08016 m. The `paper` preset follows the code's 500-point grid.

Use CSV/JSON outputs, solver status, achieved rates, and fixed seeds when comparing new optimization algorithms; comparing only rendered figures can hide infeasible or inaccurate solver solutions.

## Tests

```bash
pytest -q
ruff check src tests scripts
```

The tests cover spherical-wave derivatives, the far-field limit, FIM scaling, CRB positivity, ZF rate/power feasibility, near-field localization, far-field range ambiguity, and a small end-to-end SDR.

## Citation

If this baseline is useful in research, cite the original paper:

```bibtex
@article{wang2023nearfieldisac,
  author  = {Zhaolin Wang and Xidong Mu and Yuanwei Liu},
  title   = {Near-Field Integrated Sensing and Communications},
  journal = {IEEE Communications Letters},
  volume  = {27},
  number  = {8},
  pages   = {2048--2052},
  month   = aug,
  year    = {2023},
  doi     = {10.1109/LCOMM.2023.3280132}
}
```

## License

This Python baseline is released under the [MIT License](LICENSE). The paper and the authors' MATLAB repository remain subject to their respective licenses and copyright terms.

