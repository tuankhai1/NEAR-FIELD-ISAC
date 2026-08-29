# Near-Field ISAC — Python Reproduction Baseline

A tested Python baseline for reproducing the main numerical results of:

> Z. Wang, X. Mu, and Y. Liu, “Near-Field Integrated Sensing and Communications,” *IEEE Communications Letters*, vol. 27, no. 8, pp. 2048–2052, Aug. 2023. [DOI](https://doi.org/10.1109/LCOMM.2023.3280132) · [arXiv](https://arxiv.org/abs/2302.01153) · [authors' MATLAB code](https://github.com/zhaolin820/near-field-integrated-sensing-and-communications)

The repository implements the spherical-wave channel, joint range/angle Fisher information matrix (FIM), Cramér–Rao bound (CRB), two-dimensional MUSIC, the paper's fully digital semidefinite relaxation (SDR), and its two-stage hybrid design. A fast zero-forcing (ZF) sensing baseline is included as a non-paper comparison.

The detailed Vietnamese paper analysis is in [docs/paper_analysis_vi.md](docs/paper_analysis_vi.md).

## Implemented experiments

| Paper result | What this repository generates | Entry point |
|---|---|---|
| Full pipeline | Runs Fig. 2--4 with the full paper preset | `python main.py` |
| Fig. 2 | Range/angle RCRB versus minimum user rate for fully digital and hybrid arrays | `python main.py figure2` |
| Fig. 3 | Near-field localized MUSIC peak versus far-field range ambiguity | `python main.py figure3` |
| Fig. 4 | Range/angle RCRB versus target distance with target pathloss held fixed | `python main.py figure4` |

Two presets are available:

- `quick`: 17 antennas and 2 users. Use it for tests, debugging, and solver checks. It is not a numerical reproduction of the paper.
- `paper`: 65 antennas, 4 users, 5 RF chains, 28 GHz, 0.5 m aperture, 128 snapshots, 20 dBm transmit power, and −60 dBm noise power.

## Installation

Python 3.10 or newer is required. Python 3.11–3.13 is the most conservative choice for scientific-computing environments. The current MOSEK 11.2 Python API officially supports Python 3.9–3.14.

```bash
git clone <your-repository-url>
cd near-field-isac-reproduction
python -m pip install -e ".[optimization]"
```

The project does not require a virtual environment. A `.venv` is only an optional way to isolate dependencies and is already excluded by `.gitignore`.

For only the lightweight ZF/MUSIC path without the full SDR pipeline, CVXPY can be omitted:

```bash
python -m pip install -e .
```

Development/test tools are optional:

```bash
python -m pip install -e ".[optimization,dev]"
```

### Solver choice

`--solver auto` tries MOSEK, then CLARABEL, then SCS. CLARABEL is a good open-source choice for `quick`. The paper-size fully digital problem contains five large complex PSD variables; MOSEK is strongly recommended for Fig. 2 and Fig. 4, matching the public MATLAB code. CLARABEL/SCS can be substantially slower at `N=65`, and SCS may return `optimal_inaccurate`.

### Install and run MOSEK on Windows

MOSEK is not bundled with this repository. It requires both the Python package and a valid license. Run all commands below from the same Python environment used to launch `main.py`.

If the terminal prompt contains `(.venv)`, `python -m pip` installs MOSEK inside that environment. To use the global Python installation instead, leave it first:

```powershell
deactivate
where.exe python
python --version
```

Install the [MOSEK Python API](https://docs.mosek.com/latest/pythonapi/install-interface.html):

```powershell
python -m pip install Mosek
```

Request a [personal academic license](https://www.mosek.com/products/academic-licenses/) or a trial/commercial license. Store the downloaded license at the default Windows location documented by MOSEK:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\mosek"
Copy-Item "C:\path\to\downloaded\mosek.lic" "$env:USERPROFILE\mosek\mosek.lic"
```

The final path is normally `C:\Users\<username>\mosek\mosek.lic`. Never commit this file to GitHub.

Confirm that CVXPY detects the solver:

```powershell
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

The printed list must contain `MOSEK`. Then test both the installation and license with a tiny optimization:

```powershell
python -c "import cvxpy as cp; x=cp.Variable(); p=cp.Problem(cp.Minimize(x),[x>=1]); p.solve(solver='MOSEK'); print('status=',p.status,'x=',x.value)"
```

Run a small project smoke test before starting the paper-size sweep:

```powershell
python main.py figure3 --preset quick --optimizer sdr --solver MOSEK --grid-size 121
```

Run the complete paper reproduction explicitly with MOSEK:

```powershell
python main.py all --preset paper --solver MOSEK --workers 1 --solver-threads 14
```

Adjust `--solver-threads` to the machine. On a 28-logical-CPU machine, 14 is a conservative starting point. Once MOSEK and its license work, plain `python main.py` also selects it automatically through `--solver auto`.

`python main.py mosek` is not valid syntax: `all`, `figure2`, `figure3`, or `figure4` must be the subcommand, while MOSEK is selected by `--solver MOSEK`. See the [official license setup](https://docs.mosek.com/11.2/licensing/quickstart.html) if the solver is installed but reports a license error.

### CPU and parallel execution

Fig. 2 rate points and Fig. 4 distance points are independent. Use `--workers` to solve several points in separate processes:

```bash
python main.py all --preset paper --solver CLARABEL --workers 4 --solver-threads 1
```

Start with `--workers 2`, monitor RAM, then increase to 4 if memory permits. Each paper-size fully digital worker can use substantial memory. For MOSEK, prefer its internal parallelism first:

```bash
python main.py all --preset paper --solver MOSEK --workers 1 --solver-threads 14
```

Keep approximately `workers * solver_threads <= logical CPU count`. CLARABEL's default QDLDL linear solver is effectively single-threaded, so multiple workers usually help more than a large `--solver-threads` value. MUSIC evaluation uses a low-rank signal-subspace identity and is vectorized; the `500 x 500` paper grid is no longer the dominant runtime.

## Run the complete reproduction

Run all three experiments at the highest paper setting:

```bash
python main.py
```

This is equivalent to `python main.py all --preset paper` and runs:

- Fig. 2 at `R_min = 0, 1, ..., 10` for fully digital and hybrid arrays;
- Fig. 3 with fully digital SDR and a `500 x 500` MUSIC grid;
- Fig. 4 at target ranges `5, 10, ..., 40 m` for fully digital and hybrid arrays.

The full run solves many `65 x 65` complex SDPs. MOSEK is recommended; with only CLARABEL or SCS it can take a long time. Solver selection defaults to `auto`.

For a shorter end-to-end solver check:

```bash
python main.py all --preset quick --solver CLARABEL --workers 4 --solver-threads 1
```

## Execution modes, expected time, and minimum hardware

The table below covers every supported experiment mode. Times are wall-clock estimates, not guarantees. The default `quick` pipeline measured 14.8 s with one worker on the current 28-logical-CPU development machine; a longer sweep measured 44 s with one worker and 18 s with four. Paper-size times are conservative estimates because solver version, CPU memory bandwidth, RAM pressure, seed, and MOSEK license/settings can change them substantially.

| Command | What runs | Expected time | Practical minimum |
|---|---|---:|---|
| `python main.py` | Fig. 2--4, `paper`, SDR, 39 optimizations, `500 x 500` MUSIC | MOSEK: 25--90 min; CLARABEL: 1.5--6 h; SCS: 5--18 h | 8 cores, 16 GB RAM; 32 GB recommended |
| `python main.py all --preset paper ...` | Same full run, with explicit solver/parallel controls | Same as above; 2--4 workers can shorten the sweep portion | 16 GB for 1 worker; 24 GB for 2; 32 GB+ for 4 |
| `python main.py all --preset quick --solver CLARABEL` | Small end-to-end Fig. 2--4 solver validation | About 15--30 s with 1 worker | 4 cores, 8 GB RAM |
| `python main.py all --preset quick --solver CLARABEL --workers 4 --solver-threads 1` | Parallel small end-to-end validation | About 10--25 s | 8 logical CPUs, 8--12 GB RAM |
| `python main.py figure2 --preset quick --solver CLARABEL` | Three rate points, FD and HB | About 15--30 s | 4 cores, 8 GB RAM |
| `python main.py figure2 --preset paper --solver MOSEK` | Fig. 2, 11 rate points, FD and HB | MOSEK: 15--50 min; CLARABEL: 1--4 h | 8 cores, 16 GB RAM |
| `python main.py figure3 --preset quick --optimizer zf` | Small ZF/MUSIC diagnostic; no CVXPY | Under 2 s | 2 cores, 4 GB RAM |
| `python main.py figure3 --preset paper --optimizer zf --grid-size 500` | Paper grid with the non-paper ZF baseline; no CVXPY | About 2--10 s | 2 cores, 4 GB RAM |
| `python main.py figure3 --preset quick --optimizer sdr --solver CLARABEL` | Small fully digital SDR plus MUSIC | About 2--10 s | 4 cores, 8 GB RAM |
| `python main.py figure3 --preset paper --optimizer sdr --solver MOSEK --grid-size 500` | Paper Fig. 3 fully digital SDR plus MUSIC | MOSEK: 1--5 min; CLARABEL: 5--30 min | 8 cores, 16 GB RAM |
| `python main.py figure3 --preset quick --optimizer hybrid --solver CLARABEL` | Small two-stage hybrid SDR plus MUSIC | About 1--8 s | 4 cores, 8 GB RAM |
| `python main.py figure3 --preset paper --optimizer hybrid --solver MOSEK --grid-size 500` | Paper-size hybrid comparison plus MUSIC | MOSEK: 1--5 min; CLARABEL: 2--15 min | 8 cores, 12 GB RAM |
| `python main.py figure4 --preset quick --solver CLARABEL` | Three target distances, FD and HB | About 15--30 s | 4 cores, 8 GB RAM |
| `python main.py figure4 --preset paper --solver MOSEK` | Fig. 4, eight distances, FD and HB | MOSEK: 10--40 min; CLARABEL: 45 min--3 h | 8 cores, 16 GB RAM |

The minimums assume `--workers 1`. A worker is a complete solver process, so RAM use grows approximately with worker count. MOSEK rows require a working MOSEK installation and license. SCS is available as a fallback but is not recommended for publication-size runs because it is slower and may end with `optimal_inaccurate`.

All command forms accept `--output PATH`; generated PNG/CSV/NPZ/JSON files then go below that path. Without it, plots are in `results/figure2/`, `results/figure3/`, and `results/figure4/`.

## Command-line options

List all commands or the options for one command:

```bash
python main.py --help
python main.py all --help
python main.py figure3 --help
```

Common controls:

| Option | Meaning |
|---|---|
| `--preset quick|paper` | Small test setup or the full published setup |
| `--solver auto|MOSEK|CLARABEL|SCS` | Convex SDP solver |
| `--seed N` | Reproducible user locations and target reflection |
| `--output PATH` | Output root instead of `results/` |
| `--tolerance X` | Solver feasibility/optimality tolerance |
| `--max-iterations N` | Solver iteration limit |
| `--solver-threads N` | Threads inside CLARABEL/MOSEK |
| `--workers N` | Parallel Fig. 2/4 sweep processes (`all`, `figure2`, `figure4`) |
| `--verbose` | Print detailed solver logs |

Experiment-specific controls:

| Command | Options and defaults |
|---|---|
| `all` | `paper`; SDR; rates `0:1:10`; distances `5:5:40`; grid `500`; supports `--grid-size`, `--rates`, `--distances`, `--workers` |
| `figure2` | `quick`; rates `0 2 4`; supports `--rates` and `--workers` |
| `figure3` | `quick`; ZF; grid `121`; supports `--optimizer zf|sdr|hybrid` and `--grid-size` |
| `figure4` | `quick`; distances `5 20 40`; supports `--distances` and `--workers` |

## Individual and quick runs

Run the fast MUSIC/ZF path without a convex solver:

```bash
python main.py figure3 --preset quick --optimizer zf --grid-size 121
```

Check the paper's SDR on the small preset:

```bash
python main.py figure3 --preset quick --optimizer sdr --solver CLARABEL --grid-size 121
```

Generate short versions of the CRB sweeps:

```bash
python main.py figure2 --preset quick --solver CLARABEL --rates 0 2 5
python main.py figure4 --preset quick --solver CLARABEL --distances 5 20 40
```

Equivalent wrapper scripts are available under `scripts/`.

## Paper-parameter runs

The following commands use the parameters and grid sizes reported by the paper/author code:

```bash
python main.py figure3 --preset paper --optimizer sdr --solver MOSEK --grid-size 500
python main.py figure2 --preset paper --solver MOSEK --rates 0 1 2 3 4 5 6 7 8 9 10
python main.py figure4 --preset paper --solver MOSEK --distances 5 10 15 20 25 30 35 40
```

Generated plots, CSV/NPZ data, and JSON summaries are written to `results/<figure>/`. Generated artifacts are ignored by Git; `results/.gitkeep` preserves the directory.

The plotting code intentionally follows the paper's visual grammar: blue/open-circle distance curves, red/open-square angle curves, solid FD and dashed HB lines, two-column boxed legends, restrained major/minor grids, and scientific notation where appropriate. Fig. 3 uses matching 3D camera angles, fixed dB limits, the BS marker, and the true-target marker on both panels.

## Meaning of each figure

RCRB is the square root of a diagonal CRB entry. It is a theoretical lower bound on the standard deviation of an unbiased estimator, not the measured RMSE of MUSIC. Lower is better.

### Figure 2 — sensing/communication tradeoff

- Horizontal axis: minimum rate required from every communication user, in bit/s/Hz.
- Vertical axes: range RCRB in metres and angle RCRB in degrees.
- Curves: fully digital (FD) and hybrid beamforming (HB).

As `R_min` increases, more transmit degrees of freedom and power must satisfy communication constraints, leaving less freedom to shape a sensing-optimal covariance. The RCRBs should therefore generally increase. The gap between HB and FD quantifies the sensing cost of the lower-power hybrid hardware and its unit-modulus RF constraint.

### Figure 3 — what near-field sensing adds

The two surfaces are normalized MUSIC spectra over Cartesian position `(x,y)`. The true `(20 m, 45°)` target is at approximately `(14.14 m, 14.14 m)`.

- Near-field: spherical-wave curvature changes with both angle and range, so MUSIC produces a localized two-dimensional peak around the target.
- Far-field: the steering vector depends only on angle, so all points along `theta = 45°` are indistinguishable and form a ridge. Any single reported range at the ridge maximum is arbitrary.

This figure is the clearest demonstration of the paper's central claim: near-field propagation introduces an additional identifiable range dimension.

### Figure 4 — loss of range information with distance

- Horizontal axis: target distance.
- Vertical axes: range and angle RCRB.
- Target pathloss is held fixed during this sweep, so the plot isolates array geometry rather than simple SNR decay.

As the target moves farther away, its spherical wavefront becomes increasingly planar. Range-dependent phase curvature vanishes, so range RCRB rises rapidly. Angle estimation can improve or approach the far-field limit because the target direction becomes more uniform across the aperture. HB should remain worse than the fully digital bound.

## Codebase map

```text
.
├── main.py                         # full Fig. 2--4 paper run: python main.py
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
