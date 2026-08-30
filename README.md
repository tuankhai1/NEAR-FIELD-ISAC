# Near-Field ISAC — Python Reproduction

Python baseline for reproducing the main numerical results of:

> Z. Wang, X. Mu, and Y. Liu, “Near-Field Integrated Sensing and Communications,” *IEEE Communications Letters*, vol. 27, no. 8, pp. 2048–2052, Aug. 2023. [DOI](https://doi.org/10.1109/LCOMM.2023.3280132) · [arXiv](https://arxiv.org/abs/2302.01153) · [MATLAB code](https://github.com/zhaolin820/near-field-integrated-sensing-and-communications)

The repository implements the spherical-wave channel, range/angle CRB, 2D MUSIC, fully digital SDR, the paper's two-stage hybrid design, and a fast ZF sensing baseline. A detailed Vietnamese analysis is available in [docs/paper_analysis_vi.md](docs/paper_analysis_vi.md).

## Reproduced results

| Result | Description | Output |
|---|---|---|
| Fig. 2 | Range/angle RCRB versus minimum communication rate | `results/figure2/` |
| Fig. 3 | Near-field MUSIC peak versus far-field range ambiguity | `results/figure3/` |
| Fig. 4 | Range/angle RCRB versus target distance | `results/figure4/` |

The generated plots follow the paper's serif typography, clean grid-free axes,
colors, open markers, line styles, legends, 3D camera angles, and near-/far-field
references.

## Installation

Python 3.10 or newer is required.

```bash
git clone <your-repository-url>
cd near-field-isac-reproduction
python -m pip install -e ".[optimization]"
```

The project does not require a virtual environment. For development tools:

```bash
python -m pip install -e ".[optimization,dev]"
```

With `--solver auto`, the pipeline prefers MOSEK for the paper-size fully
digital SDP and CLARABEL for the compact hybrid SDP, with the other installed
solvers available as fallbacks. This architecture-specific policy avoids known
numerical failures in the high-rate hybrid sweep.

## Usage

Run the complete Fig. 2–4 reproduction with the highest paper preset:

```bash
python main.py
```

This is equivalent to:

```bash
python main.py all --preset paper
```

Run an accurate reduced-workload reproduction. This keeps the paper's 65
antennas, 4 users, and 5 RF chains, but uses fewer curve/grid samples and reuses
the nominal optimization across Figures 2--4:

```bash
python main.py all --preset quick --solver auto --workers 1 --solver-threads 4
```

Run a short end-to-end installation check with the intentionally tiny model:

```bash
python main.py all --preset smoke --solver CLARABEL --workers 1
```

Run one experiment:

```bash
# Fig. 2: sensing/communication tradeoff
python main.py figure2 --preset paper --solver MOSEK

# Fig. 3: paper SDR and MUSIC spectrum
python main.py figure3 --preset paper --optimizer sdr --solver MOSEK --grid-size 500

# Fig. 3: fast baseline without a convex solver
python main.py figure3 --preset paper --optimizer zf --grid-size 500

# Fig. 4: CRB versus target distance
python main.py figure4 --preset paper --solver MOSEK
```

Useful options:

| Option | Purpose |
|---|---|
| `--preset smoke|quick|paper` | Tiny validation model, reduced sampling, or full published sampling |
| `--solver auto|MOSEK|CLARABEL|SCS` | SDP solver |
| `--optimizer zf|sdr|hybrid` | Fig. 3 waveform method |
| `--rates ...` / `--distances ...` | Custom Fig. 2/4 sweep points |
| `--grid-size N` | MUSIC points per Cartesian axis |
| `--workers N` | Parallel Fig. 2/4 sweep processes |
| `--solver-threads N` | Threads used inside the solver; defaults to 4 to limit peak memory |
| `--seed N` / `--output PATH` | Random seed and output root |
| `--verbose` | Detailed solver output |

Use `python main.py <command> --help` for the complete option list.

## Runtime and hardware

Times are approximate and depend strongly on the solver, CPU, RAM, and scenario realization.

| Run | Expected time | Practical minimum |
|---|---:|---|
| Full `smoke` pipeline | 15–30 s | 4 cores, 8 GB RAM |
| Full `quick` pipeline | MOSEK: 15–50 min | 8 cores, 16 GB RAM; 32 GB recommended |
| Full `paper` pipeline | MOSEK: 25–90 min; CLARABEL: 1.5–6 h | 8 cores, 16 GB RAM; 32 GB recommended |
| Fig. 2 `paper` | MOSEK: 15–50 min; CLARABEL: 1–4 h | 8 cores, 16 GB RAM |
| Fig. 3 `paper`, ZF | 2–10 s | 2 cores, 4 GB RAM |
| Fig. 3 `paper`, SDR/hybrid | MOSEK: 1–5 min; CLARABEL: 2–30 min | 8 cores, 12–16 GB RAM |
| Fig. 4 `paper` | MOSEK: 10–40 min; CLARABEL: 45 min–3 h | 8 cores, 16 GB RAM |

Paper-size fully digital SDP can use most available RAM. Start with `--workers 1`
and `--solver-threads 4`; each additional worker launches another solver process
and can multiply memory usage. If MOSEK reports `MSK_RES_ERR_SPACE`, close
memory-heavy applications or retry with `--solver-threads 1`. With one worker,
paper-size sweep points run in disposable processes so native solver memory is
released between points.

## Output files

Each experiment saves a plot and machine-readable data:

```text
results/
├── figure2/
│   ├── figure2_rcrb_vs_rate.png
│   ├── figure2_rcrb_vs_rate.csv
│   └── figure2_summary.json
├── figure3/
│   ├── figure3_music_spectrum.png
│   ├── figure3_music_data.npz
│   └── figure3_summary.json
└── figure4/
    ├── figure4_rcrb_vs_distance.png
    ├── figure4_rcrb_vs_distance.csv
    └── figure4_summary.json
```

Generated results are ignored by Git. Use `--output PATH` to change the output root.

## Figure interpretation

RCRB is the square root of a diagonal CRB entry and represents a lower bound on estimation standard deviation. Lower is better.

### Figure 2 — sensing/communication tradeoff

Increasing the minimum user rate consumes waveform power and spatial degrees of freedom, so sensing accuracy generally degrades. The FD/HB gap shows the performance cost of hybrid hardware constraints.

### Figure 3 — near-field range information

The near-field spherical wavefront depends on both range and angle, producing a localized MUSIC peak near the target `(20 m, 45°)`. The far-field response depends only on angle and therefore forms a range-ambiguous ridge.

### Figure 4 — transition toward the far field

As target distance increases, the wavefront becomes more planar and range-dependent phase curvature disappears. Range RCRB therefore grows, while angle RCRB approaches its far-field reference. Target pathloss is held fixed to isolate this geometry effect.

## Code structure

```text
.
├── main.py                       # command-line entry point
├── src/near_field_isac/
│   ├── channels.py              # near-/far-field channel models
│   ├── communication.py         # rates and ZF baseline
│   ├── config.py                # smoke/quick/paper configurations
│   ├── fim.py                   # FIM and CRB
│   ├── music.py                 # echo simulation and MUSIC
│   ├── optimization.py          # fully digital and hybrid SDR
│   ├── experiments.py           # Fig. 2–4 pipelines and plots
│   └── cli.py                   # command-line options
├── docs/                         # paper analysis
├── tests/                        # numerical and integration tests
└── results/                      # generated artifacts
```

## Reproducibility notes

- The paper and public MATLAB code do not publish an RNG seed. Both the four
  user locations and the complex target reflection are random, so their exact
  realization materially changes the RCRB curves. This baseline uses
  `seed=2023` and records it in each JSON summary.
- The authors' public code implements fully digital Fig. 3. Fig. 2, Fig. 4, and the hybrid sweeps are reconstructed from the paper equations.
- Exact pixel-level agreement is not guaranteed; compare CSV/JSON values, solver status, achieved rates, and fixed seeds rather than only the rendered images.

## Tests

```bash
pytest -q
ruff check src tests scripts
```

## Citation

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

This Python baseline is released under the [MIT License](LICENSE). The paper and authors' MATLAB repository remain subject to their respective licenses.
