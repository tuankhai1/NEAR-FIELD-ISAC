# Comparing PSO + SDR and DE + SDR with the original Figure 4

This additional experiment optimizes the analog stage of the hybrid (HB) design.
Each analog candidate uses the existing SDR solver to optimize its digital stage.
The two reference designs, **Original FD** and **Original HB**, are recomputed
using the same realization as the reproduced Figure 4 results. The comparison
uses the original problem as implemented in this repository; values digitized
from the paper are not treated as outputs of this experiment.

## Model and comparison conditions

| Parameter | Value |
|---|---|
| Antennas / RF chains / users | 65 / 5 / 4 |
| Frequency / aperture | 28 GHz / 0.5 m |
| Transmit power / noise power | 20 dBm / −60 dBm |
| Snapshots | 128 |
| Minimum rate per user | 5 bit/s/Hz |
| Target distances | 5, 10, 15, 20, 25, 30, 35, 40 m |
| Target angle | 45° |
| Channel seed | 2023, matching the original results |
| Target gain | Fixed at the realization generated at 20 m |
| HB receive combiner | Fixed and shared by original HB, PSO and DE |
| HB CRB noise model | Baseline approximation `N sigma² I` |
| Objective | `CRB_range [m²] + CRB_angle [rad²]` |
| Inner solver | MOSEK, tolerance 1e-11, one thread per worker |

The power, rate and positive-semidefinite covariance constraints are unchanged.
Holding the gain fixed across distances implements the convention of excluding
pathloss variation in Figure 4. The paper describes FD SDR and the two-stage HB
design in [Section III](https://arxiv.org/html/2302.01153v5#S3).

## Optimization variables

Each RF column has the form `a*(r_focus, theta_focus)`. PSO and DE vary the
**virtual beam focusing locations**, while the physical user locations, target
location and propagation channels remain fixed. Five RF chains give ten
variables: five focusing ranges and five focusing angles.

Focusing ranges can vary by ±50% from their initial values. Angles can vary by
up to ±3°, within 0–180°. The response-vector entries have unit magnitude, so
every candidate RF matrix automatically satisfies this hardware constraint.
The search space includes the paper's design at the zero-offset vector.

This searches a family of focusing beams with ten parameters, rather than all
325 independent RF phases. A virtual focus outside the target sweep interval
does not change the physical target location being evaluated.

## Algorithms and budgets

| Setting | PSO | DE |
|---|---|---|
| Variant | Synchronous PSO with linearly decreasing inertia | DE/rand/1/bin with generational updates |
| Population | 12 | 12 |
| Generations after initialization | 20 | 20 |
| Search seeds | 101, 202, 303 | 101, 202, 303 |
| Fitness requests per restart | 252 | 252 |
| Coefficients | `w: 0.9 → 0.4`, `c1 = c2 = 1.5` | `F = 0.7`, `CR = 0.9` |
| Step limits | Maximum velocity 0.4 in normalized coordinates `[-1,1]` | Donor clipped to `[-1,1]` |

PSO follows the population mechanism of
[Kennedy and Eberhart](https://doi.org/10.1109/ICNN.1995.488968), with an inertia
schedule selected for this experiment. DE uses three distinct mutation donors,
excluding the target individual, and binomial crossover with at least one donor
coordinate, following [Storn and Price](https://doi.org/10.1023/A:1008202821328).

For each seed, both algorithms receive the same initial population, including
the original HB design. Each candidate is solved by SDR and accepted only if
it passes the baseline's physical checks. Rejected candidates receive infinite
fitness, with the reason recorded. Numerical rejection does not prove that a
candidate is mathematically infeasible.

The best solution found is always retained, so its total CRB cannot exceed that
of the initial HB design. Repeated evaluations use a cache. **Equal budgets mean
equal numbers of fitness requests**; actual SDR calls are recorded separately.
Neither algorithm receives an additional local search or polishing step.

## Running the experiment

From the repository directory:

```powershell
python main.py metaheuristics --preset paper --solver MOSEK --solver-threads 1 --workers 4
python scripts/validate_metaheuristics.py --original-figure4 results/figure4
```

These commands create a separate `results/metaheuristics/` directory. The
Figures 2–4 commands continue to work as before, and `python main.py all` still
runs the original pipeline.

To change the budget or save a separate run:

```powershell
python main.py metaheuristics --preset paper --population 20 --generations 40 --search-seeds 101 202 303 --solver-threads 1 --workers 4 --output results/larger-budget
python scripts/validate_metaheuristics.py --results results/larger-budget/metaheuristics --original-figure4 results/figure4
```

Increasing the budget creates a new experiment; its outcome is not guaranteed.
Increasing `--workers` processes distances in parallel without changing the
seeds or algorithm update rules. Recorded timings depend on concurrent work.
If the virtual environment is inactive, replace `python` with
`.\.venv\Scripts\python.exe` in these PowerShell commands.

## Reading the outputs

- `figure4_metaheuristic_comparison.png/.svg`: range and angle panels with four
  curves: Original FD, Original HB, HB + PSO and HB + DE. Both panels use common
  logarithmic axes across methods to compare physical values directly.
- `metaheuristic_convergence.png/.svg`: best total CRB divided by original HB
  CRB versus the fitness budget. Values below one indicate improvement.
- `comparison.csv`: mean and sample standard deviation across three search seeds.
- `runs.csv`: per-seed results, timings, SDR call counts and rejected evaluations.
- `distance_*m/`: best waveforms, RF/baseband matrices and candidate evaluation logs.
- `settings.json`, `provenance.json`, `scenario.npz`: complete settings and
  realization, package versions and source hashes.
- `validation.json`: independent post-run checks; `report.md`: comparison tables.
- `rendering_provenance.json`, when present: data and plotting-code hashes for
  presentation changes after simulation; original numerical provenance is retained.

Shading in the main comparison is ±1 standard deviation across **search seeds
on one fixed channel**. It is neither a confidence interval nor an average over
multiple channels. The convergence plot combines distances after normalization
by each corresponding HB baseline. Results do not select only the best seed.

Reducing total CRB does not guarantee lower range and angle RCRBs together,
because the two components have very different scales. Each component is
therefore reported separately. This finite budget establishes neither global
convergence nor consistent superiority of one algorithm across other
realizations. The magnitude discrepancy with the paper's Figure 4 remains a
separate reproduction issue, discussed in the
[Figure 4 analysis](figure4_discrepancies.md).

## Results at 20 m from the 8 September 2026 run

The PSO/DE rows below average all three seeds, including restarts with no improvement.

| Method | Range RCRB (m) | Angle RCRB (deg) | Range RCRB reduction vs HB |
|---|---:|---:|---:|
| Original FD | 0.03586696 | 0.0002403039 | — |
| Original HB | 0.1774955 | 0.0009962601 | 0% |
| HB + PSO | 0.1527089 | 0.001172328 | 13.96% |
| HB + DE | 0.1427424 | 0.001478270 | 19.58% |

PSO reduces mean total CRB by approximately 24.99%, and DE by 35.25%. However,
angle RCRB increases by 17.67% and 48.38%, respectively. Because the original
objective adds m² to rad², optimization prioritizes lower range variance while
accepting higher angle variance. These results do not establish improvement
in every sensing metric.

The complete run included 48 searches and 16 baseline solutions, taking about
438.4 seconds with four workers. All 64 waveforms passed independent checks of
rates, power, covariance, RF hardware and CRBs. The largest CRB discrepancy
against finite differences was 0.000257928%. The original FD/HB curves matched
the previous Figure 4 results with a maximum relative difference of about
3.11e-13%. All 36 tests available at the time passed.

Images and tables from that run are saved in
[`docs/results/2026-09-08/metaheuristics/`](results/2026-09-08/metaheuristics/).
This snapshot retains aggregate tables and provenance; complete waveforms and
candidate logs are in `results/metaheuristics/`. The newer
[12 September pipeline audit](pipeline_audit_2026-09-12.md) records the current
validation results and test count.

If the research objective is to reduce both errors together, a follow-up
experiment could constrain angle CRB to be no worse than original HB, or use a
weighted, normalized objective. That would define a different optimization
problem and should be reported separately from this comparison, which preserves
the paper's objective.
