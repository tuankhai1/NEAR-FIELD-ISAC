# Analysis of “Near-Field Integrated Sensing and Communications” (2023)

## 1. Overview and main objective

The paper by Zhaolin Wang, Xidong Mu and Yuanwei Liu proposes a narrowband,
monostatic ISAC system. A base station with a very large uniform linear array
(ULA) sends downlink data to multiple users while sensing one target. Its key
feature is the use of spherical waves instead of a plane-wave approximation:
near-field wavefront curvature makes the steering vector depend on both range
and angle.

The three main technical contributions are:

1. An exact near-field channel model based on the distance from each antenna
   element to the user or target.
2. FIM/CRB expressions for joint range and angle estimation, treating the
   target's complex reflection coefficient as a nuisance parameter.
3. ISAC covariance and beamformer design through a globally optimal SDR for the
   fully digital architecture and a two-stage heuristic for the hybrid architecture.

Primary references: [arXiv paper](https://arxiv.org/abs/2302.01153),
[DOI](https://doi.org/10.1109/LCOMM.2023.3280132), and
[author MATLAB code](https://github.com/zhaolin820/near-field-integrated-sensing-and-communications).
For the current implementation and measured validation results, see the
[numerical method](numerical_method.md) and
[12 September 2026 audit](pipeline_audit_2026-09-12.md).

## 2. System model

The ULA has an odd number of antennas, `N = 2*N_tilde + 1`, with its center at
the origin and element `n` at `(n*d, 0)`. For a point with polar coordinates
`(r, theta)`, the distance to antenna `n` is

```math
r_n(r,\theta)=\sqrt{r^2+n^2d^2-2rnd\cos\theta}.
```

The near-field steering vector removes the common phase associated with the
distance to the array center:

```math
[a(r,\theta)]_n=\exp\left[-j\frac{2\pi}{\lambda}(r_n-r)\right].
```

When `r` is much larger than the aperture, a first-order Taylor expansion gives
the far-field steering vector

```math
[a_{far}(\theta)]_n=\exp\left(j\frac{2\pi}{\lambda}nd\cos\theta\right),
```

which no longer depends on range. This explains why near-field MUSIC has a
two-dimensional peak, while far-field MUSIC produces a ridge along a fixed angle.

The Rayleigh distance is `2*D^2/lambda`. With `D=0.5 m` and `f=28 GHz`, the
paper's value is approximately 46.7 m. The target at `(20 m, 45°)` lies within
this distance.

## 3. Communication and sensing signals

The transmitted signal at snapshot `t` is

```math
x[t]=\sum_{k=1}^{K} f_k c_k[t]+s[t],
```

where `f_k` is a communication beamformer and `s[t]` is a dedicated sensing
signal with covariance `R_s`. The total covariance is

```math
R_x=\sum_k f_k f_k^H+R_s.
```

The paper uses the convention `h_k^T f_k`, so the rate of user `k` is computed
from the SINR

```math
\mathrm{SINR}_k=
\frac{|h_k^T f_k|^2}
{h_k^T R_x h_k^* - |h_k^T f_k|^2+\sigma_k^2}.
```

For monostatic sensing, the round-trip response excluding the target coefficient is

```math
\widetilde G=a(r_s,\theta_s)a^T(r_s,\theta_s),
```

and the actual channel is `G = beta_s * G_tilde`. The received echo is
`y_s[t]=Gx[t]+z_s[t]`.

## 4. Near-field MUSIC

For a single target, the eigenvector associated with the largest eigenvalue
of the echo sample covariance spans the signal subspace. The remaining
eigenvectors form the noise subspace `E_n`. The pseudospectrum is

```math
P(r,\theta)=\frac{1}{a^H(r,\theta)E_nE_n^Ha(r,\theta)}.
```

The code scans a Cartesian grid `(x,y)`, converts each point to `(r,theta)`,
and finds the maximum. In the far-field model, all points on the same ray have
the same steering vector, so range is unidentifiable.

## 5. FIM and CRB

The unknown parameter vector is

```math
\xi=[r_s,\theta_s,\Re\{\beta_s\},\Im\{\beta_s\}]^T.
```

The FIM is partitioned as

```math
J_\xi=\begin{bmatrix}J_{11}&J_{12}\\J_{12}^T&J_{22}\end{bmatrix},
```

where `J_11` contains range and angle information, `J_22` contains reflection
coefficient information, and `J_12` describes their coupling. Eliminating the
nuisance parameters with a Schur complement gives

```math
\mathrm{CRB}_{r,\theta}=
\left(J_{11}-J_{12}J_{22}^{-1}J_{12}^T\right)^{-1}.
```

The two plotted metrics are:

- Range RCRB: `sqrt(CRB[0,0])`, in meters.
- Angle RCRB: `sqrt(CRB[1,1]) * 180/pi`, in degrees.

The derivatives `da/dr` and `da/dtheta` are implemented analytically in
`channels.py` and compared against finite differences in the tests. Range
derivatives become very small at large distances, making direct finite
differences vulnerable to cancellation errors.

## 6. Fully digital optimization

The paper minimizes `trace(CRB)` subject to three sets of constraints:

1. Each user's rate is at least `R_min,k`.
2. `trace(R_x) <= P_max`.
3. `R_x - sum_k f_k f_k^H` is positive semidefinite, ensuring a valid sensing covariance.

Define `F_k=f_k f_k^H`. Temporarily dropping the rank-one constraints produces
a convex SDP. The rate constraint becomes linear in `F_k` and `R_x`:

```math
|h_k^T f_k|^2 \ge
(2^{R_{min,k}}-1)\left(h_k^T R_xh_k^*-|h_k^Tf_k|^2+\sigma_k^2\right).
```

The inverse-matrix objective is represented through a Schur complement and an
auxiliary matrix `U`. The SDR is tight for this problem: an optimal lifted
solution allows recovery of

```math
f_k^\star=
\frac{F_k h_k^*}{\sqrt{h_k^T F_k h_k^*}}
```

without changing the objective or violating the constraints. Fully digital
optimization therefore provides an upper bound on sensing performance in the
paper's model, equivalently a lower bound on the minimized CRB objective.

The FIM has strongly differing scales across its range, angle and reflection
blocks. The Python implementation in `optimization.py` uses a congruence
preconditioner and the epigraph constraint

```math
\begin{bmatrix}U&I\\I&V\end{bmatrix}\succeq0
```

to represent `V >= U^{-1}`. This preserves the original feasible set and
objective while improving numerical conditioning. Solver-specific limitations
remain, as recorded in the current pipeline audit.

## 7. Hybrid architecture

The RF beamformer has unit-modulus entries. The paper's first stage chooses
each column as a conjugated steering vector:

- The first `K` columns focus on the `K` users.
- The remaining columns focus on the target.

Once the RF beamformer is fixed, baseband covariance and beamformers are
optimized using the same SDR in `N_RF` dimensions. The receive combiner is
sampled randomly on the unit circle. The paper uses the approximation
`(1/N) W_RF W_RF^H ≈ I`, giving effective noise `N*sigma_s^2`.

The public MATLAB code does not implement the hybrid design. This repository
reconstructs it from Eqs. (14), (15) and (22), rather than porting it line by line
from the upstream code.

## 8. Interpreting the three main figures

### Figure 2 — RCRB versus minimum rate

As `R_min` increases, the covariance must allocate more resources to
communication, leaving less freedom for sensing. The minimized total CRB
therefore cannot improve as rate constraints tighten; its individual range and
angle components need not both increase monotonically. Fully digital provides
a better optimized bound than hybrid under the corresponding model. The
slopes and values depend on the user-location realization.

### Figure 3 — MUSIC spectrum

The near-field spectrum peaks near `(x,y)≈(14.14,14.14) m`, corresponding to
`(20 m,45°)`. The far-field spectrum has a ridge along `y=x`; an individual
argmax on this ridge is not a valid range estimate.

### Figure 4 — RCRB versus distance

The paper excludes pathloss variation to examine geometric effects. As the
target moves farther away, wavefront curvature decreases and range RCRB rises
rapidly. The paper's angle-estimation curves improve toward the far-field
limit as the directions seen by different antennas become more similar. The
reproduced hybrid angle curve differs, as discussed in the
[Figure 4 analysis](figure4_discrepancies.md).

## 9. Missing information for exact reproduction

The following sources of uncertainty should be recorded in experimental reports:

- The paper does not provide a seed, the four users' location realization, or
  the number of Monte Carlo trials.
- The upstream MATLAB code does not set an RNG seed and provides only the
  fully digital MUSIC pipeline.
- Upstream user ranges are sampled uniformly from zero to the Rayleigh
  distance, although the model states a Fresnel lower bound of `1.2D`.
- The upstream path-gain convention is `rho_0=lambda/(4*pi)`, followed by
  `sqrt(rho_0)/r`. This repository retains it for comparison with the code,
  rather than substituting a different Friis convention.
- Figure 4 excludes pathloss without specifying the exact normalization.
  The baseline retains the complex target gain generated at 20 m throughout
  the distance sweep.
- Figures 2 and 4 may not match the paper point by point even when trends
  agree. Reproduction claims should include the seed, solver, tolerance,
  status, achieved rates and CSV data.

## 10. Scientific limitations of the model

The model assumes narrowband operation, one target, line-of-sight propagation,
perfect communication CSI, target location information from the previous
coherent block, approximately equal gains across the aperture, and no coupling,
quantization or hardware impairments. Extremely large arrays or wide bandwidths
can make spatial non-stationarity, beam squint, per-antenna near-field path-gain
variation and target extent significant.

The CRB is also a local lower bound for unbiased estimators. It does not
guarantee that MUSIC with finitely many snapshots attains the bound, especially
at low SNR or in the presence of ambiguity and sidelobes.

## 11. Mapping the paper to the code

| Component | File/function |
|---|---|
| Eqs. (1), (4), (5) | `channels.py`: `near_field_response`, `far_field_response` |
| Eqs. (6), (7) | `channels.py`: `generate_scenario` |
| Eq. (11) | `communication.py`: `communication_rates` |
| Eq. (13), Appendix B | `fim.py`: `fisher_information_blocks`, `crb_matrix` |
| Eqs. (20), (21) | `optimization.py`: `solve_fully_digital_sdr`, rank-one recovery |
| Eq. (22) | `optimization.py`: `hybrid_analog_beamformer` |
| Eqs. (23), (24) | `music.py`: `noise_projector`, `music_spectrum_xy` |
| Figures 2–4 | `experiments.py` and `scripts/reproduce_figure*.py` |

## 12. Possible optimization extensions

The baseline includes ZF for a comparison with lower computational cost.
Possible extensions, in suggested order, are:

1. **WMMSE/SCA:** combine `trace(CRB)` and weighted sum rate into a scalar
   objective, or retain rate constraints, while alternating receiver-weight
   and waveform updates.
2. **Riemannian hybrid optimization:** optimize RF phases directly on a product
   of complex circles instead of fixing the steering columns.
3. **Robust design:** optimize expected or worst-case performance over an
   uncertain `(r,theta)` region instead of assuming an exact target location.
4. **First-order solvers for large problems:** exploit low-rank and channel
   structure to avoid many dense `N x N` PSD cones.
5. **Multiple targets:** extend the position FIM to `2M x 2M`, accounting for
   target association and correlated echoes.

New algorithms should share the same saved scenario and seed. Before comparing
CRBs, verify the power constraint, minimum achieved rate and positive
semidefiniteness of the residual sensing covariance. The additional
[PSO/DE experiment](metaheuristics.md) follows this comparison convention.

## 13. Runtime considerations

CPU utilization below 100% does not imply a fault. SDP runtime includes
canonicalization, sparse or dense factorization, and memory synchronization.
Useful thread counts depend on the selected solver and linear algebra backend.

The rate points in Figure 2 and distances in Figure 4 are independent, and the
implementation supports multiple worker processes. The current exact
transmit-subspace reduction makes the default FD SDP at most seven dimensional,
so memory and threading behavior differ from an unreduced `N x N` formulation.

The verified MOSEK paper-preset pipeline uses:

```powershell
python main.py all --preset paper --solver MOSEK --workers 1 --solver-threads 1
```

The additional PSO/DE comparison was verified with `--workers 4` and
`--solver-threads 1`. Increasing either setting should be measured on the local
machine; excessive concurrency can increase memory use and slow the run.
As a practical starting point, keep `workers * solver_threads` within the
available logical CPU count. The current CLARABEL installation cannot complete
the full published rate grid; see the audit for its rejected high-rate cases.
With an inactive virtual environment, replace `python` in PowerShell with
`.\.venv\Scripts\python.exe`.

The MUSIC grid uses `P_noise = I - E_signal E_signal^H`, avoiding multiplication
by an `N x N` projector at every grid point. This equivalent expression reduces
single-target grid evaluation from approximately `O(N^2 G)` to `O(N G)`.
