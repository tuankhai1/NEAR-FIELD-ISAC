# Numerical method and reproduction conventions

The implementation keeps the paper's spherical-wave channels, complex-gain nuisance parameters, mW units, and objective `CRB_range [m²] + CRB_angle [rad²]`. Curves are never rescaled or smoothed. The random draws the paper does not report (users, target reflection, receive combiners) are recovered from its figures; that recovery, and the evidence for the conventions below, is described in [reproduction.md](reproduction.md).

## Sensing signal in the CRB

`SimulationConfig.crb_signal` selects the covariance whose echo enters the FIM. The default `"dedicated"` uses only the dedicated sensing signal, `R_s = R_x - sum_k p_k p_kᴴ`; `"total"` uses `R_x` as Eq. (13) is written. The optimizer, every CRB evaluation (`fim.crb_covariance`) and the independent validator follow the same setting. Only `"dedicated"` reproduces the paper's Fig. 2, in which all four curves rise by one common factor; see [reproduction.md](reproduction.md).

## Exact transmit dimension reduction

Let Q span the conjugated user channels and the right row spaces of the target matrix G and its derivatives. Replacing R by `Q Qᴴ R Q Qᴴ` leaves all sensing and communication trace forms unchanged, while its trace cannot increase. Therefore an optimal fully digital solution exists in this space. With four users and one near-field target its dimension is at most seven; without rate constraints only three sensing dimensions are needed. The RF matrix remains a separate hardware restriction in the hybrid experiment.

`reduce_dimension=False` is available on the fully digital Python solver for independent checks. Regression tests compare both formulations and independently verify projection invariance. An SVD detects redundant columns in the hybrid RF matrix; the baseband mapping uses only its actual span.

## Balanced inverse-information formulation

The mean-echo derivatives are `[beta G_r, beta G_theta, G, jG]`. A derivative component parallel to G can be absorbed into the unknown complex gain. Removing that component is an invertible nuisance-parameter reparameterization, so it leaves the range/angle Schur complement unchanged. The phase of beta can likewise be absorbed by rotating its two nuisance coordinates.

The implementation balances these derivatives using their isotropic-reference information diagonals, normalizes transmit covariance by the power budget, and represents the inverse-information objective with a single Schur-complement epigraph. The parameter selector retains the original m²/rad² objective weights exactly. The SINR inequality uses normalized user channels and the equivalent form `(1 + 1/gamma) signal >= total + noise`, with rate zero handled explicitly.

The default solver tolerance is 1e-11. MOSEK receives only the relevant conic feasibility and relative-gap tolerances, avoiding CVXPY's blanket `eps` setting. Auto mode tries MOSEK, then CLARABEL, then SCS. At 1e-11 an interior-point solver works at its accuracy limit and can stall on rounding noise alone, so each solver is retried at 1e-9 before the next one (also when a looser requested tolerance misses the acceptance limits). An inaccurate solution is physically checked and alternatives are tried. Rejected attempts are saved. If no result passes the checks, the pipeline stops.

Acceptance limits: rate margin at least -1e-5 bit/s/Hz; power overrun at most 1e-7 of the power budget; covariance eigenvalue at least -1e-7 of the power budget; inverse-epigraph relative discrepancy at most 2e-5; balanced information-LMI eigenvalue at least -1e-7. These explicit numerical tolerances do not certify global optimality of every separately reported variance.

## Lexicographic angle refinement

The angle variance (rad²) is about 1e-8 of the range variance (m²), below interior-point accuracy, so the trace objective cannot resolve the angle: the solver returned an arbitrary point of the range-optimal face. At zero rate it gave a rank-2 covariance with the optimal range CRB but a 20 % worse angle CRB than target focusing, and the value changed between solvers. `solve_sdr` therefore runs a second stage by default: it caps the range CRB at `(1 + 1e-6)` and the angle CRB at `1` times their first-stage values, and minimizes the mean of the two normalized CRBs. Because the range cannot fall materially below its optimum, this effectively minimizes the angle on the range-optimal face, the limit of the trace optimum as the angle weight tends to zero. It succeeded at all 56 near-field points of the paper run. At zero rate it recovers rank-one target focusing, whose RCRB range/angle ratio (150.61 m/deg) equals the paper's. MOSEK at 1e-11 and 1e-12 and CLARABEL now agree on that angle. If the second stage fails or does not improve the angle, the first-stage solution is kept and the reason is recorded in `angle_refinement`.

Only the optimized total is guaranteed to be nondecreasing when minimum rates increase; individual CRBs need not be monotonic.

## CRB evaluation and hybrid noise

Physical CRBs are recomputed from the recovered covariance. A diagonally balanced Schur-complement inverse preserves identifiable parameters with very different units. Singular, nonfinite, or indefinite information raises an explicit error rather than returning a pseudoinverse with zero variance. Invalid variances are not clipped into apparently perfect accuracy.

For hybrid CRBs, the paper's approximation `sigma² W Wᴴ ≈ N sigma² I` is used. Figure 3 uses the fully digital receiver.

## Figure 4 pathloss and far-field reference

Section IV-C studies distance "without factoring in pathloss". The target gain generated at the nominal 20 m is held fixed over the sweep, and the user channels carry no pathloss (`generate_scenario(..., user_pathloss=False)`). The paper's 20 m FD values equal its zero-rate Fig. 2 values to 3e-5, so the 5 bit/s/Hz constraint costs no sensing power in its Fig. 4. With pathloss on the users it would cost 0.7 %.

The far-field references are **angle-only** SDPs with planar target steering, the same users, target gain and hybrid receive combiner; the hybrid target RF column uses planar steering. They reduce to planar-wave focusing; the FD reference sits 0.0071 % below the 40 m near-field value (paper 0.0073 %). For HB the gap depends on the random combiner: 2.5 % here versus 0.34 % in the paper.

## Paper reference extraction and validation

`docs/paper_reference/` contains values extracted from vector paths in the supplied arXiv v5 PDF and calibrated against each physical axis's tick positions. These are digitized graph values, not original simulation arrays. The extraction records the PDF SHA-256 and does not modify the source PDF.

Figure 2 uses the recovered rate grid `[0,1,2,3,4,5,6,7,8,9,9.6,10,10.3,10.5,10.6,10.7]`. Figure 4 uses the paper's twelve marker positions `[5,6,7,8,10,12,15,20,25,30,35,40]` m; the range panel marks only the 5 m grid, as in the paper. MUSIC uses the author's 500-point `linspace(0,40,500)` on each Cartesian axis. The literal `0:0.08:40` text in the paper would instead give 501 points; the author-code grid reproduces the stated 19.952 m estimate.

Every curve point is saved as an NPZ waveform plus JSON diagnostics. Each experiment also saves the complete scenario, configuration, software versions, and source hashes. `scripts/validate_results.py` checks those hashes, independently reconstructs communication rates, power and residual covariance, recomputes CRBs using finite differences of the mean echo, and compares matching sample coordinates with the digitized paper. Absolute and first-point-normalized comparison plots distinguish amplitude and shape differences. Normalization is confined to the explicitly labeled comparison figure.

Figures are written as 300-dpi PNG and SVG. Figure 3 retains the full numerical grid when rendering and rasterizes its surface inside SVG; text and axes remain vector objects. New runs default to `results/`, with figure2, figure3, figure4 and validation subfolders. Use `--output` to keep a separate run.
