# Numerical method and reproduction conventions

The revision keeps the supplied paper's spherical-wave channels, complex-gain nuisance parameters, mW units, and objective `CRB_range [m²] + CRB_angle [rad²]`. It does not fit a scale factor, select a random seed to match a graph, restore historical derivative bugs, or smooth the output curves.

## Exact transmit dimension reduction

Let Q span the conjugated user channels and the right row spaces of the target matrix G and its derivatives. Replacing R by `Q Qᴴ R Q Qᴴ` leaves all sensing and communication trace forms unchanged, while its trace cannot increase. Therefore an optimal fully digital solution exists in this space. With four users and one near-field target its dimension is at most seven; without rate constraints only three sensing dimensions are needed. The RF matrix remains a separate hardware restriction in the hybrid experiment.

`reduce_dimension=False` is available on the fully digital Python solver for independent checks. Regression tests compare both formulations and independently verify projection invariance. An SVD detects redundant columns in the hybrid RF matrix; the baseband mapping uses only its actual span.

## Balanced inverse-information formulation

The mean-echo derivatives are `[beta G_r, beta G_theta, G, jG]`. A derivative component parallel to G can be absorbed into the unknown complex gain. Removing that component is an invertible nuisance-parameter reparameterization, so it leaves the range/angle Schur complement unchanged. The phase of beta can likewise be absorbed by rotating its two nuisance coordinates.

The implementation balances these derivatives using their isotropic-reference information diagonals, normalizes transmit covariance by the power budget, and represents the inverse-information objective with a single Schur-complement epigraph. The parameter selector retains the original m²/rad² objective weights exactly. The SINR inequality uses normalized user channels and the equivalent form `(1 + 1/gamma) signal >= total + noise`, with rate zero handled explicitly.

The default solver tolerance is 1e-11. MOSEK receives only the relevant conic feasibility and relative-gap tolerances, avoiding CVXPY's blanket `eps` setting. Auto mode tries MOSEK, then CLARABEL, then SCS. An inaccurate solution is physically checked and alternatives are tried. Rejected attempts are saved. If no result passes the checks, the pipeline stops.

Acceptance limits: rate margin at least -1e-5 bit/s/Hz; power overrun at most 1e-7 of the power budget; covariance eigenvalue at least -1e-7 of the power budget; inverse-epigraph relative discrepancy at most 2e-5; balanced information-LMI eigenvalue at least -1e-7. These explicit numerical tolerances do not certify global optimality of every separately reported variance. A comparison with tighter tolerances and a second solver is saved by `scripts/check_solver_convergence.py`.

The angle term is extremely small relative to range in this mixed-unit objective. In particular, the zero-rate FD angle remains more sensitive across solvers than the total objective. Only the optimized total is guaranteed to be nondecreasing when minimum rates increase; individual CRBs need not be monotonic. Both facts are reported rather than concealed through a different objective or altered curves.

## CRB evaluation and hybrid noise

Physical CRBs are recomputed from the recovered covariance. A diagonally balanced Schur-complement inverse preserves identifiable parameters with very different units. Singular, nonfinite, or indefinite information raises an explicit error rather than returning a pseudoinverse with zero variance. Invalid variances are not clipped into apparently perfect accuracy.

For hybrid CRBs, the default remains the paper's approximation `sigma² W Wᴴ ≈ N sigma² I`. Hybrid MUSIC simulates actual combined antenna noise and whitens the covariance and steering vectors together. This gives a correct MUSIC calculation for that physical receiver, but its finite-array noise model is more exact than the paper-approximate hybrid CRB. The distinction is recorded in provenance. The main Figure 3 is fully digital, so it does not use this approximation.

## Figure 4 reference and target gain

The target's complex gain generated at the nominal 20 m is held fixed over the near-field range sweep to isolate geometry. The far-field references are independently optimized **angle-only** SDPs. They use planar target steering, the same target gain, fixed near-field communication channels, and the same hybrid receive combiner. Hybrid user RF columns retain near-field focusing; the target RF column uses planar steering. The result is independent of the chosen distance-sweep endpoint.

This is an explicit reconstruction convention: the original Figure 4 implementation and its exact gain normalization are not supplied by the public author example. An angle-only optimum need not equal the angle component of a joint range/angle optimum. Nor is it a universal lower bound comparing two different statistical models or two different hybrid transmit subspaces. The figure labels describe the designs rather than claiming such a bound.

## Paper reference extraction and validation

`docs/paper_reference/` contains values extracted from vector paths in the supplied arXiv v5 PDF and calibrated against each physical axis's tick positions. These are digitized graph values, not original simulation arrays. The extraction records the PDF SHA-256 and does not modify the source PDF.

Figure 2 uses the recovered rate grid `[0,1,2,3,4,5,6,7,8,9,9.6,10,10.3,10.5,10.6,10.7]`. Figure 4 uses the eight main range samples from 5 to 40 m; the paper's angle lines include additional intermediate points, which are retained in the reference CSV but not silently interpolated into comparisons. MUSIC uses the author's 500-point `linspace(0,40,500)` on each Cartesian axis. The literal `0:0.08:40` text in the paper would instead give 501 points; the author-code grid reproduces the stated 19.952 m estimate.

Every curve point is saved as an NPZ waveform plus JSON diagnostics. Each experiment also saves the complete scenario, configuration, software versions, and source hashes. `scripts/validate_results.py` checks those hashes, independently reconstructs communication rates, power and residual covariance, recomputes CRBs using finite differences of the mean echo, and compares matching sample coordinates with the digitized paper. Absolute and first-point-normalized comparison plots distinguish amplitude and shape differences. Normalization is confined to the explicitly labeled comparison figure.

Figures are written as 300-dpi PNG and SVG. Figure 3 retains the full numerical grid when rendering and rasterizes its surface inside SVG; text and axes remain vector objects. New runs default to `results/`, with figure2, figure3, figure4 and validation subfolders. Use `--output` to keep a separate run.
