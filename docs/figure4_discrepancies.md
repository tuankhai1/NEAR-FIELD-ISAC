# Why Figure 4 differs from the paper

Comparison on 8 September 2026 uses the saved seed-2023 paper-preset run and
vector-digitized coordinates from the supplied PDF. The disagreement is real,
not a rotation or axis-label issue. Its complete cause is still unresolved.

## Scale and shape are different questions

At 20 m:

| Design | Quantity | Our result | Paper plot | Ratio |
|---|---|---:|---:|---:|
| Fully digital | Range RCRB (m) | 0.03586696 | 0.005198570 | 6.899 |
| Fully digital | Angle RCRB (deg) | 0.0002403039 | 0.00003451512 | 6.962 |
| Hybrid | Range RCRB (m) | 0.1774955 | 0.02386508 | 7.437 |
| Hybrid | Angle RCRB (deg) | 0.0009962601 | 0.0002038824 | 4.886 |

Across 5–40 m, the fully digital range ratio stays between 6.896 and 6.901.
Both curves therefore have almost the same distance dependence, with a large
scale offset. Fully digital angle has a similar roughly 6.96-fold offset.
Hybrid range ratios vary from about 5.27 to 7.47, so a common scalar correction
cannot explain all four curves.

## What can affect the scale

For fixed transmit covariance and geometry, eliminating the unknown complex-gain
nuisance parameters gives information proportional to `T |beta|^2 / sigma^2`.
Consequently, RCRB scales as `sigma / (sqrt(T) |beta|)`. A different sensing gain,
noise convention or snapshot normalization can shift both range and angle curves.
The near-constant fully digital offset is consistent with such a difference;
it does not identify which parameter or convention caused it.

Our realized reflection magnitude is 0.522666. Setting it to one would reduce
fixed-covariance RCRBs to about 52.3% of their current values, an improvement of
only 1.91 times. It would not explain a 6.9-fold offset. We therefore have not
changed the reflection to force a match. The paper's exact realization and full
Figure 4 sweep implementation are unavailable in the supplied materials.

Section IV-C of the paper explicitly excludes pathloss variation from this sweep.
Our implementation holds the nominal 20 m target gain fixed over the sweep to
follow that intention. Restoring distance-dependent pathloss would change the
experiment, rather than establish the missing historical normalization.

## Why the hybrid angle shape differs

Hybrid RF columns focus on the realized communication users and the target.
Moving the target changes this restricted transmit space; the random receive
combiner also affects sensing information. The exact original realization is
needed for a strict comparison. These are plausible sources of shape differences,
not experimentally established explanations of the entire mismatch.

The optimizer minimizes `CRB_range [m^2] + CRB_angle [rad^2]`. Range dominates this
mixed-unit objective. Optimizing that sum does not require its angle component to
decrease with distance. Thus our small hybrid angle upturn is not, by itself,
evidence of an incorrect solver. The independently calculated far-field lines
use an angle-only planar model, while the near-field problem estimates both
parameters and uses a different hybrid transmit space. Our reconstruction does
not establish the universal lower-bound relationship suggested by the paper's
discussion. See [numerical conventions](numerical_method.md).

Hybrid MUSIC whitening does not explain Figure 4: it applies to MUSIC data
processing, not these CRB calculations. The hybrid CRB still uses the paper's
`N sigma^2 I` combined-noise approximation.

## What validation establishes

All 48 curve points passed independent communication-rate, power, covariance and
finite-difference CRB checks. The largest CRB discrepancy was 0.000257928%.
This supports internal numerical consistency with the implemented model, but it
does not certify the missing author settings or uniqueness of separately reported
angle variances. Cross-solver checks still show sensitivity at zero communication
rate; Figure 4 uses 5 bit/s/Hz, so that zero-rate issue alone cannot explain it.

Matching these historical curves remains an open reproduction issue. Dividing
all outputs by a fitted constant would hide the scale difference and still leave
the hybrid shape mismatch.
