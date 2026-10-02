# How the paper's figures are reproduced

Audit of 2 October 2026 against the authors'
[MATLAB code](https://github.com/zhaolin820/near-field-integrated-sensing-and-communications)
and the digitized curves in [`paper_reference/`](paper_reference/). It replaces
the earlier Figure 4 discrepancy notes. The channel, FIM and rate code was
already a faithful port of the authors' `beamfocusing.m`, `FIM.m`,
`generate_channel.m` and `rate_calculator.m`. The remaining differences came
from three numerical/setup issues and from unreported random draws.

## Result

| Curve | Largest deviation from the digitized paper |
|---|---:|
| Fig. 2, FD and HB, range and angle (64 points) | 1.22 % (at 10 bit/s/Hz) |
| Fig. 4, FD range and angle | 0.05 % |
| Fig. 4, HB range | 3.4 % (at 5 m; ≤ 1.8 % elsewhere) |
| Fig. 4, HB angle | 2.1 % (at 40 m) |
| Fig. 4, FD far-field reference | 0.05 % |
| Fig. 3, near-field MUSIC estimate | 19.952 m, 45°, identical |

Before this audit the FD curves were 6.9 times too large, the zero-rate FD
angle depended on the solver, and the HB angle in Fig. 4 rose with distance.

## Issues fixed in the code

**1. The trace objective could not resolve the angle CRB.** The objective adds
`CRB_range [m²]` to `CRB_angle [rad²]`; the angle term is ~1e-8 of the total,
below interior-point accuracy. At zero rate the solver returned a rank-2
covariance with the optimal range CRB but a 20 % worse angle CRB than the
rank-one beam focused on the target. A second, lexicographic stage now
minimizes the angle CRB with the range CRB held at its optimum (within 1e-6).
At zero rate it returns target focusing, whose range/angle RCRB ratio,
150.614 m/deg, equals the paper's 150.61 for both figures.

**2. Fig. 2 requires the CRB of the dedicated sensing signal.** In the paper,
all four Fig. 2 curves (FD/HB × range/angle) rise with `R_min` by the *same*
factor, to 1e-5:

| R_min (bit/s/Hz) | FD range | FD angle | HB range | HB angle |
|---:|---:|---:|---:|---:|
| 8 | 1.06946 | 1.06947 | 1.06947 | 1.06947 |
| 10 | 1.46536 | 1.46536 | 1.46537 | 1.46536 |
| 10.5 | 2.07246 | 2.07245 | 2.07247 | 2.07245 |

With the FIM of the whole transmit signal `R_x` (Eq. (13) as written), the
optimum reshapes the sensing beam once communication beams exist. The angle
RCRB then grew up to 24 times faster than the range RCRB, for every user
draw tested, including users nearly orthogonal to the target. When only the
dedicated sensing signal `R_s = R_x - sum_k p_k p_kᴴ` enters the FIM,
communication power simply leaves the focused sensing beam. All four factors
then agree to 1e-4 for any user draw, as in the paper. This is the default
(`crb_signal="dedicated"`); `--crb-signal total` restores the written Eq. (13).

**3. Fig. 4 excludes pathloss for the users too.** The paper's FD values at
20 m in Fig. 4 equal its *zero-rate* Fig. 2 values (5.19857e-3 vs 5.19851e-3
m; 3.45151e-5 vs 3.45150e-5 deg), although Fig. 4 uses 5 bit/s/Hz. The rate
constraint therefore cost no sensing power. With pathloss on the users it
costs 0.7 %. Fig. 4 now keeps the 20 m target gain fixed and drops pathloss
from the user channels. With that, the FD curves match the paper's shape to
three to four digits: range grows as `r^2` (×4.0003 from 20 to 40 m, paper
×4.0002), the angle falls 0.449 % from 5 to 40 m (paper 0.453 %), and the
far-field reference sits 0.0071 % below the 40 m value (paper 0.0073 %). The
far-field references previously sat 1 % below.

Two smaller fixes: each solver is retried at tolerance 1e-9 when 1e-11 stalls
on rounding noise, and Fig. 4 uses the paper's twelve marker distances.

## Random draws recovered from the figures

The paper reports neither its user locations, its target reflection
coefficient, nor its random hybrid combiner, and the MATLAB code sets no seed.
These are recovered in [`realization.py`](../src/near_field_isac/realization.py)
and selected with `--realization paper` (the default for the paper and quick
presets). Each draw comes from the model's own distribution; only the choice
among draws is informed by the figures.

| Quantity | Value | Evidence |
|---|---|---|
| Reflection magnitude | 3.59852 | FD range and angle in Figs. 2 and 4 give 3.59851 ± 0.00003 |
| Users (seed 1522) | 33.2, 23.1, 33.3, 3.6 m at 59°, 103°, 86°, 88° | Fig. 2 rise up to 10.7 bit/s/Hz |
| Fig. 2 combiner (seed 1429895) | uniform random phases | HB/FD = 3.303 (range), 5.938 (angle) |
| Fig. 4 combiner (seed 6067369) | uniform random phases | HB/FD shape over distance |

*Reflection magnitude.* At zero rate both FD and HB optimize to target
focusing, so the RCRBs scale exactly as `1/|beta_s|`. The value is 16.8 dB
above the seed-2023 draw (0.5227) used before. A unit-power Rayleigh draw this
large is unlikely (probability ~2e-6). The paper's run therefore probably used
a different gain normalization (for example one-way pathloss with
`|beta| ≈ 1.8`). The CRBs cannot distinguish those cases. The paper's MUSIC
spectrum floor, about -70 dB against -51 dB at the old gain, independently
confirms a higher echo SNR.

*Users.* Under issue 2, the users only set the communication power, which is
`(2^R - 1) × 0.052 mW` in the paper. Candidates were screened with a
zero-forcing estimate and confirmed with the SDR. The remaining 1.2 % residual
is a systematic shape difference, and no draw among ~1,600 did better.

*Combiners.* At zero rate cost the hybrid optimum is also target focusing, so
the HB/FD ratios depend only on the combiner. Across random combiners they
range from 2.9 to 7.7 (range) and 2.5 to 9.7 (angle); the paper's values are
typical. Figs. 2 and 4 must have used different draws: their HB range at
20 m and 5 bit/s/Hz differs by 38 % while FD agrees. The Fig. 4 combiner was
the best of 6.4 million draws; its HB curves stay within 3.4 %. The paper's HB
near-field angle approaches its far-field line more closely (0.34 % at 40 m
vs 2.5 % here), which no draw reproduced exactly.

## Running an independent draw

`--realization random` draws users, reflection and combiners from `--seed`, as
the authors' `generate_channel.m` does. Trends are the same but the scales are
not: the FD level follows `|beta|`, the high-rate rise follows the users and
the HB/FD gap follows the combiner.
