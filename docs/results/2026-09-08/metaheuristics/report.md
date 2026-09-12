# PSO/DE versus original Figure 4

Mean over independent search restarts on the same channel realization; shading in the figure is one sample standard deviation. Smaller CRBs are better.

| Distance (m) | Method | Range RCRB (m) | Angle RCRB (deg) | Trace reduction vs HB (%) | Search/solve time (s) |
|---:|---|---:|---:|---:|---:|
| 5 | fully-digital-sdr | 0.002237599 | 0.0002411313 | 95.160 | 1.51 |
| 5 | hybrid-two-stage-sdr | 0.01017103 | 0.00104458 | 0.000 | 0.14 |
| 5 | hybrid-pso-sdr | 0.009889798 | 0.001074557 | 5.301 | 39.95 |
| 5 | hybrid-de-sdr | 0.01005628 | 0.001158618 | 2.218 | 34.95 |
| 10 | fully-digital-sdr | 0.008962311 | 0.0002405133 | 95.835 | 1.63 |
| 10 | hybrid-two-stage-sdr | 0.04391749 | 0.0009963628 | 0.000 | 0.14 |
| 10 | hybrid-pso-sdr | 0.03857055 | 0.001083637 | 22.125 | 40.70 |
| 10 | hybrid-de-sdr | 0.03564805 | 0.001301581 | 34.060 | 34.74 |
| 15 | fully-digital-sdr | 0.02017206 | 0.0002403727 | 95.909 | 1.52 |
| 15 | hybrid-two-stage-sdr | 0.09972799 | 0.0009942659 | 0.000 | 0.14 |
| 15 | hybrid-pso-sdr | 0.08528549 | 0.001155978 | 25.793 | 40.83 |
| 15 | hybrid-de-sdr | 0.08070342 | 0.001221898 | 34.487 | 34.82 |
| 20 | fully-digital-sdr | 0.03586696 | 0.0002403039 | 95.917 | 1.54 |
| 20 | hybrid-two-stage-sdr | 0.1774955 | 0.0009962601 | 0.000 | 0.14 |
| 20 | hybrid-pso-sdr | 0.1527089 | 0.001172328 | 24.990 | 40.77 |
| 20 | hybrid-de-sdr | 0.1427424 | 0.00147827 | 35.254 | 34.76 |
| 25 | fully-digital-sdr | 0.05604703 | 0.0002402656 | 95.912 | 0.21 |
| 25 | hybrid-two-stage-sdr | 0.2771997 | 0.000998484 | 0.000 | 0.14 |
| 25 | hybrid-pso-sdr | 0.239261 | 0.001196556 | 24.547 | 34.36 |
| 25 | hybrid-de-sdr | 0.2256439 | 0.001283443 | 33.712 | 35.12 |
| 30 | fully-digital-sdr | 0.08071228 | 0.00024024 | 95.905 | 0.23 |
| 30 | hybrid-two-stage-sdr | 0.3988341 | 0.001000408 | 0.000 | 0.14 |
| 30 | hybrid-pso-sdr | 0.3455239 | 0.001132831 | 24.040 | 34.33 |
| 30 | hybrid-de-sdr | 0.3298975 | 0.001292866 | 31.533 | 35.17 |
| 35 | fully-digital-sdr | 0.1098627 | 0.0002402213 | 95.897 | 0.22 |
| 35 | hybrid-two-stage-sdr | 0.5423963 | 0.001002003 | 0.000 | 0.15 |
| 35 | hybrid-pso-sdr | 0.4717264 | 0.001158387 | 23.494 | 34.44 |
| 35 | hybrid-de-sdr | 0.4530198 | 0.001424486 | 30.209 | 35.10 |
| 40 | fully-digital-sdr | 0.1434984 | 0.0002402081 | 95.891 | 0.22 |
| 40 | hybrid-two-stage-sdr | 0.7078849 | 0.001003321 | 0.000 | 0.14 |
| 40 | hybrid-pso-sdr | 0.6188997 | 0.001199206 | 22.749 | 34.46 |
| 40 | hybrid-de-sdr | 0.6012281 | 0.001372421 | 27.700 | 35.06 |

Search runtimes exclude the shared HB initialization solve. FD/HB entries show a single solve. Timings reflect concurrent execution.

All methods minimize the original mixed-unit trace; neither individual RCRB is guaranteed to improve. This comparison does not resolve the original paper's absolute-scale discrepancy and is not a channel Monte Carlo study.
