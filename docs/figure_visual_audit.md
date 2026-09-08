# Figure legends and axes: review on 8 September 2026

Reference: the user's supplied Figures 2–4, consistent with the stored
arXiv v5 digitization. The figures did not fully match the paper's labeling
and legend layout before this review.

| Figure | Finding | Result |
|---|---|---|
| 2 | Axis titles and units already matched. The two-column legend grouped FD/HB differently. | Reordered it to distance on the left and angle on the right, with FD above HB. Blue circles, red squares, solid FD and dashed HB match the reference. |
| 3 | Both panels already had `x (m)`, `y (m)`, `Spectrum (dB)`, the BS/target legend, `(20 m, 45°)`, and near/far-field captions. | Verified the rendered figure; no label change was needed. |
| 4, upper | `RCRB (m)` and the FD/HB legend were present, but `Distance, r (m)` and distance tick numbers were absent. Shared x-axes hid the upper tick labels. | Added the title, explicitly enabled upper tick labels, and used 5 m tick intervals on both panels. |
| 4, lower | All three axis titles were correct. The legend sat outside the panel and the FD far-field line used dash-dot styling. | Moved the four-entry legend inside the upper-right corner and made both far-field lines dotted, matching the supplied reference. Adjusted panel spacing to fit the restored upper title. |

Fig. 4's bottom labels are `RCRB, FD (deg)` on the left and `RCRB, HB (deg)` on
the right. The top is logarithmic; the bottom uses two independent linear axes
with scientific multipliers. These conventions match the paper. Numerical
limits and exponents follow the actual simulated values, which differ from the
paper. Matching the paper's limits would clip data or give misleading units.

The refreshed PNG and SVG files are in `results/figure2/`, `results/figure4/`
and the README gallery at `docs/results/2026-09-08/`. These are new renders of
the saved simulation rows. The solver was not rerun. Original numerical
provenance remains intact; `visual_review.json` in the gallery records the
rendering source and input hashes separately.

Verification: inspected both refreshed PNGs, checked visible upper distance
ticks and titles, checked that the lower legend covers no curve, compared
plotted coordinates with saved rows, and passed all 27 existing tests and
the plotting module's Ruff checks. All 16 Figure 4 points also passed a fresh
independent finite-difference CRB comparison (maximum relative error
0.000257928%).

See [why Figure 4 differs](figure4_discrepancies.md) for measured offsets,
source-code checks, and the remaining reproduction uncertainty.
