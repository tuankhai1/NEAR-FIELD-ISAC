# Component-capped hybrid PSO/DE follow-up

This is a separate follow-up to the original `metaheuristics` experiment. The
original experiment minimizes the paper's mixed-unit trace objective, so a
reduction in range variance can legitimately increase angle variance. This
follow-up asks a different question: can the hybrid hardware improve the two
physical sensing variances together, relative to the original hybrid (HB)
design?

## Objective and caps

At each target distance, the ordinary two-stage HB solution supplies the
reference variances:

```text
D = diag(CRB_range^HB, CRB_angle^HB).
```

The nominal balanced SDR and every PSO/DE candidate solve:

```text
minimize  0.5 * (CRB_range / D_range + CRB_angle / D_angle)
subject to CRB_range <= D_range
           CRB_angle <= D_angle.
```

The CRBs are physical variances (metres squared and radians squared), not
display-unit RCRBs. The normalizations make the two terms dimensionless; the
constraints prevent either component from becoming worse than original HB.
The original fully-digital row remains a reference comparison and is not
capped by the hybrid baseline.

The inverse-FIM LMI applies this normalization directly. It is not a
post-search filter: infeasible candidates are rejected by the inner SDR, and
their reason is retained in the evaluation log. Saved waveforms are also
checked against the physical component caps before their fitness is accepted.

## Search space and budget

Each RF column begins at the original virtual focusing beam. In addition to
the range and angle offsets used by the original PSO/DE comparison, each
column receives three nonconstant cosine phase modes. A constant RF-column
phase is intentionally excluded because it is redundant with the baseband
matrix. All RF entries remain unit modulus.

PSO and DE share the same deterministic, multiscale initial population. The
zero position uses the balanced nominal HB solution; it is cached, remains an
incumbent, and proves that every returned result satisfies the two caps. Both
algorithms receive exactly:

```text
population * (generations + 1) + polish_evaluations
```

fitness requests. The final coordinate polish uses paired positive and
negative proposals from the same origin. Stalled populations are restarted
without replacing the incumbent. Repeated candidate positions use a cache,
so the saved `solver_calls` count can be below the fitness budget.

## Running

```powershell
python main.py pareto --preset paper --solver MOSEK --solver-threads 1 --workers 4
```

For a quick inspection without overwriting the default result folder, reduce
the workload and choose another output root:

```powershell
python main.py pareto --preset smoke --distances 20 --population 4 --generations 1 `
  --phase-modes 1 --polish-evaluations 2 --search-seeds 101 --solver auto `
  --tolerance 1e-7 --output results/pareto-smoke
```

The results appear in `results/pareto/` by default, including the five-method
comparison, search convergence, raw candidate logs, waveforms, component
ratios, and complete solver provenance. `maximum_component_ratio <= 1`
(within the documented numerical tolerance) is the direct per-run audit of
the constrained objective.

This remains an optimization comparison on one fixed realization, not a
channel-distribution result or proof that PSO/DE reached a global optimum.
