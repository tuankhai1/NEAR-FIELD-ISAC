"""Reproduce Fig. 2: RCRB (distance, angle) versus the minimum communication rate, FD vs HB.

    python scripts/run_fig2.py [--realizations 100] [--workers 8] [--solver MOSEK]

For every Monte-Carlo realisation (random user positions in the near-field
region and a random analog combiner W_RF) and every R_min in 0:1:10 bit/s/Hz:
  * FD: SDR of problem (20) + rank-one reconstruction (21)            (Sec. III-B)
  * HB: analog beamformer (22) + SDR of the digital stage              (Sec. III-C)
The CRB matrices are averaged over the realisations that are feasible for all
R_min of both architectures and RCRB = sqrt(mean CRB) is plotted (Sec. IV-A).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# One BLAS thread per worker process (parallelism is across realisations); without this
# every process spawns one OpenBLAS thread per core and can run out of memory.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nfisac.channel import alpha_value, random_scenario  # noqa: E402
from nfisac.optimization import (FORMULATIONS, InfeasibleError, design_fully_digital,  # noqa: E402
                                 design_hybrid, random_analog_combiner)
from nfisac.params import SystemParams  # noqa: E402
from nfisac.plotting import plot_fig2  # noqa: E402

RATE_TOL = 1e-3   # bit/s/Hz, accepted numerical slack on the rate constraints


def run_realization(job):
    """Solve FD and HB for all R_min of one realisation; {"infeasible": reason} if any point fails."""
    idx, seed, R_list, solver, reflection, weights, solver_opts, formulation = job
    p = SystemParams()
    rng = np.random.default_rng([seed, idx])
    scn = random_scenario(p, rng, reflection_mode=reflection)
    W_RF = random_analog_combiner(p, rng)
    out = {"idx": idx, "r_users": scn.r_users, "theta_users": scn.theta_users,
           "crb_FD": [], "crb_HB": [], "rate_FD": [], "rate_HB": [], "power_FD": [], "power_HB": [],
           "time_FD": [], "time_HB": [], "scale_FD": [], "scale_HB": []}
    kw = dict(solver=solver, crb_weights=weights, solver_opts=solver_opts, formulation=formulation)
    try:
        for R in R_list:
            for arch, fn in (("FD", lambda: design_fully_digital(scn, R, **kw)),
                             ("HB", lambda: design_hybrid(scn, R, W_RF, **kw))):
                t0 = time.perf_counter()
                try:
                    d = fn()
                except InfeasibleError as exc:
                    raise InfeasibleError(f"{arch} R_min={R:g}: {exc}") from exc
                if d.rates.min() < R - RATE_TOL or not np.all(np.isfinite(d.crb)) or np.any(np.diag(d.crb) <= 0):
                    raise InfeasibleError(f"{arch} R_min={R:g}: rate/CRB check failed")
                out[f"crb_{arch}"].append(d.crb)
                out[f"rate_{arch}"].append(d.rates)
                out[f"power_{arch}"].append(float(np.real(np.trace(d.Rx))))
                out[f"time_{arch}"].append(time.perf_counter() - t0)        # model build + solve(s) + checks
                out[f"scale_{arch}"].append(float(d.sol.info.get("scale", np.nan)))
    except InfeasibleError as exc:
        return {"idx": idx, "infeasible": str(exc)}
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in out.items()}


def output_and_reflection(args: argparse.Namespace) -> tuple[Path, object]:
    """Results folder and reflection mode: with --alpha the reflection coefficient is fixed and, unless
    --out is given, the results go to results/alpha_paper/ or results/alpha_<value>/."""
    alpha = getattr(args, "alpha", None)
    reflection = args.reflection if alpha is None else alpha_value(alpha)
    if args.out is not None:
        out = Path(args.out)
    elif alpha is None:
        out = ROOT / "results"
    else:
        out = ROOT / "results" / ("alpha_paper" if str(alpha).lower() == "paper" else f"alpha_{float(alpha):g}")
    out.mkdir(parents=True, exist_ok=True)
    return out, reflection


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realizations", type=int, default=100, help="number of feasible realisations to average")
    ap.add_argument("--rmin-max", type=float, default=10.0)
    ap.add_argument("--rmin-step", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--solver", default="MOSEK", help="CVXPY solver (MOSEK or CLARABEL)")
    ap.add_argument("--reflection", choices=["unit", "rayleigh"], default="unit")
    ap.add_argument("--alpha", default=None,
                    help="fixed reflection coefficient: 'paper' (= 3.5985, calibrated on the paper's FD levels; "
                         "NOT reported by the authors) or a number; overrides --reflection")
    ap.add_argument("--objective", choices=["trace", "normalized"], default="trace",
                    help="trace = tr(CRB) of the paper (default); normalized = sum_i CRB_ii / CRB_iso,ii "
                         "(suggested unit-free variant, NOT the paper)")
    ap.add_argument("--formulation", choices=list(FORMULATIONS), default="paper",
                    help="paper = problem (20) solved as written (full N x N, authors' scaling); "
                         "fast = exact reformulation (subspace reduction, own scaling, whitened hybrid), ~10x faster")
    ap.add_argument("--out", default=None, help="output directory (default results/, results/alpha_<...>/ with --alpha)")
    return ap


def run(args: argparse.Namespace) -> None:
    R_list = np.round(np.arange(0.0, args.rmin_max + 1e-9, args.rmin_step), 6)
    weights = None if args.objective == "trace" else "normalized"
    tag = "" if args.objective == "trace" else "_normalized"
    solver_opts = {"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}} if args.solver.upper() == "MOSEK" else {}
    out, reflection = output_and_reflection(args)

    results, infeasible, next_idx = [], [], 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        while len(results) < args.realizations and next_idx < 5 * args.realizations:
            n_jobs = max(args.workers, int(1.2 * (args.realizations - len(results))))
            jobs = [(i, args.seed, R_list, args.solver, reflection, weights, solver_opts, args.formulation)
                    for i in range(next_idx, next_idx + n_jobs)]
            next_idx += n_jobs
            for res in ex.map(run_realization, jobs):
                if "infeasible" in res:
                    infeasible.append(res)
                elif len(results) < args.realizations:
                    results.append(res)
            print(f"  feasible {len(results)}/{args.realizations}, infeasible {len(infeasible)}, "
                  f"{time.time() - t0:.0f} s", flush=True)

    if not results:
        raise SystemExit(f"no feasible realisation out of {len(infeasible)} (solver {args.solver}); "
                         f"first failure: {infeasible[0]['infeasible']}")
    results.sort(key=lambda d: d["idx"])
    crb_FD = np.stack([d["crb_FD"] for d in results])      # (M, L, 2, 2)
    crb_HB = np.stack([d["crb_HB"] for d in results])
    curves = {
        "FD_r": np.sqrt(crb_FD[:, :, 0, 0].mean(0)),
        "FD_th": np.rad2deg(np.sqrt(crb_FD[:, :, 1, 1].mean(0))),
        "HB_r": np.sqrt(crb_HB[:, :, 0, 0].mean(0)),
        "HB_th": np.rad2deg(np.sqrt(crb_HB[:, :, 1, 1].mean(0))),
    }

    np.savez_compressed(out / f"fig2_data{tag}.npz", R_min=R_list, crb_FD=crb_FD, crb_HB=crb_HB,
                        rate_FD=np.stack([d["rate_FD"] for d in results]),
                        rate_HB=np.stack([d["rate_HB"] for d in results]),
                        r_users=np.stack([d["r_users"] for d in results]),
                        theta_users=np.stack([d["theta_users"] for d in results]),
                        realization_idx=np.array([d["idx"] for d in results]),
                        time_FD=np.stack([d["time_FD"] for d in results]),
                        time_HB=np.stack([d["time_HB"] for d in results]),
                        scale_FD=np.stack([d["scale_FD"] for d in results]),
                        scale_HB=np.stack([d["scale_HB"] for d in results]), **curves)
    with open(out / f"fig2_curves{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["R_min_bps_hz", "FD_RCRB_distance_m", "FD_RCRB_angle_deg", "HB_RCRB_distance_m", "HB_RCRB_angle_deg"])
        for i, R in enumerate(R_list):
            w.writerow([R] + [f"{curves[k][i]:.6e}" for k in ("FD_r", "FD_th", "HB_r", "HB_th")])
    (out / f"fig2_summary{tag}.json").write_text(json.dumps({
        "feasible_realizations": len(results),
        "infeasible_realizations": len(infeasible),
        "infeasible_reasons": [d["infeasible"] for d in infeasible],
        "solver": args.solver, "reflection": reflection, "objective": args.objective, "seed": args.seed,
        "formulation": args.formulation, "workers": args.workers,
        "time_per_design_s": {arch: {"median": float(np.median(t)), "mean": float(np.mean(t)),
                                     "max": float(np.max(t)), "total": float(np.sum(t))}
                              for arch, t in (("FD", [d["time_FD"] for d in results]),
                                              ("HB", [d["time_HB"] for d in results]))},
        "elapsed_s": time.time() - t0,
    }, indent=2))
    plot_fig2(R_list, curves, str(out / f"fig2_rcrb_vs_rmin{tag}"))

    print("\n R_min | FD dist (m) | FD angle (deg) | HB dist (m) | HB angle (deg)")
    for i, R in enumerate(R_list):
        print(f" {R:5.1f} | {curves['FD_r'][i]:.3e}   | {curves['FD_th'][i]:.3e}      | "
              f"{curves['HB_r'][i]:.3e}   | {curves['HB_th'][i]:.3e}")
    print(f"saved to {out}")


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
