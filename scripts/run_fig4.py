"""Reproduce Fig. 4: RCRB (distance, angle) versus the target distance r_s, R_min = 5 bit/s/Hz.

    python scripts/run_fig4.py [--realizations 100] [--workers 8] [--solver MOSEK]

Sec. IV-C: the target moves along theta_s = 45 deg, r_s in [5, 40] m, "without factoring
in pathloss": beta_s keeps its value at the reference distance of Sec. IV (20 m), so
only the array geometry changes with r_s.  For every Monte-Carlo realisation (random
user positions in the near-field region and a random analog combiner W_RF):
  * near-field ISAC, FD and HB, at every r_s                          (Sec. III-B, III-C)
  * far-field ISAC [7], FD and HB: angle-only CRB of the far-field sensing model (Eq. (5)),
    same rate constraints.  Without pathloss it does not depend on r_s (horizontal lines).
    --ff-benchmark target (default): same near-field users, only the target is far-field,
    as in [7] (far-field sensing, arbitrary communication channels); this is the curve the
    near-field one tends to when r_s grows.  --ff-benchmark system: users far-field too, as
    in Fig. 3(b).
CRB matrices are averaged over the realisations that are feasible at every point and
RCRB = sqrt(mean CRB) is plotted, as for Fig. 2.
--wrf matched (variant, NOT the paper): HB receive combiner with rows (a(r_s, theta_s) .* u_i)^H,
u_i random conjugate-symmetric phases (``matched_analog_combiner``) instead of i.i.d. random phases;
the far-field HB line uses a_far(theta_s).  FD curves are unchanged.  Files tagged _wrf_matched.
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

# One BLAS thread per worker process (see run_fig2.py).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nfisac.channel import Scenario, build_scenario, draw_reflection, draw_users  # noqa: E402
from nfisac.optimization import (FORMULATIONS, InfeasibleError, conj_symmetric_phases,  # noqa: E402
                                 design_far_field, design_fully_digital, design_hybrid,
                                 matched_analog_combiner, random_analog_combiner)
from nfisac.params import SystemParams  # noqa: E402
from nfisac.plotting import plot_fig4  # noqa: E402
from scripts.run_fig2 import output_and_reflection  # noqa: E402

RATE_TOL = 1e-3   # bit/s/Hz, same slack as Fig. 2
R_S_DEFAULT = (5, 6, 7, 8, 10, 12.5, 15, 20, 25, 30, 35, 40)   # denser near the array, as the paper's markers


def _checked(label: str, fn, R_min: float):
    try:
        d = fn()
    except InfeasibleError as exc:
        raise InfeasibleError(f"{label}: {exc}") from exc
    if d.rates.min() < R_min - RATE_TOL or not np.all(np.isfinite(d.crb)) or np.any(np.diag(d.crb) <= 0):
        raise InfeasibleError(f"{label}: rate/CRB check failed")
    return d


def run_realization(job):
    """Near-field FD/HB at every r_s + far-field FD/HB of one realisation; {"infeasible": reason} on failure."""
    idx, seed, r_list, R_min, solver, reflection, weights, solver_opts, formulation, ff_benchmark, wrf = job
    p = SystemParams(R_min=R_min)
    rng = np.random.default_rng([seed, idx])
    r_u, th_u = draw_users(p, rng)
    alpha = draw_reflection(rng, reflection)
    W_RF = random_analog_combiner(p, rng)
    U = conj_symmetric_phases(p, rng)                       # drawn after users / alpha / W_RF: same realisations
    combiner = (lambda scn: W_RF) if wrf == "random" else (lambda scn: matched_analog_combiner(scn.a_s, U))
    kw = dict(solver=solver, crb_weights=weights, solver_opts=solver_opts, formulation=formulation)
    ref = build_scenario(p, r_u, th_u, alpha)               # users and beta_s at the reference r_s = 20 m
    if ff_benchmark == "system":
        ff = build_scenario(p, r_u, th_u, alpha, far_field=True)
    else:
        ff = Scenario(p, r_u, th_u, ref.H, ref.beta_s, far_field=True, far_field_users=False)
    out = {"idx": idx, "r_users": r_u, "theta_users": th_u,
           "crb_FD": [], "crb_HB": [], "rate_FD": [], "rate_HB": [], "time_FD": [], "time_HB": []}
    try:
        for r in r_list:
            scn = Scenario(p.with_(r_s=float(r)), r_u, th_u, ref.H, ref.beta_s)   # no pathloss: beta_s fixed
            for arch, fn in (("FD", lambda: design_fully_digital(scn, R_min, **kw)),
                             ("HB", lambda: design_hybrid(scn, R_min, combiner(scn), **kw))):
                t0 = time.perf_counter()
                d = _checked(f"{arch} r_s={r:g}", fn, R_min)
                out[f"crb_{arch}"].append(d.crb)
                out[f"rate_{arch}"].append(d.rates)
                out[f"time_{arch}"].append(time.perf_counter() - t0)
        out["crb_FF_FD"] = _checked("far-field FD", lambda: design_far_field(ff, R_min, **kw), R_min).crb
        out["crb_FF_HB"] = _checked("far-field HB", lambda: design_hybrid(ff, R_min, combiner(ff), **kw), R_min).crb
    except InfeasibleError as exc:
        return {"idx": idx, "infeasible": str(exc)}
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in out.items()}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realizations", type=int, default=100, help="number of feasible realisations to average")
    ap.add_argument("--r-list", type=float, nargs="+", default=list(R_S_DEFAULT), help="target distances r_s [m]")
    ap.add_argument("--rmin", type=float, default=5.0, help="minimum rate per user [bit/s/Hz]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--solver", default="MOSEK", help="CVXPY solver (MOSEK or CLARABEL)")
    ap.add_argument("--reflection", choices=["unit", "rayleigh"], default="unit")
    ap.add_argument("--alpha", default=None,
                    help="fixed reflection coefficient: 'paper' (= 3.5985) or a number; overrides --reflection")
    ap.add_argument("--objective", choices=["trace", "normalized"], default="trace",
                    help="trace = tr(CRB) of the paper (default); normalized = suggested unit-free variant, NOT the paper")
    ap.add_argument("--wrf", choices=["random", "matched"], default="random",
                    help="HB receive combiner: random = i.i.d. unit-circle phases as in the paper [15] (default); "
                         "matched = target-matched conjugate-symmetric random phases (variant, NOT the paper)")
    ap.add_argument("--ff-benchmark", choices=["target", "system"], default="target",
                    help="far-field ISAC lines: target = only the target far-field (same users); "
                         "system = users far-field too, as in Fig. 3(b)")
    ap.add_argument("--formulation", choices=list(FORMULATIONS), default="paper",
                    help="paper = problem (20) solved as written; fast = exact reformulation, ~10x faster")
    ap.add_argument("--out", default=None, help="output directory (default results/, results/alpha_<...>/ with --alpha)")
    return ap


def run(args: argparse.Namespace) -> None:
    r_list = np.asarray(sorted(args.r_list), float)
    weights = None if args.objective == "trace" else "normalized"
    tag = ("" if args.objective == "trace" else "_normalized") + ("" if args.wrf == "random" else "_wrf_matched")
    solver_opts = {"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}} if args.solver.upper() == "MOSEK" else {}
    out, reflection = output_and_reflection(args)

    results, infeasible, next_idx = [], [], 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        while len(results) < args.realizations and next_idx < 5 * args.realizations:
            n_jobs = max(args.workers, int(1.2 * (args.realizations - len(results))))
            jobs = [(i, args.seed, r_list, args.rmin, args.solver, reflection, weights, solver_opts,
                     args.formulation, args.ff_benchmark, args.wrf) for i in range(next_idx, next_idx + n_jobs)]
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
    stack =lambda key: np.stack([d[key] for d in results])
    crb_FD, crb_HB = stack("crb_FD"), stack("crb_HB")          # (M, L, 2, 2)
    crb_FF_FD, crb_FF_HB = stack("crb_FF_FD"), stack("crb_FF_HB")   # (M, 1, 1)
    rcrb_deg = lambda c: float(np.rad2deg(np.sqrt(c[:, -1, -1].mean())))
    curves = {
        "FD_r": np.sqrt(crb_FD[:, :, 0, 0].mean(0)),
        "FD_th": np.rad2deg(np.sqrt(crb_FD[:, :, 1, 1].mean(0))),
        "HB_r": np.sqrt(crb_HB[:, :, 0, 0].mean(0)),
        "HB_th": np.rad2deg(np.sqrt(crb_HB[:, :, 1, 1].mean(0))),
        "FF_FD_th": rcrb_deg(crb_FF_FD),
        "FF_HB_th": rcrb_deg(crb_FF_HB),
    }

    np.savez_compressed(out / f"fig4_data{tag}.npz", r_s=r_list, R_min=args.rmin, crb_FD=crb_FD, crb_HB=crb_HB,
                        crb_FF_FD=crb_FF_FD, crb_FF_HB=crb_FF_HB,
                        rate_FD=stack("rate_FD"), rate_HB=stack("rate_HB"),
                        r_users=stack("r_users"), theta_users=stack("theta_users"),
                        realization_idx=np.array([d["idx"] for d in results]),
                        time_FD=stack("time_FD"), time_HB=stack("time_HB"), **curves)
    with open(out / f"fig4_curves{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["r_s_m", "FD_RCRB_distance_m", "FD_RCRB_angle_deg", "HB_RCRB_distance_m", "HB_RCRB_angle_deg",
                    "FF_FD_RCRB_angle_deg", "FF_HB_RCRB_angle_deg"])
        for i, r in enumerate(r_list):
            w.writerow([f"{r:g}"] + [f"{curves[k][i]:.6e}" for k in ("FD_r", "FD_th", "HB_r", "HB_th")]
                       + [f"{curves['FF_FD_th']:.6e}", f"{curves['FF_HB_th']:.6e}"])
    (out / f"fig4_summary{tag}.json").write_text(json.dumps({
        "feasible_realizations": len(results),
        "infeasible_realizations": len(infeasible),
        "infeasible_reasons": [d["infeasible"] for d in infeasible],
        "R_min": args.rmin, "r_s": r_list.tolist(),
        "solver": args.solver, "reflection": reflection, "objective": args.objective, "seed": args.seed,
        "formulation": args.formulation, "ff_benchmark": args.ff_benchmark, "wrf": args.wrf, "workers": args.workers,
        "far_field_rcrb_angle_deg": {"FD": curves["FF_FD_th"], "HB": curves["FF_HB_th"]},
        "elapsed_s": time.time() - t0,
    }, indent=2))
    plot_fig4(r_list, curves, str(out / f"fig4_rcrb_vs_distance{tag}"),
              note=None if args.wrf == "random" else "HB: target-matched $W_{RF}$ (variant)")

    print("\n r_s (m) | FD dist (m) | FD angle (deg) | HB dist (m) | HB angle (deg)")
    for i, r in enumerate(r_list):
        print(f" {r:7.1f} | {curves['FD_r'][i]:.3e}   | {curves['FD_th'][i]:.4e}     | "
              f"{curves['HB_r'][i]:.3e}   | {curves['HB_th'][i]:.4e}")
    print(f" far-field ISAC angle: FD {curves['FF_FD_th']:.4e} deg, HB {curves['FF_HB_th']:.4e} deg")
    print(f"saved to {out}")


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
