"""Full pipeline: reproduce Fig. 2, Fig. 3 and Fig. 4 of the near-field ISAC letter in one run.

    python main.py                                  # all figures, paper configuration
    python main.py --solver CLARABEL --workers 4    # no MOSEK licence, 4 processes
    python main.py --figs 3                         # only Fig. 3

Every default is the paper configuration of scripts/run_fig2.py, run_fig3.py and run_fig4.py.
Options shared by all figures: --solver, --reflection, --alpha, --out, --formulation.

    python main.py --formulation fast --alpha paper # |alpha| = 3.5985 (paper levels), results in results/alpha_paper/
"""

from __future__ import annotations

import argparse
import time

# run_fig2 must be imported first: it pins BLAS to one thread per worker before numpy loads.
from scripts import run_fig2, run_fig3, run_fig4  # noqa: I001


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def build_parser() -> argparse.ArgumentParser:
    d2 = run_fig2.build_parser().parse_args([])
    d3 = run_fig3.build_parser().parse_args([])
    d4 = run_fig4.build_parser().parse_args([])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=_HelpFormatter)
    ap.add_argument("--figs", type=int, nargs="+", choices=[2, 3, 4], default=[2, 3, 4],
                    help="figures to generate")
    ap.add_argument("--solver", default=d2.solver, help="CVXPY solver (MOSEK or CLARABEL)")
    ap.add_argument("--reflection", choices=["unit", "rayleigh"], default=d2.reflection)
    ap.add_argument("--alpha", default=None,
                    help="fixed reflection coefficient: 'paper' (= 3.5985, calibrated on the paper's FD levels; "
                         "NOT reported by the authors) or a number; overrides --reflection; results go to "
                         "results/alpha_paper/ or results/alpha_<value>/ unless --out is given")
    ap.add_argument("--out", default=None, help="output directory (default results/)")
    ap.add_argument("--formulation", choices=["fast", "paper"], default=d2.formulation,
                    help="paper = problem (20) solved as written; fast = exact reformulation, ~10x faster")

    g2 = ap.add_argument_group("Fig. 2 (RCRB vs R_min)")
    g2.add_argument("--realizations", type=int, default=d2.realizations,
                    help="number of feasible realisations to average")
    g2.add_argument("--rmin-max", type=float, default=d2.rmin_max)
    g2.add_argument("--rmin-step", type=float, default=d2.rmin_step)
    g2.add_argument("--workers", type=int, default=d2.workers, help="processes (Fig. 2 and Fig. 4)")
    g2.add_argument("--objective", choices=["trace", "normalized"], default=d2.objective,
                    help="trace = tr(CRB) of the paper; normalized = suggested unit-free variant, NOT the paper "
                         "(Fig. 2 and Fig. 4)")
    g2.add_argument("--fig2-seed", type=int, default=d2.seed)

    g3 = ap.add_argument_group("Fig. 3 (MUSIC spectrum)")
    g3.add_argument("--rmin", type=float, default=d3.rmin, help="minimum rate per user [bit/s/Hz]")
    g3.add_argument("--step", type=float, default=d3.step, help="grid step [m]")
    g3.add_argument("--fig3-seed", type=int, default=d3.seed)

    g4 = ap.add_argument_group("Fig. 4 (RCRB vs target distance, R_min = --rmin)")
    g4.add_argument("--fig4-realizations", type=int, default=d4.realizations)
    g4.add_argument("--r-list", type=float, nargs="+", default=d4.r_list, help="target distances r_s [m]")
    g4.add_argument("--wrf", choices=["random", "matched"], default=d4.wrf,
                    help="HB receive combiner: random = as in the paper [15]; matched = target-matched "
                         "conjugate-symmetric random phases (variant, NOT the paper), files tagged _wrf_matched")
    g4.add_argument("--ff-benchmark", choices=["target", "system"], default=d4.ff_benchmark,
                    help="far-field lines: target = only the target far-field; system = users too, as Fig. 3(b)")
    g4.add_argument("--fig4-seed", type=int, default=d4.seed)
    return ap


def main():
    args = build_parser().parse_args()
    shared = dict(solver=args.solver, reflection=args.reflection, alpha=args.alpha, out=args.out,
                  formulation=args.formulation)
    t0 = time.time()
    if 2 in args.figs:
        print("=== Fig. 2 ===", flush=True)
        run_fig2.run(argparse.Namespace(realizations=args.realizations, rmin_max=args.rmin_max,
                                        rmin_step=args.rmin_step, seed=args.fig2_seed, workers=args.workers,
                                        objective=args.objective, **shared))
    if 3 in args.figs:
        print("\n=== Fig. 3 ===", flush=True)
        run_fig3.run(argparse.Namespace(seed=args.fig3_seed, rmin=args.rmin, step=args.step, **shared))
    if 4 in args.figs:
        print("\n=== Fig. 4 ===", flush=True)
        run_fig4.run(argparse.Namespace(realizations=args.fig4_realizations, r_list=args.r_list, rmin=args.rmin,
                                        seed=args.fig4_seed, workers=args.workers, objective=args.objective,
                                        ff_benchmark=args.ff_benchmark, wrf=args.wrf, **shared))
    print(f"\npipeline done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
