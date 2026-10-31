"""Reproduce Fig. 3: normalised MUSIC spectrum of near-field vs far-field ISAC (R_min = 5 bit/s/Hz).

    python scripts/run_fig3.py [--seed 2023] [--rmin 5] [--step 0.08] [--solver MOSEK]

Pipeline (Sec. IV-B): draw K users -> optimise the ISAC waveform (fully digital
SDR, Sec. III-B) -> transmit T = 128 snapshots -> MUSIC (Appendix A) over the
grid x, y in [0 : 0.08 : 40] m.  The far-field benchmark uses the same user
positions, far-field channels (Eq. (5)), the angle-only CRB design of ref. [7]
and far-field MUSIC steering vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nfisac.channel import build_scenario, draw_reflection, draw_users  # noqa: E402
from nfisac.music import music_spectrum, simulate_echo  # noqa: E402
from nfisac.optimization import FORMULATIONS, design_far_field, design_fully_digital  # noqa: E402
from nfisac.params import SystemParams  # noqa: E402
from nfisac.plotting import plot_fig3  # noqa: E402
from scripts.run_fig2 import output_and_reflection  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--rmin", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=0.08, help="grid step [m]")
    ap.add_argument("--reflection", choices=["unit", "rayleigh"], default="unit")
    ap.add_argument("--alpha", default=None,
                    help="fixed reflection coefficient: 'paper' (= 3.5985) or a number; overrides --reflection")
    ap.add_argument("--solver", default="MOSEK", help="CVXPY solver (MOSEK or CLARABEL)")
    ap.add_argument("--formulation", choices=list(FORMULATIONS), default="paper",
                    help="paper = problem (20) solved as written; fast = exact reformulation")
    ap.add_argument("--out", default=None, help="output directory (default results/, results/alpha_<...>/ with --alpha)")
    return ap


def run(args: argparse.Namespace) -> None:
    p = SystemParams(R_min=args.rmin)
    rng = np.random.default_rng(args.seed)
    r_u, th_u = draw_users(p, rng)
    out, reflection = output_and_reflection(args)
    alpha = draw_reflection(rng, reflection)

    grid = np.round(np.arange(0.0, 40.0 + 1e-9, args.step), 6)
    res, summary = {}, {"formulation": args.formulation, "reflection": reflection,
                        "users_r_m": r_u.tolist(), "users_theta_deg": np.rad2deg(th_u).tolist()}
    for tag, far, design in (("near", False, design_fully_digital), ("far", True, design_far_field)):
        scn = build_scenario(p, r_u, th_u, alpha, far_field=far)
        t0 = time.perf_counter()
        d = design(scn, p.R_min, solver=args.solver, formulation=args.formulation)
        t_design = time.perf_counter() - t0
        Y = simulate_echo(scn, d.Rx, d.P, np.random.default_rng(args.seed + 1))
        res[tag] = music_spectrum(scn, Y, grid, grid)
        summary[tag] = {
            "rates_bps_hz": d.rates.tolist(),
            "tx_power_mW": float(np.real(np.trace(d.Rx))),
            "rcrb": np.sqrt(np.diag(d.crb)).tolist(),
            "r_hat_m": res[tag].r_hat,
            "theta_hat_deg": float(np.rad2deg(res[tag].theta_hat)),
            "design_time_s": t_design,
            "scale": d.sol.info.get("scale"),
        }
        print(f"[{tag}-field ISAC] rates = {np.round(d.rates, 3)} bit/s/Hz, "
              f"r_hat = {res[tag].r_hat:.3f} m, theta_hat = {np.rad2deg(res[tag].theta_hat):.3f} deg")

    np.savez_compressed(out / "fig3_data.npz", x=grid, y=grid,
                        spectrum_near=res["near"].spectrum, spectrum_far=res["far"].spectrum)
    (out / "fig3_summary.json").write_text(json.dumps(summary, indent=2))
    plot_fig3(res["near"], res["far"], p, str(out / "fig3_music_spectrum"))
    print(f"saved to {out}")


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
