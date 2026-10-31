"""Runtime of one design at full size (N = 65) for the available formulations.

    python scripts/benchmark_runtime.py                  # fast vs paper (MOSEK Fusion), FD and HB, R_min = 0, 5, 10
    python scripts/benchmark_runtime.py --cvxpy          # also the plain CVXPY transcription of (20) at N = 65 (slow)

All runs use one realisation of Fig. 2 (seed 0, index --idx) and one MOSEK thread, so the
numbers are per-process times as in the Fig. 2 driver.  Results go to results/benchmark_runtime.json.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nfisac.channel import random_scenario  # noqa: E402
from nfisac.crb import near_field_model  # noqa: E402
from nfisac.optimization import (_paper_cvxpy, design_fully_digital, design_hybrid,  # noqa: E402
                                 random_analog_combiner)
from nfisac.params import SystemParams  # noqa: E402


def peak_memory_mb() -> float:
    """Peak working set of this process (Windows) or max RSS (POSIX), in MB."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2 ** 20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idx", type=int, default=0, help="realisation index of Fig. 2 (seed 0)")
    ap.add_argument("--rmin", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    ap.add_argument("--repeat", type=int, default=3, help="repetitions per case (median reported)")
    ap.add_argument("--cvxpy", action="store_true", help="also time the CVXPY transcription at N = 65 (R_min = 5)")
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark_runtime.json"))
    args = ap.parse_args()

    p = SystemParams()
    rng = np.random.default_rng([0, args.idx])
    scn = random_scenario(p, rng)
    W_RF = random_analog_combiner(p, rng)
    opts = {"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}}
    rows = []
    for R in args.rmin:
        for arch in ("FD", "HB"):
            for form in ("fast", "paper"):
                times = []
                for _ in range(args.repeat):
                    t0 = time.perf_counter()
                    d = (design_fully_digital(scn, R, formulation=form, solver_opts=opts) if arch == "FD"
                         else design_hybrid(scn, R, W_RF, formulation=form, solver_opts=opts))
                    times.append(time.perf_counter() - t0)
                n_var = p.N if arch == "FD" else p.N_RF
                if form == "fast":
                    n_var = d.sol.info.get("subspace_dim", n_var)
                row = {"R_min": R, "arch": arch, "formulation": form, "variable_size": n_var,
                       "time_s_median": float(np.median(times)), "time_s_all": times,
                       "rcrb_r_m": d.rcrb_r, "rcrb_theta_deg": d.rcrb_theta_deg,
                       "scale": d.sol.info.get("scale"), "scale_tries": len(d.sol.info.get("failed_scales", [])) + 1}
                rows.append(row)
                print(f"R_min={R:4.1f} {arch} {form:5s} ({n_var}x{n_var}): {row['time_s_median']:7.2f} s  "
                      f"RCRB = {d.rcrb_r:.6e} m, {d.rcrb_theta_deg:.4e} deg", flush=True)
    result = {"idx": args.idx, "mosek_threads": 1, "rows": rows, "peak_memory_mb": peak_memory_mb()}

    if args.cvxpy:
        R = 5.0
        fim = near_field_model(p, scn.beta_s)
        G = np.conj(scn.H) / np.sqrt(p.noise)
        gamma = 1.0 + 1.0 / (2.0 ** R - 1.0)
        print("CVXPY transcription of (20), FD, N = 65, R_min = 5 ...", flush=True)
        t0 = time.perf_counter()
        status, Rx, _, obj = _paper_cvxpy(fim, 1e2 * p.noise / p.T, G, gamma, np.eye(p.N), p.P_max, None,
                                          "MOSEK", False, opts)
        el = time.perf_counter() - t0
        result["cvxpy_full_size"] = {"R_min": R, "status": str(status), "time_s": el,
                                     "peak_memory_mb": peak_memory_mb(),
                                     "rcrb": None if Rx is None else np.sqrt(np.diag(fim.crb(Rx))).tolist()}
        print(f"  status {status}, {el:.0f} s, peak memory {peak_memory_mb():.0f} MB, "
              f"RCRB {result['cvxpy_full_size']['rcrb']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
