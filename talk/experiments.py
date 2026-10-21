"""Experiment batches for the talk (see results/README or the conversation log).

Usage:  python experiments.py --batch {h,eps,iter_eps,iter_h,timing} [--smoke]

Each run is appended to results/<batch>.csv as soon as it finishes.

Batches:
    h        Convergence in h at eps = 1e-3: box (axis, diagonal), wall K=8.
    eps      Convergence in eps: diagonal box, n = 256, one continuation run
             from eps = 1 to 1e-5, one row per stage.
    iter_eps Nonlinear iterations vs eps: diagonal box n=64, wall n=40.
    iter_h   Nonlinear iterations vs h at eps = 0.03.
    timing   Wall-clock time vs N: diagonal box, n = 128 ... 1024.
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

import numpy as np

from beckmann_reg import RegularizedBeckmann, box_pair, continuation, l_scheme, lagged_diffusivity

OUT = Path(__file__).parent / "results"
TOL = 1e-6
AA = 5

# ---- Batch parameters ----
H_STUDY = dict(
    eps=1e-3,
    cases={"box_axis": [32, 64, 128, 256], "box_diag": [32, 64, 128, 256], "wall": [40, 80, 160, 320]},
    variants=["N", "F", "C"],
    darsia_max_n={"box_axis": 256, "box_diag": 256, "wall": 160},
    max_time=900,
)
EPS_STUDY = dict(case="box_diag", n=256, variants=["N", "F", "C"], eps_start=1.0, eps_end=1e-5, max_time=1800)
ITER_EPS = dict(
    cases={"box_diag": 64, "wall": 40},
    eps=[0.3, 0.1, 0.03, 0.01, 0.003],
    variants=["N", "F", "C"],
    lscheme_maxiter=200_000,
    max_time=300,
)
ITER_H = dict(eps=0.03, case="box_diag", ns=[32, 64, 128, 256], variants=["N", "F", "C"], max_time=300)
TIMING = dict(
    case="box_diag",
    ns=[128, 256, 512, 1024],
    variants=["N", "F"],
    eps=0.05,
    eps_small=0.005,
    darsia_max_n=256,
    lscheme_max_time=600,
    lagged_max_time=1200,
)
EPS_START = {"box_axis": 1.0, "box_diag": 1.0, "wall": 0.05}
CONT_CMP = dict(cases={"wall": 40, "box_diag": 64}, eps=[1e-3, 1e-4], variants=["F", "C"], max_time=300)
BALANCE = dict(
    cases={"wall": [40, 80, 160], "box_diag": [32, 64, 128]},
    variants=["F", "C"],
    eps_start=0.1,
    eps_end=1e-4,
    max_time=1200,
)
WORKPREC = dict(case="wall", ns={"F": [40, 80, 160, 320], "C": [40, 80, 160]}, eps=1e-4, cont=False, max_time=1200)


# ---- Test cases ----

def make_case(case: str, n: int):
    """Returns h, mu, gamma, kinv, W_exact on an n x n grid."""
    if case in ("box_axis", "box_diag"):
        shift = (0.25, 0.0) if case == "box_axis" else (0.25, 0.25)
        mu, gamma, W = box_pair(n, shift=shift)
        return 1.0 / n, mu, gamma, np.ones((n, n)), W
    if case == "wall":
        from wall_refinement import WALL_T, exact_distance, setup

        assert n % 8 == 0
        n_, h, mu, gamma, kinv = setup(n // 8)
        return h, mu, gamma, kinv, exact_distance(WALL_T)
    raise ValueError(case)


# ---- Runners ----

def _base_row(batch, case, n, h, variant, solver, eps, W_exact):
    return dict(batch=batch, case=case, n=n, N=n * n, h=h, variant=variant, solver=solver, eps=eps, W_exact=W_exact)


def _finish(row, hists, ok, t):
    last = hists[-1]
    W0 = last.energy0[-1]
    row.update(
        iterations=sum(hh.iterations for hh in hists),
        linear_iterations=sum(sum(hh.linear_iterations) for hh in hists),
        converged=ok,
        residual=last.residual[-1],
        W_eps=last.energy[-1],
        W_0=W0,
        rel_error=(W0 - row["W_exact"]) / row["W_exact"],
        monotone=all(np.all(np.diff(hh.energy) <= 1e-10 * abs(hh.energy[0])) for hh in hists),
        aa_rejected=sum(hh.aa_rejected for hh in hists),
        time_s=t,
        time_setup=sum(hh.time_setup for hh in hists),
        time_solve=sum(hh.time_solve for hh in hists),
    )
    return row


def run_ours(batch, case, n, variant, solver, eps, *, aa=0, cont=False, eps_start=None,
             max_time=np.inf, maxiter=None, record_every=1, per_stage=False):
    h, mu, gamma, kinv, W_exact = make_case(case, n)
    P = RegularizedBeckmann((n, n), (h, h), variant, eps=eps, kappa=1.0 / kinv)
    f = mu - gamma
    if solver == "L-scheme":
        fn = l_scheme
        kw = dict(tol=TOL, maxiter=maxiter or 200_000, record_every=record_every)
        name = "L-scheme"
    else:
        fn = lagged_diffusivity
        kw = dict(tol=TOL, maxiter=maxiter or 3000, aa_depth=aa, linear_solver="minres")
        name = "lagged" + (f"+AA{aa}" if aa else "")
    name += "+cont" if cont else ""
    tic = time.perf_counter()
    if cont:
        _, _, hists, ok = continuation(fn, P, f, eps, eps_start or EPS_START[case], 0.5, max_time=max_time, **kw)
    else:
        _, _, hist, ok = fn(P, f, max_time=max_time, **kw)
        hists = [hist]
    t = time.perf_counter() - tic
    if per_stage:
        rows, t_acc = [], 0.0
        for k, hh in enumerate(hists):
            t_acc += hh.time[-1]
            r = _base_row(batch, case, n, h, variant, name, hh.eps, W_exact)
            stage_ok = hh.residual[-1] < TOL
            rows.append(_finish(r, hists[: k + 1], stage_ok, t_acc) | dict(stage_iterations=hh.iterations))
        return rows
    return [_finish(_base_row(batch, case, n, h, variant, name, eps, W_exact), hists, ok, t)]


def run_darsia(batch, case, n):
    from wall_test import run_darsia as _run

    h, mu, gamma, kinv, W_exact = make_case(case, n)
    dist, it, ok, t = _run(n, h, mu, gamma, kinv)
    row = _base_row(batch, case, n, h, "RT/cell", "DarSIA newton", 0.0, W_exact)
    row.update(iterations=it, converged=ok, W_eps=dist, W_0=dist, rel_error=(dist - W_exact) / W_exact, time_s=t)
    return [row]


# ---- Output ----

FIELDS = [
    "batch", "case", "n", "N", "h", "variant", "solver", "eps", "iterations", "stage_iterations",
    "linear_iterations", "converged", "residual", "W_eps", "W_0", "W_exact", "rel_error", "monotone",
    "aa_rejected", "time_s", "time_setup", "time_solve", "error",
]


def append(batch: str, rows: list[dict]) -> None:
    path = OUT / f"{batch}.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
            print({k: r.get(k) for k in ("case", "n", "variant", "solver", "eps", "iterations", "converged", "rel_error", "time_s")}, flush=True)


def safe(batch: str, label: dict, fn, *args, **kwargs) -> None:
    """Run one experiment; on failure, log the error as a row and continue."""
    try:
        append(batch, fn(*args, **kwargs))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        append(batch, [dict(batch=batch, **label, error=repr(e)[:300])])


# ---- Batches ----

def batch_h(smoke=False):
    cfg = H_STUDY
    for case, ns in cfg["cases"].items():
        ns = ns[:1] if smoke else ns
        for n in ns:
            for v in cfg["variants"]:
                safe("h", dict(case=case, n=n, variant=v), run_ours, "h", case, n, v, "lagged", cfg["eps"],
                     aa=AA, cont=True, max_time=cfg["max_time"])
            if n <= cfg["darsia_max_n"][case] and not smoke:
                safe("h", dict(case=case, n=n, variant="RT/cell"), run_darsia, "h", case, n)


def batch_eps(smoke=False):
    cfg = EPS_STUDY
    n = 32 if smoke else cfg["n"]
    for v in cfg["variants"]:
        safe("eps", dict(case=cfg["case"], n=n, variant=v), run_ours, "eps", cfg["case"], n, v, "lagged",
             cfg["eps_end"] if not smoke else 1e-2, aa=AA, cont=True, eps_start=cfg["eps_start"],
             max_time=cfg["max_time"], per_stage=True)


def batch_iter_eps(smoke=False):
    cfg = ITER_EPS
    for case, n in cfg["cases"].items():
        n = 16 if smoke and case != "wall" else (40 if case == "wall" else n)
        for eps in cfg["eps"][:2] if smoke else cfg["eps"]:
            for v in cfg["variants"]:
                lab = dict(case=case, n=n, variant=v, eps=eps)
                safe("iter_eps", lab, run_ours, "iter_eps", case, n, v, "L-scheme", eps,
                     maxiter=cfg["lscheme_maxiter"], max_time=cfg["max_time"], record_every=100)
                safe("iter_eps", lab, run_ours, "iter_eps", case, n, v, "lagged", eps, max_time=cfg["max_time"])
                safe("iter_eps", lab, run_ours, "iter_eps", case, n, v, "lagged", eps, aa=AA, max_time=cfg["max_time"])
                safe("iter_eps", lab, run_ours, "iter_eps", case, n, v, "lagged", eps, aa=AA, cont=True,
                     eps_start=cfg["eps"][0], max_time=cfg["max_time"])


def batch_iter_h(smoke=False):
    cfg = ITER_H
    for n in cfg["ns"][:1] if smoke else cfg["ns"]:
        for v in cfg["variants"]:
            lab = dict(case=cfg["case"], n=n, variant=v, eps=cfg["eps"])
            safe("iter_h", lab, run_ours, "iter_h", cfg["case"], n, v, "L-scheme", cfg["eps"],
                 max_time=cfg["max_time"], record_every=100)
            safe("iter_h", lab, run_ours, "iter_h", cfg["case"], n, v, "lagged", cfg["eps"], aa=AA,
                 max_time=cfg["max_time"])


def batch_timing(smoke=False):
    cfg = TIMING
    case = cfg["case"]
    for n in [32] if smoke else cfg["ns"]:
        for v in cfg["variants"]:
            lab = dict(case=case, n=n, variant=v)
            safe("timing", lab | dict(eps=cfg["eps"]), run_ours, "timing", case, n, v, "L-scheme", cfg["eps"],
                 max_time=cfg["lscheme_max_time"], record_every=1000)
            safe("timing", lab | dict(eps=cfg["eps"]), run_ours, "timing", case, n, v, "lagged", cfg["eps"],
                 aa=AA, max_time=cfg["lagged_max_time"])
            safe("timing", lab | dict(eps=cfg["eps_small"]), run_ours, "timing", case, n, v, "lagged",
                 cfg["eps_small"], aa=AA, cont=True, eps_start=cfg["eps"], max_time=cfg["lagged_max_time"])
        if n <= cfg["darsia_max_n"] and not smoke:
            safe("timing", dict(case=case, n=n, variant="RT/cell"), run_darsia, "timing", case, n)


def batch_cont_cmp(smoke=False):
    cfg = CONT_CMP
    for case, n in cfg["cases"].items():
        n = 16 if smoke and case != "wall" else n
        for eps in cfg["eps"][:1] if smoke else cfg["eps"]:
            for v in cfg["variants"]:
                lab = dict(case=case, n=n, variant=v, eps=eps)
                for cont in (False, True):
                    safe("cont_cmp", lab, run_ours, "cont_cmp", case, n, v, "lagged", eps, aa=AA, cont=cont,
                         max_time=cfg["max_time"])


def batch_balance(smoke=False):
    cfg = BALANCE
    for case, ns in cfg["cases"].items():
        for n in ns[:1] if smoke else ns:
            for v in cfg["variants"]:
                safe("balance", dict(case=case, n=n, variant=v), run_ours, "balance", case, n, v, "lagged",
                     1e-2 if smoke else cfg["eps_end"], aa=AA, cont=True, eps_start=cfg["eps_start"],
                     max_time=cfg["max_time"], per_stage=True)


def batch_workprec(smoke=False):
    cfg = WORKPREC
    for v, ns in cfg["ns"].items():
        for n in ns[:1] if smoke else ns:
            safe("workprec", dict(case=cfg["case"], n=n, variant=v), run_ours, "workprec", cfg["case"], n, v,
                 "lagged", 1e-2 if smoke else cfg["eps"], aa=AA, cont=cfg["cont"], max_time=cfg["max_time"])


BATCHES = dict(h=batch_h, eps=batch_eps, iter_eps=batch_iter_eps, iter_h=batch_iter_h, timing=batch_timing,
               cont_cmp=batch_cont_cmp, balance=batch_balance, workprec=batch_workprec)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, choices=list(BATCHES))
    ap.add_argument("--smoke", action="store_true", help="tiny version for testing the pipeline")
    args = ap.parse_args()
    if args.smoke:
        OUT = OUT / "smoke"  # noqa: F811
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"=== batch {args.batch} start {time.strftime('%H:%M:%S')}", flush=True)
    BATCHES[args.batch](smoke=args.smoke)
    print(f"=== batch {args.batch} done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
