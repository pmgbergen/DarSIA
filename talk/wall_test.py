"""Heterogeneous wall test (cf. examples/wasserstein_trygve.py::wall_test).

Domain [0, 8]^2, unit-density 2x2 boxes centred at (2, 4) and (6, 4). A vertical
wall of two cells at x = 4, height WALL_L, with kappa^{-1} = 0.5 K / h inside, so
crossing it costs K per unit mass. K = 0 means no wall (W = 16 exactly).

Runs the L-scheme for each energy variant and, as a cross-check, DarSIA's
"newton" (lagged diffusivity) solver with its default discretization.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from beckmann_reg import RegularizedBeckmann, continuation, l_scheme, lagged_diffusivity

# ---- Parameters ----
FACTOR = 5  # cells per unit length; n = 8 * FACTOR
WALL_L = 6.0
WALL_K = [0.0, 1.0, 3.0, 8.0]
VARIANTS = ["N", "F", "C"]
TOL = 1e-6
# (solver, eps, aa_depth, use_continuation); continuation starts at EPS_START
CONFIGS = [
    ("L-scheme", 0.05, 0, False),
    ("lagged", 0.05, 0, False),
    ("lagged", 0.05, 5, False),
    ("lagged", 0.005, 5, True),
    ("lagged", 0.0005, 5, True),
]
EPS_START = 0.05
MAXITER = {"L-scheme": 500_000, "lagged": 1000}
RUN_DARSIA = True
OUT = Path(__file__).parent / "results"
TAG = "lagged"


def setup(factor: int, K: float, wall_L: float):
    n = 8 * factor
    h = 1.0 / factor
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")  # axis 0 = x, axis 1 = y

    def box(cx, cy):
        return ((np.abs(X - cx) < 1) & (np.abs(Y - cy) < 1)).astype(float)

    mu, gamma = box(2, 4), box(6, 4)
    kinv = np.ones((n, n))
    if K > 0:
        kinv[(np.abs(X - 4) <= h) & (np.abs(Y - 4) <= wall_L / 2)] = 0.5 * K / h
    return n, h, mu, gamma, kinv


def run_darsia(n, h, mu, gamma, kinv):
    import darsia

    def img(a):
        return darsia.Image(a, width=n * h, height=n * h, scalar=True, dim=2, series=False)

    options = {
        "num_iter": 500,
        "tol_distance": 1e-8,
        "tol_increment": 1e-6,
        "tol_residual": 1e5,
        "regularization": 1e-16,
        "return_info": True,
    }
    tic = time.perf_counter()
    dist, info = darsia.wasserstein_distance(
        img(mu), img(gamma), weight=img(kinv), method="newton", options=options
    )
    return dist, info["number_iterations"], info["converged"], time.perf_counter() - tic


def main():
    OUT.mkdir(exist_ok=True)
    rows, fluxes = [], {}
    for K in WALL_K:
        n, h, mu, gamma, kinv = setup(FACTOR, K, WALL_L)
        for solver, eps, aa, cont in CONFIGS:
            for v in VARIANTS:
                P = RegularizedBeckmann((n, n), (h, h), v, eps=eps, kappa=1.0 / kinv)
                fn = l_scheme if solver == "L-scheme" else lagged_diffusivity
                kw = dict(tol=TOL, maxiter=MAXITER[solver])
                if solver == "lagged":
                    kw["aa_depth"] = aa
                tic = time.perf_counter()
                if cont:
                    p, phi, hists, ok = continuation(fn, P, mu - gamma, eps, EPS_START, 0.5, **kw)
                    hist = hists[-1]
                    # lagged history includes the initial state as entry 0
                    iters = sum(len(hh.residual) - (solver == "lagged") for hh in hists)
                    monotone = all(np.all(np.diff(hh.energy) <= 1e-12 * abs(hh.energy[0])) for hh in hists)
                else:
                    p, phi, hist, ok = fn(P, mu - gamma, **kw)
                    iters = len(hist.residual) - (solver == "lagged")
                    monotone = bool(np.all(np.diff(hist.energy) <= 1e-12 * abs(hist.energy[0])))
                t = time.perf_counter() - tic
                name = solver + (f"+AA{aa}" if aa else "") + ("+cont" if cont else "")
                row = dict(
                    solver=name, variant=v, K=K, n=n, eps=eps, iterations=iters,
                    converged=ok, W_eps=hist.energy[-1], W_0=hist.energy0[-1],
                    residual=hist.residual[-1], monotone=monotone, time_s=t,
                )
                rows.append(row)
                fluxes[(K, v, name, eps)] = np.linalg.norm(P.cell_flux(p), axis=-1).reshape(n, n)
                print(row, flush=True)
        if RUN_DARSIA:
            dist, it, ok, t = run_darsia(n, h, mu, gamma, kinv)
            row = dict(
                solver="DarSIA newton", variant="RT/cell", K=K, n=n, eps=0.0, iterations=it,
                converged=ok, W_eps=dist, W_0=dist, residual=np.nan, monotone=None, time_s=t,
            )
            rows.append(row)
            print(row, flush=True)

    with open(OUT / f"wall_test_{TAG}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(
        OUT / f"wall_test_flux_{TAG}.npz",
        **{f"K{K}_{v}_{name}_eps{eps}": a for (K, v, name, eps), a in fluxes.items()},
    )


if __name__ == "__main__":
    main()
