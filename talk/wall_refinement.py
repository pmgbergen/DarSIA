"""Refinement study for the impenetrable-wall case against the exact solution.

Same geometry as wall_test.py, but the wall has fixed physical thickness WALL_T
(kappa^{-1} = K / WALL_T inside), so the exact value is h-independent. For K
large enough that no mass crosses the wall (K = 8 suffices: crossing costs
>= 2 - t + K, going around <= ~7.3), all mass goes around the wall corners and

    W = 4 I + 4 t,   I = int_1^3 int_4^5 sqrt((4 - t/2 - x)^2 + (7 - y)^2) dy dx

(upper half of the source box over the top, lower half under; the cost splits
into source and target parts, so the plan within each half does not matter).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from scipy.integrate import dblquad

from beckmann_reg import RegularizedBeckmann, continuation, lagged_diffusivity

# ---- Parameters ----
FACTORS = [5, 10, 20]  # cells per unit length; n = 8 * FACTOR (t/2 must be a multiple of h)
WALL_L = 6.0
WALL_T = 0.4
WALL_K = 8.0
VARIANTS = ["N", "F", "C"]
EPS = 5e-4
EPS_START = 0.05
AA_DEPTH = 5
TOL = 1e-6
MAXITER = 1000
RUN_DARSIA = True
OUT = Path(__file__).parent / "results"


def exact_distance(t: float) -> float:
    I, _ = dblquad(lambda y, x: np.hypot(4 - t / 2 - x, 7 - y), 1, 3, 4, 5, epsabs=1e-12, epsrel=1e-12)
    return 4 * I + 4 * t


def setup(factor: int):
    n = 8 * factor
    h = 1.0 / factor
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")  # axis 0 = x, axis 1 = y

    def box(cx, cy):
        return ((np.abs(X - cx) < 1) & (np.abs(Y - cy) < 1)).astype(float)

    kinv = np.ones((n, n))
    kinv[(np.abs(X - 4) < WALL_T / 2) & (np.abs(Y - 4) < WALL_L / 2)] = WALL_K / WALL_T
    return n, h, box(2, 4), box(6, 4), kinv


def main():
    from wall_test import run_darsia

    OUT.mkdir(exist_ok=True)
    W_exact = exact_distance(WALL_T)
    print(f"W_exact = {W_exact:.8f}")
    rows = []
    for factor in FACTORS:
        n, h, mu, gamma, kinv = setup(factor)
        for v in VARIANTS:
            P = RegularizedBeckmann((n, n), (h, h), v, eps=EPS, kappa=1.0 / kinv)
            tic = time.perf_counter()
            p, phi, hists, ok = continuation(
                lagged_diffusivity, P, mu - gamma, EPS, EPS_START, 0.5,
                tol=TOL, maxiter=MAXITER, aa_depth=AA_DEPTH,
            )
            W0 = hists[-1].energy0[-1]
            row = dict(
                method=f"lagged+AA{AA_DEPTH}+cont", variant=v, n=n, h=h, eps=EPS,
                iterations=sum(len(hh.residual) - 1 for hh in hists), converged=ok,
                W=W0, W_exact=W_exact, rel_error=(W0 - W_exact) / W_exact,
                time_s=time.perf_counter() - tic,
            )
            rows.append(row)
            print(row, flush=True)
        if RUN_DARSIA:
            dist, it, ok, t = run_darsia(n, h, mu, gamma, kinv)
            row = dict(
                method="DarSIA newton", variant="RT/cell", n=n, h=h, eps=0.0, iterations=it,
                converged=ok, W=dist, W_exact=W_exact, rel_error=(dist - W_exact) / W_exact, time_s=t,
            )
            rows.append(row)
            print(row, flush=True)

    with open(OUT / "wall_refinement.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
