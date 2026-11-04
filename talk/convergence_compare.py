"""Convergence of the nonlinear solvers on one problem (figures for discussion).

Compares the explicit L-scheme (projected gradient descent), the paper's
semi-implicit L-scheme (draft eq. 27) for several L, and lagged diffusivity
(= semi-implicit L-scheme with L = 0).

Usage: python convergence_compare.py [--n 32] [--eps 1e-2] [--max-time 120] [--png]
Data: results/convergence_compare.npz; figures: results/figures/conv_*.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beckmann_reg import RegularizedBeckmann, box_pair, l_scheme, l_scheme_implicit, lagged_diffusivity
from plot_results import DARK_RED, FULL, GRAY, HALF, save

TOL = 1e-6

# Semi-implicit family: one hue, darker = smaller L; L = 0 in gray.
METHODS = [
    ("explicit", dict(color=DARK_RED, marker="s", linestyle="-", label=r"explicit L-scheme, $L=L_{\mathrm{Lip}}$")),
    ("implicit_1", dict(color="#E8A15C", marker="v", linestyle="--", label=r"semi-implicit, $L=L_{\mathrm{Lip}}$")),
    ("implicit_0.1", dict(color="#CC6600", marker="^", linestyle="--", label=r"semi-implicit, $L=L_{\mathrm{Lip}}/10$")),
    ("implicit_0.01", dict(color="#8A4500", marker="D", linestyle="--", label=r"semi-implicit, $L=L_{\mathrm{Lip}}/100$")),
    ("implicit_0", dict(color=GRAY, marker="o", linestyle="-", label=r"$L=0$ (lagged diffusivity)")),
]


def run(n: int, eps: float, max_time: float) -> dict[str, np.ndarray]:
    mu, gamma, _ = box_pair(n)
    f = (mu - gamma).ravel()
    pr = RegularizedBeckmann((n, n), (1 / n, 1 / n), "F", eps=eps)
    Lip = pr.lipschitz()

    # Reference energy from a tightly converged lagged-diffusivity run
    _, _, h_ref, _ = lagged_diffusivity(pr, f, tol=1e-11, maxiter=2000)
    data = {"J_ref": np.array(h_ref.energy[-1]), "Lip": np.array(Lip)}

    every = 100
    _, _, h, ok = l_scheme(pr, f, tol=TOL, maxiter=2_000_000, max_time=max_time, record_every=every)
    it = np.arange(len(h.residual)) * every
    it[-1] = h.iterations
    data |= _pack("explicit", h, it, ok)
    print(f"explicit: it={h.iterations} conv={ok} t={h.time[-1]:.1f}s")

    for key in ("1", "0.1", "0.01", "0"):
        L = float(key) * Lip
        _, _, h, ok = l_scheme_implicit(pr, f, L=L, tol=TOL, maxiter=100_000, max_time=max_time)
        data |= _pack(f"implicit_{key}", h, np.arange(len(h.residual)), ok)
        print(f"implicit L={L:.3g}: it={h.iterations} conv={ok} t={h.time[-1]:.1f}s")
    return data


def _pack(name, h, it, ok) -> dict[str, np.ndarray]:
    return {
        f"{name}_it": it,
        f"{name}_res": np.array(h.residual),
        f"{name}_J": np.array(h.energy),
        f"{name}_t": np.array(h.time),
        f"{name}_conv": np.array(ok),
    }


def plot(d, out: Path, png: bool, n: int, eps: float) -> None:
    J_ref = float(d["J_ref"])

    def series(ax, x_key, y_fn):
        for name, st in METHODS:
            x = d[f"{name}_{x_key}"] + (1 if x_key == "it" else 0)
            y = y_fn(name)
            ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                    markevery=0.12, linewidth=1.6, label=st["label"])
            if not bool(d[f"{name}_conv"]):
                ax.plot(x[-1], y[-1], marker="x", color=st["color"], markersize=8, mew=2)

    res = lambda name: d[f"{name}_res"]
    gap = lambda name: np.maximum(d[f"{name}_J"] - J_ref, 1e-16) / J_ref

    # Residual vs iteration and vs time, side by side
    fig, axs = plt.subplots(1, 2, figsize=FULL, sharey=True)
    series(axs[0], "it", res)
    series(axs[1], "t", res)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(TOL, color="#9A9A9A", linewidth=1.0, linestyle=":")
    axs[0].set_xlabel("iteration + 1")
    axs[1].set_xlabel("wall-clock time [s]")
    axs[0].set_ylabel("relative KKT residual")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.06),
               ncol=3, fontsize=7)
    save(fig, out, "conv_residual", png)

    # Energy gap vs iteration (monotone decrease)
    fig, ax = plt.subplots(figsize=HALF)
    series(ax, "it", gap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iteration + 1")
    ax.set_ylabel(r"$(J_\varepsilon(q^k) - J_\varepsilon^*)\,/\,J_\varepsilon^*$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=7)
    save(fig, out, "conv_energy", png)

    print(f"n={n}, eps={eps}, L_Lip={float(d['Lip']):.3g}; x = stopped at time limit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--max-time", type=float, default=120.0)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()

    base = Path(__file__).parent / "results"
    data_file = base / "convergence_compare.npz"
    if a.plot_only:
        data = dict(np.load(data_file))
    else:
        data = run(a.n, a.eps, a.max_time)
        np.savez(data_file, **data)
    plot(data, base / "figures", a.png, a.n, a.eps)
