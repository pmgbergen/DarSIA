"""Figures for the talk from results/*.csv (vector PDF, no titles, deck colours).

Usage: python plot_results.py [--dir results] [--png]

Sizes: FULL = full-width 16:9 Beamer frame, HALF = half-width column.
Every series has colour + marker (+ line style) so identity never relies on
colour alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Style ----
DARK_RED, MID, ORANGE, GRAY = "#741B1B", "#A86A6A", "#CC6600", "#6B6B6B"
VARIANT_STYLE = {
    "N": dict(color=MID, marker="s", label="(N) normal only"),
    "F": dict(color=ORANGE, marker="o", label="(F) face reconstr."),
    "C": dict(color=DARK_RED, marker="^", label="(C) cell reconstr."),
    "RT/cell": dict(color=GRAY, marker="D", label="DarSIA (RT, cell)"),
}
SOLVER_STYLE = {
    "L-scheme": dict(color=DARK_RED, marker="s", linestyle="-", label="L-scheme"),
    "lagged": dict(color=MID, marker="o", linestyle="-", label="lagged diff."),
    "lagged+AA5": dict(color=ORANGE, marker="^", linestyle="-", label="lagged + Anderson"),
    "lagged+AA5+cont": dict(color=ORANGE, marker="v", linestyle="--", label="lagged + AA + contin."),
    "DarSIA newton": dict(color=GRAY, marker="D", linestyle=":", label="DarSIA"),
}
FULL = (6.3, 3.0)
HALF = (3.1, 2.7)
REF = dict(color="#9A9A9A", linewidth=1.0, linestyle=":")

mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E3E3E3",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load(d: Path, name: str) -> pd.DataFrame | None:
    path = d / f"{name}.csv"
    if not path.exists():
        print(f"skip {name}: no data")
        return None
    df = pd.read_csv(path)
    if "error" in df:
        bad = df["error"].notna()
        if bad.any():
            print(f"{name}: {bad.sum()} failed runs ignored")
        df = df[~bad]
    return df


def save(fig, out: Path, name: str, png: bool) -> None:
    for ax in fig.axes:
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    fig.savefig(out / f"{name}.pdf")
    if png:
        fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {name}")


def ref_slope(ax, x, y0, power, label, logfactor=False, anchor=None):
    """Reference line with slope `power` on log-log axes through (anchor, y0);
    anchor defaults to min(x). logfactor multiplies by |ln x| (needs x < 1)."""
    x = np.asarray(sorted(x), dtype=float)
    a = x[0] if anchor is None else anchor
    y = y0 * (x / a) ** power
    if logfactor:
        y *= np.abs(np.log(x)) / abs(np.log(a))
    ax.plot(x, y, **REF)
    ax.annotate(label, (x[-1], y[-1]), textcoords="offset points", xytext=(3, 0), fontsize=8, color=REF["color"],
                va="center")


def observed_rates(x, e):
    x, e = np.asarray(x, float), np.abs(np.asarray(e, float))
    return np.log(e[1:] / e[:-1]) / np.log(x[1:] / x[:-1])


# ---- Figures ----

def fig_h_convergence(df, out, png):
    """Experiment 1: rel. error vs h, solid = axis shift, dashed = diagonal."""
    fig, ax = plt.subplots(figsize=HALF)
    lines = []
    for case, ls in [("box_axis", "-"), ("box_diag", "--")]:
        for v, st in VARIANT_STYLE.items():
            d = df[(df.case == case) & (df.variant == v)].sort_values("h")
            if d.empty:
                continue
            ax.loglog(d.h, d.rel_error.abs(), linestyle=ls, color=st["color"], marker=st["marker"],
                      markerfacecolor="white" if ls == "--" else st["color"])
            capped = ~d.converged.astype(bool)
            ax.loglog(d.h[capped], d.rel_error.abs()[capped], linestyle="none", marker=st["marker"], markersize=9,
                      markerfacecolor="white", markeredgecolor=st["color"])
            rates = observed_rates(d.h, d.rel_error)
            lines.append(f"{case:9s} {v:7s} " + " ".join(f"{r:5.2f}" for r in rates))
    hs = df[df.case.isin(["box_axis", "box_diag"])].h
    if len(hs):
        ref_slope(ax, sorted(hs.unique()), 0.5 * df[df.case == "box_diag"].rel_error.abs().min(), 1, r"$h$")
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"$|W_h - W|/W$")
    handles = [mpl.lines.Line2D([], [], color=st["color"], marker=st["marker"], label=st["label"])
               for v, st in VARIANT_STYLE.items() if v in set(df.variant)]
    handles += [mpl.lines.Line2D([], [], color="k", linestyle="-", label="axis shift"),
                mpl.lines.Line2D([], [], color="k", linestyle="--", marker="o", markerfacecolor="white",
                                 label="diagonal shift")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out, "exp1_h_convergence", png)
    return lines


def fig_wall_convergence(df, out, png):
    d0 = df[df.case == "wall"]
    if d0.empty:
        return []
    fig, ax = plt.subplots(figsize=HALF)
    lines = []
    for v, st in VARIANT_STYLE.items():
        d = d0[d0.variant == v].sort_values("h")
        if d.empty:
            continue
        ax.loglog(d.h, d.rel_error.abs(), color=st["color"], marker=st["marker"], label=st["label"])
        capped = ~d.converged.astype(bool)
        ax.loglog(d.h[capped], d.rel_error.abs()[capped], linestyle="none", marker=st["marker"], markersize=9,
                  markerfacecolor="white", markeredgecolor=st["color"])
        lines.append(f"wall      {v:7s} " + " ".join(f"{r:5.2f}" for r in observed_rates(d.h, d.rel_error)))
    ref_slope(ax, sorted(d0.h.unique()), 0.5 * d0.rel_error.abs().min(), 1, r"$h$")
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"$|W_h - W|/W$")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out, "exp1_wall_h_convergence", png)
    return lines


def fig_eps_convergence(df, out, png):
    """Error vs eps at fixed h: vs exact (dashed) and vs smallest-eps value (solid)."""
    fig, ax = plt.subplots(figsize=HALF)
    lines = []
    for v, st in VARIANT_STYLE.items():
        d = df[(df.variant == v) & df.converged.astype(bool)].sort_values("eps")
        if len(d) < 3:
            continue
        W_ref = d.W_0.iloc[0]  # smallest eps
        dd = d.iloc[1:]
        err_eps = (dd.W_0 - W_ref).abs() / d.W_exact.iloc[0]
        ax.loglog(dd.eps, err_eps, color=st["color"], marker=st["marker"], label=st["label"])
        ax.loglog(d.eps, d.rel_error.abs(), color=st["color"], linestyle="--", linewidth=1.2, alpha=0.8)
        lines.append(f"eps {v}: W(eps_min={d.eps.iloc[0]:.1e}) = {W_ref:.6f}, exact {d.W_exact.iloc[0]:.6f}")
    epss = [e for e in sorted(df.eps.unique())[1:] if e <= 0.1]
    if len(epss) > 1 and ax.get_lines():
        y0 = ax.get_lines()[0].get_ydata()[0]  # first solid line, smallest eps > eps_min
        ref_slope(ax, epss, 0.5 * y0, 1, r"$\varepsilon|\ln\varepsilon|$", logfactor=True)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"relative error")
    handles = ax.get_legend_handles_labels()[0]
    handles += [mpl.lines.Line2D([], [], color="k", label=r"$|W_{h,\varepsilon}-W_{h,\varepsilon_{\min}}|/W$"),
                mpl.lines.Line2D([], [], color="k", linestyle="--", linewidth=1.2, label=r"$|W_{h,\varepsilon}-W|/W$")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out, "exp1_eps_convergence", png)
    return lines


def fig_iter_eps(df, out, png):
    """Experiment 2a: iterations vs eps, one panel per variant, per case."""
    lines = []
    for case in df.case.unique():
        dc = df[df.case == case]
        variants = [v for v in ["N", "F", "C"] if v in set(dc.variant)]
        fig, axs = plt.subplots(1, len(variants), figsize=FULL, sharey=True, squeeze=False)
        for ax, v in zip(axs[0], variants):
            for s, st in SOLVER_STYLE.items():
                d = dc[(dc.variant == v) & (dc.solver == s)].sort_values("eps")
                if d.empty:
                    continue
                conv = d.converged.astype(bool)
                ax.loglog(d.eps, d.iterations, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                          label=st["label"])
                # Hollow markers: hit the iteration/time cap (lower bound)
                ax.loglog(d.eps[~conv], d.iterations[~conv], linestyle="none", marker=st["marker"],
                          markersize=9, markerfacecolor="white", markeredgecolor=st["color"])
                if s == "L-scheme" and conv.sum() >= 2:
                    dd = d[conv]
                    slope = np.polyfit(np.log(dd.eps), np.log(dd.iterations), 1)[0]
                    lines.append(f"{case} {v}: L-scheme observed slope eps^{slope:.2f}")
            Ls = dc[(dc.variant == v) & (dc.solver == "L-scheme")].sort_values("eps", ascending=False)
            if not Ls.empty:
                ref_slope(ax, dc.eps.unique(), Ls.iterations.iloc[0], -2, r"$\varepsilon^{-2}$",
                          anchor=Ls.eps.iloc[0])
            ax.set_xlabel(r"$\varepsilon$")
            ax.text(0.03, 0.03, VARIANT_STYLE[v]["label"], transform=ax.transAxes, fontsize=9)
        axs[0, 0].set_ylabel("nonlinear iterations")
        axs[0, -1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        save(fig, out, f"exp2a_iterations_vs_eps_{case}", png)
    return lines


def fig_iter_h(df, out, png):
    fig, ax = plt.subplots(figsize=HALF)
    for s in ["L-scheme", "lagged+AA5"]:
        for v, st in VARIANT_STYLE.items():
            d = df[(df.solver == s) & (df.variant == v)].sort_values("n")
            if d.empty:
                continue
            ax.loglog(d.N, d.iterations, color=st["color"], marker=st["marker"],
                      linestyle=SOLVER_STYLE[s]["linestyle"] if s == "L-scheme" else "--",
                      markerfacecolor=st["color"] if s == "L-scheme" else "white")
            conv = d.converged.astype(bool)
            ax.loglog(d.N[~conv], d.iterations[~conv], linestyle="none", marker="x", color="k")
    ax.set_xlabel("number of cells $N$")
    ax.set_ylabel("nonlinear iterations")
    handles = [mpl.lines.Line2D([], [], color=st["color"], marker=st["marker"], label=st["label"])
               for v, st in VARIANT_STYLE.items() if v in set(df.variant)]
    handles += [mpl.lines.Line2D([], [], color="k", label="L-scheme"),
                mpl.lines.Line2D([], [], color="k", linestyle="--", marker="o", markerfacecolor="white",
                                 label="lagged + Anderson")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out, "exp2_iterations_vs_h", png)


def fig_timing(df, out, png):
    """Experiment 2b: wall-clock vs N."""
    fig, ax = plt.subplots(figsize=FULL)
    for s, st in SOLVER_STYLE.items():
        for v, vls in [("N", ":"), ("F", "-"), ("RT/cell", "-")]:
            for eps in sorted(df[(df.solver == s) & (df.variant == v)].eps.unique()):
                d = df[(df.solver == s) & (df.variant == v) & (df.eps == eps)].sort_values("N")
                if d.empty:
                    continue
                conv = d.converged.astype(bool)
                lab = f"{st['label']}, ({v})" + (f", ε={eps:g}" if eps > 0 else "")
                ax.loglog(d.N, d.time_s, color=st["color"], marker=st["marker"],
                          linestyle=vls if s != "DarSIA newton" else ":", label=lab)
                ax.loglog(d.N[~conv], d.time_s[~conv], linestyle="none", marker=st["marker"], markersize=9,
                          markerfacecolor="white", markeredgecolor=st["color"])
    Ns = sorted(df.N.unique())
    if len(Ns) > 1:
        t0 = df[df.N == Ns[0]].time_s.min()
        ref_slope(ax, Ns, t0, 1, "$N$")
        x = np.array(Ns, float)
        ax.plot(x, t0 * x * np.log(x) / (x[0] * np.log(x[0])) * 3, **REF)
        ax.annotate(r"$N\log N$", (x[-1], 3 * t0 * x[-1] * np.log(x[-1]) / (x[0] * np.log(x[0]))),
                    textcoords="offset points", xytext=(3, 0), fontsize=8, color=REF["color"], va="center")
    ax.set_xlabel("number of cells $N$")
    ax.set_ylabel("wall-clock time [s]")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    save(fig, out, "exp2b_time_vs_N", png)


N_STYLE = [dict(color=MID, marker="s"), dict(color=ORANGE, marker="o"), dict(color=DARK_RED, marker="^")]


def balanced_eps(eps, err):
    """eps* where the eps-part equals the h-part: err(eps*) = 2 * plateau,
    plateau = error at the smallest eps (estimate of the h-error).
    Log-log interpolation; None if not bracketed."""
    eps, err = np.asarray(eps, float), np.abs(np.asarray(err, float))
    order = np.argsort(eps)
    eps, err = eps[order], err[order]
    if len(err) < 2 or err[1] > 1.3 * err[0]:
        return None  # no plateau at the smallest eps: h-error not resolved
    target = 2 * err[0]
    for i in range(1, len(eps)):
        if err[i - 1] <= target <= err[i]:
            t = (np.log(target) - np.log(err[i - 1])) / (np.log(err[i]) - np.log(err[i - 1]))
            return float(np.exp(np.log(eps[i - 1]) + t * (np.log(eps[i]) - np.log(eps[i - 1]))))
    return None


def fig_balance(df, out, png):
    """Error vs eps for each n: the plateau is the h-error, the slope the eps-error."""
    lines = []
    for case in df.case.unique():
        dc = df[df.case == case]
        variants = [v for v in ["F", "C"] if v in set(dc.variant)]
        fig, axs = plt.subplots(1, len(variants), figsize=FULL, sharey=True, squeeze=False)
        for ax, v in zip(axs[0], variants):
            for st, n in zip(N_STYLE, sorted(dc.n.unique())):
                d = dc[(dc.variant == v) & (dc.n == n)].sort_values("eps")
                if d.empty:
                    continue
                conv = d.converged.astype(bool)
                ax.loglog(d.eps, d.rel_error.abs(), color=st["color"], marker=st["marker"], label=f"$n = {n}$")
                ax.loglog(d.eps[~conv], d.rel_error.abs()[~conv], linestyle="none", marker=st["marker"],
                          markersize=9, markerfacecolor="white", markeredgecolor=st["color"])
                dd = d[conv]
                es = balanced_eps(dd.eps, dd.rel_error) if len(dd) > 2 else None
                plateau = abs(dd.rel_error.iloc[0]) if len(dd) else np.nan
                lines.append(f"balance {case} {v} n={n}: error at eps_min={dd.eps.min():.1e}: {plateau:.2e}; "
                             f"eps* (eps-error = h-error) = {es:.1e}" if es else
                             f"balance {case} {v} n={n}: error at eps_min: {plateau:.2e}; no plateau (eps-limited down to eps_min)")
                if es:
                    ax.axvline(es, color=st["color"], linewidth=0.8, linestyle=":")
            ax.set_xlabel(r"$\varepsilon$")
            ax.text(0.03, 0.95, VARIANT_STYLE[v]["label"], transform=ax.transAxes, fontsize=9, va="top")
        axs[0, 0].set_ylabel(r"$|W_{h,\varepsilon} - W|/W$")
        axs[0, -1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        save(fig, out, f"balance_h_eps_{case}", png)
    return lines


def fig_workprec(df, out, png):
    fig, ax = plt.subplots(figsize=HALF)
    lines = []
    for v in ["F", "C"]:
        st = VARIANT_STYLE[v]
        d = df[df.variant == v].sort_values("n")
        if d.empty:
            continue
        conv = d.converged.astype(bool)
        ax.loglog(d.time_s, d.rel_error.abs(), color=st["color"], marker=st["marker"], label=st["label"])
        ax.loglog(d.time_s[~conv], d.rel_error.abs()[~conv], linestyle="none", marker=st["marker"], markersize=9,
                  markerfacecolor="white", markeredgecolor=st["color"])
        for _, r in d.iterrows():
            ax.annotate(f"{r.n}", (r.time_s, abs(r.rel_error)), textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color="#444444")
            lines.append(f"workprec {v} n={r.n}: err={r.rel_error:.3e} time={r.time_s:.0f}s it={r.iterations} "
                         f"conv={r.converged}")
    ax.set_xlabel("wall-clock time [s]")
    ax.set_ylabel(r"$|W_h - W|/W$")
    ax.legend(loc="upper right")
    save(fig, out, "workprec_wall", png)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)
    out = d / "figures"
    out.mkdir(exist_ok=True)
    report = []
    if (df := load(d, "h")) is not None:
        report += fig_h_convergence(df, out, args.png)
        report += fig_wall_convergence(df, out, args.png)
    if (df := load(d, "eps")) is not None:
        report += fig_eps_convergence(df, out, args.png)
    if (df := load(d, "iter_eps")) is not None:
        report += fig_iter_eps(df, out, args.png)
    if (df := load(d, "iter_h")) is not None:
        fig_iter_h(df, out, args.png)
    if (df := load(d, "timing")) is not None:
        fig_timing(df, out, args.png)
    if (df := load(d, "balance")) is not None:
        report += fig_balance(df, out, args.png)
    if (df := load(d, "workprec")) is not None:
        report += fig_workprec(df, out, args.png)
    (out / "observed_rates.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
