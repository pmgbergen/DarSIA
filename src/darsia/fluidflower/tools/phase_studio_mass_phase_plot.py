"""Plot saved concentration calibrations without changing scientific results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


_SCOPES = ("individual", "shared")
_TITLES = {"individual": "Individual calibration", "shared": "Shared series calibration"}
_IDENTITY = ("dataset", "run", "candidate_id", "params_sha256", "optical_model_sha256")


def _series(result):
    rows = sorted(result.get("frames", []), key=lambda row: float(row["elapsed_h"]))
    if not rows:
        raise ValueError("Saved mass comparison needs at least one frame per calibration.")
    data = {name: np.asarray([row[name] for row in rows], float)
            for name in ("elapsed_h", "aqueous_mass_g", "gas_mass_g", "total_mass_g")}
    if any(not np.isfinite(values).all() for values in data.values()):
        raise ValueError("Saved frame times and phase masses must be finite.")
    if any(np.any(data[name] < 0) for name in ("aqueous_mass_g", "gas_mass_g", "total_mass_g")):
        raise ValueError("Saved phase masses must be nonnegative.")
    if len(np.unique(data["elapsed_h"])) != len(rows):
        raise ValueError("Saved frame times must be unique within each calibration.")
    if not np.allclose(data["aqueous_mass_g"] + data["gas_mass_g"], data["total_mass_g"], rtol=1e-6, atol=1e-7):
        raise ValueError("Saved total mass does not match the dissolved/free-gas phase split.")
    injected = [row.get("injected_mass_g") if row.get("injected_mass_g") is not None
                else row.get("nominal_injected_mass_g") for row in rows]
    data["injected_mass_g"] = np.asarray([np.nan if value is None else value for value in injected], float)
    valid = ~np.isnan(data["injected_mass_g"])
    if np.any(~np.isfinite(data["injected_mass_g"][valid])) or np.any(data["injected_mass_g"][valid] < 0):
        raise ValueError("Available nominal injected masses must be finite and nonnegative.")
    for name in ("total_mass_lower_g", "total_mass_upper_g"):
        data[name] = np.asarray([row.get(name, row["total_mass_g"]) for row in rows], float)
    lower, upper = data["total_mass_lower_g"], data["total_mass_upper_g"]
    if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower < 0) or np.any(upper < lower)
            or np.any(lower > data["total_mass_g"] + 1e-7)
            or np.any(upper + 1e-7 < data["total_mass_g"])):
        raise ValueError("Saved physical mass bounds must contain the quantified mass.")
    data["unknown"] = bool(np.any(upper - lower > 1e-10))
    data["partial"] = data["unknown"] or any(not row.get("quantification_complete", True) for row in rows)
    return data


def _identity(result):
    candidate = result.get("selected_inputs", {}).get("candidate", {})
    if not candidate.get("run") or not candidate.get("dataset"):
        raise ValueError("Saved results must identify their experiment and series.")
    return tuple(candidate.get(key) for key in _IDENTITY)


def _visible_max(data, end=None):
    indices = np.ones(data["elapsed_h"].shape, bool)
    if end is not None:
        indices = (data["elapsed_h"] >= 0) & (data["elapsed_h"] <= end)
        # The line crossing the right window boundary also belongs to the view.
        after = np.flatnonzero(data["elapsed_h"] > end)
        if after.size:
            indices[after[0]] = True
    values = np.concatenate([data[key][indices] for key in ("total_mass_upper_g", "injected_mass_g")])
    finite = values[np.isfinite(values)]
    return float(finite.max(initial=0))


def _draw(axis, data, *, xlim, ylim, title):
    times = data["elapsed_h"]
    axis.stackplot(times, data["aqueous_mass_g"], data["gas_mass_g"],
                   colors=("#39a99d", "#dc7791"), alpha=.82)
    if data["unknown"]:
        axis.fill_between(times, data["total_mass_lower_g"], data["total_mass_upper_g"],
                          color="#bdbdbd", alpha=.32, hatch="///", edgecolor="#757575", linewidth=.5)
    axis.plot(times, data["total_mass_g"], color="#182d38", linewidth=1.9, marker="o", markersize=2.5)
    if np.isfinite(data["injected_mass_g"]).any():
        axis.plot(times, data["injected_mass_g"], color="#b66b00", linestyle="--", linewidth=2.)
    else:
        axis.text(.02, .94, "Nominal injected mass unavailable", transform=axis.transAxes,
                  va="top", fontsize=9, color="#715423")
    if not np.any((times >= xlim[0]) & (times <= xlim[1])):
        axis.text(.5, .5, "No saved frames in this interval", ha="center", transform=axis.transAxes, fontsize=10)
    axis.set(xlim=xlim, ylim=ylim, title=title, ylabel="CO₂ mass · g", xlabel="Time since injection start · h")
    axis.grid(alpha=.23)
    axis.set_axisbelow(True)


def _build_mass_phase_figure(results_by_scope):
    """Internal figure builder kept separate to test plotted values and limits."""
    if not results_by_scope or any(scope not in _SCOPES for scope in results_by_scope):
        raise ValueError("Provide saved results keyed by individual and/or shared.")
    scopes = [scope for scope in _SCOPES if scope in results_by_scope]
    identities = [_identity(results_by_scope[scope]) for scope in scopes]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Compared calibrations must use the same experiment and fixed phase candidate.")
    data = {scope: _series(results_by_scope[scope]) for scope in scopes}
    times = np.concatenate([item["elapsed_h"] for item in data.values()])
    start, end = min(0., float(times.min())), max(0., float(times.max()))
    if end <= start:
        end = start + 1.
    early_end = min(6., end) if end > 0 else 6.
    full_height = max(_visible_max(item) for item in data.values())
    early_height = max(_visible_max(item, early_end) for item in data.values())
    full_ylim = (0., max(full_height * 1.08, 1e-3))
    early_ylim = (0., max(early_height * 1.08, 1e-3))
    fig = Figure(figsize=(14.5 if len(scopes) == 2 else 8., 9.2), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, len(scopes), squeeze=False)
    for column, scope in enumerate(scopes):
        _draw(axes[0, column], data[scope], xlim=(start, end), ylim=full_ylim,
              title=_TITLES[scope] + " · full timeline")
        _draw(axes[1, column], data[scope], xlim=(0., early_end), ylim=early_ylim,
              title=_TITLES[scope] + f" · first {early_end:g} h")
    partial = any(item["partial"] for item in data.values())
    handles = [Patch(facecolor="#39a99d", label="Dissolved CO₂"),
               Patch(facecolor="#dc7791", label="Free gaseous CO₂"),
               Line2D([], [], color="#182d38", linewidth=1.9,
                      label="Detected total" + (" · quantified support" if partial else "")),
               Line2D([], [], color="#b66b00", linestyle="--", linewidth=2., label="Nominal injected CO₂")]
    if any(item["unknown"] for item in data.values()):
        handles.append(Patch(facecolor="#bdbdbd", alpha=.45, hatch="///", edgecolor="#757575",
                             label="Physical bound including uncalibrated support"))
    fig.legend(handles=handles, loc="outside lower center", ncols=3 if len(scopes) == 2 else 2, fontsize=10)
    run = str(identities[0][1]).upper()
    dataset = str(identities[0][0]).upper()
    fig.suptitle(f"{dataset} · {run} · injected mass and detected phase masses\n"
                 "Fits use nominal injection; these curves are not independent concentration validation.\n"
                 "Lines connect saved frames; no new phase or concentration calculation.", fontsize=13)
    return fig


def render_mass_phase_comparison(results_by_scope: dict[str, dict], output_path: Path) -> Path:
    """Save a comparison figure from immutable result dictionaries only.

    No phase map, optical input, calibration fit or external measurement is read.
    Missing injection observations remain gaps rather than assumed zeros.
    """
    output_path = Path(output_path)
    figure = _build_mass_phase_figure(results_by_scope)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    figure.clear()
    return output_path
