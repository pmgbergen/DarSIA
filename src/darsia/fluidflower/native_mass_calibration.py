"""Automatic (Optuna) mass calibration for DarSIA's colour-to-mass analysis.

DarSIA's own mass calibration (Calibration > Mass) is interactive: sliders set
the signal values that turn a position on a sand's colour path into a CO2
signal (0 to 0.75 dissolved, 0.75 to 1 gas). This module is the automatic
alternative used in FluidFlower's earlier calibrations: an Optuna search over
the same signal values that minimises the mass-balance error over the
photographs of ``[calibration.mass] data_selection``::

    error = sum over photographs | detected CO2 mass - injected CO2 mass |

It works on the same files as the interactive tool. It starts from the
colour-to-mass calibration already stored for the embedding (written by
Calibration > Default mass, which it runs first when there is none), keeps the
flash (dissolved/gas split) as it is, and writes the best signal values back
to the same folder, where Analysis and the interactive tool find them. A copy
of the previous calibration, every trial and a mass-balance figure are kept
in a report folder next to it.

Speed: the signal model is linear in its values up to the flash, and DarSIA's
volume averaging is linear too. Each photograph is therefore analysed once,
through DarSIA's own signal model, for a unit step at each knot of each sand's
signal function; a trial then only combines those restored fields and applies
the flash and the mass formulas. The result equals DarSIA's own computation,
which the module checks at the end for the best values.

Run as ``python -m darsia.fluidflower.native_mass_calibration --config
run.toml``; the native FluidFlower window offers it as "Mass, Optuna" in
Original DarSIA workflow > Original calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Upper bound of a signal value; FluidFlower's earlier searches used the same.
DEFAULT_UPPER = 2.0
#: Restored basis values at or below this are floating-point residue of the
#: averaging filter and are left out of the sparse fields.
ACTIVE_TOLERANCE = 1e-12
#: Name of the report folder next to the colour-to-mass calibration.
REPORT_FOLDER = "mass_calibration_optuna"


class CalibrationError(RuntimeError):
    """A calibration cannot start or finish; the message says why."""


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SearchSettings:
    """Settings of one automatic calibration."""

    trials: int = 200
    """Number of Optuna trials (the first starts from the current calibration)."""
    refine: int | None = None
    """Evaluations of the local refinement after the search (None: 30 per
    parameter, at least 40; 0: no refinement)."""
    seed: int = 0
    """Seed of the sampler, so that a run can be repeated."""
    per_facies: bool = False
    """Separate signal values per sand facies instead of shared ones."""
    upper: float = DEFAULT_UPPER
    """Upper bound of the signal values."""
    objective: str = "absolute"
    """'absolute' (sum of |error|) or 'relative' (sum of |error| / injected)."""
    min_hours: float | None = None
    """Leave out photographs taken earlier (hours after the experiment start)."""
    max_hours: float | None = None
    """Leave out photographs taken later (hours after the experiment start)."""
    timeout_minutes: float | None = None
    """Stop the search after this time even if trials remain."""


@dataclass(frozen=True)
class SignalSpace:
    """Monotone signal values of the free (sand) facies, as Optuna parameters.

    Each free facies has a piecewise-linear signal function over the knots of
    its colour path. The first value stays 0 (baseline colour, no CO2). A
    function is drawn as a gain and a shape: the gain is its last value (the
    end colour of the path, in [0, upper]) and the shape says how much of the
    rise is reached at each inner knot, as cumulative fractions (shape k =
    shape k-1 + s_k * (1 - shape k-1), s_k in [0, 1]); value k = gain * shape k.
    This maps the parameter box one to one onto the monotone functions with
    values in [0, upper], and it separates the overall strength (which sets
    the detected mass) from the form (which sets how it changes as CO2
    dissolves), which the samplers handle better than raw values. With
    ``shared`` all free facies use the same function.
    """

    labels: tuple[int, ...]
    knots: tuple[int, ...]
    upper: float = DEFAULT_UPPER
    shared: bool = True

    def __post_init__(self) -> None:
        if not self.labels:
            raise CalibrationError("No facies with a colour path to calibrate.")
        if len(self.labels) != len(self.knots):
            raise ValueError("One knot count per label is required.")
        if any(count < 2 for count in self.knots):
            raise CalibrationError("Every signal function needs at least two knots.")
        if self.shared and len(set(self.knots)) > 1:
            raise CalibrationError(
                "Shared signal values need the same number of colour-path knots for "
                f"every sand facies, got {dict(zip(self.labels, self.knots))}. "
                "Use separate values per facies."
            )
        if not self.upper > 0:
            raise ValueError("The upper bound must be positive.")

    @property
    def groups(self) -> list[tuple[str, tuple[int, ...], int]]:
        """(parameter prefix, labels, knot count) per independent signal function."""
        if self.shared:
            return [("shared", self.labels, self.knots[0])]
        return [
            (f"facies{label}", (label,), count)
            for label, count in zip(self.labels, self.knots)
        ]

    @property
    def parameter_names(self) -> list[str]:
        names = []
        for prefix, _labels, count in self.groups:
            names.append(f"{prefix}_gain")
            names.extend(f"{prefix}_shape{k}" for k in range(1, count - 1))
        return names

    def bounds(self, name: str) -> tuple[float, float]:
        return (0.0, self.upper) if name.endswith("_gain") else (0.0, 1.0)

    @property
    def basis_keys(self) -> list[tuple[int, int]]:
        """(label, knot) of each unit step, in the order of the basis fields."""
        return [
            (label, k)
            for label, count in zip(self.labels, self.knots)
            for k in range(1, count)
        ]

    def values(self, params: dict[str, float]) -> dict[int, np.ndarray]:
        """Signal values per label for Optuna parameters (gain and shape)."""
        result: dict[int, np.ndarray] = {}
        for prefix, labels, count in self.groups:
            gain = min(max(float(params[f"{prefix}_gain"]), 0.0), self.upper)
            shape = np.zeros(count)
            shape[-1] = 1.0
            for k in range(1, count - 1):
                rise = min(max(float(params[f"{prefix}_shape{k}"]), 0.0), 1.0)
                shape[k] = shape[k - 1] + rise * (1.0 - shape[k - 1])
            for label in labels:
                result[label] = gain * shape
        return result

    def parameters(self, values: dict[int, Sequence[float]]) -> dict[str, float]:
        """Optuna parameters for given signal values (shared: their mean).

        Values are projected onto the search space first: the first to 0, the
        others made non-decreasing and capped at the upper bound.
        """
        params: dict[str, float] = {}
        for prefix, labels, count in self.groups:
            stacked = np.array([np.asarray(values[label], float) for label in labels])
            projected = np.maximum.accumulate(np.clip(stacked.mean(axis=0), 0.0, None))
            projected[0] = 0.0
            projected = np.minimum(projected, self.upper)
            gain = float(projected[-1])
            params[f"{prefix}_gain"] = gain
            shape = projected / gain if gain > 0 else np.linspace(0.0, 1.0, count)
            for k in range(1, count - 1):
                room = 1.0 - shape[k - 1]
                rise = 0.0 if room <= 1e-12 else (shape[k] - shape[k - 1]) / room
                params[f"{prefix}_shape{k}"] = min(max(float(rise), 0.0), 1.0)
        return params

    def coefficients(self, values: dict[int, np.ndarray]) -> np.ndarray:
        """Weights of the unit-step fields: the increments of the values."""
        return np.array(
            [values[label][k] - values[label][k - 1] for label, k in self.basis_keys],
            dtype=np.float64,
        )


# --------------------------------------------------------------------------- #
# Photographs: DarSIA's analysis once per unit step, then fast trials
# --------------------------------------------------------------------------- #


@dataclass
class PhotoFields:
    """What a trial needs of one photograph, on its active pixels only."""

    name: str
    hours: float
    injected: float
    """Injected CO2 mass [kg] when the photograph was taken."""
    shape: tuple[int, int]
    path: Path | None
    """The photograph, read again for the final check with DarSIA."""
    index: np.ndarray
    """Flat indices of the pixels where a unit step gives a signal."""
    basis: np.ndarray
    """(steps, pixels): DarSIA's restored signal for a unit step at each knot."""
    weight: np.ndarray
    """Integration weight (pixel area x depth x porosity) [m^3]."""
    density: np.ndarray
    """Density of gaseous CO2 [kg/m^3] for this photograph's pressure."""
    solubility: np.ndarray
    """Mass of dissolved CO2 per volume of saturated water [kg/m^3]."""
    label: np.ndarray
    """Facies of each active pixel (for the report)."""
    keep_g: np.ndarray | None = None
    keep_aq: np.ndarray | None = None

    @property
    def active_fraction(self) -> float:
        return self.index.size / float(self.shape[0] * self.shape[1])


@dataclass(frozen=True)
class Flash:
    """DarSIA's SimpleFlash bounds, kept fixed during the search."""

    min_aq: float
    max_aq: float
    min_g: float
    max_g: float

    @classmethod
    def of(cls, flash: Any) -> "Flash":
        return cls(
            float(flash.min_value_aq),
            float(flash.max_value_aq),
            float(flash.min_value_g),
            float(flash.max_value_g),
        )


def photo_masses(
    photo: PhotoFields,
    coefficients: np.ndarray,
    flash: Flash,
    *,
    by_label: bool = False,
) -> tuple[float, float] | dict[int, tuple[float, float]]:
    """Gaseous and dissolved CO2 mass [kg] of a photograph for given increments.

    The same formulas as SimpleFlash and CO2MassAnalysis.mass_analysis, on the
    active pixels (all others carry no signal).
    """
    signal = np.asarray(coefficients, dtype=np.float32) @ photo.basis
    signal = signal.astype(np.float64, copy=False)
    c_aq = (np.clip(signal, flash.min_aq, flash.max_aq) - flash.min_aq) / (
        flash.max_aq - flash.min_aq
    )
    s_g = (np.clip(signal, flash.min_g, flash.max_g) - flash.min_g) / (
        flash.max_g - flash.min_g
    )
    if photo.keep_aq is not None:
        c_aq = np.where(photo.keep_aq, c_aq, 0.0)
    if photo.keep_g is not None:
        s_g = np.where(photo.keep_g, s_g, 0.0)
    mass_g = photo.weight * photo.density * s_g
    mass_aq = photo.weight * photo.solubility * c_aq * np.clip(1.0 - s_g, 0.0, None)
    if not by_label:
        return float(mass_g.sum()), float(mass_aq.sum())
    return {
        int(label): (
            float(mass_g[photo.label == label].sum()),
            float(mass_aq[photo.label == label].sum()),
        )
        for label in np.unique(photo.label)
    }


def _signal_functions(analysis: Any) -> dict[int, Any]:
    """The per-facies PWTransformation objects DarSIA evaluates."""
    return analysis.signal_model.model[1]


def current_values(analysis: Any) -> dict[int, np.ndarray]:
    functions = _signal_functions(analysis)
    return {
        int(label): np.asarray(functions[label].values, dtype=np.float64).copy()
        for label in functions.keys()
    }


def set_values(analysis: Any, values: dict[int, Sequence[float]]) -> None:
    functions = _signal_functions(analysis)
    for label, label_values in values.items():
        functions[label].update(values=np.asarray(label_values, dtype=np.float64))


def free_labels(analysis: Any) -> list[int]:
    """Facies with a colour path, i.e. not ignored by the colour-to-mass model."""
    functions = _signal_functions(analysis)
    ignored = {int(label) for label in functions.ignore_labels}
    present = {int(label) for label in np.unique(analysis.labels.img)}
    return sorted(
        int(label)
        for label in functions.keys()
        if int(label) not in ignored and int(label) in present
    )


def _update_state(analysis: Any, experiment: Any, date: datetime) -> None:
    state = experiment.pressure_temperature_protocol.get_state(date)
    gradient = experiment.pressure_temperature_protocol.get_gradient(date)
    analysis.co2_mass_analysis.update_state(
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )


def _integration_weights(geometry: Any, image: Any) -> np.ndarray:
    import darsia

    shape = tuple(image.img.shape[:2])
    try:
        weights = geometry.make_extensive(darsia.full_like(image, 1.0)).img
    except (ValueError, AttributeError):
        geometry._prepare_cached_voxel_volume(list(shape))
        weights = geometry.cached_voxel_volume
    return np.broadcast_to(np.asarray(weights, dtype=np.float64), shape)


def exact_masses(
    analysis: Any, experiment: Any, interpretation: Any
) -> tuple[float, float]:
    """DarSIA's own result for the current signal values (as in its manual tool)."""
    _update_state(analysis, experiment, interpretation.date)
    signal = analysis.call_pH_analysis(interpretation)
    result = analysis.call_flash_and_mass_analysis(signal)
    return (
        float(analysis.geometry.integrate(result.mass_g)),
        float(analysis.geometry.integrate(result.mass_aq)),
    )


def prepare_photo(
    analysis: Any,
    experiment: Any,
    interpretation: Any,
    space: SignalSpace,
    *,
    name: str,
    hours: float,
    injected: float,
    path: Path | None = None,
) -> PhotoFields:
    """Analyse one photograph once per unit step with DarSIA's signal model."""
    original = current_values(analysis)
    steps = []
    try:
        for label, k in space.basis_keys:
            values = {
                other: np.zeros_like(other_values)
                for other, other_values in original.items()
            }
            values[label] = (np.arange(len(original[label])) >= k).astype(np.float64)
            set_values(analysis, values)
            restored = analysis.call_pH_analysis(interpretation).img
            steps.append(np.asarray(restored, dtype=np.float64))
    finally:
        set_values(analysis, original)

    shape = tuple(steps[0].shape[:2])
    stack = np.stack([step.reshape(-1) for step in steps])
    index = np.flatnonzero(np.any(np.abs(stack) > ACTIVE_TOLERANCE, axis=0))
    basis = np.ascontiguousarray(stack[:, index], dtype=np.float32)
    del stack, steps

    _update_state(analysis, experiment, interpretation.date)
    co2 = analysis.co2_mass_analysis
    density = np.broadcast_to(co2.density_gaseous_co2, shape).reshape(-1)[index]
    solubility = np.broadcast_to(co2.solubility_co2, shape).reshape(-1)[index]
    weight = _integration_weights(analysis.geometry, interpretation).reshape(-1)[index]

    keep_g = keep_aq = None
    adapter = getattr(analysis, "expert_knowledge_adapter", None)
    if adapter is not None:
        mask_g = adapter.mask_for(interpretation, "saturation_g")
        mask_aq = adapter.mask_for(interpretation, "concentration_aq")
        keep_g = None if mask_g is None else mask_g.reshape(-1)[index].copy()
        keep_aq = None if mask_aq is None else mask_aq.reshape(-1)[index].copy()

    return PhotoFields(
        name=name,
        hours=hours,
        injected=float(injected),
        shape=shape,
        path=path,
        index=index,
        basis=basis,
        weight=np.ascontiguousarray(weight, dtype=np.float64),
        density=np.ascontiguousarray(density, dtype=np.float64),
        solubility=np.ascontiguousarray(solubility, dtype=np.float64),
        label=np.asarray(analysis.labels.img).reshape(-1)[index].astype(np.int16),
        keep_g=keep_g,
        keep_aq=keep_aq,
    )


# --------------------------------------------------------------------------- #
# Objective and search
# --------------------------------------------------------------------------- #


def mass_error(
    detected: Sequence[float], injected: Sequence[float], objective: str
) -> float:
    detected = np.asarray(detected, dtype=np.float64)
    injected = np.asarray(injected, dtype=np.float64)
    deviation = np.abs(detected - injected)
    if objective == "absolute":
        return float(deviation.sum())
    if objective == "relative":
        floor = max(1e-3 * float(np.max(np.abs(injected), initial=0.0)), 1e-12)
        return float((deviation / np.maximum(np.abs(injected), floor)).sum())
    raise ValueError(f"Unknown objective {objective!r}; use 'absolute' or 'relative'.")


def mean_error_percent(detected: Sequence[float], injected: Sequence[float]) -> float:
    """Mean |detected - injected| in percent of the largest injected mass."""
    detected = np.asarray(detected, dtype=np.float64)
    injected = np.asarray(injected, dtype=np.float64)
    scale = float(np.max(np.abs(injected), initial=0.0))
    if scale <= 0:
        return math.nan
    return float(100.0 * np.mean(np.abs(detected - injected)) / scale)


@dataclass
class TrialRecord:
    number: int
    error: float
    error_percent: float
    values: dict[int, list[float]]
    masses: list[float]
    seconds: float
    stage: str = "optuna"
    """'optuna' (search) or 'refine' (local refinement of the best trial)."""


@dataclass
class SearchResult:
    best: TrialRecord
    start: TrialRecord
    trials: list[TrialRecord] = field(default_factory=list)


def evaluate(
    photos: Sequence[PhotoFields],
    space: SignalSpace,
    values: dict[int, np.ndarray],
    flash: Flash,
) -> list[float]:
    coefficients = space.coefficients(values)
    return [sum(photo_masses(photo, coefficients, flash)) for photo in photos]


def run_search(
    photos: Sequence[PhotoFields],
    space: SignalSpace,
    flash: Flash,
    start_values: dict[int, np.ndarray],
    settings: SearchSettings,
    *,
    on_trial: Callable[[TrialRecord, TrialRecord], None] | None = None,
) -> SearchResult:
    """Optuna (TPE) search, starting from the given values.

    The first trial is ``start_values`` projected onto the search space (with
    shared values: their mean over the sand facies).
    """
    try:
        import optuna
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise CalibrationError(
            "Optuna is not installed in this environment; install the FluidFlower "
            "extra (pip install -e .[fluidflower]) or 'optuna'."
        ) from error

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    injected = [photo.injected for photo in photos]
    records: list[TrialRecord] = []
    best: list[TrialRecord] = []

    def objective(trial: Any) -> float:
        tic = time.perf_counter()
        params = {
            name: trial.suggest_float(name, *space.bounds(name))
            for name in space.parameter_names
        }
        values = space.values(params)
        masses = evaluate(photos, space, values, flash)
        error = mass_error(masses, injected, settings.objective)
        record = TrialRecord(
            number=trial.number,
            error=error,
            error_percent=mean_error_percent(masses, injected),
            values={label: value.tolist() for label, value in values.items()},
            masses=masses,
            seconds=time.perf_counter() - tic,
        )
        records.append(record)
        if not best or record.error < best[0].error:
            best[:] = [record]
        if on_trial is not None:
            on_trial(record, best[0])
        return error

    with warnings.catch_warnings():
        # Multivariate TPE is marked experimental in Optuna 4; it is the
        # sampler that models the coupling between the knots of a function.
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=settings.seed,
            multivariate=True,
            n_startup_trials=max(1, min(10, settings.trials // 3)),
        )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(space.parameters(start_values))
    study.optimize(
        objective,
        n_trials=settings.trials,
        timeout=(
            None if settings.timeout_minutes is None else 60.0 * settings.timeout_minutes
        ),
        gc_after_trial=False,
    )
    if not records:
        raise CalibrationError("No trial finished.")
    refine(photos, space, flash, settings, records, best, on_trial=on_trial)
    return SearchResult(best=best[0], start=records[0], trials=records)


def refine(
    photos: Sequence[PhotoFields],
    space: SignalSpace,
    flash: Flash,
    settings: SearchSettings,
    records: list[TrialRecord],
    best: list[TrialRecord],
    *,
    on_trial: Callable[[TrialRecord, TrialRecord], None] | None = None,
) -> None:
    """Nelder-Mead from the best trial, in the same cumulative fractions.

    The sampler finds the right valley; the few local steps that reach its
    bottom are cheap, so they are left to a simplex method. Each evaluation is
    recorded like a trial (stage 'refine').
    """
    names = space.parameter_names
    evaluations = settings.refine
    if evaluations is None:
        evaluations = max(40, 30 * len(names))
    if evaluations <= 0 or not names:
        return
    from scipy.optimize import minimize

    injected = [photo.injected for photo in photos]
    start = space.parameters({label: np.asarray(v) for label, v in best[0].values.items()})
    # The simplex works on parameters scaled to [0, 1].
    width = np.array([space.bounds(name)[1] for name in names])
    x0 = np.array([start[name] for name in names]) / width
    scale = best[0].error if best[0].error > 0 else 1.0

    def objective(x: np.ndarray) -> float:
        tic = time.perf_counter()
        values = space.values(dict(zip(names, np.clip(x, 0.0, 1.0) * width)))
        masses = evaluate(photos, space, values, flash)
        error = mass_error(masses, injected, settings.objective)
        record = TrialRecord(
            number=len(records),
            error=error,
            error_percent=mean_error_percent(masses, injected),
            values={label: value.tolist() for label, value in values.items()},
            masses=masses,
            seconds=time.perf_counter() - tic,
            stage="refine",
        )
        records.append(record)
        if record.error < best[0].error:
            best[:] = [record]
        if on_trial is not None:
            on_trial(record, best[0])
        return error / scale

    simplex = [x0]
    for i in range(len(names)):
        vertex = x0.copy()
        vertex[i] = vertex[i] + 0.05 if vertex[i] + 0.05 <= 1.0 else vertex[i] - 0.05
        simplex.append(vertex)
    minimize(
        objective,
        x0,
        method="Nelder-Mead",
        bounds=[(0.0, 1.0)] * len(names),
        options={
            "maxfev": evaluations,
            "xatol": 1e-5,
            "fatol": 1e-7,
            "initial_simplex": np.array(simplex),
        },
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def write_trials_csv(path: Path, records: Sequence[TrialRecord], space: SignalSpace) -> None:
    columns = [f"facies{label}_v{k}" for label, count in zip(space.labels, space.knots) for k in range(count)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial", "stage", "error", "mean_error_percent", *columns, "seconds"])
        for record in records:
            values = [
                record.values[label][k]
                for label, count in zip(space.labels, space.knots)
                for k in range(count)
            ]
            writer.writerow(
                [
                    record.number,
                    record.stage,
                    f"{record.error:.9g}",
                    f"{record.error_percent:.4f}",
                    *[f"{value:.6f}" for value in values],
                    f"{record.seconds:.3f}",
                ]
            )


def plot_report(
    path: Path,
    photos: Sequence[PhotoFields],
    result: SearchResult,
    *,
    before: Sequence[float] | None,
    best_exact: Sequence[float] | None,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    hours = [photo.hours for photo in photos]
    grams = 1000.0
    fig, (ax_mass, ax_trials) = plt.subplots(1, 2, figsize=(13, 5))
    ax_mass.plot(hours, [p.injected * grams for p in photos], "k-o", label="Injected")
    if before is not None:
        ax_mass.plot(hours, [m * grams for m in before], "--s", color="0.55", label="Before (current calibration)")
    ax_mass.plot(hours, [m * grams for m in result.best.masses], "-o", color="tab:blue", label="Best trial")
    if best_exact is not None:
        ax_mass.plot(hours, [m * grams for m in best_exact], "x", color="tab:red", label="Best, DarSIA check")
    ax_mass.set_xlabel("Time after experiment start [h]")
    ax_mass.set_ylabel("CO2 mass [g]")
    ax_mass.set_title("Detected and injected CO2")
    ax_mass.grid(alpha=0.3)
    ax_mass.legend()

    numbers = np.array([record.number for record in result.trials])
    errors = np.array([record.error_percent for record in result.trials])
    refined = np.array([record.stage == "refine" for record in result.trials])
    # The objective, not the plotted percentage, decides the best trial.
    objective = np.array([record.error for record in result.trials])
    running = errors[[int(np.argmin(objective[: i + 1])) for i in range(len(objective))]]
    ax_trials.semilogy(numbers[~refined], errors[~refined], ".", color="0.6", label="Optuna trial")
    if refined.any():
        ax_trials.semilogy(numbers[refined], errors[refined], ".", color="tab:orange", label="Refinement")
    ax_trials.semilogy(numbers, running, "-", color="tab:blue", label="Best so far")
    ax_trials.set_xlabel("Trial")
    ax_trials.set_ylabel("Mean |detected - injected| [% of largest injected]")
    ax_trials.set_title("Search")
    ax_trials.grid(alpha=0.3, which="both")
    ax_trials.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Calibration of a run configuration
# --------------------------------------------------------------------------- #


@dataclass
class CalibrationOutcome:
    report_folder: Path
    calibration_folder: Path
    saved: bool
    summary: dict


def _say(message: str) -> None:
    print(message, flush=True)


def _read_photo(rig: Any, path: Path, cfg: Any, selection: Sequence[Path]) -> Any:
    """A corrected photograph, one at a time (the selection may not fit in memory).

    With ``[data] use_cache`` it uses the same cache file as DarSIA's
    load_images_with_cache for the whole selection.
    """
    import darsia

    cache_dir = getattr(cfg.data, "cache", None)
    if not (getattr(cfg.data, "use_cache", False) and cache_dir is not None):
        return rig.read_image(path)
    resolved = Path(path).resolve()
    try:
        root = Path(os.path.commonpath([str(Path(p).resolve()) for p in selection]))
        relative = resolved.relative_to(root)
    except ValueError:
        relative = Path(resolved.name)
    if not relative.name:
        relative = Path(resolved.name)
    cache_path = Path(cache_dir) / relative.with_suffix(".npz")
    if cache_path.exists():
        return darsia.imread(cache_path)
    image = rig.read_image(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(cache_path)
    return image


def _relative_hours(image: Any, experiment: Any) -> float:
    if getattr(image, "time", None) is not None:
        return float(image.time) / 3600.0
    return (image.date - experiment.experiment_start).total_seconds() / 3600.0


def _load_analysis(ctx: Any, embedding: Any) -> Any:
    """The stored colour-to-mass analysis of the embedding (as Analysis loads it)."""
    import darsia
    from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
        HeterogeneousColorToMassAnalysis,
    )

    experiment = ctx.experiment
    rig = ctx.fluidflower
    start = experiment.experiment_start
    state = experiment.pressure_temperature_protocol.get_state(start)
    gradient = experiment.pressure_temperature_protocol.get_gradient(start)
    co2 = darsia.CO2MassAnalysis(
        baseline=rig.baseline,
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )
    analysis_cfg = ctx.config.analysis.mass if ctx.config.analysis else None
    return HeterogeneousColorToMassAnalysis.load(
        folder=embedding.color_to_mass_folder,
        baseline=rig.baseline,
        labels=embedding.get_labels(rig),
        co2_mass_analysis=co2,
        geometry=rig.geometry,
        restoration=ctx.restoration,
        basis=embedding.basis,
        expert_knowledge_adapter=ctx.expert_knowledge_adapter,
        contour_smoother=getattr(analysis_cfg, "contour_smoother", None),
    )


def _default_report_root(config_paths, calibration_folder):
    """Use series Calibration storage while preserving standalone colour workflows."""
    from .series_paths import series_folder_path
    for configured in config_paths:
        for root in Path(configured).resolve().parents:
            if (root / "series.toml").is_file():
                return series_folder_path(root, "colour_calibration_reports")
    return Path(calibration_folder).parent.parent / REPORT_FOLDER


def calibrate(
    config: Sequence[str | Path],
    settings: SearchSettings,
    *,
    dry_run: bool = False,
    report_root: Path | None = None,
) -> CalibrationOutcome:
    """Run the automatic mass calibration for a run configuration."""
    import darsia
    from darsia.presets.workflows.analysis.analysis_context import (
        prepare_analysis_context,
        select_image_paths,
    )
    from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (
        calibration_color_to_mass_analysis_from_context,
    )
    from darsia.presets.workflows.rig import Rig
    from darsia.signals.color import ColorPathEmbedding

    config_paths = [Path(path) for path in config]
    started = time.time()
    _say("Automatic mass calibration (Optuna): loading the run ...")
    ctx = prepare_analysis_context(
        cls=Rig,
        path=config_paths,
        all=False,
        require_color_to_mass=False,
        section="calibration",
        require_results=False,
    )
    cfg = ctx.config
    if cfg.calibration is None or cfg.calibration.mass is None:
        raise CalibrationError("The configuration has no [calibration.mass] section.")
    mass_cfg = cfg.calibration.mass
    embedding = mass_cfg.embedding
    if not isinstance(embedding, ColorPathEmbedding):
        raise CalibrationError(
            "Automatic mass calibration needs a colour-path embedding in "
            "[calibration.mass] embedding."
        )
    if not embedding.color_paths_csv_file.exists():
        raise CalibrationError(
            f"No colour path for '{embedding.embedding_id}' yet "
            f"({embedding.color_paths_csv_file}); run Setup > Color path first."
        )
    folder = Path(embedding.color_to_mass_folder)
    if not folder.exists():
        _say(
            "No colour-to-mass calibration yet: creating the default one first "
            "(the same as Calibration > Default mass)."
        )
        calibration_color_to_mass_analysis_from_context(
            ctx, ref_path=None, reset=False, show=False, default=True
        )
    analysis = _load_analysis(ctx, embedding)
    experiment = ctx.experiment
    rig = ctx.fluidflower

    labels = free_labels(analysis)
    original = current_values(analysis)
    space = SignalSpace(
        labels=tuple(labels),
        knots=tuple(len(original[label]) for label in labels),
        upper=settings.upper,
        shared=not settings.per_facies,
    )
    flash = Flash.of(analysis.flash)
    _say(
        f"Sand facies with a colour path: {', '.join(map(str, labels))}; "
        f"{'shared' if space.shared else 'separate'} signal values, "
        f"{len(space.parameter_names)} parameters, flash kept at "
        f"{flash.min_aq:g}-{flash.max_aq:g} (dissolved), {flash.min_g:g}-{flash.max_g:g} (gas)."
    )

    paths = select_image_paths(cfg, experiment, all=False, sub_config=mass_cfg)
    roi = darsia.CoordinateArray([rig.baseline.origin, rig.baseline.opposite_corner])
    photos: list[PhotoFields] = []
    before: list[float] = []
    skipped: list[str] = []
    for number, path in enumerate(paths, start=1):
        image = _read_photo(rig, path, cfg, paths)
        hours = _relative_hours(image, experiment)
        if (settings.min_hours is not None and hours < settings.min_hours) or (
            settings.max_hours is not None and hours > settings.max_hours
        ):
            skipped.append(Path(path).name)
            continue
        interpretation = analysis.call_color_interpretation(image)
        del image
        injected = float(experiment.injection_protocol.injected_mass(date=interpretation.date, roi=roi))
        before.append(sum(exact_masses(analysis, experiment, interpretation)))
        photo = prepare_photo(
            analysis,
            experiment,
            interpretation,
            space,
            name=Path(path).name,
            hours=hours,
            injected=injected,
            path=Path(path),
        )
        photos.append(photo)
        _say(
            f"Photograph {number}/{len(paths)} {photo.name}: {hours:.2f} h, injected "
            f"{1000 * injected:.3f} g, now detected {1000 * before[-1]:.3f} g "
            f"({100 * photo.active_fraction:.0f} % of pixels carry signal)."
        )
    if skipped:
        _say(f"Left out by the time window: {', '.join(skipped)}.")
    if not photos:
        raise CalibrationError("No photograph left for the calibration.")
    if all(photo.injected <= 0 for photo in photos):
        raise CalibrationError(
            "No CO2 was injected before any of the calibration photographs; check "
            "[calibration.mass] data_selection and the injection protocol."
        )

    order = sorted(range(len(photos)), key=lambda i: photos[i].hours)
    photos = [photos[i] for i in order]
    before = [before[i] for i in order]
    injected = [photo.injected for photo in photos]
    start_values = {label: original[label] for label in labels}
    # The fast evaluation of the current values must reproduce DarSIA's result
    # (possible whenever the first signal value is 0, as DarSIA writes it).
    start_check = (
        evaluate(photos, space, start_values, flash)
        if all(abs(original[label][0]) < 1e-12 for label in labels)
        else None
    )
    _say(
        f"Before: mean deviation {mean_error_percent(before, injected):.2f} % of the "
        f"largest injected mass ({1000 * max(injected):.3f} g)."
    )
    if start_check is not None:
        difference = max(
            abs(a - b) / max(abs(b), 1e-12) for a, b in zip(start_check, before)
        )
        _say(
            f"Fast evaluation of the current values matches DarSIA's within "
            f"{100 * difference:.4f} %."
        )

    last_report = [0.0]

    def on_trial(record: TrialRecord, best: TrialRecord) -> None:
        now = time.time()
        first = record.number == 0 or (
            record.stage == "refine" and record.number == settings.trials
        )
        if first or record.number == settings.trials - 1 or now - last_report[0] > 2.0:
            last_report[0] = now
            step = (
                f"Trial {record.number + 1}/{settings.trials}"
                if record.stage == "optuna"
                else f"Refinement step {record.number + 1 - settings.trials}"
            )
            _say(
                f"{step}: {record.error_percent:.2f} % "
                f"(best {best.error_percent:.2f} %, trial {best.number + 1})"
            )

    result = run_search(photos, space, flash, start_values, settings, on_trial=on_trial)
    best_values = {label: np.asarray(result.best.values[label]) for label in labels}

    # DarSIA's own computation for the best values, photograph by photograph.
    _say("Checking the best values with DarSIA's own analysis ...")
    full_best = original_like(original, best_values)
    set_values(analysis, full_best)
    best_exact: list[float] = []
    for photo in photos:
        image = _read_photo(rig, photo.path, cfg, paths)
        interpretation = analysis.call_color_interpretation(image)
        del image
        best_exact.append(sum(exact_masses(analysis, experiment, interpretation)))
    deviation = max(
        abs(a - b) / max(abs(b), 1e-12) for a, b in zip(result.best.masses, best_exact)
    )
    start_deviation = (
        None
        if start_check is None
        else max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(start_check, before))
    )

    error_before = mass_error(before, injected, settings.objective)
    error_best = mass_error(best_exact, injected, settings.objective)
    improves = error_best < error_before

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = (Path(report_root) if report_root else _default_report_root(config_paths, folder)) / timestamp
    report.mkdir(parents=True, exist_ok=True)
    saved = False
    if improves and not dry_run:
        shutil.copytree(folder, report / "previous_calibration", dirs_exist_ok=True)
        analysis.save(folder)
        saved = True
    elif not improves:
        set_values(analysis, original)

    by_label_best = [
        photo_masses(photo, space.coefficients(best_values), flash, by_label=True)
        for photo in photos
    ]
    summary = {
        "configuration": [str(path) for path in config_paths],
        "embedding": embedding.embedding_id,
        "calibration_folder": str(folder),
        "report_folder": str(report),
        "saved": saved,
        "dry_run": dry_run,
        "improves": improves,
        "settings": asdict(settings),
        "facies": labels,
        "shared_values": space.shared,
        "flash": asdict(flash),
        "values_before": {str(label): original[label].tolist() for label in labels},
        "values_best": {str(label): best_values[label].tolist() for label in labels},
        "objective_before": error_before,
        "objective_best": error_best,
        "mean_error_percent_before": mean_error_percent(before, injected),
        "mean_error_percent_best": mean_error_percent(best_exact, injected),
        "best_trial": result.best.number,
        "trials": len(result.trials),
        "fast_vs_darsia_best_max_relative_difference": deviation,
        "fast_vs_darsia_start_max_relative_difference": start_deviation,
        "photographs": [
            {
                "name": photo.name,
                "hours": photo.hours,
                "injected_g": 1000 * photo.injected,
                "before_g": 1000 * b,
                "best_g": 1000 * m,
                "best_darsia_g": 1000 * e,
                "best_by_facies_g": {
                    str(label): {"gas": 1000 * g, "dissolved": 1000 * a}
                    for label, (g, a) in parts.items()
                },
                "active_pixel_fraction": photo.active_fraction,
            }
            for photo, b, m, e, parts in zip(photos, before, result.best.masses, best_exact, by_label_best)
        ],
        "seconds": time.time() - started,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    write_trials_csv(report / "trials.csv", result.trials, space)
    (report / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_report(
        report / "mass_balance.png",
        photos,
        result,
        before=before,
        best_exact=best_exact,
        title=f"Automatic mass calibration, {embedding.embedding_id}: "
        f"{summary['mean_error_percent_before']:.2f} % -> {summary['mean_error_percent_best']:.2f} %",
    )

    _say("")
    _say(f"Finished {len(result.trials)} trials in {summary['seconds'] / 60:.1f} min.")
    _say(
        f"Mean deviation from the injected mass: before {summary['mean_error_percent_before']:.2f} %, "
        f"best {summary['mean_error_percent_best']:.2f} % (trial {result.best.number + 1})."
    )
    for label in labels:
        _say(
            f"  facies {label}: {', '.join(f'{v:.3f}' for v in original[label])} -> "
            f"{', '.join(f'{v:.3f}' for v in best_values[label])}"
        )
    if deviation > 1e-3:
        _say(
            f"Warning: the fast evaluation differs from DarSIA's by up to {100 * deviation:.2f} %."
        )
    if saved:
        _say(f"Saved the calibration to {folder} (previous one in {report / 'previous_calibration'}).")
    elif dry_run:
        _say("Dry run: the calibration was not changed.")
    else:
        _say("The search found nothing better than the current calibration; it was not changed.")
    _say(f"Report: {report}")
    return CalibrationOutcome(report_folder=report, calibration_folder=folder, saved=saved, summary=summary)


def original_like(
    original: dict[int, np.ndarray], values: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """All signal functions, with the given labels replaced."""
    merged = {label: np.asarray(v, dtype=np.float64).copy() for label, v in original.items()}
    for label, label_values in values.items():
        merged[label] = np.asarray(label_values, dtype=np.float64).copy()
    return merged


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m darsia.fluidflower.native_mass_calibration",
        description=(
            "Automatic (Optuna) mass calibration of DarSIA's colour-to-mass "
            "analysis over the [calibration.mass] photographs."
        ),
    )
    parser.add_argument("--config", nargs="+", required=True, help="Run configuration file(s).")
    parser.add_argument("--trials", type=int, default=SearchSettings.trials)
    parser.add_argument(
        "--refine",
        type=int,
        default=None,
        help="Evaluations of the local refinement after the search (default: 30 "
        "per parameter, at least 40; 0 switches it off).",
    )
    parser.add_argument("--seed", type=int, default=SearchSettings.seed)
    parser.add_argument(
        "--per-facies",
        action="store_true",
        help="Separate signal values per sand facies (default: shared values).",
    )
    parser.add_argument("--upper", type=float, default=DEFAULT_UPPER, help="Upper bound of the signal values.")
    parser.add_argument("--objective", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--min-hours", type=float, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--timeout-minutes", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Report only; keep the calibration.")
    parser.add_argument("--report-folder", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    if args.trials < 1:
        print("--trials must be at least 1.", file=sys.stderr)
        return 2
    settings = SearchSettings(
        trials=args.trials,
        refine=args.refine,
        seed=args.seed,
        per_facies=args.per_facies,
        upper=args.upper,
        objective=args.objective,
        min_hours=args.min_hours,
        max_hours=args.max_hours,
        timeout_minutes=args.timeout_minutes,
    )
    try:
        calibrate(args.config, settings, dry_run=args.dry_run, report_root=args.report_folder)
    except CalibrationError as error:
        print(f"Error: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
