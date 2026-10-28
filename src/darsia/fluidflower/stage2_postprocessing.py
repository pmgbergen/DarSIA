"""Stage 2: generic post-processing of stored per-frame maps for pH-indicator series.

Stage 1 (the existing DarSIA method, unchanged) analyses every photograph once and
stores per-frame products. Stage 2 reads stored arrays only and never opens a
photograph. For experiments whose colour signal is a BTB/NaOH indicator it turns
the indicator blue fraction f into dissolved inorganic carbon (DIC), flags, masses
and the per-cycle budget with the existing modules:

* ``titration.TitrationModel.dic_from_blue_fraction``: f -> DIC (mol/L);
* ``concentration_flags``: saturated interval, not-quantified mask, area fractions;
* ``titration.TitrationModel.mass_per_pixel_g``: DIC and pore volume -> grams;
* ``cycle_budget``: one budget row per evaluated frame of a cycle.

Input contract of one frame (``FrameInputs``; one grid, the shape of ``labels``):

* ``blue_fraction``: float, f in [0, 1], NaN marks invalid colour. Finite values
  outside [0, 1] and infinities are refused: mapping a colour into [0, 1] is the
  calibration's job, not stage 2's.
* ``labels``: integer label map; ``Stage2Config.quantified_labels`` are the sand
  labels that have a colour path.
* ``pore_volume_m3``: pore volume per pixel (porosity x depth x area). It must be
  finite and positive wherever a pixel is quantified; NaN is allowed elsewhere.
* ``valid_colour_mask``, ``gas_mask``, ``outside_mask`` (boolean) and
  ``pixel_area_m2``: each given explicitly, None when stage 1 has no such array.

A pixel is quantified when its label is sand, its colour is valid and it is neither
free gas nor outside. Every other pixel is not quantified: NaN in every map, left
out of every mass and counted in the not-quantified area, never as zero.

Gap (19 Oct 2026): stage 1 stores neither f nor any per-pixel colour coordinate.
Its ``products.npz`` holds ``concentration_aqueous`` (g/L, after the stage-1
calibration), ``saturation_free_gas``, ``phase_labels``, the mass maps and masks;
``native_geometry.npz`` holds ``pixel_area_m2``, ``depth_m``, ``porosity``,
``facies`` and ``transport_weight_l`` (pore volume per pixel, L). The colour-path
signal and the flash's normalised colour position exist only in memory during
stage 1. The colour -> f step is therefore an explicit ``BlueFractionAdapter``
that the per-series calibration supplies later. Stage 2 never builds or guesses
one, and it does not invert ``concentration_aqueous``: that would reuse the
stage-1 gain, dilution factor and clipping as a calibration.

Every method parameter is a field of ``Stage2Config`` without a default: the BTB
form and recipe, pKa, temperature, the two titration policies, f_min, the CO2
pressure of the solubility bound and the gas densities with their declared
reference conditions.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from os import PathLike
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from darsia.fluidflower.concentration_flags import (
    dissolved_interval,
    flag_fractions,
    not_quantified_mask,
)
from darsia.fluidflower.cycle_budget import (
    CycleBudgetInputs,
    budget_rows,
    free_gas_mass_g,
    injected_mass_g,
)
from darsia.fluidflower.titration import IndicatorRecipe, TitrationModel

__all__ = [
    "PORE_VOLUME_UNITS_TO_M3",
    "Stage2Config",
    "BlueFractionAdapter",
    "blue_fraction_from_colour",
    "FrameInputs",
    "MassSummary",
    "FrameSummary",
    "FrameResult",
    "process_frame",
    "BudgetTerms",
    "CycleResult",
    "process_cycle",
    "StoredField",
    "StoredFrameMapping",
    "load_npz_fields",
    "read_stage1_frame",
    "COLOUR_COORDINATE_FORMAT",
    "COLOUR_COORDINATE_ARCHIVE",
    "COLOUR_COORDINATE_ARRAY",
    "COLOUR_COORDINATE_RECORD",
    "read_colour_coordinate",
    "blue_fraction_from_stored_coordinate",
]

PORE_VOLUME_UNITS_TO_M3 = {"m3": 1.0, "L": 1.0e-3}
"""Declared unit of a stored pore-volume field. Stage 1 stores litres
(``native_geometry.npz:transport_weight_l``)."""


def _number(name: str, value) -> float:
    """A finite real number; None (missing), booleans and non-finite values fail."""
    if value is None:
        raise ValueError(f"{name} is required; stage 2 has no default for it")
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number, not {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _labels_tuple(values) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(
            "quantified_labels is required: a collection of integer labels"
        )
    labels = tuple(values)
    if not labels or any(
        isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in labels
    ):
        raise ValueError("quantified_labels needs at least one label, all integers")
    labels = tuple(int(x) for x in labels)
    if len(set(labels)) != len(labels):
        raise ValueError("quantified_labels has duplicates")
    return labels


# -- configuration ----------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Stage2Config:
    """All stage-2 parameters. No field has a default: the experimenter sets each.

    Indicator and titration (``TitrationModel``): ``naoh_g_per_l``,
    ``btb_g_per_l`` and ``btb_form`` ('sodium_salt' or 'free_acid') give the
    recipe; ``pka_indicator``, ``temperature_c``; ``blue_fraction_floor`` and
    ``clip_negative_dic`` are the model's two policies.

    Flags (``concentration_flags``): ``f_min`` is the blue fraction below which a
    pixel is saturated; ``p_co2_atm`` the CO2 pressure of the solubility (upper)
    bound; ``quantified_labels`` the labels with a colour path.

    Budget (``cycle_budget``): ``injection_gas_density_kg_per_m3`` converts the
    protocol's injected gas volume; ``free_gas_density_kg_per_m3`` and
    ``water_density_kg_per_m3`` convert the scale's displaced water to free gas;
    ``gas_density_reference`` declares their reference conditions and is written
    to every budget row.
    """

    naoh_g_per_l: float
    btb_g_per_l: float
    btb_form: str
    pka_indicator: float
    temperature_c: float
    blue_fraction_floor: float
    clip_negative_dic: bool
    f_min: float
    p_co2_atm: float
    quantified_labels: tuple[int, ...]
    injection_gas_density_kg_per_m3: float
    free_gas_density_kg_per_m3: float
    water_density_kg_per_m3: float
    gas_density_reference: str

    def __post_init__(self) -> None:
        densities = (
            "injection_gas_density_kg_per_m3",
            "free_gas_density_kg_per_m3",
            "water_density_kg_per_m3",
        )
        for name in (
            "naoh_g_per_l",
            "btb_g_per_l",
            "pka_indicator",
            "temperature_c",
            "blue_fraction_floor",
            "f_min",
            "p_co2_atm",
            *densities,
        ):
            object.__setattr__(self, name, _number(name, getattr(self, name)))
        for name in densities:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.temperature_c <= -273.15:
            raise ValueError("temperature_c must lie above absolute zero")
        if not isinstance(self.btb_form, str):
            raise ValueError("btb_form is required: 'sodium_salt' or 'free_acid'")
        if not isinstance(self.clip_negative_dic, bool):
            raise ValueError("clip_negative_dic is required and must be True or False")
        reference = self.gas_density_reference
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                "gas_density_reference must declare the reference conditions"
            )
        object.__setattr__(
            self, "quantified_labels", _labels_tuple(self.quantified_labels)
        )
        # The recipe, the model and the interval check their own inputs (form,
        # floor, f_min, pressure, threshold below solubility); running them once
        # here makes a bad configuration fail before the first frame.
        dissolved_interval(
            np.empty(0),
            self.titration_model(),
            f_min=self.f_min,
            p_co2_atm=self.p_co2_atm,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Stage2Config:
        """Build from a configuration table (e.g. parsed TOML). Every parameter is
        required; missing and unknown keys are refused together."""
        names = [spec.name for spec in fields(cls)]
        missing = [name for name in names if name not in values]
        unknown = sorted(str(key) for key in values if key not in names)
        if missing or unknown:
            raise ValueError(
                f"stage-2 configuration: missing {missing}, unknown {unknown}"
            )
        return cls(**{name: values[name] for name in names})

    def recipe(self) -> IndicatorRecipe:
        return IndicatorRecipe(
            naoh_g_per_l=self.naoh_g_per_l,
            btb_g_per_l=self.btb_g_per_l,
            form=self.btb_form,
        )

    def titration_model(self) -> TitrationModel:
        """The titration model with every argument from this configuration."""
        return TitrationModel(
            recipe=self.recipe(),
            temperature_c=self.temperature_c,
            pka_indicator=self.pka_indicator,
            blue_fraction_floor=self.blue_fraction_floor,
            clip_negative_dic=self.clip_negative_dic,
        )


# -- colour -> blue fraction ------------------------------------------------------
class BlueFractionAdapter(Protocol):
    """Per-series calibration from a stored colour coordinate to the blue fraction.

    Stage 2 defines only this interface; the series calibration supplies the
    implementation later. It receives the stored array as loaded and returns, on
    the frame's grid, a float array with f in [0, 1] and NaN where the colour is
    invalid.
    """

    def __call__(self, colour: np.ndarray, /) -> np.ndarray:
        """Blue fraction of each pixel of ``colour``."""


def _blue_fraction(value, shape: tuple[int, ...] | None) -> np.ndarray:
    f = np.asarray(value)
    if not np.issubdtype(f.dtype, np.floating):
        raise ValueError(
            "the blue fraction must be a float array (NaN marks invalid colour)"
        )
    if shape is not None and f.shape != shape:
        raise ValueError(
            f"blue fraction shape {f.shape} differs from the label grid {shape}"
        )
    f = f.astype(np.float64)
    if np.isinf(f).any():
        raise ValueError("the blue fraction must be finite or NaN, never infinite")
    outside = (f < 0.0) | (f > 1.0)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} blue fractions lie outside [0, 1]; the adapter must "
            "map the colour into [0, 1] or to NaN"
        )
    return f


def blue_fraction_from_colour(colour, adapter: BlueFractionAdapter) -> np.ndarray:
    """Apply the series' adapter to a stored colour coordinate and enforce its
    output contract (float, f in [0, 1], NaN marks invalid colour)."""
    if not callable(adapter):
        raise ValueError("a blue-fraction adapter (the series calibration) is required")
    return _blue_fraction(adapter(np.asarray(colour)), None)


# -- one frame -----------------------------------------------------------------------
def _mask(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f"{name} must be a boolean mask on the label grid {shape}")
    return array


def _geometry(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    """Per-pixel geometry: float on the grid, never infinite or negative. NaN
    (unknown) passes here and fails later wherever a pixel needs the value."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating) or array.shape != shape:
        raise ValueError(f"{name} must be a float array on the label grid {shape}")
    array = array.astype(np.float64)
    if np.isinf(array).any() or (array < 0.0).any():
        raise ValueError(f"{name} must not be infinite or negative")
    return array


@dataclass(frozen=True, kw_only=True, eq=False)
class FrameInputs:
    """One stored frame in the stage-2 input contract (see the module docstring).

    Every field must be given: pass None for a mask or for pixel areas that the
    stage-1 products do not provide. The checks that need the configuration
    (pore volume at quantified pixels, pixel areas on the sand) run in
    ``process_frame``.
    """

    frame_id: str
    blue_fraction: np.ndarray
    labels: np.ndarray
    pore_volume_m3: np.ndarray
    valid_colour_mask: np.ndarray | None
    gas_mask: np.ndarray | None
    outside_mask: np.ndarray | None
    pixel_area_m2: np.ndarray | None

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a nonempty string")
        labels = np.asarray(self.labels)
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be a 2D integer label map")
        shape = labels.shape
        object.__setattr__(self, "labels", labels)
        object.__setattr__(
            self, "blue_fraction", _blue_fraction(self.blue_fraction, shape)
        )
        object.__setattr__(
            self,
            "pore_volume_m3",
            _geometry("pore_volume_m3", self.pore_volume_m3, shape),
        )
        if self.pixel_area_m2 is not None:
            object.__setattr__(
                self,
                "pixel_area_m2",
                _geometry("pixel_area_m2", self.pixel_area_m2, shape),
            )
        for name in ("valid_colour_mask", "gas_mask", "outside_mask"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _mask(name, getattr(self, name), shape))


@dataclass(frozen=True)
class MassSummary:
    """Dissolved mass of a pixel selection as the saturated interval, with its
    quantified, saturated and not-quantified pixels. A selection without any
    quantified pixel has an unknown (NaN) mass, never zero; the not-quantified
    area is NaN without pixel areas."""

    mass_lower_g: float
    mass_upper_g: float
    pixels: int
    quantified_pixels: int
    saturated_pixels: int
    not_quantified_pixels: int
    not_quantified_area_m2: float


@dataclass(frozen=True)
class FrameSummary:
    """Scalar results of one frame: the sand total, every label present, every
    region (its sand pixels) and the area fractions of ``flag_fractions``."""

    frame_id: str
    total: MassSummary
    by_label: dict[int, MassSummary]
    by_region: dict[str, MassSummary]
    saturated_fraction: float
    not_quantified_fraction: float
    sand_pixels: int


@dataclass(frozen=True, eq=False)
class FrameResult:
    """Per-pixel maps of one frame and its summary; NaN marks not quantified.

    ``dic_mol_per_l`` is the closed-form DIC where the colour measures it
    (quantified and unsaturated) and NaN elsewhere; saturated pixels carry only
    the interval ``dic_lower_mol_per_l`` to ``dic_upper_mol_per_l``.
    """

    summary: FrameSummary
    dic_mol_per_l: np.ndarray
    dic_lower_mol_per_l: np.ndarray
    dic_upper_mol_per_l: np.ndarray
    mass_lower_g: np.ndarray
    mass_upper_g: np.ndarray
    saturated: np.ndarray
    not_quantified: np.ndarray


def _mass_g(dic_mol_per_l, pore_volume_m3):
    """Grams per pixel. ``mass_per_pixel_g`` multiplies porosity, depth and area;
    the stored pore volume is their product, so it enters as one factor."""
    return TitrationModel.mass_per_pixel_g(dic_mol_per_l, 1.0, 1.0, pore_volume_m3)


def _summarise(
    selection, quantified, saturated, mass_lower, mass_upper, area
) -> MassSummary:
    counted = selection & quantified
    missing = selection & ~quantified
    n = int(counted.sum())
    return MassSummary(
        mass_lower_g=float(mass_lower[counted].sum()) if n else math.nan,
        mass_upper_g=float(mass_upper[counted].sum()) if n else math.nan,
        pixels=int(selection.sum()),
        quantified_pixels=n,
        saturated_pixels=int((selection & saturated).sum()),
        not_quantified_pixels=int(missing.sum()),
        not_quantified_area_m2=math.nan if area is None else float(area[missing].sum()),
    )


def process_frame(
    config: Stage2Config,
    frame: FrameInputs,
    *,
    regions: Mapping[str, np.ndarray] | None = None,
) -> FrameResult:
    """DIC, flags and masses of one stored frame.

    ``regions`` are named boolean masks (e.g. boxes); a region summary covers the
    sand pixels inside it. Label summaries cover every label present in the map.
    """
    if not isinstance(config, Stage2Config) or not isinstance(frame, FrameInputs):
        raise ValueError("process_frame needs a Stage2Config and FrameInputs")
    shape = frame.labels.shape
    checked = {}
    for name, region in (regions or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("region names must be nonempty strings")
        checked[name] = _mask(f"region {name!r}", region, shape)
    f = frame.blue_fraction
    sand = np.isin(frame.labels, config.quantified_labels)
    invalid_colour = np.isnan(f)
    if frame.valid_colour_mask is not None:
        invalid_colour |= ~frame.valid_colour_mask
    not_quantified = not_quantified_mask(
        frame.labels,
        config.quantified_labels,
        gas_mask=frame.gas_mask,
        outside_mask=frame.outside_mask,
        invalid_colour_mask=invalid_colour,
    )
    quantified = ~not_quantified
    volume = frame.pore_volume_m3
    no_volume = quantified & ~(volume > 0.0)  # NaN fails the comparison
    if no_volume.any():
        raise ValueError(
            f"{int(no_volume.sum())} quantified pixels have no finite positive pore "
            "volume; mark them outside in the stage-1 inputs"
        )
    area = frame.pixel_area_m2
    if area is not None and (sand & ~(area > 0.0)).any():
        raise ValueError("pixel_area_m2 must be finite and positive on the sand")
    interval = dissolved_interval(
        f,
        config.titration_model(),
        f_min=config.f_min,
        p_co2_atm=config.p_co2_atm,
        valid_mask=quantified,
    )
    saturated = interval.saturated
    # Where unsaturated, the lower bound is the closed-form DIC of f itself.
    dic = np.where(saturated, np.nan, interval.lower)
    mass_lower = np.where(quantified, _mass_g(interval.lower, volume), np.nan)
    mass_upper = np.where(quantified, _mass_g(interval.upper, volume), np.nan)

    def summary(selection):
        return _summarise(
            selection, quantified, saturated, mass_lower, mass_upper, area
        )

    fractions = flag_fractions(saturated, not_quantified, sand)
    frame_summary = FrameSummary(
        frame_id=frame.frame_id,
        total=summary(sand),
        by_label={
            int(label): summary(frame.labels == label)
            for label in np.unique(frame.labels)
        },
        by_region={name: summary(region & sand) for name, region in checked.items()},
        saturated_fraction=fractions["saturated_fraction"],
        not_quantified_fraction=fractions["not_quantified_fraction"],
        sand_pixels=fractions["sand_pixels"],
    )
    return FrameResult(
        summary=frame_summary,
        dic_mol_per_l=dic,
        dic_lower_mol_per_l=interval.lower,
        dic_upper_mol_per_l=interval.upper,
        mass_lower_g=mass_lower,
        mass_upper_g=mass_upper,
        saturated=saturated,
        not_quantified=not_quantified,
    )


# -- one cycle -----------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class BudgetTerms:
    """Caller-supplied budget inputs at one frame time of a cycle.

    ``injection_intervals`` lists (rate mL/min, duration min) of the gas injected
    in this cycle up to the frame time, one pair per interval and port; stage 2
    converts it with ``Stage2Config.injection_gas_density_kg_per_m3``.
    ``displaced_water_g`` is the water displaced from the rig by this cycle's free
    gas at the frame time (scale record, relative to the cycle baseline) or None
    without a scale record. ``left_overflow_g`` is the declared dissolved CO2 that
    left with the overflow, or None when unknown. Unknown terms become NaN,
    never 0.
    """

    injection_intervals: tuple[tuple[float, float], ...]
    displaced_water_g: float | None
    left_overflow_g: float | None
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        intervals = self.injection_intervals
        if isinstance(intervals, (str, bytes)) or not isinstance(intervals, Iterable):
            raise ValueError(
                "injection_intervals lists (rate mL/min, duration min) pairs"
            )
        checked = []
        for pair in intervals:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    "each injection interval is a (rate mL/min, duration min) pair"
                )
            rate = _number("injection rate", pair[0])
            duration = _number("injection duration", pair[1])
            if rate < 0.0 or duration < 0.0:
                raise ValueError("injection rate and duration must be non-negative")
            checked.append((rate, duration))
        if not checked:
            raise ValueError("injection_intervals needs at least one interval")
        object.__setattr__(self, "injection_intervals", tuple(checked))
        flags = self.quality_flags
        if not isinstance(flags, (tuple, list)) or not all(
            isinstance(flag, str) for flag in flags
        ):
            raise ValueError("quality_flags must be a tuple of strings")
        object.__setattr__(self, "quality_flags", tuple(flags))


def _unknown_as_nan(value: float | None) -> float:
    """None (unknown) becomes NaN; ``cycle_budget`` checks every other value."""
    return math.nan if value is None else value


@dataclass(frozen=True)
class CycleResult:
    """Budget rows of one cycle (one per evaluated frame) and the frame summaries."""

    cycle_run: str
    baseline: FrameSummary
    frames: tuple[FrameSummary, ...]
    rows: tuple[dict, ...]


def process_cycle(
    config: Stage2Config,
    *,
    cycle_run: str,
    baseline: FrameInputs,
    frames: Iterable[tuple[FrameInputs, BudgetTerms]],
    regions: Mapping[str, np.ndarray] | None = None,
) -> CycleResult:
    """Per-cycle budget rows with ``cycle_budget``, one row per evaluated frame.

    ``baseline`` is the cycle baseline (the frame before the cycle's injection).
    ``frames`` yields (frame, terms) pairs and may be a generator: frames are
    processed one at a time and only the baseline maps are kept.

    The new dissolved mass is frame minus baseline, pixel by pixel, over the pixels
    quantified in both, as the interval [frame lower - baseline upper, frame upper
    - baseline lower]. The background is the baseline's dissolved mass over its own
    quantified pixels: ``background_dissolved_g`` holds it when the baseline has no
    saturated pixel and NaN otherwise; its bounds are extra row columns. A frame
    without any pixel quantified in both has an unknown new dissolved mass, so the
    budget cannot close and the frame is refused.
    """
    if not isinstance(cycle_run, str) or not cycle_run.strip():
        raise ValueError("cycle_run must be a nonempty string")
    if not isinstance(baseline, FrameInputs):
        raise ValueError("baseline must be FrameInputs")
    base = process_frame(config, baseline, regions=regions)
    base_quantified = ~base.not_quantified
    background = base.summary.total
    exact = background.mass_lower_g == background.mass_upper_g
    background_g = background.mass_lower_g if exact else math.nan
    inputs, extras, summaries, seen = [], [], [], set()
    for item in frames:
        if not (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], FrameInputs)
            and isinstance(item[1], BudgetTerms)
        ):
            raise ValueError("frames must yield (FrameInputs, BudgetTerms) pairs")
        frame, terms = item
        if frame.frame_id in seen:
            raise ValueError(
                f"frame {frame.frame_id!r} appears twice in cycle {cycle_run!r}"
            )
        seen.add(frame.frame_id)
        if frame.labels.shape != baseline.labels.shape:
            raise ValueError("every frame of a cycle must share the baseline grid")
        result = process_frame(config, frame, regions=regions)
        common = base_quantified & ~result.not_quantified
        support = int(common.sum())
        if support == 0:
            raise ValueError(
                f"frame {frame.frame_id!r}: no pixel is quantified in both the frame and "
                "the cycle baseline, so the new dissolved mass is unknown"
            )
        new_lower = float(
            (result.mass_lower_g[common] - base.mass_upper_g[common]).sum()
        )
        new_upper = float(
            (result.mass_upper_g[common] - base.mass_lower_g[common]).sum()
        )
        density = config.injection_gas_density_kg_per_m3
        injected = math.fsum(
            injected_mass_g(rate, duration, density)
            for rate, duration in terms.injection_intervals
        )
        free_gas = free_gas_mass_g(
            _unknown_as_nan(terms.displaced_water_g),
            config.free_gas_density_kg_per_m3,
            water_density_kg_per_m3=config.water_density_kg_per_m3,
        )
        inputs.append(
            CycleBudgetInputs(
                cycle_run=cycle_run,
                injected_g=injected,
                dissolved_new_lower_g=new_lower,
                dissolved_new_upper_g=new_upper,
                free_gas_g=free_gas,
                left_overflow_g=_unknown_as_nan(terms.left_overflow_g),
                background_dissolved_g=background_g,
                quality_flags=terms.quality_flags,
            )
        )
        summary = result.summary
        extras.append(
            dict(
                frame_id=summary.frame_id,
                gas_density_reference=config.gas_density_reference,
                injected_volume_ml=math.fsum(
                    r * d for r, d in terms.injection_intervals
                ),
                background_dissolved_lower_g=background.mass_lower_g,
                background_dissolved_upper_g=background.mass_upper_g,
                frame_dissolved_lower_g=summary.total.mass_lower_g,
                frame_dissolved_upper_g=summary.total.mass_upper_g,
                new_dissolved_support_pixels=support,
                saturated_fraction=summary.saturated_fraction,
                not_quantified_fraction=summary.not_quantified_fraction,
                sand_pixels=summary.sand_pixels,
            )
        )
        summaries.append(summary)
    if not inputs:
        raise ValueError("a cycle needs at least one frame after its baseline")
    rows = budget_rows(inputs)
    for row, extra in zip(rows, extras, strict=True):
        row.update(extra)
    return CycleResult(cycle_run, base.summary, tuple(summaries), tuple(rows))


# -- reading stage-1 products --------------------------------------------------------
_ARRAY_ROLES = ("colour", "labels", "pore_volume", "pixel_area_m2")
_MASK_ROLES = ("valid_colour", "gas", "outside")


@dataclass(frozen=True)
class StoredField:
    """One stored array: the key of its archive and its name in that archive.

    ``true_values`` is for mask roles only. None: the stored array is the boolean
    mask itself. A tuple: the mask is True where the stored value is one of them,
    e.g. ``(2,)`` for a gas category of a phase-label map or ``(False,)`` to
    invert a stored domain mask.
    """

    archive: str
    name: str
    true_values: tuple[int, ...] | tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        for attribute in ("archive", "name"):
            value = getattr(self, attribute)
            if not isinstance(value, str) or not value:
                raise ValueError(f"StoredField.{attribute} must be a nonempty string")
        if self.true_values is not None:
            values = tuple(self.true_values)
            booleans = all(isinstance(v, (bool, np.bool_)) for v in values)
            integers = all(
                isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                for v in values
            )
            if not values or not (booleans or integers):
                raise ValueError(
                    "StoredField.true_values must be all booleans or all integers"
                )
            object.__setattr__(self, "true_values", values)


def _sources(value) -> tuple[StoredField, ...]:
    if value is None:
        return ()
    return value if isinstance(value, tuple) else (value,)


@dataclass(frozen=True, kw_only=True)
class StoredFrameMapping:
    """Which stored array feeds each stage-2 input. Every role must be given, None
    when the products have no such array; no field is looked up by a default name.

    ``colour`` is passed to the adapter as stored; ``pore_volume`` is converted
    from the declared ``pore_volume_unit``; ``pixel_area_m2`` must be stored in m2.
    A mask role takes one field or a tuple of fields, combined as a union.
    """

    colour: StoredField
    labels: StoredField
    pore_volume: StoredField
    pore_volume_unit: str
    valid_colour: StoredField | tuple[StoredField, ...] | None
    gas: StoredField | tuple[StoredField, ...] | None
    outside: StoredField | tuple[StoredField, ...] | None
    pixel_area_m2: StoredField | None

    def __post_init__(self) -> None:
        if self.pore_volume_unit not in PORE_VOLUME_UNITS_TO_M3:
            raise ValueError(
                f"pore_volume_unit must be one of {sorted(PORE_VOLUME_UNITS_TO_M3)}"
            )
        for role in _ARRAY_ROLES:
            source = getattr(self, role)
            if source is None and role != "pixel_area_m2":
                raise ValueError(f"the {role} role is required")
            if source is not None and (
                not isinstance(source, StoredField) or source.true_values is not None
            ):
                raise ValueError(f"{role} must be one StoredField without true_values")
        for role in _MASK_ROLES:
            sources = _sources(getattr(self, role))
            if getattr(self, role) is not None and (
                not sources or not all(isinstance(s, StoredField) for s in sources)
            ):
                raise ValueError(
                    f"{role} must be a StoredField, a tuple of them or None"
                )


def load_npz_fields(
    path: str | PathLike, names: Mapping[str, str]
) -> dict[str, np.ndarray]:
    """Load exactly the named arrays of one .npz archive, as ``{key: stored name}``.

    No other member of the archive is read, pickled objects are refused and a
    missing name raises ValueError (listing what the archive holds).
    """
    if not names:
        raise ValueError("no stored field names were given")
    if not Path(path).is_file():
        raise ValueError(f"{path} does not exist; the stored fields are absent")
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"{path} is not an .npz archive")
    with loaded as archive:
        missing = sorted({name for name in names.values() if name not in archive.files})
        if missing:
            held = sorted(archive.files)
            raise ValueError(
                f"{path}: stored field(s) {missing} not found; it holds {held}"
            )
        return {key: archive[name] for key, name in names.items()}


def _stored_mask(role: str, array: np.ndarray, source: StoredField) -> np.ndarray:
    values = source.true_values
    boolean = array.dtype == np.bool_
    if values is None:
        if not boolean:
            raise ValueError(
                f"{role}: {source.archive}:{source.name} is {array.dtype}, not boolean; "
                "declare its true_values"
            )
        return array
    categorical = boolean or np.issubdtype(array.dtype, np.integer)
    if not categorical or isinstance(values[0], (bool, np.bool_)) != boolean:
        raise ValueError(
            f"{role}: true_values {values} do not match {source.archive}:{source.name} "
            f"({array.dtype})"
        )
    return np.isin(array, np.asarray(values))


def read_stage1_frame(
    archives: Mapping[str, str | PathLike],
    mapping: StoredFrameMapping,
    *,
    frame_id: str,
    adapter: BlueFractionAdapter,
) -> FrameInputs:
    """Stage-2 inputs of one frame from stored stage-1 archives.

    ``archives`` maps the archive keys used in ``mapping`` to paths, e.g. one
    frame's ``products.npz`` and ``native_geometry.npz``. Only the mapped fields
    are loaded. The stored colour goes through ``adapter``; if the stored field
    already is the blue fraction, the adapter is the identity, stated by the
    caller.
    """
    if not isinstance(mapping, StoredFrameMapping):
        raise ValueError("mapping must be a StoredFrameMapping")
    requests = [
        (role, role, getattr(mapping, role))
        for role in _ARRAY_ROLES
        if getattr(mapping, role) is not None
    ]
    for role in _MASK_ROLES:
        for index, source in enumerate(_sources(getattr(mapping, role))):
            requests.append((f"{role}:{index}", role, source))
    unknown = sorted({source.archive for _, _, source in requests} - set(archives))
    if unknown:
        raise ValueError(f"no path was given for archive(s) {unknown}")
    arrays: dict[str, np.ndarray] = {}
    for archive in sorted({source.archive for _, _, source in requests}):
        names = {key: s.name for key, _, s in requests if s.archive == archive}
        arrays.update(load_npz_fields(archives[archive], names))
    masks: dict[str, np.ndarray | None] = {role: None for role in _MASK_ROLES}
    for key, role, source in requests:
        if role in _MASK_ROLES:
            mask = _stored_mask(role, arrays[key], source)
            masks[role] = mask if masks[role] is None else masks[role] | mask
    pore = arrays["pore_volume"]
    if not np.issubdtype(pore.dtype, np.floating):
        raise ValueError("the stored pore volume must be a float array")
    return FrameInputs(
        frame_id=frame_id,
        blue_fraction=blue_fraction_from_colour(arrays["colour"], adapter),
        labels=arrays["labels"],
        pore_volume_m3=pore.astype(np.float64)
        * PORE_VOLUME_UNITS_TO_M3[mapping.pore_volume_unit],
        valid_colour_mask=masks["valid_colour"],
        gas_mask=masks["gas"],
        outside_mask=masks["outside"],
        pixel_area_m2=arrays.get("pixel_area_m2"),
    )


# -- stored colour-path coordinate of stage 1 ---------------------------------------
# Stage 1 writes it only with the storage option store_colour_coordinate (default
# off): <frame>/colour_path_coordinate.npz, one float32 array on the native grid,
# NaN outside the colour-quantifiable support, recorded in the frame's result.json.
# These names equal the constants of darsia.fluidflower.storage_profile.
COLOUR_COORDINATE_FORMAT = "darsia-colour-path-coordinate-v1"
COLOUR_COORDINATE_ARCHIVE = "colour_path_coordinate.npz"
COLOUR_COORDINATE_ARRAY = "colour_path_coordinate"
COLOUR_COORDINATE_RECORD = "colour_coordinate"


def read_colour_coordinate(frame_directory: str | PathLike) -> np.ndarray:
    """Load a stage-1 frame's stored colour-path coordinate, verified.

    The frame's result.json must record it; the archive must match the recorded
    SHA-256 and size, and only the recorded array is read, from the verified
    bytes. A frame without the coordinate, a missing or changed archive and an
    array other than float32 on a 2D grid raise ValueError.
    """
    directory = Path(frame_directory)
    try:
        result = json.loads((directory / "result.json").read_bytes())
    except FileNotFoundError:
        raise ValueError(f"{directory} holds no stage-1 result.json") from None
    record = result.get(COLOUR_COORDINATE_RECORD) if isinstance(result, dict) else None
    if record is None:
        raise ValueError(
            f"{directory}: the frame stores no colour-path coordinate; stage 1 must run "
            "with the storage option store_colour_coordinate"
        )
    if (
        not isinstance(record, dict)
        or record.get("format") != COLOUR_COORDINATE_FORMAT
        or record.get("path") != COLOUR_COORDINATE_ARCHIVE
        or record.get("array") != COLOUR_COORDINATE_ARRAY
    ):
        raise ValueError(f"{directory}: unknown colour-path coordinate record")
    path = directory / COLOUR_COORDINATE_ARCHIVE
    if not path.is_file():
        raise ValueError(f"{path} is missing although result.json records it")
    data = path.read_bytes()
    if len(data) != record.get("size_bytes") or (
        hashlib.sha256(data).hexdigest() != record.get("sha256")
    ):
        raise ValueError(f"{path} differs from its recorded SHA-256 or size")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if COLOUR_COORDINATE_ARRAY not in archive.files:
            raise ValueError(f"{path} holds no {COLOUR_COORDINATE_ARRAY} array")
        value = archive[COLOUR_COORDINATE_ARRAY]
    identity = record.get("identity") or {}
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or identity.get("dtype") != value.dtype.str
        or identity.get("shape") != list(value.shape)
    ):
        raise ValueError(
            f"{path}: the coordinate must be the recorded float32 2D array"
        )
    return value


def blue_fraction_from_stored_coordinate(
    frame_directory: str | PathLike, adapter: BlueFractionAdapter
) -> np.ndarray:
    """Blue fraction of a stage-1 frame from its stored colour-path coordinate.

    The coordinate is read with ``read_colour_coordinate`` and passed through
    the per-series ``adapter`` (the calibration, supplied by the caller), whose
    output contract is enforced by ``blue_fraction_from_colour``.
    """
    return blue_fraction_from_colour(read_colour_coordinate(frame_directory), adapter)
