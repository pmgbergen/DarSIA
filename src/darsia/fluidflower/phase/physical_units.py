"""Pixel-sized phase parameters defined in physical units.

The frozen phase parameters store several rule and operator sizes as pixels of
the nominal native raster, whose pitch is ``decisions.nominal_mm_per_pixel``.
:data:`PIXEL_RULES` gives each such entry a physical definition and the
rounding its DATA value was derived with. A ``*_mm`` key of the DATA is the
definition where one exists; every other entry is defined as its native pixel
value times the nominal pitch.

:func:`parameters_on_grid` recomputes all entries for a computing grid whose
cells span ``pixels_per_cell`` native raster pixels along each axis. A rule
therefore keeps its extent on the rig at every resolution. The grid's own
nominal pitch, pixel areas and spacings follow the same factor. With one
pixel per cell the result equals the DATA exactly; the tests check this
against the bound values.

Native runs do not call this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re

PITCH = ("decisions", "nominal_mm_per_pixel")
ROUNDINGS = ("round", "floor", "float", "odd_window", "halo")


@dataclass(frozen=True)
class PixelRule:
    """One pixel-sized parameter, its physical definition and its rounding.

    ``path`` locates the value in the phase parameters. ``definition`` locates
    its length in nominal millimetres, or is None when the length is the
    native pixel value itself. ``rounding`` is one of :data:`ROUNDINGS`:
    ``round`` and ``floor`` give integer pixels as the DATA did, ``float``
    keeps fractional pixels, ``odd_window`` keeps an odd window (or a list of
    them) with the same half-width, and ``halo`` is the half-width of the
    window that ``definition`` names.
    """

    path: tuple
    definition: tuple | None
    rounding: str
    note: str


def _rule(path, definition, rounding, note):
    """Build a PixelRule from dotted paths."""
    return PixelRule(tuple(path.split(".")), None if definition is None else tuple(definition.split(".")),
                     rounding, note)


PIXEL_RULES = (
    _rule("decisions.top.band_pixels", "decisions.top.band_mm", "round", "top band"),
    _rule("decisions.top.column_gate_pixels", "decisions.top.column_gate_mm", "floor",
          "top acting-column gate"),
    _rule("decisions.top.source_rule.band_pixels", "decisions.top.source_rule.band_mm", "round",
          "recorded top band"),
    _rule("decisions.top.source_rule.column_gate_pixels", "decisions.top.source_rule.column_gate_mm",
          "floor", "recorded top acting-column gate"),
    _rule("decisions.floor.column_gate_pixels", "decisions.floor.column_gate_mm", "floor",
          "floor acting-column gate"),
    _rule("decisions.floor.floor_band_pixels", "decisions.floor.floor_band_mm", "floor", "floor band"),
    _rule("decisions.floor.connectivity_disc_radius_pixels",
          "decisions.floor.connectivity_disc_radius_mm", "float",
          "recorded floor connectivity disc; the rule divides its millimetres by the pitch"),
    _rule("decisions.floor.source_rule.column_gate_pixels", "decisions.floor.source_rule.column_gate_mm",
          "floor", "recorded floor acting-column gate"),
    _rule("decisions.floor.source_rule.floor_band_pixels", "decisions.floor.source_rule.floor_band_mm",
          "floor", "recorded floor band"),
    _rule("decisions.floor.source_rule.connectivity_disc_radius_pixels",
          "decisions.floor.source_rule.connectivity_disc_radius_mm", "float",
          "recorded floor connectivity disc"),
    _rule("decisions.floor.source_rule.short_fill_limit_pixels",
          "decisions.floor.source_rule.short_fill_limit_mm", "floor", "recorded short contact fill"),
    _rule("decisions.floor.source_rule.envelope_radius_pixels",
          "decisions.floor.source_rule.envelope_radius_mm", "float", "recorded gas-envelope radius"),
    _rule("decisions.contact.short_fill_limit_pixels", "decisions.contact.short_fill_limit_mm", "floor",
          "short contact-fill span"),
    _rule("decisions.envelopes.radius_pixels", "decisions.envelopes.radius_mm", "float",
          "recorded gas-envelope radius; the rule divides its millimetres by the pitch"),
    _rule("decisions.placement.fraction_kernel_hw", None, "odd_window", "P2 neighbourhood box"),
    _rule("features.optical.texture_sigma", None, "float", "texture Gaussian sigma"),
    _rule("features.flatfield.box_size", None, "odd_window", "flat-field box mean"),
    _rule("features.flatfield.halo_rows", "features.flatfield.box_size", "halo",
          "tile halo of the flat-field box"),
    _rule("feature_support.texture_footprint_hw", None, "odd_window", "feature-support erosion"),
)
#: The grid's own nominal pitch, pixel areas and spacings; they scale with it.
GRID_PITCHES = (PITCH, ("decisions", "floor", "source_rule", "nominal_mm_per_phase_pixel"))
GRID_AREAS = (
    ("decisions", "nominal_pixel_area_mm2"),
    ("decisions", "top", "source_rule", "nominal_pixel_area_mm2"),
    ("decisions", "floor", "source_rule", "nominal_pixel_area_mm2"),
    ("decisions", "barrier", "diagnostic_pixel_area_mm2"),
    ("domain", "nominal_rule_b_pixel_area_mm2"),
)
GRID_SPACINGS = (("chain", "rig_spacing_m"), ("chain", "probability_spacing_m"))
#: The spatial context sigma is a fraction of the optical feature shape.
CONTEXT_FRACTION = ("features", "optical", "spatial_context_fraction")
#: Pixel-like keys that stay unchanged on another grid, with the reason.
UNCHANGED = {
    "component_structure_hw": "neighbourhood adjacency (connectivity), not a distance",
    "carryover_structure_hw": "one-neighbour adjacency of an absent carryover mask",
    "top_mask_structure_hw": "neighbourhood adjacency (connectivity), not a distance",
    "apply_top_rule_structure_hw": "neighbourhood adjacency of a QA closure check",
    "label_structure_hw": "neighbourhood adjacency (connectivity), not a distance",
    "floor_core_structure_hw": "one-neighbour adjacency of an absent carryover mask",
    "guard_px": "native photo support, built on the native raster before coarsening",
    "hole_max_area_px": "native photo support, built on the native raster before coarsening",
    "vertical_gap_max_px": "native photo support, built on the native raster before coarsening",
    "grid_shape_hw": "native photo support grid",
    "working_spacing_m": "regional working grid, already in metres",
    "native_shape_hw": "grid geometry, set by the caller that builds the grid inputs",
}
_PIXEL_KEY = re.compile(r"(_pixels?|_px|_hw|_pixel_area_mm2|mm_per_pixel|mm_per_phase_pixel"
                        r"|texture_sigma|box_size|halo_rows|_spacing_m|spatial_context_fraction)$")


def _get(tree, path):
    """The value at a key path of nested mappings."""
    for key in path:
        tree = tree[key]
    return tree


def _set(tree, path, value):
    """Replace the value at a key path of nested mappings."""
    for key in path[:-1]:
        tree = tree[key]
    tree[path[-1]] = value


def _has(tree, path):
    """Whether nested mappings hold a key path."""
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return False
        tree = tree[key]
    return True


def _factor(value, name):
    """A positive finite number, or a ValueError naming it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a positive finite number")
    return float(value)


def native_length_pixels(rule, parameters):
    """The rule's length in native raster pixels, before any rounding.

    A millimetre definition is divided by the nominal pitch. A window is its
    half-width; a pixel-only value is itself.
    """
    if rule.rounding == "halo":
        raise ValueError("A halo follows its window; it has no own length")
    if rule.definition is not None:
        return _factor(_get(parameters, rule.definition), "/".join(rule.definition)) / _factor(
            _get(parameters, PITCH), "nominal pitch")
    value = _get(parameters, rule.path)
    if rule.rounding == "odd_window":
        sizes = value if isinstance(value, list) else [value]
        if any(type(size) is not int or size < 1 or size % 2 != 1 for size in sizes):
            raise ValueError("Window " + "/".join(rule.path) + " must hold odd positive sizes")
        return [(size - 1) / 2 for size in sizes] if isinstance(value, list) else (value - 1) / 2
    return _factor(value, "/".join(rule.path))


def grid_value(rule, length, pixels_per_cell):
    """Express one native length on a grid with ``pixels_per_cell`` pixels per cell.

    ``length`` is :func:`native_length_pixels` of the rule. Dividing by a
    factor of exactly one leaves the length unchanged, so the native grid
    reproduces the DATA.
    """
    factor = _factor(pixels_per_cell, "Pixels per cell")
    if rule.rounding == "round":
        return round(length / factor)
    if rule.rounding == "floor":
        return math.floor(length / factor)
    if rule.rounding == "float":
        return length / factor
    if rule.rounding == "odd_window":
        if isinstance(length, list):
            return [2 * round(half / factor) + 1 for half in length]
        return 2 * round(length / factor) + 1
    raise ValueError("Unknown pixel rounding: " + str(rule.rounding))


def pixel_sized_paths(parameters):
    """Every leaf whose key names a pixel size, area, pitch or spacing."""
    found = []

    def walk(tree, path):
        """Collect matching leaves below one mapping."""
        for key, value in tree.items():
            if isinstance(value, dict):
                walk(value, (*path, key))
            elif _PIXEL_KEY.search(key):
                found.append((*path, key))
    walk(parameters, ())
    return found


def require_covered(parameters):
    """Fail on a pixel-sized key that this module neither converts nor exempts.

    A new pixel parameter must be added to :data:`PIXEL_RULES`, to the grid
    entries, or to :data:`UNCHANGED` with its reason before a grid of another
    pitch may use the parameters.
    """
    known = {rule.path for rule in PIXEL_RULES}
    known.update(GRID_PITCHES, GRID_AREAS, GRID_SPACINGS, [CONTEXT_FRACTION])
    missing = [path for path in pixel_sized_paths(parameters)
               if path not in known and path[-1] not in UNCHANGED]
    if missing:
        raise ValueError("Pixel-sized phase parameters lack a physical definition: "
                         + ", ".join(".".join(path) for path in missing))


def parameters_on_grid(parameters, *, pixels_per_cell, feature_shape_hw):
    """Return the phase parameters for a grid of ``pixels_per_cell`` native pixels.

    Every :data:`PIXEL_RULES` entry is recomputed from its physical definition.
    The nominal pitch, the nominal pixel areas and the spacings scale with the
    factor. The spatial context fraction is rescaled so that its sigma keeps
    its length when the optical feature shape becomes ``feature_shape_hw``.
    The input is not modified. One pixel per cell and the native feature shape
    return values equal to the DATA.
    """
    factor = _factor(pixels_per_cell, "Pixels per cell")
    require_covered(parameters)
    result = deepcopy(parameters)
    for rule in PIXEL_RULES:
        if not _has(parameters, rule.path):
            continue
        if rule.rounding == "halo":
            window = _get(result, rule.definition)
            if type(window) is not int or window < 1 or window % 2 != 1:
                raise ValueError("A halo needs its converted odd window")
            _set(result, rule.path, (window - 1) // 2)
            continue
        _set(result, rule.path, grid_value(rule, native_length_pixels(rule, parameters), factor))
    for path in GRID_PITCHES:
        if _has(parameters, path):
            _set(result, path, _factor(_get(parameters, path), "/".join(path)) * factor)
    for path in GRID_AREAS:
        if _has(parameters, path):
            _set(result, path, _factor(_get(parameters, path), "/".join(path)) * (factor * factor))
    for path in GRID_SPACINGS:
        if _has(parameters, path):
            _set(result, path, [_factor(value, "/".join(path)) * factor for value in _get(parameters, path)])
    if _has(parameters, CONTEXT_FRACTION):
        native = _has(parameters, ("geometry", "native_shape_hw"))
        if not native:
            raise ValueError("The spatial context needs the native optical feature shape")
        native_min = min(map(int, _get(parameters, ("geometry", "native_shape_hw"))))
        grid_min = min(map(int, feature_shape_hw))
        if native_min < 1 or grid_min < 1:
            raise ValueError("Optical feature shapes must be positive")
        fraction = _factor(_get(parameters, CONTEXT_FRACTION), "spatial context fraction")
        _set(result, CONTEXT_FRACTION, fraction * ((native_min / grid_min) / factor))
    return result


def conversion_report(parameters, converted, *, pixels_per_cell):
    """List each converted entry with its definition, DATA and grid values.

    The rows are the provenance of a coarse run: what each pixel parameter
    means physically and which value the grid used.
    """
    pitch = _factor(_get(parameters, PITCH), "nominal pitch")
    rows = []
    for rule in PIXEL_RULES:
        if not _has(parameters, rule.path):
            continue
        if rule.definition is not None and rule.rounding != "halo":
            definition = dict(source=".".join(rule.definition), millimetres=_get(parameters, rule.definition))
        elif rule.rounding == "halo":
            definition = dict(source="half-width of " + ".".join(rule.definition))
        else:
            length = native_length_pixels(rule, parameters)
            millimetres = ([value * pitch for value in length] if isinstance(length, list)
                           else length * pitch)
            definition = dict(source="native pixels x nominal pitch",
                              millimetres=millimetres,
                              measure="half-width" if rule.rounding == "odd_window" else "length")
        rows.append(dict(path=".".join(rule.path), note=rule.note, rounding=rule.rounding,
                         definition=definition, native=_get(parameters, rule.path),
                         grid=_get(converted, rule.path)))
    for group, paths in (("pitch", GRID_PITCHES), ("area", GRID_AREAS), ("spacing", GRID_SPACINGS),
                         ("context", (CONTEXT_FRACTION,))):
        for path in paths:
            if _has(parameters, path):
                rows.append(dict(path=".".join(path), note="grid " + group, rounding="scale",
                                 definition=dict(source="scales with the pixels per cell"),
                                 native=_get(parameters, path), grid=_get(converted, path)))
    return dict(format="darsia-phase-physical-units-v1", pixels_per_cell=float(pixels_per_cell),
                nominal_mm_per_pixel=pitch, rows=rows)
