"""Phase rule sizes in physical units: native DATA reproduced, lattice grids derived.

The decisions come from the repository fixture of the frozen rule DATA. The
other pixel-sized entries are the values of the bound production phase
configuration (SHA-256 2e9821e0...); no NAS file is read here.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from darsia.fluidflower.phase import physical_units as units

FIXTURES = Path(__file__).parent / "fixtures"
BOUND = dict(
    features=dict(optical=dict(texture_sigma=2.0, spatial_context_fraction=0.0015, block_rows=128),
                  flatfield=dict(box_size=5, halo_rows=2, feature_count=4)),
    feature_support=dict(texture_footprint_hw=[17, 17], guard_px=8, hole_max_area_px=60000,
                         vertical_gap_max_px=64, grid_shape_hw=[382, 710], native_shape_hw=[2979, 5472],
                         coverage_tolerance=1e-06),
    domain=dict(nominal_rule_b_pixel_area_mm2=0.027053, gas_minimum_area_mm2=2.5),
    chain=dict(rig_spacing_m=[0.00016448, 0.00016448]),
    geometry=dict(native_shape_hw=[2979, 5472], source_box_xywh=[416, 496, 5472, 2979]),
    regional=dict(settings=dict(working_spacing_m=0.001, fine_sigma_m=0.00075, coarse_sigma_m=0.002)),
)


@pytest.fixture
def parameters():
    value = deepcopy(BOUND)
    value["decisions"] = json.loads((FIXTURES / "phase_rules_parameters.json").read_bytes())
    return value


def assert_identical(actual, expected, where="parameters"):
    """Equal values of equal types, recursively; 365 and 365.0 differ."""
    assert type(actual) is type(expected), where
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), where
        for key in expected:
            assert_identical(actual[key], expected[key], where + "." + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_identical(left, right, f"{where}[{index}]")
    else:
        assert actual == expected, where


def test_one_pixel_per_cell_reproduces_every_bound_value(parameters):
    native = units.parameters_on_grid(parameters, pixels_per_cell=1.0,
                                      feature_shape_hw=parameters["geometry"]["native_shape_hw"])
    assert_identical(native, parameters)


def test_the_millimetre_definitions_derive_the_data_with_its_rounding(parameters):
    d = parameters["decisions"]
    pitch = d["nominal_mm_per_pixel"]
    assert round(d["top"]["band_mm"] / pitch) == d["top"]["band_pixels"] == 365
    assert math.floor(d["top"]["column_gate_mm"] / pitch) == d["top"]["column_gate_pixels"] == 182
    assert math.floor(d["floor"]["floor_band_mm"] / pitch) == d["floor"]["floor_band_pixels"] == 121
    assert round(d["floor"]["floor_band_mm"] / pitch) == 122  # floor, not round, gives the DATA
    assert math.floor(d["contact"]["short_fill_limit_mm"] / pitch) == d["contact"]["short_fill_limit_pixels"] == 9
    assert d["floor"]["connectivity_disc_radius_mm"] / pitch == d["floor"]["connectivity_disc_radius_pixels"]
    assert d["envelopes"]["radius_mm"] / pitch == d["envelopes"]["radius_pixels"]
    rules = {rule.path: rule for rule in units.PIXEL_RULES}
    assert rules[("decisions", "top", "band_pixels")].rounding == "round"
    assert rules[("decisions", "floor", "floor_band_pixels")].rounding == "floor"


@pytest.mark.parametrize("cells,expected", [
    (5.995, dict(band=61, gate=30, floor=20, contact=1, kernel=[9, 9], box=1, halo=0, erosion=[3, 3])),
    (2.9975, dict(band=122, gate=60, floor=40, contact=3, kernel=[17, 17], box=3, halo=1, erosion=[7, 7])),
])
def test_a_lattice_keeps_each_rule_extent_on_the_rig(parameters, cells, expected):
    grid = units.parameters_on_grid(parameters, pixels_per_cell=cells, feature_shape_hw=(500, 913))
    d, pitch = grid["decisions"], parameters["decisions"]["nominal_mm_per_pixel"]
    assert d["top"]["band_pixels"] == expected["band"] == round(60 / pitch / cells)
    assert d["top"]["column_gate_pixels"] == d["floor"]["column_gate_pixels"] == expected["gate"]
    assert d["floor"]["floor_band_pixels"] == expected["floor"]
    assert d["contact"]["short_fill_limit_pixels"] == expected["contact"]
    assert d["placement"]["fraction_kernel_hw"] == expected["kernel"]
    assert grid["features"]["flatfield"]["box_size"] == expected["box"]
    assert grid["features"]["flatfield"]["halo_rows"] == expected["halo"]
    assert grid["feature_support"]["texture_footprint_hw"] == expected["erosion"]
    assert d["floor"]["connectivity_disc_radius_pixels"] == pytest.approx(1.5 / pitch / cells)
    assert d["envelopes"]["radius_pixels"] == pytest.approx(5.0 / pitch / cells)
    assert grid["features"]["optical"]["texture_sigma"] == pytest.approx(2.0 / cells)
    # Documentation copies follow their rules.
    assert d["top"]["source_rule"]["band_pixels"] == d["top"]["band_pixels"]
    assert d["floor"]["source_rule"]["floor_band_pixels"] == d["floor"]["floor_band_pixels"]
    # Unchanged entries: adjacency, native photo support, metric regional scales.
    assert d["top"]["component_structure_hw"] == [3, 3]
    assert grid["feature_support"]["guard_px"] == 8
    assert grid["regional"] == parameters["regional"]


def test_the_grid_pitch_areas_and_spacings_scale_with_the_cell(parameters):
    grid = units.parameters_on_grid(parameters, pixels_per_cell=4.0, feature_shape_hw=(745, 1368))
    native = parameters["decisions"]
    assert grid["decisions"]["nominal_mm_per_pixel"] == native["nominal_mm_per_pixel"] * 4.0
    for path in units.GRID_AREAS:
        assert units._get(grid, path) == units._get(parameters, path) * 16.0
    assert grid["chain"]["rig_spacing_m"] == [0.00016448 * 4.0] * 2


def test_the_spatial_context_keeps_its_length_on_another_feature_shape(parameters):
    grid = units.parameters_on_grid(parameters, pixels_per_cell=6.0, feature_shape_hw=(497, 913))
    sigma = grid["features"]["optical"]["spatial_context_fraction"] * 497
    assert sigma == pytest.approx(0.0015 * 2979 / 6.0)
    same_shape = units.parameters_on_grid(parameters, pixels_per_cell=1.0, feature_shape_hw=(2979, 5472))
    assert same_shape["features"]["optical"]["spatial_context_fraction"] == 0.0015


def test_an_unconverted_pixel_parameter_is_refused(parameters):
    parameters["decisions"]["top"]["new_margin_pixels"] = 12
    with pytest.raises(ValueError, match="decisions.top.new_margin_pixels"):
        units.parameters_on_grid(parameters, pixels_per_cell=2.0, feature_shape_hw=(10, 10))
    with pytest.raises(ValueError, match="lack a physical definition"):
        units.require_covered(parameters)


def test_pixel_sized_paths_name_pixel_areas_pitches_and_spacings(parameters):
    paths = set(units.pixel_sized_paths(parameters))
    assert ("decisions", "top", "band_pixels") in paths
    assert ("decisions", "nominal_mm_per_pixel") in paths
    assert ("chain", "rig_spacing_m") in paths
    assert ("feature_support", "guard_px") in paths
    assert ("decisions", "top", "band_mm") not in paths
    units.require_covered(parameters)


def test_the_input_is_unchanged_and_the_report_names_each_definition(parameters):
    before = deepcopy(parameters)
    grid = units.parameters_on_grid(parameters, pixels_per_cell=6.0, feature_shape_hw=(497, 913))
    assert_identical(parameters, before)
    report = units.conversion_report(parameters, grid, pixels_per_cell=6.0)
    rows = {row["path"]: row for row in report["rows"]}
    band = rows["decisions.top.band_pixels"]
    assert band["definition"] == dict(source="decisions.top.band_mm", millimetres=60)
    assert (band["native"], band["grid"], band["rounding"]) == (365, 61, "round")
    kernel = rows["decisions.placement.fraction_kernel_hw"]["definition"]
    assert kernel["measure"] == "half-width"
    assert kernel["millimetres"] == [25 * parameters["decisions"]["nominal_mm_per_pixel"]] * 2
    assert rows["features.flatfield.halo_rows"]["definition"]["source"].startswith("half-width")
    assert rows["decisions.nominal_pixel_area_mm2"]["rounding"] == "scale"
    assert report["pixels_per_cell"] == 6.0


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "6"])
def test_invalid_pixels_per_cell_are_refused(parameters, bad):
    with pytest.raises(ValueError, match="Pixels per cell"):
        units.parameters_on_grid(parameters, pixels_per_cell=bad, feature_shape_hw=(10, 10))


def test_lengths_and_grid_values_follow_each_rounding(parameters):
    rules = {rule.path: rule for rule in units.PIXEL_RULES}
    band = rules[("decisions", "top", "band_pixels")]
    length = units.native_length_pixels(band, parameters)
    assert length == 60 / parameters["decisions"]["nominal_mm_per_pixel"]
    assert units.grid_value(band, length, 1.0) == 365
    kernel = rules[("decisions", "placement", "fraction_kernel_hw")]
    assert units.native_length_pixels(kernel, parameters) == [25.0, 25.0]
    assert units.grid_value(kernel, [25.0, 25.0], 2.0) == [25, 25]  # 12.5 rounds to even 12
    sigma = rules[("features", "optical", "texture_sigma")]
    assert units.grid_value(sigma, 2.0, 4.0) == 0.5
    with pytest.raises(ValueError, match="halo"):
        units.native_length_pixels(rules[("features", "flatfield", "halo_rows")], parameters)
    parameters["decisions"]["placement"]["fraction_kernel_hw"] = [50, 51]
    with pytest.raises(ValueError, match="odd positive"):
        units.native_length_pixels(kernel, parameters)
    with pytest.raises(ValueError, match="Unknown pixel rounding"):
        units.grid_value(units.PixelRule(("x",), None, "ceil", "test"), 1.0, 1.0)
