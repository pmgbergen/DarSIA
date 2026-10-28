"""Lattice inputs for the unchanged phase chain, on a synthetic rotated canvas.

A native canvas of 720 x 560 pixels of 0.25 mm, slightly rotated against the
metric axes, carries a native centre box and a coloured plume in front of an
unchanged baseline. The lattice inputs coarsen it, and the unchanged
``infer_labels`` classifies the lattice.
"""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.computing_lattice import CellAssignment, computing_lattice
from darsia.fluidflower.mass.n1_geometry import frame_coordinates
from darsia.fluidflower.phase import infer
from darsia.fluidflower.phase.inputs import MaskedNativeInput, TemporalNativeInput
from darsia.fluidflower.pipeline import lattice_inputs as li
from test_phase_infer import _full_inputs
from test_phase_rules import parameters as rule_parameters  # noqa: F401  (fixture)
from test_phase_domain import parameters as domain_parameters  # noqa: F401  (fixture)

CANVAS = (720, 560)
BOX = (40, 30, 480, 640)  # native centre x, y, width, height on the canvas
PITCH = 0.00025
ANGLE = 0.01
EXTENT = (0.0, 0.145, 0.0, 0.183)
PLUME = (slice(420, 600), slice(150, 400))
FIELDS = dict(source_to_native="H_reference_to_frame_native",
              original_domain_to_native="H_reference_to_frame_native",
              carried_homography_provenance="original_H_reference_to_native")


def reference_matrix():
    """Reference native coordinates to metres: 0.25 mm pixels, rotated, rows running down."""
    c, s = math.cos(ANGLE), math.sin(ANGLE)
    return np.array([[PITCH * c, PITCH * s, 0.012], [PITCH * s, -PITCH * c, 0.173], [0.0, 0.0, 1.0]])


def record(homography=np.eye(3)):
    """A native full record that places the synthetic canvas."""
    return dict(phase_shape_hw=list(CANVAS), native_origin_xy=[-BOX[0], -BOX[1]],
                binding=dict(geometry={FIELDS["source_to_native"]: np.asarray(homography).tolist()}))


def log_image(rgb):
    """CHW float32 log RGB of a constant colour on the canvas."""
    return np.stack([np.full(CANVAS, math.log(value), np.float32) for value in rgb])


def centre(values):
    """The native-centre crop of a canvas array."""
    x, y, width, height = BOX
    return np.ascontiguousarray(values[..., y:y + height, x:x + width])


def synthetic(rule_parameters, domain_parameters, *, plume=True):
    """Native inputs as SeriesInputs would prepare them, and a lattice run fixture."""
    seed, bundle, p = _full_inputs(rule_parameters, domain_parameters)
    if plume:
        for head in (bundle["native"]["original_model"].co2_presence,):
            object.__setattr__(head, "intercept", 100.0)
        for name in ("regional_original", "regional_refit", "regional_edge"):
            bundle["native"][name]["heads"]["co2_presence"].update(intercept=100.0)
    p["geometry"].update(source_box_xywh=list(BOX), native_shape_hw=[BOX[3], BOX[2]],
                         floor_source_origin_xy=[-BOX[0], -BOX[1]], rig_origin_xy=[-BOX[0], -BOX[1]])
    baseline_log = log_image((300.0, 300.0, 300.0))
    current = baseline_log.copy()
    if plume:
        current[:, PLUME[0], PLUME[1]] = np.log(np.array([200.0, 300.0, 420.0], np.float32))[:, None, None]
    valid = np.ones(CANVAS, bool)
    rig = np.zeros(CANVAS, bool)
    rig[10:715, 10:550] = True
    native_input = MaskedNativeInput(centre(current), centre(valid), {"synthetic": "current"})
    base_centre = MaskedNativeInput(centre(baseline_log), centre(valid), {"synthetic": "baseline"})
    temporal = TemporalNativeInput(base_centre.log, base_centre.valid, (0, 0), {"synthetic": "baseline"})
    bundle.update(
        baseline=dict(native_input=base_centre, temporal_input=temporal, full_log=baseline_log, full_valid=valid),
        floor_curve=np.full(CANVAS[1], 712.0), zones=np.full((BOX[3], BOX[2]), 4, np.uint8),
        envelopes=dict(bundle["envelopes"], env1=np.zeros(CANVAS, bool), env2=np.zeros(CANVAS, bool)))
    previous = dict(source=dict(source_sha256="b" * 64), full_log=baseline_log, full_valid=valid,
                    current_h=np.eye(3), temporal_input=temporal)
    inputs = SimpleNamespace(parameters=p, model_bundle=bundle, previous=previous)
    source = dict(seed["source"], previous_source_sha256="b" * 64)
    config = dict(phase=dict(reference=dict(full=record(), source_sha256="r" * 64),
                             initial_previous=dict(full=record(), source=previous["source"])))
    area = np.full(CANVAS, 0.0625)
    context = SimpleNamespace(rig_domain=rig, hardware_excluded=np.zeros(CANVAS, bool),
                              pixel_area_mm2=area, parameters={"synthetic": True})
    native = dict(record=dict(full=record()), context=context)
    frame = dict(seed, source=source, full_log=current, full_valid=valid, context=context,
                 native_input=native_input, temporal_input=native_input, photo_sand=centre(valid),
                 inherited_spacing_m=[PITCH, PITCH], probability_spacing_m=[PITCH, PITCH],
                 observed=valid, original_area=area, pixel_area_mm2=area * 0.9,
                 components=dict(open_components=rig.astype(np.uint8), barrier_mask=np.zeros(CANVAS, bool)),
                 top_keep=valid)
    return config, inputs, native, frame, dict(source=source)


@pytest.fixture
def lattice():
    return computing_lattice(EXTENT, 1.0)


def test_the_canvas_matrix_is_n1s_frame_coordinates_and_its_jacobian_the_pixel_area():
    geometry, origin = record()["binding"]["geometry"], [-BOX[0], -BOX[1]]
    matrix = li.canvas_to_metric(reference_matrix(), geometry, origin, fields=FIELDS)
    x, y = frame_coordinates(reference_matrix(), geometry, dict(phase_shape_hw=list(CANVAS), native_origin_xy=origin),
                             fields=FIELDS)
    for row, column in ((0, 0), (500, 33), (719, 559)):
        mapped = matrix @ np.array([column, row, 1.0])
        np.testing.assert_allclose(mapped[:2] / mapped[2], [x[row, column], y[row, column]], rtol=1e-6)
    np.testing.assert_allclose(li.jacobian_area(matrix, CANVAS), PITCH * PITCH, rtol=1e-12)
    projective = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [1e-3, 0.0, 1.0]])
    area = li.jacobian_area(projective, (5, 5))
    step = 1e-6
    for row, column in ((1, 1), (3, 4)):
        def at(c, r):
            v = projective @ np.array([c, r, 1.0])
            return v[:2] / v[2]
        jacobian = np.column_stack([(at(column + step, row) - at(column, row)) / step,
                                    (at(column, row + step) - at(column, row)) / step])
        assert area[row, column] == pytest.approx(abs(np.linalg.det(jacobian)), rel=1e-5)


def test_acquisition_cells_follow_storage_cells_and_the_box_bounds_the_centre(lattice):
    assignment, matrix = li.acquisition_assignment(lattice, reference_matrix(), record(), fields=FIELDS)
    assert assignment.source_shape == CANVAS and assignment.pixel_count.sum() == CANVAS[0] * CANVAS[1]
    box = li.centre_box(assignment, BOX, CANVAS)
    inside = li.embed(np.ones((BOX[3], BOX[2]), bool), BOX, CANVAS, False)
    fraction = assignment.fraction(inside)
    cropped = li.crop(fraction, box) >= 0.5
    # The box bounds the cells at least half inside the rotated centre.
    assert cropped[0].any() and cropped[-1].any() and cropped[:, 0].any() and cropped[:, -1].any()
    assert cropped.mean() > 0.9
    outside = np.ones(lattice.shape, bool)
    outside[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = False
    assert (fraction[outside] < 0.5).all()
    assert box[2] == pytest.approx(BOX[2] / 4, abs=2) and box[3] == pytest.approx(BOX[3] / 4, abs=2)
    np.testing.assert_array_equal(li.embed(np.ones((2, 3), np.uint8), (1, 2, 3, 2), (5, 5), 0)[2:4, 1:4], 1)
    with pytest.raises(ValueError, match="source box"):
        li.embed(np.ones((2, 2)), (0, 0, 3, 2), (5, 5), 0)


def test_lattice_rules_convert_rule_sizes_and_describe_the_box(rule_parameters, domain_parameters):
    _, inputs, *_ = synthetic(rule_parameters, domain_parameters, plume=False)
    rules = li.lattice_rules(inputs.parameters, pixels_per_cell=4.0, box=(5, 6, 120, 160))
    assert rules["geometry"]["source_box_xywh"] == [5, 6, 120, 160]
    assert rules["geometry"]["native_shape_hw"] == [160, 120] and rules["geometry"]["rig_origin_xy"] == [-5, -6]
    assert rules["geometry"]["floor_source_origin_xy"] == [-BOX[0], -BOX[1]]
    assert rules["decisions"]["top"]["band_pixels"] == round(60 / inputs.parameters["decisions"]["nominal_mm_per_pixel"] / 4.0)
    assert rules["decisions"]["nominal_mm_per_pixel"] == inputs.parameters["decisions"]["nominal_mm_per_pixel"] * 4.0
    assert inputs.parameters["geometry"]["source_box_xywh"] == list(BOX)


def test_the_reference_homography_maps_reference_points_to_lattice_box_coordinates(lattice):
    box = (3, 4, 10, 10)
    matrix = li.reference_to_lattice_native(lattice, reference_matrix(), box)
    point = np.array([123.0, 77.0, 1.0])
    metres = reference_matrix() @ point
    cell = lattice.metric_to_cell() @ metres
    mapped = matrix @ point
    np.testing.assert_allclose(mapped[:2] / mapped[2], cell[:2] / cell[2] - np.array(box[:2]), atol=1e-9)


def test_positive_cells_fill_only_cells_without_source_pixels():
    areas = np.array([[0.0, 1.0], [2.0, 4.0]])
    np.testing.assert_array_equal(li.positive_cells(areas), [[2.0, 1.0], [2.0, 4.0]])
    with pytest.raises(ValueError, match="No lattice cell"):
        li.positive_cells(np.zeros((2, 2)))


def test_an_acquisition_coarsens_canvas_and_masked_centre_to_registered_lattice_inputs(
        rule_parameters, domain_parameters, lattice):
    config, inputs, native, frame, spec = synthetic(rule_parameters, domain_parameters)
    run = li.LatticeRun(config, inputs, lattice, reference_matrix=reference_matrix(), fields=FIELDS)
    assert run.pixels_per_cell == pytest.approx(4.0)
    assert run.canvas_pitch_m == pytest.approx(PITCH)
    lattice_frame, assignment, _ = run.frame(spec, native, frame)
    x, y, width, height = run.box
    assert lattice_frame["native_input"].log.shape == (3, height, width)
    assert lattice_frame["temporal_input"].origin_xy == (0, 0)
    np.testing.assert_array_equal(lattice_frame["current_h"], np.eye(3))
    assert lattice_frame["probability_spacing_m"] == [pytest.approx(0.001)] * 2
    assert lattice_frame["reference_to_current_h"] is run.reference_to_lattice
    full = lattice_frame["full_log"]
    np.testing.assert_array_equal(np.isfinite(full).all(0), lattice_frame["full_valid"])
    rgb = np.exp(full[:, lattice_frame["full_valid"]])
    assert rgb.min() > 199.0 and rgb.max() < 421.0
    np.testing.assert_allclose(lattice_frame["pixel_area_mm2"][assignment.pixel_count == 16], 16 * 0.0625 * 0.9)
    assert (lattice_frame["original_area"] > 0).all()
    rig = lattice_frame["context"].rig_domain
    assert rig.dtype == bool and not (rig & lattice_frame["context"].hardware_excluded).any()
    components = lattice_frame["components"]
    assert not (components["barrier_mask"] & (components["open_components"] > 0)).any()
    assert frame["full_log"].shape == (3, *CANVAS)  # The native frame is not modified.
    assert run.previous["source"]["source_sha256"] == "b" * 64
    assert run.provenance()["native_pixels_per_cell"] == pytest.approx(4.0)


def test_retained_native_caches_are_refused(rule_parameters, domain_parameters, lattice):
    config, inputs, native, frame, spec = synthetic(rule_parameters, domain_parameters, plume=False)
    run = li.LatticeRun(config, inputs, lattice, reference_matrix=reference_matrix(), fields=FIELDS)
    with pytest.raises(ValueError, match="native-only"):
        run.frame(spec, native, dict(frame, prior_cache=object()))


@pytest.mark.parametrize("resolution", [1.0, 0.5])
def test_the_unchanged_phase_chain_finds_the_plume_on_the_lattice(rule_parameters, domain_parameters, resolution):
    lattice = computing_lattice(EXTENT, resolution)
    config, inputs, native, frame, spec = synthetic(rule_parameters, domain_parameters)
    run = li.LatticeRun(config, inputs, lattice, reference_matrix=reference_matrix(), fields=FIELDS)
    lattice_frame, assignment, _ = run.frame(spec, native, frame)
    result = infer.infer_labels(lattice_frame, run.previous, run.bundle(inputs, assignment), run.rules)
    labels = result["labels"]
    assert labels.shape == lattice.shape and labels.dtype == np.uint8
    plume = np.zeros(CANVAS, bool)
    plume[PLUME] = True
    near = np.zeros(CANVAS, bool)
    near[380:640, 110:440] = True  # Regional context may widen the plume by a few millimetres.
    core = assignment.fraction(plume) == 1.0
    clear = assignment.fraction(near) == 0.0
    domain = result["classes"]["rig_domain"]
    assert core.sum() > 100
    assert (labels[core & domain] == 1).all()
    assert (labels[clear & domain] == 0).all()
    assert (labels[~domain] == 255).all()
    assert result["qa"]["phase_area"]["co2_cm2"] == pytest.approx(
        float(lattice_frame["pixel_area_mm2"][labels == 1].sum()) / 100)
    run.advance(lattice_frame)
    assert run.previous["source"] is spec["source"]
    assert run.previous["temporal_input"] is lattice_frame["temporal_input"]
