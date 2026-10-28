"""Mass on the lattice through the unchanged N1 code, against native N1.

A synthetic canvas of 96 x 128 pixels of 0.25 mm is aligned with a 1 mm
lattice (4 x 4 pixels per cell). With uniform colours, physical fields and
cell-aligned phases, the lattice mass must equal the native mass; this is the
strongest check that the lattice template, records and box masks carry the
native inputs faithfully.
"""
import json
from pathlib import Path

import darsia
import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower.computing_lattice import computing_lattice
from darsia.fluidflower.mass import lattice_mass as lm
from darsia.fluidflower.mass import n1_geometry as geometry
from darsia.fluidflower.mass.n1_pipeline import CONFIG_FORMAT, DataReader, N1Pipeline, asset, _write_npz
from darsia.fluidflower.mass.production_precision import PixelMath
from darsia.fluidflower.pipeline.lattice_inputs import acquisition_assignment

SHAPE = (96, 128)
PITCH = 0.00025
REFERENCE = np.array([[PITCH, 0.0, PITCH / 2], [0.0, -PITCH, 0.024 - PITCH / 2], [0.0, 0.0, 1.0]])
EXTENT = (0.0, 0.032, 0.0, 0.024)
FIELDS = dict(source_to_native="H_reference_to_frame_native",
              original_domain_to_native="H_reference_to_frame_native",
              carried_homography_provenance="original_H_reference_to_native")


def save(directory, name, value, kind):
    path = Path(directory) / name
    if kind == "json":
        path.write_text(json.dumps(value), encoding="utf-8")
    elif kind == "npy":
        np.save(path, value, allow_pickle=False)
    elif kind == "npz":
        np.savez(path, **value)
    else:
        Image.fromarray(value).save(path)
    return asset(path)


def raw_record(directory, name, rgb, shape=SHAPE):
    """A native record of constant linear RGB, as compose_raw reads it."""
    value = np.broadcast_to(np.asarray(rgb, np.float32), (*shape, 3)).copy()
    logs = np.moveaxis(np.log(value * np.float32(65535)), 2, 0).astype(np.float32)
    return dict(phase_shape_hw=list(shape), native_origin_xy=[0, 0],
                native_centre=dict(native_shape_hw=list(shape), log_linear_rgb=save(directory, name + ".npy", logs, "npy"),
                                   valid_mask=save(directory, name + "_valid.npy", np.ones(shape, bool), "npy")),
                ring_windows=[], binding=dict(geometry={FIELDS["source_to_native"]: np.eye(3).tolist()}))


def dataset(tmp_path):
    """Native N1 template, frame specs and the native phase on the synthetic canvas."""
    domain = np.zeros(SHAPE, bool)
    domain[4:92, 4:124] = True
    facies = np.ones(SHAPE, np.uint8)
    facies[:, 64:] = 2
    labels = np.where(domain, 0, 255).astype(np.uint8)
    labels[40:72, 32:96] = 1
    labels[48:56, 40:56] = 2
    physical = save(tmp_path, "physical.npz", dict(facies=facies, domain=np.ones(SHAPE, bool),
                                                   porosity=np.full(SHAPE, 0.3, np.float32),
                                                   depth_m=np.full(SHAPE, 0.01, np.float32)), "npz")
    colors, signals = {}, {}
    flash_bounds = dict(min_value_aq=0.0, max_value_aq=1.0, min_value_g=0.0, max_value_g=1.0)
    for label in (1, 2):
        base = tmp_path / ("color_" + str(label))
        path = darsia.ColorPath(colors=[np.array([1.0, 1.0, 1.0]), np.array([0.8, 0.7, 0.6])])
        darsia.ColorPathInterpolation(path, darsia.ColorMode.RELATIVE, [0.0, 1.0]).save(base)
        colors[str(label)] = asset(base.with_suffix(".json"))
        base = tmp_path / ("signal_" + str(label))
        darsia.PWTransformation([0.0, 1.0], [0.0, 1.0]).save(base)
        signals[str(label)] = asset(base.with_suffix(".csv"))
    base = tmp_path / "flash"
    darsia.SimpleFlash(**flash_bounds).save(base)
    metadata = save(tmp_path, "optics.json", dict(label_ids=[1, 2], ignore_labels=[], basis="facies",
                                                  flash_bounds=flash_bounds), "json")
    config = dict(
        format=CONFIG_FORMAT, run="synthetic", frames=[], geometry_fields=dict(FIELDS),
        shared=dict(physical_reference=physical, reference_matrix=REFERENCE.tolist(),
                    floor_curve=save(tmp_path, "floor.npy", np.full(SHAPE[1], 91.5), "npy"), floor_origin_xy=[0, 0],
                    depth_points=[[0.0, 0.0, 0.01], [0.04, 0.0, 0.01], [0.0, 0.03, 0.01], [0.04, 0.03, 0.01]],
                    porosity_by_facies={"1": 0.3, "2": 0.3}, swi_by_facies={"1": 0.2, "2": 0.25},
                    unquantified_facies=[], clearwater=dict(
                        rgb=save(tmp_path, "clear.npy", np.ones((*SHAPE, 3), np.float32), "npy"),
                        valid=save(tmp_path, "clear_valid.npy", np.ones(SHAPE, bool), "npy"),
                        H_reference_to_grid=np.eye(3).tolist()),
                    optics=dict(metadata=metadata, color_paths=colors, signals=signals,
                                flash=asset(base.with_suffix(".json")))),
        calibration=save(tmp_path, "calibration.json", dict(parameters=dict(candidate="N1", gain=1.2, beta=0.3)), "json"),
        physical_correction=save(tmp_path, "correction.json", dict(correction_matrix=np.eye(3).tolist()), "json"),
        anchor=dict(elapsed_h=0.5, affected_pore_volume_m3=1e-8), recipe=dict(alkalinity_M=0.001, btb_M=0.0001),
        dimensions_m=[0.024, 0.032],
        boxes=dict(box1=[[0.004, 0.004], [0.016, 0.012]], box2=[[0.016, 0.008], [0.03, 0.02]]),
        summary=dict(baseline_gain=1.1, intervals=32768, max_column_distance_m=0.02, max_depth_m=0.0155,
                     max_porosity=0.45),
        bindings=dict(amendment_sha256="a" * 64, round4_spec_sha256="b" * 64),
        contracts=dict(result="saved-products", sandmap="saved-sandmap"))
    current, baseline = raw_record(tmp_path, "current", [0.8, 0.7, 0.6]), raw_record(tmp_path, "baseline", [1.0, 1.0, 1.0])
    mass_spec = dict(
        selected_target_h=1.0, light_epoch="measured", nominal_injected_mass_g=1.0,
        atmosphere=dict(pressure_bar=1.0, temperature_c=20.0, pressure_gradient_bar_per_m=0.0,
                        temperature_gradient_c_per_m=0.0),
        photo_top=dict(save(tmp_path, "top.npz", dict(keep=np.ones(SHAPE, bool)), "npz"), array_key="keep"),
        original_domain=dict(parent_labels_map=save(tmp_path, "parent.png", np.zeros(SHAPE, np.uint8), "png"),
                             native_shape_hw=list(SHAPE), frame_index=0,
                             **{FIELDS["original_domain_to_native"]: np.eye(3).tolist()}),
        inner_polygon_native=[[2.0, 2.0], [125.0, 2.0], [125.0, 93.0], [2.0, 93.0]])
    source = dict(source_binding=dict(source_sha256="c" * 64), source_sha256="c" * 64)
    run_frame = dict(case_id="sample_1", frame_index=1, elapsed_h=1.0, source=source, mass_spec=mass_spec)
    classes = dict(rig_domain=domain, modelled=domain, filled_observed=np.zeros(SHAPE, bool),
                   filled_unobserved=np.zeros(SHAPE, bool))
    phase = dict(labels=save(tmp_path, "labels.png", labels, "png"),
                 classes=save(tmp_path, "classes.npz", classes, "npz"),
                 area=dict(save(tmp_path, "area.npz", dict(pixel_area_mm2=np.full(SHAPE, 0.0625)), "npz"),
                           array_key="pixel_area_mm2"))
    native_spec = dict(mass_spec, case_id="sample_1", frame_index=1, elapsed_h=1.0,
                       source_binding=source["source_binding"], current=current, baseline=baseline, **phase)
    config["frames"] = [native_spec]
    return config, native_spec, run_frame, dict(labels=labels, classes=classes, current=current, baseline=baseline)


@pytest.fixture
def lattice():
    return computing_lattice(EXTENT, 1.0)


def test_physical_reference_is_coarsened_by_majority_and_domain_means(lattice):
    shape = (8, 12)
    matrix = np.array([[PITCH, 0.0, PITCH / 2], [0.0, -PITCH, 0.024 - PITCH / 2], [0.0, 0.0, 1.0]])
    facies = np.ones(shape, np.uint8)
    facies[:, 4:] = 3
    facies[0, 5] = 1
    domain = np.ones(shape, bool)
    domain[4:, :] = False
    porosity = np.full(shape, 0.3, np.float32)
    porosity[:, 8:] = 0.5
    physical = dict(facies=facies, domain=domain, porosity=porosity, depth_m=np.full(shape, 0.01, np.float32))
    result = lm.physical_on_lattice(physical, matrix, np.eye(3), lattice)
    assert result["facies"].dtype == np.uint8 and result["porosity"].dtype == np.float32
    assert list(result["facies"][0, :3]) == [1, 3, 3]
    assert result["domain"][0, :3].all() and not result["domain"][1:, :].any()
    np.testing.assert_allclose(result["porosity"][0, :3], [0.3, 0.3, 0.5])
    assert (result["porosity"][~result["domain"]] == 0).all()
    shifted = lm.physical_on_lattice(physical, matrix, np.array([[1.0, 0, 4.0], [0, 1.0, 0], [0, 0, 1.0]]), lattice)
    assert list(shifted["facies"][0, :4]) == [0, 1, 3, 3]  # facies follow the correction, as N1 places them


def test_clear_water_and_floor_reach_lattice_cells(lattice):
    rgb = np.ones((8, 12, 3), np.float32)
    rgb[:, 4:8] = 2.0
    valid = np.ones((8, 12), bool)
    valid[:, 8:] = False
    result, supported = lm.clear_water_on_lattice(rgb, valid, np.eye(3), REFERENCE, lattice)
    assert result.dtype == np.float32 and result.shape == (*lattice.shape, 3)
    np.testing.assert_allclose(result[0, :2, 0], [1.0, 2.0])
    assert supported[0, :2].all() and not supported[0, 2] and not supported[3, 0]
    floor = lm.floor_on_lattice(np.full(SHAPE[1], 91.5), (0, 0), REFERENCE, lattice)
    np.testing.assert_allclose(floor[np.isfinite(floor)], 22.5)
    assert np.isfinite(floor[:32]).all()


def test_lattice_dimensions_keep_the_native_depth_per_metre_of_rig(lattice):
    matrix = np.array([[PITCH, 0.0, PITCH / 2], [0.0, -PITCH, 0.024 - PITCH / 2], [0.0, 0.0, 1.0]])
    dimensions, facts = lm.hydrostatic_dimensions([0.024, 0.032], SHAPE, matrix, lattice)
    native_per_metre = 0.024 / ((SHAPE[0] - 1) * PITCH)
    assert facts["depth_per_metre_of_rig"] == pytest.approx(native_per_metre)
    assert dimensions[0] / (lattice.shape[0] - 1) == pytest.approx(native_per_metre * lattice.spacing_m)
    assert dimensions[1] / (lattice.shape[1] - 1) == pytest.approx(0.032 / ((SHAPE[1] - 1) * PITCH) * 0.001)
    # The first lattice row's centre is canvas row 1.5, whose native depth remains.
    assert facts["first_row_native_depth_m"] == pytest.approx(0.024 * 1.5 / 95)


def test_native_box_masks_follow_the_template_roi_on_the_canvas():
    masks = lm.native_box_masks(dict(box1=[[0.004, 0.004], [0.016, 0.012]], box2=dict(roi=[[0.0, 0.0], [0.032, 0.024]])),
                                SHAPE, [0.024, 0.032])
    image = darsia.ScalarImage(np.zeros(SHAPE, np.float32), dimensions=[0.024, 0.032])
    expected = darsia.roi_to_mask(darsia.CoordinateArray([[0.004, 0.004], [0.016, 0.012]]), image, mode="voxels").img
    np.testing.assert_array_equal(masks["box1"], expected)
    assert masks["box2"].all()


def test_a_lattice_record_is_read_back_exactly_by_compose_raw(tmp_path, lattice):
    log = np.log(np.full((3, *lattice.shape), 1000.0, np.float32)).astype(np.float32)
    valid = np.ones(lattice.shape, bool)
    log[:, 0, 0], valid[0, 0] = np.nan, False
    record = lm.lattice_record(tmp_path, "current", log, valid, lattice, fields=FIELDS, provenance=dict(role="test"))
    rgb, decoded_valid = geometry.compose_raw(record, DataReader(), PixelMath("float64"))
    np.testing.assert_array_equal(decoded_valid, valid)
    np.testing.assert_allclose(rgb[valid], 1000.0 / 65535, rtol=1e-6)
    assert record["binding"]["geometry"][FIELDS["source_to_native"]] == np.eye(3).tolist()


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_lattice_mass_equals_native_mass_on_cell_aligned_uniform_fields(tmp_path, lattice, precision):
    config, native_spec, run_frame, native = dataset(tmp_path)
    math = PixelMath(precision)
    pipeline = N1Pipeline(config, precision=precision, pixel_math=math)
    expected = pipeline.evaluate(native_spec)["result"]
    record = dict(phase_shape_hw=list(SHAPE), native_origin_xy=[0, 0],
                  binding=dict(geometry={FIELDS["source_to_native"]: np.eye(3).tolist()}))
    assignment, matrix = acquisition_assignment(lattice, REFERENCE, record, fields=FIELDS)
    mass = lm.LatticeMass(pipeline, lattice, tmp_path / "lattice", canvas_shape_hw=SHAPE, canvas_matrix=matrix,
                          precision=precision, pixel_math=math, unc_only=False)
    stage = tmp_path / "stage"
    stage.mkdir()
    reader = DataReader()
    records = {}
    for name in ("current", "baseline"):
        rgb, valid = geometry.compose_raw(native[name], reader, PixelMath("float64"))
        log = np.moveaxis(np.log(rgb * 65535), 2, 0).astype(np.float32)
        lattice_log, lattice_valid, _ = assignment.log_rgb(log, valid)
        records[name] = lm.lattice_record(stage, name, lattice_log, lattice_valid, lattice, fields=FIELDS,
                                          provenance=dict(role=name))
    labels = assignment.majority(native["labels"])
    classes = {name: assignment.mask(value) for name, value in native["classes"].items()}
    np.testing.assert_array_equal(classes["rig_domain"], labels != 255)
    phase = dict(labels=save(stage, "labels.png", labels, "png"), classes=save(stage, "classes.npz", classes, "npz"),
                 area=dict(save(stage, "area.npz", dict(pixel_area_mm2=assignment.total(np.full(SHAPE, 0.0625))), "npz"),
                           array_key="pixel_area_mm2"))
    spec = mass.frame_spec(run_frame, stage, current=records["current"], baseline=records["baseline"], phase=phase,
                           assignment=assignment, canvas_matrix=matrix, native_record=record)
    # Canvas pixel 2 lies in cell 0, whose centre is canvas pixel 1.5: 0.5 px = 0.125 cell.
    assert spec["inner_polygon_native"][0] == pytest.approx([0.125, 0.125])
    boxes = mass.box_masks(assignment, SHAPE)
    native_boxes = lm.native_box_masks(config["boxes"], SHAPE, config["dimensions_m"])
    for name in ("box1", "box2"):
        np.testing.assert_array_equal(boxes[name], assignment.majority(native_boxes[name].astype(np.uint8)) == 1)
    evaluated = mass.evaluate(spec, boxes)
    actual = evaluated["result"]
    exact = 1e-12 if precision == "float64" else 1e-6
    # Volumes and colour responses do not depend on the hydrostatic depth: exact.
    assert actual["affected_pore_volume_m3"] == pytest.approx(expected["affected_pore_volume_m3"], rel=exact)
    native_colour = expected["fields"]["c_aq_relative"][44, 60]
    assert native_colour > 0
    np.testing.assert_allclose(actual["fields"]["c_aq_relative"][11, 15], native_colour, rtol=exact)
    # Masses carry the documented constant depth offset of the first lattice row.
    hydrostatics = mass.config["lattice"]["hydrostatics"]
    assert hydrostatics["first_row_native_depth_m"] == pytest.approx(0.024 * 1.5 / 95)
    for key in ("total_mass_g", "aqueous_mass_g", "gas_mass_g"):
        assert actual[key] == pytest.approx(expected[key], rel=1e-4), key
    assert actual["gas_mass_g"] > 0 and actual["aqueous_mass_g"] > 0
    for box in ("box1", "box2"):
        for key in ("total_mass_g", "aqueous_mass_g", "gas_mass_g"):
            assert actual["domains"][box][key] == pytest.approx(expected["domains"][box][key], rel=1e-4), (box, key)
    assert actual["fields"]["phase_labels"].shape == lattice.shape
    assert mass.config["lattice"]["lattice"]["shape"] == list(lattice.shape)
    template = json.loads(Path(mass.config_asset["path"]).read_bytes())
    assert template["shared"]["reference_matrix"] == lattice.cell_to_metric().tolist()
    assert template["physical_correction"]["path"].endswith("lattice_physical_correction.json")
