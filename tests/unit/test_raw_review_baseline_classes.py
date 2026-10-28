"""Baseline Change uses its own source geometry and mandatory floor cutoff."""
from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower import raw_review_baseline_classes as baseline
from darsia.fluidflower.phase_saved_assets import asset


@pytest.fixture
def source(tmp_path):
    import json
    def write(name, value):
        path = tmp_path / name
        if name.endswith(".json"):
            path.write_text(json.dumps(value), encoding="utf-8")
        else:
            np.save(path, value)
        return asset(path)
    identity = np.eye(3).tolist()
    valid = np.ones((4, 6), bool)
    valid[1, 1] = False
    top = np.ones((6, 8), bool)
    top[0] = False
    audit = write("audit.json", dict(jpeg_inputs=False,
        source_identity=dict(raw_sha256="a" * 64), renderer=dict(
            native_to_camera_u_h=identity, camera_u_wh=[6, 4])))
    row = dict(reference=True, raw_source=dict(sha256="a" * 64),
        phase_shape_hw=[6, 8], native_origin_xy=[-1, -1],
        reference_rig=dict(polygon_reference_native=[[-1, -1], [6, -1], [6, 4], [-1, 4]]),
        H_reference_rig_to_frame_native=identity, H_reference_to_frame=identity,
        area_parameters=dict(H_reference_to_frame=identity, H_reference_to_metric=identity),
        frame_photo_top_keep=write("top.npy", top), frame_fixture_polygons_native={},
        fixtures_verified_none=True)
    record = dict(native_row=write("row.json", dict(raw_producer_audit=audit)),
        full=dict(source_sha256="a" * 64, phase_shape_hw=[6, 8], native_origin_xy=[-1, -1],
            binding=dict(geometry=row, producer_audit=audit), native_centre=dict(
                native_shape_hw=[4, 6], valid_mask=write("valid.npy", valid))))
    constants = dict(determinant_floor=1e-14, point_horizon_floor=1e-12,
        area_horizon_floor=1e-10, observation_horizon_floor=1e-10,
        raster_scale=256, raster_shift=8, canvas_margin=0.5,
        square_mm_per_cm2=100, nominal_pixel_area_mm2=0.03,
        contracts=dict(rig_context="rig", independent_context="independent"),
        fields=dict(reference_polygon="polygon", reference_fixtures="fixtures",
            nominal_area_field="area", reference_metric_field="metric",
            reference_transform_field="H_reference_to_frame"))
    phase = dict(geometry_parameters=constants, geometry_fields=dict(
        reference_to_frame="H_reference_to_frame", reference_to_metric="H_reference_to_metric"),
        parameters=dict(geometry=dict(rig_origin_xy=[-1, -1], native_shape_hw=[4, 6],
            source_box_xywh=[1, 1, 6, 4], floor_source_origin_xy=[-1, -1],
            curve_homography_floor=1e-10)), chain_assets=dict(floor=write("floor.npy", np.full(8, 3.))))
    return record, phase


def test_own_raw_support_top_and_floor_make_disjoint_classes(source):
    record, phase = source
    result = baseline.baseline_rig_classes(record, phase)
    domain = result["rig_domain"]
    assert not domain[0].any()  # Own photographic sand top.
    assert not domain[4:].any() and result["below_sand_floor"][4:].any()
    assert domain[1:4].all()
    assert result["modelled"][1, 1] and not result["modelled"][2, 2]
    assert result["filled_observed"][2, 2]  # A source hole is visible fill.
    assert result["filled_unobserved"][2, 0]  # Outside the own camera footprint.
    total = sum(result[name].astype(np.uint8) for name in
                ("modelled", "filled_observed", "filled_unobserved"))
    assert np.array_equal(total, domain.astype(np.uint8))


def test_writer_is_exclusive_and_uses_real_array_payloads(tmp_path, source):
    record, phase = source
    path = tmp_path / "classes/baseline.npz"
    bound = baseline.write_baseline_rig_classes(path, record, phase)
    before = path.read_bytes()
    assert bound == asset(path)
    with np.load(path) as archive:
        expected = baseline.baseline_rig_classes(record, phase)
        assert all(np.array_equal(archive[key], value) for key, value in expected.items())
    with pytest.raises(FileExistsError):
        baseline.write_baseline_rig_classes(path, record, phase)
    assert path.read_bytes() == before


def test_missing_geometry_floor_or_acquisition_never_uses_a_substitute(source):
    record, phase = source
    altered = deepcopy(record)
    altered["full"]["binding"]["geometry"]["raw_source"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="own bound RAW"):
        baseline.baseline_rig_classes(altered, phase)
    altered = deepcopy(phase)
    del altered["chain_assets"]["floor"]
    with pytest.raises(KeyError, match="floor"):
        baseline.baseline_rig_classes(record, altered)
    altered = deepcopy(record)
    altered["full"]["native_origin_xy"] = [0, 0]
    with pytest.raises(ValueError, match="canvas"):
        baseline.baseline_rig_classes(altered, phase)
