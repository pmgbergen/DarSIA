"""Metric units, registration composition and the unchanged area recipe."""

from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower import round5_geometry as geometry


PARAMETERS = {"determinant_floor": 1e-12, "area_horizon_floor": 1e-10}


def model(h=None):
    h = [[0.2, 0.0, -10], [0.0, -0.2, 20], [0.0, 0.0, 1.0]] if h is None else h
    return geometry.reference_model(
        "example", {"H_camera_to_mm": h, "sand_area_factor14": 1.034,
                    "explicit_certificate": {"physical_width_mm": 899.0}},
        reference_image={"path": "reference.arw", "sha256": "a" * 64, "size_bytes": 20},
        camera_profile={"camera_wh": [100, 80], "profile_id": "profile"},
        source_model={"sha256": "b" * 64}, edge_review={"sha256": "c" * 64},
        baseline_configuration={"sha256": "d" * 64},
    )


def project(h, point):
    value = np.asarray(h) @ np.r_[point, 1.0]
    return value[:2] / value[2]


def test_metric_reference_and_target_placement_use_the_same_physical_point():
    reference = model()
    target = model([[0.3, 0.01, 5], [0.005, -0.3, 30], [0.0001, 0.0002, 1]])
    placement = geometry.label_placement(reference, target)
    point = np.array([40.0, 25.0])
    moved = project(placement, point)
    np.testing.assert_allclose(project(target["H_camera_u_to_metric_mm"], moved),
                               project(reference["H_camera_u_to_metric_mm"], point), atol=1e-12)
    assert reference["metric_frame"]["unit"] == "mm"
    assert reference["metric_frame"]["y_axis"] == "up"
    assert reference["area_factor"] == 1.034
    assert not reference["adopted"]


def test_frame_registration_origin_and_metre_units_are_not_confused_with_pixels():
    reference = model()
    source_to_baseline = [[1, 0, 30], [0, 1, -20], [0, 0, 1]]
    source_to_frame = [[1, 0, 36], [0, 1, -17], [0, 0, 1]]
    matrices = geometry.frame_matrices(reference, source_to_baseline, source_to_frame,
                                       fixed_area_mm2=0.027053)
    source_point = [10, 50]
    frame_point = project(source_to_frame, source_point)
    expected_mm = project(reference["H_camera_u_to_metric_mm"], project(source_to_baseline, source_point))
    np.testing.assert_allclose(project(matrices["H_frame_native_to_metric_mm"], frame_point), expected_mm)
    np.testing.assert_allclose(project(matrices["H_source_to_metric_m"], source_point), expected_mm / 1000)
    np.testing.assert_allclose(project(matrices["H_metric_mm_to_frame_native"], expected_mm), frame_point)


def test_metric_pixels_cancel_the_old_area_unit_and_optical_factor_is_applied_once():
    reference = model([[2, 0, 0], [0, -3, 0], [0, 0, 1]])
    matrices = geometry.frame_matrices(reference, np.eye(3), np.eye(3), fixed_area_mm2=0.027053)
    row = {"area_parameters": {"H_ac28_to_frame_native": np.eye(3).tolist(),
                               "fixed_area_mm2": 0.027053},
           "phase_shape_hw": [5, 7], "native_origin_xy": [-4, -2]}
    area = geometry.frame_area(row, metric_matrix=matrices["H_source_to_metric_pixel"],
                                area_factor=reference["area_factor"], geometry_parameters=PARAMETERS)
    assert area.dtype == np.float32
    np.testing.assert_allclose(area, np.float32(6.0) * np.float32(1.034), rtol=1e-7)


def test_identity_geometry_keeps_all_old_row_fields_and_caller_inputs():
    original = {"area_parameters": {"H_ac28_to_metric": np.eye(3).tolist()},
                "fixtures": {"a": [[1, 2], [3, 4]]}, "native_origin_xy": [-416, -496]}
    before = deepcopy(original)
    result = geometry.geometry_row(original, np.eye(3))
    assert result == original == before
    result["fixtures"]["a"][0][0] = 10
    assert original == before


def test_identity_area_matches_the_original_double_inverse_and_selected_dtype_rule():
    from darsia.fluidflower.geometry.context import rig_plane_area_map
    from darsia.fluidflower.geometry.areas import apply_selected_area_scale

    h = [[0.9, 0.01, 9], [-0.02, 0.95, 8], [-0.0001, 0.00002, 1]]
    row = {"area_parameters": {"H_ac28_to_frame_native": h, "fixed_area_mm2": 0.027053},
           "phase_shape_hw": [11, 13], "native_origin_xy": [-4, -5]}
    old = rig_plane_area_map(np.eye(3) @ np.linalg.inv(h), (11, 13),
                              origin_xy=(-4, -5), metric_pixel_area_mm2=0.027053,
                              parameters=PARAMETERS)
    old = apply_selected_area_scale(old, 0.9224286)
    actual = geometry.frame_area(row, metric_matrix=np.eye(3), area_factor=0.9224286,
                                  geometry_parameters=PARAMETERS)
    assert actual.tobytes() == old.tobytes()


@pytest.mark.parametrize("value", [[[1, 0], [0, 1]], [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                                  [[1, 0, float('nan')], [0, 1, 0], [0, 0, 1]]])
def test_bad_matrices_are_refused(value):
    with pytest.raises(ValueError):
        geometry.matrix(value)


def test_a_saved_model_cannot_change_behind_its_sha_binding(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    bound = geometry.descriptor(path)
    path.write_text('{"value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        geometry.read_bound_json(bound)


def bridged_model(tmp_path, *, displacement=0.2, accepted=True, baseline_sha=None):
    baseline = {"path": "baseline.arw", "sha256": "a" * 64, "size_bytes": 20}
    photograph = {"path": "model.arw", "sha256": "e" * 64, "size_bytes": 21}
    transport = np.eye(3); transport[0, 2] = displacement
    bridge = geometry.write_new_json(tmp_path / "bridge.json", {
        "accepted": accepted, "required": True,
        "baseline_raw": {**baseline, "sha256": baseline_sha or baseline["sha256"]},
        "model_raw": photograph, "H_baseline_u_to_model_u": transport.tolist(),
        "maximum_displacement_px": abs(displacement), "displacement_limit_px": 0.5,
    })
    native_to_u = [[1, 0, 57], [0, 1, 0], [0, 0, 1]]
    native = geometry.write_new_json(tmp_path / "native.json", {
        "shape_hw": [70, 100], "H_rig_corrected_native_to_baseline_camera_u": native_to_u,
        "renderer": {"raw_acquisition": baseline, "camera_u_wh": [100, 80],
                     "native_shape_hw": [70, 100], "native_to_camera_u_h": native_to_u},
    })
    return geometry.reference_model(
        "example", {"H_camera_to_mm": [[0.2, 0, -10], [0, -0.2, 20], [0, 0, 1]],
                    "sand_area_factor14": 1.034, "explicit_certificate": {}},
        reference_image=baseline, model_photograph=photograph,
        baseline_to_model=bridge, baseline_native_renderer=native,
        camera_profile={"camera_wh": [100, 80]}, source_model={}, edge_review={},
        baseline_configuration={},
    )


def test_different_photograph_uses_measured_baseline_bridge_and_separate_native_grid(tmp_path):
    value = bridged_model(tmp_path)
    np.testing.assert_allclose(project(value["H_camera_u_to_metric_mm"], [40, 25]), [-1.96, 15])
    assert value["reference_image"]["sha256"] != value["model_photograph"]["sha256"]
    native = value["baseline_native_grid"]
    assert native["shape_hw"] == [70, 100] and value["shape_hw"] == [80, 100]
    np.testing.assert_allclose(project(native["H_baseline_camera_u_to_rig_corrected_native"], [70, 25]), [13, 25])
    assert value["area_factor"] == 1.034 and not value["new_fit"]


@pytest.mark.parametrize("kwargs,match", [
    ({"accepted": False}, "not passed"),
    ({"displacement": 0.5001}, "exceeds"),
    ({"baseline_sha": "f" * 64}, "another acquisition"),
])
def test_unaccepted_large_or_wrong_acquisition_bridge_is_refused(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        bridged_model(tmp_path, **kwargs)
