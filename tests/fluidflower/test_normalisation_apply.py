"""Historical B1 application keeps source binding and finite optical support."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.native import normalisation_apply as apply
from darsia.fluidflower.native.io import sha256_file


@pytest.fixture
def bound_application(tmp_path, monkeypatch):
    paths = {}
    for name in ("camera", "raw", "review"):
        paths[name] = tmp_path / name
        paths[name].write_bytes(name.encode())
    certificate = tmp_path / "certificate.json"
    certificate.write_text(json.dumps(dict(sources=dict(review_native_jpeg_sha256="baseline-review"),
                                           review_to_camera_h=np.eye(3).tolist())))
    frame = dict(camera_jpeg_sha256=sha256_file(paths["camera"]), native_review_jpeg_sha256=sha256_file(paths["review"]),
        v5_B_arw_sha256=sha256_file(paths["raw"]), v5_B_camera_jpeg_sha256=sha256_file(paths["camera"]),
        v5_B_native_jpeg_sha256="baseline-review", v5_B_camera_jpeg_path=str(paths["camera"]),
        v5_B_registration_certificate_path=str(certificate), photo_sand_reference_mask_path=str(tmp_path / "mask"),
        photo_sand_reference_mask_sha256="mask-sha", href=np.eye(3), H_v5_B_to_frame_native=np.eye(3))
    config = dict(source=dict(asset_aliases={}, policy_ids=dict(keep="keep", raw_display="raw"),
        native_shape_hw=[2, 3], content_box_xywh=[0, 0, 3, 2], mask_guard_px=1, signal_denominator=65535.,
        renderer_source_ids=dict(ac28="camera"), profiles=dict(ac28="old"), homography_min_scale=1e-12,
        symbols=dict(reference_to_frame_h_key="href"), chart_shape=[2, 2], cell_scale=2, cell_center=.5),
        support={}, registration={}, photometry={})
    processor = SimpleNamespace(parameters=config, run="ac28", row=dict(source_manifest_sha256="manifest"),
        enclosed_hole_policy="keep", isolated_invalid_policy="keep", display_mode="raw",
        cohort=SimpleNamespace(frame=lambda index: frame), grid=SimpleNamespace(origin_cell_xy=(0, 0)),
        fit=SimpleNamespace(xy=np.array([[0., 0.], [1., 1.]], np.float32), CB=np.zeros((3, 2))),
        reference=object(), old_field=np.full((3, 2, 3), .125, np.float32))
    safe = np.array([[True, True, False], [True, True, True]])
    values = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / np.float32(4.)
    values[1, 1, 1] = np.nan
    calls = []
    monkeypatch.setattr(apply.cv2.ipp, "useIPP", lambda: True)
    monkeypatch.setattr(apply, "source_photo_grid", lambda *a, **k: safe)
    monkeypatch.setattr(apply, "project_source_photo_safe_core", lambda *a, **k: safe.copy())
    def render(*args, **kwargs):
        calls.append(("render", kwargs))
        return np.ones((2, 3, 3), np.float32), dict(review_to_camera_h=np.eye(3))
    def chart(**kwargs):
        calls.append(("chart", kwargs))
        return np.ones((2, 2), np.float32), np.array([True, False]), {}
    def correct(*args, **kwargs):
        calls.append(("correct", kwargs))
        return values, {}
    monkeypatch.setattr(apply, "registered_edge_aware_sensor_linear", render)
    monkeypatch.setattr(apply, "register_chart", chart)
    monkeypatch.setattr(apply, "correct_frame_native", correct)
    kwargs = dict(run="ac28", frame_id="frame_0", camera_jpeg=paths["camera"], camera_arw=paths["raw"],
        review_jpeg=paths["review"], source_manifest_sha256="manifest", content_shape_hw=(2, 3), content_box_xywh=(0, 0, 3, 2))
    return processor, frame, values, calls, kwargs


@pytest.mark.parametrize("baseline", [True, False])
def test_baseline_and_later_frame_keep_original_support_and_chart_binding(bound_application, baseline):
    processor, frame, values, calls, kwargs = bound_application
    if not baseline:
        frame["v5_B_camera_jpeg_sha256"] = "earlier-camera"
    before = values.copy()
    result = apply.apply_historical_frame(processor, **kwargs)
    expected_mask = np.array([[True, True, False], [True, False, True]])
    assert np.array_equal(result.required_valid_mask, expected_mask)
    assert result.log_linear_rgb.dtype == np.float32
    assert result.log_linear_rgb[:, expected_mask].tobytes() == before[:, expected_mask].tobytes()
    assert np.isnan(result.log_linear_rgb[:, ~expected_mask]).all()
    assert values.tobytes() == before.tobytes()
    assert [name for name, _ in calls] == (["render", "correct"] if baseline else ["render", "chart", "correct"])
    assert calls[-1][1]["exact_baseline_source"] is baseline
    assert calls[-1][1]["old_field_ref8"] is processor.old_field
    if not baseline:
        assert calls[1][1]["baseline_chart_log_rgb"] is processor.fit.CB
        assert calls[-1][1]["frame_chart_usable"].tolist() == [True, False]


def test_wrong_runtime_cannot_silently_reproduce_a_different_image(bound_application, monkeypatch):
    processor, _, _, calls, kwargs = bound_application
    monkeypatch.setattr(apply.cv2.ipp, "useIPP", lambda: False)
    with pytest.raises(RuntimeError, match="requires IPP in its separate process"):
        apply.apply_historical_frame(processor, **kwargs)
    assert not calls


@pytest.mark.parametrize("change", ["camera", "manifest", "policy"])
def test_changed_source_or_choices_fail_before_calculation(bound_application, change):
    processor, _, _, calls, kwargs = bound_application
    if change == "camera":
        kwargs["camera_jpeg"].write_bytes(b"changed acquisition")
    elif change == "manifest":
        kwargs["source_manifest_sha256"] = "another revision"
    else:
        processor.isolated_invalid_policy = "fill"
    with pytest.raises(ValueError):
        apply.apply_historical_frame(processor, **kwargs)
    assert not calls
