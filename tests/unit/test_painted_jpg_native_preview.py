"""Verified JPEG display never rewrites or relaxes original native DATA.

The tiny original owned producer publishes every native/archive/model/review
binding. Its existing decoder, registration and runtime fixture seams remain
synthetic. JPEG decoding, display warp and original source guards are real.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import painted_camera_jpg_display as camera
from darsia.fluidflower import painted_review_comparison as comparison

from .test_painted_analysis_exclusions import _make_gap_case, registered_setup  # noqa: F401
from .test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: F401


@pytest.fixture
def jpeg_case(registered_setup, tmp_path, monkeypatch):
    # Change explicit synthetic camera-U input BEFORE the original producer
    # writes its archive/certificate and before any review is published.
    scale = np.float32(1.01) / np.float32(.17)
    case = _make_gap_case(registered_setup, tmp_path, monkeypatch, native_rgb_scale=scale)
    for row in case.review["runs"]:
        model = gaps._checked(row["target_model"], metadata=True)
        raw = Path(model["reference_image"]["path"])
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3) * np.uint8(11)
        Image.fromarray(rgb).save(raw.with_suffix(".jpg"), quality=100, subsampling=0)
    row = next(row for row in case.review["runs"] if row["components"])
    component = row["components"][0]
    return SimpleNamespace(case=case, review=case.review, row=row, component=component,
                           run=row["run_id"], key=component["component_id"])


def test_verified_jpeg_displays_finite_native_overshoot_and_keeps_original_target_science(jpeg_case):
    fixture = jpeg_case
    row, review = fixture.row, fixture.review
    native_path = Path(row["native_photo"]["path"])
    native = geometry._load_array(native_path)
    assert np.isfinite(native).all() and native.max() > 1 and native.dtype.kind == "f"
    before = {Path(item["path"]): Path(item["path"]).read_bytes()
              for item in (row["native_photo"], row["product"], review["review_binding"])}
    original_review = deepcopy(review)
    with pytest.raises(ValueError, match="original RGB display encoding"):
        gaps.read_gap_preview(review, fixture.run, fixture.key)
    with pytest.raises(ValueError, match="original RGB display encoding"):
        gaps.read_target_paint(review, fixture.run, fixture.key)
    packet = camera.read_gap_preview(review, fixture.run, fixture.key)
    with Image.open(packet["camera_jpeg"]["path"]) as photograph:
        expected = np.array(photograph.convert("RGB"), copy=True)
    # This real fixture's target renderer is the identity camera-U/native map.
    np.testing.assert_array_equal(packet["display_transform"], np.eye(3))
    np.testing.assert_array_equal(packet["image_rgb"], expected)
    assert packet["image_encoding"] == "rgb_uint8" and packet["camera_jpg_display_only"] is True
    assert packet["image_role"] == "current" and packet["valid_support"].all()
    assert packet["display_grid"] == comparison._grid(row)
    arrays = gaps._arrays(row["product"])
    selected = next(mask for component, mask in gaps._components(arrays["gaps"])
                    if component["component_id"] == fixture.key)
    np.testing.assert_array_equal(packet["gap_mask"], selected)
    target = camera.read_target_paint(review, fixture.run, fixture.key)
    np.testing.assert_array_equal(target["target_labels"], arrays["array"].astype(np.uint8))
    np.testing.assert_array_equal(target["editable_mask"], selected)
    assert target["target_correction_basis"] == dict(run_id=fixture.run,
        review=review["review_binding"], review_row_sha256=fixture.component["review_row_sha256"],
        component_id=fixture.key, science=row["target_correction_science"])
    assert all(item["role"] == "porous_medium" for item in target["allowed_layers"])
    assert review == original_review
    for path, raw in before.items():
        assert path.read_bytes() == raw
    np.testing.assert_array_equal(geometry._load_array(native_path), native)


@pytest.mark.parametrize("fault", ["archive", "pixels", "shape"])
def test_jpeg_never_bypasses_original_native_archive_pixels_or_shape(jpeg_case, monkeypatch, fault):
    fixture = jpeg_case
    path = Path(fixture.row["native_photo"]["path"])
    if fault == "archive":
        path.write_bytes(path.read_bytes() + b"changed fixture archive")
    else:
        original = geometry._load_array
        def damaged(selected):
            image = original(selected)
            if Path(selected) == path:
                image = image.copy()
                if fault == "pixels":
                    image.flat[0] += np.float32(.125)
                else:
                    image = image.transpose(1, 0, 2)
            return image
        # Decode-fault injection only. The archive descriptor, original pixel
        # identity checker and all scientific admission predicates stay real.
        monkeypatch.setattr(geometry, "_load_array", damaged)
    with pytest.raises(ValueError):
        camera.read_gap_preview(fixture.review, fixture.run, fixture.key)
    with pytest.raises(ValueError):
        camera.read_target_paint(fixture.review, fixture.run, fixture.key)


@pytest.mark.parametrize("fault", ["nan", "infinite", "uint16", "display_float", "display_shape"])
def test_explicit_jpeg_display_contract_refuses_invalid_native_or_display_arrays(fault):
    # Pure display boundary only: these invalid arrays are not represented as
    # producer-qualified or source-authenticated artifacts.
    native = np.full((2, 3, 3), 1.01, np.float32)
    visible = np.zeros(native.shape, np.uint8)
    if fault == "nan":
        native.flat[0] = np.nan
    elif fault == "infinite":
        native.flat[0] = np.inf
    elif fault == "uint16":
        native = native.astype(np.uint16)
    elif fault == "display_float":
        visible = visible.astype(np.float32)
    else:
        visible = visible.transpose(1, 0, 2)
    with pytest.raises(ValueError):
        gaps._preview_display_rgb(native, visible)
