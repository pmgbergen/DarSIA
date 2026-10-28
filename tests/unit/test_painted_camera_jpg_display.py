"""Actual tiny JPEG/metadata files and original display warp; no RAW decoder."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import painted_camera_jpg_display as display
from darsia.fluidflower import painted_review_comparison as comparison


@pytest.fixture
def camera(tmp_path):
    import json
    raw = tmp_path / "DSC00001.ARW"
    raw.write_bytes(b"synthetic RAW identity; never decoded")
    jpeg = raw.with_suffix(".JPG")
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3) * np.uint8(5)
    Image.fromarray(pixels).save(jpeg, quality=100, subsampling=0)
    parameters = tmp_path / "profile.json"
    parameters.write_text(json.dumps(dict(registration=dict(common=dict(sensor=dict(camera_wh=[4, 3]))))), encoding="utf8")
    bound = comparison._bound(parameters)
    geometry = SimpleNamespace(model=dict(camera_profile=dict(bound, camera_wh=[4, 3])))
    convention = dict(source="saved corrected-baseline pixel centres (column,row)",
        target="camera-U pixel centres (column,row)", camera_u_axes="x right, y down", raw_orientation=1)
    grid = dict(num_voxels=[3, 4], homography_coordinates=["column", "row", "one"],
        pixel_centres="integer indices", indexing="ij", array_row_direction="down", array_column_direction="right")
    certificate = dict(renderer=dict(sensor_metadata=dict(orientation=1)), homography_convention=convention,
        inputs=dict(profile=bound, uncorrected_grid=grid))
    return SimpleNamespace(raw=raw, jpeg=jpeg, binding=comparison._bound(raw), geometry=geometry,
                           certificate=certificate, profile=bound)


def test_camera_jpg_uses_stored_camera_u_pixels_and_original_native_display_warp(camera):
    before = camera.raw.read_bytes()
    with Image.open(camera.jpeg) as image:
        expected = np.array(image.convert("RGB"), copy=True)
    rgb, binding = display._camera_rgb(camera.binding, camera.geometry, camera.certificate)
    np.testing.assert_array_equal(rgb, expected)
    assert rgb.dtype == np.uint8 and binding == comparison._bound(camera.jpeg)
    row = dict(gap=dict(shape_hw=[3, 4]), native_photo={"synthetic": "native"},
               native_photo_pixels={"synthetic": "pixels"}, target_model=camera.profile)
    original = deepcopy(row)
    h = np.array([[1., 0., 1.], [0., 1., 0.], [0., 0., 1.]])
    packet = display._mapped(row, camera.geometry, camera.certificate, camera.binding, h)
    target = np.zeros_like(expected)
    target[:, 1:] = expected[:, :-1]
    np.testing.assert_array_equal(packet["image_rgb"], target)
    assert packet["image_encoding"] == "rgb_uint8" and packet["camera_jpg_display_only"] is True
    assert not packet["valid_support"][:, 0].any() and packet["valid_support"][:, 1:].all()
    assert packet["display_grid"] == comparison._grid(row) and row == original
    assert camera.raw.read_bytes() == before


@pytest.mark.parametrize("damage", ["missing", "grid", "profile"])
def test_camera_jpg_display_refuses_wrong_exposure_grid_or_changed_bound_profile(camera, damage):
    raw = dict(camera.binding)
    if damage == "missing":
        raw["path"] = str(Path(raw["path"]).with_name("DSC00002.ARW"))
    elif damage == "grid":
        Image.new("RGB", (5, 3)).save(camera.jpeg)
    else:
        Path(camera.profile["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError):
        display._camera_rgb(raw, camera.geometry, camera.certificate)


def test_camera_jpg_mpo_container_uses_only_first_stored_frame(camera):
    first = Image.new("RGB", (4, 3), (240, 20, 10))
    second = Image.new("RGB", (4, 3), (10, 20, 240))
    first.save(camera.jpeg, format="MPO", save_all=True, append_images=[second], quality=100, subsampling=0)
    with Image.open(camera.jpeg) as photograph:
        assert photograph.format == "MPO" and photograph.n_frames == 2
        photograph.seek(0)
        expected = np.array(photograph.convert("RGB"), copy=True)
        photograph.seek(1)
        other = np.array(photograph.convert("RGB"), copy=True)
    assert not np.array_equal(expected, other)
    rgb, binding = display._camera_rgb(camera.binding, camera.geometry, camera.certificate)
    np.testing.assert_array_equal(rgb, expected)
    assert binding == comparison._bound(camera.jpeg)