"""An optional crop origin moves only the coordinates of the cropped image."""

import numpy as np
import pytest

import darsia
from darsia.corrections.shape.curvature import crop_origin
from darsia.gui.ui.schema.dataclass_introspection import _build_fields
from darsia.presets.workflows.config.corrections import (
    CropCorrectionConfig,
    CurvatureCorrectionConfig,
)


def _section(**extra):
    crop = {
        "top_left": [3, 4],
        "bottom_left": [44, 6],
        "bottom_right": [43, 73],
        "top_right": [4, 72],
        "width": 0.6,
        "height": 0.3,
        "in_meters": True,
    }
    crop.update(extra)
    return {"active": ["crop"], "crop": crop}


def _image():
    return darsia.ScalarImage(
        np.indices((50, 80)).sum(axis=0).astype(np.float32) / 128.0,
        dimensions=[0.5, 0.8],
    )


def _corrected(section):
    settings = CurvatureCorrectionConfig().load(section).to_dict()
    correction = darsia.CurvatureCorrection(config=settings)
    return settings, correction, correction(_image())


@pytest.mark.parametrize(
    "value, expected",
    [([0.1, 0.45], (0.1, 0.45)), ((-2, 3), (-2.0, 3.0)), (np.array([0.5, 1.5]), (0.5, 1.5))],
)
def test_crop_origin_reads_two_finite_numbers(value, expected):
    result = crop_origin(value)
    assert result == expected and all(type(item) is float for item in result)


@pytest.mark.parametrize(
    "value",
    [[0.1], [0.1, 0.2, 0.3], "0.1, 0.2", ["0.1", "0.2"], [True, 0.2], [float("nan"), 0.2],
     [0.1, float("inf")], None, 0.1],
)
def test_crop_origin_refuses_anything_else(value):
    with pytest.raises(ValueError, match="curvature.crop.origin"):
        crop_origin(value)


def test_without_origin_the_crop_keeps_its_zero_based_coordinates():
    settings, correction, corrected = _corrected(_section())
    assert "origin" not in settings["crop"] and "origin" not in correction.config["crop"]
    assert CropCorrectionConfig().load(_section()["crop"]).origin is None
    assert np.allclose(corrected.origin, [0.0, 0.3])
    assert np.allclose(corrected.dimensions, [0.3, 0.6])


def test_an_origin_moves_the_coordinates_and_keeps_every_pixel():
    plain = _corrected(_section())[2]
    settings, correction, shifted = _corrected(_section(origin=[0.15, 0.4]))
    assert settings["crop"]["origin"] == [0.15, 0.4]
    assert correction.config["crop"]["origin"] == [0.15, 0.4]
    assert np.array_equal(shifted.img, plain.img)
    assert np.allclose(shifted.dimensions, plain.dimensions)
    assert np.allclose(shifted.origin, [0.15, 0.4])
    assert np.allclose(shifted.opposite_corner, [0.75, 0.1])
    # The top-left voxel starts at the moved corner; its centre lies half a
    # voxel inside it.
    voxel = np.asarray(shifted.voxel_size)
    corner = shifted.coordinatesystem.coordinate(darsia.make_voxel([0, 0]))
    centre = shifted.coordinatesystem.coordinate(darsia.make_voxel_center([0, 0]))
    assert np.allclose(corner, [0.15, 0.4])
    assert np.allclose(centre, [0.15 + voxel[1] / 2, 0.4 - voxel[0] / 2])


def test_a_saved_crop_correction_keeps_its_origin(tmp_path):
    _, correction, shifted = _corrected(_section(origin=[0.15, 0.4]))
    path = tmp_path / "shape_correction_curvature.npz"
    correction.save(path)
    loaded = darsia.read_correction(path)
    again = loaded(_image())
    assert np.allclose(again.origin, shifted.origin)
    assert np.array_equal(again.img, shifted.img)


def test_the_settings_panel_does_not_offer_the_origin():
    names = [row["key"] for row in _build_fields(CropCorrectionConfig, "corrections.curvature.crop")]
    assert "corrections.curvature.crop.top_left" in names
    assert not any(name.endswith(".origin") for name in names)


def test_an_invalid_origin_is_refused_when_the_settings_are_read():
    with pytest.raises(ValueError, match="curvature.crop.origin"):
        CurvatureCorrectionConfig().load(_section(origin=[0.15]))
    with pytest.raises(ValueError, match="curvature.crop.origin"):
        darsia.CurvatureCorrection(config=_section(origin=["left", "top"]))
