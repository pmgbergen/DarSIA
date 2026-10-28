"""Review camera conventions use the original renderer's sensor metadata."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from darsia.fluidflower import painted_review_comparison as display
from darsia.fluidflower.geometry_raster_sources import descriptor


@pytest.mark.parametrize("orientation", [1, 6])
def test_camera_context_uses_recorded_orientation_with_no_profile_orientation_and_refuses_tampering(
        tmp_path, orientation):
    # The real registration profile configures the camera grid. RAW orientation
    # is actual decoded sensor metadata, recorded separately by rig_raw_build.
    path = tmp_path / "current-profile.json"
    path.write_text(json.dumps(dict(registration=dict(common=dict(
        sensor=dict(camera_wh=[40, 30]))))), encoding="utf8")
    profile = dict(descriptor(path), profile_id="synthetic-current", camera_wh=[40, 30])
    geometry = SimpleNamespace(model=dict(camera_profile=profile))
    certificate = dict(
        inputs=dict(profile=descriptor(path), uncorrected_grid=dict(
            num_voxels=[30, 40], homography_coordinates=["column", "row", "one"],
            pixel_centres="integer indices", indexing="ij",
            array_row_direction="down", array_column_direction="right")),
        renderer=dict(sensor_metadata=dict(
            orientation=orientation, white_level=4095, black_level=0)),
        homography_convention=dict(
            source="saved corrected-baseline pixel centres (column,row)",
            target="camera-U pixel centres (column,row)",
            camera_u_axes="x right, y down", raw_orientation=orientation))
    before = deepcopy(certificate)
    context = display._camera_u_context(geometry, certificate)
    assert context["raw_orientation"] == orientation
    assert type(context["raw_orientation"]) is int
    assert context["profile"] == profile and context["shape_hw"] == [30, 40]
    assert certificate == before

    missing = deepcopy(certificate)
    missing["renderer"].pop("sensor_metadata")
    with pytest.raises(ValueError, match="recorded integer RAW orientation"):
        display._camera_u_context(geometry, missing)

    boolean = deepcopy(certificate)
    boolean["renderer"]["sensor_metadata"]["orientation"] = True
    with pytest.raises(ValueError, match="recorded integer RAW orientation"):
        display._camera_u_context(geometry, boolean)

    changed = deepcopy(certificate)
    changed["renderer"]["sensor_metadata"]["orientation"] = 6 if orientation == 1 else 1
    with pytest.raises(ValueError, match="coordinate convention differs"):
        display._camera_u_context(geometry, changed)

    boolean_convention = deepcopy(certificate)
    boolean_convention["homography_convention"]["raw_orientation"] = True
    with pytest.raises(ValueError, match="coordinate convention differs"):
        display._camera_u_context(geometry, boolean_convention)

    # A legacy extra profile field cannot substitute for the required actual
    # renderer metadata, even when it happens to equal the convention.
    alternate = tmp_path / "profile-with-extra-orientation.json"
    alternate.write_text(json.dumps(dict(common=dict(sensor=dict(
        camera_wh=[40, 30], orientation=orientation)))), encoding="utf8")
    configured = SimpleNamespace(model=dict(camera_profile=dict(
        descriptor(alternate), profile_id="synthetic-current", camera_wh=[40, 30])))
    missing["inputs"]["profile"] = descriptor(alternate)
    with pytest.raises(ValueError, match="recorded integer RAW orientation"):
        display._camera_u_context(configured, missing)
