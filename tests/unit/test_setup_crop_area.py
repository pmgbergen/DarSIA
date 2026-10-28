"""Storage and geometry of the separate Setup crop area (local fixtures)."""

import json
import math

import numpy as np
import pytest

from darsia.fluidflower import setup_crop_area as crop
from darsia.fluidflower.series_geometry import (
    CROP_CORNERS,
    RigFrame,
    create_series_geometry_project,
    native_series_link,
)

# 100 px per metre in both directions; rig 2.0 m x 1.0 m.
SQUARE = {
    "top_left": (10.0, 20.0),
    "bottom_left": (110.0, 20.0),
    "bottom_right": (110.0, 220.0),
    "top_right": (10.0, 220.0),
}


def _frame(corners=SQUARE, **extra):
    values = dict(
        corners=dict(corners), width_m=2.0, height_m=1.0, run_id="ac1",
        image="ac1.png", image_size=(240, 130),
    )
    values.update(extra)
    return RigFrame(**values)


def _area(corners, **extra):
    values = dict(corners=corners, run_id="ac1", image="ac1.png", image_size=(240, 130))
    values.update(extra)
    return crop.CropArea(**values)


@pytest.fixture
def project(tmp_path):
    return create_series_geometry_project(
        series_id="local", series_name="Local", series_root=tmp_path / "series"
    )


def test_the_choice_defaults_to_the_reference_quad_and_writes_only_changes(project):
    manifest = project.root / "series.json"
    before = manifest.read_bytes()
    assert crop.saved_crop_choice(project) == crop.CROP_REFERENCE_QUAD
    assert crop.save_crop_choice(project, crop.CROP_REFERENCE_QUAD) is False
    assert manifest.read_bytes() == before
    assert crop.save_crop_choice(project, crop.CROP_SEPARATE_AREA, actor="tester") is True
    assert crop.saved_crop_choice(project) == crop.CROP_SEPARATE_AREA
    assert native_series_link(project)["crop_choice"] == "separate_area"
    written = manifest.read_bytes()
    assert crop.save_crop_choice(project, crop.CROP_SEPARATE_AREA) is False
    assert manifest.read_bytes() == written
    assert crop.save_crop_choice(project, crop.CROP_REFERENCE_QUAD) is True
    assert crop.saved_crop_choice(project) == crop.CROP_REFERENCE_QUAD
    with pytest.raises(ValueError, match="Unknown Setup crop choice"):
        crop.save_crop_choice(project, "whole photograph")


def test_an_unknown_stored_choice_is_refused_until_a_choice_replaces_it(project):
    project.update_metadata(
        "tools", {"native_series": dict(native_series_link(project), crop_choice="half")}
    )
    with pytest.raises(ValueError, match="Unknown Setup crop choice"):
        crop.saved_crop_choice(project)
    assert crop.save_crop_choice(project, crop.CROP_REFERENCE_QUAD) is True
    assert crop.saved_crop_choice(project) == crop.CROP_REFERENCE_QUAD


def test_the_separate_area_reason_follows_the_saved_geometry(project):
    assert crop.separate_crop_reason(project) == ""
    current = project.root / "overrides" / "geometry_facies" / "current.json"
    current.parent.mkdir(parents=True, exist_ok=True)
    revision = {"format": "darsia-fluidflower-geometry-facies-v1", "revision_id": "r1",
                "configuration": {"frame": "raw"}}
    current.write_text(json.dumps(revision), encoding="utf-8")
    assert crop.separate_crop_reason(project) == ""
    revision["configuration"]["frame"] = "rig-corrected"
    current.write_text(json.dumps(revision), encoding="utf-8")
    assert "already cropped" in crop.separate_crop_reason(project)
    assert crop.separate_crop_reason(project, configuration={"frame": "raw"}) == ""
    current.write_text("{", encoding="utf-8")
    assert "could not be read" in crop.separate_crop_reason(project)


def test_areas_are_stored_by_run_next_to_the_rig_frames(project):
    marked = {"top_left": (30, 50), "top_right": (32, 200), "bottom_right": (95, 198),
              "bottom_left": (94, 52)}
    area = _area(marked)
    assert crop.saved_crop_areas(project) == {}
    crop.save_crop_area(project, area, actor="tester")
    crop.save_crop_area(project, _area(marked, run_id="ac2", image="ac2.png"))
    link = native_series_link(project)
    assert set(link["crop_areas"]) == {"ac1", "ac2"} and "updated_utc" in link["crop_areas"]["ac1"]
    assert crop.saved_crop_areas(project)["ac1"] == area
    assert crop.remove_crop_area(project, "ac2") is True
    assert crop.remove_crop_area(project, "ac2") is False
    assert set(crop.saved_crop_areas(project)) == {"ac1"}
    assert crop.CropArea.from_dict(area.to_dict()) == area


def test_area_corners_are_named_by_position_and_checked():
    clicked = {"top_left": (95.0, 198.0), "top_right": (30.0, 50.0),
               "bottom_right": (32.0, 200.0), "bottom_left": (94.0, 52.0)}
    area = _area(clicked)
    assert area.corners["top_left"] == (30.0, 50.0)
    assert area.corners["bottom_right"] == (95.0, 198.0)
    assert area.to_dict()["corners"]["top_right"] == [32.0, 200.0]
    for corners, message in (
        ({name: SQUARE[name] for name in CROP_CORNERS[:3]}, "four named corners"),
        (dict(SQUARE, top_left=(math.nan, 1.0)), "Non-finite corner"),
        (dict(SQUARE, top_left=(1.0,)), "Invalid corner"),
        (dict(SQUARE, top_left="10, 20"), "Invalid corner"),
        (dict(SQUARE, top_left=SQUARE["top_right"]), "not distinct"),
    ):
        with pytest.raises(ValueError, match=message):
            _area(corners)
    with pytest.raises(ValueError, match="belongs to a run"):
        _area(SQUARE, run_id="")


def test_a_new_area_starts_as_the_reference_quad():
    frame = _frame()
    area = crop.crop_area_from_frame(frame)
    assert area.corners == {name: SQUARE[name] for name in CROP_CORNERS}
    assert (area.run_id, area.image, area.image_size) == ("ac1", "ac1.png", (240, 130))
    with pytest.raises(ValueError, match="four corners"):
        crop.crop_area_from_frame(_frame({name: SQUARE[name] for name in CROP_CORNERS[:3]}))


def test_the_extent_is_the_rectangle_between_the_marked_points():
    frame = _frame()
    # Rig x = (col - 20) / 100 and y = 1 - (row - 10) / 100.
    area = _area({"top_left": (30.0, 50.0), "top_right": (31.0, 200.0),
                  "bottom_right": (95.0, 199.0), "bottom_left": (94.0, 52.0)})
    extent = crop.crop_extent(frame, area)
    assert (extent.left_m, extent.right_m) == (0.32, 1.79)
    assert (extent.bottom_m, extent.top_m) == (0.16, 0.79)
    assert (extent.width_m, extent.height_m) == (1.47, 0.63)
    assert extent.origin_m == (0.32, 0.79)
    expected = {"top_left": (31.0, 52.0), "bottom_left": (94.0, 52.0),
                "bottom_right": (94.0, 199.0), "top_right": (31.0, 199.0)}
    for name in CROP_CORNERS:
        assert extent.corners[name] == pytest.approx(expected[name])
    table = extent.native_crop()
    assert list(table) == [*CROP_CORNERS, "width", "height", "in_meters", "origin"]
    assert table["top_left"] == [31, 52] and table["bottom_right"] == [94, 199]
    assert (table["width"], table["height"], table["in_meters"]) == (1.47, 0.63, True)
    assert table["origin"] == [0.32, 0.79]
    moved = extent.native_crop({name: (row + 2.4, col - 3.6) for name, (row, col) in expected.items()})
    assert moved["top_left"] == [33, 48] and moved["origin"] == [0.32, 0.79]


def test_the_rectangle_edges_are_rounded_inward_to_a_tenth_of_a_millimetre():
    frame = _frame()
    area = _area({"top_left": (30.004, 50.004), "top_right": (30.004, 199.996),
                  "bottom_right": (89.996, 199.996), "bottom_left": (89.996, 50.004)})
    extent = crop.crop_extent(frame, area)
    assert (extent.left_m, extent.right_m) == (0.3001, 1.7999)
    assert (extent.bottom_m, extent.top_m) == (0.2001, 0.7999)
    exact = _area({"top_left": (30.0, 50.0), "top_right": (30.0, 200.0),
                   "bottom_right": (90.0, 200.0), "bottom_left": (90.0, 50.0)})
    assert crop.crop_extent(frame, exact).origin_m == (0.3, 0.8)


def test_a_perspective_reference_quad_gives_the_coordinates():
    tilted = {"top_left": (12.0, 25.0), "bottom_left": (118.0, 14.0),
              "bottom_right": (106.0, 236.0), "top_right": (6.0, 214.0)}
    frame = _frame(tilted)
    marked = {name: frame.rig_to_pixel(x, y) for name, (x, y) in {
        "top_left": (0.25, 0.9), "bottom_left": (0.2, 0.1),
        "bottom_right": (1.8, 0.15), "top_right": (1.7, 0.95)}.items()}
    extent = crop.crop_extent(frame, _area(marked))
    assert (extent.left_m, extent.right_m, extent.bottom_m, extent.top_m) == (0.25, 1.7, 0.15, 0.9)
    for name, point in {"top_left": (0.25, 0.9), "bottom_left": (0.25, 0.15),
                        "bottom_right": (1.7, 0.15), "top_right": (1.7, 0.9)}.items():
        assert frame.pixel_to_rig(*extent.corners[name]) == pytest.approx(point)
    # Larger than the reference quad is allowed; the coordinates continue.
    outside = {name: frame.rig_to_pixel(x, y) for name, (x, y) in {
        "top_left": (-0.1, 1.05), "bottom_left": (-0.1, -0.05),
        "bottom_right": (2.1, -0.05), "top_right": (2.1, 1.05)}.items()}
    wide = crop.crop_extent(frame, _area(outside))
    assert wide.origin_m == (-0.1, 1.05) and (wide.width_m, wide.height_m) == (2.2, 1.1)


def test_the_extent_is_refused_with_its_reason():
    frame = _frame()
    area = crop.crop_area_from_frame(frame)
    for broken in (_frame({name: SQUARE[name] for name in CROP_CORNERS[:3]}),
                   _frame(width_m=None)):
        with pytest.raises(ValueError, match="reference quadrilateral needs"):
            crop.crop_extent(broken, area)
    with pytest.raises(ValueError, match="marked on run ac2"):
        crop.crop_extent(frame, _area(SQUARE, run_id="ac2"))
    with pytest.raises(ValueError, match="another size"):
        crop.crop_extent(frame, _area(SQUARE, image_size=(120, 65)))
    thin = {"top_left": (30.0, 50.0), "top_right": (30.0, 50.004),
            "bottom_right": (90.0, 50.004), "bottom_left": (90.0, 50.0)}
    with pytest.raises(ValueError, match="no rectangle"):
        crop.crop_extent(frame, _area(thin))


def test_an_export_records_an_area_only_for_the_separate_choice(project):
    frame = _frame()
    area = _area({"top_left": (30.0, 50.0), "top_right": (30.0, 200.0),
                  "bottom_right": (90.0, 200.0), "bottom_left": (90.0, 50.0)})
    crop.save_crop_area(project, area)
    assert crop.export_crop_area(project, frame) is None
    assert crop.export_crop_area(project, None) is None
    crop.save_crop_choice(project, crop.CROP_SEPARATE_AREA)
    assert crop.export_crop_area(project, frame) == area.to_dict()
    with pytest.raises(ValueError, match="reference quadrilateral of the reference run"):
        crop.export_crop_area(project, None)
    with pytest.raises(ValueError, match="Mark the separate crop area on the reference photograph of run ac2"):
        crop.export_crop_area(project, _frame(run_id="ac2", image="ac2.png"))
    crop.save_crop_area(project, _area({"top_left": (30.0, 50.0), "top_right": (30.0, 50.004),
                                        "bottom_right": (90.0, 50.004), "bottom_left": (90.0, 50.0)}))
    with pytest.raises(ValueError, match="no rectangle"):
        crop.export_crop_area(project, frame)


def _inside(outline, point, tolerance=1e-6):
    """Whether a (row, col) point lies in a convex outline, edges included."""
    crosses = [
        (c1 - c0) * (point[0] - r0) - (r1 - r0) * (point[1] - c0)
        for (r0, c0), (r1, c1) in zip(outline, outline[1:] + outline[:1])
    ]
    return all(value >= -tolerance for value in crosses) or all(
        value <= tolerance for value in crosses
    )


def test_the_extent_stays_inside_the_marked_area():
    frame = _frame({"top_left": (12.0, 25.0), "bottom_left": (118.0, 14.0),
                    "bottom_right": (106.0, 236.0), "top_right": (6.0, 214.0)})
    rng = np.random.default_rng(3)
    for _ in range(20):
        jitter = rng.uniform(-6.0, 6.0, size=(4, 2))
        marked = {name: (row + dr, col + dc) for (name, (row, col)), (dr, dc) in zip(
            {"top_left": (30.0, 60.0), "bottom_left": (95.0, 55.0),
             "bottom_right": (92.0, 190.0), "top_right": (28.0, 185.0)}.items(), jitter)}
        area = _area(marked)
        extent = crop.crop_extent(frame, area)
        outline = [area.corners[name] for name in CROP_CORNERS]
        assert all(_inside(outline, extent.corners[name]) for name in CROP_CORNERS)
