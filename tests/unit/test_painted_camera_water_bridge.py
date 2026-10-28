"""Original RAW water proposal on an actual same-exposure JPEG display.

Reuse the existing finite bound-metadata fixture and its RAW decoder seam.
The original role/model readers, native warp, basis, proposal and measurement
code run unchanged. Synthetic source pixels establish no campaign accuracy.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import painted_camera_jpg_display as camera
from darsia.fluidflower import painted_review_comparison as comparison
from darsia.fluidflower import painted_water_surface as water

from .test_painted_review_comparison import source, _clear_selection, _descriptor  # noqa: F401


@pytest.fixture
def bridge_case(source, tmp_path, monkeypatch):
    catalog = _clear_selection(source, tmp_path)
    selection = catalog["runs"]["ac18"]
    raw = selection["selected_raw"]
    jpeg = Path(raw["path"]).with_suffix(".jpg")
    Image.new("RGB", (40, 30), (90, 90, 90)).save(jpeg, quality=100, subsampling=0)
    jpeg_binding = _descriptor(jpeg, jpeg.read_bytes())
    # The existing fixture's finite metadata reader still checks exact bound
    # descriptors; JPEG bytes are independently read/hashed by the real reader.
    source.documents[str(jpeg)] = (jpeg_binding, None)
    raw_rgb = np.full((30, 40, 3), 20, np.uint8)
    raw_rgb[6:] = 200
    calls = []

    def decode(path, **options):
        assert Path(path) == Path(raw["path"])
        assert options["expected_sha256"] == raw["sha256"] and options["factor"] == 1
        assert str(options["profile_path"]) == source.models["ac18"]["camera_profile"]["path"]
        assert options["profile_sha256"] == source.models["ac18"]["camera_profile"]["sha256"]
        assert options["profile_id"] == "same"
        calls.append((str(path), deepcopy(options)))
        return raw_rgb.copy()

    from darsia.fluidflower.native import raw_photograph
    monkeypatch.setattr(raw_photograph, "load_camera_u_rgb8", decode)
    return SimpleNamespace(source=source, raw=raw, jpeg=jpeg, jpeg_binding=jpeg_binding,
        raw_rgb=raw_rgb, calls=calls, original_reader=comparison.read_comparison_preview)


def test_jpg_keeps_original_raw_water_basis_proposal_and_primary_own_geometry_height(bridge_case):
    case = bridge_case
    before = case.raw_rgb.copy(), Path(case.raw["path"]).read_bytes(), case.jpeg.read_bytes()
    display = camera.read_comparison_preview(case.source.review, "ac18", None, "clearwater")
    assert display["water_surface"] is None and not case.calls
    original = case.original_reader(case.source.review, "ac18", None, "clearwater")
    result = camera.read_water_surface_comparison_preview(case.source.review, "ac18", None, "clearwater")
    assert len(case.calls) == 2
    np.testing.assert_array_equal(result["image_rgb"], display["image_rgb"])
    assert result["source_raw"] == case.raw and result["camera_jpeg"] == case.jpeg_binding
    assert result["water_surface"] == original["water_surface"]
    assert result["water_surface"]["basis"] == comparison.expected_water_surface_basis(case.source.review, "ac18")
    assert result["water_surface"]["action"] is result["water_surface"]["measurements"] is None
    assert result["display_grid"] == original["display_grid"]
    assert result["water_surface"]["photo_to_display_h"] == result["native_to_photo_h"]
    np.testing.assert_array_equal(result["valid_support"], original["valid_support"])
    assert not np.array_equal(result["image_rgb"], original["image_rgb"])
    expected = np.zeros((12, 16, 3), np.uint8)
    expected[2:, 1:] = 90
    np.testing.assert_array_equal(result["image_rgb"], expected)
    # The visible JPEG has a different gradient: its proposal must not replace
    # the original RAW proposal or create a new scientific input path.
    basis = result["water_surface"]["basis"]
    assert water.propose_water_surface(result["image_rgb"], basis=basis) != result["water_surface"]["proposal"]
    action = water.make_water_surface_action(basis, [[4., 4.], [12., 4.]],
        actor="synthetic human", confirmed_utc="2026-10-22T00:00:00Z")
    measured = water.water_surface_measurements(action, expected_basis=basis)
    assert measured["height_basis"] == "existing-own-geometry" and measured["height_mm"] == pytest.approx(8.)
    assert measured["bolt_height_mm"] is None
    assert "observed-0-195-390-mm-left-bolts" in measured["unavailable"]
    np.testing.assert_array_equal(case.raw_rgb, before[0])
    assert Path(case.raw["path"]).read_bytes() == before[1] and case.jpeg.read_bytes() == before[2]
    assert not any("forbidden-old-" in path for path in case.source.reads)


@pytest.mark.parametrize("damage", ["exposure", "grid", "photo-transform", "support", "support-dtype",
                                    "rgb-dtype", "basis", "action"])
def test_jpg_water_bridge_refuses_mismatched_original_exposure_plane_or_basis(bridge_case, monkeypatch, damage):
    case = bridge_case

    def damaged_original(*args, **options):
        packet = case.original_reader(*args, **options)
        if damage == "exposure":
            packet["source_raw"]["sha256"] = "f" * 64
        elif damage == "grid":
            packet["water_surface"]["display_grid"]["model"]["sha256"] = "f" * 64
        elif damage == "photo-transform":
            packet["water_surface"]["photo_to_display_h"][0][2] = 1.
        elif damage == "support":
            packet["valid_support"][2, 1] = False
        elif damage == "support-dtype":
            packet["valid_support"] = packet["valid_support"].astype(np.uint8)
        elif damage == "rgb-dtype":
            packet["image_rgb"] = packet["image_rgb"].astype(np.float32)
        elif damage == "basis":
            packet["water_surface"]["basis"]["own_geometry_origin_metric_mm"][0] = 1.
        else:
            packet["water_surface"]["action"] = {"unexpected": "retained human action"}
        return packet

    # Inject a fault only AFTER the original reader has authenticated and
    # produced its packet. No scientific/source predicate is weakened.
    monkeypatch.setattr(comparison, "read_comparison_preview", damaged_original)
    with pytest.raises(ValueError, match="same selected exposure|same native display grid|original current water-line basis"):
        camera.read_water_surface_comparison_preview(case.source.review, "ac18", None, "clearwater")
    assert len(case.calls) == 1
