"""B3 saved-asset contracts; source-phase tests prepared but not executed.

Fixtures contain only small synthetic saved products. The readers must never
decode RAW inputs, refit, or replace a scientific result while displaying them.
"""

from copy import deepcopy
import json

import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage

from darsia.fluidflower.calibration_review_panels import concentration_figure, read_panel
from darsia.fluidflower.tools.phase_studio_saved_results import model_browser_metadata


def _bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _write_result(path, result):
    path.write_text(json.dumps(result), encoding="utf-8")


@pytest.fixture
def saved(tmp_path):
    candidate = dict(dataset="ffac", run="ac28", candidate_id="phase-a",
                     params_sha256="parameters-a", optical_model_sha256="optical-a")
    source = dict(native_manifest_sha256="manifest-a", photo_sha256="photo-a",
                  phase_sha256="phase-map-a", raw_source_sha256="raw-a")
    rgb = np.array([[[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]],
                    [[0.2, 0.4, 0.6], [0.8, 1.0, 0.0]]], dtype=np.float32)
    concentration = np.array([[0.0, 0.8], [1.2, 1.8]], dtype=np.float32)
    support = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    unknown = np.array([[False, False], [True, False]])
    np.savez(tmp_path / "fields.npz", photograph_rgb=rgb,
             concentration_g_l=concentration, aqueous_volume_m3=support,
             unquantified_phase_mask=unknown)
    row = dict(frame_index=7, source_name="image-007.dng", elapsed_h=1.25,
               source_reference=source, fields="fields.npz", color_limit_g_l=2.0,
               total_mass_g=4.0, aqueous_mass_g=3.0, gas_mass_g=1.0)
    result = dict(complete=True, selected_inputs={"candidate": candidate},
                  method={"settings": {"photometry_variant_id": "none_cc_off"}},
                  calibration={"parameters": {"gain": 23.859}}, frames=[row])
    path = tmp_path / "result.json"
    _write_result(path, result)
    return path, result, rgb, concentration


def test_photograph_uses_saved_rgb_and_preserves_all_scientific_bytes(saved):
    path, result, rgb, concentration = saved
    before, original = _bytes(path.parent), deepcopy(result)
    image = read_panel(result, path, 0, "photograph")
    assert (image.width(), image.height()) == (2, 2)
    assert image.pixelColor(0, 0).getRgb() == (0, 128, 255, 255)
    assert image.pixelColor(1, 0).getRgb() == (255, 0, 64, 255)
    assert result == original and _bytes(path.parent) == before
    with np.load(path.parent / "fields.npz", allow_pickle=False) as archive:
        assert archive["concentration_g_l"].dtype == np.float32
        np.testing.assert_array_equal(archive["concentration_g_l"], concentration)
        np.testing.assert_array_equal(archive["photograph_rgb"], rgb)


def test_photograph_supports_separate_immutable_rgb_archive(saved):
    path, result, rgb, _ = saved
    np.savez(path.parent / "no-photo.npz", concentration_g_l=np.zeros((2, 2), np.float32))
    np.savez(path.parent / "photo.npz", rgb=rgb)
    result["frames"][0].update(fields="no-photo.npz", photograph_arrays="photo.npz")
    _write_result(path, result)
    before = _bytes(path.parent)
    image = read_panel(result, path, 0, "photograph")
    assert image.pixelColor(0, 0).getRgb() == (0, 128, 255, 255)
    assert _bytes(path.parent) == before


def _alternate(saved):
    path, result, _, _ = saved
    alternate = deepcopy(result)
    other = deepcopy(alternate["frames"][0])
    other.update(source_name="image-003.dng", frame_index=3, elapsed_h=0.25)
    alternate["frames"] = [other, alternate["frames"][0]]
    np.savez(path.parent / "corrected.npz", photograph_rgb=np.ones((2, 2, 3), np.float32))
    alternate["frames"][1]["fields"] = "corrected.npz"
    alternate_path = path.parent / "corrected-result.json"
    _write_result(alternate_path, alternate)
    return alternate_path, alternate


def test_alternate_photograph_matches_source_and_time_in_reordered_frames(saved):
    path, result, _, _ = saved
    alternate_path, _ = _alternate(saved)
    before, original = _bytes(path.parent), deepcopy(result)
    image = read_panel(result, path, 0, "photograph", alternate_path)
    assert image.pixelColor(0, 0).getRgb() == (255, 255, 255, 255)
    assert result == original and _bytes(path.parent) == before


@pytest.mark.parametrize("section,key", [
    ("candidate", "dataset"), ("candidate", "run"),
    ("candidate", "candidate_id"), ("candidate", "params_sha256"),
    ("candidate", "optical_model_sha256"),
    ("source", "native_manifest_sha256"), ("source", "photo_sha256"),
    ("source", "phase_sha256"), ("source", "raw_source_sha256"),
    ("frame", "source_name"), ("frame", "elapsed_h"),
])
def test_alternate_photograph_rejects_source_identity_mismatch(saved, section, key):
    path, result, _, _ = saved
    alternate_path, alternate = _alternate(saved)
    target = (alternate["selected_inputs"]["candidate"] if section == "candidate"
              else alternate["frames"][1]["source_reference"] if section == "source"
              else alternate["frames"][1])
    target[key] = 9.0 if key == "elapsed_h" else "different"
    _write_result(alternate_path, alternate)
    before = _bytes(path.parent)
    with pytest.raises(ValueError, match="Photometry variants"):
        read_panel(result, path, 0, "photograph", alternate_path)
    assert _bytes(path.parent) == before


def test_alternate_photograph_rejects_incomplete_saved_result(saved):
    path, result, _, _ = saved
    alternate_path, alternate = _alternate(saved)
    alternate["complete"] = False
    _write_result(alternate_path, alternate)
    with pytest.raises(ValueError, match="incomplete"):
        read_panel(result, path, 0, "photograph", alternate_path)


def test_concentration_keeps_zero_support_mask_fixed_scale_and_float32(saved):
    path, result, _, concentration = saved
    before = _bytes(path.parent)
    figure = concentration_figure(path, result["frames"][0])
    try:
        axes = figure.axes[0]
        assert len(axes.images) == 1  # A real standalone map, without a photograph.
        image = axes.images[0]
        np.testing.assert_array_equal(image.get_array().data, concentration)
        np.testing.assert_array_equal(image.get_array().mask, [[False, True], [True, False]])
        assert image.get_array().dtype == np.float32
        assert image.get_clim() == (0, 2.0)
        assert figure.axes[1].get_ylabel() == "g/L · fixed linear scale"
    finally:
        figure.clear()
    assert _bytes(path.parent) == before


def test_concentration_panel_renders_the_saved_field_without_photo_override(saved):
    path, result, _, _ = saved
    before, original = _bytes(path.parent), deepcopy(result)
    image = read_panel(result, path, 0, "concentration", path.parent / "absent-variant.json")
    assert not image.isNull() and image.width() > 2 and image.height() > 2
    assert result == original and _bytes(path.parent) == before


@pytest.mark.parametrize("limit", [None, False, 0, -1, float("nan"), float("inf"), "2.0"])
def test_concentration_does_not_invent_a_missing_or_invalid_scale(saved, limit):
    path, result, _, _ = saved
    row = dict(result["frames"][0], color_limit_g_l=limit)
    with pytest.raises(ValueError, match="fixed positive"):
        concentration_figure(path, row)


@pytest.mark.parametrize("case,expected", [
    ("missing-support", "no aqueous support"),
    ("support-shape", "support grids differ"),
    ("unknown-shape", "unquantified mask"),
])
def test_concentration_rejects_missing_or_incompatible_masks(saved, case, expected):
    path, result, _, concentration = saved
    fields = {"concentration_g_l": concentration}
    if case != "missing-support":
        fields["av"] = np.ones((1, 2) if case == "support-shape" else (2, 2), np.float32)
    if case == "unknown-shape":
        fields["unquantified_phase_mask"] = np.zeros((1, 2), bool)
    np.savez(path.parent / "invalid.npz", **fields)
    row = dict(result["frames"][0], fields="invalid.npz")
    before = _bytes(path.parent)
    with pytest.raises(ValueError, match=expected):
        concentration_figure(path, row)
    assert _bytes(path.parent) == before


@pytest.mark.parametrize("content", ["mass_curve", "dilution_curve", "overflow_curve"])
def test_graph_reads_saved_pixels_without_using_photograph_variant(saved, content):
    path, result, _, _ = saved
    graph_path = path.parent / "graph.png"
    graph = QImage(5, 3, QImage.Format.Format_RGB888)
    graph.fill(QColor(21, 93, 174))
    assert graph.save(str(graph_path))
    result[content] = str(graph_path) if content == "overflow_curve" else "graph.png"
    before = _bytes(path.parent)
    image = read_panel(result, path, 0, content, path.parent / "absent-variant.json")
    assert image.size() == graph.size()
    assert all(image.pixelColor(x, y) == graph.pixelColor(x, y)
               for y in range(graph.height()) for x in range(graph.width()))
    assert _bytes(path.parent) == before


def test_missing_graph_is_explicitly_unavailable(saved):
    path, result, _, _ = saved
    with pytest.raises(ValueError, match="requested asset"):
        read_panel(result, path, 0, "mass_curve")
    result["mass_curve"] = "absent.png"
    with pytest.raises(ValueError, match="unavailable or unreadable"):
        read_panel(result, path, 0, "mass_curve")
    with pytest.raises(ValueError, match="Unknown"):
        read_panel(result, path, 0, "phase_map")


def test_named_metadata_groups_only_explicit_common_reference(tmp_path):
    result = {"series_calibration": "../shared/frozen.json",
              "calibration": {"display_name": "AC28 · evening", "chemistry_model": "simple_flash"}}
    before = deepcopy(result)
    ac28 = model_browser_metadata(result, tmp_path / "ac28" / "result.json")
    ac47 = model_browser_metadata(result, tmp_path / "ac47" / "result.json")
    assert ac28["review_calibration_id"] == ac47["review_calibration_id"]
    assert ac28["review_calibration_id"] == str((tmp_path / "shared" / "frozen.json").resolve())
    assert ac28["review_calibration_name"] == "AC28 · evening"
    assert ac28["review_chemistry"] == "Simple flash"
    assert result == before and not list(tmp_path.iterdir())


def test_equal_names_and_parameters_without_reference_do_not_merge_results(tmp_path):
    result = {"calibration": {"name": "Same name", "parameters": {"gain": 23.859}},
              "method": {"settings": {"chemistry": "darsia_titration_flash"}}}
    first = model_browser_metadata(result, tmp_path / "first.json")
    second = model_browser_metadata(result, tmp_path / "second.json")
    assert first["review_calibration_name"] == second["review_calibration_name"] == "Same name"
    assert first["review_calibration_id"] != second["review_calibration_id"]
    assert first["review_chemistry"] == "Titration"


def test_metadata_retains_recorded_checksum_conflicts_without_reading_reference(tmp_path):
    result = {"series_calibration": "absent.json", "bindings": {"calibration_sha256": "a" * 64},
              "calibration": {"sha256": "b" * 64}}
    metadata = model_browser_metadata(result, tmp_path / "result.json")
    assert metadata["review_calibration_binding_conflict"] is True
    assert metadata["review_calibration_sha256"] is None
    result["calibration"]["sha256"] = "a" * 64
    metadata = model_browser_metadata(result, tmp_path / "result.json")
    assert metadata["review_calibration_binding_conflict"] is False
    assert metadata["review_calibration_sha256"] == "a" * 64
    assert not list(tmp_path.iterdir())
