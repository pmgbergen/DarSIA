"""Model comparison over RAW photographic reviews (package D2e).

RAW sources give the comparison their frames, native grid and camera-RAW
identity through the shared Studio rows; historical sources are unchanged.
"""
import json

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from ..test_raw_photographic_review import NATIVE_HW, PHASE_HW, _write, inputs  # noqa: F401


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    series = tmp_path / "configured_series"
    series.mkdir()
    (series / "series.toml").write_text('schema_version = 1\nseries_id = "ffac"\n[paths]\n', encoding="utf8")
    monkeypatch.setenv("DARSIA_FFAC_PROJECT_ROOT", str(series))
    from darsia.fluidflower.tools import phase_model_comparison_view
    return phase_model_comparison_view


def test_raw_sources_pair_by_camera_raw_and_time(tmp_path, inputs, comparison):
    manifest = json.loads(_write(tmp_path, inputs).read_text(encoding="utf-8"))
    rows = comparison._frame_rows(manifest)
    assert [row["source_name"] for row in rows] == ["ac52_f1.ARW", "ac52_f2.ARW"]
    assert rows[1]["source_sha256"] == manifest["frames"][1]["acquisition"]["sha256"]
    assert comparison._content_shape(manifest) == list(NATIVE_HW)
    pair = comparison.match_frames(manifest, dict(manifest), left_digest="a" * 64, right_digest="b" * 64)
    assert [(row["left"], row["right"]) for row in pair["frames"]] == [(0, 0), (1, 1)]
    assert pair["frames"][1]["source_name"] == "ac52_f2.ARW"


def test_raw_photographs_are_cropped_to_their_native_box(app, tmp_path, inputs, comparison):
    manifest = json.loads(_write(tmp_path, inputs).read_text(encoding="utf-8"))
    canvas = QImage(PHASE_HW[1], PHASE_HW[0], QImage.Format.Format_RGB888)
    cropped = comparison._crop(canvas, manifest, frame_index=1)
    assert (cropped.width(), cropped.height()) == (NATIVE_HW[1], NATIVE_HW[0])


def test_raw_panes_offer_their_photographic_modes(tmp_path, inputs, comparison):
    manifest = json.loads(_write(tmp_path, inputs).read_text(encoding="utf-8"))
    job = dict(review_root=str(tmp_path / "review"), run="ac52")
    pair = dict(manifests=(manifest, manifest), jobs=(job, job))
    modes = comparison.available_panel_photo_modes(pair, 0)
    assert {"natural", "original", "lift1", "lift2", "raw_detail"} <= set(modes) and "change" not in modes
    assert comparison.available_brightness_modes(pair) == comparison.BASE_DISPLAY_MODES
    assert comparison._frame_rows({}) == [] and comparison._content_shape({}) is None
    for item in (comparison._frame_rows, comparison._content_shape):
        assert item.__doc__.strip()
