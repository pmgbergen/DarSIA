"""Studio desktop photographs and photo support of RAW review rows (package D2c).

A RAW row shows its configured display mode and its canvas photo support,
as a V5 row does. The desktop methods are called on a minimal window state.
"""
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import raw_photographic_review as review
from ..test_raw_photographic_review import PHASE_HW, _write, inputs  # noqa: F401


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    # The desktop resolves its configured series during import; keep it local.
    series = tmp_path / "configured_series"
    series.mkdir()
    (series / "series.toml").write_text('schema_version = 1\nseries_id = "ffac"\n[paths]\n', encoding="utf8")
    monkeypatch.setenv("DARSIA_FFAC_PROJECT_ROOT", str(series))
    from darsia.fluidflower.tools import calibration_review_desktop
    return calibration_review_desktop


@pytest.fixture
def window(tmp_path, inputs):
    _write(tmp_path, inputs)
    manifest = review.read_raw_review_manifest(tmp_path / "review", "ac52")
    return SimpleNamespace(current_review_root=tmp_path / "review", current_run="ac52", display_mode="lift1",
                           current_pair=dict(frame_sources=[dict(row) for row in manifest["frames"]]))


def test_a_raw_row_shows_its_display_mode(desktop, window):
    asset = desktop.ReviewImageAssets._frame_asset(window, 1)
    expected = window.current_review_root / "native_frames/ac52/frames/ac52_f2/lift1.png"
    assert asset.resolve().resolve() == expected.resolve()


def test_a_raw_row_loads_its_canvas_photo_support(desktop, window):
    support = desktop.ReviewImageAssets._photo_sand_loader(window, 0)()
    stored = np.asarray(Image.open(window.current_review_root / "native_frames/ac52/frames/ac52_f1/photo_support.png"))
    assert support.shape == PHASE_HW and np.array_equal(support, stored > 0)
    assert desktop.ReviewImageAssets._photo_sand_loader(window, 5) is None
