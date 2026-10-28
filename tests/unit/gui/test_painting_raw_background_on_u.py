"""A camera RAW background under a JPEG reference is shown on the camera grid U.

N106/N108: the shared renderer places the RAW on exactly the grid of the
camera's JPEG, so marks drawn on the JPEG fit it unchanged. Without a
configured camera profile for the run, the RAW is refused as before. Synthetic
80 x 60 photographs and a stub renderer; no real series or RAW file is read.
"""
import hashlib
import json

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import series_native_context as native
from darsia.fluidflower.native import raw_photograph
from .test_painting_background_photo import (  # noqa: F401  (fixtures)
    BACKGROUND_RGB, REFERENCE_RGB, _shown_path, _shown_pixel, paths, settings, view,
)

RAW_RGB = (90, 180, 120)


def series(tmp_path, *, profiles=None, camera_wh=(80, 60)):
    root = tmp_path / "series_root"
    root.mkdir(exist_ok=True)
    document = {"registration": {"common": {"profiles": {"early": {}, "late": {}},
                                            "sensor": {"camera_wh": list(camera_wh)}}}}
    if profiles is not None:
        document["source"] = {"profiles": profiles}
    raw = json.dumps(document).encode("utf-8")
    (root / "native.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (root / "series.toml").write_text(
        f'[native]\nconfiguration = "native.json"\nsha256 = "{digest}"\n', encoding="utf-8")
    return root, digest


def test_camera_u_profile_reads_the_runs_profile_from_the_bound_document(tmp_path):
    root, digest = series(tmp_path, profiles={"AC52": "late", "ac17": "early"})
    profile = native.camera_u_profile(root, "ac52")
    assert profile == native.CameraUProfile((root / "native.json").resolve(), digest, "late", (80, 60))
    assert native.camera_u_profile(root, "AC17").profile_id == "early"
    assert native.camera_u_profile(root, "ac99") is None  # This run has no profile.


def test_camera_u_profile_is_none_without_a_native_table_or_source_profiles(tmp_path):
    assert native.camera_u_profile(None, "ac52") is None
    assert native.camera_u_profile(tmp_path, "ac52") is None  # No series.toml.
    root, _digest = series(tmp_path, profiles=None)
    assert native.camera_u_profile(root, "ac52") is None


def test_camera_u_profile_refuses_an_undeclared_profile(tmp_path):
    root, _digest = series(tmp_path, profiles={"ac52": "middle"})
    with pytest.raises(ValueError, match="not declared"):
        native.camera_u_profile(root, "ac52")


@pytest.fixture
def raw_background(paths):
    path = paths["before"].with_name("DSC00441.ARW")
    path.write_bytes(b"synthetic camera RAW placeholder")
    return path


def test_a_raw_background_is_shown_on_u_under_a_jpeg_reference(view, raw_background, monkeypatch):
    profile = native.CameraUProfile(raw_background.parent / "native.json", "a" * 64, "early", (80, 60))
    calls = []
    monkeypatch.setattr(native, "camera_u_profile", lambda root, run: calls.append(run) or profile)

    def render(path, *, profile_path, profile_sha256, profile_id, **kwargs):
        assert (path, profile_path, profile_sha256, profile_id) == (
            raw_background, profile.path, profile.sha256, "early")
        return np.full((60, 80, 3), RAW_RGB, dtype=np.uint8)

    monkeypatch.setattr(raw_photograph, "load_camera_u_rgb8", render)
    assert view.set_background_photo(raw_background)
    assert view.background_shown and view.background_path == raw_background
    assert _shown_path(view) == str(raw_background)
    assert _shown_pixel(view, 40, 60) == RAW_RGB
    assert calls == ["ac52"]
    view.toggle_background()  # T switches back to the JPEG reference.
    assert _shown_pixel(view, 40, 60) == REFERENCE_RGB


def test_a_raw_background_without_a_camera_profile_is_refused_as_before(view, raw_background, monkeypatch):
    warnings = []
    monkeypatch.setattr(native, "camera_u_profile", lambda root, run: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(raw_photograph, "load_camera_u_rgb8",
                        lambda *args, **kwargs: pytest.fail("no profile, no rendering"))
    assert not view.set_background_photo(raw_background)
    assert not view.background_shown and "camera profile" in warnings[0]


def test_a_camera_profile_of_another_size_is_refused(view, raw_background, monkeypatch):
    profile = native.CameraUProfile(raw_background.parent / "native.json", "a" * 64, "early", (64, 48))
    monkeypatch.setattr(native, "camera_u_profile", lambda root, run: profile)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    monkeypatch.setattr(raw_photograph, "load_camera_u_rgb8",
                        lambda *args, **kwargs: pytest.fail("a different U size is never rendered"))
    assert not view.set_background_photo(raw_background)
