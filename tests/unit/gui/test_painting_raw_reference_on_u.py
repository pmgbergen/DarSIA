"""A camera RAW of a run with a camera profile is sized and shown on the camera grid U.

N103/N106: the painting reference, its background, the rig-frame photo and the
zoomed detail use one lattice, U, the grid of the camera's JPEG. So a session
drawn on the JPEG opens unchanged on the ARW, with no JPEG needed. Without a
profile everything stays as before. Synthetic 80 x 60 photographs and a stub
renderer; no real series or RAW file is read.
"""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QMenu

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import photograph_io
from darsia.fluidflower import series_native_context as native
from darsia.fluidflower.native import raw_photograph
from darsia.fluidflower.series_geometry import SeriesRunInputs, photo_size
from darsia.fluidflower.series_layer_view import LayerPainterView
from .test_painting_background_photo import (  # noqa: F401  (fixtures)
    BACKGROUND_RGB, _Project, _shown_path, _shown_pixel, settings,
)

RAW_RGB = (90, 180, 120)


@pytest.fixture
def profile(tmp_path):
    return native.CameraUProfile(tmp_path / "native.json", "a" * 64, "early", (80, 60))


@pytest.fixture
def renders(monkeypatch):
    """The shared renderer, stubbed; records each full render it is asked for."""

    calls = []

    def render(path, *, profile_path, profile_sha256, profile_id, **kwargs):
        calls.append((path.name, profile_id))
        return np.full((60, 80, 3), RAW_RGB, dtype=np.uint8)

    monkeypatch.setattr(raw_photograph, "load_camera_u_rgb8", render)
    photograph_io._CAMERA_U_CACHE.clear()
    return calls


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "series" / "DSC00441.ARW"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"synthetic camera RAW placeholder")
    return path


def test_photos_of_a_profiled_run_are_read_on_u_once(raw_file, profile, renders, monkeypatch):
    monkeypatch.setattr(photograph_io, "_render_raw_rgb8",
                        lambda path, factor: pytest.fail("the RAW's own grid is not used on U"))
    full = photograph_io.load_reduced_rgb8(raw_file, 1, camera_u=profile)
    half = photograph_io.load_reduced(raw_file, 2, camera_u=profile)
    assert full.shape == (60, 80, 3) and tuple(full[10, 10]) == RAW_RGB
    assert half.shape == (30, 40, 3) and np.allclose(half[5, 5], np.array(RAW_RGB) / 255.0)
    assert renders == [("DSC00441.ARW", "early")]  # One render serves both.


def test_without_a_profile_a_raw_keeps_its_own_grid(raw_file, renders, monkeypatch):
    monkeypatch.setattr(photograph_io, "_render_raw_rgb8",
                        lambda path, factor: np.zeros((7, 9, 3), dtype=np.uint8))
    assert photograph_io.load_reduced_rgb8(raw_file, 1).shape == (7, 9, 3)
    assert renders == []


def test_photo_size_is_the_u_size_for_a_raw_only(tmp_path, raw_file, profile):
    jpeg = tmp_path / "DSC00441.JPG"
    Image.new("RGB", (64, 48)).save(jpeg)
    assert photo_size(raw_file, camera_u=profile) == (80, 60)
    assert photo_size(jpeg, camera_u=profile) == (64, 48)  # A JPEG keeps its own size.


def _series(tmp_path, profiles):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    document = {"registration": {"common": {"profiles": {"early": {}},
                                            "sensor": {"camera_wh": [80, 60]}}},
                "source": {"profiles": profiles}}
    raw = json.dumps(document).encode("utf-8")
    (root / "native.json").write_bytes(raw)
    (root / "series.toml").write_text(
        f'[native]\nconfiguration = "native.json"\nsha256 = "{hashlib.sha256(raw).hexdigest()}"\n',
        encoding="utf-8")
    return root


def test_the_camera_u_of_a_photo_needs_a_raw_and_a_profiled_run(tmp_path, raw_file):
    root = _series(tmp_path, {"AC52": "early"})
    project = _Project(root)
    jpeg = tmp_path / "DSC00441.JPG"
    Image.new("RGB", (80, 60)).save(jpeg)
    found = lp.photo_camera_u(project, "ac52", raw_file)
    assert (found.profile_id, found.camera_wh) == ("early", (80, 60))
    assert lp.photo_camera_u(project, "ac52", jpeg) is None
    assert lp.photo_camera_u(project, "ac17", raw_file) is None  # No profile for that run.
    assert lp.photo_camera_u(_Project(None), "ac52", raw_file) is None
    reference = SeriesRunInputs("ac52", root / "run.toml", baseline_image=raw_file,
                                width_m=0.9, height_m=0.49)
    assert lp.reference_camera_u(project, reference) == found
    assert lp.reference_camera_u(project, None) is None


def test_a_geometry_package_finds_the_camera_u_in_its_linked_series(tmp_path, raw_file):
    # FF_AC's layout: the painting project is the geometry package inside the
    # series, and only its manifest link names the series root.
    root = _series(tmp_path, {"AC17": "early"})
    package = root / "geometry_facies"
    package.mkdir()
    linked = _Project(package)
    linked.manifest["tools"] = {"native_series": {"series_root": str(root)}}
    found = lp.run_camera_u(linked, "ac17")
    assert (found.profile_id, found.camera_wh) == ("early", (80, 60))
    assert lp.photo_camera_u(linked, "ac17", raw_file) == found
    assert lp.run_camera_u(_Project(root), "ac17") == found  # A series root itself.
    assert lp.run_camera_u(_Project(package), "ac17") is None  # No link, no series.toml.
    assert lp.run_camera_u(_Project(None), "ac17") is None


def test_a_raw_painting_image_on_u_has_the_u_size(raw_file, profile, renders):
    image = lp.load_painting_image(raw_file, camera_u=profile)
    assert image.image_size == (80, 60) and image.camera_u is profile
    assert image.display.shape[:2] == (60, 80)
    assert np.allclose(image.display[30, 40], np.array(RAW_RGB) / 255.0)


@pytest.fixture
def raw_view(qtbot, tmp_path, monkeypatch, settings, raw_file, profile, renders):
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(native, "camera_u_profile", lambda root, run: profile)
    config = raw_file.parent / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="DSC00441.ARW"\n', encoding="utf-8")
    reference = SeriesRunInputs("ac52", config, baseline_image=raw_file, width_m=0.9, height_m=0.49)
    widget = LayerPainterView(_Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    qtbot.waitExposed(widget)
    widget.enhance.setChecked(False)
    return widget


def test_a_raw_reference_opens_on_u_and_takes_a_jpeg_background(raw_view, raw_file, tmp_path):
    assert raw_view.image.image_size == (80, 60) and raw_view.image.camera_u is not None
    assert _shown_path(raw_view) == str(raw_file) and _shown_pixel(raw_view, 40, 60) == RAW_RGB
    assert tuple(raw_view.session.image_size) == (80, 60)
    jpeg = raw_file.with_suffix(".JPG")
    Image.new("RGB", (80, 60), BACKGROUND_RGB).save(jpeg)
    assert raw_view.set_background_photo(jpeg)  # The camera's JPEG is on the same grid U.
    assert np.allclose(_shown_pixel(raw_view, 40, 60), BACKGROUND_RGB, atol=3)  # JPEG is lossy.


def test_a_frame_photo_of_a_profiled_run_is_shown_on_u(qtbot, tmp_path, monkeypatch, settings,
                                                       raw_file, profile, renders):
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    folder = tmp_path / "jpeg_series"
    folder.mkdir()
    Image.new("RGB", (80, 60), BACKGROUND_RGB).save(folder / "raw.png")
    config = folder / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    reference = SeriesRunInputs("ac52", config, baseline_image=folder / "raw.png",
                                width_m=0.9, height_m=0.49)
    monkeypatch.setattr(native, "camera_u_profile", lambda root, run: profile)
    view = LayerPainterView(_Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(view)
    view.resize(1400, 900)
    view.show()
    qtbot.waitExposed(view)
    view.enhance.setChecked(False)
    assert view.show_frame_photo(raw_file)
    assert _shown_path(view) == str(raw_file) and _shown_pixel(view, 40, 60) == RAW_RGB
