"""Layer painting tab: camera RAW photographs next to JPEG/TIFF/PNG.

A RAW file (Sony ARW, Nikon NEF, ...) is shown with LibRaw's display
rendering on its own grid: the whole visible sensor area, turned upright. The
camera's JPEG is cropped and may be lens-corrected, so a RAW background is
refused for a JPEG reference and the other way round. LibRaw is replaced by a
fake here; set DARSIA_TEST_DISPLAY_ARW and DARSIA_TEST_DISPLAY_NEF to real files to run the
last test on them.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMenu, QMessageBox

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import photograph_io
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower.series_geometry import SeriesRunInputs, photo_size
from darsia.fluidflower.series_layer_view import LayerPainterView

WIDTH, HEIGHT = 120, 90


class _LibRawError(Exception):
    pass


def _rendering(seed, width=WIDTH, height=HEIGHT):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


class _FakeRaw:
    def __init__(self, library, path):
        self.library = library
        self.path = path
        rgb = library.files[path]
        flip = library.flips.get(path, 0)
        height, width = rgb.shape[:2]
        if flip in (5, 6):
            width, height = height, width
        self.sizes = types.SimpleNamespace(width=width, height=height, flip=flip)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def postprocess(self, **options):
        self.library.calls.append(options)
        rgb = self.library.files[self.path]
        return rgb[::2, ::2].copy() if options.get("half_size") else rgb.copy()


class _FakeLibRaw:
    """Stands in for rawpy: renderings are registered per path."""

    LibRawError = _LibRawError

    def __init__(self):
        self.files = {}
        self.flips = {}
        self.calls = []

    def imread(self, path):
        if path not in self.files:
            raise _LibRawError(f"cannot open {path}")
        return _FakeRaw(self, path)


@pytest.fixture
def libraw(monkeypatch):
    fake = _FakeLibRaw()
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    return fake


def _raw_file(libraw, path, rgb, flip=0):
    path.write_bytes(b"not a real raw file")
    libraw.files[str(path)] = rgb
    libraw.flips[str(path)] = flip
    return path


# ------------------------------------------------------------ the reader


@pytest.mark.parametrize("flip, expected", [(0, (WIDTH, HEIGHT)), (3, (WIDTH, HEIGHT)), (5, (HEIGHT, WIDTH)), (6, (HEIGHT, WIDTH))])
def test_raw_size_is_the_visible_area_turned_upright(tmp_path, libraw, flip, expected):
    # The fake rendering is already upright, as LibRaw's output is.
    rgb = _rendering(1, *expected)
    path = _raw_file(libraw, tmp_path / "photo.ARW", rgb, flip=flip)
    assert photograph_io.is_camera_raw(path)
    assert photograph_io.raw_photo_size(path) == expected
    # The shared size reader of DarSIA gives the same for RAW files.
    assert photo_size(path) == expected


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_raw_photographs_get_a_display_rendering(tmp_path, libraw, factor):
    rgb = _rendering(2)
    path = _raw_file(libraw, tmp_path / "photo.NEF", rgb)
    shown = photograph_io.load_reduced_rgb8(path, factor)
    options = libraw.calls[-1]
    assert options["use_camera_wb"] is True and options["output_bps"] == 8
    assert options["half_size"] is (factor >= 2)
    assert shown.dtype == np.uint8
    assert shown.shape == (HEIGHT // factor, WIDTH // factor, 3)
    if factor == 1:
        assert np.array_equal(shown, rgb)
    if factor == 2:
        assert np.array_equal(shown, rgb[::2, ::2])
    floats = photograph_io.load_reduced(path, factor)
    assert np.array_equal(floats, shown.astype(np.float32) / 255.0)


def test_an_unreadable_raw_file_is_a_value_error(tmp_path, libraw):
    path = tmp_path / "broken.ARW"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="broken.ARW"):
        photo_size(path)
    with pytest.raises(ValueError, match="broken.ARW"):
        photograph_io.load_reduced_rgb8(path, 1)


def test_raw_files_need_rawpy(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rawpy", None)
    path = tmp_path / "photo.ARW"
    path.write_bytes(b"raw")
    with pytest.raises(ValueError, match="rawpy"):
        photo_size(path)


def test_jpeg_and_png_keep_the_header_size_without_libraw(tmp_path, libraw):
    path = tmp_path / "photo.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(path)
    assert not photograph_io.is_camera_raw(path)
    assert photo_size(path) == (64, 48)
    assert libraw.calls == []


# ------------------------------------------------------- the painting tab


class _Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Keep remembered choices in a private INI file, not the user's registry.

    Message boxes are recorded instead of shown, so that a refusal can never
    block a test on a modal dialog.
    """

    path = str(tmp_path / "painting.ini")
    monkeypatch.setattr(
        slv, "QSettings", lambda *args: QSettings(path, QSettings.Format.IniFormat)
    )
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(
            QMessageBox, name, lambda *args, **kwargs: shown.append(args[2:3]) or None
        )
    return shown


@pytest.fixture
def photos(tmp_path, libraw):
    folder = tmp_path / "series"
    folder.mkdir()
    reference = _raw_file(libraw, folder / "reference.ARW", _rendering(3))
    background = _raw_file(libraw, folder / "before_indicator.ARW", _rendering(4))
    jpeg = folder / "before_indicator.jpg"
    Image.fromarray(_rendering(5)).save(jpeg, quality=95)
    config = folder / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="reference.ARW"\n', encoding="utf-8")
    return {"reference": reference, "background": background, "jpeg": jpeg, "config": config}


def _make_view(qtbot, monkeypatch, tmp_path, baseline, config):
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    # A 60 x 45 display copy, so that the full resolution has its own layer.
    monkeypatch.setattr(lp, "DISPLAY_LIMIT_PX", 60)
    original = lp.load_painting_image
    monkeypatch.setattr(
        lp,
        "load_painting_image",
        lambda path, **options: original(path, **{"display_limit": 60, **options}),
    )
    reference = SeriesRunInputs("ac52", config, baseline_image=baseline, width_m=0.9, height_m=0.49)
    widget = LayerPainterView(_Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    qtbot.waitExposed(widget)
    widget.enhance.setChecked(False)
    return widget


def _detail_pixel(view, row, col):
    colour = view.canvas.detail_item.pixmap().toImage().pixelColor(col, row)
    return (colour.red(), colour.green(), colour.blue())


def test_a_raw_reference_photo_opens_on_its_own_grid(qtbot, monkeypatch, tmp_path, settings, photos):
    view = _make_view(qtbot, monkeypatch, tmp_path, photos["reference"], photos["config"])
    assert view.image is not None
    assert view.image.image_size == (WIDTH, HEIGHT)
    assert view.image.display.shape[:2] == (HEIGHT // 2, WIDTH // 2)
    assert view.session.image_size == (WIDTH, HEIGHT)
    assert view.canvas.paint_enabled and view.background_available()
    pixmap = view.canvas.detail_item.pixmap()
    assert (pixmap.width(), pixmap.height()) == (WIDTH, HEIGHT)
    expected = _rendering(3)
    assert _detail_pixel(view, 17, 33) == tuple(int(v) for v in expected[17, 33])


def test_a_raw_background_goes_with_a_raw_reference(qtbot, monkeypatch, tmp_path, settings, photos):
    view = _make_view(qtbot, monkeypatch, tmp_path, photos["reference"], photos["config"])
    assert view.set_background_photo(photos["background"])
    assert view.background_shown
    expected = _rendering(4)
    assert _detail_pixel(view, 40, 70) == tuple(int(v) for v in expected[40, 70])


def test_raw_and_jpeg_photos_are_not_mixed(qtbot, monkeypatch, tmp_path, settings, photos):
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: messages.append(args[2]))
    view = _make_view(qtbot, monkeypatch, tmp_path, photos["reference"], photos["config"])
    assert not view.set_background_photo(photos["jpeg"])
    assert messages and "not the same kind of file" in messages[-1]
    assert view.background_path is None and not view.background_shown


def test_a_jpeg_reference_refuses_a_raw_background(qtbot, monkeypatch, tmp_path, settings, photos):
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: messages.append(args[2]))
    jpeg_reference = photos["jpeg"].with_name("reference.jpg")
    Image.fromarray(_rendering(6)).save(jpeg_reference, quality=95)
    view = _make_view(qtbot, monkeypatch, tmp_path, jpeg_reference, photos["config"])
    assert not view.set_background_photo(photos["background"])
    assert messages and "lens-corrected" in messages[-1]
    assert view.set_background_photo(photos["jpeg"])


def test_the_background_dialog_offers_the_reference_photos_kind_first(
    qtbot, monkeypatch, tmp_path, settings, photos
):
    seen = []

    def fake_dialog(parent, title, start, filters):
        seen.append(filters.split(";;"))
        return "", ""

    monkeypatch.setattr(slv.QFileDialog, "getOpenFileName", fake_dialog)
    view = _make_view(qtbot, monkeypatch, tmp_path, photos["reference"], photos["config"])
    assert not view.choose_background_photo()
    jpeg_reference = photos["jpeg"].with_name("reference.jpg")
    Image.fromarray(_rendering(6)).save(jpeg_reference, quality=95)
    view = _make_view(qtbot, monkeypatch, tmp_path, jpeg_reference, photos["config"])
    assert not view.choose_background_photo()
    raw_first, jpeg_first = seen
    assert raw_first[0].startswith("Camera RAW (") and "*.arw" in raw_first[0] and "*.nef" in raw_first[0]
    assert jpeg_first[0].startswith("Photographs (") and "*.jpg" in jpeg_first[0]
    assert raw_first[-1] == jpeg_first[-1] == "All files (*)"


def test_the_rig_frame_panel_dialog_follows_the_reference_photo_too(
    qtbot, monkeypatch, tmp_path, settings, photos
):
    seen = []
    monkeypatch.setattr(
        slv.QFileDialog,
        "getOpenFileName",
        lambda parent, title, start, filters: seen.append(filters.split(";;")) or ("", ""),
    )
    from darsia.fluidflower import series_rig_frame_panel as panel_module

    monkeypatch.setattr(
        panel_module.QFileDialog,
        "getOpenFileName",
        lambda parent, title, start, filters: seen.append(filters.split(";;")) or ("", ""),
    )
    view = _make_view(qtbot, monkeypatch, tmp_path, photos["reference"], photos["config"])
    view.rig_panel._choose_photo()
    assert seen and seen[-1][0].startswith("Camera RAW (")


@pytest.mark.parametrize(
    "reference, first",
    [("photo.ARW", "Camera RAW ("), ("photo.nef", "Camera RAW ("), ("photo.JPG", "Photographs ("), (None, "Photographs (")],
)
def test_the_photo_dialog_filter_puts_the_reference_kind_first(reference, first):
    from darsia.fluidflower.image_formats import photo_dialog_filter

    filters = photo_dialog_filter(reference).split(";;")
    assert filters[0].startswith(first)
    assert len(filters) == 3 and filters[-1] == "All files (*)"
    assert {"*.arw", "*.nef"} <= set(" ".join(filters).replace("(", " ").replace(")", " ").split())


# ------------------------------------------------------------- real files


@pytest.mark.parametrize("variable", ["DARSIA_TEST_DISPLAY_ARW", "DARSIA_TEST_DISPLAY_NEF"])
def test_real_raw_files_are_rendered_on_their_visible_area(variable):
    value = os.environ.get(variable)
    if not value or not Path(value).is_file():
        pytest.skip(f"{variable} does not name a RAW file here")
    rawpy = pytest.importorskip("rawpy")
    path = Path(value)
    with rawpy.imread(str(path)) as raw:
        expected = (raw.sizes.width, raw.sizes.height)
        if raw.sizes.flip in (5, 6):
            expected = expected[::-1]
    assert photo_size(path) == photograph_io.raw_photo_size(path) == expected
    shown = photograph_io.load_reduced_rgb8(path, 4)
    assert shown.dtype == np.uint8
    assert shown.shape == (expected[1] // 4, expected[0] // 4, 3)
    assert 10.0 < float(shown.mean()) < 245.0
