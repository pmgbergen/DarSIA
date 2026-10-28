"""Layer painting tab: the photograph shown with less colour saturation.

A slider from 0 % (grey) to 100 % (the photograph's own colours) changes the
display copy and the full-resolution detail under the marks, for the
reference and the background photo. It is display only and a setting of the
user, never stored in the painting session. Synthetic 80 x 60 photographs, a
40 x 30 display copy and a project in a temporary folder; no real series,
rig model or NAS input is used.
"""

import json

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QMenu

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_layer_view import LayerPainterView, photo_with_saturation


class _Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


def _colourful(width=80, height=60, seed=4):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """The user's settings in a private INI file, not the registry.

    QSettings(organization, application) always uses the native format, which
    is the registry on Windows, so the view's QSettings is replaced here.
    """

    path = str(tmp_path / "painting.ini")
    monkeypatch.setattr(
        slv, "QSettings", lambda *args: QSettings(path, QSettings.Format.IniFormat)
    )
    return path


@pytest.fixture
def make_view(qtbot, tmp_path, monkeypatch, settings):
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    # A 40 x 30 display copy of the 80 x 60 photographs, so that the
    # full-resolution detail exists.
    monkeypatch.setattr(lp, "DISPLAY_LIMIT_PX", 40)
    original = lp.load_painting_image
    monkeypatch.setattr(
        lp,
        "load_painting_image",
        lambda path, **options: original(path, **{"display_limit": 40, **options}),
    )
    folder = tmp_path / "series"
    folder.mkdir()
    baseline = folder / "raw.png"
    Image.fromarray(_colourful()).save(baseline)
    before = folder / "before_indicator.png"
    Image.fromarray(_colourful(seed=9)).save(before)
    config = folder / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    reference = SeriesRunInputs(
        "ac52", config, baseline_image=baseline, width_m=0.9, height_m=0.49
    )

    def make():
        widget = LayerPainterView(_Project(tmp_path / "project"), lambda: reference)
        qtbot.addWidget(widget)
        widget.resize(1400, 900)
        widget.show()
        qtbot.waitExposed(widget)
        return widget

    make.before = before
    return make


@pytest.fixture
def view(make_view):
    return make_view()


def _pixels(pixmap):
    """The RGB pixels of a pixmap as a (rows, cols, 3) uint8 array."""

    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    rows, cols = image.height(), image.width()
    data = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    # A copy, while the image that owns the bytes still exists.
    rgb = data.reshape(rows, image.bytesPerLine())[:, : 3 * cols].reshape(rows, cols, 3)
    return rgb.copy()


def _shown(view):
    return _pixels(view.canvas.pixmap_item.pixmap()), _pixels(view.canvas.detail_item.pixmap())


def _grey(rgb):
    return bool(np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]))


# ---------------------------------------------------------------------------
# The helper


def test_photo_with_saturation_blends_the_photograph_with_its_grey():
    rgb = _colourful(20, 10)
    assert photo_with_saturation(rgb, 1.0) is rgb
    assert photo_with_saturation(rgb, 1.5) is rgb
    grey = photo_with_saturation(rgb, 0.0)
    assert grey.dtype == np.uint8 and grey.shape == rgb.shape and _grey(grey)
    # OpenCV's RGB-to-grey weights (0.299, 0.587, 0.114), rounded.
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    assert np.max(np.abs(grey[..., 0].astype(float) - luma)) <= 1.0
    assert np.array_equal(photo_with_saturation(rgb, -0.5), grey)
    half = photo_with_saturation(rgb, 0.5)
    expected = 0.5 * rgb.astype(float) + 0.5 * grey.astype(float)
    assert np.max(np.abs(half.astype(float) - expected)) <= 1.0
    assert rgb.flags.writeable and np.array_equal(rgb, _colourful(20, 10))  # Unchanged.


# ---------------------------------------------------------------------------
# The slider


def test_the_slider_greys_the_photo_and_its_detail_and_100_shows_it_as_before(view):
    slider = view.saturation
    assert (slider.minimum(), slider.maximum(), slider.value()) == (0, 100, 100)
    assert view.saturation_value.text() == "100 %"
    assert "0 % (grey)" in slider.toolTip()
    # Below the photograph, in a row of its own under Enhance contrast.
    assert slider.parentWidget() is view.enhance.parentWidget()
    top = view.enhance.mapTo(view, view.enhance.rect().bottomLeft()).y()
    assert slider.mapTo(view, slider.rect().topLeft()).y() >= top
    assert view.canvas.mapTo(view, view.canvas.rect().bottomLeft()).y() < top
    for enhanced in (True, False):
        view.enhance.setChecked(enhanced)
        display, detail = _shown(view)
        expected = view._enhanced if enhanced else view._plain
        expected_detail = view._detail_enhanced if enhanced else view._detail_plain
        assert detail.shape == (60, 80, 3)
        # 100 %: byte-identical to the display without the slider.
        assert np.array_equal(display, expected)
        assert np.array_equal(detail, expected_detail)
    slider.setValue(0)
    assert view.saturation_value.text() == "0 %"
    view.apply_saturation()
    display, detail = _shown(view)
    assert _grey(display) and _grey(detail)
    assert np.array_equal(display, photo_with_saturation(view._plain, 0.0))
    assert np.array_equal(detail, photo_with_saturation(view._detail_plain, 0.0))
    slider.setValue(100)
    view.apply_saturation()
    display, detail = _shown(view)
    assert np.array_equal(display, view._plain)
    assert np.array_equal(detail, view._detail_plain)


def test_the_slider_applies_on_its_own_after_a_moment(view, qtbot):
    view.saturation.setValue(0)
    qtbot.waitUntil(lambda: _grey(_shown(view)[0]), timeout=3000)
    assert not view._saturation_timer.isActive()


def test_the_background_photo_is_shown_at_the_same_saturation(view, make_view):
    view.saturation.setValue(0)
    view.apply_saturation()
    assert view.set_background_photo(make_view.before)
    assert view.background_shown
    display, detail = _shown(view)
    assert _grey(display) and _grey(detail)
    view.saturation.setValue(100)
    view.apply_saturation()
    display, _detail = _shown(view)
    assert view.enhance.isChecked()  # The contrast stretch is on by default.
    assert np.array_equal(display, view._background_enhanced)


def test_saturation_is_display_only_and_remembered_for_the_user(make_view, settings):
    view = make_view()
    view.add_polygon(lp.Polygon(1, [(5.0, 5.0), (5.0, 30.0), (25.0, 30.0)]))
    view.save_session()
    session_file = lp.session_path(view.project, "ac52")
    saved = session_file.read_bytes()
    before = view.session.to_dict()
    steps = len(view._undo_stack)
    view.saturation.setValue(30)
    view.apply_saturation()
    # Nothing of the painting changes, and nothing is written but the setting.
    assert view.session.to_dict() == before
    assert not view._dirty and len(view._undo_stack) == steps
    view.save_session()
    assert session_file.read_bytes() == saved

    def keys(value):
        if isinstance(value, dict):
            return set(value) | {key for row in value.values() for key in keys(row)}
        if isinstance(value, list):
            return {key for row in value for key in keys(row)}
        return set()

    assert not any("saturation" in key for key in keys(json.loads(saved)))
    stored = QSettings(settings, QSettings.Format.IniFormat)
    assert int(stored.value(slv._SATURATION_KEY)) == 30
    again = make_view()
    assert again.saturation.value() == 30 and again.saturation_value.text() == "30 %"
    display, _detail = _shown(again)
    assert again.enhance.isChecked()  # The contrast stretch is on by default.
    assert np.array_equal(display, photo_with_saturation(again._enhanced, 0.3))
