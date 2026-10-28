"""Layer painting tab: the photograph at full resolution when zoomed in.

The display copy (at most DISPLAY_LIMIT_PX) still sets the scene grid for the
marks, strokes and the proposal, and the proposal still reads the same
reduced copy. Over it, the photograph's own pixels are shown when zoomed in,
placed so that the marks sit on the pixels they name. Synthetic 80 x 60
photographs, a 40 x 30 display copy and a project in a temporary folder; no
real series, rig model or NAS input is used.
"""

import math

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMenu

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import photograph_io
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_layer_view import LayerPainterView

EVEN = (200, 40, 40)
ODD = (40, 40, 200)


class _Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


def _striped(width=80, height=60):
    """One-pixel columns that a 2x reduction averages away."""

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, 0::2] = EVEN
    rgb[:, 1::2] = ODD
    rgb[10, 21] = (255, 255, 0)
    return rgb


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Keep remembered choices in a private INI file, not the user's registry."""

    path = str(tmp_path / "painting.ini")
    monkeypatch.setattr(
        slv, "QSettings", lambda *args: QSettings(path, QSettings.Format.IniFormat)
    )


@pytest.fixture
def paths(tmp_path):
    folder = tmp_path / "series"
    folder.mkdir()
    baseline = folder / "raw.png"
    Image.fromarray(_striped()).save(baseline)
    before = folder / "before_indicator.png"
    clear = np.full((60, 80, 3), (30, 160, 30), dtype=np.uint8)
    clear[40, 70] = (250, 250, 250)
    Image.fromarray(clear).save(before)
    config = folder / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    return {"baseline": baseline, "before": before, "config": config, "clear": clear}


def _make_view(qtbot, monkeypatch, tmp_path, paths):
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    reference = SeriesRunInputs(
        "ac52", paths["config"], baseline_image=paths["baseline"], width_m=0.9, height_m=0.49
    )
    widget = LayerPainterView(_Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


@pytest.fixture
def view(qtbot, tmp_path, monkeypatch, settings, paths):
    # A 40 x 30 display copy of the 80 x 60 photographs.
    monkeypatch.setattr(lp, "DISPLAY_LIMIT_PX", 40)
    original = lp.load_painting_image
    monkeypatch.setattr(
        lp,
        "load_painting_image",
        lambda path, **options: original(path, **{"display_limit": 40, **options}),
    )
    widget = _make_view(qtbot, monkeypatch, tmp_path, paths)
    widget.enhance.setChecked(False)
    return widget


def _detail_pixel(view, row, col):
    colour = view.canvas.detail_item.pixmap().toImage().pixelColor(col, row)
    return (colour.red(), colour.green(), colour.blue())


def _zoom(view, scale):
    """Zoom as the mouse wheel does: the view then keeps its scale."""

    view.canvas.resetTransform()
    view.canvas.scale(scale, scale)
    view.canvas._user_zoomed = True
    view.canvas.refresh_detail()


def test_zoomed_in_the_canvas_shows_the_photographs_own_pixels(view):
    assert view.image.display.shape[:2] == (30, 40)
    pixmap = view.canvas.detail_item.pixmap()
    assert (pixmap.width(), pixmap.height()) == (80, 60)
    assert view.canvas.pixmap_item.pixmap().width() == 40
    _zoom(view, 8.0)
    assert view.canvas.detail_shown
    assert _detail_pixel(view, 5, 20) == EVEN
    assert _detail_pixel(view, 5, 21) == ODD
    assert _detail_pixel(view, 10, 21) == (255, 255, 0)


def test_zoomed_out_the_display_copy_is_shown(view):
    # A detail pixel smaller than half a screen pixel: the reduced copy.
    _zoom(view, 0.6)
    assert not view.canvas.detail_shown
    _zoom(view, 1.0)
    assert view.canvas.detail_shown


def test_the_marks_sit_on_the_pixels_they_name(view):
    item = view.canvas.detail_item
    for row, col in ((0, 0), (10, 21), (59, 79), (31.5, 40.25)):
        point = item.mapFromScene(view.canvas.raw_to_scene(row, col))
        assert point.x() == pytest.approx(col + 0.5)
        assert point.y() == pytest.approx(row + 0.5)
        # And back: a click on a shown pixel names that raw pixel.
        raw = view.canvas.scene_to_raw(item.mapToScene(point))
        assert raw == pytest.approx((row, col))


def test_enhanced_contrast_stretches_the_full_resolution_photo_alike(view, paths):
    view.enhance.setChecked(True)
    _zoom(view, 8.0)
    lower, upper = lp.contrast_limits(view.image.display, view._display_mask())
    expected = lp.stretch_contrast_rgb8(_striped(), lower, upper)
    assert _detail_pixel(view, 5, 20) == tuple(int(v) for v in expected[5, 20])
    assert _detail_pixel(view, 5, 21) == tuple(int(v) for v in expected[5, 21])


def test_the_background_photo_is_shown_at_full_resolution_too(view, paths):
    assert view.set_background_photo(paths["before"])
    _zoom(view, 8.0)
    assert view.canvas.detail_shown
    assert _detail_pixel(view, 40, 70) == (250, 250, 250)
    assert _detail_pixel(view, 5, 20) == (30, 160, 30)
    view.toggle_background()
    assert _detail_pixel(view, 5, 20) == EVEN


def test_the_proposal_reads_the_same_reduced_copy(view, paths):
    # The display and analysis copies are the reduced reader's, as before.
    reduced = _old_load_reduced(paths["baseline"], 2)
    assert np.array_equal(view.image.display, reduced)
    assert np.array_equal(view.image.analysis, reduced)


def test_another_photo_in_the_rig_frame_panel_has_no_detail(view, paths):
    assert view.show_frame_photo(paths["before"])
    assert view.canvas.detail_item.pixmap().isNull()
    assert not view.canvas.detail_shown


def test_a_photo_that_fits_the_display_needs_no_detail(
    qtbot, tmp_path, monkeypatch, settings, paths
):
    widget = _make_view(qtbot, monkeypatch, tmp_path, paths)
    assert widget.image.display.shape[:2] == (60, 80)
    assert widget.canvas.detail_item.pixmap().isNull()
    assert not widget.canvas.detail_shown


# ------------------------------------------------- unchanged inputs and stretch


def _old_load_reduced(path, factor):
    """The reader before load_reduced_rgb8 was split out."""

    with Image.open(path) as image:
        width, height = image.size
        if factor > 1:
            image.draft("RGB", (width // factor, height // factor))
        image = image.convert("RGB")
        target = (max(1, width // factor), max(1, height // factor))
        if image.size != target:
            image = image.resize(target, Image.BOX)
        return np.asarray(image, dtype=np.float32) / 255.0


def _old_enhance_contrast(rgb, mask=None, *, low=0.5, high=99.5):
    """enhance_contrast before contrast_limits was split out."""

    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) > 1.5:
        image = image / 255.0
    sample = image[mask] if mask is not None and np.any(mask) else image.reshape(-1, 3)
    if sample.shape[0] > 400_000:
        step = int(math.ceil(sample.shape[0] / 400_000))
        sample = sample[::step]
    lower = np.percentile(sample, low, axis=0)
    upper = np.percentile(sample, high, axis=0)
    span = np.maximum(upper - lower, 1.0e-3)
    stretched = np.clip((image - lower) / span, 0.0, 1.0)
    return np.rint(255.0 * stretched).astype(np.uint8)


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
@pytest.mark.parametrize("factor", [1, 2, 3])
def test_the_reduced_reader_is_unchanged(tmp_path, suffix, factor):
    rng = np.random.default_rng(7)
    path = tmp_path / f"photo{suffix}"
    Image.fromarray(rng.integers(0, 256, (61, 83, 3), dtype=np.uint8)).save(path)
    old = _old_load_reduced(path, factor)
    new = photograph_io.load_reduced(path, factor)
    assert new.dtype == old.dtype and np.array_equal(new, old)
    rgb8 = photograph_io.load_reduced_rgb8(path, factor)
    assert rgb8.dtype == np.uint8
    assert np.array_equal(rgb8.astype(np.float32) / 255.0, old)


@pytest.mark.parametrize("masked", [False, True])
def test_enhance_contrast_is_unchanged_and_the_lookup_matches_it(masked):
    rng = np.random.default_rng(11)
    # More than 400 000 pixels, so that the sampling step is used.
    rgb8 = rng.integers(0, 256, (700, 640, 3), dtype=np.uint8)
    rgb = rgb8.astype(np.float32) / 255.0
    mask = None
    if masked:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[50:650, 40:600] = True
    stretched = lp.enhance_contrast(rgb, mask)
    assert np.array_equal(stretched, _old_enhance_contrast(rgb, mask))
    assert np.array_equal(lp.enhance_contrast(rgb8, mask), _old_enhance_contrast(rgb8, mask))
    lower, upper = lp.contrast_limits(rgb, mask)
    assert np.array_equal(lp.stretch_contrast_rgb8(rgb8, lower, upper), stretched)
