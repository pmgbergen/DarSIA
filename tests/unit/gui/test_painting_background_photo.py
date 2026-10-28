"""Layer painting tab: another photograph of the run as the background.

The background is display only. It must have the raw size of the reference
photo; marks, strokes and the proposal stay on top, painting keeps working,
and the proposal and the saved session always use the reference photo. A
synthetic 80 x 60 photograph and a project in a temporary folder; no real
series, rig model or NAS input is used.
"""

import json

import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMenu, QMessageBox

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_layer_view import LayerPainterView

REFERENCE_RGB = (30, 40, 50)
BACKGROUND_RGB = (200, 160, 40)
POLYGON = [(5.0, 5.0), (5.0, 35.0), (30.0, 35.0), (30.0, 5.0)]


class _Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Keep the remembered choice in a private INI file, not the user's registry.

    QSettings(organization, application) always uses the native format, which
    is the registry on Windows, so the view's QSettings is replaced here.
    """

    path = str(tmp_path / "painting.ini")
    monkeypatch.setattr(
        slv, "QSettings", lambda *args: QSettings(path, QSettings.Format.IniFormat)
    )


@pytest.fixture
def paths(tmp_path):
    folder = tmp_path / "series"
    folder.mkdir()
    baseline = folder / "raw.png"
    Image.new("RGB", (80, 60), REFERENCE_RGB).save(baseline)
    before = folder / "before_indicator.png"
    Image.new("RGB", (80, 60), BACKGROUND_RGB).save(before)
    other_size = folder / "other_camera.png"
    Image.new("RGB", (64, 48), BACKGROUND_RGB).save(other_size)
    config = folder / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    return {"baseline": baseline, "before": before, "other": other_size, "config": config}


@pytest.fixture
def view(qtbot, tmp_path, monkeypatch, settings, paths):
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    reference = SeriesRunInputs(
        "ac52", paths["config"], baseline_image=paths["baseline"], width_m=0.9, height_m=0.49
    )
    widget = LayerPainterView(_Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    qtbot.waitExposed(widget)
    widget.enhance.setChecked(False)  # compare plain colours
    return widget


def _shown_pixel(view, row, col):
    image = view.canvas.pixmap_item.pixmap().toImage()
    colour = image.pixelColor(col, row)
    return (colour.red(), colour.green(), colour.blue())


def _shown_path(view):
    return view.canvas.preview["path"]


def test_a_photo_of_the_same_size_is_shown_under_the_marks(view, paths):
    view.add_polygon(lp.Polygon(2, list(POLYGON)))
    assert view.set_background_photo(paths["before"])
    assert view.background_shown and view.background_path == paths["before"]
    assert _shown_path(view) == str(paths["before"])
    assert _shown_pixel(view, 40, 60) == BACKGROUND_RGB
    # Marks stay on top and painting stays possible.
    assert view.canvas._polygon_items and all(
        item.isVisible() for item in view.canvas._polygon_items
    )
    assert view.canvas.paint_enabled
    assert not view.photo_note_row.isVisible()


def test_t_switches_between_the_reference_and_the_background(view, paths):
    assert view.set_background_photo(paths["before"])
    view.toggle_background()
    assert not view.background_shown
    assert _shown_path(view) == str(paths["baseline"])
    assert _shown_pixel(view, 40, 60) == REFERENCE_RGB
    view.toggle_background()
    assert view.background_shown and _shown_pixel(view, 40, 60) == BACKGROUND_RGB
    # The T shortcut is installed on the canvas.
    from PySide6.QtGui import QKeySequence, QShortcut

    keys = [shortcut.key() for shortcut in view.canvas.findChildren(QShortcut)]
    assert QKeySequence("T") in keys


def test_the_background_box_lists_and_switches_the_photos(view, paths):
    box = view.background
    assert [box.itemText(i) for i in range(box.count())] == [
        "Reference photo",
        "Choose photo...",
    ]
    assert view.set_background_photo(paths["before"])
    assert [box.itemText(i) for i in range(box.count())] == [
        "Reference photo",
        "before_indicator.png",
        "Choose photo...",
    ]
    assert box.currentIndex() == 1
    view._background_activated(0)
    assert not view.background_shown and box.currentIndex() == 0
    view._background_activated(1)
    assert view.background_shown and box.currentIndex() == 1


def test_choose_photo_opens_a_file_dialog_in_the_run_folder(view, paths, monkeypatch):
    seen = {}

    def fake_dialog(parent, title, start, filters):
        seen["start"] = start
        return str(paths["before"]), filters

    monkeypatch.setattr(slv.QFileDialog, "getOpenFileName", fake_dialog)
    view._background_activated(view.background.count() - 1)
    assert view.background_shown and view.background_path == paths["before"]
    assert seen["start"] == str(paths["baseline"].parent.parent)


def test_a_photo_of_another_size_is_refused(view, paths, monkeypatch):
    messages = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: messages.append(args[2])
    )
    assert not view.set_background_photo(paths["other"])
    assert messages and "64 x 48" in messages[0] and "80 x 60" in messages[0]
    assert view.background_path is None and not view.background_shown
    assert _shown_path(view) == str(paths["baseline"])


def test_the_proposal_and_the_saved_session_still_use_the_reference(view, paths, tmp_path):
    view.add_polygon(lp.Polygon(2, list(POLYGON)))
    view.save_session()
    saved = lp.session_path(view.project, "ac52")
    before = json.loads(saved.read_text(encoding="utf-8"))
    assert view.set_background_photo(paths["before"])
    view.save_session()
    assert json.loads(saved.read_text(encoding="utf-8")) == before
    view.add_polygon(lp.Polygon(3, [(35.0, 5.0), (35.0, 30.0), (55.0, 30.0), (55.0, 5.0)]))
    view.save_session()
    after = json.loads(saved.read_text(encoding="utf-8"))
    # The background is not written to the session; the image is the reference.
    assert after["image"] == before["image"]
    # The reference path itself may legitimately contain "background". Check
    # the document fields rather than searching arbitrary serialized values.
    assert after.keys() == before.keys()
    # The intentional polygon edit also updates its ordinary edit timestamp.
    edited = {"polygons", "updated_utc"}
    assert {key: value for key, value in after.items() if key not in edited} == {
        key: value for key, value in before.items() if key not in edited
    }
    assert str(view.image.path) == str(paths["baseline"])


def test_drawing_on_the_background_keeps_the_background(view, paths):
    assert view.set_background_photo(paths["before"])
    for tool in ("polygon", "edit", "paint", "erase", "pan"):
        view.set_tool(tool)
        assert view.background_shown and _shown_path(view) == str(paths["before"])


def test_contrast_enhancement_applies_to_the_background(view, paths):
    assert view.set_background_photo(paths["before"])
    view.enhance.setChecked(True)
    assert view.background_shown and _shown_path(view) == str(paths["before"])
    view.enhance.setChecked(False)
    assert _shown_pixel(view, 40, 60) == BACKGROUND_RGB


def test_the_choice_is_remembered_per_run_and_restored_on_reload(view, paths):
    assert view.set_background_photo(paths["before"])
    view.reload()
    # Restored for this run, but the reference photo is shown first.
    assert view.background_path == paths["before"]
    assert not view.background_shown
    assert _shown_path(view) == str(paths["baseline"])
    view.toggle_background()
    assert view.background_shown
    # Forgetting it removes the remembered choice.
    assert view.set_background_photo(None)
    view.reload()
    assert view.background_path is None


def test_a_remembered_photo_that_is_gone_is_ignored(view, paths, tmp_path):
    gone = tmp_path / "series" / "gone.png"
    Image.new("RGB", (80, 60), BACKGROUND_RGB).save(gone)
    assert view.set_background_photo(gone)
    gone.unlink()
    view.reload()
    assert view.background_path is None and not view.background_shown
