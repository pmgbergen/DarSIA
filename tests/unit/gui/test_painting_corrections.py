"""Layer painting tab: hand corrections on top of the proposal.

Correct and Restore are brushes, Lasso fills a drawn outline and Fill takes a
clicked region. Corrections are kept in their own file, one step each to undo,
and never change the polygons, strokes, proposal or session file. A synthetic
80 x 60 photograph and a project in a temporary folder; no real series is used.
"""

import json

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu, QMessageBox

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import series_layer_view as view_module
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_layer_view import LayerPainterView

LEFT_BUTTON = Qt.MouseButton.LeftButton
NO = Qt.KeyboardModifier.NoModifier
# The default layers: 1 Free water, 2 Fine sand, 3 Coarse sand, 4 Excluded.
WATER, FINE, COARSE, EXCLUDED = 1, 2, 3, 4


class _Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


def _proposal(digest="first"):
    """Fine sand on the left, coarse sand on the right, a 2 px border outside."""

    classes = np.full((60, 80), lp.OUTSIDE, dtype=np.uint8)
    classes[2:58, 2:40] = FINE
    classes[2:58, 40:78] = COARSE
    confidence = np.full(classes.shape, 255, dtype=np.uint8)
    return lp.LayerProposal(classes, confidence, (80, 60), digest)


@pytest.fixture
def view(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    folder = tmp_path / "series"
    folder.mkdir()
    baseline = folder / "raw.png"
    Image.new("RGB", (80, 60), (30, 40, 50)).save(baseline)
    config = folder / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    reference = SeriesRunInputs(
        "ac52", config, baseline_image=baseline, width_m=0.9, height_m=0.49
    )
    widget = LayerPainterView(_Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.resize(1400, 900)
    widget.show()
    qtbot.waitExposed(widget)
    assert widget.canvas._scale() == (1.0, 1.0)
    return widget


def _show(view, proposal):
    view._proposal_ready(proposal)
    view._refresh_status()


def _corrected(view):
    return lp.corrected_classes(view.proposal, view.corrections)[0]


def _at(canvas, row, col):
    return canvas.mapFromScene(canvas.raw_to_scene(row, col))


def _stroke(canvas, points):
    QTest.mousePress(canvas.viewport(), LEFT_BUTTON, NO, _at(canvas, *points[0]))
    for point in points[1:]:
        QTest.mouseMove(canvas.viewport(), _at(canvas, *point))
    QTest.mouseRelease(canvas.viewport(), LEFT_BUTTON, NO, _at(canvas, *points[-1]))


def _session_file(view):
    path = lp.session_path(view.project, "ac52")
    return path.read_bytes() if path.exists() else None


def test_the_correction_tools_wait_for_a_proposal(view):
    view._refresh_status()
    for key in ("correct", "restore", "lasso", "fill"):
        assert not view.tool_buttons[key].isEnabled()
    before = view.canvas.tool
    view.set_tool("correct")
    assert view.canvas.tool == before
    assert "Propose the boundaries first" in view.canvas_hint.text()
    assert not view.add_correction(lp.Correction("brush", FINE, [(5, 5)], radius_px=2))
    _show(view, _proposal())
    assert all(view.tool_buttons[key].isEnabled() for key in ("correct", "fill"))
    assert view.corrections_label.text() == "No corrections."


def test_a_correction_brush_paints_its_layer_and_leaves_the_session_alone(view, qtbot):
    _show(view, _proposal())
    view.save_session()
    session_before = _session_file(view)
    view.select_layer_row(2)  # Coarse sand
    view.set_tool("correct")
    assert view.brush_size.isEnabled()
    _stroke(view.canvas, [(30.0, 10.0), (30.0, 20.0), (30.0, 30.0)])
    assert len(view.corrections) == 1
    (correction,) = view.corrections
    assert correction.kind == "brush" and correction.layer_id == COARSE
    assert correction.proposal_digest == "first" and correction.created_utc
    assert _corrected(view)[30, 20] == COARSE and view.proposal.classes[30, 20] == FINE
    assert view.corrections_label.text() == "1 correction(s)"
    qtbot.waitUntil(lambda: not view._corrections_dirty, timeout=3000)
    saved = lp.load_corrections(view.project, "ac52")
    assert saved == view.corrections
    assert _session_file(view) == session_before
    assert not lp.proposal_path(view.project, "ac52").exists()


def test_restore_lasso_and_fill_apply_in_drawing_order(view):
    _show(view, _proposal())
    view.select_layer_row(3)  # Excluded
    lasso = lp.Correction(
        "lasso", EXCLUDED, [(10.0, 10.0), (10.0, 30.0), (20.0, 30.0), (20.0, 10.0)]
    )
    assert view.add_correction(lasso)
    view.select_layer_row(0)  # Free water
    assert view.fill_region_at(40.0, 60.0)  # The coarse sand region.
    restore = lp.Correction("restore", 0, [(15.0, 15.0)], radius_px=2.0)
    assert view.add_correction(restore)
    classes = _corrected(view)
    assert classes[18, 25] == EXCLUDED and classes[15, 15] == FINE
    assert np.all(classes[2:58, 40:78] == WATER)
    assert np.all(classes[:2] == lp.OUTSIDE)
    assert [row.kind for row in view.corrections] == ["lasso", "fill", "restore"]
    assert view.corrections[1].replaced_layer == COARSE
    assert view.corrections_label.text() == "3 correction(s)"


def test_fill_refuses_the_outside_and_a_region_of_the_selected_layer(view):
    _show(view, _proposal())
    view.select_layer_row(1)  # Fine sand
    assert not view.fill_region_at(0.0, 0.0)
    assert "inside the rig frame" in view.canvas_hint.text()
    assert not view.fill_region_at(30.0, 10.0)
    assert "already has the selected layer" in view.canvas_hint.text()
    assert view.corrections == []


def test_undo_and_redo_take_back_corrections_and_marks_separately(view):
    _show(view, _proposal())
    view.add_polygon(lp.Polygon(FINE, [(5.0, 5.0), (5.0, 20.0), (20.0, 20.0)]))
    view.save_session()
    assert not view._dirty
    view.select_layer_row(2)
    assert view.add_correction(lp.Correction("brush", COARSE, [(30, 10)], radius_px=3))
    view.undo()
    assert view.corrections == [] and len(view.session.polygons) == 1
    assert not view._dirty  # Undoing a correction does not touch the session.
    view.redo()
    assert len(view.corrections) == 1 and len(view.session.polygons) == 1
    view.undo()
    view.undo()
    assert view.corrections == [] and view.session.polygons == []


def test_clear_asks_first_and_is_one_step_to_undo(view, monkeypatch):
    _show(view, _proposal())
    view.select_layer_row(2)
    for col in (10.0, 20.0):
        brush = lp.Correction("brush", COARSE, [(30.0, col)], radius_px=2.0)
        view.add_correction(brush)
    answers = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: answers.pop(0),
    )
    answers.append(QMessageBox.StandardButton.No)
    view.clear_corrections_button.click()
    assert len(view.corrections) == 2
    answers.append(QMessageBox.StandardButton.Yes)
    view.clear_corrections_button.click()
    assert view.corrections == []
    view.undo()
    assert len(view.corrections) == 2


def test_corrections_return_with_the_run_and_stay_on_a_new_proposal(view, qtbot):
    _show(view, _proposal())
    view.select_layer_row(2)
    view.add_correction(lp.Correction("brush", COARSE, [(30.0, 10.0)], radius_px=2.0))
    view.save_session()
    _proposal().save(lp.proposal_path(view.project, "ac52"))
    view.reload()
    assert view.proposal is not None and view.corrections == lp.load_corrections(
        view.project, "ac52"
    )
    assert len(view.corrections) == 1
    _show(view, _proposal(digest="second"))
    assert _corrected(view)[30, 10] == COARSE
    assert view.corrections_label.text() == (
        "1 correction(s) · 1 drawn on an earlier proposal"
    )


def test_a_fill_that_no_longer_fits_is_counted_not_applied(view):
    _show(view, _proposal())
    view.select_layer_row(0)
    assert view.fill_region_at(30.0, 60.0)
    changed = _proposal(digest="second")
    changed.classes[2:58, 40:78] = FINE  # The coarse sand is gone.
    _show(view, changed)
    assert view.corrections_label.text() == (
        "1 correction(s) · 1 drawn on an earlier proposal · 1 no longer apply"
    )
    assert not np.any(_corrected(view) == WATER)


def test_an_unreadable_correction_file_turns_correcting_off(view):
    path = lp.corrections_path(view.project, "ac52")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    _proposal().save(lp.proposal_path(view.project, "ac52"))
    view.reload()
    assert view.proposal is not None and not view.corrections_available()
    assert not view.tool_buttons["correct"].isEnabled()
    assert "could not be read" in view.corrections_label.text()
    assert not view.add_correction(lp.Correction("brush", FINE, [(5, 5)], radius_px=2))
    view.save_session()
    assert path.read_text(encoding="utf-8") == "not json"


def test_the_shown_proposal_outlines_the_corrections(view):
    _show(view, _proposal())
    view.select_layer_row(2)
    outline = [(20.0, 10.0), (20.0, 30.0), (40.0, 30.0), (40.0, 10.0)]
    view.add_correction(lp.Correction("lasso", COARSE, outline))

    def colour(row, col):
        image = view.canvas.proposal_item.pixmap().toImage()
        return QColor(image.pixel(col, row)).getRgb()[:3]

    edge = view_module._CORRECTED_EDGE[:3]
    assert colour(20, 20) == edge and colour(30, 20) != edge
    view.show_corrections.setChecked(False)
    assert colour(20, 20) != edge


def test_saving_passes_the_corrections_to_the_revision(view, monkeypatch):
    _show(view, _proposal())
    view.select_layer_row(2)
    view.add_correction(lp.Correction("brush", COARSE, [(30.0, 10.0)], radius_px=2.0))
    seen = {}
    monkeypatch.setattr(
        view_module.lp,
        "save_painted_revision",
        lambda *args, **kwargs: seen.update(kwargs),
    )
    monkeypatch.setattr(
        view_module, "export_native_setup", lambda *args, **kwargs: {"ok": True}
    )
    worker = view_module._SaveWorker(
        view.project,
        view.session,
        view.proposal,
        view.reference(),
        None,
        "",
        corrections=tuple(view.corrections),
    )
    finished = []
    worker.finished.connect(finished.append)
    worker.run()
    assert finished == [{"ok": True}]
    assert seen["corrections"] == tuple(view.corrections)


def test_the_correction_file_is_one_correction_per_line(view):
    _show(view, _proposal())
    view.select_layer_row(2)
    view.add_correction(lp.Correction("brush", COARSE, [(30.0, 10.0)], radius_px=2.0))
    view.select_layer_row(0)
    view.fill_region_at(30.0, 60.0)
    view.save_session()
    text = lp.corrections_path(view.project, "ac52").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.lstrip().startswith('{"kind"')]
    assert len(rows) == 2
    assert json.loads(text)["corrections"][1]["kind"] == "fill"
