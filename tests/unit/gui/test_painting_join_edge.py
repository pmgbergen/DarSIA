"""Preview, visibility, one undo and unchanged session settings for joining.

The dialog explains a refused join in plain words with a red ring on the
canvas, offers a distance that is checked to work, and joins only the
checked stretches.
"""

import copy
import re
import threading
import time

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMenu
from shiboken6 import isValid

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_join
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower.layer_painting_join import JoinEdgesError, join_to_neighbour
from darsia.fluidflower.series_geometry import SeriesRunInputs


TARGET = [(5.0, 15.0), (5.0, 50.0), (28.0, 50.0), (28.0, 15.0)]
MASTER = [(30.0, 5.0), (30.0, 70.0), (55.0, 70.0), (55.0, 5.0)]
# A thin layer whose top dips to 3 px above its bottom, 2 px above the
# polygon below: at 8 px the dip is joined too and the result would cross
# itself along the shared edge, at raw row 27, column 43.
THIN = [(10.0, 5.0), (10.0, 28.0), (22.0, 40.0), (10.0, 52.0), (10.0, 75.0),
        (25.0, 75.0), (25.0, 5.0)]
BELOW = [(27.0, 2.0), (27.0, 78.0), (55.0, 78.0), (55.0, 2.0)]
DIP = (22.0, 40.0)
# Two separate stretches along the bottom, either of which joins alone.
NOTCHED = [(5.0, 15.0), (5.0, 50.0), (28.0, 50.0), (28.0, 38.0), (20.0, 38.0),
           (20.0, 27.0), (28.0, 27.0), (28.0, 15.0)]
APPLY = QDialogButtonBox.StandardButton.Apply


class Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


def _painter_view(qtbot, tmp_path, monkeypatch, target, master):
    """A shown painting view with the target (layer 1) and the master (layer 2)."""
    settings = str(tmp_path / "settings.ini")
    monkeypatch.setattr(slv, "QSettings", lambda *a: QSettings(settings, QSettings.Format.IniFormat))
    monkeypatch.setattr(QMenu, "popup", lambda *a, **k: None)
    baseline = tmp_path / "raw.png"
    Image.new("RGB", (80, 60), (30, 40, 50)).save(baseline)
    config = tmp_path / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    reference = SeriesRunInputs("example", config, baseline_image=baseline,
                                width_m=0.08, height_m=0.06)
    widget = slv.LayerPainterView(Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    qtbot.waitExposed(widget)
    widget.add_polygon(lp.Polygon(1, list(target)))
    widget.add_polygon(lp.Polygon(2, list(master)))
    widget.set_tool("edit")
    return widget


@pytest.fixture
def view(qtbot, tmp_path, monkeypatch):
    return _painter_view(qtbot, tmp_path, monkeypatch, TARGET, MASTER)


@pytest.fixture
def notched_view(qtbot, tmp_path, monkeypatch):
    return _painter_view(qtbot, tmp_path, monkeypatch, NOTCHED, MASTER)


def _dialog(view, qtbot, points, master, distance):
    """The join dialog at ``distance`` mm, one raw pixel per mm."""
    dialog = slv._JoinEdgesDialog(
        view, view.canvas, list(points), [(1, "Neighbour", master)], 1.0
    )
    qtbot.addWidget(dialog)
    dialog.distance.setValue(distance)
    return dialog


def test_apply_and_one_undo_restore_both_polygons_and_all_settings(view):
    before = copy.deepcopy(view.session.to_dict())
    undo_count = len(view._undo_stack)
    plan = join_to_neighbour(TARGET, MASTER, 3.0)
    assert view._apply_join_preview(0, 1, plan)
    assert len(view._undo_stack) == undo_count + 1
    after = copy.deepcopy(view.session.to_dict())
    assert before["polygons"] != after["polygons"]
    assert {k: v for k, v in before.items() if k != "polygons"} == {
        k: v for k, v in after.items() if k != "polygons"
    }
    # Ordinary polygons carry shared edges; no join-specific session field.
    restored = lp.PaintingSession.from_dict(after)
    assert lp.shared_edges(restored.polygons)
    view.undo()
    assert view.session.to_dict() == before
    view.redo()
    assert view.session.to_dict() == after


def test_preview_draws_stretches_and_length_but_changes_nothing(view, qtbot):
    before = copy.deepcopy(view.session.to_dict())
    dialog = slv._JoinEdgesDialog(view, view.canvas, TARGET, [(1, "Neighbour", MASTER)], 1.0)
    qtbot.addWidget(dialog)
    assert dialog.distance.value() == 5
    assert "mm" in dialog.summary.text() and "1 stretch" in dialog.summary.text()
    assert not dialog.old_item.path().isEmpty()
    assert not dialog.new_item.path().isEmpty()
    assert view.session.to_dict() == before
    dialog.distance.setValue(1)
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Apply).isEnabled()
    assert "No stretch" in dialog.summary.text()
    dialog.reject()
    assert dialog.old_item.scene() is None
    assert dialog.new_item.scene() is None
    assert view.session.to_dict() == before


def test_hidden_neighbour_is_ignored_and_cannot_be_applied(view):
    plan = join_to_neighbour(TARGET, MASTER, 3)
    before = copy.deepcopy(view.session.to_dict())
    view._set_hidden_layers({2})
    view._join_polygon_dialog(0)
    assert "Show another polygon" in view.canvas_hint.text()
    assert not view._apply_join_preview(0, 1, plan)
    assert view.session.to_dict() == before


def test_selected_polygon_button_and_context_menu_reach_the_same_action(view, monkeypatch):
    calls = []
    monkeypatch.setattr(view, "_join_polygon_dialog", calls.append)
    view.canvas.set_selection([(0, 0), (0, 1)])
    view.join_edge_button.click()
    view._polygon_menu(0, -1, QPoint(0, 0))
    action = next(a for a in view._polygon_menu_widget.actions()
                  if a.text() == "Join to neighbour's edge…")
    action.trigger()
    assert calls == [0, 0]


def test_refused_self_intersection_shows_location_and_disables_apply(view, qtbot):
    crossed = [(0, 0), (10, 10), (0, 10), (10, 0)]
    dialog = slv._JoinEdgesDialog(view, view.canvas, crossed, [(1, "Neighbour", MASTER)], 1)
    qtbot.addWidget(dialog)
    assert dialog.summary.text() == (
        "The selected polygon crosses itself at the marked point. "
        "Correct its outline before joining."
    )
    assert "raw row 5.00, column 5.00" in dialog.summary.toolTip()
    assert dialog.error_item.toolTip() == "Raw row 5.00, column 5.00"
    assert not dialog.error_item.path().isEmpty()
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Apply).isEnabled()
    # A smaller distance cannot mend the selected polygon: nothing is offered.
    assert dialog.use_button.isHidden() and dialog._thread is None
    dialog.reject()


_ADVICE = "Use a smaller distance, uncheck the stretch there, or join from the other polygon."


@pytest.mark.parametrize(
    "error, words",
    [
        (
            JoinEdgesError("x", (1.0, 2.0), outline="Joined polygon",
                           reason="Ring Self-intersection[1 2]"),
            "The joined outline would cross itself at the marked point. " + _ADVICE,
        ),
        (
            JoinEdgesError("x", None, outline="Joined polygon", reason="Valid Geometry"),
            "The joined outline would not be a valid outline. " + _ADVICE,
        ),
        (
            JoinEdgesError("x", (1.0, 2.0), outline="Neighbour polygon after endpoint insertion",
                           reason="Self-intersection[1 2]"),
            "The neighbour's outline would not stay valid at the marked point. "
            "Use a smaller distance, or join from the other polygon.",
        ),
        (
            JoinEdgesError("x", (1.0, 2.0), outline="Neighbour polygon",
                           reason="Self-intersection[1 2]"),
            "This neighbour crosses itself at the marked point. "
            "Correct its outline, or choose another neighbour.",
        ),
        (
            JoinEdgesError("x", None, outline="Selected polygon", reason="Too few finite points"),
            "The selected polygon is not a valid outline. Correct its outline before joining.",
        ),
        (
            JoinEdgesError("The whole outline is within the distance. Choose a smaller distance."),
            "The whole outline is within the distance. Choose a smaller distance.",
        ),
    ],
)
def test_every_refusal_is_said_in_plain_words(error, words):
    assert slv._JoinEdgesDialog._plain_refusal(error) == words


def _ring_centre(dialog):
    """The raw (row, column) at the centre of the red ring."""
    return dialog.canvas.scene_to_raw(dialog.error_item.path().boundingRect().center())


def test_a_refused_join_is_explained_with_a_red_ring_that_goes_with_it(view, qtbot):
    before = copy.deepcopy(view.session.to_dict())
    dialog = _dialog(view, qtbot, THIN, BELOW, 8.0)
    text = dialog.summary.text()
    assert text.startswith(
        "The joined outline would cross itself at the marked point. Use a smaller distance, "
        "uncheck the stretch there, or join from the other polygon."
    )
    assert "raw" not in text.lower() and "Self-intersection" not in text
    assert "raw row 27.00, column 43.00" in dialog.summary.toolTip()
    assert dialog.error_item.toolTip() == "Raw row 27.00, column 43.00"
    assert dialog.error_item.scene() is view.canvas.scene()
    assert _ring_centre(dialog) == pytest.approx((27.0, 43.0), abs=1e-6)
    assert not dialog.buttons.button(APPLY).isEnabled()
    dialog.distance.setValue(4.0)
    assert dialog.error_item.path().isEmpty() and dialog.error_item.toolTip() == ""
    assert "1 stretch" in dialog.summary.text() and dialog.buttons.button(APPLY).isEnabled()
    dialog.distance.setValue(8.0)
    assert not dialog.error_item.path().isEmpty()
    dialog.reject()
    assert dialog.error_item.scene() is None
    assert dialog._thread is None and dialog._worker is None
    assert view.session.to_dict() == before


def test_a_distance_checked_to_work_is_offered_and_used(view, qtbot):
    dialog = _dialog(view, qtbot, THIN, BELOW, 8.0)
    button = dialog.use_button
    assert not button.isHidden() and not button.isEnabled()
    assert button.text() == "Finding a distance that works…"
    qtbot.waitUntil(lambda: dialog.offered_mm is not None, timeout=10000)
    offered = dialog.offered_mm
    assert not button.isHidden() and button.isEnabled()
    assert re.fullmatch(r"Use \d+\.\d\d mm", button.text())
    assert button.text() == f"Use {offered:.2f} mm" and offered < 8.0
    assert "checked to give a valid join" in button.toolTip()
    assert "not necessarily the largest" in button.toolTip()
    button.click()
    assert dialog.distance.value() == offered
    assert dialog.buttons.button(APPLY).isEnabled()
    assert "1 stretch" in dialog.summary.text()
    assert dialog.error_item.path().isEmpty()
    assert button.isHidden() and dialog.offered_mm is None
    dialog.reject()


def test_no_offer_when_no_smaller_distance_works(view, qtbot):
    tiny = [(10.0, 10.0), (10.0, 10.6), (10.5, 10.3)]
    box = [(12.0, 0.0), (12.0, 30.0), (30.0, 30.0), (30.0, 0.0)]
    dialog = _dialog(view, qtbot, tiny, box, 5.0)
    assert dialog.summary.text().startswith("The whole outline is within the distance.")
    qtbot.waitUntil(lambda: dialog._thread is None, timeout=10000)
    assert dialog.use_button.isHidden() and dialog.offered_mm is None
    assert dialog.summary.text().endswith("No smaller distance was found that works.")
    assert not dialog.buttons.button(APPLY).isEnabled()
    dialog.reject()


def test_the_stretch_list_changes_the_preview_at_once(view, qtbot):
    dialog = _dialog(view, qtbot, THIN, BELOW, 8.0)
    items = [dialog.stretch_list.item(row) for row in range(dialog.stretch_list.count())]
    assert len(items) == 2
    assert all(item.checkState() == Qt.CheckState.Checked for item in items)
    for number, item in enumerate(items):
        length = dialog.stretches[number].length_px
        assert item.text().startswith(f"Stretch {number + 1}: {length:.2f} mm")
    dip = next(row for row, stretch in enumerate(dialog.stretches) if DIP in stretch.before)
    assert items[dip].text().endswith(" (at the red ring)")
    assert not dialog.buttons.button(APPLY).isEnabled()
    both = dialog.old_item.path().elementCount()
    items[dip].setCheckState(Qt.CheckState.Unchecked)
    assert dialog.checked_stretches() == [1 - dip]
    assert dialog.buttons.button(APPLY).isEnabled()
    assert dialog.summary.text().startswith("1 stretch")
    assert dialog.error_item.path().isEmpty()
    assert dialog.preview.stretches == (dialog.stretches[1 - dip],)
    assert DIP in dialog.preview.target
    assert dialog.old_item.path().elementCount() < both
    assert not items[dip].text().endswith(" (at the red ring)")
    # The dip alone is refused by the same check, with the same words and ring.
    items[dip].setCheckState(Qt.CheckState.Checked)
    items[1 - dip].setCheckState(Qt.CheckState.Unchecked)
    assert dialog.summary.text().startswith(
        "The joined outline would cross itself at the marked point."
    )
    assert not dialog.error_item.path().isEmpty()
    assert not dialog.buttons.button(APPLY).isEnabled()
    assert items[dip].text().endswith(" (at the red ring)")
    items[dip].setCheckState(Qt.CheckState.Unchecked)
    assert dialog.summary.text() == "No stretch is checked. Check a stretch to join it."
    assert dialog.error_item.path().isEmpty()
    assert not dialog.buttons.button(APPLY).isEnabled()
    dialog.reject()


def test_a_new_distance_lists_its_stretches_all_checked(view, qtbot):
    dialog = _dialog(view, qtbot, THIN, BELOW, 8.0)
    dialog.stretch_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    dialog.distance.setValue(4.0)
    assert dialog.stretch_list.count() == 1
    assert dialog.stretch_list.item(0).checkState() == Qt.CheckState.Checked
    assert dialog.preview.stretches == dialog.stretches
    dialog.reject()


def test_selecting_a_stretch_highlights_its_old_outline(view, qtbot):
    dialog = _dialog(view, qtbot, THIN, BELOW, 8.0)
    assert dialog.current_item.path().isEmpty()
    dialog.stretch_list.setCurrentRow(0)
    assert not dialog.current_item.path().isEmpty()
    dialog.distance.setValue(4.0)
    assert dialog.current_item.path().isEmpty()
    dialog.reject()
    assert dialog.current_item.scene() is None


def test_long_refusals_wrap_and_keep_the_dialog_small(view, qtbot):
    dialog = _dialog(view, qtbot, THIN, BELOW, 8.0)
    assert dialog.summary.wordWrap()
    hint = dialog.minimumSizeHint()
    assert hint.width() < 600 and hint.height() < 600
    dialog.reject()


def test_closing_during_a_search_stops_it_and_frees_the_worker_here(view, qtbot, monkeypatch):
    started = threading.Event()

    def slow(points, other, distance_px, tolerance_px, *, cancelled=None):
        started.set()
        while not cancelled():
            time.sleep(0.005)
        return None

    monkeypatch.setattr(layer_painting_join, "verified_working_distance", slow)
    dialog = _dialog(view, qtbot, THIN, BELOW, 8.0)
    assert started.wait(5.0)
    worker = dialog._worker
    dialog.reject()
    assert dialog._thread is None and dialog._worker is None
    assert isValid(worker), "the worker was deleted in its own thread"
    del worker  # The last reference: the worker is freed here, in the GUI thread.


def test_apply_joins_only_the_checked_stretches_in_one_undo_step(notched_view, monkeypatch):
    view = notched_view
    before = copy.deepcopy(view.session.to_dict())
    undo_count = len(view._undo_stack)
    seen = {}

    def run(dialog):
        dialog.distance.setValue(5.0 / dialog.pixels_per_mm)
        assert dialog.stretch_list.count() == 2
        dialog.stretch_list.item(1).setCheckState(Qt.CheckState.Unchecked)
        assert dialog.buttons.button(APPLY).isEnabled()
        seen["joined"], seen["kept"] = dialog.stretches
        dialog.accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(slv._JoinEdgesDialog, "exec", run)
    view._join_polygon_dialog(0)
    assert len(view._undo_stack) == undo_count + 1
    after = copy.deepcopy(view.session.to_dict())
    joined = [tuple(point) for point in view.session.polygons[0].points]
    master = view.session.polygons[1].points
    assert all(point in joined for point in seen["joined"].after)
    assert all(point in joined for point in seen["kept"].before[1:-1])
    assert len(master) == len(MASTER) + 2
    assert {k: v for k, v in before.items() if k != "polygons"} == {
        k: v for k, v in after.items() if k != "polygons"
    }
    view.undo()
    assert view.session.to_dict() == before
    view.redo()
    assert view.session.to_dict() == after
