"""Passive measured lines keep the actual canvas and worker lifecycle intact."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import QGraphicsItem, QVBoxLayout, QWidget
from shiboken6 import isValid

from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower import painting_measured_edges_ui as ui
from darsia.fluidflower.series_geometry import NATIVE_SERIES_TOOL_KEY, RigFrame
from darsia.fluidflower.series_layer_view import LayerCanvas

from ..test_painting_measured_edges import saved, _files, LINES, SHAPE  # noqa: F401
from ..test_painting_measured_edges_display import make_display


class _Preview(QObject):
    display_changed = Signal()

    def __init__(self, index, parent):
        super().__init__(parent)
        self._index = str(index)
        self.reference = True

    def is_reference(self):
        return self.reference


@pytest.fixture
def context(saved, qtbot):
    arguments, config, config_path = make_display(saved)
    view = QWidget()
    qtbot.addWidget(view)
    layout = QVBoxLayout(view)
    view.project = SimpleNamespace(root=arguments["project_root"], project_id=arguments["project_id"],
        manifest={"tools": {NATIVE_SERIES_TOOL_KEY: {"series_root": arguments["series_root"]}}})
    view.session = painting.PaintingSession.default("sample", arguments["photograph_path"], SHAPE[::-1])
    view.session.image_stamp = arguments["photograph_stamp"]
    view.session.polygons = [painting.Polygon(2, [(5.125, 8.), (5., 23.), (18., 20.)])]
    view.session.strokes = [painting.Stroke(2, 1.25, [(8., 12.)])]
    profile = SimpleNamespace(**{key: value for key, value in saved.camera.items() if key != "size_bytes"})
    rgb = np.full((*SHAPE, 3), .4, np.float32)
    view.image = painting.PaintingImage(Path(arguments["photograph_path"]), SHAPE[::-1], rgb, rgb.copy(),
                                       camera_u=profile)
    view._reference_token = {"reference": True, "path": str(view.image.path), "raw_size": SHAPE[::-1]}
    view._other_photo = None
    view._background_shown = False
    view._workspace_frame_config = {}
    view.canvas = LayerCanvas()
    view.canvas.setParent(view)
    layout.addWidget(view.canvas)
    view.canvas.set_photo(np.full((*SHAPE, 3), 102, np.uint8), SHAPE[::-1],
                          token=view._reference_token, reset_view=True)
    view.run_preview = _Preview(saved.index_path, view)
    frame = RigFrame(corners={"top_left": (5., 8.), "top_right": (5., 23.),
        "bottom_right": (18., 23.), "bottom_left": (18., 8.)}, width_m=.015, height_m=.013,
        run_id="sample", image=str(view.image.path), image_size=SHAPE[::-1])
    view.current_frame = lambda: frame
    view.busy = lambda: False
    controller = ui.PaintingMeasuredEdgesController(view)
    layout.addWidget(controller.widget)
    view.resize(960, 740)
    view.show()
    qtbot.waitExposed(view)
    value = SimpleNamespace(view=view, controller=controller, frame=frame, saved=saved,
                            configuration=config, config_path=config_path)
    yield value
    controller.wait_for_workers()


def _load(context, qtbot):
    assert context.controller.reload()
    qtbot.waitUntil(lambda: not context.controller.busy(), timeout=10000)
    assert context.controller.result is not None, context.controller.note.text()


def test_actual_canvas_saved_lines_and_both_fits_leave_drawing_and_pixels_unchanged(context, qtbot):
    view, controller = context.view, context.controller
    before_files = _files(context.saved.folder)
    before_session, before_frame = deepcopy(view.session.to_dict()), deepcopy(context.frame.corners)
    before_pixels = view.image.display.copy(), view.image.analysis.copy()
    point = view.canvas.raw_to_scene(9.125, 12.75)
    _load(context, qtbot)
    assert not controller._items and not controller.show_edges.isChecked()
    controller.show_edges.click()
    assert len(controller._items) == 3
    for item, (name, coefficients) in zip(controller._items, controller.result.edges.lines):
        assert name in LINES and item.acceptedMouseButtons() == Qt.MouseButton.NoButton
        assert not item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        path = item.path()
        for number in (0, 1):
            element = path.elementAt(number)
            row, col = view.canvas.scene_to_raw(QPointF(element.x, element.y))
            a, b, c = coefficients
            assert abs(a*col + b*row + c) < 1.e-10
    assert controller.fit_camera_u()
    camera_scale = view.canvas.transform().m11()
    assert controller.fit_rig_frame()
    assert view.canvas.transform().m11() > camera_scale
    assert controller.fit_camera_u()
    assert view.canvas.raw_to_scene(9.125, 12.75) == point
    assert view.session.to_dict() == before_session and context.frame.corners == before_frame
    assert np.array_equal(view.image.display, before_pixels[0])
    assert np.array_equal(view.image.analysis, before_pixels[1])
    assert _files(context.saved.folder) == before_files
    assert "Fit camera U" == controller.fit_camera_button.text()
    assert "full pre-crop sensor" in controller.fit_camera_button.toolTip()


def test_construction_is_passive_and_missing_companion_error_is_explicit(context, qtbot):
    controller = context.controller
    assert controller.result is None and not controller.busy() and not controller._items
    context.view.project.root = context.saved.folder / "missing-project"
    assert controller.reload()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=10000)
    assert controller.result is None and not controller.show_edges.isEnabled()
    assert "No measured-edge display metadata" in controller.note.text()
    assert not context.view.project.root.exists()


def test_rig_fit_refuses_a_frame_from_another_run_or_grid(context):
    context.frame.run_id = "other"
    assert not context.controller.fit_rig_frame()
    context.frame.run_id = "sample"
    context.frame.image_size = (31, 24)
    assert not context.controller.fit_rig_frame()


@pytest.mark.parametrize("change", ["other_run", "other_photo", "background", "placement", "grid",
                                    "token", "session_photo", "changed_photo", "corrected", "profile"])
def test_foreign_or_unproven_canvas_hides_lines_and_disables_fits(context, qtbot, change):
    _load(context, qtbot)
    view, controller = context.view, context.controller
    controller.show_edges.click()
    assert controller._items
    if change == "other_run":
        view.run_preview.reference = False
    elif change == "other_photo":
        view._other_photo = object()
    elif change == "background":
        view._background_shown = True
    elif change == "placement":
        view.canvas.place_mode = True
    elif change == "grid":
        view.canvas._raw_size = (31, 24)
    elif change == "token":
        view.canvas.preview = {}
    elif change == "session_photo":
        view.session.image_path = "another-photo.arw"
    elif change == "changed_photo":
        view.session.photo_changed = True
    elif change == "corrected":
        view._workspace_frame_config = {"frame": "rig-corrected"}
    else:
        view.image.camera_u = None
    view.run_preview.display_changed.emit()
    assert not controller._items
    assert not controller.show_edges.isEnabled()
    assert not controller.fit_camera_u() and not controller.fit_rig_frame()


def test_worker_is_off_gui_responsive_and_released_only_after_join(context, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = ui.load_measured_edge_display
    beats, busy_events = [], []
    timer = QTimer(context.view)
    timer.setInterval(5)
    timer.timeout.connect(lambda: beats.append(1))
    controller = context.controller
    context.view.busy = controller.busy
    gui_thread = context.view.thread()

    def blocked(**kwargs):
        assert QThread.currentThread() != gui_thread
        entered.set()
        assert release.wait(10)
        return original(**kwargs)

    monkeypatch.setattr(ui, "load_measured_edge_display", blocked)
    controller.busy_changed.connect(lambda value: busy_events.append(
        (value, controller._thread, controller._worker)))
    timer.start()
    try:
        assert controller.reload()
        qtbot.waitUntil(entered.is_set, timeout=3000)
        assert controller.busy() and not controller.can_close()
        qtbot.waitUntil(lambda: len(beats) >= 3, timeout=3000)
        worker, thread = controller._worker, controller._thread
        assert worker is not None and thread.isRunning()
        release.set()
        qtbot.waitUntil(lambda: not controller.busy(), timeout=10000)
        assert controller.result is not None and controller.can_close()
        assert busy_events[-1] == (False, None, None)
        assert not isValid(thread) or not thread.isRunning()
    finally:
        release.set()
        controller.wait_for_workers()
        timer.stop()


def test_reference_switch_while_worker_runs_discards_result(context, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = ui.load_measured_edge_display

    def blocked(**kwargs):
        entered.set()
        assert release.wait(10)
        return original(**kwargs)

    monkeypatch.setattr(ui, "load_measured_edge_display", blocked)
    controller = context.controller
    try:
        assert controller.reload()
        qtbot.waitUntil(entered.is_set, timeout=3000)
        context.view.session = deepcopy(context.view.session)
        controller.refresh()
        release.set()
        qtbot.waitUntil(lambda: not controller.busy(), timeout=10000)
        assert controller.result is None and not controller._items
        assert "stale measured lines were discarded" in controller.note.text()
    finally:
        release.set()
        controller.wait_for_workers()


def test_canvas_delete_later_and_late_events_are_harmless(context, qtbot):
    _load(context, qtbot)
    controller, canvas = context.controller, context.view.canvas
    controller.show_edges.click()
    canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qtbot.waitUntil(lambda: not isValid(canvas))
    controller.refresh()
    assert controller.eventFilter(canvas, QEvent(QEvent.Type.Resize)) is False
    assert not controller.fit_camera_u() and not controller.fit_rig_frame()
    assert controller._disposed and not controller._items
