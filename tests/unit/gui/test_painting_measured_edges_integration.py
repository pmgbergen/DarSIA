"""Saved edge metadata on the actual painting view, using temporary PNG inputs."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import json
import threading

import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QTimer, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower import painting_measured_edges_display as display
from darsia.fluidflower import painting_measured_edges_ui as measured_ui
from darsia.fluidflower import painting_run_preview_ui as preview_ui
from darsia.fluidflower import series_layer_view as layer_ui
from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT
from darsia.fluidflower.series_geometry import SeriesRunInputs

from ..test_painting_measured_edges import saved, _write, _files, SHAPE, CONFIRMATION, LINES  # noqa: F401


class _Project:
    def __init__(self, root, series):
        self.root, self.project_id = root, "fixture-project"
        self.manifest = {"rig": {"type": "medium"}, "tools": {"native_series": {
            "series_root": str(series), "series_id": "fixture"}}}

    def update_metadata(self, *args, **kwargs):
        pass

    def _audit(self, *args, **kwargs):
        pass


@pytest.fixture
def context(saved, qtbot, monkeypatch):
    # Every settings writer used by the real host is redirected to an INI file.
    settings = str(saved.folder / "user-settings.ini")
    for module in (layer_ui, preview_ui):
        monkeypatch.setattr(module, "QSettings", lambda *args: QSettings(settings, QSettings.Format.IniFormat))
    series = saved.folder / "native-series"
    (series / "config").mkdir(parents=True)
    registry = saved.folder / "registry.toml"
    registry.write_text("schema_version = 1\n", encoding="utf8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    monkeypatch.setenv("DARSIA_FIXTURE_PROJECT_ROOT", str(series))
    (series / "series.toml").write_text('schema_version = 1\nseries_id = "fixture"\n'
        '[paths]\ncommon_config = "config/common.toml"\n[runs]\nids = ["sample", "target"]\n'
        '[runs.config_files]\nsample = "config/sample.toml"\ntarget = "config/target.toml"\n', encoding="utf8")
    (series / "config/common.toml").write_text("# No original analysis boxes\n", encoding="utf8")
    photo, other = series / "reference.png", series / "other.png"
    Image.new("RGB", SHAPE[::-1], (40, 90, 140)).save(photo)
    Image.new("RGB", SHAPE[::-1], (170, 80, 30)).save(other)
    photograph = painting.image_stamp(photo)
    saved.photograph.clear()
    raw = photo.read_bytes()
    saved.photograph.update(path=str(photo), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    saved.model.update(reference_image=deepcopy(saved.photograph), model_photograph=deepcopy(saved.photograph),
        H_metric_mm_to_camera_u=[[1., 0., 0.], [0., -1., 23.], [0., 0., 1.]],
        metric_frame={"unit": "mm", "x_axis": "right", "y_axis": "up", "measured_width_mm": 31.})
    saved.save()
    root = saved.folder / "geometry-package"
    project = _Project(root, series)
    current = root / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    _write(current, dict(format=GEOMETRY_REVISION_FORMAT, revision_id="geometry-fixture"))
    config_path = root / display.RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = dict(format=display.FORMAT, project_id=project.project_id, series_root=str(series),
        measured_edges=dict(geometry_index=deepcopy(saved.index_ref), expected_confirmation=CONFIRMATION))
    _write(config_path, config)
    references = {}
    corners = {"top_left": [0., 0.], "top_right": [0., 31.],
               "bottom_right": [23., 31.], "bottom_left": [23., 0.]}
    for run, path in (("sample", photo), ("target", other)):
        path_config = series / "config" / (run + ".toml")
        path_config.write_text(f'[data]\nfolder = "{series.as_posix()}"\nbaseline = "{path.name}"\n', encoding="utf8")
        references[run] = SeriesRunInputs(run, path_config, baseline_image=path,
            width_m=.031, height_m=.023, crop=deepcopy(corners))
    # A PNG is decoded through the actual host loader. Its synthetic camera-U
    # profile is attached explicitly; this test does not decode camera RAW.
    original = painting.load_painting_image
    def loaded(path, **options):
        image = original(path, **options)
        image.camera_u = SimpleNamespace(**deepcopy(saved.camera))
        return image
    monkeypatch.setattr(painting, "load_painting_image", loaded)
    QSettings(settings, QSettings.Format.IniFormat).setValue(preview_ui._INDEX, str(saved.index_path))
    gui_thread, getters = threading.get_ident(), []
    def reference():
        getters.append(threading.get_ident())
        assert getters[-1] == gui_thread, "A worker called the host reference getter"
        return references["sample"]
    view = layer_ui.LayerPainterView(project, reference, run_inputs=lambda: list(references.values()))
    qtbot.addWidget(view)
    view.resize(1450, 1050)
    view.show()
    qtbot.waitExposed(view)
    qtbot.waitUntil(lambda: not view.busy() and not view.run_preview.busy(), timeout=15000)
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    view._dirty = False
    assert view.session.image_stamp == photograph
    assert view.measured_edges.result is not None, view.measured_edges.note.text()
    value = SimpleNamespace(view=view, controller=view.measured_edges, saved=saved,
        config=config, config_path=config_path, photo=photo, other=other, gui_thread=gui_thread, getters=getters)
    yield value
    view.wait_for_workers()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    view._dirty = False


def _wait(context, qtbot):
    qtbot.waitUntil(lambda: not context.view.busy() and not context.view.run_preview.busy(), timeout=15000)


def test_actual_host_default_off_saved_three_lines_and_fits_preserve_files_arrays_and_coordinates(context):
    view, controller = context.view, context.controller
    view.step_frame.set_expanded(True)
    assert controller.widget.isVisibleTo(view) and not controller.show_edges.isChecked()
    assert not controller._items
    before = (_files(context.saved.folder), deepcopy(view.session.to_dict()),
              view.image.display.copy(), view.image.analysis.copy(), deepcopy(view.current_frame().corners))
    raw_point = view.canvas.raw_to_scene(8.125, 12.75)
    controller.show_edges.click()
    assert len(controller._items) == 3
    assert dict(controller.result.edges.lines) == {name: tuple(value) for name, value in LINES.items()}
    assert all("top" not in item.toolTip().casefold() for item in controller._items)
    assert all(item.acceptedMouseButtons() == Qt.MouseButton.NoButton for item in controller._items)
    assert controller.fit_camera_u() and controller.fit_rig_frame() and controller.fit_camera_u()
    assert view.canvas.raw_to_scene(8.125, 12.75) == raw_point
    assert _files(context.saved.folder) == before[0] and view.session.to_dict() == before[1]
    assert np.array_equal(view.image.display, before[2]) and np.array_equal(view.image.analysis, before[3])
    assert view.current_frame().corners == before[4]
    assert controller.result.configuration == context.config


def test_reference_background_other_photo_and_other_run_refresh_without_line_leak(context, qtbot):
    view, controller = context.view, context.controller
    controller.show_edges.setChecked(True)
    assert len(controller._items) == 3
    assert view.show_frame_photo(context.other)
    assert not controller._items and not controller.show_edges.isEnabled()
    assert context.other == view._other_photo
    view.show_reference_photo()
    assert len(controller._items) == 3
    assert view.set_background_photo(context.other, quiet=True)
    assert not controller._items and not controller.show_edges.isEnabled()
    view.show_background(False)
    assert len(controller._items) == 3
    view.run_preview.runs.setCurrentIndex(view.run_preview.runs.findData("target"))
    _wait(context, qtbot)
    assert not controller._items and not controller.fit_camera_button.isEnabled()
    view.run_preview.runs.setCurrentIndex(view.run_preview.runs.findData("sample"))
    assert len(controller._items) == 3


def test_changed_companion_is_visible_and_never_written(context, qtbot):
    controller = context.controller
    context.config_path.write_text(json.dumps(dict(context.config, project_id="another-project")), encoding="utf8")
    before = _files(context.saved.folder)
    assert controller.reload()
    _wait(context, qtbot)
    assert controller.result is None and not controller.show_edges.isEnabled()
    assert "another project" in controller.note.text()
    assert _files(context.saved.folder) == before


def test_absent_project_companion_disables_lines_without_creating_a_file(context, qtbot):
    view, controller = context.view, context.controller
    previous = view.project.root
    absent = context.saved.folder / "project-without-display-metadata"
    before = _files(context.saved.folder)
    try:
        view.project.root = absent
        assert controller.reload()
        _wait(context, qtbot)
        assert controller.result is None and not controller.show_edges.isEnabled()
        assert "No measured-edge display metadata" in controller.note.text()
        assert not absent.exists() and _files(context.saved.folder) == before
    finally:
        view.project.root = previous


def test_reference_reload_revalidates_on_worker_and_preserves_user_overlay_choice(context, qtbot, monkeypatch):
    view, controller = context.view, context.controller
    calls = []
    original = measured_ui.load_measured_edge_display
    def observed(**arguments):
        calls.append(threading.get_ident())
        return original(**arguments)
    monkeypatch.setattr(measured_ui, "load_measured_edge_display", observed)
    controller.show_edges.setChecked(True)
    previous = controller.result
    before = _files(context.saved.folder)
    view.reload()
    _wait(context, qtbot)
    assert calls and all(thread != context.gui_thread for thread in calls)
    assert all(thread == context.gui_thread for thread in context.getters)
    assert controller.result is not None and controller.result is not previous
    assert controller.show_edges.isChecked() and len(controller._items) == 3
    assert _files(context.saved.folder) == before


def test_busy_measured_load_blocks_host_and_parent_close_wait_drains_without_relaunch(context, qtbot, monkeypatch):
    view, controller = context.view, context.controller
    parent = QWidget()
    qtbot.addWidget(parent)
    QVBoxLayout(parent).addWidget(view)
    parent.show()
    qtbot.waitExposed(parent)
    started, release = threading.Event(), threading.Event()
    original = measured_ui.load_measured_edge_display
    calls, beats = [], []
    timer = QTimer(view)
    timer.setInterval(5)
    timer.timeout.connect(lambda: beats.append(True))
    def blocked(**arguments):
        calls.append(threading.get_ident())
        started.set()
        assert release.wait(10)
        return original(**arguments)
    monkeypatch.setattr(measured_ui, "load_measured_edge_display", blocked)
    timer.start()
    try:
        assert controller.reload()
        qtbot.waitUntil(started.is_set, timeout=3000)
        assert view.busy() and not view.can_close()
        qtbot.waitUntil(lambda: len(beats) >= 3, timeout=3000)
        event = QCloseEvent()
        QApplication.sendEvent(parent, event)
        assert not event.isAccepted() and parent.isVisible()
        # A reference reload requests a future read, but shutdown/wait must
        # not start it after the current worker has been joined.
        view._measured_edges_reload_pending = True
        release.set()
        view.wait_for_workers()
        assert not controller.busy() and view.can_close() and len(calls) == 1
        assert view._measured_edges_reload_pending
    finally:
        release.set()
        view._measured_edges_reload_pending = False
        controller.wait_for_workers()
        timer.stop()


def test_actual_canvas_deletion_invalidates_lines_and_late_events(context):
    view, controller = context.view, context.controller
    controller.show_edges.setChecked(True)
    assert controller._items
    view.canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    controller.refresh()
    assert not controller._items and not controller.reload()
    assert not controller.eventFilter(controller._viewport, QEvent(QEvent.Type.Resize))
