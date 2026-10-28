"""Inline boxes on the actual painting view, with private photos/models/configs."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
from types import SimpleNamespace
import threading

import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QVBoxLayout, QWidget

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import metric_analysis_domains as domains
from darsia.fluidflower import painting_analysis_box_sources as source_module
from darsia.fluidflower import painting_run_preview_ui as preview_ui
from darsia.fluidflower import series_layer_view as layer_ui
from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT
from darsia.fluidflower.map_placement import GEOMETRY_FORMAT
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_layer_view import LayerPainterView

from .test_native_sidebar import native_window  # noqa: F401


class _Project:
    def __init__(self, root, series):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}, "tools": {"native_series": {
            "series_root": str(series), "series_id": "testseries"}}}

    def update_metadata(self, *args, **kwargs):
        pass

    def _audit(self, *args, **kwargs):
        pass


@pytest.fixture
def context(qtbot, tmp_path, monkeypatch, request):
    settings_path = tmp_path / "private-user.ini"
    gui_thread = threading.get_ident()
    def private_settings(*args):
        assert tuple(args) == preview_ui._SETTINGS
        assert threading.get_ident() == gui_thread, "A worker accessed QSettings"
        value = QSettings(str(settings_path), QSettings.Format.IniFormat)
        value.setFallbacksEnabled(False)
        assert Path(value.fileName()) == settings_path and not value.fallbacksEnabled()
        return value
    for module in (layer_ui, preview_ui):
        monkeypatch.setattr(module, "QSettings", private_settings)
    def forbidden_picker(*_args, **_kwargs):
        pytest.fail("Known frame models must not require a file picker")
    monkeypatch.setattr(preview_ui.QFileDialog, "getOpenFileName", forbidden_picker)
    registry = tmp_path / "registry.toml"
    registry.write_text("schema_version = 1\n", encoding="utf8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    series = tmp_path / "series"
    (series / "config/runs").mkdir(parents=True)
    monkeypatch.setenv("DARSIA_TESTSERIES_PROJECT_ROOT", str(series))
    (series / "series.toml").write_text('schema_version = 1\nseries_id = "testseries"\n'
        '[paths]\ncommon_config = "config/common.toml"\n[runs]\nids = ["run_a", "run_b"]\n'
        '[runs.config_files]\nrun_a = "config/runs/first.toml"\nrun_b = "config/runs/second.toml"\n', encoding="utf8")
    common = series / "config/common.toml"
    common.write_text('[rig]\nwidth = 0.159\nheight = 0.119\n'
                      '[corrections.curvature.crop]\n'
                      'top_left = [0.0, 0.0]\nbottom_left = [119.0, 0.0]\n'
                      'bottom_right = [119.0, 159.0]\ntop_right = [0.0, 159.0]\n'
                      '[[roi]]\nname = "box1"\ncorner_1 = [0.01, 0.005]\ncorner_2 = [0.03, 0.02]\n'
                      '[[roi]]\nname = "box2"\ncorner_1 = [0.04, 0.025]\ncorner_2 = [0.06, 0.04]\n', encoding="utf8")
    project = _Project(tmp_path / "geometry-package", series)
    current = project.root / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(dict(format=GEOMETRY_REVISION_FORMAT, revision_id="geometry-1")), encoding="utf8")
    entries, references, photos = {}, {}, {}
    corners = dict(top_left=[0., 0.], bottom_left=[119., 0.],
                   bottom_right=[119., 159.], top_right=[0., 159.])
    for number, run in enumerate(("run_a", "run_b")):
        photo = series / (run + ".png")
        Image.new("RGB", (160, 120), (40 + number*100, 60, 80)).save(photo)
        config = series / "config/runs" / ("first.toml" if number == 0 else "second.toml")
        config.write_text(f'[data]\nfolder = "{series.as_posix()}"\nbaseline = "{run}.png"\n', encoding="utf8")
        references[run] = SeriesRunInputs(run, config, baseline_image=photo,
            width_m=.159, height_m=.119, crop=deepcopy(corners))
        photos[run] = photo
        model = series / (run + "-model.json")
        model.write_text(json.dumps(dict(format=GEOMETRY_FORMAT, run_id=run,
            reference_image=domains.descriptor(photo), shape_hw=[120, 160],
            H_metric_mm_to_camera_u=[[1., 0., 5.*number], [0., -1., 119.], [0., 0., 1.]],
            metric_frame=dict(unit="mm", x_axis="right", y_axis="up", measured_width_mm=159.))), encoding="utf8")
        entries[run] = dict(reference_model=domains.descriptor(model))
    index = series / "index.json"
    index.write_text(json.dumps(dict(format=GEOMETRY_FORMAT, runs=entries)), encoding="utf8")
    settings = private_settings(*preview_ui._SETTINGS)
    if getattr(request, "param", None) != "late_index":
        settings.setValue(preview_ui._INDEX, str(index))
    settings.sync()
    # The real preview policy freezes this actual temporary project authority.
    (project.root / "series.json").write_text(json.dumps(project.manifest), encoding="utf8")
    active = {"run": "run_a"}
    gui_thread, getter_threads = threading.get_ident(), []
    def reference():
        getter_threads.append(threading.get_ident())
        assert getter_threads[-1] == gui_thread, "A worker called the GUI reference getter"
        return references[active["run"]]
    view = LayerPainterView(project, reference, run_inputs=lambda: list(references.values()))
    qtbot.addWidget(view)
    view.resize(1500, 1100)
    view.show()
    qtbot.waitExposed(view)
    qtbot.waitUntil(lambda: not view.busy() and not view.run_preview.busy(), timeout=10000)
    if getattr(request, "param", None) != "late_index":
        assert view.analysis_boxes.draft is not None, view.analysis_boxes.note.text()
        assert not view.analysis_boxes._binding_error, view.analysis_boxes.note.text()
    value = SimpleNamespace(view=view, project=project, series=series, current=current, index=index,
        references=references, photos=photos, active=active, getter_threads=getter_threads,
        gui_thread=gui_thread, user_settings=settings)
    yield value
    view.wait_for_workers()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    view._dirty = False
    view.hide()


def _row(controller, role):
    return next(row for row in controller.draft.configuration()["domains"] if row["role"] == role)


def _files(context):
    return {str(path): path.read_bytes() for root in (context.series, context.project.root)
            for path in root.rglob("*") if path.is_file()}


def _wait(context, qtbot):
    qtbot.waitUntil(lambda: not context.view.busy() and not context.view.run_preview.busy(), timeout=15000)


def test_actual_view_exposes_original_rectangles_with_pending_frame_and_worker_only_provider(context, qtbot, monkeypatch):
    view, controller = context.view, context.view.analysis_boxes
    view.set_workflow_scope('analysis_boxes')
    assert controller.widget.isVisible()
    assert controller.table.rowCount() == 2
    assert _row(controller, "box1")["bounds_mm"] == [10., 30., 5., 20.]
    assert _row(controller, "box2")["bounds_mm"] == [40., 60., 25., 40.]
    assert all(row["legacy_import"]["decision"] == "pending" for row in controller.draft.configuration()["domains"])
    assert "Follow edges and Join" in controller.widget.toolTip()
    before = _files(context), deepcopy(view.session.to_dict()), deepcopy(controller.draft.configuration())
    calls = []
    original = source_module.load_legacy_box_source
    def observed(project, run_id, *, reference_config_path):
        calls.append((run_id, reference_config_path, threading.get_ident()))
        return original(project, run_id, reference_config_path=reference_config_path)
    monkeypatch.setattr(source_module, "load_legacy_box_source", observed)
    view._sync_analysis_boxes(force=True)
    _wait(context, qtbot)
    assert calls and all(run == "run_a" and path == context.references["run_a"].config_path
                         and thread != context.gui_thread for run, path, thread in calls)
    assert all(thread == context.gui_thread for thread in context.getter_threads)
    assert _files(context) == before[0] and view.session.to_dict() == before[1]
    assert controller.draft.configuration() == before[2]


def test_view_undo_routes_to_boxes_and_shared_tool_group_keeps_layer_stack_unchanged(context):
    view, controller = context.view, context.view.analysis_boxes
    view.add_polygon(lp.Polygon(2, [(5., 5.), (5., 80.), (80., 80.), (80., 5.)]))
    view.save_session()
    before = deepcopy(view.session.to_dict()), deepcopy(view._undo_stack), deepcopy(view.corrections)
    original = controller.draft.configuration()
    controller.edit_button.click()
    assert controller.mode == "edit" and view.canvas.tool == "pan"
    assert not view.fill_region_tool.active
    assert sum(button.isChecked() for button in view.tool_group.buttons()) == 1
    assert not view.follow_edges.isEnabled() and not view.join_edge_button.isEnabled()
    controller.snap_enabled.setChecked(False)
    controller.table.item(0, 3).setText("32.1234567890123")
    changed = controller.draft.configuration()
    assert changed != original and len(controller._undo) == 1
    view.undo()
    assert controller.draft.configuration() == original
    view.redo()
    assert controller.draft.configuration() == changed
    assert view.session.to_dict() == before[0] and view._undo_stack == before[1] and view.corrections == before[2]
    view.set_tool("polygon")
    assert controller.mode is None and view.canvas.tool == "polygon"
    assert view.follow_edges.isEnabled() and view.join_edge_button.isEnabled()


def test_reference_change_retains_unsaved_series_wide_draft_and_undo_without_prompt(context, qtbot, monkeypatch):
    view, controller = context.view, context.view.analysis_boxes
    controller.snap_enabled.setChecked(False)
    controller.table.item(0, 3).setText("33.1234567890123")
    draft, count = controller.draft.configuration(), len(controller._undo)
    def forbidden(*args, **kwargs):
        raise AssertionError("An automatic reference rebind must not ask to discard the draft")
    monkeypatch.setattr(QMessageBox, "question", forbidden)
    view.run_preview.runs.blockSignals(True)
    view.run_preview.runs.setCurrentIndex(view.run_preview.runs.findData("run_b"))
    view.run_preview.runs.blockSignals(False)
    context.active["run"] = "run_b"
    view.reload()
    _wait(context, qtbot)
    assert view.session.run_id == "run_b"
    assert not controller._binding_error, controller.note.text()
    assert controller.draft.configuration() == draft and controller.dirty
    assert len(controller._undo) == count
    assert controller.geometry.run_id == "run_b"
    assert controller.undo()
    assert _row(controller, "box1")["bounds_mm"] == [10., 30., 5., 20.]


def test_incompatible_saved_binding_retains_draft_read_only_with_reason(context, qtbot, monkeypatch):
    view, controller = context.view, context.view.analysis_boxes
    controller.snap_enabled.setChecked(False)
    controller.table.item(0, 3).setText("34.1234567890123")
    before, count = controller.draft.configuration(), len(controller._undo)
    def forbidden(*args, **kwargs):
        raise AssertionError("An incompatible automatic rebind must retain the draft without a prompt")
    monkeypatch.setattr(QMessageBox, "question", forbidden)
    context.current.write_text(json.dumps(dict(format=GEOMETRY_REVISION_FORMAT, revision_id="geometry-2")), encoding="utf8")
    view._sync_analysis_boxes(force=True)
    _wait(context, qtbot)
    assert controller.draft.configuration() == before and len(controller._undo) == count
    assert controller.geometry is None and not controller.draw_button.isEnabled()
    assert "retained" in controller.note.text() and "Reload boxes" in controller.note.text()


def test_actual_target_worker_completion_projects_boxes_through_explicit_native_display_proof(context, qtbot, monkeypatch):
    view, controller, preview = context.view, context.view.analysis_boxes, context.view.run_preview
    view.session.polygons = [
        lp.Polygon(2, [(0., 0.), (0., 79.), (119., 79.), (119., 0.)]),
        lp.Polygon(3, [(0., 80.), (0., 159.), (119., 159.), (119., 80.)]),
    ]
    view._render_marks()
    original = preview_ui.preview.prepare_preview
    def reduced(*args, **kwargs):
        return original(*args, **kwargs, display_limit=80)
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", reduced)
    before = _files(context), deepcopy(view.session.to_dict()), deepcopy(controller.draft.configuration())
    preview.runs.setCurrentIndex(preview.runs.findData("run_b"))
    _wait(context, qtbot)
    result = preview.last_result
    assert result is not None, preview.note.text()
    assert result.photo_path == str(context.photos["run_b"])
    assert result.native_shape_hw == (120, 160) and result.display_shape_hw == (60, 80)
    assert len(view._analysis_boxes_preview_items) == 2
    first = view._analysis_boxes_preview_items[0].path().elementAt(0)
    # box1's (10,5) mm corner: camera (15,114), then exact half-size pixel centres.
    expected = preview.canvas.raw_to_scene((114.+.5)*.5-.5, (15.+.5)*.5-.5)
    assert first.x == pytest.approx(expected.x()) and first.y == pytest.approx(expected.y())
    assert all("read only" in item.toolTip() for item in view._analysis_boxes_preview_items)
    assert controller.mode is None and not controller.draw_button.isEnabled()
    assert _files(context) == before[0] and view.session.to_dict() == before[1]
    assert controller.draft.configuration() == before[2]
    assert not list(context.project.root.rglob("approvals.jsonl"))
    # A late target-photo callback must not restore boxes outside Analysis.
    for scope in ("rig_frame", "layers", "apply"):
        view._workflow_scope = scope
        view._refresh_analysis_box_preview()
        assert not view._analysis_boxes_preview_items
    view._workflow_scope = "all"
    view._refresh_analysis_box_preview()
    assert len(view._analysis_boxes_preview_items) == 2
    # Old packets without a native proof do not acquire a guessed overlay.
    preview.last_result = replace(result, native_shape_hw=None, H_native_to_display=None)
    preview.display_changed.emit()
    assert not view._analysis_boxes_preview_items


def test_busy_metadata_blocks_parent_close_and_wait_joins_without_gui_quit_dependency(context, qtbot, monkeypatch):
    view = context.view
    window = QWidget()
    qtbot.addWidget(window)
    layout = QVBoxLayout(window)
    layout.addWidget(view)
    window.show()
    qtbot.waitExposed(window)
    started, release = threading.Event(), threading.Event()
    original = source_module.load_legacy_box_source
    def held(*args, **kwargs):
        started.set()
        if not release.wait(5.):
            raise ValueError("The private provider was not released")
        return original(*args, **kwargs)
    monkeypatch.setattr(source_module, "load_legacy_box_source", held)
    try:
        view._sync_analysis_boxes(force=True)
        qtbot.waitUntil(started.is_set, timeout=5000)
        assert view.busy() and not view.can_close()
        event = QCloseEvent()
        QApplication.sendEvent(window, event)
        assert not event.isAccepted() and window.isVisible()
    finally:
        release.set()
    # No GUI event processing between releasing the real metadata worker and
    # joining it: thread retirement must not depend on a queued GUI quit.
    view.wait_for_workers()
    assert not view.analysis_boxes.busy() and view.can_close()
    assert view.analysis_boxes._thread is None and view.analysis_boxes._worker is None
    window.close()
    assert not window.isVisible()



def test_native_window_has_one_inline_box_route_and_retains_legacy_import(native_window, tmp_path):
    # native_window installs global private INI format/UserScope and empty
    # project/series registries BEFORE constructing the actual main window.
    # The isolated launcher may additionally redirect default constructors to
    # its own private INI root. This explicit probe stays in this fixture root.
    settings = QSettings(str(tmp_path / "native-window-proof.ini"), QSettings.Format.IniFormat)
    settings.setFallbacksEnabled(False)
    assert settings.format() == QSettings.Format.IniFormat
    assert settings.scope() == QSettings.Scope.UserScope
    assert Path(settings.fileName()) == tmp_path / "native-window-proof.ini"
    assert not settings.fallbacksEnabled()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    from darsia.fluidflower.metric_analysis_domains_ui import NativeMetricBoxesWorkflow

    assert isinstance(NativeMetricBoxesWorkflow, type), "The compatibility import must remain available"
    assert not hasattr(native_window, "metric_boxes_workflow")
    assert not any(button.text().startswith("Analysis boxes [mm]")
                   for button in native_window.findChildren(QPushButton))
    assert not hasattr(LayerPainterView, "_show_separate_box_button")
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_other_photo_and_late_canvas_destruction_cannot_leave_box_interception_active(context):
    view, controller = context.view, context.view.analysis_boxes
    controller.draw_button.click()
    assert controller.mode == "draw"
    assert view.show_frame_photo(context.photos["run_b"])
    assert controller.mode is None and not controller.draw_button.isEnabled()
    view.show_reference_photo()
    assert controller.draw_button.isEnabled()
    canvas = view.canvas
    canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    controller.refresh()
    assert not controller.set_mode("draw")
    assert not controller.eventFilter(controller._viewport, QEvent(QEvent.Type.MouseMove))


@pytest.mark.parametrize("changed_run", ["run_a", "run_b"])
def test_same_index_path_changed_source_or_target_model_hides_only_unbound_boxes_until_reload(
        context, qtbot, monkeypatch, changed_run):
    view, controller, preview = context.view, context.view.analysis_boxes, context.view.run_preview
    before_binding = deepcopy(controller.geometry_binding)
    source_geometry = controller.geometry.metric_to_grid.copy()
    model_path = context.series / (changed_run + "-model.json")
    document = json.loads(model_path.read_text())
    document["H_metric_mm_to_camera_u"][0][2] += 3.
    model_path.write_text(json.dumps(document), encoding="utf8")
    index = json.loads(context.index.read_text())
    index["runs"][changed_run]["reference_model"] = domains.descriptor(model_path)
    context.index.write_text(json.dumps(index), encoding="utf8")
    # The target worker reads the replacement at the SAME path. Photo bytes,
    # paths and native grids remain unchanged, so dimensions cannot detect it.
    preview.runs.setCurrentIndex(preview.runs.findData("run_b"))
    _wait(context, qtbot)
    result = preview.last_result
    assert result is not None, preview.note.text()
    assert result.photo_path == str(context.photos["run_b"])
    assert result.geometry_binding["geometry_index"] != before_binding["geometry_index"]
    assert not view._analysis_boxes_preview_items
    assert preview.canvas.preview["path"] == result.photo_path
    assert preview.canvas.raw_size == (160, 120)
    # A new worker proof invalidates the cached reference geometry as well;
    # returning to the reference must not allow edits against the old model.
    assert controller.geometry is None and controller._binding_error
    assert not controller.draw_button.isEnabled()
    assert np.array_equal(source_geometry, np.asarray(before_binding["models"]["run_a"]["metric_to_grid"]))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    assert controller.reload()
    _wait(context, qtbot)
    assert controller.geometry_binding == result.geometry_binding
    assert len(view._analysis_boxes_preview_items) == 2


@pytest.mark.parametrize("tool", ["paint", "erase"])
def test_box_exit_restores_prior_brush_tool_button_and_size_widgets(context, tool):
    view, controller = context.view, context.view.analysis_boxes
    view.set_tool(tool)
    assert view.brush_size.isEnabled() and view.brush_label.isEnabled()
    controller.draw_button.click()
    assert controller.mode == "draw" and not view.brush_size.isEnabled()
    controller.set_mode(None)
    assert controller.mode is None and view.canvas.tool == tool
    assert view.tool_buttons[tool].isChecked()
    assert view.brush_size.isEnabled() and view.brush_label.isEnabled()


@pytest.mark.parametrize("tool", ["paint", "erase"])
def test_box_exit_on_other_photo_keeps_exact_photo_token_and_pan_without_drawing(context, tool):
    view, controller = context.view, context.view.analysis_boxes
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    view._dirty = False
    before = (_files(context), deepcopy(view.session.to_dict()),
              deepcopy(controller.draft.configuration()), deepcopy(view._undo_stack))
    view.set_tool(tool)
    controller.draw_button.click()
    assert controller.mode == "draw" and view.canvas.tool == "pan"
    other = context.photos["run_b"]
    assert view.show_frame_photo(other)
    token = view.canvas.preview
    assert controller.mode is None and view._other_photo == other
    assert token is not view._reference_token and token["path"] == str(other)
    assert not controller.draw_button.isEnabled() and not controller.edit_button.isEnabled()
    assert view.canvas.tool == "pan" and view.tool_buttons["pan"].isChecked()
    assert not view.canvas.paint_enabled
    assert not view.brush_size.isEnabled() and not view.brush_label.isEnabled()
    # A later refresh/deactivation also retains the same object and photograph.
    controller.refresh()
    controller.set_mode(None)
    assert view.canvas.preview is token and view._other_photo == other
    assert view.canvas.tool == "pan" and not view.canvas.paint_enabled
    assert _files(context) == before[0] and view.session.to_dict() == before[1]
    assert controller.draft.configuration() == before[2] and view._undo_stack == before[3]


def test_saved_user_index_enables_draw_at_actual_view_start_without_picker(context):
    assert context.view.run_preview._index == str(context.index)
    assert context.view.analysis_boxes.draw_button.isEnabled()
    assert context.view.analysis_boxes._context_key[-1] == str(context.index)


@pytest.mark.parametrize("context", ["late_index"], indirect=True)
def test_empty_cached_index_recovers_saved_user_choice_on_reference_reload_without_picker(context, qtbot):
    view, boxes = context.view, context.view.analysis_boxes
    assert view.run_preview._index == "" and not boxes.draw_button.isEnabled()
    assert "series has no measured frame models" in boxes.note.text()
    assert "Frame models" in boxes.note.text()
    before = _files(context)
    context.user_settings.setValue(preview_ui._INDEX, str(context.index))
    context.user_settings.sync()
    view.reload()
    _wait(context, qtbot)
    assert view.run_preview._index == str(context.index)
    assert boxes._context_key[-1] == str(context.index) and boxes.draw_button.isEnabled()
    assert _files(context) == before


def test_rechecking_settings_preserves_current_override_and_unsaved_box_draft(context, qtbot):
    view, boxes = context.view, context.view.analysis_boxes
    boxes.edit_button.click()
    boxes.table.item(0, 3).setText("32.125")
    assert boxes.dirty
    before = deepcopy(boxes.draft.configuration()), len(boxes._undo)
    alternative = context.series / "alternative-index.json"
    alternative.write_bytes(context.index.read_bytes())
    context.user_settings.setValue(preview_ui._INDEX, str(alternative))
    context.user_settings.sync()
    assert view.run_preview.ensure_saved_geometry_index() == str(context.index)
    assert boxes.reload(preserve_draft=True)
    _wait(context, qtbot)
    assert view.run_preview._index == str(context.index)
    assert boxes.draft.configuration() == before[0] and len(boxes._undo) == before[1]
    assert boxes.dirty and boxes.draw_button.isEnabled()


@pytest.mark.parametrize("context", ["late_index"], indirect=True)
def test_missing_saved_index_names_missing_path_and_one_click_override(context, qtbot):
    missing = context.series / "missing-index.json"
    context.user_settings.setValue(preview_ui._INDEX, str(missing))
    context.user_settings.sync()
    context.view.reload()
    _wait(context, qtbot)
    assert not context.view.analysis_boxes.draw_button.isEnabled()
    message = context.view.analysis_boxes.note.text()
    assert str(missing) in message and "Box actions > Choose measured frame models" in message
