"""Saved version selection is a bound preview, independent of the working map."""
from copy import deepcopy
from pathlib import Path
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette

from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import layer_painting_version_archive as archive
from darsia.fluidflower import painting_run_preview_ui as preview_ui

from .test_painting_photo_saturation import settings, make_view  # noqa: F401
from .test_painting_corrections import _proposal, _show
from .test_painting_manual_step4_route import _saved_manual, _select
from .test_painting_facies_display import _files
from .test_painting_chart_step2 import _fake_review


@pytest.fixture
def view(make_view):
    widget = make_view()
    yield widget
    widget.wait_for_workers()


def _ready(view, qtbot, name):
    qtbot.waitUntil(lambda: view.run_preview._version_thread is None
        and view.run_preview._last_session is not None
        and view.run_preview.last_result.source_name == name, timeout=5000)


def test_selected_manual_snapshot_remembers_exact_version_and_keeps_working_map(view, make_view, qtbot, monkeypatch):
    record = _saved_manual(view, qtbot, "V9 polygons")
    _ready(view, qtbot, "V9 polygons")
    saved, painted = manual.load_manual_version(record)
    view.session.layers[1].name = "Working draft only"
    view.session.layers[1].facies = "Working facies only"
    snapshot = deepcopy(view.session.to_dict())
    approvals, files = deepcopy(view.map_approvals), _files(view)
    view._refresh_versions()
    _select(view, record)
    _ready(view, qtbot, "V9 polygons")
    assert view.session.to_dict() == snapshot and view.map_approvals == approvals and _files(view) == files
    shown = view.run_preview.last_result
    expected = cv2.resize(painted.classes, (shown.photo.shape[1], shown.photo.shape[0]), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(shown.layers, expected)
    assert view._version_snapshot_session.to_dict() == saved.to_dict()
    assert not hasattr(view, "version_snapshot_table")
    assert view.use_map_version.currentData() == record["folder"]
    assert view.run_preview.stack.currentWidget() is view.run_preview.canvas
    view.run_preview.sources.setFocus()
    assert view._selected_version()["folder"] == record["folder"]
    palette = view.versions_table.palette()
    assert palette.color(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight) == palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight)
    assert "Selected version" in view.versions_note.text() and "read only" in view.version_snapshot_note.text()
    again = make_view()
    try:
        assert again._selected_version()["folder"] == record["folder"]
        assert again.use_map_version.currentData() == record["folder"]
        _ready(again, qtbot, "V9 polygons")
        assert again.run_preview._last_session.to_dict() == saved.to_dict()
        key = again._version_selection_key()
        original = Path.is_file
        monkeypatch.setattr(Path, "is_file", lambda path: False if path == Path(record["folder"]) / "version.json" else original(path))
        again.versions_table.blockSignals(True)
        again.versions_table.clearSelection()
        again.versions_table.blockSignals(False)
        again._refresh_versions()
        assert again._selected_version() is None
        assert "unavailable" in again.version_snapshot_note.text()
        assert again.run_preview.stack.currentIndex() == 0
        again.project.root = Path(again.project.root) / "another-project"
        assert again._version_selection_key() != key
    finally:
        again.wait_for_workers()


def test_fast_version_switch_discards_old_snapshot_and_lists_existing_drawing(view, qtbot, monkeypatch):
    _show(view, _proposal())
    first = view.save_version("First map")
    view.session.layers[1].name = "Saved drawing label"
    drawing = archive.save_drawing_version(view.project, view.session, name="Existing drawing")
    view.versions = view._read_versions(view.session.run_id)
    view._refresh_versions()
    assert any(view.versions_table.item(row, 1).text() == "drawing" for row in range(view.versions_table.rowCount()))
    before, files = deepcopy(view.session.to_dict()), _files(view)
    entered, release = threading.Event(), threading.Event()
    load = preview_ui.preview.preview_layers
    def delayed(*args, **options):
        if options.get("record", {}).get("folder") == first["folder"]:
            entered.set()
            assert release.wait(5)
        return load(*args, **options)
    monkeypatch.setattr(preview_ui.preview, "preview_layers", delayed)
    displayed = []
    metadata = view.show_version_metadata
    monkeypatch.setattr(view, "show_version_metadata", lambda session, record: (displayed.append(record["folder"]), metadata(session, record)))
    try:
        _select(view, first)
        qtbot.waitUntil(entered.is_set)
        _select(view, drawing)
        assert not hasattr(view, "version_snapshot_table")
        assert view._version_snapshot_session is None
        assert "Loading saved version Existing drawing" in view.version_snapshot_note.text()
        assert view.working_layers_note.isVisibleTo(view) and view.working_facies_note.isVisibleTo(view)
        assert view.restore_version_button.isEnabled() and view.versions_table.isEnabled()
        release.set()
        _ready(view, qtbot, "Existing drawing")
        assert displayed == [drawing["folder"]]
        assert view._version_snapshot_session.layers[1].name == "Saved drawing label"
        assert view.use_map_version.currentData() == drawing["folder"]
        assert view._selected_version()["folder"] == drawing["folder"]
        assert view.session.to_dict() == before and _files(view) == files
        view.use_map_version.setCurrentIndex(view.use_map_version.findData(first["folder"]))
        _ready(view, qtbot, 'First map')
        assert view._selected_version()['folder'] == first['folder']
        assert view.run_preview.sources.currentData()[1]["folder"] == first["folder"]
    finally:
        release.set()


def test_apply_scope_opens_existing_inline_review_and_reports_missing_inputs(view, qtbot, monkeypatch, tmp_path):
    view.set_workflow_scope("apply")
    assert "Select the saved map" in view.use_map_progress.text()
    record = _saved_manual(view, qtbot)
    view.approve_version(record)
    _ready(view, qtbot, "Selected map")
    monkeypatch.setattr(view.run_preview, "geometry_selection", lambda: SimpleNamespace(index_path=str(tmp_path / "geometry.json")))
    monkeypatch.setattr(view, "current_frame", lambda: SimpleNamespace(
        to_dict=lambda: {"fixture": "frame"}, complete=lambda: False))
    calls = _fake_review(monkeypatch)
    view.set_workflow_scope("apply")
    assert calls == [] and "No saved transfer preparation" in view.use_map_progress.text()
    view.adoption_controller._registered_gap_reviews = [dict(project_root=str(view.project.root), binding=dict(path="fixture"))]
    before = deepcopy(view.map_approvals)
    view.set_workflow_scope("apply")
    assert len(calls) == 1 and view.run_preview.stack.currentWidget() is calls[0][2]
    assert not calls[0][2].isWindow()
    view.set_workflow_scope("layers")
    view.set_workflow_scope("apply")
    assert len(calls) == 1
    assert view.map_approvals == before and view._adoption_thread is None and view._placement_thread is None
    from darsia.fluidflower import painting_chart_transfer_reader as reader
    def unavailable(*args, **kwargs):
        raise ValueError('No saved transfer review matches this selected map and series geometry.')
    monkeypatch.setattr(reader, 'selected_chart_session', unavailable)
    # The factory captured the original reader; reopen once with the injected
    # finite-reader error and ensure the status is in step 5, not another window.
    view.adoption_controller._chart_failed = True
    view.set_workflow_scope('apply')
    with pytest.raises(ValueError, match='No saved transfer review'):
        calls[-1][1]['session_factory']([])
    assert not view._transfer_review_active
    assert 'No saved transfer review matches' in view.use_map_progress.text()


def test_apply_waits_for_selected_snapshot_then_opens_review_without_another_click(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    _ready(view, qtbot, 'Selected map')
    qtbot.waitUntil(lambda: not view.busy())
    controller = view.adoption_controller
    context = dict(project=view.project, project_root=str(view.project.root), version=deepcopy(record),
                   approval=approval, frame={}, selection=SimpleNamespace(index_path=str(tmp_path / 'geometry.json')))
    monkeypatch.setattr(controller, '_findings_context_snapshot', lambda: context)
    controller._registered_gap_reviews = [dict(project_root=str(view.project.root), binding=dict(path='fixture'))]
    calls = _fake_review(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    load = preview_ui.preview.preview_layers
    def blocked(*args, **options):
        entered.set()
        if not release.wait(5):
            raise ValueError('The private test did not release its snapshot reader')
        return load(*args, **options)
    monkeypatch.setattr(preview_ui.preview, 'preview_layers', blocked)
    view.versions_table.clearSelection()
    try:
        _select(view, record)
        qtbot.waitUntil(entered.is_set)
        assert view.run_preview.busy()
        view.set_workflow_scope('apply')
        assert calls == [] and controller._automatic_chart_request is not None
        assert 'Wait for the current map operation' in view.use_map_progress.text()
        release.set()
        qtbot.waitUntil(lambda: len(calls) == 1, timeout=5000)
        assert view.run_preview._version_thread is None
        assert controller._automatic_chart_request is None
        assert view.run_preview.stack.currentWidget() is calls[0][2]
        assert not calls[0][2].isWindow()
        assert view._selected_version()['folder'] == record['folder']
        assert view._approval_for_version(record) == approval
        assert view._adoption_thread is None and view._placement_thread is None
    finally:
        release.set()
