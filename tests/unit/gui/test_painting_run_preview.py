"""Viewing another run preserves the reference drawing and approval gates."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import painting_run_preview_ui as ui
from darsia.fluidflower.painting_run_preview import RunPreview

from .test_painting_corrections import view, _proposal, _show  # noqa: F401


@pytest.fixture(autouse=True)
def private_settings(tmp_path):
    previous = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QSettings(*ui._SETTINGS).clear()
    yield
    QSettings.setDefaultFormat(previous)


def _runs(view):
    controller = view.run_preview
    controller.run_inputs = lambda: [SimpleNamespace(run_id=view.session.run_id),
                                     SimpleNamespace(run_id="ac53"), SimpleNamespace(run_id="ac54")]
    controller._index = ""
    controller.refresh()
    return controller


def _files(view):
    folder = view.project.root
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def _mock_policy_unrelated_preview(monkeypatch):
    """Synthetic lifecycle packets claim no continuation/source qualification.

    The real frame/photo/version gates are exercised by the actual preview and
    photo-continuation tests. This seam changes only metadata construction;
    worker dispatch, callbacks, GUI rendering and retirement remain real.
    """
    from darsia.fluidflower import photo_edge_continuation
    monkeypatch.setattr(photo_edge_continuation, "build_contracts",
        lambda project, index, source, runs, ids, **kwargs: {run: None for run in runs})


def test_one_chooser_keeps_reference_editable_and_other_run_read_only(view):
    controller = _runs(view)
    before = deepcopy(view.session), _files(view)
    assert "reference" in controller.runs.currentText()
    assert controller.is_reference() and view.steps_scroll.isEnabled()
    controller.runs.setCurrentIndex(1)
    assert not controller.is_reference()
    assert controller.stack.currentWidget() is controller.canvas
    assert view.steps_scroll.isEnabled()
    assert view.layer_all_on_button.isEnabled() and view.facies_all_on_button.isEnabled()
    assert all(not button.isEnabled() for name, button in view.tool_buttons.items() if name != "pan")
    assert "Frame models" in controller.note.text()
    assert view.session == before[0] and _files(view) == before[1]
    assert all(not s.isEnabled() for s in view.findChildren(QShortcut) if s not in controller.navigation_keys)
    controller.runs.setCurrentIndex(0)
    assert controller.stack.currentIndex() == 0 and view.steps_scroll.isEnabled()
    assert view.session == before[0]


def test_preview_selection_does_not_set_reference_but_explicit_action_names_original(view, monkeypatch):
    controller = _runs(view)
    chosen, asked = [], []
    controller.set_reference = chosen.append
    controller.runs.setCurrentIndex(1)
    assert chosen == []
    def confirm(*args):
        asked.append(args[2])
        return QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QMessageBox, "question", confirm)
    controller.change_reference()
    assert chosen == ["ac53"]
    assert view.session.run_id in asked[0] and "ac53" in asked[0]
    assert "no marks are copied" in asked[0]


def test_review_order_is_runtime_data_and_previous_next_do_not_write(view):
    settings = QSettings(*ui._SETTINGS)
    settings.setValue(ui._FIRST, "ac54, ac53")
    controller = _runs(view)
    assert [controller.runs.itemData(i) for i in range(3)] == [view.session.run_id, "ac54", "ac53"]
    before = _files(view), deepcopy(view.session)
    controller.next.click()
    assert controller.runs.currentData() == "ac54"
    controller.next.click()
    assert controller.runs.currentData() == "ac53"
    controller.previous.click()
    assert controller.runs.currentData() == "ac54"
    assert _files(view) == before[0] and view.session == before[1]


def test_other_run_disables_undo_without_changing_reference_stack(view):
    controller = _runs(view)
    view.add_polygon(lp.Polygon(2, [(5, 5), (5, 20), (20, 20)]))
    before = deepcopy(view.session), deepcopy(view._undo_stack)
    controller.runs.setCurrentIndex(1)
    view.undo()
    view.redo()
    assert view.session == before[0] and view._undo_stack == before[1]
    controller.runs.setCurrentIndex(0)
    view.undo()
    assert not view.session.polygons


def test_worker_uses_exact_snapshot_and_retires_in_gui_thread(view, qtbot, monkeypatch):
    _mock_policy_unrelated_preview(monkeypatch)
    controller = _runs(view)
    view.session.polygons = [lp.Polygon(2, [(5.1234567, 5.7654321), (5., 20.), (20., 20.)])]
    exact = deepcopy(view.session)
    before_files = _files(view)
    options_seen, sessions_seen = [], []
    def layers(session, proposal, corrections, frame, **kwargs):
        sessions_seen.append(deepcopy(session))
        return session, np.ones((60, 80), np.uint8)
    def prepare(session, values, **kwargs):
        options_seen.append(kwargs)
        return RunPreview("ac53", np.full((60, 80, 3), 125, np.uint8), values,
                          np.zeros(values.shape, bool), "target.png", session.run_id, "Current drawing")
    monkeypatch.setattr(ui.preview, "preview_layers", layers)
    monkeypatch.setattr(ui.preview, "prepare_preview", prepare)
    controller._index = str(view.project.root / "preview-geometry-index.json")
    controller.runs.setCurrentIndex(controller.runs.findData("ac53"))
    worker = controller.worker
    assert worker is not None
    qtbot.waitUntil(lambda: controller.thread is None, timeout=5000)
    assert controller.worker is None
    assert controller.last_result is not None, (controller.note.text(), worker.options)
    assert controller.last_result.run_id == "ac53"
    assert sessions_seen[0].polygons == exact.polygons
    assert options_seen[0]["measure_offsets"] is True
    assert options_seen[0]["config"] == view._workspace_frame_config
    assert _files(view) == before_files and view.session == exact
    assert "Read only" in controller.note.text()


def test_logged_placement_still_requires_approval_while_preview_is_available(view):
    _show(view, _proposal())
    record = view.save_version("Draft")
    controller = _runs(view)
    assert any(value is not None and value.get("sha256") == record["sha256"]
               and value.get("folder") == record["folder"]
               for value in (controller.sources.itemData(i)[1] for i in range(controller.sources.count())))
    assert not view.place_version(record, "unused.json", runs=["ac53"])
    assert "Approve" in view.versions_note.text()
    assert not view.map_approvals


def test_actual_preview_worker_places_drawing_on_own_photo_without_writes(view, qtbot):
    import hashlib
    import json
    import os
    from pathlib import Path

    from PIL import Image

    from darsia.fluidflower import map_placement
    from darsia.fluidflower.series_geometry import RigFrame

    controller = _runs(view)
    source_run, target_run = view.session.run_id, "ac53"
    folder = view.project.root / "preview-fixture"
    folder.mkdir(parents=True)
    target_photo = folder / "target.png"
    Image.new("RGB", (80, 60), (160, 110, 70)).save(target_photo)

    def descriptor(path):
        content = path.read_bytes()
        return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content)}

    entries = {}
    for run, photo, shift in [(source_run, Path(view.session.image_path), 0),
                               (target_run, target_photo, 3)]:
        model = {"format": map_placement.GEOMETRY_FORMAT, "run_id": run,
                 "reference_image": descriptor(photo), "shape_hw": [60, 80],
                 "H_metric_mm_to_camera_u": [[1., 0., float(shift)], [0., 1., 0.], [0., 0., 1.]]}
        path = folder / (run + "-model.json")
        path.write_text(json.dumps(model), encoding="utf-8")
        entries[run] = {"reference_model": descriptor(path)}
    index = folder / "index.json"
    index.write_text(json.dumps({"format": map_placement.GEOMETRY_FORMAT, "runs": entries}),
                     encoding="utf-8")
    # Supply real private own-frame authority for the actual positive policy.
    # It must be on disk, independently of the in-memory reference getter.
    (folder / "config/runs").mkdir(parents=True)
    (folder / "config/common.toml").write_text("# private own-frame fixtures\n", encoding="utf-8")
    (folder / "series.toml").write_text(
        f'schema_version = 1\nseries_id = "private-preview"\n[runs]\nids = ["{source_run}", "{target_run}"]\n',
        encoding="utf-8")
    own_frames = {}
    for run, photo in ((source_run, Path(view.session.image_path)), (target_run, target_photo)):
        (folder / "config/runs" / (run + ".toml")).write_text(
            f'[data]\nbaseline = "{photo.as_posix()}"\n', encoding="utf-8")
        frame = RigFrame(
            corners=dict(top_left=(0., 0.), top_right=(0., 79.),
                         bottom_right=(59., 79.), bottom_left=(59., 0.)),
            width_m=.079, height_m=.059, run_id=run,
            source="private saved frame", image=str(photo), image_size=(80, 60))
        own_frames[run] = frame.to_dict()
    view.project.manifest.setdefault("tools", {})["native_series"] = dict(
        series_root=str(folder), series_id="private-preview", rig_frames=own_frames)
    (view.project.root / "series.json").write_text(json.dumps(view.project.manifest), encoding="utf-8")
    view.session.polygons = [
        lp.Polygon(2, [(0, 0), (0, 39), (59, 39), (59, 0)]),
        lp.Polygon(3, [(0, 40), (0, 79), (59, 79), (59, 40)]),
    ]
    # Flush the fixture's drawing before measuring the read-only operation;
    # no delayed autosave or manual redraw may write during that snapshot.
    view._dirty = True
    view.save_session()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    qtbot.waitUntil(lambda: not view.busy() and view._manual_thread is None, timeout=10000)
    drawing_choice = next(i for i in range(controller.sources.count())
                          if controller.sources.itemData(i) == ("drawing", None))
    controller.sources.setCurrentIndex(drawing_choice)
    qtbot.waitUntil(lambda: not view.busy() and view._manual_thread is None, timeout=10000)
    controller._index = str(index)
    before_files = _files(view)
    before_session, before_approvals = deepcopy(view.session), deepcopy(view.map_approvals)
    before_photo = Path(view.session.image_path).read_bytes()
    assert not before_approvals
    assert controller.thread is None and controller.worker is None
    controller.runs.setCurrentIndex(controller.runs.findData(target_run))
    assert controller.worker is not None, controller.note.text()
    # The accepted packet also rebinds boxes through a read-only metadata worker.
    qtbot.waitUntil(lambda: controller.thread is None and not view.busy(), timeout=15000)
    result = controller.last_result
    assert result is not None, controller.note.text()
    expected = np.full((60, 80), 2, np.uint8)
    expected[:, 40:] = 3
    assert np.array_equal(result.layers[:, 3:], expected[:, :-3])
    assert result.layers[30, 20] == 2 and result.layers[30, 43] == 3
    assert result.uncovered[:, :3].all() and not result.uncovered[:, 3:].any()
    assert result.photo_path == str(target_photo)
    assert np.array_equal(result.photo, np.full((60, 80, 3), (160, 110, 70), np.uint8))
    assert result.measurements["layer_offsets"]["method"] == "geometry_offset.measure_arrays"
    assert result.measurements["offset_px"] == {"median_px": 3., "max_px": 3.}
    assert controller.measurements.rowCount() == len(result.measurements["layer_offsets"]["boundaries"])
    assert controller.measurements.rowCount() > 0
    assert controller.worker is None and controller.thread is None and not view.busy()
    assert view.session == before_session and view.map_approvals == before_approvals
    assert Path(view.session.image_path).read_bytes() == before_photo
    assert _files(view) == before_files
    assert not list(view.project.root.rglob("approvals.jsonl"))
    assert "Read only" in controller.note.text()
    visual = os.environ.get("PAINTING_PREVIEW_VISUAL")
    if visual:
        view.resize(1500, 1000)
        qtbot.waitExposed(view)
        assert view.grab().save(visual)
