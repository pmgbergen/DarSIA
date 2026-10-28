"""Private measured-box GUI drafts: exact frames, one undo, explicit legacy choice."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import toml
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt, QThread
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QButtonGroup, QGraphicsScene, QGraphicsView, QMessageBox, QToolButton, QVBoxLayout, QWidget,
)
from shiboken6 import isValid

from darsia.fluidflower import metric_analysis_domains as domains
from darsia.fluidflower import painting_analysis_boxes_ui as ui
from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT
from darsia.fluidflower.map_placement import GEOMETRY_FORMAT, read_run_geometry
from .test_painting_analysis_boxes_integration import context as painting_view  # noqa: F401


class _Canvas(QGraphicsView):
    def __init__(self, parent):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.scene().setSceneRect(0., 0., 300., 200.)
        self.tool = "pan"
        self.place_mode = False
        self.raw_size = (300, 200)
        self.preview = {}

    def raw_to_scene(self, row, col):
        return QPointF(col, row)

    def scene_to_raw(self, point):
        return point.y(), point.x()

    def set_tool(self, tool):
        self.tool = tool


@pytest.fixture
def context(qtbot, tmp_path):
    """Real store bindings, wholly within the test's private temporary project."""
    project = SimpleNamespace(root=tmp_path / "project", manifest={"created_by": "reviewer"},
                              _audit=lambda *args: None)
    current = project.root / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(dict(format=GEOMETRY_REVISION_FORMAT, revision_id="geometry-1")))
    photo = tmp_path / "reference.png"
    photo.write_bytes(b"identity only; this controller does not load pixels")
    matrix = [[1., .13, 10.], [.07, -1., 180.], [.0002, .0001, 1.]]
    model = tmp_path / "model.json"
    model.write_text(json.dumps(dict(format=GEOMETRY_FORMAT, run_id="run_a",
        H_metric_mm_to_camera_u=matrix, reference_image=dict(path=str(photo)), shape_hw=[200, 300],
        metric_frame=dict(unit="mm", x_axis="right", y_axis="up", measured_width_mm=210.))))
    index = tmp_path / "index.json"
    index.write_text(json.dumps(dict(format=GEOMETRY_FORMAT,
        runs={"run_a": dict(reference_model=domains.descriptor(model))})))
    config = domains.new_configuration(project, index)
    models = read_run_geometry(index)
    view = QWidget()
    qtbot.addWidget(view)
    view.project = project
    view.session = SimpleNamespace(run_id="run_a", image_path=str(photo), image_size=(300, 200), polygons=[])
    view.image = SimpleNamespace(image_size=(300, 200))
    view.run_preview = SimpleNamespace(_index=str(index), is_reference=lambda: True)
    view._workspace_frame_config = {}
    view._other_photo = None
    view._background_token, view._background_shown = None, False
    view.busy = lambda: False
    view._reviewer = lambda: "private-test-reviewer"
    view.canvas = _Canvas(view)
    view._reference_token = view.canvas.preview
    view.tool_group = QButtonGroup(view)
    view.tool_group.setExclusive(True)
    view.tool_buttons = {}
    for tool in ("pan", "polygon", "edit"):
        button = QToolButton(view)
        button.setCheckable(True)
        button.clicked.connect(lambda checked=False, tool=tool: view.canvas.set_tool(tool))
        view.tool_group.addButton(button)
        view.tool_buttons[tool] = button
    view.tool_buttons["pan"].setChecked(True)
    view.private_fill_button = QToolButton(view)
    view.private_fill_button.setCheckable(True)
    view.tool_group.addButton(view.private_fill_button)
    view.private_fill_active = False
    def fill_toggled(checked):
        view.private_fill_active = checked
        if checked:
            view.canvas.set_tool("pan")
    view.private_fill_button.toggled.connect(fill_toggled)
    layout = QVBoxLayout(view)
    layout.addWidget(view.canvas)
    controller = ui.PaintingAnalysisBoxesController(view, layout)
    view.resize(900, 700)
    value = SimpleNamespace(project=project, index=index, config=config, models=models,
        photo=photo, view=view, controller=controller, matrix=np.asarray(matrix))
    yield value
    controller.wait_for_workers()


def _raw(context, x, y):
    p = context.matrix @ np.array([x, y, 1.])
    return float(p[1]/p[2]), float(p[0]/p[2])


def _install(context, rows=(), **kwargs):
    config = deepcopy(context.config)
    config["domains"] = list(rows)
    context.controller.set_context(config, context.models, **kwargs)
    context.controller.snap_enabled.setChecked(False)
    return config


def test_relocated_saved_boxes_reload_display_and_keep_sidebar_outputs(context, tmp_path, monkeypatch, qtbot):
    from darsia.fluidflower import painting_run_preview as preview
    from darsia.fluidflower import workflow_row_status as status
    from darsia.fluidflower.storage_paths import refresh_storage_paths

    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    project, controller = context.project, context.controller
    old = project.root
    current = old / "overrides/geometry_facies/current.json"
    artifacts = {}
    for name in ("labels.npz", "facies.npz", "props.csv"):
        path = old / "saved" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"saved output; metadata-only test")
        artifacts[name] = str(path)
    current.write_text(json.dumps(dict(schema=1, format=GEOMETRY_REVISION_FORMAT,
        revision_id="a" * 32, configuration=dict(labels_path=artifacts["labels.npz"]),
        artifacts=dict(facies_path=artifacts["facies.npz"], facies_properties_path=artifacts["props.csv"]))))
    config = domains.new_configuration(project, context.index)
    config["domains"] = [domains.box("Lower", [1., 2., 1., 2.], role="box1", domain_id="lower"),
                         domains.box("Upper", [3., 4., 3., 4.], role="box2", domain_id="upper")]
    store = domains.MetricAnalysisDomainStore(project)
    revision = store.save_revision(config, actor="test", reason="Saved boxes")
    revision_path = store.artifact_root / revision["revision_id"] / "revision.json"
    controller.set_context(config, context.models, store=store)
    expected_mass_boxes = domains.saved_mass_boxes(revision_path)[0]
    frozen = {path.relative_to(old): path.read_bytes() for path in old.rglob("*") if path.is_file()}
    extras = {path: path.read_bytes() for path in (context.index, tmp_path / "model.json", context.photo)}
    photo_binding = domains.descriptor(context.photo)

    series, moved = tmp_path / "DarSIA", tmp_path / "Results/setup/Geometry"
    series.mkdir()
    moved.parent.mkdir(parents=True)
    old.rename(moved)
    frame_folder = moved / "frame_models"
    frame_folder.mkdir()
    aliases = {str(old): str(moved)}
    for path in extras:
        destination = frame_folder / path.name
        path.rename(destination)
        aliases[str(path)] = str(destination)
    run_config = series / "run.toml"
    run_config.write_text('run="run_a"\n', encoding="utf-8")
    (series / "series.toml").write_text(toml.dumps(dict(schema_version=1, series_id="sample",
        paths=dict(geometry_facies=str(moved)), runs=dict(ids=["run_a"], config_files=dict(run_a="run.toml")),
        storage=dict(path_aliases=aliases))), encoding="utf-8")
    refresh_storage_paths(series)
    project.root = moved

    row = status.row_status(status.load_status_context(series, "run_a", run_config), "geometry_facies", "layers")
    assert row.state == "done", row.missing
    assert all(Path(item.path).is_relative_to(moved) for item in row.artifacts)
    assert domains.saved_mass_boxes(revision_path)[0] == expected_mass_boxes
    assert domains.validate_configuration(project, config)["valid"]
    assert controller.reload()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=10000)
    assert not controller._binding_error, controller.note.text()
    assert controller.table.rowCount() == 2
    assert controller.draft.configuration() == config
    assert controller.preview_geometry_problem(controller.geometry_binding, source_run="run_a", target_run="run_a") == ""

    moved_photo = frame_folder / context.photo.name
    assert domains._verify(photo_binding) == moved_photo
    model = controller.models["run_a"]
    assert preview._photo_geometry(model, {}, moved_photo, (200, 300)) is model
    other = tmp_path / "different-photo.png"
    other.write_bytes(extras[context.photo])
    with pytest.raises(ValueError, match="does not bind"):
        preview._photo_geometry(model, {}, other, (200, 300))
    assert {name: (moved / name).read_bytes() for name in frozen} == frozen
    assert all((frame_folder / path.name).read_bytes() == raw for path, raw in extras.items())
    (frame_folder / context.index.name).write_bytes(extras[context.index] + b" ")
    assert not domains.validate_configuration(project, config)["valid"]


def test_explicit_current_frame_action_keeps_boxes_then_saves_and_reopens(context, tmp_path, monkeypatch, qtbot):
    controller = context.controller
    config = deepcopy(context.config)
    config["domains"] = [domains.box("Lower", [0., 900., 0., 150.4], role="box1", domain_id="lower"),
                         domains.box("Upper", [0., 900., 150.4, 490.], role="box2", domain_id="upper")]
    store = domains.MetricAnalysisDomainStore(context.project)
    old = store.save_revision(config, actor="test", reason="Original measured boxes")
    old_revision = store.artifact_root / old["revision_id"] / "revision.json"
    original_bytes = old_revision.read_bytes()
    geometry = context.project.root / "overrides/geometry_facies/current.json"
    geometry.write_text(json.dumps(dict(format=GEOMETRY_REVISION_FORMAT, revision_id="new-geometry")))
    selected = tmp_path / "current-frame-index.json"
    selected.write_text(json.dumps(dict(json.loads(context.index.read_text()), label="Current measured frame")))
    context.view.run_preview._index = str(selected)
    assert controller.reload()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    assert controller._binding_error and not controller.draw_button.isEnabled()
    assert controller.table.rowCount() == 2 and controller.use_current_frame_button.isEnabled()
    assert store.current()["configuration"] == config

    controller.use_current_frame_button.click()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    draft = controller.draft.configuration()
    assert not controller._binding_error, controller.note.text()
    assert controller.dirty and controller.save_button.isEnabled()
    assert controller.table.rowCount() == 2 and controller.geometry is not None and controller._items
    assert draft["domains"] == config["domains"]
    assert draft["geometry_revision_id"] == "new-geometry"
    assert draft["geometry_index"] == domains.descriptor(selected)
    assert store.current()["configuration"] == config
    assert old_revision.read_bytes() == original_bytes

    monkeypatch.setattr(ui.QInputDialog, "getText", lambda *args: ("Use current measured frame", True))
    controller.save_button.click()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    assert not controller.dirty, controller.note.text()
    assert store.current()["configuration"] == draft
    assert store.current()["revision_id"] != old["revision_id"]
    assert controller.reload()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    assert controller.draft.configuration() == draft and not controller._binding_error
    assert old_revision.read_bytes() == original_bytes


def test_current_frame_draft_survives_actual_painting_view_refresh(painting_view, qtbot):
    context = painting_view
    view, controller = context.view, context.view.analysis_boxes
    config = controller.draft.configuration()
    old = controller.store.save_revision(config, actor="test", reason="Original boxes")
    old_path = controller.store.artifact_root / old["revision_id"] / "revision.json"
    original_bytes = old_path.read_bytes()
    controller.dirty = False
    controller._saved_configuration = deepcopy(config)
    context.current.write_text(json.dumps(dict(format=GEOMETRY_REVISION_FORMAT, revision_id="new-geometry")))
    selected = context.series / "current-frame-index.json"
    selected.write_text(json.dumps(dict(json.loads(context.index.read_text()), label="Current measured frame")))
    view.run_preview._index = str(selected)
    view._sync_analysis_boxes(force=True)
    qtbot.waitUntil(lambda: not view.busy() and not view.run_preview.busy(), timeout=10000)
    assert controller._binding_error
    controller.use_current_frame_button.click()
    qtbot.waitUntil(lambda: not view.busy() and not view.run_preview.busy(), timeout=10000)
    assert controller.dirty and not controller._binding_error, controller.note.text()
    assert controller.geometry is not None and controller.table.rowCount() == 2
    assert controller.draft.configuration()["domains"] == config["domains"]
    assert old_path.read_bytes() == original_bytes


@pytest.mark.parametrize("changed", ["run_order", "image_correction_revision_id"])
def test_explicit_current_frame_action_keeps_blocking_incompatible_saved_boxes(context, tmp_path, qtbot, changed):
    config = deepcopy(context.config)
    config["domains"] = [domains.box("Saved", [10., 20., 10., 20.], role="box1", domain_id="saved")]
    store = domains.MetricAnalysisDomainStore(context.project)
    old = store.save_revision(config, actor="test", reason="Original measured box")
    old_revision = store.artifact_root / old["revision_id"] / "revision.json"
    original_bytes = old_revision.read_bytes()
    if changed == "run_order":
        value = json.loads(context.index.read_text())
        value["runs"]["run_b"] = deepcopy(value["runs"]["run_a"])
        selected = tmp_path / "different-cohort.json"
        selected.write_text(json.dumps(value))
        context.view.run_preview._index = str(selected)
    else:
        from darsia.fluidflower.image_correction import CorrectionRevisionStore
        path = CorrectionRevisionStore(context.project).current_path
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(revision_id="new-correction")))
    controller = context.controller
    assert controller.reload()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    assert controller._binding_error
    controller.use_current_frame_button.click()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    assert controller._binding_error and changed in controller.note.text()
    assert controller.draft.configuration() == config and not controller.dirty
    assert not controller.save_button.isEnabled()
    assert old_revision.read_bytes() == original_bytes


def _row(controller, domain_id):
    return next(row for row in controller.draft.configuration()["domains"] if row["domain_id"] == domain_id)


def _mouse(controller, kind, point_rc, button=Qt.MouseButton.NoButton):
    """Deliver viewport positions through the actual controller event-filter route."""
    canvas = controller.view.canvas
    viewport_point = canvas.mapFromScene(canvas.raw_to_scene(*point_rc))
    event = QMouseEvent(kind, QPointF(viewport_point),
                        QPointF(canvas.viewport().mapToGlobal(viewport_point)),
                        button, button if kind != QEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier)
    return controller.eventFilter(canvas.viewport(), event)


def test_construction_is_passive_and_default_legacy_source_is_absent(context, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Construction must not read a store or select legacy inputs")
    monkeypatch.setattr(domains, "MetricAnalysisDomainStore", forbidden)
    second = ui.PaintingAnalysisBoxesController(context.view)
    assert second.legacy_source_provider is None
    assert second.group is second.widget
    assert second.draft is None and not second.busy()
    assert not second.draw_button.isEnabled()
    assert "Import original box1 / box2" in second.menu.actions()[0].text()
    assert "do not apply to boxes" in second.widget.toolTip()


@pytest.mark.parametrize("scope", ["rig_frame", "layers", "apply"])
def test_unrelated_scope_removes_only_box_graphics_and_restores_analysis_draft(context, scope):
    controller, view = context.controller, context.view
    config = _install(context, [domains.box("Saved box", [10., 70., 20., 90.], domain_id="saved")])
    view._workflow_scope = "analysis_boxes"
    controller.refresh()
    assert controller._items and controller.set_mode("edit")
    scene = view.canvas.scene()
    reference_quad = scene.addRect(5., 5., 100., 80.)
    injection_point = scene.addEllipse(45., 35., 3., 3.)
    owned = tuple(controller._items)
    before = {path: path.read_bytes() for path in context.project.root.rglob("*") if path.is_file()}
    view._workflow_scope = scope
    controller.refresh()
    assert controller.mode is None and not controller._items
    assert not controller.draw_button.isEnabled() and not controller.edit_button.isEnabled()
    assert not controller.set_mode("draw")
    assert all(isValid(item) and item.scene() is None for item in owned)
    assert reference_quad.scene() is scene and injection_point.scene() is scene
    assert controller.draft.configuration() == config and not controller.dirty
    assert {path: path.read_bytes() for path in before} == before
    view._workflow_scope = "analysis_boxes"
    controller.refresh()
    assert controller._items and controller.draw_button.isEnabled()
    assert controller.draft.configuration() == config


def test_missing_model_keeps_saved_rows_readable_but_disables_tools(context):
    row = domains.box("Saved exact box", [10.123456789012345, 70., 20., 90.], domain_id="saved")
    config = deepcopy(context.config)
    config["domains"] = [row]
    context.controller.set_context(config, {})
    controller = context.controller
    assert controller.table.rowCount() == 1
    assert controller.table.item(0, 1).text() == row["name"]
    assert controller.draft.configuration()["domains"] == [row]
    assert "no measured frame model" in controller.note.text()
    assert not controller.draw_button.isEnabled()
    assert not controller.set_mode("draw")
    assert not controller.create_rectangle((1., 2.), (20., 30.))


def test_rectangle_on_perspective_photo_is_measured_and_never_changes_layer_polygons(context):
    _install(context)
    polygons = [SimpleNamespace(points=[(1.123456789, 2.), (3., 20.), (15., 3.)])]
    context.view.session.polygons = polygons
    before = deepcopy(polygons)
    controller = context.controller
    assert controller.create_rectangle(_raw(context, 10., 20.), _raw(context, 70., 90.), name="Measured")
    row = controller.draft.configuration()["domains"][0]
    np.testing.assert_allclose(row["bounds_mm"], [10., 70., 20., 90.], rtol=0, atol=1e-10)
    assert context.view.session.polygons == before
    assert len(controller._undo) == 1
    outline = controller.passive_outlines(context.models["run_a"], photo_path=str(context.photo), shape=(200, 300))[0]
    actual = np.asarray(outline["points_rc"])
    expected = domains.project_box(row, context.models["run_a"])[:, ::-1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    # The exact projected quadrilateral is not replaced by its camera-grid bounds.
    assert actual[0, 1] != actual[3, 1]
    assert controller.undo()
    assert controller.draft.configuration()["domains"] == []
    assert not controller.dirty
    assert controller.redo()
    assert controller.draft.configuration()["domains"] == [row]
    assert controller.dirty


def test_shared_side_move_is_one_atomic_undo_and_invalid_inversion_is_refused(context):
    left = domains.box("Left", [10., 70., 20., 90.], domain_id="left", role="box1")
    right = domains.box("Right", [70., 150., 20., 90.], domain_id="right", role="box2")
    _install(context, [left, right])
    controller = context.controller
    assert controller.edit_side("left", "x_max", _raw(context, 80., 40.))
    assert _row(controller, "left")["bounds_mm"][1] == _row(controller, "right")["bounds_mm"][0]
    assert _row(controller, "left")["bounds_mm"][1] == pytest.approx(80.)
    assert len(controller._undo) == 1
    moved = controller.draft.configuration()
    assert not controller.edit_side("left", "x_max", _raw(context, 160., 40.))
    assert controller.draft.configuration() == moved
    assert len(controller._undo) == 1
    assert controller.undo()
    assert controller.draft.configuration()["domains"] == [left, right]
    assert controller.redo()
    assert controller.draft.configuration() == moved


@pytest.mark.parametrize("target", ["box", "layer"])
def test_side_snaps_to_measured_box_side_or_nearest_layer_segment_xy(context, target):
    rows = [domains.box("Edit", [10., 70., 20., 90.], domain_id="edit")]
    if target == "box":
        rows.append(domains.box("Guide", [80., 150., 20., 90.], domain_id="guide"))
    else:
        context.view.session.polygons = [SimpleNamespace(points=[_raw(context, x, y)
            for x, y in ((80., 10.), (80., 160.), (120., 160.), (120., 10.))])]
    _install(context, rows)
    controller = context.controller
    controller.snap_enabled.setChecked(True)
    controller.snap_distance.setValue(2.)
    guides_before = deepcopy(context.view.session.polygons)
    assert controller.edit_side("edit", "x_max", _raw(context, 79.25, 45.))
    assert _row(controller, "edit")["bounds_mm"][1] == pytest.approx(80., abs=1e-10)
    assert context.view.session.polygons == guides_before
    assert len(controller._undo) == 1


def test_table_metadata_edit_does_not_round_any_coordinate(context):
    row = domains.box("Precise", [10.123456789012345, 70.98765432109876, 20.1, 90.2], domain_id="exact")
    _install(context, [row])
    controller = context.controller
    assert controller.table.item(0, 2).text() != str(row["bounds_mm"][0])
    controller.table.item(0, 1).setText("Renamed")
    assert _row(controller, "exact")["bounds_mm"] == row["bounds_mm"]
    assert _row(controller, "exact")["name"] == "Renamed"
    assert controller.undo()
    assert controller.draft.configuration()["domains"] == [row]


def test_legacy_missing_roles_pending_conflicts_retained_and_review_preserves_original_values(context, monkeypatch):
    values = dict(box1=[[.01, .09], [.07, .02]], box2={"roi": [[.08, .03], [.16, .10]]})
    existing = domains.box("Existing role", [11., 70., 20., 90.], role="box1", domain_id="existing")
    source = dict(path="actual-input.toml", sha256="f"*64, size_bytes=123)
    before = deepcopy(values)
    _install(context, [existing], legacy=dict(values=values, source=source))
    controller = context.controller
    rows = controller.draft.configuration()["domains"]
    assert rows[0] == existing
    assert len(rows) == 2 and rows[1]["role"] == "box2"
    assert rows[1]["legacy_import"]["source"] == source
    assert rows[1]["legacy_import"]["original_value"] == values["box2"]
    assert rows[1]["legacy_import"]["decision"] == "pending"
    assert set(controller.legacy_conflicts) == {"box1"}
    with pytest.raises(ValueError, match="Review imported"):
        controller.draft.mass_boxes()
    controller.table.selectAll()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    assert controller.review_legacy()
    reviewed = controller.draft.configuration()["domains"]
    assert reviewed[1]["bounds_mm"] == rows[1]["bounds_mm"]
    assert reviewed[1]["legacy_import"]["decision"] == "use_coordinates_in_measured_frame"
    assert controller.draft.mass_boxes()["box2"] == values["box2"]
    assert len(controller._undo) == 1
    assert controller.undo()
    assert controller.draft.configuration()["domains"] == rows
    assert values == before


def test_legacy_frame_question_no_keeps_pending_gate(context, monkeypatch):
    values = dict(box1=[[.01, .02], [.07, .09]], box2=[[.08, .03], [.16, .10]])
    _install(context, legacy=dict(values=values, source={"receipt": "private-source"}))
    controller = context.controller
    before = controller.draft.configuration()
    controller.table.selectAll()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    assert not controller.review_legacy()
    assert controller.draft.configuration() == before
    assert not controller._undo
    with pytest.raises(ValueError, match="Review imported"):
        controller.draft.mass_boxes()


def test_reload_calls_only_supplied_source_off_gui_and_creates_no_revision(context, qtbot):
    controller = context.controller
    calls = []
    values = dict(box1=[[.01, .02], [.07, .09]], box2=[[.08, .03], [.16, .10]])
    def actual_source(project, run):
        calls.append((project, run, QThread.currentThread() is controller.thread()))
        return dict(values=values, source={"receipt": "private-original-input"})
    controller.legacy_source_provider = actual_source
    before = sorted(path.relative_to(context.project.root) for path in context.project.root.rglob("*") if path.is_file())
    assert controller.reload()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    assert calls == [(context.project, "run_a", False)]
    assert not controller._binding_error
    assert {row["role"] for row in controller.draft.configuration()["domains"]} == {"box1", "box2"}
    assert controller.dirty
    assert sorted(path.relative_to(context.project.root) for path in context.project.root.rglob("*") if path.is_file()) == before
    assert not (context.project.root / "results/analysis_domains/measured_frame").exists()


def test_explicit_save_uses_existing_store_off_gui_and_preserves_layer_session(context, qtbot, monkeypatch):
    store = domains.MetricAnalysisDomainStore(context.project)
    first = domains.box("First", [10.123456789012345, 70., 20., 90.], domain_id="first", role="box1")
    second = domains.box("Second", [80., 150., 30., 100.], domain_id="second", role="box2")
    _install(context, [first, second], store=store)
    controller = context.controller
    context.view.session.polygons = [SimpleNamespace(points=[(1., 2.), (3., 20.), (15., 3.)])]
    polygons_before = deepcopy(context.view.session.polygons)
    controller.table.item(0, 1).setText("Reviewed first")
    candidate = controller.draft.configuration()
    calls, saved = [], []
    writer = store.save_revision
    def observed(config, **kwargs):
        calls.append(QThread.currentThread() is controller.thread())
        return writer(config, **kwargs)
    monkeypatch.setattr(store, "save_revision", observed)
    monkeypatch.setattr(ui.QInputDialog, "getText", lambda *args: ("Private measured-box review", True))
    controller.saved.connect(saved.append)
    assert controller.save_revision()
    assert not controller.can_close()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    assert calls == [False]
    assert len(saved) == 1
    assert store.current()["configuration"] == candidate
    revision_file = store.artifact_root / saved[0]["revision_id"] / "revision.json"
    assert domains.saved_mass_boxes(revision_file)[0] == domains.mass_boxes(candidate)
    assert not controller.dirty and controller.can_close()
    assert context.view.session.polygons == polygons_before
    assert candidate["domains"][0]["bounds_mm"] == first["bounds_mm"]
    assert not (context.project.root / "overrides/analysis_domains/current.json").exists()


def test_provider_failure_and_reference_change_never_install_stale_loaded_boxes(context, qtbot):
    _install(context, [domains.box("Current", [10., 70., 20., 90.], domain_id="current")])
    controller = context.controller
    before = controller.draft.configuration()
    def failed_source(*args):
        raise ValueError("original source binding is ambiguous")
    controller.legacy_source_provider = failed_source
    assert controller.reload()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    assert "source binding is ambiguous" in controller.note.text()
    assert controller.draft.configuration() == before
    started, release = threading.Event(), threading.Event()
    def delayed_source(*args):
        started.set()
        if not release.wait(5.):
            raise ValueError("private test source was not released")
        return None
    controller.legacy_source_provider = delayed_source
    try:
        assert controller.reload()
        qtbot.waitUntil(started.is_set, timeout=5000)
        context.view.session.image_path = str(context.photo.parent / "another-photo.png")
    finally:
        release.set()
    qtbot.waitUntil(lambda: not controller.busy(), timeout=5000)
    assert controller.draft.configuration() == before
    assert "reference changed" in controller.note.text().lower()
    assert not controller.draw_button.isEnabled()


def test_mouse_drawing_previews_and_release_position_is_authoritative(context):
    _install(context)
    controller = context.controller
    assert controller.set_mode("draw")
    _mouse(controller, QEvent.Type.MouseButtonPress, _raw(context, 20., 30.), Qt.MouseButton.LeftButton)
    _mouse(controller, QEvent.Type.MouseMove, _raw(context, 50., 70.))
    assert controller.draft.configuration()["domains"] == []
    assert len(controller._drag["preview"].outlines()) == 1
    assert controller._items
    # The release has another coordinate, with no intervening move event.
    endpoint = _raw(context, 80., 100.)
    canvas = context.view.canvas
    final_raw = canvas.scene_to_raw(canvas.mapToScene(canvas.mapFromScene(canvas.raw_to_scene(*endpoint))))
    start_raw = controller._drag["raw"]
    first = np.linalg.inv(context.matrix) @ np.array([start_raw[1], start_raw[0], 1.])
    last = np.linalg.inv(context.matrix) @ np.array([final_raw[1], final_raw[0], 1.])
    first, last = first[:2]/first[2], last[:2]/last[2]
    _mouse(controller, QEvent.Type.MouseButtonRelease, endpoint, Qt.MouseButton.LeftButton)
    bounds = controller.draft.configuration()["domains"][0]["bounds_mm"]
    np.testing.assert_allclose(bounds, [first[0], last[0], first[1], last[1]], rtol=0, atol=1e-10)
    assert len(controller._undo) == 1


def test_invalid_release_discards_valid_resize_preview_and_preserves_exact_click(context):
    row = domains.box("Edit", [10.123456789012345, 70., 20., 90.], domain_id="edit")
    _install(context, [row])
    controller = context.controller
    assert controller.set_mode("edit")
    _mouse(controller, QEvent.Type.MouseButtonPress, _raw(context, 70., 50.), Qt.MouseButton.LeftButton)
    _mouse(controller, QEvent.Type.MouseMove, _raw(context, 80., 50.))
    assert controller._drag["preview"].configuration() != controller.draft.configuration()
    _mouse(controller, QEvent.Type.MouseButtonRelease, _raw(context, 5., 50.), Qt.MouseButton.LeftButton)
    assert controller.draft.configuration()["domains"] == [row]
    assert not controller._undo
    point = _raw(context, 70., 50.)
    _mouse(controller, QEvent.Type.MouseButtonPress, point, Qt.MouseButton.LeftButton)
    _mouse(controller, QEvent.Type.MouseButtonRelease, point, Qt.MouseButton.LeftButton)
    assert controller.draft.configuration()["domains"] == [row]
    assert not controller._undo


def test_passive_outline_refuses_unlinked_same_sized_photo_and_foreign_run_disables_edit(context):
    _install(context, [domains.box("Read-only", [10., 70., 20., 90.], domain_id="review")])
    controller = context.controller
    with pytest.raises(ValueError, match="does not bind"):
        controller.passive_outlines(context.models["run_a"], photo_path="unlinked.png", shape=(200, 300))
    context.view.run_preview.is_reference = lambda: False
    controller.refresh()
    assert controller.table.rowCount() == 1
    assert not controller.draw_button.isEnabled()
    assert not controller._items


@pytest.mark.parametrize("display_change", ["other_photo", "place_frame", "grid", "session_grid", "unknown_preview"])
def test_unbound_display_deactivates_box_tools_and_never_intercepts_rig_placement(context, display_change):
    _install(context, [domains.box("Bound", [10., 70., 20., 90.], domain_id="bound")])
    controller = context.controller
    assert controller.set_mode("draw")
    if display_change == "other_photo":
        context.view._other_photo = context.photo.parent / "other.png"
    elif display_change == "place_frame":
        context.view.canvas.place_mode = True
    elif display_change == "grid":
        context.view.canvas.raw_size = (301, 200)
    elif display_change == "session_grid":
        context.view.session.image_size = (300, 201)
    else:
        context.view.canvas.preview = {"same_size_but_unproved_photo": True}
    before = controller.draft.configuration()
    # No host refresh is needed for the click guard itself to notice the change.
    assert not _mouse(controller, QEvent.Type.MouseButtonPress, _raw(context, 20., 30.), Qt.MouseButton.LeftButton)
    assert controller.mode is None
    controller.refresh()
    assert not controller.draw_button.isEnabled()
    assert not controller._items
    assert controller.draft.configuration() == before


def test_host_explicit_background_on_same_reference_grid_remains_a_valid_drawing_guide(context):
    _install(context)
    token = {"host_bound_background": True}
    context.view._background_token = token
    context.view._background_shown = True
    context.view.canvas.preview = token
    controller = context.controller
    controller.refresh()
    assert controller.draw_button.isEnabled()
    assert controller.create_rectangle(_raw(context, 10., 20.), _raw(context, 70., 90.))


def test_box_tools_share_host_exclusivity_and_yield_to_new_polygon_tool_without_restoring_old_tool(context):
    _install(context)
    view, controller = context.view, context.controller
    view.private_fill_button.setChecked(True)
    assert view.private_fill_active and view.canvas.tool == "pan"
    controller.draw_button.click()
    assert controller.mode == "draw" and controller.draw_button.isChecked()
    assert not view.private_fill_active
    controller.edit_button.click()
    assert controller.mode == "edit" and not controller.draw_button.isChecked()
    view.tool_buttons["polygon"].click()
    assert controller.mode is None
    assert view.canvas.tool == "polygon"
    assert view.tool_buttons["polygon"].isChecked()
    assert not controller.draw_button.isChecked() and not controller.edit_button.isChecked()
    assert not _mouse(controller, QEvent.Type.MouseButtonPress, _raw(context, 20., 30.), Qt.MouseButton.LeftButton)
    assert controller.draft.configuration()["domains"] == []


def test_canvas_delete_later_and_late_viewport_events_do_not_access_destroyed_qt_objects(context):
    _install(context, [domains.box("Bound", [10., 70., 20., 90.], domain_id="bound")])
    controller = context.controller
    assert controller.set_mode("draw")
    viewport = controller._viewport
    canvas = context.view.canvas
    canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(canvas)
    assert controller.mode is None
    assert not controller.eventFilter(viewport, QEvent(QEvent.Type.MouseMove))
    assert not controller.eventFilter(viewport, QEvent(QEvent.Type.Destroy))
    controller.refresh()
    controller._redraw()
    assert not controller.set_mode("draw")
    assert not controller.reload()
