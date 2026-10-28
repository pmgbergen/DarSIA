"""Measured boxes retain coordinates across raster grids and saved revisions."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from darsia.fluidflower import metric_analysis_domains as data
from darsia.fluidflower.analysis_domains import AnalysisDomainStore
from darsia.fluidflower.map_placement import GEOMETRY_FORMAT, read_run_geometry
from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT


@pytest.fixture
def context(tmp_path):
    project = SimpleNamespace(root=tmp_path / "package", manifest={"created_by": "reviewer"}, _audit=lambda *args: None)
    current = project.root / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(dict(format=GEOMETRY_REVISION_FORMAT, revision_id="geometry-v1")))
    rows = {}
    for run, matrix in (("run_a", [[1, .2, 10], [.1, -1, 500], [.0001, 0, 1]]),
                        ("run_b", [[.8, -.1, 20], [.05, -.9, 490], [0, .0002, 1]])):
        path = tmp_path / (run + ".json")
        path.write_text(json.dumps(dict(format=GEOMETRY_FORMAT, run_id=run,
            H_metric_mm_to_camera_u=matrix, reference_image={}, shape_hw=[600, 1000],
            metric_frame=dict(unit="mm", x_axis="right", y_axis="up", measured_width_mm=899.))))
        rows[run] = dict(reference_model=data.descriptor(path))
    index = tmp_path / "index.json"
    index.write_text(json.dumps(dict(format=GEOMETRY_FORMAT, runs=rows)))
    config = data.new_configuration(project, index)
    return project, index, config, read_run_geometry(index)


def test_physical_rectangle_projects_to_each_grid_without_nominal_rescaling(context):
    _, _, _, models = context
    value = data.box("region", [100, 899, 20, 300])
    metric = np.array([[100, 20], [899, 20], [899, 300], [100, 300]])
    for model in models.values():
        points = data.project_box(value, model)
        inverse = np.linalg.inv(model.metric_to_grid) @ np.vstack((points.T, np.ones(4)))
        assert np.allclose((inverse[:2]/inverse[2]).T, metric, rtol=0, atol=1e-10)
    assert not np.array_equal(data.project_box(value, models["run_a"]), data.project_box(value, models["run_b"]))
    assert value["bounds_mm"][1] == 899


def test_editor_roundtrip_is_display_only_and_does_not_clip_legacy_edge():
    original = data.box("legacy edge", [0, 900, 0, 490])
    display = data.ui_domain(original)
    actual = data.from_normalized(display["normalized_xyxy_top_left"], name=original["name"])
    assert actual["bounds_mm"] == original["bounds_mm"]
    assert display["physical_bounds_m"]["x_max"] == .9


@pytest.mark.parametrize("bounds", ([0, 0, 1, 2], [2, 1, 0, 1], [0, 1, 0, float("nan")], [False, 1, 0, 1]))
def test_invalid_physical_boxes_are_refused(bounds):
    with pytest.raises(ValueError):
        data.box("invalid", bounds)


def test_original_legacy_values_and_corner_order_survive_explicit_review():
    original = dict(box1=[[.1, .49], [.9, .1]], box2={"roi": [[.2, .2], [.4, .3]]})
    rows = data.import_legacy_boxes(original, source={"receipt": "original input"})
    config = dict(domains=rows)
    assert rows[0]["bounds_mm"] == [100, 900, 100, 490]
    with pytest.raises(ValueError, match="Review imported"):
        data.mass_boxes(config)
    for row in rows:
        row["legacy_import"]["decision"] = "use_coordinates_in_measured_frame"
    assert data.mass_boxes(config) == original
    config["domains"][0]["bounds_mm"][0] = 101
    with pytest.raises(ValueError, match="edited"):
        data.mass_boxes(config)
    config["domains"][0]["legacy_import"]["unmodified"] = False
    assert data.mass_boxes(config)["box1"] == [[.101, .1], [.9, .49]]
    assert original["box1"][0] == [.1, .49]


def test_mass_roles_are_exact_and_convert_once():
    first = data.box("first", [100, 899, 20, 300], role="box1")
    second = data.box("second", [200, 400, 30, 200], role="box2")
    result = data.mass_boxes(dict(domains=[first, second]))
    assert set(result) == {"box1", "box2"}
    assert result["box1"] == [[.1, .02], [.899, .3]]
    with pytest.raises(ValueError, match="exactly once"):
        data.mass_boxes(dict(domains=[first, deepcopy(first), second]))
    with pytest.raises(ValueError, match="Assign one"):
        data.mass_boxes(dict(domains=[first]))


def test_new_namespace_uses_existing_writer_preserves_legacy_and_detects_stale_editor(context):
    project, _, config, _ = context
    old = AnalysisDomainStore(project)
    old.current_path.parent.mkdir(parents=True, exist_ok=True)
    old.current_path.write_bytes(b'{"legacy":"untouched"}')
    original = old.current_path.read_bytes()
    store = data.MetricAnalysisDomainStore(project)
    stale = data.MetricAnalysisDomainStore(project)
    config["domains"] = [data.box("first", [100, 800, 20, 300], role="box1"),
                         data.box("second", [200, 400, 30, 200], role="box2")]
    result = store.save_revision(config, actor="reviewer", reason="Measured regions")
    revision_file = store.artifact_root / result["revision_id"] / "revision.json"
    before = revision_file.read_bytes()
    assert data.saved_mass_boxes(revision_file)[0] == data.mass_boxes(config)
    result2 = store.save_revision(config, actor="reviewer", reason="Second review")
    assert result2["revision_id"] != result["revision_id"]
    assert revision_file.read_bytes() == before
    assert old.current_path.read_bytes() == original
    with pytest.raises(ValueError, match="Another measured-box revision"):
        stale.save_revision(config, actor="reviewer", reason="Stale")


def test_geometry_revision_and_model_bytes_are_bound(context):
    project, index, config, _ = context
    assert data.validate_configuration(project, config)["valid"]
    model = index.parent / "run_a.json"
    model.write_bytes(model.read_bytes() + b" ")
    assert not data.validate_configuration(project, config)["valid"]
    assert not data.MetricAnalysisDomainStore(project).artifact_root.exists()


def test_changed_current_geometry_refuses_save_without_artifacts(context):
    project, _, config, _ = context
    current = project.root / "overrides/geometry_facies/current.json"
    current.write_text(json.dumps(dict(revision_id="geometry-v2", content_sha256="2"*64)))
    store = data.MetricAnalysisDomainStore(project)
    with pytest.raises(ValueError, match="blocking errors"):
        store.save_revision(config, actor="reviewer", reason="stale geometry")
    assert not store.artifact_root.exists()


def test_original_validator_remains_default(monkeypatch, tmp_path):
    import darsia.fluidflower.analysis_domains as original
    seen = []
    monkeypatch.setattr(original, "validate_analysis_domain_config", lambda project, config: seen.append(config) or {"valid": False})
    store = original.AnalysisDomainStore(SimpleNamespace(root=tmp_path))
    assert store.validate_candidate({"format": "original"}) == {"valid": False}
    assert seen == [{"format": "original"}]


def test_original_dialog_defaults_keep_old_store_grid_and_reference_loading(context, qtbot, monkeypatch):
    from darsia.fluidflower.ui import analysis_domains as old
    project, _, _, _ = context
    loaded = []
    monkeypatch.setattr(old.AnalysisDomainDialog, "_load_reference",
        lambda self, **kwargs: loaded.append(kwargs))
    dialog = old.AnalysisDomainDialog(project)
    qtbot.addWidget(dialog)
    assert type(dialog.store) is AnalysisDomainStore
    assert dialog.store.override_root == project.root / "overrides/analysis_domains"
    assert dialog.domain_table.horizontalHeaderItem(2).text() == "x min [m]"
    assert loaded == [{"show_errors": False}]
    assert type(dialog.domain_canvas) is old.AnalysisDomainImageView
    dialog.close()


def test_concurrent_editors_cannot_both_publish_from_the_same_current(context):
    from concurrent.futures import ThreadPoolExecutor
    project, _, config, _ = context
    stores = [data.MetricAnalysisDomainStore(project) for _ in range(2)]
    barrier = threading.Barrier(2)
    def save(store):
        barrier.wait()
        try:
            return store.save_revision(config, actor="reviewer", reason="Concurrent review")
        except ValueError as error:
            return str(error)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save, stores))
    assert sum(isinstance(row, dict) for row in results) == 1
    assert sum("Another measured-box revision" in row for row in results if isinstance(row, str)) == 1
    assert len(stores[0].revisions()) == 1


def test_preview_and_overlay_do_not_change_scientific_inputs(context):
    _, _, config, models = context
    model = models["run_a"]
    photo = np.full((*model.shape_hw, 3), 80, np.uint8)
    labels = np.zeros(model.shape_hw, np.uint8); labels[:, 300:] = 2
    before, labels_before, matrix = photo.copy(), labels.copy(), model.metric_to_grid.copy()
    config["domains"] = [data.box("first", [100, 500, 40, 300], role="box1")]
    preview = data.metric_preview(photo, model, labels=labels)
    overlay = data.box_overlay(photo, model, config)
    assert preview.shape == (490, 900, 3)
    assert overlay.shape == photo.shape and np.any(overlay != photo)
    assert np.array_equal(photo, before) and np.array_equal(labels, labels_before)
    assert np.array_equal(model.metric_to_grid, matrix)


def test_gui_edits_mm_and_background_callback_stays_on_gui_thread(context, qtbot):
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QThread
    from shiboken6 import isValid
    from darsia.fluidflower.metric_analysis_domains_ui import MetricAnalysisDomainDialog
    project, _, config, models = context
    config["domains"] = [data.box("first", [100, 500, 40, 300], role="box1")]
    dialog = MetricAnalysisDomainDialog(project, config, models=models, series_runs=["run_a", "run_b", "missing"])
    qtbot.addWidget(dialog); dialog.show()
    assert dialog.domain_table.horizontalHeaderItem(2).text() == "x min [mm]"
    assert dialog.domain_table.item(0, 3).text() == "500"
    dialog.domain_table.item(0, 3).setText("899")
    assert dialog.domains[0]["bounds_mm"] == [100, 899, 40, 300]
    assert dialog.run_choice.count() == 3
    gui_ident = threading.get_ident()
    observed = []
    def work():
        time.sleep(.08)
        return threading.get_ident() == gui_ident
    dialog._background(work, lambda result: observed.append((result, threading.get_ident() == gui_ident)))
    worker = dialog._worker
    qtbot.waitUntil(lambda: len(observed) == 1)
    assert observed == [(False, True)]
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert dialog._worker is None and isValid(worker)
    watcher = QObject()
    worker.destroyed.connect(watcher.deleteLater)
    del worker  # The owner and this final reference release it in the GUI thread.
    QCoreApplication.sendPostedEvents(watcher, QEvent.Type.DeferredDelete)
    assert not isValid(watcher)
    dialog.close()


def test_canvas_drag_moves_and_corner_drag_resizes_without_editing_other_boxes(context, qtbot):
    from PySide6.QtCore import QPoint, Qt
    from darsia.fluidflower.metric_analysis_domains_ui import MetricAnalysisDomainDialog
    project, _, config, models = context
    config["domains"] = [data.box("first", [100, 500, 40, 300], role="box1"),
                         data.box("second", [600, 800, 40, 300], role="box2")]
    dialog = MetricAnalysisDomainDialog(project, config, models=models)
    qtbot.addWidget(dialog); dialog.show(); qtbot.wait(20)
    canvas = dialog.domain_canvas; canvas.fit_image(); dialog._set_mode("edit")
    old = deepcopy(dialog.domains[0]["bounds_mm"])
    item = canvas._domain_items[dialog.domains[0]["domain_id"]]
    centre = canvas.mapFromScene(item.rect().center())
    qtbot.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=centre)
    qtbot.mouseMove(canvas.viewport(), centre+QPoint(18, 0))
    qtbot.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=centre+QPoint(18, 0))
    changed = dialog.domains[0]["bounds_mm"]
    assert changed[0] > old[0] and changed[1]-changed[0] == pytest.approx(400)
    assert dialog.domains[1]["bounds_mm"] == [600, 800, 40, 300]
    item = canvas._domain_items[dialog.domains[0]["domain_id"]]
    corner = canvas.mapFromScene(item.rect().bottomRight())
    qtbot.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=corner)
    qtbot.mouseMove(canvas.viewport(), corner+QPoint(12, 8))
    qtbot.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=corner+QPoint(12, 8))
    resized = dialog.domains[0]["bounds_mm"]
    assert resized[1] > changed[1] and resized[2] < changed[2]
    assert dialog.domains[0]["role"] == "box1"
    dialog.close()


def test_name_role_and_enable_edits_preserve_all_coordinate_bits(context, qtbot):
    from PySide6.QtCore import Qt
    from darsia.fluidflower.metric_analysis_domains_ui import MetricAnalysisDomainDialog
    project, _, config, models = context
    coordinates = [100.12345678901234, 500.98765432109876, 40.12345678901234, 300.98765432109876]
    config["domains"] = [data.box("first", coordinates, role="box1")]
    dialog = MetricAnalysisDomainDialog(project, config, models=models)
    qtbot.addWidget(dialog)
    dialog.domain_table.item(0, 1).setText("renamed")
    dialog.domain_table.item(0, 6).setText("box2")
    dialog.domain_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert [x.hex() for x in dialog.domains[0]["bounds_mm"]] == [x.hex() for x in coordinates]
    dialog.show(); qtbot.wait(20); dialog.domain_canvas.fit_image(); dialog._set_mode("edit")
    item = dialog.domain_canvas._domain_items[dialog.domains[0]["domain_id"]]
    centre = dialog.domain_canvas.mapFromScene(item.rect().center())
    qtbot.mouseClick(dialog.domain_canvas.viewport(), Qt.MouseButton.LeftButton, pos=centre)
    assert [x.hex() for x in dialog.domains[0]["bounds_mm"]] == [x.hex() for x in coordinates]
    dialog.domain_table.item(0, 2).setText("101")
    assert dialog.domains[0]["bounds_mm"][0] == 101
    assert [x.hex() for x in dialog.domains[0]["bounds_mm"][1:]] == [x.hex() for x in coordinates[1:]]
    dialog.close()
