"""A saved box revision can be integrated without changing Setup or its model."""
from copy import deepcopy
import json
import threading

from darsia.fluidflower import metric_analysis_domains as domains
from .test_painting_analysis_boxes_ui import context, _install  # noqa: F401


def _saved(context):
    config = _install(context, [domains.box("New area", [1., 10., 2., 8.])])
    store = domains.MetricAnalysisDomainStore(context.project)
    store.save_revision(config, actor="test", reason="Analysis after production")
    context.controller.set_context(config, context.models, store=store)
    context.view._workflow_scope = "analysis_boxes"
    context.controller.refresh()
    return deepcopy(config)


def test_saved_arbitrary_box_starts_background_integration_only(context, tmp_path, qtbot):
    config = _saved(context)
    controller = context.controller
    panel = controller.reanalysis
    main_thread = threading.get_ident()
    calls = []
    def integrate(root, boxes, output, **kwargs):
        assert threading.get_ident() != main_thread
        kwargs["check_cancelled"]()
        calls.append((root, boxes, output))
        kwargs["on_progress"]({"run": "run_a", "completed": 1, "total": 2})
        output.mkdir(parents=True)
        table = output / "run_a.csv"
        table.write_text("run,name,elapsed_h,total_mass_g,aqueous_mass_g,gas_mass_g,complete\n"
                         "run_a,New area,2.5,3,1,2,True\n")
        return dict(run_count=1, frame_count=2, tables=[dict(path=str(table))])
    panel.integrator = integrate
    panel.jobs.addItem("Saved production", str(tmp_path / "job"))
    panel.output.setText(str(tmp_path / "derived"))
    panel.refresh()
    assert panel.run.isEnabled()
    assert controller.table.isColumnHidden(6)
    assert "Assign one" not in controller.note.text()
    assert panel.start()
    qtbot.waitUntil(lambda: not controller.busy())
    assert len(calls) == 1
    assert calls[0][1] == config == controller._saved_configuration
    assert calls[0][2].parent == tmp_path / "derived"
    assert "Completed: 1 runs, 2 frames" in panel.status.text()
    assert panel.open.isEnabled()
    assert panel.results.rowCount() == 1
    assert panel.results.item(0, 0).text() == "run_a / New area"
    assert panel.results.item(0, 2).text() == "3"


def test_unsaved_boxes_cannot_start_reanalysis(context, tmp_path):
    _saved(context)
    panel = context.controller.reanalysis
    panel.jobs.addItem("Saved production", str(tmp_path / "job"))
    context.controller.dirty = True
    panel.refresh()
    assert not panel.run.isEnabled()
    assert not panel.start()


def test_cancel_releases_editor_after_current_read(context, tmp_path, qtbot):
    _saved(context)
    controller = context.controller
    panel = controller.reanalysis
    started = threading.Event()
    release = threading.Event()
    def integrate(*args, **kwargs):
        started.set()
        release.wait(3)
        kwargs["check_cancelled"]()
        raise AssertionError("Cancelled work continued")
    panel.integrator = integrate
    panel.jobs.addItem("Saved production", str(tmp_path / "job"))
    panel.refresh()
    assert panel.start()
    qtbot.waitUntil(started.is_set)
    assert panel.cancel.isEnabled() and not panel.jobs.isEnabled()
    panel.cancel_requested()
    release.set()
    qtbot.waitUntil(lambda: not controller.busy())
    assert "cancelled" in panel.status.text().lower()
    assert not panel.open.isEnabled() and panel.run.isEnabled()


def test_source_list_contains_mass_jobs_not_setup_or_p1(context, tmp_path, monkeypatch):
    from darsia.fluidflower import native_jobs_registry, series_geometry
    from darsia.fluidflower.analysis_box_reanalysis_ui import registered_jobs
    rows = []
    for pipeline in ("series-setup", "native-phase-statistics", "native-phase-n1", "n1", "lattice-phase-n1"):
        root = tmp_path / pipeline
        task_root = root / "results" / "run_a"
        task_root.mkdir(parents=True)
        (root / "job.json").write_text(json.dumps({"expected_runs": ["run_a"]}))
        (task_root / "task.json").write_text(json.dumps({"pipeline": pipeline}))
        rows.append(dict(root=str(root), name=pipeline))
    monkeypatch.setattr(series_geometry, "native_series_link", lambda project:
                        dict(series_root=str(tmp_path), series_id="series"))
    monkeypatch.setattr(native_jobs_registry.JobsUIRegistry, "jobs", lambda *_: rows)
    assert [row["name"] for row in registered_jobs(context.project)] == [
        "native-phase-n1", "n1", "lattice-phase-n1"]


def test_sand_layers_are_integrated_only_when_chosen(context, tmp_path, qtbot, monkeypatch):
    from darsia.fluidflower import analysis_box_reanalysis

    _saved(context)
    panel = context.controller.reanalysis
    monkeypatch.setattr(analysis_box_reanalysis, "layer_names", lambda project: {1: "Upper sand"})
    calls = []

    def integrate(root, boxes, output, **kwargs):
        calls.append({key: kwargs[key] for key in ("layers", "layer_names") if key in kwargs})
        output.mkdir(parents=True)
        result = dict(run_count=1, frame_count=2, tables=[])
        if kwargs.get("layers"):
            result.update(layer_tables=[], frames_without_native_facies=1)
        return result

    panel.integrator = integrate
    panel.jobs.addItem("Saved production", str(tmp_path / "job"))
    panel.output.setText(str(tmp_path / "derived"))
    panel.refresh()
    assert not panel.layers.isChecked()
    assert panel.start()
    qtbot.waitUntil(lambda: not context.controller.busy())
    assert calls == [{}]  # Off: the integrator is called exactly as before.
    panel.layers.setChecked(True)
    assert panel.start()
    qtbot.waitUntil(lambda: not context.controller.busy())
    assert calls[1] == dict(layers=True, layer_names={1: "Upper sand"})
    assert "1 frames had no native maps with facies" in panel.status.text()
