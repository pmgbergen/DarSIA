from threading import Event
from types import SimpleNamespace as NS
import json

from PySide6.QtWidgets import QWidget
import toml

from darsia.fluidflower import calibration_preparation as preparation
from darsia.fluidflower import mass_calibration_display as mass
from darsia.fluidflower import phase_model_catalog as catalog
from darsia.fluidflower import storage_paths
from darsia.fluidflower.refit_job_panel import RefitJobPanel
from darsia.fluidflower.ui import model_training as studio
from darsia.fluidflower.ui_metadata_preview import MetadataPreview


def test_disposable_reader_cancels_without_waiting_or_delivering_stale_result(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    reader = MetadataPreview(owner)
    entered, release = Event(), Event()
    delivered = []
    reader.completed.connect(lambda value, error: delivered.append((value, error)))
    def slow(cancelled):
        entered.set()
        assert release.wait(5)
        return "obsolete metadata"
    try:
        reader.start(slow)
        qtbot.waitUntil(entered.is_set)
        reader.cancel()
        assert not reader.busy
        owner.close()
        release.set()
        qtbot.wait(30)
        assert delivered == []
    finally:
        release.set()


def test_opening_calibration_reads_only_saved_series_choices_and_never_admits_cohort(qtbot, tmp_path, monkeypatch):
    path = tmp_path / "series.toml"
    path.write_text(toml.dumps(dict(series_id="sample", runs=dict(ids=["run1"]),
        image_selection=dict(runs=dict(run1=dict(selected=[dict(source_path="offline.RAW", elapsed_h=1.)]))))))
    before = path.read_bytes()
    monkeypatch.setattr(preparation, "saved_selection", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Scientific cohort admission must wait for Start")))
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    monkeypatch.setattr(panel, "_create_blocker", lambda **k: "")
    window = NS(open_image_selection_plan=lambda: None, _select_row=lambda *a: None,
                show_phase_model_workflow=lambda: None)
    widget = preparation.CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    widget.set_series(tmp_path)
    qtbot.waitUntil(lambda: not widget._preview.busy)
    assert widget.selection["preview_only"] and widget.table.rowCount() == 1
    assert widget.table.item(0, 2).text() == "1"  # Saved metadata count; no RAW is read.
    assert "checked on Start" in widget.summary.text()
    assert widget.can_close() and path.read_bytes() == before


def test_calibration_can_close_during_disposable_metadata_and_discards_it(qtbot, tmp_path, monkeypatch):
    entered, release = Event(), Event()
    def slow(*args):
        entered.set()
        assert release.wait(5)
        return dict(preview_only=True, runs=[], counts=dict(selected_frames=0))
    monkeypatch.setattr(preparation, "selection_preview", slow)
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    window = NS(open_image_selection_plan=lambda: None, _select_row=lambda *a: None,
                show_phase_model_workflow=lambda: None)
    widget = preparation.CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    try:
        widget.set_series(tmp_path)
        qtbot.waitUntil(entered.is_set)
        assert widget.can_close()
        widget.close()
        assert not widget._preview.busy
        release.set()
        qtbot.wait(30)
        assert widget.selection is None
    finally:
        release.set()


def test_native_model_metadata_does_not_run_saved_map_admission_until_requested(tmp_path, monkeypatch):
    tasks, verified, events = [], [], []
    class DeferredThread:
        def __init__(self, *, target, **kwargs):
            self.target = target
        def start(self):
            tasks.append(self.target)
    selected = dict(root=str(tmp_path / "selected"), model_id="selected")
    monkeypatch.setattr(studio, "Thread", DeferredThread)
    monkeypatch.setattr(studio, "registration_context_available", lambda: True)
    monkeypatch.setattr(catalog, "inspect_selected_catalog_entry", lambda entry: dict(entry))
    page = NS(_native_metadata_preview_only=True, model_library=NS(text=lambda: str(tmp_path)),
        validation_model=NS(currentData=lambda: selected["root"], setPlaceholderText=lambda _: None),
        _model_catalog_inflight=False, _completed_model_roots={}, _sync_completed_model_roots=lambda: None,
        review_sessions=[], _active_model_series="", model_history=[selected], _model_catalog_generation=0,
        _saved_model_preference_pending=False, _saved_model_preference_root="", _model_catalog_diagnostics=(),
        _model_metadata_ready=NS(emit=lambda result: events.append(result)),
        _model_binding_ready=NS(emit=lambda result: None), _model_catalog_ready=NS(emit=lambda result: None),
        _selected_validation_model=lambda: selected, _discover_model_entries=lambda *a: [dict(selected)],
        _catalog_validation_binding=lambda entry, **k: verified.append(entry["root"]) or {"exact": True})
    studio.SharedPhaseModelTrainingPage._request_async_model_refresh(page)
    tasks.pop(0)()
    assert not verified and events[-1][2][0]["validation_binding"] is None
    assert not events[-1][2][0]["validation_binding_loading"]
    page._model_catalog_inflight = False
    studio.SharedPhaseModelTrainingPage._request_async_model_refresh(page, bindings_only=True)
    assert not tasks and not verified
    studio.SharedPhaseModelTrainingPage._request_async_model_refresh(page, bindings_only=True, verify_saved=True)
    tasks.pop(0)()
    assert verified == [selected["root"]]


def test_mass_display_follows_declared_alias_without_touching_frozen_reader(tmp_path, monkeypatch):
    old, current = tmp_path / "old-result.json", tmp_path / "current-result.json"
    current.write_text(json.dumps(dict(parameters=dict(gain=2., beta=.5))))
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [
        (str(old).replace("\\", "/").casefold(), str(current).replace("\\", "/"))])
    value = mass.read_mass_calibration_display("phase_model", str(old), series_root=tmp_path)
    assert dict((row[0], row[1]) for row in value.rows)["Gain"] == "2.0"
    assert str(current) in value.source_label and not old.exists()


def test_calibration_preview_separates_all_preparation_runs_from_explicit_mass_fit_runs(tmp_path):
    path = tmp_path / "series.toml"
    path.write_text(toml.dumps(dict(series_id="sample", runs=dict(ids=["fit", "heldout"]),
        method=dict(active="phase_model", phase_model=dict(mass_calibration="frozen.json")),
        image_selection=dict(runs={run: dict(selected=[dict(source_path="offline.RAW", elapsed_h=1.)])
                                   for run in ("fit", "heldout")}))))
    (tmp_path / "frozen.json").write_text(json.dumps(dict(calibration_runs=["fit"])))
    value = preparation.selection_preview(tmp_path)
    assert [row["run"] for row in value["runs"]] == ["fit", "heldout"]
    assert value["mass_fit_runs"] == ["fit"]
