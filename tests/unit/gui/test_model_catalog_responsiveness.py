"""Model choices and continuation checks never wait on source assets in Qt."""
import json
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace as NS

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QLineEdit, QTextEdit, QWidget

from darsia.fluidflower import phase_model_catalog as catalog
from darsia.fluidflower import native_continuation_training as continuation
from darsia.fluidflower.ui import model_training as ui
from .test_phase_model_layout_copy import originals  # noqa: F401


def test_index_snapshot_never_opens_the_referenced_model(tmp_path, monkeypatch):
    from darsia.fluidflower import phase_model_library as library
    folder = tmp_path / "index"
    folder.mkdir()
    target = tmp_path / "offline-model"
    (folder / "M.json").write_text(json.dumps(dict(format=library.FORMAT, kind="phase",
        target=dict(path=str(target), relative_to="absolute"), source_identity=dict(model_id="M"),
        metadata=dict(display_name="Existing model", series="ffac", created_timestamp=12))), encoding="utf-8")
    monkeypatch.setattr(library, "library_root", lambda root: folder)
    monkeypatch.setattr(library, "resolve_reference", lambda *a, **k: (_ for _ in ()).throw(AssertionError("full source read")))
    rows = catalog.indexed_catalog_snapshot([tmp_path])
    assert rows[0]["root"] == str(target) and rows[0]["model_id"] == "M"
    assert rows[0]["validation_binding_loading"] and rows[0]["validation_binding"] is None
    assert "library_available" not in rows[0]  # A recorded name is not an admission.


def test_index_names_are_published_before_full_discovery(monkeypatch, tmp_path):
    tasks, events = [], []
    class DeferredThread:
        def __init__(self, *, target, **kwargs):
            self.target = target
        def start(self):
            tasks.append(self.target)
    monkeypatch.setattr(ui, "Thread", DeferredThread)
    monkeypatch.setattr(ui, "registration_context_available", lambda: False)
    monkeypatch.setattr(catalog, "indexed_catalog_snapshot", lambda roots: [dict(root="M", model_id="M")])
    page = NS(model_library=NS(text=lambda: str(tmp_path)), validation_model=NS(currentData=lambda: "", setPlaceholderText=lambda _: None),
        _model_catalog_inflight=False, _completed_model_roots={}, _sync_completed_model_roots=lambda: None,
        review_sessions=[], _active_model_series="ffac", model_history=[], _model_catalog_generation=0,
        _saved_model_preference_pending=False, _saved_model_preference_root="", _model_catalog_diagnostics=(),
        _configured_series_roots={"ffac": tmp_path},
        _model_metadata_ready=NS(emit=lambda result: events.append(("visible", result[2]))),
        _model_catalog_ready=NS(emit=lambda result: events.append(("finished", result))))
    def full(*args):
        assert events[0] == ("visible", [dict(root="M", model_id="M")])
        events.append(("full source inspection", None))
        return []
    page._discover_model_entries = full
    ui.SharedPhaseModelTrainingPage._request_async_model_refresh(page)
    tasks[0]()
    assert events[0][0] == "visible" and events[1][0] == "full source inspection"


def test_selected_binding_is_ready_before_other_models_are_inspected(monkeypatch, tmp_path):
    tasks, events = [], []
    class DeferredThread:
        def __init__(self, *, target, **kwargs):
            self.target = target
        def start(self):
            tasks.append(self.target)
    selected = dict(root=str(tmp_path / "selected"), model_id="selected", datasets=["ffac"],
                    library_source_identity={"version_sha256": "exact"})
    other = dict(root=str(tmp_path / "other"), model_id="other", datasets=["ffac"])
    monkeypatch.setattr(ui, "Thread", DeferredThread)
    monkeypatch.setattr(ui, "registration_context_available", lambda: True)
    monkeypatch.setattr(catalog, "inspect_selected_catalog_entry", lambda entry:
        events.append(("selected metadata", entry["root"])) or dict(entry))
    page = NS(model_library=NS(text=lambda: str(tmp_path)), validation_model=NS(currentData=lambda: selected["root"], setPlaceholderText=lambda _: None),
        _model_catalog_inflight=False, _completed_model_roots={}, _sync_completed_model_roots=lambda: None,
        review_sessions=[], _active_model_series="ffac", model_history=[selected, other], _model_catalog_generation=0,
        _saved_model_preference_pending=False, _saved_model_preference_root="", _model_catalog_diagnostics=(),
        _model_metadata_ready=NS(emit=lambda result: events.append(("visible", result[2]))),
        _model_binding_ready=NS(emit=lambda result: events.append(("bound", result[2]))),
        _model_catalog_ready=NS(emit=lambda result: events.append(("finished", result))))
    def bind(entry, **kwargs):
        events.append(("verify", entry["root"]))
        assert entry["root"] == selected["root"]
        return {"run_count": 1}
    def full(*args):
        assert any(kind == "bound" and value["validation_binding"] == {"run_count": 1}
                   for kind, value in events)
        events.append(("full catalogue", None))
        return [dict(selected), dict(other)]
    page._catalog_validation_binding, page._discover_model_entries = bind, full
    ui.SharedPhaseModelTrainingPage._request_async_model_refresh(page)
    tasks[0]()
    assert [item for item in events if item[0] == "verify"] == [("verify", selected["root"])]
    assert next(i for i, row in enumerate(events) if row[0] == "bound") < next(
        i for i, row in enumerate(events) if row[0] == "full catalogue")


def test_priority_inspection_keeps_exact_index_verification(tmp_path, monkeypatch):
    from darsia.fluidflower import phase_model_library as library
    reference = tmp_path / "indexed.json"
    reference.write_text(json.dumps({"source_identity": {"model_id": "M"}}), encoding="utf-8")
    reads = []
    monkeypatch.setattr(library, "resolve_reference", lambda path: reads.append(path) or tmp_path / "different")
    monkeypatch.setattr(catalog, "inspect_catalog_entry", lambda path: (_ for _ in ()).throw(AssertionError("wrong model opened")))
    import pytest
    with pytest.raises(ValueError, match="another model folder"):
        catalog.inspect_selected_catalog_entry(dict(root=str(tmp_path / "selected"), library_reference=str(reference)))
    assert reads == [str(reference)]


def test_native_combo_has_visible_choices_after_context_change_while_full_scan_is_blocked(
        originals, qtbot, monkeypatch, tmp_path):
    """Exercise the real worker -> Qt slot -> native filter -> visible combo chain."""
    from PySide6.QtCore import QSettings
    from darsia.fluidflower.native_phase_steps import NativePhaseSteps
    data = originals
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_PHASE_MODEL_DEFAULTS", str(tmp_path / "no-defaults.json"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    for name in ("DARSIA_FLUIDFLOWER_PHASE_MODEL_LIBRARY", "DARSIA_NATIVE_CONFIGURATION",
                 "DARSIA_NATIVE_CONFIGURATION_SHA256"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ui, "QSettings", lambda *_args: QSettings(
        str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(ui, "default_phase_review_roots", lambda: (data.root / "reviews",))
    first_snapshot, release_snapshot, full_scan, release_full = Event(), Event(), Event(), Event()
    original_snapshot = catalog.indexed_catalog_snapshot
    def snapshot(roots):
        if not first_snapshot.is_set():
            first_snapshot.set()
            assert release_snapshot.wait(10)
        return original_snapshot(roots)
    def blocked_full(*args):
        full_scan.set()
        assert release_full.wait(10)
        return []
    monkeypatch.setattr(catalog, "indexed_catalog_snapshot", snapshot)
    monkeypatch.setattr(ui.SharedPhaseModelTrainingPage, "_discover_model_entries", staticmethod(blocked_full))
    studio = ui.SharedPhaseModelTrainingPage()
    studio._request_async_model_refresh()
    assert first_snapshot.wait(5)
    steps = NativePhaseSteps(studio, actor_provider=lambda: "reviewer")
    steps.set_context(data.root, "ffac")
    steps.show_section("phase_model")
    steps.resize(1400, 900)
    steps.show()
    for timer in studio.findChildren(ui.QTimer):
        timer.stop()
    try:
        assert studio.validation_model.count() == 0
        assert studio.validation_model.placeholderText() == "Loading models…"
        release_snapshot.set()
        qtbot.waitUntil(full_scan.is_set, timeout=5000)
        qtbot.waitUntil(lambda: studio.validation_model.count() > 0, timeout=1000)
        combo = studio.validation_model
        assert combo.isVisible() and combo.isEnabled()
        assert combo.currentData() == str(data.model)
        assert not combo.view().isRowHidden(combo.currentIndex())
        combo.showPopup()
        qtbot.waitUntil(lambda: combo.view().isVisible())
        assert combo.view().model().rowCount() > 0
        assert data.model.name in combo.currentText()
        combo.hidePopup()
        assert studio._model_catalog_inflight and not release_full.is_set()
        choices = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
        studio._request_async_model_refresh()
        assert [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())] == choices
    finally:
        release_snapshot.set()
        release_full.set()
        qtbot.waitUntil(lambda: not studio._model_catalog_inflight and not studio._model_catalog_pending
                        and not studio._review_sessions_loading and not studio._retraction_inflight
                        and studio._pending_evidence_inflight is None, timeout=10000)
        assert studio.validation_model.count() == 0
        assert studio.validation_model.placeholderText() == "No models found"
        assert "checking" not in studio.catalog_loading_status.text().casefold()
        steps.close()
        studio.close()


def test_catalogue_failure_keeps_existing_choices_and_reports_error(qtbot):
    from .test_native_model_catalog_context import catalog_status_page
    page_type, page = catalog_status_page(qtbot, None)
    page.validation_model.addItem("Existing model", "model-root")
    page_type._apply_async_model_catalog(page, (3, "library", None, "Storage unavailable"))
    assert page.validation_model.count() == 1
    assert page.validation_model.currentData() == "model-root"
    assert page.validation_model.placeholderText() == "Models unavailable"
    assert "Storage unavailable" in page.catalog_loading_status.text()


def test_popup_paints_before_refresh_and_repeated_openings_do_not_queue_scans(qtbot, monkeypatch):
    combo = ui._RefreshingModelComboBox()
    qtbot.addWidget(combo)
    combo.addItem("Existing model", "M")
    combo.show()
    observed = []
    combo.about_to_open.connect(lambda: observed.append(combo.view().isVisible()))
    combo.showPopup()
    assert not observed
    qtbot.waitUntil(lambda: bool(observed))
    assert observed == [True]
    combo.hidePopup()
    calls = []
    page = NS(_model_catalog_root="same", model_library=NS(text=lambda: "same"),
        _model_catalog_inflight=True, _model_catalog_last_requested=100.,
        _request_async_model_refresh=lambda: calls.append("refresh"))
    monkeypatch.setattr(ui, "monotonic", lambda: 120.)
    for _ in range(3):
        ui.SharedPhaseModelTrainingPage._model_popup_opened(page)
    page._model_catalog_inflight = False
    ui.SharedPhaseModelTrainingPage._model_popup_opened(page)
    assert calls == []
    page._model_catalog_root = "previous series"
    ui.SharedPhaseModelTrainingPage._model_popup_opened(page)
    assert calls == ["refresh"]


def test_slow_continuation_is_off_thread_cached_and_drops_previous_selection(qtbot, monkeypatch, tmp_path):
    main = get_ident()
    entered, release = Event(), Event()
    reads = []
    for root in (tmp_path / "old", tmp_path / "new"):
        for name in ("training_request.json", "models/generic/co2_presence_signal.json", "models/generic/gas_within_co2_signal.json"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
    selected = dict(root=str(tmp_path / "old"), library_source_identity={})
    steps = QWidget()
    qtbot.addWidget(steps)
    steps.series_id, steps.training_routes = "ffac", {}
    steps.studio = NS(_selected_validation_model=lambda: selected, _training_finished=lambda *a: None,
        _selected_annotators=lambda: None, reviewer=QLineEdit("Reviewer"), bundles=QTextEdit("new evidence"))
    adapter = continuation.NativeContinuationTraining(steps)
    steps.refresh = adapter.refresh
    def inspect(entry):
        assert get_ident() != main
        reads.append(Path(entry["root"]).name)
        if reads[-1] == "old":
            entered.set()
            assert release.wait(5)
            raise KeyError("settings")
        return {}, dict(settings=dict(series=["ffac"])), [], {}
    monkeypatch.setattr(continuation.continuation, "inspect_parent", inspect)
    try:
        assert "background" in adapter.refresh()
        qtbot.waitUntil(entered.is_set)
        responsive = []
        QTimer.singleShot(0, lambda: responsive.append(True))
        qtbot.waitUntil(lambda: bool(responsive))
        for _ in range(3):
            adapter.refresh()
        assert reads == ["old"]
        selected = dict(root=str(tmp_path / "new"), library_source_identity={})
        adapter.refresh()
        assert not adapter.ready
        release.set()
        qtbot.waitUntil(lambda: adapter.ready)
        assert reads == ["old", "new"] and "saved weights" in adapter.reason
        for _ in range(3):
            adapter.refresh()
        assert reads == ["old", "new"]
    finally:
        release.set()
