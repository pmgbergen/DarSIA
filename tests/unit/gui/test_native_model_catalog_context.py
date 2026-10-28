"""Bounded metadata catalogue fallbacks and explicit saved-model selection."""
from contextvars import Context
import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
from PySide6.QtWidgets import QComboBox, QLabel


@pytest.fixture
def history(monkeypatch):
    from darsia.fluidflower import phase_model_history as module
    monkeypatch.delenv("DARSIA_NATIVE_CONFIGURATION", raising=False)
    monkeypatch.delenv("DARSIA_NATIVE_CONFIGURATION_SHA256", raising=False)
    return module


def test_missing_document_is_the_only_optional_registration_condition(history, monkeypatch, tmp_path):
    assert Context().run(history.registration_context_available) is False
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION", "present-without-digest")
    with pytest.raises(ValueError, match="needs both"):
        Context().run(history.registration_context_available)
    source = tmp_path / "parameters.json"
    source.write_bytes(b"{}")
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION", str(source))
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="Source artifact changed"):
        Context().run(history.registration_context_available)


def test_library_list_survives_absent_document_with_explicit_partial_reason(history, monkeypatch, tmp_path):
    from darsia.fluidflower import phase_model_catalog as catalog
    from darsia.fluidflower import phase_model_library as library
    root, model = tmp_path / "series", tmp_path / "saved-model"
    root.mkdir()
    model.mkdir()
    record = {"root": str(model), "available": True, "reference": "local-reference.json",
              "source_identity": {"model_id": "exact-model"}, "metadata": {"series": "s"}}
    monkeypatch.setattr(library, "records_for_location", lambda _root: [record])
    monkeypatch.setattr(library, "deduplicate_versions", lambda rows: list(dict.fromkeys(map(Path, rows))))
    monkeypatch.setattr(history, "_model_directories", lambda *_a, **_k: [])
    monkeypatch.setattr(history, "registered_phase_model_defaults", lambda: {})
    monkeypatch.setattr(history, "registration_context_available", lambda: False)
    monkeypatch.setattr(catalog, "inspect_catalog_entry", lambda path: {"root": str(path), "model_id": "exact-model"})
    rows = history.discover_phase_model_history(root)
    assert len(rows) == 1
    assert rows[0]["root"] == str(model)
    assert rows[0]["library_available"] is True
    assert rows[0]["library_source_identity"] == record["source_identity"]
    assert rows.diagnostics == (history.REGISTRATION_ROOTS_UNAVAILABLE,)


def test_bound_registration_failure_is_not_silently_skipped(history, monkeypatch, tmp_path):
    from darsia.fluidflower import phase_saved_registration as registration
    from darsia.fluidflower import phase_model_library as library
    monkeypatch.setattr(history, "registration_context_available", lambda: True)
    monkeypatch.setattr(history, "registered_phase_model_defaults", lambda: {})
    monkeypatch.setattr(history, "_model_directories", lambda *_a, **_k: [])
    monkeypatch.setattr(library, "records_for_location", lambda _root: [])
    def refused():
        raise ValueError("bound registration source identity changed")
    monkeypatch.setattr(registration, "registered_roots", refused)
    with pytest.raises(ValueError, match="bound registration source identity changed"):
        history.discover_phase_model_history(tmp_path)


def catalog_status_page(qtbot, history):
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage as Page
    label = QLabel()
    qtbot.addWidget(label)
    page = NS(_model_catalog_generation=3, _model_catalog_metadata_generation=3,
              _model_catalog_inflight=True, _model_catalog_pending=False,
              _model_binding_refresh_pending=False, _model_catalog_diagnostics=(),
              _saved_model_preference_pending=False, _saved_model_preference_root="",
              model_library=NS(text=lambda: "library"), catalog_loading_status=label,
              _selected_validation_model=lambda: {}, history_status=QLabel(), validation_model=QComboBox())
    qtbot.addWidget(page.history_status)
    qtbot.addWidget(page.validation_model)
    page._catalog_notice = MethodType(Page._catalog_notice, page)
    return Page, page


def test_successful_async_finish_keeps_partial_catalogue_status_visible(qtbot, history):
    Page, page = catalog_status_page(qtbot, history)
    entries = history.ModelHistoryEntries([], diagnostics=(history.REGISTRATION_ROOTS_UNAVAILABLE,))
    Page._apply_async_model_catalog(page, (3, "library", entries, None))
    assert page.catalog_loading_status.text() == history.REGISTRATION_ROOTS_UNAVAILABLE
    assert not page.catalog_loading_status.isHidden()


def test_stale_async_catalogue_cannot_replace_current_status(qtbot, history):
    Page, page = catalog_status_page(qtbot, history)
    page.catalog_loading_status.setText("current-series-status")
    entries = history.ModelHistoryEntries([], diagnostics=(history.REGISTRATION_ROOTS_UNAVAILABLE,))
    Page._apply_async_model_catalog(page, (2, "library", entries, None))
    assert page.catalog_loading_status.text() == "current-series-status"
    assert page._model_catalog_diagnostics == ()


def model_choice_page(qtbot, rows):
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage as Page
    choice = QComboBox()
    qtbot.addWidget(choice)
    page = NS(model_history=rows, _active_model_series="", validation_model=choice,
              _saved_model_preference_key=None, _saved_model_preference_root="",
              _saved_model_preference_pending=False, _model_catalog_auto_selection_root="old-auto",
              _model_catalog_priority_root="", _model_catalog_diagnostics=(),
              _model_catalog_generation=4, _model_catalog_inflight=False, _request_async_model_refresh=lambda: None,
              _validation_model_label=lambda row: row["model_id"], _validation_model_changed=lambda: None)
    page._populate_validation_models = MethodType(Page._populate_validation_models, page)
    page._catalog_notice = MethodType(Page._catalog_notice, page)
    return Page, page


def test_explicit_saved_model_older_than_newest_is_selected_and_later_click_is_preserved(qtbot, tmp_path):
    old, newest = (tmp_path / "saved-M").resolve(), (tmp_path / "newer-model").resolve()
    rows = [{"root": str(newest), "model_id": "new", "created_timestamp": 2},
            {"root": str(old), "model_id": "M", "created_timestamp": 1}]
    Page, page = model_choice_page(qtbot, rows)
    Page.set_saved_model_preference(page, old, context_key="series-a")
    assert page.validation_model.currentData() == str(old)
    assert not page._saved_model_preference_pending
    assert page._model_catalog_auto_selection_root == ""
    page.validation_model.setCurrentIndex(page.validation_model.findData(str(newest)))
    Page.set_saved_model_preference(page, old, context_key="series-a")
    assert page.validation_model.currentData() == str(newest)


def test_missing_saved_model_does_not_fall_back_to_newest(qtbot, tmp_path):
    desired, newest = (tmp_path / "absent-M").resolve(), (tmp_path / "newer").resolve()
    Page, page = model_choice_page(qtbot, [{"root": str(newest), "model_id": "new"}])
    Page.set_saved_model_preference(page, desired, context_key="series-a")
    assert page.validation_model.currentIndex() == -1
    assert page._saved_model_preference_pending
    assert str(desired) in page._catalog_notice()


def test_changed_series_context_invalidates_old_callbacks_even_with_same_library(qtbot, tmp_path):
    desired = (tmp_path / "same-model").resolve()
    Page, page = model_choice_page(qtbot, [{"root": str(desired), "model_id": "M"}])
    requested = []
    page._request_async_model_refresh = lambda: requested.append(page._model_catalog_generation)
    Page.set_saved_model_preference(page, desired, context_key=("series-a", "sha-a"))
    first = page._model_catalog_generation
    Page.set_saved_model_preference(page, desired, context_key=("series-b", "sha-b"))
    assert page._model_catalog_generation == first + 1
    assert requested == [first, first + 1]


def test_unavailable_explicit_model_is_visible_without_substitution(qtbot, tmp_path):
    desired = (tmp_path / "unavailable-reference.json").resolve()
    rows = [{"root": str(desired), "model_id": "M", "library_available": False}]
    Page, page = model_choice_page(qtbot, rows)
    Page.set_saved_model_preference(page, desired, context_key="series-a")
    assert page.validation_model.currentData() == str(desired)
    assert rows[0]["library_available"] is False


def test_human_choice_clears_missing_manifest_preference_before_next_refresh(qtbot, tmp_path):
    desired, selected = (tmp_path / "absent-M").resolve(), (tmp_path / "selected-other").resolve()
    Page, page = model_choice_page(qtbot, [{"root": str(selected), "model_id": "other"}])
    Page.set_saved_model_preference(page, desired, context_key="series-a")
    page.validation_model.setCurrentIndex(0)
    Page._note_model_catalog_selection(page, 0)
    page._populate_validation_models()
    assert page.validation_model.currentData() == str(selected)
    assert not page._saved_model_preference_pending


def test_existing_catalogue_thread_captures_launch_context_not_later_environment(monkeypatch, tmp_path, history):
    from darsia.fluidflower.ui import model_training as ui
    from darsia.fluidflower.native.config import configure_parameters, require_configuration, reset_parameters
    observed, launched, metadata = [], [], []
    class DeferredThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target
        def start(self):
            launched.append(self.target)
    monkeypatch.setattr(ui, "Thread", DeferredThread)
    page = NS(model_library=NS(text=lambda: str(tmp_path)), validation_model=NS(currentData=lambda: "", setPlaceholderText=lambda _: None),
              _model_catalog_inflight=False, _model_catalog_pending=False, _model_binding_refresh_pending=False,
              _completed_model_roots={}, _sync_completed_model_roots=lambda: None, review_sessions=[],
              _active_model_series="", model_history=[], _model_catalog_generation=0,
              _saved_model_preference_pending=False, _saved_model_preference_root="",
              _model_catalog_diagnostics=(), _model_metadata_ready=NS(emit=metadata.append),
              _model_catalog_ready=NS(emit=lambda _value: None))
    def discover(*_args):
        observed.append(require_configuration()["tag"])
        return history.ModelHistoryEntries([], diagnostics=(history.REGISTRATION_ROOTS_UNAVAILABLE,))
    page._discover_model_entries = discover
    a = configure_parameters({"tag": "launched-a"})
    try:
        ui.SharedPhaseModelTrainingPage._request_async_model_refresh(page)
        b = configure_parameters({"tag": "later-b"})
        try:
            assert len(launched) == 1
            launched[0]()
            assert observed == ["launched-a"]
            assert require_configuration()["tag"] == "later-b"
            assert metadata[0][3] == (history.REGISTRATION_ROOTS_UNAVAILABLE,)
        finally:
            reset_parameters(b)
    finally:
        reset_parameters(a)


def test_bound_a_unbound_b_bound_c_same_library_keeps_absence_and_drops_stale_ui(
    qtbot, monkeypatch, tmp_path, history
):
    from darsia.fluidflower.ui import model_training as ui
    from darsia.fluidflower.native import config as native_config
    from darsia.fluidflower.series_native_context import SeriesNativeContext
    from .test_native_series_context import series

    tasks, metadata, finished, observed, core_lookups = [], [], [], [], []
    class DeferredThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target
        def start(self):
            tasks.append(self.target)
    monkeypatch.setattr(ui, "Thread", DeferredThread)
    require_original = native_config.require_configuration
    def require_recorded(*args, **kwargs):
        result = require_original(*args, **kwargs)
        core_lookups.append(result.get("tag"))
        return result
    monkeypatch.setattr(native_config, "require_configuration", require_recorded)
    a = series(tmp_path, "a")
    b = series(tmp_path, "b", native=False)
    c = series(tmp_path, "c")
    owner = SeriesNativeContext()
    choice, label, history_label = QComboBox(), QLabel("current-C-ui"), QLabel()
    for widget in (choice, label, history_label):
        qtbot.addWidget(widget)
    model_root = str((tmp_path / "exact-indexed-model").resolve())
    choice.addItem("Exact indexed model", model_root)
    old = {"root": model_root, "model_id": "M", "library_available": True,
           "validation_binding": {"old": "A"}, "validation_binding_guard": {"old": "A"},
           "p1_final3_registered_complete": True}
    page = NS(model_library=NS(text=lambda: "one-shared-library"), validation_model=choice,
              _model_catalog_inflight=False, _model_catalog_pending=False,
              _model_binding_refresh_pending=False, _completed_model_roots={},
              _sync_completed_model_roots=lambda: None, review_sessions=[], _active_model_series="",
              model_history=[old], _model_catalog_generation=0, _model_catalog_metadata_generation=-1,
              _saved_model_preference_key=None, _saved_model_preference_root="",
              _saved_model_preference_pending=False, _model_catalog_diagnostics=(),
              _model_metadata_ready=NS(emit=metadata.append), _model_catalog_ready=NS(emit=finished.append),
              _model_binding_ready=NS(emit=lambda _value: pytest.fail("Unbound context attempted admission")),
              catalog_loading_status=label, history_status=history_label,
              _selected_validation_model=lambda: None)
    page._request_async_model_refresh = MethodType(ui.SharedPhaseModelTrainingPage._request_async_model_refresh, page)
    def discover(*_args):
        available = history.registration_context_available()
        if available:
            observed.append(native_config.require_configuration()["tag"])
            return history.ModelHistoryEntries([])
        observed.append("explicitly-unbound")
        # The real optional-root API must also bypass every require/env fallback.
        assert history.registration_context_available() is False
        return history.ModelHistoryEntries([dict(old)], diagnostics=(history.REGISTRATION_ROOTS_UNAVAILABLE,))
    page._discover_model_entries = discover
    page._saved_binding_guard_matches = lambda *_a: pytest.fail("Unbound context reused old admitted binding")
    page._catalog_validation_binding = lambda *_a, **_k: pytest.fail("Unbound context invoked original admission")
    page._catalog_notice = MethodType(ui.SharedPhaseModelTrainingPage._catalog_notice, page)

    def scenario():
        try:
            owner.bind(a)
            ui.SharedPhaseModelTrainingPage.set_saved_model_preference(page, None, context_key="a")
            assert len(tasks) == 1
            owner.bind(b)
            ui.SharedPhaseModelTrainingPage.set_saved_model_preference(page, None, context_key="b")
            tasks[0]()
            assert observed == []  # obsolete A must not delay B with a full scan
            ui.SharedPhaseModelTrainingPage._apply_async_model_catalog(page, finished.pop(0))
            assert len(tasks) == 2  # normal pending B refresh, still no OS thread
            lookups_before_b = len(core_lookups)
            tasks[1]()
            assert observed == ["explicitly-unbound"]
            assert len(core_lookups) == lookups_before_b
            b_result = metadata[-1]
            assert b_result[2][0]["library_available"] is True
            assert b_result[2][0]["validation_binding"] is None
            assert "validation_binding_guard" not in b_result[2][0]
            assert b_result[2][0]["p1_final3_registered_complete"] is False
            assert b_result[2][0]["validation_collection_error"] == history.REGISTRATION_ROOTS_UNAVAILABLE
            assert b_result[3] == (history.REGISTRATION_ROOTS_UNAVAILABLE,)
            owner.bind(c)
            ui.SharedPhaseModelTrainingPage.set_saved_model_preference(page, None, context_key="c")
            assert require_original()["tag"] == "c"
            ui.SharedPhaseModelTrainingPage._apply_async_model_metadata(page, b_result)
            assert label.text() == "current-C-ui"
            assert page.model_history == [old]  # no stale install or change to old metadata
            # Outside the catalogue-only scope, the unchanged original API sees C.
            assert history.registration_context_available() is True
            assert require_original()["tag"] == "c"
        finally:
            owner.close()
    Context().run(scenario)
