"""Analysis lists saved model metadata without granting native execution admission."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.native import config as native_config
from darsia.fluidflower import phase_model_history as history
from darsia.fluidflower.tools import phase_studio_mass_analysis_workspace as analysis


@pytest.fixture
def no_native(tmp_path, monkeypatch):
    for key in tuple(__import__("os").environ):
        if key.startswith("DARSIA_"):
            monkeypatch.delenv(key)
    for name in ("APPDATA", "LOCALAPPDATA", "USERPROFILE"):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setenv(name, str(path))
    token = native_config._configured.set(None)
    catalogue_token = history._CATALOG_NATIVE_AVAILABLE.set(None)
    try:
        with pytest.raises(ValueError, match="An explicit native parameter document is required"):
            native_config.require_configuration()
        yield
    finally:
        history._CATALOG_NATIVE_AVAILABLE.reset(catalogue_token)
        native_config._configured.reset(token)


@pytest.fixture
def saved_models(tmp_path, no_native):
    root = tmp_path / "actual-series"
    root.mkdir()
    (root / "series.toml").write_text('schema_version = 1\nseries_id = "FF_UM"\n', encoding="utf-8")
    for name, date, scope in (
        ("phase-model-earlier", "2026-01-01T00:00:00Z", "ffum"),
        ("phase-model-later", "2026-01-02T00:00:00Z", "ffum"),
        ("phase-model-foreign", "2026-01-03T00:00:00Z", "ffbm"),
    ):
        folder = root / "models" / name
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps(dict(
            campaign_id=name, created_utc=date,
            inventory=dict(datasets=[scope], runs=["run1"], series_scope=scope))), encoding="utf-8")
        (folder / "training_request.json").write_text(json.dumps(dict(
            model_id=name, created_utc=date, settings={})), encoding="utf-8")
    # Missing heads are deliberate: invalid/interrupted versions must remain
    # visible with their existing warning; listing does not make them runnable.
    return root


def test_real_catalogue_without_native_does_not_call_registration_runtime(saved_models, monkeypatch):
    from darsia.fluidflower import phase_saved_registration

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Full native registration admission was called during listing")

    monkeypatch.setattr(phase_saved_registration, "registration_warning", forbidden)
    before = {str(path): path.read_bytes() for path in saved_models.rglob("*") if path.is_file()}
    rows = analysis._series_models("ffum", saved_models)
    assert [row["model_id"] for row in rows] == ["phase-model-later", "phase-model-earlier"]
    assert all(row["catalog_admission_pending"] for row in rows)
    assert all(row["state"] == "invalid" for row in rows)
    assert rows.diagnostics == (history.REGISTRATION_ROOTS_UNAVAILABLE,)
    assert history._CATALOG_METADATA_ONLY.get() is False
    assert {str(path): path.read_bytes() for path in saved_models.rglob("*") if path.is_file()} == before


def test_cancellation_still_returns_without_model_inspection(saved_models):
    assert analysis._series_models("ffum", saved_models, lambda: True) == []


def test_owner_invalid_binding_is_reported_in_thread_even_after_environment_cleared(saved_models, monkeypatch, qtbot):
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION", str(saved_models / "missing.json"))
    lookup = analysis._ModelChoicesLookup("ffum", saved_models)
    monkeypatch.delenv("DARSIA_NATIVE_CONFIGURATION")
    with qtbot.waitSignal(lookup.ready, timeout=5000) as signal:
        lookup.start()
    assert lookup.wait(5000)
    rows, error = signal.args
    assert rows is None
    assert "Native parameter environment needs both" in error


def test_owner_verified_native_context_survives_actual_qthread(saved_models, qtbot):
    token = native_config.configure_parameters({"catalogue_test": "owner binding"})
    try:
        lookup = analysis._ModelChoicesLookup("ffum", saved_models)
    finally:
        native_config.reset_parameters(token)
    with pytest.raises(ValueError, match="An explicit native parameter document is required"):
        native_config.require_configuration()
    with qtbot.waitSignal(lookup.ready, timeout=5000) as signal:
        lookup.start()
    assert lookup.wait(5000)
    rows, error = signal.args
    assert not error
    assert len(rows) == 2
    assert rows.diagnostics == ()


def test_owner_absence_cannot_adopt_a_later_environment_document(saved_models, monkeypatch, qtbot):
    lookup = analysis._ModelChoicesLookup("ffum", saved_models)
    path = saved_models / "later-native.json"
    raw = b'{"catalogue_test": "later environment"}'
    path.write_bytes(raw)
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION", str(path))
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION_SHA256", hashlib.sha256(raw).hexdigest())
    with qtbot.waitSignal(lookup.ready, timeout=5000) as signal:
        lookup.start()
    assert lookup.wait(5000)
    rows, error = signal.args
    assert not error
    assert len(rows) == 2
    assert rows.diagnostics == (history.REGISTRATION_ROOTS_UNAVAILABLE,)


def _owner(qtbot):
    from PySide6.QtWidgets import QComboBox, QLabel

    model, status = QComboBox(), QLabel()
    qtbot.addWidget(model)
    qtbot.addWidget(status)
    calls = []
    return SimpleNamespace(_generation=7, model=model, status=status,
                           _selected_model_changed=lambda: calls.append("selected")), calls


def test_original_dropdown_shows_models_and_missing_native_without_autoselect(saved_models, qtbot):
    rows = analysis._series_models("ffum", saved_models)
    owner, calls = _owner(qtbot)
    analysis.MassAnalysisWorkspace._models_ready(owner, 7, None, rows, "")
    assert owner.model.count() == 3
    assert owner.model.currentData() is None
    assert calls == []
    assert "2 recorded model versions" in owner.status.text()
    assert history.REGISTRATION_ROOTS_UNAVAILABLE in owner.status.text()
    assert {owner.model.itemData(index) for index in (1, 2)} == {row["root"] for row in rows}


def test_existing_explicit_selection_is_preserved(saved_models, qtbot):
    rows = analysis._series_models("ffum", saved_models)
    owner, calls = _owner(qtbot)
    selected = str(saved_models / "models/phase-model-earlier")
    analysis.MassAnalysisWorkspace._models_ready(owner, 7, selected, rows, "")
    assert Path(owner.model.currentData()) == Path(selected)
    assert calls == ["selected"]


def test_selected_model_native_admission_error_is_not_promoted_to_catalogue_success(no_native, qtbot):
    from PySide6.QtWidgets import QLabel

    status = QLabel()
    qtbot.addWidget(status)
    owner = SimpleNamespace(_generation=7, status=status, phase_catalog=None,
                            saved_results=SimpleNamespace(phase_catalog=None))
    error = "An explicit native parameter document is required"
    analysis.MassAnalysisWorkspace._catalog_ready(owner, 7, None, error)
    assert owner.phase_catalog is None
    assert owner.saved_results.phase_catalog is None
    assert owner.status.text() == "Selected model is not ready for native mass analysis: " + error
