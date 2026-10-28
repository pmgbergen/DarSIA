"""Local metadata/context regressions; no native computation or real series."""
from contextvars import Context, copy_context
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from darsia.fluidflower.native.config import (
    configure_parameters, require_configuration, reset_parameters,
)
from darsia.fluidflower.series_native_context import (
    SeriesNativeContext, read_series_native_binding,
)
from .test_native_series_method import method_window


ENV = ("DARSIA_NATIVE_CONFIGURATION", "DARSIA_NATIVE_CONFIGURATION_SHA256")


def series(tmp_path, name, *, native=True):
    root = tmp_path / name
    root.mkdir()
    raw = (json.dumps({"tag": name, "numeric": [1, 1.25, True]}) + "\n").encode()
    folder = root / "config/native"
    folder.mkdir(parents=True)
    (folder / "parameters.json").write_bytes(raw)
    text = f'series_id = "{name}"\n'
    if native:
        text += ('[native]\nconfiguration = "config/native/parameters.json"\n'
                 f'sha256 = "{hashlib.sha256(raw).hexdigest()}"\n')
    (root / "series.toml").write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def no_native_environment(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)


def test_relative_binding_verifies_exact_bytes_without_context_or_writes(tmp_path, no_native_environment):
    root = series(tmp_path, "series-a")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    binding = read_series_native_binding(root)
    assert binding.path == (root / "config/native/parameters.json").resolve()
    assert binding.document == {"tag": "series-a", "numeric": [1, 1.25, True]}
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert all(os.environ.get(key) is None for key in ENV)


def test_switch_and_close_restore_previous_document_and_environment(tmp_path, monkeypatch):
    a, b = series(tmp_path, "a"), series(tmp_path, "b")
    monkeypatch.setenv(ENV[0], "original-external-document")
    monkeypatch.setenv(ENV[1], "original-external-digest")
    ambient = {"ambient": True}
    token = configure_parameters(ambient)
    owner = SeriesNativeContext()
    try:
        owner.bind(a)
        assert require_configuration()["tag"] == "a"
        assert os.environ[ENV[0]] == str((a / "config/native/parameters.json").resolve())
        owner.bind(b)
        assert require_configuration()["tag"] == "b"
        owner.close()
        assert require_configuration() is ambient
        assert tuple(os.environ[key] for key in ENV) == ("original-external-document", "original-external-digest")
    finally:
        owner.close()
        reset_parameters(token)


def test_absent_native_table_restores_owned_prior_binding(tmp_path, no_native_environment):
    a, b = series(tmp_path, "a"), series(tmp_path, "b", native=False)
    owner = SeriesNativeContext()
    def scenario():
        try:
            owner.bind(a)
            owner.bind(b)
            assert owner.binding is None
            assert all(os.environ.get(key) is None for key in ENV)
            with pytest.raises(ValueError, match="explicit native parameter document"):
                require_configuration()
        finally:
            owner.close()
    Context().run(scenario)


@pytest.mark.parametrize("replacement", [
    '[native]\nconfiguration = "config/native/parameters.json"\n',
    '[native]\nsha256 = "' + "0" * 64 + '"\n',
    'native = "wrong-table-type"\n',
])
def test_present_malformed_table_is_not_absent(tmp_path, replacement):
    root = series(tmp_path, "bad")
    (root / "series.toml").write_text(replacement, encoding="utf-8")
    with pytest.raises(ValueError, match="native"):
        read_series_native_binding(root)


def test_bad_sha_refuses_target_and_preserves_old_accepted_context(tmp_path, no_native_environment):
    a, b = series(tmp_path, "a"), series(tmp_path, "b")
    owner = SeriesNativeContext()
    try:
        owner.bind(a)
        accepted = require_configuration()
        environment = tuple(os.environ[key] for key in ENV)
        (b / "config/native/parameters.json").write_bytes(b'{"changed": true}')
        with pytest.raises(ValueError, match="Source artifact changed"):
            owner.bind(b)
        assert require_configuration() is accepted
        assert owner.binding.series_root == a.resolve()
        assert tuple(os.environ[key] for key in ENV) == environment
    finally:
        owner.close()


def test_copied_context_keeps_accepted_document_after_main_switch(tmp_path, no_native_environment):
    owner = SeriesNativeContext()
    a, b = series(tmp_path, "a"), series(tmp_path, "b")
    try:
        owner.bind(a)
        launched = copy_context()
        owner.bind(b)
        assert launched.run(require_configuration)["tag"] == "a"
        assert require_configuration()["tag"] == "b"
    finally:
        owner.close()


def test_close_does_not_clobber_an_external_later_environment_change(tmp_path, monkeypatch):
    owner = SeriesNativeContext()
    owner.bind(series(tmp_path, "a"))
    monkeypatch.setenv(ENV[0], "later-owner-path")
    monkeypatch.setenv(ENV[1], "later-owner-sha")
    owner.close()
    assert tuple(os.environ[key] for key in ENV) == ("later-owner-path", "later-owner-sha")


def test_dunder_import_lookup_never_requests_native_parameters(monkeypatch):
    from darsia.fluidflower import phase_saved_registration as registration
    def forbidden(*_args):
        pytest.fail("Dunder lookup requested native configuration")
    monkeypatch.setattr(registration, "configured_parameters", forbidden)
    for name in ("__path__", "__all__", "__spec_missing__"):
        with pytest.raises(AttributeError):
            getattr(registration, name)
    # Retained ordinary export lookup must still use the original parameter API.
    with pytest.raises(pytest.fail.Exception):
        getattr(registration, "unknown_ordinary_export")


@pytest.fixture
def workspace(qtbot, monkeypatch, tmp_path, no_native_environment):
    from darsia.fluidflower.native_darsia_app import PhaseModelWorkspace
    a, b = series(tmp_path, "a"), series(tmp_path, "b")
    owner = SeriesNativeContext()
    window = PhaseModelWorkspace(series=(("a", "A", a), ("b", "B", b)),
                                 initial_series="a", native_context_binder=owner.bind)
    qtbot.addWidget(window)
    calls = []
    studio = NS(set_model_series=lambda key: calls.append(("studio", key)))
    page = NS(studio=studio, native_numbered_steps=True,
              _active_job_notice=lambda _studio: None,
              bind_context=lambda key, _root: calls.append(("context", key, require_configuration()["tag"])),
              setEnabled=lambda _value: None)
    window.studio_page = page
    try:
        yield NS(window=window, owner=owner, roots=(a, b), page=page, calls=calls)
    finally:
        window.close()
        owner.close()


def test_real_workspace_binds_before_existing_studio_context_switch(workspace):
    workspace.window.series_selector.setCurrentIndex(1)
    assert workspace.window.series_id == "b"
    assert workspace.calls[0] == ("context", "b", "b")


def test_invalid_target_rolls_back_selector_without_binding_existing_page(workspace):
    (workspace.roots[1] / "config/native/parameters.json").write_bytes(b"{}")
    workspace.window.series_selector.setCurrentIndex(1)
    assert workspace.window.series_id == "a"
    assert require_configuration()["tag"] == "a"
    assert not workspace.calls
    assert "Source artifact changed" in workspace.window.series_support_status.text()


def test_active_job_refusal_does_not_switch_native_context(workspace, monkeypatch):
    from darsia.fluidflower import native_darsia_app as native
    workspace.page._active_job_notice = lambda _studio: ("Busy", "Keep context")
    notices = []
    monkeypatch.setattr(native.QMessageBox, "information", lambda *_args: notices.append(_args))
    workspace.window.series_selector.setCurrentIndex(1)
    assert workspace.window.series_id == "a"
    assert require_configuration()["tag"] == "a"
    assert not workspace.calls
    assert len(notices) == 1


def test_existing_save_first_refusal_precedes_new_native_context_binder(method_window, monkeypatch):
    window = method_window
    roots = window.phase_model_workspace.series_roots
    root = roots["series-a"]
    native = root / "config/native/local.json"
    native.parent.mkdir(parents=True)
    raw = b'{"tag": "accepted-series-a"}\n'
    native.write_bytes(raw)
    with (root / "series.toml").open("ab") as stream:
        stream.write(('[native]\nconfiguration = "config/native/local.json"\n'
                      f'sha256 = "{hashlib.sha256(raw).hexdigest()}"\n').encode())
    window._series_native_context.bind(root)
    accepted = require_configuration()
    environment = tuple(os.environ.get(key) for key in ENV)
    requested = []
    monkeypatch.setattr(window.reference_open_guard, "allowed",
                        lambda **kwargs: requested.append(kwargs) or False)
    for row in range(window.home_page.table.rowCount()):
        if window.home_page.table.item(row, 1).text() == "series-b":
            window.home_page.table.selectRow(row)
            break
    assert requested
    assert window.phase_model_workspace.series_id == "series-a"
    assert require_configuration() is accepted
    assert tuple(os.environ.get(key) for key in ENV) == environment
