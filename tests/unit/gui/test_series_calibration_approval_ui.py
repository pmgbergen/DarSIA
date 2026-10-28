"""Original native Use/Unlock controls with finite saved metadata only."""

from dataclasses import replace
import threading
import tomllib

import toml

from PySide6.QtCore import QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDialog, QLineEdit

import pytest

from darsia.fluidflower import native_darsia_app as native
from darsia.fluidflower import series_calibration_approval as approval
from tests.unit.test_series_calibration_approval import bound_products
from .test_native_series_method import method_window, _roots  # noqa: F401


@pytest.fixture(autouse=True)
def loaded_series(method_window, qtbot):
    """Edit the real fields only after their owned initial metadata read returns."""
    qtbot.waitUntil(lambda: not method_window._series_loading, timeout=10000)
    assert not method_window._series_load_error


def _choose(window):
    case = bound_products(_roots(window)["series-a"])
    window._refresh_method_state(force=True)
    panel = window.method_panel
    panel.phase_model_button.click()
    panel.model_version.setText(case.selection.model_version)
    panel.mass_calibration.setText(case.selection.mass_calibration)
    return case, panel


def _settle(panel, qtbot):
    qtbot.waitUntil(lambda: panel.calls.thread is None and not panel.calls.preview.busy and not panel.calls.timer.isActive(), timeout=10000)


def test_native_use_locks_bound_choice_and_reasoned_unlock_preserves_sources(
    method_window, monkeypatch, qtbot,
):
    window = method_window
    case, panel = _choose(window)
    _settle(panel, qtbot)
    before = {p: p.read_bytes() for p in case.root.rglob("*") if p.is_file()}
    assert panel.use_button.isEnabled()
    assert "Titration" in panel.approval_summary.text()
    assert approval.read_approval(case.root) is None
    panel.use_button.click()  # Actual signal -> owned original reader -> manifest CAS.
    _settle(panel, qtbot)
    first = approval.read_approval(case.root)
    assert first["locked"] and first["binding"]["chemistry_model"] == "titration"
    assert not panel.use_button.isEnabled()
    assert not panel.mass_calibration.isEnabled() and not panel.model_version.isEnabled()
    assert not panel.colour_path_button.isEnabled()
    assert "Titration" in window.method_badge.text() and "locked" in window.method_badge.text()
    window._refresh_method_state(force=True)
    _settle(panel, qtbot)
    assert not panel.model_version.isEnabled()  # Lock survives a reload.
    case.calibration.write_bytes(before[case.calibration] + b" ")
    window._refresh_method_state(force=True)
    _settle(panel, qtbot)
    assert not window._confirm_method_action("phase_model", "Run the selected method")
    assert approval.read_approval(case.root, verify=False)["locked"]
    assert not panel.use_button.isEnabled()
    case.calibration.write_bytes(before[case.calibration])
    window._refresh_method_state(force=True)
    _settle(panel, qtbot)
    monkeypatch.setattr(native.QInputDialog, "getMultiLineText",
                        lambda *_a, **_k: ("Review a new calibration.", True))
    panel.unlock_button.click()
    _settle(panel, qtbot)
    second = approval.read_approval(case.root)
    assert not second["locked"] and second["binding"] == first["binding"]
    assert second["events"][:-1] == first["events"]
    assert panel.mass_calibration.isEnabled() and panel.use_button.isEnabled()
    assert "last approved chemistry Titration" in window.method_badge.text()
    assert all(p.read_bytes() == raw for p, raw in before.items() if p != case.manifest)


@pytest.mark.parametrize("change", ["cancel", "series", "manifest"])
def test_unlock_modal_context_change_or_cancel_never_retargets_an_approval(
    method_window, monkeypatch, change, qtbot,
):
    window = method_window
    case, panel = _choose(window)
    _settle(panel, qtbot)
    panel.use_button.click()
    _settle(panel, qtbot)
    original = case.manifest.read_bytes()
    other = _roots(window)["series-b"] / "series.toml"
    other_before = other.read_bytes()
    expected = original

    def consent(*_a, **_k):
        nonlocal expected
        if change == "series":
            window._select_series("series-b")
        elif change == "manifest":
            expected = original + b'\n[external_edit]\nkeep = true\n'
            case.manifest.write_bytes(expected)
        return "Review another calibration.", change != "cancel"

    monkeypatch.setattr(native.QInputDialog, "getMultiLineText", consent)
    warnings = []
    monkeypatch.setattr(native.QMessageBox, "warning", lambda *a: warnings.append(a))
    panel.unlock_button.click()
    _settle(panel, qtbot)
    assert case.manifest.read_bytes() == expected
    assert other.read_bytes() == other_before
    assert approval.read_approval(case.root)["locked"]
    if change == "manifest":
        assert warnings  # Original expected-bytes CAS, not a fake writer seam.


def test_new_opaque_or_unsupported_choice_cannot_bypass_the_native_bound_use(
    method_window, monkeypatch, qtbot,
):
    window = method_window
    case, panel = _choose(window)
    _settle(panel, qtbot)
    original = case.manifest.read_bytes()
    panel.mass_calibration.setText("missing-frozen.json")
    _settle(panel, qtbot)
    assert not panel.use_button.isEnabled()
    assert "missing-frozen" in panel.approval_summary.text()
    warnings = []
    monkeypatch.setattr(native.QMessageBox, "warning", lambda *a: warnings.append(a))
    window._save_method_choice(replace(case.selection, mass_calibration="missing-frozen.json"))
    assert warnings and case.manifest.read_bytes() == original
    assert approval.read_approval(case.root) is None


def test_owned_verification_is_off_ui_and_stale_results_cannot_enable_use(
    method_window, monkeypatch, qtbot,
):
    window = method_window
    entered, release = threading.Event(), threading.Event()
    original = approval.selection_binding
    threads = []

    def blocked(*args, **kwargs):
        threads.append(QThread.currentThread() == window.thread())
        if threads[-1]:
            raise AssertionError("Asset verification ran on the GUI thread")
        entered.set()
        if not release.wait(5):
            raise RuntimeError("Finite verification release missing")
        return original(*args, **kwargs)

    monkeypatch.setattr(approval, "selection_binding", blocked)
    case, panel = _choose(window)
    before = case.manifest.read_bytes()
    try:
        qtbot.waitUntil(entered.is_set)
        assert panel.calls.preview.busy and panel.calls.thread is None and not panel.use_button.isEnabled()
        assert panel.can_close()
        panel.calls.stop_pending()
        assert not panel.calls.preview.busy
        panel.mass_calibration.setText("missing-frozen.json")
        release.set()
        _settle(panel, qtbot)
        assert not panel.use_button.isEnabled() and panel.calls.binding is None
        assert "missing-frozen" in panel.approval_summary.text()
        assert case.manifest.read_bytes() == before
        panel.mass_calibration.setText(case.selection.mass_calibration)
        _settle(panel, qtbot)
        assert panel.use_button.isEnabled()
        panel.use_button.click()
        _settle(panel, qtbot)
        assert approval.read_approval(case.root, verify=False)["locked"]
        assert threads and all(main is False for main in threads)
    finally:
        release.set()
        _settle(panel, qtbot)


@pytest.mark.parametrize("change", ["source", "settings", "unchanged"])
def test_actual_action_boundary_rechecks_sources_then_continues_only_unchanged_request(
    method_window, monkeypatch, qtbot, change,
):
    """Real source check; only the final action dispatch is a finite capture."""
    window = method_window
    case, panel = _choose(window)
    _settle(panel, qtbot)
    panel.use_button.click()
    _settle(panel, qtbot)
    assert panel.calls.ready()
    calls = []
    if change == "unchanged":
        # A transient mismatch dialog is not an editable run/export setting.
        transient = QDialog(window)
        transient_field = QLineEdit(transient)
        snapshot = window._method_action_snapshot()
        transient_field.setText("a modal explanation")
        assert window._same_method_action_snapshot(snapshot)
        transient.deleteLater()
        # Even a valid alternate frozen file cannot inherit the saved lock.
        alternative = case.calibration.with_name("other-frozen.json")
        alternative.write_bytes(case.calibration.read_bytes())
        panel.mass_calibration.setText(alternative.relative_to(case.root).as_posix())
        _settle(panel, qtbot)
        window._invoke_method_action(
            "phase_model", "Export phase-model analysis results",
            lambda: calls.append("wrong pre-call selection"), (), {},
        )
        _settle(panel, qtbot)
        assert calls == []
        assert "selection differs" in panel.calls.error
        panel.mass_calibration.setText(case.selection.mass_calibration)
        _settle(panel, qtbot)
    entered, release = threading.Event(), threading.Event()
    original = approval.read_approval

    def checked(*args, **kwargs):
        assert QThread.currentThread() != window.thread()
        entered.set()
        if not release.wait(5):
            raise RuntimeError("Finite action-check release missing")
        return original(*args, **kwargs)

    monkeypatch.setattr(approval, "read_approval", checked)
    if change == "source":
        case.calibration.write_bytes(case.calibration.read_bytes() + b" ")
        # No force-refresh or cache invalidation: a synchronous locked-choice
        # predicate must never grant the old cached result as action authority.
        assert not window._confirm_method_action("phase_model", "Export phase-model analysis results")
    try:
        window._invoke_method_action(
            "phase_model", "Export phase-model analysis results",
            lambda value: calls.append((value, QThread.currentThread() == window.thread())),
            ("exact original action",), {},
        )
        qtbot.waitUntil(entered.is_set)
        assert not panel.can_close() and calls == []
        if change == "settings":
            window.selected_checkbox_id = "changed while reading"
        release.set()
        _settle(panel, qtbot)
        assert calls == ([("exact original action", True)] if change == "unchanged" else [])
        assert window._pending_method_action is None
        assert original(case.root, verify=False)["locked"]
    finally:
        release.set()
        _settle(panel, qtbot)


def test_locked_native_action_continues_after_byte_preserving_results_move(method_window, monkeypatch, qtbot, tmp_path):
    from darsia.fluidflower import series_storage_migration

    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    window = method_window
    case = bound_products(_roots(window)["series-a"])
    source, target = case.root / "results", case.root.parent / "Results"
    model = source / "models" / case.model.name
    model.parent.mkdir(parents=True)
    case.model.rename(model)
    calibration = source / case.calibration.name
    case.calibration.rename(calibration)
    document = tomllib.loads(case.manifest.read_text())
    document.setdefault("paths", {})["results"] = "results"
    case.manifest.write_text(toml.dumps(document))
    window._refresh_method_state(force=True)
    panel = window.method_panel
    panel.phase_model_button.click()
    panel.model_version.setText(model.relative_to(case.root).as_posix())
    panel.mass_calibration.setText(calibration.relative_to(case.root).as_posix())
    _settle(panel, qtbot)
    panel.use_button.click()
    _settle(panel, qtbot)
    locked = approval.read_approval(case.root)
    plan = series_storage_migration.preview_results_migration(case.root, case.root.parent, target)
    series_storage_migration.apply_results_migration(plan, writers_quiescent=True)
    window._refresh_method_state(force=True)
    _settle(panel, qtbot)
    assert panel.calls.ready() and not panel.calls.error and approval.read_approval(case.root) == locked
    calls = []
    window._invoke_method_action("phase_model", "Export phase-model analysis results",
                                lambda: calls.append("original action"), (), {})
    _settle(panel, qtbot)
    assert calls == ["original action"] and window._pending_method_action is None
    assert approval.read_approval(case.root) == locked
