"""File actions give visible feedback without probing storage on routine UI updates."""
from pathlib import Path

import pytest
from PySide6.QtWidgets import QTableWidgetItem

from darsia.fluidflower.ui import model_training as ui
from .test_review_session_refresh import studio  # noqa: F401


def _select_history_folder(page, folder):
    entry = dict(root=str(folder) if folder is not None else "", model_id="saved-model")
    page.visible_model_history = [entry]
    page.history_table.setRowCount(1)
    page.history_table.setItem(0, 0, QTableWidgetItem("Saved model"))
    page.history_table.selectRow(0)
    assert page._selected_model_history() == entry
    assert page.open_model_folder.isEnabled()
    return entry


@pytest.mark.parametrize("failure", ["missing_locator", "missing", "storage_error", "desktop_rejected"])
def test_open_model_folder_reports_failure_and_keeps_selection(
    studio, monkeypatch, tmp_path, failure,
):
    page, _initial = studio
    folder = tmp_path / "selected-model"
    if failure not in {"missing", "missing_locator"}:
        folder.mkdir()
    entry = _select_history_folder(page, None if failure == "missing_locator" else folder)
    warnings, opened = [], []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda _parent, title, message: warnings.append((title, message)))

    def desktop_open(url):
        opened.append(url.toLocalFile())
        return False

    monkeypatch.setattr(ui.QDesktopServices, "openUrl", desktop_open)
    if failure == "storage_error":
        original = Path.is_dir

        def unavailable(path):
            if path == folder:
                raise OSError("The storage connection is unavailable")
            return original(path)

        monkeypatch.setattr(Path, "is_dir", unavailable)
    page.open_model_folder.click()
    assert len(warnings) == 1
    if failure == "missing_locator":
        assert "no recorded folder" in warnings[0][1]
    else:
        assert str(folder) in warnings[0][1]
    if failure == "storage_error":
        assert "OSError: The storage connection is unavailable" in warnings[0][1]
    elif failure == "desktop_rejected":
        assert "desktop could not open" in warnings[0][1]
    elif failure == "missing":
        assert "missing or unavailable" in warnings[0][1]
    assert bool(opened) == (failure == "desktop_rejected")
    assert page._selected_model_history() == entry
    assert page.open_model_folder.isEnabled()


@pytest.mark.parametrize("files_present", [False, True])
def test_action_updates_do_not_stat_decisions_but_click_still_checks_evidence(
    studio, monkeypatch, tmp_path, files_present,
):
    page, initial = studio
    decisions = tmp_path / "decisions.json"
    combined = tmp_path / "prepared"
    manifest = combined / "postflight_manifest.json"
    if files_present:
        decisions.write_text("{}", encoding="utf-8")
        combined.mkdir()
        manifest.write_text("{}", encoding="utf-8")
    initial.update(decision_path=str(decisions), combined_root=str(combined))
    probes = []
    original = Path.is_file
    click_started = False

    def observe_file_probe(path):
        if path in {decisions, manifest}:
            assert click_started, "Routine GUI updates queried saved evidence storage"
            probes.append(path)
        return original(path)

    monkeypatch.setattr(Path, "is_file", observe_file_probe)
    warnings = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda _parent, _title, message: warnings.append(message))
    page._update_action_states()
    page.reviewer.setText("reviewer")
    page._session_changed()
    assert not probes
    assert page.add_session_evidence.isEnabled()
    assert page.add_selected_session_button.isEnabled()
    assert "Check that the saved decisions" in page.add_selected_session_button.toolTip()
    click_started = True
    page.add_session_evidence.click()
    assert probes[0] == decisions
    if files_present:
        assert probes == [decisions, manifest]
        assert not warnings
        assert page.bundles.toPlainText() == f"{decisions.resolve()} | {manifest.resolve()}"
        assert page.tabs.currentWidget() is page.training_tab
    else:
        assert warnings and "both saved decisions" in warnings[0]
        assert not page.bundles.toPlainText()


def test_session_without_recorded_decisions_path_has_no_add_action(studio):
    page, initial = studio
    initial["decision_path"] = " "
    page._update_action_states()
    assert not page.add_session_evidence.isEnabled()
    assert not page.add_selected_session_button.isEnabled()
    assert "recorded decisions path" in page.add_selected_session_button.toolTip()


@pytest.mark.parametrize("failure_at", ["availability", "append_resolution"])
def test_add_session_reports_storage_failure_without_changing_inputs(studio, monkeypatch, tmp_path, failure_at):
    page, initial = studio
    decisions = tmp_path / "saved-decisions.json"
    combined = tmp_path / "saved-campaign"
    combined.mkdir()
    decisions.write_text("{}", encoding="utf-8")
    (combined / "postflight_manifest.json").write_text("{}", encoding="utf-8")
    initial.update(decision_path=str(decisions), combined_root=str(combined))
    page.bundles.setPlainText("retained manual input")
    page._update_action_states()
    original_tab = page.tabs.currentWidget()
    warnings = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda _parent, title, message: warnings.append((title, message)))
    method = "is_file" if failure_at == "availability" else "resolve"
    original = getattr(Path, method)

    def unavailable(path, *args, **kwargs):
        if path == decisions:
            raise OSError("The saved evidence storage is unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, method, unavailable)
    page.add_selected_session_button.click()
    assert warnings == [("Evidence package unavailable",
        "Could not access the selected session's saved evidence. "
        "Check its storage connection and try again.\n"
        "OSError: The saved evidence storage is unavailable")]
    assert page._selected_session() == initial
    assert page.bundles.toPlainText() == "retained manual input"
    assert page.tabs.currentWidget() is original_tab
    assert page.add_selected_session_button.isEnabled()
