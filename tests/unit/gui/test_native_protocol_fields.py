"""Protocol explanations report authored sources without inventing or saving them."""

from copy import deepcopy
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QWidget

from darsia.fluidflower import native_protocol_fields as module
from darsia.fluidflower import storage_paths


@pytest.fixture
def inputs(qtbot, monkeypatch):
    monkeypatch.setattr(storage_paths, "storage_path_remapper", lambda: lambda value: value)
    pages = {}
    for name in ("data", "imaging", "injection", "pressure_temperature", "exclusions", "overflow"):
        page = QWidget()
        QFormLayout(page)
        qtbot.addWidget(page)
        pages[name] = page
    fields = {}
    data_edits = []
    for folder in ("experiment-part-one", "experiment-part-two"):
        holder = QWidget()
        layout = QHBoxLayout(holder)
        edit = QLineEdit(folder)
        layout.addWidget(edit)
        pages["data"].layout().addRow(holder)
        data_edits.append(edit)
    fields["data.folders"] = data_edits
    for key, page_name in (("imaging", "imaging"), ("blacklist", "exclusions")):
        rows = []
        for folder in ("experiment-part-one", "experiment-part-two"):
            folder_edit, edit = QLineEdit(folder), QLineEdit()
            pages[page_name].layout().addRow(folder_edit, edit)
            rows.append((folder_edit, edit))
        fields["protocols." + key] = {"path_map": True, "rows": rows}
    for key in ("injection", "pressure_temperature", "overflow"):
        edit = QLineEdit()
        pages[key].layout().addRow("Protocol", edit)
        fields["protocols." + key] = edit
    for key, page_key in (("injection_mode", "injection"), ("pressure_temperature_mode", "pressure_temperature")):
        mode = QComboBox()
        mode.addItems(["constant", "Load from CSV"])
        pages[page_key].layout().addRow("Generation mode", mode)
        fields["protocols." + key] = mode
    window = SimpleNamespace(settings_inputs=fields, config_file=None, config_dict={"untouched": [1, 2, 3]})
    return window, pages


def test_empty_optional_sources_remain_empty_and_roles_are_not_inferred(inputs, qtbot):
    window, pages = inputs
    before = deepcopy(window.config_dict)
    notes = module.decorate_protocol_fields(window, pages)
    qtbot.waitUntil(lambda: bool(notes.rows) and not notes.reader.busy)
    statuses = [label.text() for _, label in notes.rows.values()]
    assert all("Not selected" in text for text in statuses)
    assert any("Excluded images" in text and "optional" in text for text in statuses)
    assert all(not edit.text() for _, edit in window.settings_inputs["protocols.blacklist"]["rows"])
    folder_labels = [label.text() for label in pages["data"].findChildren(QLabel)]
    assert "Experiment images 1" in folder_labels and "Experiment images 2" in folder_labels
    assert any("not automatically a washing-image folder" in text for text in folder_labels)
    assert window.config_dict == before


def test_file_presence_is_checked_off_gui_thread_once_per_file(inputs, qtbot, monkeypatch, tmp_path):
    window, pages = inputs
    workbook = tmp_path / "protocols.xlsx"
    workbook.write_bytes(b"not parsed as a workbook")
    for index, (_, edit) in enumerate(window.settings_inputs["protocols.imaging"]["rows"]):
        edit.setText(repr([str(workbook), "run-" + str(index)]))
    checked = []
    gui_thread = get_ident()
    exists = Path.is_file
    def is_file(path):
        assert get_ident() != gui_thread
        checked.append(path)
        return exists(path)
    monkeypatch.setattr(Path, "is_file", is_file)
    notes = module.decorate_protocol_fields(window, pages)
    qtbot.waitUntil(lambda: bool(notes.rows) and not notes.reader.busy)
    assert checked == [workbook]
    labels = [notes.rows[("protocols.imaging", number)][1].text() for number in (0, 1)]
    assert all("File found" in text and "worksheet:" in text and "not checked" in text for text in labels)


def test_missing_or_malformed_sources_are_visible_and_never_replaced(inputs, qtbot, tmp_path):
    window, pages = inputs
    missing = str(tmp_path / "missing.csv")
    window.settings_inputs["protocols.injection"].setText(missing)
    window.settings_inputs["protocols.pressure_temperature"].setText("['invalid-pair']")
    notes = module.decorate_protocol_fields(window, pages)
    qtbot.waitUntil(lambda: bool(notes.rows) and not notes.reader.busy)
    assert "File missing" in notes.rows[("protocols.injection", 0)][1].text()
    assert "Unavailable:" in notes.rows[("protocols.pressure_temperature", 0)][1].text()
    assert window.settings_inputs["protocols.injection"].text() == missing
    assert window.settings_inputs["protocols.pressure_temperature"].text() == "['invalid-pair']"


def test_editing_invalidates_file_presence_until_explicit_check(inputs, qtbot, tmp_path):
    window, pages = inputs
    path = tmp_path / "injection.csv"
    path.write_text("time,rate")
    edit = window.settings_inputs["protocols.injection"]
    edit.setText(str(path))
    notes = module.decorate_protocol_fields(window, pages)
    qtbot.waitUntil(lambda: bool(notes.rows) and not notes.reader.busy)
    label = notes.rows[("protocols.injection", 0)][1]
    assert "File found" in label.text()
    edit.setText(str(tmp_path / "other.csv"))
    assert "Not checked after editing" in label.text()
    edit.editingFinished.emit()
    qtbot.waitUntil(lambda: not notes.reader.busy)
    assert "File missing" in label.text()


def test_workbook_hides_only_template_rows_and_csv_restores_them(inputs, qtbot, tmp_path):
    window, pages = inputs
    before = deepcopy(window.config_dict)
    injection = window.settings_inputs["protocols.injection"]
    injection.setText(repr([str(tmp_path / "protocol.xlsx"), "test-run"]))
    notes = module.decorate_protocol_fields(window, pages)
    qtbot.waitUntil(lambda: bool(notes.rows) and not notes.reader.busy)
    injection_mode = window.settings_inputs["protocols.injection_mode"]
    pressure_mode = window.settings_inputs["protocols.pressure_temperature_mode"]
    assert not pages["injection"].layout().isRowVisible(injection_mode)
    assert not pages["pressure_temperature"].layout().isRowVisible(pressure_mode)
    assert pages["injection"].layout().isRowVisible(injection)
    assert "Workbook-backed" in notes.notes["injection"].text()
    injection.setText(str(tmp_path / "protocol.csv"))
    assert pages["injection"].layout().isRowVisible(injection_mode)
    assert pages["pressure_temperature"].layout().isRowVisible(pressure_mode)
    assert injection_mode.currentText() == "constant" and pressure_mode.currentText() == "constant"
    assert window.config_dict == before


def test_deferred_decoration_keeps_its_original_form_input_snapshot(inputs, qtbot):
    window, pages = inputs
    old_inputs = window.settings_inputs
    notes = module.decorate_protocol_fields(window, pages)
    unrelated = QLineEdit("different-run.xlsx")
    qtbot.addWidget(unrelated)
    window.settings_inputs = {"protocols.injection": unrelated}
    qtbot.waitUntil(lambda: bool(notes.rows) and not notes.reader.busy)
    assert notes.inputs is old_inputs
    assert notes.rows[("protocols.injection", 0)][0] is old_inputs["protocols.injection"]
    unrelated.setText("changed.xlsx")
    assert "Not selected" in notes.rows[("protocols.injection", 0)][1].text()


@pytest.mark.parametrize("value", ["", "No file chosen", "['', 'run']"])
def test_empty_references_require_no_storage_lookup(value, monkeypatch):
    monkeypatch.setattr(storage_paths, "storage_path_remapper", lambda: pytest.fail("Nothing is configured"))
    assert module._check_sources([("optional", value)], None, lambda: False) == {"optional": "Not selected"}
