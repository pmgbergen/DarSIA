"""Protocol View reads the selected worksheet inside the native main window."""
from types import SimpleNamespace

import openpyxl
import pytest
import toml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QStackedWidget, QVBoxLayout, QPushButton, QTableWidget

from darsia.gui.ui.file_dialog import FileDialogHelper
from darsia.gui.ui.table_viewer import TableViewerDialog, load_table_reference


@pytest.fixture
def source(tmp_path):
    file = tmp_path / "protocol.xlsx"
    book = openpyxl.Workbook()
    first = book.active
    first.title = "r1"
    first.append(["datetime", "weight_g"])
    first.append(["2023-01-01T00:00:00+01:00", 1.5])
    second = book.create_sheet("r2")
    second.append(["datetime", "weight_g"])
    second.append(["2023-02-02T01:00:00+01:00", 42.5])
    book.save(file)
    return file


def test_literal_pair_selects_only_the_requested_sheet_and_does_not_write(source):
    original = source.read_bytes()
    frame = load_table_reference(str([str(source), "r2"]))
    assert frame["weight_g"].tolist() == [42.5]
    assert frame["datetime"].tolist() == ["2023-02-02T01:00:00+01:00"]
    assert source.read_bytes() == original


def test_native_view_is_embedded_read_only_and_back_returns_to_settings(qtbot, source, monkeypatch):
    window = QWidget(); qtbot.addWidget(window)
    window.config_dict = {"protocols": {"overflow": [str(source), "r2"]}}
    window.config_file = ""
    window.chosen_files = {}
    window.settings_factory = SimpleNamespace(get_value=lambda dictionary, key: dictionary["protocols"]["overflow"])
    logs = []
    window.print_log = logs.append
    window.content_stack = QStackedWidget()
    layout = QVBoxLayout(window); layout.addWidget(window.content_stack)
    previous = QWidget()
    window.content_stack.addWidget(previous)
    helper = FileDialogHelper(window)
    _label, field = helper.create_file_chooser("Overflow", "", False,
        dict(key="protocols.overflow", name="Overflow", table_viewer="csv"))
    qtbot.addWidget(field)
    def forbidden(*args, **kwargs):
        raise AssertionError("Native table View must not open a separate dialog")
    monkeypatch.setattr(TableViewerDialog, "exec", forbidden)
    original = source.read_bytes()
    next(button for button in field.findChildren(QPushButton) if button.text() == "View").click()
    viewer = window.content_stack.currentWidget()
    assert isinstance(viewer, TableViewerDialog)
    assert not viewer.isWindow()
    table = viewer.findChild(QTableWidget)
    assert table.rowCount() == 1
    assert table.item(0, 1).text() == "42.5"
    assert not table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    next(button for button in viewer.findChildren(QPushButton) if button.text() == "Back").click()
    assert window.content_stack.currentWidget() is previous
    assert source.read_bytes() == original
    assert not logs


def test_pair_parsing_never_evaluates_an_expression(tmp_path):
    sentinel = tmp_path / "do-not-create"
    expression = f"[__import__('pathlib').Path({str(sentinel)!r}).touch(), 'r2']"
    with pytest.raises(ValueError, match="file path or"):
        load_table_reference(expression)
    assert not sentinel.exists()


def test_relative_workbook_reference_follows_the_series_folder_alias(source, monkeypatch):
    root = source.parent
    registry = root / "registry.toml"
    registry.write_text(toml.dumps(dict(schema_version=1,
        series=[dict(id="test", root=str(root))])), encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    (root / "series.toml").write_text(toml.dumps(dict(schema_version=1, series_id="test",
        storage=dict(path_aliases={str(root / "oldbooks"): str(root)}))), encoding="utf-8")
    config = root / "run.toml"
    config.write_text("[protocols]\n", encoding="utf-8")
    frame = load_table_reference(str(["oldbooks/protocol.xlsx", "r2"]), config=config)
    assert frame["weight_g"].tolist() == [42.5]
