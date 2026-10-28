"""Workbook conversion is an embedded, explicit staging action."""
from PySide6.QtWidgets import QWidget, QStackedWidget, QVBoxLayout

from darsia.fluidflower.protocol_workbooks import ProtocolSheet, ProtocolWorkbookPlan
from darsia.fluidflower import protocol_workbooks_ui as module


def test_embedded_preview_then_explicit_create_without_activation(qtbot, tmp_path, monkeypatch):
    window = QWidget(); qtbot.addWidget(window)
    layout = QVBoxLayout(window)
    window.content_stack = QStackedWidget()
    window.settings_scroll_area = QWidget()
    window.content_stack.addWidget(window.settings_scroll_area)
    layout.addWidget(window.content_stack)
    window._method_series_root = lambda: tmp_path
    plan = ProtocolWorkbookPlan(tmp_path, ("r1",),
        (ProtocolSheet("overflow", "r1", ("datetime", "volume_ml"),
            (("2023-01-01T01:00:00+01:00", "2.0"),)),),
        {"r1": {"overflow": "r1"}}, 1)
    monkeypatch.setattr(module, "plan_protocol_workbooks", lambda root: plan)
    panel = module.install_protocol_workbooks(window)
    assert module.open_protocol_workbooks(window) is panel
    assert window.content_stack.currentWidget() is panel
    assert not panel.isWindow()
    assert panel.plan is None
    panel.preview_button.click()
    qtbot.waitUntil(lambda: panel._call is None)
    assert panel.table.rowCount() == 1
    assert not (tmp_path / "protocols").exists()
    assert panel.create_button.isEnabled()
    panel.create_button.click()
    qtbot.waitUntil(lambda: panel._call is None)
    assert panel.package["activated"] is False
    assert (tmp_path / "protocols/workbooks/overflow_protocol.xlsx").is_file()
    assert "[protocols]" in panel.proposed.toPlainText()
    assert "not been switched" in panel.status.text()
    assert panel.activate_button.isEnabled()
    def activate(current, package):
        assert current is plan
        package["activated"] = True
        return dict(activated=True)
    monkeypatch.setattr(module, "activate_protocol_workbooks", activate)
    with qtbot.waitSignal(panel.activated) as emitted:
        panel.activate_button.click()
    qtbot.waitUntil(lambda: panel._call is None)
    assert emitted.args == [str(tmp_path)]
    assert not panel.activate_button.isEnabled()
    assert panel.archive_button.isEnabled()
    from darsia.fluidflower import protocol_workbook_archive
    def archive(current, package):
        assert current is plan and package["activated"]
        package["archived"] = True
        return dict(folder=tmp_path / "Archive/ProtocolMigration_test", moves=[("old", "archive")])
    monkeypatch.setattr(protocol_workbook_archive, "archive_protocol_sources", archive)
    with qtbot.waitSignal(panel.archived) as archived:
        panel.archive_button.click()
    qtbot.waitUntil(lambda: panel._call is None)
    assert archived.args == [str(tmp_path)]
    assert not panel.archive_button.isEnabled()
    panel.back_button.click()
    assert window.content_stack.currentWidget() is window.settings_scroll_area
