from PySide6.QtCore import Qt

from darsia.fluidflower.refit_job_panel import RefitJobPanel
from darsia.fluidflower.shared_calibration_cohort import SharedCalibrationCohort, read_mass_cohort
from tests.unit.test_shared_calibration_cohort import series


def test_shared_selector_saves_explicitly_and_hides_for_individual(qtbot, tmp_path, monkeypatch):
    _document, raw = series(tmp_path)
    panel = RefitJobPanel()
    qtbot.addWidget(panel)
    widget = SharedCalibrationCohort(panel)
    qtbot.addWidget(widget)
    widget.set_series(tmp_path)
    widget.show()
    qtbot.waitUntil(lambda: widget.snapshot is not None)
    assert widget.table.rowCount() == 3
    assert (tmp_path / "series.toml").read_bytes() == raw
    panel.calibration_scope_combo.setCurrentIndex(1)
    assert widget.isHidden()
    panel.calibration_scope_combo.setCurrentIndex(0)
    assert not widget.isHidden()
    widget.table.item(1, 1).setCheckState(Qt.CheckState.Unchecked)
    assert (tmp_path / "series.toml").read_bytes() == raw
    def call(operation, ready):
        ready(operation())
        return True
    monkeypatch.setattr(panel, "_call", call)
    widget.save_button.click()
    assert read_mass_cohort(tmp_path)["roles"]["fit"] == ["one"]
    assert "selection saved" in widget.status.text()
    assert "No job started" in widget.status.text()
