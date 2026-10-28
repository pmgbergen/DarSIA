"""Synthetic kept-widget tests; compile-only preparation, not runtime evidence."""
import json

from PySide6.QtWidgets import QComboBox, QDialog, QLabel, QWidget

from darsia.fluidflower.native_phase_steps import saved_method_text
from darsia.fluidflower.series_method import MethodSelection
from .test_native_phase_steps import setup


def test_step6_uses_saved_method_refs_even_when_no_normalisation_is_ready(setup, tmp_path):
    manifest = tmp_path / "series.toml"
    manifest.write_bytes(b'[method]\nactive="phase_model"\n[method.phase_model]\nmodel_version="saved-M"\nmass_calibration="saved-K"\n')
    before = manifest.read_bytes()
    setup.steps.series_root = tmp_path
    setup.steps.refresh()
    assert setup.steps.saved_method_label.text() == "phase_model · saved-M · saved-K"
    assert setup.normalisation.ready is False
    assert "choose a model in Basis" in setup.steps.review_selection_notice.text()
    assert "review_reference" not in manifest.read_bytes().decode()
    assert setup.steps.layouts[6].indexOf(setup.steps.saved_method_label) == 0
    assert manifest.read_bytes() == before
    manifest.write_bytes(b'[method]\nactive="colour_path"\n[method.colour_path]\ncalibration="other-series-C"\n')
    setup.steps.refresh()
    assert setup.steps.saved_method_label.text() == "colour_path · other-series-C"


def test_saved_method_missing_values_remain_missing_not_defaults():
    assert saved_method_text(MethodSelection(active="phase_model")) == "phase_model · model not referenced · calibration not referenced"
    assert saved_method_text(MethodSelection()) == "Method: not chosen"
    assert saved_method_text(MethodSelection(error="bad manifest")) == "Saved method unavailable — bad manifest"


def test_legacy_explicit_K_view_does_not_configure_calculation(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower.tools import phase_studio_series_ui as legacy
    from darsia.fluidflower.mass_calibration_display import MassCalibrationDisplay
    path = tmp_path / "frozen.json"
    path.write_bytes(json.dumps({"contract": "ffac-common-mass-round3-frozen-v1", "frozen": True,
                                "parameters": {"gain": 23.5, "beta": 0.65}}).encode())
    before = path.read_bytes()
    owner = QWidget()
    qtbot.addWidget(owner)
    owner.calibration_file = "previous-calibration"
    owner.scope = QComboBox(owner)
    owner.scope.addItem("local", "local")
    owner.overflow_mode = QComboBox(owner)
    owner.overflow_mode.addItem("none", "none")
    owner.calibration_label = QLabel("original", owner)
    owner.status = QLabel("original", owner)
    monkeypatch.setattr(legacy.QFileDialog, "getOpenFileName", lambda *_a: (str(path), "JSON"))
    dialogs = []
    def opened(dialog):
        view = dialog.findChild(MassCalibrationDisplay)
        dialogs.append([(view.parameters.item(i, 0).text(), view.parameters.item(i, 1).text())
                        for i in range(view.parameters.rowCount())])
        assert not view.manual_button.isEnabled() and not view.automatic_button.isEnabled()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", opened)
    legacy.SeriesMassWorkspace.load_calibration(owner)
    assert dict(dialogs[0])["Gain"] == "23.5"
    assert dict(dialogs[0])["Dilution factor"] == "Not in this calibration document"
    assert owner.calibration_file == "previous-calibration"
    assert owner.scope.currentData() == "local" and owner.overflow_mode.currentData() == "none"
    assert owner.calibration_label.text() == "original"
    assert path.read_bytes() == before
