"""Build published dataset: the Results and export action and its dialog (offscreen Qt)."""

import pytest

from darsia.fluidflower.published_dataset import builder, scan
from darsia.fluidflower.published_dataset.build_dialog import BuildDatasetDialog
from darsia.fluidflower.published_dataset.manifest import load_manifest
from tests.unit.published_dataset.synthetic import make_series
from tests.unit.published_dataset.test_export_options import software_inventory


@pytest.fixture(autouse=True)
def fictional_machine(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "machine_names", lambda: (
        scan.ScanName("example-build-host", "build machine host name"),))
    monkeypatch.setenv("DARSIA_PUBLICATION_SETTINGS", str(tmp_path / "user-settings" / "publication.toml"))


def _dialog(qtbot, series, output):
    dialog = BuildDatasetDialog(series.series)
    qtbot.addWidget(dialog)
    dialog.job_edit.setText(str(series.job))
    dialog.output_edit.setText(str(output))
    return dialog


def test_dialog_checks_then_builds_in_a_worker_thread(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "reference_machine", lambda request, job, report: (
        {"label": "reference machine", "backend": "cpu", "precision": "float32", "threads": 2,
         "seconds_per_frame": 6.0, "peak_memory_bytes": request.peak_memory_bytes,
         "peak_memory_basis": request.peak_memory_basis}, None))
    series = make_series(tmp_path)
    dialog = _dialog(qtbot, series, tmp_path / "record")
    inventory, _ = software_inventory(series, tmp_path / "software-input")
    dialog.software_edit.setText(str(inventory))
    dialog.memory_edit.setText("1.5")
    dialog.memory_basis_edit.setText("measured committed memory")
    assert dialog.request().peak_memory_bytes == int(1.5 * 1024**3)
    assert dialog.start(check_only=True)
    assert not dialog.start(check_only=True)  # One run at a time.
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=20000)
    assert dialog.report_view.toPlainText().startswith("Checked: nothing refuses the record.")
    assert not (tmp_path / "record").exists()
    assert dialog.start(check_only=False)
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=20000)
    assert dialog.report_view.toPlainText().startswith("Record built.")
    assert dialog.last_report.built and load_manifest(tmp_path / "record")["machine"]["peak_memory_bytes"] == int(
        1.5 * 1024**3)
    assert dialog.check_button.isEnabled() and dialog.build_button.isEnabled()


def test_dialog_explains_missing_choices_and_simulator_resolution(qtbot, tmp_path):
    dialog = BuildDatasetDialog(tmp_path)
    qtbot.addWidget(dialog)
    assert not dialog.resolution_edit.isEnabled()
    assert not dialog.start(check_only=True)
    assert dialog.report_view.toPlainText() == "Choose the production job and an output folder"
    dialog.job_edit.setText(str(tmp_path / "job"))
    dialog.output_edit.setText(str(tmp_path / "out"))
    dialog.record_combo.setCurrentIndex(dialog.record_combo.findData("S"))
    assert dialog.resolution_edit.isEnabled()
    assert not dialog.start(check_only=False)
    assert "no default" in dialog.report_view.toPlainText()
    dialog.resolution_edit.setText("1")
    request = dialog.request()
    assert (request.record, request.simulator_resolution_mm, request.proof) == ("S", 1.0, None)
    assert request.simulator_geometry is None and dialog.geometry_edit.isEnabled()
    assert request.require_simulator_geometry
    dialog.geometry_edit.setText(str(tmp_path / "geometry_request.json"))
    assert dialog.request().simulator_geometry == tmp_path / "geometry_request.json"
    dialog.record_combo.setCurrentIndex(dialog.record_combo.findData("R"))
    assert not dialog.geometry_edit.isEnabled() and dialog.request().simulator_geometry is None


def test_export_dialog_uses_mm_and_preserves_independent_grid_choices(qtbot, tmp_path):
    from PySide6.QtWidgets import QLabel, QScrollArea
    dialog = BuildDatasetDialog(tmp_path)
    qtbot.addWidget(dialog)
    dialog.job_edit.setText(str(tmp_path / "job"))
    dialog.output_edit.setText(str(tmp_path / "output"))
    assert not any(label.text() == "Tier" for label in dialog.findChildren(QLabel))
    assert [dialog.mass_resolution_combo.itemText(i) for i in range(4)] == ["1 mm", "2 mm", "5 mm", "10 mm"]
    assert dialog.request().tier is None
    assert dialog.request().software_mode == "references"
    assert dialog.findChildren(QScrollArea)
    dialog.software_folder_edit.setText(str(tmp_path / "software"))
    dialog.software_code_edit.setText("https://example.org/darsia")
    dialog.software_python_edit.setText("3.13.9")
    assert dialog.request().software_folder == tmp_path / "software"
    assert dialog.request().software_code_url == "https://example.org/darsia"
    dialog.software_combo.setCurrentIndex(1)
    assert dialog.request().software_mode == "include"
    dialog.record_combo.setCurrentIndex(dialog.record_combo.findData("S"))
    dialog.resolution_edit.setText("5")
    dialog.mass_resolution_combo.setCurrentIndex(dialog.mass_resolution_combo.findData(2))
    request = dialog.request()
    assert request.simulator_resolution_mm == 5 and request.mass_map_resolution_mm == 2
    assert request.software_mode is None and not dialog.software_edit.isEnabled()


def test_media_choice_opens_original_reader_without_starting_a_build(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.production_results_view import ProductionResultsPage
    opened = []
    monkeypatch.setattr(ProductionResultsPage, "open_job", lambda self, path: opened.append(path))
    dialog = BuildDatasetDialog(tmp_path)
    qtbot.addWidget(dialog)
    dialog.job_edit.setText(str(tmp_path / "job"))
    dialog.record_combo.setCurrentIndex(dialog.record_combo.findData("media"))
    assert not dialog.check_button.isEnabled()
    assert dialog.start(check_only=False)
    assert opened == [tmp_path / "job"] and dialog.thread is None
    assert isinstance(dialog.media_page, ProductionResultsPage)
    dialog.media_dialog.close()


def test_staged_inventory_and_original_receipts_are_optional_reproduction_only_choices(qtbot, tmp_path):
    dialog = BuildDatasetDialog(tmp_path)
    qtbot.addWidget(dialog)
    dialog.job_edit.setText(str(tmp_path / "job"))
    dialog.output_edit.setText(str(tmp_path / "output"))
    controls = (dialog.stage_software_edit, dialog.stage_software_button, dialog.producer_metadata_check)
    assert all(control.isEnabled() for control in controls)
    assert dialog.request().software_stage_inventory is None
    assert dialog.request().retain_producer_metadata is False
    dialog.stage_software_edit.setText("   ")
    assert dialog.request().software_stage_inventory is None
    selected = tmp_path / "stages.json"
    dialog.stage_software_edit.setText("  " + str(selected) + "  ")
    dialog.producer_metadata_check.setChecked(True)
    assert dialog.request().software_stage_inventory == selected
    assert dialog.request().retain_producer_metadata is True
    dialog.record_combo.setCurrentIndex(dialog.record_combo.findData("S"))
    dialog.resolution_edit.setText("5")
    assert not any(control.isEnabled() for control in controls)
    assert dialog.request().software_stage_inventory is None
    assert dialog.request().retain_producer_metadata is False
    dialog.record_combo.setCurrentIndex(dialog.record_combo.findData("media"))
    assert not any(control.isEnabled() for control in controls)
    with pytest.raises(ValueError, match="films and figures"):
        dialog.request()
    dialog.record_combo.setCurrentIndex(dialog.record_combo.findData("R"))
    assert all(control.isEnabled() for control in controls)
    assert dialog.request().software_stage_inventory == selected
    assert dialog.request().retain_producer_metadata is True
    assert dialog.thread is None and not (tmp_path / "output").exists()


def test_results_and_export_offers_the_action_only_in_its_view(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.tools import phase_studio_mass_analysis_workspace as module

    root = tmp_path / "series-a"
    root.mkdir()
    monkeypatch.setattr(module, "configured_series", lambda: (("series-a", "series-a", root),))
    page = module.MassAnalysisWorkspace(series="series-a", project_root=root)
    qtbot.addWidget(page)
    page.show()
    page.show_view("review")
    assert not page.publish_button.isVisible()
    page.show_view("results")
    assert page.publish_button.isVisible() and page.publish_button.text() == "Build published dataset…"
    page.publish_button.click()
    dialog = page.publish_dialog
    assert isinstance(dialog, BuildDatasetDialog) and dialog.series_root == root and dialog.isVisible()
    page.build_published_dataset()
    assert page.publish_dialog is dialog
    other = tmp_path / "series-b"
    other.mkdir()
    page.project_root = other
    page.build_published_dataset()
    assert page.publish_dialog is not dialog and page.publish_dialog.series_root == other
    page.publish_dialog.close()
    page._stop_lookups()
