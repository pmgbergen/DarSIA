from types import SimpleNamespace
from pathlib import Path
import os

import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFont, QFontDatabase, QTextDocument
from PySide6.QtWidgets import QFileDialog, QFormLayout

from darsia.fluidflower.mass_calibration_display import MassCalibrationDisplay, SavedMassCalibrationData
from darsia.fluidflower.native_mass_calibration_layout import NativeMassCalibrationDisplay


@pytest.fixture
def desktop_font(qapp):
    """Use the real Windows font when Qt offscreen exposes no system fonts."""
    original, loaded = qapp.font(), -1
    source = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/segoeui.ttf'
    if source.is_file():
        loaded = QFontDatabase.addApplicationFont(str(source))
        families = QFontDatabase.applicationFontFamilies(loaded)
        if families:
            qapp.setFont(QFont(families[0], original.pointSize()))
    yield
    qapp.setFont(original)
    if loaded >= 0:
        QFontDatabase.removeApplicationFont(loaded)


def test_native_panels_preserve_original_tables_and_explicit_series_visibility(qtbot, monkeypatch, tmp_path):
    page = NativeMassCalibrationDisplay()
    qtbot.addWidget(page)
    page.resize(1440, 1000)
    page.show()
    contexts = []
    monkeypatch.setattr(page.refit_panel, "set_series", contexts.append)
    assert list(page.sections) == ["chemistry", "job"]
    assert page.saved_calibration_page.isHidden()
    assert all(not page.sections[key].isHidden() and not page.sections[key].isEnabled()
               for key in ("chemistry", "job"))
    page.set_refit_series(tmp_path)
    assert contexts == [tmp_path]
    assert all(section.isVisible() for section in page.sections.values())
    assert page.refit_panel.isVisible()
    for table in (page.refit_panel.recipe_table, page.refit_panel.tasks):
        assert page.workflow.workspace.isAncestorOf(table)
        assert page.refit_panel.isAncestorOf(table)
    assert page.saved_calibration_page.isAncestorOf(page.parameters)
    assert page.saved_calibration_page.isAncestorOf(page.pressure_table)
    assert not page.workflow.isAncestorOf(page.parameters)
    page.saved_calibration_page.setParent(None)
    qtbot.addWidget(page.saved_calibration_page)
    page.saved_calibration_page.show()

    page.set_display(SavedMassCalibrationData((("Gain", "23.5", "Original saved source"),),
                                            "Selected saved calibration", ("Original note",)))
    page.set_pressure_sources(SimpleNamespace(rows=[SimpleNamespace(run="r1", source="Measured",
        values="1 bar", span="Whole run", status="Ready", fit="Yes", detail="Original source")],
        complete=True, notes=["Source note"]))
    assert page.parameters.item(0, 1).text() == "23.5"
    assert page.pressure_table.item(0, 1).text() == "Measured"
    assert page.pressure_box.isVisible()
    page.set_refit_series(None)
    assert contexts == [tmp_path, None]
    assert page.job_results.isHidden() and not page.sections["job"].isHidden()
    assert not page.sections["job"].isEnabled()
    assert page.parameters.isVisible() and page.pressure_table.isVisible()


def test_original_folder_and_overflow_callbacks_survive_folding(qtbot, monkeypatch, tmp_path):
    page = NativeMassCalibrationDisplay()
    qtbot.addWidget(page)
    monkeypatch.setattr(page.refit_panel, "set_series", lambda _root: None)
    page.set_refit_series(tmp_path)
    page.resize(1440, 1000)
    page.show()
    panel = page.refit_panel
    panel.series_root = tmp_path
    panel._update_actions()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(tmp_path / "chosen"))
    page.sections["job"].set_expanded(True)
    qtbot.mouseClick(panel.jobs_button, Qt.MouseButton.LeftButton)
    panel.name_edit.setText("my-calibration")
    panel.loro_check.setChecked(False)
    page.sections["job"].set_expanded(False)
    page.sections["job"].set_expanded(True)
    assert panel.jobs_edit.text() == str(tmp_path / "chosen")
    assert panel.name_edit.text() == "my-calibration" and not panel.loro_check.isChecked()
    page.sections["chemistry"].set_expanded(True)
    panel.f_check.click()
    assert panel.f_check.isChecked()
    assert "Variant" not in panel.f_check.text()
    assert panel._thread is None and panel.job_root is None


def test_overflow_parameter_and_source_controls_stay_in_model_parameters(qtbot, monkeypatch, tmp_path):
    page = NativeMassCalibrationDisplay()
    qtbot.addWidget(page)
    panel = page.refit_panel
    monkeypatch.setattr(panel, 'set_series', lambda _root: None)
    page.set_refit_series(tmp_path)
    panel.series_root = tmp_path
    panel._update_actions()
    page.show()
    overflow = page.sections['chemistry']
    assert overflow.isAncestorOf(panel.f_check)
    assert overflow.isAncestorOf(panel.source_folder_button)
    assert overflow.isAncestorOf(panel.source_folder_panel)
    assert panel.source_folder_panel.isHidden()
    overflow.set_expanded(True)
    qtbot.mouseClick(panel.source_folder_button, Qt.MouseButton.LeftButton)
    assert panel.source_folder_panel.isVisible()
    assert 'Choose the reviewed overflow inputs file first' in panel.source_folder_status.text()
    overflow.set_expanded(False)
    assert not panel.source_folder_panel.isVisible()
    overflow.set_expanded(True)
    assert panel.source_folder_panel.isVisible()


def test_actual_native_main_window_starts_without_a_selected_series(qtbot, monkeypatch, tmp_path, desktop_font):
    from darsia.fluidflower import native_darsia_app as native
    monkeypatch.setattr(native, 'configured_series', lambda: ())
    monkeypatch.setattr(native, 'ProjectRegistry', lambda: SimpleNamespace(entries=lambda: []))
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY', str(tmp_path/'unregistered.toml'))
    window = native.NativeFluidFlowerMainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(window.isVisible)
    display = window.mass_calibration_display
    assert isinstance(display, NativeMassCalibrationDisplay)
    assert display.refit_panel.series_root is None
    assert display.sections['chemistry'].isAncestorOf(display.refit_panel.source_folder_panel)
    assert not window.grab().isNull()


def test_folding_keeps_original_refresh_owner_while_page_hide_stops_it(qtbot, monkeypatch, tmp_path):
    page = NativeMassCalibrationDisplay()
    qtbot.addWidget(page)
    panel = page.refit_panel
    monkeypatch.setattr(panel, "set_series", lambda _root: None)
    page.set_refit_series(tmp_path)
    panel.job_root = tmp_path / "existing-job"
    panel.snapshot = {"status": {"tasks": {"fit": {"state": "running"}}}}
    panel._timer.setInterval(60000)
    page.show()
    qtbot.waitUntil(panel._timer.isActive)
    page.sections["job"].set_expanded(True)
    page.sections["job"].set_expanded(False)
    assert panel._timer.isActive() and panel.isVisible()
    page.hide()
    assert not panel._timer.isActive()


def test_original_display_keeps_its_original_layout(qtbot):
    page = MassCalibrationDisplay()
    qtbot.addWidget(page)
    assert not hasattr(page, "workflow")
    assert page.refit_area.widget() is page.refit_panel
    assert isinstance(page.refit_panel.layout().itemAt(1).layout(), QFormLayout)


def test_parameters_follow_method_and_original_interactive_action_stays_inline(qtbot, monkeypatch, tmp_path):
    page = NativeMassCalibrationDisplay()
    qtbot.addWidget(page)
    calls = []
    window = SimpleNamespace(
        _show_original_mass_calibration_controls=lambda target: calls.append(target),
        open_image_selection_plan=lambda: None, _select_row=lambda _target: None,
        run_combo=SimpleNamespace(currentData=lambda: "saved/run.toml"))
    page.install_common_actions(window)
    page.set_calibration_context("phase_model", tmp_path)
    assert page.model_context.currentWidget() is page.phase_parameters
    assert page.sections["chemistry"].isAncestorOf(page.refit_panel.f_check)
    assert page.sections["chemistry"].isAncestorOf(page.refit_panel.loro_check)
    assert page.overflow_choice.isHidden()
    page.set_calibration_context("colour_path", tmp_path)
    assert page.model_context.isAncestorOf(page.original_controls.trials)
    assert not page.original_controls.isAncestorOf(page.original_controls.trials)
    page.original_controls.trials.setValue(250)
    page.original_controls.interactive_button.click()
    assert calls == ["mass"]
    assert page.original_controls.options()["trials"] == 250
    assert page.refit_panel._thread is None


def test_shared_run_selection_refreshes_the_original_parameter_owner(qtbot, tmp_path):
    import tomllib

    root = tmp_path / "series"
    root.mkdir()
    manifest = root / "series.toml"
    manifest.write_text('schema_version=1\nseries_id="example"\n[runs]\nids=["r1","r2"]\n')
    page = NativeMassCalibrationDisplay()
    qtbot.addWidget(page)
    page.set_refit_series(root)
    page.show()
    choice, panel = page.shared_cohort, page.refit_panel
    qtbot.waitUntil(lambda: choice.snapshot is not None and not choice.preview.busy)
    assert panel.roles["fit"] == ["r1", "r2"]
    choice.table.item(1, 1).setCheckState(Qt.CheckState.Unchecked)
    choice.save_button.click()
    qtbot.waitUntil(lambda: panel._thread is None)
    qtbot.waitUntil(lambda: panel.roles["fit"] == ["r1"])
    assert panel.roles["heldout"] == ["r2"]
    assert tomllib.loads(manifest.read_text())["calibration"]["mass"]["fit_runs"] == ["r1"]
    assert list(root.iterdir()) == [manifest]
    assert panel.job_root is None


def test_long_nas_paths_and_descriptions_wrap_without_horizontal_scroll(qtbot, monkeypatch, tmp_path, desktop_font):
    page = NativeMassCalibrationDisplay()
    qtbot.addWidget(page)
    monkeypatch.setattr(page.refit_panel, "set_series", lambda _root: None)
    page.set_refit_series(tmp_path)
    source = (r"\\nas.example\shared\FF_AC\DarSIA\calibration\refit"
              "\\series-mass-calibration-with-completed-inputs-20261024\\"
              + "a" * 64 + r"\outputs\frozen_calibration.json")
    explanation = ("This saved calibration retains the original measured inputs and source binding. "
                   "The full selected source remains available for copying: " + source)
    page.set_display(SavedMassCalibrationData((("Gain", "23.5", source),), source, (explanation,)))
    page.refit_panel.parameter_explanation.setText(explanation)
    page.refit_panel.baseline_label.setText(source)
    page.sections["chemistry"].set_expanded(True)
    page.show()
    page.saved_calibration_page.setParent(None)
    qtbot.addWidget(page.saved_calibration_page)
    page.saved_calibration_page.show()
    labels = (page.source, page.refit_panel.parameter_explanation, page.refit_panel.baseline_label)
    def text_fits(label):
        document = label.findChild(QTextDocument)
        block = document.begin()
        while block.isValid():
            layout = block.layout()
            if not layout.lineCount() or any(
                layout.lineAt(i).naturalTextWidth() > label.width()
                for i in range(layout.lineCount())
            ):
                return False
            block = block.next()
        return label.height() >= label.heightForWidth(label.width())

    for width, column_width in ((1440, 520), (1000, 400), (1600, 620)):
        page.resize(width, 900)
        page.saved_calibration_page.resize(width, 900)
        page.workflow.setSizes([column_width, width - column_width])
        qtbot.waitUntil(lambda: page.width() == width)
        for label in labels:
            qtbot.waitUntil(lambda: text_fits(label))
        qtbot.waitUntil(lambda: page.workflow.controls_scroll.horizontalScrollBar().maximum() == 0)
        field = page.refit_panel.parameter_explanation
        form = field.parentWidget().layout()
        row, _role = form.getWidgetPosition(field)
        next_label = form.itemAt(row + 1, QFormLayout.ItemRole.LabelRole).widget()
        qtbot.waitUntil(lambda: field.geometry().bottom() < next_label.geometry().top())
        assert not page.grab().isNull()
        for label in labels:
            document = label.findChild(QTextDocument)
            assert document.toPlainText() == label.text()
            qtbot.waitUntil(lambda: text_fits(label))
    assert page.source.text() == source
    assert page.refit_panel.parameter_explanation.text() == explanation
    assert page.refit_panel.baseline_label.text() == source
    assert page.source.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
