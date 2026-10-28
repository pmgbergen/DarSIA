"""Embedded curvature master retains stage choices and the independent crop."""

from copy import deepcopy
import tomllib

import numpy as np
import pytest
import toml
from PySide6.QtWidgets import QApplication, QDialog

import darsia
from darsia.presets.workflows.config.corrections import CorrectionsConfig

from .test_native_sidebar import native_window, run_window, _select_run
from .test_crop_wizard_ui import correction_values as display_corrections


def correction_values(*, explicit=True):
    result = display_corrections(explicit=explicit)
    result["type"] = {"target_type": "float32"}
    result["resize"] = {"mode": "target_shape", "target_shape": [40, 50]}
    return result


def show_curvature(fixture, corrections):
    window = fixture.window
    _select_run(window, "series-a", "first")
    path = fixture.merged["series-a"]["first"]
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if corrections is not None:
        document["corrections"] = deepcopy(corrections)
    path.write_text(toml.dumps(document), encoding="utf-8")
    window.config_path_label.setText(str(path))
    window.config_controller.load_config()
    before = path.read_bytes()
    workflow = window.curvature_workflow
    workflow.set_active(True)
    # Root's route test owns the actual geometry host. Here the real embedded
    # form is visible without moving or replacing the native settings editor.
    window.content_stack.addWidget(workflow.widget)
    window.content_stack.setCurrentWidget(workflow.widget)
    return window, workflow, path, before


def field(workflow, stage, name):
    control = workflow.inputs[f"corrections.curvature.{stage}.{name}"]
    return control.property("value_widget") or control


def test_embedded_form_is_optional_and_does_not_take_over_native_settings(run_window):
    window, workflow, path, before = show_curvature(run_window, None)
    assert not workflow.master.isChecked() and not workflow.options.isEnabled()
    assert workflow.inputs is not window.settings_inputs
    outer = workflow.inputs["corrections.curvature"]
    assert not outer["widget"].isCheckable() and "checkbox" not in outer
    assert outer["widget"] is not window.settings_inputs["corrections.curvature"]["widget"]
    assert not any(widget.isVisible() for widget in QApplication.topLevelWidgets() if isinstance(widget, QDialog))
    assert path.read_bytes() == before
    assert not window.process_runner.active


@pytest.mark.parametrize("explicit", [False, True])
def test_crop_only_stays_active_and_other_corrections_survive_save(run_window, explicit):
    corrections = correction_values(explicit=explicit)
    corrections["curvature"]["active"] = ["crop"]
    window, workflow, path, _before = show_curvature(run_window, corrections)
    original = deepcopy(window.config_dict["corrections"])
    crop = workflow.inputs["corrections.curvature.crop"]["widget"]
    assert crop.isHidden() and not crop.isEnabled()
    assert not workflow.master.isChecked()
    workflow.save_button.click()
    saved = tomllib.loads(path.read_text(encoding="utf-8"))["corrections"]
    expected = deepcopy(original)
    expected["curvature"]["optional_stages_active"] = False
    assert saved == expected
    parsed = CorrectionsConfig().load([path])
    assert parsed.curvature is not None and parsed.curvature.crop is not None
    assert parsed.curvature.init is None and parsed.curvature.bulge is None
    assert parsed.curvature.stretch is None
    assert run_window.warnings == []


def test_master_off_save_reload_preserves_choices_and_on_restores_numeric_effect(run_window):
    corrections = correction_values()
    corrections["curvature"]["active"] = ["crop", "init"]
    corrections["curvature"]["init"]["horizontal_bulge"] = 0.0001
    window, workflow, path, _before = show_curvature(run_window, corrections)
    assert workflow.master.isChecked() and workflow.options.isEnabled()
    workflow.save_button.click()
    original = tomllib.loads(path.read_text(encoding="utf-8"))["corrections"]
    image = np.indices((40, 50)).sum(axis=0).astype(np.float32)
    on = darsia.CurvatureCorrection(config=CorrectionsConfig().load([path]).curvature.to_dict()).correct_array(image)
    workflow.master.setChecked(False)
    assert not workflow.options.isEnabled()
    assert workflow.inputs["corrections.curvature.init"]["checkbox"].isChecked()
    workflow.save_button.click()
    off = tomllib.loads(path.read_text(encoding="utf-8"))["corrections"]
    expected = deepcopy(original)
    expected["curvature"]["optional_stages_active"] = False
    assert off == expected
    window.config_controller.load_config()
    workflow.refresh()
    assert not workflow.master.isChecked() and not workflow.options.isEnabled()
    assert workflow.inputs["corrections.curvature.init"]["checkbox"].isChecked()
    parsed_off = CorrectionsConfig().load([path]).curvature
    assert parsed_off.init is None and parsed_off.crop is not None
    off_pixels = darsia.CurvatureCorrection(config=parsed_off.to_dict()).correct_array(image)
    assert not np.array_equal(on, off_pixels)
    workflow.master.setChecked(True)
    workflow.save_button.click()
    restored = CorrectionsConfig().load([path]).curvature
    assert restored.init is not None and restored.crop is not None
    assert np.array_equal(on, darsia.CurvatureCorrection(config=restored.to_dict()).correct_array(image))
    assert tomllib.loads(path.read_text(encoding="utf-8"))["corrections"] == original


def test_master_on_does_not_activate_a_previously_disabled_crop(run_window):
    corrections = correction_values()
    corrections["active"].remove("curvature")
    window, workflow, path, _before = show_curvature(run_window, corrections)
    assert not workflow.master.isChecked()
    workflow.save_button.click()
    saved_off = tomllib.loads(path.read_text(encoding="utf-8"))["corrections"]
    expected = deepcopy(corrections)
    expected["curvature"]["optional_stages_active"] = False
    assert saved_off == expected
    workflow.master.setChecked(True)
    workflow.save_button.click()
    saved = tomllib.loads(path.read_text(encoding="utf-8"))["corrections"]
    assert "curvature" in saved["active"] and "crop" not in saved["curvature"]["active"]
    assert saved["curvature"]["crop"] == corrections["curvature"]["crop"]
    assert CorrectionsConfig().load([path]).curvature.crop is None


def test_run_switch_uses_original_loader_and_saves_only_selected_run(run_window):
    window, workflow, first, _before = show_curvature(run_window, correction_values())
    first_bytes = first.read_bytes()
    _select_run(window, "series-a", "second")
    workflow.set_active(True)
    second = run_window.merged["series-a"]["second"]
    assert window.config_file == str(second)
    assert "SECOND" in workflow.context_note.text()
    assert not workflow.master.isChecked()
    workflow.master.setChecked(True)
    field(workflow, "init", "horizontal_bulge").setText("0.0002")
    workflow.save_button.click()
    saved = tomllib.loads(second.read_text(encoding="utf-8"))
    assert saved["run_marker"] == "second"
    assert saved["corrections"]["curvature"]["init"]["horizontal_bulge"] == 0.0002
    assert first.read_bytes() == first_bytes


def test_old_form_cannot_save_after_selector_changes_without_loading(run_window):
    window, workflow, first, before = show_curvature(run_window, correction_values())
    second = run_window.merged["series-a"]["second"]
    second_before = second.read_bytes()
    field(workflow, "init", "horizontal_bulge").setText("0.0002")
    window.run_combo.setCurrentIndex(window.run_combo.findText("second"))
    workflow.save_button.click()
    assert first.read_bytes() == before and second.read_bytes() == second_before


def test_deactivate_remount_preserves_pending_draft_for_same_run(run_window):
    window, workflow, path, before = show_curvature(run_window, correction_values())
    workflow.master.setChecked(False)
    field(workflow, "init", "horizontal_bulge").setText("0.0002")
    original_field = field(workflow, "init", "horizontal_bulge")
    workflow.set_active(False)
    workflow.widget.setParent(window)
    workflow.widget.hide()
    workflow.set_active(True)
    workflow.refresh()
    assert field(workflow, "init", "horizontal_bulge") is original_field
    assert not workflow.master.isChecked() and not workflow.options.isEnabled()
    assert original_field.text() == "0.0002"
    assert path.read_bytes() == before
