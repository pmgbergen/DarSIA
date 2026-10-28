"""Native wizard policy boundaries; local widgets/TOML only, no workflows."""

from copy import deepcopy

import pytest
from PySide6.QtWidgets import QAbstractButton

from .test_native_sidebar import native_window, run_window, _select_run
from darsia.fluidflower.crop_wizard_ui import prepare_crop_wizard


def correction_values(*, explicit=True):
    curvature = {
        "init": {"horizontal_bulge": 0.125},
        "crop": {
            "top_left": [1, 2],
            "bottom_left": [30, 2],
            "bottom_right": [30, 40],
            "top_right": [1, 40],
            "width": 0.9,
            "height": 0.49,
            "in_meters": True,
            "custom_provenance": "keep",
        },
        "bulge": {"vertical_bulge": 0.25},
        "stretch": {"horizontal_stretch": 0.375},
    }
    values = {
        "type": {"dtype": "float32"},
        "resize": {"shape": [40, 50]},
        "curvature": curvature,
    }
    if explicit:
        values["active"] = ["resize", "type", "curvature"]
        curvature["active"] = ["stretch", "crop", "init", "bulge"]
    return values


def dialog_for(run_window, qtbot, *, explicit=True):
    from darsia.gui.ui.preprocessing_wizard import PreprocessingWizardDialog

    window = run_window.window
    _select_run(window, "series-a", "first")
    window._select_row(("preprocessing", "protocols"))
    window.config_dict["corrections"] = correction_values(explicit=explicit)
    before = deepcopy(window.config_dict["corrections"])
    dialog = PreprocessingWizardDialog(window)
    qtbot.addWidget(dialog)
    # Exercise the composed native rig/crop policy on the original dialog.
    window.rig_context.prepare_wizard(dialog, extra_policy=prepare_crop_wizard)
    return window, dialog, before


def go(dialog, name):
    from darsia.gui.ui.preprocessing_wizard import STEPS

    dialog._goto(next(i for i, entry in enumerate(STEPS) if entry[0] == name))


@pytest.mark.parametrize("explicit", [True, False])
def test_hidden_crop_keeps_values_activation_and_unrendered_members_on_navigation(
    run_window, qtbot, explicit
):
    files = {path: path.read_bytes() for path in run_window.paths}
    window, dialog, before = dialog_for(run_window, qtbot, explicit=explicit)
    # Jump directly to Curvature before Type/Resize have been rendered.
    go(dialog, "curvature")
    inputs = window.settings_inputs
    assert inputs["corrections.curvature.crop"]["widget"].isHidden()
    assert not inputs["corrections.curvature.crop"]["widget"].isEnabled()
    for name in ("init", "bulge", "stretch"):
        group = inputs["corrections.curvature." + name]["widget"]
        assert not group.isHidden() and group.isEnabled()
    parent = inputs["corrections.curvature"]["widget"]
    load = [
        button
        for button in parent.findChildren(QAbstractButton)
        if "Load" in button.text()
    ]
    assert load and all(button.isHidden() and not button.isEnabled() for button in load)
    go(dialog, "depth")
    go(dialog, "curvature")
    window.settings_factory._sync_settings_inputs_to_config_dict()
    actual = window.config_dict["corrections"]
    assert actual["curvature"]["crop"] == before["curvature"]["crop"]
    if explicit:
        assert actual["active"] == before["active"]
        assert actual["curvature"]["active"] == before["curvature"]["active"]
    else:
        assert "active" not in actual
        assert "active" not in actual["curvature"]
    dialog._launch_crop_correction()
    for path, contents in files.items():
        assert path.read_bytes() == contents


def test_visible_stage_edits_work_while_hidden_crop_and_other_activations_survive(
    run_window, qtbot
):
    window, dialog, before = dialog_for(run_window, qtbot)
    go(dialog, "curvature")
    field = window.settings_inputs["corrections.curvature.bulge.vertical_bulge"]
    value = field.property("value_widget") or field
    value.setText("0.625")
    window.settings_inputs["corrections.curvature.init"]["checkbox"].setChecked(False)
    window.settings_factory._sync_settings_inputs_to_config_dict()
    actual = window.config_dict["corrections"]
    assert actual["curvature"]["bulge"]["vertical_bulge"] == 0.625
    assert actual["curvature"]["crop"] == before["curvature"]["crop"]
    assert actual["active"] == before["active"]
    assert actual["curvature"]["active"] == ["stretch", "crop", "bulge"]


def test_full_editor_and_ordinary_wizard_keep_original_crop_fields_and_load(
    run_window, qtbot
):
    from darsia.gui.ui.preprocessing_wizard import PreprocessingWizardDialog

    window, native_dialog, _before = dialog_for(run_window, qtbot)
    go(native_dialog, "curvature")
    native_dialog.close()
    window.settings_factory.display_full_settings()
    group = window.settings_inputs["corrections.curvature.crop"]["widget"]
    assert not group.isHidden() and group.isEnabled()
    parent = window.settings_inputs["corrections.curvature"]["widget"]
    assert any(
        button.isEnabled()
        for button in parent.findChildren(QAbstractButton)
        if "Load" in button.text()
    )
    ordinary = PreprocessingWizardDialog(window)
    qtbot.addWidget(ordinary)
    go(ordinary, "curvature")
    group = window.settings_inputs["corrections.curvature.crop"]["widget"]
    assert not group.isHidden() and group.isEnabled()
    assert not getattr(ordinary, "_native_crop_wizard_policy", False)
    assert (
        ordinary._launch_crop_correction.__func__
        is PreprocessingWizardDialog._launch_crop_correction
    )
