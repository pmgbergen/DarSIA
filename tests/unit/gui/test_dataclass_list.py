"""Test the ordered list widget (create_dataclass_list_input) with the restoration steps.

The steps are an array of tables (``[[restoration.steps]]``): entries without names,
in a given order, each with a method and the options of that method.
"""

import copy

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QFormLayout, QLineEdit, QWidget

from darsia.gui.ui.schema.dataclass_introspection import get_section_fields
from darsia.gui.ui.settings import SettingsFactory, unwrap_composite_widget
from darsia.presets.workflows.config.restoration import RestorationConfig


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class MockMainWindow:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.settings_inputs = {}

    def print_log(self, message):
        pass


def _steps_setting() -> dict:
    (section,) = get_section_fields("restoration")
    (steps,) = [f for f in section["fields"] if f["key"] == "restoration.steps"]
    return steps


def _build(config_dict):
    window = MockMainWindow(copy.deepcopy(config_dict))
    factory = SettingsFactory(window)
    parent = QWidget()
    form = QFormLayout(parent)
    setting = _steps_setting()
    _, result = factory.create_dataclass_list_input(
        setting, setting["entry_dataclass"], form_context={"form": form}
    )
    window.settings_inputs[setting["key"]] = factory._wrap_multi_row_result(result)
    return factory, window, result, parent


def _saved(factory, window) -> dict:
    factory._sync_settings_inputs_to_config_dict()
    return window.config_dict["restoration"]


def _control(entry, key):
    return unwrap_composite_widget(entry["group"]["sub_inputs"][key])


CONFIG = {
    "restoration": {
        "ignore": ["boolean_porosity"],
        "steps": [
            {"method": "resize", "options": {"mode": "scale", "scale": 0.2}},
            {"method": "tvd", "options": {"weight": 0.3}},
        ],
    }
}


def test_schema_of_the_steps_is_an_ordered_list():
    setting = _steps_setting()
    assert setting["type"] == "dataclass_list"
    assert setting["entry_dataclass"].__name__ == "RestorationStepConfig"


def test_steps_are_shown_in_their_order(qapp):
    _, _, result, parent = _build(CONFIG)

    entries = result["entries"]
    assert [entry["widget"].title() for entry in entries] == ["Step 1", "Step 2"]
    methods = [_control(entry, "entry.method").currentText() for entry in entries]
    assert methods == ["resize", "tvd"]
    scale = _control(entries[0], "entry.options.scale")
    assert isinstance(scale, QLineEdit) and scale.text() == "0.2"
    # Only the options of the selected method are shown
    parent.show()
    groups = entries[0]["group"]["sub_inputs"]
    assert groups["entry.resize_options"]["widget"].isVisible()
    assert not groups["entry.tvd_options"]["widget"].isVisible()


def test_saved_steps_load_with_the_options_of_their_method(qapp):
    factory, window, _, _ = _build(CONFIG)

    restoration = _saved(factory, window)

    steps = restoration["steps"]
    assert [step["method"] for step in steps] == ["resize", "tvd"]
    assert steps[0]["options"]["scale"] == 0.2
    assert "rev_size" not in steps[0]["options"]  # hidden groups are left out
    assert steps[1]["options"]["weight"] == 0.3
    assert restoration["ignore"] == ["boolean_porosity"]
    config = RestorationConfig().load(restoration)
    assert [step.method for step in config.steps] == ["resize", "tvd"]
    assert config.steps[0].resize_options.scale == 0.2


def test_moving_a_step_changes_the_saved_order(qapp):
    factory, window, result, _ = _build(CONFIG)

    result["entries"][1]["up"].click()

    assert [entry["widget"].title() for entry in result["entries"]] == [
        "Step 1",
        "Step 2",
    ]
    steps = _saved(factory, window)["steps"]
    assert [step["method"] for step in steps] == ["tvd", "resize"]


def test_unchecked_step_keeps_its_values(qapp):
    factory, window, result, _ = _build(CONFIG)

    result["entries"][0]["checkbox"].setChecked(False)

    step = _saved(factory, window)["steps"][0]
    assert step["active"] is False
    assert step["method"] == "resize"
    assert step["options"]["scale"] == 0.2


def test_added_and_removed_steps(qapp):
    factory, window, result, _ = _build(CONFIG)

    assert result["add_button"].text() == "Add step"
    result["add_button"].click()
    result["entries"][0]["remove"].click()

    steps = _saved(factory, window)["steps"]
    assert [step["method"] for step in steps] == ["tvd", "volume_average"]


def test_single_method_form_is_shown_as_one_step_and_saved_as_steps(qapp):
    factory, window, result, _ = _build(
        {"restoration": {"method": "tvd", "options": {"weight": 0.05}}}
    )

    assert len(result["entries"]) == 1
    method = _control(result["entries"][0], "entry.method")
    assert isinstance(method, QComboBox) and method.currentText() == "tvd"
    restoration = _saved(factory, window)
    assert "method" not in restoration and "options" not in restoration
    assert restoration["steps"][0]["options"]["weight"] == 0.05


def test_single_method_form_without_method_has_no_steps(qapp):
    factory, window, result, _ = _build({"restoration": {"method": "none"}})

    assert result["entries"] == []
    restoration = _saved(factory, window)
    assert restoration["steps"] == [] and "method" not in restoration
