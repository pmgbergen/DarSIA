"""Saving option groups shown by `depends_on`: only the visible group is written.

Two option groups share a TOML table (``toml_key``) and a field name, as the contour
smoother options of the mass calibration do.
"""

import copy
from dataclasses import dataclass, field

import pytest
from PySide6.QtWidgets import QApplication, QFormLayout, QWidget

from darsia.gui.ui.schema.dataclass_introspection import _build_fields
from darsia.gui.ui.settings import SettingsFactory, unwrap_composite_widget


@dataclass
class OptionsA:
    window: int = field(default=3, metadata={"name": "Window"})
    scale: float = field(default=1.0, metadata={"name": "Scale"})


@dataclass
class OptionsB:
    window: int = field(default=3, metadata={"name": "Window"})
    weight: float = field(default=0.1, metadata={"name": "Weight"})


@dataclass
class Section:
    method: str = field(default="a", metadata={"name": "Method", "options": ["a", "b"]})
    a_options: OptionsA | None = field(
        default=None,
        metadata={
            "toml_key": "options",
            "depends_on": {"field": "method", "value": "a"},
        },
    )
    b_options: OptionsB | None = field(
        default=None,
        metadata={
            "toml_key": "options",
            "depends_on": {"field": "method", "value": "b"},
        },
    )


class MockMainWindow:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.settings_inputs = {}

    def print_log(self, message):
        pass


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build(config_dict):
    window = MockMainWindow(copy.deepcopy(config_dict))
    factory = SettingsFactory(window)
    parent = QWidget()
    factory.build_tab_form(QFormLayout(parent), _build_fields(Section, "sec"))
    return factory, window, parent


def _edit(window, group_key, key, text):
    group = window.settings_inputs[group_key]
    unwrap_composite_widget(group["sub_inputs"][key]).setText(text)


def test_only_the_options_of_the_selected_method_are_saved(qapp):
    factory, window, _ = _build(
        {"sec": {"method": "b", "options": {"window": 5, "weight": 0.2}}}
    )

    factory._sync_settings_inputs_to_config_dict()

    options = window.config_dict["sec"]["options"]
    assert options == {"window": 5, "weight": 0.2}  # no "scale" of the hidden group


def test_shared_field_of_the_selected_method_is_saved(qapp):
    """The hidden group has a field with the same key; the visible one is saved."""
    factory, window, _ = _build(
        {"sec": {"method": "b", "options": {"window": 5, "weight": 0.2}}}
    )

    _edit(window, "sec.b_options", "sec.options.window", "7")
    factory._sync_settings_inputs_to_config_dict()

    assert window.config_dict["sec"]["options"]["window"] == 7
