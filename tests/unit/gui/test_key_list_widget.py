"""End-to-end test for the unified ``key_list`` settings widget.

Exercises the whole path that replaced the former
registry/format/roi/color ``*_key_list_input`` methods:

* ``SettingsFactory.create_key_list_input`` (reached through the real
  ``create_setting_edit`` dispatch),
* ``_wrap_multi_row_result`` tagging,
* the single unified ``key_list`` pass inside
  ``_sync_settings_inputs_to_config_dict``.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QFormLayout, QWidget

from darsia.gui.ui.settings import SettingsFactory


@pytest.fixture
def qapp():
    """Provide a (headless, via conftest) QApplication for widget construction."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _MockMainWindow:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.settings_inputs = {}

    def print_log(self, *_args, **_kwargs):
        pass


def _build(factory, setting, keepalive):
    """Render a key_list field through the real dispatch; return its result dict.

    ``keepalive`` retains the parent QWidget/QFormLayout so Qt does not garbage
    collect the form out from under the widget being built.
    """
    parent = QWidget()
    form = QFormLayout(parent)
    keepalive.append((parent, form))
    return factory.create_setting_edit(setting, form_context={"form": form})


def _options(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_key_list_widget_end_to_end(qapp):
    config_dict = {
        # two "registry" source sections (array-of-tables shape)
        "data_interval": [{"name": "interval_a"}, {"name": "interval_b"}],
        "data_path": [{"name": "path_x"}],
        # a "format" source section whose entries carry a discriminating type
        "format": [
            {"name": "png_hi", "type": "png"},
            {"name": "csv_out", "type": "csv"},
        ],
        # pre-existing selections
        "analysis": {"thresholding": {"data_selection": ["interval_b", "path_x"]}},
        "mass": {"color": ["interval_a", "interval_b"]},
    }
    factory = SettingsFactory(_MockMainWindow(config_dict))
    keepalive: list = []

    # 1. multi-source union + source_type_map + one prefilled row per selection
    registry_setting = {
        "key": "analysis.thresholding.data_selection",
        "name": "Data selection",
        "type": "key_list",
        "key_list_sources": ("data_interval", "data_path"),
    }
    _, reg = _build(factory, registry_setting, keepalive)

    assert reg["key_list"] is True
    assert reg["source_type_map"] == {
        "interval_a": "data_interval",
        "interval_b": "data_interval",
        "path_x": "data_path",
    }
    assert [c.currentText() for c in reg["rows"]] == ["interval_b", "path_x"]
    assert _options(reg["rows"][0]) == ["interval_a", "interval_b", "path_x"]

    # 2. format_types filters the dropdown by entry["type"]
    format_setting = {
        "key": "analysis.thresholding.formats",
        "name": "Formats",
        "type": "key_list",
        "key_list_sources": ("format",),
        "format_types": {"png"},
    }
    _, fmt = _build(factory, format_setting, keepalive)
    assert list(fmt["source_type_map"]) == ["png_hi"]  # csv_out filtered out
    assert _options(fmt["rows"][0]) == ["png_hi"]

    # 3. max_rows == 1 caps the pre-fill to a single row
    max_rows_setting = {
        "key": "mass.color",
        "name": "Color embedding",
        "type": "key_list",
        "key_list_sources": ("data_interval",),
        "max_rows": 1,
    }
    _, mr = _build(factory, max_rows_setting, keepalive)
    assert mr["max_rows"] == 1
    assert len(mr["rows"]) == 1
    assert mr["rows"][0].currentText() == "interval_a"

    # 4. missing source section degrades gracefully to an empty dropdown
    empty_setting = {
        "key": "misc.selection",
        "name": "Misc",
        "type": "key_list",
        "key_list_sources": ("does_not_exist",),
    }
    _, empty = _build(factory, empty_setting, keepalive)
    assert empty["source_type_map"] == {}
    assert _options(empty["rows"][0]) == []

    # 5. save round-trip through the real unified ``key_list`` save pass
    reg["rows"][0].setCurrentText("interval_a")  # row 2 stays "path_x"

    factory.main_window.settings_inputs = {
        registry_setting["key"]: factory._wrap_multi_row_result(reg),
        format_setting["key"]: factory._wrap_multi_row_result(fmt),
        max_rows_setting["key"]: factory._wrap_multi_row_result(mr),
        empty_setting["key"]: factory._wrap_multi_row_result(empty),
    }
    factory._sync_settings_inputs_to_config_dict()

    # multi-select -> list[str], order preserved
    assert config_dict["analysis"]["thresholding"]["data_selection"] == [
        "interval_a",
        "path_x",
    ]
    # single available option was auto-selected
    assert config_dict["analysis"]["thresholding"]["formats"] == ["png_hi"]
    # max_rows == 1 -> bare string, not a list
    assert config_dict["mass"]["color"] == "interval_a"
    # nothing selectable -> None
    assert config_dict["misc"]["selection"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
