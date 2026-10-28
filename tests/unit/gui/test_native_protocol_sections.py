"""Overflow is a lazy protocol overview; phase image choices stay with calibration."""
from pathlib import Path
from threading import get_ident
from copy import deepcopy

from PySide6.QtWidgets import QTabWidget, QLineEdit

from darsia.fluidflower import native_protocol_sections as module
from darsia.gui.ui.settings import unwrap_composite_widget
from darsia.fluidflower.overflow_data_page import OverflowData
from .test_native_row_scope import scope_window  # noqa: F401
from .test_native_series_method import method_window  # noqa: F401


def test_protocol_overview_is_loaded_only_when_its_vertical_page_opens(scope_window, qtbot, monkeypatch):
    window = scope_window
    calls = []
    gui_thread = get_ident()
    def read(root):
        assert get_ident() != gui_thread
        calls.append(Path(root))
        return OverflowData(summary="Configured protocol overview")
    monkeypatch.setattr(module, "read_overflow_data", read)
    window._select_row(("preprocessing", "protocols"))
    qtbot.wait(1)
    controller = window.protocol_sections
    assert controller.navigation is not None
    assert all(not tabs.isVisible() for tabs in window.settings_scroll_area.findChildren(QTabWidget))
    page = controller.overflow_page
    assert page is not None and not calls
    assert window.sidebar._sections["setup"].get_row("overflow_data:runs").isHidden()
    controller.navigation.setCurrentRow(list(controller.pages).index("overflow"))
    qtbot.waitUntil(lambda: not page.reader.busy)
    assert len(calls) == 1
    assert page.summary.text() == "Configured protocol overview"


def test_protocol_pages_keep_original_keys_values_and_dependent_fields(scope_window, qtbot):
    window = scope_window
    window.settings_inputs = {}
    window.config_dict = {
        "data": {"folders": ["images/run", "images/continuation"], "format": "NEF", "baseline": "base.NEF"},
        "protocols": {
            "imaging": {"images/run": ["protocols/imaging.xlsx", "run"]},
            "blacklist": {},
            "injection": ["protocols/injection.xlsx", "run"],
            "injection_mode": "Load from CSV",
            "pressure_temperature": ["protocols/conditions.xlsx", "run"],
            "pressure_temperature_mode": "Load from CSV",
            "overflow": ["protocols/overflow.xlsx", "run"],
        },
    }
    original = deepcopy(window.config_dict)
    window._select_row(("preprocessing", "protocols"))
    qtbot.wait(1)
    controller = window.protocol_sections
    assert list(controller.pages) == list(module.NativeProtocolSections.TITLES)
    assert controller.pages["injection"].isAncestorOf(window.settings_inputs["protocols.injection"])
    assert controller.pages["pressure_temperature"].isAncestorOf(window.settings_inputs["protocols.pressure_temperature"])
    assert controller.pages["exclusions"].isAncestorOf(window.settings_inputs["protocols.blacklist"]["rows"][0][1])
    rate = unwrap_composite_widget(window.settings_inputs["protocols.injection_rate"])
    assert isinstance(rate, QLineEdit)
    ancestors = []
    control = rate
    while control is not controller.pages["injection"]:
        ancestors.append(control)
        control = control.parentWidget()
    assert any(control.isHidden() for control in ancestors)
    for index in (2, 3, 4, 0):
        controller.navigation.setCurrentRow(index)
    assert window.config_dict == original
    # The existing save synchronizer still owns all fields, including sheet pairs.
    window.settings_factory._sync_settings_inputs_to_config_dict()
    for key in ("imaging", "injection", "pressure_temperature", "overflow"):
        assert window.config_dict["protocols"][key] == original["protocols"][key]
    assert window.config_dict["protocols"]["blacklist"] == {}
    controller.navigation.setCurrentRow(list(controller.pages).index("pressure_temperature"))
    window.settings_factory.refresh_current_view()
    assert window.protocol_sections.selected_key == "pressure_temperature"
    assert window.protocol_sections.stack.currentIndex() == list(controller.pages).index("pressure_temperature")


def test_protocol_regrouping_is_a_copy_with_every_original_field_once():
    source = {
        "data": [{"key": "data.folders", "group_name": "Input"}],
        "protocols": [{"key": key, "group_name": "Operating conditions", "depends_on": {"field": "injection_mode", "value": "constant"}}
                      for key in ("protocols.imaging", "protocols.blacklist", "protocols.injection", "protocols.injection_rate",
                                  "protocols.pressure_temperature", "protocols.temperature_celsius", "protocols.overflow")],
        "options": [{"key": "options.preprocessing"}],
    }
    before = deepcopy(source)
    result = module.NativeProtocolSections._sections(source)
    # Preparing the protocols ignores the original "Show plots" option, so it is not shown.
    assert "options" not in result
    assert sorted(field["key"] for values in result.values() for field in values) == sorted(
        field["key"] for section, values in source.items() if section != "options" for field in values)
    assert source == before
    assert "group_name" not in result["injection"][0]
    assert result["injection"][1]["depends_on"] == source["protocols"][3]["depends_on"]


def test_the_list_covers_pressure_datasets_and_workbooks_without_extra_buttons(scope_window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QPushButton
    from darsia.fluidflower import protocol_workbooks_ui

    window = scope_window
    window._select_row(("preprocessing", "protocols"))
    qtbot.wait(1)
    controller = window.protocol_sections
    titles = [controller.navigation.item(row).text() for row in range(controller.navigation.count())]
    assert titles[-1] == "Protocol workbooks" and "Protocol preparation options" not in titles
    # No action buttons above the settings: each action sits on its list page.
    above = window.content_stack.parentWidget()
    assert not [button.text() for button in above.findChildren(QPushButton)
                if not window.content_stack.isAncestorOf(button) and button.text() in
                ("Pressure datasets…", "Protocol workbooks…")]
    pressure = controller.pages["pressure_temperature"]
    assert pressure.isAncestorOf(controller.pressure_datasets_button)
    opened = []
    monkeypatch.setattr(window.running_pressure, "open", lambda: opened.append("pressure"))
    controller.pressure_datasets_button.click()
    assert opened == ["pressure"]
    controller.navigation.setCurrentRow(titles.index("Protocol workbooks"))
    assert controller.selected_key == "workbooks"
    assert controller.pages["workbooks"].isAncestorOf(controller.workbooks_button)
    monkeypatch.setattr(protocol_workbooks_ui.ProtocolWorkbooksPanel, "show_series", lambda panel, root: None)
    controller.workbooks_button.click()
    assert window.content_stack.currentWidget() is window.protocol_workbooks_panel


def test_setup_reference_selection_remains_separate_from_calibration(scope_window):
    window = scope_window
    row = window.sidebar._sections["setup"].get_row("image_selection:plan")
    window.protocol_sections.update_method("phase_model")
    assert not row.isHidden()
    assert window.sidebar._sections["setup"].get_row("setup:rig").isHidden()
    assert window.sidebar._sections["setup"].get_row("setup:all").isHidden()
    calls = []
    window.image_selection_workflow.open = lambda: calls.append(True)
    window.open_image_selection_plan()
    assert calls == [True]
    window.protocol_sections.update_method("colour_path")
    assert not row.isHidden()
    assert not window.sidebar._sections["setup"].get_row("setup:rig").isHidden()
    assert not window.sidebar._sections["setup"].get_row("setup:all").isHidden()
