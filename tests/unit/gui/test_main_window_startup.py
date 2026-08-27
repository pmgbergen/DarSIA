"""Smoke test for MainWindow startup and basic initialization.

This test verifies that the GUI's main window can be instantiated without errors
and that all major components (tabs, menu, toolbar, settings panel, log) are
properly initialized. It's a high-value regression test that catches import
errors, layout crashes, and wiring mistakes across the entire GUI module.
"""

import pytest
from PySide6.QtWidgets import QTextEdit, QToolBar


@pytest.fixture
def main_window(qtbot):
    """Fixture to instantiate and manage MainWindow lifecycle."""
    from darsia.gui.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_main_window_startup(main_window):
    """Test that MainWindow instantiates and initializes basic state."""
    assert main_window is not None
    assert main_window.windowTitle()
    assert main_window.windowTitle() == "DarSIA"


def test_main_window_has_tabs(main_window):
    """Test that the three workflow categories (Setup, Calibration, Analysis) exist in the sidebar."""
    # The sidebar-based navigation replaces the old QTabWidget architecture.
    # Check that the sidebar contains the expected workflow categories.
    assert hasattr(main_window, "sidebar"), "MainWindow does not have a sidebar"
    sidebar = main_window.sidebar
    assert sidebar is not None, "Sidebar is None"

    # The sidebar has a _sections dict mapping action names to CategorySection objects
    assert hasattr(sidebar, "_sections"), "Sidebar does not have _sections"
    section_names = list(sidebar._sections.keys())

    # Check that we have sections for Setup, Calibration, and Analysis
    assert (
        "setup" in section_names
    ), f"Setup category not found. Found categories: {section_names}"
    assert (
        "calibration" in section_names
    ), f"Calibration category not found. Found categories: {section_names}"
    assert (
        "analysis" in section_names
    ), f"Analysis category not found. Found categories: {section_names}"


def test_main_window_has_menu_bar(main_window):
    """Test that the menu bar is populated."""
    menu_bar = main_window.menuBar()
    assert menu_bar is not None
    assert menu_bar.actions(), "Menu bar has no actions"

    # Check for at least File and Help menus (strip & mnemonic from action text)
    menu_titles = [action.text().replace("&", "") for action in menu_bar.actions()]
    assert "File" in menu_titles, f"File menu not found. Menus: {menu_titles}"
    assert "Help" in menu_titles, f"Help menu not found. Menus: {menu_titles}"


def test_main_window_has_toolbar(main_window):
    """Test that the toolbar is populated."""
    toolbars = main_window.findChildren(QToolBar)
    assert len(toolbars) > 0, "No toolbar found in MainWindow"

    # Check that the toolbar has actions (buttons)
    toolbar = toolbars[0]
    toolbar_actions = toolbar.actions()
    assert len(toolbar_actions) > 0, "Toolbar has no actions"


def test_main_window_has_settings_panel(main_window):
    """Test that the settings panel (scroll area with settings widgets) exists."""
    # The settings panel should be a scroll area or widget that will be populated
    # by display_settings. We check that the main_window has a settings_inputs dict
    # that will be populated when settings are displayed.
    assert hasattr(main_window, "settings_inputs")
    assert isinstance(main_window.settings_inputs, dict)
    # Initially empty until display_settings is called
    assert len(main_window.settings_inputs) == 0


def test_main_window_has_log_panel(main_window):
    """Test that the log panel (QTextEdit) exists and initial message is logged."""
    log_edits = main_window.findChildren(QTextEdit)
    assert len(log_edits) > 0, "No QTextEdit found for log panel"

    log_text_edit = log_edits[0]
    assert log_text_edit is not None

    # Check that there's some initial log content (welcome message)
    log_content = log_text_edit.toPlainText()
    assert len(log_content) > 0, "Log panel is empty (no initial message)"


def test_main_window_initial_state(main_window):
    """Test that MainWindow initializes with expected empty/default state."""
    # Check config_dict is initialized
    assert hasattr(main_window, "config_dict")
    assert isinstance(main_window.config_dict, dict)
    assert len(main_window.config_dict) == 0

    # Check chosen_files is initialized
    assert hasattr(main_window, "chosen_files")
    assert isinstance(main_window.chosen_files, dict)
    assert len(main_window.chosen_files) == 0

    # Check settings_inputs is initialized
    assert hasattr(main_window, "settings_inputs")
    assert isinstance(main_window.settings_inputs, dict)
    assert len(main_window.settings_inputs) == 0


def test_main_window_has_settings_factory(main_window):
    """Test that SettingsFactory is properly initialized."""
    assert hasattr(main_window, "settings_factory")
    from darsia.gui.ui.settings import SettingsFactory

    assert isinstance(main_window.settings_factory, SettingsFactory)
