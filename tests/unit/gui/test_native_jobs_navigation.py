"""Native headings and Jobs routes retain their original owners and actions."""

from PySide6.QtCore import Qt

from .test_native_sidebar import native_window


def test_jobs_has_one_owner_outside_production(native_window, qtbot):
    window = native_window
    assert window.sidebar.display_selection("jobs", "overview") == ("jobs", "jobs:overview")
    assert "jobs:overview" not in window.sidebar._sections["analysis"]._rows
    assert window.sidebar.display_selection("jobs", "new_analysis")[0] == "analysis"
    section = window.sidebar._sections["jobs"]
    qtbot.mouseClick(section.header_button, Qt.MouseButton.LeftButton)
    owner = window.jobs_workspace
    assert owner is not None
    assert window.content_stack.currentWidget() is owner
    assert window.sidebar.get_selected() == ("jobs", "overview")
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()
    assert not window.run_combo.isVisible()
    window._select_row(("jobs", "sources"))
    assert owner.sources.isChecked()
    assert owner.sources.title() == "Advanced — imports and paths"
    assert owner.shared_jobs_section.is_expanded()
    assert not window.run_combo.isVisible()
    window.show_jobs()
    assert window.jobs_workspace is owner
    assert window.sidebar.get_selected() == ("jobs", "overview")
    assert owner.panel._read_thread is None
    assert owner.panel._create_thread is None
    assert owner.panel._session_requests == []


def test_analysis_keeps_original_wasserstein_routes(native_window, monkeypatch):
    window = native_window
    displayed = []
    original = window.settings_factory.display_settings

    def record(action, selected):
        displayed.append((action, selected))
        return original(action, selected)

    monkeypatch.setattr(window.settings_factory, "display_settings", record)
    assert window.sidebar.categories_data["comparison"][0] == "Analysis"
    assert window.sidebar.categories_data["analysis"][0] == "Production"
    for item in ("wasserstein_compute", "wasserstein_assemble"):
        window._select_row(("comparison", item))
        assert window.sidebar.get_selected() == ("comparison", item)
        assert displayed[-1] == ("comparison", [item])
        assert window.content_stack.currentWidget() is window.settings_scroll_area
        assert window.analysis_execution_note.isVisible()
        assert "on this computer" in window.analysis_execution_note.text()
        assert "Comparisons → Wasserstein comparison" in window.analysis_execution_note.text()
    window.show_jobs()
    assert not window.analysis_execution_note.isVisible()


def test_comparison_preparation_has_one_passive_jobs_owner(native_window):
    window = native_window
    window._select_row(("jobs", "wasserstein"))
    owner = window.jobs_workspace
    assert window.sidebar.display_selection("jobs", "wasserstein")[0] == "comparison"
    assert "jobs:wasserstein" not in window.sidebar._sections["jobs"]._rows
    assert "jobs:wasserstein" in window.sidebar._sections["comparison"]._rows
    assert window.content_stack.currentWidget() is owner
    assert owner.views.currentWidget() is owner.wasserstein
    assert not owner.wasserstein.prepare_button.isEnabled()
    assert owner.wasserstein._thread is None and owner.panel._session_requests == []
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.run_combo.isVisible()
    window.show_jobs()
    assert window.jobs_workspace is owner and owner.views.currentWidget() is owner.jobs_view
