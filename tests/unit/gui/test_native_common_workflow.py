"""One fixed navigation dispatches only to existing calculation owners."""
from PySide6.QtCore import Qt

from darsia.fluidflower.series_method import MethodSelection, save_series_method
from .test_native_series_method import method_window, _roots, _files  # noqa: F401


def select_model(window, method):
    save_series_method(_roots(window)["series-a"], MethodSelection(active=method,
        model_version="fixture-model", mass_calibration="fixture-calibration.json",
        colour_calibration="fixture-colour"))
    window._refresh_method_state(force=True)


def test_calibration_positions_are_fixed_and_inactive_rows_cannot_be_clicked(method_window, qtbot):
    window = method_window
    section = window.sidebar._sections["calibration"]
    order = list(section._rows)
    for method in ("phase_model", "colour_path"):
        select_model(window, method)
        window._select_row(("calibration", "mass"))
        before = window.sidebar.get_selected()
        row = section.get_row("calibration_controls:interactive" if method == "phase_model" else "phase_model:training")
        assert not row.isEnabled()
        assert row.isHidden() == (method == "phase_model")
        assert "Calibration Method" in row.toolTip()
        qtbot.mouseClick(row, Qt.MouseButton.LeftButton)
        assert window.sidebar.get_selected() == before
        assert list(section._rows) == order
        assert section.get_row("color") is None
        assert window.sidebar._sections["setup"].get_row("calibration:color").isEnabled()


def test_legacy_mass_routes_dispatch_saved_choice_not_unsaved_fields(method_window, monkeypatch):
    window = method_window
    calls = []
    monkeypatch.setattr(window, "_show_mass_calibration_values", calls.append)
    for method, draft in (("phase_model", "colour_path"), ("colour_path", "phase_model")):
        select_model(window, method)
        window.method_panel.set_selection(MethodSelection(active=draft), available=True)
        before = _files(_roots(window)["series-a"])
        window._select_row(("phase_model", "mass_from_phase"))
        assert calls[-1] == method
        assert window.sidebar.get_selected() == ("calibration", "mass")
        assert _files(_roots(window)["series-a"]) == before


def test_colour_paths_dispatches_both_existing_owners_without_launch(method_window, monkeypatch):
    window = method_window
    select_model(window, "phase_model")
    before = _files(_roots(window)["series-a"])
    window._select_row(("calibration", "color"))
    assert window.content_stack.currentWidget() is window.mass_calibration_display.phase_colour_paths
    assert window.sidebar.display_selection("calibration", "color") == ("setup", "calibration:color")
    preparation = window.mass_calibration_display.preparation
    assert not window.mass_calibration_display.phase_colour_paths.preparation_host.isAncestorOf(preparation)
    assert not preparation._preparation_only_view
    assert window.current_row_scope.scope == "series"
    assert not window.current_row_scope.run_selector_enabled
    finish = window.mass_calibration_display.phase_colour_paths.calibration_button
    assert not finish.isEnabled()
    assert not window.toolbar_builder.play_action.isEnabled()
    assert _files(_roots(window)["series-a"]) == before
    assert finish.isHidden()
    rows = list(window.sidebar._sections["setup"]._rows)
    assert rows.index("setup:finish") == rows.index("calibration:color") + 1
    assert _files(_roots(window)["series-a"]) == before
    select_model(window, "colour_path")
    monkeypatch.setattr(window, "_ensure_run_loaded", lambda: True)
    window._select_row(("calibration", "color"))
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert (window.selected_action, window.selected_checkbox_id) == ("calibration", "color")
    assert window.current_row_scope.scope == "run"
    assert window.current_row_scope.run_selector_enabled


def test_finish_setup_is_separate_and_opens_without_geometry_canvas_or_raw(method_window, qtbot, monkeypatch):
    from types import SimpleNamespace
    window = method_window
    select_model(window, "phase_model")
    root = _roots(window)["series-a"]
    before = _files(root)
    monkeypatch.setattr(window.geometry_workspace, "ensure_page", lambda: (_ for _ in ()).throw(
        AssertionError("Finish setup must not open a geometry canvas or RAW photograph")))
    jobs = SimpleNamespace(series_id="series-a", registry=SimpleNamespace(jobs=lambda selected_root, series: []))
    monkeypatch.setattr(window, "ensure_jobs", lambda: jobs)
    window._select_row(("setup", "finish"))
    page = window.finish_setup_page
    assert window.content_stack.currentWidget() is page
    qtbot.waitUntil(lambda: not page.preview.busy)
    assert page.apply_build_button.text() == "Finish setup"
    assert not page.calibration_button.isEnabled()
    assert "status unavailable" in page.status.text()  # This fixture has no saved geometry project.
    assert not window.toolbar_builder.play_action.isEnabled()
    assert _files(root) == before


def test_special_analysis_moves_keep_original_routes_and_new_analysis_has_one_mass_owner(method_window, monkeypatch):
    window = method_window
    for item in ("fingers", "segmentation", "thresholding", "volume"):
        assert window.sidebar.display_selection("analysis", item) == ("comparison", "analysis:" + item)
        assert window.sidebar._sections["analysis"].get_row(item) is None
    assert window.sidebar._sections["analysis"].get_row("mass").isHidden()
    assert window.sidebar._sections["export"].get_row("analysis:cropping").label == "Export prepared images"
    select_model(window, "colour_path")
    monkeypatch.setattr(window, "_ensure_run_loaded", lambda: True)
    window._select_row(("jobs", "new_analysis"))
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert (window.selected_action, window.selected_checkbox_id) == ("analysis", "mass")
    assert window.sidebar.get_selected() == ("jobs", "new_analysis")


def test_interactive_controls_require_explicit_colour_context(method_window, monkeypatch):
    window = method_window
    select_model(window, "colour_path")
    monkeypatch.setattr(window, "_ensure_run_loaded", lambda: True)
    before = _files(_roots(window)["series-a"])
    window._select_row(("calibration_controls", "interactive"))
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert window._original_mass_controls is True
    assert (window.selected_action, window.selected_checkbox_id) == ("calibration", "mass")
    window._select_row(("calibration", "mass"))
    assert window.content_stack.currentWidget() is window.mass_calibration_display
    assert window._original_mass_controls is False
    assert not window.toolbar_builder.play_action.isEnabled()
    assert _files(_roots(window)["series-a"]) == before
