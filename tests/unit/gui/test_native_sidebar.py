"""Navigation regressions for the FluidFlower extension of DarSIA's sidebar.

The real window and sidebar handle every selection. Expensive page loading is
replaced by empty local widgets, so these tests need no series, NAS or workflow.
"""

import copy
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtWidgets import QLineEdit, QPushButton


@pytest.fixture
def native_window(qtbot, monkeypatch, tmp_path, request):
    from darsia.fluidflower import native_darsia_app as native

    # Keep startup/dock preferences local to this test, including settings read
    # by the unchanged base window and theme helpers.
    old_format = QSettings.defaultFormat()
    old_organization = QCoreApplication.organizationName()
    old_application = QCoreApplication.applicationName()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    QCoreApplication.setOrganizationName("DarSIA-sidebar-tests")
    QCoreApplication.setApplicationName(tmp_path.name)

    class EmptyProjects:
        def entries(self):
            return []

    series = []
    for key in getattr(request, "param", ()):
        root = tmp_path / key
        root.mkdir()
        (root / "series.toml").write_text(
            f'schema_version = 1\nseries_id = "{key}"\nname = "{key}"\n'
            "[runs]\nids = []\n",
            encoding="utf-8",
        )
        series.append((key, key, root))
    monkeypatch.setattr(native, "configured_series", lambda: tuple(series))
    monkeypatch.setattr(native, "ProjectRegistry", EmptyProjects)
    monkeypatch.setattr(
        native.PhaseModelWorkspace,
        "ensure_studio",
        lambda self: self.studio_placeholder,
    )
    monkeypatch.setattr(
        native.SeriesGeometryWorkspace, "ensure_page", lambda self: self
    )

    window = native.NativeFluidFlowerMainWindow()
    window._test_qtbot = qtbot
    qtbot.addWidget(window, before_close_func=lambda w: qtbot.waitUntil(
        lambda: not w._series_loading and not w._series_loader.busy(), timeout=10000))

    def unexpected_workflow(*args, **kwargs):
        pytest.fail("Sidebar navigation must not start a workflow")

    monkeypatch.setattr(
        window.process_runner, "start_workflow_process", unexpected_workflow
    )
    window.show()
    qtbot.waitUntil(window.isVisible)
    # Navigation resumes only after the native startup metadata read retires.
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    try:
        yield window
    finally:
        qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
        qtbot.waitUntil(lambda: not window._series_loader.busy(), timeout=10000)
        window.close()
        QCoreApplication.setOrganizationName(old_organization)
        QCoreApplication.setApplicationName(old_application)
        QSettings.setDefaultFormat(old_format)


def _allow_reference_for_empty_routing_fixture(window, monkeypatch):
    # These historical A9 fixtures test navigation only and have no runs.
    # Actual A15 reference/refusal tests keep the real loader and guard.
    for root in window.phase_model_workspace.series_roots.values():
        manifest = tomllib.loads((root / "series.toml").read_bytes().decode("utf-8"))
        assert manifest["runs"]["ids"] == []
    monkeypatch.setattr(window, "_load_reference_configuration", lambda: True)


def _wait_metadata(window):
    """Routing assertions observe the accepted asynchronous series snapshot."""
    window._test_qtbot.waitUntil(
        lambda: not window._series_loading and not window._series_loader.busy(),
        timeout=10000)


def _custom_page(window, category):
    if category == "start":
        return window.home_page
    if category == "geometry_facies":
        return window.geometry_workspace
    if category == "phase_model":
        return window.phase_model_workspace
    return window.mass_analysis_placeholder


def test_sidebar_has_one_ordered_navigation_and_preserves_settings(native_window):
    window = native_window
    categories = window.sidebar.categories_data
    assert [entry[0].replace("&&", "&") for entry in categories.values()] == [
        "Start",
        "Setup",
        "Calibration",
        "Mass analysis",
        "Comparisons",
        "Export results",
        "Jobs",
        "Helper",
        "Utils",
        "Legacy workflow",
    ]
    assert window.content_stack.widget(0) is window.settings_scroll_area
    assert not hasattr(window, "workspace_tabs")
    assert not hasattr(window, "workspace_pages")
    assert window.settings_scroll_area.widget() is window.settings_container
    assert not window.series_header.isVisible()
    assert not window.run_combo.isVisible()
    assert not window.setup_series_label.isVisible()


def test_setup_is_one_ordered_list_with_original_action_keys(native_window):
    window = native_window
    section = window.sidebar._sections["setup"]
    assert [label for label, _rows in section.groups] == [""]
    assert [row.label for row in section._rows.values()] == [
        "Protocols",
        "Overflow data",
        "Image selection",
        "Rig Geometry",
        "Layers and facies",
        "Sand map in all runs",
        "Image correction",
        "Colour paths",
        "Finish setup",
        "Build run model",
        "Full setup",
    ]
    # The Setup step is named Image correction; no row help uses the old name.
    assert not [row.label for row in section._rows.values()
                if "image preparation" in row.help_text.casefold()]
    assert "image correction" in section.get_row("setup:finish").help_text
    expected = {
        "preprocessing": window.preprocessing_tab.sidebar_items(),
        "setup": window.setup_tab.sidebar_items(),
    }
    for action, groups in expected.items():
        for _label, rows in groups:
            for text, row_id, _icon, _help in rows:
                if action == "preprocessing" and row_id in (
                    "depth_measurements",
                    "crop",
                ):
                    continue  # The single Depth pane owns both existing steps.
                if action == "setup" and row_id in ("segmentation", "facies", "depth"):
                    continue  # Existing workflows remain, with one visible Layers entry.
                row = section.get_row(f"{action}:{row_id}")
                assert row is not None
                assert row.action == action
                assert row.checkbox_id == row_id
                assert row.help_text
    for row_id in ("rig_frame", "layers", "export"):
        row = section.get_row(f"geometry_facies:{row_id}")
        assert row is not None
        assert (row.action, row.checkbox_id) == ("geometry_facies", row_id)
    assert "preprocessing" not in window.sidebar._sections
    assert "geometry_facies" not in window.sidebar._sections


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_reference_export_does_not_claim_other_runs_geometry_is_applied(
    native_window, monkeypatch
):
    from darsia.fluidflower import setup_display, series_geometry

    window = native_window
    roots = window.phase_model_workspace.series_roots
    monkeypatch.setattr(series_geometry, "geometry_package_root", lambda root: root)
    exported = set()
    monkeypatch.setattr(
        setup_display,
        "completed_current_native_setup_export",
        lambda root: {"revision_id": "current"} if root in exported else None,
    )
    window._select_series("series-a")
    exported.add(roots["series-a"])
    window._update_painted_geometry_status()
    for key in ("segmentation", "facies"):
        assert window.sidebar._sections["setup"].get_row(f"setup:{key}") is None
    export_row = window.sidebar._sections["setup"].get_row("geometry_facies:export")
    assert export_row.label == "Sand map in all runs"
    assert "sand map to every run" in export_row.help_text
    window._select_series("series-b")
    assert export_row.label == "Sand map in all runs"
    assert "applied" not in export_row.label.casefold()


@pytest.mark.parametrize(
    ("action", "row_id"),
    [
        ("preprocessing", "protocols"),
        ("setup", "all"),
        ("setup", "rig"),
    ],
)
def test_merged_setup_click_uses_original_settings_play_and_wizard_keys(
    native_window, monkeypatch, qtbot, action, row_id
):
    from darsia.gui.ui.wizards import has_wizard

    window = native_window
    displayed = []
    original_display = window.settings_factory.display_settings

    def record_display(selected_action, selected):
        displayed.append((selected_action, list(selected)))
        return original_display(selected_action, selected)

    monkeypatch.setattr(window.settings_factory, "display_settings", record_display)
    ran = []
    monkeypatch.setattr(
        window.action_dispatch[action],
        "on_run_clicked",
        lambda: ran.append((window.selected_action, window.selected_checkbox_id)),
    )
    section = window.sidebar._sections["setup"]
    section.set_expanded(True)
    row = section.get_row(f"{action}:{row_id}")
    qtbot.mouseClick(row, Qt.MouseButton.LeftButton)

    assert window.sidebar.get_selected() == (action, row_id)
    assert window.selected_action == action
    assert window.selected_checkbox_id == row_id
    assert displayed[-1] == (action, [row_id])
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert window.series_header.isVisible()
    assert window.toolbar_builder.play_action.isEnabled()
    assert window.toolbar_builder.wizard_action.isEnabled() is has_wizard(action)
    window.toolbar_builder.play_action.trigger()
    assert ran == [(action, row_id)]

    window.sidebar.set_item_state(action, row_id, "done")
    assert row._completion_state == "done"


def test_invalid_sidebar_selection_keeps_the_current_real_row(native_window):
    window = native_window
    window._select_row(("preprocessing", "protocols"))
    _wait_metadata(window)
    selected = window.sidebar.get_selected()
    page = window.content_stack.currentWidget()
    for action, row in (
        ("preprocessing", "does-not-exist"),
        ("setup", "does-not-exist"),
        ("setup", "segmentation"),
        ("setup", "facies"),
        ("unknown-category", "protocols"),
    ):
        window.sidebar.select(action, row)
        assert window.sidebar.get_selected() == selected
        assert window.selected_action == "preprocessing"
        assert window.selected_checkbox_id == "protocols"
        assert window.content_stack.currentWidget() is page


@pytest.mark.parametrize(
    ("category", "row"),
    [
        ("start", "series"),
        ("geometry_facies", "rig_frame"),
        ("geometry_facies", "layers"),
        ("geometry_facies", "export"),
        ("phase_model", "training"),
        ("mass_analysis", "frozen"),
    ],
)
def test_custom_rows_return_to_real_preprocessing_settings(
    native_window, monkeypatch, category, row
):
    window = native_window
    displayed = []
    original_display = window.settings_factory.display_settings

    def record_display(action, selected):
        displayed.append((action, list(selected)))
        return original_display(action, selected)

    monkeypatch.setattr(window.settings_factory, "display_settings", record_display)
    if category == "phase_model":
        window.phase_model_workspace.select_calibration("mass")
    window._select_row((category, row))
    _wait_metadata(window)
    assert window.content_stack.currentWidget() is _custom_page(window, category)
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()
    # The run selector shows on run pages only; the header line names the
    # series on every page but Start (N141).
    assert window.run_combo.isVisible() is (category == "geometry_facies" and row != "layers")
    assert not window.setup_series_label.isVisible()
    assert window.status_header.line.isVisible() is (category != "start")
    if category == "phase_model":
        choices = window.phase_model_workspace.calibration_sidebar
        assert choices.get_selected() == ("calibration", "phase_model_training")
        window.show_mass_calibration()
        assert window.content_stack.currentWidget() is window.mass_calibration_display
        assert window.sidebar.get_selected() == ("calibration", "mass")
        assert not window.toolbar_builder.play_action.isEnabled()

    window._select_row(("preprocessing", "protocols"))
    _wait_metadata(window)
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert displayed[-1] == ("preprocessing", ["protocols"])
    assert window.sidebar.get_selected() == ("preprocessing", "protocols")
    assert window.toolbar_builder.wizard_action.isEnabled()
    assert window.settings_layout.count() > 0


@pytest.mark.parametrize("category", ["start", "calibration", "analysis"])
def test_category_header_routes_page_and_resets_wizard_state(
    native_window, qtbot, category
):
    window = native_window
    window._select_row(("preprocessing", "protocols"))
    assert window.toolbar_builder.wizard_action.isEnabled()

    qtbot.mouseClick(
        window.sidebar._sections[category].header_button, Qt.MouseButton.LeftButton
    )
    expected = {"start": window.home_page, "calibration": window.phase_model_workspace,
                "analysis": window.production_image_selection.panel}
    assert window.content_stack.currentWidget() is expected[category]
    assert window.sidebar.get_selected() == {"start": ("start", "series"),
        "calibration": ("method", "use_for_series"), "analysis": ("image_selection", "production")}[category]
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()

    qtbot.mouseClick(
        window.sidebar._sections["setup"].header_button,
        Qt.MouseButton.LeftButton,
    )
    _wait_metadata(window)
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert window.selected_action == "preprocessing"
    assert window.toolbar_builder.wizard_action.isEnabled()


@pytest.mark.parametrize(
    "category",
    [
        "setup",
        "calibration",
        "analysis",
        "comparison",
        "helper",
        "utils",
    ],
)
def test_series_run_selector_is_visible_for_every_category_except_start(
    native_window, qtbot, category
):
    window = native_window
    qtbot.mouseClick(
        window.sidebar._sections[category].header_button, Qt.MouseButton.LeftButton
    )
    series_page = category in {"calibration", "analysis"}
    assert window.series_header.isVisible() is not series_page
    assert window.run_combo.isVisible() is not series_page
    assert not window.setup_series_label.isVisible()
    assert window.status_header.line.isVisible()
    window._select_row(("start", "series"))
    assert not window.series_header.isVisible()
    assert not window.status_header.line.isVisible()


def test_original_menus_and_dock_controls_survive_page_navigation(native_window):
    window = native_window
    menus = tuple(window.menuBar().actions())
    titles = {action.text().replace("&", "") for action in menus}
    assert {"File", "Settings", "Wizard", "Run", "View", "Help"} <= titles
    assert "preprocessing" in window.menu_builder.wizard_actions
    central = window.centralWidget()
    splitter = window.root_splitter
    log = window.log_dock
    stream = window.streaming_dock
    assert log.widget().isAncestorOf(window.log_text)
    assert stream.widget() is window.streaming_panel

    for selection in [
        ("geometry_facies", "layers"),
        ("phase_model", "training"),
        ("start", "series"),
        ("preprocessing", "protocols"),
    ]:
        window._select_row(selection)
        assert tuple(window.menuBar().actions()) == menus
        assert window.centralWidget() is central
        assert window.root_splitter is splitter
        for button, dock in [
            (window.log_toggle_button, log),
            (window.streaming_toggle_button, stream),
        ]:
            was_visible = dock.isVisible()
            button.click()
            assert dock.isVisible() is not was_visible
            button.click()
            assert dock.isVisible() is was_visible


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
@pytest.mark.parametrize("destination", ["geometry-row", "preprocessing-header"])
def test_start_series_selection_is_applied_before_showing_destination(
    native_window, qtbot, destination
):
    window = native_window
    assert window.phase_model_workspace.series_id == "series-a"
    table = window.home_page.table
    target = next(
        row
        for row in range(table.rowCount())
        if table.item(row, 1).text() == "series-b"
    )
    table.selectRow(target)
    assert window.home_page.selected_series() == "series-b"
    shown_series = []
    window.content_stack.currentChanged.connect(
        lambda _index: shown_series.append(window.phase_model_workspace.series_id)
    )

    if destination == "geometry-row":
        window._select_row(("geometry_facies", "rig_frame"))
        _wait_metadata(window)
        assert window.content_stack.currentWidget() is window.geometry_workspace
    else:
        qtbot.mouseClick(
            window.sidebar._sections["setup"].header_button,
            Qt.MouseButton.LeftButton,
        )
        _wait_metadata(window)
        assert window.content_stack.currentWidget() is window.settings_scroll_area

    assert shown_series and set(shown_series) == {"series-b"}
    assert window.phase_model_workspace.series_id == "series-b"
    assert window.geometry_workspace.series_id == "series-b"
    assert "series-b" in window.setup_series_label.text()


def test_collapsing_active_upstream_category_disables_play(native_window, qtbot):
    window = native_window
    window._select_row(("preprocessing", "protocols"))
    _wait_metadata(window)
    assert window.toolbar_builder.play_action.isEnabled()
    qtbot.mouseClick(
        window.sidebar._sections["setup"].header_button,
        Qt.MouseButton.LeftButton,
    )
    assert window.selected_action is None
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()


@pytest.mark.parametrize(
    "selection",
    [
        ("start", "series"),
        ("geometry_facies", "rig_frame"),
        ("phase_model", "training"),
        ("mass_analysis", "frozen"),
    ],
)
def test_workflow_completion_cannot_enable_play_on_custom_pages(
    native_window, selection
):
    window = native_window
    play = window.toolbar_builder.play_action
    window._select_row(selection)
    assert not play.isEnabled()

    # This is the completion handler's exact button-state operation, including
    # its abort/error paths, exercised without launching an actual process.
    play.setEnabled(True)
    assert not play.isEnabled()

    # The custom-page guard must leave the upstream completion behavior intact.
    window._select_row(("preprocessing", "protocols"))
    _wait_metadata(window)
    assert play.isEnabled()
    play.setEnabled(False)
    play.setEnabled(True)
    assert play.isEnabled()


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_start_shows_series_list_and_current_folder_actions_after_refresh(
    native_window,
):
    window = native_window
    home = native_window.home_page

    def check_controls():
        assert home.new_button.isHidden()
        assert home.open_button.isHidden()
        assert home.open_selected_button.isVisible()
        assert home.update_run_sources_button.isHidden()
        assert home.add_runs_button.isHidden()
        assert home.table.isVisible()
        assert not window.series_header.isVisible()
        assert not hasattr(window, "image_selection_button")
        visible = {
            button
            for button in home.findChildren(QPushButton)
            if button.isVisibleTo(home)
        }
        assert visible == {
            home.new_series_button,
            home.add_existing_button,
            home.refresh_series_button,
            home.folders_button,
            home.remove_series_button,
            home.open_selected_button,
        }
        assert home.open_button.text() == "Open series"
        for button in (
            home.new_series_button,
            home.add_existing_button,
            home.refresh_series_button,
        ):
            assert button.isVisible()
            assert button.isEnabled()

    check_controls()
    home.refresh()
    check_controls()
    target = next(
        row
        for row in range(home.table.rowCount())
        if home.table.item(row, 1).text() == "series-b"
    )
    home.table.selectRow(target)
    assert home.selected_series() == "series-b"
    check_controls()
    home.refresh_series_button.click()
    check_controls()


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_single_click_selects_series_without_training_or_wizard(
    native_window, qtbot, monkeypatch
):
    window = native_window
    _allow_reference_for_empty_routing_fixture(window, monkeypatch)
    home = window.home_page
    target = next(
        row
        for row in range(home.table.rowCount())
        if home.table.item(row, 1).text() == "series-b"
    )
    window.phase_model_workspace.select_calibration("mass")
    monkeypatch.setattr(
        window.phase_model_workspace,
        "ensure_studio",
        lambda: pytest.fail("Opening a series must not load Training"),
    )
    monkeypatch.setattr(
        window,
        "open_wizard_for",
        lambda *_args, **_kwargs: pytest.fail(
            "Opening a series must not open a wizard"
        ),
    )
    original_manifests = {
        key: (root / "series.toml").read_bytes()
        for key, root in window.phase_model_workspace.series_roots.items()
    }
    assert home.open_button.isHidden()
    qtbot.mouseClick(
        home.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=home.table.visualItemRect(home.table.item(target, 0)).center(),
    )

    assert window.content_stack.currentWidget() is home
    assert window.phase_model_workspace.studio_page is None
    assert not window.toolbar_builder.wizard_action.isEnabled()
    assert {
        key: (root / "series.toml").read_bytes()
        for key, root in window.phase_model_workspace.series_roots.items()
    } == original_manifests


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_programmatic_series_selection_and_refresh_do_not_open_workspace(native_window):
    window = native_window
    window.home_page.table.selectRow(1)
    window._refresh_series()
    assert window.content_stack.currentWidget() is window.home_page


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_add_existing_folder_registers_and_opens_series(native_window, monkeypatch):
    from darsia.fluidflower import native_darsia_app as native, series_paths

    window = native_window
    _allow_reference_for_empty_routing_fixture(window, monkeypatch)
    root = window.phase_model_workspace.series_roots["series-b"]
    registered = []
    monkeypatch.setattr(
        native.QFileDialog, "getExistingDirectory", lambda *args: str(root)
    )

    def register(folder):
        registered.append(folder)
        return "series-b", "series-b", root

    monkeypatch.setattr(series_paths, "register_series", register)
    monkeypatch.setattr(
        window.phase_model_workspace,
        "ensure_studio",
        lambda: pytest.fail("Adding a series must not load Training"),
    )
    monkeypatch.setattr(
        window,
        "open_wizard_for",
        lambda *_args, **_kwargs: pytest.fail("Adding a series must not open a wizard"),
    )
    window._add_existing_series()
    _wait_metadata(window)
    assert registered == [root]
    assert window.phase_model_workspace.series_id == "series-b"
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert window.sidebar.get_selected() == ("preprocessing", "protocols")
    assert window.phase_model_workspace.studio_page is None


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_single_click_selects_and_double_click_or_open_button_enters_setup(
    native_window, monkeypatch, qtbot
):
    window = native_window
    _allow_reference_for_empty_routing_fixture(window, monkeypatch)
    home = window.home_page
    target_row = next(
        row
        for row in range(home.table.rowCount())
        if home.table.item(row, 1).text() == "series-b"
    )
    home.table.selectRow(target_row)
    assert window.content_stack.currentWidget() is window.home_page
    selected = []
    select_row = window._select_row

    def record(selection):
        selected.append(selection)
        return select_row(selection)

    monkeypatch.setattr(window, "_select_row", record)
    home.table.cellClicked.emit(target_row, 0)
    assert selected == []
    assert window.content_stack.currentWidget() is home
    home.table.cellDoubleClicked.emit(target_row, 0)
    home.table.cellClicked.emit(target_row, 0)
    qtbot.waitUntil(lambda:selected == [("preprocessing", "protocols")], timeout=10000)
    assert selected == [("preprocessing", "protocols")]
    assert window.phase_model_workspace.series_id == "series-b"
    assert window.geometry_workspace.series_id == "series-b"
    assert window.sidebar.get_selected() == ("preprocessing", "protocols")
    assert window.phase_model_workspace.studio_page is None
    # Idempotence must not suppress opening the same series after returning.
    select_row(("start", "series"))
    home.open_selected_button.click()
    qtbot.waitUntil(lambda:len(selected) == 2, timeout=10000)
    assert selected == [("preprocessing", "protocols"), ("preprocessing", "protocols")]


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_hidden_open_signal_and_compatibility_entries_use_setup(
    native_window, monkeypatch
):
    window = native_window
    _allow_reference_for_empty_routing_fixture(window, monkeypatch)
    table = window.home_page.table
    target_row = next(
        row
        for row in range(table.rowCount())
        if table.item(row, 1).text() == "series-b"
    )
    table.selectRow(target_row)
    root = window.phase_model_workspace.series_roots["series-b"]
    for opening in (
        window.home_page.open_requested.emit,
        lambda: window._show_training_for_path(str(root)),
        window._show_home_calibration,
    ):
        window._select_row(("start", "series"))
        opening()
        _wait_metadata(window)
        assert window.sidebar.get_selected() == ("preprocessing", "protocols")
        assert window.content_stack.currentWidget() is window.settings_scroll_area
        assert window.phase_model_workspace.series_id == "series-b"
        assert window.phase_model_workspace.studio_page is None


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_setup_open_respects_existing_series_change_rejection(
    native_window, monkeypatch
):
    window = native_window
    root = window.phase_model_workspace.series_roots["series-b"]
    requested = []
    monkeypatch.setattr(window, "_select_series", requested.append)
    window._open_series_from_start(str(root))
    assert requested == ["series-b"]
    assert window.phase_model_workspace.series_id == "series-a"
    assert window.content_stack.currentWidget() is window.home_page
    assert window.sidebar.get_selected() == ("start", "series")


def test_start_open_preserves_legacy_project_original_opening(
    native_window, monkeypatch
):
    window = native_window
    path = "earlier-series.json-project"
    window.home_page.series_registry.legacy_by_path[path] = {"path": path}
    opened = []
    monkeypatch.setattr(
        window,
        "_open_legacy_project",
        lambda *args, **kwargs: opened.append((args, kwargs)),
    )
    window._open_series_from_start(path)
    assert opened == [((path,), {})]
    assert window.content_stack.currentWidget() is window.home_page


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_image_selection_is_reached_from_setup_for_selected_series(
    native_window, monkeypatch, qtbot
):
    from PySide6.QtWidgets import QComboBox

    window = native_window
    table = window.home_page.table
    row = next(
        row
        for row in range(table.rowCount())
        if table.item(row, 1).text() == "series-b"
    )
    table.selectRow(row)
    root = window.phase_model_workspace.series_roots["series-b"]
    calls = []

    class SelectionDialog:
        def __init__(self, path, parent, **_options):
            calls.append((path, parent))
            self.run_combo = QComboBox(parent)
            self.saved = SimpleNamespace(connect=lambda callback: None)

        def exec(self):
            return 0

    monkeypatch.setattr(window.image_selection_workflow, "dialog_factory", SelectionDialog)
    window._select_row(("image_selection", "plan"))
    _wait_metadata(window)
    section = window.sidebar._sections["setup"]
    plan = section.get_row("image_selection:plan")
    assert plan is not None
    assert plan.label == "Image selection"
    assert window.sidebar._sections["calibration"].get_row("phase_model:selection_plan") is None
    qtbot.mouseClick(plan, Qt.MouseButton.LeftButton)

    # Selecting the row shows its panel; opening the editor remains explicit.
    assert calls == []
    assert window.sidebar.get_selected() == ("image_selection", "plan")
    assert window.content_stack.currentWidget() is window.image_selection_workflow.panel
    # A whole-series page: no run selector, the header line names the series.
    assert not window.series_header.isVisible()
    assert "Whole series" in window.status_header.line.text()
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()
    qtbot.mouseClick(window.image_selection_workflow.edit_button, Qt.MouseButton.LeftButton)
    assert calls == [(root, window)]


def test_image_selection_without_series_does_not_open_dialog(
    native_window, monkeypatch
):
    def unexpected_dialog(*args, **kwargs):
        pytest.fail("Choose a series before opening its image selection editor")

    monkeypatch.setattr(native_window.image_selection_workflow, "dialog_factory", unexpected_dialog)
    native_window._select_row(("image_selection", "plan"))
    assert not native_window.image_selection_workflow.edit_button.isEnabled()
    native_window.image_selection_workflow.open()
    assert not native_window.toolbar_builder.play_action.isEnabled()


@pytest.fixture
def run_window(native_window, monkeypatch, tmp_path):
    """Two local series with distinct run-only and premerged configuration files."""
    from darsia.fluidflower import native_darsia_app as native

    roots = {}
    common = {}
    runs = {}
    merged = {}
    paths = []
    for series_index, key in enumerate(("series-a", "series-b")):
        root = tmp_path / key
        roots[key] = root
        for folder in ("config/runs", "config/merged"):
            (root / folder).mkdir(parents=True)
        manifest = root / "series.toml"
        manifest.write_text(
            f'schema_version = 1\nseries_id = "{key}"\nname = "{key}"\n'
            '[runs]\nids = ["first", "second"]\n',
            encoding="utf-8",
        )
        common[key] = root / "config/common.toml"
        common[key].write_text(
            f'common_marker = "{key}"\n[rig]\nheight = 0.5\n',
            encoding="utf-8",
        )
        paths.extend((manifest, common[key]))
        runs[key] = {}
        merged[key] = {}
        for run_index, run in enumerate(("first", "second")):
            width = 1.0 + 2.0 * series_index + run_index
            runs[key][run] = root / f"config/runs/{run}.toml"
            merged[key][run] = root / f"config/merged/{run}.toml"
            runs[key][run].write_text(
                f'run_marker = "{run}"\n[rig]\nwidth = {width}\n',
                encoding="utf-8",
            )
            merged[key][run].write_text(
                f'common_marker = "{key}"\nrun_marker = "{run}"\n'
                f"[rig]\nheight = 0.5\nwidth = {width}\n",
                encoding="utf-8",
            )
            paths.extend((runs[key][run], merged[key][run]))
    configured = tuple((key, key, root) for key, root in roots.items())
    monkeypatch.setattr(native, "configured_series", lambda: configured)
    warnings = []
    monkeypatch.setattr(
        native.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    native_window._refresh_series(preferred_series="series-a", show_errors=True)
    _wait_metadata(native_window)
    return SimpleNamespace(
        window=native_window,
        roots=roots,
        common=common,
        runs=runs,
        merged=merged,
        paths=paths,
        warnings=warnings,
    )


def _select_run(window, series, run, *, activate=True):
    table = window.home_page.table
    row = next(
        row for row in range(table.rowCount()) if table.item(row, 1).text() == series
    )
    table.selectRow(row)
    if activate:
        table.cellDoubleClicked.emit(row, 0)
        _wait_metadata(window)
    assert window.phase_model_workspace.series_id == series
    index = window.run_combo.findText(run)
    assert index >= 0
    window.run_combo.setCurrentIndex(index)
    if activate:
        window.run_combo.activated.emit(index)
    window._update_setup_action()


def _wizard_dispatch_spy(monkeypatch):
    from darsia.gui.ui import main_window

    calls = []

    class WizardSpy:
        def __init__(self, parent):
            self.parent = parent

        def exec(self):
            calls.append(
                {
                    "path": self.parent.config_file,
                    "config": copy.deepcopy(self.parent.config_dict),
                    "selection": self.parent.sidebar.get_selected(),
                }
            )
            return 0

    def registered_dialog(category):
        assert category == "preprocessing"
        return WizardSpy

    monkeypatch.setattr(main_window, "load_wizard_dialog_class", registered_dialog)
    return calls


def _config_snapshot(window):
    return {
        "object": window.config_dict,
        "values": copy.deepcopy(window.config_dict),
        "file": window.config_file,
        "label": window.config_path_label.text(),
        "series": window._config_series,
        "stamp": window._config_stamp,
        "view": copy.deepcopy(window._last_settings_view),
    }


def _assert_config_restored(window, previous):
    assert window.config_dict is previous["object"]
    assert window.config_dict == previous["values"]
    assert window.config_file == previous["file"]
    assert window.config_path_label.text() == previous["label"]
    assert window._config_series == previous["series"]
    assert window._config_stamp == previous["stamp"]
    assert window._last_settings_view == previous["view"]


def test_preprocessing_handoff_loads_selected_premerged_run_with_real_controller(
    run_window, monkeypatch, qtbot
):
    fixture = run_window
    window = fixture.window
    calls = _wizard_dispatch_spy(monkeypatch)
    _select_run(window, "series-b", "second")
    assert window.home_page.new_button.isEnabled()
    window._open_preprocessing_for_selected_run()

    assert len(calls) == 1
    assert Path(calls[0]["path"]) == fixture.merged["series-b"]["second"]
    assert calls[0]["config"]["common_marker"] == "series-b"
    assert calls[0]["config"]["run_marker"] == "second"
    assert calls[0]["config"]["rig"]["height"] == 0.5
    assert calls[0]["config"]["rig"]["width"] == 4.0
    assert calls[0]["selection"] == ("preprocessing", "protocols")
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert fixture.warnings == []


def test_run_category_wizard_loads_current_run_with_original_preprocessing_key(
    run_window, monkeypatch
):
    fixture = run_window
    window = fixture.window
    calls = _wizard_dispatch_spy(monkeypatch)
    _select_run(window, "series-b", "second")
    window._select_row(("preprocessing", "protocols"))
    assert window.toolbar_builder.wizard_action.isEnabled()
    window.toolbar_builder.wizard_action.trigger()

    assert len(calls) == 1
    assert Path(calls[0]["path"]) == fixture.merged["series-b"]["second"]
    assert calls[0]["config"]["run_marker"] == "second"
    assert calls[0]["selection"] == ("preprocessing", "protocols")
    assert window.home_page.new_button.isHidden()


@pytest.mark.parametrize("failure", ["malformed", "missing"])
def test_invalid_selected_run_never_opens_wizard_with_previous_config(
    run_window, monkeypatch, failure
):
    fixture = run_window
    window = fixture.window
    _select_run(window, "series-a", "first")
    assert window._load_run_config(str(fixture.merged["series-a"]["first"]))
    previous = _config_snapshot(window)
    calls = _wizard_dispatch_spy(monkeypatch)
    _select_run(window, "series-a", "second", activate=False)
    assert window.home_page.new_button.isEnabled()
    target = fixture.merged["series-a"]["second"]
    if failure == "malformed":
        target.write_text('value = "unterminated', encoding="utf-8")
    else:
        target.unlink()
    before = {path: path.read_bytes() for path in fixture.paths if path.exists()}
    window._open_preprocessing_for_selected_run()

    assert calls == []
    assert fixture.warnings
    _assert_config_restored(window, previous)
    assert {path: path.read_bytes() for path in before} == before
    assert "Error loading config file:" in window.log_text.toPlainText()


def test_same_path_failed_reload_cannot_reuse_previously_parsed_values(
    run_window, monkeypatch
):
    fixture = run_window
    window = fixture.window
    target = fixture.merged["series-a"]["first"]
    _select_run(window, "series-a", "first")
    assert window._load_run_config(str(target))
    previous = _config_snapshot(window)
    calls = _wizard_dispatch_spy(monkeypatch)
    target.write_text('value = "unterminated', encoding="utf-8")
    window._open_preprocessing_for_selected_run()

    assert calls == []
    assert fixture.warnings
    _assert_config_restored(window, previous)
    assert target.read_text("utf-8") == 'value = "unterminated'


def test_post_assignment_load_failure_restores_old_config_and_editors(
    run_window, monkeypatch
):
    fixture = run_window
    window = fixture.window
    _select_run(window, "series-a", "first")
    assert window._load_run_config(str(fixture.merged["series-a"]["first"]))
    previous = _config_snapshot(window)
    calls = _wizard_dispatch_spy(monkeypatch)
    target = fixture.merged["series-a"]["second"]
    _select_run(window, "series-a", "second", activate=False)
    original_refresh = window._config_loaded
    injected = []

    def fail_after_new_editors_are_built():
        if Path(window.config_file) == target and not injected:
            assert window.config_dict["rig"]["width"] == 2.0
            assert "rig.width" in window.settings_inputs
            injected.append(True)
            raise RuntimeError("injected failure after config and editor assignment")
        return original_refresh()

    monkeypatch.setattr(
        window, "_config_loaded", fail_after_new_editors_are_built
    )
    before = {path: path.read_bytes() for path in fixture.paths}
    window._open_preprocessing_for_selected_run()

    assert injected == [True]
    assert calls == []
    assert fixture.warnings
    _assert_config_restored(window, previous)
    # A later save reads these real rebuilt editors. They must not copy B's
    # width into the restored A dictionary, even after pending Qt fills run.
    window.settings_factory._sync_settings_inputs_to_config_dict()
    assert window.config_dict["rig"]["width"] == 1.0
    assert {path: path.read_bytes() for path in fixture.paths} == before


def test_real_wizard_finish_writes_only_the_selected_merged_file(
    run_window, monkeypatch
):
    from darsia.gui.ui import main_window, preprocessing_wizard
    from darsia.gui.ui.settings import unwrap_composite_widget

    fixture = run_window
    window = fixture.window
    _select_run(window, "series-b", "second")
    target = fixture.merged["series-b"]["second"]
    before = {path: path.read_bytes() for path in fixture.paths}
    dialog_class = preprocessing_wizard.PreprocessingWizardDialog
    monkeypatch.setattr(
        dialog_class,
        "_load_catalogue",
        lambda self: preprocessing_wizard.SeriesCatalogue(),
    )
    monkeypatch.setattr(main_window, "load_wizard_dialog_class", lambda _: dialog_class)
    finished = []

    def edit_and_finish(dialog):
        assert Path(dialog.main_window.config_file) == target
        rig_index = next(
            index
            for index, step in enumerate(preprocessing_wizard.STEPS)
            if step[0] == "rig"
        )
        dialog._goto(rig_index)
        assert "rig.width" not in window.settings_inputs
        data_index = next(
            index
            for index, step in enumerate(preprocessing_wizard.STEPS)
            if step[0] == "data"
        )
        dialog._goto(data_index)
        results = unwrap_composite_widget(window.settings_inputs["data.results"])
        assert isinstance(results, QLineEdit)
        results.setText(str(target.parent / "wizard-saved-results"))
        dialog._goto(len(preprocessing_wizard.STEPS) - 1)
        dialog._run_protocols_checkbox.setChecked(False)
        dialog._run_depth_measurements_checkbox.setChecked(False)
        dialog._finish()
        finished.append(dialog.result())
        return dialog.result()

    def unexpected_preprocessing(*args, **kwargs):
        pytest.fail("Finish with both compute checkboxes off must only save")

    monkeypatch.setattr(dialog_class, "exec", edit_and_finish)
    monkeypatch.setattr(dialog_class, "_start_preprocessing", unexpected_preprocessing)
    window._open_preprocessing_for_selected_run()

    assert finished == [1]
    saved = tomllib.loads(target.read_text("utf-8"))
    original_rig = tomllib.loads(before[target].decode("utf-8"))["rig"]
    assert all(saved["rig"][key] == value for key, value in original_rig.items())
    assert saved["data"]["results"] == str(target.parent / "wizard-saved-results")
    assert saved["common_marker"] == "series-b"
    assert saved["run_marker"] == "second"
    assert target.read_bytes() != before[target]
    assert all(
        path.read_bytes() == raw for path, raw in before.items() if path != target
    )
    assert fixture.warnings == []


def test_unchanged_open_run_keeps_pending_widget_edit_for_real_wizard(
    run_window, monkeypatch
):
    from darsia.gui.ui import main_window, preprocessing_wizard
    from darsia.gui.ui.settings import unwrap_composite_widget

    fixture = run_window
    window = fixture.window
    target = fixture.merged["series-a"]["first"]
    _select_run(window, "series-a", "first")
    assert window._load_run_config(str(target))
    window.settings_factory.display_full_settings()
    QCoreApplication.processEvents()
    width = unwrap_composite_widget(window.settings_inputs["rig.width"])
    assert isinstance(width, QLineEdit)
    width.setText("1.75")
    before = {path: path.read_bytes() for path in fixture.paths}
    dialog_class = preprocessing_wizard.PreprocessingWizardDialog
    monkeypatch.setattr(
        dialog_class,
        "_load_catalogue",
        lambda self: preprocessing_wizard.SeriesCatalogue(),
    )
    monkeypatch.setattr(main_window, "load_wizard_dialog_class", lambda _: dialog_class)
    seen = []

    def inspect_then_cancel(dialog):
        # The unchanged real constructor has already synchronized the fields.
        seen.append(
            (
                Path(dialog.main_window.config_file),
                dialog.main_window.config_dict["rig"],
            )
        )
        dialog.reject()
        return dialog.result()

    monkeypatch.setattr(dialog_class, "exec", inspect_then_cancel)
    window._open_preprocessing_for_selected_run()

    assert seen and seen[0][0] == target
    assert seen[0][1]["width"] == 1.75
    assert window.config_dict["rig"]["width"] == 1.75
    assert {path: path.read_bytes() for path in fixture.paths} == before
    assert fixture.warnings == []


def test_setup_button_tracks_selected_series_run_and_missing_file(run_window):
    fixture = run_window
    window = fixture.window
    window.run_combo.setCurrentIndex(-1)
    assert not window.home_page.new_button.isEnabled()
    assert window.home_page.new_button.toolTip()

    _select_run(window, "series-a", "first")
    assert window.home_page.new_button.isEnabled()
    table = window.home_page.table
    row = next(
        row
        for row in range(table.rowCount())
        if table.item(row, 1).text() == "series-b"
    )
    table.selectRow(row)
    window._update_setup_action()
    assert window.home_page.selected_series() == "series-b"
    assert not window.home_page.new_button.isEnabled()

    _select_run(window, "series-b", "second")
    assert window.home_page.new_button.isEnabled()
    fixture.merged["series-b"]["second"].unlink()
    window._refresh_series(preferred_series="series-b", show_errors=True)
    _wait_metadata(window)
    window._update_setup_action()
    assert not window.home_page.new_button.isEnabled()
    assert window.home_page.new_button.toolTip()


def test_split_config_without_merged_stays_disabled_until_common_is_empty(run_window):
    fixture = run_window
    window = fixture.window
    fixture.merged["series-a"]["first"].unlink()
    window._populate_runs()
    first = window.run_combo.findText("first")
    assert not window.run_combo.model().item(first).isEnabled()
    assert not window.run_combo.itemData(first)
    window.run_combo.setCurrentIndex(first)
    assert not window.home_page.new_button.isEnabled()
    assert "no merged file" in window.run_combo.itemData(first, Qt.ToolTipRole)

    fixture.common["series-a"].write_text(
        "# No common settings yet\n", encoding="utf-8"
    )
    window._populate_runs()
    first = window.run_combo.findText("first")
    assert window.run_combo.model().item(first).isEnabled()
    assert Path(window.run_combo.itemData(first)) == fixture.runs["series-a"]["first"]
    window.run_combo.setCurrentIndex(first)
    assert window.home_page.new_button.isEnabled()


@pytest.fixture
def new_series_flow(native_window, monkeypatch, tmp_path):
    """Create a real local series, replacing only the modal wait of each wizard."""
    import os

    from PIL import Image

    from darsia.fluidflower import native_darsia_app as native
    from darsia.fluidflower import series_creation, series_paths, series_wizard
    from darsia.gui.ui import preprocessing_wizard

    method = native_window.home_page.new_series_method
    method.setCurrentIndex(method.findData("phase_model"))

    # Registration is a TOML file, not QSettings. Keep it and every discovery
    # source local; a workstation's registered NAS series must never be read.
    for name in tuple(os.environ):
        if (
            name.startswith("DARSIA_") and name.endswith("_PROJECT_ROOT")
        ) or name == "DARSIA_FLUIDFLOWER_DATA_ROOT":
            monkeypatch.delenv(name)
    registry = tmp_path / "registry.toml"
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(native, "configured_series", series_paths.configured_series)

    experiments = tmp_path / "experiments"
    sources = []
    for run in ("zrun", "arun"):
        folder = experiments / run
        folder.mkdir(parents=True)
        image = folder / "frame001.png"
        Image.new("RGB", (16, 12), (110, 140, 170)).save(image)
        sources.append(image)
    state = SimpleNamespace(
        window=native_window,
        experiments=experiments,
        root=experiments / "DarSIA",
        registry=registry,
        sources={path: path.read_bytes() for path in sources},
        rig_class="medium",
        physical_rig="FF AC",
        cancel_creation=False,
        finish_preprocessing=False,
        edited_results=None,
        created_originals={},
        merged_at_dialog={},
        creations=[],
        preprocessing=[],
        warnings=[],
        loads=[],
    )
    monkeypatch.setattr(
        native.QMessageBox,
        "warning",
        lambda _parent, title, message: state.warnings.append((title, message)),
    )
    monkeypatch.setattr(native.QMessageBox, "information", lambda *args: None)
    original_load = native_window._load_run_config

    def load_config(path):
        result = original_load(path)
        if result:
            state.loads.append(Path(native_window.config_file))
        return result

    monkeypatch.setattr(native_window, "_load_run_config", load_config)

    def create_dialog(wizard):
        state.creations.append(wizard)
        page = wizard.series_page
        page.name.setText("New UI series")
        page.series_id.setText("FF_UI_NEW")
        page.folder.setText(str(experiments))
        page.rig_class.setCurrentIndex(page.rig_class.findData(state.rig_class))
        if state.rig_class == "medium":
            page.physical_rig.setCurrentIndex(
                page.physical_rig.findData(state.physical_rig)
            )
        assert page.isComplete()
        if state.cancel_creation:
            wizard.reject()
            return wizard.result()
        wizard.experiments_page.runs = [
            series_creation.scan_run_folder(experiments / run)
            for run in ("zrun", "arun")
        ]
        assert wizard.experiments_page.isComplete()
        wizard.accept()
        assert wizard.created == state.root
        state.created_originals = {
            path: path.read_bytes() for path in state.root.rglob("*") if path.is_file()
        }
        return wizard.result()

    def preprocessing_dialog(dialog):
        from darsia.gui.ui.settings import unwrap_composite_widget

        preset = dialog._selected_preset
        state.preprocessing.append(
            {
                "path": Path(dialog.main_window.config_file),
                "config": copy.deepcopy(dialog.main_window.config_dict),
                "selection": dialog.main_window.sidebar.get_selected(),
                "preset_rig": preset.rig if preset is not None else None,
                "preset_pieces": set(preset.pieces()) if preset is not None else set(),
                "step": dialog._index,
                "class": type(dialog),
            }
        )
        state.merged_at_dialog = {
            path: path.read_bytes()
            for path in (state.root / "config/merged").glob("*.toml")
        }
        if state.finish_preprocessing:
            # The main settings refresh during preset application must not
            # replace the living wizard Data editors in its save registry.
            dialog.show()
            QCoreApplication.processEvents()
            results = unwrap_composite_widget(
                dialog.main_window.settings_inputs["data.results"]
            )
            assert isinstance(results, QLineEdit)
            assert dialog.isAncestorOf(results)
            assert results.isVisibleTo(dialog)
            results.setText(state.edited_results)
            dialog._on_next()
            assert (
                dialog.main_window.config_dict["data"]["results"]
                == state.edited_results
            )
            dialog._goto(len(preprocessing_wizard.STEPS) - 1)
            dialog.main_window.config_dict["id"]["comment"] = "Saved by UI wizard"
            dialog._run_protocols_checkbox.setChecked(False)
            dialog._run_depth_measurements_checkbox.setChecked(False)
            dialog._finish()
        else:
            dialog.reject()
        return dialog.result()

    def unexpected_preprocessing(*args, **kwargs):
        pytest.fail("This UI test must not start a preprocessing calculation")

    monkeypatch.setattr(series_wizard.NewSeriesFolderWizard, "exec", create_dialog)
    monkeypatch.setattr(
        preprocessing_wizard.PreprocessingWizardDialog, "exec", preprocessing_dialog
    )
    monkeypatch.setattr(
        preprocessing_wizard.PreprocessingWizardDialog,
        "_start_preprocessing",
        unexpected_preprocessing,
    )
    return state


@pytest.mark.parametrize(
    ("rig_class", "physical_rig", "expected_size"),
    [
        ("medium", "FF AC", (0.9, 0.49)),
        ("medium", "FF Bilbo", (0.92, 0.55)),
        ("large", None, (2.8, 1.5)),
    ],
)
def test_new_series_opens_real_preprocessing_on_first_manifest_run(
    new_series_flow, qtbot, rig_class, physical_rig, expected_size
):
    from darsia.gui.ui.preprocessing_wizard import PreprocessingWizardDialog

    state = new_series_flow
    state.rig_class = rig_class
    state.physical_rig = physical_rig
    qtbot.mouseClick(
        state.window.home_page.new_series_button, Qt.MouseButton.LeftButton
    )
    qtbot.waitUntil(lambda: not state.window._series_loading, timeout=10000)

    assert len(state.creations) == 1
    assert len(state.preprocessing) == 1
    first = state.root / "config/merged/zrun.toml"
    opened = state.preprocessing[0]
    assert opened["class"] is PreprocessingWizardDialog
    assert opened["path"] == first
    assert first in state.loads
    assert opened["selection"] == ("preprocessing", "protocols")
    # The explicit Start choice was already applied; the native wizard shows
    # a read-only summary instead of a second live preset question.
    assert opened["preset_rig"] is None
    assert opened["preset_pieces"] == set()
    assert opened["step"] == 0
    assert state.window.run_combo.currentText() == "zrun"
    manifest = tomllib.loads((state.root / "series.toml").read_text("utf-8"))
    assert manifest["method"] == {"active": "phase_model"}
    assert manifest["paths"]["geometry_facies"] == "../Results/setup/Geometry"
    assert manifest["rig_class"] == rig_class
    assert manifest.get("rig_profile") == physical_rig
    assert manifest["runs"]["ids"] == ["zrun", "arun"]
    for run in ("zrun", "arun"):
        path = state.root / f"config/merged/{run}.toml"
        assert path.is_file()
        values = tomllib.loads(path.read_text("utf-8"))
        assert (values["rig"]["width"], values["rig"]["height"]) == expected_size
        assert values["id"]["run_id"] == f"FF_UI_NEW {run.upper()}"
        assert "depth" not in values
        assert "corrections" not in values
        index = state.window.run_combo.findText(run)
        assert state.window.run_combo.model().item(index).isEnabled()
        assert Path(state.window.run_combo.itemData(index)) == path
    assert all(
        path.read_bytes() == raw for path, raw in state.created_originals.items()
    )
    assert all(path.read_bytes() == raw for path, raw in state.sources.items())
    assert all(path.read_bytes() == raw for path, raw in state.merged_at_dialog.items())
    assert state.warnings == []


@pytest.mark.parametrize("physical_rig", ["FF AC", "FF Bilbo"])
def test_new_series_preprocessing_finish_keeps_visible_data_edit_in_first_run(
    new_series_flow, qtbot, physical_rig
):
    state = new_series_flow
    state.physical_rig = physical_rig
    state.finish_preprocessing = True
    state.edited_results = str(state.experiments / "chosen-ui-results")
    qtbot.mouseClick(
        state.window.home_page.new_series_button, Qt.MouseButton.LeftButton
    )
    qtbot.waitUntil(lambda: not state.window._series_loading, timeout=10000)

    first = state.root / "config/merged/zrun.toml"
    second = state.root / "config/merged/arun.toml"
    assert len(state.preprocessing) == 1
    saved = tomllib.loads(first.read_text("utf-8"))
    assert saved["id"]["comment"] == "Saved by UI wizard"
    assert saved["data"]["results"] == state.edited_results
    assert first.read_bytes() != state.merged_at_dialog[first]
    assert second.read_bytes() == state.merged_at_dialog[second]
    assert all(
        path.read_bytes() == raw for path, raw in state.created_originals.items()
    )
    assert all(path.read_bytes() == raw for path, raw in state.sources.items())
    assert state.warnings == []


def test_cancelled_new_series_does_not_register_or_open_preprocessing(
    new_series_flow, qtbot
):
    state = new_series_flow
    state.cancel_creation = True
    qtbot.mouseClick(
        state.window.home_page.new_series_button, Qt.MouseButton.LeftButton
    )

    assert len(state.creations) == 1
    assert state.preprocessing == []
    assert not state.root.exists()
    assert not state.registry.exists()
    assert all(path.read_bytes() == raw for path, raw in state.sources.items())


def test_new_series_registration_failure_preserves_created_files_without_handoff(
    new_series_flow, monkeypatch, qtbot
):
    from darsia.fluidflower import series_paths

    state = new_series_flow

    def rejected_registration(_root):
        raise ValueError("injected duplicate registered series")

    monkeypatch.setattr(series_paths, "register_series", rejected_registration)
    qtbot.mouseClick(
        state.window.home_page.new_series_button, Qt.MouseButton.LeftButton
    )

    assert len(state.creations) == 1
    assert state.preprocessing == []
    assert not (state.root / "config/merged").exists()
    assert all(
        path.read_bytes() == raw for path, raw in state.created_originals.items()
    )
    assert state.warnings and "injected duplicate" in state.warnings[-1][1]


def test_failed_new_run_load_never_opens_preprocessing_with_previous_values(
    new_series_flow, monkeypatch, qtbot
):
    state = new_series_flow
    original_prepare = state.window._prepare_new_series_runs

    def corrupt_new_run(root, **kwargs):
        run, path = original_prepare(root, **kwargs)
        path.write_text('value = "unterminated', encoding="utf-8")
        return run, path

    monkeypatch.setattr(state.window, "_prepare_new_series_runs", corrupt_new_run)
    qtbot.mouseClick(
        state.window.home_page.new_series_button, Qt.MouseButton.LeftButton
    )
    qtbot.waitUntil(lambda: not state.window._series_loading, timeout=10000)

    assert len(state.creations) == 1
    assert state.preprocessing == []
    assert state.warnings
    assert "Error loading config file:" in state.window.log_text.toPlainText()
    assert all(
        path.read_bytes() == raw for path, raw in state.created_originals.items()
    )
    assert all(path.read_bytes() == raw for path, raw in state.sources.items())


def test_medium_requires_a_physical_rig_without_changing_the_rig_class(
    native_window, qtbot, tmp_path
):
    from darsia.fluidflower.series_wizard import NewSeriesFolderWizard

    wizard = NewSeriesFolderWizard(native_window)
    qtbot.addWidget(wizard)
    wizard.show()
    page = wizard.series_page
    page.name.setText("Local UI series")
    page.series_id.setText("FF_UI_PROFILE")
    page.folder.setText(str(tmp_path))
    page.rig_class.setCurrentIndex(page.rig_class.findData("medium"))
    assert page.rig_class.currentData() == "medium"
    assert page.rig_class.currentText() == "Medium FluidFlower"
    assert page.physical_rig.isVisible()
    assert page.physical_rig.currentData() in (None, "")
    assert not page.isComplete()
    for profile in ("FF AC", "FF Bilbo"):
        index = page.physical_rig.findData(profile)
        assert index >= 0
        page.physical_rig.setCurrentIndex(index)
        assert page.isComplete()
        assert wizard.draft().rig_class == "medium"
    page.rig_class.setCurrentIndex(page.rig_class.findData("large"))
    assert not page.physical_rig.isVisible()
    assert page.isComplete()
    assert wizard.draft().rig_class == "large"
    wizard.reject()


def test_comparisons_show_one_wasserstein_row_and_keep_the_original_steps_routable(native_window):
    window = native_window
    window._order_method_groups()
    section = window.sidebar._sections["comparison"]
    visible = [row.label for row in section._rows.values() if not row.isHidden()]
    assert [label for label in visible if "Wasserstein" in label] == ["Wasserstein comparison"]
    assert {"Fingers", "Segmentation", "Thresholding", "Volume", "Events"} <= set(visible)
    for item in ("wasserstein_compute", "wasserstein_assemble"):
        assert section.get_row(item) is not None and section.get_row(item).isHidden()


def test_comparisons_events_opens_the_events_page_and_keeps_the_original_step_routable(native_window):
    window = native_window
    window._order_method_groups()
    section = window.sidebar._sections["comparison"]
    visible = [row.label for row in section._rows.values() if not row.isHidden()]
    assert visible.count("Events") == 1
    assert section.get_row("events") is not None and section.get_row("events").isHidden()
    window._on_sidebar_selection("events", "question")
    assert window.content_stack.currentWidget() is window.events_page
    assert window.events_page.status.text() == "Choose a registered series on Start."
