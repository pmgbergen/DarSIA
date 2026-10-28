"""Native method selection: local manifests, real controls, no calculations."""

import copy
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, QSignalBlocker, Qt
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower.series_method import MethodSelection, save_series_method


@pytest.fixture
def method_window(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower import native_darsia_app as native

    old = (
        QSettings.defaultFormat(),
        QCoreApplication.organizationName(),
        QCoreApplication.applicationName(),
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )
    QCoreApplication.setOrganizationName("DarSIA-method-tests")
    QCoreApplication.setApplicationName(tmp_path.name)
    series = []
    for key in ("series-a", "series-b"):
        root = tmp_path / key
        root.mkdir()
        (root / "series.toml").write_text(
            f'schema_version = 1\nseries_id = "{key}"\nname = "{key}"\n'
            "[runs]\nids = []\n[future]\nopaque = [1, 2, 3]\n",
            encoding="utf-8",
        )
        (root / "calibration-original.json").write_text(
            '{"preserve":true}', encoding="utf-8"
        )
        series.append((key, key, root))
    monkeypatch.setattr(native, "configured_series", lambda: tuple(series))
    monkeypatch.setattr(
        native, "ProjectRegistry", lambda: SimpleNamespace(entries=lambda: [])
    )
    monkeypatch.setattr(
        native.PhaseModelWorkspace,
        "ensure_studio",
        lambda self: self.studio_placeholder,
    )
    monkeypatch.setattr(
        native.SeriesGeometryWorkspace, "ensure_page", lambda self: self
    )
    monkeypatch.setattr(
        native.NativeFluidFlowerMainWindow,
        "ensure_mass_analysis",
        lambda self: self.mass_analysis_placeholder,
    )
    window = native.NativeFluidFlowerMainWindow(initial_series="series-a")
    # The stubbed results page accepts the saved-results view request.
    window.mass_analysis_placeholder.show_view = lambda view: None
    def retire_method_reader(owned):
        # QtBot closes/deletes registered widgets before fixture finalizers.
        # Retire the original owned read while its receiver is still alive.
        owned.method_panel.calls.stop_pending()
        qtbot.waitUntil(owned.method_panel.can_close, timeout=10000)
        owned.method_panel.calls.stop_pending()

    qtbot.addWidget(window, before_close_func=retire_method_reader)
    window.show()
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    monkeypatch.setattr(
        window.process_runner,
        "start_workflow_process",
        lambda *_a, **_k: pytest.fail("Unexpected numerical workflow"),
    )
    try:
        yield window
    finally:
        qtbot.waitUntil(lambda: not window._series_loader.busy(), timeout=10000)
        window.method_panel.calls.stop_pending()
        qtbot.waitUntil(window.method_panel.can_close, timeout=10000)
        window.method_panel.calls.stop_pending()
        window.close()
        QCoreApplication.setOrganizationName(old[1])
        QCoreApplication.setApplicationName(old[2])
        QSettings.setDefaultFormat(old[0])


def _roots(window):
    return window.phase_model_workspace.series_roots


def _files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _phase_choice(root):
    save_series_method(
        root,
        MethodSelection(
            active="phase_model",
            model_version="model <v1>",
            mass_calibration="../fits/frozen.json",
        ),
    )


def test_prepared_model_publication_uses_guarded_writer_and_retains_old_fit(method_window, qtbot):
    window = method_window
    root = _roots(window)["series-a"]
    save_series_method(root, MethodSelection(active="phase_model", model_version="models/old",
                                            mass_calibration="calibration-original.json"))
    window._refresh_method_state(force=True)
    qtbot.waitUntil(lambda: not window.method_panel.calls.preview.busy and window.method_panel.can_close())
    before = _files(root)
    expected = window._method_manifest_bytes
    outcomes = []
    assert window._publish_prepared_model(root, "models/new", expected_bytes=expected,
        expected_selection=window._saved_method, finished=lambda *args: outcomes.append(args))
    qtbot.waitUntil(lambda: bool(outcomes), timeout=10000)
    assert outcomes == [(True, "")]
    after = _files(root)
    assert {key: data for key, data in after.items() if key != "series.toml"} == {
        key: data for key, data in before.items() if key != "series.toml"}
    saved = tomllib.loads(after["series.toml"].decode())
    assert saved["method"]["phase_model"] == dict(model_version="models/new", mass_calibration="",
        previous_mass_calibrations=["calibration-original.json"])
    assert "approval" not in saved["method"]
    assert window._saved_method.model_version == "models/new"
    assert window._saved_method.mass_calibration == ""
    rejected = []
    assert not window._publish_prepared_model(root, "models/newer", expected_bytes=expected,
        expected_selection=window._saved_method, finished=lambda *args: rejected.append(args))
    assert rejected and rejected[0][0] is False
    assert _files(root) == after


def test_method_choice_changes_only_manifest_and_survives_refresh_and_series_switch(
    method_window, qtbot,
):
    from tests.unit.test_series_calibration_approval import bound_products

    window = method_window
    root = _roots(window)["series-a"]
    bound = bound_products(root)
    save_series_method(root, bound.selection)
    from darsia.fluidflower.series_method import save_series_calibration_method
    save_series_calibration_method(root, "colour_path")
    window._refresh_method_state(force=True)
    before = _files(root)
    config = copy.deepcopy(window.config_dict)
    window._select_row(("method", "use_for_series"))
    panel = window.method_panel
    panel.phase_model_button.click()
    assert not panel.model_version.isVisible() and not panel.mass_calibration.isVisible()
    assert not panel.calls.timer.isActive()
    assert _files(root) == before
    panel.use_button.click()
    qtbot.waitUntil(lambda: panel.can_close() and not panel.calls.timer.isActive(), timeout=10000)
    after = _files(root)
    assert set(after) == set(before)
    assert after["calibration-original.json"] == before["calibration-original.json"]
    manifest = tomllib.loads(after["series.toml"].decode("utf-8"))
    assert manifest["future"] == {"opaque": [1, 2, 3]}
    assert manifest["method"]["phase_model"]["model_version"] == bound.selection.model_version
    assert "approval" not in manifest["method"]
    assert window.config_dict == config
    window._refresh_method_state(force=True)
    assert panel.selection().active == "phase_model"
    assert bound.selection.model_version in window.method_badge.text()
    assert window.method_badge.textFormat() == Qt.TextFormat.PlainText
    window._select_series("series-b")
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    assert window._saved_method.active == ""
    assert window.method_badge.text() == "Method: not chosen"
    window._select_series("series-a")
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    assert window._saved_method.active == "phase_model"
    assert _files(root) == after


def test_method_badge_is_visible_in_workflow_categories_and_analysis_repeats_it(
    method_window, qtbot,
):
    window = method_window
    _phase_choice(_roots(window)["series-a"])
    window._refresh_method_state()
    assert list(window.sidebar._sections) == [
        "start",
        "setup",
        "calibration",
        "analysis",
        "comparison",
        "export",
        "jobs",
        "helper",
        "utils",
        "legacy",
    ]
    for category in window.sidebar._sections:
        section = window.sidebar._sections[category]
        logical = window.sidebar._logical_selection(category, next(iter(section._rows)))
        window._select_row(logical)
        qtbot.waitUntil(lambda: not window._series_loading and not window._series_loader.busy(), timeout=10000)
        # The badges are details of the one header line on every page but Start (N141).
        native_header = category not in {"start", "legacy"}
        assert (not window.method_badge.isHidden()) is native_header, category
        assert window.status_header.line.isVisible() is native_header, category
        if native_header:
            assert " · Phase-map method · model " in window.status_header.line.text()
        assert "Method: phase-map method" in window.method_badge.text()
        assert window.method_panel.isVisible() == (logical == ("method", "use_for_series"))
        assert (not window.analysis_method_badge.isHidden()) is (category == "analysis")
        if category == "start":
            assert not window.series_header.isVisible()
        if category == "analysis":
            assert window.analysis_method_badge.text() == window.method_badge.text()


def test_start_lists_each_series_saved_method_without_writing_or_opening_it(
    method_window,
):
    window = method_window
    roots = _roots(window)
    _phase_choice(roots["series-a"])
    before = {key: _files(root) for key, root in roots.items()}
    home = window.home_page
    home.refresh()
    assert home.table.horizontalHeaderItem(4).text() == "Method"
    assert not home.table.isColumnHidden(4)
    displayed = {
        home.table.item(row, 1).text(): home.table.item(row, 4).text()
        for row in range(home.table.rowCount())
    }
    assert displayed == {"series-a": "phase-map method", "series-b": "–"}
    assert window.content_stack.currentWidget() is home
    assert not window.method_badge.isVisible()
    assert not window.series_header.isVisible()
    assert {key: _files(root) for key, root in roots.items()} == before
    save_series_method(
        roots["series-b"],
        MethodSelection(active="colour_path", colour_calibration="colour"),
    )
    home.refresh()
    assert home.table.item(1, 4).text() == "direct colour method"


def test_unchanged_manifest_refresh_preserves_unsaved_panel_draft(method_window):
    window = method_window
    root = _roots(window)["series-a"]
    original = (root / "series.toml").read_bytes()
    window._select_row(("calibration", "color"))
    window.method_panel.phase_model_button.click()
    window.method_panel.model_version.setText("unsaved draft model")
    window.method_panel.mass_calibration.setText("relative unsaved fit")
    drafted = window.method_panel.selection()
    window._select_row(("setup", "depth"))
    window._select_row(("calibration", "color"))
    assert window.method_panel.selection() == drafted
    assert (root / "series.toml").read_bytes() == original
    assert window._saved_method.active == ""


@pytest.mark.parametrize("category", ["comparison", "helper", "utils"])
def test_original_category_header_hides_method_panel(method_window, qtbot, category):
    window = method_window
    window._select_row(("calibration", "color"))
    assert not window.method_panel.isVisible()  # No permanent method panel in Calibration.
    section = window.sidebar._sections[category]
    section.set_expanded(False)
    qtbot.mouseClick(section.header_button, Qt.MouseButton.LeftButton)
    assert not window.method_panel.isVisible()
    assert window.analysis_method_badge.isHidden()
    assert not window.method_badge.isHidden()
    assert window.status_header.line.isVisible()


@pytest.mark.parametrize("method", ["colour_path", "phase_model"])
def test_mass_calibration_uses_one_readonly_display_and_cannot_launch(
    method_window, method
):
    window = method_window
    root = _roots(window)["series-a"]
    frozen = root / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "parameters": {"gain": 23.859, "beta": 0.6513},
                "metrics": {"mean_run_nmae": 0.02},
            }
        ),
        encoding="utf-8",
    )
    colour = root / "colour"
    (colour / "flash").mkdir(parents=True)
    (colour / "flash/flash.json").write_text('{"min_value_aq":0.15}', encoding="utf-8")
    for active in ("phase_model", "colour_path"):
        save_series_method(
            root,
            MethodSelection(
                active=active,
                model_version="v1",
                mass_calibration="frozen.json",
                colour_calibration="colour",
            ),
        )
    save_series_method(root, MethodSelection(active=method, model_version="v1",
        mass_calibration="frozen.json", colour_calibration="colour"))
    window._refresh_method_state(force=True)
    before = _files(root)
    config = copy.deepcopy(window.config_dict)
    key = ("calibration", "saved")
    window._select_row(key)
    panel = window.mass_calibration_display
    assert window.content_stack.currentWidget() is panel.saved_calibration_page
    assert window.sidebar.get_selected() == key
    assert not panel.manual_button.isEnabled()
    assert not panel.automatic_button.isEnabled()
    assert "saved calibration" in panel.mode_notice.text().lower()
    assert not window.toolbar_builder.play_action.isEnabled()
    window.toolbar_builder.play_action.setEnabled(True)
    assert not window.toolbar_builder.play_action.isEnabled()
    window.run_selected_workflow()
    for hidden_action in ("mass", "mass_optuna", "default_mass"):
        window.selected_checkbox_id = hidden_action
        window.calibration_tab.on_run_clicked()
    assert window.config_dict == config
    assert _files(root) == before
    rows = {
        panel.parameters.item(i, 0).text(): panel.parameters.item(i, 1).text()
        for i in range(panel.parameters.rowCount())
    }
    if method == "phase_model":
        assert rows["Gain"] == "23.859"
        assert rows["Beta"] == "0.6513"
        assert "mean_run_nmae = 0.02" in rows["Saved mass-balance error"]
        assert "Not reported" in rows["Dilution factor"]
    else:
        assert rows["Flash min_value_aq"] == "0.15"
        assert "Not reported" in rows["Saved mass-balance error"]


def test_one_fixed_calibration_list_preserves_original_compatibility_routes(method_window):
    window = method_window
    section = window.sidebar._sections["calibration"]
    assert [row.label for row in section._rows.values() if not row.isHidden()] == [
        "Image selection", "Calibration Method", "Phase model", "Mass calibration",
        "Saved calibrations", "Review", "Validation"]
    assert window.sidebar.display_selection("phase_model", "mass_from_phase") == ("calibration", "mass")
    assert window.sidebar.display_selection("phase_model", "overflow") == ("calibration", "mass")
    for key in ("mass_optuna", "default_mass", "reset", "delete"):
        assert section.get_row(key).isHidden()


def test_mass_display_follows_series_switch_without_reusing_previous_values(
    method_window, qtbot,
):
    window = method_window
    root = _roots(window)["series-a"]
    (root / "frozen.json").write_text(
        '{"parameters":{"gain":23.859,"beta":0.6513}}', encoding="utf-8"
    )
    save_series_method(
        root,
        MethodSelection(
            active="phase_model", model_version="v1", mass_calibration="frozen.json"
        ),
    )
    window._refresh_method_state(force=True)
    window._select_row(("phase_model", "mass_from_phase"))
    window._select_series("series-b")
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    panel = window.mass_calibration_display
    assert window.content_stack.currentWidget() is panel
    assert "Not reported" in panel.parameters.item(0, 1).text()


def test_loaded_incomplete_choice_warns_without_fallback(method_window):
    window = method_window
    root = _roots(window)["series-a"]
    path = root / "series.toml"
    path.write_bytes(path.read_bytes() + b'\n[method]\nactive = "phase_model"\n')
    before = path.read_bytes()
    window._refresh_method_state()
    assert window._saved_method.active == "phase_model"
    assert "missing model version and mass calibration" in window.method_badge.text()
    assert window.method_panel.phase_model_button.isChecked()
    assert not window.method_panel.use_button.isEnabled()
    assert path.read_bytes() == before


def test_production_rows_keep_fixed_positions_and_disable_other_model_actions(method_window, qtbot):
    window = method_window
    root = _roots(window)["series-a"]
    _phase_choice(root)
    window._refresh_method_state()
    section = window.sidebar._sections["analysis"]
    order = list(section._rows)
    original = window.sidebar._sections["comparison"].get_row("analysis:fingers")
    phase = window.sidebar._sections["export"].get_row("mass_analysis:frozen")
    assert next(iter(section._rows)) == "jobs:new_analysis"
    assert not original.isHidden() and not original.isEnabled()
    assert "Colour signal" in original.toolTip()
    assert not phase.isHidden() and phase.isEnabled()
    section.set_expanded(False)
    qtbot.mouseClick(section.header_button, Qt.MouseButton.LeftButton)
    assert window.sidebar.get_selected() == ("jobs", "new_analysis")
    save_series_method(root, MethodSelection(active="colour_path", colour_calibration="existing-colour-fit"))
    window._refresh_method_state()
    assert list(section._rows) == order
    assert original.isEnabled() and not original.isHidden()
    assert not phase.isEnabled() and not phase.isHidden()


@pytest.mark.parametrize("accept", [False, True])
def test_nonchosen_original_analysis_confirms_before_settings_sync_or_process(
    method_window, monkeypatch, accept
):
    window = method_window
    root = _roots(window)["series-a"]
    _phase_choice(root)
    window._refresh_method_state()
    before = _files(root)
    config_path = root / "unchanged-run.toml"
    window.config_path_label.setText(str(config_path))
    window.selected_action = "analysis"
    window.selected_checkbox_id = "mass"
    synced = []
    calls = []
    monkeypatch.setattr(
        window.settings_factory,
        "_sync_settings_inputs_to_config_dict",
        lambda: synced.append(True),
    )
    monkeypatch.setattr(window.settings_factory, "get_value", lambda *_: False)
    monkeypatch.setattr(
        window.process_runner,
        "start_workflow_process",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_a, **_k: (
            QMessageBox.StandardButton.Yes if accept else QMessageBox.StandardButton.No
        ),
    )
    window.analysis_tab.on_run_clicked()
    assert bool(synced) is accept
    assert bool(calls) is accept
    if accept:
        args, keywords = calls[0]
        assert args[0][-4:] == [
            "--config",
            str(config_path.resolve()),
            "--mass",
            "--progress",
        ]
        assert keywords["workflow"] == "analysis"
        assert keywords["actions"] == ["mass"]
    assert _files(root) == before
    assert window._saved_method.active == "phase_model"


def test_phase_export_no_prevents_original_export_and_yes_preserves_arguments(
    method_window, monkeypatch
):
    window = method_window
    calls = []
    saved = SimpleNamespace(export=lambda *a, **k: calls.append((a, k)))
    window._guard_phase_exports(SimpleNamespace(saved_results=saved))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.No
    )
    saved.export("csv")
    assert not calls
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
    )
    saved.export("csv")
    assert calls == [(("csv",), {})]
    assert window._saved_method.active == ""


def test_method_references_are_only_explicit_manifest_choices_not_legacy_selections(
    method_window,
):
    window = method_window
    root = _roots(window)["series-a"]
    original = (root / "series.toml").read_bytes()
    # Retained viewers can hold old selectors after updating their series label.
    window.mass_analysis_page = SimpleNamespace(
        series="series-a",
        project_root=root,
        phase_catalog=None,
        model=SimpleNamespace(currentData=lambda: "old-other-series-model"),
        saved_results=SimpleNamespace(
            calibration_file=Path("old-other-series-fit.json")
        ),
    )
    window.phase_model_workspace.studio_page = SimpleNamespace(
        studio=SimpleNamespace(
            _active_model_series="series-a",
            _selected_validation_model=lambda: {"model_id": "old-model"},
        )
    )
    try:
        window._refresh_method_state(force=True)
        assert window.method_panel.selection() == MethodSelection()
        assert (root / "series.toml").read_bytes() == original
        _phase_choice(root)
        window._refresh_method_state(force=True)
        assert window.method_panel.selection().model_version == "model <v1>"
        assert window.method_panel.selection().mass_calibration == "../fits/frozen.json"
    finally:
        window.mass_analysis_page = None
        window.phase_model_workspace.studio_page = None


def test_original_compatibility_entry_opens_visible_readonly_mass_row(method_window):
    window = method_window
    window.show_original_workflow()
    assert window.sidebar.get_selected() == ("calibration", "mass")
    assert window.content_stack.currentWidget() is window.mass_calibration_display
    assert not window.toolbar_builder.play_action.isEnabled()


def test_new_phase_series_records_method_before_models_or_calibration_exist():
    from darsia.fluidflower.series_creation import _series_toml

    draft = SimpleNamespace(
        runs=[SimpleNamespace(run_id="ac52")],
        series_id="demo",
        name="Demo",
        rig_class="medium",
        created_by="test",
    )
    document = tomllib.loads(_series_toml(draft, "2026-10-15T00:00:00Z"))
    assert document["method"] == {"active": "phase_model"}
    assert document["paths"]["geometry_facies"] == "../Results/setup/Geometry"
