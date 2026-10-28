"""Real offscreen native widgets over local-only metadata fixtures.

The native source and metadata adapters come from the active checkout.
The numerical runner, Training load, wizard and foreign/NAS paths are forbidden.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtTest import QTest


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def local_window(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower import workflow_status_ui as ui
    from darsia.fluidflower import native_darsia_app as native

    old_settings = (
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
    QCoreApplication.setOrganizationName("DarSIA-A14-A15-local-integration")
    QCoreApplication.setApplicationName(tmp_path.name)
    root = tmp_path / "series"
    put(
        root / "series.toml",
        'schema_version=1\nseries_id="fixture"\nname="Fixture"\n[runs]\nids=["ac28","ac52"]\n',
    )
    for run in ("ac28", "ac52"):
        result = (root / f"results/{run}").as_posix()
        put(
            root / f"config/merged/{run}.toml",
            f'run_marker="{run}"\n[data]\nresults="{result}"\n[rig]\nwidth=0.9\nheight=0.5\n',
        )
    other = tmp_path / "other-series"
    put(
        other / "series.toml",
        'schema_version=1\nseries_id="other"\nname="Other"\n[runs]\nids=["ac28"]\n',
    )
    put(
        other / "config/merged/ac28.toml",
        'run_marker="other-ac28"\n[rig]\nwidth=0.8\nheight=0.5\n',
    )
    monkeypatch.setattr(
        native,
        "configured_series",
        lambda: (("fixture", "Fixture", root), ("other", "Other", other)),
    )
    warnings = []
    monkeypatch.setattr(
        native.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    monkeypatch.setattr(
        native, "ProjectRegistry", lambda: SimpleNamespace(entries=lambda: [])
    )
    studio_loads = []
    monkeypatch.setattr(
        native.PhaseModelWorkspace,
        "ensure_studio",
        lambda self: studio_loads.append(True) or self.studio_placeholder,
    )
    monkeypatch.setattr(
        native.SeriesGeometryWorkspace, "ensure_page", lambda self: self
    )
    monkeypatch.setattr(
        native.NativeFluidFlowerMainWindow,
        "ensure_mass_analysis",
        lambda self: self.mass_analysis_placeholder,
    )
    window = native.NativeFluidFlowerMainWindow(initial_series="fixture")
    qtbot.addWidget(window)
    executions = []

    def forbidden(*args, **kwargs):
        executions.append((args, kwargs))
        pytest.fail("UI navigation must not start a workflow or wizard")

    monkeypatch.setattr(window.process_runner, "start_workflow_process", forbidden)
    monkeypatch.setattr(window, "run_selected_workflow", forbidden)
    monkeypatch.setattr(window, "_open_preprocessing_for_selected_run", forbidden)
    opened = []
    monkeypatch.setattr(
        ui, "open_in_file_explorer", lambda path: opened.append(Path(path))
    )
    window.resize(1600, 1100)
    window.show()
    qtbot.waitUntil(window.isVisible)
    value = SimpleNamespace(
        window=window,
        root=root,
        opened=opened,
        executions=executions,
        studio_loads=studio_loads,
        warnings=warnings,
        other=other,
    )
    try:
        yield value
    finally:
        window.close()
        QCoreApplication.setOrganizationName(old_settings[1])
        QCoreApplication.setApplicationName(old_settings[2])
        QSettings.setDefaultFormat(old_settings[0])
        assert not executions


def test_start_click_loads_saved_reference_through_original_loader(local_window, qtbot):
    fixture = local_window
    window = fixture.window
    put(
        fixture.root / "geometry_facies/series.json",
        json.dumps({"tools": {"native_series": {"reference_run": "ac52"}}}),
    )
    assert not window.config_file
    table = window.home_page.table
    rect = table.visualItemRect(table.item(0, 0))
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    qtbot.waitUntil(lambda: window.config_file is not None)
    assert Path(window.config_file) == fixture.root / "config/merged/ac52.toml"
    assert window.config_dict["run_marker"] == "ac52"
    assert window.run_combo.currentText() == "ac52"
    assert window.sidebar.get_selected() == ("preprocessing", "protocols")
    assert (
        window.reference_configuration_choice.source
        == "saved native geometry reference"
    )
    assert not fixture.studio_loads and not fixture.executions


def test_invalid_saved_reference_never_loads_another_run(local_window, qtbot):
    fixture = local_window
    window = fixture.window
    put(
        fixture.root / "geometry_facies/series.json",
        json.dumps({"tools": {"native_series": {"reference_run": "ac99"}}}),
    )
    table = window.home_page.table
    rect = table.visualItemRect(table.item(0, 0))
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert not window.config_file
    assert window.run_combo.currentIndex() == -1
    assert window.reference_configuration_choice.run_id == "ac99"
    assert window.reference_configuration_choice.issues
    assert not fixture.studio_loads and not fixture.executions


def test_clean_reference_reopen_is_idempotent_without_warning(
    local_window, qtbot, monkeypatch
):
    f = local_window
    w = f.window
    put(
        f.root / "geometry_facies/series.json",
        json.dumps({"tools": {"native_series": {"reference_run": "ac52"}}}),
    )
    w._open_series_from_start(str(f.root))
    assert Path(w.config_file) == f.root / "config/merged/ac52.toml"
    loads = []
    original = w.config_controller.load_config
    monkeypatch.setattr(
        w.config_controller, "load_config", lambda: loads.append(True) or original()
    )
    w._open_series_from_start(str(f.root))
    assert not loads and not f.warnings


def _edit_width(f):
    from darsia.gui.ui.settings import unwrap_composite_widget

    w = f.window
    assert w._load_run_config(str(f.root / "config/merged/ac28.toml"))
    field = unwrap_composite_widget(w.settings_inputs["rig.width"])
    field.setText("1.234")
    return field


def test_unsaved_other_run_reopen_preserves_widget_context_and_files(
    local_window, qtbot, monkeypatch
):
    f = local_window
    w = f.window
    put(
        f.root / "geometry_facies/series.json",
        json.dumps({"tools": {"native_series": {"reference_run": "ac52"}}}),
    )
    field = _edit_width(f)
    old_dict = w.config_dict
    original_values = copy.deepcopy(old_dict)
    old_file = w.config_file
    old_run = w.run_combo.currentText()
    old_selection = w.sidebar.get_selected()
    before = {p: p.read_bytes() for p in f.root.rglob("*") if p.is_file()}
    loads = []
    monkeypatch.setattr(w.config_controller, "load_config", lambda: loads.append(True))
    w._open_series_from_start(str(f.root))
    assert not loads and f.warnings and "Save" in f.warnings[-1][0]
    assert w.config_dict is old_dict and w.config_dict == original_values
    assert w.config_file == old_file and w.run_combo.currentText() == old_run
    assert w.sidebar.get_selected() == old_selection and field.text() == "1.234"
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_unsaved_cross_series_click_preserves_selected_context(
    local_window, qtbot, monkeypatch
):
    f = local_window
    w = f.window
    field = _edit_width(f)
    loads = []
    monkeypatch.setattr(w.config_controller, "load_config", lambda: loads.append(True))
    table = w.home_page.table
    row = next(i for i in range(table.rowCount()) if table.item(i, 1).text() == "other")
    rect = table.visualItemRect(table.item(row, 0))
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert (
        w.phase_model_workspace.series_id == "fixture"
        and w.home_page.selected_series() == "fixture"
    )
    assert not loads and field.text() == "1.234" and len(f.warnings) == 1


def test_invalid_reference_refuses_foreign_config_protocols_and_play(
    local_window, qtbot
):
    f = local_window
    w = f.window
    foreign = put(
        f.root / "foreign.toml", 'run_marker="foreign"\n[rig]\nwidth=0.9\nheight=0.5\n'
    )
    assert w._load_run_config(str(foreign))
    put(
        f.root / "geometry_facies/series.json",
        json.dumps({"tools": {"native_series": {"reference_run": "ac99"}}}),
    )
    w._open_series_from_start(str(f.root))
    assert w.reference_configuration_choice.issues and not f.warnings
    assert Path(w.config_file) == foreign
    assert w.sidebar.get_selected() == ("start", "series")
    assert not w.toolbar_builder.play_action.isEnabled()


def test_successful_actual_save_resets_guard_then_reference_may_open(
    local_window, qtbot
):
    import tomllib

    f = local_window
    w = f.window
    field = _edit_width(f)
    assert not w.reference_open_guard.allowed()
    old = Path(w.config_file)
    w.settings_factory.save_settings()
    assert tomllib.loads(old.read_bytes().decode("utf-8"))["rig"]["width"] == 1.234
    assert w.reference_open_guard.allowed()
    f.warnings.clear()
    put(
        f.root / "geometry_facies/series.json",
        json.dumps({"tools": {"native_series": {"reference_run": "ac52"}}}),
    )
    w._open_series_from_start(str(f.root))
    assert Path(w.config_file) == f.root / "config/merged/ac52.toml"
    assert not f.warnings


def test_cancelled_save_as_does_not_reset_dirty_guard(local_window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    f = local_window
    w = f.window
    field = _edit_width(f)
    old = Path(w.config_file)
    raw = old.read_bytes()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    w.config_controller.save_config_as()
    assert not w.reference_open_guard.allowed()
    assert field.text() == "1.234" and old.read_bytes() == raw


def test_failed_save_does_not_reset_dirty_guard(local_window, qtbot, monkeypatch):
    import builtins

    f = local_window
    w = f.window
    field = _edit_width(f)
    old = Path(w.config_file)
    raw = old.read_bytes()
    original = builtins.open

    def fail_write(path, mode="r", *args, **kwargs):
        if Path(path) == old and "w" in mode:
            raise PermissionError("Local fixture save refused")
        return original(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", fail_write)
        with pytest.raises(PermissionError, match="Local fixture save refused"):
            w.settings_factory.save_settings()
    assert not w.reference_open_guard.allowed()
    assert field.text() == "1.234" and old.read_bytes() == raw


def test_external_disk_change_is_refused_without_writes(local_window, qtbot):
    f = local_window
    w = f.window
    assert w._load_run_config(str(f.root / "config/merged/ac28.toml"))
    assert w.reference_open_guard.allowed()
    old = Path(w.config_file)
    raw = old.read_bytes()
    old.write_bytes(raw + b"\n# external change\n[external]\nchanged=true\n")
    changed = old.read_bytes()
    assert not w.reference_open_guard.allowed()
    assert old.read_bytes() == changed and "external" not in w.config_dict


def test_new_optional_widget_field_stays_dirty_across_rerender(local_window, qtbot):
    from darsia.gui.ui.settings import unwrap_composite_widget
    import tomllib

    f = local_window
    w = f.window
    assert w._load_run_config(str(f.root / "config/merged/ac28.toml"))
    assert (
        "dim"
        not in tomllib.loads(Path(w.config_file).read_bytes().decode("utf-8"))["rig"]
    )
    combo = unwrap_composite_widget(w.settings_inputs["rig.dim"])
    combo.setCurrentText("3")
    w.settings_factory.display_full_settings()
    assert str(w.config_dict["rig"]["dim"]) == "3"
    assert not w.reference_open_guard.allowed()
    assert str(w.config_dict["rig"]["dim"]) == "3"


def test_same_reference_start_reopen_after_external_disk_change_refuses_without_reload(
    local_window, qtbot, monkeypatch
):
    f = local_window
    w = f.window
    put(
        f.root / "geometry_facies/series.json",
        json.dumps({"tools": {"native_series": {"reference_run": "ac52"}}}),
    )
    w._open_series_from_start(str(f.root))
    path = Path(w.config_file)
    old_dict = w.config_dict
    path.write_bytes(path.read_bytes() + b"\n[external]\nchanged=true\n")
    changed = path.read_bytes()
    loads = []
    monkeypatch.setattr(w.config_controller, "load_config", lambda: loads.append(True))
    w._open_series_from_start(str(f.root))
    assert (
        f.warnings
        and not loads
        and w.config_dict is old_dict
        and path.read_bytes() == changed
    )


def test_crop_apply_reuses_real_native_guard_with_one_unsaved_warning(
    local_window, monkeypatch
):
    from darsia.fluidflower import crop_setup_ui

    fixture = local_window
    window = fixture.window
    field = _edit_width(fixture)
    old_config = window.config_dict
    original_values = copy.deepcopy(old_config)
    old_file = window.config_file
    old_run = window.run_combo.currentText()
    old_selection = window.sidebar.get_selected()
    before = {
        path: path.read_bytes() for path in fixture.root.rglob("*") if path.is_file()
    }
    fixture.warnings.clear()
    task = window.crop_workflow
    assert window.reference_open_guard is not None
    # The valid geometry/export context is outside this Save-first regression;
    # the native window, real form edits, shared guard and Crop.start stay real.
    monkeypatch.setattr(task, "_selected_context", lambda _export: object())
    monkeypatch.setattr(
        crop_setup_ui,
        "prepare_crop_plan",
        lambda *_a, **_k: pytest.fail("unsaved edit must stop before registration"),
    )
    monkeypatch.setattr(
        crop_setup_ui,
        "apply_crop_plan",
        lambda *_a, **_k: pytest.fail("unsaved edit must stop before writing"),
    )
    assert not task.start(object())
    assert not task.running and window.isEnabled()
    assert len(fixture.warnings) == 1
    assert fixture.warnings[0][0] == "Save current configuration first"
    assert window.config_dict is old_config and window.config_dict == original_values
    assert field.text() == "1.234"
    assert window.config_file == old_file and window.run_combo.currentText() == old_run
    assert window.sidebar.get_selected() == old_selection
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not fixture.executions
