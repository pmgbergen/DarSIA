"""Scope UI uses local manifests/configuration only and never runs anything."""

import tomllib
from pathlib import Path

import pytest
from PySide6.QtCore import QSignalBlocker, Qt

from .test_native_series_method import method_window


@pytest.fixture
def scope_window(method_window, monkeypatch):
    window = method_window
    for key, root in window.phase_model_workspace.series_roots.items():
        for folder in ("config/runs", "config/merged"):
            (root / folder).mkdir(parents=True)
        (root / "series.toml").write_text(
            f'schema_version = 1\nseries_id = "{key}"\nname = "{key}"\n'
            '[runs]\nids = ["ac28", "ac47"]\n',
            encoding="utf-8",
        )
        common = '[calibration.color]\nembedding = "shared-colour"\n[calibration.mass]\nembedding = "shared-mass"\n'
        (root / "config/common.toml").write_text(common, encoding="utf-8")
        for run in ("ac28", "ac47"):
            override = (
                '[calibration.mass]\nembedding = "run-only"\n'
                if run == "ac47"
                else 'run_marker = "ac28"\n'
            )
            (root / f"config/runs/{run}.toml").write_text(override, encoding="utf-8")
            merged = (
                common.replace('embedding = "shared-mass"', 'embedding = "run-only"')
                if run == "ac47"
                else common
            )
            (root / f"config/merged/{run}.toml").write_text(merged, encoding="utf-8")
    # These are scope rendering tests, not ConfigController tests. Opening a
    # scope must never cause an extra controller load or synchronize a form.
    monkeypatch.setattr(window, "_ensure_run_loaded", lambda: None)
    monkeypatch.setattr(
        window,
        "_load_run_config",
        lambda *_a: pytest.fail("Scope display must not load a GUI configuration"),
    )
    window.config_file = None
    window._config_series = None
    window._populate_runs()
    with QSignalBlocker(window.run_combo):
        window.run_combo.setCurrentIndex(window.run_combo.findText("ac28"))
    window._refresh_scope_display()
    return window


def _select_run(window, run):
    window.config_file = None
    window._config_series = None
    window.run_combo.setCurrentIndex(window.run_combo.findText(run))


def _snapshot(window):
    return {
        str(path): path.read_bytes()
        for root in window.phase_model_workspace.series_roots.values()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_series_scope_hides_run_selector_and_panel_has_independent_series_scope(
    scope_window,
):
    window = scope_window
    before = _snapshot(window)
    window._select_row(("phase_model", "training"))
    # The scope line sits in the header details; the header line names the
    # whole series and the run selector is hidden on a series page (N141).
    assert not window.scope_label.isHidden()
    assert "whole series series-a (2 runs)" in window.scope_label.text()
    assert not window.run_combo.isEnabled()
    assert not window.run_combo.isVisible()
    assert window.status_header.line.text().endswith(" · Whole series (2 runs)")
    window._select_row(("analysis", "mass"))
    assert window.run_combo.isEnabled()
    assert window.run_combo.isVisible()
    assert window.scope_label.text() == "Applies to: run AC28 only"
    assert window.status_header.line.text().endswith(" · Run AC28 only")
    window._select_row(("calibration", "delete"))
    assert window.scope_label.text() == "Applies to: run AC28 only"
    assert "whole series series-a (2 runs)" in window.method_scope_label.text()
    assert _snapshot(window) == before


def test_common_scope_is_data_bound_and_changes_with_selected_run_override(
    scope_window,
):
    window = scope_window
    window._select_row(("calibration", "mass"))
    # Common settings, but the action runs on the selected run (N141).
    assert window.scope_label.text() == "Settings: whole series (common.toml) · Runs on: AC28"
    assert window.run_combo.isEnabled()
    # Changing the chosen run elsewhere is reflected when returning here.
    window._select_row(("analysis", "mass"))
    _select_run(window, "ac47")
    window._select_row(("calibration", "mass"))
    assert window.scope_label.text() == "Applies to: run AC47 only"
    assert window.run_combo.isEnabled()
    # Colour has no run override, independently of mass.
    window._select_row(("calibration", "color"))
    assert window.scope_label.text() == "Settings: whole series (common.toml) · Runs on: AC47"
    assert window.run_combo.isEnabled()


def test_one_context_updates_all_help_including_existing_popup_without_accumulation(
    scope_window, monkeypatch
):
    from darsia.fluidflower import row_scope
    from darsia.gui.ui.help import HelpPopup

    window = scope_window
    section = window.sidebar._sections["calibration"]
    row = section.get_row("mass")
    base = window._scope_base_help[id(row)]
    row._help_popup = HelpPopup(row.help_text)
    window.run_combo.setCurrentIndex(window.run_combo.findText("ac47"))
    calls = []
    load = row_scope.load_scope_context

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return load(*args, **kwargs)

    monkeypatch.setattr(row_scope, "load_scope_context", record)
    window._refresh_scope_display()
    assert len(calls) == 1
    for category in ("calibration", "analysis"):
        for candidate in window.sidebar._sections[category]._rows.values():
            lines = candidate.help_text.count("Applies to:") + candidate.help_text.count("Settings: whole series")
            assert lines == 1, candidate.help_text
    assert row.help_text.startswith(base + "\n")
    assert "run AC47 only" in row.help_text
    assert row._help_popup._help_text == row.help_text
    assert row._help_popup._help_label.text() == row.help_text
    window._refresh_scope_display()
    assert row.help_text.count("Applies to:") == 1
    row._help_popup.close()


def test_actual_loaded_registered_config_wins_and_external_config_is_not_bound(
    scope_window,
):
    window = scope_window
    root = window.phase_model_workspace.series_roots["series-a"]
    _select_run(window, "ac47")
    loaded = root / "config/merged/ac28.toml"
    window.config_file = str(loaded)
    window._config_series = "series-a"
    window.config_dict = tomllib.loads(loaded.read_text(encoding="utf-8"))
    window._select_row(("analysis", "mass"))
    assert window.scope_context.selected_run == "ac28"
    assert window.scope_label.text() == "Applies to: run AC28 only"
    # Original actions use the actual open config. A foreign file must never
    # be labelled as the selected run just because the combo still names it.
    window.config_file = str(root.parent / "not-this-series.toml")
    window._config_series = "series-a"
    window._refresh_scope_display()
    assert window.scope_context.selected_run == "ac47"
    assert window.scope_context.loaded_path == root.parent / "not-this-series.toml"
    assert (
        window.scope_label.text()
        == "Applies to: selected configuration (run not confirmed)"
    )
    window._select_row(("calibration", "mass"))
    assert "via common.toml" not in window.scope_label.text()


@pytest.mark.parametrize(
    "category", ["setup", "comparison", "helper", "utils", "start"]
)
def test_other_category_headers_restore_run_selector_and_hide_scope_line(
    scope_window, qtbot, category
):
    window = scope_window
    window._select_row(("phase_model", "training"))
    assert not window.run_combo.isEnabled()
    section = window.sidebar._sections[category]
    section.set_expanded(False)
    qtbot.mouseClick(section.header_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not window._series_loading)
    assert window.run_combo.isEnabled()
    assert window.run_combo.isVisible() is (category != "start")
    assert window.scope_label.isHidden()


def test_series_switch_updates_top_panel_and_all_help_scope_without_writes(
    scope_window, qtbot,
):
    window = scope_window
    window._select_row(("phase_model", "training"))
    before = _snapshot(window)
    window._select_series("series-b")
    qtbot.waitUntil(lambda: not window._series_loading)
    assert "whole series series-b (2 runs)" in window.scope_label.text()
    assert "whole series series-b (2 runs)" in window.method_scope_label.text()
    assert (
        "whole series series-a"
        not in window.sidebar._sections["calibration"]
        .get_row("phase_model:training")
        .help_text
    )
    assert _snapshot(window) == before
