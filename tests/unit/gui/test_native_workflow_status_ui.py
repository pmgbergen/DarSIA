"""Real offscreen native widgets over local-only metadata fixtures.

The native source and metadata adapters come from the active checkout.
The numerical runner, Training load, wizard and foreign/NAS paths are forbidden.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton


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
    qtbot.addWidget(window, before_close_func=lambda w:
                    qtbot.waitUntil(lambda: not w._series_loader.busy()))
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
    qtbot.waitUntil(lambda: not window._series_loading)
    qtbot.waitUntil(lambda: not window._series_loader.busy())
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
        from shiboken6 import isValid
        if isValid(window):
            qtbot.waitUntil(lambda: not window._series_loader.busy())
            window.close()
        QCoreApplication.setOrganizationName(old_settings[1])
        QCoreApplication.setApplicationName(old_settings[2])
        QSettings.setDefaultFormat(old_settings[0])
        assert not executions


def test_actual_rows_have_status_widgets_and_missing_note_keeps_settings(
    local_window, qtbot
):
    fixture = local_window
    window = fixture.window
    window._select_row(("setup", "depth"))
    qtbot.waitUntil(lambda: window.workflow_status.panel.isVisible())
    adapter = window.workflow_status
    assert window.content_stack.currentWidget() is window.depth_workflow.panel
    assert "Not done" in adapter.label.text()
    assert "Produced by:" in adapter.missing_label.text()
    assert adapter.missing_label.isVisible()
    for logical in (("setup", "depth"), ("calibration", "color"), ("analysis", "mass")):
        row, label = adapter.rows[logical]
        assert label.parentWidget().layout().indexOf(row._text_label) >= 0
        assert label.text() and not row.findChildren(QPushButton)
    assert not fixture.studio_loads


@pytest.mark.parametrize('target', ['name', 'status', 'space', 'enter'])
def test_whole_sidebar_row_navigates_without_running(local_window, qtbot, target):
    fixture = local_window
    window = fixture.window
    window._select_row(("setup", "depth"))
    row, status = window.workflow_status.rows[("setup", "depth")]
    selections = []
    window.sidebar.selection_changed.connect(lambda *value: selections.append(value))
    if target in {'name', 'status'}:
        child = row._text_label if target == 'name' else status
        QTest.mouseClick(row, Qt.MouseButton.LeftButton,
                         pos=child.mapTo(row, child.rect().center()))
    else:
        row.setFocus()
        QTest.keyClick(row, Qt.Key.Key_Space if target == 'space' else Qt.Key.Key_Return)
    qtbot.waitUntil(lambda: not window._series_loader.busy())
    assert selections == [('setup', 'depth')]
    assert window.sidebar.get_selected() == ("setup", "depth")
    assert window.content_stack.currentWidget() is window.depth_workflow.panel
    assert not window.depth_workflow.sequence.running
    assert not fixture.executions and not fixture.studio_loads


def test_done_row_selects_settings_and_content_open_keeps_saved_file_action(local_window, qtbot):
    fixture = local_window
    window = fixture.window
    output = put(
        fixture.root / "results/ac28/sparse_data/integrated_mass.csv",
        "time,mass\n0,0\n",
    )
    assert window._load_run_config(str(fixture.root / "config/merged/ac28.toml"))
    window._select_row(("analysis", "mass"))
    adapter = window.workflow_status
    qtbot.waitUntil(lambda: not adapter.checking, timeout=10000)
    state = adapter.states[("analysis", "mass")]
    assert (
        state.state == "done"
        and state.date
        and state.date_source == "file modification time"
    )
    row, _ = adapter.rows[("analysis", "mass")]
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not window._series_loader.busy())
    assert window.sidebar.get_selected() == ('analysis', 'mass')
    assert not row.findChildren(QPushButton) and not fixture.opened
    QTest.mouseClick(adapter.open_button, Qt.MouseButton.LeftButton)
    assert fixture.opened == [output]
    assert not fixture.executions


def test_disabled_other_model_row_shows_only_not_applicable(local_window):
    from darsia.fluidflower.workflow_row_status import RowStatus
    from darsia.fluidflower.series_method import MethodSelection
    window = local_window.window
    adapter = window.workflow_status
    window._saved_method = MethodSelection(active="phase_model")
    row, label = adapter.rows[("analysis", "cropping")]
    row.setEnabled(False)
    original = RowStatus("not_applicable", "Not applicable for method phase_model.",
        selected_run_text="Analysis exclusion scope unavailable: Every excluded run needs a registered ID and nonempty reason")
    states = dict(adapter.states)
    states[("analysis", "cropping")] = original
    adapter.refresh(context=adapter.context, states=states)
    assert label.text() == "— Not applicable"
    assert "exclusion" not in label.toolTip().lower()
    assert adapter.states[("analysis", "cropping")].selected_run_text == ""
    assert "exclusion" in original.selected_run_text.lower()  # Display-only suppression.
    assert not local_window.executions


@pytest.mark.parametrize('logical', [('phase_model', 'training'), ('phase_model', 'mass_from_phase')])
def test_calibration_row_click_preserves_original_route_without_extra_buttons(local_window, qtbot, logical):
    window = local_window.window
    category, key = window.sidebar.display_selection(*logical)
    section = window.sidebar._sections[category]
    section.set_expanded(True)
    row = section.get_row(key)
    assert not row.findChildren(QPushButton)
    QTest.mouseClick(row, Qt.MouseButton.LeftButton,
                     pos=row._text_label.mapTo(row, row._text_label.rect().center()))
    qtbot.waitUntil(lambda: not window._series_loader.busy())
    assert window.sidebar.get_selected() == (("calibration", "mass")
        if logical == ("phase_model", "mass_from_phase") else logical)
    assert not local_window.opened and not local_window.executions


def test_phase_model_status_has_no_duplicate_open_but_other_pages_keep_it(
    local_window, qtbot, monkeypatch
):
    from darsia.fluidflower.workflow_row_status import Artifact, RowStatus

    fixture = local_window
    window = fixture.window
    qtbot.waitUntil(lambda: not window._series_loading)
    adapter = window.workflow_status
    output = put(fixture.root / "saved-model.json", "{}")
    saved = RowStatus("done", "Saved model", artifacts=(
        Artifact(path=str(output), kind="model", size_bytes=2),))
    states = dict(adapter.states)
    states[("phase_model", "training")] = saved
    states[("analysis", "mass")] = saved
    selected = ("phase_model", "training")
    monkeypatch.setattr(window.sidebar, "get_selected", lambda: selected)
    for selected in (("phase_model", "training"), ("analysis", "mass"), ("phase_model", "training")):
        adapter.refresh(context=adapter.context, states=states)
        assert adapter.open_button.isHidden() is (selected == ("phase_model", "training"))
        assert str(output) in adapter.label.toolTip()
    # The exact saved-file callback remains available to existing callers.
    adapter.open_button.click()
    assert fixture.opened == [output]
    assert not fixture.executions and not fixture.studio_loads


def _edit_width(f):
    from darsia.gui.ui.settings import unwrap_composite_widget

    w = f.window
    assert w._load_run_config(str(f.root / "config/merged/ac28.toml"))
    field = unwrap_composite_widget(w.settings_inputs["rig.width"])
    field.setText("1.234")
    return field


def test_comparison_utils_keep_original_shallow_dots_without_streaming_scan(
    local_window, qtbot, monkeypatch
):
    from darsia.presets.workflows import results_folder

    f = local_window
    w = f.window
    assert w._load_run_config(str(f.root / "config/merged/ac28.toml"))
    output = f.root / "comparison-output"
    put(output / "existing.log", "saved original output")
    monkeypatch.setattr(
        results_folder, "suggested_workflow_results_folder", lambda *a: output
    )
    monkeypatch.setattr(
        w.streaming_panel,
        "refresh_results",
        lambda: pytest.fail("No passive recursive streaming refresh"),
    )
    helper = w.sidebar._sections["helper"].get_row("color")
    helper.set_completion_state("error")
    w.refresh_sidebar_progress()
    for category in ("comparison", "utils"):
        assert all(
            row._completion_state == "done"
            for row in w.sidebar._sections[category]._rows.values()
            if row.action == category
        )
    assert helper._completion_state == "error"


def _prepared_review(f, folder):
    from dataclasses import replace

    model = f.root / "models/model1"
    put(model / "summary.json", json.dumps({"model_id": "model1"}))
    f.window._saved_method = replace(f.window._saved_method, model_version=str(model))
    binding = {
        "format": "fluidflower-phase-review-collection-v1",
        "source_model_id": "model1",
        "combined_root": str(folder),
    }
    put(
        folder / "postflight_manifest.json",
        json.dumps({**binding, "jobs": [{"dataset": "fixture", "run": "ac52"}]}),
    )
    return binding


def _review_sender(f, qtbot, started):
    import ast
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QWidget, QLabel, QLineEdit
    from collections.abc import Mapping
    import inspect
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage

    tree = ast.parse(inspect.getsource(SharedPhaseModelTrainingPage))
    original = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SharedPhaseModelTrainingPage"
    )
    method = next(
        n
        for n in original.body
        if isinstance(n, ast.FunctionDef) and n.name == "_start_review_process"
    )
    declaration = ast.parse("review_opened = Signal(object)").body[0]
    cls = ast.ClassDef(
        name="ActualReviewLaunchFixture",
        bases=[ast.Name(id="QWidget", ctx=ast.Load())],
        keywords=[],
        body=[declaration, method],
        decorator_list=[],
    )
    ns = {
        "QWidget": QWidget,
        "Signal": Signal,
        "Path": Path,
        "Mapping": Mapping,
        "QProcess": SimpleNamespace(
            startDetached=lambda *a: (started, 31415 if started else 0)
        ),
        "QMessageBox": SimpleNamespace(critical=lambda *a: None),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            "actual_UI_review_launch",
            "exec",
        ),
        ns,
    )
    sender = ns["ActualReviewLaunchFixture"]()
    qtbot.addWidget(sender)
    sender.review_save_root = QLineEdit(str(f.root / "local-review-save"))
    sender.review_status = QLabel()
    sender._review_command = lambda *a, **k: ["unused-no-child"]
    sender._active_model_series = "fixture"
    return sender


def test_actual_review_success_signal_binds_exact_manifest_without_discovery(
    local_window, qtbot
):
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QWidget

    f = local_window
    w = f.window
    binding = _prepared_review(f, f.root / "prepared-review")
    sender = _review_sender(f, qtbot, True)
    sender._selected_validation_model = lambda: {"validation_binding": binding}

    class Page(QWidget):
        studio_loaded = Signal()

    page = Page()
    qtbot.addWidget(page)
    page.studio = None
    w.workflow_status.bind_studio_page(page)
    page.studio = sender
    page.studio_loaded.emit()
    sender._start_review_process("open_correction", binding)
    key = w.workflow_status._series_key()
    assert w.workflow_status.selected_references[key]["review"] == str(
        f.root / "prepared-review/postflight_manifest.json"
    )
    qtbot.waitUntil(lambda: not w.workflow_status.checking, timeout=10000)
    assert w.workflow_status.states[("mass_analysis", "review")].state == "done"


def test_actual_failed_review_launch_does_not_replace_previous_reference(
    local_window, qtbot
):
    f = local_window
    w = f.window
    old = _prepared_review(f, f.root / "old-review")
    w.workflow_status.set_selected_reference(
        "review", str(f.root / "old-review/postflight_manifest.json")
    )
    new = _prepared_review(f, f.root / "new-review")
    sender = _review_sender(f, qtbot, False)
    sender._selected_validation_model = lambda: {"validation_binding": new}
    w.workflow_status.bind_review_studio(sender)
    sender._start_review_process("open_correction", new)
    assert w.workflow_status.selected_references[w.workflow_status._series_key()][
        "review"
    ] == str(f.root / "old-review/postflight_manifest.json")


def _saved_viewer(f, path, receipt, fail=False):
    class Viewer:
        _workflow_status_bound = False
        result_path = None
        entries = [{"path": str(path), "receipt_path": str(receipt)}] if receipt else []
        result = {
            "selected_inputs": {"candidate": {"dataset": "fixture", "run": "ac52"}}
        }

        def load_result(self, selected):
            if fail:
                raise ValueError("Synthetic original result-load refusal")
            self.result_path = Path(selected)

    saved = Viewer()
    f.window.workflow_status.bind_mass_workspace(SimpleNamespace(saved_results=saved))
    return saved


def _receipt(f, name):
    path = put(f.root / (name + ".json"), "saved result stays unread for status")
    receipt = put(
        f.root / (name + "-receipt.json"),
        json.dumps(
            {
                "format": "studio-mass-variant-delivery-v1",
                "candidate": {"dataset": "fixture", "run": "ac52"},
                "result_sha256": "a" * 64,
                "result": str(path),
            }
        ),
    )
    return path, receipt


def test_successful_selected_result_load_binds_actual_registered_receipt(
    local_window, qtbot
):
    f = local_window
    path, receipt = _receipt(f, "result")
    saved = _saved_viewer(f, path, receipt)
    saved.load_result(path)
    assert f.window.workflow_status.selected_references[
        f.window.workflow_status._series_key()
    ]["result_receipt"] == str(receipt)
    qtbot.waitUntil(lambda: not f.window.workflow_status.checking, timeout=10000)
    assert f.window.workflow_status.states[("mass_analysis", "frozen")].state == "done"


def test_failed_original_result_load_preserves_previous_receipt(local_window, qtbot):
    f = local_window
    path, receipt = _receipt(f, "old")
    f.window.workflow_status.set_selected_reference("result_receipt", str(receipt))
    other, new = _receipt(f, "new")
    saved = _saved_viewer(f, other, new, True)
    with pytest.raises(ValueError, match="original result-load refusal"):
        saved.load_result(other)
    assert f.window.workflow_status.selected_references[
        f.window.workflow_status._series_key()
    ]["result_receipt"] == str(receipt)


def test_unregistered_selected_result_clears_earlier_receipt(local_window, qtbot):
    f = local_window
    path, receipt = _receipt(f, "old")
    f.window.workflow_status.set_selected_reference("result_receipt", str(receipt))
    new = put(f.root / "unregistered.json", "not read for status")
    saved = _saved_viewer(f, new, None)
    saved.load_result(new)
    assert not f.window.workflow_status.selected_references[
        f.window.workflow_status._series_key()
    ]["result_receipt"]
    qtbot.waitUntil(lambda: not f.window.workflow_status.checking, timeout=10000)
    assert (
        f.window.workflow_status.states[("mass_analysis", "frozen")].state == "missing"
    )
