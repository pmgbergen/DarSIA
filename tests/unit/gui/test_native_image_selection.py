"""Real native Qt integration with local-only metadata and no computations."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtTest import QTest

from darsia.fluidflower import series_image_selection as model
from darsia.fluidflower import image_selection_widget as ui
from darsia.fluidflower import image_selection_status as summary
from darsia.fluidflower import image_selection_ui as adapter
from darsia.fluidflower import workflow_row_status as row_status
from darsia.fluidflower import workflow_status_ui as status_ui
from darsia.fluidflower import row_scope, rig_display as rig
from darsia.fluidflower import native_darsia_app as native


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def local_window(qtbot, tmp_path, monkeypatch):
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
    QCoreApplication.setOrganizationName("B4-local-metadata-native")
    QCoreApplication.setApplicationName(tmp_path.name)
    root = tmp_path / "series"
    put(
        root / "series.toml",
        'schema_version=1\nseries_id="fixture"\nname="Local fixture"\n[runs]\nids=["ac17","ac18"]\n',
    )
    for run in ("ac17", "ac18"):
        put(
            root / f"config/merged/{run}.toml",
            f'run_marker="{run}"\n[rig]\nwidth=0.9\nheight=0.5\n',
        )
    monkeypatch.setattr(
        native, "configured_series", lambda: (("fixture", "Local fixture", root),)
    )
    monkeypatch.setattr(
        native, "ProjectRegistry", lambda: SimpleNamespace(entries=lambda: [])
    )
    calls = []
    warnings = []
    monkeypatch.setattr(
        native.QMessageBox, "warning", lambda *args: warnings.append(args[1:])
    )
    monkeypatch.setattr(
        native.PhaseModelWorkspace,
        "ensure_studio",
        lambda self: calls.append("studio") or self.studio_placeholder,
    )
    monkeypatch.setattr(
        native.SeriesGeometryWorkspace, "ensure_page", lambda self: self
    )
    monkeypatch.setattr(
        native.NativeFluidFlowerMainWindow,
        "ensure_mass_analysis",
        lambda self: self.mass_analysis_placeholder,
    )
    w = native.NativeFluidFlowerMainWindow(initial_series="fixture")
    qtbot.addWidget(w, before_close_func=lambda window:
                    qtbot.waitUntil(lambda: not window._series_loader.busy()))

    def forbidden(*a, **k):
        pytest.fail(
            "Native image selection cannot start computation, wizard or history discovery"
        )

    monkeypatch.setattr(w.process_runner, "start_workflow_process", forbidden)
    monkeypatch.setattr(w, "run_selected_workflow", forbidden)
    monkeypatch.setattr(w, "_open_preprocessing_for_selected_run", forbidden)
    monkeypatch.setattr(status_ui, "open_in_file_explorer", forbidden)
    w.resize(1600, 1100)
    w.show()
    qtbot.waitUntil(w.isVisible)
    qtbot.waitUntil(lambda: not w._series_loader.busy())
    try:
        yield SimpleNamespace(window=w, root=root, calls=calls, warnings=warnings)
    finally:
        from shiboken6 import isValid
        if isValid(w):
            qtbot.waitUntil(lambda: not w._series_loader.busy())
            w.close()
        QCoreApplication.setOrganizationName(old[1])
        QCoreApplication.setApplicationName(old[2])
        QSettings.setDefaultFormat(old[0])


def metadata_for(root, run):
    source = put(root / f"{run}_protocol.csv", "local metadata binding")
    return model.RunImageMetadata(
        run,
        (
            {
                "source_path": str(root / f"{run}_pre.png"),
                "frame_index": 0,
                "image_id": f"{run}_pre.png",
                "elapsed_h": -0.1,
            },
            {
                "source_path": str(root / f"{run}_post.png"),
                "frame_index": 1,
                "image_id": f"{run}_post.png",
                "elapsed_h": 2.0,
            },
        ),
        0,
        1,
        (model._descriptor(source),),
        (),
        str(root),
    )


def test_moved_setup_row_opens_passive_series_panel_without_protocol_or_config_load(
    local_window, qtbot, monkeypatch
):
    f = local_window
    w = f.window
    old = (f.root / "series.toml").read_bytes()
    monkeypatch.setattr(
        adapter,
        "read_series_image_selection",
        lambda root: model.read_series_image_selection(root),
    )
    w._select_row(("image_selection", "plan"))
    qtbot.waitUntil(w.image_selection_workflow.panel.isVisible)
    assert w.sidebar.display_selection("image_selection", "plan")[0] == "setup"
    assert w.content_stack.currentWidget() is w.image_selection_workflow.panel
    assert w.current_row_scope.scope == "series" and not w.run_combo.isEnabled()
    # The rig and method lines are details of the one header line (N141).
    assert not w.rig_label.isHidden() and w.status_header.line.isVisible()
    assert not w.method_badge.isHidden() and not w.method_panel.isVisible()
    assert (
        not w.toolbar_builder.play_action.isEnabled()
        and not w.toolbar_builder.wizard_action.isEnabled()
    )
    assert w.image_selection_workflow.edit_button.isEnabled()
    assert "0 of 2 active runs" in w.image_selection_workflow.status.text()
    assert w.workflow_status.states[("image_selection", "plan")].state == "missing"
    assert ("phase_model", "selection_plan") not in [
        (r.action, r.checkbox_id)
        for r in w.sidebar._sections["calibration"]._rows.values()
    ]
    assert not w.config_file and not f.calls and not f.warnings
    assert (f.root / "series.toml").read_bytes() == old


def test_native_edit_uses_exact_series_and_real_passive_editor(local_window, qtbot):
    f = local_window
    w = f.window
    opened = []

    def factory(root, parent, **kwargs):
        dialog = ui.SeriesImageSelectionDialog(
            root,
            parent,
            metadata_reader=lambda *a: pytest.fail("Load was not clicked"),
            **kwargs,
        )
        dialog.exec = (
            lambda: opened.append((Path(root), dialog.suggest_button.isEnabled())) or 0
        )
        return dialog

    w.image_selection_workflow.dialog_factory = factory
    w._select_row(("image_selection", "plan"))
    old = (f.root / "series.toml").read_bytes()
    QTest.mouseClick(w.image_selection_workflow.edit_button, Qt.MouseButton.LeftButton)
    assert opened == [(f.root, False)]
    assert w.image_selection_workflow.dialog.table.rowCount() == 0
    assert (f.root / "series.toml").read_bytes() == old and not f.calls


def test_saved_choices_status_opens_editor_not_explorer_or_computation(
    local_window, qtbot
):
    f = local_window
    w = f.window
    for run in ("ac17", "ac18"):
        meta = metadata_for(f.root, run)
        model.save_run_image_selection(
            f.root,
            meta,
            [row["source_path"] for row in meta.frames],
            expected_bytes=(f.root / "series.toml").read_bytes(),
        )
    # Removed runs remain historical exclusions, not missing active choices.
    with (f.root / "series.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[runs.excluded]\nremoved_run="Removed from the active series"\n')
    w._select_series('fixture')
    qtbot.waitUntil(lambda: not w._series_loader.busy())
    w._select_row(("image_selection", "plan"))
    qtbot.waitUntil(lambda: not w.workflow_status.checking, timeout=10000)
    state = w.workflow_status.states[("image_selection", "plan")]
    assert state.state == "done" and "4 images in 2 of 2 active runs" in state.summary
    assert "not a computed result" in state.summary and state.date
    assert state.selected_run_text == ""
    assert "Analysis exclusion scope unavailable" not in w.workflow_status.label.text()
    choices = w.image_selection_workflow.saved_choices
    assert choices.rowCount() == 2
    assert [choices.item(0, column).text() for column in range(4)] == [
        "AC17", "2", "-0.1 to 2", "Yes"
    ]
    assert [choices.item(1, column).text() for column in range(4)] == [
        "AC18", "2", "-0.1 to 2", "Yes"
    ]
    assert "Rig Geometry" in w.image_selection_workflow.note.text()
    assert "Calibration determines" in w.image_selection_workflow.note.text()
    assert "no separate saved grid" in w.image_selection_workflow.production_status.text()
    opened = []

    def factory(root, parent, **kwargs):
        dialog = ui.SeriesImageSelectionDialog(
            root,
            parent,
            metadata_reader=lambda *a: pytest.fail("Load not clicked"),
            **kwargs,
        )
        dialog.exec = lambda: opened.append(Path(root)) or 0
        return dialog

    w.image_selection_workflow.dialog_factory = factory
    before = (f.root / "series.toml").read_bytes()
    button = w.workflow_status.open_button
    assert button.text() == "Open"
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert opened == [f.root] and not f.calls
    assert (f.root / "series.toml").read_bytes() == before


def test_summary_partial_and_malformed_tables_never_claim_computed_output():
    doc = {"schema_version": 1, "runs": {"ids": ["ac17", "ac18"]}}
    info = summary.saved_selection_summary(doc)
    assert not info["complete"] and info["missing_runs"] == ["ac17", "ac18"]
    bad = deepcopy(doc)
    bad["image_selection"] = {"format": "wrong", "schema_version": 1, "runs": {}}
    with pytest.raises(ValueError):
        summary.saved_selection_summary(bad)


def test_passive_summary_rejects_arbitrary_source_binding_not_done():
    doc = {
        "schema_version": 1,
        "runs": {"ids": ["ac17"]},
        "image_selection": {
            "format": model.FORMAT,
            "schema_version": 1,
            "runs": {
                "ac17": {
                    "selected": [
                        {
                            "source_path": "pre.png",
                            "image_id": "pre.png",
                            "frame_index": 0,
                            "elapsed_h": -0.1,
                        }
                    ],
                    "source_metadata": [{}],
                }
            },
        },
    }
    with pytest.raises(ValueError, match="source metadata binding"):
        summary.saved_selection_summary(doc)
