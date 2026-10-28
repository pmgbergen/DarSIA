"""Local Qt fixtures, fake registration/apply, no processes or project data."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QSettings
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from darsia.fluidflower import crop_setup_ui as ui
from darsia.fluidflower.series_geometry import NativeSetupExport


class LocalWindow(QWidget):
    def __init__(self, root, package, config):
        super().__init__()
        self.messages = []
        self.reloads = []
        self.config_file = str(config)
        self.config_dict = {}
        self.settings_factory = SimpleNamespace(
            _sync_settings_inputs_to_config_dict=lambda: None
        )
        self.phase_model_workspace = SimpleNamespace(
            series_id="test", series_roots={"test": root}
        )
        self.geometry_workspace = SimpleNamespace(
            _loaded_root=root,
            project=SimpleNamespace(root=package),
            reference=lambda: SimpleNamespace(run_id="ac52"),
            busy=lambda: False,
        )
        self.process_runner = SimpleNamespace(active={})
        self.depth_workflow = SimpleNamespace(sequence=SimpleNamespace(running=False))
        self.print_log = self.messages.append

    def _load_run_config(self, path):
        self.reloads.append(path)
        return True

    def _update_painted_geometry_status(self):
        pass


@pytest.fixture
def prepared(qtbot, monkeypatch, tmp_path):
    root, package = tmp_path / "series", tmp_path / "geometry"
    root.mkdir()
    config = root / "run.toml"
    config.write_text("# unchanged local config\n", encoding="utf-8")
    revision = "a" * 32
    destination = package / "exports/native_setup" / revision
    destination.mkdir(parents=True)
    current = package / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_bytes(b'{"revision_id":"initial"}')
    (destination / "native_setup.toml").write_text(
        "# reference export\n", encoding="utf-8"
    )
    export = NativeSetupExport(
        revision,
        destination,
        destination / "colored_labels.png",
        destination / "native_labels.npz",
        destination / "facies_properties.csv",
        destination / "native_setup.toml",
        "# reference export",
        3,
        2,
        0.001,
        "raw baseline",
        [],
    )
    receipt = {
        "revision_id": revision,
        "reference_run": "ac52",
        "region_count": 3,
        "facies_count": 2,
        "rtol": 0.001,
        "grid": "raw baseline",
    }
    monkeypatch.setattr(ui, "geometry_package_root", lambda _root: package)
    monkeypatch.setattr(
        ui, "completed_current_native_setup_export", lambda _package: receipt
    )
    warnings = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *args: warnings.append(args))
    window = LocalWindow(root, package, config)
    qtbot.addWidget(window)
    task = ui.NativeCropWorkflow(window)
    yield task, window, export, current, warnings
    assert not task.running


def test_export_dialog_attachment_has_no_registration_or_write(
    prepared, qtbot, monkeypatch
):
    task, window, export, _, _ = prepared
    monkeypatch.setattr(
        ui,
        "prepare_crop_plan",
        lambda *_a, **_k: pytest.fail("navigation must not prepare"),
    )
    monkeypatch.setattr(
        ui,
        "apply_crop_plan",
        lambda *_a, **_k: pytest.fail("navigation must not write"),
    )
    dialog = QDialog(window)
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QDialogButtonBox(QDialogButtonBox.StandardButton.Close))
    task.attach_export_dialog(dialog, export)
    assert dialog.apply_corners_button.text() == "Apply rig corners to Setup"
    assert "remain separate exports" in dialog.apply_corners_note.text()
    assert not task.running


def test_open_current_export_only_opens_existing_dialog(prepared, monkeypatch):
    task, window, export, _, _ = prepared
    monkeypatch.setattr(
        ui, "prepare_crop_plan", lambda *_a, **_k: pytest.fail("must not register")
    )
    opened = []
    window.geometry_workspace._show_export_dialog = opened.append
    assert task.open_current_export()
    assert len(opened) == 1 and opened[0].revision_id == export.revision_id
    assert not task.running


def test_explicit_apply_reloads_only_actual_applied_open_config(
    prepared, qtbot, monkeypatch
):
    task, window, export, _, _ = prepared
    calls = []
    monkeypatch.setattr(
        ui,
        "prepare_crop_plan",
        lambda *args, **kwargs: calls.append((args, kwargs)) or object(),
    )
    receipt = export.destination / "test-receipt.json"

    def apply(plan, **kwargs):
        receipt.write_text(
            json.dumps(
                {
                    "state": "complete",
                    "files": [{"path": window.config_file, "state": "applied"}],
                }
            ),
            encoding="utf-8",
        )
        return receipt

    monkeypatch.setattr(ui, "apply_crop_plan", apply)
    with qtbot.waitSignal(task.finished, timeout=3000) as result:
        assert task.start(export)
        assert task.running and not window.isEnabled()
    assert result.args[0] is True
    assert len(calls) == 1
    assert calls[0][1]["expected_revision_id"] == export.revision_id
    assert window.reloads == [window.config_file]
    assert window.isEnabled() and task.last_receipt == receipt


def test_background_failure_is_reported_unchanged_without_retry(
    prepared, qtbot, monkeypatch
):
    task, window, export, _, warnings = prepared
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise ValueError("The rig frame gives no consistent camera move.")

    monkeypatch.setattr(ui, "prepare_crop_plan", broken)
    monkeypatch.setattr(
        ui,
        "apply_crop_plan",
        lambda *_a, **_k: pytest.fail("registration failure must not write"),
    )
    with qtbot.waitSignal(task.finished, timeout=3000) as result:
        task.start(export)
    assert result.args[0] is False
    assert calls == [1] and window.reloads == []
    assert (
        "ValueError: The rig frame gives no consistent camera move." in result.args[1]
    )
    assert "UTC" in result.args[1] and warnings
    assert window.isEnabled()


def test_current_revision_change_during_plan_blocks_all_writes(prepared, monkeypatch):
    task, window, export, current, _ = prepared
    context = task._selected_context(export)

    def prepare(*args, **kwargs):
        current.write_bytes(b"changed revision")
        return object()

    monkeypatch.setattr(ui, "prepare_crop_plan", prepare)
    monkeypatch.setattr(
        ui,
        "apply_crop_plan",
        lambda *_a, **_k: pytest.fail("stale revision must not write"),
    )
    worker = ui._CropWorker(context)
    failures = []
    worker.failed.connect(lambda trace, receipt: failures.append((trace, receipt)))
    worker.run()
    assert "saved geometry revision changed" in failures[0][0]
    assert failures[0][1] == ""


def test_wrong_series_or_changed_reference_never_prepares(prepared, monkeypatch):
    task, window, export, _, warnings = prepared
    monkeypatch.setattr(
        ui,
        "prepare_crop_plan",
        lambda *_a, **_k: pytest.fail("stale context must not prepare"),
    )
    window.geometry_workspace.reference = lambda: SimpleNamespace(run_id="another")
    assert not task.start(export)
    assert not task.running and warnings
    assert "reference run changed" in window.messages[-1].lower()


def test_depth_between_stages_prevents_crop_apply(prepared, monkeypatch):
    task, window, export, _, _ = prepared
    window.depth_workflow.sequence.running = True
    monkeypatch.setattr(
        ui,
        "prepare_crop_plan",
        lambda *_a, **_k: pytest.fail("another calculation is active"),
    )
    assert not task.start(export)
    assert not task.running


def test_close_is_ignored_only_while_apply_thread_is_owned(prepared):
    task, window, _, _, _ = prepared
    task._thread = object()
    event = QCloseEvent()
    QCoreApplication.sendEvent(window, event)
    assert not event.isAccepted()
    task._thread = None


def test_unapplied_open_config_is_not_reloaded(prepared):
    task, window, export, _, _ = prepared
    receipt = export.destination / "test-receipt.json"
    receipt.write_text(
        json.dumps({"files": [{"path": window.config_file, "state": "staged"}]}),
        encoding="utf-8",
    )
    task._reload_changed_config(str(receipt), task._selected_context(export))
    assert window.reloads == []


def test_unrelated_pending_widget_edit_is_preserved_and_blocks_before_plan(
    prepared, monkeypatch
):
    task, window, export, _, warnings = prepared
    Path(window.config_file).write_text("unrelated = 1\n", encoding="utf-8")
    window.config_dict = {"unrelated": 1}
    window.settings_factory._sync_settings_inputs_to_config_dict = (
        lambda: window.config_dict.update(unrelated=2)
    )
    monkeypatch.setattr(
        ui,
        "prepare_crop_plan",
        lambda *_a, **_k: pytest.fail("unsaved settings must stop before registration"),
    )
    monkeypatch.setattr(
        ui,
        "apply_crop_plan",
        lambda *_a, **_k: pytest.fail("unsaved settings must stop before writing"),
    )
    assert not task.start(export)
    assert window.config_dict["unrelated"] == 2
    assert Path(window.config_file).read_text(encoding="utf-8") == "unrelated = 1\n"
    assert window.reloads == [] and not task.running
    assert "Save it first" in warnings[-1][-1]


def test_native_sidebar_has_one_corner_row_and_no_navigation_calculation(
    qtbot, monkeypatch, tmp_path
):
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
    QCoreApplication.setOrganizationName("DarSIA-crop-ui-tests")
    QCoreApplication.setApplicationName(tmp_path.name)
    monkeypatch.setattr(native, "configured_series", lambda: ())
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
        ui,
        "prepare_crop_plan",
        lambda *_a, **_k: pytest.fail("navigation must not register"),
    )
    monkeypatch.setattr(
        ui,
        "apply_crop_plan",
        lambda *_a, **_k: pytest.fail("navigation must not write"),
    )
    window = native.NativeFluidFlowerMainWindow()
    qtbot.addWidget(window)
    try:
        qtbot.waitUntil(lambda: not window._series_loader.busy(), timeout=10000)
        rows = window.sidebar._sections["setup"]._rows
        assert "preprocessing:crop" not in rows
        assert (
            rows["geometry_facies:rig_frame"].label
            == "Rig Geometry"
        )
        assert rows["geometry_facies:export"].label == "Sand map in all runs"
        opened = []
        monkeypatch.setattr(
            window.crop_workflow,
            "open_current_export",
            lambda: opened.append(1) or False,
        )
        scopes = []
        monkeypatch.setattr(window, "_set_geometry_workflow_scope", scopes.append)
        window._select_row(("geometry_facies", "rig_frame"))
        assert not window.toolbar_builder.play_action.isEnabled()
        window._select_row(("geometry_facies", "export"))
        assert scopes[-2:] == ["rig_frame", "export"]
        assert opened == []
        assert not window.crop_workflow.running
        assert all("is applied" not in row.label for row in rows.values())
    finally:
        window.close()
        QCoreApplication.setOrganizationName(old[1])
        QCoreApplication.setApplicationName(old[2])
        QSettings.setDefaultFormat(old[0])


def test_native_export_dialog_marks_label_files_as_reference_only(prepared, qtbot):
    from darsia.fluidflower.series_geometry_workspace import NativeSetupExportDialog

    task, window, export, _, _ = prepared
    dialog = NativeSetupExportDialog(export, window)
    qtbot.addWidget(dialog)
    task.attach_export_dialog(dialog, export)
    labels = [label.text() for label in dialog.findChildren(ui.QLabel)]
    assert any("only into its matching run configuration" in text for text in labels)
    assert not any("config/common.toml for all runs" in text for text in labels)
    assert dialog.apply_corners_button.text() == "Apply rig corners to Setup"
