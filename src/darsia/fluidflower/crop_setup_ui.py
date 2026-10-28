"""Explicit native-window UI workflow for applying exported rig corners."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
import tomllib

from PySide6.QtCore import QEvent, QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel, QMessageBox

from darsia.fluidflower.series_geometry import NativeSetupExport, geometry_package_root
from darsia.fluidflower.setup_display import completed_current_native_setup_export
from darsia.fluidflower.setup_geometry_apply import apply_crop_plan, prepare_crop_plan


def _same_path(first, second) -> bool:
    if not first or not second:
        return False
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
            os.path.abspath(second)
        )


@dataclass(frozen=True)
class CropContext:
    series_id: str
    root: Path
    export: NativeSetupExport
    current_json: Path
    current_bytes: bytes


class _CropWorker(QObject):
    completed = Signal(str)
    failed = Signal(str, str)

    def __init__(self, context: CropContext) -> None:
        super().__init__()
        self.context = context

    @Slot()
    def run(self) -> None:
        context = self.context
        try:
            plan = prepare_crop_plan(
                context.root,
                context.export.destination / "export.json",
                expected_revision_id=context.export.revision_id,
            )
            if context.current_json.read_bytes() != context.current_bytes:
                raise RuntimeError(
                    "The saved geometry revision changed during registration. No crop configuration was applied."
                )
            receipt = apply_crop_plan(
                plan,
                receipt_directory=context.export.destination / "crop_apply_receipts",
            )
        except Exception as error:
            receipt = str(getattr(error, "receipt", ""))
            self.failed.emit(traceback.format_exc(), receipt)
        else:
            self.completed.emit(str(receipt))


class NativeCropWorkflow(QObject):
    """No planning or writes until the user clicks the explicit Apply button."""

    finished = Signal(bool, str)

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self._thread: QThread | None = None
        self._worker: _CropWorker | None = None
        self._context: CropContext | None = None
        self._outcome = None
        self._button = None
        self._note = None
        self._previous_enabled = True
        self.last_receipt: Path | None = None
        window.installEventFilter(self)

    @property
    def running(self) -> bool:
        return self._thread is not None

    def eventFilter(self, watched, event):
        if (
            watched is self.window
            and event.type() == QEvent.Type.Close
            and self.running
        ):
            event.ignore()
            self.window.print_log(
                "Rig-corner application is still running. Let its current write finish before closing."
            )
            return True
        return super().eventFilter(watched, event)

    def _selected_context(self, export: NativeSetupExport) -> CropContext:
        workspace = self.window.phase_model_workspace
        series_id = workspace.series_id
        root = workspace.series_roots.get(series_id)
        geometry = self.window.geometry_workspace
        if root is None or not series_id:
            raise ValueError("Choose a series before applying rig corners.")
        root = Path(root)
        if not _same_path(getattr(geometry, "_loaded_root", None), root):
            raise ValueError(
                "The geometry workspace belongs to another selected series."
            )
        package = geometry_package_root(root)
        if geometry.project is None or not _same_path(geometry.project.root, package):
            raise ValueError(
                "The loaded geometry package is not bound to the selected series."
            )
        receipt = completed_current_native_setup_export(package)
        if receipt is None or receipt.get("revision_id") != export.revision_id:
            raise ValueError(
                "Save and export the current geometry revision before applying its corners."
            )
        destination = package / "exports" / "native_setup" / export.revision_id
        if not _same_path(export.destination, destination):
            raise ValueError("The export belongs to another geometry package.")
        reference = geometry.reference()
        if reference is None or receipt.get("reference_run") != reference.run_id:
            raise ValueError(
                "The reference run changed. Export the current reference geometry first."
            )
        current_json = package / "overrides" / "geometry_facies" / "current.json"
        return CropContext(
            series_id, root, export, current_json, current_json.read_bytes()
        )

    def attach_export_dialog(self, dialog, export: NativeSetupExport) -> None:
        """Add a button to the existing dialog; never register or apply here."""
        from darsia.fluidflower.corrected_setup import CorrectedSetupExport

        if isinstance(export, CorrectedSetupExport):
            return  # Its own selected-run Apply never performs raw registration.
        if getattr(dialog, "apply_corners_button", None) is not None:
            return
        buttons = dialog.findChild(QDialogButtonBox)
        if buttons is None:
            raise RuntimeError("The native geometry export dialog has no button box.")
        for label in dialog.findChildren(QLabel):
            if label.text().startswith("Add these sections to config/common.toml"):
                label.setText(
                    "The label and facies files belong to the reference photograph. "
                    "Copy their sections only into its matching run configuration; "
                    "they have not been registered for other runs. Existing "
                    "injection protocols are unchanged."
                )
        note = QLabel(
            "Apply registers these rig corners to each run's raw baseline and "
            "writes its crop settings. Labels, facies and injection protocols "
            "remain separate exports; they are not applied to other runs here."
        )
        note.setTextFormat(Qt.TextFormat.PlainText)
        note.setWordWrap(True)
        dialog.layout().insertWidget(dialog.layout().count() - 1, note)
        button = buttons.addButton(
            "Apply rig corners to Setup", QDialogButtonBox.ButtonRole.ActionRole
        )
        dialog.apply_corners_button = button
        dialog.apply_corners_note = note
        button.clicked.connect(lambda: self.start(export, button=button, note=note))

    def open_current_export(self) -> bool:
        """Read-only opening of an existing export from the Apply sidebar row."""
        from darsia.fluidflower.corrected_setup_ui import open_current_corrected_export

        if open_current_corrected_export(self.window):
            return True
        workspace = self.window.phase_model_workspace
        root = workspace.series_roots.get(workspace.series_id)
        if root is None:
            return False
        package = geometry_package_root(Path(root))
        receipt = completed_current_native_setup_export(package)
        if receipt is None:
            self.window.print_log(
                "Save and export the current geometry revision first; opening this row has not changed any configuration."
            )
            return False
        destination = package / "exports" / "native_setup" / receipt["revision_id"]
        try:
            export = NativeSetupExport(
                revision_id=receipt["revision_id"],
                destination=destination,
                colored_image=destination / "colored_labels.png",
                native_labels=destination / "native_labels.npz",
                facies_properties=destination / "facies_properties.csv",
                setup_toml=destination / "native_setup.toml",
                toml_text=(destination / "native_setup.toml").read_text(
                    encoding="utf-8"
                ),
                region_count=int(receipt["region_count"]),
                facies_count=int(receipt["facies_count"]),
                rtol=float(receipt["rtol"]),
                grid=str(receipt["grid"]),
                warnings=list(receipt.get("warnings") or []),
                reused=True,
            )
            self._selected_context(export)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.window.print_log(
                f"Could not open the current geometry export: {error}"
            )
            return False
        # The native adapter already wraps this unchanged export-dialog method
        # and attaches our explicit button when the dialog has been created.
        self.window.geometry_workspace._show_export_dialog(export)
        return True

    def start(self, export: NativeSetupExport, *, button=None, note=None) -> bool:
        if self.running:
            return False
        depth = getattr(self.window, "depth_workflow", None)
        if self.window.process_runner.active or getattr(
            getattr(depth, "sequence", None), "running", False
        ):
            self.window.print_log(
                "Wait for the current calculation before changing run crop settings."
            )
            return False
        geometry = self.window.geometry_workspace
        if geometry.busy():
            self.window.print_log(
                "Let the geometry export finish before applying its corners."
            )
            return False
        try:
            context = self._selected_context(export)
            if not self._require_saved_open_settings():
                return False  # The shared guard already explained the refusal.
        except Exception as error:
            self.window.print_log(f"Could not apply rig corners: {error}")
            QMessageBox.warning(self.window, "Rig corners were not applied", str(error))
            return False
        self._context = context
        self._button, self._note = button, note
        self._outcome = None
        self._previous_enabled = self.window.isEnabled()
        self.window.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        if note is not None:
            note.setText(
                "Registering raw baseline photographs and preparing the run crop settings…"
            )
        self.window.print_log(
            "Applying rig corners: registering raw baseline photographs; original run files will be preserved in the application receipt."
        )
        thread = QThread(self)
        worker = _CropWorker(context)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._capture_complete)
        worker.failed.connect(self._capture_failure)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._finish)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        thread.start()
        return True

    def _require_saved_open_settings(self) -> bool:
        """Reuse the native Save-first guard, retaining the standalone fallback."""
        guard = getattr(self.window, "reference_open_guard", None)
        if guard is not None:
            return guard.allowed()
        opened = getattr(self.window, "config_file", None)
        if not opened:
            return True
        # The native factory's existing sync also settles deferred field fills.
        # It does not save. Keep all edits in config_dict if Save-first stops us.
        self.window.settings_factory._sync_settings_inputs_to_config_dict()
        saved = tomllib.loads(Path(opened).read_bytes().decode("utf-8-sig"))
        if self.window.config_dict != saved:
            raise ValueError(
                "The open configuration has unsaved settings. Save it first, then "
                "apply rig corners. Your edits remain in the editor; no crop "
                "registration or configuration write has started."
            )
        return True

    @Slot(str)
    def _capture_complete(self, receipt: str) -> None:
        self._outcome = (True, receipt, "")

    @Slot(str, str)
    def _capture_failure(self, trace: str, receipt: str) -> None:
        self._outcome = (False, receipt, trace)

    def _reload_changed_config(self, receipt: str, context: CropContext) -> None:
        if not receipt:
            return
        data = json.loads(Path(receipt).read_bytes())
        applied = [
            item["path"]
            for item in data.get("files") or []
            if item.get("state") == "applied"
        ]
        opened = self.window.config_file
        if opened and any(_same_path(opened, path) for path in applied):
            workspace = self.window.phase_model_workspace
            if workspace.series_id == context.series_id and _same_path(
                workspace.series_roots.get(workspace.series_id), context.root
            ):
                if not self.window._load_run_config(opened):
                    self.window.print_log(
                        "The crop settings were written, but the open configuration could not be refreshed. Reopen that configuration before running Setup."
                    )

    @Slot()
    def _finish(self) -> None:
        success, receipt, trace = self._outcome or (
            False,
            "",
            "The crop worker exited without a completion result.",
        )
        context = self._context
        if self._thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            self._thread.wait()
        self._thread, self._worker = None, None
        self.window.setEnabled(self._previous_enabled)
        QApplication.restoreOverrideCursor()
        if receipt:
            self.last_receipt = Path(receipt)
            try:
                self._reload_changed_config(receipt, context)
            except Exception as error:
                self.window.print_log(
                    f"Could not reload a changed crop configuration: {type(error).__name__}: {error}"
                )
        if success:
            message = f"Rig corners applied to run crop configurations. Receipt and original files: {receipt}"
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            message = f"Rig-corner application stopped at {stamp}.\n{trace}"
            if receipt:
                message += f"\nPartial-application receipt and originals: {receipt}"
        self.window.print_log(message)
        try:
            if self._note is not None:
                self._note.setText(message)
            if self._button is not None:
                self._button.setEnabled(not success)
        except RuntimeError:
            pass  # The user may have closed the export dialog after completion.
        if not success:
            QMessageBox.warning(self.window, "Rig-corner application stopped", message)
        self.window._update_painted_geometry_status()
        self.finished.emit(success, message)
