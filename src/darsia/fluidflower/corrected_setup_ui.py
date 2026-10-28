"""Native-only corrected export dialog; writes require an explicit Apply."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib
import traceback

from PySide6.QtCore import QEvent, QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QMessageBox, QPlainTextEdit, QVBoxLayout,
)

from darsia.fluidflower.corrected_setup import (
    FORMAT, EXPORT_FILES, CorrectedSetupExport, _small, _canonical_location,
    _identity, _replacements, _same, apply_corrected_selected_run,
)
from darsia.fluidflower.geometry_facies import GeometryRevisionStore
from darsia.fluidflower.geometry_workspace_frame import current_configuration, frame_kind
from darsia.fluidflower.project import utc_now
from darsia.fluidflower.series_geometry import native_series_link


class _Apply(QObject):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, project, export):
        super().__init__()
        self.project, self.export = project, export

    @Slot()
    def run(self):
        try:
            result = apply_corrected_selected_run(self.project, self.export)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.completed.emit(str(result))


class _ApplyAll(QObject):
    """Export and apply the current corrected revision for every run."""

    progress = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, project, references, actor):
        super().__init__()
        self.project, self.references, self.actor = project, list(references), actor

    @Slot()
    def run(self):
        from darsia.fluidflower.corrected_setup_all_runs import apply_geometry_to_all_runs

        try:
            path, _result = apply_geometry_to_all_runs(
                self.project, self.references, actor=self.actor, progress=self.progress.emit
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.completed.emit(str(path))


class _CheckRebuilt(QObject):
    """Check every run's built rig against its adopted labels (read only)."""

    progress = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, project, references):
        super().__init__()
        self.project, self.references = project, list(references)

    @Slot()
    def run(self):
        from darsia.fluidflower.corrected_setup_all_runs import check_rebuilt_runs

        try:
            path, _result = check_rebuilt_runs(self.project, self.references, progress=self.progress.emit)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.completed.emit(str(path))


class CorrectedSetupExportDialog(QDialog):
    """Keep the existing raw export dialog/crop application entirely separate."""
    def __init__(self, export, workspace):
        super().__init__(workspace)
        self.export, self.workspace, self.project = export, workspace, workspace.project
        self._thread = self._worker = self._outcome = None
        self._mode = "one"
        self._previous_enabled = True
        self._window = workspace.window()
        self._window.installEventFilter(self)
        self.setWindowTitle("Corrected geometry for the selected run")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        self.note = QLabel(
            f"Run {export.run_id}. Original labels and facies are preserved as byte copies. "
            "Apply changes only this run's label/facies input references. Its depth, "
            "crop and protocols stay unchanged. Build run model remains an explicit action."
        )
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        text = QPlainTextEdit(export.toml_text)
        text.setReadOnly(True)
        layout.addWidget(text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.apply_button = buttons.addButton(
            f"Apply to {export.run_id}", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.apply_button.clicked.connect(self.start)
        self.apply_all_button = buttons.addButton(
            "Apply to all runs…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.apply_all_button.setToolTip(
            "Export and apply this revision for every run of the series, the display "
            "reference run first, with one receipt; Build run model is not run"
        )
        self.apply_all_button.clicked.connect(self.start_all)
        self.check_button = buttons.addButton(
            "Check re-export…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.check_button.setToolTip(
            "After Build run model: check that every run's built rig holds exactly the labels "
            "and facies its applied export adopted; reads only, one receipt"
        )
        self.check_button.clicked.connect(self.start_check)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _ready(self):
        """The main window when a write may start now; None after saying why."""

        window = self.workspace.window()
        reference = self.workspace.reference()
        if self.workspace.project is not self.project or reference is None or reference.run_id != self.export.run_id or self.workspace._loaded_series != self.workspace.series_id:
            self.note.setText("The selected series or run changed. Reopen its corrected export.")
            return None
        if getattr(window, "process_runner", None) is not None and window.process_runner.active:
            self.note.setText("Wait for the current calculation before applying setup inputs.")
            return None
        if self.workspace.busy():
            self.note.setText("Wait for the geometry operation to finish before applying setup inputs.")
            return None
        guard = getattr(window, "reference_open_guard", None)
        if guard is None:
            self.note.setText("Save-first protection is unavailable; no configuration was changed.")
            return None
        if not guard.allowed():
            return None
        return window

    def start(self):
        if self._thread is not None:
            return
        window = self._ready()
        if window is None:
            return
        self._run(window, _Apply(self.project, self.export), "one")

    def start_all(self):
        """Ask, then export and apply this revision for every run, with one receipt."""

        if self._thread is not None:
            return
        from darsia.fluidflower.corrected_setup_all_runs import current_corrected_revision
        from darsia.fluidflower.series_geometry import default_actor

        references = list(getattr(self.workspace, "run_inputs", None) or [])
        if not references:
            self.note.setText("The series lists no runs; nothing was applied.")
            return
        try:
            revision = current_corrected_revision(self.project)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.note.setText(f"Could not read the current geometry revision: {error}")
            return
        if revision["revision_id"] != self.export.revision_id:
            self.note.setText("The geometry revision changed. Reopen its corrected export.")
            return
        window = self._ready()
        if window is None:
            return
        reason = f" ({revision['reason']})" if revision["reason"] else ""
        answer = QMessageBox.question(
            self,
            "Apply to all runs",
            f"Apply geometry revision {revision['revision_id'][:8]}{reason} to all "
            f"{len(references)} runs? The display reference run "
            f"{revision['display_reference_run']} goes first. For each run its corrected "
            "Setup inputs are exported as byte copies and only its label and facies "
            "references change; depth, crop and protocols stay unchanged, and the "
            "originals are kept beside each export. The first run that cannot be applied "
            "stops the sequence. Build run model is not run. One receipt lists every run.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._run(window, _ApplyAll(self.project, references, default_actor()), "all")

    def start_check(self):
        """Check every run's built rig against its adopted labels, in a worker."""

        if self._thread is not None:
            return
        references = list(getattr(self.workspace, "run_inputs", None) or [])
        if not references:
            self.note.setText("The series lists no runs; nothing was checked.")
            return
        window = self.workspace.window()
        if getattr(window, "process_runner", None) is not None and window.process_runner.active:
            self.note.setText("Wait for the current calculation before checking the built runs.")
            return
        self._run(window, _CheckRebuilt(self.project, references), "check")

    def _run(self, window, worker, mode):
        """Run one apply worker in its own thread, the main window disabled."""

        self._mode = mode
        self._previous_enabled = window.isEnabled()
        window.setEnabled(False)
        self._outcome = None
        self.apply_button.setEnabled(False)
        self.apply_all_button.setEnabled(False)
        self.check_button.setEnabled(False)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        if mode in ("all", "check"):
            worker.progress.connect(self.note.setText)
        worker.completed.connect(self._capture_success)
        worker.failed.connect(self._capture_failure)
        worker.completed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._finish)
        self._thread, self._worker = thread, worker
        thread.start()

    @Slot(str)
    def _capture_success(self, value):
        self._outcome = (True, value)

    @Slot(str)
    def _capture_failure(self, value):
        self._outcome = (False, value)

    @Slot()
    def _finish(self):
        thread = self._thread
        if thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            thread.wait()
        self._thread = self._worker = None
        if thread is not None:
            thread.deleteLater()
        window = self.workspace.window()
        window.setEnabled(self._previous_enabled)
        # Checking reads only, so it is available again after any step.
        self.check_button.setEnabled(True)
        success, value = self._outcome or (False, "The setup writer exited without a result.")
        if self._mode == "all":
            self._finish_all(window, success, value)
            return
        if self._mode == "check":
            self._finish_check(window, success, value)
            return
        if success:
            receipt = json.loads(_small(value))
            opened = getattr(window, "config_file", None)
            if opened and any(Path(row["path"]).resolve() == Path(opened).resolve() for row in receipt["files"]):
                if not window._load_run_config(opened):
                    self.note.setText(f"Inputs were applied; reopen the run configuration. Receipt: {value}")
                    return
            self.note.setText(f"Corrected inputs applied to {self.export.run_id}. Build run model has not run. Receipt: {value}")
            refresh = getattr(window, "_update_painted_geometry_status", None)
            if callable(refresh):
                refresh()
        else:
            message = f"Corrected setup application stopped at {utc_now()}.\n{value}"
            self.note.setText(message)
            window.print_log(message)
            self.apply_button.setEnabled(True)
            self.apply_all_button.setEnabled(True)

    def _finish_check(self, window, success, value):
        """Say what the re-export check found, from its receipt."""

        self.check_button.setEnabled(True)
        self.apply_all_button.setEnabled(True)
        if not success:
            message = f"The re-export check stopped at {utc_now()} before its receipt.\n{value}"
            self.note.setText(message)
            window.print_log(message)
            return
        receipt = json.loads(_small(value))
        groups = {}
        for row in receipt["runs"]:
            groups.setdefault(row["status"], []).append(row["run_id"])
        text = f"Re-export check: identical in {len(groups.get('identical', []))} of {len(receipt['runs'])} runs."
        for status in ("differs", "not built", "not applied"):
            if groups.get(status):
                text += f" {status.capitalize()}: {', '.join(groups[status])}."
        text += f" Receipt: {value}"
        self.note.setText(text)
        window.print_log(text)

    def _finish_all(self, window, success, value):
        """Say what the all-runs apply did, from its receipt."""

        if not success:
            message = f"Applying to all runs stopped at {utc_now()} before its receipt.\n{value}"
            self.note.setText(message)
            window.print_log(message)
            self.apply_button.setEnabled(True)
            self.apply_all_button.setEnabled(True)
            return
        receipt = json.loads(_small(value))
        rows = receipt["runs"]
        applied = [row["run_id"] for row in rows if row["status"] == "applied"]
        stopped = [row for row in rows if row["status"] != "applied"]
        total = len(rows) + len(receipt["not_attempted"])
        text = f"Applied to {len(applied)} of {total} runs. Build run model has not run."
        if stopped:
            text += f" Stopped at {stopped[0]['run_id']}: {stopped[0]['error']}"
        if receipt["not_attempted"]:
            text += f" Not attempted: {', '.join(receipt['not_attempted'])}."
        text += f" Receipt: {value}"
        self.note.setText(text)
        window.print_log(text)
        opened = getattr(window, "config_file", None)
        if opened and applied and not window._load_run_config(opened):
            self.note.setText(text + " Reopen the run configuration.")
        refresh = getattr(window, "_update_painted_geometry_status", None)
        if callable(refresh):
            refresh()
        if stopped:
            self.apply_all_button.setEnabled(True)

    def eventFilter(self, watched, event):
        if watched is self._window and event.type() == QEvent.Type.Close and self._thread is not None:
            event.ignore()
            self._window.print_log("Corrected setup inputs are still being written; wait for the write to finish.")
            return True
        return super().eventFilter(watched, event)

    def reject(self):
        if self._thread is None:
            super().reject()

    def closeEvent(self, event):
        if self._thread is not None:
            event.ignore()
        else:
            super().closeEvent(event)


def open_current_corrected_export(window):
    """Metadata-only Open route; True means this corrected route handled it."""
    workspace = window.geometry_workspace
    project = workspace.project
    if project is None:
        return False
    try:
        config = current_configuration(project)
        if frame_kind(config) != "rig-corrected":
            return False
    except (OSError, ValueError, KeyError, TypeError) as error:
        window.print_log(f"Could not inspect the current geometry revision: {error}")
        return True
    try:
        reference = workspace.reference()
        if reference is None:
            raise ValueError("Choose the run whose corrected export is to be opened.")
        path = GeometryRevisionStore(project).current_path
        current_bytes = _small(path)
        current = json.loads(current_bytes)
        revision = current["revision_id"]
        if len(revision) != 32 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Invalid workspace revision UUID.")
        root, destination, display, planned = _canonical_location(
            project.root, current["configuration"], revision, reference.run_id
        )
        receipt_path = destination / "export.json"
        if not receipt_path.is_file():
            window.print_log(f"No saved corrected export is available for run {reference.run_id}: {receipt_path}")
            return True
        receipt = json.loads(_small(receipt_path))
        current_sha = hashlib.sha256(current_bytes).hexdigest()
        if receipt.get("schema") != 1 or receipt.get("format") != FORMAT or receipt.get("revision_id") != revision or receipt.get("run_id") != reference.run_id or receipt.get("frame") != "rig-corrected" or receipt.get("reference_run") != display or receipt.get("current_sha256") != current_sha:
            raise ValueError("The corrected export is not bound to the current revision and run.")
        if reference.run_id == display:
            if receipt.get("runs") != planned or receipt.get("root_export") is not None:
                raise ValueError("The reference export does not bind its planned canonical children.")
        else:
            root_path = root / "export.json"
            root_bytes = _small(root_path)
            root_receipt = json.loads(root_bytes)
            expected_root = {"path": str(root_path), "sha256": hashlib.sha256(root_bytes).hexdigest(), "size_bytes": len(root_bytes)}
            if (root_receipt.get("schema") != 1 or root_receipt.get("format") != FORMAT
                    or root_receipt.get("revision_id") != revision or root_receipt.get("run_id") != display
                    or root_receipt.get("reference_run") != display or root_receipt.get("frame") != "rig-corrected"
                    or root_receipt.get("current_sha256") != current_sha or root_receipt.get("runs") != planned
                    or receipt.get("root_export") != expected_root):
                raise ValueError("This selected run's child is not bound to the actual reference export.")
        saved = current["configuration"]["frame_binding"]["runs"][reference.run_id]
        binding = receipt["binding"]
        if (binding["configuration"] != _identity(saved["configuration"])
                or binding["properties"] != _identity(saved["facies_properties"])
                or binding["configured_depth_source"] != _identity(saved["configured_depth_map"])
                or binding["replacements"] != _replacements(destination)
                or any(binding["sources"][name] != _identity(saved["fields"][name])
                       for name in ("shape_corrected_baseline", "labels", "facies", "depth"))):
            raise ValueError("The corrected export inputs belong to a different selected-run binding.")
        series = Path(native_series_link(project)["series_root"])
        bundles = binding.get("configuration_bundles") or {}
        selected_bundle = bundles.get(reference.run_id) or []
        if (not _same(binding.get("series_root", ""), series)
                or not _same(binding["series_manifest"]["path"], series / "series.toml")
                or not selected_bundle or not any(_same(binding["configuration"]["path"], item) for item in selected_bundle)
                or any(_same(item, other) for run, rows in bundles.items() if run != reference.run_id
                       for item in selected_bundle for other in rows)):
            raise ValueError("The corrected export has no unique bundle for the selected series and run.")
        if set(receipt["files"]) != EXPORT_FILES:
            raise ValueError("The corrected export does not name its exact five canonical files.")
        for name, row in receipt["files"].items():
            item = destination / name
            digest = row.get("sha256")
            if (Path(row["path"]).resolve() != item.resolve() or not item.is_file()
                    or type(row.get("size_bytes")) is not int or row["size_bytes"] < 0
                    or item.stat().st_size != row["size_bytes"]
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError("A corrected export file is unavailable or belongs elsewhere.")
        setup_bytes = _small(destination / "native_setup.toml")
        if hashlib.sha256(setup_bytes).hexdigest() != receipt["files"]["native_setup.toml"]["sha256"]:
            raise ValueError("The selected run's native Setup text changed.")
        text = setup_bytes.decode("utf-8")
        if tomllib.loads(text) != binding["replacements"]:
            raise ValueError("The selected run's native Setup text names other input paths.")
        if _small(path) != current_bytes:
            raise RuntimeError("The workspace revision changed while opening its selected export.")
        export = CorrectedSetupExport(revision, reference.run_id, destination, text, path, current_bytes, receipt, True)
        workspace._show_export_dialog(export)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        window.print_log(f"Could not open this run's corrected export: {error}")
    return True
