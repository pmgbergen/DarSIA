"""Saved transfer chooser: bounded metadata only, read off the GUI thread."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QListWidget, QVBoxLayout

_LIMIT = 4 * 1024 * 1024
_TOTAL = 16 * 1024 * 1024
_FORMATS = {"darsia-fluidflower-map-placement-v1", "darsia-fluidflower-map-placement-v2"}


def _safe(path):
    """Require an absolute metadata path without linked or reparse ancestors."""
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Saved transfers need an absolute project path")
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if part.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Saved transfer metadata refuses linked or reparse paths")
    return path


def saved_transfers(root):
    """List display metadata; opening a row still uses the original verifier."""
    root = _safe(root)
    if not root.exists():
        return [], []
    children = list(root.iterdir())
    if len(children) > 4096:
        raise ValueError("Too many saved transfer folders to list safely")
    rows, errors, consumed = [], [], 0
    for folder in children:
        if QThread.currentThread().isInterruptionRequested():
            return [], ["Loading the saved transfer list was cancelled."]
        try:
            _safe(folder)
            if not folder.is_dir():
                continue
            path = _safe(folder / "index.json")
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                raw = stream.read(_LIMIT + 1)
            consumed += len(raw)
            if len(raw) > _LIMIT or consumed > _TOTAL:
                raise ValueError("Saved transfer metadata exceeds the bounded read")
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError("Repeated saved-transfer metadata key")
                    result[key] = value
                return result
            def invalid(_value):
                raise ValueError("Nonfinite saved-transfer metadata")
            value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)
            version = value.get("map_version") if isinstance(value, dict) else None
            if value.get("format") not in _FORMATS or not isinstance(version, dict):
                raise ValueError("This index is not a saved map transfer")
            if ("approval" in version and version["approval"] is not None
                    and not isinstance(version["approval"], dict)):
                raise ValueError("The saved transfer approval metadata must be an object")
            if (not isinstance(version.get("name"), str) or not version["name"] or len(version["name"]) > 256
                    or not isinstance(value.get("created_utc"), str) or not value["created_utc"]
                    or len(value["created_utc"]) > 64):
                raise ValueError("The saved transfer needs its bounded version name and timestamp")
            rows.append(dict(folder=str(folder), version=dict(version),
                created_utc=str(value.get("created_utc") or ""), reference_run=str(value.get("reference_run") or "")))
        except (OSError, ValueError, TypeError, AttributeError) as error:
            errors.append(f"{folder.name}: {error}")
            if consumed > _TOTAL:
                break
    return sorted(rows, key=lambda row: (row["created_utc"], row["folder"]), reverse=True), errors


class _Read(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, root):
        """Bind the project metadata root for one finite background read."""
        super().__init__()
        self.root = root

    @Slot()
    def run(self):
        """Read display metadata once and return the worker to the GUI thread."""
        try:
            self.finished.emit(saved_transfers(self.root))
        except Exception as error:
            self.failed.emit(f"The saved transfer list could not be read: {error}")
        finally:
            # The finished worker is destroyed in the GUI thread, after its
            # finite metadata thread has retired.
            self.moveToThread(QCoreApplication.instance().thread())


class SavedTransferChooser(QDialog):
    transfer_selected = Signal(str)
    loading_changed = Signal(bool)

    def __init__(self, root, approved_version=None, parent=None):
        """Build the saved-transfer list and start its finite metadata reader."""
        super().__init__(parent)
        self.setWindowTitle("Review a saved map transfer")
        self.resize(700, 420)
        self.rows = []
        self.preferred = dict(approved_version or {})
        if not isinstance(self.preferred.get("folder"), str):
            self.preferred = {}
        self._close_when_done = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose a saved transfer by map version and time. The summary is read only."))
        self.list = QListWidget()
        layout.addWidget(self.list, 1)
        self.status = QLabel("Loading saved transfers…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Close transfer list")
        self.open_button = buttons.addButton("Open transfer summary", QDialogButtonBox.ButtonRole.ActionRole)
        self.open_button.setEnabled(False)
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self._open)
        self.list.currentRowChanged.connect(lambda row: self.open_button.setEnabled(not self.busy() and row >= 0))
        self.list.itemClicked.connect(lambda _item: self._open())
        self.list.itemActivated.connect(lambda _item: self._open())
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._thread = QThread(self)
        self._worker = _Read(root)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._loaded)
        self._worker.failed.connect(self.status.setText)
        self._worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._worker.failed.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._thread.finished.connect(self._retired)
        self._thread.start()

    def busy(self):
        """Report whether the finite metadata reader still needs retirement."""
        return self._thread is not None

    def _loaded(self, result):
        """Display verified list metadata and prefer the selected approved identity."""
        self.rows, errors = result
        preferred = 0
        for index, row in enumerate(self.rows):
            version = row["version"]
            name = version.get("name") or "Saved map"
            when = row["created_utc"][:16].replace("T", " ")
            from .layer_painting import approval_withdrawal
            withdrawn = approval_withdrawal(getattr(self.parent(), "map_approvals", []), version.get("approval"))
            suffix = " · approval withdrawn " + withdrawn["recorded_utc"] if withdrawn else ""
            self.list.addItem(f"{name} – {when} UTC" + suffix)
            self.list.item(index).setToolTip(f"Map SHA-256: {version.get('sha256') or 'not recorded'}")
            if (self.preferred.get("folder") and version.get("sha256") == self.preferred.get("sha256")
                    and (version.get("folder") == self.preferred.get("folder") or
                         (version.get("approval") or {}).get("version") == Path(self.preferred["folder"]).name)):
                if not any((old["version"].get("folder") == self.preferred.get("folder") or
                            (old["version"].get("approval") or {}).get("version") == Path(self.preferred["folder"]).name)
                           and old["version"].get("sha256") == self.preferred.get("sha256") for old in self.rows[:index]):
                    preferred = index
        if self.rows:
            self.list.setCurrentRow(preferred)
        self.status.setText((f"{len(self.rows)} saved transfer(s)." if self.rows else "No saved transfers in this project yet.")
                            + (" Some entries are unavailable: " + "; ".join(errors[:4]) if errors else ""))

    def _retired(self):
        """Wait for native thread retirement before releasing its Qt objects."""
        if self._thread is None:
            return
        thread, worker = self._thread, self._worker
        thread.wait()
        self._thread = self._worker = None
        del worker
        thread.deleteLater()
        self.open_button.setEnabled(self.list.currentRow() >= 0)
        self.loading_changed.emit(False)
        if self._close_when_done:
            super().reject()

    def _open(self):
        """Emit one selected folder after metadata loading has retired."""
        row = self.list.currentRow()
        if self.busy() or row < 0 or getattr(self, "_selected_once", False):
            return
        self._selected_once = True
        self.transfer_selected.emit(self.rows[row]["folder"])
        self.accept()

    def wait_for_workers(self):
        """Wait for and retire the outstanding finite metadata reader."""
        if self._thread is not None:
            self._thread.wait()
            self._retired()

    def reject(self):
        """Defer closing until an interrupted metadata reader has retired."""
        if self.busy():
            self._close_when_done = True
            self._thread.requestInterruption()
            self.status.setText("Closing after the current metadata read finishes…")
            return
        super().reject()

    def closeEvent(self, event):
        """Keep the dialog alive while its finite reader is retiring."""
        if self.busy():
            self.reject()
            event.ignore()
        else:
            event.accept()
