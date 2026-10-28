"""Where this computer opens each shared storage root (a per-user setting).

Shared records name files on the network share by the share's own path, for
example ``\\\\server\\share\\<series>\\...``. A computer that opens the share
elsewhere, such as ``/Volumes/share`` on a Mac, keeps one row per shared root
here. The GUI installs these rows at start (:func:`install_machine_path_map`);
reading then finds stored paths below this computer's folder, and paths saved
below that folder are written with the shared root, so other computers read
them (``darsia.fluidflower.storage_paths``). Workers never read this setting,
and without rows nothing changes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterable
from urllib.parse import unquote

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import storage_paths

SETTINGS = ("DarSIA", "Storage locations")
KEY = "machine_path_map"
ACTION_TEXT = "Shared storage on this computer…"
_MOUNT = re.compile(
    r"^//(?:[^@/]*@)?(?P<host>[^/]+)/(?P<share>[^ ]+(?: [^ ]+)*?) on (?P<folder>/.+?) "
    r"\((?P<kind>smbfs|cifs)[,)]"
)


def _settings(settings: QSettings | None) -> QSettings:
    return QSettings(*SETTINGS) if settings is None else settings


def read_machine_path_map(settings: QSettings | None = None) -> tuple[tuple[str, str], ...]:
    """Return the saved (shared root, folder on this computer) rows.

    Rows that cannot be read or fail
    :func:`~darsia.fluidflower.storage_paths.validate_machine_path_map` give
    an empty map, so a damaged setting never redirects a read.
    """
    text = _settings(settings).value(KEY, "", type=str)
    if not text:
        return ()
    try:
        rows = json.loads(text)
        return storage_paths.validate_machine_path_map(
            (str(canonical), str(local)) for canonical, local in rows)
    except (TypeError, ValueError):
        return ()


def save_machine_path_map(rows: Iterable, settings: QSettings | None = None) -> tuple[tuple[str, str], ...]:
    """Check, save and install the rows; return them as saved.

    Raises ValueError (and saves nothing) when a row is incomplete or
    inconsistent. An empty sequence removes the setting's rows.
    """
    checked = storage_paths.validate_machine_path_map(rows)
    target = _settings(settings)
    target.setValue(KEY, json.dumps([list(row) for row in checked], ensure_ascii=False))
    target.sync()
    return install_machine_path_map(target)


def install_machine_path_map(settings: QSettings | None = None) -> tuple[tuple[str, str], ...]:
    """Install the saved rows for this process and refresh cached locations.

    Called once by each GUI at start and after a change. Returns the rows
    installed (empty when none are saved).
    """
    rows = read_machine_path_map(settings)
    if rows or storage_paths.machine_path_map():
        storage_paths.set_machine_path_map(rows)
        from .review_storage import review_aliases

        try:
            review_aliases(refresh=True)
        except (OSError, ValueError):
            pass  # A series that cannot be read yet reports itself when opened.
    return rows


def mounted_shares(mount_output: str) -> list[tuple[str, str]]:
    """Return (shared root, folder) rows for the SMB shares in ``mount`` output.

    Reads lines such as ``//user@server._smb._tcp.local/share on /Volumes/share
    (smbfs, ...)`` and proposes ``\\\\server\\share`` for ``/Volumes/share``.
    The proposal names the server as the system mounted it; the spelling the
    series' records use may differ and is chosen by the user.
    """
    rows = []
    for line in mount_output.splitlines():
        match = _MOUNT.match(line.strip())
        if match is None:
            continue
        host = re.sub(r"(?:\._smb\._tcp)?\.local\.?$", "", unquote(match["host"]), flags=re.IGNORECASE)
        share = unquote(match["share"])
        if host and share and "/" not in share:
            rows.append((f"\\\\{host}\\{share}", match["folder"]))
    return rows


def _system_mount_output() -> str:
    """The output of this computer's ``mount`` command ("" where unavailable)."""
    if sys.platform != "darwin":
        return ""
    try:
        return subprocess.run(["/sbin/mount"], capture_output=True, text=True, timeout=10,
                              check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


class MachinePathMapDialog(QDialog):
    """Edit where this computer opens each shared storage root."""

    def __init__(self, parent: QWidget | None = None, *, settings: QSettings | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("Shared storage on this computer")
        layout = QVBoxLayout(self)
        note = QLabel(
            "Saved records name files on the network share by the share's own path, "
            "for example \\\\server\\share\\... . Give the folder where this computer "
            "opens each share, for example /Volumes/share on a Mac. Files are then read "
            "from that folder, and paths saved below it are written with the shared "
            "path, so the other computers read them. Leave the list empty on a computer "
            "that opens the shared paths directly."
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(note)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Shared root", "Folder on this computer"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)
        buttons_row = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.remove_button = QPushButton("Remove")
        self.browse_button = QPushButton("Choose folder…")
        self.suggest_button = QPushButton("Add mounted shares")
        self.suggest_button.setToolTip("Propose one row for each network share this computer has mounted")
        self.suggest_button.setVisible(sys.platform == "darwin")
        for button in (self.add_button, self.remove_button, self.browse_button, self.suggest_button):
            buttons_row.addWidget(button)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)
        self.add_button.clicked.connect(lambda: self.add_row("", ""))
        self.remove_button.clicked.connect(self._remove_selected)
        self.browse_button.clicked.connect(self._choose_folder)
        self.suggest_button.clicked.connect(self._suggest)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for canonical, local in read_machine_path_map(settings):
            self.add_row(canonical, local)
        self.resize(760, 360)

    def add_row(self, canonical: str, local: str) -> None:
        """Append one editable row."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(canonical))
        self.table.setItem(row, 1, QTableWidgetItem(local))

    def rows(self) -> list[tuple[str, str]]:
        """The table's rows, without rows left completely empty."""
        result = []
        for row in range(self.table.rowCount()):
            values = tuple((self.table.item(row, column).text() if self.table.item(row, column) else "").strip()
                           for column in (0, 1))
            if any(values):
                result.append(values)
        return result

    def _remove_selected(self) -> None:
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def _choose_folder(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            self.add_row("", "")
            row = self.table.rowCount() - 1
        folder = QFileDialog.getExistingDirectory(self, "Folder on this computer", "/Volumes" if sys.platform == "darwin" else "")
        if folder:
            self.table.setItem(row, 1, QTableWidgetItem(folder))

    def _suggest(self) -> None:
        present = {canonical.casefold() for canonical, _local in self.rows()}
        for canonical, local in mounted_shares(_system_mount_output()):
            if canonical.casefold() not in present:
                self.add_row(canonical, local)

    def save(self) -> None:
        """Save and install the rows, or explain why they cannot be saved."""
        try:
            save_machine_path_map(self.rows(), self._settings)
        except ValueError as error:
            QMessageBox.warning(self, "Shared storage on this computer", str(error))
            return
        self.accept()


def add_machine_path_action(window) -> QAction:
    """Add the dialog to the window's Settings menu (or its menu bar)."""
    action = QAction(ACTION_TEXT, window)
    action.setMenuRole(QAction.MenuRole.NoRole)

    def open_dialog() -> None:
        if MachinePathMapDialog(window).exec() == QDialog.DialogCode.Accepted:
            refresh = getattr(window, "_refresh_series", None)
            if callable(refresh):
                refresh(show_errors=True)

    action.triggered.connect(open_dialog)
    menus = [entry.menu() for entry in window.menuBar().actions() if entry.menu() is not None]
    settings_menu = next((menu for menu in menus if menu.title().replace("&", "") == "Settings"), None)
    (settings_menu or window.menuBar()).addAction(action)
    return action
