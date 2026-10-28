"""Version actions for the painting page; immutable content and reversible moves."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QHBoxLayout, QInputDialog, QMenu, QMessageBox, QPushButton

from . import layer_painting as lp
from . import layer_painting_manual as manual
from . import layer_painting_polygons_only as polygons
from . import layer_painting_version_archive as versions


class PaintingVersionsActions(QObject):
    """The drawing and metadata actions beside the existing version workflow."""

    def __init__(self, view, layout):
        super().__init__(view)
        self.view = view
        self.menu = None
        self.buttons = {}
        for choices in (
            (("save_drawing", "Save drawing…", self.save_drawing),
             ("load_drawing", "Load drawing…", self.load_drawing)),
            (("rename", "Rename…", self.rename), ("note", "Note…", self.note)),
            (("withdraw", "Withdraw approval…", self.withdraw),
             ("archive", "Delete version…", self.archive),
             ("restore_deleted", "Restore deleted version…", self.restore_deleted)),
        ):
            row = QHBoxLayout()
            for key, text, callback in choices:
                button = QPushButton(text)
                button.clicked.connect(callback)
                row.addWidget(button)
                self.buttons[key] = button
            row.addStretch(1)
            layout.addLayout(row)
        view.versions_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.versions_table.customContextMenuRequested.connect(self.context_menu)

    def available(self):
        chooser = getattr(self.view, "run_preview", None)
        return (self.view.session is not None and not self.view.busy()
                and (chooser is None or chooser.is_reference()))

    def refresh(self):
        available = self.available()
        selected = self.view._selected_version()
        for key, button in self.buttons.items():
            button.setEnabled(available and (
                key in {"save_drawing", "load_drawing", "restore_deleted"} or selected is not None
            ) and (key in {"save_drawing", "load_drawing"} or self.history_available())
                and (key != "withdraw" or self.view._approval_for_version(selected) is not None))

    def history_available(self):
        return not (getattr(self.view, "_map_approvals_error", "")
                    or any(row.get("history_error") for row in self.view.versions))

    def reload(self, message=""):
        view = self.view
        if view.session is not None:
            view.versions = view._read_versions(view.session.run_id)
            view._read_approvals(view.session.run_id)
        view._refresh_versions()
        dialog = getattr(view, "_placement_dialog", None)
        if dialog is not None and hasattr(dialog, "refresh_approval_state"):
            dialog.refresh_approval_state(view.map_approvals)
        if message:
            view.versions_note.setText(message)

    def _failure(self, error):
        self.view.versions_note.setText(str(error))

    def _record(self):
        if not self.history_available():
            self._failure("Version history is unreadable; no approval or history event was changed.")
            return None
        return self.view._selected_version() if self.available() else None

    @staticmethod
    def name(record):
        return str(record.get("display_name", record.get("name", "")))

    def save_drawing(self):
        view = self.view
        if not self.available():
            return None
        name, accepted = QInputDialog.getText(view, "Save drawing", "Drawing name:")
        if not accepted or not name.strip():
            return None
        try:
            row = versions.save_drawing_version(view.project, view.session, name=name)
        except (OSError, ValueError, KeyError) as error:
            self._failure(error)
            return None
        self.reload(f"Saved drawing {name}: SHA-256 {row['sha256']}")
        return row

    def use_drawing(self, record):
        view = self.view
        if not self.available():
            return False
        current = view.session
        try:
            from .painting_run_preview import _same_path
            loaded = versions.load_drawing_version(view.project, current.run_id, record)
            if (tuple(loaded.image_size or ()) != tuple(current.image_size or ())
                    or not _same_path(loaded.image_path, current.image_path)
                    or loaded.image_stamp != current.image_stamp):
                raise ValueError("This drawing belongs to another reference photograph; it was not moved.")
            loaded, _backup = versions.use_drawing_version(view.project, current, record)
            polygons.save_current_session(view.project, loaded)
        except (OSError, ValueError, KeyError) as error:
            self._failure(error)
            return False
        view._dirty = False
        view.session = loaded
        view.reload()
        self.reload(f"Loaded drawing {self.name(record)}; the previous drawing was saved first.")
        return True

    def load_drawing(self):
        view = self.view
        if not self.available():
            return False
        rows = [row for row in view.versions if row.get("kind") == versions.DRAWING_KIND]
        if not rows:
            self._failure("No drawing versions have been saved for this reference run.")
            return False
        labels = [f"{self.name(row)} · {row['sha256'][:12]} · {Path(row['folder']).name}" for row in rows]
        choice, accepted = QInputDialog.getItem(view, "Load drawing", "Drawing:", labels, 0, False)
        if not accepted:
            return False
        return self.use_drawing(rows[labels.index(choice)])

    def rename(self):
        view, record = self.view, self._record()
        if record is None:
            return
        name, accepted = QInputDialog.getText(view, "Rename version", "Display name:", text=self.name(record))
        if not accepted:
            return
        try:
            versions.rename_version(view.project, view.session.run_id, record, name)
        except (OSError, ValueError, KeyError) as error:
            self._failure(error)
            return
        self.reload("The display name changed; the content SHA and approval are unchanged.")

    def note(self):
        view, record = self.view, self._record()
        if record is None:
            return
        text, accepted = QInputDialog.getMultiLineText(view, "Version note", "Note:", str(record.get("note", "")))
        if not accepted:
            return
        try:
            versions.set_version_note(view.project, view.session.run_id, record, text)
        except (OSError, ValueError, KeyError) as error:
            self._failure(error)
            return
        self.reload("The note was recorded; the content SHA and approval are unchanged.")

    def withdraw(self):
        """Confirm once, then append the withdrawal through the existing log writer."""
        view, record = self.view, self._record()
        if record is None:
            return False
        try:
            context = versions.withdrawal_context(view.project, view.session.run_id, record)
            placement = ("\nIts logged placement(s) will be marked approval withdrawn; their saved evidence stays."
                         if context["placements"] else "")
            answer = QMessageBox.question(view, "Withdraw approval",
                f"Withdraw approval of {self.name(record)} ({record['kind']})?\n"
                f"SHA-256 {record['sha256']}" + placement
                + "\nThe approval history and saved version remain unchanged; a withdrawal is appended.")
            if answer != QMessageBox.StandardButton.Yes:
                return False
            versions.withdraw_approval(view.project, view.session.run_id, record)
        except (OSError, ValueError, KeyError) as error:
            self._failure(error)
            return False
        self.reload("Approval withdrawn. The saved version and earlier approval history are retained.")
        return True

    def archive(self):
        view, record = self.view, self._record()
        if record is None:
            return False
        try:
            blockers = versions.archive_blockers(view.project, view.session.run_id, record)
            if blockers:
                raise ValueError("This version cannot be deleted: " + "; ".join(map(str, blockers)))
            answer = QMessageBox.question(
                view, "Delete version",
                f"Delete {self.name(record)} ({record['kind']}) from the list?\n"
                f"SHA-256 {record['sha256']}\nIts files move to the deleted folder and can be restored.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
            versions.archive_version(view.project, view.session.run_id, record)
        except (OSError, ValueError, KeyError) as error:
            self._failure(error)
            return False
        self.reload("The version moved to the deleted folder. No files were removed.")
        return True

    def restore_deleted(self):
        view = self.view
        if not self.available() or not self.history_available():
            if not self.history_available():
                self._failure("Version history is unreadable; no deleted version was moved.")
            return False
        try:
            rows = versions.list_version_metadata(view.project, view.session.run_id, deleted=True)
            if not rows:
                self._failure("There are no deleted versions for this reference run.")
                return False
            labels = [f"{self.name(row)} · {row['kind']} · {row['sha256'][:12]} · {Path(row['folder']).name}" for row in rows]
            choice, accepted = QInputDialog.getItem(view, "Restore deleted version", "Version:", labels, 0, False)
            if not accepted:
                return False
            versions.restore_deleted_version(view.project, view.session.run_id, rows[labels.index(choice)])
        except (OSError, ValueError, KeyError) as error:
            self._failure(error)
            return False
        self.reload("The deleted version was restored with the same content SHA.")
        return True

    def context_menu(self, position):
        view = self.view
        index = view.versions_table.indexAt(position)
        if index.isValid():
            view.versions_table.selectRow(index.row())
        menu = QMenu(view)
        for text, callback in (
            ("Use this version", view._restore_version_clicked),
            ("Rename…", self.rename), ("Note…", self.note),
            ("Withdraw approval…", self.withdraw),
            ("Delete version…", self.archive),
            ("Restore deleted version…", self.restore_deleted),
        ):
            action = menu.addAction(text)
            action.setEnabled(self.available() and (
                text == "Restore deleted version…" or view._selected_version() is not None)
                and (text == "Use this version" or self.history_available())
                and (text != "Withdraw approval…" or self.view._approval_for_version(view._selected_version()) is not None))
            action.triggered.connect(callback)
        self.menu = menu
        menu.popup(view.versions_table.viewport().mapToGlobal(position))


def settings_text(record):
    """A compact table summary; full settings remain available in the tooltip."""
    method = record.get("proposal_method")
    if record.get("kind") == versions.DRAWING_KIND:
        return "drawing only"
    if record.get("kind") == manual.MANUAL_MAP_KIND:
        return "as drawn"
    values = record.get("settings", {})
    if method == polygons.METHOD:
        return "Polygons only"
    fields = []
    for key in ("smoothing", "min_region_cm2", "snap_mm", "detail"):
        if key in values:
            fields.append(f"{key.replace('_', ' ')}: {values[key]}")
    return " · ".join(fields) or str(method or "Classifier")
