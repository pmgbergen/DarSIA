"""Shared Qt controls, formatting helpers, and quality-gate page base."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.modules import (
    MODULE_BY_KEY,
    dependencies_satisfied,
)
from darsia.fluidflower.project import FluidFlowerProject

FILE_DIALOG_OPTIONS = QFileDialog.Option.DontUseNativeDialog
DIRECTORY_DIALOG_OPTIONS = (
    QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog
)

STATUS_LABELS = {
    "not_started": "Not started",
    "ready": "Ready",
    "in_progress": "In progress",
    "review_required": "Review required",
    "approved": "Approved",
    "complete": "Complete",
    "stale": "Outdated",
    "blocked": "Blocked",
}

STATUS_COLORS = {
    "not_started": "#8f9aa3",
    "ready": "#73b9e6",
    "in_progress": "#e3b65c",
    "review_required": "#e59a55",
    "approved": "#66c69a",
    "complete": "#66c69a",
    "stale": "#e27979",
    "blocked": "#d36b6b",
}

INJECTOR_RECORD_ROLE = int(Qt.ItemDataRole.UserRole) + 1

STATE_LABEL_OBJECTS = {
    "saved": "savedState",
    "pending": "pendingState",
    "incomplete": "incompleteState",
}


def set_state_label(label: QLabel, state: str, text: str) -> None:
    """Set text and refresh the visual style of a persistent-state label."""

    label.setObjectName(STATE_LABEL_OBJECTS[state])
    label.setText(text)
    label.style().unpolish(label)
    label.style().polish(label)


class PathField(QWidget):
    """One path editor with a native Qt chooser."""

    changed = Signal(str)

    def __init__(
        self,
        *,
        directory: bool,
        file_filter: str = "All files (*)",
        placeholder: str = "Optional",
    ) -> None:
        super().__init__()
        self.directory = directory
        self.file_filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed)
        layout.addWidget(self.edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        layout.addWidget(browse)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:  # noqa: N802 - Qt naming convention
        self.edit.setText(value)
        self.edit.setToolTip(value)
        self.edit.setCursorPosition(0)

    def _browse(self) -> None:
        initial = self.text() or str(Path.home())
        if self.directory:
            selected = QFileDialog.getExistingDirectory(
                self,
                "Choose folder",
                initial,
                options=DIRECTORY_DIALOG_OPTIONS,
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Choose file",
                initial,
                self.file_filter,
                options=FILE_DIALOG_OPTIONS,
            )
        if selected:
            self.setText(selected)


class ModulePage(QWidget):
    status_changed = Signal()

    def __init__(self, project: FluidFlowerProject, module_key: str) -> None:
        super().__init__()
        self.project = project
        self.module_key = module_key
        self.definition = MODULE_BY_KEY[module_key]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        title = QLabel(self.definition.title)
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        description = QLabel(self.definition.description)
        description.setWordWrap(True)
        description.setObjectName("secondaryText")
        layout.addWidget(description)
        self.work_area = QWidget()
        self.work_layout = QVBoxLayout(self.work_area)
        self.work_layout.setContentsMargins(0, 0, 0, 0)
        self.work_layout.setSpacing(10)
        self.work_area.setVisible(False)
        layout.addWidget(self.work_area, 1)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)
        self.status = QLabel()
        self.status.setObjectName("moduleStatus")
        layout.addWidget(self.status)
        self.dependencies = QLabel()
        self.dependencies.setWordWrap(True)
        layout.addWidget(self.dependencies)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Reason or review note")
        layout.addWidget(self.reason)
        actions = QHBoxLayout()
        self.start = QPushButton("Start work")
        self.start.clicked.connect(lambda: self._set_status("in_progress"))
        actions.addWidget(self.start)
        self.review = QPushButton("Request review")
        self.review.clicked.connect(lambda: self._set_status("review_required"))
        actions.addWidget(self.review)
        self.approve = QPushButton("Approve quality gate")
        self.approve.clicked.connect(lambda: self._set_status("approved"))
        actions.addWidget(self.approve)
        self.complete = QPushButton("Mark complete")
        self.complete.clicked.connect(lambda: self._set_status("complete"))
        actions.addWidget(self.complete)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.integration = QLabel(self._integration_text())
        self.integration.setObjectName("integrationNotice")
        self.integration.setWordWrap(True)
        layout.addWidget(self.integration)
        layout.addStretch(1)
        self._refresh_gate()

    def _integration_text(self) -> str:
        if self.module_key == "setup":
            return (
                "Project manifest, source registry, rig template, protocol source, "
                "and discovered overflow candidates are active in this first "
                "application sequence. The overflow source is selected after data "
                "integrity has been reviewed."
            )
        return (
            "This module has a stable project boundary and quality gate. Its "
            "dedicated Qt work surface is connected in the next implementation "
            "sequences."
        )

    def _refresh_gate(self) -> None:
        state = self.project.module_states[self.module_key]
        status = str(state.get("status") or "not_started")
        color = STATUS_COLORS.get(status, "#d7dce0")
        self.status.setText(
            f"Current status: <span style='color:{color}; font-weight:600'>"
            f"{STATUS_LABELS.get(status, status)}</span>"
        )
        if self.definition.dependencies:
            parts = []
            for key in self.definition.dependencies:
                dependency_status = str(
                    self.project.module_states[key].get("status") or "not_started"
                )
                parts.append(
                    f"{MODULE_BY_KEY[key].title}: "
                    f"{STATUS_LABELS.get(dependency_status, dependency_status)}"
                )
            self.dependencies.setText("Required quality gates: " + " | ".join(parts))
        else:
            self.dependencies.setText("No upstream quality gates.")
        available = dependencies_satisfied(self.module_key, self.project.module_states)
        self.start.setEnabled(available)
        self.review.setEnabled(status in {"in_progress", "ready", "stale"})
        self.approve.setEnabled(
            status in {"review_required", "in_progress", "ready", "stale"}
        )
        self.complete.setEnabled(
            status in {"approved", "review_required", "in_progress"}
        )

    def refresh(self) -> None:
        self._refresh_gate()

    def _set_status(self, status: str) -> None:
        try:
            invalidated = self.project.set_module_status(
                self.module_key,
                status,
                reason=self.reason.text().strip(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not update quality gate", str(exc))
            return
        if invalidated:
            QMessageBox.information(
                self,
                "Downstream results are outdated",
                "These modules were marked outdated:\n"
                + "\n".join(MODULE_BY_KEY[key].title for key in invalidated),
            )
        self.reason.clear()
        self.refresh()
        self.status_changed.emit()


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "Not set"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def json_pretty(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)


def _gui_float(value: Any) -> float | None:
    from darsia.fluidflower.analysis_domains import _number
    return _number(value)


def _gui_int(value: Any) -> int | None:
    number = _gui_float(value)
    return int(number) if number is not None and float(number).is_integer() else None


def _table_number(value: Any) -> str:
    number = _gui_float(value)
    return "" if number is None else f"{number:.17g}"


def _percent(value: Any) -> str:
    number = _gui_float(value)
    return "Not available" if number is None else f"{100.0 * number:.2f}%"


def _millimetres(value: Any) -> str:
    number = _gui_float(value)
    return "n/a" if number is None else f"{1000.0 * number:.3f} mm"


def deepcopy_settings(value: Any) -> dict[str, Any]:
    from copy import deepcopy

    return deepcopy(dict(value))
