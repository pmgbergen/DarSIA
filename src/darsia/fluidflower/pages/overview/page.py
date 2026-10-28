"""Qt workflow-overview page."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.modules import MODULE_BY_KEY, MODULE_DEFINITIONS
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.ui.common import STATUS_COLORS, STATUS_LABELS


class OverviewPage(QWidget):
    module_requested = Signal(str)

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__()
        self.project = project
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        title = QLabel("Workflow overview")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        self.summary = QLabel()
        self.summary.setObjectName("secondaryText")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(len(MODULE_DEFINITIONS), 4)
        self.table.setHorizontalHeaderLabels(
            ["Module", "Quality gate", "Dependencies", "Last change"]
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._open_module)
        layout.addWidget(self.table, 1)
        self.refresh()

    def refresh(self) -> None:
        source_count = len(self.project.source_manifest().get("sources") or [])
        rig = self.project.manifest.get("rig") or {}
        template = self.project.manifest.get("template") or {}
        self.summary.setText(
            f"{self.project.name} | {rig.get('type_label', '')} | "
            f"template {template.get('name', '')} | {source_count} registered sources"
        )
        for row, definition in enumerate(MODULE_DEFINITIONS):
            state = self.project.module_states[definition.key]
            status = str(state.get("status") or "not_started")
            module_item = QTableWidgetItem(definition.title)
            module_item.setData(Qt.ItemDataRole.UserRole, definition.key)
            status_item = QTableWidgetItem(STATUS_LABELS.get(status, status))
            status_item.setForeground(QColor(STATUS_COLORS.get(status, "#d7dce0")))
            dependencies = (
                ", ".join(MODULE_BY_KEY[key].title for key in definition.dependencies)
                or "None"
            )
            self.table.setItem(row, 0, module_item)
            self.table.setItem(row, 1, status_item)
            self.table.setItem(row, 2, QTableWidgetItem(dependencies))
            self.table.setItem(
                row,
                3,
                QTableWidgetItem(str(state.get("updated_utc") or "")),
            )

    def _open_module(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.module_requested.emit(str(item.data(Qt.ItemDataRole.UserRole)))
