"""Drop-in geometry-and-facies subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPushButton,
        QTableWidget,
        QVBoxLayout,
        QWidget,
)


def build(parent_page: Any) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        history_row = QHBoxLayout()
        parent_page.revisions = QComboBox()
        history_row.addWidget(parent_page.revisions, 1)
        load = QPushButton("Load selected")
        load.clicked.connect(parent_page._load_selected_revision)
        history_row.addWidget(load)
        layout.addLayout(history_row)
        parent_page.validation_details = QLabel("No validation report.")
        parent_page.validation_details.setWordWrap(True)
        parent_page.validation_details.setObjectName("secondaryText")
        layout.addWidget(parent_page.validation_details)
        parent_page.issues = QTableWidget(0, 3)
        parent_page.issues.setHorizontalHeaderLabels(["Severity", "Finding", "Details"])
        parent_page.issues.verticalHeader().setVisible(False)
        parent_page.issues.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        issue_header = parent_page.issues.horizontalHeader()
        issue_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        issue_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        issue_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(parent_page.issues, 1)
        return tab
