"""Drop-in reporting subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


def build(parent_page: Any) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    quality_title = QLabel("Current quality gates")
    quality_title.setObjectName("sectionTitle")
    layout.addWidget(quality_title)
    parent_page.reporting_quality = QTableWidget(0, 4)
    parent_page.reporting_quality.setHorizontalHeaderLabels(
        ["Module", "Status", "Reviewed by", "Reason"]
    )
    parent_page.reporting_quality.verticalHeader().setVisible(False)
    parent_page.reporting_quality.setEditTriggers(
        QAbstractItemView.EditTrigger.NoEditTriggers
    )
    parent_page.reporting_quality.horizontalHeader().setSectionResizeMode(
        0, QHeaderView.ResizeMode.ResizeToContents
    )
    parent_page.reporting_quality.horizontalHeader().setSectionResizeMode(
        1, QHeaderView.ResizeMode.ResizeToContents
    )
    parent_page.reporting_quality.horizontalHeader().setSectionResizeMode(
        2, QHeaderView.ResizeMode.ResizeToContents
    )
    parent_page.reporting_quality.horizontalHeader().setSectionResizeMode(
        3, QHeaderView.ResizeMode.Stretch
    )
    parent_page.reporting_quality.setMaximumHeight(210)
    layout.addWidget(parent_page.reporting_quality)

    uncertainty_title = QLabel("Uncertainty register")
    uncertainty_title.setObjectName("sectionTitle")
    layout.addWidget(uncertainty_title)
    parent_page.reporting_uncertainties = QTableWidget(0, 6)
    parent_page.reporting_uncertainties.setHorizontalHeaderLabels(
        [
            "Category",
            "Status",
            "Scientific statement",
            "Effect on results",
            "Mitigation or evidence",
            "Reviewed by",
        ]
    )
    parent_page.reporting_uncertainties.verticalHeader().setVisible(False)
    for column in (0, 1, 5):
        parent_page.reporting_uncertainties.horizontalHeader().setSectionResizeMode(
            column, QHeaderView.ResizeMode.ResizeToContents
        )
    for column in (2, 3, 4):
        parent_page.reporting_uncertainties.horizontalHeader().setSectionResizeMode(
            column, QHeaderView.ResizeMode.Stretch
        )
    parent_page.reporting_uncertainties.setMinimumHeight(260)
    layout.addWidget(parent_page.reporting_uncertainties)

    review_note = QLabel(
        "Generation-ready means the selected artifacts are reproducible and "
        "available. The status summary also shows whether uncertainties and "
        "relevant module gates have been reviewed."
    )
    review_note.setObjectName("secondaryText")
    review_note.setWordWrap(True)
    layout.addWidget(review_note)
    layout.addStretch(1)
    return tab
