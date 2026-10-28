"""Drop-in geometry-and-facies subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
        QAbstractItemView,
        QHBoxLayout,
        QHeaderView,
        QPushButton,
        QTableWidget,
        QVBoxLayout,
        QWidget,
)


def build(parent_page: Any) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        parent_page.layers = QTableWidget(0, 7)
        parent_page.layers.setHorizontalHeaderLabels(
            ["Use", "Label", "Facies", "Name", "Role", "Porosity", "Permeability"]
        )
        parent_page.layers.verticalHeader().setVisible(False)
        parent_page.layers.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = parent_page.layers.horizontalHeader()
        for column in (0, 1, 2, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(parent_page.layers, 1)
        layer_actions = QHBoxLayout()
        apply_facies = QPushButton("Apply selected properties to same facies")
        apply_facies.clicked.connect(parent_page._apply_properties_to_facies)
        layer_actions.addWidget(apply_facies)
        layer_actions.addStretch(1)
        layout.addLayout(layer_actions)
        return tab
