"""Drop-in geometry-and-facies subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.geometry_facies import RUN_GEOMETRY_POLICIES


def build(parent_page: Any) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        note = QLabel(
            "Choose whether one approved geometry applies to the series or reviewed "
            "layer maps may be selected per run. The change screen compares registered "
            "layer boundaries and only raises an advisory warning. It never changes "
            "geometry, porosity, or permeability automatically."
        )
        note.setObjectName("secondaryText")
        note.setWordWrap(True)
        layout.addWidget(note)
        policy_form = QFormLayout()
        parent_page.run_geometry_policy = QComboBox()
        for label, value in (
            ("Use one approved series geometry", "series_reference_only"),
            ("Allow reviewed geometry per run", "allow_reviewed_run_overrides"),
            ("Require reviewed geometry per run", "require_reviewed_run_overrides"),
        ):
            if value in RUN_GEOMETRY_POLICIES:
                parent_page.run_geometry_policy.addItem(label, value)
        parent_page.run_geometry_policy.currentIndexChanged.connect(parent_page._mark_dirty)
        policy_form.addRow("Geometry policy", parent_page.run_geometry_policy)
        detection_row = QWidget()
        detection_layout = QHBoxLayout(detection_row)
        detection_layout.setContentsMargins(0, 0, 0, 0)
        parent_page.run_geometry_change_detection = QCheckBox(
            "Warn about substantial registered boundary changes"
        )
        parent_page.run_geometry_change_detection.setChecked(True)
        parent_page.run_geometry_change_detection.toggled.connect(parent_page._mark_dirty)
        detection_layout.addWidget(parent_page.run_geometry_change_detection)
        detection_layout.addWidget(QLabel("Displacement"))
        parent_page.run_geometry_displacement_threshold = QDoubleSpinBox()
        parent_page.run_geometry_displacement_threshold.setRange(0.01, 25.0)
        parent_page.run_geometry_displacement_threshold.setDecimals(2)
        parent_page.run_geometry_displacement_threshold.setValue(0.5)
        parent_page.run_geometry_displacement_threshold.setSuffix(" % of height")
        parent_page.run_geometry_displacement_threshold.valueChanged.connect(parent_page._mark_dirty)
        detection_layout.addWidget(parent_page.run_geometry_displacement_threshold)
        detection_layout.addWidget(QLabel("Affected boundary"))
        parent_page.run_geometry_affected_threshold = QDoubleSpinBox()
        parent_page.run_geometry_affected_threshold.setRange(0.0, 100.0)
        parent_page.run_geometry_affected_threshold.setDecimals(1)
        parent_page.run_geometry_affected_threshold.setValue(10.0)
        parent_page.run_geometry_affected_threshold.setSuffix(" %")
        parent_page.run_geometry_affected_threshold.valueChanged.connect(parent_page._mark_dirty)
        detection_layout.addWidget(parent_page.run_geometry_affected_threshold)
        detection_layout.addStretch(1)
        policy_form.addRow("Advisory screen", detection_row)
        layout.addLayout(policy_form)
        parent_page.run_geometry_table = QTableWidget(0, 9)
        parent_page.run_geometry_table.setHorizontalHeaderLabels(
            [
                "Include",
                "Run",
                "Acquisition",
                "Authoritative geometry",
                "Registered source image",
                "Layer labels",
                "Facies map",
                "Boundary-change screen",
                "Physical properties",
            ]
        )
        parent_page.run_geometry_table.verticalHeader().setVisible(False)
        parent_page.run_geometry_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        header = parent_page.run_geometry_table.horizontalHeader()
        for column in (0, 1, 2, 3, 7, 8):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for column in (4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        parent_page.run_geometry_table.cellDoubleClicked.connect(
            parent_page._browse_run_geometry_cell
        )
        layout.addWidget(parent_page.run_geometry_table, 1)
        actions = QHBoxLayout()
        add_runs = QPushButton("Add project runs")
        add_runs.clicked.connect(parent_page._add_project_run_geometry_rows)
        actions.addWidget(add_runs)
        add_custom = QPushButton("Add run")
        add_custom.clicked.connect(parent_page._add_custom_run_geometry_row)
        actions.addWidget(add_custom)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(parent_page._remove_run_geometry_rows)
        actions.addWidget(remove)
        clear = QPushButton("Use series map for selected")
        clear.clicked.connect(parent_page._clear_selected_run_geometry_maps)
        actions.addWidget(clear)
        preview = QPushButton("Preview selected overlay")
        preview.clicked.connect(parent_page._preview_selected_run_geometry)
        actions.addWidget(preview)
        actions.addStretch(1)
        layout.addLayout(actions)
        path_note = QLabel(
            "Double-click an image, layer-label, or facies-map cell to choose a file. "
            "A run map may be attached for change screening without selecting it as "
            "authoritative. Clearwater is preferred; indicator-filled sources receive "
            "a carryover warning."
        )
        path_note.setObjectName("secondaryText")
        path_note.setWordWrap(True)
        layout.addWidget(path_note)
        return tab
