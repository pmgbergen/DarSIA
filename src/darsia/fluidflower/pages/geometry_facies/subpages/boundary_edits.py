"""Drop-in geometry-and-facies subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
        QAbstractItemView,
        QButtonGroup,
        QComboBox,
        QDoubleSpinBox,
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

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target layer"))
        parent_page.edit_target = QComboBox()
        parent_page.edit_target.setMinimumWidth(220)
        target_row.addWidget(parent_page.edit_target, 1)
        parent_page.add_edit_label = QPushButton("New layer")
        parent_page.add_edit_label.clicked.connect(parent_page._add_edit_target)
        target_row.addWidget(parent_page.add_edit_label)
        layout.addLayout(target_row)

        mode_row = QHBoxLayout()
        parent_page.boundary_modes = QButtonGroup(parent_page)
        parent_page.boundary_modes.setExclusive(True)
        parent_page.pan_boundaries = QPushButton("Pan")
        parent_page.polygon_boundaries = QPushButton("Polygon")
        parent_page.brush_boundaries = QPushButton("Brush")
        parent_page.pan_boundaries.setToolTip("Pan the image without changing a layer.")
        parent_page.polygon_boundaries.setToolTip(
            "Click vertices, then press Enter or double-click to apply the polygon."
        )
        parent_page.brush_boundaries.setToolTip(
            "Drag over the image with the selected physical brush radius."
        )
        for button, mode in (
            (parent_page.pan_boundaries, "pan"),
            (parent_page.polygon_boundaries, "polygon"),
            (parent_page.brush_boundaries, "brush"),
        ):
            button.setCheckable(True)
            button.toggled.connect(
                lambda checked, selected=mode: (
                    checked and parent_page._boundary_mode_changed(selected)
                )
            )
            parent_page.boundary_modes.addButton(button)
            mode_row.addWidget(button)
        parent_page.pan_boundaries.setChecked(True)
        mode_row.addWidget(QLabel("Radius"))
        parent_page.brush_radius_mm = QDoubleSpinBox()
        parent_page.brush_radius_mm.setRange(0.1, 100.0)
        parent_page.brush_radius_mm.setDecimals(2)
        parent_page.brush_radius_mm.setValue(2.0)
        parent_page.brush_radius_mm.setSuffix(" mm")
        parent_page.brush_radius_mm.setMaximumWidth(110)
        mode_row.addWidget(parent_page.brush_radius_mm)
        layout.addLayout(mode_row)

        edit_actions = QHBoxLayout()
        parent_page.finish_polygon = QPushButton("Finish polygon")
        parent_page.finish_polygon.clicked.connect(parent_page.canvas.finish_edit)
        parent_page.finish_polygon.setToolTip(
            "Finish the current polygon (Enter or double-click)."
        )
        edit_actions.addWidget(parent_page.finish_polygon)
        cancel = QPushButton("Cancel stroke")
        cancel.clicked.connect(parent_page.canvas.cancel_edit)
        cancel.setToolTip("Discard the unfinished polygon or brush stroke (Escape).")
        edit_actions.addWidget(cancel)
        parent_page.undo_label_edit = QPushButton("Undo")
        parent_page.undo_label_edit.clicked.connect(parent_page._undo_label_edit)
        edit_actions.addWidget(parent_page.undo_label_edit)
        parent_page.redo_label_edit = QPushButton("Redo")
        parent_page.redo_label_edit.clicked.connect(parent_page._redo_label_edit)
        edit_actions.addWidget(parent_page.redo_label_edit)
        edit_actions.addStretch(1)
        layout.addLayout(edit_actions)

        parent_page.materialize_label_edits = QPushButton("Materialize revised layer map")
        parent_page.materialize_label_edits.clicked.connect(parent_page._materialize_label_edits)
        parent_page.materialize_label_edits.setEnabled(False)
        layout.addWidget(parent_page.materialize_label_edits)
        parent_page.label_edit_status = QLabel(
            "No boundary edits. Imported label maps remain immutable."
        )
        parent_page.label_edit_status.setObjectName("secondaryText")
        parent_page.label_edit_status.setWordWrap(True)
        layout.addWidget(parent_page.label_edit_status)

        parent_page.label_edit_history = QTableWidget(0, 4)
        parent_page.label_edit_history.setHorizontalHeaderLabels(
            ["Time", "Action", "Target", "Pixels"]
        )
        parent_page.label_edit_history.verticalHeader().setVisible(False)
        parent_page.label_edit_history.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        history_header = parent_page.label_edit_history.horizontalHeader()
        history_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        history_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        history_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        history_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(parent_page.label_edit_history, 1)
        parent_page._boundary_mode_changed("pan")
        parent_page._refresh_label_edit_controls()
        return tab
