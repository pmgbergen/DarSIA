"""Drop-in geometry-and-facies subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.ui.common import PathField


def build(parent_page: Any) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        note = QLabel(
            "Injection-point coordinates are optional in the Excel protocol. Keep each "
            "location_id, then either enter x/y below or select a baseline or early "
            "CO2 image and place the point by clicking. x is measured from the "
            "left and y from the bottom of the registered 2D rig."
        )
        note.setObjectName("secondaryText")
        note.setWordWrap(True)
        layout.addWidget(note)

        protocol_actions = QHBoxLayout()
        parent_page.load_protocol_points = QPushButton("Load / update from Excel protocol")
        parent_page.load_protocol_points.setToolTip(
            "Read ip1, ip2, and optional x/y coordinates from the selected Excel "
            "protocol. Manual GUI positions are kept and compared with Excel."
        )
        parent_page.load_protocol_points.clicked.connect(
            parent_page._load_injection_points_from_protocol
        )
        protocol_actions.addWidget(parent_page.load_protocol_points)
        protocol_actions.addStretch(1)
        layout.addLayout(protocol_actions)

        image_filter = (
            "Images and DarSIA files "
            "(*.npz *.npy *.jpg *.jpeg *.png *.tif *.tiff);;All files (*)"
        )
        reference_form = QFormLayout()
        parent_page.review_frame_path = PathField(
            directory=False,
            file_filter=image_filter,
            placeholder="Optional early CO2 image; leave blank to use the baseline",
        )
        reference_form.addRow("Placement image", parent_page.review_frame_path)
        layout.addLayout(reference_form)
        reference_actions = QHBoxLayout()
        parent_page.preview_injector_reference = QPushButton("Preview placement image")
        parent_page.preview_injector_reference.clicked.connect(
            parent_page._preview_injector_reference
        )
        reference_actions.addWidget(parent_page.preview_injector_reference)
        parent_page.show_injector_baseline = QPushButton("Show baseline")
        parent_page.show_injector_baseline.clicked.connect(parent_page._show_injector_baseline)
        reference_actions.addWidget(parent_page.show_injector_baseline)
        parent_page.place_selected_injector = QPushButton("Click image to place selected")
        parent_page.place_selected_injector.setCheckable(True)
        parent_page.place_selected_injector.toggled.connect(parent_page.place_injector.setChecked)
        parent_page.place_injector.toggled.connect(parent_page.place_selected_injector.setChecked)
        reference_actions.addWidget(parent_page.place_selected_injector)
        reference_actions.addStretch(1)
        layout.addLayout(reference_actions)

        parent_page.injector_table = QTableWidget(0, 6)
        parent_page.injector_table.setHorizontalHeaderLabels(
            [
                "Location ID",
                "Name",
                "x from left [m]",
                "y above bottom [m]",
                "Uncertainty [m]",
                "Source",
            ]
        )
        parent_page.injector_table.verticalHeader().setVisible(False)
        parent_page.injector_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        header = parent_page.injector_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        parent_page.injector_table.itemSelectionChanged.connect(
            parent_page._update_injector_position_status
        )
        layout.addWidget(parent_page.injector_table, 1)
        buttons = QHBoxLayout()
        add = QPushButton("Add injection point")
        add.clicked.connect(parent_page._add_injector)
        buttons.addWidget(add)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(parent_page._remove_injector)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        parent_page.injector_position_status = QLabel(
            "No injection point selected. Coordinates entered in Excel will appear in this table."
        )
        parent_page.injector_position_status.setObjectName("secondaryText")
        parent_page.injector_position_status.setWordWrap(True)
        layout.addWidget(parent_page.injector_position_status)
        return tab
