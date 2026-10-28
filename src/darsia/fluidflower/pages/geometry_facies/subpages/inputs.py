"""Drop-in geometry-and-facies subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from darsia.fluidflower.ui.common import PathField


def build(parent_page: Any) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        array_filter = "DarSIA and array files (*.npz *.npy);;All files (*)"
        image_filter = "Images and DarSIA files (*.npz *.npy *.jpg *.jpeg *.png *.tif *.tiff);;All files (*)"
        parent_page.rig_folder = PathField(
            directory=True, placeholder="Optional imported rig folder"
        )
        parent_page.baseline_path = PathField(
            directory=False, file_filter=image_filter, placeholder="Corrected baseline"
        )
        parent_page.depth_path = PathField(
            directory=False, file_filter=array_filter, placeholder="Depth map"
        )
        parent_page.depth_path.setToolTip(
            "Select the approved rig-depth map. Saving geometry copies it into "
            "the immutable geometry revision; the selected source is never changed."
        )
        parent_page.labels_path = PathField(
            directory=False,
            file_filter=array_filter,
            placeholder="Detailed physical-layer labels",
        )
        parent_page.facies_path = PathField(
            directory=False,
            file_filter=array_filter,
            placeholder="Optional existing facies map",
        )
        parent_page.properties_path = PathField(
            directory=False,
            file_filter="Tables (*.csv *.xlsx *.xls);;All files (*)",
            placeholder="Optional facies properties",
        )
        parent_page.protocol_path = PathField(
            directory=False,
            file_filter=(
                "Excel protocol workbooks (*.xlsx *.xls);;"
                "Legacy protocol tables (*.csv);;All files (*)"
            ),
            placeholder="Optional protocol; injection-point x/y may be blank",
        )
        form.addRow("Rig folder", parent_page.rig_folder)
        form.addRow("Baseline", parent_page.baseline_path)
        form.addRow("Depth map", parent_page.depth_path)
        form.addRow("Layer labels", parent_page.labels_path)
        form.addRow("Facies source", parent_page.facies_path)
        form.addRow("Properties", parent_page.properties_path)
        form.addRow("Protocol", parent_page.protocol_path)
        dimensions = QWidget()
        dimensions_layout = QHBoxLayout(dimensions)
        dimensions_layout.setContentsMargins(0, 0, 0, 0)
        parent_page.width_m = QDoubleSpinBox()
        parent_page.width_m.setRange(0.0, 100.0)
        parent_page.width_m.setDecimals(9)
        parent_page.width_m.setSuffix(" m")
        parent_page.height_m = QDoubleSpinBox()
        parent_page.height_m.setRange(0.0, 100.0)
        parent_page.height_m.setDecimals(9)
        parent_page.height_m.setSuffix(" m")
        dimensions_layout.addWidget(QLabel("Width"))
        dimensions_layout.addWidget(parent_page.width_m)
        dimensions_layout.addWidget(QLabel("Height"))
        dimensions_layout.addWidget(parent_page.height_m)
        form.addRow("Dimensions", dimensions)
        parent_page.grouping_mode = QComboBox()
        parent_page.grouping_mode.addItem("Shared properties by facies", "shared_facies")
        parent_page.grouping_mode.addItem("Unique properties per physical layer", "per_layer")
        form.addRow("Material grouping", parent_page.grouping_mode)
        parent_page.permeability_unit = QComboBox()
        parent_page.permeability_unit.setEditable(True)
        for value in ("source_unit", "darcy", "mD", "m2"):
            parent_page.permeability_unit.addItem(value)
        form.addRow("Permeability unit", parent_page.permeability_unit)

        draft_heading = QLabel("Unknown geometry")
        draft_heading.setObjectName("sectionTitle")
        form.addRow(draft_heading)
        draft_note = QLabel(
            "Generate a reproducible optical draft from a baseline image. "
            "Every proposed region starts excluded until reviewed."
        )
        draft_note.setWordWrap(True)
        draft_note.setObjectName("secondaryText")
        form.addRow(draft_note)
        parent_page.draft_source_path = PathField(
            directory=False,
            file_filter=image_filter,
            placeholder="Baseline image for an unknown rig",
        )
        form.addRow("Draft source", parent_page.draft_source_path)
        draft_settings = QWidget()
        draft_layout = QHBoxLayout(draft_settings)
        draft_layout.setContentsMargins(0, 0, 0, 0)
        parent_page.draft_classes = QSpinBox()
        parent_page.draft_classes.setRange(2, 24)
        parent_page.draft_classes.setValue(6)
        parent_page.draft_depth_mm = QDoubleSpinBox()
        parent_page.draft_depth_mm.setRange(0.01, 1000.0)
        parent_page.draft_depth_mm.setDecimals(3)
        parent_page.draft_depth_mm.setValue(10.0)
        parent_page.draft_depth_mm.setSuffix(" mm")
        draft_layout.addWidget(QLabel("Classes"))
        draft_layout.addWidget(parent_page.draft_classes)
        draft_layout.addWidget(QLabel("Depth"))
        draft_layout.addWidget(parent_page.draft_depth_mm)
        parent_page.generate_draft = QPushButton("Generate optical draft")
        parent_page.generate_draft.clicked.connect(parent_page._generate_optical_draft)
        draft_layout.addStretch(1)
        form.addRow("Draft settings", draft_settings)
        form.addRow("", parent_page.generate_draft)
        return tab
