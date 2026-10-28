"""Drop-in geometry-and-facies subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.facies_learning import FACIES_SPLITS
from darsia.fluidflower.ui.common import PathField


def build(parent_page: Any) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        form = QFormLayout()
        parent_page.facies_dataset_root = PathField(
            directory=True,
            placeholder="Local or shared facies dataset folder",
        )
        parent_page.facies_dataset_root.setText(
            str(parent_page.project.root / "annotations" / "facies")
        )
        form.addRow("Dataset folder", parent_page.facies_dataset_root)
        parent_page.facies_group_id = QLineEdit(
            str(parent_page.project.manifest.get("series_id") or parent_page.project.project_id)
        )
        parent_page.facies_group_id.setToolTip(
            "All samples sharing one rig and acquisition context must use the same "
            "group so they cannot cross dataset splits."
        )
        form.addRow("Leakage group", parent_page.facies_group_id)
        parent_page.facies_split = QComboBox()
        for split in FACIES_SPLITS:
            parent_page.facies_split.addItem(split.title(), split)
        form.addRow("Dataset split", parent_page.facies_split)
        parent_page.include_facies_review_frame = QCheckBox("Registered experiment frame")
        parent_page.include_facies_review_frame.setToolTip(
            "Include the selected registered frame as contextual data. The corrected "
            "baseline remains the default training image."
        )
        form.addRow("Additional image", parent_page.include_facies_review_frame)
        layout.addLayout(form)

        actions = QHBoxLayout()
        parent_page.export_facies_bundle = QPushButton("Export annotation bundle")
        parent_page.export_facies_bundle.clicked.connect(parent_page._export_facies_annotations)
        actions.addWidget(parent_page.export_facies_bundle)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(parent_page._refresh_facies_exports)
        actions.addWidget(refresh)
        open_folder = QPushButton("Open folder")
        open_folder.clicked.connect(parent_page._open_facies_dataset)
        actions.addWidget(open_folder)
        actions.addStretch(1)
        layout.addLayout(actions)

        parent_page.facies_export_status = QLabel("No facies annotations exported.")
        parent_page.facies_export_status.setObjectName("secondaryText")
        parent_page.facies_export_status.setWordWrap(True)
        layout.addWidget(parent_page.facies_export_status)
        parent_page.facies_dataset_summary = QLabel("")
        parent_page.facies_dataset_summary.setObjectName("secondaryText")
        layout.addWidget(parent_page.facies_dataset_summary)

        parent_page.facies_exports = QTableWidget(0, 6)
        parent_page.facies_exports.setHorizontalHeaderLabels(
            ["Time", "Series", "Group", "Split", "Status", "Bundle"]
        )
        parent_page.facies_exports.verticalHeader().setVisible(False)
        parent_page.facies_exports.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        parent_page.facies_exports.cellDoubleClicked.connect(parent_page._open_selected_facies_bundle)
        export_header = parent_page.facies_exports.horizontalHeader()
        export_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        export_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        export_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for column in (3, 4, 5):
            export_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        parent_page.facies_exports.setColumnWidth(1, 105)
        parent_page.facies_exports.setMaximumHeight(170)
        layout.addWidget(parent_page.facies_exports)

        model_heading = QLabel("Supervised facies baseline")
        model_heading.setObjectName("sectionTitle")
        layout.addWidget(model_heading)
        model_note = QLabel(
            "Train universal spatial-role, named-material, and transferable hydraulic "
            "heads from approved bundles. Entire leakage groups remain isolated between "
            "train, validation, and holdout. This CPU baseline does not perform phase, "
            "3D, concentration, or mass interpretation."
        )
        model_note.setObjectName("secondaryText")
        model_note.setWordWrap(True)
        layout.addWidget(model_note)

        model_form = QFormLayout()
        parent_page.facies_model_dataset_roots = QLineEdit(
            str(parent_page.project.root / "annotations" / "facies")
        )
        parent_page.facies_model_dataset_roots.setPlaceholderText(
            "One or more annotation folders separated by semicolons"
        )
        parent_page.facies_model_dataset_roots.setToolTip(
            "Combine approved bundles from multiple projects or experimental series. "
            "Use semicolons between folders."
        )
        model_form.addRow("Annotation folders", parent_page.facies_model_dataset_roots)
        parent_page.facies_model_root = PathField(
            directory=True,
            placeholder="Immutable supervised facies models",
        )
        parent_page.facies_model_root.setText(
            str(parent_page.project.root / "results" / "facies_models")
        )
        model_form.addRow("Model folder", parent_page.facies_model_root)
        parent_page.facies_model_split_plan = PathField(
            directory=False,
            file_filter="Facies split plans (*.json);;All files (*)",
            placeholder="Optional versioned group assignment",
        )
        parent_page.facies_model_split_plan.setToolTip(
            "A split plan can assign bundles exported once to Pool into train, "
            "validation, and holdout for a reproducible campaign."
        )
        model_form.addRow("Split plan", parent_page.facies_model_split_plan)

        parent_page.facies_model_estimators = QSpinBox()
        parent_page.facies_model_estimators.setRange(8, 4096)
        parent_page.facies_model_estimators.setValue(128)
        parent_page.facies_model_estimators.setMinimumWidth(120)
        parent_page.facies_model_pixels = QSpinBox()
        parent_page.facies_model_pixels.setRange(100, 2_000_000)
        parent_page.facies_model_pixels.setSingleStep(10_000)
        parent_page.facies_model_pixels.setValue(120_000)
        parent_page.facies_model_pixels.setMinimumWidth(120)
        parent_page.facies_model_cpu_workers = QSpinBox()
        parent_page.facies_model_cpu_workers.setRange(0, 256)
        parent_page.facies_model_cpu_workers.setSpecialValueText("Auto")
        parent_page.facies_model_cpu_workers.setValue(0)
        parent_page.facies_model_cpu_workers.setMinimumWidth(120)
        model_form.addRow("Trees", parent_page.facies_model_estimators)
        model_form.addRow("Pixels per bundle", parent_page.facies_model_pixels)
        model_form.addRow("CPU workers", parent_page.facies_model_cpu_workers)

        model_options = QWidget()
        model_options_layout = QHBoxLayout(model_options)
        model_options_layout.setContentsMargins(0, 0, 0, 0)
        parent_page.facies_model_calibrate = QCheckBox("Calibrate uncertainty")
        parent_page.facies_model_calibrate.setChecked(True)
        parent_page.facies_model_calibrate.setToolTip(
            "Uses group-isolated sigmoid calibration only when at least three "
            "independent training groups contain every class."
        )
        model_options_layout.addWidget(parent_page.facies_model_calibrate)
        parent_page.facies_model_position = QCheckBox("Use normalized image position")
        parent_page.facies_model_position.setToolTip(
            "Disabled by default because absolute position may leak rig geometry and "
            "reduce transfer to other FluidFlower layouts."
        )
        model_options_layout.addWidget(parent_page.facies_model_position)
        model_options_layout.addStretch(1)
        model_form.addRow("Options", model_options)
        layout.addLayout(model_form)

        model_actions = QGridLayout()
        parent_page.train_facies_model = QPushButton("Train supervised baseline")
        parent_page.train_facies_model.clicked.connect(parent_page._train_facies_baseline)
        model_actions.addWidget(parent_page.train_facies_model, 0, 0, 1, 2)
        refresh_models = QPushButton("Refresh models")
        refresh_models.clicked.connect(parent_page._refresh_facies_models)
        model_actions.addWidget(refresh_models, 1, 0)
        open_models = QPushButton("Open model folder")
        open_models.clicked.connect(parent_page._open_facies_models)
        model_actions.addWidget(open_models, 1, 1)
        parent_page.facies_model_busy = QProgressBar()
        parent_page.facies_model_busy.setRange(0, 1)
        parent_page.facies_model_busy.setValue(0)
        model_actions.addWidget(parent_page.facies_model_busy, 2, 0, 1, 2)
        model_actions.setColumnStretch(0, 1)
        model_actions.setColumnStretch(1, 1)
        layout.addLayout(model_actions)

        parent_page.facies_model_status = QLabel("No supervised model trained.")
        parent_page.facies_model_status.setObjectName("secondaryText")
        parent_page.facies_model_status.setWordWrap(True)
        layout.addWidget(parent_page.facies_model_status)
        parent_page.facies_models = QTableWidget(0, 5)
        parent_page.facies_models.setHorizontalHeaderLabels(
            [
                "Time",
                "Model",
                "Validation hydraulic mIoU",
                "Holdout hydraulic mIoU",
                "Status",
            ]
        )
        parent_page.facies_models.verticalHeader().setVisible(False)
        parent_page.facies_models.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        parent_page.facies_models.cellDoubleClicked.connect(parent_page._open_selected_facies_model)
        model_header = parent_page.facies_models.horizontalHeader()
        model_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        model_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        model_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        model_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        model_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(parent_page.facies_models, 1)
        parent_page._refresh_facies_exports()
        parent_page._refresh_facies_models()
        return tab
