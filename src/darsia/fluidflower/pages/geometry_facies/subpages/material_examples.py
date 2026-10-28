"""Drop-in few-shot material and facies proposal subpage."""

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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.ui.common import PathField


def build(parent_page: Any) -> QWidget:
    tab = QScrollArea()
    tab.setWidgetResizable(True)
    content = QWidget()
    layout = QVBoxLayout(content)
    tab.setWidget(content)

    note = QLabel(
        "Define the sand vocabulary once, then mark representative interior "
        "examples. New or unknown series should start guided. A mature compatible "
        "model may instead predict blindly before local corrections are added. "
        "Every result remains an unreviewed 2D proposal."
    )
    note.setObjectName("integrationNotice")
    note.setWordWrap(True)
    layout.addWidget(note)

    form = QFormLayout()
    parent_page.facies_example_strategy = QComboBox()
    parent_page.facies_example_strategy.addItem(
        "Guide with named examples", "guided_examples"
    )
    parent_page.facies_example_strategy.addItem(
        "Blind compatible-model prediction first", "blind_model_first"
    )
    parent_page.facies_example_strategy.setToolTip(
        "Use guided setup while the shared model is immature. Blind-first is an "
        "honest transfer test: inspect its proposal before adding local examples."
    )
    form.addRow("Starting strategy", parent_page.facies_example_strategy)

    parent_page.facies_example_source = PathField(
        directory=False,
        file_filter=(
            "Images and arrays (*.jpg *.jpeg *.png *.tif *.tiff *.npy *.npz);;"
            "All files (*)"
        ),
        placeholder="Registered corrected or representative rig image",
    )
    form.addRow("Example image", parent_page.facies_example_source)
    source_actions = QWidget()
    source_layout = QHBoxLayout(source_actions)
    source_layout.setContentsMargins(0, 0, 0, 0)
    parent_page.facies_example_use_baseline = QPushButton("Use current baseline")
    parent_page.facies_example_use_baseline.clicked.connect(
        parent_page._facies_example_use_current_baseline
    )
    source_layout.addWidget(parent_page.facies_example_use_baseline)
    parent_page.facies_example_load = QPushButton("Show image")
    parent_page.facies_example_load.clicked.connect(
        parent_page._load_facies_example_source
    )
    source_layout.addWidget(parent_page.facies_example_load)
    source_layout.addStretch(1)
    form.addRow("Image actions", source_actions)

    parent_page.facies_example_model_manifest = PathField(
        directory=False,
        file_filter="DarSIA facies model manifest (*.json);;All files (*)",
        placeholder="Required only for blind-first prediction",
    )
    form.addRow("Pretrained model", parent_page.facies_example_model_manifest)
    parent_page.facies_example_use_latest_model = QPushButton(
        "Use latest compatible project model"
    )
    parent_page.facies_example_use_latest_model.clicked.connect(
        parent_page._facies_example_select_latest_model
    )
    form.addRow("Model selection", parent_page.facies_example_use_latest_model)

    parent_page.facies_example_minimum = QSpinBox()
    parent_page.facies_example_minimum.setRange(1, 1000)
    parent_page.facies_example_minimum.setValue(8)
    parent_page.facies_example_minimum.setToolTip(
        "Eight spatially distributed boxes per sand type is the default minimum. "
        "Use pure interiors and avoid boundaries, tubes, reflections, and CO2."
    )
    form.addRow("Minimum boxes per sand type", parent_page.facies_example_minimum)
    layout.addLayout(form)

    material_heading = QLabel("Named sand types")
    material_heading.setObjectName("sectionTitle")
    layout.addWidget(material_heading)
    count_row = QGridLayout()
    count_row.addWidget(QLabel("Number of sand types"), 0, 0)
    parent_page.facies_material_count = QSpinBox()
    parent_page.facies_material_count.setRange(0, 128)
    parent_page.facies_material_count.setValue(0)
    count_row.addWidget(parent_page.facies_material_count, 0, 1)
    parent_page.facies_material_apply_count = QPushButton("Create / update list")
    parent_page.facies_material_apply_count.clicked.connect(
        parent_page._set_facies_material_count
    )
    count_row.addWidget(parent_page.facies_material_apply_count, 0, 2)
    parent_page.facies_material_add = QPushButton("Add sand type")
    parent_page.facies_material_add.clicked.connect(parent_page._add_facies_material)
    count_row.addWidget(parent_page.facies_material_add, 1, 0, 1, 2)
    parent_page.facies_material_remove = QPushButton("Remove selected")
    parent_page.facies_material_remove.clicked.connect(
        parent_page._remove_facies_material
    )
    count_row.addWidget(parent_page.facies_material_remove, 1, 2)
    count_row.setColumnStretch(1, 1)
    layout.addLayout(count_row)

    parent_page.facies_materials = QTableWidget(0, 5)
    parent_page.facies_materials.setHorizontalHeaderLabels(
        ["Sand type", "Role", "Boxes", "Spatial areas", "Ready"]
    )
    parent_page.facies_materials.setSelectionBehavior(
        QAbstractItemView.SelectionBehavior.SelectRows
    )
    parent_page.facies_materials.verticalHeader().setVisible(False)
    material_header = parent_page.facies_materials.horizontalHeader()
    material_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for column in range(1, 5):
        material_header.setSectionResizeMode(
            column, QHeaderView.ResizeMode.ResizeToContents
        )
    parent_page.facies_materials.setMaximumHeight(210)
    layout.addWidget(parent_page.facies_materials)

    examples_heading = QLabel("Representative boxes")
    examples_heading.setObjectName("sectionTitle")
    layout.addWidget(examples_heading)
    example_actions = QGridLayout()
    parent_page.facies_example_material = QComboBox()
    parent_page.facies_example_material.setMinimumWidth(180)
    example_actions.addWidget(QLabel("Sand type"), 0, 0)
    example_actions.addWidget(parent_page.facies_example_material, 0, 1, 1, 2)
    parent_page.draw_facies_example = QPushButton("Draw representative box")
    parent_page.draw_facies_example.setCheckable(True)
    parent_page.draw_facies_example.setToolTip(
        "Drag a tight rectangle inside one visually homogeneous sand region. "
        "The box is normalized to the full registered camera image."
    )
    parent_page.draw_facies_example.toggled.connect(
        parent_page._facies_example_drawing_toggled
    )
    example_actions.addWidget(parent_page.draw_facies_example, 1, 0, 1, 2)
    parent_page.delete_facies_examples = QPushButton("Delete selected boxes")
    parent_page.delete_facies_examples.clicked.connect(
        parent_page._delete_facies_examples
    )
    example_actions.addWidget(parent_page.delete_facies_examples, 1, 2)
    parent_page.show_facies_examples = QCheckBox("Show boxes")
    parent_page.show_facies_examples.setChecked(True)
    parent_page.show_facies_examples.stateChanged.connect(
        parent_page._refresh_facies_example_overlay
    )
    example_actions.addWidget(parent_page.show_facies_examples, 2, 0, 1, 3)
    example_actions.setColumnStretch(1, 1)
    layout.addLayout(example_actions)

    parent_page.facies_examples = QTableWidget(0, 6)
    parent_page.facies_examples.setHorizontalHeaderLabels(
        ["Sand type", "Image", "x", "y", "w x h", "Author"]
    )
    parent_page.facies_examples.setSelectionBehavior(
        QAbstractItemView.SelectionBehavior.SelectRows
    )
    parent_page.facies_examples.setEditTriggers(
        QAbstractItemView.EditTrigger.NoEditTriggers
    )
    parent_page.facies_examples.verticalHeader().setVisible(False)
    examples_header = parent_page.facies_examples.horizontalHeader()
    examples_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    examples_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    for column in range(2, 6):
        examples_header.setSectionResizeMode(
            column, QHeaderView.ResizeMode.ResizeToContents
        )
    parent_page.facies_examples.setMaximumHeight(220)
    layout.addWidget(parent_page.facies_examples)

    parent_page.facies_example_quality_status = QLabel("")
    parent_page.facies_example_quality_status.setObjectName("secondaryText")
    parent_page.facies_example_quality_status.setWordWrap(True)
    layout.addWidget(parent_page.facies_example_quality_status)

    prediction_actions = QGridLayout()
    parent_page.generate_guided_facies = QPushButton("Generate guided proposal")
    parent_page.generate_guided_facies.clicked.connect(
        lambda: parent_page._run_facies_example_prediction("guided_examples")
    )
    prediction_actions.addWidget(parent_page.generate_guided_facies, 0, 0)
    parent_page.generate_blind_facies = QPushButton("Generate blind-first proposal")
    parent_page.generate_blind_facies.clicked.connect(
        lambda: parent_page._run_facies_example_prediction("blind_model_first")
    )
    prediction_actions.addWidget(parent_page.generate_blind_facies, 0, 1)
    parent_page.apply_facies_example_prediction = QPushButton(
        "Use as unreviewed working draft"
    )
    parent_page.apply_facies_example_prediction.setEnabled(False)
    parent_page.apply_facies_example_prediction.clicked.connect(
        parent_page._apply_facies_example_prediction
    )
    prediction_actions.addWidget(
        parent_page.apply_facies_example_prediction, 1, 0, 1, 2
    )
    prediction_actions.setColumnStretch(0, 1)
    prediction_actions.setColumnStretch(1, 1)
    layout.addLayout(prediction_actions)

    parent_page.facies_example_busy = QProgressBar()
    parent_page.facies_example_busy.setRange(0, 1)
    parent_page.facies_example_busy.setValue(0)
    layout.addWidget(parent_page.facies_example_busy)
    parent_page.facies_example_status = QLabel(
        "No few-shot material proposal generated."
    )
    parent_page.facies_example_status.setObjectName("secondaryText")
    parent_page.facies_example_status.setWordWrap(True)
    layout.addWidget(parent_page.facies_example_status)
    layout.addStretch(1)

    parent_page.facies_example_strategy.currentIndexChanged.connect(
        parent_page._facies_example_session_changed
    )
    parent_page.facies_example_source.edit.editingFinished.connect(
        parent_page._facies_example_session_changed
    )
    parent_page.facies_example_model_manifest.edit.editingFinished.connect(
        parent_page._facies_example_session_changed
    )
    parent_page.facies_example_minimum.valueChanged.connect(
        parent_page._facies_example_session_changed
    )
    parent_page.facies_materials.itemChanged.connect(
        parent_page._facies_material_item_changed
    )
    return tab


__all__ = ["build"]
