"""Drop-in phase-interpretation subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.ui.common import PathField


def build(parent_page: Any) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    explanation = QLabel(
        "Inspect the selected model on the complete corrected 2D image sequence, "
        "without A/B comparison or annotation choices. Approve every experiment "
        "only when the visible phase map is scientifically credible. If it is not, "
        "open the separate contribution workspace and add localized corrections "
        "that can later be exported to shared model training."
    )
    explanation.setWordWrap(True)
    explanation.setObjectName("secondaryText")
    layout.addWidget(explanation)
    form = QFormLayout()
    parent_page.phase_model_validation_root = PathField(
        directory=True,
        placeholder="Prepared campaign root containing postflight_manifest.json",
    )
    parent_page.phase_model_validation_save_root = PathField(
        directory=True,
        placeholder="Reviewer-owned local validation folder",
    )
    parent_page.phase_model_validation_save_root.setText(
        str(
            parent_page.project.root
            / "annotations"
            / "phase"
            / "validation_sessions"
        )
    )
    form.addRow("Prepared campaign", parent_page.phase_model_validation_root)
    form.addRow("Local review record", parent_page.phase_model_validation_save_root)
    layout.addLayout(form)
    actions = QHBoxLayout()
    use_campaign = QPushButton("Use inspected campaign")
    use_campaign.clicked.connect(
        lambda: parent_page._use_inspected_review_campaign("model_validation")
    )
    actions.addWidget(use_campaign)
    parent_page.open_model_validation = QPushButton("Open 2D model inspection")
    parent_page.open_model_validation.clicked.connect(
        lambda: parent_page._launch_phase_review("model_validation")
    )
    actions.addWidget(parent_page.open_model_validation)
    parent_page.open_phase_contribution = QPushButton(
        "Add local corrections / training evidence"
    )
    parent_page.open_phase_contribution.clicked.connect(
        parent_page._open_phase_contribution
    )
    actions.addWidget(parent_page.open_phase_contribution)
    actions.addStretch(1)
    layout.addLayout(actions)
    parent_page.phase_model_validation_status = QLabel(
        "No inspection session open. Project approval is kept separate from "
        "optional training contributions."
    )
    parent_page.phase_model_validation_status.setWordWrap(True)
    parent_page.phase_model_validation_status.setObjectName("secondaryText")
    layout.addWidget(parent_page.phase_model_validation_status)

    heading = QLabel("Per-experiment visible-2D approval")
    heading.setObjectName("sectionTitle")
    layout.addWidget(heading)
    parent_page.phase_run_review_table = QTableWidget(0, 5)
    parent_page.phase_run_review_table.setHorizontalHeaderLabels(
        ["Experiment", "Selected model", "2D phase status", "Reviewer", "Note"]
    )
    parent_page.phase_run_review_table.setSelectionBehavior(
        QAbstractItemView.SelectionBehavior.SelectRows
    )
    parent_page.phase_run_review_table.setSelectionMode(
        QAbstractItemView.SelectionMode.SingleSelection
    )
    parent_page.phase_run_review_table.setEditTriggers(
        QAbstractItemView.EditTrigger.NoEditTriggers
    )
    parent_page.phase_run_review_table.verticalHeader().setVisible(False)
    header = parent_page.phase_run_review_table.horizontalHeader()
    for column in (0, 2, 3):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
    layout.addWidget(parent_page.phase_run_review_table, 1)

    review_actions = QHBoxLayout()
    parent_page.phase_run_review_note = QLineEdit()
    parent_page.phase_run_review_note.setPlaceholderText(
        "Optional reason or required local correction"
    )
    review_actions.addWidget(parent_page.phase_run_review_note, 1)
    approve = QPushButton("Approve selected experiment")
    approve.clicked.connect(
        lambda: parent_page._set_selected_phase_run_review("approved")
    )
    review_actions.addWidget(approve)
    repair = QPushButton("Needs local correction")
    repair.clicked.connect(
        lambda: parent_page._set_selected_phase_run_review(
            "needs_local_correction"
        )
    )
    review_actions.addWidget(repair)
    reset = QPushButton("Reset to pending")
    reset.clicked.connect(
        lambda: parent_page._set_selected_phase_run_review("pending")
    )
    review_actions.addWidget(reset)
    layout.addLayout(review_actions)
    return tab
