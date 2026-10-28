"""Drop-in phase-interpretation subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.ui.common import PathField


def build(parent_page: Any) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        explanation = QLabel(
            "Review model-proposed regions near uncertain phase decisions. The "
            "model answer stays hidden until you respond. The full-resolution "
            "timeline, zoom and pan, phase layers, speed loop, reference frame, "
            "manual boxes/freehand areas, guides, filters, and continuous local "
            "autosave remain available."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)
        form = QFormLayout()
        parent_page.phase_active_learning_root = PathField(
            directory=True,
            placeholder="Prepared campaign root containing postflight_manifest.json",
        )
        parent_page.phase_active_learning_save_root = PathField(
            directory=True,
            placeholder="Reviewer-owned local annotation folder",
        )
        parent_page.phase_active_learning_save_root.setText(
            str(parent_page.project.root / "annotations" / "phase" / "review_sessions")
        )
        form.addRow("Prepared campaign", parent_page.phase_active_learning_root)
        form.addRow("Local autosave root", parent_page.phase_active_learning_save_root)
        layout.addLayout(form)
        actions = QHBoxLayout()
        use_campaign = QPushButton("Use inspected campaign")
        use_campaign.clicked.connect(
            lambda: parent_page._use_inspected_review_campaign("active_learning")
        )
        actions.addWidget(use_campaign)
        parent_page.open_active_learning_review = QPushButton("Open active learning review")
        parent_page.open_active_learning_review.clicked.connect(
            lambda: parent_page._launch_phase_review("active_learning")
        )
        actions.addWidget(parent_page.open_active_learning_review)
        actions.addStretch(1)
        layout.addLayout(actions)
        parent_page.phase_active_learning_status = QLabel(
            "No review session open. New answers are isolated from blind validation."
        )
        parent_page.phase_active_learning_status.setWordWrap(True)
        parent_page.phase_active_learning_status.setObjectName("secondaryText")
        layout.addWidget(parent_page.phase_active_learning_status)
        layout.addStretch(1)
        return tab
