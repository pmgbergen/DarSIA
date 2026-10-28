"""Drop-in reporting subpage."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.reporting import (
    REPORT_EXPORT_FORMATS,
    REPORT_SECTIONS,
    REQUIRED_REPORT_EXPORTS,
    REQUIRED_REPORT_SECTIONS,
)
from darsia.fluidflower.ui.common import PathField


def build(parent_page: Any) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    result_panel = QFrame()
    result_panel.setFrameShape(QFrame.Shape.StyledPanel)
    result_layout = QFormLayout(result_panel)
    parent_page.reporting_post_result = PathField(
        directory=False,
        file_filter="Post-analysis result (*.json);;All files (*)",
        placeholder="Optional for a plan draft; required for generation",
    )
    result_layout.addRow("Post-analysis result", parent_page.reporting_post_result)
    result_note = QLabel(
        "Every selected movie, figure, table, and global provenance file must "
        "match the current post-analysis task and pass SHA-256 verification."
    )
    result_note.setObjectName("secondaryText")
    result_note.setWordWrap(True)
    result_layout.addRow("Artifact contract", result_note)
    layout.addWidget(result_panel)

    identity_panel = QFrame()
    identity_panel.setFrameShape(QFrame.Shape.StyledPanel)
    identity_layout = QGridLayout(identity_panel)
    parent_page.reporting_title = QLineEdit()
    parent_page.reporting_subtitle = QLineEdit()
    parent_page.reporting_version = QLineEdit()
    parent_page.reporting_language = QComboBox()
    parent_page.reporting_language.addItem("English", "English")
    parent_page.reporting_language.addItem("Norwegian Bokmal", "Norwegian Bokmal")
    parent_page.reporting_audience = QComboBox()
    for label, value in (
        ("Scientific internal review", "scientific_internal_review"),
        ("Supervisor and developer review", "supervisor_and_developer_review"),
        ("Publication supplement", "publication_supplement"),
        ("Public release", "public_release"),
    ):
        parent_page.reporting_audience.addItem(label, value)
    parent_page.reporting_authors = QLineEdit()
    parent_page.reporting_authors.setPlaceholderText("Semicolon-separated names")
    parent_page.reporting_organization = QLineEdit()
    identity_layout.addWidget(QLabel("Title"), 0, 0)
    identity_layout.addWidget(parent_page.reporting_title, 0, 1, 1, 3)
    identity_layout.addWidget(QLabel("Subtitle"), 1, 0)
    identity_layout.addWidget(parent_page.reporting_subtitle, 1, 1, 1, 3)
    identity_layout.addWidget(QLabel("Version"), 2, 0)
    identity_layout.addWidget(parent_page.reporting_version, 2, 1)
    identity_layout.addWidget(QLabel("Language"), 2, 2)
    identity_layout.addWidget(parent_page.reporting_language, 2, 3)
    identity_layout.addWidget(QLabel("Audience"), 3, 0)
    identity_layout.addWidget(parent_page.reporting_audience, 3, 1)
    identity_layout.addWidget(QLabel("Organization"), 3, 2)
    identity_layout.addWidget(parent_page.reporting_organization, 3, 3)
    identity_layout.addWidget(QLabel("Authors"), 4, 0)
    identity_layout.addWidget(parent_page.reporting_authors, 4, 1, 1, 3)
    layout.addWidget(identity_panel)

    content_panel = QWidget()
    content_layout = QVBoxLayout(content_panel)
    content_layout.setContentsMargins(0, 0, 0, 0)
    sections_panel = QFrame()
    sections_panel.setFrameShape(QFrame.Shape.StyledPanel)
    sections_layout = QGridLayout(sections_panel)
    sections_title = QLabel("Report sections")
    sections_title.setObjectName("sectionTitle")
    sections_layout.addWidget(sections_title, 0, 0, 1, 2)
    parent_page.reporting_sections: dict[str, QCheckBox] = {}
    for index, key in enumerate(REPORT_SECTIONS):
        checkbox = QCheckBox(parent_page._SECTION_LABELS[key])
        if key in REQUIRED_REPORT_SECTIONS:
            checkbox.setChecked(True)
            checkbox.setEnabled(False)
        parent_page.reporting_sections[key] = checkbox
        sections_layout.addWidget(checkbox, 1 + index // 2, index % 2)
    content_layout.addWidget(sections_panel)

    exports_panel = QFrame()
    exports_panel.setFrameShape(QFrame.Shape.StyledPanel)
    exports_layout = QGridLayout(exports_panel)
    exports_title = QLabel("Exports")
    exports_title.setObjectName("sectionTitle")
    exports_layout.addWidget(exports_title, 0, 0, 1, 2)
    parent_page.reporting_exports: dict[str, QCheckBox] = {}
    for index, key in enumerate(REPORT_EXPORT_FORMATS):
        checkbox = QCheckBox(parent_page._EXPORT_LABELS[key])
        if key in REQUIRED_REPORT_EXPORTS:
            checkbox.setChecked(True)
            checkbox.setEnabled(False)
        parent_page.reporting_exports[key] = checkbox
        exports_layout.addWidget(checkbox, 1 + index // 2, index % 2)
    parent_page.reporting_embed_limit = QSpinBox()
    parent_page.reporting_embed_limit.setRange(0, 4096)
    parent_page.reporting_embed_limit.setSuffix(" MB")
    exports_layout.addWidget(QLabel("Embed limit"), 5, 0)
    exports_layout.addWidget(parent_page.reporting_embed_limit, 5, 1)
    media_note = QLabel(
        "Large movies remain verified links with thumbnails. Small artifacts "
        "are copied into the versioned provenance package."
    )
    media_note.setObjectName("secondaryText")
    media_note.setWordWrap(True)
    exports_layout.addWidget(media_note, 6, 0, 1, 2)
    content_layout.addWidget(exports_panel)
    layout.addWidget(content_panel)

    evidence_panel = QFrame()
    evidence_panel.setFrameShape(QFrame.Shape.StyledPanel)
    evidence_layout = QVBoxLayout(evidence_panel)
    evidence_title = QLabel("Additional evidence")
    evidence_title.setObjectName("sectionTitle")
    evidence_layout.addWidget(evidence_title)
    parent_page.reporting_evidence = QTableWidget(0, 3)
    parent_page.reporting_evidence.setHorizontalHeaderLabels(["Type", "Title", "File"])
    parent_page.reporting_evidence.verticalHeader().setVisible(False)
    parent_page.reporting_evidence.horizontalHeader().setSectionResizeMode(
        0, QHeaderView.ResizeMode.ResizeToContents
    )
    parent_page.reporting_evidence.horizontalHeader().setSectionResizeMode(
        1, QHeaderView.ResizeMode.Stretch
    )
    parent_page.reporting_evidence.horizontalHeader().setSectionResizeMode(
        2, QHeaderView.ResizeMode.Stretch
    )
    parent_page.reporting_evidence.setMaximumHeight(210)
    evidence_layout.addWidget(parent_page.reporting_evidence)
    evidence_actions = QHBoxLayout()
    add_evidence = QPushButton("Add files")
    add_evidence.clicked.connect(parent_page._add_reporting_evidence)
    evidence_actions.addWidget(add_evidence)
    remove_evidence = QPushButton("Remove selected")
    remove_evidence.clicked.connect(parent_page._remove_reporting_evidence)
    evidence_actions.addWidget(remove_evidence)
    evidence_actions.addStretch(1)
    evidence_layout.addLayout(evidence_actions)
    layout.addWidget(evidence_panel)

    execution_panel = QFrame()
    execution_panel.setFrameShape(QFrame.Shape.StyledPanel)
    execution_layout = QGridLayout(execution_panel)
    parent_page.reporting_output_root = PathField(
        directory=True, placeholder="Versioned report-package directory"
    )
    parent_page.reporting_local_cache = QCheckBox("Local cache")
    parent_page.reporting_retries = QSpinBox()
    parent_page.reporting_retries.setRange(1, 64)
    execution_layout.addWidget(QLabel("Output root"), 0, 0)
    execution_layout.addWidget(parent_page.reporting_output_root, 0, 1, 1, 3)
    execution_layout.addWidget(parent_page.reporting_local_cache, 1, 0)
    execution_layout.addWidget(QLabel("I/O attempts"), 1, 2)
    execution_layout.addWidget(parent_page.reporting_retries, 1, 3)
    layout.addWidget(execution_panel)
    layout.addStretch(1)
    return tab
