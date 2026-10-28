"""Rig-and-protocol setup step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWizardPage,
)

from darsia.fluidflower.pages.setup.steps.identity import SeriesIdentityPage
from darsia.fluidflower.protocol_schema import (
    write_protocol_file_set_template,
    write_protocol_workbook_template,
)
from darsia.fluidflower.templates import RIG_TYPES, template_by_id
from darsia.fluidflower.ui.common import (
    DIRECTORY_DIALOG_OPTIONS,
    FILE_DIALOG_OPTIONS,
    PathField,
)


class RigProtocolPage(QWizardPage):
    def __init__(self, identity: SeriesIdentityPage) -> None:
        super().__init__()
        self.identity = identity
        self.setTitle("Apparatus and protocol")
        self.setSubTitle(
            "The universal FluidFlower event-table schema is independent of apparatus "
            "class and series; measured values remain run-specific."
        )
        layout = QFormLayout(self)
        self.rig_type = QComboBox()
        for key, label in RIG_TYPES.items():
            self.rig_type.addItem(label, key)
        apparatus_help = (
            "Select the physical FluidFlower apparatus class and approximate size. "
            "This does not define the sand-layer geometry or facies; those are "
            "established and reviewed later in Geometry and facies."
        )
        self.rig_type.setToolTip(apparatus_help)
        apparatus_label = QLabel("FluidFlower apparatus")
        apparatus_label.setToolTip(apparatus_help)
        layout.addRow(apparatus_label, self.rig_type)
        self.protocol_layout = QComboBox()
        self.protocol_layout.addItem(
            "One Excel workbook inside each experiment folder (recommended)",
            "per_run_files",
        )
        self.protocol_layout.addItem("One Excel workbook for all runs", "combined")
        self.protocol_layout.addItem("One Excel sheet per run", "per_run_sheets")
        layout_help = {
            "combined": (
                "One Events table contains every run. Each row must identify its run "
                "with the run_id column."
            ),
            "per_run_sheets": (
                "One Excel workbook contains a separate sheet for each run. The sheet "
                "name identifies the run."
            ),
            "per_run_files": (
                "Each experiment normally contains its own Excel protocol workbook. "
                "DarSIA discovers and reviews it together with that experiment's "
                "images; no common protocol folder is required."
            ),
        }
        for index in range(self.protocol_layout.count()):
            key = str(self.protocol_layout.itemData(index))
            self.protocol_layout.setItemData(
                index, layout_help[key], Qt.ItemDataRole.ToolTipRole
            )
        self.protocol_layout.currentIndexChanged.connect(self._protocol_layout_changed)
        organization_label = QLabel("Protocol organization")
        organization_label.setToolTip(
            "Choose how runs are separated in the protocol source. The editable "
            "example below follows this selection."
        )
        layout.addRow(organization_label, self.protocol_layout)
        self.protocol = PathField(
            directory=False,
            file_filter=(
                "Excel protocol workbooks (*.xlsx *.xls);;"
                "Legacy protocol files (*.csv *.tsv *.toml *.json *.yaml *.yml);;"
                "All files (*)"
            ),
            placeholder="Optional; can be established later",
        )
        self.protocol_source_label = QLabel("Protocol file")
        layout.addRow(self.protocol_source_label, self.protocol)
        self.protocol_discovery_note = QLabel(
            "Protocol workbooks are selected per experiment in the source mapping "
            "below. They may sit directly in the experiment folder or in one of its "
            "subfolders."
        )
        self.protocol_discovery_note.setWordWrap(True)
        self.protocol_discovery_note.setObjectName("secondaryText")
        layout.addRow("Per-experiment protocol", self.protocol_discovery_note)
        self.create_protocol_template = QPushButton("Create editable example...")
        self.create_protocol_template.clicked.connect(self._create_protocol_template)
        layout.addRow("Editable example", self.create_protocol_template)
        note = QLabel(
            "This selection supplies apparatus-level defaults only. Exact dimensions, "
            "depth, injection points, sand-layer geometry, and facies are reviewed in "
            "later modules."
        )
        note.setWordWrap(True)
        note.setObjectName("secondaryText")
        layout.addRow("", note)
        self._protocol_layout_changed()

    def _protocol_layout_changed(self) -> None:
        layout = str(self.protocol_layout.currentData())
        per_run_files = layout == "per_run_files"
        self.protocol.directory = False
        self.protocol.file_filter = (
            "Excel protocol workbooks (*.xlsx *.xls);;"
            "Legacy protocol tables (*.csv *.tsv);;All files (*)"
        )
        self.protocol_source_label.setText("Shared protocol workbook")
        self.protocol_source_label.setVisible(not per_run_files)
        self.protocol.setVisible(not per_run_files)
        self.protocol_discovery_note.setVisible(per_run_files)
        button_text = {
            "combined": "Create combined-workbook example...",
            "per_run_sheets": "Create per-run-sheet workbook...",
            "per_run_files": "Create per-run-file examples...",
        }
        self.create_protocol_template.setText(button_text[layout])
        help_text = str(
            self.protocol_layout.itemData(
                self.protocol_layout.currentIndex(), Qt.ItemDataRole.ToolTipRole
            )
            or ""
        )
        self.protocol_layout.setToolTip(help_text)
        self.create_protocol_template.setToolTip(
            "Create an editable example with experiment dates, actual camera "
            "intervals, injection points and rate segments, pH measurements, and "
            f"interventions. {help_text}"
        )

    def _create_protocol_template(self) -> None:
        layout = str(self.protocol_layout.currentData())
        if layout == "per_run_files":
            selected = QFileDialog.getExistingDirectory(
                self,
                "Choose where to create the per-run protocol example folder",
                str(Path.home()),
                options=DIRECTORY_DIALOG_OPTIONS,
            )
            if not selected:
                return
            base = Path(selected) / "fluidflower_protocol_per_run_example"
            path = base
            suffix = 2
            while path.exists():
                path = base.with_name(f"{base.name}_{suffix}")
                suffix += 1
            try:
                write_protocol_file_set_template(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(
                    self, "Could not create protocol examples", str(exc)
                )
                return
            QMessageBox.information(
                self,
                "Protocol examples created",
                "Two editable per-run Excel workbooks and a short README were "
                f"written to:\n{path}",
            )
            return

        filename = (
            "fluidflower_protocol_per_run_sheets.xlsx"
            if layout == "per_run_sheets"
            else "fluidflower_protocol_combined.xlsx"
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Create universal FluidFlower protocol template",
            str(Path.home() / filename),
            "Excel workbooks (*.xlsx)",
            options=FILE_DIALOG_OPTIONS,
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.casefold() != ".xlsx":
            path = path.with_suffix(".xlsx")
        try:
            write_protocol_workbook_template(path, layout=layout)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not create protocol template", str(exc))
            return
        QMessageBox.information(
            self,
            "Protocol template created",
            f"The editable workbook was written to:\n{path}",
        )

    def initializePage(self) -> None:  # noqa: N802
        template = template_by_id(self.identity.template_id())
        index = self.rig_type.findData(template.default_rig_type)
        if index >= 0:
            self.rig_type.setCurrentIndex(index)


def build(wizard: Any) -> RigProtocolPage:
    return RigProtocolPage(wizard.identity)
