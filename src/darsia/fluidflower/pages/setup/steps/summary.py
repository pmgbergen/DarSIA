"""Project-summary setup step."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWizardPage

from darsia.fluidflower.project import slugify
from darsia.fluidflower.templates import RIG_TYPES, template_by_id


class SummaryPage(QWizardPage):
    def __init__(self, wizard: Any) -> None:
        super().__init__()
        self.owner = wizard
        self.setTitle("Create project")
        self.setSubTitle(
            "Review the setup. Source data remain read-only and are only "
            "referenced from the project."
        )
        layout = QVBoxLayout(self)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary)

    def initializePage(self) -> None:  # noqa: N802
        owner = self.owner
        template = template_by_id(owner.identity.template_id())
        rig = RIG_TYPES[str(owner.rig_protocol.rig_type.currentData())]
        reference_settings = owner.images.reference_settings()
        indicator_reference = owner.images.indicator_reference_contract()
        source_rows = [
            (
                "Raw experiment-data folder (read-only)",
                owner.images.experiment_images.text(),
            ),
            ("Shared baseline override", owner.images.baseline_images.text()),
            ("Shared clear-water override", owner.images.clearwater_images.text()),
            (
                "ColorChecker source",
                str(reference_settings["colorchecker_location"]).replace("_", " "),
            ),
            (
                "ColorChecker reference folder",
                owner.images.colorchecker_images.text() or "not set",
            ),
            (
                "Shared measured-pH buffer folder",
                owner.images.indicator_reference_images.text(),
            ),
            (
                "Shared other-color-state folder",
                owner.images.alternate_color_reference_images.text(),
            ),
        ]
        mappings = owner.images.source_mapping()
        reviewed = sum(
            str(row.get("candidate_status") or "") == "user_confirmed"
            for row in mappings
        )
        overflow_runs = sum(
            bool(row.get("overflow_image_folders") or row.get("overflow_table"))
            for row in mappings
        )
        protocol_runs = sum(
            bool(row.get("protocol_tables") or row.get("protocol_table"))
            for row in mappings
        )
        indicator_reference_runs = sum(
            bool(row.get("indicator_reference_folders")) for row in mappings
        )
        alternate_color_runs = sum(
            bool(row.get("alternate_color_reference_folders")) for row in mappings
        )
        mapping_notes = sum(
            len(row.get("discovery_warnings") or []) for row in mappings
        )
        protocol_layout = str(owner.rig_protocol.protocol_layout.currentData())
        protocol_source = (
            "discovered inside each experiment folder"
            if protocol_layout == "per_run_files"
            else owner.rig_protocol.protocol.text() or "not set"
        )
        sources = "\n".join(
            f"  {label}: {value or 'not set'}" for label, value in source_rows
        )
        self.summary.setPlainText(
            f"Project: {owner.identity.name.text().strip()}\n"
            f"Series ID: {slugify(owner.identity.series_id.text())}\n"
            f"Template: {template.name}\n"
            f"FluidFlower apparatus: {rig}\n"
            f"Protocol source: {protocol_source}\n"
            f"Protocol organization: {owner.rig_protocol.protocol_layout.currentText()}\n"
            f"Overflow: configure per experiment after Data and integrity\n"
            f"DarSIA project folder: {owner.identity.target_path()}\n"
            f"Default generated-results folder: "
            f"{owner.identity.default_result_path()}\n"
            f"Raw source policy: referenced read-only; never used as project storage\n\n"
            f"Registered raw-data sources:\n{sources}\n"
            f"  Light/color-correction model: selected later under Image correction\n"
            f"\nPer-run source review:\n"
            f"  Runs reviewed: {reviewed}/{len(mappings)}\n"
            f"  Runs with protocol workbook: {protocol_runs}/{len(mappings)}\n"
            f"  Runs with overflow source: {overflow_runs}/{len(mappings)}\n"
            f"  Active overflow source can later be changed between images, "
            f"CSV/Excel, and none\n"
            f"  Runs with indicator-reference folders: "
            f"{indicator_reference_runs}/{len(mappings)}\n"
            f"  Runs with other-color-state folders: "
            f"{alternate_color_runs}/{len(mappings)}\n"
            f"  Indicator-reference images: {indicator_reference['entry_count']} "
            f"({str(indicator_reference['status']).replace('_', ' ')})\n"
            f"  Discovery notes requiring inspection: {mapping_notes}\n"
            f"  Default baseline: first image in each CO2 sequence\n"
        )


def build(wizard: Any) -> SummaryPage:
    return SummaryPage(wizard)
