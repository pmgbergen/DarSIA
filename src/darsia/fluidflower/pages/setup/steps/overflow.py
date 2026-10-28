"""Optional overflow-source setup step."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QWizardPage

from darsia.fluidflower.pages.setup.steps.identity import SeriesIdentityPage
from darsia.fluidflower.templates import template_by_id
from darsia.fluidflower.ui.common import PathField


class OverflowPage(QWizardPage):
    def __init__(self, wizard: Any) -> None:
        super().__init__()
        self.owner = wizard
        self.identity: SeriesIdentityPage = wizard.identity
        self._initialized = False
        self.setTitle("Overflow")
        self.setSubTitle(
            "Overflow is optional and can be omitted, imported, or interpreted "
            "from scale-display images."
        )
        layout = QFormLayout(self)
        self.mode = QComboBox()
        self.mode.addItem("Not relevant", "none")
        self.mode.addItem("Use sources discovered inside each run", "per_run")
        self.mode.addItem("Import CSV or Excel", "table")
        self.mode.addItem("Read scale-display images", "scale_images")
        self.mode.setToolTip(
            "Per-run mode uses the overflow image folders or tables reviewed on "
            "the Image sources page. A shared source is only needed for legacy layouts."
        )
        self.mode.currentIndexChanged.connect(self._mode_changed)
        layout.addRow("Overflow mode", self.mode)
        self.source = PathField(
            directory=True,
            placeholder="Folder containing scale images",
        )
        layout.addRow("Overflow source", self.source)
        self.per_run_summary = QLabel()
        self.per_run_summary.setWordWrap(True)
        self.per_run_summary.setObjectName("secondaryText")
        layout.addRow("", self.per_run_summary)
        self._mode_changed()

    def initializePage(self) -> None:  # noqa: N802
        mappings = self.owner.images.source_mapping()
        discovered = sum(
            bool(row.get("overflow_image_folders") or row.get("overflow_table"))
            for row in mappings
        )
        if not self._initialized:
            preferred_mode = "per_run" if discovered else template_by_id(
                self.identity.template_id()
            ).overflow_mode
            index = self.mode.findData(preferred_mode)
            if index >= 0:
                self.mode.setCurrentIndex(index)
            self._initialized = True
        self.per_run_summary.setText(
            f"Per-run overflow sources are available for {discovered}/{len(mappings)} "
            "reviewed run(s). Runs without usable overflow remain explicitly marked "
            "and must use the documented fallback during calibration."
            if mappings
            else "No reviewed per-run source mapping is available yet."
        )
        self._mode_changed()

    def _mode_changed(self) -> None:
        mode = str(self.mode.currentData())
        self.source.directory = mode != "table"
        self.source.file_filter = (
            "Overflow tables (*.csv *.xlsx *.xls);;All files (*)"
            if mode == "table"
            else "All files (*)"
        )
        self.source.setEnabled(mode in {"table", "scale_images"})


def build(wizard: Any) -> OverflowPage:
    return OverflowPage(wizard)
