"""Series-identity setup step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QLineEdit, QWizardPage

from darsia.fluidflower.project import default_project_root, slugify
from darsia.fluidflower.templates import EXPERIMENTAL_SERIES_TEMPLATES, template_by_id
from darsia.fluidflower.ui.common import PathField


class SeriesIdentityPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self._applying_template = False
        self.setTitle("Project identity and storage")
        self.setSubTitle(
            "Choose the experimental-series template and where DarSIA will create its "
            "editable project. Raw experiment data are selected separately on the "
            "next page."
        )
        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.template = QComboBox()
        for item in EXPERIMENTAL_SERIES_TEMPLATES:
            self.template.addItem(item.name, item.template_id)
        self.template.currentIndexChanged.connect(self._template_changed)
        layout.addRow("Template", self.template)
        self.template_description = QLabel()
        self.template_description.setWordWrap(True)
        self.template_description.setObjectName("secondaryText")
        layout.addRow("", self.template_description)
        self.name = QLineEdit()
        self.name.setPlaceholderText("For example, pilot experiment series")
        self.name.textChanged.connect(self._identity_edited)
        layout.addRow("Project name", self.name)
        self.series_id = QLineEdit()
        self.series_id.setPlaceholderText("Stable short identifier")
        self.series_id.textChanged.connect(self._identity_edited)
        layout.addRow("Series ID", self.series_id)
        self.created_by = QLineEdit()
        self.created_by.setPlaceholderText("Full name")
        layout.addRow("Created by", self.created_by)
        storage_heading = QLabel("DarSIA project and analysis storage")
        storage_heading.setObjectName("sectionTitle")
        layout.addRow(storage_heading)
        self.storage_explanation = QLabel(
            "DarSIA creates a new project folder here for configuration, review data, "
            "cache, and generated analyses. This is not the folder containing the raw "
            "experiment data; those files are only referenced and remain read-only."
        )
        self.storage_explanation.setWordWrap(True)
        self.storage_explanation.setObjectName("secondaryText")
        layout.addRow("", self.storage_explanation)
        self.parent_folder = PathField(
            directory=True,
            placeholder="Choose the parent folder for this DarSIA project",
        )
        self.parent_folder.setToolTip(
            "The new project gets its own named subfolder at this location. It does "
            "not change or move the raw experiment-data folder selected later."
        )
        self.parent_folder.changed.connect(self._update_preview)
        layout.addRow("Project storage location", self.parent_folder)
        self.target_preview = QLineEdit()
        self.target_preview.setReadOnly(True)
        self.target_preview.setObjectName("pathPreview")
        self.target_preview.setToolTip(
            "The dedicated DarSIA project folder that will be created."
        )
        layout.addRow("New DarSIA project folder", self.target_preview)
        self.result_preview = QLineEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setObjectName("pathPreview")
        self.result_preview.setToolTip(
            "Generated results use this structured project folder by default. Large "
            "outputs, such as mass-analysis NPZ files, can be redirected per analysis "
            "later under Production."
        )
        layout.addRow("Default generated results", self.result_preview)
        self.parent_folder.setText(str(default_project_root()))
        self._template_changed()

    def template_id(self) -> str:
        return str(self.template.currentData())

    def target_path(self) -> Path:
        parent = Path(self.parent_folder.text() or default_project_root())
        return parent / slugify(self.series_id.text() or self.name.text())

    def default_result_path(self) -> Path:
        return self.target_path() / "results"

    def _template_changed(self) -> None:
        template = template_by_id(self.template_id())
        self.template_description.setText(template.description)
        self._applying_template = True
        if not self.name.text().strip() or self.name.property("template_default"):
            self.name.setText(template.name if template.template_id != "custom" else "")
            self.name.setProperty("template_default", True)
        if not self.series_id.text().strip() or self.series_id.property(
            "template_default"
        ):
            value = template.template_id if template.template_id != "custom" else ""
            self.series_id.setText(value)
            self.series_id.setProperty("template_default", True)
        self._applying_template = False
        self._update_preview()

    def _identity_edited(self, _value: str = "") -> None:
        if not self._applying_template:
            self.name.setProperty("template_default", False)
            self.series_id.setProperty("template_default", False)
        self._update_preview()

    def _update_preview(self, _value: str = "") -> None:
        self.target_preview.setText(str(self.target_path()))
        self.result_preview.setText(str(self.default_result_path()))
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802
        return bool(
            self.name.text().strip()
            and self.series_id.text().strip()
            and self.parent_folder.text()
        )


def build(_wizard: Any) -> SeriesIdentityPage:
    return SeriesIdentityPage()
