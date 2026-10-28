"""Resumable new-series setup wizard."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QMessageBox, QWidget, QWizard

from darsia.fluidflower.image_formats import DEFAULT_IMAGE_FILE_POLICY
from darsia.fluidflower.page_registry import (
    FluidFlowerSubpageRegistry,
    default_subpage_registry,
)
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.setup_drafts import SetupDraftStore


class NewSeriesWizard(QWizard):
    project_created = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        draft_id: str = "",
        draft_store: SetupDraftStore | None = None,
        subpage_registry: FluidFlowerSubpageRegistry | None = None,
    ) -> None:
        super().__init__(parent)
        self.subpage_registry = subpage_registry or default_subpage_registry()
        self.draft_store = draft_store or SetupDraftStore()
        self.draft_id = str(draft_id).strip() or uuid.uuid4().hex
        self._restoring_draft = False
        self.setWindowTitle("Set up FluidFlower experimental series")
        self.resize(1240, 800)
        self.setMinimumSize(980, 700)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.subpage_registry.populate_pages("setup_wizard", self, self)
        self._draft_timer = QTimer(self)
        self._draft_timer.setSingleShot(True)
        self._draft_timer.setInterval(350)
        self._draft_timer.timeout.connect(self._save_draft)
        self.currentIdChanged.connect(self._schedule_draft_save)
        for signal in (
            self.identity.template.currentIndexChanged,
            self.identity.name.textChanged,
            self.identity.series_id.textChanged,
            self.identity.created_by.textChanged,
            self.identity.parent_folder.changed,
            self.rig_protocol.rig_type.currentIndexChanged,
            self.rig_protocol.protocol.changed,
            self.rig_protocol.protocol_layout.currentIndexChanged,
            self.images.experiment_images.changed,
            self.images.baseline_images.changed,
            self.images.clearwater_images.changed,
            self.images.colorchecker_images.changed,
            self.images.indicator_reference_images.changed,
            self.images.alternate_color_reference_images.changed,
            self.images.reference_settings_changed,
            self.images.structure.currentIndexChanged,
            self.images.image_file_policy.currentIndexChanged,
            self.images.flat_run_id.textChanged,
            self.images.example_run.currentIndexChanged,
            self.images.run_definitions_changed,
            self.images.layout_template_changed,
            self.images.mapping_table.itemChanged,
        ):
            signal.connect(self._schedule_draft_save)
        if draft_id:
            self._apply_draft(self.draft_store.load(self.draft_id))
        else:
            self._save_draft()

    def _schedule_draft_save(self, _value: Any = None) -> None:
        if not self._restoring_draft:
            self._draft_timer.start()

    def _draft_values(self) -> dict[str, Any]:
        return {
            "wizard_layout_version": 4,
            "template_id": self.identity.template_id(),
            "name": self.identity.name.text(),
            "series_id": self.identity.series_id.text(),
            "created_by": self.identity.created_by.text(),
            "parent_folder": self.identity.parent_folder.text(),
            "rig_type": str(self.rig_protocol.rig_type.currentData()),
            "protocol_path": self.rig_protocol.protocol.text(),
            "protocol_layout": str(self.rig_protocol.protocol_layout.currentData()),
            "experiment_images": self.images.experiment_images.text(),
            "baseline_images": self.images.baseline_images.text(),
            "clearwater_images": self.images.clearwater_images.text(),
            "colorchecker_images": self.images.colorchecker_images.text(),
            "reference_settings": self.images.reference_settings(),
            "image_structure": str(self.images.structure.currentData()),
            "image_file_policy": self.images.image_file_policy_value(),
            "flat_run_id": self.images.flat_run_id.text().strip() or "run",
            "run_definitions": self.images.run_definitions_value(),
            "run_layout_template": self.images.layout_template(),
            "source_mapping": self.images.source_mapping(),
            "current_page": self.currentId(),
        }

    def _save_draft(self) -> None:
        self.draft_store.save(self._draft_values(), draft_id=self.draft_id)

    def _apply_draft(self, record: Mapping[str, Any]) -> None:
        values = dict(record.get("values") or {})
        self._restoring_draft = True
        try:
            for combo, key in (
                (self.identity.template, "template_id"),
                (self.rig_protocol.rig_type, "rig_type"),
                (self.rig_protocol.protocol_layout, "protocol_layout"),
            ):
                index = combo.findData(str(values.get(key) or ""))
                if index >= 0:
                    combo.setCurrentIndex(index)
            self.identity.name.setText(str(values.get("name") or ""))
            self.identity.series_id.setText(str(values.get("series_id") or ""))
            self.identity.created_by.setText(str(values.get("created_by") or ""))
            self.identity.parent_folder.setText(str(values.get("parent_folder") or ""))
            self.rig_protocol.protocol.setText(str(values.get("protocol_path") or ""))
            self.images.experiment_images.setText(
                str(values.get("experiment_images") or "")
            )
            self.images.baseline_images.setText(
                str(values.get("baseline_images") or "")
            )
            self.images.clearwater_images.setText(
                str(values.get("clearwater_images") or "")
            )
            reference_settings = values.get("reference_settings")
            if isinstance(reference_settings, Mapping):
                self.images.set_reference_settings(reference_settings)
            else:
                legacy_colorchecker = str(values.get("colorchecker_images") or "")
                self.images.set_reference_settings(
                    {
                        "radiometric_mode": (
                            "colorchecker_uniform24" if legacy_colorchecker else "none"
                        ),
                        "colorchecker_root": legacy_colorchecker,
                    }
                )
            structure_index = self.images.structure.findData(
                str(values.get("image_structure") or "run_subfolders")
            )
            if structure_index >= 0:
                self.images.structure.setCurrentIndex(structure_index)
            self.images.flat_run_id.setText(str(values.get("flat_run_id") or "run"))
            self.images.set_image_file_policy(
                str(
                    values.get("image_file_policy")
                    or DEFAULT_IMAGE_FILE_POLICY
                )
            )
            run_definitions = [
                dict(row)
                for row in values.get("run_definitions") or []
                if isinstance(row, Mapping)
            ]
            if run_definitions:
                self.images.set_run_definitions(run_definitions)
            run_layout_template = values.get("run_layout_template")
            if isinstance(run_layout_template, Mapping):
                self.images.set_layout_template(dict(run_layout_template))
            source_mapping = [
                dict(row)
                for row in values.get("source_mapping") or []
                if isinstance(row, Mapping)
            ]
            if source_mapping:
                self.images.set_mapping(source_mapping)
            page_id = int(values.get("current_page") or 0)
            layout_version = int(values.get("wizard_layout_version") or 1)
            if layout_version < 2:
                page_id = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}.get(page_id, 0)
            if layout_version < 4:
                page_id = {0: 0, 1: 1, 2: 1, 3: 2}.get(page_id, 0)
            if page_id in self.pageIds():
                self.setStartId(page_id)
        finally:
            self._restoring_draft = False

    def reject(self) -> None:
        self._draft_timer.stop()
        self._save_draft()
        super().reject()

    @staticmethod
    def _optional_path(value: str) -> Path | None:
        return Path(value) if value.strip() else None

    def _shared_protocol_path(self) -> Path | None:
        if str(self.rig_protocol.protocol_layout.currentData()) == "per_run_files":
            return None
        return self._optional_path(self.rig_protocol.protocol.text())

    def accept(self) -> None:
        target = self.identity.target_path()
        reference_settings = self.images.reference_settings()
        indicator_reference = self.images.indicator_reference_contract()
        source_mapping = self.images.source_mapping()
        has_overflow_sources = any(
            row.get("overflow_image_folders")
            or row.get("overflow_tables")
            or row.get("overflow_table")
            for row in source_mapping
        )
        try:
            project = FluidFlowerProject.create(
                target,
                name=self.identity.name.text().strip(),
                series_id=self.identity.series_id.text().strip(),
                template_id=self.identity.template_id(),
                rig_type=str(self.rig_protocol.rig_type.currentData()),
                protocol_path=self._shared_protocol_path(),
                image_sources={
                    "experiment_images": self._optional_path(
                        self.images.experiment_images.text()
                    ),
                    "baseline_images": self._optional_path(
                        self.images.baseline_images.text()
                    ),
                    "clearwater_images": self._optional_path(
                        self.images.clearwater_images.text()
                    ),
                    "colorchecker_images": self._optional_path(
                        reference_settings["colorchecker_root"]
                    ),
                    "indicator_reference_images": self._optional_path(
                        self.images.indicator_reference_images.text()
                    ),
                    "alternate_color_reference_images": self._optional_path(
                        self.images.alternate_color_reference_images.text()
                    ),
                },
                overflow_mode="per_run" if has_overflow_sources else "none",
                created_by=self.identity.created_by.text().strip(),
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self,
                "Could not create project",
                f"{type(exc).__name__}: {exc}",
            )
            return
        project.update_metadata(
            "protocol",
            {
                "organization": str(self.rig_protocol.protocol_layout.currentData()),
                "per_run_source_mapping": str(
                    self.rig_protocol.protocol_layout.currentData()
                )
                == "per_run_files",
                "runs_with_protocol_workbooks": [
                    str(row.get("run_id") or "")
                    for row in self.images.source_mapping()
                    if row.get("protocol_tables") or row.get("protocol_table")
                ],
            },
            updated_by=self.identity.created_by.text().strip(),
            reason="Selected during project setup.",
        )
        mapping_approved = bool(source_mapping) and all(
            str(row.get("candidate_status") or "") == "user_confirmed"
            for row in source_mapping
        )
        project.update_metadata(
            "chemistry",
            {"indicator_response_reference": indicator_reference},
            updated_by=self.identity.created_by.text().strip(),
            reason=(
                "Registered optional pH-buffer and known-state indicator references "
                "during project setup."
            ),
        )
        project.update_metadata(
            "tools",
            {
                "image_reference": {
                    "correction_model_selection": "image_correction_module",
                    "colorchecker_location": reference_settings[
                        "colorchecker_location"
                    ],
                    "colorchecker_root": reference_settings["colorchecker_root"],
                    "indicator_reference_root": reference_settings[
                        "indicator_reference_root"
                    ],
                    "alternate_color_reference_root": reference_settings[
                        "alternate_color_reference_root"
                    ],
                    "buffer_images_are_radiometric_targets": False,
                },
                "source_mapping": {
                    "structure": str(self.images.structure.currentData()),
                    "image_file_policy": self.images.image_file_policy_value(),
                    "run_definitions": self.images.run_definitions_value(),
                    "run_layout_template": self.images.layout_template(),
                    "rows": source_mapping,
                    "baseline_and_clearwater_candidates_approved": mapping_approved,
                    "reference_candidates_approved": mapping_approved,
                },
            },
            updated_by=self.identity.created_by.text().strip(),
            reason="Non-destructive image-source discovery during setup.",
        )
        runs_with_images = [
            str(row.get("run_id") or "")
            for row in source_mapping
            if row.get("overflow_image_folders")
        ]
        runs_with_tables = [
            str(row.get("run_id") or "")
            for row in source_mapping
            if row.get("overflow_tables") or row.get("overflow_table")
        ]
        project.update_metadata(
            "overflow",
            {
                "mode": "per_run" if has_overflow_sources else "none",
                "per_run_source_mapping": True,
                "configuration_stage": "after_data_integrity",
                "run_count": len(source_mapping),
                "runs_with_scale_images": runs_with_images,
                "runs_with_data_tables": runs_with_tables,
            },
            updated_by=self.identity.created_by.text().strip(),
            reason=(
                "Overflow candidates discovered during setup; active sources are "
                "selected after data-integrity review."
            ),
        )
        if source_mapping:
            project.set_runs(
                [
                    {
                        "run_id": row["run_id"],
                        "label": str(row["run_id"]).upper(),
                        "indicator_state": "unknown",
                        "source_mapping": row,
                    }
                    for row in source_mapping
                ],
                updated_by=self.identity.created_by.text().strip(),
                reason="Initialized from the reviewed image-folder mapping.",
            )
        self._draft_timer.stop()
        self.draft_store.delete(self.draft_id)
        self.project_created.emit(project)
        super().accept()
