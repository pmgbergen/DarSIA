"""Incremental run-source management for established FluidFlower projects."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.image_formats import (
    IMAGE_FILE_POLICY_ALL,
    IMAGE_FILE_POLICY_PROCESSED,
    IMAGE_FILE_POLICY_PROCESSED_ONLY,
    IMAGE_FILE_POLICY_RAW,
    IMAGE_FILE_POLICY_RAW_ONLY,
)
from darsia.fluidflower.pages.setup.global_run_structure import (
    GlobalRunStructureDialog,
)
from darsia.fluidflower.pages.setup.source_layout_dialog import (
    SourceMappingDetailsDialog,
)
from darsia.fluidflower.pages.setup.steps.image_sources import ImageSourcesPage
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.source_discovery import discover_image_source_mapping
from darsia.fluidflower.templates import RIG_TYPES
from darsia.fluidflower.ui.common import set_state_label


def _project_run_rows(project: FluidFlowerProject) -> list[dict[str, Any]]:
    return [
        deepcopy(dict(row))
        for row in project.manifest.get("runs") or []
        if isinstance(row, Mapping)
    ]


def _project_mapping_config(project: FluidFlowerProject) -> dict[str, Any]:
    tools = project.manifest.get("tools") or {}
    config = tools.get("source_mapping") if isinstance(tools, Mapping) else {}
    return deepcopy(dict(config)) if isinstance(config, Mapping) else {}


def _project_run_folders(project: FluidFlowerProject) -> set[str]:
    folders: set[str] = set()
    for row in _project_run_rows(project):
        mapping = row.get("source_mapping") or {}
        folder = str(mapping.get("experiment_folder") or "").strip()
        if folder:
            folders.add(str(Path(folder).expanduser().resolve()))
    return folders


def _registered_source_path(project: FluidFlowerProject, kind: str) -> str:
    matches = [
        str(row.get("path") or "")
        for row in project.source_manifest().get("sources") or []
        if isinstance(row, Mapping)
        and str(row.get("kind") or "") == kind
        and not str(row.get("run_id") or "")
    ]
    return matches[-1] if matches else ""


def _default_series_root(project: FluidFlowerProject) -> str:
    registered = _registered_source_path(project, "experiment_images")
    if registered and Path(registered).is_dir():
        return registered
    parents = [Path(folder).parent for folder in _project_run_folders(project)]
    if parents and all(parent == parents[0] for parent in parents):
        return str(parents[0])
    return ""


def _source_count(mapping: Mapping[str, Any]) -> int:
    paths: set[str] = set()
    for key in (
        "experiment_folder",
        "baseline_candidate",
        "clearwater_candidate",
        "protocol_table",
        "overflow_table",
    ):
        value = str(mapping.get(key) or "").strip()
        if value:
            paths.add(value.casefold())
    for key in (
        "experiment_folders",
        "clearwater_folders",
        "indicator_reference_folders",
        "alternate_color_reference_folders",
        "protocol_tables",
        "overflow_image_folders",
        "overflow_tables",
    ):
        for value in mapping.get(key) or []:
            text = str(value).strip()
            if text:
                paths.add(text.casefold())
    return len(paths)


def _editable_mapping_signature(mapping: Mapping[str, Any]) -> str:
    """Return a stable comparison value without generated source-record IDs."""

    comparable = deepcopy(dict(mapping))
    comparable.pop("source_record_ids", None)
    comparable.pop("source_revision_id", None)
    return json.dumps(comparable, sort_keys=True, ensure_ascii=False, default=str)


def _mapping_paths(mapping: Mapping[str, Any]) -> list[Path]:
    """Return paths that must belong to the selected run itself."""

    values: list[str] = []
    for key in (
        "protocol_table",
        "overflow_table",
    ):
        value = str(mapping.get(key) or "").strip()
        if value:
            values.append(value)
    for key in (
        "experiment_folders",
        "protocol_tables",
        "overflow_image_folders",
        "overflow_tables",
    ):
        values.extend(
            str(value).strip() for value in mapping.get(key) or [] if str(value).strip()
        )
    return [Path(value).expanduser().resolve() for value in dict.fromkeys(values)]


def _is_within(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
    except ValueError:
        return False
    return True


def _cross_run_mapping_issues(
    project: FluidFlowerProject,
    run_id: str,
    mapping: Mapping[str, Any],
) -> tuple[str, ...]:
    """Reject a single-run edit that accidentally absorbs neighboring runs."""

    selected_root_text = str(mapping.get("experiment_folder") or "").strip()
    if not selected_root_text:
        return ("The experiment folder is missing.",)
    selected_root = Path(selected_root_text).expanduser().resolve()
    paths = _mapping_paths(mapping)
    issues: list[str] = []
    for row in _project_run_rows(project):
        other_id = str(row.get("run_id") or "")
        if other_id == run_id:
            continue
        other_mapping = row.get("source_mapping") or {}
        other_root_text = str(other_mapping.get("experiment_folder") or "").strip()
        if not other_root_text:
            continue
        other_root = Path(other_root_text).expanduser().resolve()
        if _is_within(other_root, selected_root):
            issues.append(
                f"The selected folder also contains registered experiment "
                f"{other_id.upper()}: {other_root}"
            )
            continue
        foreign_paths = [path for path in paths if _is_within(path, other_root)]
        if foreign_paths:
            issues.append(
                f"{run_id.upper()} currently includes source data from "
                f"{other_id.upper()}: {foreign_paths[0]}"
            )
    return tuple(dict.fromkeys(issues))


class RunImportPreviewDialog(QDialog):
    """Final, read-only preview before run metadata is committed."""

    def __init__(
        self,
        mappings: Sequence[Mapping[str, Any]],
        parent: Any = None,
        *,
        title: str = "Review experiment import",
        confirm_label: str = "Add experiments",
        note_text: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.mappings = [deepcopy(dict(mapping)) for mapping in mappings]
        self.setWindowTitle(title)
        self.resize(1220, 560)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "This is the exact source mapping that will be registered. Original "
            "files are referenced read-only. Existing runs, approvals, and result "
            "files are not replaced."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)

        self.table = QTableWidget(len(self.mappings), 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Run",
                "CO2 images",
                "CO2 folders",
                "Registered sources",
                "Baseline",
                "Clear-water",
                "Measured-pH buffers",
                "Other color states",
                "Protocol",
                "Overflow",
                "Discovery notes",
            ]
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in (0, 1, 2, 3, 8, 9, 10):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for column in (4, 5, 6, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        for row_index, mapping in enumerate(self.mappings):
            overflow_tables = mapping.get("overflow_tables") or []
            overflow_image_count = len(mapping.get("overflow_image_folders") or [])
            overflow_table_count = len(
                {
                    str(value)
                    for value in [
                        *overflow_tables,
                        mapping.get("overflow_table"),
                    ]
                    if str(value or "").strip()
                }
            )
            overflow_parts = []
            if overflow_image_count:
                overflow_parts.append(f"{overflow_image_count} image folder(s)")
            if overflow_table_count:
                overflow_parts.append(f"{overflow_table_count} table(s)")
            values = (
                str(mapping.get("run_id") or "").upper(),
                int(mapping.get("image_count") or 0),
                len(mapping.get("experiment_folders") or []),
                _source_count(mapping),
                str(mapping.get("baseline_candidate") or "Not found"),
                str(mapping.get("clearwater_candidate") or "Not found"),
                len(mapping.get("indicator_reference_folders") or []),
                len(mapping.get("alternate_color_reference_folders") or []),
                str(mapping.get("protocol_table") or "Not found"),
                " + ".join(overflow_parts) if overflow_parts else "Not registered",
                f"{len(mapping.get('discovery_warnings') or [])} note(s)",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 10:
                    item.setToolTip(
                        "\n".join(mapping.get("discovery_warnings") or [])
                        or "No discovery notes"
                    )
                else:
                    item.setToolTip(str(value))
                self.table.setItem(row_index, column, item)
        self.table.cellDoubleClicked.connect(self._inspect_mapping)
        layout.addWidget(self.table, 1)

        details_help = QLabel(
            "Double-click an experiment to inspect every matched folder and the "
            "complete discovery notes before confirming."
        )
        details_help.setWordWrap(True)
        details_help.setObjectName("secondaryText")
        layout.addWidget(details_help)

        note = QLabel(
            note_text
            or (
                "After import, each new run starts at run-specific data-integrity "
                "review. Previously processed runs retain their current state."
            )
        )
        note.setWordWrap(True)
        note.setObjectName("secondaryText")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        confirm = buttons.addButton(
            confirm_label, QDialogButtonBox.ButtonRole.AcceptRole
        )
        confirm.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _inspect_mapping(self, row: int, _column: int) -> None:
        if 0 <= row < len(self.mappings):
            SourceMappingDetailsDialog(self.mappings[row], self).exec()


class AddRunsDialog(QDialog):
    """Discover, review, and append experiments to an existing series."""

    def __init__(self, project: FluidFlowerProject, parent: Any = None) -> None:
        super().__init__(parent)
        self.project = project
        self.report: dict[str, Any] | None = None
        self.setWindowTitle(f"Add experiments to {project.name}")
        self.resize(1480, 900)
        layout = QVBoxLayout(self)
        heading = QLabel(f"Add experiments to {project.name}")
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        explanation = QLabel(
            f"The project currently contains {len(_project_run_rows(project))} "
            "experiment(s). Select the series folder, include only new experiment "
            "folders, and review every proposed mapping. Already registered folders "
            "are locked."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)

        self.sources = ImageSourcesPage()
        self.sources.indicator_system_override = str(
            (project.manifest.get("chemistry") or {}).get("indicator_system")
            or "unspecified"
        )
        existing_rows = _project_run_rows(project)
        self.sources.set_reserved_runs(
            run_ids={str(row.get("run_id") or "") for row in existing_rows},
            folders=_project_run_folders(project),
        )
        root = _default_series_root(project)
        if root:
            self.sources.experiment_images.setText(root)
        self.sources.baseline_images.setText(
            _registered_source_path(project, "baseline_images")
        )
        self.sources.clearwater_images.setText(
            _registered_source_path(project, "clearwater_images")
        )
        reference_tools = dict(
            ((project.manifest.get("tools") or {}).get("image_reference") or {})
        )
        reference_contract = dict(
            (
                (project.manifest.get("chemistry") or {}).get(
                    "indicator_response_reference"
                )
                or {}
            )
        )
        colorchecker_root = _registered_source_path(project, "colorchecker_images")
        self.sources.set_reference_settings(
            {
                "colorchecker_location": str(
                    reference_tools.get("colorchecker_location") or ""
                ),
                "radiometric_mode": str(
                    reference_tools.get("radiometric_mode")
                    or ("colorchecker_uniform24" if colorchecker_root else "none")
                ),
                "colorchecker_root": colorchecker_root,
                "indicator_reference_root": _registered_source_path(
                    project, "indicator_reference_images"
                ),
                "alternate_color_reference_root": _registered_source_path(
                    project, "alternate_color_reference_images"
                ),
                "indicator_reference_entries": list(
                    reference_contract.get("entries") or []
                ),
            }
        )
        config = _project_mapping_config(project)
        self.sources.set_layout_template(config.get("run_layout_template") or {})
        self.sources.set_image_file_policy(project.image_file_policy())
        self.sources.image_file_policy.setEnabled(False)
        self.sources.image_file_policy.setToolTip(
            "This is a global series setting. Review the established project setup "
            "to change it for every experiment before adding new experiments."
        )
        layout.addWidget(self.sources, 1)

        provenance = QFormLayout()
        self.actor = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.actor.setPlaceholderText("Name of the person adding the experiments")
        provenance.addRow("Added by", self.actor)
        layout.addLayout(provenance)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.add_button = buttons.addButton(
            "Review and add experiments...",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.add_button.clicked.connect(self._review_and_add)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.sources.completeChanged.connect(self._refresh_actions)
        self.actor.textChanged.connect(self._refresh_actions)
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        self.add_button.setEnabled(
            bool(self.actor.text().strip()) and self.sources.isComplete()
        )

    def _review_and_add(self) -> None:
        mappings = self.sources.source_mapping()
        issues = self.project.validate_run_additions(mappings)
        if issues:
            QMessageBox.warning(
                self,
                "Import is not ready",
                "Resolve these items before importing:\n\n" + "\n".join(issues),
            )
            return
        preview = RunImportPreviewDialog(
            mappings,
            self,
            confirm_label=f"Add {len(mappings)} experiment(s)",
        )
        if preview.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.report = self.project.append_runs(
                mappings,
                updated_by=self.actor.text().strip(),
                reason="Added through the incremental series-import dialog.",
            )
            self.project.update_metadata(
                "chemistry",
                {
                    "indicator_response_reference": (
                        self.sources.indicator_reference_contract()
                    )
                },
                updated_by=self.actor.text().strip(),
                reason="Updated indicator-reference metadata while adding experiments.",
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self,
                "Could not add experiments",
                "No existing run or result was replaced.\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return
        QMessageBox.information(
            self,
            "Experiments added",
            f"Added {self.report['run_count']} experiment(s): "
            + ", ".join(str(value).upper() for value in self.report["run_ids"])
            + "\n\nRun-specific data-integrity review is now required.",
        )
        self.accept()


class EstablishedSourceSetupEditor(QWidget):
    """Edit an established series and version its run-source mappings."""

    project_changed = Signal()
    individual_mapping_saved = Signal()

    def __init__(
        self,
        project: FluidFlowerProject,
        parent: Any = None,
        *,
        initial_run_id: str = "",
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.rows = _project_run_rows(project)
        self.report: dict[str, Any] | None = None
        self.changed = False
        self._loaded_run_id = ""
        self._loaded_mapping_signature = ""
        layout = QVBoxLayout(self)
        heading = QLabel("Review or correct the established source setup")
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        explanation = QLabel(
            "Use global settings for the normal series workflow: one representative "
            "experiment defines the expected folder structure for the selected runs. "
            "Open individual overrides only when one experiment genuinely differs. "
            "Saved changes create new revisions; raw files are never modified."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)

        actor_row = QHBoxLayout()
        actor_row.addWidget(QLabel("Updated by"))
        self.actor = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.actor.setPlaceholderText("Name of the person changing the setup")
        actor_row.addWidget(self.actor, 1)
        layout.addLayout(actor_row)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Setup scope"))
        scope_row.setSpacing(0)
        self.scope_buttons = QButtonGroup(self)
        self.scope_buttons.setExclusive(True)
        self.global_scope_button = QPushButton("Global settings")
        self.global_scope_button.setObjectName("scopeModeButton")
        self.global_scope_button.setCheckable(True)
        self.scope_buttons.addButton(self.global_scope_button)
        scope_row.addWidget(self.global_scope_button)
        self.individual_scope_button = QPushButton("Individual experiment overrides")
        self.individual_scope_button.setObjectName("scopeModeButton")
        self.individual_scope_button.setCheckable(True)
        self.scope_buttons.addButton(self.individual_scope_button)
        scope_row.addWidget(self.individual_scope_button)
        scope_row.addStretch(1)
        layout.addLayout(scope_row)
        self.scope_explanation = QLabel()
        self.scope_explanation.setWordWrap(True)
        self.scope_explanation.setObjectName("secondaryText")
        layout.addWidget(self.scope_explanation)

        self.content_scroll = QScrollArea()
        self.content_scroll.setObjectName("establishedSetupScroll")
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.content_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.content_widget = QWidget()
        self.content_widget.setMinimumWidth(1160)
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.content_scroll.setWidget(self.content_widget)
        layout.addWidget(self.content_scroll, 1)

        self.global_group = QGroupBox("Global series settings")
        global_layout = QVBoxLayout(self.global_group)
        global_explanation = QLabel(
            "These values belong to the experimental series and therefore apply to "
            "every included experiment. They are not stored separately for AC17, "
            "AC18, or any other run."
        )
        global_explanation.setWordWrap(True)
        global_explanation.setObjectName("secondaryText")
        global_layout.addWidget(global_explanation)
        series_settings = QFormLayout()
        self.rig_type = QComboBox()
        for key, label in RIG_TYPES.items():
            self.rig_type.addItem(label, key)
        current_rig_type = str((project.manifest.get("rig") or {}).get("type") or "")
        rig_index = self.rig_type.findData(current_rig_type)
        if rig_index >= 0:
            self.rig_type.setCurrentIndex(rig_index)
        rig_row = QHBoxLayout()
        rig_row.addWidget(self.rig_type, 1)
        self.apply_rig_button = QPushButton("Apply apparatus-class change...")
        self.apply_rig_button.clicked.connect(self._apply_rig_type)
        rig_row.addWidget(self.apply_rig_button)
        series_settings.addRow("Apparatus class (all experiments)", rig_row)
        self.image_file_policy = QComboBox()
        self.image_file_policy.addItem(
            "Prefer JPEG/TIFF/PNG; ignore paired RAW files (recommended)",
            IMAGE_FILE_POLICY_PROCESSED,
        )
        self.image_file_policy.addItem(
            "Prefer camera RAW; ignore paired processed copies",
            IMAGE_FILE_POLICY_RAW,
        )
        self.image_file_policy.addItem(
            "Use both processed and RAW files (advanced)",
            IMAGE_FILE_POLICY_ALL,
        )
        self.image_file_policy.addItem(
            "Only camera RAW; never JPEG/TIFF/PNG, no fallback",
            IMAGE_FILE_POLICY_RAW_ONLY,
        )
        self.image_file_policy.addItem(
            "Only JPEG/TIFF/PNG; never camera RAW, no fallback",
            IMAGE_FILE_POLICY_PROCESSED_ONLY,
        )
        saved_image_policy = project.image_file_policy()
        image_policy_index = self.image_file_policy.findData(saved_image_policy)
        if image_policy_index >= 0:
            self.image_file_policy.setCurrentIndex(image_policy_index)
        self.image_file_policy.setToolTip(
            "Series-wide choice for paired camera files. A changed choice is saved "
            "only through 'Review or replace saved structure', which rebuilds and "
            "previews every affected experiment before replacing active mappings."
        )
        series_settings.addRow("Image files (all experiments)", self.image_file_policy)
        self.run_count_label = QLabel()
        self.run_count_label.setObjectName("secondaryText")
        registered_row = QHBoxLayout()
        registered_row.addWidget(self.run_count_label, 1)
        self.add_runs_button = QPushButton("Add experiments...")
        self.add_runs_button.clicked.connect(self._add_experiments)
        registered_row.addWidget(self.add_runs_button)
        series_settings.addRow("Registered experiments", registered_row)
        structure_row = QHBoxLayout()
        self.structure_summary = QLabel()
        self.structure_summary.setWordWrap(True)
        structure_row.addWidget(self.structure_summary, 1)
        self.review_global_structure = QPushButton(
            "Review and apply expected structure..."
        )
        self.review_global_structure.setToolTip(
            "Use the same structure editor as first-time setup, then choose all or "
            "selected registered experiments for re-evaluation."
        )
        self.review_global_structure.clicked.connect(self._review_global_run_structure)
        structure_row.addWidget(self.review_global_structure)
        series_settings.addRow("Expected folder structure", structure_row)
        global_layout.addLayout(series_settings)
        content_layout.addWidget(self.global_group)

        self.individual_group = QGroupBox("Settings for one selected experiment")
        individual_layout = QVBoxLayout(self.individual_group)
        individual_explanation = QLabel(
            "Only the selected experiment is edited below. Changing its folders or "
            "folder roles does not alter the mappings of the other experiments."
        )
        individual_explanation.setWordWrap(True)
        individual_explanation.setObjectName("secondaryText")
        individual_layout.addWidget(individual_explanation)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Selected experiment"))
        self.run_selector = QComboBox()
        selector_row.addWidget(self.run_selector, 1)
        self.revision_label = QLabel()
        self.revision_label.setObjectName("secondaryText")
        selector_row.addWidget(self.revision_label)
        self.reload_mapping_button = QPushButton("Reload saved mapping")
        self.reload_mapping_button.clicked.connect(self._load_selected_run)
        selector_row.addWidget(self.reload_mapping_button)
        individual_layout.addLayout(selector_row)

        role_help = QLabel(
            "To correct folder roles, select the mapping row below and choose "
            "'Categorize selected run folders...'. Multiple folders can be assigned "
            "to the same role. Paths in the mapping table can also be edited directly."
        )
        role_help.setWordWrap(True)
        role_help.setObjectName("integrationNotice")
        individual_layout.addWidget(role_help)

        self.sources = ImageSourcesPage(embedded=True)
        self.sources.indicator_system_override = str(
            (project.manifest.get("chemistry") or {}).get("indicator_system")
            or "unspecified"
        )
        update_reference_tools = dict(
            ((project.manifest.get("tools") or {}).get("image_reference") or {})
        )
        update_reference_contract = dict(
            (
                (project.manifest.get("chemistry") or {}).get(
                    "indicator_response_reference"
                )
                or {}
            )
        )
        update_colorchecker_root = _registered_source_path(
            project, "colorchecker_images"
        )
        self.sources.set_reference_settings(
            {
                "colorchecker_location": str(
                    update_reference_tools.get("colorchecker_location") or ""
                ),
                "radiometric_mode": str(
                    update_reference_tools.get("radiometric_mode")
                    or (
                        "colorchecker_uniform24" if update_colorchecker_root else "none"
                    )
                ),
                "colorchecker_root": update_colorchecker_root,
                "indicator_reference_root": _registered_source_path(
                    project, "indicator_reference_images"
                ),
                "alternate_color_reference_root": _registered_source_path(
                    project, "alternate_color_reference_images"
                ),
                "indicator_reference_entries": list(
                    update_reference_contract.get("entries") or []
                ),
            }
        )
        single_index = self.sources.structure.findData("single_run")
        self.sources.structure.setCurrentIndex(single_index)
        self.sources.structure.setEnabled(False)
        self.sources.structure.setToolTip(
            "An established mapping is edited one experiment at a time. Use "
            "'Add experiments...' to scan the parent series folder."
        )
        self.sources.discover_sources.setText("Re-scan this experiment")
        self.sources.discover_sources.setToolTip(
            "Rebuild only the selected experiment's mapping from the folder shown "
            "above. A series parent folder is rejected here."
        )
        self.sources.mapping_table.setMinimumHeight(300)
        individual_layout.addWidget(self.sources)

        self.pending_change_status = QLabel()
        self.pending_change_status.setWordWrap(True)
        individual_layout.addWidget(self.pending_change_status)
        content_layout.addWidget(self.individual_group)

        self.included_group = QGroupBox("Experiments included in this project")
        included_layout = QVBoxLayout(self.included_group)
        included_help = QLabel(
            "This is the complete project membership, not a discovery proposal. "
            "Click a row to review that experiment above."
        )
        included_help.setWordWrap(True)
        included_help.setObjectName("secondaryText")
        included_layout.addWidget(included_help)
        self.included_table = QTableWidget(0, 6)
        self.included_table.setHorizontalHeaderLabels(
            [
                "Experiment",
                "CO2 images",
                "Source revision",
                "Folder structure",
                "Data review",
                "Run folder",
            ]
        )
        self.included_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.included_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.included_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.included_table.verticalHeader().setVisible(False)
        included_header = self.included_table.horizontalHeader()
        for column in (0, 1, 2, 3, 4):
            included_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        included_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.included_table.setMinimumHeight(320)
        self.included_table.cellClicked.connect(self._included_run_clicked)
        included_layout.addWidget(self.included_table)
        content_layout.addWidget(self.included_group)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.update_button = QPushButton("Review changes for selected experiment...")
        self.update_button.clicked.connect(self._review_and_update)
        actions.addWidget(self.update_button)
        layout.addLayout(actions)
        self.run_selector.currentIndexChanged.connect(self._load_selected_run)
        self.rig_type.currentIndexChanged.connect(self._refresh_actions)
        self.image_file_policy.currentIndexChanged.connect(self._refresh_actions)
        self.sources.completeChanged.connect(self._refresh_actions)
        self.actor.textChanged.connect(self._refresh_actions)
        self.global_scope_button.clicked.connect(lambda: self.set_setup_scope("global"))
        self.individual_scope_button.clicked.connect(
            lambda: self.set_setup_scope("individual")
        )
        self._reload_rows(initial_run_id)
        self.set_setup_scope("individual" if initial_run_id else "global")

    def set_setup_scope(self, scope: str) -> None:
        """Show global series settings or the exceptional per-run editor."""

        if scope not in {"global", "individual"}:
            raise ValueError(f"Unknown setup scope: {scope}")
        global_scope = scope == "global"
        self.global_scope_button.setChecked(global_scope)
        self.individual_scope_button.setChecked(not global_scope)
        self.global_group.setVisible(global_scope)
        self.individual_group.setVisible(not global_scope)
        self.update_button.setVisible(not global_scope)
        self.scope_explanation.setText(
            (
                "Recommended for structurally consistent series. Review shared "
                "settings, choose one representative experiment, and apply its "
                "expected folder structure to all or selected experiments."
            )
            if global_scope
            else (
                "Use this only for an exception. Changes below affect the selected "
                "experiment alone and create a new run-specific source revision."
            )
        )

    def refresh(self) -> None:
        """Reload the saved project state while retaining the selected experiment."""

        self._reload_rows(str(self.run_selector.currentData() or ""))

    def select_run(self, run_id: str) -> None:
        """Select a run when another workflow links back to source setup."""

        index = self.run_selector.findData(str(run_id or "").strip().lower())
        if index >= 0:
            self.run_selector.setCurrentIndex(index)
            self.set_setup_scope("individual")

    def _mark_project_changed(self) -> None:
        self.changed = True
        self.project_changed.emit()

    def _reload_rows(self, selected_run: str = "") -> None:
        current = selected_run or str(self.run_selector.currentData() or "")
        self.rows = _project_run_rows(self.project)
        self.run_selector.blockSignals(True)
        self.run_selector.clear()
        for row in self.rows:
            run_id = str(row.get("run_id") or "")
            self.run_selector.addItem(str(row.get("label") or run_id.upper()), run_id)
        restored = self.run_selector.findData(current)
        if restored >= 0:
            self.run_selector.setCurrentIndex(restored)
        self.run_selector.blockSignals(False)
        self.run_count_label.setText(
            f"{len(self.rows)} experiment(s) are currently registered in this project."
        )
        self._refresh_global_structure_summary()
        self._populate_included_table()
        self._load_selected_run()

    def _refresh_global_structure_summary(self) -> None:
        config = _project_mapping_config(self.project)
        template = config.get("run_layout_template") or {}
        if not template:
            set_state_label(
                self.structure_summary,
                "incomplete",
                "INCOMPLETE: No shared expected folder structure is saved.",
            )
            self.review_global_structure.setText(
                "Define and save expected structure..."
            )
            return
        revision = int(config.get("run_layout_template_revision") or 1)
        applied = sum(
            int((row.get("source_mapping") or {}).get("layout_template_revision") or 0)
            == revision
            for row in self.rows
        )
        example = str(template.get("example_run_name") or "an earlier experiment")
        roles = len(template.get("folders") or [])
        image_policy = str(
            config.get("image_file_policy")
            or template.get("image_file_policy")
            or self.project.image_file_policy()
        )
        set_state_label(
            self.structure_summary,
            "saved",
            f"SAVED r{revision}: Learned from {example}; {roles} folder role(s). "
            f"Application is recorded for {applied} of {len(self.rows)} "
            f"experiment(s). Image-file policy: {image_policy}.",
        )
        self.review_global_structure.setText("Review or replace saved structure...")

    def _populate_included_table(self) -> None:
        self.included_table.setRowCount(len(self.rows))
        config = _project_mapping_config(self.project)
        current_layout_revision = int(
            config.get("run_layout_template_revision")
            or (1 if config.get("run_layout_template") else 0)
        )
        for row_index, row in enumerate(self.rows):
            mapping = row.get("source_mapping") or {}
            workflow = row.get("workflow") or {}
            integrity = workflow.get("data_integrity") or {}
            run_id = str(row.get("run_id") or "")
            applied_layout_revision = int(mapping.get("layout_template_revision") or 0)
            layout_status = (
                f"Current (r{applied_layout_revision})"
                if current_layout_revision
                and applied_layout_revision == current_layout_revision
                else (
                    f"Older (r{applied_layout_revision})"
                    if applied_layout_revision
                    else "Not recorded"
                )
            )
            values = (
                str(row.get("label") or run_id.upper()),
                int(mapping.get("image_count") or 0),
                int(row.get("source_revision") or 1),
                layout_status,
                str(integrity.get("status") or "not started").replace("_", " "),
                str(mapping.get("experiment_folder") or ""),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, run_id)
                item.setToolTip(str(value))
                self.included_table.setItem(row_index, column, item)

    def _included_run_clicked(self, row: int, _column: int) -> None:
        item = self.included_table.item(row, 0)
        run_id = str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""
        index = self.run_selector.findData(run_id)
        if index >= 0:
            self.run_selector.setCurrentIndex(index)

    def _select_included_run(self, run_id: str) -> None:
        for row in range(self.included_table.rowCount()):
            item = self.included_table.item(row, 0)
            if item and str(item.data(Qt.ItemDataRole.UserRole) or "") == run_id:
                self.included_table.selectRow(row)
                return

    def _add_experiments(self) -> None:
        current = str(self.run_selector.currentData() or "")
        dialog = AddRunsDialog(self.project, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._mark_project_changed()
        self._reload_rows(current)

    def _review_global_run_structure(self) -> None:
        actor = self.actor.text().strip()
        if not actor:
            QMessageBox.warning(
                self,
                "Name required",
                "Enter the name of the person reviewing the project setup first.",
            )
            return
        config = _project_mapping_config(self.project)
        requested_image_policy = str(self.image_file_policy.currentData() or "")
        active_image_policy = self.project.image_file_policy()
        template = deepcopy(dict(config.get("run_layout_template") or {}))
        template["image_file_policy"] = requested_image_policy
        dialog = GlobalRunStructureDialog(
            self.rows,
            self,
            template=template,
            template_revision=int(
                config.get("run_layout_template_revision")
                or (1 if config.get("run_layout_template") else 0)
            ),
            current_run_id=str(self.run_selector.currentData() or ""),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_ids = dialog.selected_run_ids()
        template = dialog.layout_template()
        template["image_file_policy"] = requested_image_policy
        if requested_image_policy != active_image_policy and set(selected_ids) != {
            str(row.get("run_id") or "") for row in self.rows
        }:
            QMessageBox.warning(
                self,
                "Image choice applies to the full series",
                "The image-file preference is a global series setting. Select every "
                "registered experiment when changing it, then review the complete "
                "mapping preview before saving.",
            )
            return
        rows_by_id = {str(row.get("run_id") or ""): row for row in self.rows}
        mappings: list[dict[str, Any]] = []
        errors: list[str] = []
        progress = QProgressDialog(
            "Re-evaluating experiment folders...",
            "Cancel",
            0,
            len(selected_ids),
            self,
        )
        progress.setWindowTitle("Build source-mapping preview")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        protocol_layout = str(
            (self.project.manifest.get("protocol") or {}).get("organization") or ""
        )
        shared_baseline = _registered_source_path(self.project, "baseline_images")
        shared_clearwater = _registered_source_path(self.project, "clearwater_images")
        for index, run_id in enumerate(selected_ids, start=1):
            if progress.wasCanceled():
                progress.close()
                return
            progress.setLabelText(f"Inspecting {run_id.upper()}...")
            QApplication.processEvents()
            row = rows_by_id.get(run_id) or {}
            current_mapping = row.get("source_mapping") or {}
            root = Path(str(current_mapping.get("experiment_folder") or ""))
            try:
                discovered = discover_image_source_mapping(
                    root,
                    structure="single_run",
                    baseline_root=Path(shared_baseline) if shared_baseline else None,
                    clearwater_root=(
                        Path(shared_clearwater) if shared_clearwater else None
                    ),
                    flat_run_id=run_id,
                    layout_template=template,
                    protocol_layout=protocol_layout,
                    image_file_policy=requested_image_policy,
                )
                if len(discovered) != 1:
                    raise ValueError("The run did not produce one source mapping.")
                mapping = dict(discovered[0])
                mapping["candidate_status"] = "user_confirmed"
                ownership_issues = _cross_run_mapping_issues(
                    self.project, run_id, mapping
                )
                if ownership_issues:
                    errors.extend(ownership_issues)
                mappings.append(mapping)
            except Exception as exc:  # noqa: BLE001 - visible GUI boundary
                errors.append(f"{run_id.upper()}: {type(exc).__name__}: {exc}")
            progress.setValue(index)
        progress.close()
        errors.extend(self.project.validate_run_source_updates(mappings))
        errors = list(dict.fromkeys(errors))
        if errors:
            QMessageBox.warning(
                self,
                "The expected structure could not be applied",
                "Resolve these source-mapping problems before saving:\n\n"
                + "\n".join(errors[:12])
                + (
                    f"\n\n...and {len(errors) - 12} more problem(s)."
                    if len(errors) > 12
                    else ""
                ),
            )
            return

        preview = RunImportPreviewDialog(
            mappings,
            self,
            title="Review mappings created from the expected folder structure",
            confirm_label=(f"Save structure and update {len(mappings)} experiment(s)"),
            note_text=(
                f"Scope: {len(mappings)} of {len(self.rows)} registered experiments: "
                + ", ".join(run_id.upper() for run_id in selected_ids)
                + ". Only these experiments receive new source revisions. Their "
                "previous mappings and all raw files are retained. Any changed source "
                "mapping reopens Data and integrity and makes the previous report "
                "historical."
            ),
        )
        if preview.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            report = self.project.apply_run_layout_template(
                mappings,
                template,
                updated_by=actor,
                reason=(
                    "Expected folder structure reviewed and applied through project "
                    "setup."
                ),
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self,
                "Could not update experiment mappings",
                "The previous mappings remain active.\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return
        self._mark_project_changed()
        current = str(self.run_selector.currentData() or "")
        self._reload_rows(current)
        changed_ids = [str(run_id).upper() for run_id in report["changed_run_ids"]]
        changed_scope = (
            f"all {len(self.rows)} registered experiments"
            if len(changed_ids) == len(self.rows)
            else f"{len(changed_ids)} selected experiment(s)"
        )
        changed_list = ", ".join(changed_ids) or "None"
        QMessageBox.information(
            self,
            "Expected folder structure applied",
            f"Saved structure revision r{report['template_revision']}. "
            f"Updated {changed_scope}; {len(report['unchanged_run_ids'])} already "
            f"matched.\n\nChanged source mappings: {changed_list}.\n\n"
            "Data and integrity now requires a new report for every changed "
            "experiment. The previous report remains available as history.",
        )

    def _apply_rig_type(self) -> None:
        requested = str(self.rig_type.currentData() or "")
        current = str((self.project.manifest.get("rig") or {}).get("type") or "")
        if requested == current:
            return
        choice = QMessageBox.question(
            self,
            "Change FluidFlower apparatus class",
            f"Change the apparatus class from {RIG_TYPES[current]} to "
            f"{RIG_TYPES[requested]}?\n\nExisting geometry values are retained. If "
            "geometry or downstream analyses were already approved, they will require "
            "renewed review.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        try:
            report = self.project.update_rig_type(
                requested,
                updated_by=self.actor.text().strip(),
                reason="Apparatus class corrected through Review project setup.",
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self,
                "Could not update apparatus class",
                f"{type(exc).__name__}: {exc}",
            )
            return
        self._mark_project_changed()
        self._refresh_actions()
        affected = sum(
            bool(value) for value in report["run_modules_invalidated"].values()
        )
        QMessageBox.information(
            self,
            "Apparatus class updated",
            f"The active apparatus class is now {RIG_TYPES[requested]}. "
            f"Geometry review was reopened for {affected} experiment(s).",
        )

    def _selected_row(self) -> dict[str, Any] | None:
        run_id = str(self.run_selector.currentData() or "")
        return next(
            (row for row in self.rows if str(row.get("run_id") or "") == run_id),
            None,
        )

    def _load_selected_run(self) -> None:
        row = self._selected_row()
        if row is None:
            self._loaded_run_id = ""
            self._loaded_mapping_signature = ""
            self.sources.set_mapping([])
            self.revision_label.clear()
            self._refresh_actions()
            return
        run_id = str(row.get("run_id") or "")
        self._select_included_run(run_id)
        self.revision_label.setText(
            f"Active source revision: {int(row.get('source_revision') or 1)}"
        )
        mapping = deepcopy(dict(row.get("source_mapping") or {}))
        mapping["run_id"] = run_id
        mapping["candidate_status"] = "user_confirmed"
        self.sources.flat_run_id.setText(run_id)
        self.sources.experiment_images.setText(
            str(mapping.get("experiment_folder") or "")
        )
        config = _project_mapping_config(self.project)
        self.sources.set_layout_template(config.get("run_layout_template") or {})
        self.sources.set_image_file_policy(self.project.image_file_policy())
        self.sources.image_file_policy.setEnabled(False)
        self.sources.image_file_policy.setToolTip(
            "This is a global series setting. Change it in Global series settings "
            "above and rebuild the expected folder mapping."
        )
        self.sources.set_mapping([mapping] if mapping else [])
        loaded = self.sources.source_mapping()
        self._loaded_run_id = run_id
        self._loaded_mapping_signature = (
            _editable_mapping_signature(loaded[0]) if len(loaded) == 1 else ""
        )
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        actor_present = bool(self.actor.text().strip())
        run_id = str(self.run_selector.currentData() or "")
        mappings = self.sources.source_mapping()
        mapping_complete = self.sources.isComplete()
        current_signature = (
            _editable_mapping_signature(mappings[0]) if len(mappings) == 1 else ""
        )
        individual_change = bool(
            run_id
            and run_id == self._loaded_run_id
            and current_signature
            and current_signature != self._loaded_mapping_signature
        )
        self.update_button.setText(
            f"Review changes for {run_id.upper()}..."
            if run_id
            else "Review changes for selected experiment..."
        )
        self.update_button.setEnabled(
            actor_present and mapping_complete and individual_change
        )
        policy_changed = (
            str(self.image_file_policy.currentData() or "")
            != self.project.image_file_policy()
        )
        if policy_changed:
            self.review_global_structure.setText(
                "Review and apply image choice to all experiments..."
            )
        else:
            config = _project_mapping_config(self.project)
            self.review_global_structure.setText(
                "Review or replace saved structure..."
                if config.get("run_layout_template")
                else "Define and save expected structure..."
            )
        if not run_id:
            set_state_label(
                self.pending_change_status,
                "incomplete",
                "INCOMPLETE: No experiment is selected.",
            )
        elif not mapping_complete:
            set_state_label(
                self.pending_change_status,
                "incomplete",
                f"INCOMPLETE: The individual mapping for {run_id.upper()} must be "
                "reviewed before it can be saved.",
            )
        elif individual_change:
            set_state_label(
                self.pending_change_status,
                "pending",
                f"NOT SAVED: Individual changes were detected for "
                f"{run_id.upper()}. The button below saves only {run_id.upper()}.",
            )
        else:
            set_state_label(
                self.pending_change_status,
                "saved",
                f"SAVED: No unsaved individual changes for {run_id.upper()}. Global "
                "folder-structure changes are saved in the global workflow above "
                "and do not require a second confirmation here.",
            )
        self.review_global_structure.setEnabled(actor_present and bool(self.rows))
        self.apply_rig_button.setEnabled(
            actor_present
            and str(self.rig_type.currentData() or "")
            != str((self.project.manifest.get("rig") or {}).get("type") or "")
        )

    def _review_and_update(self) -> None:
        run_id = str(self.run_selector.currentData() or "")
        mappings = self.sources.source_mapping()
        current_signature = (
            _editable_mapping_signature(mappings[0]) if len(mappings) == 1 else ""
        )
        if (
            run_id != self._loaded_run_id
            or not current_signature
            or current_signature == self._loaded_mapping_signature
        ):
            QMessageBox.information(
                self,
                "No individual changes",
                f"There are no unsaved source changes for {run_id.upper()}. Global "
                "folder-structure changes are confirmed in the global workflow.",
            )
            return
        if len(mappings) != 1:
            QMessageBox.warning(
                self,
                "Source mapping required",
                "Inspect and review one source mapping before continuing.",
            )
            return
        mapping = mappings[0]
        if str(mapping.get("run_id") or "") != run_id:
            QMessageBox.warning(
                self,
                "Run ID cannot change here",
                "Restore the selected experiment's Run ID. Renaming is a separate "
                "project operation.",
            )
            return
        cross_run_issues = _cross_run_mapping_issues(self.project, run_id, mapping)
        if cross_run_issues:
            QMessageBox.warning(
                self,
                "Selected folder contains other experiments",
                "This mapping cannot be saved as one experiment:\n\n"
                + "\n".join(cross_run_issues[:8])
                + (
                    f"\n\n...and {len(cross_run_issues) - 8} more conflict(s)."
                    if len(cross_run_issues) > 8
                    else ""
                )
                + "\n\nChoose the individual experiment folder, or use "
                "'Add experiments...' to scan the series parent folder.",
            )
            return
        preview = RunImportPreviewDialog(
            mappings,
            self,
            title=f"Review source update for {run_id.upper()}",
            confirm_label=f"Save new source revision for {run_id.upper()}",
        )
        if preview.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.report = self.project.update_run_sources(
                run_id,
                mapping,
                updated_by=self.actor.text().strip(),
                reason="Updated through the established-series source dialog.",
            )
            self.project.update_metadata(
                "chemistry",
                {
                    "indicator_response_reference": (
                        self.sources.indicator_reference_contract()
                    )
                },
                updated_by=self.actor.text().strip(),
                reason="Updated indicator-reference metadata with run sources.",
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self,
                "Could not update experiment",
                f"The previous mapping remains active.\n\n{type(exc).__name__}: {exc}",
            )
            return
        QMessageBox.information(
            self,
            "Project setup updated",
            f"{run_id.upper()} now uses source revision "
            f"{self.report['source_revision']}. Its previous mapping was retained, "
            "and this experiment must pass data-integrity review again.",
        )
        self._mark_project_changed()
        self._reload_rows(run_id)
        self.individual_mapping_saved.emit()


class UpdateRunSourcesDialog(QDialog):
    """Compatibility dialog wrapping the established-series setup editor."""

    def __init__(
        self,
        project: FluidFlowerProject,
        parent: Any = None,
        *,
        initial_run_id: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Review project setup for {project.name}")
        self.resize(1480, 900)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.editor = EstablishedSourceSetupEditor(
            project,
            self,
            initial_run_id=initial_run_id,
        )
        layout.addWidget(self.editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.editor.individual_mapping_saved.connect(self.accept)

    @property
    def changed(self) -> bool:
        return self.editor.changed

    @property
    def report(self) -> dict[str, Any] | None:
        return self.editor.report

    def __getattr__(self, name: str) -> Any:
        editor = self.__dict__.get("editor")
        if editor is not None:
            try:
                return getattr(editor, name)
            except AttributeError:
                pass
        raise AttributeError(name)


__all__ = [
    "AddRunsDialog",
    "EstablishedSourceSetupEditor",
    "RunImportPreviewDialog",
    "UpdateRunSourcesDialog",
]
