"""Image-source discovery and per-run mapping setup step."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from darsia.fluidflower.image_formats import (
    DEFAULT_IMAGE_FILE_POLICY,
    IMAGE_FILE_POLICY_ALL,
    IMAGE_FILE_POLICY_PROCESSED,
    IMAGE_FILE_POLICY_PROCESSED_ONLY,
    IMAGE_FILE_POLICY_RAW,
    IMAGE_FILE_POLICY_RAW_ONLY,
    normalize_image_file_policy,
)
from darsia.fluidflower.indicator_references import (
    REFERENCE_KIND_ALTERNATE_STATE,
    REFERENCE_KIND_PH_BUFFER,
    build_indicator_reference_contract,
    discover_indicator_reference_images,
    merge_indicator_reference_entries,
)
from darsia.fluidflower.pages.setup.indicator_reference_dialog import (
    IndicatorReferenceDialog,
)
from darsia.fluidflower.pages.setup.source_layout_dialog import (
    RunLayoutDialog,
    SeriesRunsDialog,
    SourceMappingDetailsDialog,
)
from darsia.fluidflower.pages.setup.steps.identity import SeriesIdentityPage
from darsia.fluidflower.pages.setup.steps.rig_protocol import RigProtocolPage
from darsia.fluidflower.source_discovery import discover_image_source_mapping
from darsia.fluidflower.templates import template_by_id
from darsia.fluidflower.ui.common import PathField, deepcopy_settings


def _source_explanation(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("secondaryText")
    return label


class SourceDiscoveryWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, settings: Mapping[str, Any]) -> None:
        super().__init__()
        self.settings = deepcopy_settings(settings)

    @Slot()
    def run(self) -> None:
        try:
            rows = discover_image_source_mapping(
                Path(str(self.settings["experiment_root"])),
                structure=str(self.settings["structure"]),
                baseline_root=(
                    Path(str(self.settings["baseline_root"]))
                    if self.settings.get("baseline_root")
                    else None
                ),
                clearwater_root=(
                    Path(str(self.settings["clearwater_root"]))
                    if self.settings.get("clearwater_root")
                    else None
                ),
                flat_run_id=str(self.settings.get("flat_run_id") or "run"),
                layout_template=(
                    dict(self.settings["layout_template"])
                    if isinstance(self.settings.get("layout_template"), Mapping)
                    else None
                ),
                run_definitions=(
                    list(self.settings.get("run_definitions") or []) or None
                ),
                protocol_layout=str(self.settings.get("protocol_layout") or ""),
                image_file_policy=str(
                    self.settings.get("image_file_policy") or DEFAULT_IMAGE_FILE_POLICY
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(rows)


class ImageSourcesPage(QWizardPage):
    layout_template_changed = Signal()
    run_definitions_changed = Signal()
    reference_settings_changed = Signal()

    def __init__(
        self,
        identity: SeriesIdentityPage | None = None,
        *,
        embedded: bool = False,
    ) -> None:
        super().__init__()
        self.mapping_rows: list[dict[str, Any]] = []
        self.identity = identity
        self.indicator_system_override = ""
        self.indicator_reference_entries: list[dict[str, Any]] = []
        self.run_layout_template: dict[str, Any] = {}
        self.run_definitions: list[dict[str, Any]] = []
        self.reserved_run_ids: set[str] = set()
        self.reserved_run_folders: set[str] = set()
        self._thread: QThread | None = None
        self._worker: SourceDiscoveryWorker | None = None
        self._populating_mapping = False
        self.rig_protocol: RigProtocolPage | None = None
        self.setTitle(
            "Apparatus, protocol and image sources"
            if identity is not None
            else "Image sources"
        )
        self.setSubTitle(
            "Choose the apparatus and protocol organization, then review protocol, "
            "image, and optional overflow sources for every experiment. Source files "
            "remain read-only."
            if identity is not None
            else (
                "Choose a series folder, define one representative run structure, "
                "and review the proposed sources for every run. Source files remain "
                "read-only."
            )
        )
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area: QScrollArea | None = None
        self.scroll_content = QWidget()
        self.scroll_content.setMinimumWidth(1120)
        layout = QFormLayout(self.scroll_content)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        if embedded:
            page_layout.addWidget(self.scroll_content)
        else:
            self.scroll_area = QScrollArea()
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self.scroll_area.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self.scroll_area.setWidget(self.scroll_content)
            page_layout.addWidget(self.scroll_area)

        if identity is not None:
            self.rig_protocol = RigProtocolPage(identity)
            layout.addRow(self.rig_protocol)
            source_heading = QLabel("Read-only experiment folders and per-run sources")
            source_heading.setObjectName("sectionTitle")
            layout.addRow(source_heading)

        self.experiment_images = PathField(
            directory=True,
            placeholder="Existing raw-data folder containing one or more runs",
        )
        self.experiment_images.setToolTip(
            "Select the existing raw experiment-data folder. DarSIA registers it as a "
            "read-only source; it is not the project folder or a result destination. "
            "Each immediate subfolder is treated as one run unless single-run mode is "
            "selected."
        )
        self.baseline_images = PathField(
            directory=True,
            placeholder="Optional shared override; normally leave empty",
        )
        self.baseline_images.setToolTip(
            "A baseline shows the rig filled with the experiment's actual pH-indicator "
            "solution before any CO2 is present. Normally this is the first image in "
            "the CO2 sequence because the camera starts before injection."
        )
        self.clearwater_images = PathField(
            directory=True,
            placeholder="Optional shared override; normally found inside each run",
        )
        self.clearwater_images.setToolTip(
            "A clear-water reference shows the rig filled with clear, indicator-free "
            "liquid and no CO2. It is normally found at one end of a wash or "
            "indicator-flush sequence."
        )
        self.colorchecker_images = PathField(
            directory=True,
            placeholder=(
                "Folder with ColorChecker reference frames from this camera setup"
            ),
        )
        self.colorchecker_source = QComboBox()
        self.colorchecker_source.addItem(
            "Visible in the experiment images", "embedded_in_source_images"
        )
        self.colorchecker_source.addItem(
            "Available in a separate reference folder", "dedicated_folder"
        )
        self.colorchecker_source.addItem("No ColorChecker available", "not_available")
        self.colorchecker_source.setToolTip(
            "This only records where the ColorChecker can be read. The correction "
            "model is selected and validated later under Image correction."
        )
        self.indicator_reference_images = PathField(
            directory=True,
            placeholder="Optional shared folder with the same indicator at measured pH",
        )
        self.indicator_reference_images.setToolTip(
            "Full-rig images containing the same indicator mixture as the experiment "
            "at explicitly measured pH values. These fit the chemical color response; "
            "they do not normalize illumination."
        )
        self.alternate_color_reference_images = PathField(
            directory=True,
            placeholder="Optional shared folder with other known full-rig colors",
        )
        self.alternate_color_reference_images.setToolTip(
            "Images of the rig filled with another known liquid, dye, or color state "
            "that is not a measured-pH state of the experiment's indicator. They are "
            "stored as labelled optical controls, not as a pH curve or ColorChecker."
        )
        self.configure_indicator_references = QPushButton(
            "Review pH / color-state metadata..."
        )
        self.configure_indicator_references.clicked.connect(
            self._configure_indicator_reference_entries
        )
        self.indicator_reference_summary = QLabel("No indicator references registered.")
        self.indicator_reference_summary.setWordWrap(True)
        self.indicator_reference_summary.setObjectName("secondaryText")
        indicator_reference_row = QHBoxLayout()
        indicator_reference_row.addWidget(self.indicator_reference_images, 1)
        indicator_reference_row.addWidget(self.configure_indicator_references)
        layout.addRow("Raw experiment-data folder (read-only)", self.experiment_images)
        self.raw_data_explanation = _source_explanation(
            "This location tells DarSIA where the original camera images, run folders, "
            "protocols, and optional overflow sources are stored. Nothing is written "
            "here; it is a read-only source. The DarSIA project and all default "
            "generated results use the "
            "separate storage location selected on the first setup page."
        )
        layout.addRow("", self.raw_data_explanation)
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
        self.image_file_policy.setToolTip(
            "Choose one series-wide image family before source discovery. When a "
            "folder contains paired JPG and ARW files, the non-preferred copy is "
            "excluded from image counts, integrity reads, and analysis. The preferred "
            "family falls back to the other family only when it is unavailable. "
            "The two 'Only' choices never fall back: a folder without that kind "
            "of file is reported as an error."
        )
        layout.addRow("Image files used by DarSIA", self.image_file_policy)
        layout.addRow(
            "",
            _source_explanation(
                "Most cameras save one processed image and one RAW file for the same "
                "exposure. Selecting both normally duplicates the time series. The "
                "recommended processed-image option is immediately supported by all "
                "DarSIA image stages; select RAW only when the project has a validated "
                "RAW decoding workflow. The selected policy is stored with the project."
            ),
        )
        layout.addRow("Shared baseline override", self.baseline_images)
        layout.addRow(
            "",
            _source_explanation(
                "Use this override only for a separate shared collection. A valid "
                "baseline shows the rig filled with the correct pH-indicator solution "
                "but before CO2 enters the system. Normally leave it empty: the first "
                "frame in each CO2 sequence is used because imaging starts before "
                "injection."
            ),
        )
        layout.addRow("Shared clear-water override", self.clearwater_images)
        layout.addRow(
            "",
            _source_explanation(
                "This must show the rig clear and without indicator or CO2. It is "
                "usually the last image of a wash that changes indicator-filled water "
                "to clear water, or the first image of a flush that changes clear water "
                "to indicator-filled water."
            ),
        )
        layout.addRow("ColorChecker source", self.colorchecker_source)
        layout.addRow(
            "",
            _source_explanation(
                "Register whether the 24-patch chart is visible in the ordinary "
                "experiment frames, supplied through a separate folder, or unavailable. "
                "This page records source availability only. Selection and comparison "
                "of light/color-correction models take place under Image correction."
            ),
        )
        layout.addRow("ColorChecker reference folder", self.colorchecker_images)
        layout.addRow(
            "",
            _source_explanation(
                "Use this only when reference frames are stored separately. They must "
                "come from the same camera and lighting setup, with the chart at the "
                "same registered position as in the experiment images."
            ),
        )
        layout.addRow("Measured-pH indicator buffers", indicator_reference_row)
        layout.addRow(
            "",
            _source_explanation(
                "These contain the same indicator mixture used in the experiments at "
                "known pH values. They constrain the color-to-pH response after image "
                "correction and are never substituted for a ColorChecker."
            ),
        )
        layout.addRow(
            "Other known full-rig color states",
            self.alternate_color_reference_images,
        )
        layout.addRow(
            "",
            _source_explanation(
                "Use this for a rig filled with another known color, dye, or liquid "
                "that is not the experiment indicator at measured pH. These images are "
                "kept as labelled optical controls and do not enter the pH-response "
                "curve or radiometric normalization."
            ),
        )
        layout.addRow("", self.indicator_reference_summary)

        self.structure = QComboBox()
        self.structure.addItem(
            "Multiple runs under one parent folder", "run_subfolders"
        )
        self.structure.addItem("Selected folder is one run", "single_run")
        self.structure.setToolTip(
            "For a series, DarSIA learns the internal structure from one example run "
            "and applies it to the other immediate subfolders."
        )
        self.flat_run_id = QLineEdit("run")
        self.flat_run_id.setPlaceholderText("Run ID for single-run mode")
        structure_row = QHBoxLayout()
        structure_row.addWidget(self.structure)
        structure_row.addWidget(self.flat_run_id)
        layout.addRow("Folder organization", structure_row)

        self.scan_runs = QPushButton("Scan and review run folders...")
        self.scan_runs.clicked.connect(self._review_run_folders)
        self.run_summary = QLabel("Run folders have not been reviewed.")
        self.run_summary.setWordWrap(True)
        self.run_summary.setObjectName("secondaryText")
        runs_row = QHBoxLayout()
        runs_row.addWidget(self.scan_runs)
        runs_row.addWidget(self.run_summary, 1)
        layout.addRow("Experimental runs", runs_row)

        self.example_run = QComboBox()
        self.example_run.setMinimumContentsLength(24)
        self.example_run.setToolTip(
            "Choose a representative run whose internal folder structure DarSIA should learn."
        )
        refresh_examples = QPushButton("Refresh")
        refresh_examples.clicked.connect(self._refresh_example_runs)
        self.define_layout = QPushButton("Define structure...")
        self.define_layout.clicked.connect(self._define_run_layout)
        example_row = QHBoxLayout()
        example_row.addWidget(self.example_run, 1)
        example_row.addWidget(refresh_examples)
        example_row.addWidget(self.define_layout)
        layout.addRow("Representative run", example_row)
        self.layout_summary = QLabel(
            "No shared structure has been defined. Automatic per-run discovery remains available."
        )
        self.layout_summary.setWordWrap(True)
        self.layout_summary.setObjectName("secondaryText")
        layout.addRow("", self.layout_summary)

        self.discover_sources = QPushButton("Apply structure and inspect all runs")
        self.discover_sources.clicked.connect(self._discover)
        layout.addRow("Source mapping", self.discover_sources)
        self.mapping_summary = QLabel("No source mapping has been proposed.")
        self.mapping_summary.setWordWrap(True)
        self.mapping_summary.setObjectName("secondaryText")
        layout.addRow("", self.mapping_summary)

        self.mapping_table = QTableWidget(0, 12)
        self.mapping_table.setHorizontalHeaderLabels(
            [
                "Reviewed",
                "Run ID",
                "CO2 images",
                "Ordered CO2 folders",
                "Baseline",
                "Clear-water",
                "Measured-pH buffers",
                "Other color states",
                "Protocol workbook",
                "Overflow images",
                "Overflow table",
                "Status",
            ]
        )
        self.mapping_table.verticalHeader().setVisible(False)
        self.mapping_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.mapping_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.mapping_table.cellDoubleClicked.connect(self._inspect_mapping_row)
        self.mapping_table.itemChanged.connect(self._mapping_item_changed)
        mapping_header = self.mapping_table.horizontalHeader()
        for column in (0, 1, 2, 11):
            mapping_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        for column in (3, 4, 5, 6, 7, 8, 9, 10):
            mapping_header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.mapping_table.setMinimumHeight(220)
        layout.addRow(self.mapping_table)

        mapping_actions = QHBoxLayout()
        self.inspect_mapping = QPushButton("Inspect selected run")
        self.inspect_mapping.clicked.connect(self._inspect_selected_mapping)
        self.inspect_mapping.setEnabled(False)
        mapping_actions.addWidget(self.inspect_mapping)
        self.edit_mapping = QPushButton("Categorize selected run folders...")
        self.edit_mapping.clicked.connect(self._categorize_selected_run)
        self.edit_mapping.setEnabled(False)
        mapping_actions.addWidget(self.edit_mapping)
        self.review_all = QPushButton("Mark all visible mappings reviewed")
        self.review_all.clicked.connect(self._mark_all_reviewed)
        self.review_all.setEnabled(False)
        mapping_actions.addWidget(self.review_all)
        mapping_actions.addStretch(1)
        layout.addRow("", mapping_actions)

        self.experiment_images.changed.connect(self._source_root_changed)
        self.structure.currentIndexChanged.connect(self._source_root_changed)
        self.image_file_policy.currentIndexChanged.connect(self._invalidate_mapping)
        self.flat_run_id.textChanged.connect(self._invalidate_mapping)
        for signal in (
            self.baseline_images.changed,
            self.clearwater_images.changed,
        ):
            signal.connect(self._invalidate_mapping)
        self.colorchecker_source.currentIndexChanged.connect(
            self._colorchecker_source_changed
        )
        self.colorchecker_images.changed.connect(self._reference_settings_edited)
        self.indicator_reference_images.changed.connect(self._reference_settings_edited)
        self.alternate_color_reference_images.changed.connect(
            self._reference_settings_edited
        )
        if self.rig_protocol is not None:
            self.rig_protocol.protocol_layout.currentIndexChanged.connect(
                self._invalidate_mapping
            )
        self._structure_changed()
        self._colorchecker_source_changed()

    def _settings(self) -> dict[str, Any]:
        return {
            "experiment_root": self.experiment_images.text(),
            "baseline_root": self.baseline_images.text(),
            "clearwater_root": self.clearwater_images.text(),
            "structure": str(self.structure.currentData()),
            "flat_run_id": self.flat_run_id.text().strip() or "run",
            "layout_template": deepcopy(self.run_layout_template),
            "run_definitions": self.run_definitions_value(),
            "protocol_layout": self.protocol_layout(),
            "image_file_policy": self.image_file_policy_value(),
        }

    def image_file_policy_value(self) -> str:
        return normalize_image_file_policy(self.image_file_policy.currentData())

    def set_image_file_policy(self, policy: str) -> None:
        normalized = normalize_image_file_policy(policy)
        index = self.image_file_policy.findData(normalized)
        if index < 0:
            return
        self.image_file_policy.blockSignals(True)
        self.image_file_policy.setCurrentIndex(index)
        self.image_file_policy.blockSignals(False)

    def _colorchecker_source_changed(self, _value: Any = None) -> None:
        uses_separate_folder = (
            str(self.colorchecker_source.currentData()) == "dedicated_folder"
        )
        self.colorchecker_images.setEnabled(uses_separate_folder)
        self._reference_settings_edited()

    def _reference_settings_edited(self, _value: Any = None) -> None:
        self._update_indicator_reference_summary()
        self.reference_settings_changed.emit()
        self.completeChanged.emit()

    def _indicator_reference_sources(self) -> list[tuple[Path, str]]:
        """Return sources that are expected to contain measured-pH buffers."""

        sources: list[tuple[Path, str]] = []
        shared = self.indicator_reference_images.text().strip()
        if shared:
            sources.append((Path(shared), "series"))
        rows = (
            self.source_mapping()
            if self.mapping_table.rowCount()
            else self.mapping_rows
        )
        for row in rows:
            run_id = str(row.get("run_id") or "run")
            for folder in row.get("indicator_reference_folders") or []:
                if str(folder).strip():
                    sources.append((Path(str(folder)), run_id))
        return sources

    def _alternate_color_reference_sources(self) -> list[tuple[Path, str]]:
        """Return sources containing known non-indicator full-rig color states."""

        sources: list[tuple[Path, str]] = []
        shared = self.alternate_color_reference_images.text().strip()
        if shared:
            sources.append((Path(shared), "series"))
        rows = (
            self.source_mapping()
            if self.mapping_table.rowCount()
            else self.mapping_rows
        )
        for row in rows:
            run_id = str(row.get("run_id") or "run")
            for folder in row.get("alternate_color_reference_folders") or []:
                if str(folder).strip():
                    sources.append((Path(str(folder)), run_id))
        return sources

    def _merged_indicator_reference_entries(self) -> list[dict[str, Any]]:
        discovered = discover_indicator_reference_images(
            self._indicator_reference_sources(),
            default_kind=REFERENCE_KIND_PH_BUFFER,
        )
        discovered = merge_indicator_reference_entries(
            discovered,
            discover_indicator_reference_images(
                self._alternate_color_reference_sources(),
                default_kind=REFERENCE_KIND_ALTERNATE_STATE,
            ),
        )
        return merge_indicator_reference_entries(
            self.indicator_reference_entries,
            discovered,
        )

    def _configure_indicator_reference_entries(self) -> None:
        dialog = IndicatorReferenceDialog(
            self,
            entries=self.indicator_reference_entries,
            ph_buffer_folders=self._indicator_reference_sources(),
            alternate_state_folders=self._alternate_color_reference_sources(),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.indicator_reference_entries = dialog.normalized_entries()
        self._reference_settings_edited()

    def _indicator_system(self) -> str:
        if self.indicator_system_override:
            return self.indicator_system_override
        if self.identity is None:
            return "unspecified"
        return template_by_id(self.identity.template_id()).indicator_system

    def indicator_reference_contract(self) -> dict[str, Any]:
        return build_indicator_reference_contract(
            self._merged_indicator_reference_entries(),
            indicator_system=self._indicator_system(),
        )

    def reference_settings(self) -> dict[str, Any]:
        colorchecker_location = str(
            self.colorchecker_source.currentData() or "not_available"
        )
        colorchecker_root = (
            self.colorchecker_images.text()
            if colorchecker_location == "dedicated_folder"
            else ""
        )
        return {
            "correction_model_selection": "image_correction_module",
            "colorchecker_location": colorchecker_location,
            "colorchecker_root": colorchecker_root,
            "indicator_reference_root": self.indicator_reference_images.text(),
            "alternate_color_reference_root": (
                self.alternate_color_reference_images.text()
            ),
            "indicator_reference_entries": deepcopy(self.indicator_reference_entries),
        }

    def set_reference_settings(self, values: Mapping[str, Any] | None) -> None:
        settings = dict(values or {})
        colorchecker_root = str(settings.get("colorchecker_root") or "")
        location = str(settings.get("colorchecker_location") or "")
        if location not in {
            "embedded_in_source_images",
            "dedicated_folder",
            "not_available",
        }:
            legacy_mode = str(settings.get("radiometric_mode") or "none")
            location = (
                "dedicated_folder"
                if colorchecker_root
                else (
                    "embedded_in_source_images"
                    if legacy_mode == "colorchecker_uniform24"
                    else "not_available"
                )
            )
        index = self.colorchecker_source.findData(location)
        if index >= 0:
            self.colorchecker_source.setCurrentIndex(index)
        self.colorchecker_images.setText(colorchecker_root)
        self.indicator_reference_images.setText(
            str(settings.get("indicator_reference_root") or "")
        )
        self.alternate_color_reference_images.setText(
            str(settings.get("alternate_color_reference_root") or "")
        )
        self.indicator_reference_entries = [
            deepcopy(dict(row))
            for row in settings.get("indicator_reference_entries") or []
            if isinstance(row, Mapping)
        ]
        self._colorchecker_source_changed()

    def _update_indicator_reference_summary(self) -> None:
        contract = build_indicator_reference_contract(
            self.indicator_reference_entries,
            indicator_system=self._indicator_system(),
        )
        ph_source_count = len(self._indicator_reference_sources())
        alternate_source_count = len(self._alternate_color_reference_sources())
        self.indicator_reference_summary.setText(
            f"{contract['known_ph_buffer_count']} measured-pH image(s) and "
            f"{contract['alternate_state_count']} other color-state image(s) have "
            f"metadata; {ph_source_count} pH-buffer folder(s) and "
            f"{alternate_source_count} other color-state folder(s) are available. "
            "Status: "
            f"{str(contract['status']).replace('_', ' ')}. Buffer references are "
            "reserved for the chemistry response and never normalize illumination."
        )

    def protocol_layout(self) -> str:
        if self.rig_protocol is None:
            return ""
        return str(self.rig_protocol.protocol_layout.currentData() or "")

    def initializePage(self) -> None:
        if self.rig_protocol is not None:
            self.rig_protocol.initializePage()

    def _structure_changed(self) -> None:
        multiple = str(self.structure.currentData()) == "run_subfolders"
        self.example_run.setEnabled(True)
        self.flat_run_id.setVisible(not multiple)
        self.scan_runs.setVisible(multiple)
        self.run_summary.setVisible(multiple)
        self._refresh_example_runs()

    def _source_root_changed(self, _value: Any = None) -> None:
        self.run_definitions = []
        self.run_summary.setText("Run folders have not been reviewed.")
        self.run_definitions_changed.emit()
        self.run_layout_template = {}
        self.define_layout.setText("Define structure...")
        self.layout_summary.setText(
            "No shared structure has been defined. Automatic per-run discovery remains available."
        )
        self._structure_changed()
        self._invalidate_mapping()

    def _refresh_example_runs(self) -> None:
        current = str(self.example_run.currentData() or "")
        self.example_run.blockSignals(True)
        self.example_run.clear()
        root = Path(self.experiment_images.text()).expanduser()
        if root.is_dir() and str(self.structure.currentData()) == "run_subfolders":
            definitions = [
                row for row in self.run_definitions if bool(row.get("included", True))
            ]
            if definitions:
                for row in definitions:
                    self.example_run.addItem(
                        str(row.get("run_id") or Path(str(row["folder"])).name),
                        str(row["folder"]),
                    )
            else:
                for folder in sorted(
                    root.iterdir(), key=lambda path: path.name.casefold()
                ):
                    if folder.is_dir():
                        self.example_run.addItem(folder.name, str(folder.resolve()))
        elif root.is_dir():
            self.example_run.addItem(
                self.flat_run_id.text().strip() or root.name, str(root.resolve())
            )
        restored = self.example_run.findData(current)
        if restored >= 0:
            self.example_run.setCurrentIndex(restored)
        self.example_run.blockSignals(False)
        self.define_layout.setEnabled(self.example_run.count() > 0)

    def _review_run_folders(self) -> bool:
        root = Path(self.experiment_images.text()).expanduser()
        if not root.is_dir():
            QMessageBox.warning(
                self,
                "Raw experiment-data folder required",
                "Select the existing folder containing the experimental runs first.",
            )
            return False
        try:
            dialog = SeriesRunsDialog(
                root,
                self,
                definitions=self.run_definitions,
                reserved_run_ids=self.reserved_run_ids,
                reserved_folders=self.reserved_run_folders,
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self, "Could not scan run folders", f"{type(exc).__name__}: {exc}"
            )
            return False
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        self.set_run_definitions(dialog.definitions())
        self._invalidate_mapping()
        return True

    def set_reserved_runs(
        self,
        *,
        run_ids: set[str] | None = None,
        folders: set[str] | None = None,
    ) -> None:
        """Prevent an incremental import from selecting existing runs again."""

        self.reserved_run_ids = {
            str(value).strip().casefold() for value in run_ids or set()
        }
        self.reserved_run_folders = {
            str(Path(value).expanduser().resolve())
            for value in folders or set()
            if str(value).strip()
        }

    def set_run_definitions(self, definitions: list[Mapping[str, Any]] | None) -> None:
        self.run_definitions = [deepcopy(dict(row)) for row in definitions or []]
        selected = sum(bool(row.get("included", True)) for row in self.run_definitions)
        if self.run_definitions:
            self.run_summary.setText(
                f"Detected {len(self.run_definitions)} folder(s); "
                f"{selected} selected as experimental runs."
            )
        else:
            self.run_summary.setText("Run folders have not been reviewed.")
        self._refresh_example_runs()
        self.run_definitions_changed.emit()

    def run_definitions_value(self) -> list[dict[str, Any]]:
        return deepcopy(self.run_definitions)

    def _define_run_layout(self) -> None:
        path = Path(str(self.example_run.currentData() or ""))
        if not path.is_dir():
            QMessageBox.warning(
                self,
                "Representative run required",
                "Choose the series folder and a representative run first.",
            )
            return
        specifications = [
            dict(row)
            for row in self.run_layout_template.get("folders") or []
            if isinstance(row, Mapping)
        ]
        assignments = {
            str(row.get("relative_path") or ""): str(row.get("role") or "ignore")
            for row in specifications
        }
        sequence_orders = {
            str(row.get("relative_path") or ""): int(row["sequence_order"])
            for row in specifications
            if str(row.get("sequence_order") or "").isdigit()
        }
        selected_table_files = {
            str(row.get("relative_path") or ""): str(
                row.get("selected_table_name") or ""
            )
            for row in specifications
            if str(row.get("selected_table_name") or "")
        }
        table_assignments = {
            str(row.get("relative_path") or ""): str(row.get("role") or "ignore")
            for row in self.run_layout_template.get("table_files") or []
            if isinstance(row, Mapping)
        }
        try:
            dialog = RunLayoutDialog(
                path,
                self,
                assignments=assignments,
                sequence_orders=sequence_orders,
                selected_table_files=selected_table_files,
                table_assignments=table_assignments,
                clearwater_selector=str(
                    self.run_layout_template.get("clearwater_selector") or "first"
                ),
                image_file_policy=self.image_file_policy_value(),
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self, "Could not inspect run structure", f"{type(exc).__name__}: {exc}"
            )
            return
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_layout_template(dialog.template())
            self._invalidate_mapping()

    def set_layout_template(self, template: Mapping[str, Any] | None) -> None:
        self.run_layout_template = deepcopy(dict(template or {}))
        if self.run_layout_template.get("image_file_policy"):
            self.set_image_file_policy(
                str(self.run_layout_template["image_file_policy"])
            )
        if not self.run_layout_template:
            self.define_layout.setText("Define structure...")
            self.layout_summary.setText(
                "No shared structure has been defined. Automatic per-run discovery remains available."
            )
            self.layout_template_changed.emit()
            return
        folders = list(self.run_layout_template.get("folders") or [])
        example = str(self.run_layout_template.get("example_run_name") or "example run")
        clearwater = str(self.run_layout_template.get("clearwater_selector") or "first")
        table_files = [
            dict(row)
            for row in self.run_layout_template.get("table_files") or []
            if isinstance(row, Mapping)
        ]
        protocol_count = sum(
            str(row.get("role") or "") == "protocol_table" for row in table_files
        )
        overflow_count = sum(
            str(row.get("role") or "") == "overflow_table" for row in table_files
        )
        table_summary = (
            f" Separate table assignments: {protocol_count} protocol and "
            f"{overflow_count} overflow file(s)."
            if table_files
            else ""
        )
        self.define_layout.setText("Review or change structure...")
        self.layout_summary.setText(
            f"Structure learned from {example}: {len(folders)} source-bearing folder "
            f"role(s). Baseline = first image in the CO2 injection sequence; "
            f"clear-water = {clearwater} image in its selected sequence."
            f"{table_summary}"
        )
        example_path = str(self.run_layout_template.get("example_run_root") or "")
        index = self.example_run.findData(example_path)
        if index >= 0:
            self.example_run.setCurrentIndex(index)
        self.layout_template_changed.emit()

    def layout_template(self) -> dict[str, Any]:
        return deepcopy(self.run_layout_template)

    def _invalidate_mapping(self, _value: Any = None) -> None:
        if self.mapping_rows:
            self.mapping_rows = []
            self.mapping_table.setRowCount(0)
            self.mapping_summary.setText(
                "Source settings changed. Inspect all runs again before project creation."
            )
            self.inspect_mapping.setEnabled(False)
            self.edit_mapping.setEnabled(False)
            self.review_all.setEnabled(False)
            self.completeChanged.emit()

    def _discover(self) -> None:
        if self._thread is not None:
            return
        if not Path(self.experiment_images.text()).expanduser().is_dir():
            QMessageBox.warning(
                self,
                "Raw experiment-data folder required",
                "Select the existing folder containing the experimental runs first.",
            )
            return
        if (
            str(self.structure.currentData()) == "run_subfolders"
            and not any(bool(row.get("included", True)) for row in self.run_definitions)
            and not self._review_run_folders()
        ):
            return
        self.discover_sources.setEnabled(False)
        self.mapping_summary.setText(
            "Inspecting run folders without reading or modifying image content..."
        )
        thread = QThread(self)
        worker = SourceDiscoveryWorker(self._settings())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._discovery_ready)
        worker.failed.connect(self._discovery_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._discovery_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _discovery_finished(self) -> None:
        if self._thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            self._thread.wait()
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.discover_sources.setEnabled(True)

    @Slot(object)
    def _discovery_ready(self, raw_rows: object) -> None:
        self.set_mapping([dict(row) for row in raw_rows if isinstance(row, Mapping)])

    @Slot(str)
    def _discovery_failed(self, message: str) -> None:
        self.mapping_summary.setText("Source discovery failed; no mapping was saved.")
        QMessageBox.critical(self, "Could not discover image sources", message)

    @staticmethod
    def _read_only_item(value: Any) -> QTableWidgetItem:
        item = QTableWidgetItem(str(value or ""))
        item.setToolTip(str(value or ""))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    @staticmethod
    def _editable_item(value: Any) -> QTableWidgetItem:
        item = QTableWidgetItem(str(value or ""))
        item.setToolTip(str(value or ""))
        return item

    def set_mapping(self, rows: list[dict[str, Any]]) -> None:
        self._populating_mapping = True
        try:
            self.mapping_rows = deepcopy(rows)
            self.mapping_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                reviewed = QTableWidgetItem("")
                reviewed.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                reviewed.setCheckState(
                    Qt.CheckState.Checked
                    if str(row.get("candidate_status") or "")
                    in {"approved", "user_confirmed"}
                    else Qt.CheckState.Unchecked
                )
                self.mapping_table.setItem(row_index, 0, reviewed)
                self.mapping_table.setItem(
                    row_index, 1, self._editable_item(row.get("run_id"))
                )
                self.mapping_table.setItem(
                    row_index, 2, self._read_only_item(row.get("image_count"))
                )
                sequences = row.get("co2_sequences") or []
                sequence_item = self._read_only_item(
                    "\n".join(
                        f"{entry.get('order')}. [{entry.get('kind')}] "
                        f"{entry.get('folder')}"
                        for entry in sequences
                        if isinstance(entry, Mapping)
                    )
                    if sequences
                    else "\n".join(row.get("experiment_folders") or [])
                )
                if sequences:
                    sequence_item.setToolTip(
                        "\n".join(
                            f"{entry.get('order')}. {entry.get('kind')}: "
                            f"{entry.get('folder')}"
                            for entry in sequences
                            if isinstance(entry, Mapping)
                        )
                    )
                self.mapping_table.setItem(row_index, 3, sequence_item)
                self.mapping_table.setItem(
                    row_index, 4, self._editable_item(row.get("baseline_candidate"))
                )
                self.mapping_table.setItem(
                    row_index, 5, self._editable_item(row.get("clearwater_candidate"))
                )
                self.mapping_table.setItem(
                    row_index,
                    6,
                    self._editable_item(
                        "\n".join(row.get("indicator_reference_folders") or [])
                    ),
                )
                self.mapping_table.setItem(
                    row_index,
                    7,
                    self._editable_item(
                        "\n".join(row.get("alternate_color_reference_folders") or [])
                    ),
                )
                self.mapping_table.setItem(
                    row_index,
                    8,
                    self._editable_item(
                        "\n".join(
                            row.get("protocol_tables")
                            or (
                                [row.get("protocol_table")]
                                if row.get("protocol_table")
                                else []
                            )
                        )
                    ),
                )
                self.mapping_table.setItem(
                    row_index,
                    9,
                    self._editable_item(
                        "\n".join(row.get("overflow_image_folders") or [])
                    ),
                )
                self.mapping_table.setItem(
                    row_index,
                    10,
                    self._editable_item(
                        "\n".join(
                            row.get("overflow_tables")
                            or (
                                [row.get("overflow_table")]
                                if row.get("overflow_table")
                                else []
                            )
                        )
                    ),
                )
                warnings = list(row.get("discovery_warnings") or [])
                status = str(row.get("mapping_status") or "unknown")
                status_text = (
                    f"{status}; {len(warnings)} note(s)" if warnings else status
                )
                status_item = self._read_only_item(status_text)
                status_item.setToolTip("\n".join(warnings) or status)
                self.mapping_table.setItem(row_index, 11, status_item)
                visible_lines = max(
                    len(row.get("experiment_folders") or []),
                    len(row.get("indicator_reference_folders") or []),
                    len(row.get("alternate_color_reference_folders") or []),
                    len(row.get("overflow_image_folders") or []),
                    2,
                )
                self.mapping_table.setRowHeight(
                    row_index, min(110, 22 * visible_lines + 8)
                )
        finally:
            self._populating_mapping = False
        self.inspect_mapping.setEnabled(bool(rows))
        self.edit_mapping.setEnabled(bool(rows))
        self.review_all.setEnabled(bool(rows))
        if rows:
            self.mapping_table.selectRow(0)
        self._update_mapping_summary()
        self._update_indicator_reference_summary()
        self.completeChanged.emit()

    @staticmethod
    def _path_list(value: str) -> list[str]:
        return [
            entry.strip()
            for line in value.splitlines()
            for entry in line.split(";")
            if entry.strip()
        ]

    def source_mapping(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row_index in range(self.mapping_table.rowCount()):
            source = deepcopy(self.mapping_rows[row_index])
            stored_sequences = [
                dict(entry)
                for entry in source.get("co2_sequences") or []
                if isinstance(entry, Mapping)
            ]
            experiment_folders = (
                [
                    str(entry.get("folder") or "")
                    for entry in sorted(
                        stored_sequences,
                        key=lambda entry: int(entry.get("order") or 10**9),
                    )
                    if str(entry.get("folder") or "")
                ]
                if stored_sequences
                else self._path_list(self.mapping_table.item(row_index, 3).text())
            )
            overflow_folders = self._path_list(
                self.mapping_table.item(row_index, 9).text()
            )
            overflow_tables = self._path_list(
                self.mapping_table.item(row_index, 10).text()
            )
            overflow_table = overflow_tables[0] if overflow_tables else ""
            protocol_tables = self._path_list(
                self.mapping_table.item(row_index, 8).text()
            )
            protocol_table = protocol_tables[0] if protocol_tables else ""
            warnings = [
                str(value)
                for value in source.get("discovery_warnings") or []
                if str(value) != "No per-experiment protocol workbook was found."
            ]
            if self.protocol_layout() == "per_run_files" and not protocol_table:
                warnings.append("No per-experiment protocol workbook was found.")
            existing_sequences = {
                str(entry.get("folder") or "").casefold(): deepcopy(dict(entry))
                for entry in stored_sequences
                if isinstance(entry, Mapping)
            }
            co2_sequences: list[dict[str, Any]] = []
            for order, folder in enumerate(experiment_folders, start=1):
                sequence = existing_sequences.get(folder.casefold(), {})
                sequence.update(
                    {
                        "order": order,
                        "kind": str(
                            sequence.get("kind")
                            or ("injection" if order == 1 else "continuation")
                        ),
                        "folder": folder,
                    }
                )
                co2_sequences.append(sequence)
            source.update(
                {
                    "run_id": self.mapping_table.item(row_index, 1)
                    .text()
                    .strip()
                    .casefold(),
                    "image_file_policy": self.image_file_policy_value(),
                    "experiment_folders": experiment_folders,
                    "co2_image_folders": experiment_folders,
                    "co2_sequences": co2_sequences,
                    "baseline_candidate": self.mapping_table.item(row_index, 4)
                    .text()
                    .strip(),
                    "clearwater_candidate": self.mapping_table.item(row_index, 5)
                    .text()
                    .strip(),
                    "indicator_reference_folders": self._path_list(
                        self.mapping_table.item(row_index, 6).text()
                    ),
                    "alternate_color_reference_folders": self._path_list(
                        self.mapping_table.item(row_index, 7).text()
                    ),
                    "protocol_table": protocol_table,
                    "protocol_tables": protocol_tables,
                    "discovery_warnings": warnings,
                    "overflow_image_folders": overflow_folders,
                    "overflow_table": overflow_table,
                    "overflow_tables": overflow_tables,
                    "candidate_status": (
                        "user_confirmed"
                        if self.mapping_table.item(row_index, 0).checkState()
                        == Qt.CheckState.Checked
                        else "requires_user_confirmation"
                    ),
                }
            )
            rows.append(source)
        return rows

    def _mapping_item_changed(self, _item: QTableWidgetItem) -> None:
        if self._populating_mapping:
            return
        self._update_mapping_summary()
        self._update_indicator_reference_summary()
        self.completeChanged.emit()

    def _update_mapping_summary(self) -> None:
        rows = self.source_mapping() if self.mapping_table.rowCount() else []
        reviewed = sum(row.get("candidate_status") == "user_confirmed" for row in rows)
        images = sum(int(row.get("image_count") or 0) for row in rows)
        overflow = sum(
            bool(row.get("overflow_image_folders") or row.get("overflow_table"))
            for row in rows
        )
        protocols = sum(
            bool(row.get("protocol_tables") or row.get("protocol_table"))
            for row in rows
        )
        indicator_references = sum(
            bool(row.get("indicator_reference_folders")) for row in rows
        )
        alternate_color_references = sum(
            bool(row.get("alternate_color_reference_folders")) for row in rows
        )
        warnings = sum(len(row.get("discovery_warnings") or []) for row in rows)
        processed_files = sum(
            int((row.get("image_file_counts") or {}).get("processed") or 0)
            for row in rows
        )
        raw_files = sum(
            int((row.get("image_file_counts") or {}).get("raw") or 0) for row in rows
        )
        if rows:
            self.mapping_summary.setText(
                f"Found {len(rows)} run(s) and {images} CO2 images. Per-run overflow "
                f"was found for {overflow} run(s); per-run protocols for {protocols}. "
                f"Measured-pH buffers were found for {indicator_references} run(s); "
                f"other color states for {alternate_color_references}. "
                f"Image choice: {self.image_file_policy.currentText()} "
                f"({processed_files} processed and {raw_files} RAW CO2 files found). "
                f"Reviewed: {reviewed}/{len(rows)}. "
                f"Discovery notes: {warnings}. Double-click a row for full details."
            )

    def _mark_all_reviewed(self) -> None:
        for row_index in range(self.mapping_table.rowCount()):
            self.mapping_table.item(row_index, 0).setCheckState(Qt.CheckState.Checked)

    def _inspect_mapping_row(self, row: int, _column: int) -> None:
        if 0 <= row < self.mapping_table.rowCount():
            SourceMappingDetailsDialog(self.source_mapping()[row], self).exec()

    def _inspect_selected_mapping(self) -> None:
        row = self.mapping_table.currentRow()
        if row < 0:
            QMessageBox.information(
                self, "Select a run", "Select one mapping row to inspect."
            )
            return
        self._inspect_mapping_row(row, 0)

    def _categorize_selected_run(self) -> None:
        row_index = self.mapping_table.currentRow()
        if row_index < 0:
            QMessageBox.information(
                self, "Select a run", "Select one mapping row to edit."
            )
            return
        current = self.source_mapping()[row_index]
        run_root = Path(str(current.get("experiment_folder") or ""))
        if not run_root.is_dir():
            QMessageBox.warning(
                self,
                "Run folder unavailable",
                "The selected run folder cannot currently be read.",
            )
            return
        assignments = {
            str(item.get("relative_path") or ""): str(item.get("role") or "ignore")
            for item in current.get("source_folder_matches") or []
            if isinstance(item, Mapping)
        }
        sequence_orders = {
            str(item.get("relative_path") or ""): int(item["sequence_order"])
            for item in current.get("source_folder_matches") or []
            if isinstance(item, Mapping)
            and str(item.get("sequence_order") or "").isdigit()
        }
        selected_table_files = {
            str(item.get("relative_path") or ""): str(item.get("selected_table") or "")
            for item in current.get("source_folder_matches") or []
            if isinstance(item, Mapping) and str(item.get("selected_table") or "")
        }
        table_assignments = {
            str(item.get("relative_path") or ""): str(item.get("role") or "ignore")
            for item in current.get("source_table_matches") or []
            if isinstance(item, Mapping)
        }
        try:
            dialog = RunLayoutDialog(
                run_root,
                self,
                assignments=assignments,
                sequence_orders=sequence_orders,
                selected_table_files=selected_table_files,
                table_assignments=table_assignments,
                clearwater_selector=str(current.get("clearwater_selector") or "first"),
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self, "Could not inspect run structure", f"{type(exc).__name__}: {exc}"
            )
            return
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            rebuilt = discover_image_source_mapping(
                run_root,
                structure="single_run",
                baseline_root=(
                    Path(self.baseline_images.text())
                    if self.baseline_images.text().strip()
                    else None
                ),
                clearwater_root=(
                    Path(self.clearwater_images.text())
                    if self.clearwater_images.text().strip()
                    else None
                ),
                flat_run_id=str(current.get("run_id") or run_root.name),
                layout_template=dialog.template(),
                protocol_layout=self.protocol_layout(),
            )[0]
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self, "Could not rebuild run mapping", f"{type(exc).__name__}: {exc}"
            )
            return
        rebuilt["candidate_status"] = "requires_user_confirmation"
        rows = self.source_mapping()
        rows[row_index] = rebuilt
        self.set_mapping(rows)
        self.mapping_table.selectRow(row_index)

    def isComplete(self) -> bool:
        rows = self.source_mapping() if self.mapping_table.rowCount() else []
        return bool(rows) and all(
            row.get("candidate_status") == "user_confirmed" for row in rows
        )


def build(wizard: Any) -> ImageSourcesPage:
    page = ImageSourcesPage(wizard.identity)
    wizard.rig_protocol = page.rig_protocol
    return page
