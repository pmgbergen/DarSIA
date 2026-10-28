"""Qt dialogs for defining and reviewing per-run source-folder mappings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from darsia.fluidflower.image_formats import DEFAULT_IMAGE_FILE_POLICY
from darsia.fluidflower.source_discovery import (
    ROLE_ALTERNATE_COLOR_REFERENCE,
    ROLE_CLEARWATER,
    ROLE_EXPERIMENT,
    ROLE_IGNORE,
    ROLE_INDICATOR_REFERENCE,
    ROLE_INJECTION,
    ROLE_OVERFLOW_COMBINED,
    ROLE_OVERFLOW_IMAGES,
    ROLE_OVERFLOW_TABLE,
    ROLE_PROTOCOL_TABLE,
    build_run_layout_template,
    discover_run_candidates,
    inspect_run_structure,
)

ROLE_OPTIONS = (
    ("Ignore", ROLE_IGNORE),
    ("CO2 injection sequence (baseline starts here)", ROLE_INJECTION),
    ("CO2 continuation / resting sequence", ROLE_EXPERIMENT),
    ("Clear-water / indicator-flush sequence", ROLE_CLEARWATER),
    (
        "Measured-pH indicator buffer sequence",
        ROLE_INDICATOR_REFERENCE,
    ),
    (
        "Other known full-rig color state (not a pH buffer)",
        ROLE_ALTERNATE_COLOR_REFERENCE,
    ),
    ("Overflow scale images", ROLE_OVERFLOW_IMAGES),
)

TABLE_ROLE_OPTIONS = (
    ("Ignore this table", ROLE_IGNORE),
    ("Experimental protocol workbook", ROLE_PROTOCOL_TABLE),
    ("Overflow measurement table", ROLE_OVERFLOW_TABLE),
)


class SeriesRunsDialog(QDialog):
    """Review which immediate child folders represent experimental runs."""

    def __init__(
        self,
        series_root: Path,
        parent: Any = None,
        *,
        definitions: list[Mapping[str, Any]] | None = None,
        reserved_run_ids: set[str] | None = None,
        reserved_folders: set[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.series_root = Path(series_root).expanduser().resolve()
        self.reserved_run_ids = {
            str(value).strip().casefold() for value in reserved_run_ids or set()
        }
        self.reserved_folders = {
            str(Path(value).expanduser().resolve()).casefold()
            for value in reserved_folders or set()
            if str(value).strip()
        }
        candidates = discover_run_candidates(self.series_root)
        previous = {
            str(row.get("folder") or "").casefold(): dict(row)
            for row in definitions or []
        }
        self.candidates = [
            {**row, **previous.get(str(row["folder"]).casefold(), {})}
            for row in candidates
        ]
        self.setWindowTitle("Review detected experimental runs")
        self.resize(920, 620)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Each immediate subfolder is proposed as one run. Exclude unrelated "
            "folders and correct run IDs before DarSIA inspects their contents."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)

        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.table = QTableWidget(len(self.candidates), 4)
        self.table.setHorizontalHeaderLabels(
            ["Include", "Run ID", "Run folder", "Project status"]
        )
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        for row_index, candidate in enumerate(self.candidates):
            folder_identity = str(
                Path(str(candidate.get("folder") or "")).expanduser().resolve()
            ).casefold()
            already_registered = folder_identity in self.reserved_folders
            id_collision = (
                str(candidate.get("run_id") or "").strip().casefold()
                in self.reserved_run_ids
            )
            included = QTableWidgetItem("")
            included_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if not already_registered:
                included_flags |= Qt.ItemFlag.ItemIsUserCheckable
            included.setFlags(included_flags)
            included.setCheckState(
                Qt.CheckState.Checked
                if bool(candidate.get("included", True)) and not already_registered
                else Qt.CheckState.Unchecked
            )
            self.table.setItem(row_index, 0, included)
            run_item = QTableWidgetItem(str(candidate.get("run_id") or ""))
            if already_registered:
                run_item.setFlags(run_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_index, 1, run_item)
            folder = QTableWidgetItem(str(candidate.get("folder") or ""))
            folder.setToolTip(folder.text())
            folder.setFlags(folder.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_index, 2, folder)
            status = QTableWidgetItem(
                "Already in project"
                if already_registered
                else (
                    "Rename Run ID before including"
                    if id_collision
                    else "New experiment"
                )
            )
            status.setFlags(status.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_index, 3, status)
        self.table.itemChanged.connect(self._update_summary)
        layout.addWidget(self.table, 1)

        actions = QHBoxLayout()
        select_all = QPushButton("Include all")
        select_all.clicked.connect(lambda: self._set_all(True))
        actions.addWidget(select_all)
        select_none = QPushButton("Exclude all")
        select_none.clicked.connect(lambda: self._set_all(False))
        actions.addWidget(select_none)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        use_selection = buttons.addButton(
            "Use selected experiments", QDialogButtonBox.ButtonRole.AcceptRole
        )
        use_selection.clicked.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_summary()

    def _set_all(self, included: bool) -> None:
        state = Qt.CheckState.Checked if included else Qt.CheckState.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "run_id": self.table.item(row, 1).text().strip().casefold(),
                "folder": self.table.item(row, 2).text().strip(),
                "included": self.table.item(row, 0).checkState()
                == Qt.CheckState.Checked,
            }
            for row in range(self.table.rowCount())
        ]

    def _update_summary(self, _item: QTableWidgetItem | None = None) -> None:
        included = sum(row["included"] for row in self.definitions())
        self.summary.setText(
            f"Detected {self.table.rowCount()} folder(s); {included} selected as runs."
        )

    def _validate_and_accept(self) -> None:
        selected = [row for row in self.definitions() if row["included"]]
        if not selected:
            QMessageBox.warning(
                self, "No runs selected", "Select at least one experimental run."
            )
            return
        run_ids = [str(row["run_id"]) for row in selected]
        if any(not run_id for run_id in run_ids):
            QMessageBox.warning(
                self, "Run ID required", "Every included folder needs a run ID."
            )
            return
        if len(set(run_ids)) != len(run_ids):
            QMessageBox.warning(
                self, "Unique run IDs required", "Included run IDs must be unique."
            )
            return
        collisions = sorted(set(run_ids) & self.reserved_run_ids)
        if collisions:
            QMessageBox.warning(
                self,
                "Run ID already exists",
                "Rename these selected runs before continuing: "
                + ", ".join(collisions),
            )
            return
        self.accept()


class RunLayoutDialog(QDialog):
    """Assign source roles in one example run and export a reusable template."""

    def __init__(
        self,
        run_root: Path,
        parent: Any = None,
        *,
        assignments: Mapping[str, str] | None = None,
        sequence_orders: Mapping[str, int] | None = None,
        selected_table_files: Mapping[str, str] | None = None,
        table_assignments: Mapping[str, str] | None = None,
        clearwater_selector: str = "first",
        image_file_policy: str = DEFAULT_IMAGE_FILE_POLICY,
    ) -> None:
        super().__init__(parent)
        self.run_root = Path(run_root).expanduser().resolve()
        self.image_file_policy = image_file_policy
        inspected_records = inspect_run_structure(
            self.run_root, image_file_policy=image_file_policy
        )
        self.records = [
            dict(record)
            for record in inspected_records
            if int(record.get("image_count") or 0) > 0
        ]
        self.table_records = [
            dict(candidate)
            for record in inspected_records
            for candidate in record.get("table_candidates") or []
            if isinstance(candidate, Mapping)
        ]
        self.role_boxes: list[QComboBox] = []
        self.sequence_items: list[QTableWidgetItem] = []
        self.table_role_boxes: list[QComboBox] = []
        self.setWindowTitle(f"Define run structure from {self.run_root.name}")
        self.resize(1220, 820)
        layout = QVBoxLayout(self)

        explanation = QLabel(
            "Assign image folders and table files independently. DarSIA stores the "
            "relative patterns and proposes corresponding sources in other runs. "
            "A protocol workbook and an overflow workbook can therefore be selected "
            "separately even when both files are in the same folder."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)

        folder_heading = QLabel("Image folders")
        folder_heading.setObjectName("sectionTitle")
        layout.addWidget(folder_heading)
        self.table = QTableWidget(len(self.records), 6)
        self.table.setHorizontalHeaderLabels(
            [
                "Content",
                "CO2 order",
                "Relative folder",
                "Images",
                "First image",
                "Last image",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setMinimumHeight(220)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        reserved_orders = {
            int(value) for value in (sequence_orders or {}).values() if int(value) > 0
        }
        next_sequence_order = 1
        for row_index, record in enumerate(self.records):
            role = QComboBox()
            for label, value in ROLE_OPTIONS:
                role.addItem(label, value)
            selected_role = str(
                (assignments or {}).get(
                    str(record["relative_path"]), record["suggested_role"]
                )
            )
            if selected_role not in {value for _label, value in ROLE_OPTIONS}:
                selected_role = ROLE_IGNORE
            suggested = role.findData(selected_role)
            role.setCurrentIndex(max(0, suggested))
            self.role_boxes.append(role)
            self.table.setCellWidget(row_index, 0, role)
            order = QTableWidgetItem()
            order.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if selected_role in {ROLE_INJECTION, ROLE_EXPERIMENT}:
                relative = str(record["relative_path"])
                if relative in (sequence_orders or {}):
                    selected_order = int((sequence_orders or {})[relative])
                else:
                    while next_sequence_order in reserved_orders:
                        next_sequence_order += 1
                    selected_order = next_sequence_order
                    reserved_orders.add(selected_order)
                order.setText(str(selected_order))
                next_sequence_order = max(next_sequence_order, selected_order + 1)
            self.sequence_items.append(order)
            self.table.setItem(row_index, 1, order)
            values = (
                record["relative_path"],
                record["image_count"],
                record["first_image"],
                record["last_image"],
            )
            columns = (2, 3, 4, 5)
            for column, value in zip(columns, values, strict=True):
                item = QTableWidgetItem(str(value or ""))
                item.setToolTip(str(value or ""))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row_index, column, item)
            role.currentIndexChanged.connect(
                lambda _index, row=row_index: self._role_changed(row)
            )
            self._role_changed(row_index)
        layout.addWidget(self.table, 1)

        category_actions = QHBoxLayout()
        self.bulk_role = QComboBox()
        for label, value in ROLE_OPTIONS:
            self.bulk_role.addItem(label, value)
        category_actions.addWidget(QLabel("Assign selected folders as"))
        category_actions.addWidget(self.bulk_role, 1)
        apply_role = QPushButton("Apply")
        apply_role.clicked.connect(self._apply_bulk_role)
        category_actions.addWidget(apply_role)
        move_earlier = QPushButton("Move CO2 source earlier")
        move_earlier.clicked.connect(lambda: self._move_selected_sequence(-1))
        category_actions.addWidget(move_earlier)
        move_later = QPushButton("Move CO2 source later")
        move_later.clicked.connect(lambda: self._move_selected_sequence(1))
        category_actions.addWidget(move_later)
        layout.addLayout(category_actions)

        table_heading = QLabel("Protocol and overflow table files")
        table_heading.setObjectName("sectionTitle")
        layout.addWidget(table_heading)
        table_help = QLabel(
            "Classify every CSV or Excel file separately. A folder does not impose "
            "one shared role on all of its workbooks."
        )
        table_help.setWordWrap(True)
        table_help.setObjectName("secondaryText")
        layout.addWidget(table_help)
        self.table_files = QTableWidget(len(self.table_records), 4)
        self.table_files.setHorizontalHeaderLabels(
            ["Purpose", "Table file", "Folder", "Format"]
        )
        self.table_files.verticalHeader().setVisible(False)
        self.table_files.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        table_header = self.table_files.horizontalHeader()
        table_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        requested_roles = {
            str(key).replace("\\", "/").casefold(): str(value)
            for key, value in (table_assignments or {}).items()
        }
        legacy_selected = {
            str(key): str(value)
            for key, value in (selected_table_files or {}).items()
            if str(value).strip()
        }
        for row_index, record in enumerate(self.table_records):
            role = QComboBox()
            for label, value in TABLE_ROLE_OPTIONS:
                role.addItem(label, value)
            relative = str(record.get("relative_path") or "")
            absolute = str(record.get("path") or "")
            selected_role = next(
                (
                    requested_roles[key]
                    for key in (
                        relative.replace("\\", "/").casefold(),
                        absolute.replace("\\", "/").casefold(),
                        Path(absolute).name.casefold(),
                    )
                    if key in requested_roles
                ),
                str(record.get("suggested_role") or ROLE_IGNORE),
            )
            folder_relative = str(record.get("folder_relative_path") or ".")
            legacy_path = legacy_selected.get(folder_relative, "")
            if legacy_path and (
                Path(legacy_path).name.casefold() == Path(absolute).name.casefold()
            ):
                legacy_role = str((assignments or {}).get(folder_relative, ROLE_IGNORE))
                if legacy_role == ROLE_OVERFLOW_COMBINED:
                    legacy_role = ROLE_OVERFLOW_TABLE
                if legacy_role in {ROLE_PROTOCOL_TABLE, ROLE_OVERFLOW_TABLE}:
                    selected_role = legacy_role
            role.setCurrentIndex(max(0, role.findData(selected_role)))
            role.setToolTip(
                "Choose what this individual file contains; the choice is independent "
                "of other files in the same folder."
            )
            self.table_role_boxes.append(role)
            self.table_files.setCellWidget(row_index, 0, role)
            for column, value in enumerate(
                (
                    relative,
                    folder_relative,
                    str(record.get("suffix") or "").lstrip(".").upper(),
                ),
                start=1,
            ):
                item = QTableWidgetItem(value)
                item.setToolTip(absolute if column == 1 else value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table_files.setItem(row_index, column, item)
        self.table_files.setMinimumHeight(180)
        layout.addWidget(self.table_files)
        table_actions = QHBoxLayout()
        open_table = QPushButton("Open selected table file")
        open_table.clicked.connect(self._open_selected_table)
        table_actions.addWidget(open_table)
        table_actions.addStretch(1)
        layout.addLayout(table_actions)

        options = QFormLayout()
        self.baseline_rule = QComboBox()
        self.baseline_rule.addItem(
            "First image in the CO2 injection sequence (recommended)",
            "first_injection_image",
        )
        self.baseline_rule.setToolTip(
            "The baseline normally is the first pre-injection frame in the CO2 "
            "image sequence; it does not require a separate folder."
        )
        options.addRow("Baseline", self.baseline_rule)
        self.clearwater_selector = QComboBox()
        self.clearwater_selector.addItem("First image in selected sequence", "first")
        self.clearwater_selector.addItem("Last image in selected sequence", "last")
        selected_clearwater = self.clearwater_selector.findData(clearwater_selector)
        if selected_clearwater >= 0:
            self.clearwater_selector.setCurrentIndex(selected_clearwater)
        self.clearwater_selector.setToolTip(
            "Use first for a clear-to-indicator flush and last for an "
            "indicator-to-clear wash sequence."
        )
        options.addRow("Clear-water candidate", self.clearwater_selector)
        layout.addLayout(options)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        use_structure = buttons.addButton(
            "Use this folder structure", QDialogButtonBox.ButtonRole.AcceptRole
        )
        use_structure.clicked.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def assignments(self) -> dict[str, str]:
        return {
            str(record["relative_path"]): str(role.currentData())
            for record, role in zip(self.records, self.role_boxes, strict=True)
        }

    def sequence_orders(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for record, role, item in zip(
            self.records, self.role_boxes, self.sequence_items, strict=True
        ):
            if str(role.currentData()) not in {ROLE_INJECTION, ROLE_EXPERIMENT}:
                continue
            try:
                result[str(record["relative_path"])] = int(item.text())
            except ValueError:
                continue
        return result

    def selected_table_files(self) -> dict[str, str]:
        """Return a legacy one-table-per-folder view for older callers."""

        selected: dict[str, str] = {}
        for record, role in zip(self.table_records, self.table_role_boxes, strict=True):
            if str(role.currentData()) == ROLE_IGNORE:
                continue
            selected.setdefault(
                str(record.get("folder_relative_path") or "."),
                str(record.get("path") or ""),
            )
        return selected

    def table_assignments(self) -> dict[str, str]:
        return {
            str(record.get("relative_path") or ""): str(role.currentData())
            for record, role in zip(
                self.table_records, self.table_role_boxes, strict=True
            )
        }

    def template(self) -> dict[str, Any]:
        return build_run_layout_template(
            self.run_root,
            assignments=self.assignments(),
            sequence_orders=self.sequence_orders(),
            table_assignments=self.table_assignments(),
            baseline_rule=str(self.baseline_rule.currentData()),
            clearwater_selector=str(self.clearwater_selector.currentData()),
            image_file_policy=self.image_file_policy,
        )

    def _validate_and_accept(self) -> None:
        if not {ROLE_INJECTION, ROLE_EXPERIMENT} & set(self.assignments().values()):
            QMessageBox.warning(
                self,
                "CO2 images required",
                "Assign at least one folder as a CO2 experiment image sequence.",
            )
            return
        selected_orders = self.sequence_orders()
        selected_count = sum(
            role in {ROLE_INJECTION, ROLE_EXPERIMENT}
            for role in self.assignments().values()
        )
        if len(selected_orders) != selected_count or any(
            value < 1 for value in selected_orders.values()
        ):
            QMessageBox.warning(
                self,
                "CO2 sequence order required",
                "Every CO2 source folder needs a positive integer sequence order.",
            )
            return
        if len(set(selected_orders.values())) != len(selected_orders):
            QMessageBox.warning(
                self,
                "Unique CO2 sequence order required",
                "Each CO2 source folder needs a different sequence order.",
            )
            return
        self.accept()

    def _role_changed(self, row: int) -> None:
        role = str(self.role_boxes[row].currentData())
        item = self.sequence_items[row]
        if role not in {ROLE_INJECTION, ROLE_EXPERIMENT}:
            item.setText("")
            return
        if item.text().strip():
            return
        existing = list(self.sequence_orders().values())
        item.setText(str(max(existing, default=0) + 1))

    def _open_selected_table(self) -> None:
        row = self.table_files.currentRow()
        if row < 0:
            QMessageBox.information(
                self,
                "Select a table file",
                "Select one protocol or overflow table row first.",
            )
            return
        path = Path(str(self.table_records[row].get("path") or ""))
        if not path.is_file():
            QMessageBox.information(
                self,
                "No table file selected",
                "This row does not contain a selectable table file.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _apply_bulk_role(self) -> None:
        rows = sorted(
            {index.row() for index in self.table.selectionModel().selectedRows()}
        )
        target = str(self.bulk_role.currentData())
        for row in rows:
            index = self.role_boxes[row].findData(target)
            if index >= 0:
                self.role_boxes[row].setCurrentIndex(index)

    def _move_selected_sequence(self, direction: int) -> None:
        rows = sorted(
            {index.row() for index in self.table.selectionModel().selectedRows()}
        )
        if len(rows) != 1:
            return
        row = rows[0]
        role = str(self.role_boxes[row].currentData())
        if role not in {ROLE_INJECTION, ROLE_EXPERIMENT}:
            return
        ordered = sorted(
            (
                (int(item.text()), index)
                for index, (candidate_role, item) in enumerate(
                    zip(self.role_boxes, self.sequence_items, strict=True)
                )
                if str(candidate_role.currentData())
                in {ROLE_INJECTION, ROLE_EXPERIMENT}
                and item.text().strip().isdigit()
            ),
            key=lambda value: value[0],
        )
        positions = [index for _, index in ordered]
        current = positions.index(row)
        target = current + direction
        if not 0 <= target < len(ordered):
            return
        other_row = positions[target]
        current_text = self.sequence_items[row].text()
        self.sequence_items[row].setText(self.sequence_items[other_row].text())
        self.sequence_items[other_row].setText(current_text)


def _mapping_paths(mapping: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "experiment_folder",
        "experiment_folders",
        "baseline_candidate",
        "clearwater_candidate",
        "clearwater_folders",
        "indicator_reference_folders",
        "alternate_color_reference_folders",
        "protocol_table",
        "protocol_tables",
        "overflow_image_folders",
        "overflow_table",
        "overflow_tables",
    ):
        value = mapping.get(key)
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text and text not in values:
                values.append(text)
    return values


class SourceMappingDetailsDialog(QDialog):
    """Show one complete per-run proposal and provide direct source access."""

    def __init__(self, mapping: Mapping[str, Any], parent: Any = None) -> None:
        super().__init__(parent)
        self.mapping = dict(mapping)
        run_id = str(mapping.get("run_id") or "run")
        self.setWindowTitle(f"Inspect source mapping: {run_id.upper()}")
        self.resize(900, 620)
        layout = QVBoxLayout(self)
        summary = QLabel(
            "Inspect the complete proposal below. Opening a source only displays "
            "the existing file or folder; DarSIA never changes it."
        )
        summary.setWordWrap(True)
        summary.setObjectName("secondaryText")
        layout.addWidget(summary)

        experiment_folders = mapping.get("experiment_folders") or []
        clearwater_folders = mapping.get("clearwater_folders") or []
        indicator_reference_folders = mapping.get("indicator_reference_folders") or []
        alternate_color_reference_folders = (
            mapping.get("alternate_color_reference_folders") or []
        )
        overflow_folders = mapping.get("overflow_image_folders") or []
        co2_sequences = mapping.get("co2_sequences") or []
        warnings = mapping.get("discovery_warnings") or []
        folder_matches = mapping.get("source_folder_matches") or []
        table_matches = mapping.get("source_table_matches") or []
        body = [
            f"Run: {run_id.upper()}",
            f"Status: {mapping.get('mapping_status') or 'unknown'}",
            f"Image-file policy: {mapping.get('image_file_policy') or 'prefer_processed'}",
            (
                "Available CO2 files: "
                f"{int((mapping.get('image_file_counts') or {}).get('processed') or 0)} "
                "processed, "
                f"{int((mapping.get('image_file_counts') or {}).get('raw') or 0)} RAW"
            ),
            f"CO2 images: {int(mapping.get('image_count') or 0)}",
            "Ordered CO2 image sequence:",
            *(
                f"  {int(value.get('order') or 0)}. "
                f"{value.get('kind') or 'unknown'} | "
                f"{int(value.get('image_count') or 0)} images | "
                f"{value.get('folder') or ''}"
                for value in co2_sequences
                if isinstance(value, Mapping)
            ),
            *(
                f"  {index}. {value}"
                for index, value in enumerate(experiment_folders, start=1)
                if not co2_sequences
            ),
            "",
            f"Baseline ({mapping.get('baseline_rule') or 'manual'}):",
            f"  {mapping.get('baseline_candidate') or 'not found'}",
            "",
            f"Clear-water ({mapping.get('clearwater_selector') or 'manual'}):",
            f"  {mapping.get('clearwater_candidate') or 'not found'}",
            *(f"  folder: {value}" for value in clearwater_folders),
            "",
            "Indicator-response reference images:",
            f"  Images: {int(mapping.get('indicator_reference_image_count') or 0)}",
            *(f"  folder: {value}" for value in indicator_reference_folders),
            *(["  not provided"] if not indicator_reference_folders else []),
            "",
            "Other known full-rig color states:",
            f"  Images: {int(mapping.get('alternate_color_reference_image_count') or 0)}",
            *(f"  folder: {value}" for value in alternate_color_reference_folders),
            *(["  not provided"] if not alternate_color_reference_folders else []),
            "",
            "Experimental protocol workbook:",
            f"  {mapping.get('protocol_table') or 'not found'}",
            "",
            f"Overflow scale images: {int(mapping.get('overflow_image_count') or 0)}",
            *(f"  {value}" for value in overflow_folders),
            "Overflow table:",
            f"  {mapping.get('overflow_table') or 'not found'}",
            "",
            "Folder interpretation:",
            *(
                "  "
                f"{value.get('relative_path') or '.'} -> "
                f"{value.get('role') or 'ignore'} "
                f"[{value.get('match') or 'unknown'}; "
                f"{int(value.get('image_count') or 0)} images, "
                f"{int(value.get('table_count') or 0)} tables]"
                for value in folder_matches
                if isinstance(value, Mapping)
            ),
            "",
            "Table-file interpretation:",
            *(
                "  "
                f"{value.get('relative_path') or Path(str(value.get('path') or '')).name} -> "
                f"{value.get('role') or 'ignore'} "
                f"[{value.get('match') or 'unknown'}]"
                for value in table_matches
                if isinstance(value, Mapping)
            ),
            *([] if table_matches else ["  no CSV or Excel files found"]),
            "",
            "Discovery notes:",
            *(f"  {value}" for value in warnings),
            *([] if warnings else ["  none"]),
        ]
        details = QPlainTextEdit("\n".join(body))
        details.setReadOnly(True)
        layout.addWidget(details, 1)

        source_row = QHBoxLayout()
        self.sources = QComboBox()
        for path in _mapping_paths(mapping):
            self.sources.addItem(path, path)
        source_row.addWidget(self.sources, 1)
        open_source = QPushButton("Open selected source")
        open_source.setEnabled(self.sources.count() > 0)
        open_source.clicked.connect(self._open_source)
        source_row.addWidget(open_source)
        layout.addLayout(source_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _open_source(self) -> None:
        path = Path(str(self.sources.currentData() or ""))
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


__all__ = [
    "ROLE_OPTIONS",
    "TABLE_ROLE_OPTIONS",
    "RunLayoutDialog",
    "SeriesRunsDialog",
    "SourceMappingDetailsDialog",
]
