"""Reusable Qt dialog for applying one folder structure to selected runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from darsia.fluidflower.pages.setup.source_layout_dialog import RunLayoutDialog
from darsia.fluidflower.ui.common import set_state_label


def _template_editor_values(
    template: Mapping[str, Any], run_root: Path
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Return saved editor values only when they belong to this example run."""

    saved_root = str(template.get("example_run_root") or "").strip()
    try:
        same_example = (
            bool(saved_root) and Path(saved_root).resolve() == run_root.resolve()
        )
    except OSError:
        same_example = False
    if not same_example:
        return {}, {}, {}
    rows = [
        dict(row) for row in template.get("folders") or [] if isinstance(row, Mapping)
    ]
    assignments = {
        str(row.get("relative_path") or ""): str(row.get("role") or "ignore")
        for row in rows
    }
    sequence_orders = {
        str(row.get("relative_path") or ""): int(row["sequence_order"])
        for row in rows
        if str(row.get("sequence_order") or "").isdigit()
    }
    selected_tables = {
        str(row.get("relative_path") or ""): str(row.get("selected_table_name") or "")
        for row in rows
        if str(row.get("selected_table_name") or "")
    }
    return assignments, sequence_orders, selected_tables


def _template_table_assignments(
    template: Mapping[str, Any], run_root: Path
) -> dict[str, str]:
    """Return file-level table roles for the saved representative run."""

    saved_root = str(template.get("example_run_root") or "").strip()
    try:
        same_example = (
            bool(saved_root) and Path(saved_root).resolve() == run_root.resolve()
        )
    except OSError:
        same_example = False
    if not same_example:
        return {}
    return {
        str(row.get("relative_path") or ""): str(row.get("role") or "ignore")
        for row in template.get("table_files") or []
        if isinstance(row, Mapping)
    }


class GlobalRunStructureDialog(QDialog):
    """Choose a reusable run layout and the registered runs it should update."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        parent: Any = None,
        *,
        template: Mapping[str, Any] | None = None,
        template_revision: int = 0,
        current_run_id: str = "",
    ) -> None:
        super().__init__(parent)
        self.rows = [deepcopy(dict(row)) for row in rows]
        self._template = deepcopy(dict(template or {}))
        self._template_changed = False
        self.template_revision = int(template_revision or 0)
        self.current_run_id = str(current_run_id or "")
        self.setWindowTitle("Review the expected folder structure")
        self.resize(1120, 720)
        layout = QVBoxLayout(self)

        heading = QLabel("Expected folder structure for the experimental series")
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        explanation = QLabel(
            "Use one representative experiment to define the meaning and order of "
            "its source folders. The same expected pattern can then be tested on all "
            "registered experiments or only the experiments you select below. No raw "
            "file is moved or modified."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)

        representative_row = QHBoxLayout()
        representative_row.addWidget(QLabel("Representative experiment"))
        self.representative = QComboBox()
        for row in self.rows:
            run_id = str(row.get("run_id") or "")
            mapping = row.get("source_mapping") or {}
            self.representative.addItem(
                str(row.get("label") or run_id.upper()),
                {
                    "run_id": run_id,
                    "root": str(mapping.get("experiment_folder") or ""),
                },
            )
        saved_example_root = str(self._template.get("example_run_root") or "")
        saved_example_name = str(
            self._template.get("example_run_name") or ""
        ).casefold()
        selected_index = next(
            (
                index
                for index in range(self.representative.count())
                if self._matches_saved_example(
                    self.representative.itemData(index) or {},
                    root=saved_example_root,
                    name=saved_example_name,
                )
            ),
            -1,
        )
        if selected_index < 0:
            selected_index = next(
                (
                    index
                    for index in range(self.representative.count())
                    if str(
                        (self.representative.itemData(index) or {}).get("run_id") or ""
                    )
                    == self.current_run_id
                ),
                -1,
            )
        if selected_index >= 0:
            self.representative.setCurrentIndex(selected_index)
        representative_row.addWidget(self.representative, 1)
        self.review_structure = QPushButton("Review folder roles and order...")
        self.review_structure.clicked.connect(self._review_structure)
        representative_row.addWidget(self.review_structure)
        layout.addLayout(representative_row)

        self.template_summary = QLabel()
        self.template_summary.setWordWrap(True)
        layout.addWidget(self.template_summary)

        selection_heading = QLabel("Experiments that will be re-evaluated")
        selection_heading.setObjectName("sectionTitle")
        layout.addWidget(selection_heading)
        selection_help = QLabel(
            "Checked experiments receive a new, versioned source mapping after the "
            "proposed matches have been reviewed. Unchecked experiments keep their "
            "current mapping and results unchanged."
        )
        selection_help.setWordWrap(True)
        selection_help.setObjectName("secondaryText")
        layout.addWidget(selection_help)

        self.table = QTableWidget(len(self.rows), 6)
        self.table.setHorizontalHeaderLabels(
            [
                "Apply",
                "Experiment",
                "CO2 images",
                "Source revision",
                "Structure status",
                "Run folder",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in range(5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        for row_index, row in enumerate(self.rows):
            run_id = str(row.get("run_id") or "")
            mapping = row.get("source_mapping") or {}
            applied_revision = int(mapping.get("layout_template_revision") or 0)
            selection = QTableWidgetItem("")
            selection.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            selection.setCheckState(Qt.CheckState.Checked)
            selection.setData(Qt.ItemDataRole.UserRole, run_id)
            self.table.setItem(row_index, 0, selection)
            values = (
                str(row.get("label") or run_id.upper()),
                int(mapping.get("image_count") or 0),
                int(row.get("source_revision") or 1),
                (
                    f"Current (r{applied_revision})"
                    if self.template_revision
                    and applied_revision == self.template_revision
                    else (
                        f"Older (r{applied_revision})"
                        if applied_revision
                        else "Not recorded"
                    )
                ),
                str(mapping.get("experiment_folder") or ""),
            )
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(str(value))
                self.table.setItem(row_index, column, item)
        self.table.itemChanged.connect(self._refresh_actions)
        layout.addWidget(self.table, 1)

        selection_actions = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_selected("all"))
        selection_actions.addWidget(select_all)
        select_current = QPushButton("Only selected experiment")
        select_current.clicked.connect(lambda: self._set_selected("current"))
        selection_actions.addWidget(select_current)
        clear = QPushButton("Clear selection")
        clear.clicked.connect(lambda: self._set_selected("none"))
        selection_actions.addWidget(clear)
        selection_actions.addStretch(1)
        self.selection_summary = QLabel()
        self.selection_summary.setObjectName("secondaryText")
        selection_actions.addWidget(self.selection_summary)
        layout.addLayout(selection_actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.continue_button = buttons.addButton(
            "Preview mappings before saving...",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.continue_button.clicked.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh_template_summary()
        self._refresh_actions()

    @staticmethod
    def _matches_saved_example(
        data: Mapping[str, Any], *, root: str, name: str
    ) -> bool:
        candidate_root = str(data.get("root") or "").strip()
        if root.strip() and candidate_root:
            try:
                if Path(candidate_root).resolve() == Path(root).resolve():
                    return True
            except OSError:
                if candidate_root.casefold() == root.strip().casefold():
                    return True
        run_id = str(data.get("run_id") or "").casefold()
        return bool(name) and run_id == name

    def _selected_example(self) -> tuple[str, Path] | None:
        data = self.representative.currentData() or {}
        root = Path(str(data.get("root") or "")).expanduser()
        if not root.is_dir():
            return None
        return str(data.get("run_id") or ""), root.resolve()

    def _review_structure(self) -> None:
        selected = self._selected_example()
        if selected is None:
            QMessageBox.warning(
                self,
                "Representative experiment unavailable",
                "Select an experiment whose source folder is currently accessible.",
            )
            return
        _run_id, run_root = selected
        assignments, sequence_orders, selected_tables = _template_editor_values(
            self._template, run_root
        )
        table_assignments = _template_table_assignments(self._template, run_root)
        try:
            dialog = RunLayoutDialog(
                run_root,
                self,
                assignments=assignments,
                sequence_orders=sequence_orders,
                selected_table_files=selected_tables,
                table_assignments=table_assignments,
                clearwater_selector=str(
                    self._template.get("clearwater_selector") or "first"
                ),
                image_file_policy=str(
                    self._template.get("image_file_policy") or "prefer_processed"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self,
                "Could not inspect the representative experiment",
                f"{type(exc).__name__}: {exc}",
            )
            return
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._template = dialog.template()
            self._template_changed = True
            self._refresh_template_summary()
            self._refresh_actions()

    def _set_selected(self, mode: str) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            run_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
            checked = mode == "all" or (
                mode == "current" and run_id == self.current_run_id
            )
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )

    def _refresh_template_summary(self) -> None:
        folders = [
            row
            for row in self._template.get("folders") or []
            if isinstance(row, Mapping)
        ]
        tables = [
            row
            for row in self._template.get("table_files") or []
            if isinstance(row, Mapping)
        ]
        if not folders:
            set_state_label(
                self.template_summary,
                "incomplete",
                "INCOMPLETE: No expected structure is defined or saved. Review one "
                "representative experiment before continuing.",
            )
            return
        example = str(self._template.get("example_run_name") or "unknown")
        if self._template_changed:
            set_state_label(
                self.template_summary,
                "pending",
                f"NOT SAVED: The edited structure is based on {example} and contains "
                f"{len(folders)} image-folder and {len(tables)} table-file role(s). "
                "Continue to preview the "
                "mappings, then use the Save button in that preview.",
            )
        else:
            revision = f" r{self.template_revision}" if self.template_revision else ""
            set_state_label(
                self.template_summary,
                "saved",
                f"SAVED{revision}: The active structure is based on {example} and "
                f"contains {len(folders)} image-folder and {len(tables)} table-file "
                "role(s). Nothing changes "
                "until mappings are previewed and explicitly saved.",
            )

    def _refresh_actions(self, _item: QTableWidgetItem | None = None) -> None:
        count = len(self.selected_run_ids())
        self.selection_summary.setText(f"{count} of {len(self.rows)} selected")
        self.continue_button.setEnabled(
            count > 0 and bool(self._template.get("folders"))
        )

    def selected_run_ids(self) -> list[str]:
        return [
            str(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) or "")
            for row in range(self.table.rowCount())
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        ]

    def layout_template(self) -> dict[str, Any]:
        return deepcopy(self._template)

    def _validate_and_accept(self) -> None:
        if not self.selected_run_ids():
            QMessageBox.warning(
                self, "No experiments selected", "Select at least one experiment."
            )
            return
        if not self._template.get("folders"):
            QMessageBox.warning(
                self,
                "Folder structure required",
                "Review the folder roles in one representative experiment first.",
            )
            return
        self.accept()


__all__ = ["GlobalRunStructureDialog"]
