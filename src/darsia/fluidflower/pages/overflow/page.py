"""Qt workflow for selecting, reading, and revising overflow measurements."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.modules import dependencies_satisfied
from darsia.fluidflower.overflow import (
    OverflowRevisionStore,
    apply_registered_overflow_mode,
    normalize_overflow_config,
    overflow_run_source_digest,
    scan_overflow_run,
    validate_overflow_config,
)
from darsia.fluidflower.overflow_workflow import scale_reader_review_progress
from darsia.fluidflower.pages.overflow.reader_dialog import ScaleImageReaderDialog
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.ui.common import (
    DIRECTORY_DIALOG_OPTIONS,
    ModulePage,
    PathField,
)

MODE_LABELS = {
    "none": "No overflow measurements",
    "scale_images": "Read scale-display images",
    "table": "Import CSV or Excel",
}


class OverflowScanWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__()
        self.rows = deepcopy(rows)
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    @Slot()
    def run(self) -> None:
        results: list[dict[str, Any]] = []
        try:
            total = len(self.rows)
            for index, row in enumerate(self.rows, start=1):
                if self.cancel_requested:
                    self.cancelled.emit()
                    return
                run_id = str(row.get("run_id") or "")
                self.progress.emit(index - 1, total, run_id.upper())
                results.append({"run_id": run_id, "scan": scan_overflow_run(row)})
            self.progress.emit(total, total, "Source scan complete")
        except Exception as exc:  # noqa: BLE001 - visible worker boundary
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(results)


class OverflowPage(ModulePage):
    """Per-experiment overflow source selection after data-integrity review."""

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__(project, "overflow")
        self.store = OverflowRevisionStore(project)
        self.config = normalize_overflow_config(project, self.store.current_config())
        self._loaded_run_id = ""
        self._updating = False
        self._global_updating = False
        self._scope = "global"
        self._dirty = self.store.has_draft()
        self._thread: QThread | None = None
        self._worker: OverflowScanWorker | None = None
        self._active_scan_run_ids: tuple[str, ...] = ()
        self._reader_dialog: ScaleImageReaderDialog | None = None
        self._draft_timer = QTimer(self)
        self._draft_timer.setSingleShot(True)
        self._draft_timer.setInterval(500)
        self._draft_timer.timeout.connect(self._persist_working_draft)

        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "Setup only discovers possible overflow sources. This module decides which "
            "source is active for each experiment, validates it, and keeps every change "
            "as a new revision. Original images and tables remain read-only."
        )
        explanation = QLabel(
            "Choose a source per experiment. Scan source checks availability and file "
            "contents. For scale images, Open scale-image reader opens the OCR and "
            "manual-verification workspace inside this application. You can later "
            "replace those images with an existing Excel file without rebuilding the "
            "project."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        self.work_layout.addWidget(explanation)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Overflow settings"))
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
        self.work_layout.addLayout(scope_row)
        self.scope_explanation = QLabel()
        self.scope_explanation.setWordWrap(True)
        self.scope_explanation.setObjectName("secondaryText")
        self.work_layout.addWidget(self.scope_explanation)

        self.global_group = QGroupBox("Global overflow-source assignment")
        global_layout = QVBoxLayout(self.global_group)
        global_note = QLabel(
            "Apply one source policy to all or selected experiments. DarSIA resolves "
            "the registered source separately for every experiment and records those "
            "resolved paths in the final revision. Experiments without the requested "
            "source are never changed silently."
        )
        global_note.setWordWrap(True)
        global_note.setObjectName("secondaryText")
        global_layout.addWidget(global_note)
        global_form = QFormLayout()
        self.global_mode = QComboBox()
        self.global_mode.addItem(
            "Use registered Excel or CSV file for each experiment", "table"
        )
        self.global_mode.addItem(
            "Use registered scale-display image folders for each experiment",
            "scale_images",
        )
        self.global_mode.addItem("No overflow measurements", "none")
        self.global_mode.setToolTip(
            "This bulk choice does not assume that every experiment uses the same "
            "file. It selects each experiment's own source registered during setup."
        )
        global_form.addRow("Source policy", self.global_mode)
        global_layout.addLayout(global_form)
        self.global_table = QTableWidget(0, 5)
        self.global_table.setHorizontalHeaderLabels(
            [
                "Apply",
                "Experiment",
                "Current source",
                "Registered source for this policy",
                "Assignment",
            ]
        )
        self.global_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.global_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.global_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.global_table.verticalHeader().setVisible(False)
        global_header = self.global_table.horizontalHeader()
        for column in (0, 1, 2, 4):
            global_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        global_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.global_table.setMinimumHeight(280)
        self.global_table.itemChanged.connect(self._global_selection_changed)
        global_layout.addWidget(self.global_table, 1)
        global_actions = QHBoxLayout()
        self.select_all_eligible = QPushButton("Select all available")
        self.select_all_eligible.clicked.connect(
            lambda: self._set_global_selection(True)
        )
        global_actions.addWidget(self.select_all_eligible)
        self.clear_global_selection = QPushButton("Clear selection")
        self.clear_global_selection.clicked.connect(
            lambda: self._set_global_selection(False)
        )
        global_actions.addWidget(self.clear_global_selection)
        self.apply_global_source = QPushButton("Review and apply to selected...")
        self.apply_global_source.clicked.connect(self._apply_global_source)
        global_actions.addWidget(self.apply_global_source)
        global_actions.addStretch(1)
        self.global_selection_summary = QLabel()
        self.global_selection_summary.setObjectName("secondaryText")
        global_actions.addWidget(self.global_selection_summary)
        global_layout.addLayout(global_actions)
        self.work_layout.addWidget(self.global_group, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.individual_splitter = splitter
        self.run_table = QTableWidget(0, 8)
        self.run_table.setHorizontalHeaderLabels(
            [
                "Experiment",
                "Active source",
                "Location",
                "Scan",
                "Source items",
                "Reviewed images",
                "Reviewed time span",
                "Notes",
            ]
        )
        self.run_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.run_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.run_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.run_table.verticalHeader().setVisible(False)
        header = self.run_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.run_table.itemSelectionChanged.connect(self._selection_changed)
        splitter.addWidget(self.run_table)

        self.editor = QGroupBox("Selected experiment")
        editor_layout = QVBoxLayout(self.editor)
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem(MODE_LABELS["none"], "none")
        self.mode.addItem(MODE_LABELS["scale_images"], "scale_images")
        self.mode.addItem(MODE_LABELS["table"], "table")
        self.mode.setToolTip(
            "This choice is per experiment and may be revised later. Changing it "
            "does not modify or delete the previous source."
        )
        self.mode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("Active source", self.mode)
        editor_layout.addLayout(form)

        self.source_stack = QStackedWidget()
        none_panel = QWidget()
        none_layout = QVBoxLayout(none_panel)
        none_layout.setContentsMargins(0, 4, 0, 4)
        none_note = QLabel(
            "Use this when no overflow measurements exist or overflow is not relevant."
        )
        none_note.setWordWrap(True)
        none_layout.addWidget(none_note)
        none_layout.addStretch(1)
        self.source_stack.addWidget(none_panel)

        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 4, 0, 4)
        image_note = QLabel(
            "Folders containing camera images of the scale display. Every selected "
            "folder belongs to this experiment."
        )
        image_note.setWordWrap(True)
        image_layout.addWidget(image_note)
        self.image_folders = QListWidget()
        self.image_folders.setMinimumHeight(105)
        image_layout.addWidget(self.image_folders)
        image_actions = QHBoxLayout()
        self.add_image_folder = QPushButton("Add folder...")
        self.add_image_folder.clicked.connect(self._add_image_folder)
        image_actions.addWidget(self.add_image_folder)
        self.remove_image_folder = QPushButton("Remove selected")
        self.remove_image_folder.clicked.connect(self._remove_image_folder)
        image_actions.addWidget(self.remove_image_folder)
        image_actions.addStretch(1)
        image_layout.addLayout(image_actions)
        self.source_stack.addWidget(image_panel)

        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 4, 0, 4)
        table_note = QLabel(
            "Select an existing overflow table. CSV and Excel are accepted; time and "
            "weight columns are mapped later during temporal synchronization."
        )
        table_note.setWordWrap(True)
        table_layout.addWidget(table_note)
        self.table_path = PathField(
            directory=False,
            file_filter=(
                "Overflow tables (*.xlsx *.xls *.csv *.tsv *.txt);;All files (*)"
            ),
            placeholder="CSV or Excel file",
        )
        self.table_path.changed.connect(self._source_changed)
        table_layout.addWidget(self.table_path)
        table_layout.addStretch(1)
        self.source_stack.addWidget(table_panel)
        editor_layout.addWidget(self.source_stack)

        source_actions = QHBoxLayout()
        self.restore_discovered = QPushButton("Use discovered source")
        self.restore_discovered.clicked.connect(self._restore_discovered_source)
        source_actions.addWidget(self.restore_discovered)
        self.use_reviewed_table = QPushButton("Use reviewed workbook")
        self.use_reviewed_table.clicked.connect(self._use_reviewed_workbook)
        source_actions.addWidget(self.use_reviewed_table)
        source_actions.addStretch(1)
        editor_layout.addLayout(source_actions)

        self.scan_report = QTextEdit()
        self.scan_report.setReadOnly(True)
        self.scan_report.setMinimumHeight(130)
        editor_layout.addWidget(self.scan_report)
        splitter.addWidget(self.editor)
        splitter.setSizes([760, 500])
        self.work_layout.addWidget(splitter, 1)

        scan_actions = QHBoxLayout()
        self.scan_selected = QPushButton("Scan selected source")
        self.scan_selected.clicked.connect(lambda: self._start_scan(False))
        scan_actions.addWidget(self.scan_selected)
        self.scan_all = QPushButton("Scan all experiments")
        self.scan_all.clicked.connect(lambda: self._start_scan(True))
        scan_actions.addWidget(self.scan_all)
        self.cancel_scan = QPushButton("Cancel")
        self.cancel_scan.setEnabled(False)
        self.cancel_scan.clicked.connect(self._cancel_scan)
        scan_actions.addWidget(self.cancel_scan)
        self.open_reader = QPushButton("Open scale-image reader")
        self.open_reader.clicked.connect(self._open_scale_reader)
        scan_actions.addWidget(self.open_reader)
        self.open_output = QPushButton("Open reader output")
        self.open_output.clicked.connect(self._open_reader_output)
        scan_actions.addWidget(self.open_output)
        scan_actions.addStretch(1)
        self.work_layout.addLayout(scan_actions)

        progress_row = QHBoxLayout()
        self.scan_progress = QProgressBar()
        self.scan_progress.setRange(0, 1)
        self.scan_progress.setValue(0)
        progress_row.addWidget(self.scan_progress, 1)
        self.scan_message = QLabel("No source scan is running.")
        progress_row.addWidget(self.scan_message)
        self.work_layout.addLayout(progress_row)

        revision_group = QGroupBox("Save overflow decision")
        revision_layout = QVBoxLayout(revision_group)
        reviewer_row = QHBoxLayout()
        self.reviewer = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.reviewer.setPlaceholderText("Reviewer name")
        reviewer_row.addWidget(QLabel("Reviewer"))
        reviewer_row.addWidget(self.reviewer)
        self.revision_note = QLineEdit()
        self.revision_note.setPlaceholderText(
            "Reason, for example: replaced camera readings with validated workbook"
        )
        reviewer_row.addWidget(QLabel("Revision note"))
        reviewer_row.addWidget(self.revision_note, 1)
        self.save_revision = QPushButton("Save overflow revision")
        self.save_revision.clicked.connect(self._save_revision)
        reviewer_row.addWidget(self.save_revision)
        revision_layout.addLayout(reviewer_row)
        self.revision_status = QLabel()
        self.revision_status.setWordWrap(True)
        revision_layout.addWidget(self.revision_status)
        self.work_layout.addWidget(revision_group)

        self.global_scope_button.clicked.connect(
            lambda: self.set_overflow_scope("global")
        )
        self.individual_scope_button.clicked.connect(
            lambda: self.set_overflow_scope("individual")
        )
        self.global_mode.currentIndexChanged.connect(self._global_mode_changed)

        self._populate_table()
        self.set_overflow_scope("global")
        self.refresh()

    def _rows(self) -> list[dict[str, Any]]:
        return self.config.setdefault("runs", [])

    def _row(self, run_id: str) -> dict[str, Any] | None:
        normalized = str(run_id).strip().casefold()
        return next(
            (
                row
                for row in self._rows()
                if str(row.get("run_id") or "").strip().casefold() == normalized
            ),
            None,
        )

    def _selected_run_id(self) -> str:
        selected = self.run_table.selectedItems()
        if not selected:
            return ""
        return str(selected[0].data(Qt.ItemDataRole.UserRole) or "")

    @staticmethod
    def _registered_source_for_mode(
        row: dict[str, Any], mode: str
    ) -> tuple[bool, str, str]:
        if mode == "none":
            return True, "No source will be used", "Explicitly disable overflow"
        if mode == "table":
            sources = [
                str(value)
                for value in row.get("discovered_tables") or []
                if str(value).strip()
            ]
            missing = "No Excel or CSV overflow file is registered"
        else:
            sources = [
                str(value)
                for value in row.get("discovered_image_folders") or []
                if str(value).strip()
            ]
            missing = "No scale-display image folder is registered"
        if not sources:
            return False, missing, missing
        suffix = f" (+{len(sources) - 1} more)" if len(sources) > 1 else ""
        return True, sources[0] + suffix, "\n".join(sources)

    @staticmethod
    def _assignment_label(row: dict[str, Any]) -> str:
        return {
            "global_registered_source": "Global policy",
            "individual_override": "Individual override",
            "setup_discovery": "Initial setup",
        }.get(str(row.get("source_origin") or ""), "Saved selection")

    def _global_selected_run_ids(self) -> list[str]:
        selected: list[str] = []
        for row_index in range(self.global_table.rowCount()):
            item = self.global_table.item(row_index, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(str(item.data(Qt.ItemDataRole.UserRole) or ""))
        return selected

    def _populate_global_table(self, *, select_eligible: bool = False) -> None:
        previous = set(self._global_selected_run_ids())
        first_population = self.global_table.rowCount() == 0
        mode = str(self.global_mode.currentData() or "table")
        self._global_updating = True
        try:
            rows = self._rows()
            self.global_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                run_id = str(row.get("run_id") or "")
                available, registered, registered_tooltip = (
                    self._registered_source_for_mode(row, mode)
                )
                selection = QTableWidgetItem("")
                selection.setData(Qt.ItemDataRole.UserRole, run_id)
                if available:
                    selection.setFlags(
                        Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    checked = select_eligible or first_population or run_id in previous
                    selection.setCheckState(
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                    )
                else:
                    selection.setFlags(Qt.ItemFlag.ItemIsSelectable)
                    selection.setCheckState(Qt.CheckState.Unchecked)
                    selection.setToolTip(registered_tooltip)
                self.global_table.setItem(row_index, 0, selection)
                values = (
                    run_id.upper(),
                    MODE_LABELS.get(str(row.get("mode") or "none"), "Unknown"),
                    registered,
                    self._assignment_label(row),
                )
                for column, value in enumerate(values, start=1):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if column == 3:
                        item.setToolTip(registered_tooltip)
                    self.global_table.setItem(row_index, column, item)
        finally:
            self._global_updating = False
        self._refresh_global_selection_summary()

    def _refresh_global_selection_summary(self) -> None:
        mode = str(self.global_mode.currentData() or "table")
        available = sum(
            self._registered_source_for_mode(row, mode)[0] for row in self._rows()
        )
        selected = len(self._global_selected_run_ids())
        unavailable = len(self._rows()) - available
        suffix = f"; {unavailable} unavailable" if unavailable else ""
        self.global_selection_summary.setText(
            f"{selected} selected of {available} available{suffix}"
        )
        self.apply_global_source.setEnabled(
            bool(selected)
            and dependencies_satisfied("overflow", self.project.module_states)
            and not (self._thread is not None and self._thread.isRunning())
        )

    def _global_selection_changed(self, _item: QTableWidgetItem) -> None:
        if not self._global_updating:
            self._refresh_global_selection_summary()

    def _set_global_selection(self, selected: bool) -> None:
        self._global_updating = True
        try:
            for row_index in range(self.global_table.rowCount()):
                item = self.global_table.item(row_index, 0)
                if item is None or not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
                    continue
                item.setCheckState(
                    Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked
                )
        finally:
            self._global_updating = False
        self._refresh_global_selection_summary()

    def _global_mode_changed(self) -> None:
        if not self._global_updating:
            self._populate_global_table(select_eligible=True)

    def _apply_global_source(self) -> None:
        self._commit_editor(self._loaded_run_id)
        run_ids = self._global_selected_run_ids()
        if not run_ids:
            QMessageBox.information(
                self,
                "No experiments selected",
                "Select at least one available experiment first.",
            )
            return
        mode = str(self.global_mode.currentData() or "table")
        candidate, summary = apply_registered_overflow_mode(
            self.config, mode=mode, run_ids=run_ids
        )
        applied = [str(value).upper() for value in summary["applied_run_ids"]]
        changed = [str(value).upper() for value in summary["changed_run_ids"]]
        skipped = {
            str(key).upper(): str(value)
            for key, value in dict(summary["skipped"]).items()
        }
        lines = [
            f"Policy: {MODE_LABELS[mode]}",
            f"Experiments selected: {len(run_ids)}",
            f"Sources that will change: {len(changed)}",
            f"Already using this source: {len(applied) - len(changed)}",
        ]
        if skipped:
            lines.append(f"Unavailable and left unchanged: {len(skipped)}")
            lines.extend(f"- {run_id}: {reason}" for run_id, reason in skipped.items())
        lines.extend(
            [
                "",
                "This updates the recoverable working state only. Scan the selected "
                "sources and save an overflow revision afterwards.",
            ]
        )
        answer = QMessageBox.question(
            self,
            "Apply global overflow policy?",
            "\n".join(lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not applied:
            QMessageBox.warning(
                self,
                "No compatible sources",
                "None of the selected experiments has the requested registered source.",
            )
            return
        self.config = candidate
        self._dirty = True
        self._persist_working_draft()
        self._populate_table(self._loaded_run_id)
        self.set_overflow_scope("global")
        self.scan_message.setText(
            f"Global policy applied to {len(applied)} experiment(s). "
            "Scan the checked sources, then save an overflow revision."
        )
        self.refresh()

    def set_overflow_scope(self, scope: str) -> None:
        if scope not in {"global", "individual"}:
            raise ValueError(f"Unknown overflow-settings scope: {scope}")
        self._scope = scope
        global_scope = scope == "global"
        self.global_scope_button.setChecked(global_scope)
        self.individual_scope_button.setChecked(not global_scope)
        self.global_group.setVisible(global_scope)
        self.individual_splitter.setVisible(not global_scope)
        self.open_reader.setVisible(not global_scope)
        self.scan_selected.setText(
            "Scan checked experiments" if global_scope else "Scan selected source"
        )
        self.scope_explanation.setText(
            (
                "Use one policy across the series, either for every compatible "
                "experiment or for a checked subset. Each experiment still retains "
                "its own resolved file or image folders for provenance."
            )
            if global_scope
            else (
                "Use this for exceptions. Changes affect only the selected experiment "
                "and are saved together with the next overflow revision."
            )
        )
        self._refresh_action_state()

    @staticmethod
    def _source_location(row: dict[str, Any]) -> str:
        mode = str(row.get("mode") or "none")
        if mode == "scale_images":
            folders = [str(value) for value in row.get("image_folders") or []]
            if not folders:
                return "Not selected"
            suffix = f" (+{len(folders) - 1} more)" if len(folders) > 1 else ""
            return folders[0] + suffix
        if mode == "table":
            return str(row.get("table_path") or "Not selected")
        return "Not used"

    @staticmethod
    def _scan_values(row: dict[str, Any]) -> tuple[str, str, str]:
        mode = str(row.get("mode") or "none")
        if mode == "none":
            return (
                "Not applicable",
                "-",
                "No overflow measurements are configured; no source scan is required.",
            )
        scan = dict(row.get("scan") or {})
        status = str(scan.get("status") or "Not scanned").replace("_", " ")
        if mode == "scale_images":
            count = str(scan.get("image_count") or 0) if scan else "-"
        elif mode == "table":
            count = str(scan.get("row_count") or 0) if scan else "-"
        else:
            count = "-"
        notes = [str(value) for value in scan.get("errors") or []]
        notes.extend(str(value) for value in scan.get("warnings") or [])
        return status.title(), count, " | ".join(notes)

    @staticmethod
    def _reader_progress_values(
        progress: dict[str, Any],
    ) -> tuple[str, str, str]:
        if not progress.get("available"):
            return "-", "-", "No scale-image review has been started."
        if progress.get("error"):
            return "Unavailable", "-", str(progress["error"])

        reviewed = int(progress.get("reviewed_count") or 0)
        selected = int(progress.get("selected_count") or 0)
        review_text = f"{reviewed} / {selected}" if selected else str(reviewed)
        first_h = progress.get("first_time_h")
        last_h = progress.get("last_time_h")
        if first_h is None or last_h is None:
            span_text = "-"
        elif abs(float(last_h) - float(first_h)) < 1e-9:
            span_text = f"t = {float(first_h):.2f} h"
        else:
            span_text = f"{float(first_h):.2f} to {float(last_h):.2f} h"

        details = [
            f"Manually reviewed images: {reviewed}",
            f"Selected for review: {selected}",
            f"Exact readings: {int(progress.get('exact_count') or 0)}",
            f"Uncertain readings: {int(progress.get('partial_count') or 0)}",
            f"Unreadable images: {int(progress.get('unreadable_count') or 0)}",
        ]
        first_image_time = str(progress.get("first_image_time") or "")
        last_image_time = str(progress.get("last_image_time") or "")
        if first_image_time and last_image_time:
            details.append(
                f"Reviewed camera times: {first_image_time} to {last_image_time}"
            )
        reviewers = [str(value) for value in progress.get("reviewers") or []]
        if reviewers:
            details.append(f"Reviewed by: {', '.join(reviewers)}")
        if progress.get("updated_at"):
            details.append(f"Last saved: {progress['updated_at']}")
        return review_text, span_text, "\n".join(details)

    def _populate_table(self, selected_run_id: str = "") -> None:
        selected_run_id = selected_run_id or self._selected_run_id()
        self._updating = True
        try:
            rows = self._rows()
            self.run_table.setRowCount(len(rows))
            selected_index = 0 if rows else -1
            for row_index, row in enumerate(rows):
                run_id = str(row.get("run_id") or "")
                scan_status, item_count, notes = self._scan_values(row)
                reader_progress = scale_reader_review_progress(
                    self.project.root, run_id
                )
                reviewed, reviewed_span, reader_tooltip = self._reader_progress_values(
                    reader_progress
                )
                values = (
                    run_id.upper(),
                    MODE_LABELS.get(str(row.get("mode") or "none"), "Unknown"),
                    self._source_location(row),
                    scan_status,
                    item_count,
                    reviewed,
                    reviewed_span,
                    notes,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.ItemDataRole.UserRole, run_id)
                    if column in {5, 6}:
                        item.setToolTip(reader_tooltip)
                    elif column in {2, 7}:
                        item.setToolTip(value)
                    self.run_table.setItem(row_index, column, item)
                if run_id == selected_run_id:
                    selected_index = row_index
            if selected_index >= 0:
                self.run_table.selectRow(selected_index)
        finally:
            self._updating = False
        self._populate_global_table()
        self._load_selected_editor()

    def _selection_changed(self) -> None:
        if self._updating:
            return
        self._commit_editor(self._loaded_run_id)
        self._load_selected_editor()

    def _load_selected_editor(self) -> None:
        run_id = self._selected_run_id()
        row = self._row(run_id)
        self._loaded_run_id = run_id
        self._updating = True
        try:
            self.image_folders.clear()
            if row is None:
                self.scan_report.setPlainText("No experiment is selected.")
                self.editor.setTitle("Selected experiment")
                return
            self.editor.setTitle(f"Selected experiment: {run_id.upper()}")
            mode = str(row.get("mode") or "none")
            index = self.mode.findData(mode)
            self.mode.setCurrentIndex(max(0, index))
            for value in row.get("image_folders") or []:
                self.image_folders.addItem(str(value))
            self.table_path.setText(str(row.get("table_path") or ""))
            self._set_source_panel(mode)
            self._show_scan(row)
        finally:
            self._updating = False
        self._refresh_action_state()
        self._show_selected_scan_progress(row)

    def _commit_editor(self, run_id: str = "") -> None:
        if self._updating or not run_id:
            return
        row = self._row(run_id)
        if row is None:
            return
        before = overflow_run_source_digest(row)
        row["mode"] = str(self.mode.currentData() or "none")
        row["image_folders"] = [
            self.image_folders.item(index).text()
            for index in range(self.image_folders.count())
        ]
        row["table_path"] = self.table_path.text()
        after = overflow_run_source_digest(row)
        if before != after:
            row["source_origin"] = "individual_override"
            row["scan"] = {}
            self._dirty = True
            self._draft_timer.start()

    @Slot()
    def _persist_working_draft(self) -> None:
        if not self._dirty:
            return
        try:
            self.store.save_draft(
                self.config,
                actor=self.reviewer.text().strip(),
                reason="Overflow source selection or scan updated.",
            )
        except OSError as exc:
            self.scan_message.setText(
                f"Could not save overflow working state: {type(exc).__name__}: {exc}"
            )

    def _set_source_panel(self, mode: str) -> None:
        self.source_stack.setCurrentIndex(
            {"none": 0, "scale_images": 1, "table": 2}.get(mode, 0)
        )

    def _mode_changed(self) -> None:
        if self._updating:
            return
        mode = str(self.mode.currentData() or "none")
        self._set_source_panel(mode)
        self._source_changed()

    def _source_changed(self, _value: str = "") -> None:
        if self._updating:
            return
        run_id = self._loaded_run_id
        self._commit_editor(run_id)
        self._populate_table(run_id)

    def _add_image_folder(self) -> None:
        initial = (
            self.image_folders.item(self.image_folders.count() - 1).text()
            if self.image_folders.count()
            else str(Path.home())
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose scale-image folder",
            initial,
            options=DIRECTORY_DIALOG_OPTIONS,
        )
        if not selected:
            return
        existing = {
            self.image_folders.item(index).text()
            for index in range(self.image_folders.count())
        }
        if selected not in existing:
            self.image_folders.addItem(selected)
            self._source_changed()

    def _remove_image_folder(self) -> None:
        for item in self.image_folders.selectedItems():
            self.image_folders.takeItem(self.image_folders.row(item))
        self._source_changed()

    def _restore_discovered_source(self) -> None:
        row = self._row(self._loaded_run_id)
        if row is None:
            return
        images = [str(value) for value in row.get("discovered_image_folders") or []]
        tables = [str(value) for value in row.get("discovered_tables") or []]
        self._updating = True
        try:
            mode = "scale_images" if images else "table" if tables else "none"
            self.mode.setCurrentIndex(self.mode.findData(mode))
            self.image_folders.clear()
            self.image_folders.addItems(images)
            self.table_path.setText(tables[0] if tables else "")
            self._set_source_panel(mode)
        finally:
            self._updating = False
        self._source_changed()

    def _reviewed_workbook(self, run_id: str) -> Path:
        return (
            self.project.root
            / "results"
            / "overflow"
            / "scale_review"
            / run_id
            / f"{run_id}_overflow_scale_review.xlsx"
        )

    def _use_reviewed_workbook(self) -> None:
        run_id = self._loaded_run_id
        workbook = self._reviewed_workbook(run_id)
        if not workbook.is_file():
            QMessageBox.information(
                self,
                "No reviewed workbook",
                "Complete and save image review first. The expected workbook is:\n"
                + str(workbook),
            )
            return
        self._updating = True
        try:
            self.mode.setCurrentIndex(self.mode.findData("table"))
            self.table_path.setText(str(workbook))
            self._set_source_panel("table")
        finally:
            self._updating = False
        self._source_changed()

    def _show_scan(self, row: dict[str, Any]) -> None:
        if str(row.get("mode") or "none") == "none":
            self.scan_report.setPlainText(
                "No overflow measurements are configured for this experiment. "
                "This is a valid, explicit decision and no source scan is required."
            )
            return
        scan = dict(row.get("scan") or {})
        if not scan:
            self.scan_report.setPlainText(
                "This source has not been scanned, or it changed after its last scan."
            )
            return
        lines = [
            f"Status: {str(scan.get('status') or '').replace('_', ' ')}",
            f"Scanned: {scan.get('scanned_utc') or 'unknown'}",
        ]
        if scan.get("kind") == "scale_images":
            lines.extend(
                [
                    f"Images: {scan.get('image_count', 0)}",
                    (
                        f"Images with filename timestamps: "
                        f"{scan.get('timestamped_image_count', 0)}"
                    ),
                    (
                        f"Time range: {scan.get('first_timestamp') or 'unknown'} to "
                        f"{scan.get('last_timestamp') or 'unknown'}"
                    ),
                ]
            )
        elif scan.get("kind") == "table":
            lines.extend(
                [
                    f"Data rows: {scan.get('row_count', 0)}",
                    f"Sheets: {', '.join(scan.get('sheets') or []) or 'none'}",
                    f"Columns: {', '.join(scan.get('columns') or []) or 'none'}",
                ]
            )
        issues = [str(value) for value in scan.get("errors") or []]
        issues.extend(str(value) for value in scan.get("warnings") or [])
        if issues:
            lines.append("\nFindings:\n- " + "\n- ".join(issues))
        self.scan_report.setPlainText("\n".join(lines))

    def _show_selected_scan_progress(self, row: dict[str, Any] | None) -> None:
        """Represent the selected run when no scan operation is active."""

        if self._active_scan_run_ids:
            return
        self.scan_progress.setRange(0, 1)
        if row is None:
            self.scan_progress.setValue(0)
            self.scan_message.setText("No experiment is selected.")
            return
        run_id = str(row.get("run_id") or "").upper()
        mode = str(row.get("mode") or "none")
        scan = dict(row.get("scan") or {})
        if mode == "none":
            self.scan_progress.setValue(1)
            save_hint = (
                " Save the overflow revision to record this decision."
                if self._dirty
                else ""
            )
            self.scan_message.setText(
                f"{run_id}: no overflow measurements are configured; "
                f"source scan is not required.{save_hint}"
            )
            return
        if not scan:
            self.scan_progress.setValue(0)
            self.scan_message.setText(
                f"{run_id}: selected source has not been scanned."
            )
            return
        self.scan_progress.setValue(1)
        status = str(scan.get("status") or "unknown").replace("_", " ")
        item_key = "image_count" if mode == "scale_images" else "row_count"
        item_label = "images" if mode == "scale_images" else "table rows"
        count = int(scan.get(item_key) or 0)
        self.scan_message.setText(
            f"{run_id}: scan finished with status {status}; {count} {item_label}."
        )

    def _scan_targets(self, scan_all: bool) -> list[dict[str, Any]]:
        if scan_all:
            return [
                deepcopy(row)
                for row in self._rows()
                if str(row.get("mode") or "none") != "none"
            ]
        if self._scope == "global":
            selected = set(self._global_selected_run_ids())
            return [
                deepcopy(row)
                for row in self._rows()
                if str(row.get("run_id") or "") in selected
                and str(row.get("mode") or "none") != "none"
            ]
        row = self._row(self._loaded_run_id)
        return (
            [deepcopy(row)]
            if row is not None and str(row.get("mode") or "none") != "none"
            else []
        )

    def _start_scan(self, scan_all: bool) -> None:
        self._commit_editor(self._loaded_run_id)
        if not dependencies_satisfied("overflow", self.project.module_states):
            QMessageBox.warning(
                self,
                "Data review required",
                "Approve Data and integrity before scanning overflow sources.",
            )
            return
        if self._thread is not None and self._thread.isRunning():
            return
        rows = self._scan_targets(scan_all)
        if not rows:
            QMessageBox.information(
                self,
                "No source scan required",
                "No selected experiment has an active overflow source. Experiments "
                "configured with 'No overflow measurements' are already resolved and "
                "do not require scanning.",
            )
            return
        try:
            self.project.set_module_status(
                "overflow",
                "in_progress",
                updated_by=self.reviewer.text().strip(),
                reason="Scanning selected overflow sources.",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not start overflow scan", str(exc))
            return
        self._active_scan_run_ids = tuple(
            str(row.get("run_id") or "").strip().casefold() for row in rows
        )
        self._set_scanning(True)
        self.scan_progress.setRange(0, max(1, len(rows)))
        self.scan_progress.setValue(0)
        self.scan_message.setText("Preparing source scan...")
        self._thread = QThread(self)
        self._worker = OverflowScanWorker(rows)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._scan_progressed)
        self._worker.finished.connect(self._scan_finished)
        self._worker.failed.connect(self._scan_failed)
        self._worker.cancelled.connect(self._scan_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        self.status_changed.emit()

    def _cancel_scan(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_scan.setEnabled(False)
            self.scan_message.setText("Cancelling after the current experiment...")

    @Slot(int, int, str)
    def _scan_progressed(self, current: int, total: int, label: str) -> None:
        self.scan_progress.setRange(0, max(1, total))
        self.scan_progress.setValue(current)
        if label == "Source scan complete":
            self.scan_message.setText("Finishing source scan...")
            return
        position = min(max(1, current + 1), max(1, total))
        self.scan_message.setText(
            f"Scanning {label} ({position} of {max(1, total)})..."
        )

    @Slot(object)
    def _scan_finished(self, raw_results: object) -> None:
        selected = self._loaded_run_id
        for result in list(raw_results):
            row = self._row(str(result.get("run_id") or ""))
            if row is not None:
                row["scan"] = deepcopy(dict(result.get("scan") or {}))
        self._dirty = True
        self._persist_working_draft()
        self.scan_progress.setValue(self.scan_progress.maximum())
        self.scan_message.setText("Scan complete. Save an overflow revision.")
        self._finish_scan()
        self._populate_table(selected)
        self.refresh()

    @Slot(str)
    def _scan_failed(self, message: str) -> None:
        self.scan_message.setText("Source scan failed; no partial revision was saved.")
        self._finish_scan()
        QMessageBox.critical(self, "Overflow scan failed", message)

    @Slot()
    def _scan_cancelled(self) -> None:
        self.scan_message.setText("Source scan cancelled; no partial scan was applied.")
        self._finish_scan()

    def _finish_scan(self) -> None:
        self._worker = None
        self._thread = None
        self._active_scan_run_ids = ()
        self._set_scanning(False)

    def _set_scanning(self, active: bool) -> None:
        self.cancel_scan.setEnabled(active)
        self.run_table.setEnabled(not active)
        self.editor.setEnabled(not active)
        self.global_group.setEnabled(not active)
        self._refresh_action_state(active_scan=active)

    def _open_scale_reader(self) -> None:
        self._commit_editor(self._loaded_run_id)
        row = self._row(self._loaded_run_id)
        if row is None or str(row.get("mode") or "") != "scale_images":
            QMessageBox.information(
                self,
                "Scale images not selected",
                "Choose Read scale-display images for this experiment first.",
            )
            return
        if not row.get("image_folders"):
            QMessageBox.warning(
                self, "No image folder", "Add at least one scale-image folder first."
            )
            return
        self.scan_message.setText(
            f"Opening integrated scale-image reader for {self._loaded_run_id.upper()}..."
        )
        if self._reader_dialog is not None and self._reader_dialog.isVisible():
            self._reader_dialog.raise_()
            self._reader_dialog.activateWindow()
            return
        dialog = ScaleImageReaderDialog(
            self.project,
            self.config,
            self._loaded_run_id,
            self.window(),
        )
        dialog.finished.connect(self._reader_closed)
        self._reader_dialog = dialog
        dialog.showMaximized()

    @Slot()
    def _reader_closed(self) -> None:
        self._reader_dialog = None
        self._populate_table(self._loaded_run_id)
        self._show_selected_scan_progress(self._row(self._loaded_run_id))

    def _open_reader_output(self) -> None:
        output = self.project.root / "results" / "overflow" / "scale_review"
        output.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))

    def _save_revision(self) -> None:
        self._commit_editor(self._loaded_run_id)
        if self._draft_timer.isActive():
            self._draft_timer.stop()
        self._persist_working_draft()
        actor = self.reviewer.text().strip()
        if not actor:
            QMessageBox.warning(
                self, "Reviewer required", "Enter the name of the reviewer first."
            )
            return
        expected = [
            str(row.get("run_id") or "")
            for row in self.project.manifest.get("runs") or []
        ]
        report = validate_overflow_config(self.config, expected_run_ids=expected)
        errors = [
            str(issue.get("message") or "")
            for issue in report.get("issues") or []
            if str(issue.get("severity") or "") == "error"
        ]
        if errors:
            QMessageBox.warning(
                self,
                "Overflow sources need attention",
                "Resolve these items before saving:\n\n- " + "\n- ".join(errors),
            )
            return
        try:
            revision = self.store.save_revision(
                self.config,
                actor=actor,
                reason=self.revision_note.text().strip()
                or "Overflow sources scanned and reviewed.",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save overflow revision", str(exc))
            return
        self.config = normalize_overflow_config(
            self.project, dict(revision.get("configuration") or {})
        )
        self._dirty = False
        self.revision_note.clear()
        self.scan_message.setText("Overflow revision saved.")
        self._populate_table(self._loaded_run_id)
        self.refresh()
        self.status_changed.emit()

    def _refresh_action_state(self, *, active_scan: bool | None = None) -> None:
        if active_scan is None:
            active_scan = self._thread is not None and self._thread.isRunning()
        available = dependencies_satisfied("overflow", self.project.module_states)
        has_rows = bool(self._rows())
        selected = self._row(self._loaded_run_id)
        global_targets = bool(self._global_selected_run_ids())
        global_selected = set(self._global_selected_run_ids())
        global_scan_targets = any(
            str(row.get("run_id") or "") in global_selected
            and str(row.get("mode") or "none") != "none"
            for row in self._rows()
        )
        selected_has_source = bool(
            selected and str(selected.get("mode") or "none") != "none"
        )
        active_rows_exist = any(
            str(row.get("mode") or "none") != "none" for row in self._rows()
        )
        image_mode = bool(
            selected and str(selected.get("mode") or "") == "scale_images"
        )
        has_scan_target = (
            global_scan_targets if self._scope == "global" else selected_has_source
        )
        self.scan_selected.setEnabled(available and has_scan_target and not active_scan)
        self.scan_all.setEnabled(available and active_rows_exist and not active_scan)
        self.save_revision.setEnabled(available and not active_scan)
        self.open_reader.setEnabled(available and image_mode and not active_scan)
        self.open_output.setEnabled(not active_scan)
        self.use_reviewed_table.setEnabled(
            available and bool(selected) and not active_scan
        )
        self.apply_global_source.setEnabled(
            available and global_targets and not active_scan
        )

    def refresh(self) -> None:
        super().refresh()
        revisions = self.store.revisions()
        current = revisions[-1] if revisions else {}
        dirty_text = " Unsaved source changes are present." if self._dirty else ""
        if current:
            self.revision_status.setText(
                f"Saved revisions: {len(revisions)}. Current revision: "
                f"{str(current.get('revision_id') or '')[:12]}." + dirty_text
            )
        else:
            self.revision_status.setText(
                "No overflow revision has been saved." + dirty_text
            )
        if self.store.has_draft():
            self.revision_status.setText(
                self.revision_status.text()
                + " The incomplete working state is saved and will be restored after restart."
            )
        available = dependencies_satisfied("overflow", self.project.module_states)
        self.mode.setEnabled(available)
        self.restore_discovered.setEnabled(available)
        self.add_image_folder.setEnabled(available)
        self.remove_image_folder.setEnabled(available)
        self.table_path.setEnabled(available)
        self.global_mode.setEnabled(available)
        self.select_all_eligible.setEnabled(available)
        self.clear_global_selection.setEnabled(available)
        self._refresh_global_selection_summary()
        self._refresh_action_state()

    def _set_status(self, status: str) -> None:
        run_ids: list[str] = []
        if status in {"review_required", "approved", "complete"}:
            self._commit_editor(self._loaded_run_id)
            if self._dirty or not self.store.revisions():
                QMessageBox.warning(
                    self,
                    "Save overflow revision",
                    "Save the current overflow decision before changing its quality gate.",
                )
                return
            report = validate_overflow_config(
                self.config,
                expected_run_ids=[
                    str(row.get("run_id") or "")
                    for row in self.project.manifest.get("runs") or []
                ],
            )
            if int(report["summary"]["error_count"]):
                QMessageBox.warning(
                    self,
                    "Overflow validation failed",
                    "One or more active overflow sources must be scanned again.",
                )
                return
            run_ids = [
                str(row.get("run_id") or "").strip().casefold()
                for row in self.config.get("runs") or []
                if isinstance(row, dict) and str(row.get("run_id") or "").strip()
            ]
        if run_ids:
            try:
                self.project.set_run_module_status(
                    run_ids,
                    "overflow",
                    status,
                    updated_by=self.reviewer.text().strip(),
                    reason=(
                        "Covered by the reviewed overflow revision, including explicit "
                        "no-overflow decisions."
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - visible GUI boundary
                QMessageBox.critical(
                    self, "Could not update experiment reviews", str(exc)
                )
                return
        super()._set_status(status)


__all__ = ["OverflowPage", "OverflowScanWorker"]
