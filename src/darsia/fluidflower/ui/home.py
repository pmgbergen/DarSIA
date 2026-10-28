"""Home page for project and unfinished-setup selection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.project import ProjectRegistry, slugify
from darsia.fluidflower.setup_drafts import SetupDraftStore
class HomePage(QWidget):
    new_requested = Signal()
    resume_setup_requested = Signal(str)
    open_requested = Signal()
    recent_requested = Signal(str)
    add_runs_requested = Signal(str)
    update_run_sources_requested = Signal(str)

    def __init__(
        self,
        registry: ProjectRegistry,
        *,
        draft_store: SetupDraftStore | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.draft_store = draft_store or SetupDraftStore()
        home_layout = QVBoxLayout(self)
        home_layout.setContentsMargins(40, 30, 40, 30)
        home_layout.setSpacing(16)
        title = QLabel("FluidFlower Experiments")
        title.setObjectName("homeTitle")
        home_layout.addWidget(title)
        self.intro = QLabel(
            "Set up, open, and continue a FluidFlower experimental series."
        )
        self.intro.setObjectName("secondaryText")
        home_layout.addWidget(self.intro)
        projects_tab = QWidget()
        outer = QVBoxLayout(projects_tab)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)
        home_layout.addWidget(projects_tab, 1)
        actions = QHBoxLayout()
        self.new_button = QPushButton("Set up new experimental series")
        self.new_button.setIcon(
            QApplication.style().standardIcon(
                QStyle.StandardPixmap.SP_FileDialogNewFolder
            )
        )
        self.new_button.clicked.connect(self.new_requested)
        actions.addWidget(self.new_button)
        self.open_button = QPushButton("Import or open project package")
        self.open_button.setIcon(
            QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        self.open_button.clicked.connect(self.open_requested)
        actions.addWidget(self.open_button)
        actions.addStretch(1)
        outer.addLayout(actions)

        self.draft_heading = QLabel("Incomplete setup")
        self.draft_heading.setObjectName("sectionTitle")
        outer.addWidget(self.draft_heading)
        self.draft_note = QLabel()
        self.draft_note.setWordWrap(True)
        self.draft_note.setObjectName("secondaryText")
        outer.addWidget(self.draft_note)
        self.draft_table = QTableWidget(0, 4)
        self.draft_table.setHorizontalHeaderLabels(
            ["Project", "Last saved", "Continue from", "Planned location"]
        )
        self.draft_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.draft_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.draft_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.draft_table.verticalHeader().setVisible(False)
        draft_header = self.draft_table.horizontalHeader()
        draft_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        draft_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        draft_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        draft_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.draft_table.setMaximumHeight(180)
        self.draft_table.cellDoubleClicked.connect(self._resume_draft_row)
        self.draft_table.itemSelectionChanged.connect(self._draft_selection_changed)
        outer.addWidget(self.draft_table)
        draft_actions = QHBoxLayout()
        self.resume_setup_button = QPushButton("Continue selected setup")
        self.resume_setup_button.clicked.connect(self._resume_latest_setup)
        draft_actions.addWidget(self.resume_setup_button)
        self.discard_setup_button = QPushButton("Discard selected setup")
        self.discard_setup_button.clicked.connect(self._discard_selected_setup)
        draft_actions.addWidget(self.discard_setup_button)
        draft_actions.addStretch(1)
        self.draft_actions_widget = QWidget()
        self.draft_actions_widget.setLayout(draft_actions)
        outer.addWidget(self.draft_actions_widget)

        heading = QLabel("Recent experimental series")
        heading.setObjectName("sectionTitle")
        outer.addWidget(heading)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Project", "Series", "Runs", "Template", "Last opened", "Location"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self._open_row)
        self.table.itemSelectionChanged.connect(self._project_selection_changed)
        outer.addWidget(self.table, 1)
        project_actions = QHBoxLayout()
        self.open_selected_button = QPushButton("Open selected series")
        self.open_selected_button.clicked.connect(self._open_selected_project)
        project_actions.addWidget(self.open_selected_button)
        self.add_runs_button = QPushButton("Add experiments to selected series")
        self.add_runs_button.clicked.connect(self._add_runs_to_selected_project)
        project_actions.addWidget(self.add_runs_button)
        self.update_run_sources_button = QPushButton("Open selected series setup")
        self.update_run_sources_button.setToolTip(
            "Open the complete Series setup workspace. Review global settings and "
            "any experiment, correct folder roles, and save new source revisions "
            "without changing the raw files."
        )
        self.update_run_sources_button.clicked.connect(
            self._update_selected_project_sources
        )
        project_actions.addWidget(self.update_run_sources_button)
        project_actions.addStretch(1)
        outer.addLayout(project_actions)
        self.package_note = QLabel(
            "To import a shared package, select its top-level folder; DarSIA finds "
            "the enclosed project and adds it to this computer. Raw images and "
            "imported measurements remain external and read-only."
        )
        self.package_note.setWordWrap(True)
        self.package_note.setObjectName("secondaryText")
        outer.addWidget(self.package_note)
        self.refresh()

    def refresh(self) -> None:
        drafts = self.draft_store.drafts()
        has_drafts = bool(drafts)
        self.draft_note.setText(
            "Select an autosaved setup to continue from its last page, or discard "
            "it. Discarding only archives the setup draft; source files are untouched."
            if has_drafts
            else "No unfinished setup is stored."
        )
        self.draft_table.setVisible(has_drafts)
        self.draft_actions_widget.setVisible(has_drafts)
        self.draft_table.setRowCount(len(drafts))
        step_labels = {
            0: "Series identity",
            1: "Apparatus, protocol and sources",
            2: "Review and create",
        }
        for row_index, draft in enumerate(drafts):
            values = dict(draft.get("values") or {})
            draft_id = str(draft.get("draft_id") or "")
            project_label = str(
                values.get("name") or values.get("series_id") or "Untitled setup"
            )
            parent = str(values.get("parent_folder") or "")
            series_id = slugify(str(values.get("series_id") or project_label))
            location = str(Path(parent) / series_id) if parent else "Not selected"
            current_page = int(values.get("current_page") or 0)
            layout_version = int(values.get("wizard_layout_version") or 1)
            if layout_version < 2:
                current_page = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}.get(current_page, 0)
            if layout_version < 4:
                current_page = {0: 0, 1: 1, 2: 1, 3: 2}.get(current_page, 0)
            display_values = (
                project_label,
                draft.get("updated_utc"),
                step_labels.get(current_page, "Setup"),
                location,
            )
            for column, value in enumerate(display_values):
                item = QTableWidgetItem(str(value or ""))
                item.setData(Qt.ItemDataRole.UserRole, draft_id)
                self.draft_table.setItem(row_index, column, item)
        if has_drafts:
            self.draft_table.selectRow(0)
        self._draft_selection_changed()
        self.registry.forget_missing()
        rows = self.registry.entries()
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.get("name"),
                row.get("series_id"),
                row.get("run_count", ""),
                row.get("template_id"),
                row.get("last_opened_utc"),
                row.get("path"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value or ""))
                item.setData(Qt.ItemDataRole.UserRole, str(row.get("path") or ""))
                self.table.setItem(row_index, column, item)
        self._project_selection_changed()

    def _resume_latest_setup(self) -> None:
        draft_id = self._selected_draft_id()
        if draft_id:
            self.resume_setup_requested.emit(draft_id)

    def _resume_draft_row(self, row: int, _column: int) -> None:
        self.draft_table.selectRow(row)
        self._resume_latest_setup()

    def _selected_draft_id(self) -> str:
        row = self.draft_table.currentRow()
        item = self.draft_table.item(row, 0) if row >= 0 else None
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""

    def _draft_selection_changed(self) -> None:
        selected = bool(self._selected_draft_id())
        self.resume_setup_button.setEnabled(selected)
        self.discard_setup_button.setEnabled(selected)

    def _discard_selected_setup(self) -> None:
        draft_id = self._selected_draft_id()
        if not draft_id:
            return
        draft = self.draft_store.load(draft_id)
        values = dict(draft.get("values") or {})
        label = str(values.get("name") or values.get("series_id") or "this setup")
        choice = QMessageBox.question(
            self,
            "Discard incomplete setup",
            f"Discard the incomplete setup for {label}?\n\n"
            "Only the autosaved setup draft is archived. Source files and existing "
            "projects are not changed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        self.draft_store.discard(draft_id)
        self.refresh()

    def _open_row(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.recent_requested.emit(str(item.data(Qt.ItemDataRole.UserRole) or ""))

    def _selected_project_path(self) -> str:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""

    def _project_selection_changed(self) -> None:
        selected = bool(self._selected_project_path())
        self.open_selected_button.setEnabled(selected)
        self.add_runs_button.setEnabled(selected)
        self.update_run_sources_button.setEnabled(selected)

    def _open_selected_project(self) -> None:
        path = self._selected_project_path()
        if path:
            self.recent_requested.emit(path)

    def _add_runs_to_selected_project(self) -> None:
        path = self._selected_project_path()
        if path:
            self.add_runs_requested.emit(path)

    def _update_selected_project_sources(self) -> None:
        path = self._selected_project_path()
        if path:
            self.update_run_sources_requested.emit(path)
