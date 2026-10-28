"""Qt application shell and project navigation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.modules import MODULE_BY_KEY, MODULE_DEFINITIONS
from darsia.fluidflower.page_registry import (
    FluidFlowerPageRegistry,
    default_page_registry,
)
from darsia.fluidflower.pages.setup.wizard import NewSeriesWizard
from darsia.fluidflower.pages.setup.incremental_runs import (
    AddRunsDialog,
)
from darsia.fluidflower.project import (
    FluidFlowerProject,
    ProjectRegistry,
    default_project_root,
    resolve_project_package_root,
)
from darsia.fluidflower.setup_drafts import SetupDraftStore
from darsia.fluidflower.ui.common import (
    DIRECTORY_DIALOG_OPTIONS,
    STATUS_COLORS,
    STATUS_LABELS,
)
from darsia.fluidflower.ui.home import HomePage
from darsia.fluidflower.ui.theme import application_icon, apply_application_theme

WORKSPACE_CONTEXT_HELP: dict[str, tuple[str, str]] = {
    "overview": (
        "Workflow overview",
        "Shows the state, dependencies, and latest change for every analysis module. "
        "Use it to see what is ready, blocked, stale, or awaiting review. No source "
        "data or analysis results are changed here.",
    ),
    "execution": (
        "Production",
        "Review a saved model and mass-analysis plan before calculation. Detect local "
        "hardware and use the existing local or shared task queue. One mass-analysis "
        "task runs on one host; dividing its runs among several hosts requires separate "
        "immutable tasks. Temporary cache stays local and final results use the "
        "configured series result destination.",
    ),
}


SUBSECTION_CONTEXT_HELP: dict[tuple[str, str], str] = {
    ("geometry_facies", "Inputs"): (
        "Register the baseline, labels, depth, facies, and optional protocol references "
        "that define the spatial analysis grid."
    ),
    ("geometry_facies", "Layers"): (
        "Review individual physical layers, group equivalent layers into shared facies, "
        "and assign analysis roles, porosity, and permeability."
    ),
    ("geometry_facies", "Injection points"): (
        "Place and verify injection points on the corrected rig image. Protocol events "
        "refer to their stable ip1, ip2, and subsequent identities."
    ),
    ("geometry_facies", "Run geometry"): (
        "Choose whether all runs share one approved geometry or require run-specific "
        "review when layer compression or displacement is detected."
    ),
    ("geometry_facies", "Boundary edits"): (
        "Correct proposed layer boundaries on the native spatial grid. Edits are saved "
        "as revisions and never overwrite imported labels."
    ),
    ("geometry_facies", "Training data"): (
        "Export reviewed facies and material annotations with leakage-safe train, "
        "validation, and holdout grouping."
    ),
    ("geometry_facies", "QC and history"): (
        "Inspect validation findings, compare revisions, and approve the exact geometry "
        "artifact that downstream modules may use."
    ),
    ("phase_interpretation", "Active learning review"): (
        "Answer model-selected regions where phase classification is uncertain, and add "
        "targeted boxes or freehand annotations where the proposals miss useful evidence."
    ),
    ("phase_interpretation", "Model validation"): (
        "Review a complete phase interpretation through time, including false positives, "
        "missed CO2, implausible phase behavior, and indicator carryover."
    ),
}


class ProjectWorkspace(QWidget):
    home_requested = Signal()
    project_changed = Signal(object)

    def __init__(
        self,
        project: FluidFlowerProject,
        *,
        page_registry: FluidFlowerPageRegistry | None = None,
    ) -> None:
        super().__init__()
        self.project = project
        self.page_registry = page_registry or default_page_registry()
        self.pages: dict[str, QWidget] = {}
        self.page_hosts: dict[str, QWidget] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.setContentsMargins(14, 10, 14, 10)
        back = QPushButton("Projects")
        back.setIcon(
            QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack)
        )
        back.clicked.connect(self.home_requested)
        header.addWidget(back)
        title = QLabel(project.name)
        title.setObjectName("workspaceTitle")
        header.addWidget(title)
        self.current_section_label = QLabel("Overview")
        self.current_section_label.setObjectName("secondaryText")
        header.addWidget(self.current_section_label)
        self.context_help_button = QPushButton()
        self.context_help_button.setObjectName("contextHelpButton")
        self.context_help_button.setIcon(
            QApplication.style().standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxInformation
            )
        )
        self.context_help_button.setFixedSize(34, 34)
        self.context_help_button.setAccessibleName("Information about current section")
        self.context_help_button.clicked.connect(self._show_context_help)
        header.addWidget(self.context_help_button)
        header.addStretch(1)
        location = QLabel(str(project.root))
        location.setObjectName("pathPreview")
        location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(location)
        folder = QPushButton("Open folder")
        folder.setIcon(
            QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(project.root)))
        )
        header.addWidget(folder)
        self.add_runs_button = QPushButton("Add experiments")
        self.add_runs_button.setToolTip(
            "Scan the series source folder for experiments that are not yet "
            "registered in this project."
        )
        self.add_runs_button.clicked.connect(self._add_experiments)
        header.addWidget(self.add_runs_button)
        outer.addLayout(header)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        outer.addWidget(separator)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        self.navigation = QListWidget()
        self.navigation.setObjectName("moduleNavigation")
        self.navigation.setFixedWidth(340)
        overview_item = QListWidgetItem("Overview")
        overview_item.setData(Qt.ItemDataRole.UserRole, "overview")
        self.navigation.addItem(overview_item)
        for definition in MODULE_DEFINITIONS:
            item = QListWidgetItem(definition.title)
            item.setData(Qt.ItemDataRole.UserRole, definition.key)
            item.setToolTip(definition.description)
            self.navigation.addItem(item)
            if definition.key == "mass_analysis":
                execution_item = QListWidgetItem("Production")
                execution_item.setData(Qt.ItemDataRole.UserRole, "execution")
                execution_item.setToolTip(
                    "Preflight, saved plans, local or shared task queue, and workers."
                )
                self.navigation.addItem(execution_item)
        self.navigation.currentItemChanged.connect(self._navigation_changed)
        body.addWidget(self.navigation)
        self.stack = QStackedWidget()
        overview = self.page_registry.create("overview", project)
        if not isinstance(overview, QWidget):
            raise TypeError("The registered overview page must be a QWidget")
        overview.module_requested.connect(self.select_module)
        self.pages["overview"] = overview
        self.page_hosts["overview"] = overview
        self.stack.addWidget(overview)
        for definition in MODULE_DEFINITIONS:
            page = self.page_registry.create(definition.key, project)
            if not isinstance(page, QWidget):
                raise TypeError(
                    f"The registered {definition.key!r} page must be a QWidget"
                )
            status_changed = getattr(page, "status_changed", None)
            if status_changed is not None and hasattr(status_changed, "connect"):
                status_changed.connect(self.refresh)
            page_project_changed = getattr(page, "project_changed", None)
            if page_project_changed is not None and hasattr(
                page_project_changed, "connect"
            ):
                page_project_changed.connect(self._page_project_changed)
            module_requested = getattr(page, "module_requested", None)
            if module_requested is not None and hasattr(module_requested, "connect"):
                module_requested.connect(self.select_module)
            review_setup_requested = getattr(page, "review_setup_requested", None)
            if review_setup_requested is not None and hasattr(
                review_setup_requested, "connect"
            ):
                review_setup_requested.connect(self._review_setup_for_run)
            self.pages[definition.key] = page
            plugin = self.page_registry.plugin(definition.key)
            host: QWidget = page
            if plugin.scrollable:
                scroll_host = QScrollArea()
                scroll_host.setFrameShape(QFrame.Shape.NoFrame)
                scroll_host.setWidgetResizable(True)
                scroll_host.setWidget(page)
                host = scroll_host
            self.page_hosts[definition.key] = host
            self.stack.addWidget(host)
        execution = self.page_registry.create("execution", project)
        if not isinstance(execution, QWidget):
            raise TypeError("The registered execution page must be a QWidget")
        self.pages["execution"] = execution
        execution_plugin = self.page_registry.plugin("execution")
        execution_host: QWidget = execution
        if execution_plugin.scrollable:
            execution_scroll = QScrollArea()
            execution_scroll.setFrameShape(QFrame.Shape.NoFrame)
            execution_scroll.setWidgetResizable(True)
            execution_scroll.setWidget(execution)
            execution_host = execution_scroll
        self.page_hosts["execution"] = execution_host
        self.stack.addWidget(execution_host)
        for page in self.pages.values():
            for tabs in page.findChildren(QTabWidget):
                tabs.currentChanged.connect(self._context_tab_changed)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        self.navigation.setCurrentRow(0)
        self.refresh()

    def _add_experiments(self) -> None:
        dialog = AddRunsDialog(self.project, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh()
            self.project_changed.emit(self.project)

    def _review_setup_for_run(self, run_id: str = "") -> None:
        self.select_module("setup")
        setup_page = self.pages.get("setup")
        select_run = getattr(setup_page, "select_run", None)
        if callable(select_run) and run_id:
            select_run(run_id)

    def _page_project_changed(self) -> None:
        self.refresh()
        self.project_changed.emit(self.project)

    def select_module(self, module_key: str) -> None:
        for row in range(self.navigation.count()):
            item = self.navigation.item(row)
            if str(item.data(Qt.ItemDataRole.UserRole)) == module_key:
                self.navigation.setCurrentRow(row)
                return

    def refresh(self) -> None:
        for row in range(1, self.navigation.count()):
            item = self.navigation.item(row)
            key = str(item.data(Qt.ItemDataRole.UserRole))
            if key == "execution":
                item.setText("Production")
                item.setForeground(QColor("#73b9e6"))
                continue
            status = str(self.project.module_states[key].get("status") or "not_started")
            item.setText(
                f"{MODULE_BY_KEY[key].title}  ·  {STATUS_LABELS.get(status, status)}"
            )
            item.setForeground(QColor(STATUS_COLORS.get(status, "#d7dce0")))
        for page in self.pages.values():
            refresh = getattr(page, "refresh", None)
            if callable(refresh):
                refresh()

    def _navigation_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        key = str(current.data(Qt.ItemDataRole.UserRole))
        host = self.page_hosts.get(key)
        if host is not None:
            self.stack.setCurrentWidget(host)
        if key == "execution":
            self.pages["execution"].refresh()
        self._update_context_help(key)

    def _context_tab_changed(self, _index: int) -> None:
        current = self.navigation.currentItem()
        if current is not None:
            self._update_context_help(
                str(current.data(Qt.ItemDataRole.UserRole) or "overview")
            )

    def _active_subsection(self, module_key: str) -> str:
        page = self.pages.get(module_key)
        if page is None:
            return ""
        for tabs in page.findChildren(QTabWidget):
            if tabs.count() and tabs.isVisibleTo(page):
                return tabs.tabText(tabs.currentIndex())
        return ""

    def _context_help(self, module_key: str) -> tuple[str, str]:
        if module_key in MODULE_BY_KEY:
            definition = MODULE_BY_KEY[module_key]
            title = definition.title
            body = definition.help_text or definition.description
        else:
            title, body = WORKSPACE_CONTEXT_HELP.get(
                module_key,
                ("FluidFlower Experiments", "No additional guidance is available."),
            )
        subsection = self._active_subsection(module_key)
        subsection_help = SUBSECTION_CONTEXT_HELP.get((module_key, subsection), "")
        if subsection:
            title = f"{title} / {subsection}"
        if subsection_help:
            body = f"{body}\n\nCurrent subsection\n{subsection_help}"
        return title, body

    def _update_context_help(self, module_key: str) -> None:
        title, _body = self._context_help(module_key)
        self.current_section_label.setText(title)
        self.context_help_button.setToolTip(f"About {title}")

    def _show_context_help(self) -> None:
        current = self.navigation.currentItem()
        module_key = (
            str(current.data(Qt.ItemDataRole.UserRole) or "overview")
            if current is not None
            else "overview"
        )
        title, body = self._context_help(module_key)
        QMessageBox.information(self, title, body)


class FluidFlowerMainWindow(QMainWindow):
    """Main entry point for modular FluidFlower workflows."""

    def __init__(
        self,
        *,
        registry: ProjectRegistry | None = None,
        draft_store: SetupDraftStore | None = None,
        page_registry: FluidFlowerPageRegistry | None = None,
        initial_project: Path | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry or ProjectRegistry()
        self.draft_store = draft_store or SetupDraftStore()
        self.page_registry = page_registry or default_page_registry()
        self.wizard: NewSeriesWizard | None = None
        self.workspace: ProjectWorkspace | None = None
        self._workspaces: dict[str, ProjectWorkspace] = {}
        self.setWindowTitle("DarSIA · FluidFlower Experiments")
        self.resize(1500, 930)
        self.setMinimumSize(1050, 700)
        self.stack = QStackedWidget()
        self.home = HomePage(self.registry, draft_store=self.draft_store)
        self.home.new_requested.connect(self._new_project)
        self.home.resume_setup_requested.connect(self._resume_project_setup)
        self.home.open_requested.connect(self._open_project_dialog)
        self.home.recent_requested.connect(lambda path: self.open_project(Path(path)))
        self.home.add_runs_requested.connect(self._add_runs_to_project)
        self.home.update_run_sources_requested.connect(
            self._update_project_run_sources
        )
        self.stack.addWidget(self.home)
        self.setCentralWidget(self.stack)
        self._apply_style()
        if initial_project is not None:
            self.open_project(initial_project)

    def _apply_style(self) -> None:
        apply_application_theme(self)

    def _new_project(self) -> None:
        self.wizard = NewSeriesWizard(self, draft_store=self.draft_store)
        self.wizard.project_created.connect(self._project_created)
        self.wizard.finished.connect(lambda _result: self.home.refresh())
        self.wizard.open()

    def _resume_project_setup(self, draft_id: str) -> None:
        try:
            self.wizard = NewSeriesWizard(
                self,
                draft_id=draft_id,
                draft_store=self.draft_store,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not resume setup", str(exc))
            return
        self.wizard.project_created.connect(self._project_created)
        self.wizard.finished.connect(lambda _result: self.home.refresh())
        self.wizard.open()

    def _project_created(self, project: FluidFlowerProject) -> None:
        self.registry.remember(project)
        self.show_project(project)

    def _open_project_dialog(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Import or open a FluidFlower project package",
            str(default_project_root()),
            options=DIRECTORY_DIALOG_OPTIONS,
        )
        if selected:
            self.open_project(Path(selected))

    def _load_selected_project(self, path: str) -> FluidFlowerProject | None:
        key = self._workspace_key(Path(path))
        existing = self._workspaces.get(key)
        if existing is not None:
            return existing.project
        try:
            return FluidFlowerProject.load(resolve_project_package_root(Path(path)))
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self,
                "Could not open project",
                f"{type(exc).__name__}: {exc}",
            )
            return None

    def _add_runs_to_project(self, path: str) -> None:
        project = self._load_selected_project(path)
        if project is None:
            return
        dialog = AddRunsDialog(project, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.registry.remember(project)
            self.home.refresh()

    def _update_project_run_sources(self, path: str) -> None:
        project = self._load_selected_project(path)
        if project is None:
            return
        self.show_project(project)
        if self.workspace is not None:
            self.workspace.select_module("setup")

    def open_project(self, root: Path) -> None:
        try:
            project = FluidFlowerProject.load(resolve_project_package_root(root))
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self,
                "Could not open project",
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.registry.remember(project)
        self.show_project(project)

    def show_project(self, project: FluidFlowerProject) -> None:
        key = self._workspace_key(project.root)
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = ProjectWorkspace(
                project,
                page_registry=self.page_registry.copy(),
            )
            workspace.home_requested.connect(self.show_home)
            workspace.project_changed.connect(self._workspace_project_changed)
            self._workspaces[key] = workspace
            self.stack.addWidget(workspace)
        self.workspace = workspace
        self.workspace.refresh()
        self.stack.setCurrentWidget(self.workspace)

    def _workspace_project_changed(self, project: FluidFlowerProject) -> None:
        self.registry.remember(project)
        self.home.refresh()

    def show_home(self) -> None:
        self.home.refresh()
        self.stack.setCurrentWidget(self.home)

    @staticmethod
    def _workspace_key(root: Path) -> str:
        return str(resolve_project_package_root(root).resolve()).casefold()

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        for workspace in self._workspaces.values():
            for page in workspace.pages.values():
                shutdown = getattr(page, "shutdown_background_work", None)
                if callable(shutdown):
                    shutdown()
        super().closeEvent(event)
