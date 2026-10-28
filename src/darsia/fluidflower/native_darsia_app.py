"""Native DarSIA navigation for original and phase-model FluidFlower workflows.

Run with ``python -m darsia.fluidflower.native_darsia_app``.  The ordinary
``python -m darsia.gui`` entry point and its main window remain unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from copy import deepcopy
from dataclasses import replace
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QEventLoop, QSettings, QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.calibration_header import NativeStatusHeader, row_category, status_line
from darsia.fluidflower.image_selection_ui import NativeImageSelectionWorkflow
from darsia.fluidflower.depth_setup_ui import NativeDepthWorkflow
from darsia.fluidflower.rig_display import NativeRigContext
from darsia.fluidflower.crop_setup_ui import NativeCropWorkflow
from darsia.fluidflower.coloured_map_ui import NativeColouredMapWorkflow
from darsia.fluidflower.crop_wizard_ui import prepare_crop_wizard
from darsia.fluidflower.curvature_setup_ui import NativeCurvatureWorkflow
from darsia.fluidflower.project import ProjectRegistry
from darsia.fluidflower.series_geometry import series_rig_class
from darsia.fluidflower.series_geometry_workspace import SeriesGeometryWorkspace
from darsia.fluidflower.series_paths import configured_series, series_folder_path
from darsia.fluidflower.series_method import (
    METHOD_DESCRIPTIONS,
    METHOD_HEADERS,
    METHOD_NAMES,
    MethodSelection,
    SeriesMethodPanel,
    missing_method_references,
    read_series_method,
    save_series_method,
)
from darsia.fluidflower.templates import RIG_CLASSES, rig_class_rank
from darsia.fluidflower.ui.home import HomePage
from darsia.gui.ui.main_window import MainWindow
from darsia.gui.ui.sidebar import GroupHeaderLabel, Sidebar, SidebarRow
from darsia.gui.ui.theme import apply_theme, get_theme


class _NativeSeriesRegistry:
    """Present current series and earlier project packages on Start."""

    def __init__(
        self,
        series: Sequence[tuple[str, str, Path]],
        *,
        project_registry: ProjectRegistry | None = None,
    ) -> None:
        self.project_registry = project_registry or ProjectRegistry()
        self.replace(series)

    def replace(self, series: Sequence[tuple[str, str, Path]]) -> None:
        self.series = tuple(series)
        self.series_by_path = {str(root): key for key, _label, root in self.series}

    def forget_missing(self) -> None:
        # An unavailable folder must remain visible so users can reconnect it.
        pass

    def entries(self) -> list[dict[str, str]]:
        """Series folders, then earlier projects, each sorted by rig class."""

        current = []
        for key, label, root in self.series:
            rig_class = series_rig_class(root)
            current.append(
                {
                    "name": label,
                    "series_id": key,
                    "run_count": self._run_count(root),
                    "path": str(root),
                    "template_id": _rig_class_text(rig_class),
                    "rig_class": rig_class,
                }
            )
        current.sort(key=_by_rig_class)
        self.legacy_by_path: dict[str, dict] = {}
        seen_paths = {str(root).casefold() for _key, _label, root in self.series}
        legacy = []
        for project in self.project_registry.entries():
            path = str(project.get("path") or "")
            if not path or path.casefold() in seen_paths:
                continue
            seen_paths.add(path.casefold())
            self.legacy_by_path[path] = project
            label = str(project.get("name") or project.get("series_id") or path)
            rig_class = _project_rig_type(path)
            legacy.append(
                {
                    "name": f"{label} · earlier project",
                    "series_id": str(project.get("series_id") or ""),
                    "run_count": str(project.get("run_count") or ""),
                    "path": path,
                    "template_id": _rig_class_text(rig_class),
                    "rig_class": rig_class,
                    "last_opened_utc": str(project.get("last_opened_utc") or ""),
                }
            )
        legacy.sort(key=_by_rig_class)
        self.last_entries = current + legacy
        return self.last_entries

    @staticmethod
    def _run_count(root: Path) -> str:
        try:
            with (root / "series.toml").open("rb") as stream:
                runs = tomllib.load(stream).get("runs", {})
        except (OSError, ValueError):
            return ""
        ids = runs.get("ids") if isinstance(runs, dict) else None
        return str(len(ids)) if isinstance(ids, list) else ""


def _rig_class_text(key: str) -> str:
    rig_class = RIG_CLASSES.get(key)
    return rig_class.short if rig_class is not None else "Not set"


def _by_rig_class(row: dict[str, str]) -> tuple[int, str]:
    return rig_class_rank(row.get("rig_class", "")), str(row.get("name", "")).casefold()


def _project_rig_type(path: str) -> str:
    """Apparatus class recorded in an earlier project package ("" if unknown)."""

    import json

    try:
        manifest = json.loads((Path(path) / "series.json").read_text("utf-8-sig"))
    except (OSError, ValueError):
        return ""
    rig = manifest.get("rig") if isinstance(manifest, dict) else None
    value = str(rig.get("type") or "") if isinstance(rig, dict) else ""
    return value if value in RIG_CLASSES else ""


class _NoLegacyDrafts:
    """Configured series do not use the former series.json draft store."""

    def drafts(self) -> list:
        return []


class NativeHomePage(HomePage):
    """Start page for current series folders and earlier project packages."""

    def __init__(
        self,
        *,
        series: Sequence[tuple[str, str, Path]] | None = None,
        initial_series: str | None = None,
        project_registry: ProjectRegistry | None = None,
    ) -> None:
        self.series_registry = _NativeSeriesRegistry(
            configured_series() if series is None else series,
            project_registry=project_registry,
        )
        self.initial_series = initial_series or (
            self.series_registry.series[0][0] if self.series_registry.series else ""
        )
        super().__init__(self.series_registry, draft_store=_NoLegacyDrafts())
        self.new_button.hide()
        self.open_button.setText("Open series")
        self.open_button.hide()
        self.open_selected_button.setText("Open series")
        self.open_selected_button.show()
        self.update_run_sources_button.setText("Open experiment setup")
        self.update_run_sources_button.hide()
        self.add_runs_button.hide()
        self.draft_heading.hide()
        self.draft_note.hide()
        self.table.setHorizontalHeaderLabels(
            ["Series", "ID", "Runs", "Rig class", "Method", "Project folder"]
        )
        # A single click selects a row for folder settings or removal.
        # The inherited double click and Open series button open Setup.
        self.table.setToolTip("Select a series for folder settings. Double-click or choose Open series to continue to Setup.")
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.package_note.setText(
            "Series folders contain current configuration, reviews, models and "
            "results. Earlier project packages open in their original workspace."
        )
        self.intro.setText(
            "Choose an existing experimental series or create a new series."
        )
        self.intro.setWordWrap(True)
        self.new_series_method = QComboBox()
        for method in ("colour_path", "phase_model"):
            self.new_series_method.addItem(METHOD_HEADERS[method], method)
        self.new_series_method.setAccessibleName("New series method")
        self.new_series_method.setToolTip(
            "Original DarSIA creates one TOML configuration. The phase-map method "
            "creates a series folder."
        )
        self.new_series_button = QPushButton("New series…")
        self.new_series_button.setToolTip(
            "Create a project using the selected method."
        )
        self.add_existing_button = QPushButton("Add existing series folder…")
        self.add_existing_button.setToolTip(
            "Register a series folder or its DarSIA child containing series.toml. "
            "The folder may be local or on a shared drive."
        )
        self.remove_series_button = QPushButton("Remove from Start")
        self.remove_series_button.setToolTip("Remove this local registration. Project files and running jobs are kept.")
        self.folders_button = QPushButton("Series folders…")
        self.folders_button.setToolTip("Edit folders for the selected series, including images, Results and this computer’s working folder.")
        self.refresh_series_button = QPushButton("Refresh series")
        self.refresh_series_button.setToolTip(
            "Reload registered series and earlier projects from local settings."
        )
        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Method:"))
        add_row.addWidget(self.new_series_method)
        add_row.addWidget(self.new_series_button)
        add_row.addWidget(self.add_existing_button)
        add_row.addWidget(self.refresh_series_button)
        add_row.addWidget(self.folders_button)
        add_row.addWidget(self.remove_series_button)
        add_row.addStretch(1)
        self.layout().insertLayout(2, add_row)
        if not self.series_registry.series:
            self.package_note.setText(
                "No experimental series is configured. Add an existing DarSIA "
                "series folder, or choose New series… to create one."
            )
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text() == self.initial_series:
                self.table.selectRow(row)
                break
        self._project_selection_changed()

    def refresh(self) -> None:
        super().refresh()
        rows = getattr(self.series_registry, "last_entries", [])
        for index, row in enumerate(rows[: self.table.rowCount()]):
            method_item = self.table.item(index, 4)
            root = row.get("path", "")
            folder_item = self.table.item(index, 5)
            if root in self.series_registry.series_by_path:
                try:
                    folder_item.setText(os.path.abspath(series_folder_path(root, "project")))
                    folder_item.setToolTip("DarSIA setup folder: " + root)
                except (OSError, ValueError, TypeError) as error:
                    folder_item.setToolTip("Could not read the project folder: " + str(error))
            choice = (
                read_series_method(Path(root))
                if root in self.series_registry.series_by_path
                else MethodSelection()
            )
            method_item.setText(METHOD_NAMES.get(choice.active, "–").lower())
            method_item.setToolTip(
                choice.error or "Chosen method saved in series.toml."
            )
            item = self.table.item(index, 3)
            rig = RIG_CLASSES.get(row.get("rig_class", ""))
            if item is None:
                continue
            if rig is None:
                item.setToolTip(
                    "No rig class recorded. Choose one in Setup > Geometry and "
                    "facies; it is stored in series.toml."
                )
            else:
                item.setToolTip(rig.title() + (f". {rig.source}" if rig.source else ""))

    def selected_series(self) -> str:
        if self.selected_legacy_project():
            return ""
        return self.series_registry.series_by_path.get(
            self._selected_project_path(), self.initial_series
        )

    def selected_legacy_project(self) -> str:
        path = self._selected_project_path()
        return path if path in self.series_registry.legacy_by_path else ""

    def _project_selection_changed(self) -> None:
        super()._project_selection_changed()
        legacy = bool(self.selected_legacy_project())
        self.new_button.setText(
            "Edit earlier project setup" if legacy else "Set up an experiment"
        )
        self.open_button.setText("Open series")
        if hasattr(self, "folders_button"):
            self.folders_button.setEnabled(bool(self._selected_project_path()) and not legacy)
            self.remove_series_button.setEnabled(bool(self._selected_project_path()))
        self.open_selected_button.setText(
            "Open earlier project" if legacy else "Open series"
        )
        self.update_run_sources_button.setText(
            "Open project setup" if legacy else "Open experiment setup"
        )

    def update_series(
        self, series: Sequence[tuple[str, str, Path]], *, selected: str
    ) -> None:
        self.series_registry.replace(series)
        self.initial_series = selected
        self.refresh()
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text() == selected:
                self.table.selectRow(row)
                break
        if series:
            self.package_note.setText(
                "Series folders contain current configuration, reviews, models and "
                "results. Earlier project packages open in their original workspace."
            )


# Sections of the shared native page; preparation is routed from Setup.
PHASE_PAGE_SECTIONS = {
    ("phase_model", "normalisation"): "normalisation",
    ("phase_model", "training"): "phase_model",
    ("method", "use_for_series"): "use",
}


def _native_row(aliases, category, text, source, item_id, help_text, icon="fa5s.circle"):
    """One sidebar row of a native page, routed by its logical (source, item_id)."""
    display_id = f"{source}:{item_id}"
    aliases[category][display_id] = (source, item_id)
    return (text, display_id, icon, help_text)


class PhaseModelWorkspace(QWidget):
    """Training and calibration backed by one lazily loaded Studio instance."""

    def __init__(
        self,
        *,
        series: Sequence[tuple[str, str, Path]] | None = None,
        initial_series: str | None = None,
        native_context_binder=None,
        defer_native_context=False,
    ) -> None:
        super().__init__()
        self.configured_series = tuple(
            configured_series() if series is None else series
        )
        self.series_roots = {key: root for key, _label, root in self.configured_series}
        initial_series = initial_series or (
            self.configured_series[0][0] if self.configured_series else ""
        )
        self.setObjectName("phaseModelWorkspace")
        self.studio_page: QWidget | None = None
        self.native_context_binder = native_context_binder
        self.defer_native_context = defer_native_context
        self._requested_mode = "training"
        self._requested_section = "phase_model"

        layout = QVBoxLayout(self)
        heading = QLabel("Phase-model workflow · training and calibration")
        heading_font = heading.font()
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading)

        controls = QFormLayout()
        self.series_selector = QComboBox()
        for key, label, _root in self.configured_series:
            self.series_selector.addItem(label, key)
        index = self.series_selector.findData(initial_series)
        if index >= 0:
            self.series_selector.setCurrentIndex(index)
        elif initial_series:
            raise ValueError(f"Unknown experimental series: {initial_series}")
        controls.addRow("Experimental series", self.series_selector)

        self.series_folder = QLineEdit()
        self.series_folder.setReadOnly(True)
        self.series_folder.setToolTip("Configured DarSIA folder for this series")
        controls.addRow("Series folder", self.series_folder)
        for field in (self.series_selector, self.series_folder):
            controls.labelForField(field).hide()
            field.hide()
        heading.hide()
        layout.addLayout(controls)
        self.series_support_status = QLabel()
        self.series_support_status.setWordWrap(True)
        self.series_support_status.hide()
        layout.addWidget(self.series_support_status)

        workspace = QHBoxLayout()
        self.calibration_sidebar = Sidebar(
            {
                "calibration": (
                    "Phase-model calibration",
                    "fa5s.balance-scale",
                    [
                        (
                            "Phase-model methods",
                            [
                                (
                                    "Phase model training",
                                    "phase_model_training",
                                    "fa5s.circle",
                                    "Review evidence, train models and inspect validation.",
                                ),
                                (
                                    "Mass calibration from phase model",
                                    "mass_from_phase",
                                    "fa5s.circle",
                                    "Fit and inspect mass calibration from a selected phase model.",
                                ),
                            ],
                        )
                    ],
                )
            }
        )
        self.calibration_sidebar.setMinimumWidth(230)
        self.calibration_sidebar.setMaximumWidth(340)
        self.calibration_sidebar._sections["calibration"].set_expanded(True)
        mass_row = self.calibration_sidebar._sections["calibration"].get_row(
            "mass_from_phase"
        )
        mass_row._text_label.setWordWrap(True)
        mass_row.setMinimumHeight(40)
        self.calibration_sidebar.select("calibration", "phase_model_training")
        self.calibration_sidebar.selection_changed.connect(self._calibration_selected)
        workspace.addWidget(self.calibration_sidebar)

        studio_panel = QWidget()
        self.studio_host = QVBoxLayout(studio_panel)
        self.studio_host.setContentsMargins(0, 0, 0, 0)
        self.studio_placeholder = QLabel(
            "Choose phase-model training or mass calibration."
        )
        self.studio_placeholder.setWordWrap(True)
        self.studio_host.addWidget(self.studio_placeholder)
        workspace.addWidget(studio_panel, 1)
        layout.addLayout(workspace, 1)

        self._selected_series = initial_series
        self._update_series_folder()
        if not self.defer_native_context:
            self._bind_native_context()
        self.series_selector.currentIndexChanged.connect(self._series_changed)

    @property
    def series_id(self) -> str:
        return str(self.series_selector.currentData() or "")

    def _update_series_folder(self) -> None:
        root = self.series_roots.get(self.series_id)
        self.series_folder.setText(str(root) if root is not None else "")

    def update_series(
        self, series: Sequence[tuple[str, str, Path]], *, selected: str
    ) -> None:
        self.configured_series = tuple(series)
        self.series_roots = {key: root for key, _label, root in self.configured_series}
        self.series_selector.blockSignals(True)
        self.series_selector.clear()
        for key, label, _root in self.configured_series:
            self.series_selector.addItem(label, key)
        index = self.series_selector.findData(selected)
        if index >= 0:
            self.series_selector.setCurrentIndex(index)
        self.series_selector.blockSignals(False)
        self._series_changed(-1)

    def _bind_native_context(self) -> bool:
        """Bind before Studio construction/catalogue requests; no fallback input."""
        if self.native_context_binder is None:
            return True
        previous_error = getattr(self, "_native_context_error", "")
        try:
            self.native_context_binder(self.series_roots.get(self.series_id))
        except (OSError, ValueError, TypeError) as error:
            self._native_context_error = f"Could not bind the series native configuration: {type(error).__name__}: {error}"
            self.series_support_status.setText(self._native_context_error)
            self.series_support_status.show()
            return False
        self._native_context_error = ""
        if previous_error and self.series_support_status.text() == previous_error:
            self.series_support_status.clear()
            self.series_support_status.hide()
        if getattr(self, "_native_context_disabled", False) and self.studio_page is not None:
            self._native_context_disabled = False
            self.studio_page.setEnabled(True)
            studio = getattr(self.studio_page, "studio", None)
            if studio is not None:
                self._bind_studio_series(studio)
        return True

    def ensure_studio(self) -> QWidget:
        """Create the adapter once; it creates SharedPhaseModelTrainingPage on show."""
        if not self.series_id:
            self.studio_placeholder.setText(
                "Add an existing experimental series folder on Start before "
                "opening phase-model training or mass calibration."
            )
            return self.studio_placeholder
        if not self._bind_native_context():
            if self.studio_placeholder is not None:
                self.studio_placeholder.setText(self._native_context_error)
                return self.studio_placeholder
            self._native_context_disabled = True
            self.studio_page.setEnabled(False)
            return self.studio_page
        if getattr(self, '_context_needs_apply', False):
            deferred = self.defer_native_context
            self.defer_native_context = False
            try:
                self._series_changed(-1)
                self._context_needs_apply = False
            finally:
                self.defer_native_context = deferred
        if self.studio_page is None:
            from darsia.fluidflower.native_phase_subpage import NativePhaseModelStudioSubpage

            self.studio_page = NativePhaseModelStudioSubpage(
                self.series_id, project_root=self.series_folder.text(),
                step_options=getattr(self, "native_step_options", None),
            )
            self.studio_page.studio_loaded.connect(self._verify_studio_series)
            self.studio_page.studio_loaded.connect(self._show_selected_studio_mode)
            if self.studio_placeholder is not None:
                self.studio_host.removeWidget(self.studio_placeholder)
                self.studio_placeholder.deleteLater()
                self.studio_placeholder = None
            self.studio_host.addWidget(self.studio_page)
        return self.studio_page

    def _bind_studio_series(self, studio: QWidget) -> None:
        """Keep an unsupported series from operating on a different model filter."""
        try:
            studio.set_model_series(self.series_id)
        except ValueError:
            self.studio_page.setEnabled(False)
            self.series_support_status.setText(
                "The phase-model Studio does not yet support this series. "
                "Its training and mass controls are disabled for this selection."
            )
            self.series_support_status.show()
        else:
            self.studio_page.setEnabled(True)
            self.series_support_status.clear()
            self.series_support_status.hide()

    def _verify_studio_series(self) -> None:
        studio = getattr(self.studio_page, "studio", None)
        if (
            studio is not None
            and getattr(studio, "_active_model_series", self.series_id)
            != self.series_id
        ):
            self._bind_studio_series(studio)

    def _calibration_selected(self, _action: str, item_id: str) -> None:
        self._requested_mode = "mass" if item_id == "mass_from_phase" else "training"
        if self.isVisible():
            self.ensure_studio()
        self._show_selected_studio_mode()

    def select_calibration(self, mode: str) -> None:
        if mode not in {"training", "mass"}:
            raise ValueError(f"Unknown calibration mode: {mode}")
        item_id = {
            "training": "phase_model_training",
            "mass": "mass_from_phase",
        }[mode]
        self.calibration_sidebar.select("calibration", item_id)

    def _show_selected_studio_mode(self) -> None:
        """Navigate existing Studio tabs without duplicating its model/job state."""
        if self.studio_page is not None:
            self.studio_page.show()
        if getattr(self.studio_page, "native_numbered_steps", False):
            show_steps = getattr(self.studio_page, "show_steps", None)
            if show_steps is not None:
                show_steps()
            steps = getattr(self.studio_page, "native_steps", None)
            if steps is not None:
                steps.show_section(self._requested_section)
            return
        studio = getattr(self.studio_page, "studio", None)
        tabs = getattr(studio, "tabs", None)
        mass_tab = getattr(studio, "mass_workspace_tab", None)
        if not isinstance(tabs, QTabWidget) or mass_tab is None:
            return
        mass_index = tabs.indexOf(mass_tab)
        if mass_index < 0:
            return
        if self._requested_mode == "mass":
            tabs.setTabVisible(mass_index, True)
            tabs.setCurrentIndex(mass_index)
            for index in range(mass_index):
                tabs.setTabVisible(index, False)
            tabs.tabBar().hide()
        else:
            for index in range(mass_index):
                tabs.setTabVisible(index, True)
            if tabs.currentIndex() == mass_index:
                tabs.setCurrentWidget(studio.evidence_tab)
            tabs.setTabVisible(mass_index, False)
            tabs.tabBar().show()

    def _series_changed(self, _index: int) -> None:
        page = self.studio_page
        studio = getattr(page, "studio", None)
        review_guard = getattr(page, "review_context_block_reason", None)
        if callable(review_guard) and (reason := review_guard()):
            studio.review_status.setText(reason)
            page.inline_review.notice.setText(reason)
            with QSignalBlocker(self.series_selector):
                self.series_selector.setCurrentIndex(
                    self.series_selector.findData(self._selected_series))
            return
        if studio is not None:
            # A running job must retain its series context until it finishes.
            notice = page._active_job_notice(studio)
            if notice is not None:
                QMessageBox.information(self, *notice)
                self.series_selector.blockSignals(True)
                self.series_selector.setCurrentIndex(
                    self.series_selector.findData(self._selected_series)
                )
                self.series_selector.blockSignals(False)
                return

        if not self.defer_native_context and not self._bind_native_context():
            with QSignalBlocker(self.series_selector):
                self.series_selector.setCurrentIndex(self.series_selector.findData(self._selected_series))
            return
        self._selected_series = self.series_id
        self._update_series_folder()
        if self.defer_native_context:
            if page is not None:
                page.setEnabled(False)
            return
        if page is None:
            return
        page.series_id = self.series_id
        page.project_root = Path(self.series_folder.text())
        page._mass_series_bound = False
        if studio is not None:
            if getattr(page, "native_numbered_steps", False):
                page.bind_context(self.series_id, page.project_root)
            elif hasattr(studio, "set_project_root"):
                studio.set_project_root(page.project_root, fixed=True)
            self._bind_studio_series(studio)
            workspace = getattr(studio, "_mass_workspace", None)
            control = getattr(workspace, "series", None)
            if control is not None:
                index = control.findData(self.series_id)
                if index >= 0:
                    control.setCurrentIndex(index)


def _same_executable(first: str | Path, second: str | Path) -> bool:
    try:
        return Path(first).resolve() == Path(second).resolve()
    except OSError:
        return str(first) == str(second)


def use_environment_python(runner) -> None:
    """Start the original DarSIA workflows with the environment's interpreter.

    The Windows launcher runs this application with a base CPython and adds the
    environment's site-packages at run time, so ``sys.executable`` cannot import
    darsia. The original DarSIA pages start Setup, calibration and analysis as
    ``sys.executable -m darsia...``; this swaps that interpreter for the one of
    the environment (``DARSIA_FLUIDFLOWER_PYTHON`` or ``VIRTUAL_ENV``).
    """

    from darsia.fluidflower.phase_workflow import phase_python_executable

    start = runner.start_workflow_process

    def start_workflow_process(argv, *args, **kwargs):
        argv = [str(value) for value in argv]
        if argv[1:3] == ["-m", "darsia.presets.workflows.user_interface_setup"]:
            argv[2] = "darsia.fluidflower.setup_rig"
        if argv and _same_executable(argv[0], sys.executable):
            argv[0] = phase_python_executable()
        return start(argv, *args, **kwargs)

    start_workflow_process.__wrapped__ = start
    runner.start_workflow_process = start_workflow_process


def clear_config_placeholder(window) -> None:
    """Let the original pages see that no configuration is open.

    DarSIA's main window keeps the open configuration file in a hidden label
    whose text is "No config loaded." until a file is opened, while Setup,
    calibration, analysis and the helpers treat only an empty label (or "No
    file chosen") as missing. ▶ before File > Open Config… would therefore
    start the workflow with that text as the file name and end in a
    traceback. An empty label makes them ask for a configuration instead.
    """

    if not window.config_file:
        window.config_path_label.setText("")


def _read_toml(path: Path) -> dict:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def _has_settings(path: Path) -> bool:
    """Whether a TOML file holds any key or table (comments do not count)."""
    try:
        return bool(_read_toml(path))
    except FileNotFoundError:
        return False
    except (OSError, tomllib.TOMLDecodeError):
        return True


def run_configurations(root: str | Path) -> list[tuple[str, Path | None, str]]:
    """The file the settings panel opens for each run of a series.

    DarSIA's settings panel opens one configuration file, while a series keeps
    the settings of a run in ``config/common.toml`` and
    ``config/runs/<run>.toml``. A merged single file in ``config/merged`` holds
    both (FF_AC, FF_UM and FF_BM); without one, the run file serves alone when
    the common file holds no settings (FF_RE). Returns ``(run, file, note)``
    in the order of ``series.toml``; ``file`` is None when no single file holds
    the run's settings, and ``note`` says why.
    """

    root = Path(root)
    try:
        manifest = _read_toml(root / "series.toml")
    except (OSError, tomllib.TOMLDecodeError):
        return []
    paths = manifest.get("paths") or {}
    runs = manifest.get("runs") or {}
    files = runs.get("config_files") or {}
    merged_files = runs.get("merged_files") or {}
    merged_folder = root / str(paths.get("merged_configs", "config/merged"))
    common = root / str(paths.get("common_config", "config/common.toml"))
    run_folder = str(paths.get("run_configs", "config/runs"))
    common_has_settings: bool | None = None
    choices: list[tuple[str, Path | None, str]] = []
    for run in [str(value) for value in runs.get("ids") or []]:
        if run in merged_files:
            name = merged_files[run]
            if not isinstance(name, str) or not name.strip():
                choices.append((run, None, "Registered merged configuration path is invalid"))
                continue
            merged = Path(name)
            if not merged.is_absolute():
                merged = root / merged
            if merged.is_file():
                choices.append((run, merged, "Registered merged configuration (common and run settings)"))
            else:
                choices.append((run, None, f"{name} does not exist"))
            continue
        merged = merged_folder / f"{run}.toml"
        if merged.is_file():
            choices.append(
                (run, merged, "Merged configuration (common and run settings)")
            )
            continue
        name = str(files.get(run) or f"{run_folder}/{run}.toml")
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file():
            choices.append((run, None, f"{name} does not exist"))
            continue
        if common_has_settings is None:
            common_has_settings = _has_settings(common)
        if common_has_settings:
            choices.append(
                (
                    run,
                    None,
                    f"The settings are split between {common.name} and {name}, the "
                    "settings panel opens one file, and there is no merged file "
                    f"{merged.relative_to(root).as_posix()}",
                )
            )
        else:
            choices.append((run, path, "Run configuration"))
    return choices


def _same_file(first: str | Path | None, second: str | Path | None) -> bool:
    if not first or not second:
        return False
    if os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
            os.path.abspath(second)
        )


def _file_stamp(path: str | Path) -> tuple[int, int] | None:
    try:
        stat = Path(path).stat()
    except (OSError, ValueError):
        return None
    return stat.st_mtime_ns, stat.st_size


def _ask_config_reload(parent: QWidget, path: str) -> str:
    """Ask what to do with a configuration that changed on disk.

    Returns ``"reload"``, ``"overwrite"`` or ``"cancel"``.
    """

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Configuration changed on disk")
    box.setText(
        f"{Path(path).name} changed on disk after it was opened here, for example "
        "because a script regenerated it."
    )
    box.setInformativeText(
        "Reload it and run with the settings in the file, or keep the settings "
        "shown here and write them over the file."
    )
    reload_button = box.addButton("Reload and run", QMessageBox.ButtonRole.AcceptRole)
    keep_button = box.addButton(
        "Overwrite and run", QMessageBox.ButtonRole.DestructiveRole
    )
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(reload_button)
    box.exec()
    clicked = box.clickedButton()
    if clicked is reload_button:
        return "reload"
    if clicked is keep_button:
        return "overwrite"
    return "cancel"


def guard_config_file(window) -> None:
    """Keep the settings panel from writing an old copy over a changed file.

    DarSIA's settings panel writes the configuration it shows to the opened
    file before every workflow. When the file changed on disk after it was
    opened, for example because a series script regenerated it, that would put
    the older copy back. The window therefore records the file's modification
    time and size whenever it opens or saves the file, and before a workflow
    whose file has changed since, it asks whether to reload the file or to
    overwrite it.
    """

    controller = window.config_controller
    factory = window.settings_factory
    runner = window.process_runner
    load = controller.load_config
    save = factory.save_settings
    start = runner.start_workflow_process

    def record() -> None:
        path = window.config_file
        stamp = _file_stamp(path) if path else None
        window._config_stamp = (str(path), stamp) if stamp is not None else None

    def load_config(*args, **kwargs):
        result = load(*args, **kwargs)
        record()
        hook = getattr(window, "_config_loaded", None)
        if callable(hook):
            hook()
        return result

    def save_settings(*args, **kwargs):
        result = save(*args, **kwargs)
        record()
        return result

    def start_workflow_process(argv, *args, **kwargs):
        path = window.config_file
        recorded = getattr(window, "_config_stamp", None)
        if path and recorded is not None and recorded[0] == str(path):
            current = _file_stamp(path)
            if current is not None and current != recorded[1]:
                choice = _ask_config_reload(window, str(path))
                if choice == "cancel":
                    window.print_log(
                        "Run cancelled: the configuration changed on disk."
                    )
                    return None
                if choice == "reload":
                    selection = window.sidebar.get_selected()
                    controller.load_config()
                    if selection is not None:
                        window.sidebar.select(*selection)
                    window.print_log(f"Reloaded {path}, which changed on disk.")
        return start(argv, *args, **kwargs)

    load_config.__wrapped__ = load
    save_settings.__wrapped__ = save
    start_workflow_process.__wrapped__ = start
    controller.load_config = load_config
    factory.save_settings = save_settings
    runner.start_workflow_process = start_workflow_process
    # File > Save Config was connected to the unwrapped method.
    menu = getattr(window, "menu_builder", None)
    save_action = getattr(menu, "save_action", None)
    if save_action is not None:
        try:
            save_action.triggered.disconnect()
        except (RuntimeError, TypeError):
            pass
        save_action.triggered.connect(lambda _checked=False: factory.save_settings())
    window._config_stamp = None


def guard_published_dataset(window) -> None:
    """Refuse settings saves and workflows while the series is a published dataset.

    A published dataset opens read-only (its folder is protected in this
    process). Saving settings and DarSIA's workflows would write into it, so
    they are refused with a note; New analysis writes to the user's results
    location instead.
    """

    factory = window.settings_factory
    runner = window.process_runner
    save = factory.save_settings
    start = runner.start_workflow_process

    def save_settings(*args, **kwargs):
        """Save as before, unless the series is a published dataset."""
        if window.dataset_series_root() is not None:
            window.print_log(
                "This series is a published dataset and opens read-only; "
                "settings are not saved."
            )
            return None
        return save(*args, **kwargs)

    def start_workflow_process(argv, *args, **kwargs):
        """Start as before, unless the series is a published dataset."""
        if window.dataset_series_root() is not None:
            window.print_log(
                "Workflows write into the series folder, and a published dataset is "
                "read-only. Use New analysis, which writes to your results location."
            )
            return None
        return start(argv, *args, **kwargs)

    save_settings.__wrapped__ = save
    start_workflow_process.__wrapped__ = start
    factory.save_settings = save_settings
    runner.start_workflow_process = start_workflow_process


def settle_settings_before_sync(window) -> None:
    """Let DarSIA fill its settings fields before their values are read back.

    DarSIA's settings panel fills several kinds of fields one event-loop turn
    after it builds them (``QTimer.singleShot(0, ...)``): data selections, the
    per-folder imaging and blacklist protocols and the facies label lists.
    Rebuilding the panel (selecting a sidebar row) or saving first copies every
    field back into the configuration. Read in the same turn as the build, those
    fields are still empty, so the copy drops data selections, all but the first
    protocol folder and the facies sections, and the next save writes that to
    the file. DarSIA's own flow never hits this, because a person clicks after
    the event loop has run; this window opens a run's file and selects its step
    in one call. Processing the pending events (not user input) before each copy
    makes the copy read filled fields.
    """

    factory = window.settings_factory
    sync = factory._sync_settings_inputs_to_config_dict

    def _sync_settings_inputs_to_config_dict(*args, **kwargs):
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        return sync(*args, **kwargs)

    _sync_settings_inputs_to_config_dict.__wrapped__ = sync
    factory._sync_settings_inputs_to_config_dict = _sync_settings_inputs_to_config_dict


# --------------------------------------------------------------------------- #
# Mass calibration: manual (DarSIA's sliders) or automatic (Optuna)
# --------------------------------------------------------------------------- #

#: Sidebar row of the automatic (Optuna) mass calibration, next to DarSIA's
#: interactive "mass" row in the calibration section.
OPTUNA_MASS_ROW = "mass_optuna"
_OPTUNA_HELP = (
    "Automatic mass calibration: an Optuna search for the signal values of the "
    "sand facies that makes the detected CO2 mass match the injected mass in the "
    "photographs of [calibration.mass] data_selection. It starts from the stored "
    "colour-to-mass calibration (and creates Default mass first when there is "
    "none), keeps the flash, saves the best values in the same place and keeps "
    "the previous calibration, every trial and a mass-balance figure in "
    "mass_calibration_optuna next to the colour path. Uses the same settings as "
    "Mass, manual."
)


class AutomaticMassCalibrationDialog(QDialog):
    """Options of an automatic (Optuna) mass calibration before it starts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mass, Optuna (automatic calibration)")
        layout = QVBoxLayout(self)
        text = QLabel(
            "Searches the signal values of the sand facies so that the detected CO2 "
            "mass matches the injected mass in the [calibration.mass] photographs. "
            "The best values replace the stored calibration; the previous one and a "
            "report are kept in mass_calibration_optuna."
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        form = QFormLayout()
        settings = QSettings()
        self.trials = QSpinBox()
        self.trials.setRange(10, 5000)
        self.trials.setSingleStep(50)
        self.trials.setValue(settings.value("native_app/optuna/trials", 200, type=int))
        self.trials.setToolTip(
            "Optuna trials (a fraction of a second each), followed by a short local "
            "refinement of the best one"
        )
        form.addRow("Trials", self.trials)
        self.values = QComboBox()
        self.values.addItem("Shared by the sand facies", False)
        self.values.addItem("Separate per sand facies", True)
        self.values.setCurrentIndex(
            1 if settings.value("native_app/optuna/per_facies", False, type=bool) else 0
        )
        self.values.setToolTip(
            "Shared: one signal function for all sands with a colour path (robust "
            "when a sand holds little CO2 in the photographs). Separate: one per sand."
        )
        form.addRow("Signal values", self.values)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(settings.value("native_app/optuna/seed", 0, type=int))
        self.seed.setToolTip("Seed of the sampler; the same seed repeats a run")
        form.addRow("Seed", self.seed)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> dict:
        return {
            "trials": self.trials.value(),
            "per_facies": bool(self.values.currentData()),
            "seed": self.seed.value(),
        }

    def remember(self) -> None:
        settings = QSettings()
        options = self.options()
        settings.setValue("native_app/optuna/trials", options["trials"])
        settings.setValue("native_app/optuna/per_facies", options["per_facies"])
        settings.setValue("native_app/optuna/seed", options["seed"])


def _ask_optuna_options(parent: QWidget) -> dict | None:
    dialog = AutomaticMassCalibrationDialog(parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    dialog.remember()
    return dialog.options()


def optuna_mass_calibration_argv(config_file: str | Path, options: dict) -> list[str]:
    """Command line of darsia.fluidflower.native_mass_calibration."""
    argv = [
        sys.executable,
        "-m",
        "darsia.fluidflower.native_mass_calibration",
        "--config",
        str(Path(config_file).resolve()),
        "--trials",
        str(int(options["trials"])),
        "--seed",
        str(int(options["seed"])),
    ]
    if options.get("per_facies"):
        argv.append("--per-facies")
    return argv


def add_automatic_mass_calibration(window) -> None:
    """Offer the mass calibration as a choice: manual (sliders) or automatic.

    DarSIA's calibration section has one interactive "Mass" row. This renames it
    "Mass, manual" and adds "Mass, Optuna" (automatic) below it, which shows the
    same settings and runs :mod:`darsia.fluidflower.native_mass_calibration`
    through DarSIA's process runner (log, stop button and done dialog as for the
    other steps).
    """

    section = window.sidebar._sections["calibration"]
    manual = section.get_row("mass")
    if manual is None or section.get_row(OPTUNA_MASS_ROW) is not None:
        return
    manual.label = "Mass, manual"
    manual._text_label.setText(manual.label)
    row = SidebarRow(
        "calibration",
        OPTUNA_MASS_ROW,
        "Mass, Optuna",
        manual.icon_name,
        _OPTUNA_HELP,
    )
    row.clicked.connect(lambda: section._on_row_clicked(OPTUNA_MASS_ROW))
    section.items_layout.insertWidget(section.items_layout.indexOf(manual) + 1, row)
    # Keep the row table in screen order (code iterates it for dots and rows).
    rows = dict(section._rows)
    rows[OPTUNA_MASS_ROW] = row
    section._rows = {
        widget.checkbox_id: widget
        for widget in (
            section.items_layout.itemAt(i).widget()
            for i in range(section.items_layout.count())
        )
        if isinstance(widget, SidebarRow) and widget.checkbox_id in rows
    }

    factory = window.settings_factory
    display = factory.display_settings

    def display_settings(action, checked_ids):
        if action == "calibration":
            checked_ids = ["mass" if i == OPTUNA_MASS_ROW else i for i in checked_ids]
        return display(action, checked_ids)

    display_settings.__wrapped__ = display
    factory.display_settings = display_settings

    tab = window.calibration_tab
    run = tab.run_calibration

    def run_calibration(*args, **kwargs):
        if window.selected_checkbox_id != OPTUNA_MASS_ROW:
            return run(*args, **kwargs)
        config_file = window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            window.print_log("Please select a config file first.")
            return None
        options = _ask_optuna_options(window)
        if options is None:
            window.print_log("Automatic mass calibration cancelled.")
            return None
        window.print_log(
            f"Starting automatic mass calibration (Optuna): {options['trials']} trials, "
            f"{'separate' if options['per_facies'] else 'shared'} signal values."
        )
        toolbar = window.toolbar_builder
        tab.process = window.process_runner.start_workflow_process(
            optuna_mass_calibration_argv(config_file, options),
            toolbar.play_action,
            toolbar.stop_action,
            cwd=Path.cwd(),
            workflow="calibration",
            actions=["mass"],
            config_path=Path(config_file),
        )
        return tab.process

    run_calibration.__wrapped__ = run
    tab.run_calibration = run_calibration

    refresh = window.refresh_sidebar_progress

    def refresh_sidebar_progress(*args, **kwargs):
        result = refresh(*args, **kwargs)
        state = getattr(section.get_row("mass"), "_completion_state", "none")
        section.set_item_state(OPTUNA_MASS_ROW, state)
        return result

    refresh_sidebar_progress.__wrapped__ = refresh
    window.refresh_sidebar_progress = refresh_sidebar_progress


def _update_group_headers(section) -> None:
    """Hide the group headers of a sidebar section whose rows are all hidden."""

    layout = section.items_layout
    header = None
    shown = False
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if isinstance(widget, GroupHeaderLabel):
            if header is not None:
                header.setVisible(shown)
            header, shown = widget, False
        elif widget is not None and not widget.isHidden():
            shown = True
    if header is not None:
        header.setVisible(shown)


class _NativeSidebar(Sidebar):
    """Display merged categories while preserving the original action keys."""

    def __init__(self, categories_data: dict, setup_rows: dict, aliases=None) -> None:
        self._setup_rows = setup_rows
        self._logical_rows = {"setup": setup_rows, **(aliases or {})}
        self._display_rows = {
            logical: (category, key)
            for category, rows in self._logical_rows.items()
            for key, logical in rows.items()
        }
        # The old advanced Sources route still opens its persistent form,
        # while the sidebar now has one Jobs entry and one shared-folder menu.
        self._display_rows[("jobs", "sources")] = ("jobs", "jobs:overview")
        super().__init__(categories_data)
        for category, rows in self._logical_rows.items():
            for display_id, (action, item_id) in rows.items():
                row = self._sections[category].get_row(display_id)
                row.action = action
                row.checkbox_id = item_id
        for header in self._sections["setup"].findChildren(GroupHeaderLabel):
            if not header.text():
                header.hide()

    def display_selection(self, action: str, item_id: str) -> tuple[str, str]:
        if (action, item_id) in {("preprocessing", "depth_measurements"), ("setup", "depth")}:
            action, item_id = "geometry_facies", "rig_frame"
        if (action, item_id) in {("phase_model", "overflow"), ("phase_model", "mass_from_phase")}:
            return "calibration", "mass"
        return self._display_rows.get((action, item_id), (action, item_id))

    def _logical_selection(self, action: str, item_id: str) -> tuple[str, str]:
        return self._logical_rows.get(action, {}).get(item_id, (action, item_id))

    def _on_selection_changed(self, action: str, checkbox_id: str) -> None:
        # Let Sidebar maintain one visual selection, then emit the original keys.
        row = self._sections[action].get_row(checkbox_id)
        if row is None or not row.isEnabled():
            return
        with QSignalBlocker(self):
            super()._on_selection_changed(action, checkbox_id)
        if self._selected_row is not None:
            self.selection_changed.emit(*self._logical_selection(action, checkbox_id))

    def select(self, action: str, checkbox_id: str) -> None:
        if (action, checkbox_id) in {("preprocessing", "depth_measurements"), ("setup", "depth")}:
            self.selection_changed.emit("setup", "depth")
            return
        if (action, checkbox_id) == ("jobs", "sources"):
            super().select(*self.display_selection(action, checkbox_id))
            self.selection_changed.emit(action, checkbox_id)
            return
        super().select(*self.display_selection(action, checkbox_id))

    def get_selected(self) -> tuple[str, str] | None:
        selected = super().get_selected()
        return self._logical_selection(*selected) if selected is not None else None

    def set_item_state(self, action: str, checkbox_id: str, state: str) -> None:
        if (action, checkbox_id) in {("preprocessing", "depth_measurements"), ("setup", "depth")}:
            return  # Depth is an embedded section, not the whole Rig Geometry status.
        action, checkbox_id = self.display_selection(action, checkbox_id)
        super().set_item_state(action, checkbox_id, state)


class NativeFluidFlowerMainWindow(MainWindow):
    """FluidFlower pages alongside DarSIA's original sidebar categories."""

    # One ordered list owns the navigation; original rows come from their managers.
    SIDEBAR_CATEGORIES = (
        ("start", "Start", "fa5s.home", (("Series", "series"),)),
        ("setup", "Setup", "fa5s.cogs", None),
        ("calibration", "Calibration", "fa5s.balance-scale", None),
        ("analysis", "Mass analysis", "fa5s.chart-line", None),
        ("comparison", "Comparisons", "fa5s.exchange-alt", None),
        ("export", "Export results", "fa5s.file-export", None),
        ("jobs", "Jobs", "fa5s.tasks", None),
        ("helper", "Helper", "fa5s.life-ring", None),
        ("utils", "Utils", "fa5s.toolbox", None),
        ("legacy", "Legacy workflow", "fa5s.history", (("Open workflow", "workspace"),)),
    )

    SETUP_WORKFLOW = (
        (
            "Protocols",
            "preprocessing",
            "protocols",
            "Read the run protocol and produce the injection and timing records.",
        ),
        (
            "Overflow data",
            "overflow_data",
            "runs",
            "Show each run's overflow source (none, scale images or table), file and status as the series records them; opening this page writes nothing.",
        ),
        (
            "Image selection",
            "image_selection",
            "plan",
            "Choose images per run for review and analysis; edit saved protocol-based choices in series.toml without starting a calculation.",
        ),
        (
            "Rig Geometry",
            "geometry_facies",
            "rig_frame",
            "Define reference geometry, injection points and depth; curvature correction is optional. Rig corners are preferred; other measured quadrilaterals are supported.",
        ),
        (
            "Layers and facies",
            "geometry_facies",
            "layers",
            "Edit rig geometry, layers, facies and saved map versions on the reference photograph.",
        ),
        (
            "Sand map in all runs",
            "geometry_facies",
            "export",
            "Transfer the series' sand map to every run, review and approve it on each run's own photograph, check and adopt it. Finish setup applies it to the runs.",
        ),
        (
            "Image correction",
            "phase_model",
            "normalisation",
            "Choose lighting normalisation and ColorChecker colour correction, applied to the full photograph before the common crop. Crop and alignment come from Setup geometry; saving choices starts no calculation.",
        ),
        (
            "Colour paths",
            "calibration",
            "color",
            "Choose representative experiment images for optical colour paths; Finish setup performs the calculation after image corrections.",
        ),
        (
            "Finish setup",
            "setup",
            "finish",
            "Run all saved Setup steps, including image correction and colour paths. Model training and phase maps belong to Calibration.",
        ),
        (
            "Build run model",
            "setup",
            "rig",
            "Combine the configured labels, facies and corrections into the rig geometry.",
        ),
        (
            "Full setup",
            "setup",
            "all",
            "Run all original DarSIA Setup steps using the selected run configuration.",
        ),
    )

    def __init__(self, *, initial_series: str | None = None) -> None:
        super().__init__()
        use_environment_python(self.process_runner)
        clear_config_placeholder(self)
        guard_config_file(self)
        guard_published_dataset(self)
        settle_settings_before_sync(self)
        from .settings_save_guard import guard_lossless_settings_save

        guard_lossless_settings_save(self)
        from .settings_storage_aliases import display_storage_aliases

        display_storage_aliases(self)
        self.dataset_analysis_page = None
        self._config_series: str | None = None
        self._current_category = "start"
        self._last_sidebar_selection = ("start", "series")
        self._selecting_sidebar = False
        self._method_root = None
        self._method_manifest_bytes = None
        self._saved_method = MethodSelection()
        self._method_line_parts = ["Method not chosen"]
        # True only while All runs runs, after its one method confirmation (N141).
        self._all_runs_confirmed = False
        self._method_panel_seed = MethodSelection()
        self._pending_method_action = None
        self._scope_base_help = {}
        self.scope_context = None
        self.current_row_scope = None
        series = tuple(configured_series())
        from darsia.fluidflower.published_dataset.dataset_series import protect_datasets

        # A published dataset is read-only in this process from the start.
        protect_datasets(root for _key, _label, root in series)
        initial_series = initial_series or (series[0][0] if series else "")
        if initial_series and initial_series not in {
            key for key, _label, _root in series
        }:
            raise ValueError(f"Unknown experimental series: {initial_series}")
        self.home_page = NativeHomePage(series=series, initial_series=initial_series)
        self._legacy_windows: dict[str, QWidget] = {}
        self.legacy_workspace = None

        # Retain MainWindow's splitter, menus, toolbar, log and streaming handles.
        right_layout = self.settings_scroll_area.parentWidget().layout()
        right_layout.removeWidget(self.settings_scroll_area)
        self.method_badge = QLabel("Method: not chosen")
        self.method_badge.setTextFormat(Qt.TextFormat.PlainText)
        self.method_badge.setWordWrap(True)
        right_layout.addWidget(self.method_badge)
        self.series_header = QWidget()
        top = QHBoxLayout(self.series_header)
        top.setContentsMargins(0, 0, 0, 0)
        self.setup_series_label = QLabel()
        top.addWidget(self.setup_series_label)
        top.addSpacing(16)
        self.run_label = QLabel("Run")
        top.addWidget(self.run_label)
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(90)
        self.run_combo.setPlaceholderText("Choose a run")
        self.run_combo.setToolTip(
            "Run whose configuration DarSIA's steps open: config/merged/<run>.toml "
            "when the series has one, else config/runs/<run>.toml"
        )
        top.addWidget(self.run_combo)
        self.run_path = QLabel()
        self.run_path.setObjectName("secondaryText")
        self.run_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        top.addWidget(self.run_path)
        top.addStretch(1)
        right_layout.addWidget(self.series_header)
        self.method_panel = SeriesMethodPanel(method_choice_only=True)
        self.method_scope_label = QLabel()
        self.method_scope_label.setTextFormat(Qt.TextFormat.PlainText)
        self.method_scope_label.setWordWrap(True)
        self.method_panel.layout().insertWidget(1, self.method_scope_label)
        # Keep the choice together at the top of the page; extra height stays below it.
        self.method_panel.layout().addStretch(1)
        self.method_panel.hide()
        self.method_panel.use_requested.connect(self._save_method_choice)
        self.method_panel.unlock_requested.connect(self._unlock_method_choice)
        self.method_panel.calls.changed.connect(self._method_choice_reader_changed)
        self.method_panel.calls.action_finished.connect(self._method_choice_finished)
        self.analysis_method_badge = QLabel("Method: not chosen")
        self.analysis_execution_note = QLabel(
            "Comparisons compare existing results. Events and Wasserstein use the selected "
            "configuration and run on this computer. To create a Wasserstein job for local or shared workers, "
            "open Comparisons → Wasserstein comparison."
        )
        self.analysis_execution_note.setWordWrap(True)
        self.analysis_execution_note.setTextFormat(Qt.TextFormat.PlainText)
        self.analysis_execution_note.hide()
        right_layout.addWidget(self.analysis_execution_note)
        self.analysis_method_badge.setTextFormat(Qt.TextFormat.PlainText)
        self.analysis_method_badge.setWordWrap(True)
        self.analysis_method_badge.hide()
        right_layout.addWidget(self.analysis_method_badge)
        from darsia.fluidflower.frozen_reference_line import FrozenReferenceLine

        self.frozen_reference_line = FrozenReferenceLine()
        right_layout.addWidget(self.frozen_reference_line)
        self.scope_label = QLabel()
        self.scope_label.setTextFormat(Qt.TextFormat.PlainText)
        self.scope_label.setWordWrap(True)
        self.scope_label.hide()
        right_layout.addWidget(self.scope_label)
        self.rig_label = QLabel()
        self.rig_label.setTextFormat(Qt.TextFormat.PlainText)
        self.rig_label.setWordWrap(True)
        self.rig_label.hide()
        right_layout.addWidget(self.rig_label)
        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(self.settings_scroll_area)
        self.content_stack.addWidget(self.home_page)
        self.content_stack.addWidget(self.method_panel)
        from darsia.fluidflower.native_mass_calibration_layout import NativeMassCalibrationDisplay

        self.mass_calibration_display = NativeMassCalibrationDisplay()
        self.mass_calibration_display.install_preparation(self)
        self.mass_calibration_display.install_common_actions(self)
        self.content_stack.addWidget(self.mass_calibration_display.phase_colour_paths)
        self._displayed_mass_method = None
        self.content_stack.addWidget(self.mass_calibration_display)
        self.content_stack.addWidget(self.mass_calibration_display.saved_calibration_page)
        from darsia.fluidflower.native_calibration_image_selection import CalibrationImageSelection

        self.calibration_image_selection = CalibrationImageSelection(self)
        self.calibration_image_selection.saved.connect(lambda _root: self._refresh_method_state(force=True))
        self.content_stack.addWidget(self.calibration_image_selection)
        self.overflow_page = self.mass_calibration_display.overflow_choice
        self.overflow_page.mode_changed.connect(self._set_overflow_diagnostic)
        self.overflow_page.comparison_requested.connect(self._show_saved_overflow_comparison)
        from darsia.fluidflower.overflow_data_page import OverflowDataPage

        self.overflow_data_page = OverflowDataPage()
        self.content_stack.addWidget(self.overflow_data_page)
        self.geometry_workspace = SeriesGeometryWorkspace(
            series=series, initial_series=initial_series
        )
        self.geometry_workspace.series_label.hide()
        self.crop_workflow = NativeCropWorkflow(self)
        self.coloured_map_workflow = NativeColouredMapWorkflow(self)
        self.coloured_map_workflow.install_workspace(self.geometry_workspace)
        show_geometry_page = self.geometry_workspace._show_page

        def show_geometry_page_with_import():
            result = show_geometry_page()
            self.coloured_map_workflow.install(self.geometry_workspace.simple_view)
            if getattr(self, 'selected_action', '') == 'geometry_facies':
                self._set_geometry_workflow_scope(self.selected_checkbox_id)
            elif getattr(self, 'selected_action', '') == 'analysis_boxes':
                self._set_geometry_workflow_scope('analysis_boxes')
            return result

        self.geometry_workspace._show_page = show_geometry_page_with_import
        discard_geometry_page = self.geometry_workspace._discard_page

        def discard_geometry_page_with_curvature():
            # The geometry view is replaced on series/revision changes. Keep
            # its native-owned editor alive rather than deleting it as a child.
            curvature = getattr(self, "curvature_workflow", None)
            if curvature is not None:
                curvature.set_active(False)
                curvature.widget.setParent(self)
                curvature.widget.hide()
            depth = getattr(self, "depth_workflow", None)
            if depth is not None:
                depth.set_embedded_active(False)
                depth.panel.setParent(self)
                depth.panel.hide()
            return discard_geometry_page()

        self.geometry_workspace._discard_page = discard_geometry_page_with_curvature
        self.coloured_map_workflow.install(self.geometry_workspace.simple_view)
        show_export = self.geometry_workspace._show_export_dialog

        def export_finished(export):
            show_export(export)
            self.crop_workflow.attach_export_dialog(
                self.geometry_workspace.export_dialog, export
            )
            self._update_painted_geometry_status()

        self.geometry_workspace._show_export_dialog = export_finished
        self.content_stack.addWidget(self.geometry_workspace)
        from darsia.fluidflower.series_native_context import SeriesNativeContext

        self._series_native_context = SeriesNativeContext()
        self._series_snapshot = None
        self._series_loading = False
        self._series_load_error = ''
        self._pending_open_series = None
        self._pending_new_series_setup = None
        self._pending_series_navigation = None
        from .series_selection_loader import SeriesSelectionLoader
        self._series_loader = SeriesSelectionLoader(self)
        self._series_loader.ready.connect(self._series_metadata_ready)
        self.phase_model_workspace = PhaseModelWorkspace(
            series=series, initial_series=initial_series,
            native_context_binder=self._prepared_native_context,
            defer_native_context=True,
        )
        self.phase_model_workspace.calibration_sidebar.hide()
        self.phase_model_workspace.native_step_options = {
            "method_panel": self.method_panel,
            "method_use": self._save_method_choice,
            "model_for_calibration": self._select_phase_model_for_calibration,
            "phase_review": self._open_existing_phase_review,
            "run_provider": lambda: self.run_combo.currentText() if self.run_combo.currentIndex() >= 0 else "",
        }
        self.content_stack.addWidget(self.phase_model_workspace)
        self.depth_workflow = NativeDepthWorkflow(self)
        self.curvature_workflow = NativeCurvatureWorkflow(self)
        self.content_stack.addWidget(self.depth_workflow.panel)
        self.image_selection_workflow = NativeImageSelectionWorkflow(self)
        self.content_stack.addWidget(self.image_selection_workflow.panel)
        from .production_image_selection_ui import ProductionImageSelectionPage
        self.production_image_selection = ProductionImageSelectionPage(self)
        self.content_stack.addWidget(self.production_image_selection.panel)
        from .events_page import EventsPage
        self.events_page = EventsPage()
        self.content_stack.addWidget(self.events_page)
        self.rig_context = NativeRigContext(self)
        self.mass_analysis_page: QWidget | None = None
        self.mass_analysis_placeholder = QLabel(
            "Choose an experimental series before mass analysis."
        )
        self.content_stack.addWidget(self.mass_analysis_placeholder)
        self.jobs_workspace = None
        from .native_finish_setup import FinishSetupPage
        self.finish_setup_page = FinishSetupPage(self)
        self.content_stack.addWidget(self.finish_setup_page)
        right_layout.addWidget(self.content_stack, 1)

        sidebar_data = {}
        setup_rows = {}
        aliases = {"calibration": {}, "analysis": {}, "comparison": {}, "export": {}, "jobs": {}}
        # Method key of each group, in header order, so grouping never reads display text.
        self._method_group_keys = {}
        for action, label, icon, rows in self.SIDEBAR_CATEGORIES:
            if action == "setup":
                display_items = []
                for text, source, item_id, help_text in self.SETUP_WORKFLOW:
                    display_id = f"{source}:{item_id}"
                    setup_rows[display_id] = (source, item_id)
                    display_items.append((text, display_id, "fa5s.circle", help_text))
                groups = [("", display_items)]
            elif action == "calibration":
                from .native_common_workflow import CALIBRATION_ROWS
                original = {item_id: (text, item_id, icon, help_text)
                    for _group, items in self.calibration_tab.sidebar_items()
                    for text, item_id, icon, help_text in items}
                shared = []
                for title, source, item_id, help_text in CALIBRATION_ROWS:
                    if source == "calibration":
                        shared.append((title, item_id, original.get(item_id, (None, None, "fa5s.archive"))[2], help_text))
                    else:
                        shared.append(_native_row(aliases, action, title, source, item_id, help_text))
                # Keep old settings/actions addressable without duplicate menu rows.
                shared.extend(original[key] for key in ("default_mass", "reset", "delete"))
                groups = [("", shared)]
            elif action == "analysis":
                originals = {row[1]: row for _group, items in self.analysis_tab.sidebar_items() for row in items}
                common_rows = [
                    _native_row(aliases, action, "Image selection", "image_selection", "production",
                                "Choose the images per run that mass analysis computes: one shared time grid, "
                                "separate from the calibration images. Opening this page starts no calculation."),
                    _native_row(aliases, action, "New analysis", "jobs", "new_analysis",
                                "Open the selected calculation model's existing production settings; opening starts nothing."),
                    originals["mass"],
                    _native_row(aliases, action, "Results", "jobs", "production_results",
                                "Open a finished production job read-only, including frames, maps and mass curves."),
                    _native_row(aliases, action, "Phase review window", "mass_analysis", "review",
                                "Open the existing prepared model-correction review window."),
                ]
                groups = [("", common_rows)]
            elif action == "comparison":
                originals = {row[1]: row for _group, items in self.analysis_tab.sidebar_items() for row in items}
                special = [_native_row(aliases, action, title, "analysis", item_id, help_text, icon)
                    for title, item_id, icon, help_text in
                    (originals[key] for key in ("fingers", "segmentation", "thresholding", "volume"))]
                comparison = [row for _group, items in self.action_dispatch[action].sidebar_items() for row in items]
                groups = [("", [_native_row(
                    aliases, action, "Analysis boxes", "analysis_boxes", "edit",
                    "Define optional measured boxes before or after production. "
                    "Recalculate their mass from saved mass maps without rebuilding images or calibrating again."),
                    *special,
                    _native_row(aliases, action, "Events", "events", "question",
                                "Find when something happens in an area, for every run of a finished "
                                "mass analysis, for example when dissolved CO2 is first detected in a box."),
                    *comparison,
                    _native_row(aliases, action, "Wasserstein comparison", "jobs", "wasserstein",
                                "Compare existing mass results between runs as a job, then follow its local or shared workers in Jobs."),
                    ])]
            elif action == "export":
                originals = {row[1]: row for _group, items in self.analysis_tab.sidebar_items() for row in items}
                groups = [("", [
                    _native_row(aliases, action, "Results and export", "mass_analysis", "frozen",
                                "Browse finished model/calibration results per run and export them; opening starts nothing."),
                    _native_row(aliases, action, "Export prepared images", "analysis", "cropping",
                                "Export processed/cropped images through DarSIA's original image reader.",
                                originals["cropping"][2]),
                ])]
            elif action == "jobs":
                groups = [("", [
                    _native_row(aliases, action, "Jobs and workers", "jobs", "overview",
                                "Follow registered jobs, choose this computer or a registered shared-compute machine, and choose the workers to start on that machine."),
                ])]
            else:
                groups = (
                    self.action_dispatch[action].sidebar_items()
                    if rows is None
                    else [
                        (
                            label,
                            [
                                (text, row_id, "fa5s.circle", text)
                                for text, row_id in rows
                            ],
                        )
                    ]
                )
            sidebar_data[action] = (label, icon, groups)
        old_sidebar = self.sidebar
        self.sidebar = _NativeSidebar(sidebar_data, setup_rows, aliases)
        self.sidebar.setMinimumWidth(180)
        self.sidebar.setMaximumWidth(600)
        old_sidebar.parentWidget().layout().replaceWidget(old_sidebar, self.sidebar)
        # Bound the splitter pane as well as its child. Otherwise the original
        # proportional stretch leaves empty space beside a max-width sidebar.
        sidebar_pane = self.root_splitter.widget(0)
        margins = sidebar_pane.layout().contentsMargins()
        sidebar_pane.setMaximumWidth(self.sidebar.maximumWidth() + margins.left() + margins.right())
        self.root_splitter.setStretchFactor(0, 0)
        self.root_splitter.setStretchFactor(1, 1)
        self._native_sidebar_initialized = False
        QTimer.singleShot(0, self, self._initialize_sidebar_width)
        old_sidebar.hide()
        old_sidebar.deleteLater()
        self.sidebar.selection_changed.connect(self._on_sidebar_selection)
        self.sidebar.expanded_changed.connect(self._on_sidebar_expanded)
        # The helper captures its calibration section, so install it after replacement.
        add_automatic_mass_calibration(self)
        section = self.sidebar._sections["calibration"]
        mass_row = section.get_row("mass")
        mass_row.label = "Mass calibration"
        mass_row._text_label.setText(mass_row.label)
        original_calibration = self.calibration_tab.run_calibration

        def run_calibration(*args, **kwargs):
            if (self.selected_checkbox_id in {"mass", OPTUNA_MASS_ROW, "default_mass"}
                    and not (getattr(self, "_original_mass_controls", False)
                             and self._saved_method.active == "colour_path")):
                self.print_log(
                    "Mass calibration is read-only here. Manual and automatic calibration runs will be available on Jobs."
                )
                return None
            return original_calibration(*args, **kwargs)

        run_calibration.__wrapped__ = original_calibration
        self.calibration_tab.run_calibration = run_calibration
        self._capture_method_groups()
        # The Phase review window is a button on Phase model > Review and label
        # (N141). Keep its old Analysis route for external callers, out of the UI.
        self.sidebar._sections["analysis"].get_row("mass_analysis:review").hide()
        self._install_method_analysis_guard()
        from darsia.fluidflower.native_analysis_sections import NativeAnalysisSections

        self.analysis_sections = NativeAnalysisSections(self)
        from darsia.fluidflower.native_protocol_sections import NativeProtocolSections
        self.protocol_sections = NativeProtocolSections(self)
        self.toolbar_builder.play_action.changed.connect(self._guard_custom_play)
        from darsia.fluidflower.workflow_status_ui import NativeWorkflowStatus

        self.workflow_status = NativeWorkflowStatus(self)
        self.workflow_status.prepared_context_provider = self._prepared_series_status
        self.rig_context.prepared_display_provider = lambda: (
            self._current_series_snapshot()['rigs'].get(self.run_combo.currentText(),
                self._current_series_snapshot()['rigs'][''])
            if self._current_series_snapshot() is not None else None)
        from darsia.fluidflower.all_runs import NativeAllRuns

        self.all_runs = NativeAllRuns(self)
        self.content_stack.currentChanged.connect(lambda _index: self.all_runs.refresh())

        self.run_combo.activated.connect(self._run_chosen)
        self.run_combo.currentIndexChanged.connect(self._update_setup_action)
        self.run_combo.currentIndexChanged.connect(self._refresh_scope_display)
        self.run_combo.currentIndexChanged.connect(self._refresh_phase_review_requirements)
        from darsia.fluidflower.reference_open_guard import ReferenceOpenGuard

        self.reference_open_guard = ReferenceOpenGuard(self)
        self.phase_model_workspace.series_selector.currentIndexChanged.connect(
            self._sync_series_display
        )
        self.home_page.new_requested.connect(self._open_preprocessing_for_selected_run)
        self.home_page.new_series_button.clicked.connect(self._new_series)
        self.home_page.add_existing_button.clicked.connect(self._add_existing_series)
        self.home_page.folders_button.clicked.connect(self._show_series_folders)
        self.home_page.remove_series_button.clicked.connect(self._remove_series_from_start)
        self.home_page.refresh_series_button.clicked.connect(
            lambda: self._refresh_series(show_errors=True)
        )
        self.home_page.update_run_sources_requested.connect(self._show_setup_for_path)
        self.home_page.open_requested.connect(self._open_series_from_start)
        self.home_page.recent_requested.connect(self._open_series_from_start)
        self._sync_series_display()
        self._select_row(("start", "series"))
        self.status_header = NativeStatusHeader(self)
        from darsia.fluidflower.running_pressure_view import ProtocolsRunningPressure

        # Setup > Protocols: a read-only view of every run's running pressure.
        self.running_pressure = ProtocolsRunningPressure(self)
        from darsia.fluidflower.protocol_workbooks_ui import install_protocol_workbooks

        protocol_workbooks = install_protocol_workbooks(self)
        protocol_workbooks.activated.connect(self._open_series_from_start)
        protocol_workbooks.archived.connect(self._open_series_from_start)

    def _initialize_sidebar_width(self):
        """Choose the initial width after layout; later user resizing is retained."""
        if self._native_sidebar_initialized:
            return
        self._native_sidebar_initialized = True
        pane = self.root_splitter.widget(0)
        margins = pane.layout().contentsMargins()
        horizontal_margins = margins.left() + margins.right()
        minimum = 360 + horizontal_margins
        maximum = pane.maximumWidth()
        try:
            saved = int(QSettings().value(self._SIDEBAR_WIDTH_SETTINGS_KEY, 0))
        except (TypeError, ValueError):
            saved = 0
        width = saved if minimum <= saved <= maximum else min(
            maximum, max(minimum, round(self.root_splitter.width() * 0.24) + horizontal_margins)
        )
        available = self.root_splitter.width() - self.root_splitter.handleWidth()
        # A zero request collapses the work pane persistently, including when
        # the first layout has not yet expanded the splitter to the window.
        self.root_splitter.setSizes([width, max(1, available - width)])

    def refresh_sidebar_progress(self):
        """Native metadata status; preserve unrelated shallow completion dots."""
        adapter = getattr(self, "workflow_status", None)
        if adapter is None:
            return super().refresh_sidebar_progress()
        # A load, a save or a change seen elsewhere: read the rows' status anew.
        adapter.invalidate()
        adapter.refresh()
        if not self.config_file or not Path(self.config_file).is_file():
            return
        from darsia.presets.workflows.results_folder import (
            suggested_workflow_results_folder,
        )

        for category, manager in (
            ("comparison", self.comparison_tab),
            ("utils", self.utils_tab),
        ):
            for _group, items in manager.sidebar_items():
                for _label, item_id, _icon, _help in items:
                    done = False
                    try:
                        folder = suggested_workflow_results_folder(
                            category,
                            Path(self.config_file),
                            [self.action_label_for(category, item_id)],
                        )
                        if folder is not None:
                            with os.scandir(folder) as listing:
                                done = next(listing, None) is not None
                    except (OSError, ValueError, TypeError):
                        pass
                    self.sidebar.set_item_state(
                        category, item_id, "done" if done else "none"
                    )

    def _refresh_scope_display(self, *_args) -> None:
        """Render one read-only context across row help, header and selector."""
        legacy = getattr(self, "legacy_workspace", None)
        if legacy is not None and legacy.active:
            legacy.hide_series_context()
            return
        if not all(
            hasattr(self, name)
            for name in (
                "scope_label",
                "phase_model_workspace",
                "sidebar",
                "run_combo",
                "method_scope_label",
                "_scope_base_help",
            )
        ):
            return
        from darsia.fluidflower.row_scope import (
            load_scope_context,
            row_scope,
            series_scope,
        )

        workspace = self.phase_model_workspace
        if getattr(self, '_series_loading', False) or getattr(self, '_series_load_error', ''):
            return
        series_id = workspace.series_id
        root = workspace.series_roots.get(series_id)
        run_id = (
            self.run_combo.currentText() if self.run_combo.currentIndex() >= 0 else ""
        )
        path = self.run_combo.currentData() or None
        values = None
        if self.config_file:
            # Original actions use the actual open file. An unrelated File >
            # Open Config must never be relabelled as a registered series run.
            path = self.config_file
            values = self.config_dict
            loaded_index = self._run_index(path)
            if self._config_series == series_id and loaded_index >= 0:
                run_id = self.run_combo.itemText(loaded_index)
        snapshot = self._current_series_snapshot()
        context = (replace(snapshot['scopes'].get(run_id, snapshot['scopes']['']),
            loaded=values if values is not None else snapshot['scopes'].get(run_id, snapshot['scopes']['']).loaded,
            loaded_path=Path(path) if path else None) if snapshot is not None else load_scope_context(
            root,
            series_id,
            run_id,
            path,
            values,
            series_label=workspace.series_selector.currentText() or series_id,
        ))
        self.scope_context = context
        for category in ("setup", "calibration", "analysis", "comparison", "export", "jobs"):
            section = self.sidebar._sections.get(category)
            if section is None:
                continue
            for row in section._rows.values():
                base = self._scope_base_help.setdefault(id(row), row.help_text)
                scope = row_scope(context, *self._effective_workflow_selection((row.action, row.checkbox_id)))
                if scope is None:
                    continue
                text = base + "\n" + scope.line
                if scope.note:
                    text += "\n" + scope.note
                row.help_text = text
                row.setToolTip(text)
                popup = getattr(row, "_help_popup", None)
                if popup is not None:
                    popup._help_text = text
                    popup._help_label.setTextFormat(Qt.TextFormat.PlainText)
                    popup._help_label.setText(text)
                    popup.adjustSize()
        whole_series = series_scope(context)
        self.method_scope_label.setText(whole_series.line)
        self.method_scope_label.setToolTip(whole_series.note)
        category = row_category(self._current_category)
        selected = self.sidebar.get_selected()
        scope = (
            row_scope(context, *self._effective_workflow_selection(selected))
            if selected is not None and category in {"setup", "calibration", "analysis", "comparison", "export"}
            else None
        )
        self.current_row_scope = scope
        if getattr(self, "workflow_status", None) is not None:
            self.workflow_status.refresh(**({'context': snapshot['statuses'].get(run_id, snapshot['statuses'][''])[0],
                'states': snapshot['statuses'].get(run_id, snapshot['statuses'][''])[1]}
                if snapshot is not None else {}))
        self.scope_label.setText(scope.line if scope is not None else "")
        self.scope_label.setToolTip(scope.note if scope is not None else "")
        self.scope_label.setVisible(
            scope is not None and category in {"setup", "calibration", "analysis", "comparison", "export"}
        )
        # A series page hides the run selector; the header line names the whole
        # series instead (N141, layout rule 5).
        self.run_combo.setEnabled(scope is None or scope.run_selector_enabled)
        if getattr(self, "depth_workflow", None) is not None:
            self.depth_workflow.keep_selection_disabled_if_running()
        if getattr(self, "rig_context", None) is not None:
            self.rig_context.refresh(**({'display': snapshot['rigs'].get(run_id, snapshot['rigs'][''])}
                if snapshot is not None else {}))
        if getattr(self, "status_header", None) is not None:
            self.status_header.refresh()
        if getattr(self, "all_runs", None) is not None:
            self.all_runs.refresh()

    def _capture_method_groups(self) -> None:
        """Retain one fixed row order, including hidden compatibility actions."""
        self._method_row_labels = getattr(self, "_method_row_labels", {})
        self._method_row_help = getattr(self, "_method_row_help", {})
        self._method_group_order = {}
        self._method_first_rows = {}
        self._method_group_widgets = {}
        for category in ("calibration", "analysis", "comparison", "export"):
            section = self.sidebar._sections[category]
            widgets = [section.items_layout.itemAt(i).widget() for i in range(section.items_layout.count())]
            rows = {}
            for widget in widgets:
                if isinstance(widget, GroupHeaderLabel):
                    widget.hide()
                if isinstance(widget, SidebarRow):
                    _category, key = self.sidebar.display_selection(widget.action, widget.checkbox_id)
                    if (widget.action, widget.checkbox_id) == ("analysis", "mass"):
                        key = "mass"  # Hidden original settings owner, not the common row.
                    rows[key] = widget
                    self._method_row_labels.setdefault(id(widget), widget.label)
                    self._method_row_help.setdefault(id(widget), widget.help_text)
            section._rows = rows
            self._method_group_widgets[category] = {"common": widgets}

    def _method_series_root(self) -> Path | None:
        workspace = self.phase_model_workspace
        root = workspace.series_roots.get(workspace.series_id)
        return Path(root) if root is not None else None

    def _show_mass_calibration_values(self, method: str) -> None:
        """Read only the explicitly referenced saved calibration, never fit."""
        from darsia.fluidflower.mass_calibration_display import (
            read_mass_calibration_display,
            read_overflow_use,
        )
        from darsia.fluidflower.pressure_source import iter_series_pressure_sources

        selection = self._saved_method
        reference = (
            selection.mass_calibration
            if method == "phase_model"
            else selection.colour_calibration
        )
        root = self._method_series_root()
        data = read_mass_calibration_display(method, reference, series_root=root)
        self.mass_calibration_display.set_display(data)
        self.mass_calibration_display.restore_calibration_preparation()
        self.mass_calibration_display.follow_pressure_sources(
            None if root is None else iter_series_pressure_sources(
                root, method=method, calibration=reference))
        self.mass_calibration_display.set_refit_series(root if method == "phase_model" else None)
        self.mass_calibration_display.preparation.set_series(root if method == "phase_model" else None)
        self.mass_calibration_display.set_calibration_context(method, root)
        if method == "phase_model":
            self.overflow_page.set_use(*read_overflow_use(reference, root))
            saved = getattr(self.mass_analysis_page, "saved_results", None)
            self.overflow_page.set_comparison_available(bool((getattr(saved, "result", None) or {}).get("overflow_curve")))
        self._displayed_mass_method = method
        self.content_stack.setCurrentWidget(self.mass_calibration_display)

    def _show_saved_calibration(self):
        from .mass_calibration_display import read_mass_calibration_display
        choice = self._saved_method
        reference = choice.mass_calibration if choice.active == "phase_model" else choice.colour_calibration
        self.mass_calibration_display.set_display(read_mass_calibration_display(
            choice.active, reference, series_root=self._method_series_root()))
        self.mass_calibration_display.mode_notice.setText(
            "Read-only saved calibration. New calibration choices do not change this saved result.")
        self.content_stack.setCurrentWidget(self.mass_calibration_display.saved_calibration_page)

    def _refresh_method_state(self, *, force: bool = False) -> None:
        if not hasattr(self, "method_panel") or not hasattr(
            self, "phase_model_workspace"
        ):
            return
        if getattr(self, '_series_loading', False) or getattr(self, '_series_load_error', ''):
            return
        root = self._method_series_root()
        snapshot = self._current_series_snapshot() if not force else None
        try:
            raw = snapshot['raw'] if snapshot is not None else ((root / "series.toml").read_bytes() if root is not None else None)
        except OSError:
            raw = None
        changed = root != self._method_root or raw != self._method_manifest_bytes
        pristine = self.method_panel.selection() == self._method_panel_seed
        if changed or force:
            self._method_root = root
            self._method_manifest_bytes = raw
            self._saved_method = (
                snapshot['method'] if snapshot is not None else (read_series_method(root) if root is not None else MethodSelection())
            )
        if changed or force or pristine:
            self.method_panel.set_selection(
                self._saved_method,
                available=root is not None,
            )
            self._method_panel_seed = self.method_panel.selection()
        self._method_approval = None
        self._method_approval_error = ""
        if raw is not None:
            from darsia.fluidflower.series_calibration_approval import _record

            try:
                self._method_approval = _record(tomllib.loads(raw.decode("utf-8-sig")))
            except (ValueError, TypeError, KeyError) as error:
                self._method_approval_error = str(error)
        self.method_panel.set_approval_context(
            root, self._method_approval, self._method_approval_error,
            manifest_bytes=raw, force=force,
        )
        if self._method_approval is not None and self._method_approval["locked"]:
            self._method_approval_error = self._method_approval_error or self.method_panel.calls.error
        self.frozen_reference_line.set_reference(root, raw)
        choice = self._saved_method
        if choice.active == "phase_model":
            from darsia.fluidflower.model_index_names import model_reference_label

            model_label = snapshot['model_label'] if snapshot is not None else model_reference_label(root, choice.model_version)
            text = f"Method: {METHOD_NAMES['phase_model'].lower()} · model {model_label} · mass calibration {choice.mass_calibration or 'not referenced'}"
            parts = [METHOD_NAMES["phase_model"], f"model {model_label}",
                     f"calibration {choice.mass_calibration or 'not referenced'}"]
        elif choice.active == "colour_path":
            text = f"Method: {METHOD_NAMES['colour_path'].lower()} · calibration {choice.colour_calibration or 'not referenced'}"
            parts = [METHOD_NAMES["colour_path"], f"calibration {choice.colour_calibration or 'not referenced'}"]
        else:
            text = "Method: not chosen"
            parts = ["Method not chosen"]
        if self._method_approval is not None:
            binding = self._method_approval["binding"]
            label = "Titration" if binding.get("chemistry_model") == "titration" else "Not reported"
            lock = "approved and locked" if self._method_approval["locked"] else "unlocked"
            scope = "chemistry" if self._method_approval["locked"] else "last approved chemistry"
            text += f" · {scope} {label} · {lock}"
            parts.extend([f"{scope} {label}", lock])
        if self._method_approval_error:
            text += " · " + self._method_approval_error
            parts.append(self._method_approval_error)
        missing = missing_method_references(choice)
        if missing:
            text += " · missing " + " and ".join(missing)
            parts.append("missing " + " and ".join(missing))
        if choice.error:
            text += " · " + choice.error
            parts.append(choice.error)
        self._method_line_parts = parts
        for badge in (self.method_badge, self.analysis_method_badge):
            badge.setText(text)
            badge.setStyleSheet(
                "color: #b77900;"
                if not choice.active or missing or choice.error
                else ""
            )
        self._order_method_groups()
        if getattr(self, "protocol_sections", None) is not None:
            self.protocol_sections.update_method(choice.active)
        if (self.content_stack.currentWidget() is self.mass_calibration_display
                and not getattr(self, "_prepared_model_publication", False)):
            self.show_common_mass_calibration()
        elif self.content_stack.currentWidget() is self.mass_calibration_display.saved_calibration_page:
            self._show_saved_calibration()

    def _order_method_groups(self) -> None:
        """Update availability in place; method changes never move menu rows."""
        if not hasattr(self, "_method_group_widgets"):
            return
        from .native_common_workflow import availability, HIDDEN_COMPATIBILITY_ROWS
        for category in ("calibration", "analysis", "comparison", "export"):
            section = self.sidebar._sections[category]
            for row in section._rows.values():
                logical = (row.action, row.checkbox_id)
                enabled, reason = availability(logical, self._saved_method.active)
                row.setVisible(logical not in HIDDEN_COMPATIBILITY_ROWS)
                row.setEnabled(enabled)
                base = self._method_row_help[id(row)]
                row.help_text = base + ("\n" + reason if reason else "")
                self._scope_base_help[id(row)] = row.help_text
                row.setToolTip(row.help_text)
            self._method_group_order[category] = ("common",)
            self._method_first_rows[category] = next(
                key for key,row in section._rows.items() if not row.isHidden() and row.isEnabled())

    def _effective_workflow_selection(self, selection):
        if selection == ("calibration", "saved"):
            selection = ("calibration", "mass")
        if selection == ("calibration_controls", "interactive"):
            return "calibration", "mass"
        if selection == ("calibration", "color") and self._saved_method.active == "phase_model":
            return "phase_model", "colour_paths"
        if selection == ("calibration", "mass") and self._saved_method.active == "phase_model":
            return "phase_model", "mass_from_phase"
        if selection == ("jobs", "new_analysis") and self._saved_method.active == "colour_path":
            return "analysis", "mass"
        return selection

    def show_common_mass_calibration(self) -> None:
        """Use the saved model, never the transient editor choice or old route."""
        self._original_mass_controls = False
        self.selected_action, self.selected_checkbox_id = "calibration", "mass"
        self._show_mass_calibration_values(self._saved_method.active)

    def _show_original_mass_calibration_controls(self, item_id="mass") -> None:
        """Open original manual settings only in the saved colour-signal route."""
        if self._saved_method.active != "colour_path":
            return
        with QSignalBlocker(self.sidebar):
            self.sidebar.select("calibration", "mass")
        self._last_sidebar_selection = ("calibration", "mass")
        self._original_mass_controls = True
        self._current_category = "calibration"
        self.selected_action, self.selected_checkbox_id = "calibration", item_id
        self.content_stack.setCurrentWidget(self.settings_scroll_area)
        self._ensure_run_loaded()
        super()._on_sidebar_selection("calibration", item_id)
        self._refresh_scope_display()
        self._sync_wizard_action_state()
        self._update_run_action()

    def _set_method_category_visibility(self, action: str, item_id: str) -> None:
        category, _item = self.sidebar.display_selection(action, item_id)
        self.method_panel.setVisible((action, item_id) == ("method", "use_for_series"))
        self.analysis_method_badge.setVisible(category == "analysis")
        self.analysis_execution_note.setVisible(category == "comparison" and action == "comparison")
        self.frozen_reference_line.show_for_analysis(category == "analysis")
        self.method_badge.setVisible(category != "start")

    def _save_method_choice(self, selection: MethodSelection, *, prepared_model=False,
                            expected_bytes=None):
        root = self._method_series_root()
        if root is None or root != self._method_root:
            self.print_log("Choose a series before recording its method.")
            return
        if self.dataset_series_root() is not None:
            self.print_log(
                "This series is a published dataset and opens read-only; "
                "its method is not changed."
            )
            return
        calls = self.method_panel.calls
        original = self._method_manifest_bytes
        if prepared_model:
            context = (str(root), self._saved_method, original)
            if (expected_bytes != original or calls.context != context
                    or self.method_panel.selection() != self._saved_method
                    or selection.active != "phase_model" or selection.mass_calibration):
                return False
            calls.stop_pending()
            generation = calls.generation
            return calls.start_action("prepared_model", lambda: save_series_method(
                root, selection, expected_bytes=original, prepared_model=True,
                context_current=lambda: calls.context == context and calls.generation == generation), context)
        if self.method_panel.method_choice_only:
            from .series_method import save_series_calibration_method
            context = (str(root), selection, original)
            if calls.context != context:
                return False
            calls.stop_pending()
            generation = calls.generation
            return calls.start_action("calibration_method", lambda: save_series_calibration_method(
                root, selection.active, expected_bytes=original,
                context_current=lambda: calls.context == context and calls.generation == generation), context)
        context = (str(root), selection, original)
        if context != calls.context or (selection.active == "phase_model" and not calls.ready()):
            QMessageBox.warning(self, "Choice not ready", "Wait for the current selected files to be checked.")
            return
        binding = calls.binding

        def save():
            if selection.active == "phase_model":
                from darsia.fluidflower.series_calibration_approval import approve_series_method

                return approve_series_method(root, selection, expected_bytes=original,
                                             expected_binding=binding)
            return save_series_method(root, selection, expected_bytes=original)

        calls.start_action("approve", save, context)

    def _method_choice_reader_changed(self) -> None:
        if (self.method_panel.calls.thread is None
                and not getattr(self, "_prepared_model_publication", False)):
            self._refresh_method_state()

    def _method_choice_finished(self, outcome) -> None:
        if outcome["kind"] == "prepared_model":
            if (not outcome["error"] and outcome["current"]
                    and str(self._method_series_root()) == outcome["root"]):
                self._refresh_method_state(force=True)
            return  # The preparation owner reports failure or continues after acknowledgement.
        if outcome["kind"] == "method_action":
            request, self._pending_method_action = self._pending_method_action, None
            if request is None:
                return
            if outcome["error"] or not outcome["current"]:
                self.print_log("The requested action was not started: " + (outcome["error"] or "the selection changed."))
                return
            if not self._same_method_action_snapshot(request["snapshot"], request["owner"]):
                self.print_log("The requested action was not started because its series, selection or settings changed.")
                return
            request["continuation"]()
            return
        if outcome["error"]:
            QMessageBox.warning(self, "Could not record calibration choice", outcome["error"])
            return
        if (not outcome["current"] or str(self._method_series_root()) != outcome["root"]
                or str(self._method_root) != outcome["root"]):
            self.print_log("The explicit choice was recorded; the newer displayed selection was preserved.")
            return
        self._refresh_method_state(force=True)
        self.print_log("Calibration choice " + ("unlocked with its reason." if outcome["kind"] == "unlock"
                                               else "recorded. No fit or run configuration was changed."))

    def _select_phase_model_for_calibration(self, reference):
        """Record the explicitly chosen catalogue model; calibration remains separate."""
        page = getattr(self.phase_model_workspace, "studio_page", None)
        steps = getattr(page, "native_steps", None)
        studio = getattr(page, "studio", None)
        if steps is None or studio is None:
            return
        entry = studio._selected_validation_model() or {}
        if (str(entry.get("root") or "") != str(reference)
                or self._saved_method.active != "phase_model"):
            steps.set_model_selection_result(False, "Choose a saved model for the current calibration method.")
            return
        root = self._method_series_root()
        if root is None:
            steps.set_model_selection_result(False, "Choose an experimental series first.")
            return
        if reference == self._saved_method.model_version:
            steps.set_model_selection_result(True)
            return
        self._publish_prepared_model(root, reference, expected_bytes=self._method_manifest_bytes,
            expected_selection=self._saved_method, finished=steps.set_model_selection_result)

    def _publish_prepared_model(self, root, reference, *, expected_bytes,
                               expected_selection, finished):
        """Record the verified new version through the existing explicit save owner."""
        from dataclasses import replace
        root = Path(root)
        if (root != self._method_series_root() or root != self._method_root
                or expected_bytes != self._method_manifest_bytes
                or expected_selection != self._saved_method
                or expected_selection.active != "phase_model"
                or self.dataset_series_root() is not None):
            finished(False, "The series or saved model choice changed before the new model could be recorded.")
            return False
        calls = self.method_panel.calls
        if calls.thread is not None:
            finished(False, "Another model-setting action is running; the new model was retained without changing the choice.")
            return False
        chosen = replace(expected_selection, model_version=str(reference), mass_calibration="")
        self._prepared_model_publication = True
        def acknowledged(outcome):
            if outcome["kind"] != "prepared_model":
                return
            calls.action_finished.disconnect(acknowledged)
            current = (not outcome["error"] and outcome["current"]
                       and root == self._method_series_root() and self._saved_method == chosen)
            QTimer.singleShot(0, lambda: setattr(self, "_prepared_model_publication", False))
            finished(current, outcome["error"] or ("" if current else "The selected series changed during model publication."))
        calls.action_finished.connect(acknowledged)
        if not self._save_method_choice(chosen, prepared_model=True, expected_bytes=expected_bytes):
            calls.action_finished.disconnect(acknowledged)
            self._prepared_model_publication = False
            finished(False, "The model-setting save was not started; the previous choice is retained.")
            return False
        return True

    def _unlock_method_choice(self) -> None:
        """An explicit reason unlocks the choice, without replacing its files."""
        root = self._method_series_root()
        if root is None or root != self._method_root or self.dataset_series_root() is not None:
            return
        original = self._method_manifest_bytes
        selected = self.method_panel.selection()
        reason, accepted = QInputDialog.getMultiLineText(
            self, "Unlock calibration choice",
            "Why do you need to change the calibration or model for this series?",
        )
        if not accepted:
            return
        if (self._method_series_root() != root or self._method_root != root
                or self.method_panel.selection() != selected
                or self.dataset_series_root() is not None):
            self.print_log("The series or selection changed; the choice was not unlocked.")
            return
        calls = self.method_panel.calls
        context = (str(root), selected, original)
        if context != calls.context:
            return

        def unlock():
            from darsia.fluidflower.series_calibration_approval import unlock_series_method

            return unlock_series_method(root, reason, expected_bytes=original)

        calls.start_action("unlock", unlock, context)

    def _confirm_method_action(self, method: str, description: str) -> bool:
        self._refresh_method_state()
        if getattr(self, "_method_approval", None) and self._method_approval["locked"]:
            # Actual callers below use the immediate fresh-check continuation;
            # a general synchronous predicate never grants cached authority.
            return False
        if getattr(self, "_method_approval_error", ""):
            return False
        return self._confirm_method_selection(method, description)

    def _confirm_method_selection(self, method: str, description: str) -> bool:
        active = self._saved_method.active
        if (
            active == method
            and not missing_method_references(self._saved_method)
            and not self._saved_method.error
        ):
            return True
        prefix = (
            "No complete method is chosen for this series."
            if not active or missing_method_references(self._saved_method)
            else "This is not the chosen method for this series."
        )
        answer = QMessageBox.question(
            self,
            "Confirm analysis method",
            prefix
            + "\n"
            + description
            + " anyway?\nChoose the series method under Calibration. This confirmation will not change it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _method_action_snapshot(self, owner=None):
        """Pure GUI values used by the original two action seams; no file I/O."""
        from copy import deepcopy
        from PySide6.QtWidgets import (
            QAbstractButton, QAbstractSpinBox, QAbstractSlider, QPlainTextEdit, QTextEdit,
        )

        controls = []
        for widget in self.findChildren(QWidget):
            if QWidget.window(widget) is not self:
                continue  # Transient dialogs are not consumed action settings.
            if isinstance(widget, QLineEdit):
                value = widget.text()
            elif isinstance(widget, QComboBox):
                value = (widget.currentIndex(), widget.currentText())
            elif isinstance(widget, QAbstractButton):
                value = widget.isChecked()
            elif isinstance(widget, (QAbstractSpinBox, QAbstractSlider)):
                read = getattr(widget, "value", None)
                value = read() if callable(read) else widget.text()
            elif isinstance(widget, (QPlainTextEdit, QTextEdit)):
                value = widget.toPlainText()
            else:
                continue
            controls.append((id(widget), type(widget).__name__, value))
        return dict(root=str(self._method_series_root()), context=self.method_panel.calls.context,
                    generation=self.method_panel.calls.generation,
                    action=self.selected_action, item=self.selected_checkbox_id,
                    config_path=self.config_path_label.text(), config=deepcopy(self.config_dict),
                    controls=controls, result=deepcopy(getattr(owner, "result", None)),
                    result_path=str(getattr(owner, "result_path", "")))

    def _same_method_action_snapshot(self, expected, owner=None):
        from darsia.fluidflower.series_method import _same_values

        return _same_values(expected, self._method_action_snapshot(owner))

    def _invoke_method_action(self, method, description, original, args, kwargs, *, owner=None):
        """One fresh owned check, then immediate original action if unchanged."""
        from copy import deepcopy

        self._refresh_method_state()
        record = getattr(self, "_method_approval", None)
        if record is None or not record["locked"]:
            return original(*args, **kwargs) if self._confirm_method_action(method, description) else None
        calls = self.method_panel.calls
        if calls.thread is not None or self._pending_method_action is not None:
            self.print_log("Wait for the current calibration choice reader before starting this action.")
            return None
        captured = self._method_action_snapshot(owner)
        # Keep the original mismatch question BEFORE the fresh file check, so
        # no nested modal wait can age a successful binding before continuation.
        if not self._confirm_method_selection(method, description):
            return None
        if not self._same_method_action_snapshot(captured, owner):
            self.print_log("The action selection changed; nothing was started.")
            return None
        root, selection, manifest = calls.context
        expected = deepcopy(record["binding"])
        copied_args, copied_kwargs = deepcopy(args), deepcopy(kwargs)

        def verify():
            from darsia.fluidflower.series_calibration_approval import read_approval, selection_binding, same_selection_binding

            path = Path(root) / "series.toml"
            if path.read_bytes() != manifest:
                raise ValueError("The approved series changed before the action.")
            actual = read_approval(root)
            if not actual or not actual["locked"] or actual["binding"] != expected:
                raise ValueError("The action no longer has the exact approved files.")
            if not same_selection_binding(root, selection_binding(root, selection), expected):
                raise ValueError("The action selection differs from the approved files.")
            if path.read_bytes() != manifest:
                raise ValueError("The approved series changed during the action check.")
            return actual["binding"]

        self._pending_method_action = dict(snapshot=captured, owner=owner,
            continuation=lambda: original(*copied_args, **copied_kwargs))
        if not calls.start_action("method_action", verify, calls.context):
            self._pending_method_action = None
        return None

    def _install_method_analysis_guard(self) -> None:
        original = self.analysis_tab.run_analysis

        def run_analysis(*args, **kwargs):
            # All runs retains its original once-per-sequence decision.
            if self._all_runs_confirmed:
                return original(*args, **kwargs)
            return self._invoke_method_action(
                "colour_path", "Run original DarSIA analysis", original, args, kwargs)

        run_analysis.__wrapped__ = original
        self.analysis_tab.run_analysis = run_analysis

    def _guard_phase_exports(self, page) -> None:
        saved = getattr(page, "saved_results", page)
        if getattr(saved, "_native_method_guard", False):
            return
        original = saved.export

        def export(*args, **kwargs):
            return self._invoke_method_action(
                "phase_model", "Export phase-model analysis results", original, args, kwargs,
                owner=saved)

        export.__wrapped__ = original
        saved.export = export
        saved._native_method_guard = True

    def _open_existing_phase_review(self) -> None:
        page = self.phase_model_workspace.studio_page
        studio = getattr(page, "studio", None)
        steps = getattr(page, "native_steps", None)
        if steps is not None and steps.review_launch_reason():
            if hasattr(self, "content_stack"):
                self.content_stack.setCurrentWidget(self.phase_model_workspace)
            steps.show_review_requirement()
            return
        if (
            studio is not None
            and str(getattr(studio, "_active_model_series", "") or "").casefold().replace("_", "").replace("-", "")
            == str(self.phase_model_workspace.series_id).casefold().replace("_", "").replace("-", "")
        ):
            selected = studio._selected_validation_model() or {}
            if isinstance(selected.get("validation_binding"), dict):
                try:
                    studio._launch_review("open_correction")
                except Exception as error:
                    message = "Could not open phase review: " + str(error)
                    studio.review_status.setText(message)
                    QMessageBox.warning(self, "Could not open phase review", message)
                return
        if steps is not None:
            if hasattr(self, "content_stack"):
                self.content_stack.setCurrentWidget(self.phase_model_workspace)
            steps.show_review_requirement()
        else:
            QMessageBox.information(self, "Phase review needs a saved model",
                "Open Phase model, choose a model in Basis, and bind its saved phase maps for the selected run.")

    def _refresh_phase_review_requirements(self, *_args):
        workspace = getattr(self, "phase_model_workspace", None)
        page = getattr(workspace, "studio_page", None)
        steps = getattr(page, "native_steps", None)
        if steps is not None:
            steps.refresh()

    def _series_for_path(self, path: str) -> str:
        return self.home_page.series_registry.series_by_path.get(
            path, self.home_page.selected_series()
        )

    def _refresh_series(
        self, *, preferred_series: str | None = None, show_errors: bool = False
    ) -> None:
        """Reload user settings when returning to Start or requesting a refresh."""
        try:
            series = tuple(configured_series())
        except (OSError, ValueError, KeyError) as error:
            if show_errors:
                QMessageBox.warning(self, "Could not refresh series", str(error))
            return
        from darsia.fluidflower.published_dataset.dataset_series import protect_datasets

        protect_datasets(root for _key, _label, root in series)
        workspace = self.phase_model_workspace
        if series != workspace.configured_series:
            page = workspace.studio_page
            studio = getattr(page, "studio", None)
            notice = page._active_job_notice(studio) if studio is not None else None
            if notice is not None:
                if show_errors:
                    QMessageBox.information(self, *notice)
                return
        selected_path = self.home_page._selected_project_path()
        selected = preferred_series or workspace.series_id
        if selected not in {key for key, _label, _root in series}:
            selected = series[0][0] if series else ""
        self.home_page.update_series(series, selected=selected)
        if series != workspace.configured_series:
            workspace.update_series(series, selected=selected)
        self.geometry_workspace.update_series(series, selected=workspace.series_id)
        self._sync_series_display()
        if preferred_series is None and selected_path:
            for row in range(self.home_page.table.rowCount()):
                item = self.home_page.table.item(row, 0)
                if (
                    item is not None
                    and item.data(Qt.ItemDataRole.UserRole) == selected_path
                ):
                    self.home_page.table.selectRow(row)
                    break

    def _open_legacy_project(self, path: str, *, setup: bool = False) -> None:
        """Open an existing series.json package in its original project UI."""
        if path not in self.home_page.series_registry.legacy_by_path:
            return
        window = self._legacy_windows.get(path)
        if window is None:
            from darsia.fluidflower.ui.shell import FluidFlowerMainWindow

            window = FluidFlowerMainWindow(
                registry=self.home_page.series_registry.project_registry,
                initial_project=Path(path),
            )
            if window.workspace is None:
                window.close()
                return
            self._legacy_windows[path] = window
        if setup and window.workspace is not None:
            window.workspace.select_module("setup")
        window.show()
        window.raise_()
        window.activateWindow()

    def _select_series(self, series_id: str) -> None:
        selector = self.phase_model_workspace.series_selector
        index = selector.findData(series_id)
        if index >= 0:
            selector.setCurrentIndex(index)

    def _sync_series_display(self, _index: int = -1) -> None:
        selector = self.phase_model_workspace.series_selector
        label = selector.currentText()
        self.setup_series_label.setText(
            f"Series: {label}" if label else "No series selected"
        )
        for row in range(self.home_page.table.rowCount()):
            if self.home_page.table.item(row, 1).text() == selector.currentData():
                self.home_page.table.selectRow(row)
                break
        workspace = self.phase_model_workspace
        self._series_loading, self._series_load_error = True, ''
        if self._series_snapshot is not None:
            self._accepted_series_snapshot = self._series_snapshot
        self._series_snapshot = None
        self._pending_open_series = self._pending_series_navigation = None
        self._pending_new_series_setup = None
        with QSignalBlocker(self.run_combo):
            self.run_combo.clear()
        self.setup_series_label.setText(f'Series: {label} — loading metadata…')
        self.run_path.setText('Loading series metadata…')
        self.run_combo.setEnabled(False)
        self.home_page.new_button.setEnabled(False)
        self.toolbar_builder.play_action.setEnabled(False)
        self._series_generation = self._series_loader.request(workspace.series_id,
            workspace.series_roots.get(workspace.series_id), label,
            tuple((row.action, row.checkbox_id) for category in ('setup', 'calibration', 'analysis')
                for row in self.sidebar._sections[category]._rows.values()), str(self.config_file or ''))

    def _show_series_loading(self, target):
        self._pending_series_navigation = target
        if not hasattr(self, '_series_loading_page'):
            self._series_loading_page = QWidget()
            layout = QVBoxLayout(self._series_loading_page)
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self._series_loading_text = QLabel()
            self._series_loading_text.setWordWrap(True)
            layout.addWidget(self._series_loading_text)
            self._series_loading_back = QPushButton('Back to Start')
            self._series_loading_back.setToolTip('Return to the series list. The current reader continues in the background.')
            self._series_loading_back.clicked.connect(self._leave_series_loading)
            layout.addWidget(self._series_loading_back, alignment=Qt.AlignmentFlag.AlignLeft)
            self.content_stack.addWidget(self._series_loading_page)
        self._series_loading_text.setText('Loading series metadata…')
        self.content_stack.setCurrentWidget(self._series_loading_page)

    def _leave_series_loading(self):
        self._pending_open_series = self._pending_series_navigation = None
        self._returning_from_series_loading = True
        try:
            self._select_row(('start', 'series'))
        finally:
            self._returning_from_series_loading = False

    def _prepared_series_status(self):
        snapshot = self._current_series_snapshot()
        return (snapshot['statuses'].get(self.run_combo.currentText(), snapshot['statuses'][''])
            if snapshot is not None else None)

    def _prepared_guard_values(self, path):
        snapshot = self._current_series_snapshot()
        prepared = snapshot['configs'].get(str(path)) if snapshot is not None else None
        return prepared[0] if prepared is not None else None

    def _current_series_snapshot(self):
        if not getattr(self, '_using_series_snapshot', False):
            return None
        packet = getattr(self, '_series_snapshot', None)
        workspace = getattr(self, 'phase_model_workspace', None)
        return (packet[1] if packet is not None and workspace is not None
            and packet[0] == workspace.series_id else None)

    def _prepared_native_context(self, root):
        if self._current_series_snapshot() is None:
            if self._series_loading or self._series_load_error:
                raise ValueError(self._series_load_error or 'Series metadata is still loading')
            return self._series_native_context.bind(root)
        return self._series_native_context.binding

    def _series_metadata_ready(self, generation, packet, error):
        if generation != self._series_generation:
            return
        workspace = self.phase_model_workspace
        if packet is not None and (packet.series_id != workspace.series_id
                or packet.root != workspace.series_roots.get(workspace.series_id)):
            return
        self._series_loading = False
        self._series_load_error = error
        if error:
            self._pending_open_series = self._pending_series_navigation = None
            self.setup_series_label.setText('Series could not be loaded: ' + error)
            self.run_path.setText(error)
            if hasattr(self, '_series_loading_page'):
                self._series_loading_text.setText('Series could not be loaded: ' + error + '\nUse Back to Start to choose a series or refresh the list.')
            self.print_log('Series could not be loaded: ' + error)
            return
        self._using_series_snapshot = True
        try:
            self._apply_loaded_series_packet(packet)
        finally:
            self._using_series_snapshot = False

    def _apply_loaded_series_packet(self, packet):
        workspace = self.phase_model_workspace
        data = packet.unpack()
        depth = getattr(self, 'depth_workflow', None)
        if self.process_runner.active or (depth is not None and depth.sequence.running):
            self.print_log('Reference configuration stays unchanged while the current workflow is running.')
            self._restore_accepted_series()
            return
        guard_path, guard_values = data['guard']
        if str(self.config_file or '') != guard_path:
            self._series_load_error = 'The open configuration changed while series metadata was loading. Choose the series again.'
            self.setup_series_label.setText(self._series_load_error)
            return
        if not self.reference_open_guard.allowed(disk_values=guard_values):
            self._restore_accepted_series()
            return
        self._series_snapshot = (packet.series_id, data)
        self._series_native_context.apply(data['native'])
        workspace.defer_native_context = False
        try:
            if workspace.isVisible() or workspace.studio_page is None:
                workspace._series_changed(-1)
            else:
                workspace._context_needs_apply = True
        finally:
            workspace.defer_native_context = True
        self.setup_series_label.setText('Series: ' + workspace.series_selector.currentText())
        self.geometry_workspace.series_id = workspace.series_id
        self.geometry_workspace._update_series_label()
        self._apply_series_display()
        pending, self._pending_new_series_setup = self._pending_new_series_setup, None
        if pending is not None and pending[:2] == (self._series_generation, workspace.series_id):
            self._last_sidebar_selection = ("preprocessing", "protocols")
            self.run_combo.setCurrentIndex(self.run_combo.findText(pending[2]))
            self._open_preprocessing_for_selected_run(rig_preset=pending[3])
            return
        if self._pending_open_series == workspace.series_id:
            self._pending_open_series = None
            self._pending_series_navigation = None
            self._resuming_series_open = True
            try:
                self._open_series_from_start(str(packet.root))
            finally:
                self._resuming_series_open = False
        elif self._pending_series_navigation is not None:
            target, self._pending_series_navigation = self._pending_series_navigation, None
            self._select_row(target)

    def _restore_accepted_series(self):
        accepted = getattr(self, '_accepted_series_snapshot', None)
        workspace = self.phase_model_workspace
        if accepted is not None:
            with QSignalBlocker(workspace.series_selector):
                workspace.series_selector.setCurrentIndex(workspace.series_selector.findData(accepted[0]))
            workspace._selected_series = accepted[0]
            workspace._update_series_folder()
            self._series_snapshot = accepted
            self.setup_series_label.setText('Series: ' + workspace.series_selector.currentText())
            with QSignalBlocker(self.home_page.table):
                for row in range(self.home_page.table.rowCount()):
                    if self.home_page.table.item(row, 1).text() == accepted[0]:
                        self.home_page.table.selectRow(row)
                        break
            self._apply_series_display()
        self._pending_open_series = self._pending_series_navigation = None
        self.content_stack.setCurrentWidget(self.home_page)

    def _apply_series_display(self):
        self._update_painted_geometry_status()
        self._populate_runs()
        if self._darsia_visible():
            self._ensure_run_loaded()
        if getattr(self, "depth_workflow", None) is not None:
            self.depth_workflow.refresh_if_visible(reload_selected=True)
        if self.mass_analysis_page is not None and self.mass_analysis_page.isVisible():
            series_id = self.phase_model_workspace.series_id
            root = self.phase_model_workspace.series_roots.get(series_id)
            if root is not None and (
                self.mass_analysis_page.series != series_id
                or self.mass_analysis_page.project_root != Path(root)
            ):
                self.mass_analysis_page.set_series(series_id, root)
        self._refresh_method_state()
        self._refresh_scope_display()
        if self.jobs_workspace is not None and self.jobs_workspace.isVisible():
            self._refresh_jobs_context()

    def _refresh_jobs_context(self) -> None:
        if self.jobs_workspace is None:
            return
        workspace = self.phase_model_workspace
        self.jobs_workspace.set_series(
            workspace.series_id, workspace.series_roots.get(workspace.series_id),
            series_label=workspace.series_selector.currentText(),
        )

    def ensure_jobs(self) -> QWidget:
        if self.jobs_workspace is None:
            from darsia.fluidflower.native_jobs_workspace import NativeJobsWorkspace
            from darsia.fluidflower.production_page import ProductionPage
            from darsia.fluidflower.production_selection import create_production_job
            from darsia.fluidflower.resource_advisor_panel import ResourceAdvisorPanel

            self.jobs_workspace = NativeJobsWorkspace(
                self, resource_factory=ResourceAdvisorPanel,
                new_analysis_factory=ProductionPage, job_creator=create_production_job,
                raw_review_model_provider=self._selected_jobs_review_model,
            )
            self.jobs_workspace.new_analysis_requested.connect(
                lambda: self._select_row(("jobs", "new_analysis"))
            )
            self.jobs_workspace.jobs_requested.connect(lambda: self._select_row(("jobs", "overview")))
            self.jobs_workspace.raw_model_requested.connect(self.show_phase_model_workflow)
            self.jobs_workspace.raw_review_registered.connect(self._jobs_review_registered)
            self.jobs_workspace.series_locations_changed.connect(
                lambda _root: self._refresh_series(preferred_series=self.phase_model_workspace.series_id)
            )
            self.jobs_workspace.panel.phase_one_launch_accepted.connect(
                self._phase_one_launch_accepted
            )
            self.content_stack.addWidget(self.jobs_workspace)
        self._refresh_jobs_context()
        return self.jobs_workspace

    def _phase_one_launch_accepted(self, launch):
        """Hand the accepted original Main P1 job to the existing refit waiter."""
        panel = self.mass_calibration_display.refit_panel
        key = (str(Path(launch["job_root"]).resolve()), tuple(launch["required_runs"]))
        try:
            if (not launch.get("series_root") or panel.series_root is None
                    or Path(panel.series_root).resolve() != Path(launch["series_root"]).resolve()):
                raise ValueError("Mass-calibration choices must belong to the series selected for phase map generation")
            active = panel._automatic
            if (active is not None and active["generation"] == panel._generation
                    and tuple(active["required_runs"]) == key[1]
                    and str(Path(active["arguments"]["phase_job"]).resolve()) == key[0]):
                return True
            if not panel.start_after_phase_one(key[0], list(key[1])):
                raise ValueError("The mass-calibration panel is busy; automatic refit was not attached")
        except Exception as error:
            self.jobs_workspace.panel._error(
                "Phase map generation was requested, but automatic mass calibration could not be attached: "
                + str(error)
            )
            return False
        return True

    def _selected_jobs_review_model(self):
        """Return the exact model currently selected for production review."""
        page = getattr(self.phase_model_workspace, "studio_page", None)
        studio = getattr(page, "studio", None)
        return studio._selected_validation_model() if studio is not None else None

    def _jobs_review_registered(self, _binding):
        """Refresh model bindings after a production review is registered."""
        page = getattr(self.phase_model_workspace, "studio_page", None)
        studio = getattr(page, "studio", None)
        if studio is not None:
            studio._request_async_model_refresh(bindings_only=True)

    def show_jobs(self, *, start_analysis: bool = False) -> None:
        """Open Jobs, or New analysis, which replaces the start dialog (N141)."""
        self._select_row(("jobs", "new_analysis" if start_analysis else "overview"))

    def _series_method_text(self) -> str:
        """The series and method as the header line names them, for New analysis."""
        series = self.phase_model_workspace.series_selector.currentText()
        return status_line(series or "No series selected", self._method_line_parts, "")

    def closeEvent(self, event) -> None:
        preparation = getattr(getattr(self, "mass_calibration_display", None), "preparation", None)
        if preparation is not None and hasattr(preparation, "cancel_preview"):
            preparation.cancel_preview()
        preview_steps = getattr(getattr(getattr(self, "phase_model_workspace", None), "studio_page", None), "native_steps", None)
        if preview_steps is not None and hasattr(preview_steps, "cancel_metadata_previews"):
            preview_steps.cancel_metadata_previews()
        if preparation is not None and not preparation.can_close():
            self.print_log("Calibration preparation is finishing. Close after the current metadata operation returns.")
            event.ignore()
            return
        folders = getattr(self, "_series_folders_panel", None)
        if folders is not None and not folders.can_close():
            self.print_log("Wait for the current folder settings operation to finish.")
            event.ignore()
            return
        legacy = getattr(self, "legacy_workspace", None)
        if legacy is not None and not legacy.can_close():
            self.print_log(legacy.close_block_reason())
            legacy.legacy_window.print_log(legacy.close_block_reason())
            event.ignore()
            return
        if getattr(self, '_series_loader', None) is not None and self._series_loader.busy():
            self.print_log('Series metadata is still loading. Close after the reader returns; the reader was not stopped.')
            event.ignore()
            return
        page = getattr(getattr(self, "phase_model_workspace", None), "studio_page", None)
        steps = getattr(page, "native_steps", None)
        can_close = getattr(steps, "can_close", None)
        if callable(can_close) and not can_close():
            reason = getattr(steps, "close_block_reason", None)
            self.print_log((reason() if callable(reason) else "") or
                "Phase-model file preview or copying is still finishing; close after it returns.")
            log_dock = getattr(self, "log_dock", None)
            if log_dock is not None:
                log_dock.show()
            event.ignore()
            return
        checks = getattr(self, "calibration_checks_page", None)
        if checks is not None and not checks.can_close():
            self.print_log("Calibration checks are still finishing. Close after the original reader returns. No worker was stopped.")
            event.ignore()
            return
        method_panel = getattr(self, "method_panel", None)
        if method_panel is not None and not method_panel.can_close():
            self.print_log("The calibration choice reader is still finishing. Close after it returns. No worker was stopped.")
            event.ignore()
            return
        image_selection = getattr(self, "calibration_image_selection", None)
        if image_selection is not None and not image_selection.can_close():
            self.print_log("Calibration image choices are being saved; close after the save returns.")
            event.ignore()
            return
        choices = getattr(getattr(self.mass_calibration_display, "phase_colour_paths", None), "choices", None)
        if choices is not None and not choices.can_close():
            self.print_log("Colour-path choices are being saved; close after the save returns.")
            event.ignore()
            return
        finish = getattr(self, "finish_setup_page", None)
        if finish is not None and not finish.can_close():
            self.print_log("Finish setup metadata is being prepared; close after it returns.")
            event.ignore()
            return
        jobs = getattr(self, "jobs_workspace", None)
        if jobs is not None:
            jobs.prepare_close()
        if jobs is not None and not jobs.can_close():
            self.print_log(jobs.close_block_reason())
            event.ignore()
            return
        if jobs is not None:
            controller = getattr(self, "_jobs_close_controller", None)
            if controller is None:
                from .native_jobs_close import NativeJobsCloseController
                controller = self._jobs_close_controller = NativeJobsCloseController(self, jobs)
            if not controller.allow_close():
                event.ignore()
                return
        super().closeEvent(event)
        if event.isAccepted():
            if legacy is not None:
                legacy.finish_close()
            if method_panel is not None:
                method_panel.calls.stop_pending()
            self._series_native_context.close()

    def ensure_mass_analysis(self) -> QWidget:
        """Create production mass analysis as its own top-level workspace."""
        if self.mass_analysis_page is None:
            from darsia.fluidflower.tools.phase_studio_mass_analysis_workspace import (
                MassAnalysisWorkspace,
            )

            series_id = self.phase_model_workspace.series_id
            root = self.phase_model_workspace.series_roots.get(series_id)
            if root is None:
                raise ValueError("Choose an experimental series before mass analysis")
            self.mass_analysis_page = MassAnalysisWorkspace(
                series=series_id, project_root=root
            )
            self.mass_analysis_page.series_selected.connect(
                self._mass_analysis_series_selected
            )
            self.content_stack.addWidget(self.mass_analysis_page)
            self._guard_phase_exports(self.mass_analysis_page)
            self.ensure_jobs().install_mass_action(
                self.mass_analysis_page, lambda: self.show_jobs(start_analysis=True)
            )
            from darsia.fluidflower.native_results_export_layout import NativeResultsExportLayout

            self.mass_analysis_page.results_export_layout = NativeResultsExportLayout(
                self.mass_analysis_page
            )
        return self.mass_analysis_page

    def _show_overflow_use(self) -> None:
        """Compatibility route: overflow is now part of Mass calibration."""
        self.show_common_mass_calibration()

    def _set_overflow_diagnostic(self, mode: str) -> None:
        """A local saved-figure visibility choice, without altering calibration settings."""
        if self.mass_analysis_page is not None:
            self.mass_analysis_page.saved_results.set_overflow_comparison_mode(mode)

    def _show_saved_overflow_comparison(self) -> None:
        """Use the original explicit saved-results browser; never compute a comparison."""
        self._show_mass_analysis_view("review")
        if self.mass_analysis_page is not None:
            saved = getattr(self.mass_analysis_page, "saved_results", None)
            if saved is not None:
                saved.show_saved_overflow_comparison()

    def _show_mass_analysis_view(self, view: str) -> None:
        """Calibration review and Results and export share the saved-result page."""
        if not self.phase_model_workspace.series_id:
            self.content_stack.setCurrentWidget(self.mass_analysis_placeholder)
            return
        page = self.ensure_mass_analysis()
        page.show_view(view)
        mode = "compare"
        if self.mass_calibration_display._refit_series == self._method_series_root():
            mode = self.overflow_page.mode or mode
        saved = getattr(page, "saved_results", None)
        if saved is not None:
            saved.set_overflow_comparison_mode(mode)
            if view == "review" and self.mass_calibration_display._refit_series == self._method_series_root():
                panel = self.mass_calibration_display.refit_panel
                snapshot = panel.snapshot or {}
                frozen = snapshot.get("frozen")
                if frozen is not None:
                    saved.attach_local_calibration(frozen[1], name=panel.name_edit.text() or "Saved local calibration")
                for run, value in snapshot.get("individual_calibrations", {}).items():
                    saved.attach_local_calibration(value[1],
                        name=(panel.name_edit.text() or "Individual calibration") + " · " + run.upper())
        self.content_stack.setCurrentWidget(page)

    def _mass_analysis_series_selected(self, series_id: str) -> None:
        self._select_series(series_id)
        # Studio can reject a series change while training. Keep both pages
        # on the same series in that case.
        actual = self.phase_model_workspace.series_id
        if actual != series_id and self.mass_analysis_page is not None:
            root = self.phase_model_workspace.series_roots.get(actual)
            if root is not None:
                self.mass_analysis_page.set_series(actual, root)

    def _job_notice(self) -> tuple[str, str] | None:
        page = self.phase_model_workspace.studio_page
        studio = getattr(page, "studio", None)
        return page._active_job_notice(studio) if studio is not None else None

    def _new_series(self) -> None:
        """Use the selected method's existing project-creation entry point."""

        if self.home_page.new_series_method.currentData() == "colour_path":
            self._select_row(("legacy", "workspace"))
            self.legacy_workspace.legacy_window.config_controller.new_config()
            return

        notice = self._job_notice()
        if notice is not None:
            QMessageBox.information(self, *notice)
            return
        from darsia.fluidflower.series_wizard import NewSeriesFolderWizard

        wizard = NewSeriesFolderWizard(
            self,
            registered_ids=[key for key, _label, _root in configured_series()],
        )
        self.new_series_wizard = wizard
        if wizard.exec() != QDialog.DialogCode.Accepted or wizard.created is None:
            return
        series_id = self.register_new_series(wizard.created, refresh=False)
        if not series_id:
            return
        page = wizard.series_page
        rig_preset = (
            (page.physical_rig.currentText(), str(page.physical_rig.currentData()))
            if page.rig_class.currentData() == "medium"
            else None
        )
        try:
            first_run, _path = self._prepare_new_series_runs(
                wizard.created, rig_preset=rig_preset
            )
        except (OSError, ValueError, KeyError) as error:
            self._refresh_series(preferred_series=series_id, show_errors=True)
            message = f"Series created, but Preprocessing could not be opened: {error}"
            self.print_log(message)
            QMessageBox.warning(self, "Series created; setup not opened", message)
            return
        # Read the new merged configurations before handing the selected run to
        # the original wizard, with one metadata request after their creation.
        self._refresh_series(preferred_series=series_id, show_errors=True)
        self._pending_new_series_setup = (self._series_generation, series_id, first_run, rig_preset)
        self._show_series_loading(None)

    @staticmethod
    def _prepare_new_series_runs(
        root: Path, *, rig_preset: tuple[str, str] | None = None
    ) -> tuple[str, Path]:
        """Prepare controller files only for a just-created series, in manifest order."""
        from copy import deepcopy
        import toml
        from darsia.presets.workflows.config.utils import _deep_merge

        root = Path(root)
        manifest = _read_toml(root / "series.toml")
        paths = manifest.get("paths") or {}
        runs = manifest.get("runs") or {}
        ordered_runs = [str(run) for run in runs.get("ids") or []]
        if not ordered_runs:
            raise ValueError("The new series contains no experiments.")
        common = _read_toml(
            root / str(paths.get("common_config", "config/common.toml"))
        )
        files = runs.get("config_files") or {}
        run_folder = str(paths.get("run_configs", "config/runs"))
        merged_folder = root / str(paths.get("merged_configs", "config/merged"))
        rig = None
        if rig_preset is not None:
            from darsia.presets.workflows.config.catalogue import SeriesPreset

            pieces = SeriesPreset(rig=rig_preset[1]).pieces()
            if "rig" not in pieces:
                raise ValueError(
                    f"The existing rig profile is unavailable: {rig_preset[1]}"
                )
            rig = pieces["rig"][1]
        merged_folder.mkdir(parents=True, exist_ok=True)
        for run in ordered_runs:
            name = Path(str(files.get(run) or f"{run_folder}/{run}.toml"))
            source = name if name.is_absolute() else root / name
            merged = _deep_merge(deepcopy(common), _read_toml(source))
            if rig is not None:
                merged["rig"] = deepcopy(rig)
            target = merged_folder / f"{run}.toml"
            # Never overwrite a file if creation raced or this is not a fresh series.
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                toml.dump(merged, stream)
        return ordered_runs[0], merged_folder / f"{ordered_runs[0]}.toml"

    def register_new_series(self, root: Path, *, refresh: bool = True) -> str:
        """Register a created series folder, select it and show the next steps."""

        from darsia.fluidflower.series_paths import register_series

        try:
            series_id, label, _root = register_series(Path(root))
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(
                self,
                "Series created but not registered",
                f"{error}\n\nAdd {root} on Start with 'Add existing series folder'.",
            )
            return ""
        if refresh:
            self._refresh_series(preferred_series=series_id, show_errors=True)
            self._select_series(series_id)
        self.print_log(f"Series {label} created and registered in {root}.")
        return series_id

    def _add_existing_series(self) -> None:
        page = self.phase_model_workspace.studio_page
        studio = getattr(page, "studio", None)
        if studio is not None:
            notice = page._active_job_notice(studio)
            if notice is not None:
                QMessageBox.information(self, *notice)
                return
        folder = QFileDialog.getExistingDirectory(
            self, "Select a DarSIA series folder", str(Path.home())
        )
        if not folder:
            return
        from darsia.fluidflower.series_paths import register_series
        from darsia.fluidflower.published_dataset import dataset_series, read_only

        # A published dataset opens read-only; its results need a location outside it.
        dataset = read_only.dataset_root(folder)
        results = None
        if dataset is not None:
            if not (dataset / "series.toml").is_file():
                QMessageBox.warning(
                    self,
                    "Dataset not unpacked",
                    "Unpack series.zip and expected.zip into the dataset folder, "
                    "then add it again.",
                )
                return
            results = self._choose_dataset_results(dataset)
            if results is None:
                return
        try:
            series_id, _label, _root = register_series(
                dataset if dataset is not None else Path(folder)
            )
            if dataset is not None:
                dataset_series.register_dataset(_root, results)
                read_only.protect(_root)
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(
                self,
                "Could not add series",
                f"{error}\n\nSelect an existing folder containing series.toml. "
                "Use Setup to begin a new experiment.",
            )
            return
        self._refresh_series(preferred_series=series_id, show_errors=True)

        self._open_series_from_start(str(_root))

    def dataset_series_root(self) -> Path | None:
        """The selected series' folder when it is a published dataset, else None."""
        from darsia.fluidflower.published_dataset.layout import MANIFEST

        workspace = getattr(self, "phase_model_workspace", None)
        if workspace is None:
            return None
        root = workspace.series_roots.get(workspace.series_id)
        snapshot = self._current_series_snapshot()
        if snapshot is not None:
            return Path(root) if snapshot['dataset'] else None
        if getattr(self, '_series_loading', False):
            return None
        if root is None or not (Path(root) / MANIFEST).is_file():
            return None
        return Path(root)

    def _choose_dataset_results(self, dataset: Path) -> Path | None:
        """Ask for a results location outside a published dataset; None when cancelled."""
        from darsia.fluidflower.published_dataset import dataset_series

        while True:
            folder = QFileDialog.getExistingDirectory(
                self,
                "Choose a results location outside the published dataset",
                str(Path(dataset).parent),
            )
            if not folder:
                return None
            problems = dataset_series.results_location_problems(dataset, folder)
            if not problems:
                return Path(folder)
            QMessageBox.warning(self, "Choose another results location", problems[0])

    def ensure_dataset_analysis(self) -> QWidget:
        """The New analysis page of published datasets (created once)."""
        if self.dataset_analysis_page is None:
            from darsia.fluidflower.published_dataset import dataset_analysis

            self.dataset_analysis_page = dataset_analysis.DatasetAnalysisPage(self)
            self.content_stack.addWidget(self.dataset_analysis_page)
        return self.dataset_analysis_page

    def ensure_calibration_checks(self) -> QWidget:
        """The read-only Checks page of the Phase-map method (created once)."""
        page = getattr(self, "calibration_checks_page", None)
        if page is None:
            from darsia.fluidflower.calibration_checks_view import CalibrationChecksPage

            page = self.calibration_checks_page = CalibrationChecksPage()
            self.content_stack.addWidget(page)
        return page

    def ensure_production_results(self) -> QWidget:
        """The Production results page of finished production jobs (created once)."""
        page = getattr(self, "production_results_page", None)
        if page is None:
            from darsia.fluidflower.production_results_view import ProductionResultsPage

            page = self.production_results_page = ProductionResultsPage()
            self.content_stack.addWidget(page)
        return page

    def _show_dataset_analysis(self) -> bool:
        """Show New analysis for a published dataset series; False for other series."""
        root = self.dataset_series_root()
        if root is None:
            return False
        from darsia.fluidflower.published_dataset import dataset_series

        results = dataset_series.dataset_results(root)
        if results is None:
            results = self._choose_dataset_results(root)
            if results is None:
                return True
            dataset_series.register_dataset(root, results)
        page = self.ensure_dataset_analysis()
        if page.root != root.resolve() or page.results != results:
            page.set_dataset(root.resolve(), results)
        else:
            page.offer_fresh_results()
        self.content_stack.setCurrentWidget(page)
        self.print_log(
            "Published dataset: the series is read-only, so its other pages stay closed. "
            f"New analysis runs its plan into {results}."
        )
        return True

    def open_image_selection_plan(self) -> None:
        """Open the series' explicit image editor; no numerical work starts."""
        self.image_selection_workflow.open()

    def open_calibration_image_selection(self) -> None:
        """Open calibration observations independently of Setup's source pool."""
        self._select_row(("calibration", "images"))

    def _show_setup_for_path(self, path: str) -> None:
        if path in self.home_page.series_registry.legacy_by_path:
            self._open_legacy_project(path, setup=True)
            return
        self._select_series(self._series_for_path(path))
        self.show_setup()

    def _show_training_for_path(self, path: str) -> None:
        """Historical Start-opening entry point; opening now lands on Setup."""
        self._open_series_from_start(path)

    def _show_home_calibration(self) -> None:
        """Historical hidden Open-button entry point; preserve the Start route."""
        self._open_series_from_start()

    def _remove_series_from_start(self) -> None:
        """Remove only the selected local registration, retaining files and jobs."""
        path = self.home_page._selected_project_path()
        registry = self.home_page.series_registry
        if path not in registry.series_by_path and path not in registry.legacy_by_path:
            return
        from .series_folders_ui import remove_series_registration
        try:
            if path in registry.legacy_by_path:
                registry.project_registry.forget_path(path)
            else:
                remove_series_registration(path)
        except (OSError, ValueError, RuntimeError) as error:
            self.print_log(str(error))
            return
        self._refresh_series(show_errors=True)
        self.print_log("Removed from Start. Project files and running jobs are unchanged.")

    def _show_series_folders(self) -> None:
        """Edit selected-series folders in the main window, without a workflow."""
        path = self.home_page._selected_project_path()
        if path not in self.home_page.series_registry.series_by_path:
            return
        previous = getattr(self, "_series_folders_panel", None)
        if previous is not None and not previous.can_close():
            self.print_log("Wait for the current folder settings operation to finish.")
            return
        from .series_folders_ui import SeriesFoldersPanel
        panel = SeriesFoldersPanel(Path(path), self)
        self._series_folders_panel = panel
        self.content_stack.addWidget(panel)
        panel.back_requested.connect(lambda:self._select_row(("start", "series")))
        panel.folders_saved.connect(lambda _root:self._refresh_series(preferred_series=self.home_page.selected_series()))
        def setup_selected(_root):
            selected = self.home_page.selected_series()
            self._select_row(("start", "series"))
            self._refresh_series(preferred_series=selected)
        panel.setup_selected.connect(setup_selected)
        self.content_stack.setCurrentWidget(panel)
        if previous is not None:
            self.content_stack.removeWidget(previous)
            previous.deleteLater()

    def _open_series_from_start(self, path: str | None = None) -> None:
        """Open one series' reference config and Setup, without a workflow."""
        path = path or self.home_page._selected_project_path()
        if path in self.home_page.series_registry.legacy_by_path:
            self._open_legacy_project(path)
            return
        series_id = self._series_for_path(path)
        if not series_id:
            return
        target = ("preprocessing", "protocols")
        previous_series = self.phase_model_workspace.series_id
        self._select_series(series_id)
        if self.phase_model_workspace.series_id != series_id:
            return
        if not getattr(self, '_resuming_series_open', False):
            if previous_series == series_id:
                self._sync_series_display()
            self._pending_open_series = series_id
            self._show_series_loading(target)
            return
        if self._series_loading:
            self._pending_open_series = series_id
            self._show_series_loading(('preprocessing', 'protocols'))
            return
        if self._series_load_error:
            self._sync_series_display()
            self._pending_open_series = series_id
            self._show_series_loading(('preprocessing', 'protocols'))
            return
        if self.dataset_series_root() is not None:
            # A published dataset opens on New analysis; its folder is never written.
            self._select_row(("jobs", "new_analysis"))
            if self.content_stack.currentWidget() is not self.dataset_analysis_page:
                self._show_dataset_analysis()
            return
        self._last_sidebar_selection = target
        if not self._load_reference_configuration():
            # An old/foreign configuration must never look like successful
            # Protocols for the requested series, or remain runnable there.
            self._select_row(("start", "series"))
            return
        if (
            self.sidebar.get_selected() != target
            or self.content_stack.currentWidget() is not self.settings_scroll_area
        ):
            self._select_row(target)

    def _reference_switch_allowed(self) -> bool:
        return self.reference_open_guard.allowed()

    def _load_reference_configuration(self) -> bool:
        """Use the existing safe loader; never skip to another usable run."""
        from darsia.fluidflower.series_reference_selection import reference_choice

        depth = getattr(self, "depth_workflow", None)
        if self.process_runner.active or (depth is not None and depth.sequence.running):
            self.print_log(
                "Reference configuration stays unchanged while the current workflow is running."
            )
            return False
        workspace = self.phase_model_workspace
        root = workspace.series_roots.get(workspace.series_id)
        if root is None:
            return False
        snapshot = self._current_series_snapshot()
        choice = snapshot['reference'] if snapshot is not None else reference_choice(root, run_configurations(root))
        self.reference_configuration_choice = choice
        if not choice.config_path:
            self.print_log(choice.summary + " " + " ".join(choice.issues))
            return False
        index = self.run_combo.findText(choice.run_id)
        if index < 0:
            self.print_log(
                "Reference run is not present in the current run selector: "
                + choice.run_id
            )
            return False
        with QSignalBlocker(self.run_combo):
            self.run_combo.setCurrentIndex(index)
        if self._config_series == workspace.series_id and _same_file(
            self.config_file, choice.config_path
        ):
            self._update_run_path()
            self._refresh_scope_display()
            return True
        if not self._load_run_config(choice.config_path):
            self._update_run_path()
            self._refresh_scope_display()
            self.print_log(
                "Reference configuration could not be loaded: " + choice.config_path
            )
            return False
        self.print_log(choice.summary)
        return True

    def _home_series_selected(self) -> None:
        """Keep the per-run categories on the series selected on Start."""
        selected = self.home_page.selected_series()
        if selected and selected != self.phase_model_workspace.series_id:
            if not self.reference_open_guard.allowed(notify=False,
                    disk_values=self.reference_open_guard.disk):
                current = self.phase_model_workspace.series_id
                with QSignalBlocker(self.home_page.table):
                    for row in range(self.home_page.table.rowCount()):
                        if self.home_page.table.item(row, 1).text() == current:
                            self.home_page.table.selectRow(row)
                            break
                return
            self._select_series(selected)
        self._update_setup_action()

    def _update_setup_action(self, _index: int = -1) -> None:
        if getattr(self, '_series_loading', False) or getattr(self, '_series_load_error', ''):
            self.home_page.new_button.setEnabled(False)
            return
        selected = self.home_page.selected_series()
        path = (
            self.run_combo.currentData()
            if selected and selected == self.phase_model_workspace.series_id
            else None
        )
        snapshot = self._current_series_snapshot()
        enabled = bool(path and (str(path) in snapshot['configs'] if snapshot is not None else Path(path).is_file()))
        button = self.home_page.new_button
        button.setText("Set up an experiment")
        button.setEnabled(enabled)
        button.setToolTip(
            f"Open Preprocessing for {self.run_combo.currentText()}."
            if enabled
            else "Choose a series and an available experiment in Run above first."
        )

    def _open_preprocessing_for_selected_run(
        self, *, rig_preset: tuple[str, str] | None = None
    ) -> None:
        """Load the selected run and use DarSIA's unchanged per-run wizard."""
        self._home_series_selected()
        if self._show_dataset_analysis():
            return  # A published dataset is read-only; it opens on New analysis.
        selected = self.home_page.selected_series()
        path = self.run_combo.currentData()
        if not selected or selected != self.phase_model_workspace.series_id or not path:
            self._update_setup_action()
            return
        recorded = getattr(self, "_config_stamp", None)
        already_loaded = (
            _same_file(self.config_file, path)
            and self._config_series == selected
            and recorded is not None
            and recorded[1] == _file_stamp(path)
        )
        # The unchanged wizard syncs pending form edits. Do not discard them by
        # reloading a file which is already open and unchanged on disk.
        if not already_loaded and not self._load_run_config(str(path)):
            message = f"Could not load the selected experiment configuration: {path}"
            self.print_log(message)
            QMessageBox.warning(self, "Could not set up experiment", message)
            return
        self._select_row(("preprocessing", "protocols"))
        if rig_preset is None:
            self.open_wizard_for("preprocessing")
            return
        from darsia.gui.ui.main_window import load_wizard_dialog_class
        from darsia.gui.ui.preprocessing_wizard import STEPS
        from darsia.presets.workflows.config.catalogue import SeriesPreset

        dialog = load_wizard_dialog_class("preprocessing")(self)
        wizard_inputs = self.settings_inputs.copy()
        # Adapt this instance only; neither the upstream wizard nor its catalogue changes.
        label, profile = rig_preset
        preset_name = f"{label} (rig only)"
        dialog._catalogue.presets[preset_name] = SeriesPreset(
            description=f"Existing {label} rig profile ({profile}); depth and corrections are set separately.",
            rig=profile,
        )
        series_page = next(
            index for index, step in enumerate(STEPS) if step[0] == "series"
        )
        dialog._ensure_page(series_page)
        dialog._series_combo.setCurrentText(preset_name)
        # Rig-only selection leaves Data unchanged but refreshes MainWindow's
        # registry. Keep the live first-page fields owned by this wizard.
        self.settings_inputs.clear()
        self.settings_inputs.update(wizard_inputs)
        self.rig_context.refresh()
        self.rig_context.prepare_wizard(dialog, extra_policy=prepare_crop_wizard)
        dialog.exec()

    def open_wizard_for(self, category: str) -> None:
        if self.dataset_series_root() is not None:
            self.print_log(
                "Wizards change the series folder, and a published dataset is read-only."
            )
            return None
        if category != "preprocessing":
            return super().open_wizard_for(category)
        from darsia.gui.ui.main_window import load_wizard_dialog_class

        if not self.config_file:
            self.print_log("Wizard: create or open a config file first.")
            self.config_controller.new_config()
            if not self.config_file:
                return
        dialog = load_wizard_dialog_class(category)(self)
        self.rig_context.refresh()
        self.rig_context.prepare_wizard(dialog, extra_policy=prepare_crop_wizard)
        dialog.exec()

    def show_setup(self, step: str | None = None) -> None:
        """Open the requested existing setup category for the selected series."""
        legacy_path = self.home_page.selected_legacy_project()
        if legacy_path:
            self._open_legacy_project(legacy_path, setup=True)
            return
        self._select_series(self.home_page.selected_series())
        selection = {
            "protocols": ("preprocessing", "protocols"),
            "native": ("setup", "all"),
            "color": ("calibration", "color"),
        }.get(step, ("geometry_facies", "rig_frame"))
        self._select_row(selection)

    def show_original_workflow(self) -> None:
        """Compatibility entry point for the original Calibration category."""
        self._select_row(("calibration", "mass"))

    def show_geometry_facies(self) -> None:
        self._select_row(("geometry_facies", "rig_frame"))

    def show_color_path(self) -> None:
        self._select_row(("calibration", "color"))

    def _update_painted_geometry_status(self) -> None:
        """An export proves reference files exist, not that all geometry is applied."""
        if getattr(self, '_series_loading', False):
            return
        from darsia.fluidflower.series_geometry import geometry_package_root
        from darsia.fluidflower.setup_display import (
            completed_current_native_setup_export,
        )

        root = self.phase_model_workspace.series_roots.get(
            self.phase_model_workspace.series_id
        )
        snapshot = self._current_series_snapshot()
        completed = snapshot['completed'] if snapshot is not None else (
            completed_current_native_setup_export(geometry_package_root(root))
            if root is not None
            else None
        )
        for item in ("segmentation", "facies"):
            row = self.sidebar._sections["setup"].get_row(f"setup:{item}")
            if row is None:
                continue
            title = next(
                (
                    text
                    for text, source, key, _help in self.SETUP_WORKFLOW
                    if source == "setup" and key == item
                ),
                row.label,
            )
            row.label = title + (" — reference export available" if completed else "")
            row._text_label.setText(row.label)
            row._text_label.setWordWrap(True)
            row.setToolTip(
                "Reference labels and facies were exported. They have not been applied to other runs."
                if completed
                else row.help_text
            )
            row.setEnabled(True)
            effect = row.graphicsEffect()
            if effect is not None:
                effect.setOpacity(1.0)
        self._update_run_action()

    def _sidebar_selection_enabled(self) -> bool:
        if not hasattr(self, "sidebar"):
            return False
        selected = self.sidebar.get_selected()
        if selected is None:
            return False
        action, item = self.sidebar.display_selection(*selected)
        row = self.sidebar._sections[action].get_row(item)
        return row is not None and row.isEnabled()

    def _darsia_visible(self) -> bool:
        return (
            getattr(self, "_current_category", None) in self.action_dispatch
            and hasattr(self, "content_stack")
            and self.content_stack.currentWidget() is self.settings_scroll_area
        )

    def _all_runs_running(self) -> bool:
        """True while All runs starts the selected action run after run."""
        return getattr(self, "all_runs", None) is not None and self.all_runs.sequence.running

    def _guard_custom_play(self) -> None:
        """A background workflow may finish after navigation to a custom page."""
        action = self.toolbar_builder.play_action
        if action.isEnabled() and (
            self.selected_action not in self.action_dispatch
            or not self._darsia_visible()
            or not self._sidebar_selection_enabled()
            or (
                getattr(self, "depth_workflow", None) is not None
                and self.depth_workflow.sequence.running
            )
            or self._all_runs_running()
        ):
            action.setEnabled(False)

    def _update_run_action(self) -> None:
        if getattr(self, "depth_workflow", None) is not None:
            self.depth_workflow.sync_play_help()
        self.toolbar_builder.play_action.setEnabled(
            not getattr(self, '_series_loading', False)
            and not getattr(self, '_series_load_error', '')
            and self._darsia_visible()
            and self.selected_action in self.action_dispatch
            and self.selected_checkbox_id is not None
            and not self.process_runner.active
            and not (
                getattr(self, "depth_workflow", None) is not None
                and self.depth_workflow.sequence.running
            )
            and not self._all_runs_running()
            and self._sidebar_selection_enabled()
        )

    def _select_row(self, selection: tuple[str, str]) -> None:
        if selection in {("setup", "depth"), ("preprocessing", "depth_measurements")}:
            self._pending_depth_section = True
            selection = ("geometry_facies", "rig_frame")
        if selection == ("geometry_controls", "curvature"):
            selection = ("geometry_facies", "rig_frame")
        action, item_id = selection
        display_action, display_id = self.sidebar.display_selection(action, item_id)
        section = self.sidebar._sections.get(display_action)
        if section is None or section.get_row(display_id) is None:
            return
        self._selecting_sidebar = True
        try:
            section.set_expanded(True)
        finally:
            self._selecting_sidebar = False
        self.sidebar.select(action, item_id)
        if getattr(self, "_pending_depth_section", False) and selection == ("geometry_facies", "rig_frame"):
            self._open_embedded_depth()

    def _restore_selection(self) -> None:
        """Opening a configuration clears selection in the original controller."""
        self._select_row(self._last_sidebar_selection)

    def _sync_home_series_before_navigation(self) -> None:
        if self._current_category == "start":
            selected = self.home_page.selected_series()
            if selected and selected != self.phase_model_workspace.series_id:
                self._select_series(selected)

    def _on_sidebar_expanded(self, action: str, is_expanded: bool) -> None:
        if action == "legacy" and hasattr(self, "content_stack"):
            if is_expanded and not self._selecting_sidebar:
                self.sidebar.select("legacy", "workspace")
            return
        if action in {"calibration", "analysis", "comparison", "export"} and hasattr(self, "content_stack"):
            if is_expanded and not self._selecting_sidebar:
                self._sync_home_series_before_navigation()
                self._refresh_method_state()
                first = self._method_first_rows.get(action) or next(
                    iter(self.sidebar._sections[action]._rows)
                )
                logical = self.sidebar._logical_selection(action, first)
                self.sidebar.select(*logical)
                return
        if action == "setup" and hasattr(self, "content_stack"):
            self._update_painted_geometry_status()
            if is_expanded and not self._selecting_sidebar:
                self._sync_home_series_before_navigation()
                self.sidebar.select("preprocessing", "protocols")
                return
            if not is_expanded and self.selected_action in {
                "preprocessing",
                "setup",
                "geometry_facies",
                "image_selection",
            }:
                self.selected_action = None
                self.selected_checkbox_id = None
                self._sync_wizard_action_state()
            else:
                super()._on_sidebar_expanded(action, is_expanded)
        else:
            super()._on_sidebar_expanded(action, is_expanded)
        if not hasattr(self, "content_stack"):
            return
        if is_expanded:
            self._sync_home_series_before_navigation()
            self._set_method_category_visibility(action, "")
            if not self._selecting_sidebar:
                section = self.sidebar._sections[action]
                if action not in self.action_dispatch:
                    self.sidebar.select(action, next(iter(section._rows)))
                else:
                    self._current_category = action
                    self.series_header.show()
                    self.content_stack.setCurrentWidget(self.settings_scroll_area)
        self._update_run_action()
        self._refresh_scope_display()

    def _set_geometry_workflow_scope(self, item_id: str) -> None:
        """Route the geometry menu within its existing shared editor instance."""
        view = self.geometry_workspace.simple_view
        if view is None:
            return
        if not getattr(view, '_native_workflow_scope_connected', False):
            view.workflow_scope_requested.connect(lambda scope: self._select_row(
                ('geometry_facies', 'export' if scope == 'apply' else 'layers')))
            view._native_workflow_scope_connected = True
        scope = {'layers': 'layers', 'export': 'apply', 'rig_frame': 'rig_frame',
                 'analysis_boxes': 'analysis_boxes'}.get(item_id)
        if scope is not None:
            view.set_workflow_scope(scope)
        curvature = getattr(self, "curvature_workflow", None)
        host = getattr(view, "rig_curvature_host", None)
        if curvature is not None and host is not None:
            if curvature.widget.parentWidget() is not host:
                host.layout().addWidget(curvature.widget)
            host.setVisible(scope == "rig_frame")
            curvature.widget.setVisible(scope == "rig_frame")
            curvature.set_active(scope == "rig_frame")
        depth = getattr(self, "depth_workflow", None)
        depth_host = getattr(view, "rig_depth_host", None)
        depth_section = getattr(view, "step_depth", None)
        if depth is not None and depth_host is not None and depth_section is not None:
            if depth.panel.parentWidget() is not depth_host:
                depth_host.layout().addWidget(depth.panel)
                depth.panel.show()
            if not getattr(depth_section, "_native_depth_connected", False):
                depth_section.toggle.toggled.connect(lambda expanded: depth.set_embedded_active(
                    expanded and getattr(view, "_workflow_scope", None) == "rig_frame"))
                depth_section._native_depth_connected = True
            depth.set_embedded_active(scope == "rig_frame" and depth_section.is_expanded())
            if scope == "rig_frame" and getattr(self, "_pending_depth_section", False):
                self._open_embedded_depth()

    def _open_embedded_depth(self) -> None:
        view = self.geometry_workspace.simple_view
        section = getattr(view, "step_depth", None)
        if section is None:
            return
        self._pending_depth_section = False
        section.set_expanded(True)
        self.depth_workflow.set_embedded_active(True)
        view.steps_scroll.ensureWidgetVisible(section)

    def _on_sidebar_selection(self, action: str, item_id: str) -> None:
        if (action, item_id) in {("setup", "depth"), ("preprocessing", "depth_measurements")}:
            self._select_row(("setup", "depth"))
            return
        if (action, item_id) == ("geometry_controls", "curvature"):
            self._select_row(("geometry_facies", "rig_frame"))
            return
        self._pending_new_series_setup = None
        self._original_mass_controls = False
        curvature = getattr(self, "curvature_workflow", None)
        if curvature is not None and (action, item_id) != ("geometry_facies", "rig_frame"):
            curvature.set_active(False)
        depth = getattr(self, "depth_workflow", None)
        if depth is not None and (action, item_id) != ("geometry_facies", "rig_frame"):
            depth.set_embedded_active(False)
        if action == "legacy":
            self._current_category = action
            self._last_sidebar_selection = (action, item_id)
            self.selected_action, self.selected_checkbox_id = action, item_id
            # Opening this independent workspace must not queue a series action.
            self._pending_open_series = self._pending_series_navigation = None
            self.content_stack.setCurrentWidget(self.ensure_legacy_workflow())
            self._update_run_action()
            return
        if self.legacy_workspace is not None:
            self.legacy_workspace.set_active(False)
        self._sync_home_series_before_navigation()
        if getattr(self, '_series_load_error', '') and action != 'start':
            self._sync_series_display()
        if getattr(self, '_series_loading', False) and action != 'start':
            self._show_series_loading((action, item_id))
            return
        self._current_category = action
        self._last_sidebar_selection = (action, item_id)
        self.series_header.setVisible(action != "start")
        self._set_method_category_visibility(action, item_id)
        self._refresh_method_state()
        self._refresh_scope_display()
        open_page = action == "start" or (action, item_id) == ("jobs", "jobs")
        if not open_page and self._show_dataset_analysis():
            # A published dataset is read-only: pages that change the series stay closed.
            self.selected_action, self.selected_checkbox_id = "jobs", "new_analysis"
            self._update_run_action()
            return
        if (action, item_id) in {
            ("calibration", "mass"),
            ("phase_model", "mass_from_phase"),
        }:
            self.selected_action = action
            self.selected_checkbox_id = item_id
            self.show_common_mass_calibration()
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if (action, item_id) == ("method", "use_for_series"):
            self.selected_action, self.selected_checkbox_id = action, item_id
            self.content_stack.setCurrentWidget(self.method_panel)
            self.method_panel.show()
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if (action, item_id) == ("calibration", "saved"):
            self.selected_action, self.selected_checkbox_id = action, item_id
            self._show_saved_calibration()
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if (action, item_id) == ("calibration", "images"):
            self.selected_action, self.selected_checkbox_id = action, item_id
            self.calibration_image_selection.set_series(self._method_series_root())
            self.content_stack.setCurrentWidget(self.calibration_image_selection)
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if (action, item_id) == ("setup", "finish"):
            self.selected_action, self.selected_checkbox_id = action, item_id
            self.finish_setup_page.set_series(self._method_series_root())
            self.content_stack.setCurrentWidget(self.finish_setup_page)
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if (action, item_id) == ("calibration", "color") and self._saved_method.active == "phase_model":
            self.selected_action, self.selected_checkbox_id = action, item_id
            self.mass_calibration_display.show_setup_colour_paths(self._method_series_root())
            self.content_stack.setCurrentWidget(self.mass_calibration_display.phase_colour_paths)
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if (action, item_id) == ("calibration_controls", "interactive"):
            self._show_original_mass_calibration_controls("mass")
            return
        if (action, item_id) == ("jobs", "new_analysis") and self._saved_method.active == "colour_path":
            self.selected_action, self.selected_checkbox_id = "analysis", "mass"
            self._current_category = "analysis"
            self.content_stack.setCurrentWidget(self.settings_scroll_area)
            self._ensure_run_loaded()
            super()._on_sidebar_selection("analysis", "mass")
            self._refresh_scope_display()
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if (action, item_id) == ("image_selection", "plan"):
            self.selected_action, self.selected_checkbox_id = action, item_id
            self.image_selection_workflow.show()
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if (action, item_id) == ("image_selection", "production"):
            self.selected_action, self.selected_checkbox_id = action, item_id
            self.production_image_selection.show()
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if action in self.action_dispatch:
            self.content_stack.setCurrentWidget(self.settings_scroll_area)
            self._ensure_run_loaded()
            super()._on_sidebar_selection(action, item_id)
        else:
            self.selected_action = action
            self.selected_checkbox_id = item_id
            if action == "start":
                self.content_stack.setCurrentWidget(self.home_page)
                if not self._series_loading and not getattr(self, '_returning_from_series_loading', False):
                    self._refresh_series()
            elif action == "geometry_facies":
                self.content_stack.setCurrentWidget(self.geometry_workspace)
                self.geometry_workspace.ensure_page()
                self._set_geometry_workflow_scope(item_id)
                view = self.geometry_workspace.simple_view
                target = getattr(
                    view,
                    {
                        "rig_frame": "rig_panel",
                        "layers": "canvas",
                        "export": "chart_review_button",
                    }[item_id],
                    None,
                )
                if target is not None:
                    target.setFocus()
            elif action == "analysis_boxes":
                self.content_stack.setCurrentWidget(self.geometry_workspace)
                self.geometry_workspace.ensure_page()
                view = self.geometry_workspace.simple_view
                if view is not None:
                    self._set_geometry_workflow_scope("analysis_boxes")
                    view.analysis_boxes.widget.setFocus()
            elif action == "overflow_data":
                self.overflow_data_page.show_series(self._method_series_root())
                self.content_stack.setCurrentWidget(self.overflow_data_page)
            elif action == "events":
                self.events_page.set_series(self._method_series_root(), self.phase_model_workspace.series_id)
                self.content_stack.setCurrentWidget(self.events_page)
            elif (action, item_id) == ("phase_model", "calibration_review"):
                self._show_mass_analysis_view("review")
            elif (action, item_id) == ("phase_model", "checks"):
                self.content_stack.setCurrentWidget(self.ensure_calibration_checks())
            elif (action, item_id) == ("phase_model", "overflow"):
                self._show_overflow_use()
            elif action in {"phase_model", "method"}:
                workspace = self.phase_model_workspace
                if item_id == "selection_plan":
                    if workspace.series_id:
                        self.open_calibration_image_selection()
                    else:
                        self.print_log(
                            "Choose a series on Start before planning images."
                        )
                    self._sync_wizard_action_state()
                    self._update_run_action()
                    return
                # Select Training even without a series, without a lazy Studio load.
                mode = "mass" if item_id == "mass_from_phase" else "training"
                with QSignalBlocker(workspace.calibration_sidebar):
                    workspace.select_calibration(mode)
                workspace._requested_mode = mode
                # Setup preparation and Calibration share the same page object.
                workspace._requested_section = PHASE_PAGE_SECTIONS.get(
                    (action, item_id), "phase_model"
                )
                self.content_stack.setCurrentWidget(workspace)
                if workspace.series_id:
                    workspace.ensure_studio()
                workspace._show_selected_studio_mode()
            elif (action, item_id) == ("jobs", "production_results"):
                self.content_stack.setCurrentWidget(self.ensure_production_results())
            elif action == "jobs":
                jobs = self.ensure_jobs()
                if item_id == "new_analysis":
                    jobs.new_analysis.set_context(method_text=self._series_method_text())
                jobs.show_view(item_id if item_id in {"new_analysis", "wasserstein"} else "jobs")
                if item_id == "sources":
                    jobs.sources.setChecked(True)
                self.content_stack.setCurrentWidget(jobs)
            elif action == "mass_analysis":
                if item_id == "review":
                    self._open_existing_phase_review()
                    self._sync_wizard_action_state()
                    self._update_run_action()
                    return
                self._show_mass_analysis_view("results")
            self._sync_wizard_action_state()
        self._update_run_action()

    def ensure_legacy_workflow(self) -> QWidget:
        if self.legacy_workspace is None:
            from .native_legacy_workspace import NativeLegacyWorkspace

            self.legacy_workspace = NativeLegacyWorkspace(self)
            self.content_stack.addWidget(self.legacy_workspace)
            self.content_stack.currentChanged.connect(lambda _index: (
                self.legacy_workspace.set_active(
                    self.content_stack.currentWidget() is self.legacy_workspace
                )
            ))
        return self.legacy_workspace

    def _on_open_full_config(self) -> None:
        self.series_header.hide()
        self.method_panel.hide()
        self.analysis_method_badge.hide()
        self.analysis_execution_note.hide()
        self.frozen_reference_line.show_for_analysis(False)
        self.scope_label.hide()
        self.rig_label.hide()
        self.run_combo.setEnabled(True)
        self.content_stack.setCurrentWidget(self.settings_scroll_area)
        self.status_header.hide()
        self.series_header.hide()
        super()._on_open_full_config()
        self._update_run_action()

    # -------------------------------------------------------------- run files

    def _set_run_controls_visible(self, visible: bool) -> None:
        for widget in (self.run_label, self.run_combo, self.run_path):
            widget.setVisible(visible)

    def _populate_runs(self) -> None:
        """List the runs of the selected series and the file each one opens."""
        if getattr(self, '_series_loading', False):
            return
        workspace = self.phase_model_workspace
        series_id = workspace.series_id
        root = workspace.series_roots.get(series_id)
        snapshot = self._current_series_snapshot()
        choices = snapshot['choices'] if snapshot is not None else (run_configurations(root) if root is not None else [])
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run, path, note in choices:
            self.run_combo.addItem(run, str(path) if path is not None else "")
            index = self.run_combo.count() - 1
            self.run_combo.setItemData(
                index,
                f"{path}\n{note}" if path is not None else note,
                Qt.ItemDataRole.ToolTipRole,
            )
            if path is None:
                self.run_combo.model().item(index).setEnabled(False)
        current = self._run_index(self.config_file)
        if current < 0 or self._config_series != series_id:
            remembered = QSettings().value(f"native_app/run/{series_id}", "", type=str)
            current = self.run_combo.findText(remembered) if remembered else -1
            if current >= 0 and not self.run_combo.itemData(current):
                current = -1
        self.run_combo.setCurrentIndex(current)
        self.run_combo.blockSignals(False)
        self._update_run_path()
        self._update_setup_action()
        self._refresh_scope_display()

    def _run_index(self, path: str | None) -> int:
        for index in range(self.run_combo.count()):
            candidate = self.run_combo.itemData(index)
            if path and candidate and os.path.normcase(os.path.abspath(candidate)) == os.path.normcase(os.path.abspath(path)):
                return index
        return -1

    def _update_run_path(self) -> None:
        path = self.config_file
        if not path:
            self.run_path.setText("No configuration open")
            self.run_path.setToolTip("")
            return
        root = self.phase_model_workspace.series_roots.get(
            self.phase_model_workspace.series_id
        )
        shown = str(path)
        if root is not None:
            try:
                shown = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root))).as_posix()
            except (OSError, ValueError):
                pass
        if self.run_combo.currentIndex() < 0 or not _same_file(
            self.run_combo.currentData(), path
        ):
            shown = f"Open: {shown}"
        self.run_path.setText(shown)
        self.run_path.setToolTip(str(path))

    def _run_chosen(self, index: int) -> None:
        path = self.run_combo.itemData(index)
        if not path:
            return
        series_id = self.phase_model_workspace.series_id
        QSettings().setValue(
            f"native_app/run/{series_id}", self.run_combo.itemText(index)
        )
        self._load_run_config(path)
        self._refresh_scope_display()

    def _load_run_config(self, path: str) -> bool:
        """Open a run without letting a failed load reuse the previous config."""
        previous = (
            self.config_dict,
            self.config_file,
            self.config_path_label.text(),
            self._config_series,
            getattr(self, "_config_stamp", None),
            self._last_settings_view,
        )
        errors = []

        def loading_message(text: str) -> None:
            # ConfigController catches loading/rendering errors and returns None
            # for both success and failure. Preserve that controller unchanged.
            if text.startswith("Error loading config file:"):
                errors.append(text)

        self.log_message.connect(loading_message)
        try:
            self.config_path_label.setText(str(path))
            snapshot = self._current_series_snapshot()
            prepared = snapshot['configs'].get(str(path)) if snapshot is not None else None
            if prepared is None:
                self.config_controller.load_config()
            else:
                values, stamp, error = prepared
                if error:
                    raise ValueError(error)
                self.config_dict = deepcopy(values)
                self.config_file = str(path)
                self.sidebar.deselect_all()
                self.clear_selection()
                self.settings_inputs.clear()
                self.settings_factory.display_full_settings()
                self._config_stamp = (str(path), stamp)
                self._config_loaded()
        except Exception as error:
            errors.append(str(error))
            self.print_log(f"Error loading config file: {error}")
        finally:
            self.log_message.disconnect(loading_message)
        loaded = (
            not errors
            and self.config_dict is not previous[0]
            and _same_file(self.config_file, path)
        )
        if not loaded:
            # Rendering can fail after the controller replaced both the config
            # and its widgets. Never sync the new widgets into the old config.
            self.settings_inputs.clear()
            (
                self.config_dict,
                self.config_file,
                label,
                self._config_series,
                self._config_stamp,
                self._last_settings_view,
            ) = previous
            self.config_path_label.setText(label)
            if self._last_settings_view is None:
                self.settings_factory._render_settings_tabs({})
            else:
                self.settings_factory.refresh_current_view()
            self._update_run_path()
            self._update_run_action()
            self._refresh_scope_display()
            return False
        self._restore_selection()
        return True

    def _ensure_run_loaded(self) -> None:
        """Open the chosen run when no file of the selected series is open."""
        path = self.run_combo.currentData()
        if not path:
            return
        if (
            not self.config_file
            or self._config_series != self.phase_model_workspace.series_id
        ):
            self._load_run_config(path)

    def _config_loaded(self) -> None:
        """Called after DarSIA opened a configuration (see guard_config_file)."""
        if not self.config_file:
            return
        self._config_series = self.phase_model_workspace.series_id
        index = self._run_index(self.config_file)
        if index >= 0:
            self.run_combo.blockSignals(True)
            self.run_combo.setCurrentIndex(index)
            self.run_combo.blockSignals(False)
        self._update_run_path()
        self._refresh_method_state()
        self._refresh_scope_display()
        if getattr(self, "depth_workflow", None) is not None:
            self.depth_workflow.refresh_if_visible()
        curvature = getattr(self, "curvature_workflow", None)
        if curvature is not None and (
            self.selected_action, self.selected_checkbox_id
        ) == ("geometry_facies", "rig_frame"):
            curvature.refresh()

    def abort_selected_workflow(self) -> None:
        depth = getattr(self, "depth_workflow", None)
        if depth is not None and depth.sequence.running:
            depth.sequence.abort()
            return
        if self._all_runs_running():
            self.all_runs.sequence.abort()
            return
        super().abort_selected_workflow()

    def run_selected_workflow(self) -> None:
        """Play applies only to the original DarSIA categories."""
        if (
            getattr(self, "depth_workflow", None) is not None
            and self.depth_workflow.sequence.running
        ):
            return
        if self._darsia_visible():
            super().run_selected_workflow()

    def show_phase_model_workflow(self) -> None:
        self._select_row(("phase_model", "training"))

    def show_mass_calibration(self) -> None:
        self._select_row(("phase_model", "mass_from_phase"))

    def show_mass_analysis(self) -> None:
        self._select_row(("mass_analysis", "frozen"))


def create_application() -> tuple[QApplication, bool]:
    """Use the upstream saved theme and an existing QApplication when embedded."""
    existing = QApplication.instance()
    app = existing if existing is not None else QApplication(sys.argv[:1])
    if existing is None:
        app.setOrganizationName("DarSIA")
        app.setApplicationName("DarSIA GUI")
    apply_theme(app, get_theme())
    return app, existing is None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series", choices=[key for key, _label, _root in configured_series()]
    )
    parser.add_argument(
        "--tab",
        choices=(
            "home",
            "setup",
            "protocols",
            "geometry",
            "facies",
            "native-setup",
            "color",
            "color-path",
            "original",
            "phase",
            "mass",
            "analysis",
            "mass-analysis",
            "darsia",
            "calibration",
            "fluidflower",
        ),
        default="home",
    )
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--exit-after-ms", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from darsia.fluidflower.machine_paths import add_machine_path_action, install_machine_path_map

    install_machine_path_map()
    args = build_parser().parse_args(argv)
    app, owns_app = create_application()
    window = NativeFluidFlowerMainWindow(initial_series=args.series)
    add_machine_path_action(window)
    from darsia.fluidflower.helper_process import add_helper_process_action, install as install_helper

    # Pages read their files in a helper process when the user turns it on (Settings).
    install_helper(app if owns_app else None, log=window.print_log)
    add_helper_process_action(window)
    if args.tab == "setup":
        window.show_setup()
    elif args.tab == "protocols":
        window.show_setup(step="protocols")
    elif args.tab == "native-setup":
        window.show_setup(step="native")
    elif args.tab in {"geometry", "facies"}:
        window.show_geometry_facies()
    elif args.tab in {"color", "color-path"}:
        window.show_color_path()
    elif args.tab in {"original", "darsia"}:
        window.show_original_workflow()
    elif args.tab in {"phase", "calibration", "fluidflower"}:
        window.show_phase_model_workflow()
    elif args.tab == "mass":
        window.show_mass_calibration()
    elif args.tab in {"analysis", "mass-analysis"}:
        window.show_mass_analysis()
    window.show()
    # Keep a reference when called from an already running Qt application.
    app._native_fluidflower_window = window

    if args.screenshot is not None:
        target = args.screenshot.expanduser().resolve()

        def save_screenshot() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            app.processEvents()
            window.repaint()
            if not window.grab().save(str(target)):
                raise RuntimeError(f"Could not save screenshot to {target}")
            if not args.exit_after_ms:
                app.quit()

        # The Studio performs its first model-catalog discovery on opening.
        # Let that finish before a QA screenshot of the phase-model tab.
        QTimer.singleShot(
            (
                5000
                if args.tab
                in {
                    "phase",
                    "calibration",
                    "fluidflower",
                    "mass",
                    "analysis",
                    "mass-analysis",
                }
                else 500
            ),
            save_screenshot,
        )
    if args.exit_after_ms:
        QTimer.singleShot(max(1, args.exit_after_ms), app.quit)
    return int(app.exec()) if owns_app else 0


if __name__ == "__main__":
    raise SystemExit(main())
