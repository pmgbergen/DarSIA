"""Native workspace for the geometry and facies of a registered series.

The workspace opens a simple view for the series selected on Start: paint
layers on the reference run's raw baseline photograph, let a model propose the
boundaries, group layers into facies and export them as native Setup inputs.
The full geometry-and-facies page (optical drafts, material examples, boundary
edits, training data and facies models) is the advanced view. The project
package lives in the series folder and is created on request.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from . import geometry_workspace_frame as workspace_frame
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.series_geometry import (
    NativeSetupExport,
    SeriesRunInputs,
    create_series_geometry_project,
    default_actor,
    export_native_setup,
    geometry_package_root,
    load_series_geometry_project,
    native_series_link,
    series_rig_class,
    series_run_inputs,
    set_reference_run,
    set_series_rig_class,
)
from darsia.fluidflower.templates import RIG_CLASSES

_PAGE_THREADS = (
    "_thread",
    "_facies_export_thread",
    "_facies_model_thread",
    "_facies_example_thread",
)
# Series folders do not use the project quality gates; revisions and their
# validation remain visible in the page itself.
_GATE_WIDGETS = ("status", "dependencies", "start", "review", "approve", "complete")
_NATIVE_DESCRIPTION = (
    "Draw layer boundaries on the raw baseline photograph of the reference run, "
    "group layers into facies and give them porosity and permeability. Native "
    "Setup corrects the labels with each run's rig corrections."
)


class _ExportWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        project: FluidFlowerProject,
        reference: SeriesRunInputs | None,
        actor: str,
    ) -> None:
        super().__init__()
        self.project = project
        self.reference = reference
        self.actor = actor

    @Slot()
    def run(self) -> None:
        try:
            result = export_native_setup(
                self.project, reference=self.reference, actor=self.actor
            )
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class NativeSetupExportDialog(QDialog):
    """Show where the export went and the sections to add to the configuration."""

    def __init__(
        self, export: NativeSetupExport, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.export = export
        self.setWindowTitle("Exported for native Setup")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        reused = " (existing export of this revision)" if export.reused else ""
        summary = QLabel(
            f"{export.region_count} regions in {export.facies_count} facies"
            f"{reused}.<br>Label grid: {export.grid}.<br>Folder: {export.destination}"
        )
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        summary.setWordWrap(True)
        layout.addWidget(summary)
        if export.warnings:
            warnings = QLabel("Check: " + " ".join(export.warnings))
            warnings.setWordWrap(True)
            warnings.setObjectName("integrationNotice")
            layout.addWidget(warnings)
        note = QLabel(
            "Add these sections to config/common.toml for all runs, or to one run "
            "file. Then run Segment colored image, Setup facies and Setup rig in "
            "Setup. Native Setup derives the same labels from the colored image."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.text = QPlainTextEdit(export.toml_text)
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.copy_button = buttons.addButton(
            "Copy sections", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.open_button = buttons.addButton(
            "Open folder", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.copy_button.clicked.connect(self.copy_sections)
        self.open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.export.destination))
            )
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def copy_sections(self) -> None:
        QGuiApplication.clipboard().setText(self.export.toml_text)


def series_page_inputs(root: Path | None) -> dict[str, Any]:
    """Read what the geometry page of the series in ``root`` opens with.

    Returns the rig class, the run inputs (files are not checked), the
    geometry project (None when there is none), the checked inputs of its
    reference run, the geometry package folder and the problems found. It
    reads files only and changes nothing, so it can run off the GUI thread.
    """

    inputs: dict[str, Any] = dict(rig_class="", run_inputs=[], project=None, checked={},
                                  package=None, problems=[])
    if root is None:
        return inputs
    inputs["rig_class"] = series_rig_class(root)
    inputs["package"] = geometry_package_root(root)
    try:
        # Image and rig files are only checked for the selected run.
        inputs["run_inputs"] = series_run_inputs(root, check_files=False)
    except (OSError, ValueError, KeyError) as error:
        inputs["problems"].append(f"Could not read the series configuration: {error}")
    try:
        inputs["project"] = load_series_geometry_project(root)
    except (OSError, ValueError) as error:
        inputs["problems"].append(f"Could not open the geometry workspace: {error}")
    if inputs["project"] is not None:
        reference = str(native_series_link(inputs["project"]).get("reference_run") or "")
        if any(row.run_id == reference for row in inputs["run_inputs"]):
            try:
                rows = series_run_inputs(root, runs=[reference])
            except (OSError, ValueError, KeyError):
                rows = []
            if rows:
                inputs["checked"][reference] = rows[0]
    return inputs


class SeriesGeometryWorkspace(QWidget):
    """Geometry and facies of the series selected on Start."""

    reference_changed = Signal(str)
    # The rig class recorded in series.toml (a key of RIG_CLASSES).
    rig_class_changed = Signal(str)

    def __init__(
        self,
        *,
        series: Sequence[tuple[str, str, Path]] = (),
        initial_series: str = "",
    ) -> None:
        super().__init__()
        self.setObjectName("seriesGeometryWorkspace")
        self.series_roots: dict[str, Path] = {}
        self.series_labels: dict[str, str] = {}
        self._set_series_table(series)
        self.series_id = initial_series or (series[0][0] if series else "")
        self.project: FluidFlowerProject | None = None
        self.page: QWidget | None = None
        self.run_inputs: list[SeriesRunInputs] = []
        self._checked_inputs: dict[str, SeriesRunInputs] = {}
        self._loaded_series: str | None = None
        # Folder of the loaded series; series_root may already name the next one
        # while a job of the loaded series finishes.
        self._loaded_root: Path | None = None
        self._reference_shown = ""
        self._prefilled: dict[str, str] = {}
        self._export_thread: QThread | None = None
        self._export_worker: _ExportWorker | None = None
        self.export_dialog: NativeSetupExportDialog | None = None
        # Rig frame panel of the advanced page; the simple view has its own.
        self.rig_panel: QWidget | None = None
        self.simple_view: Any = None
        self._rig_class_key = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        controls = QHBoxLayout()
        self.series_label = QLabel()
        controls.addWidget(self.series_label)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Rig class"))
        self.rig_class = QComboBox()
        self.rig_class.setToolTip(
            "Class of the rig used in this series, with its nominal inner size. "
            "Stored in series.toml and shown on Start."
        )
        self.rig_class.setMinimumWidth(150)
        controls.addWidget(self.rig_class)
        controls.addSpacing(12)
        self.reference_run_label = QLabel("Reference run")
        controls.addWidget(self.reference_run_label)
        self.reference_run = QComboBox()
        self.reference_run.setToolTip(
            "Run whose raw baseline photograph the layers are drawn on. Native "
            "Setup corrects the labels with each run's rig corrections."
        )
        self.reference_run.setMinimumWidth(90)
        controls.addWidget(self.reference_run)
        self.advanced_button = QPushButton("Advanced view")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setToolTip(
            "The full geometry-and-facies page: optical drafts, material examples, "
            "boundary edits, layer tables, training data and facies models"
        )
        self.advanced_button.setEnabled(False)
        self.advanced_button.toggled.connect(self.show_advanced)
        controls.addWidget(self.advanced_button)
        self.export_button = QPushButton("Export to native Setup…")
        self.export_button.setToolTip(
            "Write the colored label image, facies properties and the [labeling] "
            "and [facies] sections for the current saved revision"
        )
        self.export_button.setEnabled(False)
        self.export_button.hide()
        controls.addWidget(self.export_button)
        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.setToolTip("Open the geometry workspace folder")
        self.open_folder_button.setEnabled(False)
        controls.addWidget(self.open_folder_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.status = QLabel()
        self.status.setObjectName("secondaryText")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.status)

        self.pages = QStackedWidget()
        self.create_panel = QWidget()
        create_layout = QVBoxLayout(self.create_panel)
        create_layout.setContentsMargins(0, 12, 0, 0)
        heading = QLabel("Geometry and facies")
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        create_layout.addWidget(heading)
        self.create_text = QLabel()
        self.create_text.setWordWrap(True)
        create_layout.addWidget(self.create_text)
        self.create_button = QPushButton("Create geometry workspace")
        create_row = QHBoxLayout()
        create_row.addWidget(self.create_button)
        create_row.addStretch(1)
        create_layout.addLayout(create_row)
        create_layout.addStretch(1)
        self.pages.addWidget(self.create_panel)
        self.page_host = QWidget()
        host_layout = QVBoxLayout(self.page_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        self.view_stack = QStackedWidget()
        host_layout.addWidget(self.view_stack)
        self.advanced_host = QWidget()
        self.page_layout = QVBoxLayout(self.advanced_host)
        self.page_layout.setContentsMargins(0, 0, 0, 0)
        self.view_stack.addWidget(self.advanced_host)
        self.pages.addWidget(self.page_host)
        # Shown while a slow reading of the series' files runs (ensure_page).
        self.loading_panel = QWidget()
        loading_layout = QVBoxLayout(self.loading_panel)
        loading_layout.setContentsMargins(0, 12, 0, 0)
        self.loading_text = QLabel()
        self.loading_text.setWordWrap(True)
        loading_layout.addWidget(self.loading_text)
        loading_layout.addStretch(1)
        self.pages.addWidget(self.loading_panel)
        layout.addWidget(self.pages, 1)

        self.create_button.clicked.connect(self.create_workspace)
        self.export_button.clicked.connect(self.start_export)
        self.open_folder_button.clicked.connect(self._open_folder)
        self.reference_run.currentIndexChanged.connect(self._reference_selected)
        self.rig_class.currentIndexChanged.connect(self._rig_class_selected)
        self._update_series_label()
        self._load_rig_class()
        self._show_create_panel()

    # ------------------------------------------------------------------ series

    def _set_series_table(self, series: Sequence[tuple[str, str, Path]]) -> None:
        self.series_roots = {key: Path(root) for key, _label, root in series}
        self.series_labels = {key: label for key, label, _root in series}

    def _update_series_label(self) -> None:
        label = self.series_labels.get(self.series_id, "")
        self.series_label.setText(f"Series: {label}" if label else "No series selected")

    def update_series(
        self, series: Sequence[tuple[str, str, Path]], *, selected: str
    ) -> None:
        loaded_root = self.series_roots.get(self._loaded_series or "")
        self._set_series_table(series)
        if self._loaded_series is not None and (
            self.series_roots.get(self._loaded_series) != loaded_root
        ):
            self._loaded_series = None
        self.set_series(selected)

    def set_series(self, series_id: str) -> None:
        self.series_id = str(series_id or "")
        self._update_series_label()
        if self.isVisible():
            self.ensure_page()

    # --------------------------------------------------------------- rig class

    def _load_rig_class(self, key: str | None = None) -> None:
        """Show the rig class recorded in the selected series' series.toml.

        ``key`` is that class when it has already been read.
        """

        root = self.series_root
        if key is None:
            key = series_rig_class(root) if root is not None else ""
        box = self.rig_class
        box.blockSignals(True)
        try:
            box.clear()
            if not key:
                box.addItem("Not set", "")
            for rig in RIG_CLASSES.values():
                box.addItem(rig.title(), rig.key)
                box.setItemData(
                    box.count() - 1,
                    rig.source or rig.label,
                    Qt.ItemDataRole.ToolTipRole,
                )
            box.setCurrentIndex(max(0, box.findData(key)))
            box.setEnabled(root is not None)
        finally:
            box.blockSignals(False)
        self._rig_class_key = key
        if self.simple_view is not None:
            self.simple_view.rig_panel.set_rig_class(key)
        if self.rig_panel is not None and hasattr(self.rig_panel, "set_rig_class"):
            self.rig_panel.set_rig_class(key)

    def rig_class_key(self) -> str:
        return self._rig_class_key

    def set_rig_class(self, key: str) -> bool:
        """Record the series' rig class in series.toml; False if it failed."""

        root = self.series_root
        if root is None or key not in RIG_CLASSES:
            return False
        if key == self._rig_class_key:
            return True
        try:
            set_series_rig_class(root, key)
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self,
                "Could not record the rig class",
                f"{error}\n\nThe rig class is stored in {Path(root) / 'series.toml'}.",
            )
            self._load_rig_class()
            return False
        if self.project is not None:
            current = str((self.project.manifest.get("rig") or {}).get("type") or "")
            if current != key:
                try:
                    self.project.update_rig_type(
                        key,
                        updated_by=default_actor(),
                        reason=f"Rig class of the series: {RIG_CLASSES[key].label}.",
                    )
                except (OSError, ValueError):
                    pass  # the workspace keeps its earlier apparatus class
        self._load_rig_class()
        self.rig_class_changed.emit(key)
        return True

    def _rig_class_selected(self, _index: int) -> None:
        key = str(self.rig_class.currentData() or "")
        if key:
            self.set_rig_class(key)

    @property
    def series_root(self) -> Path | None:
        return self.series_roots.get(self.series_id)

    def busy(self) -> bool:
        dialog = getattr(self, "export_dialog", None)
        if getattr(dialog, "_thread", None) is not None:
            return True
        if self.simple_view is not None and self.simple_view.busy():
            return True
        return self._page_busy() or self._thread_running(self._export_thread)

    @staticmethod
    def _thread_running(thread: QThread | None) -> bool:
        try:
            return thread is not None and thread.isRunning()
        except RuntimeError:  # deleted after finishing
            return False

    def _page_busy(self) -> bool:
        return any(
            self._thread_running(getattr(self.page, name, None))
            for name in _PAGE_THREADS
        )

    def ensure_page(self) -> QWidget:
        """Load the selected series; keep the current one while a job runs.

        The series' files are read off the GUI thread (series_page_inputs). A
        quick reading opens the page at once. A slow one, such as many run
        files on a series share, shows a loading note until it answers
        (_page_inputs_read).
        """

        if self._loaded_series == self.series_id:
            return self.pages.currentWidget()
        if self.busy():
            self.status.setText(
                "A geometry job is still running for "
                f"{self.series_labels.get(self._loaded_series or '', 'the previous series')}"
                ". The selected series opens when it has finished."
            )
            return self.pages.currentWidget()
        root = self.series_root
        key = (self.series_id, root)
        reader = self.__dict__.get("_page_reader")
        if reader is None:
            from .background_read import BackgroundRead
            reader = self._page_reader = BackgroundRead(self)
            reader.answered.connect(self._page_inputs_read)
        if reader.busy() and reader.key == key:
            return self.pages.currentWidget()
        self._discard_page()
        self.project = None
        self.run_inputs = []
        self._checked_inputs = {}
        self._loaded_series = None
        self._loaded_root = None
        done, inputs, error = reader.read(key, lambda: series_page_inputs(root))
        if done:
            self._open_page(key, inputs, error)
        else:
            label = self.series_labels.get(self.series_id, "the series")
            self.loading_text.setText(f"Reading the runs and geometry of {label}…")
            self.pages.setCurrentWidget(self.loading_panel)
            self._refresh_actions()
        return self.pages.currentWidget()

    def _page_inputs_read(self, key, inputs, error) -> None:
        """Open the page of a slow reading, unless another series was chosen meanwhile."""

        if key == (self.series_id, self.series_root) and self._loaded_series is None:
            self._open_page(key, inputs, error)

    def _open_page(self, key, inputs, error) -> None:
        """Show the series' page from series_page_inputs' answer."""

        series_id, root = key
        if error:
            inputs = dict(rig_class=None, run_inputs=[], project=None, checked={}, package=None,
                          problems=[f"Could not read the series: {error}"])
        self._loaded_series = series_id
        self._loaded_root = root
        self._package_root = (root, inputs["package"])
        self._load_rig_class(inputs["rig_class"])
        if root is None:
            self._set_reference_runs([], "")
            self._show_create_panel()
            return
        self.run_inputs = list(inputs["run_inputs"])
        self._checked_inputs = dict(inputs["checked"])
        self.project = inputs["project"]
        preferred = ""
        if self.project is not None:
            preferred = str(native_series_link(self.project).get("reference_run") or "")
        self._set_reference_runs(self.run_inputs, preferred)
        if self.project is None:
            self._show_create_panel(inputs["problems"])
        else:
            self._show_page()
            if inputs["problems"]:
                self.status.setText(" ".join(inputs["problems"]))

    def _geometry_package(self, root: Path) -> Path:
        """The series' geometry package folder; read with the page when it opened."""

        known = self.__dict__.get("_package_root")
        if known is not None and known[0] == root and known[1] is not None:
            return known[1]
        return geometry_package_root(root)

    def _discard_page(self) -> None:
        self._discard_advanced_page()
        if self.simple_view is not None:
            self.simple_view.wait_for_workers()
            self.view_stack.removeWidget(self.simple_view)
            self.simple_view.hide()
            self.simple_view.deleteLater()
            self.simple_view = None
        self.advanced_button.blockSignals(True)
        self.advanced_button.setChecked(False)
        self.advanced_button.blockSignals(False)
        self.advanced_button.setEnabled(False)
        self.export_button.hide()

    def _discard_advanced_page(self) -> None:
        self.rig_panel = None
        if self.page is not None:
            self.page_layout.removeWidget(self.page)
            self.page.hide()
            self.page.deleteLater()
            self.page = None
        self._prefilled = {}

    # --------------------------------------------------------------- reference

    def _set_reference_runs(
        self, rows: Sequence[SeriesRunInputs], preferred: str
    ) -> None:
        self.reference_run.blockSignals(True)
        self.reference_run.clear()
        for row in rows:
            self.reference_run.addItem(row.run_id, row.run_id)
        index = self.reference_run.findData(preferred)
        self.reference_run.setCurrentIndex(max(0, index) if rows else -1)
        self.reference_run.setEnabled(bool(rows))
        self.reference_run.blockSignals(False)
        self._reference_shown = str(self.reference_run.currentData() or "")
        self._update_status()

    def reference(self) -> SeriesRunInputs | None:
        run_id = str(self.reference_run.currentData() or "")
        listed = next((row for row in self.run_inputs if row.run_id == run_id), None)
        root = self._loaded_root
        if listed is None or root is None:
            return listed
        if run_id not in self._checked_inputs:
            try:
                rows = series_run_inputs(root, runs=[run_id])
            except (OSError, ValueError, KeyError):
                rows = []
            if not rows:
                return listed
            self._checked_inputs[run_id] = rows[0]
        return self._checked_inputs[run_id]

    def _reference_selected(self, _index: int) -> None:
        if self.simple_view is not None and self.simple_view.busy():
            # The running proposal or save belongs to the shown run.
            self.reference_run.blockSignals(True)
            self.reference_run.setCurrentIndex(
                max(0, self.reference_run.findData(self._reference_shown))
            )
            self.reference_run.blockSignals(False)
            self.status.setText(
                "Wait until the proposal or export has finished before choosing "
                "another reference run."
            )
            return
        reference = self.reference()
        if reference is None:
            return
        self._reference_shown = reference.run_id
        if self.project is not None:
            try:
                set_reference_run(self.project, reference.run_id, actor=default_actor())
            except (OSError, ValueError) as error:
                QMessageBox.warning(
                    self, "Could not save the reference run", str(error)
                )
        if self.simple_view is not None:
            self.simple_view.reload()
        if self.page is not None:
            self._prefill(self.page, reference)
            if self.rig_panel is not None:
                self.rig_panel.load()
        self._update_status()
        self.reference_changed.emit(reference.run_id)

    def _update_status(self) -> None:
        root = self.series_root
        if root is None:
            self.status.setText("")
            return
        parts: list[str] = []
        reference = self.reference()
        if reference is not None:
            frame_error = ""
            try:
                config = workspace_frame.current_configuration(self.project) if self.project is not None else {}
            except (OSError, ValueError) as error:
                config = {}
                frame_error = str(error)
            if frame_error:
                parts.append("Workspace coordinate frame unavailable: " + frame_error)
            elif workspace_frame.frame_kind(config) == "rig-corrected":
                parts.append(f"Saved corrected rig frame of {reference.run_id}. Original production maps are unchanged; production export and Build are not qualified yet.")
            elif reference.baseline_image is not None:
                parts.append(
                    f"Raw baseline of {reference.run_id}: {reference.baseline_image.name}."
                )
            if frame_error or workspace_frame.frame_kind(config) == "rig-corrected":
                pass
            elif reference.rig_folder is not None:
                parts.append(
                    "A native rig exists for this run; its maps are on the corrected "
                    "grid and cannot be exported back to Setup."
                )
            else:
                parts.append(
                    "Draw the layers on the raw baseline photograph; native Setup "
                    "corrects them with the rig corrections."
                )
            parts.extend(reference.issues)
        package = self._geometry_package(root)
        parts.append(f"Workspace folder: {package}")
        self.status.setText(" ".join(parts))
        self.open_folder_button.setEnabled(self.project is not None)
        self._refresh_actions()

    # ------------------------------------------------------------------ pages

    def _show_create_panel(self, problems: Sequence[str] = ()) -> None:
        root = self.series_root
        if root is None:
            self.create_text.setText(
                "Add an existing series folder on Start before working on its "
                "geometry and facies."
            )
            self.create_button.hide()
        else:
            package = self._geometry_package(root)
            self.create_text.setText(
                "Mark sand layers on a raw baseline photograph, let a model propose "
                "the boundaries, group layers into facies with shared properties and "
                "export the result to native Setup. This series has no geometry "
                f"workspace yet; it will be created in {package}. Images and "
                "configuration files are not changed."
                + ("\n\n" + "\n".join(problems) if problems else "")
            )
            self.create_button.show()
            self.create_button.setEnabled(True)
        self.pages.setCurrentWidget(self.create_panel)
        self._refresh_actions()

    def create_workspace(self) -> None:
        root = self.series_root
        if root is None or self.project is not None:
            return
        try:
            self.project = create_series_geometry_project(
                series_id=self.series_id,
                series_name=self.series_labels.get(self.series_id, self.series_id),
                series_root=root,
                rig_type=self._rig_class_key or "custom",
                reference=self.reference(),
                created_by=default_actor(),
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self, "Could not create the geometry workspace", str(error)
            )
            return
        self._show_page()
        self._update_status()

    def _show_page(self) -> None:
        """Open the simple view; the advanced page is created when needed."""

        from darsia.fluidflower.series_layer_view import LayerPainterView

        assert self.project is not None
        self._discard_page()
        view = LayerPainterView(self.project, self.reference,
            run_inputs=lambda: self.run_inputs, set_reference=self._set_painting_reference)
        view.rig_panel.set_rig_class(self._rig_class_key)
        view.exported.connect(self._show_export_dialog)
        view.busy_changed.connect(self._view_busy_changed)
        self.simple_view = view
        if getattr(self, "local_setup_build_request", None) is not None:
            from .local_setup_ui import LocalSetupController
            self.setup_jobs_controller = LocalSetupController(self, view)
        else:
            from .series_setup_jobs_ui import SeriesSetupJobsController
            self.setup_jobs_controller = SeriesSetupJobsController(self, view)
        self.view_stack.insertWidget(0, view)
        self.view_stack.setCurrentWidget(view)
        self.pages.setCurrentWidget(self.page_host)
        self.advanced_button.setEnabled(True)
        self._refresh_actions()

    def show_advanced(self, advanced: bool) -> None:
        """Switch between the simple view and the full page."""

        if self.project is None:
            return
        self.advanced_button.blockSignals(True)
        self.advanced_button.setChecked(bool(advanced))
        self.advanced_button.blockSignals(False)
        if advanced:
            if self.simple_view is not None:
                self.simple_view.save_session()
            if self.page is None or not self._page_busy():
                # Recreated so that it shows the latest revision.
                self._create_advanced_page()
            self.view_stack.setCurrentWidget(self.advanced_host)
            self.export_button.show()
        else:
            if self.simple_view is not None:
                self.view_stack.setCurrentWidget(self.simple_view)
                self.simple_view.rig_panel.load()
                self.simple_view._refresh_export_info()
                self.simple_view._refresh_status()
            self.export_button.hide()
        self._refresh_actions()

    def _view_busy_changed(self, busy: bool) -> None:
        # The revision writer also updates the package metadata, so the inputs
        # and the advanced page wait until the job has finished.
        self.reference_run.setEnabled(not busy and self.reference_run.count() > 0)
        self.advanced_button.setEnabled(not busy and self.project is not None)
        self._refresh_actions()
        if not busy and self._loaded_series != self.series_id and self.isVisible():
            # A series chosen on Start while the job ran opens now.
            self.ensure_page()

    def _set_painting_reference(self, run_id: str) -> None:
        """The explicit reference action; preview selection never reaches it."""
        index = self.reference_run.findData(run_id)
        if index >= 0:
            self.reference_run.setCurrentIndex(index)

    @Slot(object)
    def _show_export_dialog(self, export: object) -> None:
        from darsia.fluidflower.corrected_setup import CorrectedSetupExport

        if isinstance(export, CorrectedSetupExport):
            from darsia.fluidflower.corrected_setup_ui import CorrectedSetupExportDialog

            self.export_dialog = CorrectedSetupExportDialog(export, self)
            self.export_dialog.open()
            return
        self.export_dialog = NativeSetupExportDialog(export, self)  # type: ignore[arg-type]
        self.export_dialog.open()

    def _create_advanced_page(self) -> None:
        from darsia.fluidflower.pages.geometry_facies.page import GeometryFaciesPage

        assert self.project is not None
        self._discard_advanced_page()
        page = GeometryFaciesPage(self.project)
        for name in _GATE_WIDGETS:
            widget = getattr(page, name, None)
            if widget is not None:
                widget.hide()
        description = getattr(getattr(page, "definition", None), "description", None)
        for label in page.findChildren(QLabel):
            if description and label.text() == description:
                label.setText(_NATIVE_DESCRIPTION)
                break
        page.reason.setPlaceholderText("Revision note")
        page.status_changed.connect(self._refresh_actions)
        self._install_rig_frame_panel(page)
        self.page = page
        self.page_layout.addWidget(page)
        reference = self.reference()
        if reference is not None:
            self._prefill(page, reference)
        self._refresh_actions()

    def _install_rig_frame_panel(self, page: Any) -> None:
        """Replace the project injection-point tab with rig coordinates."""

        from darsia.fluidflower.series_rig_frame_panel import RigFramePanel

        try:
            page.canvas.physical_point_selected.disconnect(page._injector_placed)
        except (RuntimeError, TypeError):
            pass
        legacy = getattr(page, "place_injector", None)
        if legacy is not None:
            legacy.setChecked(False)
            legacy.hide()
        index = next(
            (
                position
                for position in range(page.tabs.count())
                if page.tabs.tabText(position) == "Injection points"
            ),
            page.tabs.count(),
        )
        if index < page.tabs.count():
            page.tabs.removeTab(index)
        assert self.project is not None
        self.rig_panel = RigFramePanel(page, self.project, self.reference)
        self.rig_panel.set_rig_class(self._rig_class_key)
        self.rig_panel.changed.connect(self._refresh_actions)
        page.tabs.insertTab(index, self.rig_panel, "Reference points, injection points and analysis boxes")

    def has_revision(self) -> bool:
        if self.project is None:
            return False
        from darsia.fluidflower.geometry_facies import (
            GEOMETRY_REVISION_FORMAT,
            GeometryRevisionStore,
        )

        try:
            current = GeometryRevisionStore(self.project).current()
        except (OSError, ValueError):
            return False
        return current.get("format") == GEOMETRY_REVISION_FORMAT

    def _prefill(self, page: Any, reference: SeriesRunInputs) -> None:
        """Suggest the reference run's inputs until a revision is saved.

        Only empty fields and fields still holding an earlier suggestion change.
        """

        if self.has_revision():
            return
        baseline = str(reference.baseline_image or "")
        suggestions = {
            "baseline_path": baseline,
            "draft_source_path": baseline,
            "facies_example_source": baseline,
            "rig_folder": str(reference.rig_folder or ""),
        }
        for name, value in suggestions.items():
            field = getattr(page, name, None)
            if field is None:
                continue
            current = field.text()
            if not current or current == self._prefilled.get(name):
                field.setText(value)
                self._prefilled[name] = value
        for name, value in (
            ("width_m", reference.width_m),
            ("height_m", reference.height_m),
        ):
            spin = getattr(page, name, None)
            if spin is not None and value and spin.value() <= 0.0:
                spin.setValue(float(value))

    # ----------------------------------------------------------------- export

    def _refresh_actions(self) -> None:
        painting = (self.simple_view is not None
                    and self.view_stack.currentWidget() is self.simple_view
                    and self.pages.currentWidget() is self.page_host)
        self.reference_run_label.setVisible(not painting)
        self.reference_run.setVisible(not painting)
        exporting = self._export_thread is not None
        self.export_button.setEnabled(
            self.project is not None and not exporting and self.has_revision()
        )
        self.open_folder_button.setEnabled(self.project is not None)

    def start_export(self) -> None:
        if self.project is None or self._export_thread is not None:
            return
        if not self.has_revision():
            QMessageBox.information(
                self,
                "Geometry revision required",
                "Validate and save a geometry revision before exporting it.",
            )
            return
        self.export_button.setEnabled(False)
        self.export_button.setText("Exporting…")
        thread = QThread(self)
        worker = _ExportWorker(self.project, self.reference(), default_actor())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._export_finished)
        worker.failed.connect(self._export_failed)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._export_thread_finished)
        self._export_thread = thread
        self._export_worker = worker
        thread.start()

    @Slot(object)
    def _export_finished(self, result: object) -> None:
        self._show_export_dialog(result)

    @Slot(str)
    def _export_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Could not export to native Setup", message)

    def _export_thread_finished(self) -> None:
        thread = self._export_thread
        self._export_thread = None
        self._export_worker = None
        if thread is not None:
            thread.deleteLater()
        self.export_button.setText("Export to native Setup…")
        self._refresh_actions()
        if self._loaded_series != self.series_id and self.isVisible():
            self.ensure_page()

    def _open_folder(self) -> None:
        if self.project is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.root)))


__all__ = ["NativeSetupExportDialog", "SeriesGeometryWorkspace"]
