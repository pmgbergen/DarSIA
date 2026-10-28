"""Native owner wiring for explicit Jobs registrations; no implicit execution."""

from copy import deepcopy
from pathlib import Path
import tomllib
import traceback

from PySide6.QtCore import QStandardPaths, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .jobs_metadata import create_job_from_plan
from .jobs_panel import JobsPanel, NewAnalysisPage
from .native_jobs_registry import JobsUIRegistry
from .pipeline.job_queue import name
from .platform_support import JOBS_ON_WINDOWS, jobs_supported
from .store_map_ui import StoreAvailabilityPanel
from .workflow_section import WorkflowSection, WorkflowWorkspace


class _SourcesSection(WorkflowSection):
    """Keep the existing navigation entry points for the folder choices."""

    def setChecked(self, checked):
        self.set_expanded(checked)

    def isChecked(self):
        return self.is_expanded()

    def title(self):
        return self.toggle.text()


class NativeJobsWorkspace(QWidget):
    """One passive series-scoped UI owner, created lazily by native navigation.

    Registration/selection is local UI state, not evidence that a scientific job
    was built for a series. Explicit creation uses the original B5 exact-plan
    admission or the original preset Comparison producer. The three views are
    the job list, New analysis, and Wasserstein comparison preparation.
    """

    new_analysis_requested = Signal()
    jobs_requested = Signal()
    raw_model_requested = Signal()
    raw_review_registered = Signal(object)
    series_locations_changed = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        registry=None,
        measurement_registry=None,
        local_machine=None,
        reader=None,
        job_creator=None,
        local_launcher=None,
        remote_launcher=None,
        protocol_reader=None,
        selection_resolver=None,
        inventory_provider=None,
        resource_factory=None,
        store_panel_factory=StoreAvailabilityPanel,
        raw_review_model_provider=None,
        preset_wasserstein_preparer=None,
        new_analysis_factory=NewAnalysisPage,
    ):
        super().__init__(parent)
        default = (
            Path(
                QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppLocalDataLocation
                )
            )
            / "native_jobs"
        )
        self.registry = registry or JobsUIRegistry(default)
        self._creator = job_creator or create_job_from_plan
        self.protocol_reader, self.selection_resolver = (
            protocol_reader,
            selection_resolver,
        )
        self.series_root = self.series_id = None
        self.series_label = ""
        self._local_inventory = {}
        self._discovery_thread = self._discovery_worker = None
        self._discovery_pending = False
        self._closing = False
        layout = QVBoxLayout(self)
        self.association_note = QLabel(
            "Choose a series and its shared Jobs folder. Published jobs for that series appear here; "
            "standalone jobs and adopted plans are available under advanced options."
        )
        self.association_note.setWordWrap(True)
        self.association_note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.association_note)
        self.jobs_view = WorkflowWorkspace()
        self.result_views = QStackedWidget()
        results_layout = QVBoxLayout(self.jobs_view.workspace)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addWidget(self.result_views)
        self.sources_body = QWidget()
        body_layout = QVBoxLayout(self.sources_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        folders_note = QLabel(
            "Advanced: register a standalone job.json or an adopted execution plan. "
            "These actions do not copy job data or start workers. Machine path mappings "
            "translate the shared locations for this computer."
        )
        folders_note.setWordWrap(True)
        folders_note.setTextFormat(Qt.TextFormat.PlainText)
        body_layout.addWidget(folders_note)
        self.sources = _SourcesSection(
            None, "Advanced — imports and paths", self.sources_body, expanded=False)
        self.jobs_view.controls_layout.addWidget(self.sources)
        self.stores = store_panel_factory(self.sources_body)
        body_layout.addWidget(self.stores)
        actions = QVBoxLayout()
        self.open_job_button = QPushButton("Open existing job…")
        self.import_plan_button = QPushButton("Import adopted execution plan…")
        self.open_job_button.clicked.connect(self.open_existing_job)
        self.import_plan_button.clicked.connect(self.import_execution_plan)
        actions.addWidget(self.open_job_button)
        actions.addWidget(self.import_plan_button)
        actions.addStretch()
        body_layout.addLayout(actions)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)
        self.panel = JobsPanel(
            local_machine=local_machine,
            reader=reader,
            job_creator=self._create_job,
            local_launcher=local_launcher,
            remote_launcher=remote_launcher,
            known_jobs_provider=self.registry.known_jobs,
            local_stores_path_provider=lambda: self.stores.path,
            raw_review_model_provider=raw_review_model_provider,
        )
        self.panel.raw_model_requested.connect(self.raw_model_requested)
        self.panel.raw_review_registered.connect(self.raw_review_registered)
        self.resources = None
        if resource_factory is not None:
            self.attach_resources(
                resource_factory(
                    measurement_registry=measurement_registry,
                    inventory_provider=inventory_provider,
                )
            )
        self.job_results = self.panel
        self.result_views.addWidget(self.job_results)
        self.new_analysis = new_analysis_factory(self.panel)
        self.panel.attach_new_analysis(self.new_analysis)
        self.panel.new_analysis_requested.connect(self.new_analysis_requested)
        if hasattr(self.new_analysis, "jobs_requested"):
            self.new_analysis.jobs_requested.connect(self.jobs_requested)
        from .preset_wasserstein_preparation_ui import PresetWassersteinPreparationPage
        self.wasserstein = PresetWassersteinPreparationPage(
            self.panel, self, preparer=preset_wasserstein_preparer,
            register_job=self._job_created,
        )
        self.panel.comparison_preparation = self.wasserstein
        self.wasserstein.jobs_requested.connect(self._show_comparison_job)
        self.views = QStackedWidget()
        self.views.addWidget(self.jobs_view)
        self.views.addWidget(self.new_analysis)
        self.views.addWidget(self.wasserstein)
        layout.addWidget(self.views, 1)
        self.view = "jobs"
        self.panel.job_created.connect(self._job_created)
        self.panel.job_table.cellClicked.connect(self._job_clicked)
        self.panel.job_table.itemActivated.connect(
            lambda item: self._job_clicked(item.row(), item.column())
        )
        self.panel.job_table.setToolTip(
            "Click or activate a job to remember it as active for this registered series."
        )
        self.panel.error_reported.connect(self.status_label.setText)
        self.panel.error_cleared.connect(self._error_cleared)
        self._clear_context()
        from .native_jobs_sections import NativeJobsSections
        self.inline_layout = NativeJobsSections(self)
        # Jobs controls remain visible when an alternate result hides panel.
        self.panel.set_refresh_visibility_owner(self.jobs_view)
        from .job_management_ui import JobManagementControls
        self.job_management = JobManagementControls(self)
        self.panel.job_table.parentWidget().layout().addWidget(self.job_management)
        from .shared_jobs_ui import SharedJobsControls
        self.shared_jobs = SharedJobsControls(self)
        self._shared_jobs_folder = self.shared_jobs.folder.text().strip()
        self._folder_refresh = QTimer(self)
        self._folder_refresh.setSingleShot(True)
        self._folder_refresh.setInterval(300)
        self._folder_refresh.timeout.connect(self._discover_shared_jobs)
        self.shared_jobs.folder.textChanged.connect(self._shared_folder_changed)
        self.panel.refresh_button.clicked.connect(lambda: self._folder_refresh.start(300))
        self.shared_jobs_section = WorkflowSection(None, "Shared job folder", self.shared_jobs)
        self.jobs_view.controls_layout.insertWidget(0, self.shared_jobs_section)
        self.result_views.addWidget(self.shared_jobs.results)
        self.inline_layout.attach_shared_jobs(self.shared_jobs_section, self.shared_jobs)
        self.shared_jobs.show_status_requested.connect(
            lambda: self.result_views.setCurrentWidget(self.shared_jobs.results))
        self.shared_jobs.role_changed.connect(self._shared_role_changed)
        def shared_folder_saved(root):
            editor = getattr(self.inline_layout, "store_editor", None)
            if editor is not None and editor.series_root == Path(root):
                editor.load_locations()
        self.shared_jobs.folder_saved.connect(shared_folder_saved)
        original_snapshot = self.panel._show_snapshot

        def show_snapshot(data):
            original_snapshot(data)
            self.shared_jobs.sync()

        self.panel._show_snapshot = show_snapshot
        # Reparenting the persistent pages invalidates the former two-pane
        # minimum. Refresh it before a first show can widen the entire window.
        self.panel.layout().invalidate()
        self.panel.updateGeometry()
        self.result_views.layout().invalidate()
        self.jobs_view.workspace.layout().invalidate()
        self.views.layout().invalidate()
        self.layout().invalidate()
        self.layout().activate()
        self.setMinimumSize(0, 0)
        if not jobs_supported():
            for widget in (self.panel, self.shared_jobs, self.open_job_button, self.import_plan_button):
                widget.setEnabled(False)
            self._shared_role_changed("worker")

    def _shared_role_changed(self, role):
        """Worker participation never exposes master launch or editing actions.

        Without the Windows job layer every starting action stays disabled and
        the note says where jobs run.
        """
        worker_only = role == "worker" or not jobs_supported()
        for widget in (self.sources, self.new_analysis, self.wasserstein, self.job_management,
                       self.inline_layout.legacy_section, self.inline_layout.machine_section,
                       self.inline_layout.master_controls,
                       self.inline_layout.machine, self.sources_body):
            widget.setEnabled(not worker_only)
        self.inline_layout.sync_job_controls(worker_only=worker_only)
        if self.inline_layout.store_editor is not None:
            self.inline_layout.store_editor.setEnabled(not worker_only)
        self.association_note.setText(
            JOBS_ON_WINDOWS if not jobs_supported() else
            "Worker role: this computer takes jobs from the shared folder. The publishing computer remains master."
            if worker_only else
            "Start a selected job in the shared folder, or join the same folder from another computer.")

    def _shared_folder_changed(self, folder):
        # Copy the UI choice on its owning thread. The normal metadata reader
        # consumes this string, never a Qt input, after typing has settled.
        self._shared_jobs_folder = folder.strip()
        if self.series_root is not None:
            self._folder_refresh.start()

    @Slot()
    def _discover_shared_jobs(self):
        from .shared_jobs_discovery import discover_published_jobs
        if self._closing or self.series_root is None:
            return
        if self._discovery_thread is not None:
            self._discovery_pending = True
            return
        self._discovery_pending = False
        root, series_id, folder = self.series_root, self.series_id, self._shared_jobs_folder
        stores_path = self.stores.path
        archived = self.job_management.show_archived.isChecked()
        def discover():
            from .series_locations import machine_jobs_folder
            local = machine_jobs_folder(root, folder, stores_path) if folder else ""
            rows = discover_published_jobs(local, root, series_id, self.registry,
                                          include_archived=archived)
            return dict(context=(root, series_id, folder), rows=rows,
                        missing=self.registry.missing_jobs(root, series_id))
        self._discovery_thread, self._discovery_worker = self.panel._start_call(
            discover, self._discovery_ready, self._discovery_closed,
            failed=self._discovery_failed)

    @Slot(object)
    def _discovery_ready(self, result):
        if not self._closing and result["context"] == (
                self.series_root, self.series_id, self._shared_jobs_folder):
            self.panel.set_jobs(result["rows"])
            self.panel.refresh()
            missing = result.get("missing", [])
            self.status_label.setText(
                f"{len(missing)} registered job folder(s) no longer exist and are not listed: "
                + ", ".join(Path(path).name for path in missing) if missing else "")
            if not result["rows"]:
                self.panel.progress_label.setText(
                    "No jobs are registered for this series. Choose its shared Jobs folder or prepare "
                    "a job from Setup, Calibration or Production. Standalone jobs can be opened under "
                    "Shared job folder → Advanced — imports and paths.")

    @Slot(str)
    def _discovery_failed(self, error):
        self.status_label.setText("Shared job folder: " + error)

    @Slot()
    def _discovery_closed(self):
        self._discovery_thread.wait()
        self._discovery_thread = self._discovery_worker = None
        if self._discovery_pending and not self._closing:
            self._discovery_pending = False
            self._folder_refresh.start()

    def attach_resources(self, resources):
        """Attach an explicitly supplied optional UI panel; B5 has no B6 import."""
        if self.resources is not None:
            raise ValueError("Resource panel already attached")
        self.panel.set_resource_panel(resources)
        self.resources = resources
        self.resources.hardware_inspected.connect(self._hardware_ready)
        if hasattr(self, "inline_layout"):
            self.inline_layout.sync_result_buttons()

    def _clear_context(self):
        self.series_root = self.series_id = None
        self.series_label = ""
        self.stores.set_context(None, None)
        if hasattr(self, "shared_jobs"):
            self.shared_jobs.set_series(None)
        self.open_job_button.setEnabled(False)
        self.import_plan_button.setEnabled(False)
        self.wasserstein.set_context(None, None)
        self.panel.set_context(
            series_label="",
            series_root=None,
            series_id=None,
            jobs_provider=lambda: [],
            plans_provider=lambda: [],
            machines_provider=self._machines_provider,
        )

    def set_series(self, series_id, series_root, *, series_label=""):
        if not series_root or not series_id:
            self._clear_context()
            return
        try:
            root = Path(series_root)
            raw = (root / "series.toml").read_bytes()
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("Series manifest is too large for Jobs UI metadata")
            doc = tomllib.loads(raw.decode("utf-8-sig"))
            actual = doc.get("series_id")
            if (
                doc.get("schema_version") != 1
                or not isinstance(actual, str)
                or not actual
            ):
                raise ValueError(
                    "A registered schema-1 series with its own series_id is required"
                )
            # The registered folder's literal manifest ID is authoritative for
            # B4, not a displayed alias or a previously loaded run config.
            self.series_root, self.series_id = root, actual
            self.series_label = str(series_label or actual)
            self.stores.set_context(root, actual)
            if hasattr(self, "shared_jobs"):
                from .series_paths import series_folder_path
                self.shared_jobs.set_series(root, series_folder_path(root, "jobs"))
            self.open_job_button.setEnabled(True)
            self.import_plan_button.setEnabled(True)
            self.wasserstein.set_context(root, actual, series_label=self.series_label)
            self.association_note.setText(
                f"Series: {self.series_label}"
            )
            self.panel.set_context(
                series_label=self.series_label,
                series_root=root,
                series_id=actual,
                jobs_provider=lambda: self.registry.jobs(root, actual, include_archived=(
                    hasattr(self, "job_management") and self.job_management.show_archived.isChecked())),
                plans_provider=lambda: self.registry.plans(root, actual),
                machines_provider=self._machines_provider,
                selection_provider=self._selection_provider(root, actual),
            )
        except Exception:
            text = traceback.format_exc()
            self._clear_context()
            self.status_label.setText(text)

    def _selection_provider(self, root, series_id):
        def resolve(runs):
            protocol_reader, resolver = self.protocol_reader, self.selection_resolver
            if protocol_reader is None or resolver is None:
                from .series_image_selection import (
                    load_run_protocol_metadata,
                    resolve_saved_selection,
                )

                protocol_reader = protocol_reader or load_run_protocol_metadata
                resolver = resolver or resolve_saved_selection
            result = []
            for run in runs:
                protocols = protocol_reader(root, run)
                row = resolver(root, run, protocols, check_files=False)
                if row.get("series_id") != series_id:
                    raise ValueError("B4 resolver belongs to another registered series")
                result.append(row)
            return result

        return resolve

    def _machines_provider(self):
        local = self.panel.local_machine
        rows = {
            name(local): dict(
                machine=local, inventory=deepcopy(self._local_inventory), history=[]
            )
        }
        for fact in (self.panel.current or {}).get("registered_machines", []):
            machine = fact["machine"]
            key = name(machine)
            rows.setdefault(key, dict(machine=machine, inventory={}, history=[]))
            rows[key].setdefault("registrations", []).append(deepcopy(fact["source"]))
        return list(rows.values())

    def _hardware_ready(self, inventory):
        self._local_inventory = deepcopy(inventory)
        if self.resources is not None:
            self.resources.refresh_known_machines()

    def open_existing_job(self):
        if self.series_root is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open an explicit UNC job.json", "", "Job metadata (job.json)"
        )
        if not path:
            return
        if Path(path).name.casefold() != "job.json":
            self.status_label.setText("Choose the existing job.json itself.")
            return
        self.register_existing_job(Path(path).parent)

    def register_existing_job(self, root):
        if self.series_root is None:
            return
        try:
            self.registry.register_job(
                self.series_root, self.series_id, root, activate=True
            )
            self.status_label.setText(
                "Existing job registered locally. No task, result or worker was changed."
            )
            self.refresh_registered()
        except Exception:
            self.status_label.setText(traceback.format_exc())

    def import_execution_plan(self):
        if self.series_root is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import an existing adopted UNC execution plan", "", "TOML (*.toml)"
        )
        if not path:
            return
        try:
            self.registry.register_plan(self.series_root, self.series_id, path)
            self.status_label.setText(
                "Existing adopted plan registered locally. Create still requires an exact saved B4 source/time/order match; a new selection needs its authorized producer."
            )
            self.refresh_registered()
        except Exception:
            self.status_label.setText(traceback.format_exc())

    def refresh_registered(self):
        if self.series_root is None:
            return
        self.set_series(
            self.series_id, self.series_root, series_label=self.series_label
        )

    def _job_clicked(self, row, _column):
        if self.series_root is None or not 0 <= row < len(self.panel.snapshots):
            return
        try:
            self.registry.set_active(
                self.series_root, self.series_id, self.panel.snapshots[row]["root"]
            )
            for registration in self.panel.registrations:
                registration["active"] = (
                    registration["root"] == self.panel.snapshots[row]["root"]
                )
        except Exception:
            self.status_label.setText(traceback.format_exc())

    def _create_job(self, plan, job_root, **kwargs):
        result = self._creator(plan, job_root, **kwargs)
        # UI-only result fields preserve the actual captured context even when
        # a finite background create finishes after navigation to another series.
        return dict(
            result,
            ui_series_root=str(kwargs["series_root"]),
            ui_series_id=kwargs["series_id"],
        )

    def _job_created(self, registration):
        try:
            root = registration["ui_series_root"]
            series_id = registration["ui_series_id"]
            self.registry.register_job(
                root,
                series_id,
                registration["root"],
                label=registration.get("name", ""),
                activate=True,
            )
            self.status_label.setText(
                f"Job metadata creation recorded locally for series {series_id}; no worker is started automatically."
            )
            if (
                self.series_root is not None
                and Path(root) == self.series_root
                and series_id == self.series_id
            ):
                self.refresh_registered()
            return True
        except Exception:
            self.status_label.setText(traceback.format_exc())
            return False

    def show_view(self, name):
        """Show Jobs, New analysis or original Comparison preparation.

        The New analysis page borrows the job toolbar's machine, backend and
        workers choice and the resource advisor while it is shown.
        """
        if name not in ("jobs", "new_analysis", "wasserstein"):
            raise ValueError(f"Unknown Jobs view: {name}")
        page = self.new_analysis if name == "new_analysis" else None
        self.panel.place_target_controls(page)
        if page is not None:
            self.panel.prepare_new_analysis()
        self.views.setCurrentWidget(self.wasserstein if name == "wasserstein" else page or self.jobs_view)
        self.view = name

    def _show_comparison_job(self):
        registration = self.wasserstein.registration
        if registration is None or self.series_root is None:
            return
        try:
            self.registry.set_active(self.series_root, self.series_id, registration["root"])
            self.refresh_registered()
        except Exception:
            self.status_label.setText(traceback.format_exc())
            return
        self.show_view("jobs")
        self.jobs_requested.emit()

    def start_analysis(self):
        if self.series_root is None:
            self.status_label.setText(
                "Choose a registered series before starting analysis."
            )
            return
        self.panel.open_create_dialog()

    def install_mass_action(self, page, callback=None):
        if hasattr(page, "native_start_analysis_button"):
            return page.native_start_analysis_button
        button = QPushButton("Start analysis…", page)
        button.setToolTip(
            "Choose an already adopted execution plan that exactly matches this series' saved image selection. No job starts by opening the dialog."
        )
        button.clicked.connect(callback or self.start_analysis)
        page.layout().insertWidget(0, button)
        page.native_start_analysis_button = button
        return button

    def prepare_close(self):
        """Abandon display reads without abandoning mutation/worker authority."""
        self.panel.prepare_close()
        self._closing = True
        self._folder_refresh.stop()
        cancel = getattr(self.stores, "prepare_close", None)
        if cancel is not None:
            cancel()
        if self.resources is not None:
            cancel = getattr(self.resources, "prepare_close", None)
            if cancel is not None:
                cancel()
        for name in ('shared_jobs', 'job_management'):
            cancel = getattr(getattr(self, name, None), 'prepare_close', None)
            if callable(cancel):
                cancel()

    def cancel_close(self):
        self._closing = False
        if self.series_root is not None:
            self._folder_refresh.start()
        for name in ('shared_jobs', 'job_management'):
            resume = getattr(getattr(self, name, None), 'cancel_close', None)
            if callable(resume):
                resume()

    def can_close(self):
        return self._discovery_thread is None and self.panel.can_close() and self.stores.can_close() and (
            self.resources is None or self.resources.can_close()
        ) and (not hasattr(self, "job_management") or self.job_management.can_close()) and (
            not hasattr(self, "shared_jobs") or self.shared_jobs.can_close())

    def _error_cleared(self, text):
        """Clear a Jobs refusal that no longer applies, if the status line still shows it."""
        if self.status_label.text() == text:
            self.status_label.clear()

    def close_block_reason(self):
        editor = self.inline_layout.store_editor
        if editor is not None and editor._thread is None and editor.has_edits():
            return (
                "Machine path mappings has unsaved settings edits. Open Jobs → "
                "Shared job folder → Advanced — imports and paths → Machine path mappings and choose "
                "Save local settings or Discard local settings edits before closing."
            )
        return "Job/store metadata or resource work is still finishing. Close after it returns."

    def closeEvent(self, event):
        self.prepare_close()
        if not self.can_close():
            self.status_label.setText(self.close_block_reason())
            event.ignore()
            return
        super().closeEvent(event)
