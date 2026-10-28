"""Prepare typed Setup metadata from the current workspace and existing Jobs choices."""
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
import traceback
import weakref
from uuid import uuid4

from PySide6.QtCore import QMetaObject, QObject, Qt, Slot
from PySide6.QtWidgets import QPushButton


def suggested_setup_job_root(series_root, *, jobs_folder=None):
    """Choose a fresh child of the Jobs queue or the declared series default."""
    from .series_paths import series_folder_path
    root = PureWindowsPath(str(jobs_folder or series_folder_path(series_root, "jobs")))
    if not root.drive.startswith("\\\\"):
        return ""
    return str(root / ("setup-" + uuid4().hex))


def _setup_job_root(jobs, series_root):
    """Reuse one fresh Setup child while this series and Jobs queue stay selected."""
    from .series_paths import series_folder_path
    shared = getattr(getattr(jobs, "shared_jobs", None), "folder", None)
    folder = shared.text().strip() if shared is not None else ""
    folder = folder or str(series_folder_path(series_root, "jobs"))
    parent = PureWindowsPath(folder)
    if not parent.drive.startswith("\\\\"):
        raise ValueError("Choose a shared UNC folder in Jobs > Shared job folder.")
    key = (str(Path(series_root)), str(parent).casefold())
    saved = getattr(jobs, "_setup_job_choice", None)
    if not isinstance(saved, dict) or saved.get("key") != key:
        saved = dict(key=key, root=suggested_setup_job_root(series_root, jobs_folder=folder))
        jobs._setup_job_choice = saved
    return saved["root"]


def current_version_job_root(job_root, commit, series_id):
    """An explicit rebuild captures new controls and outputs, even on the same code."""
    root = Path(job_root)
    path = root / "job.json"
    if path.is_file():
        from .pipeline.job_queue import read_json
        old = read_json(path)
        if old.get("pipeline") != "series-setup" or old.get("series_id") != series_id:
            raise ValueError("The selected job folder belongs to another job or series")
        # A new Prepare is a rebuild from current saved Setup choices. Never
        # inherit the preceding Apply completion, Build receipts or image cache.
        # The original Apply transaction replaces the active configuration
        # pointers only after these fresh job-owned exports are ready.
        return root.parent / ("setup-" + uuid4().hex)
    return root


def current_setup_job_context(workspace, *, include_identity=True):
    """Capture actual saved workspace and Jobs choices without inventing locations."""
    project = workspace.project
    root = workspace._loaded_root
    if project is None or root is None or workspace._loaded_series != workspace.series_id:
        raise ValueError("The selected series has no current loaded geometry workspace")
    current = Path(project.root) / "overrides" / "geometry_facies" / "current.json"
    with current.open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("The adopted geometry revision exceeds the metadata limit")
    revision = json.loads(raw.decode("utf-8-sig"))
    if (revision.get("format") != "darsia-fluidflower-geometry-facies-v1"
            or revision.get("schema") != 1 or not revision.get("revision_id")
            or (revision.get("configuration") or {}).get("frame") != "rig-corrected"):
        raise ValueError("Setup needs the current saved corrected geometry revision")
    window = workspace.window()
    jobs = getattr(window, "jobs_workspace", None)
    if jobs is None:
        ensure = getattr(window, "ensure_jobs", None)
        jobs = ensure() if callable(ensure) else None
    if jobs is None:
        raise ValueError("The existing Jobs workspace is unavailable")
    if jobs.series_root is None or Path(jobs.series_root) != Path(root):
        raise ValueError("Jobs is bound to another loaded series")
    job_root = _setup_job_root(jobs, root)
    from .pipeline.job_queue import name, require_unc
    require_unc(job_root)
    selected = jobs.panel.target_combo.currentData() or {}
    target = selected.get("machine")
    if not target:
        raise ValueError("Jobs has no selected worker target")
    if jobs.panel.backend_combo.currentData() != "cpu":
        raise ValueError("The selected Jobs backend must be CPU for Setup")
    if selected.get("local"):
        report = getattr(jobs.stores, "report", None) or {}
        if report.get("machine_id"):
            target = report["machine_id"]
    context = dict(project=project, series_root=Path(root), target_machine=name(target),
                   job_root=Path(job_root), adopted_revision_id=revision["revision_id"])
    if include_identity:
        from .analysis_preparation import job_identity
        identity = job_identity(Path(__file__).resolve().parents[3])
        context.update(commit=identity["commit"], environment_lock=identity["environment_lock"])
    stores_path = getattr(jobs.stores, "path", None)
    if stores_path is not None:
        context["stores_path"] = stores_path
    return context


def _prepare_current(project, series_root, **choices):
    """Call only the current-series metadata assembler after the explicit action."""
    from .series_setup_plan import prepare_current_series_setup
    from .pipeline.setup_image_preparation import independent_choice
    from .mass.n1_pipeline import asset
    authority = asset(Path(series_root) / "series.toml")
    document, image_choice = independent_choice(authority)
    if image_choice is not None:
        from .pipeline.setup_colour_paths import preflight_setup_colour_paths
        preflight_setup_colour_paths(authority, document["runs"]["ids"])
    return prepare_current_series_setup(project, series_root, **choices)


class SeriesSetupJobsController(QObject):
    """Bind the existing action to finite metadata preparation and original Jobs."""

    def __init__(self, workspace, view, *, context_provider=None, preparer=None):
        """Connect existing status signals while preserving the view's busy boundary."""
        super().__init__(view)
        self._workspace, self._view = weakref.ref(workspace), weakref.ref(view)
        self.context_provider = context_provider or current_setup_job_context
        self._default_context = context_provider is None
        self.preparer = preparer or _prepare_current
        self._thread = self._worker = None
        self._starting = False
        self._context = self._key = self._outcome = None
        self._wired_jobs = self._wired_dialog = None
        self._original_reason = view.apply_build_reason.text()
        self._original_busy = weakref.WeakMethod(view.busy)
        view.busy = self.busy
        for name in ("apply_build_job_label", "apply_build_job_edit"):
            field = getattr(view, name, None)
            if field is not None:
                field.hide()
        view.apply_build_button.setText("Prepare Apply and Build job")
        view.apply_build_button.setToolTip(
            "Create a unique job in Jobs > Shared job folder, then start its workers in Jobs.")
        view.apply_build_button.clicked.connect(self.prepare)
        self.colour_paths_button = QPushButton("Continue to colour paths", view)
        self.colour_paths_button.setToolTip(
            "Open Setup > Colour paths after Apply/Build and image correction are complete. Opening starts no calculation.")
        self.colour_paths_button.clicked.connect(self._open_colour_paths)
        parent = view.apply_build_button.parentWidget()
        layout = parent.layout() if parent is not None else None
        if layout is not None and hasattr(layout, "insertWidget"):
            layout.insertWidget(layout.indexOf(view.apply_build_button) + 1, self.colour_paths_button)
            self.colour_paths_button.setVisible(callable(getattr(workspace.window(), "_select_row", None)))
        else:
            self.colour_paths_button.setParent(view)
            self.colour_paths_button.hide()
        native = workspace.window()
        finish = getattr(native, "finish_setup_page", None)
        method = getattr(getattr(native, "_saved_method", None), "active", "")
        if finish is not None and view is not finish and method == "phase_model":
            view.apply_build_button.hide()
            view.apply_build_reason.hide()
            self.colour_paths_button.hide()
        view.busy_changed.connect(self.refresh)
        workspace.reference_changed.connect(self.refresh)
        status = getattr(view, "series_use_status", None)
        if status is not None:
            status.changed.connect(self.refresh)
        adoption = getattr(view, "adoption_controller", None)
        findings_changed = getattr(adoption, "findings_changed", None)
        if findings_changed is not None:
            findings_changed.connect(self.refresh)
        self.refresh()

    def _open_colour_paths(self):
        workspace = self._workspace()
        window = workspace.window() if workspace is not None else None
        navigate = getattr(window, "_select_row", None)
        if callable(navigate):
            navigate(("calibration", "color"))

    def busy(self):
        """Keep the original view busy while its metadata callback is retained."""
        original = self._original_busy()
        return self._starting or self._thread is not None or bool(original and original())

    def _jobs(self):
        """Use the native window's existing Jobs owner and call runner."""
        window = self._workspace().window()
        jobs = getattr(window, "jobs_workspace", None)
        if jobs is None and callable(getattr(window, "ensure_jobs", None)):
            jobs = window.ensure_jobs()
        if jobs is None:
            raise ValueError("The existing Jobs workspace is unavailable")
        return jobs

    def _findings_ready(self, view):
        """Keep new Setup preparation behind this view's current findings."""
        adoption = getattr(view, "adoption_controller", None)
        check = getattr(adoption, "findings_block_reason", None)
        reason = check() if callable(check) else ""
        from .manual_map_approval import has_manual_map_approval
        prepared = (getattr(getattr(view, 'series_use_status', None), 'result', None) or {}).get('preparation')
        if not reason and has_manual_map_approval(prepared or {}):
            # An approved transfer is ready for Check/Save, not yet Build.
            # Read the actual current revision so stale passive status cannot
            # authorize applying the preceding geometry revision.
            try:
                from .painted_adoption_table import _bound, _checked
                from .storage_paths import same_storage_descriptor
                workspace = self._workspace()
                current = _checked(_bound(Path(workspace.project.root) /
                    'overrides/geometry_facies/current.json'), metadata=True)
                configuration = current.get('configuration') or {}
                bound = (configuration.get('painted_adoption_table') or {}).get('preparation')
                if not same_storage_descriptor(bound, _bound(Path(prepared['folder']) / 'index.json')):
                    reason = 'Save the checked approved map as the current geometry revision before Apply and Build.'
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                reason = 'The selected approved map has no current saved geometry revision: ' + str(error)
        if not reason:
            return True
        self._context = self._outcome = None
        view.apply_build_button.setEnabled(False)
        view.apply_build_reason.setText(
            "Review gaps and findings before preparing Apply and Build. " + reason)
        return False

    def _wire_jobs_choices(self):
        """Observe each existing Jobs owner once, without creating another owner."""
        workspace = self._workspace()
        jobs = getattr(workspace.window(), "jobs_workspace", None) if workspace is not None else None
        if jobs is None or (self._wired_jobs is not None and self._wired_jobs() is jobs):
            return
        self._wired_jobs = weakref.ref(jobs)
        shared = getattr(getattr(jobs, "shared_jobs", None), "folder", None)
        if shared is not None:
            shared.textChanged.connect(self.refresh)
        jobs.panel.target_combo.currentIndexChanged.connect(self.refresh)
        jobs.panel.backend_combo.currentIndexChanged.connect(self.refresh)
        jobs.stores.check_button.clicked.connect(self._watch_store_check)
        jobs.stores.settings_button.clicked.connect(self._watch_store_settings)

    @Slot()
    def _watch_store_check(self):
        """Refresh after the existing availability writer publishes its report."""
        jobs = self._wired_jobs() if self._wired_jobs is not None else None
        task = getattr(jobs.stores, "_thread", None) if jobs is not None else None
        if task is not None:
            task.finished.connect(self.refresh, Qt.ConnectionType.QueuedConnection)
            if task.isFinished():
                QMetaObject.invokeMethod(self, "refresh", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _watch_store_settings(self):
        """Observe the existing settings dialog's actual saved-path signal once."""
        jobs = self._wired_jobs() if self._wired_jobs is not None else None
        dialog = getattr(jobs.stores, "_dialog", None) if jobs is not None else None
        if dialog is not None and (self._wired_dialog is None or self._wired_dialog() is not dialog):
            self._wired_dialog = weakref.ref(dialog)
            dialog.saved.connect(self.refresh)

    @Slot()
    def refresh(self, *_):
        """Offer preparation only for the currently captured genuine context."""
        workspace, view = self._workspace(), self._view()
        if workspace is None or view is None:
            return
        self._wire_jobs_choices()
        view.apply_build_button.setEnabled(False)
        from .platform_support import JOBS_ON_WINDOWS, jobs_supported
        if not jobs_supported():
            view.apply_build_reason.setText(JOBS_ON_WINDOWS)
            return
        if self._starting or self._thread is not None:
            return
        if not self._findings_ready(view):
            return
        original = self._original_busy()
        if original and original():
            view.apply_build_reason.setText("Wait for the current geometry operation before preparing Setup metadata.")
            return
        try:
            context = (self.context_provider(workspace, include_identity=False)
                       if self._default_context else self.context_provider(workspace))
            self._wire_jobs_choices()
            if context is None:
                self._context = None
                view.apply_build_reason.setText(self._original_reason)
                return
            if (context["project"] is not workspace.project
                    or Path(context["series_root"]) != Path(workspace._loaded_root)):
                raise ValueError("The Setup context belongs to another loaded series")
            key = (str(context["project"].root), str(context["series_root"]),
                   str(context["job_root"]), context["target_machine"], context.get("commit"),
                   json.dumps(context.get("environment_lock"), sort_keys=True),
                   str(context.get("stores_path") or ""), context.get("adopted_revision_id"))
        except Exception as error:
            self._wire_jobs_choices()
            self._context = None
            view.apply_build_reason.setText(str(error))
            return
        if key != self._key:
            self._outcome = None
        self._context, self._key = dict(context), key
        view.apply_build_button.setEnabled(True)
        view.apply_build_reason.setText(self._outcome or
            "Prepare the current series' Setup job metadata, then start its workers in Jobs to run Apply and Build.")

    @Slot()
    def prepare(self):
        """Freeze existing choices and prepare metadata without starting a process."""
        view = self._view()
        from .platform_support import JOBS_ON_WINDOWS, jobs_supported
        if view is not None and not jobs_supported():
            view.apply_build_reason.setText(JOBS_ON_WINDOWS)
            return
        if view is None or not self._findings_ready(view):
            return
        self.refresh()
        view, workspace = self._view(), self._workspace()
        if view is None or workspace is None or not view.apply_build_button.isEnabled():
            return
        context = dict(self._context)
        try:
            if self._default_context:
                context = dict(self.context_provider(workspace))
            jobs = self._jobs()
            if jobs.series_root is None or Path(jobs.series_root) != Path(context["series_root"]):
                raise ValueError("Jobs is bound to another loaded series")
            series_id = jobs.series_id
            fresh = current_version_job_root(context["job_root"], context["commit"], series_id)
            if fresh != Path(context["job_root"]):
                context["job_root"] = fresh
                saved = getattr(jobs, "_setup_job_choice", None)
                if isinstance(saved, dict) and Path(saved.get("root", "")) == Path(self._context["job_root"]):
                    saved["root"] = str(fresh)
        except Exception:
            self._failed(traceback.format_exc())
            return
        if not self._findings_ready(view):
            return
        preparer = self.preparer

        def prepare_metadata():
            try:
                choices = {key: context[key] for key in
                           ("target_machine", "job_root", "commit", "environment_lock")}
                if "stores_path" in context:
                    choices["stores_path"] = context["stores_path"]
                registration = preparer(context["project"], context["series_root"], **choices)
                return dict(context=context, series_id=series_id, registration=registration)
            except Exception:
                return dict(error=traceback.format_exc())

        self._starting = True
        view.apply_build_button.setEnabled(False)
        view.apply_build_reason.setText("Preparing Setup job metadata… No worker has been started.")
        view.busy_changed.emit(True)
        self._thread, self._worker = jobs.panel._start_call(prepare_metadata, self._ready, self._closed)

    @Slot(object)
    def _ready(self, value):
        """Register the captured job and open Jobs only after preparation succeeds."""
        if value.get("error"):
            self._failed(value["error"])
            return
        try:
            context, registration = value["context"], value["registration"]
            if not isinstance(registration, dict) or registration.get("complete") is not True:
                raise ValueError("Setup metadata preparation did not report a complete job")
            root = Path(context["job_root"])
            if registration.get("root") is not None and Path(registration["root"]) != root:
                raise ValueError("Prepared Setup job belongs to another selected folder")
            if registration.get("job") is not None and Path(registration["job"]["path"]) != root / "job.json":
                raise ValueError("Prepared Setup job descriptor belongs to another selected folder")
            jobs = self._jobs()
            jobs.registry.register_job(context["series_root"], value["series_id"], root, activate=True)
            if jobs.series_root is not None and Path(jobs.series_root) == Path(context["series_root"]):
                jobs.refresh_registered()
            self._outcome = "Setup job metadata created. Start its workers in Jobs to run Apply and Build."
            if "selected_runs" in registration and "remaining_runs" in registration:
                selected, remaining = registration["selected_runs"], registration["remaining_runs"]
                self._outcome = (
                    f"Setup job metadata created for {len(selected)} selected run(s): "
                    + (", ".join(map(str, selected)) or "none")
                    + f". Remaining series runs ({len(remaining)}): "
                    + (", ".join(map(str, remaining)) or "none")
                    + ". Start its workers in Jobs to run Apply and Build.")
            self._view().apply_build_reason.setText(self._outcome)
            jobs.status_label.setText(self._outcome)
            self._workspace().window().show_jobs()
        except Exception:
            self._failed(traceback.format_exc())

    def _failed(self, text):
        """Keep the literal preparation error beside the action and its traceback in Jobs."""
        self._outcome = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), text)
        view = self._view()
        if view is not None:
            view.apply_build_reason.setText(self._outcome)
        workspace = self._workspace()
        window = workspace.window() if workspace is not None else None
        jobs = getattr(window, "jobs_workspace", None)
        if jobs is not None:
            jobs.panel._error(text)
        elif callable(getattr(window, "print_log", None)):
            window.print_log(text)

    @Slot()
    def _closed(self):
        """Retire the finite call before allowing workspace navigation again."""
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self._starting = False
        view = self._view()
        if view is not None:
            view.busy_changed.emit(False)
            self.refresh()
