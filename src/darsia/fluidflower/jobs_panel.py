"""Native Jobs UI for the existing whole-run queue; no legacy execution engine."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import traceback
import uuid

from PySide6.QtCore import (
    QEvent,
    QObject,
    QProcess,
    QProcessEnvironment,
    QSignalBlocker,
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .job_control import publish_control, read_control
from .jobs_metadata import JobSnapshotReader, create_job_from_plan
from .staged_jobs import create_analysis_job
from .storage_profile import (
    DEFAULT_STORAGE_PROFILE,
    ESTIMATE_LABEL,
    PHASE_TWO_ROUTE,
    SERIES_ROUTE,
    estimate_job_bytes,
    job_storage_profile,
)
from .pipeline.job_queue import descriptor, name, require_unc, write_immutable
from .remote_worker_launch import (
    RemoteWorkerTarget,
    build_remote_worker_launch,
    launch_remote_worker,
)
from .job_launch_state import (
    UNKNOWN,
    begin_launch_request,
    declare_requests_stopped,
    launch_key,
    read_launch_requests,
    read_registered_launch_requests,
    record_launch_error,
)

CONTROL_BOUNDARY = "Pause and stop exit after the current committed frame. Requested control is not proof of worker retirement or completed results; outputs and the original claim are preserved."
SETUP_CONTROL_BOUNDARY = "Pause and stop exit at the next operation checkpoint. Requested control is not proof of worker retirement or completed results; outputs and the original claim are preserved."
COMPARISON_CONTROL_BOUNDARY = "Pause and stop exit after the current comparison result. Retrying restarts the whole requested-time task; no frame-prefix resume applies. Requested control is not proof of worker retirement or scientific acceptance."
COMPARISON_LAUNCH_SCOPE = "Each Start uses one CPU worker for one admitted requested time, then the service exits. Remaining times stay queued; start again after the service closes. Each time needs this machine's applicable qualification."
CONTROL_LEASE = "Waiting for the existing claim lease to expire (up to 10 minutes)"
REMOTE_GAP = "Remote start needs explicit existing SSH/checkout/interpreter facts and a confirmed PowerShell SSH default shell. Registration never contacts or installs a machine."


def _label(text=""):
    value = QLabel(text)
    value.setTextFormat(Qt.TextFormat.PlainText)
    value.setWordWrap(True)
    return value


# Job kind shown in the job list, from the job's own typed pipeline.
JOB_KINDS = {"series-setup": "Setup", "native-phase-statistics": "Phase maps",
             "n1-calibration-refit": "Calibration", "local-overflow-calibration-refit": "Calibration",
             "native-phase-n1": "Mass analysis", "lattice-phase-n1": "Mass analysis", "n1": "Mass analysis",
             "preset-wasserstein": "Comparisons", "wasserstein-comparison": "Comparisons"}


def job_kind(facts):
    """The job's category, named like the main workflow steps (Setup, Calibration, Mass analysis, Comparisons).

    Phase maps jobs make the phase maps and calibration statistics that both
    Calibration and Mass analysis use.
    """
    pipeline = (facts or {}).get("pipeline")
    return JOB_KINDS.get(pipeline, str(pipeline) if pipeline else "Not recorded")

def _table(headers):
    value = QTableWidget(0, len(headers))
    value.setHorizontalHeaderLabels(headers)
    value.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    value.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    value.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    value.horizontalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.ResizeToContents
    )
    value.horizontalHeader().setStretchLastSection(True)
    return value


def _fill(table, row, values):
    for column, value in enumerate(values):
        item = QTableWidgetItem(str(value))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(row, column, item)


def _rate(value):
    return (
        f"{value:.2f} frames/min" if isinstance(value, (int, float)) else "Not recorded"
    )


def _eta(value):
    return f"{value / 60:.1f} min" if isinstance(value, (int, float)) else "Unknown"


def _size(value):
    """Decimal units, as the documented per-file sizes (1 GB = 10^9 bytes)."""
    for unit, scale in (("TB", 10**12), ("GB", 10**9), ("MB", 10**6)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value} bytes"


def choice_estimate_text(profile, free_space=None):
    """The chosen profile's shared estimate and known free space, before Create."""
    series = estimate_job_bytes(profile, 1, pipeline=SERIES_ROUTE)
    phase_two = estimate_job_bytes(profile, 1, pipeline=PHASE_TWO_ROUTE)
    text = (
        f"Storage estimate for {profile} (documented reference size; larger image grids scale up): "
        f"{_size(series['per_frame_bytes'])} per frame in the native-phase-N1 series route, "
        f"{_size(phase_two['per_frame_bytes'])} per frame in phase two (N1)"
    )
    if profile == "lean":
        dropped = estimate_job_bytes(
            "lean", 1, pipeline=SERIES_ROUTE, colour_intermediates="drop"
        )
        text += (
            f", plus {_size(series['per_run_bytes'])} per run for its constant inputs "
            f"({_size(dropped['per_frame_bytes'])} per frame in the series route if the run "
            "configuration drops the colour intermediates)"
        )
    text += (
        f". {ESTIMATE_LABEL}. Create reports the whole-job estimate; no worker starts."
    )
    if free_space:
        text += (
            "\nFree space on this machine's results volume (last check): "
            + _size(free_space["free_bytes"])
        )
    else:
        text += "\nFree space: not checked on this machine yet; the worker checks fresh free space before every claim."
    return text


class _CallWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, call):
        super().__init__()
        self.call = call

    @Slot()
    def run(self):
        try:
            self.finished.emit(self.call())
        except Exception:
            self.failed.emit(traceback.format_exc())


START_ANALYSIS_NOTE = "Choose an already adopted, frozen execution-series plan. Create writes job files only; no worker is started and no image is computed."
LEAN_NOTE = "Lean uses the same computation as Full and keeps fewer intermediate files."

# Computed at (N144). Native is today's computation and the default. The coarse
# grids are the placeholder of spec section 3.5 and wait for the project owner's decision, so
# they are offered disabled. storage_resolution.COMPUTING_RESOLUTIONS_MM takes
# over from this tuple when N146 is integrated.
COARSE_COMPUTING_MM = (0.5, 1.0)
NOT_AVAILABLE_YET = "Not available yet"


def computing_resolution_combo():
    """The Computed at choice: native, and the coarse grids disabled."""
    combo = QComboBox()
    combo.addItem("Native (full resolution)", None)
    for step in COARSE_COMPUTING_MM:
        combo.addItem(f"{step:g} mm — {NOT_AVAILABLE_YET}", step)
        item = combo.model().item(combo.count() - 1)
        item.setEnabled(False)
        item.setToolTip("Computing the phase-map method on a coarser grid waits for a decision.")
    combo.setCurrentIndex(0)
    combo.setToolTip("Native is today's computation. The coarse grids are not available yet.")
    return combo


def stored_not_finer(stored_mm, computed_mm):
    """Stored cells may equal or exceed the computing grid, never be finer.

    None stands for native, the finest grid, in both arguments.
    """
    if computed_mm is None:
        return True
    return stored_mm is not None and float(stored_mm) >= float(computed_mm)


def stored_resolution_choices(computed_mm):
    """The stored sizes New analysis offers: native and the storage grids.

    None stands for native. A size finer than computed_mm is never offered.
    """
    from .storage_resolution import RESOLUTIONS_MM

    return [step for step in (None, *RESOLUTIONS_MM) if stored_not_finer(step, computed_mm)]


def read_plan_facts(plan, *, reader=None):
    """The adopted plan's runs, images, precision and bound storage request.

    The plan and its run configurations are read through the pipeline's own
    byte-bound reader; nothing is computed or written. ``storage_mm`` holds one
    value per run: None for native, else the stored cell size in millimetres.
    """
    if reader is None:
        from .staged_jobs import read_execution_plan as reader
    from .storage_resolution import resolution_mm

    value, configurations, _bound = reader(
        Path(plan["path"]), expected_sha256=plan["sha256"]
    )
    storage = []
    for config in configurations:
        request = config.get("storage")
        storage.append(
            resolution_mm(request.get("resolution_mm"))
            if isinstance(request, dict)
            else None
        )
    job = value.get("job") or {}
    facts = dict(
        runs=[config.get("run") for config in configurations],
        frames=sum(len(config.get("frames") or ()) for config in configurations),
        precision=job.get("precision"),
        storage_mm=storage,
    )
    if value.get("pipeline") == "native-phase-statistics":
        from .staged_jobs import phase1_route_blocker
        facts.update(pipeline=value["pipeline"], route_blocker=phase1_route_blocker(),
            gas_probability=all("gas_probability_storage" in config for config in configurations),
            running_pressure_runs=[config["run"] for config in configurations
                if "atmosphere_series" in config["statistics"]],
            phase_configuration=[config["phase"]["configuration"] for config in configurations],
            native_configuration=[config["native"]["configuration"] for config in configurations])
    elif value.get("pipeline") == "n1":
        facts.update(pipeline="n1", running_pressure_runs=[config["run"] for config in configurations
            if "atmosphere_series" in config], gas_probability=False, route_blocker="")
    return facts


def stored_at_text(storage_mm):
    """Stored at, as the adopted plan binds it for its runs (read-only)."""
    labels = ["native" if step is None else f"{step:g} mm" for step in storage_mm]
    distinct = list(dict.fromkeys(labels))
    if not distinct:
        return "Not bound: the plan has no runs."
    if len(distinct) == 1:
        return f"{distinct[0]}, bound in the adopted plan for all {len(labels)} runs (read-only)"
    return "Differs per run: " + ", ".join(distinct) + ", bound in the adopted plan (read-only)"


class StartAnalysisChoices(QObject):
    """The plan, job name, fresh job folder and file profile chosen before Create.

    The one implementation behind the New analysis page and the Start analysis
    dialog (N141): the same controls, labels, default, estimate and choices.
    The host is only the Qt parent, never a Python attribute: a reference back
    to it would put a parentless dialog in a reference cycle, and the cyclic
    garbage collector could then destroy it in a Jobs worker thread (Q160).
    """

    changed = Signal()

    def __init__(self, plans=(), host=None, *, free_space=None):
        super().__init__(host)
        self.plans, self.free_space, self._imported = [], None, []
        self.plan_combo = QComboBox()
        self.name_edit, self.job_edit = QLineEdit(), QLineEdit()
        self.job_edit.setPlaceholderText(r"\\server\share\DarSIA\jobs\fresh-job-name")
        # The job's storage profile is chosen here, once, before any worker.
        self.profile_combo = QComboBox()
        self.profile_combo.addItem(
            "Full — today's layout (references, qualification, exact gates)", "full"
        )
        self.profile_combo.addItem(
            "Lean — production layout: same computation, fewer stored files", "lean"
        )
        self.profile_combo.setCurrentIndex(
            self.profile_combo.findData(DEFAULT_STORAGE_PROFILE)
        )
        self.import_button = QPushButton("Import saved execution plan…")
        self.import_button.clicked.connect(self._import_plan)
        self.binding_label = _label()
        self.estimate_label = _label()
        self.plan_combo.currentIndexChanged.connect(self._update_binding)
        self.job_edit.textChanged.connect(self._update_binding)
        self.profile_combo.currentIndexChanged.connect(self._update_estimate)
        self.set_plans(plans, free_space=free_space)

    def set_plans(self, plans, *, free_space=None):
        """Offer these plans, and those imported here, with the known free space."""
        previous = (self.plan_combo.currentData() or {}).get("path")
        self.plans = deepcopy(list(plans))
        self.free_space = deepcopy(free_space)
        listed = {row["path"] for row in self.plans}
        with QSignalBlocker(self.plan_combo):
            self.plan_combo.clear()
            for row in self.plans + [r for r in self._imported if r["path"] not in listed]:
                self.plan_combo.addItem(str(row.get("label") or row["path"]), row)
            paths = [self.plan_combo.itemData(i)["path"] for i in range(self.plan_combo.count())]
            self.plan_combo.setCurrentIndex(paths.index(previous) if previous in paths else 0)
        self._update_binding()
        self._update_estimate()

    def ready(self):
        """A plan and a fresh job folder are chosen, so Create can be offered."""
        return bool(self.plan_combo.currentData()) and bool(self.job_edit.text().strip())

    def _update_estimate(self, *_):
        self.estimate_label.setText(
            choice_estimate_text(self.profile_combo.currentData(), self.free_space)
        )
        self.changed.emit()

    def _update_binding(self, *_):
        plan = self.plan_combo.currentData() or {}
        self.binding_label.setText(
            f"Plan: {plan.get('path') or 'Not selected'}\nSHA256: {plan.get('sha256') or 'Not selected'}\nThe existing producer validates adoption, ordered runs and all scientific bindings."
        )
        self.changed.emit()

    def _import_plan(self):
        host = self.parent()
        path, _ = QFileDialog.getOpenFileName(
            host, "Choose an existing adopted execution plan", "", "TOML (*.toml)"
        )
        if not path:
            return
        try:
            require_unc(path)
            bound = descriptor(Path(path))
        except Exception:
            QMessageBox.warning(host, "Could not select plan", traceback.format_exc())
            return
        row = dict(bound, label=Path(path).name)
        self._imported.append(row)
        self.plan_combo.addItem(row["label"], row)
        self.plan_combo.setCurrentIndex(self.plan_combo.count() - 1)

    def choices(self):
        return {
            "plan": deepcopy(self.plan_combo.currentData()),
            "job_root": self.job_edit.text().strip(),
            "display_name": self.name_edit.text().strip(),
            "storage_profile": self.profile_combo.currentData(),
        }


class StartAnalysisDialog(QDialog):
    """Choose an already adopted saved execution plan and a fresh UNC job root.

    The app's New analysis page replaces this dialog. A JobsPanel without that
    page still opens it, with the same StartAnalysisChoices.
    """

    def __init__(self, plans, parent=None, *, free_space=None):
        super().__init__(parent)
        self.setWindowTitle("Start analysis — create job metadata")
        layout = QVBoxLayout(self)
        layout.addWidget(_label(START_ANALYSIS_NOTE))
        self.form = StartAnalysisChoices(plans, self, free_space=free_space)
        for name in ("plans", "free_space", "plan_combo", "name_edit", "job_edit",
                     "profile_combo", "binding_label", "estimate_label"):
            setattr(self, name, getattr(self.form, name))
        form = QFormLayout()
        form.addRow("Saved execution plan", self.plan_combo)
        form.addRow("Job name", self.name_edit)
        form.addRow("Fresh UNC job folder", self.job_edit)
        form.addRow("Storage profile", self.profile_combo)
        layout.addLayout(form)
        layout.addWidget(self.form.import_button)
        layout.addWidget(self.binding_label)
        layout.addWidget(self.estimate_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Create job files"
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.form.changed.connect(self._update_ok)
        self._update_ok()

    def _update_ok(self):
        """Offer Create job files once a plan and a fresh job folder are chosen."""
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.form.ready()
        )

    def choices(self):
        """The chosen plan, job folder, job name and storage profile."""
        return self.form.choices()


class NewAnalysisPage(QWidget):
    """New analysis (N141, spec 3.1): What, Resolution and files, Where, Actions.

    It replaces the Start analysis dialog in the app, with the dialog's choices
    (StartAnalysisChoices) and the same two explicit actions: Create job files,
    then Start. The Jobs controls row lends its machine, backend and workers
    choice and the resource advisor while the page is shown. Opening the page
    starts nothing.
    """

    def __init__(self, panel, parent=None, *, facts_reader=read_plan_facts):
        super().__init__(parent)
        self.panel = panel
        self.facts_reader = facts_reader
        self.form = StartAnalysisChoices((), self)
        self.created_root = None
        self._pending_action = None
        self.facts = self.facts_error = None
        self._facts_key = None
        self._facts_generation = 0
        self._facts_thread = self._facts_worker = None
        self._facts_pending = False
        self._preparation_dialog = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # The page's one scroll area; the lent resource advisor has none here.
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(_label(START_ANALYSIS_NOTE))

        self.what = QGroupBox("What")
        what = QFormLayout(self.what)
        self.method_label = _label("Not known yet.")
        what.addRow("Series and method", self.method_label)
        plan_row = QHBoxLayout()
        plan_row.addWidget(self.form.plan_combo, 1)
        plan_row.addWidget(self.form.import_button)
        what.addRow("Saved execution plan", plan_row)
        what.addRow(self.form.binding_label)
        self.prepare_plan_button = QPushButton("Prepare a new analysis plan…")
        self.prepare_plan_button.clicked.connect(self.prepare_plan)
        what.addRow(self.prepare_plan_button)
        self.selection_label = _label()
        what.addRow("Image selection", self.selection_label)
        self.pipeline_label = _label()
        what.addRow("Analysis stage", self.pipeline_label)
        self.pressure_label = _label()
        what.addRow("Running pressure", self.pressure_label)
        self.probability_label = _label()
        what.addRow("Gas probability", self.probability_label)
        layout.addWidget(self.what)

        self.files = QGroupBox("Resolution and files")
        files = QFormLayout(self.files)
        self.computed_combo = computing_resolution_combo()
        files.addRow("Computed at", self.computed_combo)
        stored_row = QHBoxLayout()
        self.stored_combo = QComboBox()
        self.stored_combo.setEnabled(False)
        self.stored_combo.setToolTip(
            "The adopted plan binds the stored size. Another size needs a new plan "
            "revision for adoption, which New analysis cannot write yet."
        )
        stored_row.addWidget(self.stored_combo)
        self.stored_label = _label()
        stored_row.addWidget(self.stored_label, 1)
        files.addRow("Stored at", stored_row)
        files.addRow("File profile", self.form.profile_combo)
        files.addRow(_label(LEAN_NOTE))
        files.addRow(self.form.estimate_label)
        self.precision_label = _label()
        files.addRow("Precision", self.precision_label)
        layout.addWidget(self.files)

        self.where = QGroupBox("Where")
        where = QVBoxLayout(self.where)
        names = QFormLayout()
        names.addRow("Job name", self.form.name_edit)
        # The series' Jobs folder is set once (Jobs > Shared job folder); each
        # new job gets its own fresh folder there, shown rather than typed.
        self._suggested_job_root = ""
        self.form.job_edit.setReadOnly(True)
        self.form.job_edit.setToolTip(
            "Made in the series' Jobs folder (Jobs > Shared job folder) when the job files are created.")
        names.addRow("Job folder", self.form.job_edit)
        where.addLayout(names)
        self.machine_note = _label(
            "Create the job files first. The machine, backend and workers are then chosen for that job."
        )
        where.addWidget(self.machine_note)
        self.machine_box = QWidget()
        machine = QVBoxLayout(self.machine_box)
        machine.setContentsMargins(0, 0, 0, 0)
        self.target_layout = QHBoxLayout()
        machine.addLayout(self.target_layout)
        self.resource_layout = QVBoxLayout()
        machine.addLayout(self.resource_layout)
        where.addWidget(self.machine_box)
        layout.addWidget(self.where)

        self.actions = QGroupBox("Actions")
        actions = QVBoxLayout(self.actions)
        row = QHBoxLayout()
        self.create_button = QPushButton("Create job files")
        self.create_button.clicked.connect(self.create)
        self.start_button = QPushButton("Start workers for this job")
        self.start_button.clicked.connect(self.start)
        row.addWidget(self.create_button)
        row.addWidget(self.start_button)
        row.addStretch()
        actions.addLayout(row)
        self.action_label = _label("No job files created on this page yet.")
        actions.addWidget(self.action_label)
        layout.addWidget(self.actions)

        self.scroll.setWidget(body)
        outer.addWidget(self.scroll)
        self.form.changed.connect(self._choices_changed)
        self.computed_combo.currentIndexChanged.connect(self._show_facts)
        panel.job_created.connect(self._job_created)
        panel.job_creation_failed.connect(self._creation_failed)
        panel.local_worker_launch_requested.connect(self._start_requested)
        panel.remote_worker_launch_requested.connect(self._start_requested)
        panel.error_reported.connect(self._action_failed)
        self._show_facts()
        self.suggest_job_folder()

    def suggest_job_folder(self, *, fresh=False):
        """Fill a fresh ``production-<id>`` folder in the series' shared Jobs folder.

        The folder is only named here; the existing job creation makes it and
        checks it as before. Without a shared (UNC) Jobs folder the field stays
        empty and says where that folder is set. ``fresh`` names a new folder
        even when the shown one is still in the Jobs folder.
        """
        from pathlib import PureWindowsPath
        edit = self.form.job_edit
        root = getattr(self.panel, "series_root", None)
        folder = ""
        if root is not None:
            from .series_paths import series_folder_path
            try:
                folder = str(series_folder_path(root, "jobs"))
            except (OSError, ValueError, KeyError):
                folder = ""
        parent = PureWindowsPath(folder) if folder else None
        if parent is None or not parent.drive.startswith("\\\\"):
            if self._suggested_job_root and edit.text().strip() == self._suggested_job_root:
                edit.clear()
            self._suggested_job_root = ""
            edit.setPlaceholderText("No shared Jobs folder: set it under Jobs > Shared job folder.")
            return
        current = PureWindowsPath(edit.text().strip()) if edit.text().strip() else None
        if fresh or current is None or str(current.parent).casefold() != str(parent).casefold():
            self._suggested_job_root = str(parent / ("production-" + uuid.uuid4().hex))
            edit.setText(self._suggested_job_root)

    def set_context(self, *, method_text=""):
        """Series and method as Calibration recorded them (read-only)."""
        self.method_label.setText(method_text or "Not known yet.")

    def prepare_plan(self):
        from .analysis_preparation_ui import AnalysisPreparationDialog
        if self._preparation_dialog is not None:
            self._preparation_dialog.raise_()
            self._preparation_dialog.activateWindow()
            return
        dialog = AnalysisPreparationDialog(self.panel, self)
        self._preparation_dialog = dialog
        try:
            dialog.adopted.connect(self._prepared_adopted)
            dialog.exec()
        finally:
            self._preparation_dialog = None
            dialog.deleteLater()

    @Slot(object)
    def _prepared_adopted(self, bound):
        row = dict(bound, label=Path(bound["path"]).name)
        self.form._imported.append(row)
        self.form.plan_combo.addItem(row["label"], row)
        self.form.plan_combo.setCurrentIndex(self.form.plan_combo.count() - 1)

    def set_plans(self, plans, *, free_space=None):
        """Offer the adopted plans and the known free space, then read the plan."""
        with QSignalBlocker(self.form):
            self.form.set_plans(plans, free_space=free_space)
            self.suggest_job_folder()
        self._read_facts()

    def reset(self):
        """Forget the job created here, for another series."""
        self.created_root = None
        self._pending_action = None
        self.action_label.setText("No job files created on this page yet.")
        self.suggest_job_folder(fresh=True)
        self.sync_actions()

    def stored_allowed(self):
        """False only when the plan stores finer than the chosen computing grid."""
        if self.facts is None:
            return True
        computed = self.computed_combo.currentData()
        return all(stored_not_finer(step, computed) for step in self.facts["storage_mm"])

    def _choices_changed(self):
        """Read a newly chosen plan and update the actions."""
        plan = self.form.plan_combo.currentData() or {}
        if (plan.get("path"), plan.get("sha256")) != self._facts_key:
            self._read_facts()
        self.sync_actions()

    def _inspect_resources(self, plan, *, series_root, stores_path, storage_profile):
        """Preserve the staged default; Production supplies its profile admission."""
        from .staged_jobs import inspect_staged_resources
        return inspect_staged_resources(plan, series_root=series_root, stores_path=stores_path)

    def _read_facts(self):
        """Read the chosen plan's facts off the UI thread; one read at a time."""
        self._facts_generation += 1
        plan = self.form.plan_combo.currentData()
        self.facts = self.facts_error = None
        self._facts_key = ((plan or {}).get("path"), (plan or {}).get("sha256"))
        self._show_facts()
        if not plan:
            return
        if self._facts_thread is not None:
            self._facts_pending = True
            return
        key, generation, reader = self._facts_key, self._facts_generation, self.facts_reader
        series_root = self.panel.series_root
        provider = self.panel.local_stores_path_provider
        stores_path = provider() if provider is not None else None
        resource_reader = self._inspect_resources
        storage_profile = self.form.profile_combo.currentData()

        def read():
            try:
                facts = reader(plan)
                if facts.get("pipeline") in {"native-phase-statistics", "n1"} and not facts.get("route_blocker"):
                    admission = resource_reader(plan, series_root=series_root,
                        stores_path=stores_path, storage_profile=storage_profile)
                    facts["resource_admission"] = admission
                    facts["route_blocker"] = "" if admission["ready"] else admission["reason"]
                return dict(key=key, generation=generation, facts=facts)
            except Exception as error:
                return dict(key=key, generation=generation, error=f"{type(error).__name__}: {error}")

        self._facts_thread, self._facts_worker = self.panel._start_metadata_call(
            read, self._facts_ready, self._facts_closed
        )

    @Slot(object)
    def _facts_ready(self, result):
        """Show facts only for the latest request and its selected plan."""
        plan = self.form.plan_combo.currentData() or {}
        if (result["generation"] != self._facts_generation or result["key"] != self._facts_key
                or result["key"] != (plan.get("path"), plan.get("sha256"))):
            return
        self.facts, self.facts_error = result.get("facts"), result.get("error")
        self._show_facts()

    @Slot()
    def _facts_closed(self):
        """Retire the finished read before starting the latest queued request."""
        self._facts_thread = self._facts_worker = None
        if self._facts_pending:
            self._facts_pending = False
            self._read_facts()

    def _show_stored_choices(self):
        """Offer only stored sizes not finer than Computed at; select the plan's size."""
        steps = stored_resolution_choices(self.computed_combo.currentData())
        bound = set((self.facts or {}).get("storage_mm") or ())
        with QSignalBlocker(self.stored_combo):
            self.stored_combo.clear()
            for step in steps:
                self.stored_combo.addItem("native" if step is None else f"{step:g} mm", step)
            index = self.stored_combo.findData(next(iter(bound))) if len(bound) == 1 else -1
            if len(bound) == 1 and next(iter(bound)) is None:
                index = 0 if steps and steps[0] is None else -1
            self.stored_combo.setCurrentIndex(index)

    def _show_facts(self, *_):
        """Show the image selection, Stored at and Precision of the chosen plan."""
        self._show_stored_choices()
        facts = self.facts
        phase1 = (facts or {}).get("pipeline") == "native-phase-statistics"
        phase2 = (facts or {}).get("pipeline") == "n1"
        self.form.profile_combo.setEnabled(not (phase1 or phase2))
        if phase1 or phase2:
            self.form.profile_combo.setCurrentIndex(self.form.profile_combo.findData("full"))
        self.form.profile_combo.setToolTip("Full is required for this staged route." if phase1 or phase2 else "")
        self.pipeline_label.setText("Generate phase maps from model and collect calibration statistics" if phase1
                                    else "Mass from generated phase maps" if phase2
                                    else "As bound by the selected execution plan")
        pressure_runs = (facts or {}).get("running_pressure_runs", [])
        self.pressure_label.setText(
            f"Bound running values for {len(pressure_runs)} runs" if pressure_runs else "As bound by the execution plan")
        self.probability_label.setText(
            ("Stored: exact float32 decision values" if facts.get("gas_probability") else "Not requested")
            if phase1 else "As bound by the execution plan")
        if facts is None:
            if self.facts_error:
                text = "Not read: " + self.facts_error
            elif self.form.plan_combo.currentData():
                text = "Reading the adopted plan…"
            else:
                text = "Choose a saved execution plan."
            for label in (self.selection_label, self.stored_label, self.precision_label):
                label.setText(text)
        else:
            self.selection_label.setText(
                f"{len(facts['runs'])} runs, {facts['frames']} images, fixed by the adopted plan"
            )
            stored = stored_at_text(facts["storage_mm"])
            if not self.stored_allowed():
                stored += ". Finer than Computed at, which is not allowed."
            self.stored_label.setText(stored)
            self.precision_label.setText(
                f"{facts['precision'] or 'Not recorded'}, bound in the adopted plan (read-only)"
            )
        self.sync_actions()

    def sync_actions(self):
        """Create needs the choices; Start needs the job created on this page."""
        panel = self.panel
        plan = self.form.plan_combo.currentData() or {}
        facts_ready = (self.facts is not None and self.facts_error is None
            and self._facts_key == (plan.get("path"), plan.get("sha256")))
        blocker = (self.facts or {}).get("route_blocker", "")
        self.create_button.setEnabled(
            self.form.ready() and panel.create_button.isEnabled() and facts_ready
            and self.stored_allowed() and not blocker
        )
        if not blocker:
            if self.facts_error:
                blocker = "Could not read the selected execution plan: " + self.facts_error
            elif plan and not facts_ready:
                blocker = "Wait while the selected execution plan is read."
            elif not plan:
                blocker = "Choose a saved execution plan."
        self.create_button.setToolTip(blocker)
        current = (panel.current or {}).get("root")
        own = bool(self.created_root and current and Path(current) == Path(self.created_root))
        self.machine_box.setEnabled(own)
        self.machine_note.setVisible(not own)
        self.start_button.setEnabled(own and panel.start_button.isEnabled())
        self.start_button.setToolTip(
            panel.start_button.toolTip() if own else "Create the job files on this page first."
        )

    def create(self):
        """Create job files with the page's choices; no worker starts."""
        if not self.create_button.isEnabled():
            return
        self._pending_action = "create"
        self.action_label.setText("Creating job files… No worker is started.")
        self.panel.create_job(**self.form.choices())
        self.sync_actions()

    @Slot(object)
    def _job_created(self, registration):
        """Remember the job created here, so Start can follow."""
        if self._pending_action != "create":
            return
        self._pending_action = None
        self.created_root = (registration or {}).get("root")
        self.action_label.setText(
            f"Job files created: {self.created_root}. Choose the machine, backend and workers, then Start."
        )
        self.suggest_job_folder(fresh=True)  # A next job needs its own fresh folder.
        self.sync_actions()

    def start(self):
        """Start the created job on the chosen machine, as Jobs does."""
        if not self.start_button.isEnabled():
            return
        self._pending_action = "start"
        self.action_label.setText("Requesting start…")
        self.panel.start_selected_machine()
        if self._pending_action == "start":
            self._pending_action = None
            self.action_label.setText(
                "Start was not requested. " + self.panel.start_button.toolTip()
            )
        self.sync_actions()

    @Slot(object)
    def _start_requested(self, record):
        """Confirm only a start requested from this page."""
        if self._pending_action != "start":
            return
        self._pending_action = None
        self.action_label.setText("Start requested. Follow the progress on Jobs.")
        self.sync_actions()

    @Slot(str)
    def _action_failed(self, text):
        """Replace only a pending start with its synchronous reported failure."""
        if self._pending_action != "start":
            return
        self._show_action_failure("Start request failed. ", text)

    @Slot(str)
    def _creation_failed(self, text):
        """Show a failure reported by the active creator only."""
        if self._pending_action != "create":
            return
        self._show_action_failure("Job file creation failed. ", text)

    def _show_action_failure(self, prefix, text):
        """Retire pending feedback and show the final nonempty error line."""
        self._pending_action = None
        detail = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
        self.action_label.setText(prefix + detail)
        self.sync_actions()

    def can_close(self):
        """Wait for the selected-plan read and the active preparation dialog."""
        return self._facts_thread is None and (
            self._preparation_dialog is None or self._preparation_dialog.can_close())

    def prepare_close(self):
        if self._facts_thread is not None:
            self._facts_thread.cancel()
            self._facts_thread.deleteLater()
        self._facts_thread = self._facts_worker = None
        self._facts_pending = False
        self._facts_generation += 1


class RegisterMachineDialog(QDialog):
    """Record existing host/checkout/environment facts; never install or connect."""

    def __init__(self, parent=None, *, local_machine=None):
        super().__init__(parent)
        self.setWindowTitle("Register an existing machine")
        layout = QVBoxLayout(self)
        layout.addWidget(_label(REMOTE_GAP))
        form = QFormLayout()
        self.local_machine = local_machine or socket.gethostname()
        (
            self.machine_edit,
            self.host_edit,
            self.checkout_edit,
            self.python_edit,
            self.ssh_edit,
        ) = [QLineEdit() for _ in range(5)]
        for label, edit in [
            ("Machine name", self.machine_edit),
            ("Existing host", self.host_edit),
            ("Existing Git checkout", self.checkout_edit),
            ("Existing Python interpreter", self.python_edit),
            ("Existing SSH alias", self.ssh_edit),
        ]:
            form.addRow(label, edit)
        self.ssh_executable_edit = QLineEdit()
        self.ssh_executable_edit.setReadOnly(True)
        ssh_row = QHBoxLayout()
        ssh_row.addWidget(self.ssh_executable_edit)
        browse = QPushButton("Choose existing local ssh.exe…")
        browse.clicked.connect(self._choose_ssh)
        ssh_row.addWidget(browse)
        form.addRow("Local SSH executable", ssh_row)
        self.default_shell_combo = QComboBox()
        self.default_shell_combo.addItem("Not specified — remote start unavailable", "")
        self.default_shell_combo.addItem(
            "Confirmed SSH default shell: Windows PowerShell", "windows-powershell"
        )
        form.addRow("SSH server's actual command interpreter", self.default_shell_combo)
        self.new_version = QCheckBox(
            "Save as a new immutable version; keep earlier registration"
        )
        form.addRow(self.new_version)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("Register facts only")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for edit in (
            self.machine_edit,
            self.host_edit,
            self.checkout_edit,
            self.python_edit,
            self.ssh_edit,
            self.ssh_executable_edit,
        ):
            edit.textChanged.connect(self._validate)
        self.default_shell_combo.currentIndexChanged.connect(self._validate)
        self._validate()

    def _choose_ssh(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an already installed local ssh.exe", "", "ssh.exe (ssh.exe)"
        )
        if path and Path(path).is_file():
            self.ssh_executable_edit.setText(path)

    def _validate(self, *_):
        facts = self.facts()
        try:
            local = name(facts["machine"]) == name(self.local_machine)
        except ValueError:
            local = False
        basic = all(facts.get(key) for key in ("machine", "host", "checkout", "python"))
        remote = all(
            facts.get(key) for key in ("ssh_alias", "ssh_executable", "default_shell")
        )
        self.ok_button.setEnabled(bool(basic and (local or remote)))

    def facts(self):
        return {
            "machine": self.machine_edit.text().strip(),
            "host": self.host_edit.text().strip(),
            "checkout": self.checkout_edit.text().strip(),
            "python": self.python_edit.text().strip(),
            "ssh_alias": self.ssh_edit.text().strip(),
            "ssh_executable": self.ssh_executable_edit.text().strip(),
            "default_shell": self.default_shell_combo.currentData(),
            "remote_contacted": False,
            "installed": False,
        }


def _start_detached(process):
    outcome = process.startDetached()
    if isinstance(outcome, tuple):
        started, pid = outcome
    else:
        started, pid = bool(outcome), None
    return bool(started), int(pid) if started and pid is not None and int(pid) > 0 else None


def launch_local_worker(
    job_root,
    *,
    backend,
    workers,
    machine,
    python=sys.executable,
    cwd=None,
    launch_directory=None,
    stores=None,
    once=False,
):
    """Explicit existing worker command via detached QProcess, never a generated script.

    Detaching preserves valid workers when the GUI closes. Fresh launch console
    files retain startup failures before the service's own job log is opened.
    Launch PID is only a launcher fact, never evidence of numerical progress.
    """
    _admit_gui_job_root(job_root)
    if backend not in {"cpu", "cuda"} or type(workers) is not int or workers < 1:
        raise ValueError("Explicit CPU/CUDA backend and positive workers required")
    from .store_map import read_store_map
    from .pipeline.job_queue import read_json
    from darsia_cache.adapters import independent_worker_environment
    import os
    selected_job=read_json(Path(job_root)/"job.json")
    comparison = selected_job.get("pipeline") == "preset-wasserstein"
    if comparison and (backend != "cpu" or workers != 1):
        raise ValueError("Original Comparison starts require CPU and exactly one worker per service")
    selected_stores=read_store_map(stores)
    from darsia_cache.folders import folder_mode
    ordinary_cache = (selected_job.get("series_id") is not None
                      and folder_mode(selected_stores, selected_job["series_id"]))
    setup_service = (selected_job.get("pipeline") == "series-setup"
                     and "setup_cache" in selected_stores.document
                     and not ordinary_cache)
    if selected_job.get("pipeline") == "series-setup" and str(python) == str(sys.executable):
        from .phase_workflow import phase_python_executable
        python = phase_python_executable()
    if setup_service:
        from darsia_cache.setup_service import read_host
        read_host(selected_stores.document)  # Explicit existing policy; no guessed default.
        if selected_stores.machine_id.casefold() != str(machine).casefold() or backend != "cpu":
            raise ValueError("Setup requires its selected local CPU machine and store map")
        selected_stores.roots(selected_job["series_id"])
        child_environment = dict(os.environ)
    else:
        child_environment=independent_worker_environment(selected_stores,machine=machine,
            series_id=selected_job.get("series_id"),explicit_stores=stores is not None,
            parent_environment=os.environ, **({"job_root": job_root} if ordinary_cache else {}))
    if selected_job.get("pipeline") == "series-setup":
        import getpass
        from .job_control import ensure_setup_master
        ensure_setup_master(job_root, machine=machine,
            actor=selected_stores.document.get("operator", {}).get("name")
                  or os.environ.get("DARSIA_OPERATOR_NAME") or getpass.getuser())
    if cwd is None:
        from .pipeline import worker as implementation

        cwd = Path(implementation.__file__).resolve().parents[3]
    root = Path(job_root)
    job_ref = _job_reference(root, root / "job.json")
    launch_dir = (
        Path(launch_directory)
        if launch_directory is not None
        else root / "ui" / "launches" / uuid.uuid4().hex
    )
    console = launch_dir / "console.log"
    # Alias launch receipts are portable metadata, not copies of this host's argv.
    alias_console = (
        console.relative_to(root).as_posix() if "relative_path" in job_ref else None
    )
    launch_dir.mkdir(parents=True)
    worker_command = [
        str(python),
        "-m",
        "darsia.fluidflower.worker",
        "--job",
        str(root),
        "--backend",
        backend,
        "--workers",
        str(workers),
        "--machine",
        machine,
        "--watchdog",
    ]
    if comparison:
        worker_command.insert(1, "-B")
    if comparison or once:
        worker_command.append("--once")
    if setup_service:
        worker_command.insert(1, "-B")
        worker_command.append("--exit-when-idle")
        stores = selected_stores.source["path"]
    elif selected_job.get("pipeline") == "series-setup":
        worker_command.append("--exit-when-idle")
    if stores is not None:
        if not isinstance(stores, (str, Path)) or not str(stores).strip():
            raise ValueError("Choose an existing local store map before Start")
        worker_command.extend(["--stores", str(stores)])
    # N109: the stdlib supervisor starts the unchanged worker command once more
    # when it stops before the service started; it imports no DarSIA code.
    service_log = root / "logs" / (name(machine + "-" + backend) + ".log")
    command = [str(python), "-B", "-m", "darsia_service.prestart",
               "--service-log", str(service_log), "--", *worker_command]
    if setup_service:
        from .pipeline import worker as implementation
        command = [str(python), "-B", "-m", "darsia_cache.setup_service",
                   "--stores", str(stores), "--repository",
                   str(Path(implementation.__file__).resolve().parents[4]),
                   "--launch-directory", str(launch_dir),
                   "--", *worker_command]
    process = QProcess()
    environment=QProcessEnvironment()
    for key,value in child_environment.items():
        environment.insert(key,value)
    process.setProcessEnvironment(environment)
    process.setProgram(command[0])
    process.setArguments(command[1:])
    process.setWorkingDirectory(str(cwd))
    process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
    process.setStandardOutputFile(str(console))
    started, pid = _start_detached(process)
    record = {
        "format": "darsia-job-local-launch-v1",
        "job_root": str(root),
        "command": command,
        "cwd": str(cwd),
        "requested_utc": datetime.now(timezone.utc).isoformat(),
        "started": bool(started),
        "launcher_pid": int(pid) if started and pid is not None else None,
        "console": str(console),
        "numerical_progress_claimed": False,
        "detached": True,
        "error": process.errorString() if not started else "",
    }
    if "relative_path" in job_ref:
        # Keep actual local argv/cwd/stores only in this process. The shared
        # receipt identifies the request without leaking its machine-local map.
        record = dict(
            format="darsia-job-local-launch-v2",
            job=job_ref,
            machine=machine, backend=backend, workers=workers,
            entrypoint=("python -B -m darsia_cache.setup_service -- python -B -m darsia.fluidflower.worker"
                        if setup_service else "python -B -m darsia_service.prestart -- python -m darsia.fluidflower.worker"),
            requested_utc=record["requested_utc"],
            started=record["started"], launcher_pid=record["launcher_pid"],
            console=alias_console, detached=True,
            worker_liveness_verified=False, numerical_progress_claimed=False,
            error=record["error"],
        )
    write_immutable(launch_dir / "launch.json", record)
    if not started:
        raise RuntimeError(record["error"])
    return record


def register_machine_facts(job_root, facts, *, local_machine=None, new_version=False):
    """Explicit metadata registration only. Existing registrations are not replaced."""
    _admit_gui_job_root(job_root)
    machine = name(facts["machine"])
    if not all(
        isinstance(facts.get(key), str) and facts[key].strip()
        for key in ("host", "checkout", "python")
    ):
        raise ValueError(
            "Supply the existing host, checkout and interpreter explicitly"
        )
    if machine != name(local_machine or socket.gethostname()):
        target = RemoteWorkerTarget(
            **{
                key: facts.get(key, "")
                for key in (
                    "machine",
                    "ssh_alias",
                    "python",
                    "checkout",
                    "ssh_executable",
                    "default_shell",
                )
            }
        )
        build_remote_worker_launch(
            str(job_root), target=target, backend="cpu", workers=1
        )
        if not Path(target.ssh_executable).is_file():
            raise FileNotFoundError(
                "The explicitly chosen local ssh.exe does not exist"
            )
    record = {
        **deepcopy(facts),
        "format": "darsia-job-machine-registration-v1",
        "job": _job_reference(Path(job_root), Path(job_root) / "job.json"),
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "remote_contacted": False,
        "installed": False,
    }
    path = (
        Path(job_root)
        / "ui"
        / "machines"
        / (machine + ("." + uuid.uuid4().hex if new_version else "") + ".json")
    )
    write_immutable(path, record)
    return descriptor(path)


class JobsPanel(QWidget):
    """All automatic work is bounded metadata reading from explicit registrations."""

    job_created = Signal(object)
    job_creation_failed = Signal(str)
    machine_registered = Signal(object)
    local_worker_launch_requested = Signal(object)
    phase_one_launch_accepted = Signal(object)
    remote_worker_launch_requested = Signal(object)
    operator_declaration_recorded = Signal(object)
    workers_selected = Signal(object)
    job_selection_changed = Signal()
    error_reported = Signal(str)
    # A refusal reported with _refuse no longer applies (the next read completed).
    error_cleared = Signal(str)
    new_analysis_requested = Signal()
    # Emitted from the preparation thread; shown on the GUI thread.
    phase_map_progress = Signal(object)
    raw_model_requested = Signal()
    raw_review_registered = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        jobs_provider=None,
        plans_provider=None,
        machines_provider=None,
        reader=None,
        job_creator=None,
        local_launcher=None,
        local_machine=None,
        selection_provider=None,
        remote_launcher=None,
        known_jobs_provider=None,
        control_reader=None,
        control_publisher=None,
        local_stores_path_provider=None,
        raw_review_runner=None,
        raw_review_model_provider=None,
        raw_review_registrar=None,
        preset_wasserstein_exporter=None,
    ):
        super().__init__(parent)
        self.jobs_provider, self.plans_provider, self.machines_provider = (
            jobs_provider,
            plans_provider,
            machines_provider,
        )
        self.reader = reader or JobSnapshotReader()
        self.job_creator = job_creator or create_analysis_job
        self.local_launcher = local_launcher or launch_local_worker
        self.local_stores_path_provider = local_stores_path_provider
        self.remote_launcher = remote_launcher or launch_remote_worker
        self.local_machine = local_machine or socket.gethostname()
        self.selection_provider = selection_provider
        self.series_root = self.series_id = None
        self.known_jobs_provider = known_jobs_provider
        self.control_reader = control_reader or read_control
        self.control_publisher = control_publisher or publish_control
        self._known_launch_requests = []
        self._known_launch_errors = []
        self.series_label = ""
        self.registrations = []
        self.snapshots = []
        self.current = None
        self._read_thread = self._read_worker = self._create_thread = (
            self._create_worker
        ) = None
        self._generation = 0
        self.resource_panel = None
        self._launch_records = []
        self._refresh_pending = False
        self._session_requests = []
        self._control_thread = self._control_worker = None
        self._control_pending = False
        self._capacity_thread = self._capacity_worker = None
        self._known_free_space = None
        self._creation_estimates = {}
        self._build_worker_defaults = set()
        self._resume_after_control = None
        self.new_analysis = None
        self.comparison_preparation = None
        self._target_borrower = None
        self._raw_review_runner = raw_review_runner
        self._raw_review_model_provider = raw_review_model_provider
        self._raw_review_registrar = raw_review_registrar
        self._raw_review_seen = {}
        self._raw_review_observed = set()
        self._raw_review_queued = set()
        self._raw_review_pending = []
        self._raw_review_results = {}
        self._raw_review_thread = self._raw_review_worker = None
        self._raw_review_active = None
        layout = QVBoxLayout(self)
        self.context_label = _label("Jobs — no series selected")
        layout.addWidget(self.context_label)
        self.refresh_state_label = _label("Job metadata has not been refreshed.")
        layout.addWidget(self.refresh_state_label)
        # The job list at full height on the left, the selected job on the
        # right; no table, log or error text has a height cap (N141).
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        listing = QWidget()
        list_layout = QVBoxLayout(listing)
        list_layout.setContentsMargins(0, 0, 0, 0)
        details = QWidget()
        detail_layout = QVBoxLayout(details)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self.detail_tabs = QTabWidget()
        progress, files = QWidget(), QWidget()
        progress_layout, files_layout = QVBoxLayout(progress), QVBoxLayout(files)
        actions = QHBoxLayout()
        self.create_button, self.refresh_button = QPushButton(
            "Start analysis…"
        ), QPushButton("Refresh jobs")
        self.create_button.clicked.connect(self.open_create_dialog)
        self.refresh_button.clicked.connect(self.refresh)
        actions.addWidget(self.create_button)
        actions.addWidget(self.refresh_button)
        self.phase_map_purpose = QComboBox()
        self.phase_map_purpose.addItem("Calibration images", "calibration")
        self.phase_map_purpose.addItem("Production image grid", "production")
        self.phase_map_purpose.setToolTip(
            "The photographs the phase map job covers: the saved calibration selection, or the saved "
            "production image grid that mass analysis computes.")
        self.phase_map_shared = QCheckBox("Let other connected machines run it")
        self.phase_map_shared.setToolTip(
            "Off: only this machine runs the job. On: every connected machine may claim its runs; "
            "every input must then be in the shared folders.")
        actions.addWidget(self.phase_map_purpose)
        actions.addWidget(self.phase_map_shared)
        self.prepare_build_p1_button = QPushButton("Prepare phase map job")
        self.prepare_build_p1_button.setToolTip(
            "Prepare job metadata for the selected model and completed Setup. Start its workers in Jobs "
            "to generate phase maps in time order and collect calibration statistics.")
        self.prepare_build_p1_button.setEnabled(False)
        self.prepare_build_p1_button.clicked.connect(self.prepare_build_p1)
        actions.addWidget(self.prepare_build_p1_button)
        self.phase_map_progress.connect(self._phase_map_progress)
        actions.addStretch()
        list_layout.addLayout(actions)
        self.job_table = _table(
            ["Name", "Category", "Series", "Runs", "Frames", "Created", "State", "Execution reports"]
        )
        self.job_table.itemSelectionChanged.connect(self._selection_changed)
        list_layout.addWidget(self.job_table, 1)
        self.master_label, self.progress_label = _label("No active job"), _label("")
        detail_layout.addWidget(self.master_label)
        detail_layout.addWidget(self.progress_label)
        self.execution_label = _label()
        self.job_folder_label = _label()
        detail_layout.addWidget(self.execution_label)
        detail_layout.addWidget(self.job_folder_label)
        self.operation_label = _label()
        self.operation_label.hide()
        detail_layout.addWidget(self.operation_label)
        self.capacity_button = QPushButton("Check expected space on this machine")
        self.capacity_button.clicked.connect(self.check_capacity)
        self.capacity_label = _label("Expected space not checked. The worker checks fresh free space before claiming a run.")
        files_layout.addWidget(self.capacity_button)
        files_layout.addWidget(self.capacity_label)
        files_layout.addStretch()
        # Read-only: an existing job keeps the profile stored in its job.json.
        self.profile_label = _label("Storage profile: no job selected.")
        detail_layout.addWidget(self.profile_label)
        self.raw_review_button = QPushButton("Prepare RAW review…")
        self.raw_review_button.clicked.connect(self.prepare_raw_review)
        self.raw_review_status = _label(
            "RAW review is prepared after a completed run of this series, using its saved RAW acquisition preparations."
        )
        self.raw_review_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        files_layout.insertWidget(0, self.raw_review_button)
        files_layout.insertWidget(1, self.raw_review_status)
        self.raw_review_attach_button = QPushButton("Attach RAW review to selected model")
        self.raw_review_attach_button.clicked.connect(self.attach_raw_review)
        files_layout.insertWidget(1, self.raw_review_attach_button)
        from .preset_wasserstein_preparation_ui import PresetWassersteinExportPanel
        self.comparison_export = PresetWassersteinExportPanel(self, exporter=preset_wasserstein_exporter)
        files_layout.insertWidget(0, self.comparison_export)
        controls = QHBoxLayout()
        self.backend_combo, self.workers_spin = QComboBox(), QSpinBox()
        self.backend_combo.addItem("CPU", "cpu")
        self.backend_combo.addItem("CUDA", "cuda")
        self.workers_spin.setRange(1, 256)
        self.workers_spin.setPrefix("Workers: ")
        self.target_combo = QComboBox()
        self.target_combo.setMinimumContentsLength(16)
        self.target_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.target_combo.setToolTip("Machine on which to start this job's workers")
        self.backend_combo.setToolTip("Computation backend for the selected machine")
        self.workers_spin.setToolTip("Number of workers to start on the selected machine")
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.start_button, self.add_machine_button = QPushButton(
            "Start selected machine"
        ), QPushButton("Add machine…")
        self.start_button.clicked.connect(self.start_selected_machine)
        self.add_machine_button.clicked.connect(self.open_machine_dialog)
        controls.addWidget(self.target_combo)
        controls.addWidget(self.backend_combo)
        controls.addWidget(self.workers_spin)
        controls.addWidget(self.start_button)
        controls.addWidget(self.add_machine_button)
        control_actions = QHBoxLayout()
        self.pause_button, self.resume_button, self.stop_button = (
            QPushButton("Pause job after current frame"),
            QPushButton("Resume job"),
            QPushButton("Stop job after current frame"),
        )
        for button in (self.pause_button, self.resume_button, self.stop_button):
            button.setEnabled(False)
            button.setToolTip(CONTROL_BOUNDARY)
            control_actions.addWidget(button)
        self.pause_button.clicked.connect(self.pause_job)
        self.resume_button.clicked.connect(self.resume_job)
        self.stop_button.clicked.connect(self.stop_job)
        self.cancel_button = self.stop_button
        self.reset_button = QPushButton("Reset to Pending")
        self.restart_button = QPushButton("Restart unfinished work")
        self.reset_button.clicked.connect(lambda: self._request_recovery(reset=True))
        self.restart_button.clicked.connect(lambda: self._request_recovery(reset=False))
        for button in (self.reset_button, self.restart_button):
            button.setEnabled(False)
            control_actions.addWidget(button)
        self.declare_stopped_button = QPushButton("Declare previous service stopped…")
        self.declare_stopped_button.clicked.connect(
            self.declare_previous_service_stopped
        )
        self._controls = controls
        self.target_label = _label()
        detail_layout.addWidget(self.target_label)
        self.start_reason_label = _label()
        self.start_reason_label.hide()
        detail_layout.addWidget(self.start_reason_label)
        detail_layout.addLayout(controls)
        detail_layout.addLayout(control_actions)
        declaration = QHBoxLayout()
        declaration.addWidget(self.declare_stopped_button)
        declaration.addStretch()
        detail_layout.addLayout(declaration)
        self.launch_state_label = _label("No launch request for the selected target.")
        detail_layout.addWidget(self.launch_state_label)
        self.control_state_label = _label("Control unavailable: no job selected.")
        detail_layout.addWidget(self.control_state_label)
        self.control_note = _label(CONTROL_BOUNDARY + " " + REMOTE_GAP)
        detail_layout.addWidget(self.control_note)
        self.backend_combo.currentIndexChanged.connect(self._target_changed)
        self.workers_spin.valueChanged.connect(self._target_changed)
        self.machine_progress_note = _label("Per-machine operation counts, throughput and ETA are not recorded.")
        self.machine_progress_note.hide()
        progress_layout.addWidget(self.machine_progress_note)
        self.task_progress_table = _table([
            "Task", "Operation", "Machine (reported)", "Worker (reported)", "Progress",
            "Elapsed", "State (reported)", "Completed work", "Backend", "Current stage",
        ])
        self.task_progress_table.setToolTip(
            "Percentages estimate completed work, not remaining time. Stage counts are reported separately. "
            "Elapsed time uses original claims/start/end reports. Remote reports do not prove liveness.")
        self.task_progress_table.setWordWrap(False)
        self.task_progress_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column, width in enumerate((100, 70, 110, 125, 125, 90, 150, 110, 75, 220)):
            self.task_progress_table.setColumnWidth(column, width)
        progress_layout.addWidget(self.task_progress_table, 1)
        self.machine_table = _table(
            ["Machine", "Committed frames", "Average throughput", "ETA", "Scope"]
        )
        progress_layout.addWidget(self.machine_table, 1)
        self.worker_table = _table(
            [
                "Machine",
                "Worker",
                "Run",
                "Reported state",
                "Last actual progress",
                "Verbatim error",
            ]
        )
        progress_layout.addWidget(self.worker_table, 1)
        self.resource_slot = QVBoxLayout()
        self.resource_page = QWidget()
        self.resource_page.setLayout(self.resource_slot)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(2500)
        self.error_text = QPlainTextEdit()
        self.error_text.setReadOnly(True)
        self.detail_tabs.addTab(progress, "Progress")
        self.detail_tabs.addTab(self.log_text, "Log")
        self.detail_tabs.addTab(self.error_text, "Errors")
        self.detail_tabs.addTab(files, "Files")
        detail_layout.addWidget(self.detail_tabs, 1)
        self.splitter.addWidget(listing)
        self.splitter.addWidget(details)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        layout.addWidget(self.splitter, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(10000)
        self.timer.timeout.connect(self.refresh)
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(1000)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self._refresh_visibility_owner = self
        self._show_snapshot(None)

    def set_context(
        self,
        *,
        series_label="",
        series_root=None,
        series_id=None,
        jobs_provider=None,
        plans_provider=None,
        machines_provider=None,
        selection_provider=None,
    ):
        self._generation += 1
        changed = (series_root, series_id) != (self.series_root, self.series_id)
        self.series_label = str(series_label)
        self.series_root, self.series_id = series_root, series_id
        if changed and self.new_analysis is not None:
            self.new_analysis.reset()
        for key, value in (
            ("jobs_provider", jobs_provider),
            ("plans_provider", plans_provider),
            ("machines_provider", machines_provider),
        ):
            if value is not None:
                setattr(self, key, value)
        if selection_provider is not None:
            self.selection_provider = selection_provider
        self.create_button.setEnabled(
            bool(self.selection_provider and self.series_root and self.series_id)
        )
        self.create_button.setToolTip(
            ""
            if self.create_button.isEnabled()
            else "Create is blocked until the saved current-series B4 selection is bound to the existing execution plan."
        )
        self._sync_new_analysis()
        self.context_label.setText(
            "Jobs — " + (self.series_label or "No series selected")
        )
        self.set_jobs([])
        if self.jobs_provider is not None:
            try:
                self.set_jobs(self.jobs_provider())
            except Exception:
                self._error(traceback.format_exc())
        self.refresh()

    def set_jobs(self, registrations):
        rows = deepcopy(list(registrations or []))
        if any(
            not isinstance(row, dict) or not isinstance(row.get("root"), str)
            for row in rows
        ):
            raise ValueError("Explicit registered job roots required")
        if len({str(Path(row["root"])) for row in rows}) != len(rows):
            raise ValueError("Duplicate registered job roots")
        if sum(bool(row.get("active")) for row in rows) > 1:
            raise ValueError("More than one explicitly active job")
        self.registrations = rows
        if not rows:
            self.snapshots = []
            self.job_table.setRowCount(0)
            self._show_snapshot(None)

    def refresh(self, *_):
        if self._read_thread is not None or self._control_pending:
            self._refresh_pending = True
            self.refresh_state_label.setText("Job metadata refresh queued; the current metadata operation is finishing.")
            return
        if not self.registrations:
            self.refresh_state_label.setText("No registered job metadata to refresh.")
            self.progress_label.setText(
                "No jobs registered for this series. Open an existing job in Sources "
                "or use Start analysis to create job files."
            )
            return
        rows, generation = deepcopy(self.registrations), self._generation
        series_root = self.series_root
        self.refresh_state_label.setText("Reading registered job metadata…")

        def read_all(cancelled):
            result = []
            for row in rows:
                if cancelled():
                    return None
                try:
                    snapshot = dict(self.reader(row))
                    if cancelled():
                        return None
                    snapshot["archived"] = bool(row.get("archived"))
                    snapshot["control"] = self._read_control_snapshot(row["root"])
                    if snapshot["control"].get("error"):
                        snapshot["state"] = "Failed"
                    elif (Path(row["root"]) / "job.json").is_file():
                        from .job_lifecycle import user_status
                        snapshot = user_status(snapshot, local_machine=self.local_machine,
                            stores_path=self.local_stores_path_provider() if self.local_stores_path_provider else None)
                    if self._raw_review_applicable(snapshot):
                        from .raw_review_postrun import declared_preparations
                        facts = snapshot.get("job_facts") or {}
                        try:
                            snapshot["raw_review_preparations"] = declared_preparations(
                                series_root, expected_series=facts.get("series_id"),
                                runs=facts.get("expected_runs") or ())
                        except (OSError, ValueError, TypeError, KeyError) as error:
                            snapshot["raw_review_reason"] = "RAW review is unavailable: " + str(error)
                    result.append(snapshot)
                except Exception:
                    result.append(
                        {
                            "root": row["root"],
                            "name": row.get("name", Path(row["root"]).name),
                            "series": row.get("series_label", self.series_label),
                            "state": "Metadata error",
                            "errors": [traceback.format_exc()],
                        }
                    )
            if cancelled():
                return None
            requests, errors = self._read_known_launches()
            return {
                "generation": generation,
                "snapshots": result,
                "known_launch_requests": requests,
                "known_launch_errors": errors,
            }

        self._read_thread, self._read_worker = self._start_metadata_call(
            read_all, self._read_ready, self._read_closed, failed=self._read_failed,
            cancellable=True,
        )

    def _start_metadata_call(self, call, ready, closed, *, failed=None, cancellable=False):
        """Disposable reads cannot own a QThread that keeps the window alive.

        Only reads use this route. Creation, launch, control and publication
        retain _start_call and their existing close guards.
        """
        from .ui_metadata_preview import MetadataPreview
        preview = MetadataPreview(self)
        def returned(value, error):
            try:
                if error:
                    (failed or self._error)(error)
                else:
                    ready(value)
            finally:
                closed()
                preview.deleteLater()
        preview.completed.connect(returned)
        preview.start(call if cancellable else lambda cancelled: call())
        return preview, preview

    def _start_call(self, call, ready, closed, *, failed=None):
        thread = QThread(self)
        worker = _CallWorker(call)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(ready)
        worker.failed.connect(failed or self._error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        # No deleteLater for the worker. Deleted in its own thread, it needs the
        # GIL while Qt holds a signal-slot lock (the Python QThread's
        # disconnectNotify), and a GUI thread that holds the GIL and waits for
        # that lock deadlocks (Q185). The *_closed slots drop its Python owner
        # after wait(), which frees it in the GUI thread.
        thread.finished.connect(closed)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        return thread, worker

    @Slot(object)
    def _read_ready(self, result):
        try:
            self._show_read_result(result)
        except Exception:
            self._read_failed(traceback.format_exc())
        else:
            self._clear_transient_error()

    @Slot(str)
    def _read_failed(self, text):
        message = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), text)
        self.refresh_state_label.setText("Job metadata refresh failed: " + message[:240])
        self._error(text)

    def _show_read_result(self, result):
        from .jobs_execution_display import execution_report_text

        if result["generation"] != self._generation:
            return
        previous = self.current.get("root") if self.current else None
        self.snapshots = result["snapshots"]
        self._known_launch_requests = result.get("known_launch_requests", [])
        self._known_launch_errors = result.get("known_launch_errors", [])
        with QSignalBlocker(self.job_table):
            self.job_table.setRowCount(len(self.snapshots))
            setup_jobs = [self._is_setup_job(data) for data in self.snapshots]
            comparison_jobs = [self._is_comparison_job(data) for data in self.snapshots]
            self.job_table.horizontalHeaderItem(3).setText(
                ("Requested times" if all(comparison_jobs) else
                 "Runs / operations / times" if any(setup_jobs) else "Runs / times")
                if any(comparison_jobs) else
                "Operations" if setup_jobs and all(setup_jobs) else "Runs / operations" if any(setup_jobs) else "Runs")
            self.job_table.horizontalHeaderItem(4).setText(
                ("Comparison result records" if all(comparison_jobs) else "Progress")
                if any(comparison_jobs) else
                "Progress" if setup_jobs and all(setup_jobs) else "Images / operations" if any(setup_jobs) else "Frames")
            for row, data in enumerate(self.snapshots):
                frames = f"{data.get('committed', '?')} / {data.get('total_frames') if data.get('total_frames') is not None else '?'}"
                runs = f"{data.get('completed_runs', '?')} / {data.get('run_count', '?')}"
                if self._is_setup_job(data):
                    frames = f"{data.get('completed_operations', '?')} / {data.get('total_operations', '?')} operations"
                    runs = str(data.get("total_operations", "?"))
                elif self._is_comparison_job(data):
                    frames = f"{data.get('completed_comparisons', '?')} / {data.get('total_comparisons', '?')} comparison result records"
                _fill(
                    self.job_table,
                    row,
                    [
                        data["name"],
                        job_kind(data.get("job_facts")),
                        (data.get("job_facts") or {}).get("series_id")
                        or data.get("series", "Not recorded"),
                        runs,
                        frames,
                        data.get("created_utc", "Not recorded"),
                        data["state"],
                        execution_report_text(data, self.local_machine, compact=True),
                    ],
                )
                self.job_table.item(row, 7).setToolTip(
                    execution_report_text(data, self.local_machine)
                )
            active = next(
                (r["root"] for r in self.registrations if r.get("active")), None
            )
            desired = previous or active
            index = next(
                (i for i, s in enumerate(self.snapshots) if s["root"] == desired), 0
            )
            if self.snapshots:
                self.job_table.selectRow(index)
        self._selection_changed()

        self._observe_raw_review_completion()
        self.refresh_state_label.setText("Job metadata refreshed: "
            + datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @staticmethod
    def _raw_review_applicable(data):
        facts = (data or {}).get("job_facts") or {}
        return (facts.get("format") == "darsia-alias-job-v1"
                and facts.get("pipeline") in (None, "native-phase-n1"))

    @staticmethod
    def _raw_review_eligible(data):
        """Require verified completion and declared preparations for every source run."""
        facts = (data or {}).get("job_facts") or {}
        preparations = (data or {}).get("raw_review_preparations")
        runs = facts.get("expected_runs") or ()
        return bool(data and data.get("complete") is True and not data.get("errors")
            and facts.get("format") == "darsia-alias-job-v1"
            and isinstance(preparations, dict) and runs and set(preparations) == set(runs)
            and facts.get("pipeline") in (None, "native-phase-n1"))


    def _observe_raw_review_completion(self):
        """Observe actual transitions; never start an old complete job on navigation."""
        for data in self.snapshots:
            root = data["root"]
            complete = self._raw_review_eligible(data)
            was_complete = self._raw_review_seen.get(root)
            self._raw_review_seen[root] = complete
            if was_complete is False and complete and root not in self._raw_review_observed:
                self._raw_review_observed.add(root)
                self._queue_raw_review(root, interactive=False)


    def _queue_raw_review(self, root, *, interactive, preparations=None):
        """Queue one review preparation with a snapshot of its declared inputs."""
        if root in self._raw_review_queued:
            return
        if preparations is None:
            snapshot = next((row for row in self.snapshots if row["root"] == root), None)
            snapshot = snapshot or (self.current if (self.current or {}).get("root") == root else {})
            declared = (snapshot or {}).get("raw_review_preparations") or {}
            preparations = tuple(deepcopy(list(declared.values())))
        self._raw_review_queued.add(root)
        self._raw_review_pending.append(dict(root=root, interactive=interactive,
            preparations=preparations, series_root=self.series_root))
        self._start_raw_review()


    def _start_raw_review(self):
        """Start the next queued review operation on a finite background thread."""
        if self._raw_review_thread is not None or not self._raw_review_pending:
            return
        task = self._raw_review_pending.pop(0)
        self._raw_review_active = task
        try:
            stores = self.local_stores_path_provider() if self.local_stores_path_provider else None
        except Exception as error:
            self._raw_review_results[task["root"]] = "RAW review was not prepared: " + str(error)
            self._raw_review_queued.discard(task["root"])
            self._raw_review_active = None
            self._show_raw_review_state()
            self._start_raw_review()
            return
        runner = self._raw_review_runner
        def run():
            from .raw_review_postrun import (ReviewInputsMissing, prepare_and_run, default_output)
            try:
                if runner is not None:
                    result = runner(task["root"], preparations=task["preparations"],
                        stores_path=stores, series_root=task["series_root"])
                else:
                    if stores is None or task["series_root"] is None:
                        raise ValueError("Choose this machine's registered local store map and this job's series first")
                    result = prepare_and_run(task["root"], stores, task["preparations"],
                        series_root=task["series_root"],
                        output_root=default_output(task["root"], task["series_root"]))
                location = result.get("review_root")
                return dict(root=task["root"], complete=True, result=result,
                    text=("RAW photographs are ready" + (" at " + str(location) if location else "")
                          + ". The original production result is unchanged."))
            except ReviewInputsMissing as error:
                return dict(root=task["root"], complete=False, inputs_missing=True,
                    interactive=task["interactive"], text=str(error))
            except Exception as error:
                return dict(root=task["root"], complete=False,
                    text="RAW review was not prepared: " + str(error) + ". Existing files are preserved; there is no automatic retry.")
        self._raw_review_results[task["root"]] = "Preparing the finished job's RAW review; production results remain unchanged."
        self._raw_review_thread, self._raw_review_worker = self._start_call(
            run, self._raw_review_ready, self._raw_review_closed)
        self._show_raw_review_state()


    @Slot(object)
    def _raw_review_ready(self, value):
        """Display the completed operation's result for its original job."""
        self._raw_review_results[value["root"]] = value["text"]
        self._show_raw_review_state()


    @Slot()
    def _raw_review_closed(self):
        """Retire the finished thread before releasing its job and starting another."""
        if self._raw_review_thread is not None:
            self._raw_review_thread.wait()
        self._raw_review_thread = self._raw_review_worker = None
        finished = self._raw_review_active
        self._raw_review_active = None
        if finished is not None:
            self._raw_review_queued.discard(finished["root"])
        self._show_raw_review_state()
        self._start_raw_review()


    @Slot()
    def prepare_raw_review(self):
        """Queue the selected eligible job or explain which input is missing."""
        if not self._raw_review_eligible(self.current):
            self.raw_review_status.setText((self.current or {}).get("raw_review_reason") or
                "RAW review needs a verified completed run and this series' declared RAW acquisition preparations.")
            return
        self._queue_raw_review(self.current["root"], interactive=True)


    def _show_raw_review_state(self):
        """Refresh review actions and explain eligibility or operation progress."""
        root = (self.current or {}).get("root")
        applicable = self._raw_review_applicable(self.current)
        for widget in (self.raw_review_button, self.raw_review_attach_button,
                       self.raw_review_status):
            widget.setVisible(applicable)
        eligible = self._raw_review_eligible(self.current)
        reason = (self.current or {}).get("raw_review_reason") or (
            "This action needs a completed run with verified final publications and this series' declared RAW acquisition preparations.")
        self.raw_review_button.setEnabled(eligible and root not in self._raw_review_queued)
        self.raw_review_attach_button.setEnabled(
            eligible and self._raw_review_thread is None and not self._raw_review_pending)
        self.raw_review_attach_button.setToolTip(
            "Attach already prepared RAW photographs and published phase maps to the exact model selected in Phase model → Basis."
            if eligible else reason)
        self.raw_review_button.setToolTip(
            "Use the original RAW acquisition preparations to prepare this finished job's review images."
            if eligible else reason)
        self.raw_review_status.setText(self._raw_review_results.get(root,
            "Choose Prepare RAW review for an already completed job. With the GUI closed, run the same saved post-run request explicitly."
            if eligible else reason))

    def attach_raw_review(self):
        """Bind the original predictions only after an explicit model choice."""
        if self._raw_review_thread is not None or self._raw_review_pending:
            self.raw_review_status.setText("Wait for the current RAW review operation to finish before attaching a model.")
            return
        if not self._raw_review_eligible(self.current):
            self.raw_review_status.setText((self.current or {}).get("raw_review_reason") or
                "Prepare a completed run's RAW review before attaching it to a model.")
            return
        try:
            model = self._raw_review_model_provider() if self._raw_review_model_provider else None
        except Exception as error:
            self.raw_review_status.setText("Could not read the selected model: " + str(error))
            return
        if not model or not model.get("root"):
            self.raw_review_status.setText("Choose the original saved model in Phase model → Basis, then attach this review.")
            self.raw_model_requested.emit()
            return
        root, series_root, model_root = self.current["root"], self.series_root, model["root"]
        if root in self._raw_review_queued:
            return
        registrar = self._raw_review_registrar
        def register():
            from .raw_review_postrun import default_output
            from .raw_production_review import register_review
            try:
                completion = default_output(root, series_root) / "postrun_completed.json"
                binding = (registrar or register_review)(completion, model_root)
                return dict(root=root, complete=True, binding=binding,
                    text="RAW review attached to the selected model. Open Review and label in Phase model; scientific acceptance remains separate.")
            except Exception as error:
                return dict(root=root, complete=False,
                    text="RAW review was not attached: " + str(error))
        self._raw_review_queued.add(root)
        self._raw_review_active = dict(root=root)
        self._raw_review_results[root] = "Checking the original prediction and model bindings before attaching this RAW review."
        self._raw_review_thread, self._raw_review_worker = self._start_call(
            register, self._raw_review_registration_ready, self._raw_review_closed)
        self._show_raw_review_state()

    @Slot(object)
    def _raw_review_registration_ready(self, value):
        """Report registration and refresh model metadata after a successful binding."""
        self._raw_review_ready(value)
        if value.get("complete") and value.get("binding"):
            self.raw_review_registered.emit(value["binding"])


    @Slot()
    def _read_closed(self):
        self._read_thread = self._read_worker = None
        try:
            self._target_changed()
        except Exception:
            self._read_failed(traceback.format_exc())
        if self._refresh_pending:
            self._refresh_pending = False
            self.refresh()

    def _cancel_metadata_refresh(self):
        """Retire only passive display ownership, never a control publication."""
        preview = self._read_thread
        if preview is not None:
            preview.cancel()
            preview.deleteLater()
        self._read_thread = self._read_worker = None
        self._refresh_pending = False

    def _selection_changed(self):
        row = self.job_table.currentRow()
        self._show_snapshot(
            self.snapshots[row] if 0 <= row < len(self.snapshots) else None
        )

    @Slot()
    def check_capacity(self):
        if self._capacity_thread is not None or self.current is None:
            return
        root = self.current["root"]
        generation = self._generation
        selected_workers = self.workers_spin.value()
        stores_path = self.local_stores_path_provider() if self.local_stores_path_provider else None
        forbidden = tuple(p for p in (self.series_root, Path(__file__).resolve().parents[3]) if p)
        self.capacity_button.setEnabled(False)
        self.capacity_label.setText("Reading expected allocation and fresh free space on this machine…")
        def read_capacity():
            from .store_map import read_store_map
            from .pipeline.capacity import report_job_root
            try:
                if stores_path is None:
                    raise ValueError("Choose this machine's local store map")
                stores = read_store_map(stores_path,forbidden_roots=forbidden)
                job = self.reader.read(Path(root) / "job.json")
                if job.get("execution_mode") == "current_build":
                    from .pipeline.current_build_job import worker_recommendation
                    result = worker_recommendation(job, stores)
                else:
                    result = report_job_root(root, stores)
            except (OSError,ValueError) as error:
                result = dict(available=False,reason=str(error))
            return dict(generation=generation,root=root,result=result,selected_workers=selected_workers)
        self._capacity_thread,self._capacity_worker=self._start_metadata_call(
            read_capacity,self._capacity_ready,self._capacity_closed)
        self._sync_capacity_action()

    @Slot(object)
    def _capacity_ready(self, value):
        if value["generation"] != self._generation or value["root"] != (self.current or {}).get("root"):
            return
        result = value["result"]
        current_build = ((self.current or {}).get("job_facts") or {}).get("execution_mode") == "current_build"
        first_default = value["root"] not in self._build_worker_defaults
        if current_build:
            self._build_worker_defaults.add(value["root"])
        if current_build and result.get("available") is not False:
            suggested = result["recommended_workers"]
            if (first_default and suggested > 0
                    and self.workers_spin.value() == value.get("selected_workers", self.workers_spin.value())):
                self.workers_spin.setValue(suggested)
            self.capacity_label.setText(result["message"])
            self.workers_spin.setToolTip(result["message"])
            self._target_changed()
            return
        if result.get("available") is False:
            text = "Expected space unavailable: " + result["reason"]
        else:
            text = result["message"] + (" — enough by this estimate." if result["sufficient"] else " — insufficient by this estimate.")
            text += " Includes preserved native outputs, optional storage additions, estimated temporary space and reserve. Results: estimate, not coordinated across hosts. Fresh check repeats before claim; cache admission is separate."
            if type(result.get("free_bytes")) is int:
                # Free space this panel knows, shown when choosing a new job's profile.
                self._known_free_space = dict(free_bytes=result["free_bytes"],
                                              series_id=self.series_id,
                                              results_root=result.get("results_root"))
        self.capacity_label.setText(text)

    @Slot()
    def _capacity_closed(self):
        self._capacity_thread=self._capacity_worker=None
        self._sync_capacity_action()
        if (((self.current or {}).get("job_facts") or {}).get("execution_mode") == "current_build"
                and self.current["root"] not in self._build_worker_defaults):
            self.check_capacity()

    def _sync_capacity_action(self):
        if not self.current:
            reason = "Select a job before checking expected space on this machine."
        elif not self.local_stores_path_provider:
            reason = "Configure this machine's local storage settings before checking expected space."
        elif ((self.current.get("job_facts") or {}).get("format") != "darsia-alias-job-v1"
                and (self.current.get("job_facts") or {}).get("execution_mode") != "current_build"):
            reason = "This job has no supported execution metadata. Choose a job created from a saved execution plan."
        elif self._capacity_thread is not None:
            reason = "Checking expected space. Wait for the result."
        else:
            reason = ""
        self.capacity_button.setEnabled(not reason)
        self.capacity_button.setToolTip(reason or
            "Check this job's expected space against the current free space on this machine.")

    def _show_snapshot(self, data):
        from .jobs_execution_display import execution_report_text

        old_root = (self.current or {}).get("root")
        self.current = data
        self.prepare_build_p1_button.setEnabled(bool(self._is_setup_job(data)
            and data.get("complete") and self._create_thread is None))
        self.comparison_export.set_snapshot(data)
        self.execution_label.setText(execution_report_text(data, self.local_machine))
        self.job_folder_label.setText("Job folder: " + str(data["root"]) if data else "")
        self._show_raw_review_state()
        if (data or {}).get("root") != old_root:
            self.capacity_label.setText(
                "Expected space not checked in this view. The Setup worker checks fresh free space before claiming an operation."
                if self._is_setup_job(data) else
                "Expected space not checked. The worker checks fresh free space before claiming a run.")
        self.profile_label.setText(self._profile_text(data))
        self._sync_capacity_action()
        enabled = bool(data and data.get("job_facts"))
        self.start_button.setEnabled(enabled)
        self.add_machine_button.setEnabled(enabled)
        self._refresh_targets()
        self.master_label.setText(
            f"Master: {data.get('master_machine', 'Not recorded')} · {data.get('name')} · {data.get('state')}"
            if data
            else "No active job"
        )
        self._show_progress_view()
        operation_job = self._is_setup_job(data) or self._is_comparison_job(data)
        self.operation_label.setVisible(False)
        stages = {"apply": "Apply geometry", "build": "Build run model", "image-preparation": "Image correction"}
        self.operation_label.setText("\n".join(
            f"{stages.get(row.get('operation'), 'Setup operation')}"
            + ("" if row.get("operation") == "image-preparation" else f" · {row['run']}")
            + f": {row['state']}"
            for row in (data or {}).get("runs", [])
        ) if self._is_setup_job(data) else "")
        if self._is_comparison_job(data):
            self.operation_label.setText("\n".join(
                f"Requested time {(row.get('comparison_time') or {}).get('time_hours', '?')} h"
                f" · tolerance {(row.get('comparison_time') or {}).get('tolerance_hours', '?')} h"
                f" · {row['run']}: {row['state']}" + self._comparison_result_evidence(row)
                for row in (data or {}).get("runs", [])))
        self.machine_table.hide()
        self.worker_table.hide()
        self.machine_progress_note.hide()
        self.machine_progress_note.setText(
            "Per-machine comparison result counts, throughput and ETA are not recorded."
            if self._is_comparison_job(data) else
            "Per-machine operation counts, throughput and ETA are not recorded.")
        self.machine_table.setRowCount(len((data or {}).get("machines", [])))
        for row, facts in enumerate((data or {}).get("machines", [])):
            _fill(
                self.machine_table,
                row,
                [
                    facts["machine"],
                    facts["committed"],
                    _rate(facts.get("frames_per_minute")),
                    _eta(facts.get("eta_seconds")),
                    facts.get("rate_scope", ""),
                ],
            )
        self.worker_table.setRowCount(len((data or {}).get("workers", [])))
        for row, facts in enumerate((data or {}).get("workers", [])):
            _fill(
                self.worker_table,
                row,
                [
                    facts["machine"],
                    facts["worker"],
                    facts["run"],
                    facts["state"],
                    facts["last_progress_utc"],
                    facts["error"],
                ],
            )
        logs = list((data or {}).get("logs", []))
        launches = list(self._launch_records) + self._job_requests()
        console_errors = []
        shown = set()
        for launch in launches:
            if launch.get("job_root") == (data or {}).get("root") and launch.get(
                "console"
            ):
                path = Path(launch["console"])
                if str(path) in shown:
                    continue
                try:
                    if not path.exists():
                        continue
                    shown.add(str(path))
                    with path.open("rb") as stream:
                        stream.seek(0, 2)
                        size = stream.tell()
                        stream.seek(max(0, size - 128 * 1024))
                        logs.append(
                            {
                                "path": str(path),
                                "tail": stream.read().decode("utf-8", errors="replace"),
                            }
                        )
                except OSError as error:
                    shown.add(str(path))
                    console_errors.append(f"Launch console unavailable: {path}\n{error}")
                    logs.append(dict(path=str(path), tail="Launch console unavailable: " + str(error)))
        self.log_text.setPlainText(
            "\n".join(f"{item['path']}\n{item['tail']}" for item in logs)
        )
        failures = [
            row.get("error", "")
            for row in (data or {}).get("runs", [])
            if row.get("error")
        ]
        failures.extend(
            row["error_traceback"]
            for row in self._job_requests()
            if row.get("error_traceback")
        )
        failures.extend(console_errors)
        self.error_text.setPlainText(
            "\n".join((data or {}).get("errors", []) + failures)
        )
        if self.resource_panel is not None:
            self.resource_panel.set_context(
                (data or {}).get("job_facts"),
                series_label=self.series_label,
                machines_provider=self.machines_provider,
            )
        if (((data or {}).get("job_facts") or {}).get("execution_mode") == "current_build"
                and data["root"] not in self._build_worker_defaults and self._capacity_thread is None):
            self.check_capacity()
        self.job_selection_changed.emit()

    def _show_progress_view(self):
        from .jobs_progress import progress_view

        view = progress_view(self.current)
        self.progress_label.setText(view["summary"])
        self._progress_summary = view["summary"]
        self.task_progress_table.setRowCount(len(view["rows"]))
        keys = ("task", "operation", "machine", "worker", "percent", "elapsed", "state", "work", "backend", "stage")
        for index, row in enumerate(view["rows"]):
            _fill(self.task_progress_table, index, [row[key] for key in keys])
            for column in range(len(keys)):
                self.task_progress_table.item(index, column).setToolTip(
                    str(row[keys[column]]) + "\nStarted: " + str(row["started_utc"] or "Unknown")
                    + "\nFinished: " + str(row["finished_utc"] or "Unknown")
                    + "\n" + self.task_progress_table.toolTip())
        if self._is_comparison_job(self.current):
            self.progress_label.setToolTip(
                "Queue completion does not certify finite distances or scientific acceptance."
                + self._comparison_result_evidence(self.current))
        else:
            self.progress_label.setToolTip(self.task_progress_table.toolTip())

    def _update_elapsed(self):
        """Advance visible clocks from retained timestamps; never refresh metadata."""
        if not self._refresh_visibility_owner.isVisible() or self.current is None:
            return
        from .jobs_progress import progress_view

        view = progress_view(self.current)
        if self.progress_label.text() == self._progress_summary:
            self.progress_label.setText(view["summary"])
            self._progress_summary = view["summary"]
        for index, row in enumerate(view["rows"]):
            item = self.task_progress_table.item(index, 5)
            if item is not None:
                item.setText(row["elapsed"])

    @staticmethod
    def _setup_progress_text(data):
        """Show original reported stages separately from verified completions."""
        reports = [row["reported_progress"] for row in data.get("runs", [])
                   if row.get("reported_progress")]
        if not reports:
            return ""
        lines = []
        for report in reports:
            timestamp = datetime.fromisoformat(report["reported_utc"].replace("Z", "+00:00"))
            lines.append("Last report: " + report["operation"]
                + " · " + report["run"] + " · " + report["step"]
                + (" · " + report["message"] if report.get("message") else "")
                + " · " + timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
        return "\n" + "\n".join(lines)

    def _profile_text(self, data):
        """The selected job's stored storage profile; never editable here."""
        if not data:
            return "Storage profile: no job selected."
        facts = data.get("job_facts")
        if not facts:
            return "Storage profile: not shown, the job metadata could not be read."
        if self._is_setup_job(data):
            return "Setup outputs use the limits recorded in each operation. No image-frame storage profile applies."
        if self._is_comparison_job(data):
            return "Original preset Comparison outputs · CPU · original input and solver types. No image-frame storage profile applies."
        if facts.get("pipeline") == "native-phase-statistics" and "storage_profile" not in facts:
            return "Storage profile: not applicable; phase map generation writes no mass-map products."
        try:
            profile = job_storage_profile(facts)
        except ValueError as error:
            return "Storage profile: invalid job metadata: " + str(error)
        source = (
            "stored in job.json"
            if "storage_profile" in facts
            else "legacy job.json without the field; old jobs stay full"
        )
        text = f"Storage profile: {profile} — {source}. Fixed for an existing job (read-only)."
        estimate = self._creation_estimates.get(str(data.get("root")))
        if estimate and estimate.get("complete"):
            text += (
                f"\nEstimate at creation: {_size(estimate['job_bytes'])} for "
                f"{estimate['frames']} frames, plus {_size(estimate['inflight_bytes'])} in flight. "
                f"{ESTIMATE_LABEL}."
            )
        else:
            text += "\nCheck expected space on this machine for this job's estimate against fresh free space."
        return text

    @staticmethod
    def _is_setup_job(data):
        """Use only the typed pipeline identity to select operation presentation."""
        return ((data or {}).get("job_facts") or {}).get("pipeline") == "series-setup"

    @staticmethod
    def _is_comparison_job(data):
        return ((data or {}).get("job_facts") or {}).get("pipeline") == "preset-wasserstein"

    @staticmethod
    def _comparison_result_evidence(data):
        """Keep supplied scientific result status separate from queue progress."""
        rows = []
        if "nonfinite_distances" in data:
            rows.append("Nonfinite distances: " + str(data["nonfinite_distances"]))
        if "result_statuses" in data:
            rows.append("Result statuses: " + json.dumps(data["result_statuses"], ensure_ascii=False, sort_keys=True))
        if data.get("result_warnings"):
            rows.append("Result warnings: " + str(data["result_warnings"]))
        return "\n" + "\n".join(rows) if rows else ""

    def set_resource_panel(self, panel):
        if self.resource_panel is not None:
            raise ValueError("Resource panel already connected")
        self.resource_panel = panel
        self.resource_slot.addWidget(panel)
        self.detail_tabs.addTab(self.resource_page, "Resources")
        panel.workers_selected.connect(self._resource_choice)
        self._show_snapshot(self.current)

    def _resource_choice(self, choice):
        try:
            chosen = name(choice["machine"])
        except (ValueError, KeyError):
            return
        matches = [
            i
            for i in range(self.target_combo.count())
            if name((self.target_combo.itemData(i) or {})["machine"]) == chosen
        ]
        if len(matches) == 1:
            self.target_combo.setCurrentIndex(matches[0])
            self.workers_spin.setValue(choice["workers"])
            self.backend_combo.setCurrentIndex(
                max(0, self.backend_combo.findData(choice["backend"]))
            )
            feedback = (
                f"Applied {self.workers_spin.value()} worker(s) using "
                f"{self.backend_combo.currentText()} to the job controls. "
                "Start selected machine starts the workers."
            )
        elif not matches:
            feedback = (
                "Worker count not applied: this machine has no registered launch target "
                "for this job. Choose or add a machine in Jobs. No worker was started."
            )
        else:
            feedback = (
                "Worker count not applied: this machine has multiple registered launch "
                "targets. Choose the exact target in Jobs. No worker was started."
            )
        self.progress_label.setText(feedback)
        selection_label = getattr(self.resource_panel, "selection_label", None)
        if selection_label is not None:
            selection_label.setText(feedback)
        self.workers_selected.emit(deepcopy(choice))

    def attach_new_analysis(self, page):
        """Let the app's New analysis page replace the Start analysis dialog (N141)."""
        if self.new_analysis is not None:
            raise ValueError("New analysis page already attached")
        self.new_analysis = page

    def _new_analysis_inputs(self):
        """The plans and known free space offered for a new job, or None if blocked."""
        if not self.selection_provider or not self.series_root or not self.series_id:
            self._error(
                "Create is blocked until the current-series B4 image selection is connected and exactly matches the saved execution plan."
            )
            return None
        try:
            plans = self.plans_provider() if self.plans_provider else []
        except Exception:
            self._error(traceback.format_exc())
            return None
        known = self._known_free_space
        return plans, (known if known and known["series_id"] == self.series_id else None)

    def prepare_new_analysis(self):
        """Offer the current plans and known free space on the New analysis page."""
        inputs = self._new_analysis_inputs()
        plans, free_space = inputs if inputs is not None else ([], None)
        self.new_analysis.set_plans(plans, free_space=free_space)

    def place_target_controls(self, page=None):
        """Lend the machine, backend and workers choice and the resource advisor.

        With a page they move to its Where section; with None they return to
        the job toolbar and the Resources tab. The same objects and signals
        move; nothing is copied.
        """
        if page is self._target_borrower:
            return
        source = self._controls if self._target_borrower is None else self._target_borrower.target_layout
        for index, widget in enumerate((self.target_combo, self.backend_combo, self.workers_spin)):
            hidden = widget.isHidden()
            source.removeWidget(widget)
            if page is None:
                self._controls.insertWidget(index, widget)
            else:
                page.target_layout.addWidget(widget)
            widget.setHidden(hidden)
        if self.resource_panel is not None:
            hidden = self.resource_panel.isHidden()
            if page is None:
                self._target_borrower.resource_layout.removeWidget(self.resource_panel)
                self.resource_slot.addWidget(self.resource_panel)
            else:
                self.resource_slot.removeWidget(self.resource_panel)
                page.resource_layout.addWidget(self.resource_panel)
            self.resource_panel.setHidden(hidden)
        self._target_borrower = page

    def _sync_new_analysis(self):
        """Let the New analysis page follow Create and Start availability."""
        if self.new_analysis is not None:
            self.new_analysis.sync_actions()

    def open_create_dialog(self):
        if self.new_analysis is not None:
            # The app shows its New analysis page instead of the dialog.
            self.new_analysis_requested.emit()
            return
        inputs = self._new_analysis_inputs()
        if inputs is None:
            return
        plans, free_space = inputs
        dialog = StartAnalysisDialog(plans, self, free_space=free_space)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.create_job(**dialog.choices())

    def create_job(self, plan, job_root, display_name, storage_profile=None):
        if self._create_thread is not None:
            return
        self.create_button.setEnabled(False)
        self._sync_new_analysis()
        kwargs = {
            "display_name": display_name,
            "series_label": self.series_label,
            "series_root": self.series_root,
            "series_id": self.series_id,
            "selection_provider": self.selection_provider,
        }
        if storage_profile is not None:
            kwargs["storage_profile"] = storage_profile
        if self.job_creator is create_analysis_job and self.local_stores_path_provider is not None:
            kwargs["stores_path"] = self.local_stores_path_provider()
        generation = self._generation
        self._create_thread, self._create_worker = self._start_call(
            lambda: {
                "generation": generation,
                "registration": self.job_creator(plan, job_root, **kwargs),
            },
            self._created,
            self._create_closed,
        )

    @Slot(object)
    def _created(self, result):
        registration = result["registration"]
        receipt = (registration or {}).get("producer_receipt")
        if isinstance(receipt, dict) and isinstance(receipt.get("storage_estimate"), dict):
            self._creation_estimates[str(registration.get("root"))] = deepcopy(
                receipt["storage_estimate"]
            )
        self.job_created.emit(deepcopy(registration))
        if result["generation"] != self._generation:
            return
        self._generation += 1
        self.current = None
        for row in self.registrations:
            row["active"] = False
        self.registrations.append(deepcopy(registration))
        self.refresh()

    @Slot()
    def _create_closed(self):
        if self._create_thread is not None:
            self._create_thread.wait()
        self._create_thread = self._create_worker = None
        self.create_button.setEnabled(
            bool(self.selection_provider and self.series_root and self.series_id)
        )
        self._sync_new_analysis()

    def start_local_workers(self):
        if not self.current or not self.current.get("job_facts"):
            return
        if self.current.get("errors"):
            self._error(
                "Worker start unavailable: selected job has metadata errors.\n"
                + "\n".join(self.current["errors"])
            )
            self._target_changed()
            return
        if not self._start_allowed(self.local_machine):
            return
        request = None
        launch_context = dict(
            job_root=self.current["root"], series_root=self.series_root,
            pipeline=self.current["job_facts"].get("pipeline"),
            required_runs=deepcopy(self.current["job_facts"].get("expected_runs", [])),
            backend=self.backend_combo.currentData(),
        )
        try:
            local_options = {}
            if (((self.current.get("job_facts") or {}).get("format")
                    == "darsia-alias-job-v1" or
                    (self.current.get("job_facts") or {}).get("execution_mode") == "current_build")
                    and self.local_stores_path_provider is not None):
                selected_stores = self.local_stores_path_provider()
                if not isinstance(selected_stores, (str, Path)) or not str(selected_stores).strip():
                    raise ValueError("Choose an existing local store map before Start")
                local_options["stores"] = selected_stores
            request = self._begin_request(self.local_machine, "local")
            record = self.local_launcher(
                self.current["root"],
                backend=self.backend_combo.currentData(),
                workers=self.workers_spin.value(),
                machine=self.local_machine,
                launch_directory=request["launch_directory"],
                **local_options,
            )
        except Exception:
            self._launch_failed(request, traceback.format_exc())
            return
        self._launch_records.append(deepcopy(record))
        self.local_worker_launch_requested.emit(deepcopy(record))
        if (record.get("started") is True
                and launch_context["pipeline"] == "native-phase-statistics"
                and launch_context["backend"] == "cpu"):
            self.phase_one_launch_accepted.emit(launch_context)
        self.progress_label.setText(
            "Local worker launch requested. Only subsequent published comparison result records count as progress. " + COMPARISON_LAUNCH_SCOPE
            if self._is_comparison_job(self.current) else
            "Local worker launch requested. Only subsequent verified operation completions count as progress."
            if self._is_setup_job(self.current) else
            "Local worker launch requested. Only subsequent actual frame publications count as progress."
        )
        self.refresh()

    def prepare_build_p1(self):
        """Use the selected completed Build and the active series, without a picker.

        The job covers the chosen photographs (calibration images or the
        production image grid), and other connected machines may claim its
        runs only when that box is checked.
        """
        if self._create_thread is not None or not self.current or not self.current.get("complete"):
            return
        from .pipeline.current_build_job import prepare_current_build_job
        snapshot = deepcopy(self.current)
        generation = self._generation
        stores = self.local_stores_path_provider() if self.local_stores_path_provider else None
        purpose, shared = self.phase_map_purpose.currentData(), self.phase_map_shared.isChecked()
        series_root, series_id, machine = self.series_root, self.series_id, self.local_machine
        progress = self.phase_map_progress.emit
        self.prepare_build_p1_button.setEnabled(False)
        self.progress_label.setText("Preparing production phase maps from the completed Setup…"
                                    if purpose == "production" else
                                    "Preparing phase map generation from the completed Setup…")

        def prepare():
            try:
                return dict(generation=generation, registration=prepare_current_build_job(snapshot,
                    series_root=series_root, series_id=series_id, stores_path=stores, machine=machine,
                    purpose=purpose, on_progress=progress, any_connected_machine=shared))
            except ValueError as error:
                return dict(generation=generation, refused=str(error))

        self._create_thread, self._create_worker = self._start_call(
            prepare, self._phase_map_prepared, self._create_closed)

    @Slot(object)
    def _phase_map_prepared(self, result):
        """Register the prepared job, or show in plain words why it was refused."""
        if "refused" in result:
            self.progress_label.setText("Phase map job not prepared: " + result["refused"])
            self._error(result["refused"])
            return
        self._created(result)

    @Slot(object)
    def _phase_map_progress(self, value):
        """Count the production photographs whose identity has been read."""
        if self._create_thread is None or not isinstance(value, dict) or value.get("state") != "raw_identity":
            return
        self.progress_label.setText("Preparing production phase maps: reading photograph identities, "
                                    f"{value.get('completed', '?')} of {value.get('total', '?')}…")

    def _job_requests(self):
        if not self.current:
            return []
        rows = {
            row["source"]["path"]: row
            for row in self._session_requests
            if launch_key(row["job_root"], row["machine"], row["backend"])[0]
            == launch_key(self.current["root"], self.local_machine, "cpu")[0]
        }
        for row in self.current.get("launch_requests", []):
            previous = rows.get(row["source"]["path"], {})
            rows[row["source"]["path"]] = dict(
                row,
                error_traceback=row.get("error_traceback")
                or previous.get("error_traceback", ""),
            )
        return sorted(
            rows.values(), key=lambda row: (row["requested_utc"], row["source"]["path"])
        )

    def _pending_requests(self, machine):
        if not self.current:
            return []
        key = launch_key(
            self.current["root"], machine, self.backend_combo.currentData()
        )[1:]
        rows = {row["source"]["path"]: row for row in self._session_requests}
        for snapshot in self.snapshots:
            for row in snapshot.get("launch_requests", []):
                rows[row["source"]["path"]] = row
        # These are freshly bound reads from the optional all-series provider.
        for row in self._known_launch_requests:
            rows[row["source"]["path"]] = row
        for row in self._job_requests():
            if row["source"]["path"] not in rows:
                rows[row["source"]["path"]] = row
        return sorted(
            (
                row
                for row in rows.values()
                if not row["released"]
                and launch_key(row["job_root"], row["machine"], row["backend"])[1:]
                == key
            ),
            key=lambda row: (row["requested_utc"], row["source"]["path"]),
        )

    def _read_known_launches(self):
        if self.known_jobs_provider is None:
            return [], []
        try:
            return read_registered_launch_requests(list(self.known_jobs_provider()))
        except Exception:
            return [], [
                dict(
                    root="Local registered jobs", error_traceback=traceback.format_exc()
                )
            ]

    def _start_allowed(self, machine):
        if (self.current or {}).get("archived"):
            self._error("Restore the archived job before starting its workers")
            return False
        facts = (self.current or {}).get("job_facts") or {}
        if facts.get("execution_mode") == "current_build" and (
                self.backend_combo.currentData() != "cpu"
                or name(machine) != name(facts["target_machine"])):
            self._error("Phase map generation uses CPU workers on the machine that completed Build.")
            return False
        if self._is_comparison_job(self.current) and self.backend_combo.currentData() != "cpu":
            self._error("This original Comparison job requires CPU. Choose CPU before starting workers.")
            return False
        if self._is_comparison_job(self.current) and self.workers_spin.value() != 1:
            self._error("This original Comparison job requires exactly one worker per service. " + COMPARISON_LAUNCH_SCOPE)
            return False
        if (facts.get("format") == "darsia-alias-job-v1"
                and (facts.get("execution_ready") is not True
                     or facts.get("runtime_support") != "darsia-role-io-runtime-v1")):
            self._error(facts.get("execution_blocker") or "J3 alias-job runtime is not available")
            return False
        if self._control_pending:
            self._error(
                "A control command is still being published; wait for its result."
            )
            return False
        try:
            control = self._read_control_snapshot(self.current["root"])
            if control.get("error"):
                raise ValueError(control["reason"])
            if control["state"]["action"] != "run":
                raise ValueError(
                    "The job requests pause or stop. Use Resume on its recorded "
                    "master machine before starting a service."
                )
        except Exception:
            self._error(traceback.format_exc())
            return False
        # Explicit Start revalidates only known roots, never searches the NAS.
        self._known_launch_requests, self._known_launch_errors = (
            self._read_known_launches()
        )
        if self._pending_requests(machine) or self._known_launch_errors:
            self._target_changed()
            return False
        return True

    def _begin_request(self, machine, kind):
        from .job_management import JobActionLock
        with JobActionLock(self.current["root"]):
            request = begin_launch_request(
                self.current["root"], machine=machine,
                backend=self.backend_combo.currentData(), workers=self.workers_spin.value(), kind=kind)
        self._session_requests.append(request)
        self._target_changed()
        return request

    def _launch_failed(self, request, text):
        if request is not None:
            try:
                record_launch_error(request, text)
            except Exception:
                text += (
                    "\nThe UI failure receipt could not be written:\n"
                    + traceback.format_exc()
                )
            request["error_traceback"] = text
        self._error(text)
        self._target_changed()
        self.refresh()

    def declare_previous_service_stopped(self):
        selected = self.target_combo.currentData() or {}
        pending = self._pending_requests(selected.get("machine", self.local_machine))
        if not pending:
            return
        local_setup = bool(self.current and self._is_setup_job(self.current)
            and all(row["job_root"] == self.current["root"] and row.get("kind") == "local"
                    and name(row.get("machine")) == name(self.local_machine) for row in pending))
        reply = QMessageBox.question(
            self,
            "Verify stopped local processes" if local_setup else "Operator declaration only",
            ("Verify the recorded local worker and supervisor processes have stopped?\n"
             "This checks their original identities and preserves completed Apply and all saved results.\nRegistered jobs:\n"
             if local_setup else "Read the launch consoles first. Do you declare that the previous services for this machine and backend have stopped?\nRegistered jobs:\n")
            + "\n".join(sorted({row["job_root"] for row in pending}))
            + ("\nOnly a successful process check permits updating this job; no scientific completion is recorded."
               if local_setup else "\nThis records your statement only; it does not verify worker retirement, completed results or numerical progress."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if local_setup:
            if self._control_pending:
                self._refuse("A control publication is still finishing.")
                return
            root, generation = self.current["root"], self._generation
            stores = self.local_stores_path_provider() if self.local_stores_path_provider else None
            self._control_pending = True
            self._cancel_metadata_refresh()
            self._target_changed()
            def verify():
                from .job_local_stop import confirm_local_stop
                pin = confirm_local_stop(root, machine=self.local_machine,
                    actor=self.local_machine, stores_path=stores)
                return dict(root=root, generation=generation,
                    control=self._read_control_snapshot(root), resume=None, confirmation=pin)
            def verified(result):
                self._control_ready(result)
                self.operator_declaration_recorded.emit([result["confirmation"]])
                self.control_note.setText("Original local processes verified stopped. Completed work is retained; Update job can now recheck the saved inputs.")
            self._control_thread, self._control_worker = self._start_call(
                verify, verified, self._control_closed)
            return
        try:
            written = []
            roots = {row["job_root"] for row in pending}
            for root in sorted(roots):
                written.extend(
                    declare_requests_stopped(
                        root, [row for row in pending if row["job_root"] == root]
                    )
                )
                updated = read_launch_requests(root)
                self._session_requests = [
                    row
                    for row in self._session_requests
                    if launch_key(row["job_root"], row["machine"], row["backend"])[0]
                    != launch_key(root, self.local_machine, "cpu")[0]
                ] + updated
                for snapshot in self.snapshots:
                    if (
                        launch_key(snapshot["root"], self.local_machine, "cpu")[0]
                        == launch_key(root, self.local_machine, "cpu")[0]
                    ):
                        snapshot["launch_requests"] = updated
            self._known_launch_requests, self._known_launch_errors = (
                self._read_known_launches()
            )
        except Exception:
            self._error(traceback.format_exc())
            return
        self.operator_declaration_recorded.emit(written)
        self._target_changed()
        self.refresh()

    def _refresh_targets(self):
        previous = (self.target_combo.currentData() or {}).get("key")
        rows = [dict(key="local", machine=self.local_machine, local=True)]
        for record in (self.current or {}).get("registered_machines", []):
            if name(record["machine"]) == name(self.local_machine):
                continue
            rows.append(dict(record, key=record["source"]["path"], local=False))
        with QSignalBlocker(self.target_combo):
            self.target_combo.clear()
            for row in rows:
                label = row["machine"] + (
                    " (this machine)"
                    if row["local"]
                    else " · "
                    + str(row.get("registered_utc") or "saved facts")
                    + " · "
                    + row["source"]["sha256"][:8]
                )
                self.target_combo.addItem(label, row)
            index = next((i for i, row in enumerate(rows) if row["key"] == previous), 0)
            self.target_combo.setCurrentIndex(index)
        self._target_changed()

    def _target(self, record):
        return RemoteWorkerTarget(
            **{
                key: record.get(key, "")
                for key in (
                    "machine",
                    "ssh_alias",
                    "python",
                    "checkout",
                    "ssh_executable",
                    "default_shell",
                )
            }
        )

    def _target_changed(self, *_):
        self._show_execution_target()
        selected = self.target_combo.currentData() or {}
        job = self.current or {}
        enabled = bool(job.get("job_facts")) and not job.get("errors")
        reason = ""
        if job.get("archived"):
            enabled = False
            reason = "Restore the archived job before starting its workers"
        if not job.get("job_facts"):
            reason = "Choose a job with readable execution metadata before starting workers. See Errors for details."
        if job.get("errors"):
            detail = next((line.strip() for line in reversed(str(job["errors"][0]).splitlines())
                           if line.strip()), "")
            if len(detail) > 240:
                detail = detail[:237] + "..."
            reason = "Worker start unavailable: selected job has metadata errors."
            if detail:
                reason += " " + detail
            reason += " See Errors for details."
        if ((job.get("job_facts") or {}).get("format") == "darsia-alias-job-v1"
                and (job["job_facts"].get("execution_ready") is not True
                     or job["job_facts"].get("runtime_support") != "darsia-role-io-runtime-v1")):
            enabled = False
            reason = reason or job["job_facts"].get("execution_blocker") or "J3 alias-job runtime is not available"
        if self._is_comparison_job(job) and self.backend_combo.currentData() != "cpu":
            enabled = False
            reason = "This original Comparison job requires CPU. Choose CPU before starting workers."
        elif self._is_comparison_job(job) and self.workers_spin.value() != 1:
            enabled = False
            reason = "This original Comparison job requires exactly one worker per service. " + COMPARISON_LAUNCH_SCOPE
        if (job.get("job_facts") or {}).get("execution_mode") == "current_build" and (
                self.backend_combo.currentData() != "cpu"
                or not selected.get("local")):
            enabled = False
            reason = "Phase map generation uses CPU workers on the machine that completed Build."
        if enabled and not selected.get("local"):
            try:
                build_remote_worker_launch(
                    job["root"],
                    target=self._target(selected),
                    backend=self.backend_combo.currentData(),
                    workers=self.workers_spin.value(),
                    pipeline=(job.get("job_facts") or {}).get("pipeline"),
                )
            except Exception:
                enabled = False
                reason = traceback.format_exc()
        pending = self._pending_requests(selected.get("machine", self.local_machine))
        self.declare_stopped_button.setEnabled(bool(pending))
        if pending:
            enabled = False
            reason = (
                UNKNOWN
                + ". Read the saved launch receipt/console; a client exit is not worker retirement."
            )
            self.launch_state_label.setText(
                reason
                + "\n"
                + "\n".join(
                    "Registered job: "
                    + row["job_root"]
                    + "\nConsole: "
                    + row["console"]
                    for row in pending
                )
            )
        elif self._known_launch_errors:
            enabled = False
            reason = "Launch state unavailable for a registered job; Start remains blocked until its own metadata can be read."
            self.launch_state_label.setText(
                reason
                + "\n"
                + "\n".join(
                    row["root"] + "\n" + row["error_traceback"]
                    for row in self._known_launch_errors
                )
            )
        else:
            rows = [
                row
                for row in self._job_requests()
                if launch_key(row["job_root"], row["machine"], row["backend"])
                == launch_key(
                    job.get("root", ""),
                    selected.get("machine", self.local_machine),
                    self.backend_combo.currentData(),
                )
            ]
            self.launch_state_label.setText(
                rows[-1]["state"]
                if rows
                else "No launch request for the selected target."
            )
        control = job.get("control") or {}
        if self._control_pending or control.get("error") or (
            control.get("state", {}).get("action", "run") != "run"
        ):
            enabled = False
            reason = reason or (
                "The job is paused/stopped or its control metadata is unavailable. "
                "Resume requires its recorded master machine."
            )
        self.start_button.setEnabled(enabled)
        self.start_reason_label.setText(reason)
        self.start_reason_label.setVisible(bool(reason))
        self.start_button.setToolTip(
            reason
            or (COMPARISON_LAUNCH_SCOPE + " Published comparison result records alone show queue progress."
                if self._is_comparison_job(job) else "")
            or ("Explicit start of the existing whole-run worker; verified operation completions alone show progress."
                if self._is_setup_job(job) else
                "Explicit start of the existing whole-run worker; frame publications alone show progress.")
        )
        boundary = (COMPARISON_CONTROL_BOUNDARY if self._is_comparison_job(job) else
                    SETUP_CONTROL_BOUNDARY if self._is_setup_job(job) else CONTROL_BOUNDARY)
        self.control_note.setText(boundary + " " + (reason or REMOTE_GAP))
        self._show_control_state()
        self._sync_new_analysis()

    def _show_execution_target(self, *_):
        from .jobs_execution_display import selected_target_text

        # Presentation only. The existing admission and launch paths remain
        # authoritative when the user explicitly starts the selected machine.
        if hasattr(self, "target_label"):
            self.target_label.setText(selected_target_text(
                self.target_combo.currentData(), self.workers_spin.value(),
                self.backend_combo.currentData(),
            ))

    def _read_control_snapshot(self, root):
        """Read commands and the bound master; never initialize a UI namespace."""
        try:
            state = self.control_reader(root)
            path = Path(root) / "ui" / "master.json"
            if not path.is_file():
                return dict(
                    available=False,
                    state=state,
                    reason="No recorded master is present for this job; frame controls are unavailable.",
                )
            asset = descriptor(path)
            master = JobSnapshotReader().read(path, asset)
            from .job_control import _bindings
            reference, master_format, _control_format = _bindings(Path(root))
            if (master.get("format") != master_format
                    or master.get("job") != reference(Path(root) / "job.json")):
                raise ValueError("Job UI master is not bound to this immutable job")
            machine = master.get("machine")
            if not isinstance(machine, str) or not machine.strip():
                raise ValueError("Job UI master machine is missing")
            return dict(
                available=True, state=state, master=asset, master_machine=machine
            )
        except Exception:
            return dict(available=False, error=True, reason=traceback.format_exc())

    def _control_reason(self, action):
        job = self.current or {}
        control = job.get("control") or {}
        if not job.get("job_facts") or job.get("errors"):
            return "Control unavailable: selected job has no valid execution metadata."
        if not control.get("available"):
            return control.get("reason") or (
                "Refresh the job's control metadata before applying a command."
            )
        if control.get("master_machine") != self.local_machine:
            return "Only the job's recorded master machine writes commands: " + str(
                control.get("master_machine")
            )
        if self._control_pending:
            return "A control publication is still finishing."
        if job.get("complete"):
            return "The job is complete; no remaining run needs control."
        current = control["state"]["action"]
        if action == "run" and current not in {"pause", "stop"}:
            return "Resume requires a displayed pause or stop command."
        if action == current:
            return "This requested state is already displayed."
        return ""

    def _show_control_state(self):
        setup = self._is_setup_job(self.current)
        comparison = self._is_comparison_job(self.current)
        boundary = COMPARISON_CONTROL_BOUNDARY if comparison else SETUP_CONTROL_BOUNDARY if setup else CONTROL_BOUNDARY
        self.pause_button.setText("Pause job after current comparison result" if comparison else
                                 "Pause job at next operation checkpoint" if setup else "Pause job after current frame")
        self.stop_button.setText("Cancel job")
        control = (self.current or {}).get("control") or {}
        if not control.get("available"):
            self.control_state_label.setText(
                "Control unavailable: "
                + str(control.get("reason") or "no job selected.")
            )
        else:
            state = control["state"]
            command = state.get("command")
            text = (
                "Requested state: "
                + ("Pending (waiting for explicit Start or Restart)" if (self.current or {}).get("pending_reset")
                   else {"run": "Ready to run", "pause": "Pause", "stop": "Cancel"}[state["action"]])
                + " · command sequence "
                + str(state["sequence"])
            )
            text += (
                "\nRecorded master: "
                + control["master_machine"]
                + " · This machine: "
                + self.local_machine
            )
            if command:
                text += (
                    "\nDisplayed command: "
                    + str(command.get("relative_path", command.get("path")))
                    + " · SHA-256 "
                    + command["sha256"]
                    + " · "
                    + str(command.get("bytes", command.get("size_bytes")))
                    + " bytes"
                )
            else:
                text += "\nNo command has been recorded (sequence 0)."
            if (
                state["action"] == "run"
                and state["sequence"]
                and not (self.current or {}).get("complete")
            ):
                text += (
                    "\nIf a previous claim remains: "
                    + CONTROL_LEASE
                    + ". Normal services retain the original 600 s actual-progress lease. "
                    "Restart unfinished work can retry immediately after verified original retirement."
                )
            elif state["action"] in {"pause", "stop"}:
                text += "\nThe command is a request; worker exit is not inferred from it."
            self.control_state_label.setText(text)
        for action, button in (
            ("pause", self.pause_button),
            ("run", self.resume_button),
            ("stop", self.stop_button),
        ):
            reason = self._control_reason(action)
            button.setEnabled(not reason)
            button.setToolTip(
                reason or (CONTROL_LEASE if action == "run" else boundary)
            )
        job = self.current or {}
        base_reason = self._control_reason("run")
        # Recovery also supports a failed startup whose control is already run.
        if control.get("available") and control["state"]["action"] == "run":
            base_reason = self._control_reason("stop")
        ready = not base_reason and bool(job.get("recovery_available", True))
        self.restart_button.setEnabled(ready)
        self.reset_button.setEnabled(ready and bool(job.get("reset_available", True)))
        self.restart_button.setToolTip(base_reason or job.get("status_detail") or
            "Restart this job's pending work after original retirement. Keep completed products and verified checkpoints.")
        self.reset_button.setToolTip(base_reason or job.get("status_detail") or
            "Return an unstarted or failed-startup job to Pending. Saved work cannot be erased; use Restart unfinished work to retain it. Full rewind requires a new prepared job.")

    def pause_job(self):
        self._request_control("pause")

    def stop_job(self):
        self._request_control("stop")

    def resume_job(self):
        self._request_recovery(reset=False)

    def _request_recovery(self, *, reset):
        job = self.current or {}
        control = job.get("control") or {}
        reason = self._control_reason("stop" if control.get("state", {}).get("action") == "run" else "run")
        if reason:
            self._refuse(reason)
            return
        root, generation = job["root"], self._generation
        displayed = deepcopy(control)
        stores_path = self.local_stores_path_provider() if self.local_stores_path_provider else None
        resume = None if reset else dict(root=root, generation=generation,
            target=deepcopy(self.target_combo.currentData()), backend=self.backend_combo.currentData(),
            workers=self.workers_spin.value())
        self._resume_after_control = None
        self._control_pending = True
        # Like pause and stop, an explicit command retires only the passive
        # display read. The 10 s refresh starts the next read as soon as a slow
        # one returns, so waiting for an idle moment can refuse Resume and
        # Restart indefinitely; recover_job re-checks control and retirement.
        self._cancel_metadata_refresh()
        self._target_changed()

        def recover():
            from .job_lifecycle import recover_job
            asset = recover_job(root, machine=self.local_machine,
                expected_previous=displayed["state"].get("command"), reset=reset,
                stores_path=stores_path,
                publisher=self.control_publisher)
            current = self._read_control_snapshot(root)
            if not current.get("available") or current["state"].get("command") != asset:
                raise ValueError("Job control changed after recovery; refresh before starting")
            return dict(root=root, generation=generation, control=current, resume=resume)

        self._control_thread, self._control_worker = self._start_call(
            recover, self._control_ready, self._control_closed)

    def _request_control(self, action):
        reason = self._control_reason(action)
        if reason:
            self._refuse(reason)
            return
        root, generation = self.current["root"], self._generation
        displayed = deepcopy(self.current["control"])
        resume = (
            dict(
                root=root,
                generation=generation,
                target=deepcopy(self.target_combo.currentData()),
                backend=self.backend_combo.currentData(),
                workers=self.workers_spin.value(),
            )
            if action == "run"
            else None
        )
        self._resume_after_control = None
        self._control_pending = True
        self._cancel_metadata_refresh()
        self._target_changed()

        def publish():
            asset = self.control_publisher(
                root,
                action,
                machine=self.local_machine,
                expected_previous=displayed["state"].get("command"),
            )
            control = self._read_control_snapshot(root)
            if not control.get("available") or control["state"].get("command") != asset:
                raise ValueError(
                    "Job control changed after publication; refresh before starting a service."
                )
            return dict(root=root, generation=generation, control=control, resume=resume)

        self._control_thread, self._control_worker = self._start_call(
            publish, self._control_ready, self._control_closed
        )

    @Slot(object)
    def _control_ready(self, result):
        if (
            result["generation"] != self._generation
            or not self.current
            or self.current["root"] != result["root"]
        ):
            return
        self.current["control"] = result["control"]
        self._resume_after_control = result["resume"]
        self._target_changed()

    @Slot()
    def _control_closed(self):
        if self._control_thread is not None:
            self._control_thread.wait()
        self._control_thread = self._control_worker = None
        self._control_pending = False
        resume, self._resume_after_control = self._resume_after_control, None
        self._target_changed()
        if (
            resume is not None
            and resume["generation"] == self._generation
            and self.current
            and resume["root"] == self.current["root"]
        ):
            if (
                resume["target"] == self.target_combo.currentData()
                and resume["backend"] == self.backend_combo.currentData()
                and resume["workers"] == self.workers_spin.value()
            ):
                self.start_selected_machine()
            else:
                self._error(
                    "Run was requested, but the selected launch target changed; choose Start explicitly."
                )
        self._refresh_pending = False
        self.refresh()

    def start_selected_machine(self):
        selected = self.target_combo.currentData() or {}
        if self.current and self.current.get("errors"):
            self._error(
                "Worker start unavailable: selected job has metadata errors.\n"
                + "\n".join(self.current["errors"])
            )
            self._target_changed()
            return
        if selected.get("local"):
            return self.start_local_workers()
        if not self.current or not self.current.get("job_facts"):
            return
        if not self._start_allowed(selected.get("machine", self.local_machine)):
            return
        request = None
        try:
            bound = selected["source"]
            actual = JobSnapshotReader().read(Path(bound["path"]), bound)
            if actual.get("job") != descriptor(Path(self.current["root"]) / "job.json"):
                raise ValueError("Remote registration belongs to another job")
            target = self._target(actual)
            build_remote_worker_launch(
                self.current["root"],
                target=target,
                backend=self.backend_combo.currentData(),
                workers=self.workers_spin.value(),
                pipeline=(self.current.get("job_facts") or {}).get("pipeline"),
            )
            request = self._begin_request(actual["machine"], "ssh")
            record = self.remote_launcher(
                self.current["root"],
                target=target,
                backend=self.backend_combo.currentData(),
                workers=self.workers_spin.value(),
                launch_directory=request["launch_directory"],
            )
        except Exception:
            self._launch_failed(request, traceback.format_exc())
            return
        self._launch_records.append(deepcopy(record))
        self.remote_worker_launch_requested.emit(deepcopy(record))
        self.progress_label.setText(
            "SSH transport launch requested. Worker startup and comparison results are not inferred from transport startup. " + COMPARISON_LAUNCH_SCOPE
            if self._is_comparison_job(self.current) else
            "SSH transport launch requested. Worker startup and operation progress are not inferred from transport startup."
            if self._is_setup_job(self.current) else
            "SSH transport launch requested. Worker startup and frame progress are not inferred from transport startup."
        )
        self.refresh()

    def open_machine_dialog(self):
        if not self.current:
            return
        dialog = RegisterMachineDialog(self, local_machine=self.local_machine)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            result = register_machine_facts(
                self.current["root"],
                dialog.facts(),
                local_machine=self.local_machine,
                new_version=dialog.new_version.isChecked(),
            )
        except Exception:
            self._error(traceback.format_exc())
            return
        self.machine_registered.emit(result)
        self.control_note.setText(REMOTE_GAP)
        self.refresh()

    def _refuse(self, text):
        """Report why a command cannot run now; it is cleared once the next metadata read completes."""
        self._error(text)
        self._transient_error = text

    def _clear_transient_error(self):
        """Withdraw a refusal from ``_refuse`` after a completed read (``error_cleared``).

        The read itself rewrites the Errors tab; the signal lets other views
        that echoed the refusal, such as the Jobs status line, clear it too.
        """
        text, self._transient_error = getattr(self, "_transient_error", None), None
        if not text:
            return
        if self.error_text.toPlainText() == text:
            self.error_text.clear()
        self.error_cleared.emit(text)

    @Slot(str)
    def _error(self, text):
        source = self.sender()
        self._transient_error = None
        self.error_text.setPlainText(text)
        if self._create_worker is not None and source is self._create_worker:
            self.job_creation_failed.emit(text)
        self.error_reported.emit(text)

    def prepare_close(self):
        """Discard passive reads; keep every write/launch/copy close guard."""
        self._refresh_pending = False
        for kind in ("read", "capacity"):
            preview = getattr(self, f"_{kind}_thread")
            if preview is not None:
                preview.cancel()
                preview.deleteLater()
            setattr(self, f"_{kind}_thread", None)
            setattr(self, f"_{kind}_worker", None)
        self._sync_capacity_action()
        if self.new_analysis is not None:
            cancel = getattr(self.new_analysis, "prepare_close", None)
            if cancel is not None:
                cancel()
        if self.resource_panel is not None:
            cancel = getattr(self.resource_panel, "prepare_close", None)
            if cancel is not None:
                cancel()

    def can_close(self):
        return (
            self._read_thread is None
            and self._create_thread is None
            and self._capacity_thread is None
            and self._control_thread is None
            and self._raw_review_thread is None
            and not self._raw_review_pending
            and not self._control_pending
            and (self.resource_panel is None or self.resource_panel.can_close())
            and (self.new_analysis is None or self.new_analysis.can_close())
            and (self.comparison_preparation is None or self.comparison_preparation.can_close())
            and self.comparison_export.can_close()
        )

    def set_refresh_visibility_owner(self, owner):
        """Follow the visible Jobs page when its controls are reparented."""
        previous = self._refresh_visibility_owner
        if previous is not self:
            previous.removeEventFilter(self)
        self._refresh_visibility_owner = owner
        if owner is not self:
            owner.installEventFilter(self)
        self._sync_refresh_visibility()

    def _sync_refresh_visibility(self):
        if self._refresh_visibility_owner.isVisible():
            self.elapsed_timer.start()
            if not self.timer.isActive():
                self.timer.start()
                self.refresh()
        else:
            self.timer.stop()
            self.elapsed_timer.stop()

    def eventFilter(self, watched, event):
        if (watched is self._refresh_visibility_owner
                and event.type() in (QEvent.Type.Show, QEvent.Type.Hide)):
            self._sync_refresh_visibility()
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_refresh_visibility()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._sync_refresh_visibility()

    def closeEvent(self, event):
        self.prepare_close()
        if not self.can_close():
            self.progress_label.setText(
                "A metadata read/create, hardware inspection, export or RAW review preparation is still finishing; close after it returns."
            )
            event.ignore()
            return
        self.timer.stop()
        self.elapsed_timer.stop()
        super().closeEvent(event)


def _job_reference(root, path):
    from .job_control import _bindings
    return _bindings(root)[0](path)


def _admit_gui_job_root(root):
    from .pipeline.job_queue import read_json
    if read_json(Path(root) / "job.json").get("format") == "darsia-alias-job-v1":
        from .job_filesystem import inspect_shared_folder
        admission = inspect_shared_folder(root)
        if not admission.allowed:
            raise ValueError(admission.reason + ": " + admission.evidence)
    else:
        require_unc(str(root))
