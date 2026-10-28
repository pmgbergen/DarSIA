"""Mass calibration's original refit, with a direct single-computer GUI route.

The local route prepares bound plans and runs the original N1 gain/beta fitter
serially on this computer. Main may hand over its exact phase-one job and run
order for automatic fitting after all required products complete and the
preparation process retires. Input-reading, fitting and grid-audit progress
remain distinct. Existing distributed jobs retain their original read/launch
route; no local job claims a distributed publication.

Roles, rules, measured statistics and baseline come from the original selected
configuration. Explicit overflow inputs enable the separate F target route;
the ordinary route stays unchanged and no calibration is adopted here.
One background call runs at a time; the GUI retains its worker until thread
retirement and protects closing while it is active.
"""

from __future__ import annotations

import json
import math
import socket
import hashlib
from copy import deepcopy
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from PySide6.QtCore import QEvent, QThread, Qt, QTimer, QSettings, QSignalBlocker, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

#: Refit roles (the ``refit_jobs`` keys) and the Calibration review role each takes.
ROLE_KEYS = (
    ("fit", "fit"),
    ("heldout", "held-out"),
    ("descriptive", "descriptive"),
    ("excluded", "excluded"),
)
TASK_COLUMNS = ("Task", "State", "Step", "Parameters", "Stop required")
FINAL_STATES = {"completed", "failed", "stopped"}
MAX_RULES_BYTES = 1024 * 1024
REFRESH_MS = 15000
#: A nonempty route-level refusal remains available to callers which close a route.
START_BLOCKER = ""
ADMITTED_NOTICE = "Create prepares local inputs. Start runs the original mass fit on this computer. When linked to phase map generation, it starts automatically after every required run completes. Results remain unadopted."
NOTICE = ADMITTED_NOTICE
F_WAITS = "Use measured overflow as the free CO₂-gas target"
NO_JOB = "No mass calibration job is open. Create one, or open an existing mass calibration job."
TITRATION_EXPLANATION = (
    "Gain is one common factor that turns the titration colour response into dissolved CO₂ concentration, "
    "using each run's measured recipe. Beta is the exponent of the dilution law. "
    "The anchor is the end-of-injection reference: the observed affected pore volume and its time in each run. "
    "The dilution factor is calculated for each image from beta and that anchor; it is 1 before the anchor "
    "and follows the growth in affected volume afterwards. It is not another fitted parameter."
)


def fit_image_explanation(roles, rules) -> str:
    """Describe the actual selected rule windows; no image selection or fit."""
    runs = ", ".join((roles or {}).get("fit", ())) or "none recorded"
    text = f"Fit runs: {runs}. Control/validation, descriptive and excluded runs do not fit these parameters. "
    if rules is None:
        return text + "Choose the rules to see the gain and beta image windows and the anchor time."
    text += (
        f"Gain uses eligible injection images at {_hours(rules['stage1_window_h'])}, with dilution off. "
        f"Beta uses eligible images from {float(rules['stage2_start_h']):g} h, with that gain fixed. "
        f"The anchor uses the image nearest {float(rules['anchor_target_h']):g} h "
        f"within {float(rules['anchor_tolerance_h']):g} h in each run. "
        "Eligibility also requires the original source, geometry, injected-mass and no-outflow evidence; "
        "the late-image uncertainty rule still applies. "
    )
    if rules.get("target_times_h") is not None:
        text += "Only the target times listed in these rules are eligible. "
    return text + "Leave-one-run-out repeats the fit with one fit run omitted, then predicts that run."


def series_refit_roles(series_root, *, scope="shared") -> tuple[dict[str, list[str]], str]:
    """Read this scope's saved calibration cohort without changing selections."""
    from .shared_calibration_cohort import mass_cohort_roles

    return mass_cohort_roles(series_root, scope=scope)


def baseline_calibration(series_root) -> dict:
    """Active or explicitly retained starting parameters for a new calibration fit.

    Raises ValueError when the series references none, OSError when it cannot be read.
    """
    from darsia.fluidflower.pipeline.job_queue import descriptor
    from darsia.fluidflower.series_method import calibration_ancestry_reference
    import tomllib
    document = tomllib.loads((Path(series_root) / "series.toml").read_text(encoding="utf-8-sig"))
    reference = calibration_ancestry_reference(document.get("method", {}).get("phase_model", {}))
    if not reference:
        raise ValueError("The series references no baseline mass calibration.")
    path = Path(reference)
    if not path.is_absolute():
        path = Path(series_root) / path
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(path, series_roots=(Path(series_root),))
    return descriptor(path)


def _small_json(path: Path, *, limit=MAX_RULES_BYTES) -> object:
    """Bounded UTF-8 JSON; rules retain their original 1 MiB default."""
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"The file exceeds {limit} bytes: {path}")
    return json.loads(raw.decode("utf-8-sig"))


def _hours(values, separator: str = "-") -> str:
    """Hours as short text, such as '0.5-2.375 h' or, with ', ', '8, 48 h'."""
    return separator.join(f"{float(value):g}" for value in values) + " h"


def read_rules(path, *, roles: Mapping, baseline: Mapping) -> tuple[dict, str]:
    """Read a refit rules file and check it against the run roles and the baseline.

    The rules must be ``darsia-n1-refit-rules-v1``, admitted with these roles by
    the refit's own rule reader, and name the baseline calibration's parameters
    as their baseline. Returns the rules and a summary line; raises ValueError,
    KeyError, TypeError or OSError otherwise.
    """
    from darsia.fluidflower.mass.n1_refit import RULES_FORMAT, RefitRules

    from .storage_paths import resolve_storage_path
    rules = _small_json(resolve_storage_path(path))
    if not isinstance(rules, dict) or rules.get("format") != RULES_FORMAT:
        raise ValueError(f"Choose a {RULES_FORMAT} file.")
    checked = RefitRules.from_plan({"rules": rules, "runs": dict(roles)})
    document = _small_json(resolve_storage_path(baseline["path"]))
    parameters = document.get("parameters") if isinstance(document, dict) else None
    if parameters != rules.get("baseline"):
        raise ValueError("The rules' baseline differs from the parameters of the series' mass calibration.")
    parts = [
        f"Stage 1 {_hours(checked.stage1_window_h)}",
        f"stage 2 from {checked.stage2_start_h:g} h",
        f"post-injection bins split at {_hours(checked.post_bin_edges_h, ', ')}",
        f"anchor {checked.anchor_target_h:g} ± {checked.anchor_tolerance_h:g} h",
    ]
    if checked.target_times_h is not None:
        parts.append(f"{len(checked.target_times_h)} target times")
    return rules, " · ".join(parts) + f" · baseline gain {checked.baseline['gain']:g}, beta {checked.baseline['beta']:g}."


def phase_job_readiness(phase_job, fit_runs: Iterable[str]) -> tuple[list[str], list[str]]:
    """The fit runs with a published phase-one result in ``phase_job``, and the others' problems.

    It reads each run's completion record and checks its manifest's SHA-256, as
    the refit's own lookup does; it reads no statistics file.
    """
    from darsia.fluidflower.pipeline.refit_jobs import completed_attempt

    ready, problems = [], []
    for run in fit_runs:
        try:
            completed_attempt(phase_job, run)
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"{run}: {error}")
        else:
            ready.append(run)
    return ready, problems


def phase_job_summary(phase_job, fit_runs: Iterable[str]):
    """The original readiness check plus read-only, explicitly bound recipe metadata.

    Some historical/alias-only manifests do not expose their configuration as
    a local descriptor. Those rows remain unavailable here; job admission is
    still performed by the original pipeline and never by this display.
    """
    from darsia.fluidflower.pipeline.refit_jobs import completed_attempt, _descriptor
    from darsia.fluidflower.pipeline.job_inputs import MAX_METADATA_BYTES

    fit_runs = tuple(fit_runs)
    ready, problems = phase_job_readiness(phase_job, fit_runs)
    recipes = []
    for run in fit_runs:
        source, alkalinity, indicator = "No readable bound recipe metadata", "Not reported", "Not reported"
        if run in ready:
            try:
                attempt = completed_attempt(phase_job, run)
                manifest = _small_json(attempt / "manifest.json")
                bound = manifest["configuration"]
                if not isinstance(bound, dict) or set(bound) != {"path", "sha256", "size_bytes"}:
                    raise ValueError("The manifest does not expose a full local configuration descriptor")
                if _descriptor(bound["path"], max_bytes=MAX_METADATA_BYTES) != bound:
                    raise ValueError("The configuration differs from the completed manifest's binding")
                document = _small_json(Path(bound["path"]), limit=MAX_METADATA_BYTES)
                if (not isinstance(document, dict) or document.get("run") != run
                        or _descriptor(bound["path"], max_bytes=MAX_METADATA_BYTES) != bound
                        or completed_attempt(phase_job, run) != attempt):
                    raise ValueError("The configuration binding or run differs from the completed manifest")
                recipe = document.get("statistics", document)["recipe"]
                values = [recipe[key] for key in ("alkalinity_M", "btb_M")]
                if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError("The recipe does not declare finite measured inputs")
                alkalinity, indicator = map(str, values)
                source = bound["path"]
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
                source += f": {error}"
        recipes.append((str(run), alkalinity, indicator, source))
    return ready, problems, recipes


def default_job_id(now: datetime | None = None) -> str:
    """A new refit job name from the UTC time, such as ``refit-20261021T010203Z``."""
    moment = now or datetime.now(timezone.utc)
    return "refit-" + moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gui_checkout() -> Path:
    """The checkout this GUI runs from; a new job takes its commit (``job_identity``)."""
    import darsia

    return Path(darsia.__file__).resolve().parents[2]


def create_job(series_root, *, roles, rules, baseline, phase_job, jobs_root, job_id, loro=True,
               job_identity=None, repository=None, chemistry_model=None, local=False, overflow=None,
               calibration_scope="shared") -> dict:
    """Create the queued refit job and return its prepared record; it starts nothing.

    ``job_identity`` defaults to ``refit_jobs.job_identity`` of ``repository``
    (this GUI's checkout), with the baseline's SHA-256 as the job's calibration.
    """
    if local:
        from darsia.fluidflower.pipeline.local_refit import create_local_refit_job
        return create_local_refit_job(
            series_root, roles=roles, rules=rules, baseline=baseline, phase_job=phase_job,
            jobs_root=jobs_root, job_id=job_id, loro=loro, job_identity=job_identity,
            repository=repository or gui_checkout(), chemistry_model=chemistry_model,
            **({} if overflow is None else {"overflow": overflow}),
            **({} if calibration_scope == "shared" else {"calibration_scope": calibration_scope}))
    if calibration_scope != "shared":
        raise ValueError("Individual calibrations use the local calibration job route")
    if overflow is not None:
        raise ValueError("Overflow-target calibration currently uses the explicit local route")
    from darsia.fluidflower.pipeline import refit_jobs
    from darsia.fluidflower.pipeline.job_queue import descriptor, name

    name(job_id)
    identity = job_identity or refit_jobs.job_identity
    job = identity(repository or gui_checkout(), calibration_sha256=baseline["sha256"])
    return refit_jobs.create_refit_job(
        descriptor(Path(series_root) / "series.toml"), phase_job=str(phase_job), roles=dict(roles),
        rules=dict(rules), baseline_calibration=dict(baseline), job_id=job_id, jobs_root=str(jobs_root),
        job=job, loro=bool(loro), chemistry_model=chemistry_model)


def read_refit_job(job_root) -> dict:
    """``refit_status`` of a refit job, plus its frozen calibration once published.

    ``frozen`` is (document, descriptor) from ``read_frozen_calibration``;
    ``frozen_error`` says why a published result could not be read.
    """
    from darsia.fluidflower.pipeline.local_refit import FORMAT, read_local_refit_job
    if _small_json(Path(job_root) / "job.json").get("format") == FORMAT:
        return read_local_refit_job(job_root)
    from darsia.fluidflower.pipeline.refit_jobs import read_frozen_calibration, refit_status, refit_start_state

    status = refit_status(job_root)
    frozen, error = None, ""
    if status.get("frozen_calibration"):
        try:
            frozen = read_frozen_calibration(job_root)
        except (OSError, ValueError) as problem:
            error = str(problem)
    return dict(status=status, frozen=frozen, frozen_error=error,
                start=refit_start_state(job_root, status=status))


def parameters_text(parameters) -> str:
    """Fitted parameters as short text, such as 'gain 23.86 · beta 0.6513'; '—' when none."""
    if not isinstance(parameters, Mapping) or not parameters:
        return "—"
    parts = []
    for key in ("gain", "beta"):
        value = parameters.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            parts.append(f"{key} {value:.4g}")
    others = {key: value for key, value in parameters.items() if key not in {"gain", "beta"}}
    parts += [f"{key} {value}" for key, value in sorted(others.items())]
    return " · ".join(parts)


def status_rows(status: Mapping) -> list[tuple[str, str, str, str, str]]:
    """One display row per task of a refit status: task, state, step, parameters, stop."""
    rows = []
    for task, value in (status.get("tasks") or {}).items():
        stop = value.get("stop_required")
        rows.append((
            "Mass calibration" if task == "refit" else str(task),
            str(value.get("state") or "—").capitalize(),
            str(value.get("step") or "—"),
            parameters_text(value.get("parameters")),
            "—" if stop is None else ("Yes" if stop else "No"),
        ))
    return rows


def status_summary(status: Mapping) -> str:
    """One line on a refit job: tasks by state and whether the refit has finished."""
    tasks = status.get("tasks") or {}
    states = [value.get("state") for value in tasks.values()]
    counts = [f"{states.count(state)} {state}" for state in ("completed", "running", "waiting", "failed")
              if state in states]
    text = f"{len(tasks)} tasks: " + ", ".join(counts or ["none"]) + "."
    if status.get("stop_required"):
        text += " Mass calibration stopped at its grid check: no frozen calibration."
    elif status.get("complete"):
        text += " Every task has finished."
    return text


def frozen_text(frozen, error: str = "") -> str:
    """The frozen calibration's path, SHA-256, size and parameters, or why it is not shown."""
    if error:
        return f"The published frozen calibration could not be read: {error}"
    if frozen is None:
        return ""
    document, bound = frozen
    runs = document.get("calibration_runs") if isinstance(document, Mapping) else None
    fitted = f" · fitted on {len(runs)} runs" if isinstance(runs, list) else ""
    chemistry = " · chemistry: Titration" if document.get("chemistry_model") == "titration" else ""
    return (f"Frozen calibration: {bound['path']}\nSHA-256 {bound['sha256']} · {bound['size_bytes']} bytes · "
            f"{parameters_text(document.get('parameters'))}{fitted}{chemistry}.\n"
            "It is not adopted: choosing it for the series stays a separate step.")


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _row(*widgets, below: QWidget | None = None) -> QWidget:
    """Widgets side by side in one form field, the first taking the free width.

    ``below`` goes under them in the same field, so a wrapped summary line
    takes the field's own width and leaves no gap in the form.
    """
    holder = QWidget()
    column = QVBoxLayout(holder)
    column.setContentsMargins(0, 0, 0, 0)
    line = QHBoxLayout()
    for index, widget in enumerate(widgets):
        line.addWidget(widget, 1 if index == 0 else 0)
    column.addLayout(line)
    if below is not None:
        column.addWidget(below)
    return holder


class RefitJobPanel(QGroupBox):
    """Create, start and follow the series' refit job (whole series, no run selector)."""

    local_progress = Signal(object)
    input_choices_changed = Signal()

    def __init__(self, parent=None, *, job_identity=None, machine: str | None = None):
        """An empty panel; :meth:`set_series` gives it a series."""
        super().__init__("Mass calibration job — fit to the injected mass", parent)
        self._identity = job_identity
        self.machine = machine or socket.gethostname()
        self.series_root: Path | None = None
        self._default_jobs_root: str | None = None
        self.roles: dict[str, list[str]] | None = None
        self.baseline: dict | None = None
        self.rules: dict | None = None
        self._restoring_inputs = False
        self._overflow_candidate = None
        self.phase_job = ""
        self.phase_ready = False
        self.job_root: Path | None = None
        self.snapshot: dict | None = None
        self.last_launch: dict | None = None
        self._thread = self._worker = None
        self._local_window = None
        self._read_window = None
        self._local_previous_enabled = True
        self._automatic = None
        self.local_progress.connect(self._local_progress)
        self._phase_wait_timer = QTimer(self)
        self._phase_wait_timer.setSingleShot(True)
        self._phase_wait_timer.setInterval(REFRESH_MS)
        self._phase_wait_timer.timeout.connect(self._poll_phase_one)
        self._refresh_pending = False
        # A result read for another series than the current one is dropped.
        self._generation = self._call_generation = 0
        layout = QVBoxLayout(self)
        self.notice = _label(NOTICE if START_BLOCKER else ADMITTED_NOTICE)
        self.notice.setStyleSheet("color: #b77900;")
        layout.addWidget(self.notice)
        form = QFormLayout()
        self.calibration_scope_combo = QComboBox()
        self.calibration_scope_combo.addItem("Shared calibration — all fit runs", "shared")
        self.calibration_scope_combo.addItem("Individual calibration — one per fit run", "individual")
        self.calibration_scope_combo.setToolTip(
            "Shared fits one gain/beta pair using all fit runs. Individual fits a separate pair for each fit run; "
            "control/validation runs do not enter either fit.")
        form.addRow("Calibration", self.calibration_scope_combo)
        self.chemistry_combo = QComboBox()
        self.chemistry_combo.addItem("Titration", "titration")
        self.chemistry_combo.addItem("Simple flash — Not available yet", "simple_flash")
        self.chemistry_combo.addItem("Both for comparison — Not available yet", "both")
        for index in (1, 2):
            self.chemistry_combo.model().item(index).setEnabled(False)
        self.chemistry_combo.setToolTip(
            "This mass-calibration job uses the selected series' saved Titration model. "
            "Simple flash and comparison of both models are not available yet.")
        self.chemistry_combo.currentIndexChanged.connect(self._update_actions)
        form.addRow("Chemistry model", self.chemistry_combo)
        self.parameter_explanation = _label(TITRATION_EXPLANATION)
        form.addRow("Fitted parameters and dilution", self.parameter_explanation)
        self.measured_inputs = _label(
            "Measured inputs — read only: each run's alkalinity and indicator concentration, "
            "together with its pressure, temperature and geometry, come from the generated phase maps and their calibration statistics. "
            "The recipe is not fitted here; this job reuses the saved titration response.")
        form.addRow("Measured inputs", self.measured_inputs)
        self.recipe_table = QTableWidget(0, 4)
        self.recipe_table.setHorizontalHeaderLabels(("Fit run", "Alkalinity [M]", "Indicator [M]", "Phase-map configuration"))
        self.recipe_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.recipe_table.horizontalHeader().setStretchLastSection(True)
        self.recipe_table.verticalHeader().hide()
        self.recipe_table.setMaximumHeight(150)
        form.addRow("Measured recipes — read only", self.recipe_table)
        self.fit_images = _label(fit_image_explanation(None, None))
        form.addRow("Images used by the fit", self.fit_images)
        self.roles_label = _label("—")
        form.addRow("Run roles", self.roles_label)
        self.baseline_label = _label("—")
        form.addRow("Starting parameters", self.baseline_label)
        self.rules_edit = QLineEdit()
        self.rules_edit.setReadOnly(True)
        self.rules_edit.setPlaceholderText("Mass calibration rules JSON")
        self.rules_button = QPushButton("Choose…")
        self.rules_button.setToolTip("Choose the mass calibration rules file.")
        self.rules_button.clicked.connect(self.choose_rules)
        self.rules_label = _label("Choose the mass calibration rules file.")
        form.addRow("Rules", _row(self.rules_edit, self.rules_button, below=self.rules_label))
        self.phase_edit = QLineEdit()
        self.phase_edit.setReadOnly(True)
        self.phase_edit.setPlaceholderText("Completed phase-map generation job folder")
        self.phase_button = QPushButton("Choose…")
        self.phase_button.setToolTip("Choose the completed phase-map generation job folder for the fit runs.")
        self.phase_button.clicked.connect(self.choose_phase_job)
        self.phase_label = _label("Choose the completed phase-map generation job for the fit runs.")
        form.addRow("Phase-one job", _row(self.phase_edit, self.phase_button, below=self.phase_label))
        # The series names its calibration jobs folder; it is shown, not typed
        # again. Other folder… keeps a deliberate local or other location.
        self.jobs_edit = QLineEdit()
        self.jobs_edit.setReadOnly(True)
        self.jobs_edit.setPlaceholderText("The series' calibration jobs folder")
        self.jobs_edit.setToolTip("New mass calibration jobs are made in the series' calibration jobs folder.")
        self.jobs_edit.textChanged.connect(self._update_actions)
        self.jobs_button = QPushButton("Other folder…")
        self.jobs_button.setToolTip("Make the new mass calibration job in another folder, for example on local disk.")
        self.jobs_button.clicked.connect(self.choose_jobs_folder)
        form.addRow("Jobs folder", _row(self.jobs_edit, self.jobs_button))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name this calibration before creating or starting its job")
        self.name_edit.setToolTip("Use letters, numbers, dots, underscores or hyphens; start with a letter or number.")
        self.name_edit.textChanged.connect(self._update_actions)
        form.addRow("Calibration name", self.name_edit)
        self.loro_check = QCheckBox("Leave-one-run-out fits")
        self.loro_check.setToolTip("One more task per fit run: the fit without that run, and its predictions.")
        self.loro_check.setChecked(True)
        form.addRow("Folds", self.loro_check)
        self.overflow_inputs = None
        self.f_check = QCheckBox(F_WAITS)
        self.f_check.setChecked(False)
        self.f_check.setEnabled(True)
        self.f_check.setToolTip("Use each run's own reviewed measurement directly for its phase split. Missing measurements use explicitly labelled pressure-matched fit-run estimates when comparable data exist; total mass is still fitted to injected mass.")
        self.f_check.toggled.connect(self._overflow_choice_changed)
        form.addRow("Free CO₂-gas target", self.f_check)
        self.overflow_inputs_button = QPushButton("Choose reviewed overflow inputs…")
        self.overflow_inputs_button.clicked.connect(self.choose_overflow_inputs)
        form.addRow("Overflow sources", self.overflow_inputs_button)
        self.source_folder_button = QPushButton("Locate moved source folder…")
        self.source_folder_button.clicked.connect(self._show_source_folder)
        form.addRow(self.source_folder_button)
        self.source_folder_panel = QGroupBox("Moved source folder")
        locations = QFormLayout(self.source_folder_panel)
        self.source_old_folder, self.source_new_folder = QLineEdit(), QLineEdit()
        locations.addRow("Original source folder", self.source_old_folder)
        browse = QPushButton("Choose folder…")
        browse.clicked.connect(self._choose_source_folder)
        locations.addRow("Current source folder", _row(self.source_new_folder, browse))
        self.source_folder_use = QPushButton("Verify and use folder")
        self.source_folder_use.clicked.connect(self._use_source_folder)
        locations.addRow(self.source_folder_use)
        self.source_folder_status = _label(
            "Verifies the selected sources' existing SHA-256 and saves a shared folder alias. No file is moved.")
        locations.addRow(self.source_folder_status)
        self.source_folder_panel.hide()
        form.addRow(self.source_folder_panel)
        layout.addLayout(form)
        first, second = QHBoxLayout(), QHBoxLayout()
        self.create_button = QPushButton("Create mass calibration job")
        self.create_button.setToolTip("Prepares local plans; starts no calculation.")
        self.create_button.clicked.connect(self.create)
        self.open_button = QPushButton("Open mass calibration job…")
        self.open_button.setToolTip("Follow a mass calibration job created earlier.")
        self.open_button.clicked.connect(self.choose_job)
        self.start_button = QPushButton("Start mass calibration")
        self.start_button.clicked.connect(self.start)
        # Why Start is disabled, shown beside it while the queue does not admit refit jobs.
        self.start_reason = QLabel(START_BLOCKER)
        self.start_reason.setStyleSheet("color: #b77900;")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        for row, items in ((first, (self.create_button, self.start_button, self.start_reason)),
                           (second, (self.open_button, self.refresh_button))):
            for widget in items:
                row.addWidget(widget)
            row.addStretch(1)
            layout.addLayout(row)
        self.job_label = _label(NO_JOB)
        layout.addWidget(self.job_label)
        self.tasks = QTableWidget(0, len(TASK_COLUMNS))
        self.tasks.setHorizontalHeaderLabels(TASK_COLUMNS)
        self.tasks.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tasks.verticalHeader().hide()
        header = self.tasks.horizontalHeader()
        for column in range(len(TASK_COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(TASK_COLUMNS.index("Parameters"), QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tasks, 1)
        self.result_label = _label()
        layout.addWidget(self.result_label)
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self.calibration_scope_combo.currentIndexChanged.connect(self._scope_changed)
        self._update_actions()

    # ------------------------------------------------------------ inputs
    def set_overflow_enabled(self, enabled, inputs=None):
        """Choose F explicitly; absent inputs block creation instead of fitting B1."""
        if inputs is not None:
            from darsia.fluidflower.pipeline.local_overflow_refit import read_request
            from darsia.fluidflower.pipeline.job_queue import descriptor
            bound = descriptor(inputs) if isinstance(inputs, (str, Path)) else dict(inputs)
            self._overflow_candidate = bound
            from .pipeline.current_inputs import CurrentInputReader
            from .shared_calibration_cohort import current_mass_cohort
            cohort = current_mass_cohort(self.series_root, self.roles)
            from .mass.overflow_phase_split import POLICY
            read_request(bound, roles=self.roles,
                         **({"cohort": cohort} if cohort is not None else {}),
                         policy=POLICY,
                         reader=CurrentInputReader(series_roots=(self.series_root,)))
            self.overflow_inputs = bound
            self.overflow_inputs_button.setToolTip(bound["path"])
        self.f_check.setChecked(bool(enabled))
        self._remember_inputs()
        self._update_actions()

    @staticmethod
    def _input_settings():
        return QSettings("DarSIA", "Mass calibration")

    def _input_settings_key(self):
        if self.series_root is None:
            return None
        from .storage_paths import resolve_storage_path
        path = str(resolve_storage_path(self.series_root)).replace("\\", "/").casefold()
        return "series_inputs/" + hashlib.sha256(path.encode("utf-8")).hexdigest()

    def _remember_inputs(self):
        """Remember explicit local GUI choices, never edit a series or choose new rules."""
        if self._restoring_inputs or self.baseline is None or self.roles is None:
            return
        from .pipeline.job_queue import descriptor
        from .storage_paths import resolve_storage_path
        key = self._input_settings_key()
        if key is None:
            return
        rules = None
        rules_text = self.rules_edit.text().strip()
        if self.rules is not None and rules_text:
            try:
                rules_path = resolve_storage_path(rules_text)
                if rules_path.is_file():
                    rules = descriptor(rules_path)
            except (OSError, ValueError):
                # An incomplete or unavailable choice is not a remembered file.
                pass
        value = dict(baseline_sha256=self.baseline["sha256"], roles=self.roles,
                     rules=rules, overflow=self.overflow_inputs,
                     overflow_enabled=self.f_check.isChecked(), calibration_scope=self.calibration_scope_combo.currentData())
        self._input_settings().setValue(key, json.dumps(value, sort_keys=True))

    def _restore_inputs(self):
        """Reuse only the user's exact sources for this same baseline and run roles."""
        key = self._input_settings_key()
        if key is None or self.baseline is None or self.roles is None:
            return
        raw = self._input_settings().value(key, "", type=str)
        if not raw:
            return
        self._restoring_inputs = True
        try:
            value = json.loads(raw)
            if (value.get("baseline_sha256") != self.baseline["sha256"]
                    or value.get("roles") != self.roles):
                return
            from .pipeline.current_inputs import CurrentInputReader
            if value.get("rules"):
                try:
                    self.load_rules(CurrentInputReader().path(value["rules"]))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.rules_label.setText(f"The previously chosen rules cannot be used: {error}")
            scope_index = self.calibration_scope_combo.findData(value.get("calibration_scope", "shared"))
            self.calibration_scope_combo.setCurrentIndex(max(0, scope_index))
            self.f_check.setChecked(bool(value.get("overflow_enabled", False)))
            if value.get("overflow"):
                try:
                    self.set_overflow_enabled(self.f_check.isChecked(), value["overflow"])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.overflow_inputs_button.setToolTip(
                        f"The previously chosen overflow sources cannot be used: {error}")
        except (ValueError, TypeError, AttributeError):
            return
        finally:
            self._restoring_inputs = False

    def _overflow_choice_changed(self, *_args):
        self._remember_inputs()
        self._update_actions()

    def _scope_changed(self, *_args):
        self._read_scope_roles()
        if self.rules_edit.text():
            self.load_rules(self.rules_edit.text())
        if self.phase_job:
            self.check_phase_job(self.phase_job)
        self._remember_inputs()
        self._update_actions()

    def _read_scope_roles(self):
        self.roles = None
        if self.series_root is None:
            self.roles_label.setText("—")
            return
        try:
            self.roles, summary = series_refit_roles(
                self.series_root, scope=self.calibration_scope_combo.currentData())
        except (OSError, ValueError, TypeError) as error:
            summary = f"The run roles could not be read: {error}"
        self.roles_label.setText(summary)
        self.roles_label.setToolTip("\n".join(
            f"{key}: {', '.join(runs) or '—'}" for key, runs in (self.roles or {}).items()))

    def _scope_options(self):
        return ({} if self.calibration_scope_combo.currentData() == "shared" else
                dict(calibration_scope="individual", loro=False))

    def choose_overflow_inputs(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose reviewed overflow inputs", "", "JSON (*.json)")
        if path:
            try:
                self.set_overflow_enabled(True, path)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.source_folder_panel.show()
                self.source_folder_status.setText(str(error))
                self._show_source_folder()
                self.source_folder_status.setText(
                    f"These inputs cannot yet be used: {error}. Locate their moved source folder below.")
                self._update_actions()

    def _show_source_folder(self):
        bound = self._overflow_candidate or self.overflow_inputs
        self.source_folder_panel.show()
        if bound is None:
            self.source_folder_status.setText("Choose the reviewed overflow inputs file first.")
            return
        try:
            from .calibration_source_locations import overflow_source_folder
            self.source_old_folder.setText(overflow_source_folder(bound, series_root=self.series_root))
            self.source_folder_status.setText(
                "Choose the current folder; its sources must match the selected request exactly.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.source_folder_status.setText(str(error))

    def _choose_source_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Locate the moved source folder")
        if path:
            self.source_new_folder.setText(path)

    def _use_source_folder(self):
        bound = self._overflow_candidate or self.overflow_inputs
        if bound is None or self.series_root is None or self._thread is not None:
            return False
        from .calibration_source_locations import relocate_overflow_sources
        root, source = self.series_root, self.source_old_folder.text().strip()
        destination = self.source_new_folder.text().strip()
        self.source_folder_status.setText("Verifying the existing selected sources…")
        return self._call(lambda: relocate_overflow_sources(root, bound, source, destination),
                          self._source_folder_saved)

    def _source_folder_saved(self, result):
        if self._call_generation != self._generation:
            return
        # The worker has run the original complete request reader before saving.
        self.overflow_inputs = self._overflow_candidate = result["request"]
        self.overflow_inputs_button.setToolTip(self.overflow_inputs["path"])
        self.f_check.setChecked(True)
        self.source_folder_status.setText(
            f"Verified {result['verified_sources']} unchanged sources. Shared folder location saved.")
        self._remember_inputs()
        self._update_actions()

    def set_series(self, series_root) -> None:
        """Read the roles and the baseline of this series; None clears the panel.

        Another series clears the chosen rules, phase-one job and open job. The
        same series keeps them and only re-reads what the series records.
        """
        root = None if series_root is None else Path(series_root)
        changed = root != self.series_root
        if changed:
            self.overflow_inputs = None
            self._overflow_candidate = None
            self.source_folder_panel.hide()
            with QSignalBlocker(self.f_check):
                self.f_check.setChecked(False)
            self._automatic = None
            self._phase_wait_timer.stop()
            self._timer.stop()
            self._generation += 1
            self._refresh_pending = False
            self.series_root = root
            self.rules = None
            self.rules_edit.clear()
            self.rules_label.setText("Choose the mass calibration rules file.")
            self.phase_job, self.phase_ready = "", False
            self.phase_edit.clear()
            self.phase_label.setText("Choose the completed phase-map generation job for the fit runs.")
            self.job_root = self.snapshot = self.last_launch = None
            self.name_edit.clear()
            self.recipe_table.setRowCount(0)
            self._show_snapshot(None)
        self.roles = self.baseline = None
        if root is None:
            self.roles_label.setText("—")
            self.baseline_label.setText("—")
            self._update_actions()
            return
        if not self.jobs_edit.text().strip() or self.jobs_edit.text() == self._default_jobs_root:
            from .series_paths import series_folder_path
            self._default_jobs_root = str(series_folder_path(root, "calibration_jobs"))
            self.jobs_edit.setText(self._default_jobs_root or "")
        self._read_scope_roles()
        try:
            self.baseline = baseline_calibration(root)
            self.baseline_label.setText(
                f"Starting values for a new fit · {Path(self.baseline['path']).name} · SHA-256 {self.baseline['sha256'][:12]}…")
            self.baseline_label.setToolTip(self.baseline["path"] +
                "\nStarting parameters for a new fit. The new fit uses its completed colour paths and statistics from generated phase maps; this does not activate a historical calibration for production.")
        except (OSError, ValueError) as error:
            self.baseline_label.setText(f"No baseline: {error}")
        if self.rules_edit.text():
            self.load_rules(self.rules_edit.text())
        elif changed:
            self._restore_inputs()
        self._update_actions()

    def choose_rules(self) -> None:
        """Choose and check the refit rules file."""
        path, _ = QFileDialog.getOpenFileName(self, "Choose the mass calibration rules", "", "JSON (*.json)")
        if path:
            self.load_rules(path)

    def load_rules(self, path) -> bool:
        """Check a rules file against the roles and baseline; True when it is admitted."""
        self.rules = None
        self.rules_edit.setText(str(path))
        if not self.roles or not self.baseline:
            self.rules_label.setText("The rules are checked once the run roles and the baseline are read.")
            self._update_actions()
            return False
        try:
            self.rules, summary = read_rules(path, roles=self.roles, baseline=self.baseline)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.rules_label.setText(f"These rules cannot be used: {error}")
        else:
            self.rules_label.setText(summary)
            self._remember_inputs()
        self._update_actions()
        return self.rules is not None

    def choose_phase_job(self) -> None:
        """Choose the completed phase-map generation job folder and check its fit runs."""
        path = QFileDialog.getExistingDirectory(self, "Choose the completed phase-map generation job")
        if path:
            self.check_phase_job(path)

    def check_phase_job(self, path) -> bool:
        """Check in the background that every fit run has a published phase-one result."""
        fit = list((self.roles or {}).get("fit") or [])
        if not fit:
            self.phase_label.setText("The series records no fit run.")
            return False
        self.phase_job, self.phase_ready = str(path), False
        self.phase_edit.setText(str(path))
        self.phase_label.setText(f"Checking {len(fit)} fit runs…")
        self.recipe_table.setRowCount(0)
        return self._call(lambda: phase_job_summary(str(path), fit), self._phase_checked)

    @Slot(object)
    def _phase_checked(self, result) -> None:
        if self._call_generation != self._generation:
            return
        ready, problems, recipes = result
        self.recipe_table.setRowCount(len(recipes))
        for row, values in enumerate(recipes):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(value)
                self.recipe_table.setItem(row, column, item)
        total = len(ready) + len(problems)
        self.phase_ready = not problems and bool(ready)
        text = f"{len(ready)} of {total} fit runs have completed phase maps and calibration statistics."
        if problems:
            text += " Missing: " + "; ".join(problems[:3]) + (" …" if len(problems) > 3 else "")
        self.phase_label.setText(text)

    def choose_jobs_folder(self) -> None:
        """Choose the folder in which the new refit job folder is made."""
        path = QFileDialog.getExistingDirectory(self, "Choose the jobs folder")
        if path:
            self.jobs_edit.setText(path)

    def _create_blocker(self, *, require_phase=True) -> str:
        """Why Create cannot be offered yet, or '' when it can."""
        from darsia.fluidflower.pipeline.job_queue import name

        if self.series_root is None:
            return "No series."
        if self.f_check.isChecked() and self.overflow_inputs is None:
            return "Choose the reviewed overflow method and scale-source inputs for the free-gas target."
        if self.chemistry_combo.currentData() != "titration":
            return "This chemistry model is not available yet. Choose Titration."
        if not self.roles or not self.roles.get("fit"):
            return "The series records no fit run."
        if self.baseline is None:
            return "The series references no readable mass calibration."
        if self.rules is None:
            return "Choose a rules file that fits the roles and the baseline."
        if require_phase and not self.phase_ready:
            return "Choose a phase-map generation job with completed results for every fit run."
        if not self.jobs_edit.text().strip():
            return "Choose the jobs folder."
        if not self.name_edit.text().strip():
            return "Name the calibration first."
        try:
            name(self.name_edit.text().strip())
        except ValueError:
            return "Use a calibration name of 1–128 letters, numbers, dots, underscores or hyphens; avoid reserved filesystem names."
        return ""

    # ----------------------------------------------------------- actions
    def create(self) -> bool:
        """Prepare local refit plans in the background; start no calculation."""
        blocker = self._create_blocker()
        if blocker or self._thread is not None:
            if blocker:
                QMessageBox.information(self, "Create mass calibration job", blocker)
            return False
        job_id = self.name_edit.text().strip()
        requested_root, requested_generation = self.series_root, self._generation
        jobs_root = self.jobs_edit.text().strip()
        if (self.series_root != requested_root or self._generation != requested_generation
                or self.name_edit.text().strip() != job_id or self._create_blocker()):
            return False
        arguments = dict(roles=dict(self.roles), rules=dict(self.rules), baseline=dict(self.baseline),
                         phase_job=self.phase_job, jobs_root=jobs_root, job_id=job_id,
                         loro=self.loro_check.isChecked(), job_identity=self._identity,
                         chemistry_model=self.chemistry_combo.currentData(), local=True)
        arguments.update(self._scope_options())
        if self.f_check.isChecked():
            arguments["overflow"] = deepcopy(self.overflow_inputs)
        root = self.series_root
        self.job_label.setText(f"Creating {job_id}: reading the fit runs' statistics and writing the plans…")
        return self._call(lambda: dict(created=create_job(root, **arguments),
                                       job_root=str(Path(jobs_root) / job_id)), self._created)

    @Slot(object)
    def _created(self, result) -> None:
        if self._call_generation != self._generation:
            return
        self.job_root = Path(result["job_root"])
        tasks = result["created"].get("tasks") or []
        self.job_label.setText(f"Created {self.job_root.name}: {len(tasks)} tasks prepared. Nothing runs before "
                               "Start mass calibration.")
        self.name_edit.setText(self.job_root.name)
        self._refresh_pending = True
        if self._automatic is not None and self._automatic["state"] == "creating":
            self._automatic["state"] = "starting"

    def choose_job(self) -> None:
        """Choose an existing refit job folder to follow."""
        path = QFileDialog.getExistingDirectory(self, "Open an existing mass calibration job")
        if path:
            self.open_job(path)

    def open_job(self, path) -> bool:
        """Follow the refit job in ``path``; another job is refused when it is read."""
        self.job_root, self.snapshot, self.last_launch = Path(path), None, None
        self.name_edit.setText(self.job_root.name)
        self._show_snapshot(None)
        self.job_label.setText(f"Reading {self.job_root.name}…")
        return self.refresh()

    def start(self, *, confirm=True) -> bool:
        """Start the open local refit; legacy worker jobs keep their launcher.

        Refused, with nothing asked or started, while :data:`START_BLOCKER`
        names a reason.
        """
        from darsia.fluidflower.pipeline.refit_jobs import start_refit_job
        from darsia.fluidflower.store_map import default_store_map_path

        if self._start_blocker(automatic=not confirm):
            return False
        requested_root, requested_generation = self.job_root, self._generation
        from darsia.fluidflower.pipeline.local_refit import FORMAT, run_local_refit_job
        try:
            local = _small_json(Path(requested_root) / "job.json").get("format") == FORMAT
        except (OSError, ValueError, TypeError, AttributeError) as error:
            self._failed(str(error))
            return False
        if local:
            window = self.window()
            runner = getattr(window, "process_runner", None)
            depth = getattr(getattr(window, "depth_workflow", None), "sequence", None)
            if (runner is not None and runner.active) or bool(getattr(depth, "running", False)) or not window.isEnabled():
                QMessageBox.information(self, "Start mass calibration", "Wait for the current calculation to finish.")
                return False
            if (self.job_root != requested_root or self._generation != requested_generation
                    or self._start_blocker(automatic=not confirm)
                    or (runner is not None and runner.active)
                    or bool(getattr(depth, "running", False)) or not window.isEnabled()):
                return False
            self._local_window = window
            self._local_previous_enabled = window.isEnabled()
            window.installEventFilter(self)
            window.setEnabled(False)
            self.job_label.setText("Reading generated phase maps and calibration statistics; the mass optimizer has not started.")
            try:
                started = self._call(lambda: run_local_refit_job(
                    requested_root, on_progress=self.local_progress.emit), self._refreshed)
            except Exception:
                self._release_local_window()
                raise
            if not started:
                self._release_local_window()
            return started
        stores = Path(default_store_map_path())
        if not stores.is_file():
            QMessageBox.warning(self, "Start mass calibration",
                                "This machine has no data-store settings yet. Set them on the Jobs page "
                                f"(Machine stores…) before Start.\nExpected: {stores}")
            return False
        answer = QMessageBox.question(
            self, "Start mass calibration",
            f"Start one CPU worker on this machine ({self.machine}) for the mass calibration job "
            f"{requested_root.name} now, as the Jobs page's Start does?\n\n"
            "Nothing is adopted: a finished mass calibration shows its frozen calibration here, and choosing it "
            "for the series stays a separate step.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        if (self.job_root != requested_root or self._generation != requested_generation
                or self._start_blocker()):
            return False  # A nested question cannot approve a changed job or stale read.
        try:
            self.last_launch = start_refit_job(requested_root, machine=self.machine, stores=stores, workers=1)
        except Exception as error:  # The launcher's own refusal, shown as it is.
            QMessageBox.warning(self, "Start mass calibration", f"The worker could not be started: {error}")
            self.refresh()  # A failed launch keeps its original unresolved reservation visible.
            return False
        console = self.last_launch.get("console") or "the job's launch folder"
        self.job_label.setText(f"Start requested on {self.machine}; the launcher's console is {console}. "
                               "The follow view below shows what the worker records.")
        self.refresh()
        return True

    def start_after_phase_one(self, phase_job, required_runs) -> bool:
        """Main's already-authorized P1 start hands its exact job and run list here.

        This never launches P1. It waits for every required original completion,
        then creates and starts the unchanged refit without another question.
        """
        from darsia.fluidflower.pipeline.job_queue import descriptor, name
        if self._thread is not None or self._automatic is not None:
            return False
        runs = list(required_runs)
        if (not runs or len(set(runs)) != len(runs) or any(name(run) != run for run in runs)
                or not self.roles or not set(self.roles.get("fit", ())).issubset(runs)
                or not self.roles.get("fit") or self.rules is None or self.baseline is None
                or self.series_root is None or not self.jobs_edit.text().strip()
                or self.chemistry_combo.currentData() != "titration"):
            raise ValueError("The phase-map run list and complete mass-calibration choices are required")
        if self.f_check.isChecked() and self.overflow_inputs is None:
            raise ValueError("Choose reviewed overflow inputs before starting overflow-target calibration")
        job_id = name(self.name_edit.text().strip())
        phase_root = Path(phase_job).resolve()
        bound = descriptor(phase_root / "job.json")
        phase = _small_json(phase_root / "job.json")
        if list(phase.get("expected_runs", ())) != runs:
            raise ValueError("The required phase-map run order differs from its job")
        self.phase_job = str(phase_root)
        self._timer.stop()
        self._refresh_pending = False
        self.job_root = self.snapshot = None
        self.phase_edit.setText(self.phase_job)
        self.phase_ready = False
        self._automatic = dict(state="waiting", generation=self._generation,
            root=self.series_root, required_runs=runs, phase_bound=bound,
            arguments=deepcopy(dict(roles=self.roles, rules=self.rules, baseline=self.baseline,
                phase_job=self.phase_job, jobs_root=self.jobs_edit.text().strip(), job_id=job_id,
                loro=self.loro_check.isChecked(), chemistry_model="titration", local=True)))
        self._automatic["arguments"].update(self._scope_options())
        if self.f_check.isChecked():
            self._automatic["arguments"]["overflow"] = deepcopy(self.overflow_inputs)
        self.job_label.setText("Phase map generation is running; the mass optimizer has not started.")
        self._update_actions()
        self._poll_phase_one()
        return True

    def _poll_phase_one(self):
        request = self._automatic
        if request is None or request["generation"] != self._generation:
            return
        if self._thread is not None:
            self._phase_wait_timer.start()
            return
        if request["state"] == "start_ready":
            window = self.window()
            runner = getattr(window, "process_runner", None)
            depth = getattr(getattr(window, "depth_workflow", None), "sequence", None)
            if (runner is not None and runner.active) or bool(getattr(depth, "running", False)) or not window.isEnabled():
                self.job_label.setText("Phase-one products are complete; waiting for preparation to finish closing. The mass optimizer has not started.")
                self._phase_wait_timer.start()
                return
            request["state"] = "running"
            if not self.start(confirm=False):
                self._failed("Mass calibration could not start after phase map generation completed.")
            return
        if request["state"] != "waiting":
            return
        def read():
            from darsia.fluidflower.mass.n1_pipeline import DataReader
            DataReader(unc_only=False).path(request["phase_bound"])
            root = Path(request["arguments"]["phase_job"])
            ready, problems = phase_job_readiness(root, request["required_runs"])
            for run in request["required_runs"]:
                if (root / "failed" / (run + ".json")).exists():
                    raise ValueError(f"Phase one failed for {run}; the mass optimizer has not started")
                if run not in ready and (root / "results" / run / "completed.json").exists():
                    raise ValueError("A phase-map generation result is invalid: " + "; ".join(problems))
            return ready
        self._call(read, self._phase_one_waited)

    def _phase_one_waited(self, ready):
        request = self._automatic
        if request is None or request["generation"] != self._generation:
            return
        total = len(request["required_runs"])
        self.job_label.setText(f"Phase map generation: {len(ready)} of {total} runs complete. The mass optimizer has not started.")
        if len(ready) == total:
            self.phase_ready = True
            request["state"] = "create"
        else:
            self._phase_wait_timer.start()

    @Slot(object)
    def _local_progress(self, event):
        step = {"statistics_read": "reading calibration statistics from generated phase maps",
                "overflow_current_volume": "reading current free-gas volumes for the overflow target",
                "overflow_derive": "preparing the overflow-adjusted gas and water capacities",
                "overflow_own_measurement": "using this run's MEASURED overflow for the phase split",
                "overflow_pressure_estimate": "using pressure-matched ESTIMATED overflow where comparable data exist; model split elsewhere",
                "stage1": "optimizing gain", "stage1_grid": "checking the gain grid",
                "stage2": "optimizing beta", "stage2_grid": "checking the beta grid",
                "freeze": "saving the frozen calibration", "complete": "fit finished"}.get(
                    event.get("step"), str(event.get("step") or "reading inputs"))
        self.job_label.setText(f"Mass calibration: {event.get('run', 'refit')} — {step}.")

    def refresh(self) -> bool:
        """Read the open job's status in the background (one read at a time)."""
        if self.job_root is None:
            return False
        if self._thread is not None:
            self._refresh_pending = True
            return False
        root = str(self.job_root)
        return self._call(lambda: read_refit_job(root), self._refreshed)

    @Slot(object)
    def _refreshed(self, result) -> None:
        if self._call_generation != self._generation:
            return
        self.snapshot = result
        self._show_snapshot(result)

    def _show_snapshot(self, snapshot) -> None:
        """Fill the follow view from a read status; None empties it."""
        if snapshot is None:
            self.tasks.setRowCount(0)
            self.result_label.clear()
            if self.job_root is None:
                self.job_label.setText(NO_JOB)
            self._timer.stop()
            self._update_actions()
            return
        status = snapshot["status"]
        rows = status_rows(status)
        self.tasks.setRowCount(len(rows))
        for index, cells in enumerate(rows):
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(value)
                self.tasks.setItem(index, column, item)
        self.job_label.setText(f"Mass calibration job {Path(status['job_root']).name}: {status_summary(status)}")
        self.result_label.setText(frozen_text(snapshot.get("frozen"), snapshot.get("frozen_error", "")))
        if snapshot.get("calibration_scope") == "individual":
            values = snapshot.get("individual_calibrations", {})
            self.result_label.setText("Individual calibrations: " + ("; ".join(
                run + ": " + parameters_text(value[0].get("parameters")) + " · " + value[1]["path"]
                for run, value in values.items()) or "No completed results yet."))
        if self._finished(status):
            self._timer.stop()
        elif self.isVisible():
            self._timer.start()
        self._update_actions()

    @staticmethod
    def _finished(status) -> bool:
        tasks = (status or {}).get("tasks") or {}
        return bool(tasks) and all(value.get("state") in FINAL_STATES for value in tasks.values())

    def _update_actions(self, *_args) -> None:
        explanation = fit_image_explanation(self.roles, self.rules)
        if self.calibration_scope_combo.currentData() == "individual":
            explanation = explanation.replace("Leave-one-run-out repeats the fit with one fit run omitted, then predicts that run.",
                "Each selected fit run receives its own gain/beta fit. Control/validation runs do not enter these fits.")
            self.parameter_explanation.setText("A separate gain and beta are fitted for each selected fit run. "
                + TITRATION_EXPLANATION.replace("Gain is one common factor", "Gain is a factor"))
        else:
            self.parameter_explanation.setText(TITRATION_EXPLANATION)
        self.fit_images.setText(explanation)
        idle = self._thread is None and self._automatic is None
        series = self.series_root is not None
        self.create_button.setEnabled(idle and series and not self._create_blocker())
        for widget in (self.chemistry_combo, self.calibration_scope_combo, self.name_edit, self.jobs_edit, self.f_check):
            widget.setEnabled(idle)
        self.loro_check.setEnabled(idle and self.calibration_scope_combo.currentData() == "shared")
        self.create_button.setToolTip(self._create_blocker() or "Prepares local plans; starts no calculation.")
        for button in (self.rules_button, self.phase_button, self.jobs_button, self.open_button, self.overflow_inputs_button):
            button.setEnabled(idle and series)
        self.source_folder_button.setEnabled(idle and series)
        self.source_folder_panel.setEnabled(idle and series)
        self.source_folder_use.setEnabled(idle and series and self._overflow_candidate is not None)
        blocker = self._start_blocker()
        self.start_button.setEnabled(not blocker)
        self.start_button.setToolTip(blocker or "Runs the original mass fit on this computer.")
        self.start_reason.setText(blocker)
        self.start_reason.setVisible(bool(blocker))
        self.refresh_button.setEnabled(idle and self.job_root is not None)
        self.input_choices_changed.emit()

    def _start_blocker(self, *, automatic=False) -> str:
        if START_BLOCKER:
            return START_BLOCKER
        if self._thread is not None:
            return "Wait for the current job read to finish."
        if self._automatic is not None and not (automatic and self._automatic["state"] == "running"):
            return "Waiting for phase map generation and mass calibration."
        if self.chemistry_combo.currentData() != "titration":
            return "This chemistry model is not available yet. Choose Titration."
        if not self.name_edit.text().strip():
            return "Name the calibration first."
        if self.job_root is None or self.snapshot is None or "start" not in self.snapshot:
            return "Read an existing mass calibration job before Start."
        if self.name_edit.text().strip() != self.job_root.name:
            return "The name differs from the open calibration job. Create that named job, or reopen the existing one."
        if self.f_check.isChecked() != bool(self.snapshot.get("local_variant_f", False)):
            return "The open job uses a different overflow choice. Create a job with the selected free-gas target."
        if self.calibration_scope_combo.currentData() != self.snapshot.get("calibration_scope", "shared"):
            return "The open job uses a different shared/individual calibration choice. Create a job with the selected choice."
        return self.snapshot["start"]["blocker"]

    # ------------------------------------------------- background calls
    def _call(self, call, ready) -> bool:
        """Run ``call`` in a thread and hand its result to ``ready`` in the GUI thread."""
        from darsia.fluidflower.jobs_panel import _CallWorker

        if self._thread is not None:
            return False
        thread = QThread(self)
        worker = _CallWorker(call)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(ready)
        worker.failed.connect(self._failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        # No deleteLater for the worker (Q185): _closed drops it after wait(),
        # which frees it in the GUI thread.
        thread.finished.connect(self._closed)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self._read_window = self.window()
        self._read_window.installEventFilter(self)
        self._call_generation = self._generation
        self._update_actions()
        thread.start()
        return True

    @Slot(str)
    def _failed(self, text) -> None:
        self._automatic = None
        self._phase_wait_timer.stop()
        self.snapshot = None  # A failed read is not fresh evidence that Start is safe.
        last = text.strip().splitlines()[-1] if text.strip() else "unknown error"
        self.job_label.setText(f"That did not work: {last}")
        self.job_label.setToolTip(text)
        self._update_actions()

    @Slot()
    def _closed(self) -> None:
        if self._thread is not None:
            self._thread.wait()
        local_finished = self._local_window is not None
        self._thread = self._worker = None
        read_window, self._read_window = self._read_window, None
        if read_window is not None:
            read_window.removeEventFilter(self)
        self._release_local_window()
        if local_finished:
            self._automatic = None
        request = self._automatic
        if request is not None and request["generation"] == self._generation:
            if request["state"] == "create":
                request["state"] = "creating"
                args = request["arguments"]
                self._call(lambda: dict(created=create_job(request["root"], **args),
                    job_root=str(Path(args["jobs_root"]) / args["job_id"])), self._created)
                return
            if request["state"] == "starting":
                request["state"] = "start_ready"
                self._refresh_pending = False
                self._call(lambda: read_refit_job(self.job_root), self._refreshed)
                return
            if request["state"] == "start_ready":
                self._poll_phase_one()
                return
        self._update_actions()
        if self._refresh_pending:
            self._refresh_pending = False
            self.refresh()

    def _release_local_window(self):
        window, self._local_window = self._local_window, None
        if window is not None:
            window.removeEventFilter(self)
            window.setEnabled(self._local_previous_enabled)

    def eventFilter(self, watched, event):
        if (watched is self._local_window or watched is self._read_window) and event.type() == QEvent.Type.Close:
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        status = (self.snapshot or {}).get("status")
        if self.job_root is not None and status is not None and not self._finished(status):
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._timer.stop()
        super().hideEvent(event)
