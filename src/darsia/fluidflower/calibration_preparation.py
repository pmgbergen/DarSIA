"""Calibration's explicit entry to the existing Build, P1 and refit route."""
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFormLayout, QHBoxLayout, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QSpinBox, QVBoxLayout, QWidget)

from .pipeline.current_model_basis import UNBOUND_MODEL_REASON, require_current_model_basis


def require_colour_path_choice(selection, *, reader=None):
    """Use the already bound current choice; do not infer a historical image basis."""
    from .pipeline.setup_image_preparation import independent_choice
    _document, choice = independent_choice(selection["input_authority"]["current_series"], reader=reader)
    if choice is None:
        raise ValueError("Save the current image-preparation choices in Setup and complete their Apply/Build image operation before preparing new colour paths.")


def registered_preparation_snapshots(registrations, series_id, snapshot_reader):
    """The active explicit Setup edition takes precedence over old completions."""
    snapshots, active_setup = [], None
    for row in registrations:
        if row.get("archived"):
            continue
        snapshot = snapshot_reader(row)
        snapshots.append(snapshot)
        facts = snapshot.get("job_facts", {})
        if (row.get("active") and facts.get("pipeline") == "series-setup"
                and facts.get("series_id") == series_id):
            active_setup = snapshot
    if active_setup is not None:
        if not active_setup.get("complete") or active_setup.get("errors"):
            raise ValueError("Complete the active Apply/Build job before phase preparation. An older completed Build will not replace this newly prepared Setup edition.")
        snapshots = [snapshot for snapshot in snapshots
                     if snapshot is active_setup or snapshot.get("job_facts", {}).get("pipeline") != "series-setup"]
    return snapshots


def saved_selection(root, *, actor, reader=None):
    from .pipeline.current_calibration_selection import current_calibration_selection
    value = current_calibration_selection(root, actor=actor,
        reason="Prepare calibration from the saved image selection", reader=reader)
    try:
        value["input_authority"]["selected_model"] = require_current_model_basis(root, value, reader=reader)
    except (OSError, ValueError, KeyError, TypeError) as error:
        value["model_basis_error"] = str(error)
    return value


def selection_preview(root, cancelled=lambda: False):
    """Read the saved image choices only; scientific admission belongs to Start."""
    import tomllib
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(Path(root) / "series.toml")
    with path.open("rb") as stream:
        raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError("Series metadata exceeds the preview limit")
    document = tomllib.loads(raw.decode("utf-8-sig"))
    if cancelled():
        return None
    chosen = document.get("image_selection", {}).get("runs", {})
    runs = document.get("runs", {}).get("ids", [])
    all_runs = list(runs)
    role_groups = {}
    method = document.get("method", {})
    from .series_method import calibration_ancestry_reference
    reference = calibration_ancestry_reference(method.get(method.get("active"), {}))
    if reference:
        import json
        calibration_path = Path(reference)
        if not calibration_path.is_absolute():
            calibration_path = Path(root) / calibration_path
        calibration_path = resolve_storage_path(calibration_path, series_roots=(Path(root),))
        with calibration_path.open("rb") as stream:
            metadata = stream.read(1024**2 + 1)
        if len(metadata) > 1024**2:
            raise ValueError("Saved calibration metadata exceeds the preview limit")
        calibration = json.loads(metadata)
        wanted = calibration.get("calibration_runs")
        role_groups = {"Control / validation": calibration.get("heldout_runs", []),
                       "Descriptive / control": calibration.get("descriptive_runs", []),
                       "Excluded": calibration.get("excluded_runs", [])}
        if wanted is not None:
            if (not isinstance(wanted, list) or not all(isinstance(run, str) for run in wanted)
                    or len(wanted) != len(set(wanted)) or not set(wanted) <= set(runs)):
                raise ValueError("The saved calibration names another or malformed run selection")
            runs = [run for run in runs if run in wanted]
    fit_runs = set(runs)
    included = [run for run in all_runs if run not in role_groups.get("Excluded", ())]
    records = []
    for run in included:
        if cancelled():
            return None
        saved = chosen.get(run)
        if saved is None:
            continue
        rows = saved.get("selected", [])
        selected = len(rows)
        records.append(dict(run=run, calibration_role="fit" if run in fit_runs else "control",
            counts=dict(selected_frames=selected,
            fit_candidates=None, earlier_history=None)))
    display = []
    for run in all_runs:
        rows = chosen.get(run, {}).get("selected", [])
        times = [float(row["elapsed_h"]) for row in rows if isinstance(row.get("elapsed_h"), (int, float))]
        baselines = sum(row.get("frame_index") == 0 for row in rows)
        role = ("Fit" if run in fit_runs else next((key for key, values in role_groups.items() if run in values),
                                                  "Outside fit / control"))
        display.append(dict(run=run, role=role, selected_images=len(rows)-baselines, baselines=baselines,
                            time_range_h=[min(times), max(times)] if times else None))
    from .pipeline.current_optical_preparation import colour_path_image_count
    return dict(preview_only=True, runs=records, mass_fit_runs=[run for run in included if run in fit_runs],
        colour_path_count=colour_path_image_count(document), colour_path_manifest_bytes=raw,
        colour_path_choice_explicit="color_paths" in document.get("image_selection", {}), counts=dict(
        selected_frames=sum(row["counts"]["selected_frames"] for row in records),
        fit_candidates=None), model_basis_error="", display_runs=display)


def prepare_calibration_phase(root, series_id, registrations, *, actor, machine,
                              stores_path=None, snapshot_reader=None, reader=None,
                              producer=None, model_reference=None, require_fresh_paths=False):
    """Use current saved choices and a completed Build; reuse matching P1 work.

    Reads only explicitly registered jobs. P1 creation remains the original
    producer, and this function never launches a worker or fits a calibration.
    """
    from .jobs_metadata import JobSnapshotReader
    from .mass.n1_pipeline import DataReader
    from .pipeline.current_build_job import prepare_current_build_job
    reader = reader or DataReader()
    snapshot_reader = snapshot_reader or JobSnapshotReader()
    selection = saved_selection(root, actor=actor, reader=reader)
    if require_fresh_paths:
        require_colour_path_choice(selection, reader=reader)
    binding_options = dict(reader=reader)
    if model_reference is not None:
        binding_options["model_reference"] = model_reference
    selection["input_authority"]["selected_model"] = require_current_model_basis(root, selection, **binding_options)
    selection.pop("model_basis_error", None)
    snapshots = registered_preparation_snapshots(registrations, series_id, snapshot_reader)
    from .pipeline.current_training_preparation import find_completed_build
    build = find_completed_build(root, series_id, snapshots, selection, reader=reader)
    selection["input_authority"]["selected_model"] = require_current_model_basis(
        root, selection, build_snapshot=build, **binding_options)
    operations = {row["run"]: row["result"]["files"]["operation_result.json"]
                  for row in build["runs"] if row.get("operation") == "build" and row.get("state") == "Complete"}
    runs = [row["run"] for row in selection["runs"]]
    for candidate in reversed(snapshots):
        facts = candidate.get("job_facts", {})
        if (facts.get("execution_mode") != "current_build" or candidate.get("errors")
                or facts.get("series_id") != series_id or facts.get("expected_runs") != runs):
            continue
        if require_fresh_paths and not facts.get("fresh_optical_paths"):
            continue
        if {row["run"]: row["operation_result"] for row in facts.get("inputs", [])} != {
                run: operations.get(run) for run in runs}:
            continue
        previous = reader.json(facts["selection"])
        from .pipeline.current_calibration_selection import same_preparation_selection
        if not same_preparation_selection(previous, selection):
            continue
        if candidate.get("complete"):
            return dict(snapshot=candidate, selection=selection, reused=True, registration=None)
        if candidate.get("workers"):
            return dict(snapshot=candidate, selection=selection, reused=True, registration=None)
    registration = (producer or prepare_current_build_job)(build, series_root=root,
        series_id=series_id, stores_path=stores_path, machine=machine, actor=actor, reader=reader,
        **({"model_reference": model_reference} if model_reference is not None else {}))
    return dict(snapshot=snapshot_reader(registration), selection=selection,
                reused=False, registration=registration)


def prepare_calibration_training(root, series_id, registrations, *, actor, workers=1,
                                 on_progress=None, template_reference=None):
    """Prepare real corrected RAW reviews and verified derivative labels once."""
    from .jobs_metadata import JobSnapshotReader
    from .pipeline.current_training_preparation import (find_completed_build,
        prepare_training_from_completed_build)
    from .series_locations import read_series_locations
    from .mass.n1_pipeline import DataReader
    from .phase_training.raw_review_training import canonical_digest
    from .native.raw_rig_source import execution_identity
    from .native.full_photometry import ORDER
    reader = DataReader()
    selection = saved_selection(root, actor=actor, reader=reader)
    snapshots = registered_preparation_snapshots(registrations, series_id, JobSnapshotReader())
    build = find_completed_build(root, series_id, snapshots, selection, reader=reader)
    extra = {}
    if template_reference:
        from .pipeline.current_training_preparation import selected_training_template
        extra["training_template"] = selected_training_template(root, reader=reader,
            model_reference=template_reference)
    identity = canonical_digest(dict(series=series_id,
        setup=selection["input_authority"]["current_series"]["sha256"],
        build=Path(build["root"]).name, pixel_producer=execution_identity(), correction_order=ORDER,
        **({"template": extra["training_template"]["request"]} if template_reference else {})))
    output = read_series_locations(root)["jobs_folder"] / "calibration-preparation" / identity
    result = prepare_training_from_completed_build(build, series_root=root, series_id=series_id,
        output=output, actor=actor, on_progress=on_progress, reader=reader, workers=workers, **extra)
    model = Path(result["request"]["output_root"])
    if (model / "summary.json").is_file():
        require_current_model_basis(root, selection, reader=reader, model_reference=str(model))
        result["model_ready"] = True
    return result


def training_worker_count(jobs):
    """Use the visible Shared Jobs CPU policy for CPU-only image preparation."""
    shared = getattr(jobs, "shared_jobs", None)
    if shared is None:
        raise ValueError("Open Shared Jobs and choose worker capacity before preparing training images.")
    if shared.automatic.isChecked():
        return 0  # The existing preparer bounds automatic CPU workers by RAM and cores.
    count = shared.cpu.value()
    if count < 1:
        raise ValueError("Training image preparation needs CPU workers. Enable Automatic or set CPU workers above zero.")
    return count


class CalibrationPreparation(QWidget):
    """Show the saved cohort and start its prerequisite without a P1 picker."""
    training_progress = Signal(object)
    setup_completion_changed = Signal(bool)

    def __init__(self, window, panel, parent=None):
        super().__init__(parent)
        self.window_owner, self.panel = window, panel
        self.root = None
        self.selection = None
        self._thread = self._worker = None
        from .ui_metadata_preview import MetadataPreview
        self._preview = MetadataPreview(self)
        self._preview.completed.connect(self._preview_ready)
        self._generation = 0
        self._training_context = None
        self._candidate_reference = None
        self._phase_start_pending = None
        self._preparation_only_view = False
        self._preparation_only = False
        self._preparation_job = None
        self._watched_jobs = None
        self._model_retraining_attempted = False
        self._retry_model_basis = False
        self._model_publication_pending = False
        self._model_publication_context = None
        self._published_candidate = None
        self._ready_callback = None
        self._call_generation = 0
        self._colour_manifest_bytes = None
        self._saved_colour_count = 15
        self._colour_choice_explicit = False
        self._saving_colour_count = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.order = QLabel("Completed Finish setup → train or select the phase model → "
                            "generate phase maps from model → mass calibration.")
        self.order.setWordWrap(True)
        layout.addWidget(self.order)
        prerequisites = QHBoxLayout()
        self.images = QPushButton("Image normalisation")
        self.images.clicked.connect(lambda: window._select_row(("phase_model", "normalisation")))
        self.model = QPushButton("Phase model")
        self.model.clicked.connect(window.show_phase_model_workflow)
        prerequisites.addWidget(self.images)
        prerequisites.addWidget(self.model)
        layout.addLayout(prerequisites)
        # One visible owner in Model and parameters; preparation reads that
        # same choice instead of presenting another synchronised selector.
        self.scope = panel.calibration_scope_combo
        self.basis = QLabel()
        self.basis.setWordWrap(True)
        layout.addWidget(self.basis)
        self.colour_choices = QWidget()
        colour_form = QFormLayout(self.colour_choices)
        colour_form.setContentsMargins(0, 0, 0, 0)
        self.colour_count = QSpinBox()
        self.colour_count.setRange(1, 1000)
        self.colour_count.setValue(15)
        self.colour_count.valueChanged.connect(self._colour_count_changed)
        colour_form.addRow("Colour-path images per run", self.colour_count)
        colour_note = QLabel("Up to this many representative post-injection images from every included run. "
            "The saved clear-water and pre-injection references remain separate. "
            "Mass calibration and production keep their own image selections.")
        colour_note.setWordWrap(True)
        colour_form.addRow(colour_note)
        self.save_colour_count = QPushButton("Save colour-path selection")
        self.save_colour_count.clicked.connect(self._save_colour_count)
        colour_form.addRow(self.save_colour_count)
        self.colour_status = QLabel()
        self.colour_status.setWordWrap(True)
        colour_form.addRow(self.colour_status)
        layout.addWidget(self.colour_choices)
        self.summary = QLabel("Saved calibration image selection is not loaded.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(("Run", "Mass-calibration role", "Saved source images", "Baselines", "Time range [h]", "History images"))
        self.table.setColumnHidden(1, True)  # The operative fit/holdout choice lives in Model parameters.
        self._display_runs = []
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(190)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.edit = QPushButton("Edit image selection")
        self.edit.clicked.connect(lambda: (
            window.open_image_selection_plan() if self._preparation_only_view
            else window.open_calibration_image_selection()))
        self.start = QPushButton("Prepare and start calibration")
        self.start.clicked.connect(lambda: self.prepare_and_start(preparation_only=self._preparation_only_view))
        buttons.addWidget(self.edit)
        buttons.addWidget(self.start)
        layout.addLayout(buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.training_progress.connect(self._show_training_progress)
        panel.input_choices_changed.connect(self._actions)
        self._actions()

    def set_preparation_only_view(self, enabled):
        """Present the same producer in Setup without changing an active action's intent."""
        self._preparation_only_view = bool(enabled)
        self.colour_choices.setVisible(bool(enabled))
        self.order.setText(
            "Completed Finish setup with corrections, crop and colour paths → retrain the selected model when needed → "
            "generate phase maps and calibration statistics. Mass calibration is a later explicit action."
            if enabled else "Completed Finish setup → train or select the phase model → "
            "generate phase maps from model → mass calibration.")
        self.start.setText("Generate phase maps from model" if enabled else "Prepare and start calibration")
        self.images.setText("Image correction")
        self._actions()

    def _basis_blocker(self):
        if self.root is None:
            return "Choose a series."
        if self.selection is None:
            return "Reading the saved image selection and model basis…"
        reason = self.selection.get("model_basis_error", "")
        # A historical model is a valid architecture template. The explicit
        # Start action now prepares its inputs and fits new weights first.
        return "" if reason == UNBOUND_MODEL_REASON else reason

    def _actions(self):
        idle = (self._thread is None and self._training_context is None and self._phase_start_pending is None
                and not self._model_publication_pending
                and self.panel._thread is None and self.panel._automatic is None)
        reason = self._basis_blocker()
        self.basis.setText(reason or (self.selection or {}).get("model_basis_error", ""))
        if not self._preparation_only_view:
            reason = reason or self.panel._create_blocker(require_phase=False)
        elif self._colour_manifest_bytes is not None and self.colour_count.value() != self._saved_colour_count:
            reason = reason or "Save the colour-path image count before preparing."
        self.start.setEnabled(idle and self.selection is not None and not reason)
        self.start.setToolTip(reason or (
            "Use the saved model and image selection after completed Finish setup. Retrain when needed, then generate phase maps and calibration statistics; do not start mass calibration."
            if self._preparation_only_view else
            "Reuse matching generated phase maps and statistics, or generate them first, then fit this named calibration."))
        self.edit.setEnabled(idle and self.root is not None)
        self.images.setEnabled(idle and self.root is not None)
        self.model.setEnabled(idle and self.root is not None)
        self.scope.setEnabled(idle and self.root is not None)
        self.colour_count.setEnabled(idle and self._colour_manifest_bytes is not None)
        self.save_colour_count.setEnabled(idle and self._colour_manifest_bytes is not None
            and (not self._colour_choice_explicit or self.colour_count.value() != self._saved_colour_count))

    def _colour_count_changed(self):
        self._preparation_job = None
        self.setup_completion_changed.emit(False)
        if self._colour_manifest_bytes is not None:
            self.colour_status.setText("Unsaved colour-path selection. Save explicitly; no job has started."
                if self.colour_count.value() != self._saved_colour_count else
                "Saved colour-path selection." if self._colour_choice_explicit else
                "Default: 15 images per run. Save to record this choice for the series.")
        self._actions()

    def _save_colour_count(self):
        if not self.save_colour_count.isEnabled():
            return
        from .colour_path_selection import save_colour_path_selection
        root, generation, original = self.root, self._generation, self._colour_manifest_bytes
        count = self.colour_count.value()
        self._saving_colour_count = True
        self.colour_status.setText("Saving colour-path selection…")
        self._call(lambda: save_colour_path_selection(root, count, expected_bytes=original,
            context_current=lambda: self.root == root and self._generation == generation),
            lambda raw: self._colour_count_saved(raw, count))

    def _colour_count_saved(self, raw, count):
        self._saving_colour_count = False
        self._colour_manifest_bytes = raw
        self._saved_colour_count, self._colour_choice_explicit = count, True
        self._preparation_job = None
        self.setup_completion_changed.emit(False)
        if self.selection is not None:
            self.selection["preview_only"] = True
        refresh = getattr(self.window_owner, "_refresh_method_state", None)
        if callable(refresh):
            refresh(force=True)
        self.colour_status.setText(f"✓ Colour-path selection saved: up to {count} images per included run, "
            "plus the saved references. Mass-calibration and production selections are unchanged. No job started.")

    def set_series(self, root):
        root = Path(root) if root is not None else None
        if root == self.root and self._thread is not None:
            return
        self._preparation_job = None
        self._colour_manifest_bytes = None
        self._saving_colour_count = False
        self.colour_status.clear()
        self.setup_completion_changed.emit(False)
        self.setVisible(root is not None)
        if root != self.root:
            self._candidate_reference = None
            self._published_candidate = None
            self._display_runs = []
        self.root = root
        self._generation += 1
        self._preview.cancel()
        self.selection = None
        self.table.setRowCount(0)
        self._actions()
        if root is None:
            return
        if self._thread is not None:
            self.status.setText("The previous preparation read is finishing. Reopen Calibration to refresh the selection.")
            return
        self.summary.setText("Reading saved calibration image choices…")
        self._preview.start(lambda cancelled: selection_preview(root, cancelled))

    def _preview_ready(self, value, error):
        if error:
            self.status.setText(error)
        elif value is not None:
            self._selected(value)
        self._actions()

    def cancel_preview(self):
        self._preview.cancel()

    def hideEvent(self, event):
        self.cancel_preview()
        super().hideEvent(event)

    def _selected(self, value):
        self.selection = value
        if "colour_path_manifest_bytes" in value:
            self._colour_manifest_bytes = value["colour_path_manifest_bytes"]
            self._saved_colour_count = value["colour_path_count"]
            self._colour_choice_explicit = value.get("colour_path_choice_explicit", False)
            with QSignalBlocker(self.colour_count):
                self.colour_count.setValue(self._saved_colour_count)
            self.colour_status.setText(
                f"Saved: up to {self._saved_colour_count} colour-path images per run."
                if self._colour_choice_explicit else "Default: 15 images per run. Save to record this choice for the series.")
        rows = value["runs"]
        if value.get("display_runs"):
            self._display_runs = deepcopy(value["display_runs"])
        display = {row["run"]: dict(row) for row in self._display_runs}
        for row in rows:
            role = {"fit": "Fit", "heldout": "Control / validation", "descriptive": "Descriptive / control",
                    "control": "Outside fit / control"}.get(row.get("calibration_role"), "Fit")
            shown = display.setdefault(row["run"], dict(run=row["run"], role=role, baselines=1 if row.get("baseline") else 0))
            if not value.get("preview_only") or "selected_images" not in shown:
                shown["selected_images"] = row["counts"]["selected_frames"]
            shown["history"] = row["counts"]["earlier_history"]
        self.table.setRowCount(len(display))
        for index, row in enumerate(display.values()):
            coverage = row.get("time_range_h")
            interval = "—" if coverage is None else f"{coverage[0]:.2f} to {coverage[1]:.2f}"
            for column, cell in enumerate((row["run"], row["role"], row["selected_images"],
                                           row["baselines"], interval, row.get("history"))):
                self.table.setItem(index, column, QTableWidgetItem("Checked on Start" if cell is None else str(cell)))
        self.table.resizeColumnsToContents()
        counts = value["counts"]
        if value.get("preview_only"):
            self.summary.setText(f"Colour paths: up to {self._saved_colour_count} post-injection images per run "
                f"across {len(rows)} included runs, plus the saved references. "
                "The table shows the larger saved source pool. "
                "Choose shared or individual calibration and its fit/holdout runs under Mass calibration → Model parameters. "
                "The model basis and protocols are checked on Start.")
            self.status.clear()
            return
        self.summary.setText(f"Colour paths: up to {self._saved_colour_count} post-injection images per run "
            f"across {len(rows)} included runs, plus the saved references. "
            f"Phase/statistics source pool: {counts['selected_frames']} selected images. "
            "Mass-calibration scope and fit/holdout runs are shown in Model parameters. Baselines and saved history are retained. "
            "Matching phase results are reused; otherwise they are prepared before the mass fit. "
            + ("Setup preparation stops here; mass calibration is a later explicit action."
               if self._preparation_only_view else "Keep DarSIA open for the automatic transition to fitting."))
        self.status.clear()

    def _admitted_selection(self, value):
        self._selected(value)
        # Continue after the owned admission thread retires; the next action
        # still uses the original full scientific preparation path.
        self._start_after_selection = True

    @Slot()
    def prepare_and_start(self, *, preparation_only=False, _continuing=False):
        self._actions()
        if not self.start.isEnabled():
            return
        if not _continuing:
            self.setup_completion_changed.emit(False)
            self._model_retraining_attempted = False
            self._candidate_reference = self._published_candidate = None
            self._model_publication_context = (
                self.root, getattr(self.window_owner, "_method_manifest_bytes", None),
                getattr(self.window_owner, "_saved_method", None))
        self._preparation_only = bool(preparation_only)
        self._preparation_job = None
        if self.selection.get("preview_only") or (self._preparation_only and not self.selection.get("setup_colour_paths_ready")):
            root = self.root
            preparation_only = self._preparation_only
            def admit():
                value = saved_selection(root, actor=self.panel.machine)
                if preparation_only:
                    require_colour_path_choice(value)
                    value["setup_colour_paths_ready"] = True
                return value
            self._call(admit, self._admitted_selection)
            return
        jobs = self.window_owner.ensure_jobs()
        root, series_id = self.root, jobs.series_id
        registrations = deepcopy(jobs.registry.jobs(root, series_id))
        stores_path = jobs.stores.path
        if self.selection.get("model_basis_error") and self._candidate_reference is None:
            self._model_retraining_attempted = True
            self.status.setText("Preparing corrected and cropped model inputs from the completed Build...")
            try:
                workers = training_worker_count(jobs)
            except ValueError as error:
                self.status.setText(str(error))
                return
            self._call(lambda: prepare_calibration_training(root, series_id, registrations,
                actor=self.panel.machine, workers=workers, on_progress=self.training_progress.emit),
                lambda result: self._training_prepared(jobs, result, root, series_id), deliver_stale=True)
            return
        self._prepare_phase(jobs, root, series_id, registrations, stores_path)

    def _prepare_phase(self, jobs, root, series_id, registrations=None, stores_path=None):
        if registrations is None:
            registrations = deepcopy(jobs.registry.jobs(root, series_id))
        stores_path = stores_path or jobs.stores.path
        self.status.setText("Preparing calibration from the saved image selection and completed Build...")
        self._call(lambda: prepare_calibration_phase(root, series_id, registrations,
            actor=self.panel.machine, machine=jobs.panel.local_machine, stores_path=stores_path,
            require_fresh_paths=self._preparation_only,
            **({"model_reference": self._candidate_reference} if self._candidate_reference is not None else {})),
            lambda result: self._prepared(jobs, result, root, series_id), deliver_stale=True)

    @Slot(object)
    def _show_training_progress(self, value):
        if not isinstance(value, dict):
            self.status.setText(str(value))
            return
        state = str(value.get("state", "Preparing model inputs")).replace("_", " ")
        run = str(value.get("run", ""))
        counts = (f"{value['position']}/{value['total']}" if "position" in value and "total" in value else "")
        workers = f"{value['workers']} workers" if "workers" in value else ""
        frames = f"{value['frames_prepared']} frames prepared" if "frames_prepared" in value else ""
        self.status.setText(" - ".join(part for part in (state, run, counts, workers, frames) if part))

    def _training_prepared(self, jobs, result, root, series_id):
        if self.root != root or jobs.series_id != series_id:
            self.status.setText("Prepared model inputs were retained; this series is no longer active.")
            return
        if result.get("model_ready"):
            self._candidate_reference = str(Path(result["request"]["output_root"]).resolve())
            self.status.setText("Reusing the completed model fitted on these prepared images.")
            self._continue_with_model(jobs, root, series_id)
            return
        workspace = self.window_owner.phase_model_workspace
        page = workspace.ensure_studio()
        if page.studio is None:
            page._load_studio()
        steps, studio = page.native_steps, page.studio
        if steps is None or studio is None:
            raise ValueError("Could not load the phase-model trainer")
        expected = Path(result["request"]["output_root"]).resolve()
        context = dict(jobs=jobs, root=root, series_id=series_id, studio=studio,
                       expected=expected, generation=self._generation)
        self._training_context = context
        completed = lambda path: self._model_trained(context, path)
        context["completed"] = completed
        studio.training_completed.connect(completed)
        try:
            from .calibration_model_training import launch_prepared_training
            process = launch_prepared_training(steps, result)
            finished = lambda code, status: self._model_process_finished(context, code)
            context["process"], context["finished"] = process, finished
            process.finished.connect(finished)
        except Exception:
            self._disconnect_training(context)
            raise
        self.status.setText("Refitting the selected model on the current corrected and cropped images. "
                            + ("Phase map generation and calibration statistics follow; mass calibration will not start. "
                               if self._preparation_only else "Phase map generation and mass calibration follow automatically; ")
                            + "Keep DarSIA open.")
        self._actions()

    def _disconnect_training(self, context):
        try:
            context["studio"].training_completed.disconnect(context["completed"])
        except (RuntimeError, TypeError):
            pass
        if context.get("process") is not None:
            try:
                context["process"].finished.disconnect(context["finished"])
            except (RuntimeError, TypeError):
                pass
        if self._training_context is context:
            self._training_context = None
        self._actions()

    def _model_process_finished(self, context, code):
        # Studio emits training_completed only after inspecting actual model
        # artifacts. Exit 0 alone is not enough to advance the workflow.
        if self._training_context is context:
            self._disconnect_training(context)
            self.status.setText("Model refitting did not produce a usable completed model. "
                                + context["studio"].status.text())

    def _model_trained(self, context, path):
        if self._training_context is not context or Path(path).resolve() != context["expected"]:
            return
        self._disconnect_training(context)
        if self.root != context["root"] or self._generation != context["generation"]:
            self.status.setText("The completed model was retained; calibration was not started after changing series.")
            return
        self._candidate_reference = str(context["expected"])
        self.status.setText("The new phase model is fitted. Preparing its phase maps and "
                            + ("colour paths..." if self._preparation_only else "mass calibration..."))
        self._continue_with_model(context["jobs"], context["root"], context["series_id"])

    def _continue_with_model(self, jobs, root, series_id):
        generation = self._generation
        def continue_when_idle():
            if self.root != root or generation != self._generation:
                return
            if self._thread is not None:
                QTimer.singleShot(25, continue_when_idle)
                return
            if self._published_candidate == self._candidate_reference:
                self._prepare_phase(jobs, root, series_id)
                return
            reference = self._candidate_reference
            chosen = deepcopy(self.selection)
            self.status.setText("Checking the completed model before recording its new version…")
            self._call(lambda: require_current_model_basis(root, chosen, model_reference=reference),
                lambda _binding: self._publish_model(jobs, root, series_id, reference, generation))
        QTimer.singleShot(0, continue_when_idle)

    def _publish_model(self, jobs, root, series_id, reference, generation):
        context = self._model_publication_context
        if (self.root != root or self._generation != generation or not context or context[0] != root):
            self.status.setText("The completed model was retained; the series context changed before publication.")
            return
        publish = getattr(self.window_owner, "_publish_prepared_model", None)
        if not callable(publish):
            self.status.setText("The completed model was retained; the model-setting save owner is unavailable.")
            return
        self._model_publication_pending = True
        self.status.setText("Recording the completed model version; previous calibration results remain historical…")
        self._actions()
        def saved(success, error):
            self._model_publication_pending = False
            if not success or self.root != root or self._generation != generation:
                self.status.setText("The new model was retained, but phase preparation was not started: "
                                    + (error or "the series context changed."))
                self._actions()
                return
            self._published_candidate = reference
            self._model_publication_context = (root,
                getattr(self.window_owner, "_method_manifest_bytes", None),
                getattr(self.window_owner, "_saved_method", None))
            self.status.setText("✓ New model version recorded. Preparing phase map generation…")
            self._continue_with_model(jobs, root, series_id)
        try:
            publish(root, reference, expected_bytes=context[1], expected_selection=context[2], finished=saved)
        except Exception as error:
            saved(False, str(error))

    def _prepared(self, jobs, result, root, series_id):
        snapshot = result["snapshot"]
        if result["registration"] is not None and not jobs._job_created(result["registration"]):
            raise ValueError("The phase-map generation job could not be registered in Jobs.")
        if self.root != root or jobs.series_id != series_id:
            self.status.setText("The previous series' prepared job was retained in Jobs; it was not started after the selection changed.")
            return
        self._selected(result["selection"])
        runs = list(snapshot["job_facts"]["expected_runs"])
        if self._preparation_only:
            if not snapshot["job_facts"].get("fresh_optical_paths"):
                raise ValueError("This prepared job does not produce new colour paths for the selected image basis.")
            self._preparation_job = (root, series_id, str(snapshot["root"]))
            if self._watched_jobs is not jobs:
                jobs.panel.job_selection_changed.connect(lambda: self._preparation_progress(jobs))
                self._watched_jobs = jobs
        if snapshot.get("complete") or snapshot.get("workers"):
            if self._preparation_only:
                self._preparation_progress(jobs, snapshot)
            else:
                self.panel.start_after_phase_one(snapshot["root"], runs)
                self.status.setText("Using matching generated phase maps and statistics for this calibration.")
            return
        jobs.registry.set_active(jobs.series_root, jobs.series_id, snapshot["root"])
        jobs.panel._show_snapshot(snapshot)
        shared = getattr(jobs, "shared_jobs", None)
        if shared is None:
            raise ValueError("The phase-map generation job is saved. Open Shared Jobs to start its workers.")
        token = (self._generation, root, series_id, snapshot["root"])
        self._phase_start_pending = token
        self._actions()
        self.status.setText("Starting phase map generation through the selected Shared Jobs worker capacity…")
        def start_when_idle():
            if self._phase_start_pending != token:
                return
            if (self.root != root or jobs.series_id != series_id or self._generation != token[0]
                    or str((jobs.panel.current or {}).get("root")) != str(snapshot["root"])):
                self._phase_start_pending = None
                self.status.setText("The phase-map generation job was retained; its selection changed before worker start.")
                self._actions()
                return
            if shared._thread is not None:
                QTimer.singleShot(50, self, start_when_idle)
                return
            self._phase_start_pending = None
            try:
                shared.start_selected_job()
                if shared._thread is None:
                    self.status.setText("Phase map generation was not started. " + shared.status.text())
                elif self._preparation_only:
                    self.status.setText("Phase map generation and calibration statistics were requested in Shared Jobs. Follow their progress in Jobs. Mass calibration will not start.")
                elif self.panel.start_after_phase_one(snapshot["root"], runs):
                    self.status.setText("Phase map generation was requested in Shared Jobs. The same master and selected worker capacity are used; mass calibration follows completed results.")
                else:
                    self.status.setText("Phase map generation was requested in Shared Jobs; mass calibration is already busy.")
            except Exception as error:
                self.status.setText("The phase-map generation job was saved, but its start or mass-calibration handoff failed: " + str(error))
            self._actions()
        start_when_idle()

    def _preparation_progress(self, jobs, snapshot=None):
        wanted = self._preparation_job
        if wanted is None or self.root != wanted[0] or jobs.series_id != wanted[1]:
            return
        snapshot = snapshot if snapshot is not None else jobs.panel.current
        if not snapshot or str(snapshot.get("root")) != wanted[2]:
            return
        complete = bool(snapshot.get("complete") and not snapshot.get("errors") and not snapshot.get("failed_runs"))
        self.setup_completion_changed.emit(complete)
        if snapshot.get("errors") or snapshot.get("failed_runs"):
            self.status.setText("Colour-path preparation has reported errors. Open the job in Jobs to inspect them.")
        elif complete:
            self.status.setText("✓ Setup complete — ready for calibration. Colour paths and phase inputs completed. Mass calibration has not been started by this action.")
        else:
            self.status.setText("Colour paths and phase inputs are prepared or running in Jobs. Mass calibration will not start.")

    def _call(self, call, ready, *, deliver_stale=False):
        from .jobs_panel import _CallWorker
        if self._thread is not None:
            return
        self._call_generation = self._generation
        self._ready_callback = ready
        self._deliver_stale = deliver_stale
        thread = QThread(self)
        worker = _CallWorker(call)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._ready)
        worker.failed.connect(self._failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._closed)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self._actions()
        thread.start()

    @Slot(object)
    def _ready(self, value):
        if self._call_generation == self._generation or self._deliver_stale:
            try:
                self._ready_callback(value)
            except Exception as error:
                self.status.setText(str(error))

    @Slot(str)
    def _failed(self, error):
        if self._call_generation == self._generation:
            self.status.setText(error.strip().splitlines()[-1])
            if self._saving_colour_count:
                self._saving_colour_count = False
                self.colour_status.setText("Colour-path selection was not saved: " + error.strip().splitlines()[-1])
            if UNBOUND_MODEL_REASON in error and not self._model_retraining_attempted:
                # The exact completed Build can expose a grid mismatch that
                # the lightweight initial model preview could not inspect.
                self.selection["model_basis_error"] = UNBOUND_MODEL_REASON
                self._candidate_reference = None
                self._retry_model_basis = True

    @Slot()
    def _closed(self):
        self._thread.wait()
        self._thread = self._worker = None
        self._actions()
        if getattr(self, "_start_after_selection", False):
            self._start_after_selection = False
            QTimer.singleShot(0, lambda: self.prepare_and_start(preparation_only=self._preparation_only, _continuing=True))
        elif self._retry_model_basis:
            self._retry_model_basis = False
            QTimer.singleShot(0, lambda: self.prepare_and_start(preparation_only=self._preparation_only, _continuing=True))

    def can_close(self):
        return self._thread is None and self._training_context is None and not self._model_publication_pending
