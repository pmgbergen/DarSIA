"""Independent DarSIA mass-analysis page for a selected frozen phase model.

This page selects frozen phase models and embeds the retained saved-result
browser. Opening it never fits a calibration, starts workers, or publishes mass output.
The native app shows it as Calibration review and as Results and export (N141).
"""

from __future__ import annotations

from contextvars import copy_context
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from darsia.fluidflower.calibration_review_runs import (
    CalibrationReviewRunList, SliderPlayer, insert_after, review_run_rows)
from darsia.fluidflower.model_display import friendly_model_name
from darsia.fluidflower.series_paths import configured_series, model_versions_folder
from darsia.fluidflower.phase_model_history import (
    _model_directories, inspect_phase_model_history_entry, metadata_only_catalog,
    ModelHistoryEntries, registration_context_available, REGISTRATION_ROOTS_UNAVAILABLE,
    run_with_registration_context,
)
from darsia.fluidflower.tools.phase_studio_series_ui import ModelMassLookup
from darsia.fluidflower.calibration_review_workspace import CalibrationReviewWorkspace, approval_blocker


@metadata_only_catalog
def _series_models(series: str, project_root: Path,
                   cancelled: Callable[[], bool] | None = None) -> list[dict]:
    """Show every recorded model in this series, including QA-fail candidates."""
    native_available = registration_context_available()
    root = Path(project_root).resolve()
    found: dict[str, dict] = {}
    # Scan only this series' model library and bounded result tree. The general
    # history helper also scans registered defaults and external review roots;
    # those paths repeated NAS work and delayed this selector. Users can add an
    # external model folder explicitly below.
    paths = [*_model_directories(model_versions_folder(root), max_depth=4),
             *(path for path in _model_directories(root / "results", max_depth=3)
               if path.name.startswith("phase-model-"))]
    candidates = []
    for path in paths:
        if cancelled is not None and cancelled():
            return []
        candidates.append(inspect_phase_model_history_entry(path))
    for entry in candidates:
        model_root = Path(str(entry["root"])).resolve()
        scope = str(entry.get("series_scope") or "").casefold().replace("_", "")
        datasets = {str(value).casefold().replace("_", "")
                    for value in (entry.get("datasets") or [])}
        belongs_here = model_root.is_relative_to(root) or series.replace("_", "") in datasets
        if scope and scope not in {"all", "shared", "crossseries", series.replace("_", "")}:
            belongs_here = False
        if belongs_here:
            found[str(model_root).casefold()] = entry
    return ModelHistoryEntries(
        sorted(found.values(), key=lambda row: (float(row.get("created_timestamp") or 0),
                                              str(row.get("model_id") or "")), reverse=True),
        diagnostics=(() if native_available else (REGISTRATION_ROOTS_UNAVAILABLE,)))


class _ModelChoicesLookup(QThread):
    ready = Signal(object, str)

    def __init__(self, series: str, project_root: Path, parent=None):
        super().__init__(parent)
        self.series = series
        self.project_root = Path(project_root)
        # Match Studio catalogue refresh: capture the verified owner context,
        # including explicit absence/error, before entering a Qt worker thread.
        self._catalogue_context = copy_context()
        self._registration_error = None
        try:
            self._registration_available = registration_context_available()
        except Exception as error:
            self._registration_available, self._registration_error = None, error

    def run(self):
        try:
            if self._registration_error is not None:
                raise self._registration_error
            rows = self._catalogue_context.run(
                run_with_registration_context,
                lambda: _series_models(self.series, self.project_root, self.isInterruptionRequested),
                available=self._registration_available,
            )
            if not self.isInterruptionRequested():
                self.ready.emit(rows, "")
        except Exception as error:
            if not self.isInterruptionRequested():
                self.ready.emit(None, str(error))


class MassAnalysisWorkspace(QWidget):
    """Choose a model and saved native fit, then coordinate a frozen campaign.

    Two views share the page (N141): "review" (Calibration review) shows the
    viewer; "results" (Results and export) adds the preparation section, the
    export buttons and any Start analysis action. The page is the one scroll
    area: the viewer's tabs fill the remaining height and their figures rescale
    without scroll areas of their own. The page scrolls only where the
    retained controls are wider or taller than the window. In Calibration
    review a list of every run of the series sits beside that scroll area.
    """

    VIEW_TITLES = {"review": "Calibration review", "results": "Results and export"}

    series_selected = Signal(str)

    def __init__(self, parent=None, *, series: str, project_root: str | Path):
        super().__init__(parent)
        self.series = str(series).casefold()
        self.project_root = Path(project_root)
        self.series_roots = {
            key: Path(root) for key, _label, root in configured_series()
        }
        self.series_roots[self.series] = self.project_root
        self.phase_catalog = None
        self._generation = 0
        self.view, self._closing = None, False
        self._native_approval_request = self._native_approval_binding = None
        self._native_approval_error = ""
        from darsia.fluidflower.calibration_review_native_approval_ui import NativeApprovalLookup
        self._native_approval_lookup = NativeApprovalLookup(self)
        self._native_approval_lookup.ready.connect(self._native_approval_ready)
        self._model_lookup = None
        self._catalog_lookup = None
        self._background_lookups: list[QThread] = []

        outer = QHBoxLayout(self)
        # Calibration review's run list (N141 spec 2.5): every run, beside the
        # one scroll area rather than inside it.
        self.run_list = CalibrationReviewRunList(self)
        self.run_list.run_selected.connect(self._view_listed_run)
        self.run_list.hide()
        self._run_rows = None
        outer.addWidget(self.run_list)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)
        content = QWidget(self.scroll)
        self.scroll.setWidget(content)
        layout = QVBoxLayout(content)
        self.title = title = QLabel("Mass analysis · frozen model and calibration")
        font = title.font(); font.setBold(True); title.setFont(font)
        layout.addWidget(title)
        self.introduction = note = QLabel("Choose a phase model, saved native calibration and experiments here. "
                      "Image exclusions can come from a source-bound TOML selection receipt below. "
                      "Measured overflow can be compared with gas mass without refitting. "
                      "Opening this page never starts a calculation; validation-only models "
                      "remain ineligible for production mass analysis.")
        note.setWordWrap(True); layout.addWidget(note)
        series_row = QHBoxLayout()
        self.series_selector = QComboBox()
        for key, label, _root in configured_series():
            self.series_selector.addItem(label, key)
        if self.series_selector.findData(self.series) < 0:
            self.series_selector.addItem(self.series.upper(), self.series)
        self.series_selector.setCurrentIndex(self.series_selector.findData(self.series))
        self.series_selector.currentIndexChanged.connect(self._series_selection_changed)
        series_row.addWidget(QLabel("Experimental series"))
        series_row.addWidget(self.series_selector, 1)
        layout.addLayout(series_row)
        self.model_controls = QWidget()
        row = QHBoxLayout(self.model_controls)
        row.setContentsMargins(0, 0, 0, 0)
        self.model = QComboBox(); self.model.setMinimumWidth(440)
        self.model.currentIndexChanged.connect(self._selected_model_changed)
        row.addWidget(QLabel("Phase model")); row.addWidget(self.model, 1)
        self.refresh_button = QPushButton("Refresh models")
        self.refresh_button.clicked.connect(self.refresh_models)
        row.addWidget(self.refresh_button)
        self.browse_button = QPushButton("Add model folder…")
        self.browse_button.clicked.connect(self._browse_model)
        row.addWidget(self.browse_button)
        layout.addWidget(self.model_controls)
        self.status = QLabel("Loading model versions…")
        self.status.setWordWrap(True); layout.addWidget(self.status)
        # Approve run (N141 spec 2.5, Part 2), shown in Calibration review only.
        self.approval = QWidget()
        approval = QHBoxLayout(self.approval)
        approval.setContentsMargins(0, 0, 0, 0)
        self.approval_label = QLabel()
        self.approval_label.setWordWrap(True)
        approval.addWidget(self.approval_label, 1)
        self.approve_button = QPushButton("Approve run")
        self.approve_button.setToolTip(
            "Records the series' reviewer setting (if any), the time and the calibration for "
            "this run in the series folder. It never changes a result.")
        self.approve_button.clicked.connect(self.approve_selected_run)
        approval.addWidget(self.approve_button)
        layout.addWidget(self.approval)
        self.approval.hide()
        # The viewed run's pressure and temperature source, Calibration review only.
        self.pressure_line = QLabel()
        self.pressure_line.setWordWrap(True)
        self.pressure_line.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.pressure_line)
        self.pressure_line.hide()
        self._pressure_choices = None
        # Build published dataset (N169), shown in Results and export only.
        self.publish_button = QPushButton("Build published dataset…")
        self.publish_button.setToolTip(
            "Package this series and a finished production job as a published dataset record. "
            "It checks first and writes only into a new or empty folder outside the series.")
        self.publish_button.clicked.connect(self.build_published_dataset)
        self.publish_dialog = None
        layout.addWidget(self.publish_button)
        self.publish_button.hide()

        self.saved_results = CalibrationReviewWorkspace(self, selected_model=self._selected_model,
                                                       review_series_root=self.project_root)
        # Put the calibration choice before approval and image controls.
        layout.insertWidget(layout.indexOf(self.model_controls), self.saved_results.review_header)
        self.saved_results.series.setCurrentIndex(
            self.saved_results.series.findData(self.series))
        layout.addWidget(self.saved_results, 1)
        self.saved_results.view_experiment.currentIndexChanged.connect(self.refresh_approval)
        self.saved_results.review_selection_changed.connect(self.refresh_approval)
        self._follow_viewed_run()
        self.view = None
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_lookups)
        self.refresh_models()

    def _follow_viewed_run(self) -> None:
        """Add Play beside the image slider and keep the run list on the viewed run."""
        saved, experiments = self.saved_results, self.saved_results.view_experiment
        self.play_button = QPushButton("Play")
        self.play_button.setToolTip("Step through this result's images; stops at the last one.")
        # Beside "Previous", so it stays in view when the wide viewer row scrolls.
        insert_after(saved.previous_image, self.play_button)
        self.player = SliderPlayer(saved.slider, self.play_button, parent=self)
        experiments.currentIndexChanged.connect(self.player.stop)
        experiments.currentIndexChanged.connect(self._sync_run_list)
        model = experiments.model()
        for signal in (model.rowsInserted, model.rowsRemoved, model.modelReset):
            signal.connect(self._sync_run_list)
        load_result = saved.load_result

        def load_result_and_follow(path):
            load_result(path)
            self.refresh_approval()  # A different fit can retain the same run.
            self._sync_run_list()

        saved.load_result = load_result_and_follow

    def _sync_run_list(self, *_args) -> None:
        """Mark the viewed run, and which runs have a saved result, in the run list."""
        experiments = self.saved_results.view_experiment
        self.run_list.set_available(
            [experiments.itemData(index) for index in range(experiments.count())])
        self.run_list.select_run(experiments.currentData())

    def _view_listed_run(self, run: str) -> None:
        """View the run chosen in the run list through the Experiment selector."""
        experiments = self.saved_results.view_experiment
        for index in range(experiments.count()):
            if str(experiments.itemData(index)).casefold() == str(run).casefold():
                experiments.setCurrentIndex(index)
                return

    def _refresh_run_list(self) -> None:
        """Every run with its role and review status; read once per display."""
        if self._run_rows is None:
            self._run_rows = review_run_rows(self.project_root)
            self.run_list.set_rows(self._run_rows)
        self._sync_run_list()

    def show_view(self, view: str) -> None:
        """Show the page as Calibration review ("review") or Results and export ("results")."""
        if view not in self.VIEW_TITLES:
            raise ValueError(f"Unknown mass-analysis view: {view}")
        # Returning from the series-choice page can change its lock or source
        # without replacing this saved reader. Always qualify a fresh visit.
        self._invalidate_native_approval()
        self.view = view
        self.title.setText(self.VIEW_TITLES[view])
        self.run_list.setVisible(view == "review")
        self._run_rows = None
        results = view == "results"
        saved = self.saved_results
        self.introduction.setVisible(results)
        self.model_controls.setVisible(results)
        self.status.setVisible(results)
        saved.set_review_mode(not results)
        for widget in (saved.setup_toggle, saved.export_toggle,
                       getattr(self, "native_start_analysis_button", None), self.publish_button):
            if widget is not None:
                widget.setVisible(results)
        saved.setup_body.setVisible(results and saved.setup_toggle.isChecked())
        if results:
            saved.export_toggle.setChecked(True)
        saved.export_body.setVisible(results and saved.export_toggle.isChecked())
        self._pressure_choices = None
        self.refresh_approval()

    def build_published_dataset(self) -> None:
        """Open Build published dataset for this series (one dialog at a time)."""
        from darsia.fluidflower.published_dataset.build_dialog import BuildDatasetDialog

        previous = self.publish_dialog
        if previous is None or previous.series_root != self.project_root:
            if previous is not None and previous.busy():
                previous.raise_()
                return
            if previous is not None:
                previous.close()
                previous.deleteLater()
            self.publish_dialog = BuildDatasetDialog(self.project_root, self)
        self.publish_dialog.show()
        self.publish_dialog.raise_()

    def _approval_calibration(self):
        """The mass calibration series.toml references: its path and SHA-256."""
        from darsia.fluidflower.calibration_approvals import calibration_identity
        from darsia.fluidflower.series_method import read_series_method

        choice = read_series_method(self.project_root)
        if choice.error:
            raise ValueError(choice.error)
        return calibration_identity(self.project_root, choice.mass_calibration)

    def _show_pressure_line(self) -> None:
        """One line: the viewed run's configured pressure and temperature source."""
        from darsia.fluidflower.pressure_source import (
            calibration_pressure, pressure_line, run_configuration_choices, run_pressure_source)
        from darsia.fluidflower.series_method import read_series_method

        self.pressure_line.show()
        run = self.saved_results.view_experiment.currentData()
        if not run:
            self.pressure_line.setText(pressure_line(None))
            return
        if self._pressure_choices is None:
            # The run files once per series display; each run then reads its own two files.
            choices = {str(name).casefold(): (str(name), path, note)
                       for name, path, note in run_configuration_choices(self.project_root)}
            fit = calibration_pressure(
                "phase_model", read_series_method(self.project_root).mass_calibration,
                self.project_root)
            self._pressure_choices = choices, fit
        choices, fit = self._pressure_choices
        entry = choices.get(str(run).casefold())
        row = None if entry is None else run_pressure_source(
            entry[0], entry[1], note=entry[2], fit=fit)
        self.pressure_line.setText(pressure_line(row, run))

    def refresh_approval(self, *_args) -> None:
        """Show whether the viewed run is approved for the current calibration."""
        if self.view != "review":
            self._invalidate_native_approval()
            self.approval.hide()
            self.pressure_line.hide()
            return
        from darsia.fluidflower.calibration_approvals import read_run_approvals, run_approval

        self._show_pressure_line()
        self._refresh_run_list()
        self.approval.show()
        run = self.saved_results.view_experiment.currentData()
        self.approve_button.setText(f"Approve {str(run).upper()}" if run else "Approve run")
        self.approve_button.setEnabled(False)
        if self.saved_results._native_review is not None:
            self._refresh_native_approval(run)
            return
        self._invalidate_native_approval()
        try:
            calibration = self._approval_calibration()
            records = read_run_approvals(self.project_root)
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.approval_label.setText(f"Approval unavailable: {error}")
            return
        sha = calibration["sha256"]
        blocker = approval_blocker(self.saved_results.result, self.saved_results.result_path,
                                   run, self.project_root, calibration, series_id=self.series,
                                   native_review=self.saved_results._native_review)
        approved = {str(record.get("run", "")).casefold() for record in records
                    if (record.get("calibration") or {}).get("sha256") == sha}
        label = f"calibration {calibration['reference']} (SHA-256 {sha[:12]}…)"
        if not run:
            self.approval_label.setText(
                f"Choose a run to approve. {len(approved)} run(s) approved for {label}.")
            return
        record = run_approval(records, run, sha)
        if record is not None:
            by = f" by {record['reviewer']}" if record.get("reviewer") else ""
            text = f"{str(run).upper()}: approved{by} at {record.get('recorded_utc')} for {label}."
        else:
            text = f"{str(run).upper()}: not approved for {label}."
            self.approve_button.setEnabled(not blocker)
        self.approval_label.setText(
            text + f" {len(approved)} run(s) approved for this calibration. Approval never changes a result."
            + ("\n" + blocker if blocker else ""))

    def _native_approval_context(self):
        return (str(self.project_root.resolve()), self.series,
                self.saved_results.view_experiment.currentData(),
                self.saved_results._native_review, str(self.saved_results.result_path))

    def _invalidate_native_approval(self):
        if self._native_approval_request is not None:
            self._native_approval_lookup.invalidate()
        self._native_approval_request = self._native_approval_binding = None
        self._native_approval_error = ""

    def _refresh_native_approval(self, run):
        from darsia.fluidflower.calibration_approvals import read_run_approvals, run_approval

        registration = self.saved_results.native_registration()
        if registration is None:
            self._invalidate_native_approval()
            self.approval_label.setText("Approval unavailable: this native result has no registered source binding.")
            return
        context = self._native_approval_context()
        if context != self._native_approval_request:
            self._native_approval_request, self._native_approval_binding = context, None
            self._native_approval_error = ""
            self._native_approval_lookup.request(context, series_root=self.project_root,
                series_id=self.series, run=run, native_review=self.saved_results._native_review,
                registration=registration, source_registrations=self.saved_results.native_source_registrations())
        if self._native_approval_error:
            self.approval_label.setText("Approval unavailable: " + self._native_approval_error)
            return
        binding = self._native_approval_binding
        if binding is None:
            self.approval_label.setText("Checking this saved result against the current series choice and original source…")
            return
        try:
            calibration = binding["calibration"]
            record = run_approval(read_run_approvals(self.project_root), run, calibration["sha256"],
                                  native_evidence=binding["native_evidence"])
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.approval_label.setText(f"Approval unavailable: {error}")
            return
        label = f"calibration {calibration['reference']} (SHA-256 {calibration['sha256'][:12]}…)"
        if record is None:
            self.approve_button.setEnabled(True)
            text = f"{str(run).upper()}: not approved for {label}."
        else:
            by = f" by {record['reviewer']}" if record.get("reviewer") else ""
            text = f"{str(run).upper()}: approved{by} at {record.get('recorded_utc')} for {label}."
        self.approval_label.setText(text + " Approval applies to this exact saved result and source; it never changes a result.")

    def _native_approval_ready(self, context, binding, error):
        if (self._closing or self.view != "review" or context != self._native_approval_request
                or context != self._native_approval_context()):
            return
        self._native_approval_binding, self._native_approval_error = binding, error
        self.refresh_approval()

    def _approve_native_selected_run(self, run):
        from darsia.fluidflower.calibration_review_native_approval import approve_native_run
        from darsia.fluidflower.reviewer_preferences import load_reviewer_name

        context, binding = self._native_approval_request, self._native_approval_binding
        def current_action():
            return (not self._closing and self.view == "review" and context is not None
                    and context == self._native_approval_context()
                    and self._native_approval_binding is binding)
        if binding is None or self._native_approval_error or not current_action():
            raise ValueError("Wait for the current native result and source check before approving.")
        return approve_native_run(self.project_root, self.series, run,
            self.saved_results._native_review, self.saved_results.native_registration(),
            expected_binding=binding, reviewer=load_reviewer_name(self.project_root),
            action_guard=current_action, source_registrations=self.saved_results.native_source_registrations())

    def approve_selected_run(self) -> None:
        """Append one approval record for the viewed run; results stay as they are."""
        from darsia.fluidflower.calibration_approvals import approve_run
        from darsia.fluidflower.reviewer_preferences import load_reviewer_name

        run = self.saved_results.view_experiment.currentData()
        if not run or self.view != "review":
            return
        try:
            if self.saved_results._native_review is not None:
                self._approve_native_selected_run(run)
                self._run_rows = None
                self.refresh_approval()
                return
            calibration = self._approval_calibration()
            blocker = approval_blocker(self.saved_results.result, self.saved_results.result_path,
                                       run, self.project_root, calibration, series_id=self.series,
                                       native_review=self.saved_results._native_review)
            if blocker:
                raise ValueError(blocker)
            # The reviewer setting names the reviewer (N175); never the OS login.
            approve_run(self.project_root, run, calibration,
                        result=self.saved_results.result_path,
                        reviewer=load_reviewer_name(self.project_root))
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.approval_label.setText(f"The approval was not recorded: {error}")
            return
        self._run_rows = None
        self.refresh_approval()

    def set_series(self, series: str, project_root: str | Path) -> None:
        if self._closing:
            return
        self.series = str(series).casefold()
        self.project_root = Path(project_root)
        self.series_roots[self.series] = self.project_root
        index = self.series_selector.findData(self.series)
        if index < 0:
            self.series_selector.addItem(self.series.upper(), self.series)
            index = self.series_selector.findData(self.series)
        self.series_selector.blockSignals(True)
        self.series_selector.setCurrentIndex(index)
        self.series_selector.blockSignals(False)
        self._generation += 1
        self.phase_catalog = None
        self.saved_results.phase_catalog = None
        self.saved_results.set_native_series_root(self.project_root)
        self.saved_results.series.setCurrentIndex(
            self.saved_results.series.findData(self.series))
        self.refresh_models()
        self._pressure_choices = self._run_rows = None
        self.refresh_approval()

    def _series_selection_changed(self, _index: int) -> None:
        if self._closing:
            return
        series = self.series_selector.currentData()
        root = self.series_roots.get(series)
        if not series or root is None:
            return
        self.set_series(series, root)
        self.series_selected.emit(series)

    def refresh_models(self) -> None:
        if self._closing:
            return
        self._generation += 1
        generation = self._generation
        previous = self.model.currentData()
        self.status.setText("Reading model versions for this series…")
        self.refresh_button.setEnabled(False)
        if self._model_lookup is not None and self._model_lookup.isRunning():
            self._model_lookup.requestInterruption()
        lookup = _ModelChoicesLookup(self.series, self.project_root, self)
        self._model_lookup = lookup
        self._background_lookups.append(lookup)
        lookup.ready.connect(lambda rows, error: self._models_ready(
            generation, previous, rows, error))
        lookup.finished.connect(self._lookup_finished, Qt.ConnectionType.QueuedConnection)
        lookup.start()

    @Slot()
    def _lookup_finished(self) -> None:
        lookup = self.sender()
        if lookup is None:
            return
        if not self._closing and lookup is self._model_lookup:
            self.refresh_button.setEnabled(True)
        self._forget_lookup(lookup)

    def _forget_lookup(self, lookup: QThread) -> None:
        if lookup not in self._background_lookups:
            return
        self._background_lookups.remove(lookup)
        if self._model_lookup is lookup:
            self._model_lookup = None
        if self._catalog_lookup is lookup:
            self._catalog_lookup = None
        lookup.deleteLater()

    def _stop_lookups(self) -> None:
        # The selected NAS may still be loading when the user closes DarSIA.
        # End Qt threads before their owning widget is destroyed.
        self._closing = True
        self._generation += 1
        lookups, self._background_lookups = self._background_lookups, []
        self._model_lookup = self._catalog_lookup = None
        self._native_approval_lookup.stop()
        self.saved_results.stop_review_reads()
        for lookup in lookups:
            lookup.requestInterruption()
        for lookup in lookups:
            lookup.wait()
            lookup.deleteLater()

    def closeEvent(self, event) -> None:
        self._stop_lookups()
        super().closeEvent(event)

    @Slot(int, object, object, str)
    def _models_ready(self, generation: int, previous, rows, error: str) -> None:
        if getattr(self, "_closing", False) or generation != self._generation:
            return
        if error:
            self.status.setText("Model discovery failed: " + error)
            return
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem("Choose a phase model…", None)
        for entry in rows:
            name = friendly_model_name(entry) or str(entry.get("model_id") or Path(entry["root"]).name)
            label = f"{name} · {entry.get('state_label') or 'Status unknown'} · {entry.get('model_id') or Path(entry['root']).name}"
            self.model.addItem(label, str(entry["root"]))
            self.model.setItemData(self.model.count()-1, str(entry["root"]),
                                   role=3)  # Qt.ToolTipRole
        index = self.model.findData(previous) if previous else -1
        self.model.setCurrentIndex(max(index, 0))
        self.model.blockSignals(False)
        diagnostics = tuple(getattr(rows, "diagnostics", ()))
        self.status.setText(f"{len(rows)} recorded model versions. QA warnings remain visible; choose one to load its fixed maps."
                           + ("\n" + "\n".join(diagnostics) if diagnostics else ""))
        if index > 0:
            self._selected_model_changed()

    def _browse_model(self) -> None:
        if self._closing:
            return
        path = QFileDialog.getExistingDirectory(self, "Phase-model folder", str(model_versions_folder(self.project_root)))
        if self._closing:
            return
        if path:
            index = self.model.findData(path)
            if index < 0:
                entry = inspect_phase_model_history_entry(Path(path))
                self.model.addItem(
                    f"{friendly_model_name(entry) or entry['model_id']} · {entry.get('state_label')} · {entry['model_id']}",
                    str(Path(path).resolve()))
                index = self.model.count()-1
            self.model.setCurrentIndex(index)

    def _selected_model_changed(self, _index: int = -1) -> None:
        if self._closing:
            return
        path = self.model.currentData()
        self._generation += 1
        generation = self._generation
        self.phase_catalog = None
        self.saved_results.phase_catalog = None
        if not path:
            return
        self.status.setText("Checking fixed phase maps and registered mass inputs for " + Path(path).name + "…")
        lookup = ModelMassLookup(Path(path), self)
        self._catalog_lookup = lookup
        self._background_lookups.append(lookup)
        lookup.ready.connect(lambda catalog, error: self._catalog_ready(generation, catalog, error))
        lookup.finished.connect(self._lookup_finished, Qt.ConnectionType.QueuedConnection)
        lookup.start()

    @Slot(int, object, str)
    def _catalog_ready(self, generation: int, catalog, error: str) -> None:
        if getattr(self, "_closing", False) or generation != self._generation:
            return
        if error or catalog is None:
            self.status.setText("Selected model is not ready for native mass analysis: " + error)
            return
        if catalog["dataset"].casefold() != self.series:
            self.status.setText("Selected model belongs to another experimental series")
            return
        self.phase_catalog = catalog
        self.saved_results.model_results_ready(catalog, "")
        ready = sum(bool(job.get("concentration", {}).get("ready")) for job in catalog["jobs"])
        self.status.setText(f"{catalog['model_id']} · {ready}/{len(catalog['jobs'])} experiments have registered native inputs. "
                            "Inspect saved concentration and mass results below.")

    def _selected_model(self) -> dict | None:
        """Return the selected saved model for the retained result browser."""
        root = self.model.currentData()
        return {"root": root} if root else None
