"""Evidence-review queues within the Studio review page."""

from darsia.fluidflower.series_paths import tool_arguments, local_cache_root
from datetime import UTC, datetime
from pathlib import Path
import uuid

from PySide6.QtCore import QObject, QProcess, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                              QMessageBox, QPushButton, QVBoxLayout, QWidget)

from darsia.fluidflower.execution import default_execution_cache_root
from darsia.fluidflower.phase_evidence_audit import discover_queues, load_queue, queue_directory
from darsia.fluidflower.phase_workflow import (build_phase_review_command,
    build_phase_review_resume_command, default_phase_review_roots,
    discover_phase_review_sessions, phase_python_executable)
from .common import FILE_DIALOG_OPTIONS


class _InventorySignals(QObject):
    finished = Signal(object, object, str)


class _InventoryTask(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = _InventorySignals()

    def run(self):
        try:
            queues, sessions = discover_queues(), discover_phase_review_sessions()
        except Exception as exc:
            self.signals.finished.emit([], [], f'{type(exc).__name__}: {exc}')
        else:
            self.signals.finished.emit(queues, sessions, '')


class EvidenceAuditLibrary(QWidget):
    def __init__(self, owner):
        super().__init__()
        self.owner, self.process = owner, None
        self.operation = None
        self.pending_check = None
        self.worker_output = ""
        self.inventory, self.sessions = [], []
        self._inventory_loaded = False
        self._inventory_task = None
        self._inventory_selected = None
        self._inventory_refresh_again = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Evidence disagreements"))
        self.model_details = QLabel()
        self.model_details.setWordWrap(True)
        layout.addWidget(self.model_details)
        self.queues = QComboBox()
        self.queues.setMinimumContentsLength(35)
        self.queues.setToolTip("Each queue records its own evaluated model, original labels and image timeline.")
        self.queues.currentIndexChanged.connect(self.selection_changed)
        layout.addWidget(self.queues)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        self.start = QPushButton("Start new disagreement review")
        self.resume = QPushButton("Resume latest disagreement review")
        refresh = QPushButton("Refresh queues")
        self.start.clicked.connect(self.start_review)
        self.resume.clicked.connect(self.resume_review)
        refresh.clicked.connect(lambda: self.refresh())
        for button in (self.start, self.resume, refresh):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.import_button = QPushButton("Import evaluation report...")
        self.import_button.setToolTip("Build a ranked queue from a completed region-evaluation JSON report and its frozen training evidence.")
        self.import_button.clicked.connect(self.import_report)
        layout.addWidget(self.import_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)
        owner.reviewer.textChanged.connect(self.selection_changed)
        owner.validation_model.currentIndexChanged.connect(self.update_check_state)
        self.selection_changed()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._inventory_loaded and self._inventory_task is None and self.process is None:
            self.refresh()

    def refresh(self, selected_path=None):
        self._inventory_selected = selected_path or self.queues.currentData()
        if self._inventory_task is not None:
            self._inventory_refresh_again = True
            return
        task = _InventoryTask()
        task.signals.finished.connect(self._inventory_finished)
        self._inventory_task = task
        self.details.setText('Loading saved disagreement queues…')
        QThreadPool.globalInstance().start(task)

    def _inventory_finished(self, queues, sessions, error):
        self._inventory_task = None
        if self._inventory_refresh_again:
            self._inventory_refresh_again = False
            self.refresh(self._inventory_selected)
            return
        if error:
            self.details.setText('Could not load disagreement queues: ' + error)
            return
        self.inventory, self.sessions = queues, sessions
        self._inventory_loaded = True
        self.queues.blockSignals(True)
        self.queues.clear()
        for queue in self.inventory:
            self.queues.addItem(f"{queue['model_id']} | {len(queue['entries'])} initial regions | {queue['created_utc'][:10]}", queue["path"])
        self.queues.setCurrentIndex(max(0, self.queues.findData(self._inventory_selected)))
        self.queues.blockSignals(False)
        self.selection_changed()

    def selected(self):
        return next((q for q in self.inventory if q["path"] == self.queues.currentData()), None)

    def latest_session(self):
        path = self.queues.currentData()
        if not path:
            return None
        reviewer = self.owner.reviewer.text().strip().casefold()
        matches = [s for s in self.sessions if s.get("evidence_audit_path")
                   and Path(s["evidence_audit_path"]).resolve() == Path(path).resolve()
                   and str(s.get("reviewer", "")).casefold() == reviewer]
        return max(matches, key=lambda s: s.get("modified_timestamp", 0), default=None)

    def selection_changed(self, *_):
        self.update_check_state()
        queue = self.selected()
        ready = bool(queue and self.owner.reviewer.text().strip())
        self.start.setEnabled(ready)
        self.resume.setEnabled(bool(self.latest_session()))
        if not queue:
            self.details.setText("No registered disagreement queue")
            return
        hard = sum("temporal_rule_conflict" in r["reasons"] for r in queue["entries"])
        runs = ", ".join(sorted({r["run"].upper() for r in queue["entries"]}))
        self.details.setText(f"Model: {queue['model_id']}\nExperiments: {runs}\n"
            f"Temporal-rule conflicts: {hard}; initially selected regions: {len(queue['entries'])}\n"
            f"Experimental rule: {queue.get('experimental_rule') or 'None'}\n"
            + (self.coverage_summary(queue) + "\n" if queue.get("coverage") else "") +
            "Original training evidence: locked. Re-review decisions: separate local session.")
        self.start.setToolTip("The initial selection is not a limit. Use Load more disagreements in the review window to fetch further ranked regions.")

    def update_check_state(self, *_):
        model = self.owner._selected_validation_model()
        ready = bool(model and not model.get("validation_collection_error")
                     and self.owner.reviewer.text().strip()
                     and self.owner.review_save_root.text().strip() and self.process is None)
        self.model_details.setText(
            "Check model: " + self.owner._validation_model_label(model) if model else
            "Select a model in Review and label evidence to check its saved regions and validations.")

    @staticmethod
    def coverage_summary(queue):
        coverage = queue.get("coverage") or {}
        if not coverage:
            return ""
        summary = ("Checked" if coverage.get("complete") else "Partial check:") + (
            f" {coverage.get('scored_regions', 0)} of {coverage.get('total_effective_regions', 0)} regions")
        if coverage.get("missing_maps"):
            summary += f"; {coverage['missing_maps']} missing frame maps"
        if coverage.get("unmapped_runs"):
            summary += "; unmapped experiments: " + ", ".join(str(run).upper() for run in coverage["unmapped_runs"])
        skipped = coverage.get("skipped_reason_counts") or {}
        if skipped.get("annotation_has_no_registered_domain_pixels"):
            summary += f"; {skipped['annotation_has_no_registered_domain_pixels']} regions have no pixels inside the analysis area"
        remaining = sum(int(count) for reason, count in skipped.items() if reason not in {
            "no_saved_predictions_for_run", "saved_phase_map_missing", "annotation_has_no_registered_domain_pixels"})
        if remaining:
            summary += f"; {remaining} other regions could not be checked"
        conflicts = coverage.get("inherited_geometry_conflicts") or []
        if conflicts:
            runs = ", ".join(sorted({str(row["run"]).upper() for row in conflicts}))
            summary += (f"; {len(conflicts)} unresolved saved-label conflicts ({runs}). "
                        "Both annotations are preserved; resolve them in the original saved reviews")
        return summary





    def launch(self, command):
        started, pid = QProcess.startDetached(command[0], command[1:])
        if not started:
            raise RuntimeError("The native review window could not start")
        self.status.setText(f"Opening native disagreement review (process {pid})")

    def start_review(self, _checked=False, *, queue_path=None, launch_context=None):
        """Check the saved queue and every pinned source in the background, then open the review.

        The check reads and hashes every pinned phase map, which takes seconds
        to minutes on the shared drive.
        """
        from darsia.fluidflower.ui_metadata_preview import MetadataPreview
        try:
            queue_path = Path(queue_path or self.queues.currentData())
            context = launch_context or dict(reviewer=self.owner.reviewer.text().strip(),
                save_root=Path(self.owner.review_save_root.text()), use_gpu=self.owner.review_gpu.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open evidence review", str(exc))
            return
        reader = self.__dict__.get("_review_reader")
        if reader is None:
            reader = self._review_reader = MetadataPreview(self)
            reader.completed.connect(self._review_queue_checked)
        elif reader.busy:
            return
        self.start.setEnabled(False)
        self.status.setText("Checking the saved disagreement queue and its phase maps…")
        reader.start(lambda _cancelled: (queue_path, context, load_queue(queue_path, verify_sources=True)))

    def _review_queue_checked(self, value, error):
        """Open the review for a checked queue, or say why it cannot open."""
        self.status.setText("")
        self.selection_changed()
        if error or value is None:
            QMessageBox.critical(self, "Cannot open evidence review", error or "no answer was returned")
            return
        queue_path, context, queue = value
        try:
            if queue.get("automatic_review") and queue.get("phase_review_basis") not in {
                    "studio-optical-without-context-v1", "baseline-relative-optical-regions-v1"}:
                raise ValueError("This saved queue evaluated context-supported maps. Run Check disagreements again for the current optical review basis; the old queue and its labels are preserved.")
            binding = queue["binding"]
            command = build_phase_review_command(mode="open_correction",
                reviewer=context["reviewer"],
                save_root=context["save_root"],
                cache_root=default_execution_cache_root()/"phase-review",
                ffac_root=Path(binding["ffac_root"]), ffum_root=Path(binding["ffum_root"]),
                combined_root=Path(binding["combined_root"]),
                target_model_id=queue["model_id"], selected_candidates=binding["selected_candidates"],
                review_session_id=f"audit-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
                evidence_audit_path=queue_path, use_gpu=context["use_gpu"])
            self.launch(command)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open evidence review", str(exc))

    def resume_review(self):
        session = self.latest_session()
        if session is None:
            return
        try:
            self.launch(build_phase_review_resume_command(session,
                cache_root=default_execution_cache_root()/"phase-review",
                use_gpu=self.owner.review_gpu.isChecked(), read_only=bool(session.get("editing_locked"))))
        except Exception as exc:
            QMessageBox.critical(self, "Cannot resume evidence review", str(exc))

    def import_report(self):
        if self.process is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Completed phase-region evaluation", "",
            "JSON reports (*.json)", options=FILE_DIALOG_OPTIONS)
        if not path:
            return
        output = default_phase_review_roots()[0]/"evidence_audits"/f"audit-{uuid.uuid4().hex}.json"
        process = QProcess(self)
        process.setProgram(phase_python_executable())
        process.setArguments(["-m", "darsia.fluidflower.phase_evidence_audit",
                              "--evaluation", path, "--output", str(output)])
        process.finished.connect(self.import_finished)
        process.errorOccurred.connect(self.import_failed)
        self.process, self.operation = process, "import"
        self.import_button.setEnabled(False)
        self.update_check_state()
        self.status.setText("Building disagreement queue from frozen evidence...")
        process.start()

    def import_finished(self, code, _status):
        if self.process is None:
            return
        error = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        process, self.process, self.operation = self.process, None, None
        process.deleteLater()
        self.import_button.setEnabled(True)
        self.status.setText("Queue imported" if code == 0 else f"Import failed: {error[-1000:]}")
        self.refresh()

    def import_failed(self, error):
        if self.process is None:
            return
        self.status.setText(f"Import failed: {self.process.errorString()}")
        if error == QProcess.ProcessError.FailedToStart:
            self.process.deleteLater()
            self.process = None
            self.operation = None
            self.import_button.setEnabled(True)
            self.update_check_state()
