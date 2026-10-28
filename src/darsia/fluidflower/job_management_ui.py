"""Inline archive/restore and reviewed deletion, using the existing Jobs runner."""
from pathlib import Path
import os
from PySide6.QtCore import QTimer, Signal, Slot

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLabel, QLineEdit

from .job_management import UnfinishedClaimError, archive_job, deletion_plan, delete_job
from .job_management import stop_delete_review, stop_for_deletion, stop_delete_status


def _stopped_image_admission(current, image_rows, image_only):
    """Offer the original recovery review; do not establish claim ownership."""
    if (not image_only or current.get("state") != "Starting" or current.get("errors")
            or current.get("failed_runs") or not current.get("root")
            or image_rows[0].get("detail_state") != "Claiming (owner not published)"):
        return False
    from .job_control import read_control
    from .job_lifecycle import require_service_retirement
    try:
        root = Path(current["root"])
        if read_control(root)["action"] != "pause":
            return False
        require_service_retirement(root)
        return True
    except (OSError, ValueError, KeyError):
        return False


class JobManagementControls(QWidget):
    reprepare_message = Signal(str)
    reprepare_state_changed = Signal()

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.thread = self.worker = self.review = None
        self.stop_review = None
        self.retirement_timer = QTimer(self)
        self.retirement_timer.setInterval(3000)
        self.retirement_timer.timeout.connect(self._check_retirement)
        self._selected_root = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.show_archived = QCheckBox("Show archived jobs")
        self.show_archived.toggled.connect(lambda _checked: owner.refresh_registered())
        layout.addWidget(self.show_archived)
        actions = QHBoxLayout()
        self.archive = QPushButton("Archive job")
        self.restore = QPushButton("Restore job")
        self.delete = QPushButton("Delete job…")
        self.stop_delete = QPushButton("Stop and delete…")
        for button in (self.archive, self.restore, self.delete, self.stop_delete):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.reprepare = QPushButton("Update job")
        self.reprepare.setToolTip("Use the installed application version and saved settings. Keep completed work and old metadata in history. Press Start to continue.")
        self.reprepare.hide()
        self.reprepare.clicked.connect(self._reprepare_setup)
        layout.addWidget(self.reprepare)
        self._source_commit = None
        self._repreparing = False
        self._reprepare_root = None
        self.actor = QLineEdit(os.environ.get("DARSIA_OPERATOR_NAME", ""))
        self.actor.setPlaceholderText("Name for job actions")
        layout.addWidget(self.actor)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.archive_instead = QPushButton("Archive job (keep all files)")
        self.archive_instead.hide()
        layout.addWidget(self.archive_instead)
        self.confirmation = QWidget()
        review_layout = QVBoxLayout(self.confirmation)
        self.review_text = QLabel("")
        self.review_text.setWordWrap(True)
        review_layout.addWidget(self.review_text)
        self.destructive_ack = QCheckBox("I understand deletion is permanent and can break the listed dependent jobs.")
        review_layout.addWidget(self.destructive_ack)
        self.destructive_ack.toggled.connect(lambda checked: self.confirm_stop_delete.setEnabled(checked))
        row = QHBoxLayout()
        self.confirm_delete = QPushButton("Delete job folder permanently")
        self.confirm_stop_delete = QPushButton("Request Stop for listed jobs")
        self.check_retirement = QPushButton("Check original retirement")
        self.cancel_delete = QPushButton("Cancel")
        row.addWidget(self.confirm_delete)
        row.addWidget(self.confirm_stop_delete)
        row.addWidget(self.check_retirement)
        row.addWidget(self.cancel_delete)
        review_layout.addLayout(row)
        layout.addWidget(self.confirmation)
        self.confirmation.hide()
        self.archive.clicked.connect(lambda: self._archive(True))
        self.archive_instead.clicked.connect(lambda: self._archive(True))
        self.restore.clicked.connect(lambda: self._archive(False))
        self.delete.clicked.connect(self._review_delete)
        self.stop_delete.clicked.connect(self._review_stop_delete)
        self.confirm_stop_delete.clicked.connect(self._stop_for_deletion)
        self.check_retirement.clicked.connect(self._check_retirement)
        self.confirm_delete.clicked.connect(self._delete)
        self.cancel_delete.clicked.connect(self.cancel)
        owner.panel.job_selection_changed.connect(self.selection_changed)
        self.selection_changed()

    def can_close(self):
        return self.thread is None

    def prepare_close(self):
        self.retirement_timer.stop()

    def cancel_close(self):
        if self.stop_review is not None and self.thread is None:
            self.retirement_timer.start()

    def cancel(self):
        self.review = None
        self.stop_review = None
        self.retirement_timer.stop()
        self.confirmation.hide()

    def selection_changed(self):
        current = self.owner.panel.current or {}
        if current.get("root") != self._selected_root:
            self.cancel()
            self.archive_instead.hide()
        self._selected_root = current.get("root")
        enabled = bool(current.get("root") and self.owner.series_root and self.thread is None)
        archived = bool(current.get("archived"))
        self.archive.setEnabled(enabled and not archived)
        self.archive_instead.setEnabled(enabled and not archived)
        if archived:
            self.archive_instead.hide()
        self.restore.setEnabled(enabled and archived)
        self.delete.setEnabled(enabled)
        self.stop_delete.setEnabled(enabled)
        self.show_archived.setEnabled(self.thread is None)
        source = self.setup_source_status(current)
        self.reprepare.setText(source.get("action_label", "Update job"))
        self.reprepare.setVisible(source["can_reprepare"])
        self.reprepare.setEnabled(enabled and not self._repreparing)

    def source_commit(self):
        """Cache this installed checkout's identity without preparing any job."""
        if self._source_commit is None:
            import subprocess
            try:
                self._source_commit = subprocess.check_output(["git", "-C",
                    str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                    text=True, timeout=3).strip()
            except (OSError, subprocess.SubprocessError):
                self._source_commit = ""
        return self._source_commit

    def setup_source_status(self, current=None):
        """Display guidance only; the original recovery backend grants admission."""
        current = (self.owner.panel.current or {}) if current is None else current
        facts = current.get("job_facts") or {}
        installed = self.source_commit() if facts.get("pipeline") == "series-setup" else ""
        recorded = facts.get("commit") or ""
        mismatch = bool(installed and recorded and recorded != installed)
        failed_rows = [row for row in current.get("runs", []) if row.get("state") == "Failed"]
        early_failure = bool(facts.get("pipeline") == "series-setup"
            and current.get("failed_runs") == 1 and len(failed_rows) == 1
            and failed_rows[0].get("run") == "setup-apply"
            and failed_rows[0].get("operation") == "apply")
        complete_rows = [row for row in current.get("runs", [])
                         if row.get("state") in {"Complete", "Completed"}]
        early_build = bool(facts.get("pipeline") == "series-setup" and failed_rows
            and len(failed_rows) == current.get("failed_runs")
            and all(row.get("operation") == "build" for row in failed_rows)
            and any(row.get("run") == "setup-apply" and row.get("operation") == "apply"
                    for row in complete_rows)
            and all(row.get("operation") == "build" or row.get("run") == "setup-apply"
                    for row in complete_rows))
        completed_setup = bool(facts.get("pipeline") == "series-setup" and complete_rows
            and not failed_rows and not current.get("failed_runs")
            and all(row.get("operation") in {"build", "image-preparation"}
                    or row.get("run") == "setup-apply" and row.get("operation") == "apply"
                    for row in complete_rows))
        image_rows = [row for row in current.get("runs", []) if row.get("operation") == "image-preparation"]
        completed_images = bool(completed_setup and len(image_rows) == 1
            and image_rows[0] in complete_rows and len(complete_rows) == len(current.get("runs", [])))
        image_only = bool(len(image_rows) == 1 and complete_rows
            and any(row.get("operation") == "apply" for row in complete_rows)
            and len(complete_rows) + 1 == len(current.get("runs", []))
            and all(row.get("operation") in {"apply", "build"} for row in complete_rows))
        failed_image = bool(image_only and failed_rows == image_rows and current.get("failed_runs") == 1)
        stopped_admission = _stopped_image_admission(current, image_rows, image_only)
        recoverable = early_failure or early_build or failed_image or stopped_admission
        retired = (current.get("recovery_available") is True
                   and current.get("state") in {"Cancelled", "Paused"})
        work = bool(current.get("committed") or current.get("workers") and not (retired or stopped_admission)
            or not (early_build or completed_setup or image_only) and (current.get("completed_runs") or current.get("completed_operations"))
            or current.get("failed_runs") and not recoverable)
        allowed_rows = ([*image_rows, *complete_rows] if stopped_admission else
                        [*failed_rows, *complete_rows] if early_build or failed_image else complete_rows if completed_setup
                        else failed_rows if early_failure else [])
        work |= any(row.get("committed") or str(row.get("state", "")).startswith(
            ("Complete", "Failed", "Working", "Publishing", "Claimed", "Running"))
            for row in current.get("runs", []) if row not in allowed_rows)
        candidate = bool(facts.get("pipeline") == "series-setup" and not current.get("archived") and not work
            and (current.get("recovery_available") is not False
                 or completed_setup and len(complete_rows) == len(current.get("runs", []))
                    and current.get("state") in {"Complete", "Completed"}
                 or failed_image and not current.get("workers") and current.get("state") == "Failed"
                 or stopped_admission)
            and (current.get("state") in {"Pending", "Failed", "Cancelled", "Paused", "Waiting"}
                 or completed_setup and len(complete_rows) == len(current.get("runs", []))
                    and current.get("state") in {"Complete", "Completed"}
                 or stopped_admission))
        message = ""
        if mismatch:
            message = f"This Setup job uses code {recorded[:8]}; this app uses {installed[:8]}. "
            message += ("Reprepare this unstarted job with current code, then click Start. "
                        "Its folder and original approved inputs are kept." if candidate else
                        "Start is blocked. Work or unresolved worker activity is recorded, so reprepare is unavailable. "
                        "Continue with the job's recorded code version.")
        if stopped_admission and candidate:
            message = ("Image admission stopped before worker ownership was published. Update this job, then click Start. "
                       "The recovery review preserves completed Apply and Builds.")
        elif failed_image and candidate:
            message = ("Retry image correction in this job, then click Start. Completed geometry is retained; "
                       "the original failed attempt and partial image files are kept in job history.")
        elif recoverable and candidate:
            message = ("Update this failed Setup job, then click Start. The original failure and retired attempt are retained in history. "
                       "Saved Apply files are retained for reuse. Owned incomplete Build files are archived; completed Builds are preserved.")
        elif recoverable:
            message = "This failed Setup job still has unresolved worker activity or other saved work. Its history is retained; wait for retirement before updating."
        elif completed_images and candidate:
            message = ("Rebuild separate colour paths per run from the completed corrected image samples, then click Start. "
                       "The existing Apply, Builds, image corrections and historical outputs are retained.")
        elif completed_setup and candidate:
            message = ("Update the image correction in this job, then click Start. Completed Builds and saved Apply files are retained; "
                       "only the selected image correction is added, including Setup colour paths. Model training and mass calibration remain under Calibration.")
        return dict(mismatch=mismatch, blocked=mismatch or recoverable,
                    can_reprepare=candidate, message=message, commit=installed,
                    action_label="Rebuild colour paths per run" if completed_images else
                        "Retry image correction" if image_only else
                        "Update image correction" if completed_setup
                        and current.get("state") in {"Complete", "Completed"} else "Update job")

    def _reprepare_setup(self):
        try:
            context = self._context()
            source = self.setup_source_status()
            if not source["can_reprepare"]:
                raise ValueError(source["message"] or "This job does not need Setup source-version recovery.")
            actor = self.actor.text().strip() or self.owner.panel.local_machine
            from .setup_job_reprepare import reprepare_unstarted_setup
            def done(result):
                message = ("Completed image corrections are retained. Click Start to rebuild separate colour paths per run."
                           if result.get("reused_image_basis") else
                           "Completed geometry is retained. Click Start to run image correction with the saved choices."
                           if result.get("image_preparation_added") else
                           "This job already uses the current code and retained Apply. Click Start to run it."
                           if result.get("unchanged") else
                           "Setup reprepared in the same job folder; old metadata is retained in history. Click Start to run it.")
                self.note.setText(message)
                self.reprepare_message.emit(message)
                if (self.owner.panel.current or {}).get("root") == str(context[-1]):
                    self.owner.refresh_registered()
            self._repreparing = True
            self._reprepare_root = str(context[-1])
            self.reprepare_message.emit("Updating this Setup job from the current code and saved settings… No worker is being started.")
            self._run(lambda: reprepare_unstarted_setup(*context,
                machine=self.owner.panel.local_machine, actor=actor,
                stores_path=self.owner.stores.path), done)
            self.reprepare_state_changed.emit()
        except Exception as error:
            self.note.setText(str(error))
            self.reprepare_message.emit(str(error))
            if self.thread is None:
                self._repreparing = False
                self.reprepare_state_changed.emit()

    def _name(self):
        value = self.actor.text().strip()
        if not value:
            from .store_map import read_store_map
            value = str(read_store_map(self.owner.stores.path).document.get("operator", {}).get("name", "")).strip()
        if not value:
            from .reviewer_preferences import load_reviewer_name
            value = load_reviewer_name(self.owner.series_root)
        if not value:
            raise ValueError("Enter your name for this job action")
        self.actor.setText(value)
        return value

    def _context(self):
        owner = self.owner
        current = owner.panel.current or {}
        if self.thread is not None or not current.get("root") or owner.series_root is None:
            raise ValueError("Select a registered job")
        return owner.registry, owner.series_root, owner.series_id, Path(current["root"])

    def _run(self, operation, ready, *, cancel_review=True):
        if cancel_review:
            self.cancel()
        self.archive_instead.hide()
        selected_root = self._selected_root
        def call():
            try:
                return dict(value=operation())
            except Exception as error:
                return dict(error=str(error), archive_instead=isinstance(error, UnfinishedClaimError),
                            selected_root=selected_root)
        self._on_ready = ready
        self.note.setText("Working…")
        self.thread, self.worker = self.owner.panel._start_call(call, self._ready, self._closed)
        self.selection_changed()

    @Slot(object)
    def _ready(self, value):
        if "error" in value:
            self.note.setText(value["error"])
            if self._repreparing:
                self.reprepare_message.emit(value["error"])
            self.retirement_timer.stop()
            if value.get("archive_instead") and value.get("selected_root") == self._selected_root:
                self.archive_instead.show()
        else:
            self._on_ready(value["value"])

    @Slot()
    def _closed(self):
        if self.thread is not None:
            self.thread.wait()
        self.thread = self.worker = None
        repreparing = self._repreparing
        self._repreparing = False
        # Preserve a deletion preview returned by this completed read.
        review = self.review
        self.selection_changed()
        if review is not None or self.stop_review is not None:
            self.review = review
            self.confirmation.show()
        if repreparing:
            self.reprepare_state_changed.emit()

    def _show_delete_plan(self, context, plan):
        if (self.owner.panel.current or {}).get("root") != str(context[-1]):
            self.note.setText("Selection changed. Review deletion again.")
            return
        self.stop_review = None
        self.retirement_timer.stop()
        self.review = context, plan
        self.review_text.setText(
            f"Permanently delete {plan['file_count']} file(s) ({plan['bytes'] / 1024**2:.1f} MiB) in:\n"
            + plan["root"] + "\nOnly this job folder is deleted. Files stored outside it are kept.")
        if plan.get("retirement_review"):
            consumers = [row["name"] for row in plan["retirement_review"]["affected"] if row["root"] != plan["root"]]
            self.review_text.setText(self.review_text.text()
                + "\nFinal deletion permanently cancels this job's retired unfinished claims."
                + ("\nThese stopped dependent jobs will lose their source files: " + ", ".join(consumers) if consumers else ""))
        self.destructive_ack.hide()
        self.confirm_stop_delete.hide()
        self.check_retirement.hide()
        self.confirm_delete.show()
        self.confirmation.show()
        self.note.setText("Review this folder before deleting it.")

    def _review_stop_delete(self):
        try:
            context = self._context()
            self._name()
            def done(review):
                if (self.owner.panel.current or {}).get("root") != str(context[-1]):
                    self.note.setText("Selection changed. Review Stop and delete again.")
                    return
                self.stop_review = context, review
                rows = [row["name"] + "\n" + row["root"] + "\nRecorded Stop master: "
                        + str(row.get("master_machine", "unavailable")) for row in review["affected"]]
                self.review_text.setText(
                    "Delete target: " + review["root"] + "\n\nJobs affected by Stop (including registered dependent consumers):\n"
                    + "\n\n".join(rows)
                    + "\n\nDeleting the target can break these dependent jobs and permanently abandons its unfinished work. Stop is a request; deletion waits for original retirement evidence and a separate folder review. "
                    "RAW, approved geometry and other sources stored outside the selected job folder are kept.")
                self.confirm_delete.hide()
                self.check_retirement.hide()
                self.destructive_ack.setChecked(False)
                self.destructive_ack.show()
                self.confirm_stop_delete.setEnabled(False)
                self.confirm_stop_delete.show()
                self.confirmation.show()
                self.note.setText("Review every affected job before requesting Stop. No files have been deleted.")
            self._run(lambda: stop_delete_review(*context), done)
        except Exception as error:
            self.note.setText(str(error))

    def _stop_for_deletion(self):
        if self.stop_review is None or not self.destructive_ack.isChecked() or self.thread is not None:
            return
        try:
            context, review = self.stop_review
            if self._context() != context:
                raise ValueError("Selection changed. Review Stop and delete again.")
            actor = self._name()
            def done(updated):
                if self.stop_review is None or (self.owner.panel.current or {}).get("root") != str(context[-1]):
                    self.note.setText("Stop requested for the previously reviewed jobs. Selection changed; no files have been deleted.")
                    return
                self.stop_review = context, updated
                self.confirm_stop_delete.hide()
                self.destructive_ack.hide()
                self.check_retirement.show()
                self.note.setText("Stop requested for the listed jobs. Waiting for original service and claim retirement; no files have been deleted.")
                self.retirement_timer.start()
            self._run(lambda: stop_for_deletion(context[0], review,
                machine=self.owner.panel.local_machine, actor=actor), done, cancel_review=False)
        except Exception as error:
            self.note.setText(str(error))

    def _check_retirement(self):
        if self.stop_review is None or self.thread is not None:
            return
        context, review = self.stop_review
        def check():
            status = stop_delete_status(context[0], review, stores_path=self.owner.stores.path)
            if status["ready"]:
                status["plan"] = deletion_plan(*context, retirement_review=review,
                                                stores_path=self.owner.stores.path)
            return status
        def done(status):
            if self.stop_review is None or (self.owner.panel.current or {}).get("root") != str(context[-1]):
                return
            if status["ready"]:
                self._show_delete_plan(context, status["plan"])
            else:
                self.note.setText("\n".join(status["waiting"]))
                if not status["can_wait"]:
                    self.retirement_timer.stop()
                    self.archive_instead.show()
        self._run(check, done, cancel_review=False)

    def _archive(self, archived):
        try:
            context, actor = self._context(), self._name()
            def done(_value):
                self.note.setText("Job archived. Its files are kept." if archived else "Job restored.")
                self.owner.refresh_registered()
            self._run(lambda: archive_job(*context, actor=actor, archived=archived), done)
        except Exception as error:
            self.note.setText(str(error))

    def _review_delete(self):
        try:
            context = self._context()
            self._name()
            self._run(lambda: deletion_plan(*context), lambda plan: self._show_delete_plan(context, plan))
        except Exception as error:
            self.note.setText(str(error))

    def _delete(self):
        if self.review is None or self.thread is not None:
            return
        try:
            context, plan = self.review
            if (self.owner.panel.current or {}).get("root") != str(context[-1]):
                self.cancel()
                raise ValueError("Selection changed. Review deletion again.")
            actor = self._name()
            def done(_result):
                self.cancel()
                self.note.setText("Job folder deleted.")
                self.owner.refresh_registered()
            self._run(lambda: delete_job(*context[:3], plan, actor=actor,
                stores_path=self.owner.stores.path), done)
        except Exception as error:
            self.note.setText(str(error))
