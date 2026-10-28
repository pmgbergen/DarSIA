"""Own the native correction process and its original review session."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout


class NativePhaseReviewOwner:
    def __init__(self, workflow):
        self.workflow, self.steps, self.studio = workflow, workflow.steps, workflow.studio
        self.process = None
        self.process_factory = QProcess
        self.active_model_root = ""
        self._error_output = bytearray()
        self._training_session = None
        self._original_start = self.studio._start_review_process
        self.studio._start_review_process = self.start

    def identity(self):
        raw = self.studio.review_save_root.text().strip()
        if not raw or not Path(raw).expanduser().is_absolute():
            raise ValueError("Choose an absolute review save folder before continuing")
        return (Path(raw).expanduser().resolve(),
                dict(series=self.steps.series_id, reviewer=self.studio.reviewer.text().strip()))

    def close_block_reason(self):
        inline_reason = getattr(self.process, "close_block_reason", None)
        if inline_reason is not None:
            return inline_reason()
        return ("Close the phase correction window and let its saved work finish before continuing."
                if self.process is not None else "")

    def start(self, mode, binding, *, session_id=""):
        if mode not in {"open_correction", "blind_region_labeling", "active_learning"}:
            return self._original_start(mode, binding, session_id=session_id)
        if self.process is not None:
            present = getattr(self.process, "show_review", None)
            if present is not None:
                present()
            self.studio.review_status.setText(
                getattr(self.process, "already_open_message",
                    "The phase correction window is already open. Switch between labels and blind validation in that window."))
            return
        if self.studio.training_process is not None or self.studio._series_batch_active:
            self.studio.review_status.setText("Wait for training to finish before opening editable review.")
            return
        process = None
        try:
            from .native_phase_review_session import reserve_review_session
            from .phase_workflow import discover_phase_review_sessions
            root, identity = self.identity()
            model = self.studio._selected_validation_model() or {}
            scratch_control = getattr(self.steps, "scratch_basis", None)
            scratch = scratch_control is not None and scratch_control.isChecked()
            source = {}
            if scratch:
                from .native_phase_scratch_review import source_identity
                source = dict(review_basis="scratch", source_identity=source_identity(
                    binding, series=self.steps.series_id))
            # Re-discover small manifests only at the explicit launch boundary.
            # The page's review root is already selected for this series.
            session = reserve_review_session(root, **identity,
                target_model_id="" if scratch else str(model.get("model_id") or ""),
                existing_sessions=discover_phase_review_sessions([root]), **source)
            command = self.studio._review_command("active_learning", binding_override=binding,
                session_id=session["review_session_id"])
            if not command or not str(command[0]).strip():
                raise ValueError("The original review command is unavailable")
            if scratch:
                # The original command supplies only the verified photograph
                # collection. A fresh label session has no target model.
                for flag in ("--target-model-id", "--review-current-model-root"):
                    while flag in command:
                        index = command.index(flag)
                        del command[index:index + 2]
            command.extend(("--review-surface", "blind_regions", "--phase-workflow-role",
                "blind" if mode == "blind_region_labeling" else "label",
                "--phase-workflow-series", str(self.steps.series_id)))
            if not scratch and not model.get("root"):
                raise ValueError("The selected model has no saved source folder")
            if not scratch:
                command.extend(("--review-current-model-root", str(model["root"])))
            process = self.process_factory(self.steps)
            process.review_basis = "scratch" if scratch else "model"
            process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
            self.process = process
            self.active_model_root = str(model.get("root") or "")
            self._error_output.clear()
            process.readyReadStandardOutput.connect(lambda: process.readAllStandardOutput())
            process.readyReadStandardError.connect(lambda: self._read_error(process))
            process.errorOccurred.connect(lambda error: self._process_error(process, error))
            process.finished.connect(lambda code, status: self._finished(process, code, status))
            process.started.connect(lambda: self._started(process, binding))
            process.start(command[0], command[1:])
            self.workflow.refresh()
        except Exception as error:  # noqa: BLE001 - visible UI boundary
            if process is not None and process.state() == QProcess.ProcessState.NotRunning:
                self._retire(process)
            self.studio.review_status.setText("Could not open phase correction: " + str(error))

    def _started(self, process, binding):
        if process is not self.process:
            return
        self.studio.review_status.setText(
            "Phase correction opened. Both roles share this saved session; finish training or explicitly discard it before starting another.")
        self.studio.review_opened.emit(dict(binding))

    def _read_error(self, process):
        if process is self.process:
            self._error_output.extend(bytes(process.readAllStandardError()))
            del self._error_output[:-16384]

    def _process_error(self, process, error):
        if process is self.process and error == QProcess.ProcessError.FailedToStart:
            message = process.errorString()
            self._retire(process)
            self.studio.review_status.setText(
                "Phase correction did not start. Its session reservation is retained for retry or explicit discard: " + message)

    def _finished(self, process, exit_code, exit_status):
        if process is not self.process:
            return
        self._read_error(process)
        detail = self._error_output.decode("utf-8", errors="replace").strip()
        self._retire(process)
        self.studio._refresh_review_sessions()
        self.studio.review_status.setText(
            "Phase correction closed. Saved labels and validation boxes remain in the active session."
            if int(exit_code) == 0 and exit_status == QProcess.ExitStatus.NormalExit else
            "Phase correction stopped with an error; saved files and its session reservation are retained. " + detail[-2000:])

    def _retire(self, process):
        if process is not self.process:
            return
        self.process = None
        self.active_model_root = ""
        if process is not None:
            process.deleteLater()
        self.workflow.refresh()

    def run_training(self, callback, *, saved_labels_only=False):
        from .native_phase_review_session import (
            read_current_review_session, require_training_session_admission, review_editing_lease)
        from .phase_workflow import parse_training_bundles
        if reason := self.close_block_reason():
            raise ValueError(reason)
        root, identity = self.identity()
        with review_editing_lease(root, **identity):
            bundles = parse_training_bundles(self.studio.bundles.toPlainText())
            require_training_session_admission(bundles)
            session = None if saved_labels_only else read_current_review_session(root, **identity)
            if session is not None:
                sources = {str(Path(decisions).resolve()) for decisions, _manifest in bundles}
                expected = {str(Path(path).parent / "pairwise_review_decisions.json")
                            for path in session["manifest_paths"]}
                if session["status"] != "active" or not expected or not expected <= sources:
                    raise ValueError("Select all saved evidence in the active session for training, or explicitly discard the session first.")
            self._training_session = (root, identity, session)
            try:
                return callback()
            finally:
                self._training_session = None

    def before_launch(self, request):
        if self._training_session is None:
            raise ValueError("Start native training through its session admission check")
        root, identity, session = self._training_session
        if session is not None:
            from .native_phase_review_session import complete_review_session_training
            complete_review_session_training(root, **identity,
                review_session_id=session["review_session_id"], request=request,
                reason="Submitted the saved phase labels and blind validation boxes to this named model version.")

    def discard_dialog(self):
        if reason := self.close_block_reason():
            self.studio.review_status.setText(reason)
            return
        from .native_phase_review_session import read_current_review_session
        from .phase_workflow import discover_phase_review_sessions
        from .reviewer_preferences import reviewer_identifier
        sections = getattr(self.workflow, "management_sections", None)
        try:
            root, identity = self.identity()
            session = read_current_review_session(root, **identity)
            legacy = [] if session is not None else [row for row in discover_phase_review_sessions([root])
                if not row.get("editing_locked") and row.get("include_in_training") is not False
                and reviewer_identifier(row.get("reviewer") or row.get("reviewer_id") or "")
                == reviewer_identifier(identity["reviewer"])]
            if session is None and not legacy:
                self.studio.review_status.setText("There is no active review session to discard.")
                return
            context = root, identity, session, tuple(str(row["session_path"]) for row in legacy)
            previous = sections.sections.get("discard") if sections is not None else None
            if (previous is not None and not previous.isHidden()
                    and context == getattr(self, "_discard_context", None)):
                sections.show("discard")
                return
        except (OSError, ValueError, RuntimeError) as error:
            self.studio.review_status.setText(str(error))
            return
        dialog = QDialog(self.steps)
        dialog.setWindowTitle("Discard active phase review")
        layout = QVBoxLayout(dialog)
        notice = QLabel("Discarding excludes this review from future training and closes its session. "
            "All labels, validation boxes and history remain saved. Close any older review windows first.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        if legacy:
            details = QLabel("Older editable reviews to retire:\n" + "\n".join(str(row["session_path"]) for row in legacy))
            details.setWordWrap(True)
            layout.addWidget(details)
        reason = QLineEdit()
        reason.setPlaceholderText("Why is this session being discarded?")
        layout.addWidget(reason)
        status = QLabel("")
        status.setWordWrap(True)
        layout.addWidget(status)
        button = QPushButton("Discard session and preserve its saved evidence")
        button.setEnabled(False)
        reason.textChanged.connect(lambda value: button.setEnabled(bool(value.strip())))
        layout.addWidget(button)

        def apply():
            from .native_phase_review_session import discard_review_session, discard_legacy_review_sessions
            try:
                if self.identity() != (root, identity) or self.process is not None:
                    raise ValueError("The review context changed. Close this review section and open it again.")
                if session is not None:
                    discard_review_session(root, **identity, review_session_id=session["review_session_id"], reason=reason.text())
                else:
                    discard_legacy_review_sessions(root, **identity,
                        session_paths=[row["session_path"] for row in legacy], reason=reason.text())
                self.studio._refresh_review_sessions()
                self.studio.review_status.setText("Review session discarded. Its saved evidence and history remain intact.")
                dialog.accept()
            except (OSError, ValueError, RuntimeError) as error:
                status.setText(str(error))

        button.clicked.connect(apply)
        if sections is None:
            dialog.show()
        else:
            sections.present("discard", dialog.windowTitle(), dialog, 3)
        self._discard_dialog = dialog
        self._discard_context = context
