"""Workflow process management for DarSIA GUI."""

import psutil
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QMessageBox

from darsia.presets.workflows.analysis.progress import PROGRESS_LINE_PREFIX
from darsia.presets.workflows.analysis.streaming import STREAM_LINE_PREFIX
from darsia.presets.workflows.results_folder import (
    open_in_file_explorer,
    suggested_workflow_results_folder,
)


class ProcessRunner:
    """Manages launching and aborting workflow processes with live logging."""

    def __init__(self, main_window):
        self.main_window = main_window
        # Maps workflow name (e.g. "setup") to (QProcess, workflow label) for
        # whatever is currently running; used by the status-bar dashboard.
        self.active = {}

    def start_workflow_process(
        self,
        argv,
        run_button,
        abort_button,
        cwd=None,
        *,
        workflow=None,
        actions=None,
        config_paths=None,
        on_stream_line=None,
        on_progress_line=None,
    ):
        """Launch argv as a QProcess, streaming merged stdout/stderr to the log.
        Disables run_button and shows/enables abort_button while running; restores
        button state and logs completion/abort/error when the process finishes.
        Returns the QProcess (caller must keep a reference alive, e.g. on the tab,
        and call abort_workflow_process(process) to abort it).

        Args:
            argv: Command and arguments as list of strings.
            run_button: Button to disable during process and enable after.
            abort_button: Button to show during process and hide after.
            cwd: Working directory for the process (optional).
            workflow: Workflow name (e.g. "setup"), used for the dashboard and
                the terminal-state Done/Error dialog. Dialog is skipped if None.
            actions: Enabled action labels for this run, used to infer a results
                folder for the Done dialog's "Open in folder" button.
            config_paths: Config paths for this run, used for the same purpose.
            on_stream_line: Called with each raw output line that starts with
                STREAM_LINE_PREFIX; such lines are not logged or included in
                the error-dialog detail text. All other lines are handled as
                before (logged + buffered for the error dialog).
            on_progress_line: Same, but for lines starting with
                PROGRESS_LINE_PREFIX.
        """
        process = QProcess(self.main_window)
        process.setProgram(argv[0])
        process.setArguments(argv[1:])
        if cwd:
            process.setWorkingDirectory(str(cwd))
        process.setProcessChannelMode(QProcess.MergedChannels)

        output_lines = []
        pending = bytearray()

        def handle_output():
            pending.extend(bytes(process.readAllStandardOutput()))
            *complete_lines, remainder = pending.split(b"\n")
            pending[:] = remainder
            for raw_line in complete_lines:
                # Strip a trailing \r left over from \r\n (e.g. Windows stdout
                # text-mode translation), matching str.splitlines()'s handling
                # of \r\n as a single line terminator.
                line = raw_line.rstrip(b"\r").decode(errors="replace")
                if not line:
                    continue
                if on_stream_line is not None and line.startswith(STREAM_LINE_PREFIX):
                    on_stream_line(line)
                    continue
                if on_progress_line is not None and line.startswith(
                    PROGRESS_LINE_PREFIX
                ):
                    on_progress_line(line)
                    continue
                output_lines.append(line)
                self.main_window.print_log(line)

        def _is_side_channel_line(line):
            return (
                on_stream_line is not None and line.startswith(STREAM_LINE_PREFIX)
            ) or (
                on_progress_line is not None and line.startswith(PROGRESS_LINE_PREFIX)
            )

        def handle_finished(exit_code, exit_status):
            if pending:
                line = bytes(pending).rstrip(b"\r").decode(errors="replace")
                pending.clear()
                if line and not _is_side_channel_line(line):
                    output_lines.append(line)
                    self.main_window.print_log(line)
            run_button.setEnabled(True)
            abort_button.setVisible(False)
            abort_button.setEnabled(False)
            self.active.pop(workflow, None)

            aborted = bool(process.property("darsia_aborted"))
            if aborted:
                self.main_window.print_log("Process aborted.")
                return
            if exit_status == QProcess.CrashExit:
                self.main_window.print_log("Process exited unexpectedly.")
                self._show_error_dialog(workflow, exit_code, output_lines)
            elif exit_code != 0:
                self.main_window.print_log(f"Process exited with code {exit_code}.")
                self._show_error_dialog(workflow, exit_code, output_lines)
            else:
                self.main_window.print_log("Completed successfully!")
                self.main_window.config_controller.load_config()
                self._show_done_dialog(workflow, actions, config_paths)

        process.readyReadStandardOutput.connect(handle_output)
        process.finished.connect(handle_finished)
        run_button.setEnabled(False)
        abort_button.setVisible(True)
        abort_button.setEnabled(True)

        # Save current GUI state to disk before launching subprocess
        # so it reads fresh config with any unsaved widget edits
        self.main_window.settings_factory.save_settings()

        process.start()
        if workflow:
            self.active[workflow] = (process, workflow)
        return process

    def _show_done_dialog(self, workflow, actions, config_paths):
        """Show a modal completion dialog with an optional 'Open in folder' button."""
        if workflow is None:
            return
        box = QMessageBox(self.main_window)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Done")
        box.setText(f"{workflow.capitalize()} workflow completed.")
        open_folder_button = None
        folder = self._suggested_results_folder(workflow, actions, config_paths)
        if folder is not None:
            open_folder_button = box.addButton("Open in folder", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if open_folder_button is not None and box.clickedButton() is open_folder_button:
            try:
                open_in_file_explorer(folder)
            except (FileNotFoundError, RuntimeError) as e:
                self.main_window.print_log(f"Could not open results folder: {e}")

    def _show_error_dialog(self, workflow, exit_code, output_lines):
        """Show a modal error dialog with captured output as expandable detail."""
        if workflow is None:
            return
        box = QMessageBox(self.main_window)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Error")
        box.setText(
            f"{workflow.capitalize()} workflow failed with exit code {exit_code}."
        )
        detail = "\n".join(output_lines).strip()
        box.setDetailedText(detail or "No output captured.")
        box.exec()

    @staticmethod
    def _suggested_results_folder(workflow, actions, config_paths):
        if not config_paths:
            return None
        try:
            return suggested_workflow_results_folder(
                workflow, config_paths, actions or []
            )
        except (OSError, ValueError):
            return None

    def abort_workflow_process(self, process):
        """Abort a process started via start_workflow_process, killing its whole tree."""
        if process is None or process.state() == QProcess.NotRunning:
            return
        process.setProperty("darsia_aborted", True)
        self._kill_process_tree(process.processId())

    @staticmethod
    def _kill_process_tree(pid):
        """Best-effort kill of a process and all its descendants (children, grandchildren)."""
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
