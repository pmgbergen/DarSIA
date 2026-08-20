"""Workflow process management for DarSIA GUI."""

import psutil
from PySide6.QtCore import QProcess


class ProcessRunner:
    """Manages launching and aborting workflow processes with live logging."""

    def __init__(self, main_window):
        self.main_window = main_window

    def start_workflow_process(self, argv, run_button, abort_button, cwd=None):
        """Launch argv as a QProcess, streaming merged stdout/stderr to the log.
        Disables run_button and shows/enables abort_button while running; restores
        button state and logs completion/abort/error when the process finishes.
        Returns the QProcess (caller must keep a reference alive, e.g. on the tab,
        and call abort_workflow_process(process) to abort it).
        """
        process = QProcess(self.main_window)
        process.setProgram(argv[0])
        process.setArguments(argv[1:])
        if cwd:
            process.setWorkingDirectory(str(cwd))
        process.setProcessChannelMode(QProcess.MergedChannels)

        def handle_output():
            data = bytes(process.readAllStandardOutput()).decode(errors="replace")
            for line in data.splitlines():
                if line:
                    self.main_window.print_log(line)

        def handle_finished(exit_code, exit_status):
            run_button.setEnabled(True)
            abort_button.setVisible(False)
            abort_button.setEnabled(False)
            if exit_status == QProcess.CrashExit:
                self.main_window.print_log("Process aborted.")
            elif exit_code != 0:
                self.main_window.print_log(f"Process exited with code {exit_code}.")
            else:
                self.main_window.print_log("Completed successfully!")

        process.readyReadStandardOutput.connect(handle_output)
        process.finished.connect(handle_finished)
        run_button.setEnabled(False)
        abort_button.setVisible(True)
        abort_button.setEnabled(True)
        process.start()
        return process

    def abort_workflow_process(self, process):
        """Abort a process started via start_workflow_process, killing its whole tree."""
        if process is None or process.state() == QProcess.NotRunning:
            return
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
