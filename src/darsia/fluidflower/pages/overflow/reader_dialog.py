"""In-application host for the established overflow scale-image reader."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.overflow_workflow import launch_scale_image_reader

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - depends on the optional Qt WebEngine wheel
    QWebEngineView = None  # type: ignore[assignment,misc]


class ScaleReaderLaunchWorker(QObject):
    """Start and verify the reader without blocking Qt's GUI thread."""

    ready = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, project: Any, config: dict[str, Any], run_id: str) -> None:
        super().__init__()
        self.project = project
        self.config = deepcopy(config)
        self.run_id = str(run_id)

    @Slot()
    def run(self) -> None:
        try:
            session = launch_scale_image_reader(
                self.project,
                self.config,
                run_ids=[self.run_id],
            )
        except Exception as exc:  # noqa: BLE001 - worker boundary shown in the GUI
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.ready.emit(session)
        finally:
            self.finished.emit()


class ScaleImageReaderDialog(QDialog):
    """Show the mature HTML reader as an integrated Qt work surface."""

    def __init__(
        self,
        project: Any,
        config: dict[str, Any],
        run_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.run_id = str(run_id).strip().casefold()
        self._session: dict[str, Any] = {}
        self._process: subprocess.Popen[Any] | None = None
        self._close_requested = False
        self._allow_close = False

        self.setWindowTitle(f"DarSIA scale-image reader - {self.run_id.upper()}")
        self.setModal(False)
        self.resize(1500, 920)
        self.setMinimumSize(980, 680)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        header = QHBoxLayout()
        title = QLabel(f"Scale-image reader | {self.run_id.upper()}")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.reload_button = QPushButton("Reload")
        self.reload_button.setEnabled(False)
        self.reload_button.clicked.connect(self._reload)
        header.addWidget(self.reload_button)
        self.output_button = QPushButton("Open saved output")
        self.output_button.clicked.connect(self._open_output)
        header.addWidget(self.output_button)
        self.close_button = QPushButton("Close reader")
        self.close_button.clicked.connect(self.close)
        header.addWidget(self.close_button)
        layout.addLayout(header)

        self.status = QLabel(
            "Starting the validated OCR and manual-review engine. No external "
            "browser will be opened."
        )
        self.status.setWordWrap(True)
        self.status.setObjectName("secondaryText")
        layout.addWidget(self.status)

        self.startup_progress = QProgressBar()
        self.startup_progress.setRange(0, 0)
        layout.addWidget(self.startup_progress)

        self.browser = QWebEngineView(self) if QWebEngineView is not None else None
        if self.browser is None:
            unavailable = QLabel(
                "Qt WebEngine is not installed. Install PySide6 with Qt WebEngine "
                "to use the integrated scale-image reader."
            )
            unavailable.setWordWrap(True)
            unavailable.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(unavailable, 1)
            self.startup_progress.hide()
            self.status.setText("Integrated reader is unavailable in this runtime.")
            return
        self.browser.hide()
        self.browser.loadFinished.connect(self._page_loaded)
        layout.addWidget(self.browser, 1)

        self._monitor = QTimer(self)
        self._monitor.setInterval(1000)
        self._monitor.timeout.connect(self._check_process)

        self._launch_thread = QThread(self)
        self._launch_worker = ScaleReaderLaunchWorker(
            project,
            config,
            self.run_id,
        )
        self._launch_worker.moveToThread(self._launch_thread)
        self._launch_thread.started.connect(self._launch_worker.run)
        self._launch_worker.ready.connect(self._reader_ready)
        self._launch_worker.failed.connect(self._reader_failed)
        self._launch_worker.finished.connect(self._launch_thread.quit)
        self._launch_thread.finished.connect(self._launch_finished)
        self._launch_thread.finished.connect(self._launch_thread.deleteLater)
        self._launch_thread.start()

    @Slot(object)
    def _reader_ready(self, raw_session: object) -> None:
        self._session = dict(raw_session)  # type: ignore[arg-type]
        process = self._session.get("_process")
        if isinstance(process, subprocess.Popen):
            self._process = process
        if self._close_requested:
            self._stop_process()
            return
        self.status.setText(
            f"{self.run_id.upper()}: reader engine is ready. Loading the review "
            "workspace inside DarSIA..."
        )
        self.browser.setUrl(QUrl(str(self._session["url"])))
        self.browser.show()
        self._monitor.start()

    @Slot(str)
    def _reader_failed(self, message: str) -> None:
        self.startup_progress.hide()
        self.status.setText(
            "The scale-image reader did not start. No review process is running.\n\n"
            + message
        )
        self.reload_button.setEnabled(False)

    @Slot()
    def _launch_finished(self) -> None:
        # After wait() the worker is freed here, in the GUI thread (Q185).
        self._launch_thread.wait()
        self._launch_worker = None
        if self._close_requested:
            self._allow_close = True
            self.close()

    @Slot(bool)
    def _page_loaded(self, successful: bool) -> None:
        self.startup_progress.hide()
        if successful:
            self.status.setText(
                f"{self.run_id.upper()}: integrated scale-image reader is ready. "
                "Saved reviews remain in this DarSIA project."
            )
            self.reload_button.setEnabled(True)
            return
        self.status.setText(
            "The reader engine started, but its Qt view could not load. Close and "
            "reopen the reader; the diagnostic log has been retained in the project."
        )

    def _reload(self) -> None:
        if self.browser is not None:
            self.browser.reload()

    def _open_output(self) -> None:
        output = Path(
            self._session.get("output_path")
            or self.project.root / "results" / "overflow" / "scale_review"
        )
        output.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))

    def _check_process(self) -> None:
        if self._process is None or self._process.poll() is None:
            return
        self._monitor.stop()
        self.reload_button.setEnabled(False)
        self.status.setText(
            "The scale-image reader stopped unexpectedly. Its saved review data was "
            "retained; close this window and reopen it to continue."
        )

    def _stop_process(self) -> None:
        self._monitor.stop()
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

    def closeEvent(self, event: QCloseEvent) -> None:
        if (
            not self._allow_close
            and hasattr(self, "_launch_thread")
            and self._launch_thread.isRunning()
        ):
            self._close_requested = True
            self.close_button.setEnabled(False)
            self.status.setText("Finishing reader startup before closing safely...")
            event.ignore()
            return
        self._stop_process()
        super().closeEvent(event)


__all__ = ["ScaleImageReaderDialog", "ScaleReaderLaunchWorker"]
