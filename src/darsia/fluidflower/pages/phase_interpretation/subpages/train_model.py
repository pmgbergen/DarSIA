"""Lazy adapter for the canonical phase-model studio page."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QObject, QProcess, QTimer, Signal
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton, QTabWidget, QVBoxLayout, QWidget


class PhaseModelStudioSubpage(QWidget):
    """Load the shared Studio only when its workflow tab is opened."""

    studio_loaded = Signal()

    def __init__(self, series_id: str = "", project_root: Path | str | None = None) -> None:
        super().__init__()
        self.series_id = str(series_id or "")
        self.project_root = Path(project_root) if project_root is not None else None
        self.studio: QWidget | None = None
        self._guarded_window: QWidget | None = None
        self._load_pending = False
        self._load_failed = False
        self._mass_series_bound = False
        self.setMinimumHeight(700)
        self._layout = QVBoxLayout(self)
        self._optical_basis_notice = QLabel(
            "Studio uses the radiometry recorded with each phase model. A correction "
            "revision selected on the Image correction page is not automatically "
            "applied here; check the optical basis before training or calibrating."
        )
        self._optical_basis_notice.setObjectName("secondaryText")
        self._optical_basis_notice.setWordWrap(True)
        self._layout.addWidget(self._optical_basis_notice)
        self._placeholder = QLabel(
            "Opening phase review, training and mass calibration…"
        )
        self._placeholder.setObjectName("secondaryText")
        self._layout.addWidget(self._placeholder)
        self._retry_button = QPushButton("Retry opening Studio")
        self._retry_button.hide()
        self._retry_button.clicked.connect(self._load_studio)
        self._layout.addWidget(self._retry_button)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self.studio is None and not self._load_pending and not self._load_failed:
            self._load_pending = True
            QTimer.singleShot(0, self._load_studio)

    def _load_studio(self) -> None:
        self._load_pending = False
        if self.studio is not None:
            return
        self._load_failed = False
        self._retry_button.hide()
        self._placeholder.setText("Opening phase review, training and mass calibration…")
        try:
            from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage

            studio = SharedPhaseModelTrainingPage()
            if self.series_id:
                try:
                    studio.set_model_series(self.series_id)
                except ValueError:
                    # Invalid IDs keep Studio's ordinary series chooser.
                    pass
            if self.project_root is not None and hasattr(studio, "set_project_root"):
                studio.set_project_root(self.project_root, fixed=True)
        except Exception as error:  # noqa: BLE001 - Qt otherwise hides callback failures
            self._load_failed = True
            if sys.stderr is not None:
                traceback.print_exc()
            self._placeholder.setText(
                "Could not open Studio: " + f"{type(error).__name__}: {error}"
            )
            self._retry_button.show()
            return
        self.studio = studio
        tabs = getattr(studio, "tabs", None)
        if isinstance(tabs, QTabWidget):
            tabs.currentChanged.connect(self._bind_initial_mass_series)
            self._bind_initial_mass_series(tabs.currentIndex())
        self._layout.removeWidget(self._placeholder)
        self._placeholder.deleteLater()
        self._layout.removeWidget(self._retry_button)
        self._retry_button.deleteLater()
        self._layout.addWidget(self.studio)
        self._install_close_guard()
        self.studio_loaded.emit()

    def _bind_initial_mass_series(self, index: int) -> None:
        """Show the host project's series before a phase model has been loaded."""
        if self._mass_series_bound or self.studio is None:
            return
        tabs = getattr(self.studio, "tabs", None)
        if not isinstance(tabs, QTabWidget) or tabs.widget(index) is not getattr(
            self.studio, "mass_workspace_tab", None
        ):
            return
        workspace = getattr(self.studio, "_mass_workspace", None)
        if workspace is None:
            return
        if getattr(workspace, "phase_catalog", None):
            self._mass_series_bound = True
            return
        series_control = getattr(workspace, "series", None)
        if series_control is None:
            return
        canonical = self.series_id.casefold().replace("-", "").replace("_", "")
        target = series_control.findData(canonical)
        if target >= 0:
            series_control.setCurrentIndex(target)
            self._mass_series_bound = True

    def _install_close_guard(self) -> None:
        window = self.window()
        if window is self._guarded_window:
            return
        if self._guarded_window is not None:
            self._guarded_window.removeEventFilter(self)
        self._guarded_window = window
        window.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        guarded_window = getattr(self, "_guarded_window", None)
        if watched is guarded_window and event.type() == QEvent.Type.Close:
            studio = getattr(self, "studio", None)
            if studio is not None:
                notice = self._active_job_notice(studio)
                if notice is not None:
                    QMessageBox.information(guarded_window, *notice)
                    event.ignore()
                    return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _active_job_notice(studio: QWidget) -> tuple[str, str] | None:
        if getattr(studio, "training_process", None) is not None:
            return (
                "Phase-model training is still running",
                "The Studio remains open so the active model version is not "
                "interrupted accidentally. Use 'Stop training now' "
                "before closing the window.",
            )
        if getattr(studio, "_series_batch_active", False):
            return (
                "Phase maps are still being prepared",
                "The Studio remains open while the series phase-map batch is "
                "being prepared. You can close it when preparation finishes.",
            )
        if getattr(studio, "validation_process", None) is not None:
            return (
                "Phase maps are still being prepared",
                "The Studio remains open while full-resolution phase maps are "
                "being prepared. You can close it when preparation finishes.",
            )
        workspace = getattr(studio, "_mass_workspace", None)
        process = getattr(workspace, "process", None)
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            return (
                "Concentration or mass calculation is still running",
                "The Studio remains open while the calculation finishes. "
                "Stop the calculation before closing the window.",
            )
        return None


def build(parent_page: Any) -> QWidget:
    project = getattr(parent_page, "project", None)
    manifest = getattr(project, "manifest", {})
    return PhaseModelStudioSubpage(
        str(manifest.get("series_id") or ""),
        getattr(project, "root", None),
    )


__all__ = ["PhaseModelStudioSubpage", "build"]
