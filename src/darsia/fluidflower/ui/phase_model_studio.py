"""Standalone shell for transferable FluidFlower phase-model development."""

from __future__ import annotations

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox

from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage
from darsia.fluidflower.ui.theme import apply_application_theme


class PhaseModelStudioWindow(QMainWindow):
    """Host the reusable phase-training page outside the production app."""

    def __init__(self) -> None:
        super().__init__()
        from darsia.fluidflower.ui.interaction_safety import install_interaction_safety
        install_interaction_safety()
        self.setWindowTitle("DarSIA · Phase Model Studio")
        self.resize(1450, 920)
        self.setMinimumSize(1000, 680)
        self.training_page = SharedPhaseModelTrainingPage()
        self.setCentralWidget(self.training_page)
        apply_application_theme(self)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.training_page.training_process is not None:
            QMessageBox.information(
                self,
                "Phase-model training is still running",
                "The studio remains open so the active model version is not "
                "interrupted accidentally. Use 'Stop training now' "
                "before closing the studio.",
            )
            event.ignore()
            return
        if self.training_page.validation_process is not None:
            QMessageBox.information(
                self,
                "Phase maps are still being prepared",
                "The studio remains open while full-resolution phase maps are "
                "being prepared. You can close it when preparation finishes; "
                "the trained model is already saved in the model library.",
            )
            event.ignore()
            return
        super().closeEvent(event)


__all__ = ["PhaseModelStudioWindow"]
