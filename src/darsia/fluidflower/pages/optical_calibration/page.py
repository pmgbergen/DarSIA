"""Qt page for correction-linked optical and indicator-response calibration."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

from darsia.fluidflower.image_correction import CorrectionRevisionStore
from darsia.fluidflower.indicator_response import IndicatorResponseModelStore
from darsia.fluidflower.pages.calibration.indicator_response_panel import (
    IndicatorResponsePanel,
)
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.ui.common import ModulePage


class OpticalCalibrationPage(ModulePage):
    """Own optical-response fitting before visible-phase interpretation."""

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__(project, "optical_calibration")
        self.correction_store = CorrectionRevisionStore(project)
        self.response_store = IndicatorResponseModelStore(project)
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "The active image-correction revision and the fitted optical response form "
            "one traceable branch. A different correction requires a new compatible "
            "response and phase interpretation. Known-pH buffers constrain the aqueous "
            "response; they are never used as illumination references."
        )
        self.branch_status = QLabel()
        self.branch_status.setObjectName("integrationNotice")
        self.branch_status.setWordWrap(True)
        self.work_layout.addWidget(self.branch_status)
        self.indicator_response_panel = IndicatorResponsePanel(project)
        self.indicator_response_panel.model_changed.connect(self._model_changed)
        self.work_layout.addWidget(self.indicator_response_panel)
        self.refresh()

    def _model_changed(self, _revision: object) -> None:
        self.project.set_module_status(
            "optical_calibration",
            "review_required",
            reason="A new measured indicator-response model was fitted.",
        )
        self.refresh()
        self.status_changed.emit()

    def refresh(self) -> None:
        super().refresh()
        correction = self.correction_store.current()
        response = self.response_store.current()
        correction_text = (
            f"correction {str(correction.get('revision_id') or '')[:12]}"
            if correction
            else "no saved image-correction revision"
        )
        response_text = (
            f"response {str(response.get('model_id') or '')[:12]} "
            f"({str(response.get('status') or 'unknown').replace('_', ' ')})"
            if response
            else "generic corrected-image response; no measured buffer model"
        )
        self.branch_status.setText(
            f"Current optical branch: {correction_text}; {response_text}."
        )
        if hasattr(self, "indicator_response_panel"):
            self.indicator_response_panel.load_current()


__all__ = ["OpticalCalibrationPage"]
