"""Native-only lazy presentation adapter; standalone Studio is unchanged."""

from __future__ import annotations

import sys
import traceback

from darsia.fluidflower.pages.phase_interpretation.subpages.train_model import (
    PhaseModelStudioSubpage,
)


class NativePhaseModelStudioSubpage(PhaseModelStudioSubpage):
    def __init__(self, series_id="", project_root=None, *, step_options=None):
        super().__init__(series_id, project_root)
        # The native steps scroll themselves; the standalone page's 700 px
        # minimum would set the height of every page of the main window.
        self.setMinimumHeight(0)
        self.step_options = dict(step_options or {})
        self.native_steps = None
        self.inline_review = None
        self.panel_navigation = None
        self.native_numbered_steps = True
        self._optical_basis_notice.hide()

    def _load_studio(self):
        """Keep the original lazy/close lifecycle, supplying a native layout."""
        self._load_pending = False
        if self.studio is not None:
            return
        self._load_failed = False
        self._retry_button.hide()
        self._placeholder.setText("Opening the phase-model steps…")
        try:
            from darsia.fluidflower.raw_production_review import install_readers
            install_readers()
            from darsia.fluidflower.ui.model_training import (
                SharedPhaseModelTrainingPage,
            )
            from darsia.fluidflower.native_phase_steps import NativePhaseSteps

            studio = SharedPhaseModelTrainingPage(metadata_preview_only=True)
            options = dict(self.step_options)
            model_for_calibration = options.pop("model_for_calibration", None)
            if "normalisation" not in options:
                from darsia.fluidflower.normalisation_step import NormalisationStep

                options["normalisation"] = NormalisationStep()
            from darsia.fluidflower.native_training_photometry import NativeTrainingPhotometry

            photometry_factory = options.pop("training_photometry", NativeTrainingPhotometry)
            steps = NativePhaseSteps(studio, **options)
            if callable(model_for_calibration):
                steps.model_for_calibration_requested.connect(model_for_calibration)
            if photometry_factory is not None:
                steps.training_photometry = photometry_factory(steps)
            from darsia.fluidflower.native_continuation_training import NativeContinuationTraining

            NativeContinuationTraining(steps)
            steps.set_context(self.project_root, self.series_id)
        except Exception as error:
            self._load_failed = True
            if sys.stderr is not None:
                traceback.print_exc()
            self._placeholder.setText(
                f"Could not open phase-model steps: {type(error).__name__}: {error}"
            )
            self._retry_button.show()
            return
        self.studio, self.native_steps = studio, steps
        studio.setParent(self)
        studio.hide()
        self._layout.removeWidget(self._placeholder)
        self._placeholder.deleteLater()
        self._layout.removeWidget(self._retry_button)
        self._retry_button.deleteLater()
        from .native_inline_phase_review import NativeInlinePhaseReview

        self.content_stack = steps.workspace_stack
        self.inline_review = NativeInlinePhaseReview(self, self.content_stack, steps)
        self.content_stack.addWidget(self.inline_review)
        steps.phase_workflow.review_owner.process_factory = self.inline_review.process_factory
        from .native_phase_panel_navigation import NativePhasePanelNavigation

        self.panel_navigation = NativePhasePanelNavigation(steps, self.inline_review)
        self._layout.addWidget(steps)
        self._install_close_guard()
        self.studio_loaded.emit()

    def show_steps(self):
        if self.native_steps is not None:
            self.content_stack.setCurrentWidget(self.native_steps.workspace_overview)

    def review_context_block_reason(self):
        if self.native_steps is not None:
            return self.native_steps.phase_workflow.close_block_reason()
        return ""

    def bind_context(self, series_id, project_root):
        if ((str(series_id), project_root) != (self.series_id, self.project_root)
                and (reason := self.review_context_block_reason())):
            raise ValueError(reason)
        if ((str(series_id), project_root) != (self.series_id, self.project_root)
                and (navigation := getattr(self, "panel_navigation", None)) is not None):
            navigation.cancel()
        self.series_id, self.project_root = str(series_id), project_root
        if self.native_steps is not None:
            self.native_steps.set_context(project_root, series_id)
