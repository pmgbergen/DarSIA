"""Controller of the interactive mass calibration: callbacks of the window.

The controller connects the window (`CalibrationView`) with the calculations
(`CalibrationState`): it switches labels and images, applies the slider values, saves,
and decides what happens when the window is closed.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt

from darsia.presets.workflows.calibration.calibration_view import CalibrationView
from darsia.presets.workflows.windows.background import BUSY_MESSAGE, BackgroundRunner
from darsia.utils.figure_window import ask_yes_no

if TYPE_CHECKING:
    from darsia.presets.workflows.calibration.manual_calibration import (
        CalibrationState,
    )
    from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
        HeterogeneousColorToMassAnalysis,
    )

logger = logging.getLogger(__name__)


class CalibrationController:
    """Interactive tuning of the signal values, flash and contours, per image and label.

    Parameters
    ----------
    state : CalibrationState
        State of the calibration (data, parameters, mass series).
    analysis : HeterogeneousColorToMassAnalysis
        The calibrated analysis.
    cmap : optional
        Colormap of the signal panel.
    save : Callable, optional
        Stores the current calibration; called by the Save button, and when the window
        is closed with unsaved changes which the user wants to keep.

    """

    def __init__(
        self,
        state: "CalibrationState",
        analysis: "HeterogeneousColorToMassAnalysis",
        cmap=None,
        save: Callable[[], None] | None = None,
    ) -> None:
        self.state = state
        self.analysis = analysis
        self.save = save
        self.view = CalibrationView(state, analysis, cmap)
        """The window."""
        self.pending = False
        """Whether a value or flash slider was moved since the last applied values.

        The sliders round to their step, so their untouched values are not applied (they
        would change the stored calibration).

        """
        self.runner: BackgroundRunner | None = None
        """Runs the calculations (in a worker thread for windows with an event
        loop)."""

    @property
    def busy(self) -> bool:
        """Whether a calculation runs in the background."""
        return self.runner is not None and self.runner.busy

    def _refuse(self) -> bool:
        """Whether an action must wait, because a calculation runs."""
        return self.runner is not None and self.runner.refuse()

    def _thresholds(self) -> list[float]:
        """Contour thresholds of the sliders (read in the window's thread)."""
        return [slider.val for slider in self.view.threshold_sliders]

    def _prepare_display(self, thresholds: list[float]) -> None:
        """Compute the maps shown by `refresh` in advance (signal, contour overlay).

        They are computed on first use and cached in the state, so `refresh` then
        only draws; without this, the window's thread would compute them.

        """
        self.state.coarse_signal
        self.state.contour_overlay(thresholds)

    def run(self) -> bool:
        """Show the window until it is closed.

        Returns
        -------
        bool
            Whether the stored calibration equals the calibration at the end.

        """
        state, view = self.state, self.view
        if state.label_idx not in view.labels:
            state.next_label()
        if state.maps is None or state.maps["image_idx"] != state.image_idx:
            state.compute_maps()
        if not state.mass_analysis_done:
            state.update_mass_analysis()
        view.build()
        self.runner = BackgroundRunner(
            view.fig,
            show_busy=lambda message, now: view.show_busy(message, now=now),
            show_refusal=view.show_values_status,
            refusal_shown=lambda: view.values_status.get_text() == BUSY_MESSAGE,
        )
        view.show_label(state.label_idx)
        self._connect()
        plt.show()
        self.runner.shutdown()
        return not state.has_unsaved_changes()

    def _connect(self) -> None:
        """Connect the callbacks of the window."""
        view = self.view
        for slider in view.value_sliders:
            slider.on_changed(lambda val: view.show_values_status(self.check_values()))
        for slider in view.value_sliders + view.flash_sliders:
            slider.on_changed(self.on_value_slider)
        for slider in view.all_sliders:
            slider.on_changed(view.mark_unsaved)

        view.buttons["Update"].on_clicked(self.update_analysis)
        view.buttons["Reset"].on_clicked(self.reset)
        view.buttons["Save"].on_clicked(self.on_save)
        view.buttons["Next Signal"].on_clicked(self.next_signal)
        view.buttons["Next Label"].on_clicked(self.next_label)
        view.buttons["Next Image"].on_clicked(self.next_image)
        view.fig.canvas.mpl_connect("button_press_event", self.on_click)
        view.fig.canvas.mpl_connect("close_event", self.on_close)

    # ! ---- PARAMETERS ----

    def check_values(self) -> str | None:
        """Message if the signal values of the label decrease, else None."""
        return self.state.check_values(self.view.values(), self.view.label_idx)

    def apply_parameters(self) -> bool | None:
        """Apply the slider values.

        Returns
        -------
        bool or None
            Whether any value changed, or None if the values are invalid, in which case
            nothing is applied.

        """
        view = self.view
        message = self.check_values()
        view.show_values_status(message)
        if message is not None:
            return None
        changed = self.state.apply_parameters(
            view.values(), view.flash_values(), view.label_idx
        )
        self.pending = False
        return changed

    def on_value_slider(self, *args) -> None:
        """Remember that a value or flash slider was moved."""
        self.pending = True

    # ! ---- LABEL AND IMAGE ----

    def switch_label(self) -> None:
        """Show the label selected in the state; moves not applied are dropped."""
        self.pending = False
        self.view.show_values_status(None)
        self.view.show_label(self.state.label_idx)

    def on_click(self, event) -> None:
        """A click in the image panel selects the label at that pixel."""
        view = self.view
        if event.inaxes is not view.ax_coarse_image or event.button != 1:
            return
        toolbar = view.fig.canvas.toolbar
        if toolbar is not None and toolbar.mode:
            return  # zooming or panning
        if self._refuse():
            return
        row, col = int(round(event.ydata)), int(round(event.xdata))
        if self.state.select_label_at(row, col) is None:
            view.show_values_status("This label has no signal function.")
            return
        self.switch_label()

    def next_label(self, event) -> None:
        """Move to the next label."""
        if self._refuse():
            return
        self.state.next_label()
        self.switch_label()

    def next_signal(self, event) -> None:
        """Show the next signal of the current image."""
        if self._refuse():
            return
        self.state.next_signal()
        self.view.refresh(recompute=False)

    def next_image(self, event) -> None:
        """Analyze and show the next image."""
        if self._refuse():
            return
        state = self.state

        thresholds = self._thresholds()

        def work(progress) -> None:
            state.next_image()
            state.compute_maps()
            self._prepare_display(thresholds)

        self.runner.run("Computing", work, lambda: self.view.refresh(recompute=False))

    # ! ---- UPDATE, RESET, SAVE, CLOSE ----

    def update_analysis(self, event=None) -> None:
        """Apply the slider values and update everything."""
        if self._refuse():
            return
        state, view = self.state, self.view
        changed = self.apply_parameters()
        if changed is None:
            return
        level = view.selected_level()
        new_level = level != state.resolution_level
        if changed:
            view.mark_unsaved()

        thresholds = self._thresholds()

        def work(progress) -> None:
            if new_level:
                state.set_resolution(level)
            if changed or new_level:
                state.update_mass_analysis(progress=progress)
            state.compute_maps()
            self._prepare_display(thresholds)

        def done() -> None:
            view.set_resolution_title()
            view.refresh(recompute=False, signal_function=True, mass_plot=True)

        message = f"Computing at resolution 1/{level}" if new_level else "Computing"
        self.runner.run(message, work, done)

    def reset(self, event) -> None:
        """Restore the initial values of this label, flash, contours."""
        if self._refuse():
            return
        state, view = self.state, self.view
        for slider, value in zip(
            view.value_sliders, state.initial_values[view.label_idx]
        ):
            slider.set_val(value)
        if self.analysis.has_gas:
            for slider, value in zip(view.flash_sliders, state.initial_flash):
                slider.set_val(value)
        for slider in view.threshold_sliders:
            slider.set_val(state.default_contour_threshold)
        self.update_analysis()

    def save_calibration(self) -> bool:
        """Store the calibration; return if it was successful."""
        if self.save is None:
            logger.warning("No way to save the calibration provided.")
            return False
        try:
            self.save()
        except Exception:
            logger.exception("Saving the calibration failed.")
            return False
        self.state.mark_saved()
        logger.info("Saved the calibration.")
        return True

    def on_save(self, event) -> None:
        """Apply the current values and save; the window stays open.

        The parameters are applied before the update calculation starts, so the saved
        calibration does not depend on it.

        """
        if self._refuse():
            return
        view = self.view
        message = self.check_values()
        if message is not None:
            view.show_values_status(message)
            view.show_save_status("Not saved: fix the values first.", "red")
            return
        self.update_analysis()
        saved = self.save_calibration()
        view.show_save_status(
            "Saved." if saved else "Saving failed, see log.",
            "green" if saved else "red",
        )

    def on_close(self, event) -> None:
        """Closing the window ends the calibration; unsaved changes are saved or
        discarded, as the user decides."""
        if self.runner is not None:
            self.runner.wait()
        try:
            message = self.check_values() if self.pending else None
            if self.pending and message is None:
                self.apply_parameters()
            elif self.pending:
                logger.warning(f"{message}. The values of the sliders are not applied.")
            if self.state.has_unsaved_changes():
                keep = ask_yes_no(
                    "Unsaved calibration",
                    "The calibration has unsaved changes.",
                    yes="Save",
                    no="Discard",
                )
                # Without a dialog (e.g. a script), keep the changes
                if keep is None or keep:
                    self.save_calibration()
                else:
                    logger.info("Discarded the unsaved changes of the calibration.")
        except Exception:
            logger.exception("Closing the calibration window failed.")
