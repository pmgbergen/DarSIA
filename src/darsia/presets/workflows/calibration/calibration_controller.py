"""Controller of the interactive mass calibration: selection, callbacks and loops.

The controller connects the window (`CalibrationView`) with the calculations
(`CalibrationState`): it lets the user pick an image and a label, builds the window for
them, applies the slider values, saves, and decides what happens when the window is
closed.

NOTE: The order in which the callbacks are connected is checked by
`scripts/calib_reference.py` in ff_bilbo; keep it when changing the window.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt

from darsia.presets.workflows.calibration.calibration_view import (
    CalibrationView,
    select_image,
    select_label,
)
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
        Colormap of the signal panel and the label selection.
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
        self.cmap = cmap
        self.save = save
        self.view: CalibrationView | None = None
        """Window of the current image and label."""
        self.done = False
        """Whether the calibration is finished (window closed)."""
        self.pick_image = True
        """Whether the user picks a new image before the next window."""
        self.pick_label = True
        """Whether the user picks a new label before the next window."""
        self.rebuilding = False
        """Whether the window is closed to be rebuilt, not to leave."""
        self.done_tuning = False
        """Whether the current window is finished."""
        self.pending = False
        """Whether a value or flash slider was moved since the last applied values.

        The sliders round to their step, so their untouched values are not applied (they
        would change the stored calibration).

        """

    def run(self) -> bool:
        """Run the calibration until the window is closed.

        Returns
        -------
        bool
            Whether the stored calibration equals the calibration at the end.

        """
        state = self.state
        while not self.done:
            # General design:
            # 1. User needs to pick a focus image (interactively)
            # 2. User picks a label in the image (interactively)
            # 3. Mass analysis is computed for current settings
            # 4. User tunes values for the color path for this label (interactively)
            # 5. User can control to rerun any of the above steps through buttons.

            # Pick image
            if self.pick_image:
                select_image(state)
                self.pick_image = False
                self.pick_label = True

            # Fetch the current image and analyze it
            if state.maps is None or state.maps["image_idx"] != state.image_idx:
                state.compute_maps()

            # Pick label interactively
            if self.pick_label:
                select_label(state, self.cmap)
                self.pick_label = False

            # Compute mass analysis
            if not state.mass_analysis_done:
                state.update_mass_analysis()

            # Figure for this image and label
            self.view = CalibrationView(state, self.analysis, self.cmap)
            self.view.build()
            self.rebuilding = False

            # Tune values for this label
            self.done_tuning = False
            while not self.done_tuning:
                self._connect()
                plt.show()

        return not state.has_unsaved_changes()

    def _connect(self) -> None:
        """Connect the callbacks of the window."""
        view = self.view
        for slider in view.value_sliders:
            slider.on_changed(lambda val: view.show_values_status(self.check_values()))

        self.pending = False
        for slider in view.value_sliders + view.flash_sliders:
            slider.on_changed(self.on_value_slider)

        for slider in view.all_sliders:
            slider.on_changed(view.mark_unsaved)

        # Connect button callbacks
        view.buttons["Update"].on_clicked(self.update_analysis)
        view.buttons["Reset"].on_clicked(self.reset)
        view.buttons["New Label"].on_clicked(self.new_label)
        view.buttons["New Image"].on_clicked(self.new_image)
        view.buttons["Save"].on_clicked(self.on_save)
        view.fig.canvas.mpl_connect("close_event", self.on_close)
        view.buttons["Next Signal"].on_clicked(self.next_signal)
        view.buttons["Next Label"].on_clicked(self.next_label)
        view.buttons["Next Image"].on_clicked(self.next_image)

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

    # ! ---- BUTTONS ----

    def update_analysis(self, event=None) -> None:
        """Apply the slider values and update everything."""
        state, view = self.state, self.view
        changed = self.apply_parameters()
        if changed is None:
            return
        level = view.selected_level()
        new_level = level != state.resolution_level
        if changed:
            view.mark_unsaved()
        if changed or new_level:
            view.show_busy(
                f"Computing at resolution 1/{level} ..."
                if new_level
                else "Computing ..."
            )
        if new_level:
            state.set_resolution(level)
            view.set_resolution_title()
        if changed or new_level:
            state.update_mass_analysis()
        view.refresh(recompute=True, signal_function=True, mass_plot=True)
        view.show_busy("")

    def reset(self, event) -> None:
        """Restore the initial values of this label, flash, contours."""
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

    def new_label(self, event) -> None:
        """Let the user pick a new label and rebuild the figure for it."""
        self.done_tuning = True
        self.pick_label = True
        self.rebuilding = True
        plt.close("all")

    def new_image(self, event) -> None:
        """Let the user pick a new image and label and rebuild the figure."""
        self.done_tuning = True
        self.pick_image = True
        self.rebuilding = True
        plt.close("all")

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
        """Apply the current values and save; the window stays open."""
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
        if self.rebuilding:
            return
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
        finally:
            self.done_tuning = True
            self.done = True

    def next_signal(self, event) -> None:
        """Show the next signal of the current image."""
        self.state.next_signal()
        self.view.refresh(recompute=False)

    def next_label(self, event) -> None:
        """Move to the next label and rebuild the figure for it."""
        self.state.next_label()
        self.pick_label = False
        self.done_tuning = True
        self.rebuilding = True
        plt.close("all")

    def next_image(self, event) -> None:
        """Analyze and show the next image."""
        self.view.show_busy("Computing ...")
        self.state.next_image()
        self.view.refresh(recompute=True)
        self.view.show_busy("")
