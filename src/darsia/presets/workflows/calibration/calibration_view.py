"""View of the interactive mass calibration: figure, widgets and drawing.

The window shows the current image and label, the signal, the phase segmentation, the
signal function of the label, the mass evolution of all images, and the controls
(signal values, flash, contour thresholds, resolution, buttons). The calculations live in
`CalibrationState` (`calibration/manual_calibration.py`); the callbacks are connected by
`HeterogeneousColorToMassAnalysis.manual_calibration`.

NOTE: The creation order of axes, artists, texts and widgets is checked by
`scripts/calib_reference.py` in ff_bilbo; keep it when changing the layout.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, Slider

import darsia
from darsia.signals.models.evaluation import curve_inverse, signal_curve
from darsia.utils.figure_window import maximize_figure_window
from darsia.utils.units import (
    choose_mass_unit,
    choose_time_unit,
    choose_volume_unit,
    convert_mass,
    convert_time,
    convert_volume,
    mass_label,
    time_label,
    volume_label,
)

if TYPE_CHECKING:
    from darsia.presets.workflows.calibration.manual_calibration import (
        CalibrationState,
    )
    from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
        HeterogeneousColorToMassAnalysis,
    )

UNSAVED_TEXT = "Changes not saved."
"""Save status while the shown values differ from the saved ones."""


def _dot_values(signal_func) -> np.ndarray:
    """Values shown as dots in the signal function plot.

    Key colors which do not influence the signal (unused supports of a kernel) are not
    shown (NaN).

    """
    values = np.asarray(signal_func.values, dtype=float).copy()
    used = getattr(signal_func, "support_indices", None)
    if used is not None:
        values[[i for i in range(len(values)) if i not in used]] = np.nan
    return values


def select_image(state: "CalibrationState") -> int:
    """Interactive selection of plotted images - allow to click for showing the next
    coarse images - keep track of the idx in the list and return when finishing.

    Parameters
    ----------
    state : CalibrationState
        State of the calibration; its current image is changed.

    Returns
    -------
    int
        Index of the selected image.

    """
    num_images = len(state.images)
    # Create a simple figure for image selection
    fig, ax = plt.subplots(figsize=(8, 6))
    img_display = ax.imshow(state.coarse_image.img)
    ax.set_title(
        f"""Image {state.image_idx + 1}/{num_images} - Click to cycle through """
        f"""images, close window to finish"""
    )
    ax.axis("off")

    def on_click(event):
        if event.inaxes == ax:
            # Cycle to next image
            state.next_image()
            img_display.set_data(state.coarse_image.img)
            ax.set_title(
                f"""Image {state.image_idx + 1}/{num_images} - Click to cycle """
                f"""through images, close window to finish"""
            )
            fig.canvas.draw_idle()

    # Connect click event
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()
    return state.image_idx


def select_label(state: "CalibrationState", cmap=None) -> int:
    """Interactive selection of the label: the user selects a rectangle in the image.

    Parameters
    ----------
    state : CalibrationState
        State of the calibration; its current label is changed. The maps of the current
        image must be computed.
    cmap : optional
        Colormap of the shown image.

    Returns
    -------
    int
        Index of the selected label.

    """
    assistant = darsia.RectangleSelectionAssistant(
        state.maps["pH"], labels=state.labels, cmap=cmap
    )
    label_box: tuple[slice, slice] = assistant()
    return state.select_label_in(label_box)


class CalibrationView:
    """Figure of the calibration window for the current image and label.

    The figure is built for one label (the number of value sliders follows its key
    colors); it shows the state and offers the widgets, but changes no parameters.

    Parameters
    ----------
    state : CalibrationState
        State of the calibration (data, maps, mass series).
    analysis : HeterogeneousColorToMassAnalysis
        The calibrated analysis; read only (signal functions, flash, units).
    cmap : optional
        Colormap of the signal panel.

    """

    def __init__(
        self,
        state: "CalibrationState",
        analysis: "HeterogeneousColorToMassAnalysis",
        cmap=None,
    ) -> None:
        self.state = state
        self.analysis = analysis
        self.cmap = cmap
        self.label_idx = state.label_idx
        """Label the figure is built for."""
        self.buttons: dict[str, Button] = {}
        """Buttons by their label."""

    # ! ---- BUILDING ----

    def build(self) -> None:
        """Create figure, panels, controls and buttons."""
        state = self.state
        # The figure is built for this image and label
        self.image_idx = state.image_idx
        coarse_image = state.coarse_image
        coarse_signal = state.coarse_signal
        coarse_contour = state.contour_overlay()
        self._build_panels(coarse_image, coarse_signal, coarse_contour)
        self._build_controls()
        self._build_buttons()

    def _build_panels(self, coarse_image, coarse_signal, coarse_contour) -> None:
        state, analysis, cmap = self.state, self.analysis, self.cmap
        image_idx, label_idx = self.image_idx, self.label_idx
        rois = state.rois

        # Use gridspec to create a 2x3 grid with proper ratios
        fig = plt.figure(figsize=(20, 12))
        self.fig = fig
        maximize_figure_window(fig)
        gs = gridspec.GridSpec(
            nrows=2,
            ncols=3,
            width_ratios=[1, 1, 1],  # Three equal columns
            height_ratios=[1, 1],  # Two equal rows
            left=0.05,
            right=0.95,
            bottom=0.08,
            top=0.92,
            wspace=0.2,
            hspace=0.25,
        )

        # Create the 2x3 grid of subplots
        # Top left: Coarse image
        ax_coarse_image = plt.subplot(gs[0, 0])
        ax_coarse_image.axis("off")
        ax_coarse_image.set_title(f"Current Image {image_idx} and Label {label_idx}")
        self.ax_coarse_image = ax_coarse_image

        # Top middle: Density visualization
        ax_signal = plt.subplot(gs[0, 1])
        ax_signal.axis("off")
        ax_signal.set_title(state.signal_title)
        self.ax_signal = ax_signal

        # Top right: Contour/Secondary visualization
        ax_contour = plt.subplot(gs[0, 2])
        ax_contour.axis("off")
        ax_contour.set_title(
            "Phase Segmentation" if analysis.has_gas else "Concentration Contours"
        )

        # Bottom left: Signal function plot
        ax_signal_function = plt.subplot(gs[1, 0])
        ax_signal_function.set_title(f"Signal Function - Label {label_idx}")
        self.ax_signal_function = ax_signal_function

        # Bottom middle: Mass evolution plots
        ax_mass = plt.subplot(gs[1, 1])
        ax_mass.set_title("Mass Evolution" if analysis.has_gas else "Volume Evolution")
        self.ax_mass = ax_mass

        # Bottom right: Parameter sliders (will be subdivided)
        ax_sliders = plt.subplot(gs[1, 2])
        ax_sliders.axis("off")  # Turn off axis for slider area
        ax_sliders.set_title("Parameter Controls")
        self.ax_sliders = ax_sliders

        # ! ---- PLOTTING ----

        # Plot the coarse image with highlighted label - turn the rest to gray
        self.coarse_image_img = ax_coarse_image.imshow(
            state.label_highlight(coarse_image, label_idx)
        )

        # Plot the coarse signal with rois
        # TODO
        coarse_signal_img = ax_signal.imshow(coarse_signal.img, cmap=cmap)
        self.coarse_signal_img = coarse_signal_img

        # Add colorbar
        signal_pos = ax_signal.get_position()
        colorbar_width = 0.01
        colorbar_height = signal_pos.height
        colorbar_x = signal_pos.x1 + 0.01
        colorbar_y = signal_pos.y0
        colorbar_ax = fig.add_axes(
            [colorbar_x, colorbar_y, colorbar_width, colorbar_height]
        )
        self.coarse_signal_colorbar = fig.colorbar(coarse_signal_img, cax=colorbar_ax)

        # Set initial color scale for the signal
        initial_signal_min, initial_signal_max = state.signal_range()
        coarse_signal_img.set_clim(vmin=initial_signal_min, vmax=initial_signal_max)
        self.coarse_signal_colorbar.update_normal(coarse_signal_img)  # Plot the contour
        self.coarse_contour_img = ax_contour.imshow(coarse_contour.img)

        # Plot the signal function for the selected label
        signal_func: darsia.PWTransformation = analysis.signal_model.model[1][label_idx]
        self.signal_function_line = None
        if hasattr(signal_func, "values") and len(signal_func.values) > 0:
            self._plot_signal_function(signal_func)

        # Combine plot and scatter for integrated mass over time.
        # Decompose into total, gas, and aqueous mass, and add
        # expected mass using a dashed line.

        # Determine a representative mass/time magnitude up front, so the
        # unit (fixed, or auto-chosen to keep displayed numbers readable)
        # is known before any series is drawn.
        max_expected = state.max_expected_mass()
        max_time = state.max_time()
        # Quantity compared with the injection protocol: the mass (CO2), or the
        # volume (tracer; the "mass" series below then hold volumes in m3).
        if analysis.has_gas:
            mass_unit = (
                choose_mass_unit(max_expected)
                if analysis.mass_unit == "auto"
                else analysis.mass_unit
            )
            convert_quantity, quantity_label = convert_mass, mass_label
            total_name = "total"
        else:
            mass_unit = (
                choose_volume_unit(max_expected)
                if analysis.volume_unit == "auto"
                else analysis.volume_unit
            )
            convert_quantity, quantity_label = convert_volume, volume_label
            total_name = "detected"
        time_unit = (
            choose_time_unit(max_time)
            if analysis.time_unit == "auto"
            else analysis.time_unit
        )
        ax_mass.set_xlabel(time_label(time_unit))
        ax_mass.set_ylabel(quantity_label(mass_unit))

        # Store plot objects for updates
        self.integrated_mass_plots = {}
        self.integrated_mass_plots_g = {}
        self.integrated_mass_plots_aq = {}
        self.expected_mass_plots = {}
        self.integrated_mass_scatters = {}
        self.integrated_mass_scatters_g = {}
        self.integrated_mass_scatters_aq = {}

        # Generate distinct colors for each ROI
        num_rois = len(rois)
        _cmap = matplotlib.colormaps["tab10" if num_rois <= 10 else "tab20"]
        roi_colors = [_cmap(i / max(1, num_rois - 1)) for i in range(num_rois)]

        # Plot each ROI as separate series
        for i, (roi_key, roi_times) in enumerate(state.times.items()):
            if len(roi_times) > 0:
                color = roi_colors[i % len(roi_colors)]
                roi_label = roi_key if roi_key else "entire_frame"
                plot_times = convert_time(roi_times, time_unit)

                # Plot lines
                [self.integrated_mass_plots[roi_key]] = ax_mass.plot(
                    plot_times,
                    convert_quantity(state.integrated_mass[roi_key], mass_unit),
                    color=color,
                    label=f"{total_name} ({roi_label})",
                    linewidth=2,
                )
                if analysis.has_gas:
                    [self.integrated_mass_plots_g[roi_key]] = ax_mass.plot(
                        plot_times,
                        convert_quantity(state.integrated_mass_g[roi_key], mass_unit),
                        color=color,
                        label=f"gas ({roi_label})",
                        linestyle=":",
                        linewidth=2,
                    )
                    [self.integrated_mass_plots_aq[roi_key]] = ax_mass.plot(
                        plot_times,
                        convert_quantity(state.integrated_mass_aq[roi_key], mass_unit),
                        color=color,
                        label=f"aqueous ({roi_label})",
                        linestyle="-.",
                        linewidth=2,
                    )
                [self.expected_mass_plots[roi_key]] = ax_mass.plot(
                    plot_times,
                    convert_quantity(state.expected_mass[roi_key], mass_unit),
                    linestyle="--",
                    color=color,
                    label=f"injected ({roi_label})",
                    alpha=0.7,
                )

                # Plot scatter points
                self.integrated_mass_scatters[roi_key] = ax_mass.scatter(
                    plot_times,
                    convert_quantity(state.integrated_mass[roi_key], mass_unit),
                    color=color,
                    s=20,
                    alpha=0.8,
                )
                if analysis.has_gas:
                    self.integrated_mass_scatters_g[roi_key] = ax_mass.scatter(
                        plot_times,
                        convert_quantity(state.integrated_mass_g[roi_key], mass_unit),
                        color=color,
                        s=15,
                        alpha=0.6,
                        marker="^",
                    )
                    self.integrated_mass_scatters_aq[roi_key] = ax_mass.scatter(
                        plot_times,
                        convert_quantity(state.integrated_mass_aq[roi_key], mass_unit),
                        color=color,
                        s=15,
                        alpha=0.6,
                        marker="s",
                    )

        # Set y-axis limits based on maximum expected mass across all ROIs
        if max_expected > 0:
            ax_mass.set_ylim(0.0, convert_quantity(2.0 * max_expected, mass_unit))
        else:
            ax_mass.autoscale()
        ax_mass.legend()
        ax_mass.grid(True, alpha=0.3)

    def _plot_signal_function(self, signal_func) -> None:
        """Signal function of the label, with the flash lines (gas phase)."""
        analysis = self.analysis
        ax_signal_function = self.ax_signal_function
        extents = analysis.signal_model_extents
        flash = analysis.flash

        # Get the color path interpretation for gradient background
        _color_path_interpretation: darsia.ColorPathInterpolation = (
            analysis.color_path_interpretation[self.label_idx]
        )

        # Create background gradient using the color path colormap
        x_gradient = np.linspace(
            extents[0][0],
            extents[0][1],
            256,
        ).reshape(1, -1)
        y_gradient = np.ones((10, 1))  # Height for visualization
        gradient_data = x_gradient * y_gradient

        # Get the colormap from the color path
        color_path_cmap = _color_path_interpretation.color_path.get_color_map()

        # Display the gradient as background
        ax_signal_function.imshow(
            gradient_data,
            extent=[
                extents[0][0],
                extents[0][1],
                extents[1][0],
                extents[1][1],
            ],
            aspect="auto",
            cmap=color_path_cmap,
            alpha=0.3,  # Semi-transparent
            zorder=0,  # Behind everything else
        )

        # Plot the signal function evaluated on a fine grid along the color
        # path, with the values at the key colors as dots
        x_vals = signal_func.supports
        t_fine, y_fine = signal_curve(signal_func)
        [self.signal_function_line] = ax_signal_function.plot(
            t_fine, y_fine, "gray", linewidth=2, zorder=2
        )

        colors = _color_path_interpretation.color_path.colors
        facecolors = to_rgba_array(np.clip(colors, 0, 1))
        self.signal_function_scatter = ax_signal_function.scatter(
            x_vals,
            _dot_values(signal_func),
            c=facecolors,
            zorder=3,
            s=50,
            edgecolors="black",
            linewidth=1,
        )

        if analysis.has_gas:
            # Indicate cut-off and max value from flash model
            self.signal_function_aq_min_y = ax_signal_function.axhline(
                y=flash.min_value_aq,
                color="k",
                linestyle="--",
                label="aq_min",
                zorder=1,
            )
            self.signal_function_aq_max_y = ax_signal_function.axhline(
                y=flash.max_value_aq,
                color="k",
                linestyle="--",
                label="aq_max",
                zorder=1,
            )
            self.signal_function_g_max_y = ax_signal_function.axhline(
                y=flash.max_value_g,
                color="k",
                linestyle="--",
                label="g_max",
                zorder=1,
            )

            self.signal_function_g_min_y = ax_signal_function.axhline(
                y=flash.min_value_g,
                color="k",
                linestyle="--",
                label="g_min",
                zorder=1,
            )

            # Find the corresponding x-values for the cut-off and max value lines as
            # solution to the signal_function(x) = cut_off and
            # signal_function(x) = max_value
            aq_min_x = curve_inverse(t_fine, y_fine, flash.min_value_aq)
            aq_max_x = curve_inverse(t_fine, y_fine, flash.max_value_aq)
            g_max_x = curve_inverse(t_fine, y_fine, flash.max_value_g)
            g_min_x = curve_inverse(t_fine, y_fine, flash.min_value_g)

            # Add vertical lines for cut-off and max value
            self.signal_function_aq_min_x = ax_signal_function.axvline(
                x=aq_min_x,
                color="k",
                linestyle="--",
                label="min value aq",
                zorder=1,
            )
            self.signal_function_aq_max_x = ax_signal_function.axvline(
                x=aq_max_x,
                color="k",
                linestyle="--",
                label="max value aq",
                zorder=1,
            )
            self.signal_function_g_max_x = ax_signal_function.axvline(
                x=g_max_x,
                color="k",
                linestyle="--",
                label="max value g",
                zorder=1,
            )
            self.signal_function_g_min_x = ax_signal_function.axvline(
                x=g_min_x,
                color="k",
                linestyle="--",
                label="min value g",
                zorder=1,
            )

            # Annotate "CO2(aq)" and "CO2(g)" below and above the cut-off
            ax_signal_function.text(
                x=aq_max_x - 0.05,
                y=0.04,
                s="CO2(aq)",
                ha="center",
                va="center",
                fontsize=8,
                color="k",
            )
            ax_signal_function.text(
                x=aq_max_x + 0.05,
                y=0.04,
                s="CO2(g)",
                ha="center",
                va="center",
                fontsize=8,
                color="k",
            )
            ax_signal_function.text(
                x=g_max_x + 0.1,
                y=0.04,
                s="CO2(g) - 100%",
                ha="center",
                va="center",
                fontsize=8,
                color="k",
            )
        # Single-phase (tracer): no phase transition, hence no flash lines;
        # the signal is the concentration itself.
        # NOTE: Keep after the annotations above, which are accessed by index
        used_indices = getattr(signal_func, "support_indices", None)
        if used_indices is not None and len(used_indices) < len(signal_func.values):
            ax_signal_function.text(
                x=0.01,
                y=0.99,
                s="Unused key colors (no dot): the signal is affine in the "
                "color and bends at every key color.",
                transform=ax_signal_function.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="k",
                zorder=4,
            )

        ax_signal_function.set_xlabel("Color Interpretation")
        ax_signal_function.set_ylabel("Signal Value")
        ax_signal_function.grid(True, alpha=0.3, zorder=1)
        ax_signal_function.set_xlim(extents[0][0], extents[0][1])
        ax_signal_function.set_ylim(extents[1][0], extents[1][1])

    def _build_controls(self) -> None:
        state, analysis, fig = self.state, self.analysis, self.fig
        label_idx = self.label_idx
        contour_thresholds = state.contour_thresholds

        # ! ---- PARAMETER CONTROLS ----

        # Grouped rows (label | slider | value box). The number of signal value
        # rows follows the number of key colors of the selected label.
        controls_box = self.ax_sliders.get_position()
        self._controls_box = controls_box
        signal_values = analysis.signal_model.model[1][label_idx].values
        used_values = getattr(
            analysis.signal_model.model[1][label_idx],
            "support_indices",
            range(len(signal_values)),
        )
        path_colors = analysis.color_path_interpretation[label_idx].color_path.colors
        num_values = len(signal_values)
        # Rows: signal values (title, values), flash (title, 4 sliders; CO2 only),
        # contours (title, one threshold per phase), resolution (title, level)
        levels = state.available_levels
        num_rows = num_values + (9 if analysis.has_gas else 3)
        if len(levels) > 1:
            num_rows += 2
        row_height = controls_box.height / num_rows

        def make_row(row, name, valinit, vmin, vmax, color, active=True, valstep=0.01):
            """Create a labeled slider (with value text) in a row."""
            x0, width = controls_box.x0, controls_box.width
            y0 = controls_box.y1 - (row + 1) * row_height
            label_text = fig.text(
                x0, y0 + 0.5 * row_height, name, fontsize=8, ha="left", va="center"
            )
            ax_row_slider = fig.add_axes(
                [
                    x0 + 0.30 * width,
                    y0 + 0.2 * row_height,
                    0.55 * width,
                    0.6 * row_height,
                ]
            )
            slider = Slider(
                ax_row_slider,
                "",
                vmin,
                vmax,
                valinit=valinit,
                valstep=valstep,
                color=color,
            )
            slider.valtext.set_fontsize(8)
            if not active:
                # Values without influence cannot be changed and appear faded
                slider.set_active(False)
                for artist in (
                    label_text,
                    slider.poly,
                    slider._handle,
                    slider.valtext,
                ):
                    artist.set_alpha(0.35)
            return slider

        def section_title(row, title):
            return fig.text(
                controls_box.x0,
                controls_box.y1 - (row + 0.5) * row_height,
                title,
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
            )

        section_title(0, f"Signal values (label {label_idx})")
        self.value_sliders = [
            make_row(
                1 + i,
                f"Value {i}" + ("" if i in used_values else " (unused)"),
                val,
                0.0,
                2.0 if analysis.has_gas else 1.0,
                np.clip(path_colors[i], 0, 1) if i < len(path_colors) else "gray",
                active=i in used_values,
            )
            for i, val in enumerate(signal_values)
        ]

        flash_row = num_values + 1
        if analysis.has_gas:
            section_title(flash_row, "Flash")
            self.flash_sliders = [
                make_row(flash_row + 1 + i, name, value, 0.0, 2.0, "darkgray")
                for i, (name, value) in enumerate(
                    [
                        ("Min aq", analysis.flash.min_value_aq),
                        ("Max aq", analysis.flash.max_value_aq),
                        ("Min g", analysis.flash.min_value_g),
                        ("Max g", analysis.flash.max_value_g),
                    ]
                )
            ]
            contour_row = flash_row + 5
        else:
            # Single-phase: no flash parameters, the signal is the concentration
            self.flash_sliders = []
            contour_row = flash_row
        section_title(contour_row, "Contours")
        self.threshold_sliders = [
            make_row(
                contour_row + 1,
                "Threshold c_aq",
                contour_thresholds[0],
                0.0,
                1.0,
                (1, 0, 0),
            )
        ]
        if analysis.has_gas:
            self.threshold_sliders.append(
                make_row(
                    contour_row + 2,
                    "Threshold s_g",
                    contour_thresholds[1],
                    0.0,
                    1.0,
                    (0, 1, 0),
                )
            )

        for i, slider in enumerate(self.threshold_sliders):
            slider.on_changed(lambda val, i=i: contour_thresholds.__setitem__(i, val))

        # Slider updates only repaint the sliders (blitting). Otherwise, every
        # slider step re-renders the entire figure, including all images. The
        # moving parts of the sliders are animated artists, which are excluded
        # from the stored background and drawn on top of it.
        self.all_sliders = (
            self.value_sliders + self.flash_sliders + self.threshold_sliders
        )

        # Resolution of the calculations: a discrete slider over the levels. Like
        # the parameters, the selected level is applied by the Update button.
        self.resolution_slider = None
        self.resolution_status = None
        if len(levels) > 1:
            resolution_row = contour_row + 1 + len(self.threshold_sliders)

            self.resolution_status = section_title(
                resolution_row, self._resolution_title()
            )
            slider_resolution = make_row(
                resolution_row + 1,
                "Level",
                levels.index(state.resolution_level),
                0,
                len(levels) - 1,
                "steelblue",
                valstep=1,
            )
            # Show the level instead of its position
            slider_resolution._format = lambda val: f"1/{levels[int(round(val))]}"
            slider_resolution.valtext.set_text(
                slider_resolution._format(slider_resolution.val)
            )
            self.resolution_slider = slider_resolution

        blit_sliders = self.all_sliders + (
            [] if self.resolution_slider is None else [self.resolution_slider]
        )
        self._animated_artists = [
            artist
            for slider in blit_sliders
            for artist in (slider.poly, slider._handle, slider.valtext)
        ]
        for artist in self._animated_artists:
            artist.set_animated(True)
        self._background = None

        fig.canvas.mpl_connect("draw_event", self._store_background)

        for slider in blit_sliders:
            slider.drawon = False
            slider.on_changed(lambda val, slider=slider: self._blit_slider(slider))

    def _build_buttons(self) -> None:
        fig = self.fig
        controls_box = self._controls_box

        # Position buttons at the top of the figure
        button_width = 0.08
        button_height = 0.04
        button_y = 0.95
        button_spacing = 0.01

        # Button order from left to right (Reset, Update at bottom right):
        # - Next Signal
        # - Next Label
        # - New Label
        # - Next Image
        # - New Image
        # - Save
        # Calculate positions for 6 buttons with proper spacing
        total_buttons = 5
        button_start_x = (
            0.98 - total_buttons * button_width - (total_buttons - 1) * button_spacing
        )

        # Position Update button at bottom right corner
        update_button_x = 0.98 - button_width  # Right side with small margin
        update_button_y = 0.02  # Bottom with small margin

        # Add "Reset" button to the left of "Update"
        reset_button_x = update_button_x - button_width - button_spacing
        ax_reset = plt.axes(
            [reset_button_x, update_button_y, button_width, button_height]
        )
        self.buttons["Reset"] = Button(ax_reset, "Reset")

        # Add "Update" button
        ax_update = plt.axes(
            [
                update_button_x,
                update_button_y,
                button_width,
                button_height,
            ]
        )
        self.buttons["Update"] = Button(ax_update, "Update")
        self.busy_status = fig.text(
            update_button_x + button_width,
            update_button_y + button_height + 0.004,
            "",
            fontsize=8,
            color="darkorange",
            ha="right",
            va="bottom",
        )
        """Message shown while a calculation blocks the window."""
        self.values_status = fig.text(
            controls_box.x0,
            update_button_y + 0.5 * button_height,
            "",
            fontsize=8,
            color="red",
            ha="left",
            va="center",
        )
        """Message about invalid (decreasing) signal values."""

        for position, name in enumerate(
            ["Next Signal", "Next Label", "New Label", "Next Image", "New Image"]
        ):
            ax_button = plt.axes(
                [
                    button_start_x + position * (button_width + button_spacing),
                    button_y,
                    button_width,
                    button_height,
                ]
            )
            self.buttons[name] = Button(ax_button, name)

        save_button_x = 0.02  # Left side with small margin
        ax_save = plt.axes(
            [
                save_button_x,
                button_y,
                button_width,
                button_height,
            ]
        )
        self.buttons["Save"] = Button(ax_save, "Save")
        self.save_status = fig.text(
            save_button_x + button_width + button_spacing,
            button_y + 0.5 * button_height,
            "",
            fontsize=9,
            color="green",
            va="center",
        )

    # ! ---- SLIDERS ----

    def _resolution_title(self) -> str:
        return f"Resolution (in use: 1/{self.state.resolution_level})"

    def set_resolution_title(self) -> None:
        """Show the resolution level in use."""
        if self.resolution_status is not None:
            self.resolution_status.set_text(self._resolution_title())

    def selected_level(self) -> int:
        """Resolution level chosen with the slider."""
        if self.resolution_slider is None:
            return self.state.resolution_level
        levels = self.state.available_levels
        return levels[int(round(self.resolution_slider.val))]

    def values(self) -> list[float]:
        """Signal values of the sliders."""
        return [slider.val for slider in self.value_sliders]

    def flash_values(self) -> list[float]:
        """Flash parameters of the sliders (empty without a gas phase)."""
        return [slider.val for slider in self.flash_sliders]

    def _store_background(self, event) -> None:
        fig = self.fig
        self._background = fig.canvas.copy_from_bbox(fig.bbox)
        for artist in self._animated_artists:
            fig.draw_artist(artist)

    def _blit_slider(self, slider) -> None:
        fig = self.fig
        if self._background is None or not fig.canvas.supports_blit:
            fig.canvas.draw_idle()
            return
        handle_radius = 0.5 * slider._handle.get_markersize() * fig.dpi / 72
        region = Bbox.union(
            [slider.ax.bbox, slider.valtext.get_window_extent()]
        ).padded(handle_radius + 1)
        fig.canvas.restore_region(self._background)
        for artist in self._animated_artists:
            fig.draw_artist(artist)
        fig.canvas.blit(region)

    # ! ---- DRAWING ----

    def update_signal_function_plot(self) -> None:
        """Update the signal function plot (curve, flash lines)."""
        analysis = self.analysis
        flash = analysis.flash
        ax_signal_function = self.ax_signal_function
        signal_func = analysis.signal_model.model[1][self.label_idx]
        t_fine, y_fine = signal_curve(signal_func)
        # Update signal function plot
        if self.signal_function_line is not None:
            x_vals = signal_func.supports
            y_vals = signal_func.values.copy()
            self.signal_function_line.set_data(t_fine, y_fine)
            self.signal_function_scatter.set_offsets(
                np.c_[x_vals, _dot_values(signal_func)]
            )

            # Rescale y-axis to fit the curve, floored at the
            # current flash max (never zoom in tighter than the
            # flash thresholds, but grow if the curve exceeds
            # them)
            flash_max = (
                max(flash.max_value_aq, flash.max_value_g) if analysis.has_gas else 1.0
            )
            ax_signal_function.set_ylim(
                analysis.signal_model_extents[1][0],
                max(
                    flash_max,
                    float(np.max(y_fine)),
                    float(np.max(y_vals)),
                ),
            )

        if analysis.has_gas:
            # Update flash cut-off and max value lines
            self.signal_function_aq_min_y.set_ydata([flash.min_value_aq])
            self.signal_function_aq_max_y.set_ydata([flash.max_value_aq])
            self.signal_function_g_max_y.set_ydata([flash.max_value_g])
            self.signal_function_g_min_y.set_ydata([flash.min_value_g])
            aq_max_x = curve_inverse(t_fine, y_fine, flash.max_value_aq)
            aq_min_x = curve_inverse(t_fine, y_fine, flash.min_value_aq)
            g_max_x = curve_inverse(t_fine, y_fine, flash.max_value_g)
            g_min_x = curve_inverse(t_fine, y_fine, flash.min_value_g)
            self.signal_function_aq_max_x.set_xdata([aq_max_x])
            self.signal_function_aq_min_x.set_xdata([aq_min_x])
            self.signal_function_g_max_x.set_xdata([g_max_x])
            self.signal_function_g_min_x.set_xdata([g_min_x])

            # Update the position of the annotaion
            ax_signal_function.texts[0].set_position((aq_max_x - 0.05, 0.04))
            ax_signal_function.texts[1].set_position((aq_max_x + 0.05, 0.04))
            ax_signal_function.texts[2].set_position((g_max_x + 0.1, 0.04))

    def update_mass_plot(self) -> None:
        """Update the mass evolution plots."""
        state, analysis = self.state, self.analysis
        # Update mass plots
        max_expected = state.max_expected_mass()
        max_time = state.max_time()
        if analysis.has_gas:
            mass_unit = (
                choose_mass_unit(max_expected)
                if analysis.mass_unit == "auto"
                else analysis.mass_unit
            )
            convert_quantity, quantity_label = convert_mass, mass_label
        else:
            mass_unit = (
                choose_volume_unit(max_expected)
                if analysis.volume_unit == "auto"
                else analysis.volume_unit
            )
            convert_quantity, quantity_label = (
                convert_volume,
                volume_label,
            )
        time_unit = (
            choose_time_unit(max_time)
            if analysis.time_unit == "auto"
            else analysis.time_unit
        )
        for roi_key in state.rois:
            plot_times = convert_time(state.times[roi_key], time_unit)
            self.integrated_mass_plots[roi_key].set_data(
                plot_times,
                convert_quantity(state.integrated_mass[roi_key], mass_unit),
            )
            if analysis.has_gas:
                self.integrated_mass_plots_g[roi_key].set_data(
                    plot_times,
                    convert_quantity(state.integrated_mass_g[roi_key], mass_unit),
                )
                self.integrated_mass_plots_aq[roi_key].set_data(
                    plot_times,
                    convert_quantity(state.integrated_mass_aq[roi_key], mass_unit),
                )
            self.integrated_mass_scatters[roi_key].set_offsets(
                np.c_[
                    plot_times,
                    convert_quantity(state.integrated_mass[roi_key], mass_unit),
                ]
            )
            if analysis.has_gas:
                self.integrated_mass_scatters_g[roi_key].set_offsets(
                    np.c_[
                        plot_times,
                        convert_quantity(state.integrated_mass_g[roi_key], mass_unit),
                    ]
                )
                self.integrated_mass_scatters_aq[roi_key].set_offsets(
                    np.c_[
                        plot_times,
                        convert_quantity(state.integrated_mass_aq[roi_key], mass_unit),
                    ]
                )
        if max_expected > 0:
            self.ax_mass.set_ylim(0.0, convert_quantity(2.0 * max_expected, mass_unit))
        else:
            self.ax_mass.autoscale()
        self.ax_mass.set_xlabel(time_label(time_unit))
        self.ax_mass.set_ylabel(quantity_label(mass_unit))

    def refresh(
        self,
        recompute: bool = True,
        signal_function: bool = False,
        mass_plot: bool = False,
    ) -> None:
        """Update the displayed panels.

        Parameters
        ----------
        recompute : bool
            Analyze the current image anew (after changes of the
            parameters or the image); otherwise reuse the last maps.
        signal_function : bool
            Also update the signal function plot.
        mass_plot : bool
            Also update the mass evolution plot.

        """
        state = self.state
        if recompute:
            state.compute_maps()

        # Image panel and signal panel
        self.ax_coarse_image.set_title(
            f"Current Image {state.image_idx} and Label {self.label_idx}"
        )
        self.coarse_image_img.set_data(
            state.label_highlight(state.coarse_image, self.label_idx)
        )
        self.coarse_signal_img.set_data(state.coarse_signal.img)
        self.ax_signal.set_title(state.signal_title)
        signal_min, signal_max = state.signal_range()
        self.coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
        self.coarse_signal_colorbar.update_normal(self.coarse_signal_img)

        # Phase segmentation panel
        self.coarse_contour_img.set_data(
            state.contour_overlay([slider.val for slider in self.threshold_sliders]).img
        )

        if signal_function and self.signal_function_line is not None:
            self.update_signal_function_plot()
        if mass_plot:
            self.update_mass_plot()
        self.fig.canvas.draw_idle()

    # ! ---- MESSAGES ----

    def show_values_status(self, message: str | None) -> None:
        """Show (or clear) the message about invalid signal values."""
        text = "" if message is None else message.replace(": ", ":\n", 1)
        if self.values_status.get_text() != text:
            self.values_status.set_text(text)
            self.fig.canvas.draw_idle()

    def mark_unsaved(self, *args) -> None:
        """Show that the displayed values differ from the saved ones."""
        if self.save_status.get_text() != UNSAVED_TEXT:
            self.save_status.set_text(UNSAVED_TEXT)
            self.save_status.set_color("darkorange")
            self.fig.canvas.draw_idle()

    def show_save_status(self, text: str, color: str) -> None:
        """Show the result of saving."""
        self.save_status.set_text(text)
        self.save_status.set_color(color)
        self.fig.canvas.draw_idle()

    def show_busy(self, message: str) -> None:
        """Show (or clear) a message and draw it right away.

        The calculations run in the window's event loop, so the
        window does not react until they are done.

        """
        if self.busy_status.get_text() == message:
            return
        self.busy_status.set_text(message)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
