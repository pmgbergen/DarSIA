"""View of the interactive mass calibration: one window, updated in place.

The window shows the current image and label, the signal, the phase segmentation, the
signal function of the label, the mass evolution of all images, and the controls
(signal values, flash, contour thresholds, resolution, buttons). It is built once; a
change of the label or the image updates it. The calculations live in `CalibrationState`
(`calibration/manual_calibration.py`); the callbacks are connected by
`CalibrationController` (`calibration/calibration_controller.py`).

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array
from matplotlib.widgets import Button, Slider

from darsia.presets.workflows.windows.blitting import SliderBlitter, StatusText
from darsia.presets.workflows.windows.screen_image import ScreenImage
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

FADED = 0.35
"""Transparency of slider rows whose values have no influence."""


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


class FlashMarkers:
    """Bounds of the flash in the signal function plot.

    Horizontal lines at the minimal and maximal signal of the aqueous and the gas ramp,
    vertical lines where the signal function reaches them, and annotations of the
    phases.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the signal function plot.

    """

    NAMES = ("aq_min", "aq_max", "g_max", "g_min")
    """Order of the bounds (lines)."""

    def __init__(self, ax) -> None:
        style = {"color": "k", "linestyle": "--", "zorder": 1}
        self.horizontal = [
            ax.axhline(y=0.0, label=name, **style) for name in self.NAMES
        ]
        self.vertical = [
            ax.axvline(x=0.0, label=f"{name} x", **style) for name in self.NAMES
        ]
        self.lines = self.horizontal + self.vertical
        """All lines: horizontal, then vertical, each in the order of `NAMES`."""
        text_style = {"ha": "center", "va": "center", "fontsize": 8, "color": "k"}
        self.texts = [
            ax.text(x=0.0, y=0.04, s=s, **text_style)
            for s in ("CO2(aq)", "CO2(g)", "CO2(g) - 100%")
        ]
        """Annotations: dissolved below the cut-off, gas above, gas only."""

    def update(self, flash, t_fine: np.ndarray, y_fine: np.ndarray) -> None:
        """Place the markers for a flash and a signal function curve.

        Parameters
        ----------
        flash : darsia.SimpleFlash
            Flash with the bounds.
        t_fine, y_fine : np.ndarray
            Signal function on a fine grid along the color path.

        """
        bounds = (
            flash.min_value_aq,
            flash.max_value_aq,
            flash.max_value_g,
            flash.min_value_g,
        )
        # Where the signal function reaches the bounds
        positions = [curve_inverse(t_fine, y_fine, bound) for bound in bounds]
        for line, bound in zip(self.horizontal, bounds):
            line.set_ydata([bound])
        for line, position in zip(self.vertical, positions):
            line.set_xdata([position])
        aq_max_x, g_max_x = positions[1], positions[2]
        self.texts[0].set_position((aq_max_x - 0.05, 0.04))
        self.texts[1].set_position((aq_max_x + 0.05, 0.04))
        self.texts[2].set_position((g_max_x + 0.1, 0.04))


class MassSeries:
    """Integrated mass over time per region: lines and points per component.

    Components: total (detected for a tracer), gas, aqueous (gas phase only), and the
    injected mass from the protocol (line only).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the mass evolution plot.
    state : CalibrationState
        State with the mass series.
    has_gas : bool
        Whether the fluid has a gas phase.
    units : tuple
        Display units, see `CalibrationView._display_units`.

    """

    def __init__(self, ax, state: "CalibrationState", has_gas: bool, units) -> None:
        self.state = state
        self.has_gas = has_gas
        self.lines: dict[str, dict] = {}
        """Line per region and component."""
        self.points: dict[str, dict] = {}
        """Points (scatter) per region and component."""

        total_name = "total" if has_gas else "detected"
        num_rois = len(state.rois)
        cmap = matplotlib.colormaps["tab10" if num_rois <= 10 else "tab20"]
        colors = [cmap(i / max(1, num_rois - 1)) for i in range(num_rois)]
        components = ["total", "gas", "aqueous"] if has_gas else ["total"]
        line_styles = {
            "total": {"label": total_name, "linewidth": 2},
            "gas": {"label": "gas", "linestyle": ":", "linewidth": 2},
            "aqueous": {"label": "aqueous", "linestyle": "-.", "linewidth": 2},
            "injected": {"label": "injected", "linestyle": "--", "alpha": 0.7},
        }
        point_styles = {
            "total": {"s": 20, "alpha": 0.8},
            "gas": {"s": 15, "alpha": 0.6, "marker": "^"},
            "aqueous": {"s": 15, "alpha": 0.6, "marker": "s"},
        }
        for i, (roi, roi_times) in enumerate(state.times.items()):
            if len(roi_times) == 0:
                continue
            color = colors[i % len(colors)]
            roi_label = roi if roi else "entire_frame"
            times, series = self._data(roi, units)
            self.lines[roi] = {}
            for component in components + ["injected"]:
                style = dict(line_styles[component])
                style["label"] = f"{style['label']} ({roi_label})"
                [self.lines[roi][component]] = ax.plot(
                    times, series[component], color=color, **style
                )
            self.points[roi] = {
                component: ax.scatter(
                    times, series[component], color=color, **point_styles[component]
                )
                for component in components
            }

    def _data(self, roi: str, units) -> tuple[np.ndarray, dict]:
        """Times and values of all components of a region in display units."""
        state = self.state
        mass_unit, convert_quantity, _, time_unit = units
        series = {
            "total": state.integrated_mass[roi],
            "injected": state.expected_mass[roi],
        }
        if self.has_gas:
            series["gas"] = state.integrated_mass_g[roi]
            series["aqueous"] = state.integrated_mass_aq[roi]
        times = convert_time(state.times[roi], time_unit)
        return times, {
            key: convert_quantity(value, mass_unit) for key, value in series.items()
        }

    def update(self, units) -> None:
        """Show the current mass series.

        Parameters
        ----------
        units : tuple
            Display units, see `CalibrationView._display_units`.

        """
        for roi, lines in self.lines.items():
            times, series = self._data(roi, units)
            for component, line in lines.items():
                line.set_data(times, series[component])
            for component, points in self.points[roi].items():
                points.set_offsets(np.c_[times, series[component]])

    def artists(self) -> dict:
        """All lines and points by "region/component/kind"."""
        artists = {}
        for roi, lines in self.lines.items():
            for component, line in lines.items():
                artists[f"{roi}/{component}/line"] = line
            for component, points in self.points[roi].items():
                artists[f"{roi}/{component}/points"] = points
        return artists


class CalibrationView:
    """Window of the calibration: built once, updated for each label and image.

    The view shows the state and offers the widgets, but changes no parameters.

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
        """Label shown."""
        self.buttons: dict[str, Button] = {}
        """Buttons by their label."""

    @property
    def labels(self) -> list:
        """Labels with a signal function."""
        return sorted(
            label for label in self.analysis.signal_model.model[1].keys() if label >= 0
        )

    def signal_function(self, label: int):
        """Signal function of a label."""
        return self.analysis.signal_model.model[1][label]

    # ! ---- BUILDING ----

    def build(self) -> None:
        """Create figure, panels, controls and buttons for the current label."""
        self._build_panels()
        self._build_controls()
        self._build_buttons()

    def _build_panels(self) -> None:
        state, analysis = self.state, self.analysis

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

        # Top left: image with the selected label highlighted
        self.ax_coarse_image = plt.subplot(gs[0, 0])
        self.ax_coarse_image.axis("off")
        self.coarse_image_img = ScreenImage(
            self.ax_coarse_image,
            state.label_highlight(state.coarse_image, self.label_idx),
        )
        self._shown_image = (state.image_idx, self.label_idx)
        """Image and label shown in the image panel."""
        image_box = self.ax_coarse_image.get_position()
        fig.text(
            image_box.x0,
            image_box.y0 - 0.01,
            "Click in any top panel to select a label.",
            fontsize=8,
            ha="left",
            va="top",
        )

        # Top middle: signal with colorbar
        self.ax_signal = plt.subplot(gs[0, 1])
        self.ax_signal.axis("off")
        self.ax_signal.set_title(state.signal_title)
        self.coarse_signal_img = ScreenImage(
            self.ax_signal, state.coarse_signal.img, cmap=self.cmap
        )
        signal_box = self.ax_signal.get_position()
        colorbar_ax = fig.add_axes(
            [signal_box.x1 + 0.01, signal_box.y0, 0.01, signal_box.height]
        )
        self.coarse_signal_colorbar = fig.colorbar(
            self.coarse_signal_img.image, cax=colorbar_ax
        )
        signal_min, signal_max = state.signal_range()
        self.coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
        self.coarse_signal_colorbar.update_normal(self.coarse_signal_img.image)

        # Top right: phase segmentation (CO2) or concentration contours (tracer)
        ax_contour = plt.subplot(gs[0, 2])
        ax_contour.axis("off")
        ax_contour.set_title(
            "Phase Segmentation" if analysis.has_gas else "Concentration Contours"
        )
        self.ax_contour = ax_contour
        self._shown_contour = state.contour_overlay()
        """Contour overlay shown in the contour panel."""
        self.coarse_contour_img = ScreenImage(ax_contour, self._shown_contour.img)

        # The top row shows the same coarse grid: zoom and pan apply to all three, and
        # a click in any of them selects a label
        self.top_row_axes = (self.ax_coarse_image, self.ax_signal, ax_contour)
        """Panels of the coarse grid (without the colorbar)."""
        for ax in (self.ax_signal, ax_contour):
            ax.sharex(self.ax_coarse_image)
            ax.sharey(self.ax_coarse_image)

        # Bottom left: signal function of the label
        self.ax_signal_function = plt.subplot(gs[1, 0])
        self._build_signal_function()

        # Bottom middle: mass (CO2) or volume (tracer) over time
        self.ax_mass = plt.subplot(gs[1, 1])
        self.ax_mass.set_title(
            "Mass Evolution" if analysis.has_gas else "Volume Evolution"
        )
        units = self._display_units()
        self.mass_series = MassSeries(self.ax_mass, state, analysis.has_gas, units)
        self._set_mass_axes(units)
        self.ax_mass.legend()
        self.ax_mass.grid(True, alpha=0.3)

        # Bottom right: area of the controls
        self.ax_sliders = plt.subplot(gs[1, 2])
        self.ax_sliders.axis("off")
        self.ax_sliders.set_title("Parameter Controls")

    def _build_signal_function(self) -> None:
        """Artists of the signal function plot; their data is set by `show_label`."""
        analysis = self.analysis
        ax = self.ax_signal_function
        extents = analysis.signal_model_extents

        # Background: the colors of the color path of the label
        gradient = np.linspace(extents[0][0], extents[0][1], 256).reshape(1, -1)
        self.signal_function_gradient = ax.imshow(
            gradient * np.ones((10, 1)),
            extent=[extents[0][0], extents[0][1], extents[1][0], extents[1][1]],
            aspect="auto",
            alpha=0.3,  # Semi-transparent
            zorder=0,  # Behind everything else
        )

        # Signal function on a fine grid along the color path, values at the key colors
        # as dots
        [self.signal_function_line] = ax.plot([], [], "gray", linewidth=2, zorder=2)
        self.signal_function_scatter = ax.scatter(
            [], [], zorder=3, s=50, edgecolors="black", linewidth=1
        )

        # Flash bounds (gas phase only; for a tracer the signal is the concentration)
        self.flash_markers = FlashMarkers(ax) if analysis.has_gas else None

        self.unused_note = ax.text(
            x=0.01,
            y=0.99,
            s="Unused key colors (no dot): the signal is affine in the\n"
            "color and bends at every key color.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="k",
            zorder=4,
        )
        ax.set_xlabel("Color Interpretation")
        ax.set_ylabel("Signal Value")
        ax.grid(True, alpha=0.3, zorder=1)

    def _build_controls(self) -> None:
        """Rows of sliders: signal values, flash, contours, resolution.

        There is a value row for the largest number of key colors of all labels; rows
        which the shown label does not use are hidden (see `_layout_rows`).

        """
        state, analysis, fig = self.state, self.analysis, self.fig
        contour_thresholds = state.contour_thresholds
        levels = state.available_levels
        self._controls_box = self.ax_sliders.get_position()

        max_values = max(
            len(self.signal_function(label).values) for label in self.labels
        )
        num_rows = max_values + (9 if analysis.has_gas else 3)
        if len(levels) > 1:
            num_rows += 2
        self._row_height = self._controls_box.height / num_rows
        self._layout: list[tuple] = []
        """Section titles and rows from top to bottom, see `_layout_rows`."""

        self.values_title = self._section_title("")
        self.value_sliders = [
            self._slider_row(f"Value {i}", 0.0, 0.0, 2.0 if analysis.has_gas else 1.0)
            for i in range(max_values)
        ]
        for i, slider in enumerate(self.value_sliders):
            self._layout[-max_values + i] += (i,)

        self.flash_sliders = []
        if analysis.has_gas:
            self._section_title("Flash")
            flash = analysis.flash
            self.flash_sliders = [
                self._slider_row(name, value, 0.0, 2.0, "darkgray")
                for name, value in [
                    ("Min aq", flash.min_value_aq),
                    ("Max aq", flash.max_value_aq),
                    ("Min g", flash.min_value_g),
                    ("Max g", flash.max_value_g),
                ]
            ]

        self._section_title("Contours")
        self.threshold_sliders = [
            self._slider_row(
                "Threshold c_aq", contour_thresholds[0], 0.0, 1.0, (1, 0, 0)
            )
        ]
        if analysis.has_gas:
            self.threshold_sliders.append(
                self._slider_row(
                    "Threshold s_g", contour_thresholds[1], 0.0, 1.0, (0, 1, 0)
                )
            )
        for i, slider in enumerate(self.threshold_sliders):
            slider.on_changed(lambda val, i=i: contour_thresholds.__setitem__(i, val))
        self.all_sliders = (
            self.value_sliders + self.flash_sliders + self.threshold_sliders
        )

        # Resolution of the calculations: a discrete slider over the levels. Like the
        # parameters, the selected level is applied by the Update button.
        self.resolution_slider = None
        self.resolution_status = None
        if len(levels) > 1:
            self.resolution_status = StatusText(
                self._section_title(self._resolution_title())
            )
            slider = self._slider_row(
                "Level",
                levels.index(state.resolution_level),
                0,
                len(levels) - 1,
                "steelblue",
                valstep=1,
            )
            # Show the level instead of its position
            slider._format = lambda val: f"1/{levels[int(round(val))]}"
            slider.valtext.set_text(slider._format(slider.val))
            self.resolution_slider = slider

        self.blitter = SliderBlitter(fig)
        self.blitter.add(
            self.all_sliders
            + ([] if self.resolution_slider is None else [self.resolution_slider])
        )
        # Panels repainted alone after calculations (`refresh`)
        self.blitter.add_panels(
            [
                self.ax_coarse_image,
                self.ax_signal,
                self.coarse_signal_colorbar.ax,
                self.ax_contour,
                self.ax_signal_function,
                self.ax_mass,
            ]
        )

    def _section_title(self, title: str):
        """Bold title of a group of rows."""
        text = self.fig.text(
            self._controls_box.x0,
            0.0,
            title,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
        )
        self._layout.append(("title", text))
        return text

    def _slider_row(
        self, name: str, valinit, vmin, vmax, color="gray", valstep=0.01
    ) -> Slider:
        """Labeled slider (with value text) in a row."""
        box = self._controls_box
        label_text = self.fig.text(
            box.x0, 0.0, name, fontsize=8, ha="left", va="center"
        )
        slider = Slider(
            self.fig.add_axes([box.x0 + 0.30 * box.width, 0.0, 0.55 * box.width, 0.01]),
            "",
            vmin,
            vmax,
            valinit=valinit,
            valstep=valstep,
            color=color,
        )
        slider.valtext.set_fontsize(8)
        slider.row_label = label_text
        self._layout.append(("row", label_text, slider))
        return slider

    def _layout_rows(self, num_values: int) -> None:
        """Place titles and rows from top to bottom; hide unused value rows."""
        box, height = self._controls_box, self._row_height
        row = 0
        for entry in self._layout:
            if entry[0] == "title":
                entry[1].set_position((box.x0, box.y1 - (row + 0.5) * height))
                row += 1
                continue
            _, label_text, slider, *value_index = entry
            visible = not value_index or value_index[0] < num_values
            for artist in (label_text, slider.ax, slider.poly, slider._handle):
                artist.set_visible(visible)
            slider.valtext.set_visible(visible)
            if not visible:
                slider.set_active(False)
                continue
            y0 = box.y1 - (row + 1) * height
            label_text.set_position((box.x0, y0 + 0.5 * height))
            slider.ax.set_position(
                [
                    box.x0 + 0.30 * box.width,
                    y0 + 0.2 * height,
                    0.55 * box.width,
                    0.6 * height,
                ]
            )
            row += 1

    def _build_buttons(self) -> None:
        fig = self.fig
        button_width = 0.08
        button_height = 0.04
        button_y = 0.95
        button_spacing = 0.01

        def add_button(name: str, x: float, y: float) -> None:
            ax = fig.add_axes([x, y, button_width, button_height])
            self.buttons[name] = Button(ax, name)

        # Bottom right: Reset and Update, with the messages of the calculations
        update_x = 0.98 - button_width
        bottom_y = 0.02
        add_button("Reset", update_x - button_width - button_spacing, bottom_y)
        add_button("Update", update_x, bottom_y)
        self.busy_status = StatusText(
            fig.text(
                update_x + button_width,
                bottom_y + button_height + 0.004,
                "",
                fontsize=8,
                color="darkorange",
                ha="right",
                va="bottom",
            ),
            self.blitter,
        )
        """Message shown while a calculation blocks the window."""
        self.values_status = StatusText(
            fig.text(
                self._controls_box.x0,
                bottom_y + 0.5 * button_height,
                "",
                fontsize=8,
                color="red",
                ha="left",
                va="center",
            ),
            self.blitter,
        )
        """Message about invalid (decreasing) signal values."""

        # Top right: navigation
        names = ["Next Signal", "Next Label", "Next Image"]
        start_x = 0.98 - len(names) * button_width - (len(names) - 1) * button_spacing
        for position, name in enumerate(names):
            add_button(
                name, start_x + position * (button_width + button_spacing), button_y
            )

        # Top left: Save with its status
        save_x = 0.02
        add_button("Save", save_x, button_y)
        self.save_status = StatusText(
            fig.text(
                save_x + button_width + button_spacing,
                button_y + 0.5 * button_height,
                "",
                fontsize=9,
                color="green",
                va="center",
            ),
            self.blitter,
        )

    # ! ---- LABEL ----

    def show_label(self, label: int) -> None:
        """Show a label with the parameters in use.

        Values of the sliders which were not applied are dropped: the value rows show
        the signal values of the label, the flash rows the flash in use, the resolution
        slider the level in use. The contour thresholds stay.

        Parameters
        ----------
        label : int
            Label with a signal function.

        """
        state, analysis = self.state, self.analysis
        self.label_idx = label
        signal_func = self.signal_function(label)
        self._num_values = len(signal_func.values)

        # Image panel
        self.ax_coarse_image.set_title(
            f"Current Image {state.image_idx} and Label {label}"
        )
        self.coarse_image_img.set_data(state.label_highlight(state.coarse_image, label))
        self._shown_image = (state.image_idx, label)

        # Signal function panel
        ax = self.ax_signal_function
        ax.set_title(f"Signal Function - Label {label}")
        color_path = analysis.color_path_interpretation[label].color_path
        self.signal_function_gradient.set_cmap(color_path.get_color_map())
        t_fine, y_fine = signal_curve(signal_func)
        self.signal_function_line.set_data(t_fine, y_fine)
        self.signal_function_scatter.set_offsets(
            np.c_[signal_func.supports, _dot_values(signal_func)]
        )
        self.signal_function_scatter.set_facecolors(
            to_rgba_array(np.clip(color_path.colors, 0, 1))
        )
        if self.flash_markers is not None:
            self.flash_markers.update(analysis.flash, t_fine, y_fine)
        used = getattr(signal_func, "support_indices", None)
        self.unused_note.set_visible(
            used is not None and len(used) < len(signal_func.values)
        )
        extents = analysis.signal_model_extents
        ax.set_xlim(extents[0][0], extents[0][1])
        ax.set_ylim(extents[1][0], extents[1][1])

        # Value rows: one per key color of the label
        self.values_title.set_text(f"Signal values (label {label})")
        used = range(len(signal_func.values)) if used is None else used
        path_colors = color_path.colors
        for i, (slider, value) in enumerate(
            zip(self.value_sliders, signal_func.values)
        ):
            active = i in used
            slider.row_label.set_text(f"Value {i}" + ("" if active else " (unused)"))
            slider.poly.set_facecolor(
                np.clip(path_colors[i], 0, 1) if i < len(path_colors) else "gray"
            )
            for artist in (
                slider.row_label,
                slider.poly,
                slider._handle,
                slider.valtext,
            ):
                artist.set_alpha(None if active else FADED)
            self._set_slider(slider, value)
        self._layout_rows(self._num_values)
        for slider in self.value_sliders[: self._num_values]:
            slider.set_active(slider.row_label.get_alpha() is None)

        # Parameters in use for the rest
        if analysis.has_gas:
            flash = analysis.flash
            for slider, value in zip(
                self.flash_sliders,
                [
                    flash.min_value_aq,
                    flash.max_value_aq,
                    flash.min_value_g,
                    flash.max_value_g,
                ],
            ):
                self._set_slider(slider, value)
        if self.resolution_slider is not None:
            levels = state.available_levels
            self._set_slider(
                self.resolution_slider, levels.index(state.resolution_level)
            )

        # Repaint the image panel, the signal function and the controls (rows moved)
        rows = self.all_sliders + (
            [] if self.resolution_slider is None else [self.resolution_slider]
        )
        self.blitter.redraw(
            [self.ax_coarse_image, self.ax_signal_function, self.ax_sliders]
            + [slider.ax for slider in rows],
            artists=[entry[1] for entry in self._layout],
        )

    @staticmethod
    def _set_slider(slider: Slider, value: float) -> None:
        """Set a slider (rounded to its step, as at creation) without callbacks."""
        slider.eventson = False
        try:
            slider.set_val(slider._value_in_bounds(value))
        finally:
            slider.eventson = True

    # ! ---- SLIDERS ----

    def _resolution_title(self) -> str:
        return f"Resolution (in use: 1/{self.state.resolution_level})"

    def set_resolution_title(self) -> None:
        """Show the resolution level in use."""
        if self.resolution_status is not None:
            self.resolution_status.set(self._resolution_title())

    def selected_level(self) -> int:
        """Resolution level chosen with the slider."""
        if self.resolution_slider is None:
            return self.state.resolution_level
        levels = self.state.available_levels
        return levels[int(round(self.resolution_slider.val))]

    def values(self) -> list[float]:
        """Signal values of the sliders of the shown label."""
        return [slider.val for slider in self.value_sliders[: self._num_values]]

    def flash_values(self) -> list[float]:
        """Flash parameters of the sliders (empty without a gas phase)."""
        return [slider.val for slider in self.flash_sliders]

    # ! ---- DRAWING ----

    def _display_units(self) -> tuple[str, Callable, Callable, str]:
        """Units of the mass panel: quantity unit, its conversion and label, time unit.

        The quantity compared with the injection protocol is the mass (CO2), or the
        volume (tracer; the mass series then hold volumes in m3). Units are fixed, or
        chosen to keep the displayed numbers readable.

        """
        analysis, state = self.analysis, self.state
        max_expected = state.max_expected_mass()
        if analysis.has_gas:
            unit = analysis.mass_unit
            if unit == "auto":
                unit = choose_mass_unit(max_expected)
            convert_quantity, quantity_label = convert_mass, mass_label
        else:
            unit = analysis.volume_unit
            if unit == "auto":
                unit = choose_volume_unit(max_expected)
            convert_quantity, quantity_label = convert_volume, volume_label
        time_unit = analysis.time_unit
        if time_unit == "auto":
            time_unit = choose_time_unit(state.max_time())
        return unit, convert_quantity, quantity_label, time_unit

    def _set_mass_axes(self, units) -> None:
        """Labels and quantity range (twice the injected maximum) of the mass panel."""
        unit, convert_quantity, quantity_label, time_unit = units
        max_expected = self.state.max_expected_mass()
        if max_expected > 0:
            self.ax_mass.set_ylim(0.0, convert_quantity(2.0 * max_expected, unit))
        else:
            self.ax_mass.autoscale()
        self.ax_mass.set_xlabel(time_label(time_unit))
        self.ax_mass.set_ylabel(quantity_label(unit))

    def update_signal_function_plot(self) -> None:
        """Update the signal function plot (curve, dots, flash markers)."""
        analysis = self.analysis
        flash = analysis.flash
        signal_func = self.signal_function(self.label_idx)
        t_fine, y_fine = signal_curve(signal_func)
        self.signal_function_line.set_data(t_fine, y_fine)
        self.signal_function_scatter.set_offsets(
            np.c_[signal_func.supports, _dot_values(signal_func)]
        )
        # Rescale the y-axis to fit the curve, floored at the current flash maximum
        # (never zoom in tighter than the flash thresholds, but grow if the curve
        # exceeds them)
        flash_max = (
            max(flash.max_value_aq, flash.max_value_g) if analysis.has_gas else 1.0
        )
        self.ax_signal_function.set_ylim(
            analysis.signal_model_extents[1][0],
            max(
                flash_max,
                float(np.max(y_fine)),
                float(np.max(signal_func.values)),
            ),
        )
        if self.flash_markers is not None:
            self.flash_markers.update(flash, t_fine, y_fine)

    def update_mass_plot(self) -> None:
        """Update the mass evolution plots."""
        units = self._display_units()
        self.mass_series.update(units)
        self._set_mass_axes(units)

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

        # Image panel (only if another image or label is shown)
        changed = []
        shown_image = (state.image_idx, self.label_idx)
        if shown_image != self._shown_image:
            self.ax_coarse_image.set_title(
                f"Current Image {state.image_idx} and Label {self.label_idx}"
            )
            self.coarse_image_img.set_data(
                state.label_highlight(state.coarse_image, self.label_idx)
            )
            self._shown_image = shown_image
            changed.append(self.ax_coarse_image)

        # Signal panel
        self.coarse_signal_img.set_data(state.coarse_signal.img)
        self.ax_signal.set_title(state.signal_title)
        signal_min, signal_max = state.signal_range()
        self.coarse_signal_img.set_clim(vmin=signal_min, vmax=signal_max)
        self.coarse_signal_colorbar.update_normal(self.coarse_signal_img.image)
        changed += [self.ax_signal, self.coarse_signal_colorbar.ax]

        # Phase segmentation panel (the state keeps the overlay while it is valid)
        overlay = state.contour_overlay(
            [slider.val for slider in self.threshold_sliders]
        )
        if overlay is not self._shown_contour:
            self.coarse_contour_img.set_data(overlay.img)
            self._shown_contour = overlay
            changed.append(self.ax_contour)

        if signal_function:
            self.update_signal_function_plot()
            changed.append(self.ax_signal_function)
        if mass_plot:
            self.update_mass_plot()
            changed.append(self.ax_mass)
        self.blitter.redraw(changed)

    # ! ---- MESSAGES ----

    def show_values_status(self, message: str | None) -> None:
        """Show (or clear) the message about invalid signal values."""
        self.values_status.set(
            "" if message is None else message.replace(": ", ":\n", 1)
        )

    def mark_unsaved(self, *args) -> None:
        """Show that the displayed values differ from the saved ones."""
        if self.save_status.get_text() != UNSAVED_TEXT:
            self.save_status.set(UNSAVED_TEXT, "darkorange")

    def show_save_status(self, text: str, color: str) -> None:
        """Show the result of saving."""
        self.save_status.set(text, color)

    def show_busy(self, message: str, now: bool = True) -> None:
        """Show (or clear) the message about a running calculation.

        Parameters
        ----------
        message : str
            Message; empty to clear.
        now : bool
            Draw right away, for a calculation that blocks the window's event loop;
            otherwise (calculation in the background) when the window is idle.

        """
        self.busy_status.set(message, now=now)
