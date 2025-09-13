"""Interactive calibration of multiphase transformations for mass analysis."""

import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider

import darsia

logger = logging.getLogger(__name__)


def calibrate_transformations(
    transformation_g: darsia.PWTransformation,
    transformation_aq: darsia.PWTransformation,
    paths: list[Path],
    multiphase_time_series_analysis: darsia.MultiphaseTimeSeriesAnalysis,
    upper_time_limit: float,
    read_image: Callable[[Path], darsia.Image],
    pre_mass_analysis: Callable[[darsia.Image], dict],
    mass_analysis_from_pre: Callable[[dict], darsia.MassAnalysisResults],
    log: Path,
) -> None:
    """Interactive calibration of multiphase transformations for mass analysis.

    The goal will be to match the detected and expected total masses
    of CO2 over time. The transformation_g and transformation_aq
    will be adjusted iteratively to minimize the difference between
    the detected and expected masses. The routine requires three callables:
    - `read_image`: a function to read an image from a given path and convert to a darsia.Image,
    - `pre_mass_analysis`: a function to perform pre-mass analysis on the image,
    - `mass_analysis_from_pre`: a function to perform mass analysis from the pre-mass analysis results.

    Args:
        transformation_g (darsia.PWTransformation): Transformation for gas phase.
        transformation_aq (darsia.PWTransformation): Transformation for aqueous phase.
        paths (list[Path]): List of paths to images to be analyzed.
        multiphase_time_series_analysis (darsia.MultiphaseTimeSeriesAnalysis): Analysis object to track results.
        upper_time_limit (float): Upper time limit for the analysis in hours.
        read_image (Callable[[Path], darsia.Image]): Function to read an image from a path.
        pre_mass_analysis (Callable[[darsia.Image], dict]): Function to perform pre-mass analysis on

    """

    # Cache the number of values in transformations
    num_values_g = len(transformation_g.values)
    num_values_aq = len(transformation_aq.values)

    # Prepare log folder
    log.mkdir(parents=True, exist_ok=True)

    # Step 1: Pre-mass analysis
    pre_mass_results = {}
    for i, path in enumerate(paths):
        img = read_image(path)
        pre_mass_result = pre_mass_analysis(img)
        pre_mass_results[path] = pre_mass_result
        print(f"Pre-mass analysis for {path.name} done. {i + 1}/{len(paths)}")

    # Step 2: Initialize multiphase time series analysis
    run_time = []
    ind = 0
    integrated_mass = []
    integrated_mass_g = []
    integrated_mass_aq = []
    expected_mass = []
    square_error = []
    early_square_error = []
    late_square_error = []

    def update_multiphase_time_series_analysis():
        """Auxiliary function to update multiphase time series analysis."""
        nonlocal multiphase_time_series_analysis
        nonlocal run_time
        nonlocal ind
        nonlocal integrated_mass
        nonlocal integrated_mass_g
        nonlocal integrated_mass_aq
        nonlocal expected_mass
        nonlocal square_error
        nonlocal early_square_error
        nonlocal late_square_error

        multiphase_time_series_analysis.reset()
        for path in paths:
            # Mass analysis
            mass_analysis_result = mass_analysis_from_pre(pre_mass_results[path])

            # Track result
            multiphase_time_series_analysis.track(mass_analysis_result)

            # Clean data
            multiphase_time_series_analysis.clean(threshold=1.0)

        # Monitor mass evolution over time
        run_time = multiphase_time_series_analysis.data.time
        ind = np.argmax(np.where(np.array(run_time) < upper_time_limit)[0]) + 1
        integrated_mass = multiphase_time_series_analysis.data.mass_tot
        integrated_mass_g = multiphase_time_series_analysis.data.mass_g
        integrated_mass_aq = multiphase_time_series_analysis.data.mass_aq
        expected_mass = multiphase_time_series_analysis.data.exact_mass_tot

        # Errors
        square_error = np.square(np.array(integrated_mass) - np.array(expected_mass))
        early_square_error = square_error[:ind]
        late_square_error = square_error[ind:]

    # Log errors and calibration iteration
    log_calibration_iteration = 0
    log_error = []
    log_early_error = []
    log_late_error = []
    log_transformation_g_supports = np.empty((0, num_values_g))
    log_transformation_g_values = np.empty((0, num_values_g))
    log_transformation_aq_supports = np.empty((0, num_values_aq))
    log_transformation_aq_values = np.empty((0, num_values_aq))

    def log_iteration():
        """Auxiliary function to log calibration iteration."""
        nonlocal num_values_g  # not modified
        nonlocal num_values_aq  # not modified
        nonlocal log_error
        nonlocal log_early_error
        nonlocal log_late_error
        nonlocal log_transformation_g_supports
        nonlocal log_transformation_g_values
        nonlocal log_transformation_aq_supports
        nonlocal log_transformation_aq_values

        log_error.append(np.sqrt(np.sum(square_error)))
        log_early_error.append(np.sqrt(np.sum(early_square_error)))
        log_late_error.append(np.sqrt(np.sum(late_square_error)))

        # Log calibration iteration (supports and values)
        log_transformation_g_supports = np.vstack(
            (log_transformation_g_supports, transformation_g.supports)
        )
        log_transformation_g_values = np.vstack(
            (log_transformation_g_values, transformation_g.values)
        )
        log_transformation_aq_supports = np.vstack(
            (log_transformation_aq_supports, transformation_aq.supports)
        )
        log_transformation_aq_values = np.vstack(
            (log_transformation_aq_values, transformation_aq.values)
        )

        # Log the results in a csv file
        df_dict = {
            "error": log_error,
            "early_error": log_early_error,
            "late_error": log_late_error,
        }
        for i in range(num_values_g):
            df_dict[f"g_support_{i}"] = log_transformation_g_supports[:, i]
            df_dict[f"g_value_{i}"] = log_transformation_g_values[:, i]
        for i in range(num_values_aq):
            df_dict[f"aq_support_{i}"] = log_transformation_aq_supports[:, i]
            df_dict[f"aq_value_{i}"] = log_transformation_aq_values[:, i]
        df = pd.DataFrame(df_dict)
        df.to_csv(log / "pw_transformations.csv", index=False)

        # Log the current active transformations
        transformation_g.log(log / "transformation_g.png")
        transformation_aq.log(log / "transformation_aq.png")

    # Update multiphase time series analysis and log iteration
    update_multiphase_time_series_analysis()
    log_iteration()

    # Make one annotation and four plots
    # 0. Error as text in the top left corner
    # 1. PWTransformation for gas with sliders
    # 2. PWTransformation for aqueous with sliders
    # 3. Integrated mass over time, entire run, updated upon activation
    # 4. Integrated mass over time, first 12 hours, updated upon activation

    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(18, 10)
    fig.subplots_adjust(
        left=0.05,
        bottom=(max(num_values_g, num_values_aq) + 1) * 0.05,
        right=0.95,
    )

    # ! ---- ANNOTATION 0: ERROR ----

    # Print the error in the top left corner
    error_txt = fig.text(
        0.5,
        1.25,
        f"""Error: {np.sqrt(np.sum(square_error)):.2e} | """
        f"""Early: {np.sqrt(np.sum(early_square_error)):.2e} | """
        f"""Late: {np.sqrt(np.sum(late_square_error)):.2e}""",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
    )

    # ! ---- PLOT 1: PW TRANSFORMATION FOR GAS ----

    # Pre-define a signal domain [0, 1] for plotting of transformations
    signal_domain = np.arange(0.0, 1.0, 0.001)

    # Set the size of the subfigure to leave space for the sliders
    [pw_transformation_g] = ax[0].plot(
        signal_domain,
        transformation_g(signal_domain),
        color="green",
        label="gas",
    )
    pw_transformation_scatter_g = ax[0].scatter(
        transformation_g.supports,
        transformation_g.values,
        color="green",
    )
    ax[0].set_xlabel("Signal")
    ax[0].set_ylabel("Converted signal")
    ax[0].set_title("PWTransformations")
    ax[0].set_ylim(0, 1)
    ax[0].legend()

    # ! ---- SLIDERS FOR GAS TRANSFORMATION ----

    # Define an axes area and draw sliders for gas transformation
    g_slider_ax = {}
    g_slider = {}
    for i in range(num_values_g):
        g_slider_ax[i] = fig.add_axes([0.05, (num_values_g - i) * 0.05, 0.4, 0.03])
        g_slider[i] = Slider(
            g_slider_ax[i],
            f"Support {i}\n{transformation_g.supports[i]:.2f}",
            0.0,
            1.0,
            valinit=transformation_g.values[i],
        )

    # Define an action for modifying the line when any slider's value changes
    def sliders_g_on_changed(val):
        """Update the transformation based on slider values and redraw the plot."""
        transformation_g.update(values=[g_slider[i].val for i in range(num_values_g)])
        pw_transformation_g.set_ydata(transformation_g(signal_domain))
        pw_transformation_scatter_g.set_offsets(
            np.c_[
                transformation_g.supports,
                transformation_g.values,
            ]
        )
        fig.canvas.draw_idle()

    # Connect the sliders to the update function
    for i in range(len(g_slider)):
        g_slider[i].on_changed(sliders_g_on_changed)

    # ! ---- PLOT 2: PW TRANSFORMATION FOR AQUEOUS ----

    # Set the size of the subfigure to leave space for the sliders
    [pw_transformation_aq] = ax[1].plot(
        signal_domain,
        transformation_aq(signal_domain),
        color="orange",
        label="aqueous",
    )
    pw_transformation_scatter_aq = ax[1].scatter(
        transformation_aq.supports,
        transformation_aq.values,
        color="orange",
    )
    ax[1].set_xlabel("Signal")
    ax[1].set_ylabel("Converted signal")
    ax[1].set_title("PWTransformations")
    ax[1].set_ylim(0, 1)
    ax[1].legend()

    # ! ---- SLIDERS FOR AQUEOUS TRANSFORMATION ----

    # Identical to the gas transformation, but placement is different.

    # Define an axes area and draw sliders for aqueous transformation
    aq_slider_ax = {}
    aq_slider = {}
    for i in range(len(transformation_aq.supports)):
        aq_slider_ax[i] = fig.add_axes([0.55, (num_values_aq - i) * 0.05, 0.4, 0.03])
        aq_slider[i] = Slider(
            aq_slider_ax[i],
            f"Support {i}\n{transformation_aq.supports[i]:.2f}",
            0.0,
            1.0,
            valinit=transformation_aq.values[i],
        )

    # Define an action for modifying the line when any slider's value changes
    def sliders_aq_on_changed(val):
        """Update the transformation based on slider values and redraw the plot."""
        transformation_aq.update(
            values=[aq_slider[i].val for i in range(num_values_aq)]
        )
        pw_transformation_aq.set_ydata(transformation_aq(signal_domain))
        pw_transformation_scatter_aq.set_offsets(
            np.c_[
                transformation_aq.supports,
                transformation_aq.values,
            ]
        )
        fig.canvas.draw_idle()

    # Connect the sliders to the update function
    for i in range(len(aq_slider)):
        aq_slider[i].on_changed(sliders_aq_on_changed)

    # ! ---- PLOT 3: INTEGRATED MASS OVER TIME, ENTIRE RUN ----

    # Combine plot and scatter for integrated mass over time.
    # Decompose into total, gas, and aqueous mass, and add
    # expected mass using a dashed line.
    ax[2].set_xlabel("Time (h)")
    ax[2].set_ylabel("Mass (g)")
    [integrated_mass_plot] = ax[2].plot(
        run_time,
        integrated_mass,
        color="blue",
        label="total",
    )
    [integrated_mass_plot_g] = ax[2].plot(
        run_time,
        integrated_mass_g,
        color="green",
        label="gas",
    )
    [integrated_mass_plot_aq] = ax[2].plot(
        run_time,
        integrated_mass_aq,
        color="orange",
        label="aqueous",
    )
    ax[2].plot(
        run_time,
        expected_mass,
        linestyle="--",
        color="red",
        label="injected",
    )
    integrated_mass_scatter = ax[2].scatter(
        run_time,
        integrated_mass,
        color="blue",
    )
    integrated_mass_scatter_g = ax[2].scatter(
        run_time,
        integrated_mass_g,
        color="green",
    )
    integrated_mass_scatter_aq = ax[2].scatter(
        run_time,
        integrated_mass_aq,
        color="orange",
    )
    ax[2].set_ylim(0.0, 0.01)
    ax[2].legend()
    ax[2].set_title("Integrated mass over time, entire run")

    # ! ---- PLOT 4: INTEGRATED MASS OVER TIME, FIRST 12 HOURS ----

    # Same as plot 3, but only for the first 12 hours.
    ax[3].set_xlabel("Time (h)")
    ax[3].set_ylabel("Mass (g)")
    [early_integrated_mass_plot] = ax[3].plot(
        run_time[:ind],
        integrated_mass[:ind],
        color="blue",
        label="total",
    )
    [early_integrated_mass_plot_g] = ax[3].plot(
        run_time[:ind],
        integrated_mass_g[:ind],
        color="green",
        label="gas",
    )
    [early_integrated_mass_plot_aq] = ax[3].plot(
        run_time[:ind],
        integrated_mass_aq[:ind],
        color="orange",
        label="aqueous",
    )
    ax[3].plot(
        run_time[:ind],
        expected_mass[:ind],
        linestyle="--",
        color="red",
        label="injected",
    )
    early_integrated_mass_scatter = ax[3].scatter(
        run_time[:ind],
        integrated_mass[:ind],
        color="blue",
    )
    early_integrated_mass_scatter_g = ax[3].scatter(
        run_time[:ind],
        integrated_mass_g[:ind],
        color="green",
    )
    early_integrated_mass_scatter_aq = ax[3].scatter(
        run_time[:ind],
        integrated_mass_aq[:ind],
        color="orange",
    )
    ax[3].set_ylim(0.0, 0.01)
    ax[3].legend()
    ax[3].set_title("Integrated mass over time, first 12 hours")

    # ! ---- UPDATING THE MASS PLOTS ----

    # Add a button for resetting the parameters
    update_button_ax = fig.add_axes([0.85, 0.95, 0.1, 0.04])
    update_button = Button(
        update_button_ax,
        "Update",
        color="lightgoldenrodyellow",
        hovercolor="0.975",
    )

    def update_button_on_clicked(mouse_event):
        """Update the mass plots based on the current transformations."""
        nonlocal log_calibration_iteration
        log_calibration_iteration += 1

        # Deactivate the mass plots by plotting additional text 'Loading...' in the center of the plot
        loading_txt = ax[2].text(
            np.max(run_time) // 2,
            0.008,
            "Loading...",
            horizontalalignment="center",
            # color="r",
        )
        early_loading_txt = ax[3].text(
            np.max(run_time[:ind]) // 2,
            0.008,
            "Loading...",
            # color="r",
            horizontalalignment="center",
        )
        fig.canvas.draw_idle()

        # Update the multiphase time series analysis
        update_multiphase_time_series_analysis()

        # Update the mass plots (Plot 3)
        integrated_mass_plot.set_ydata(integrated_mass)
        integrated_mass_plot_g.set_ydata(integrated_mass_g)
        integrated_mass_plot_aq.set_ydata(integrated_mass_aq)

        integrated_mass_scatter.set_offsets(np.c_[run_time, integrated_mass])
        integrated_mass_scatter_g.set_offsets(np.c_[run_time, integrated_mass_g])
        integrated_mass_scatter_aq.set_offsets(np.c_[run_time, integrated_mass_aq])

        # Update the mass plots (Plot 4)
        early_integrated_mass_plot.set_ydata(integrated_mass[:ind])
        early_integrated_mass_plot_g.set_ydata(integrated_mass_g[:ind])
        early_integrated_mass_plot_aq.set_ydata(integrated_mass_aq[:ind])

        early_integrated_mass_scatter.set_offsets(
            np.c_[run_time[:ind], integrated_mass[:ind]]
        )
        early_integrated_mass_scatter_g.set_offsets(
            np.c_[run_time[:ind], integrated_mass_g[:ind]]
        )
        early_integrated_mass_scatter_aq.set_offsets(
            np.c_[run_time[:ind], integrated_mass_aq[:ind]]
        )

        # Log the results
        log_iteration()

        # Replace the error text with the updated error
        error_txt.set_text(
            f"""Error: {np.sqrt(np.sum(square_error)):.2e} | """
            f"""Early: {np.sqrt(np.sum(early_square_error)):.2e} | """
            f"""Late: {np.sqrt(np.sum(late_square_error)):.2e}"""
        )

        # Remove the text again
        loading_txt.remove()
        early_loading_txt.remove()

        # Redraw the canvas
        fig.canvas.draw_idle()

        # Plot the results in a png file
        plt.savefig(
            log / f"pw_transformations_{log_calibration_iteration}.png",
            dpi=500,
        )

    # Connect the button to the update function
    update_button.on_clicked(update_button_on_clicked)

    # Show the plot
    plt.show()
