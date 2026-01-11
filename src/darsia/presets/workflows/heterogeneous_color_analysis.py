import copy
import json
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from darsia.presets.workflows.mass_computation import MassComputation
from darsia.presets.workflows.simple_run_analysis import (
    SimpleMassAnalysisResults,
    SimpleRunAnalysis,
)
import time

import darsia

logger = logging.getLogger(__name__)


def get_mean_color(
    image: darsia.Image, mask: darsia.Image | np.ndarray | None = None
) -> np.ndarray:
    """Calculate the mean color of an image, optionally masked by a mask.

    Args:
        image (darsia.Image): The image from which to calculate the mean color.
        mask (darsia.Image | np.ndarray | None): Optional mask to apply on the image.
            If `None`, the entire image is used.

    Returns:
        np.ndarray: The mean color of the image, calculated as the average of RGB values.

    """
    if isinstance(mask, darsia.Image):
        subimage = image.img[mask.img]
    elif isinstance(mask, np.ndarray):
        subimage = image.img[mask]
    else:
        subimage = image.img
    return np.mean(subimage.reshape(-1, 3), axis=0)


class HeterogeneousColorAnalysis(darsia.ConcentrationAnalysis):
    """Color-based concentration analysis."""

    # TODO use dynamic heterogeneous model, if same labels have the same model.

    def __init__(
        self,
        baseline: darsia.Image,
        labels: darsia.Image,
        color_mode: darsia.ColorMode,
        color_path_functions: dict[int, darsia.ColorPathFunction],
        # color_paths: darsia.LabelColorPathMap | None = None,
        restoration: darsia.Model | None = None,
        ignore_labels: list[int] | None = None,
    ):
        # TODO remove combined model?

        # Define non-calibrated model in a heterogeneous fashion
        # Allow for simple scaling of the concentration e.g.
        # due to changes in light conditions
        # Clip values to [0, inf] - not strictly necessary, but useful for visualization
        model = darsia.CombinedModel(
            [
                darsia.HeterogeneousModel(
                    darsia.ColorPathInterpolation(
                        color_path=darsia.ColorPath(
                            colors=[
                                0.0 * np.ones(3),
                                0.5 * np.ones(3),
                                1.0 * np.ones(3),
                            ],
                            base_color=np.zeros(3),
                            mode="rgb",
                        ),
                        color_mode=color_mode,
                    ),
                    labels,
                    ignore_labels=ignore_labels,
                ),
                darsia.ClipModel(min_value=0.0, max_value=None),
                # darsia.ScalingModel(),
            ]
        )

        # Define general config options
        config = {
            "diff option": "plain",
            "restoration -> model": False,
        }

        # Define general ConcentrationAnalysis.
        super().__init__(
            base=baseline if color_mode == darsia.ColorMode.RELATIVE else None,
            restoration=restoration,
            labels=labels,
            model=model,
            **config,
        )

        self.color_path_associations = np.zeros(
            np.unique(self.labels.img).size, dtype=int
        )
        self.color_path_functions = []
        if color_path_functions:
            self.color_path_associations = np.unique(self.labels.img).astype(int)
            self.color_path_functions = list(color_path_functions.values())
            for label, color_path_function in color_path_functions.items():
                self.model[0][label] = copy.copy(color_path_function)

    # def update_color_path_function(
    #    self, label: int, color_path_function: darsia.ColorPathFunction
    # ) -> None:
    #    """Update the color path for a specific label in the model.

    #    Args:
    #        label (int): The label for which to update the color path.
    #        color_path (darsia.ColorPath): The new color path to set for the label.

    #    """
    #    # Update the model with the new color path
    #    # self.model[label].color_path = color_path  # copy.copy(color_path)
    #    # self.color_paths.append(color_path_function)
    #    self.model[0][
    #        label
    #    ].color_path_function = color_path_function  # copy.copy(color_path)
    #    self.color_path_functions.append(color_path_function)
    #    self.color_path_associations[label] = len(self.color_path_functions) - 1

    #    print(self.color_path_associations)

    # TODO any previous color path to be removed? need to clean up.

    def define_color_path(
        self, image: darsia.Image, mask: darsia.Image
    ) -> darsia.ColorPath:
        """Interactive definition of a color path based on a given image and mask.

        Instructions:
        - Select a rectangular area in the image using the mouse.
        - The mean color of the selected area will be added to the color path.
        - For the first selection, the background color will also be added.
        - Continue selecting areas until you decide to stop (e.g., by closing
            the figure without selecting a new area).

        Args:
            image (darsia.Image): The image from which to define the color path.
            mask (darsia.Image): The mask to apply on the image for color path definition.

        Returns:
            darsia.ColorPath: The defined color path with selected colors.

        """
        # Sanity checks
        assert mask.img.dtype == bool, "Mask must be a boolean mask."

        colors: list[np.ndarray] = []
        while True:
            assistant = darsia.RectangleSelectionAssistant(image, labels=self.labels)

            # Pick a box in the image and convert it to a mask
            box: Tuple[slice, slice] = assistant()

            if box is None and len(colors) != 0:
                print("No box selected. Exiting color path definition.")
                break

            boxed_mask = darsia.zeros_like(mask, dtype=bool)
            boxed_mask.img[box] = mask.img[box]

            # Determine the mean color in the box
            if len(colors) == 0:
                # Add in addition the background color
                mean_color = get_mean_color(self.base, mask=boxed_mask)
                colors.append(mean_color)
            mean_color = get_mean_color(image, mask=boxed_mask)
            colors.append(mean_color)

        return darsia.ColorPath(colors=colors, base_color=colors[0], mode="rgb")

    def global_calibration_colors(
        self, image: darsia.Image, mask: darsia.Image
    ) -> None:
        """Define a global color path for the calibration.

        Assign a single color path to all labels. For this an interactive
        selection of the color path is performed, cf. `define_color_path`.

        Args:
            image (darsia.Image): The image from which to define the global color path.
            mask (darsia.Image): The mask to apply on the image for color path definition.

        """
        # Interactivee definition of the color path
        color_path = self.define_color_path(image, mask)

        # Set global color path
        self.global_color_path = color_path
        self.color_paths = [self.global_color_path]
        self.color_path_associations = np.zeros(
            np.unique(self.labels.img).size, dtype=int
        )

        # Update the model with the global color path for each label
        for label in np.unique(self.labels.img):
            # Fetch the associated color path
            color_path = self.color_paths[self.color_path_associations[label]]
            self.model[0][label].color_path = copy.copy(color_path)

    def local_calibration_colors(self, image: darsia.Image, mask: darsia.Image) -> None:
        """Define local color paths for each label.

        Instructions:
            - Pick a label in the image using the mouse and select a rectangle.
            - Define a new color path just for this label.

        """
        # Add local color paths for separate labels
        while True:
            # Pick label
            concentration = self(image)
            assistant = darsia.RectangleSelectionAssistant(
                concentration, labels=self.labels
            )

            # Identify the label of interest
            label_box: Tuple[slice, slice] = assistant()
            label = np.argmax(np.bincount(self.labels.img[label_box].ravel()))

            # Define a new color path
            color_path = self.define_color_path(image, mask)
            self.color_paths.append(color_path)

            # Associate the new color path with the label
            self.color_path_associations[label] = len(self.color_paths) - 1

            # Update the model with the local color path for the label
            self.model[0][label].color_path = copy.copy(color_path)

            add_more = (
                input("Do you want to add another color path? (y/n) ").strip().lower()
            )
            if add_more != "y":
                break

    def local_calibration_values(
        self, images: darsia.Image | list[darsia.Image], mask: darsia.Image, cmap=None
    ) -> None:
        """Define a local color path for a specific label.

        Instructions:
            - Pick a label in the image using the mouse and select a rectangle.
            - Define a new color path just for this label.
            - Tune the values for the color path.

        Args:
            image (darsia.Image): The image from which to define the local color path.
            mask (darsia.Image): The mask to apply on the image for color path definition.

        """
        if not isinstance(images, list):
            images = [images]

        # Set up parameters for coarse visualization
        coarse_rows = max(200, self.labels.img.shape[0] // 4)
        coarse_cols = int(
            self.labels.img.shape[1] / self.labels.img.shape[0] * coarse_rows
        )
        coarse_shape = (coarse_rows, coarse_cols)

        # Coarsen the image and labels for better visualization
        coarse_labels = darsia.resize(
            self.labels,
            shape=coarse_shape,
            interpolation="inter_nearest",
        )
        coarse_images = [darsia.resize(image, shape=coarse_shape) for image in images]

        image_idx = 0

        def image_idx_selector() -> int:
            """Interactive selection of plotted images - allow to click for showing the next coarse images - keep track of the idx in the list and return when finishing."""
            nonlocal image_idx

            # Create a simple figure for image selection
            fig, ax = plt.subplots(figsize=(8, 6))
            current_image = coarse_images[image_idx]
            img_display = ax.imshow(current_image.img)
            ax.set_title(
                f"Image {image_idx + 1}/{len(images)} - Click to cycle through images, close window to finish"
            )
            ax.axis("off")

            def on_click(event):
                nonlocal image_idx
                if event.inaxes == ax:
                    # Cycle to next image
                    image_idx = (image_idx + 1) % len(images)
                    current_image = coarse_images[image_idx]
                    img_display.set_data(current_image.img)
                    ax.set_title(
                        f"Image {image_idx + 1}/{len(images)} - Click to cycle through images, close window to finish"
                    )
                    fig.canvas.draw_idle()

            # Connect click event
            fig.canvas.mpl_connect("button_press_event", on_click)

            plt.show()
            return image_idx

        image_idx = image_idx_selector()

        # Interactive tuning of values for each color path (chosen by user)
        done_picking_new_labels = False
        while not done_picking_new_labels:
            # Pick image
            image = images[image_idx]
            coarse_image = coarse_images[image_idx]

            # Pick label
            concentration = self(image)
            assistant = darsia.RectangleSelectionAssistant(
                concentration, labels=self.labels, cmap=cmap
            )

            # Identify the label of interest
            label_box: Tuple[slice, slice] = assistant()
            label = np.argmax(np.bincount(self.labels.img[label_box].ravel()))

            done_tuning_values = False
            while not done_tuning_values:

                def show_tuner(idx):
                    nonlocal done_tuning_values, done_picking_new_labels
                    fig, ax_conc = plt.subplots(figsize=(8, 4))
                    ax_image = plt.axes([0.05, 0.5, 0.15, 0.4])
                    plt.subplots_adjust(left=0.25, bottom=0.25)
                    ax_image.imshow(coarse_image.img)
                    mask = np.zeros_like(coarse_labels.img, dtype=np.uint8)
                    mask[coarse_labels.img == idx] = 1
                    ax_image.imshow(mask, alpha=0.5, cmap="gray", vmin=0, vmax=1)
                    ax_conc.set_title(f"Tune values for color path #{idx}")

                    sliders = []
                    slider_height = 0.03
                    for i, val in enumerate(self.model[0][idx].values):
                        ax_slider = plt.axes(
                            [0.25, 0.15 - i * slider_height, 0.65, slider_height]
                        )
                        slider = Slider(
                            ax_slider,
                            f"Value {i}",
                            -0.5,
                            1.5,
                            valinit=val,
                            valstep=0.05,
                        )
                        sliders.append(slider)

                    ax_update = plt.axes([0.8, 0.925, 0.1, 0.04])
                    btn_update = Button(ax_update, "Update values")
                    ax_close = plt.axes([0.68, 0.925, 0.1, 0.04])
                    btn_close = Button(ax_close, "Next layer")
                    ax_next_image = plt.axes([0.56, 0.925, 0.1, 0.04])
                    btn_next_image = Button(ax_next_image, "Switch image")
                    ax_finish = plt.axes([0.44, 0.925, 0.1, 0.04])
                    btn_finish = Button(ax_finish, "Finish")

                    coarse_conc = darsia.resize(self(image), shape=coarse_shape)
                    conc_img = ax_conc.imshow(coarse_conc.img, cmap=cmap)

                    def update_concentration(event=None):
                        new_values = [slider.val for slider in sliders]
                        self.model[0][idx].update_model_parameters(new_values)
                        conc = self(image)
                        coarse_conc = darsia.resize(conc, shape=coarse_shape)
                        conc_img.set_data(coarse_conc.img)
                        fig.canvas.draw_idle()

                    btn_update.on_clicked(update_concentration)

                    def close(event):
                        nonlocal done_tuning_values
                        done_tuning_values = True
                        plt.close("all")

                    btn_close.on_clicked(close)

                    def finish(event):
                        nonlocal done_tuning_values, done_picking_new_labels
                        done_tuning_values = True
                        done_picking_new_labels = True
                        plt.close("all")

                    btn_finish.on_clicked(finish)

                    def next_image(event):
                        nonlocal image_idx, done_tuning_values
                        image_idx = (image_idx + 1) % len(images)
                        done_tuning_values = True
                        plt.close("all")

                    btn_next_image.on_clicked(next_image)

                    plt.show()

                show_tuner(label)

    def global_calibration_flash(
        self,
        mass_computation: MassComputation,
        mask: darsia.Image,
        calibration_images: list[darsia.Image],
        experiment: darsia.ProtocolledExperiment,
        cmap=None,
        show: bool = False,
    ) -> None:
        """Coarse tuning of the color signal analysis."""

        # Check expected status

        # Step 1: pre-mass analysis (nothing to do as images already converted to signals)
        analysis = SimpleRunAnalysis(mass_computation.geometry)

        # # Tracking...
        # folder = Path("calibration_mass")  # TODO?
        # # Remove everything in the folder
        # if folder.exists():
        #     for file in folder.iterdir():
        #         if file.is_file():
        #             file.unlink()
        # folder.mkdir(parents=True, exist_ok=True)

        # Step 2: Initialize multiphase time series
        times = []
        expected_mass = []
        integrated_mass = []
        integrated_mass_g = []
        integrated_mass_aq = []
        square_error = []

        def update_analysis():
            """Auxiliary function to update multiphase time series analysis."""
            nonlocal analysis
            nonlocal times
            nonlocal expected_mass
            nonlocal integrated_mass
            nonlocal integrated_mass_g
            nonlocal integrated_mass_aq
            nonlocal square_error

            analysis.reset()
            for img in calibration_images:
                # Preliminaries
                img_time = img.time
                # TODO update co2_mass_analysis/flash etc based on img_time.

                # Signal analysis
                tic_local = time.time()
                signal = self(img)
                print(f"Color analysis took {time.time() - tic_local} seconds")

                # Mass computation
                tic_local = time.time()
                mass_analysis_result: SimpleMassAnalysisResults = mass_computation(
                    signal
                )
                print(f"Mass computation took {time.time() - tic_local} seconds")

                # Compute expected mass
                exact_mass = experiment.injection_protocol.injected_mass(img.date)

                # Track result
                tic_local = time.time()
                analysis.track(
                    mass_analysis_result,
                    exact_mass=exact_mass,
                )
                print(f"Tracking took {time.time() - tic_local} seconds")

                # # Clean data - TODO?
                # analysis.clean(threshold=1.0)

            # Monitor mass evolution over time
            times = analysis.data.time
            expected_mass = analysis.data.exact_mass_tot
            integrated_mass = analysis.data.mass_tot
            integrated_mass_g = analysis.data.mass_g
            integrated_mass_aq = analysis.data.mass_aq

            # Errors
            square_error = np.square(
                np.array(integrated_mass) - np.array(expected_mass)
            )

        tic = time.time()
        update_analysis()
        toc = time.time()

        from icecream import ic

        ic(f"First analysis took {toc - tic} seconds")

        # TODO log iteration?

        # Make one annotation and four plots
        # 0. Error as text in the top left corner
        # 1. PWTransformation for gas with sliders
        # 2. PWTransformation for aqueous with sliders
        # 3. Integrated mass over time, entire run, updated upon activation
        # 4. Integrated mass over time, first 12 hours, updated upon activation

        # Set up the figure layout: sliders on the left, plots on the right
        fig = plt.figure(figsize=(18, 10))

        # Use gridspec to allocate space for sliders and plots
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(
            nrows=1,
            ncols=2,
            width_ratios=[1, 2],  # Sliders:Plots
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
            wspace=0.3,
        )
        # Sliders area (left)
        slider_area = plt.subplot(gs[0, 0])
        slider_area.axis("off")
        # Plots area (right, 2x1)
        gs_plots = gridspec.GridSpecFromSubplotSpec(
            nrows=2,
            ncols=1,
            subplot_spec=gs[0, 1],
            height_ratios=[1, 1],
            hspace=0.3,
        )
        ax = [plt.subplot(gs_plots[0, 0]), plt.subplot(gs_plots[1, 0])]

        # ! ---- PLOT 3: INTEGRATED MASS OVER TIME, ENTIRE RUN ----

        # Combine plot and scatter for integrated mass over time.
        # Decompose into total, gas, and aqueous mass, and add
        # expected mass using a dashed line.
        ax[1].set_xlabel("Time (h)")
        ax[1].set_ylabel("Mass (g)")
        [integrated_mass_plot] = ax[1].plot(
            times,
            integrated_mass,
            color="blue",
            label="total",
        )
        [integrated_mass_plot_g] = ax[1].plot(
            times,
            integrated_mass_g,
            color="green",
            label="gas",
        )
        [integrated_mass_plot_aq] = ax[1].plot(
            times,
            integrated_mass_aq,
            color="orange",
            label="aqueous",
        )
        ax[1].plot(
            times,
            expected_mass,
            linestyle="--",
            color="red",
            label="injected",
        )
        integrated_mass_scatter = ax[1].scatter(
            times,
            integrated_mass,
            color="blue",
        )
        integrated_mass_scatter_g = ax[1].scatter(
            times,
            integrated_mass_g,
            color="green",
        )
        integrated_mass_scatter_aq = ax[1].scatter(
            times,
            integrated_mass_aq,
            color="orange",
        )
        ax[1].set_ylim(0.0, 0.01)
        ax[1].legend()
        ax[1].set_title("Integrated mass over time, entire run")

        plt.show()

    def local_calibration_flash(
        self,
        mass_computation: MassComputation,
        mask: darsia.Image,
        calibration_images: list[darsia.Image],
        cmap=None,
        show: bool = False,
    ) -> None:
        """Define a local color path for a specific label.

        Instructions:
            - Pick a label in the image using the mouse and select a rectangle.
            - Define a new color path just for this label.
            - Tune the values for the color path based on mass computation

        Args:
            mass_computation (darsia.MassComputation): The mass computation tool.
            mask (darsia.Image): The mask to apply on the image for color path definition.
            calibration_images (list[darsia.Image]): The images used for calibration.
            cmap: Optional colormap for visualization.
            show (bool): Whether to display plots during processing.

        """
        assert False, "continue here"

    def local_calibration_color_path(
        self, image: darsia.Image, mask: darsia.Image
    ) -> None:
        """Define a local color path for a specific label.

        Instructions:
            - Pick a label in the image using the mouse and select a rectangle.
            - Define a new color path just for this label.
            - Tune the values for the color path.

        Args:
            image (darsia.Image): The image from which to define the local color path.
            mask (darsia.Image): The mask to apply on the image for color path definition.

        """
        while True:
            # Pick label
            concentration = self(image)
            assistant = darsia.RectangleSelectionAssistant(
                concentration, labels=self.labels
            )

            # Identify the label of interest
            label_box: Tuple[slice, slice] = assistant()
            label = np.argmax(np.bincount(self.labels.img[label_box].ravel()))

            # Define a new color path
            color_path = self.define_color_path(image, mask)
            self.color_paths.append(color_path)

            # Associate the new color path with the label
            new_color_path_id = len(self.color_paths) - 1
            self.color_path_associations[label] = new_color_path_id

            # Update the model with the local color path for the label
            self.model[0][label].color_path = copy.copy(color_path)

            # Tune values

            # Set up parameters for coarse visualization
            coarse_rows = max(200, image.img.shape[0] // 4)
            coarse_cols = int(image.img.shape[1] / image.img.shape[0] * coarse_rows)
            coarse_shape = (coarse_rows, coarse_cols)

            # Coarsen the image and labels for better visualization
            coarse_image = darsia.resize(image, shape=coarse_shape)
            coarse_labels = darsia.resize(
                self.labels,
                shape=coarse_shape,
                interpolation="inter_nearest",
            )

            done = False
            while not done:

                def show_tuner(idx):
                    nonlocal done
                    color_path = self.color_paths[idx]
                    fig, ax_conc = plt.subplots(figsize=(8, 4))
                    ax_image = plt.axes([0.05, 0.5, 0.15, 0.4])
                    plt.subplots_adjust(left=0.25, bottom=0.25)
                    ax_image.imshow(coarse_image.img)
                    mask = np.zeros_like(coarse_labels.img, dtype=np.uint8)
                    for label in np.where(self.color_path_associations == idx)[0]:
                        mask[coarse_labels.img == label] = 1
                    ax_image.imshow(mask, alpha=0.5, cmap="gray", vmin=0, vmax=1)
                    ax_conc.set_title(f"Tune values for color path #{idx}")

                    sliders = []
                    slider_height = 0.03
                    for i, val in enumerate(color_path.values):
                        ax_slider = plt.axes(
                            [0.25, 0.15 - i * slider_height, 0.65, slider_height]
                        )
                        slider = Slider(
                            ax_slider,
                            f"Value {i}",
                            -0.5,
                            1.5,
                            valinit=val,
                            valstep=0.05,
                        )
                        sliders.append(slider)

                    ax_update = plt.axes([0.8, 0.025, 0.1, 0.04])
                    btn_update = Button(ax_update, "Update")
                    ax_close = plt.axes([0.68, 0.025, 0.1, 0.04])
                    btn_close = Button(ax_close, "Close")

                    coarse_conc = darsia.resize(self(image), shape=coarse_shape)
                    conc_img = ax_conc.imshow(coarse_conc.img)

                    def update_concentration(event=None):
                        new_values = [slider.val for slider in sliders]
                        color_path.update_values(new_values)
                        for label in np.where(self.color_path_associations == idx)[0]:
                            self.model[0][label].color_path = copy.copy(color_path)
                        conc = self(image)
                        coarse_conc = darsia.resize(conc, shape=coarse_shape)
                        conc_img.set_data(coarse_conc.img)
                        fig.canvas.draw_idle()

                    btn_update.on_clicked(update_concentration)

                    def close(event):
                        nonlocal done
                        done = True
                        plt.close("all")

                    btn_close.on_clicked(close)

                    plt.show()

            show_tuner(new_color_path_id)

    def calibration_values(
        self, image: darsia.Image, initial_color_path_idx: int = 0
    ) -> None:
        """Interactively tune the calibration values for each color path.

        Instructions:
            - Use the sliders to adjust the values for each color path.
            - Click "Update" to apply the changes and visualize the updated concentration.
            - Use "Prev" and "Next" to navigate through color paths.
            - Click "Close" to finish tuning.

        Args:
            image (darsia.Image): The image to visualize the color paths.
            initial_color_path_idx (int): The index of the initial color path to start tuning.

        """
        color_path_idx = initial_color_path_idx

        # Set up parameters for coarse visualization
        coarse_rows = max(200, image.img.shape[0] // 4)
        coarse_cols = int(image.img.shape[1] / image.img.shape[0] * coarse_rows)
        coarse_shape = (coarse_rows, coarse_cols)

        # Coarsen the image and labels for better visualization
        coarse_image = darsia.resize(image, shape=coarse_shape)
        coarse_labels = darsia.resize(
            self.labels,
            shape=coarse_shape,
            interpolation="inter_nearest",
        )

        done = False
        while not done:
            print("start calibration values")
            # Show the tuner for the current color path

            def show_tuner(idx):
                nonlocal done, color_path_idx
                color_path = self.color_paths[idx]
                fig, ax_conc = plt.subplots(figsize=(8, 4))
                ax_image = plt.axes([0.05, 0.5, 0.15, 0.4])
                plt.subplots_adjust(left=0.25, bottom=0.25)
                ax_image.imshow(coarse_image.img)
                mask = np.zeros_like(coarse_labels.img, dtype=np.uint8)
                for label in np.where(self.color_path_associations == idx)[0]:
                    mask[coarse_labels.img == label] = 1
                ax_image.imshow(mask, alpha=0.5, cmap="gray", vmin=0, vmax=1)
                ax_conc.set_title(f"Tune values for color path #{idx}")

                sliders = []
                slider_height = 0.03
                for i, val in enumerate(color_path.values):
                    ax_slider = plt.axes(
                        [0.25, 0.15 - i * slider_height, 0.65, slider_height]
                    )
                    slider = Slider(
                        ax_slider, f"Value {i}", -0.5, 1.5, valinit=val, valstep=0.05
                    )
                    sliders.append(slider)

                ax_update = plt.axes([0.8, 0.025, 0.1, 0.04])
                btn_update = Button(ax_update, "Update")
                ax_prev = plt.axes([0.25, 0.025, 0.1, 0.04])
                btn_prev = Button(ax_prev, "Prev")
                ax_next = plt.axes([0.37, 0.025, 0.1, 0.04])
                btn_next = Button(ax_next, "Next")
                ax_close = plt.axes([0.68, 0.025, 0.1, 0.04])
                btn_close = Button(ax_close, "Close")

                coarse_conc = darsia.resize(self(image), shape=coarse_shape)
                conc_img = ax_conc.imshow(coarse_conc.img)

                def update_concentration(event=None):
                    new_values = [slider.val for slider in sliders]
                    color_path.update_values(new_values)
                    for label in np.where(self.color_path_associations == idx)[0]:
                        self.model[0][label].color_path = copy.copy(color_path)
                    conc = self(image)
                    coarse_conc = darsia.resize(conc, shape=coarse_shape)
                    conc_img.set_data(coarse_conc.img)
                    fig.canvas.draw_idle()

                btn_update.on_clicked(update_concentration)

                def switch_path(event, direction):
                    nonlocal done, color_path_idx
                    color_path_idx = (idx + direction) % len(self.color_paths)
                    plt.close(fig)

                btn_prev.on_clicked(lambda event: switch_path(event, -1))
                btn_next.on_clicked(lambda event: switch_path(event, 1))

                def close(event):
                    nonlocal done
                    done = True
                    plt.close("all")

                btn_close.on_clicked(close)

                plt.show()

            show_tuner(color_path_idx)

            if not plt.fignum_exists(plt.gcf().number):  # Window closed by user
                break
            if not (0 <= color_path_idx < len(self.color_paths)):
                raise ValueError

        logging.info("Calibration values tuning completed.")

    def save(self, path: Path) -> None:
        """Save the calibration data to json file.

        Args:
            path (Path): The path to save the calibration data.

        """

        # Save the color paths and their associations
        color_paths = {}
        for color_path_id, color_path in enumerate(self.color_paths):
            color_paths[color_path_id] = {
                "base_color": color_path.base_color.tolist(),
                "colors": [c.tolist() for c in color_path.colors],
                "values": [float(v) for v in color_path.values],
                "labels": [],
            }

        # Invert color_path associations
        for label in np.unique(self.labels.img):
            color_path_id = self.color_path_associations[label]
            # self.labels has dtype uint8. Cast
            color_paths[color_path_id]["labels"].append(int(label))

        # Print all types of all ingredients of color_paths
        for color_path_id, value in color_paths.items():
            print(f"Color path ID: {color_path_id}")
            print(f"  Base color type: {type(value['base_color'])}")
            for i, c in enumerate(value["colors"]):
                print(f"  Color {i} type: {type(c)}")
                for j, comp in enumerate(c):
                    print(f"    Component {j} type: {type(comp)}")
            for i, v in enumerate(value["values"]):
                print(f"  Value {i} type: {type(v)}")
            for i, label in enumerate(value["labels"]):
                print(f"  Label {i} type: {type(label)}")

        # Save the data to json file
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix(".json")
        with open(path, "w") as f:
            json.dump(color_paths, f, indent=4)

        logger.info(f"Calibration data saved to {path}")

    def load(self, path: Path) -> None:
        """Load the calibration data from json file.

        Args:
            path (Path): path to load the model

        """
        # Load the json file
        with open(path, "r") as f:
            color_paths = json.load(f)

        # Update the model with the loaded data label by label
        self.color_paths = []
        self.color_path_associations = np.zeros(
            np.unique(self.labels.img).size, dtype=int
        )
        for color_path_id, value in color_paths.items():
            color_path = darsia.ColorPath(
                colors=[np.array(c) for c in value["colors"]],
                values=value["values"],
                base_color=np.array(value["base_color"]),
                mode="rgb",
            )
            self.color_paths.append(color_path)
            for label in value["labels"]:
                self.color_path_associations[label] = color_path_id
                self.model[0][label].color_path = copy.copy(color_path)

    # TODO I/O csv
