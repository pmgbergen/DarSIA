from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import json
import copy
import darsia
import logging

from matplotlib.widgets import Button, Slider

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
        # color_path: list[darsia.ColorPath] | None = None, # TODO allow for initialization
        restoration: darsia.Model | None = None,
        relative: bool = True,
        ignore_labels: list[int] | None = None,
    ):
        # Define non-calibrated model in a heterogeneous fashion
        # Allow for simple scaling of the concentration e.g.
        # due to changes in light conditions
        # Clip values to [0, 1] - not strictly necessary, but useful for visualization
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
                            values=[0.0, 0.5, 1.0],
                            base_color=np.zeros(3),
                            mode="rgb",
                        ),
                        interpolation="relative" if relative else "absolute",
                    ),
                    labels,
                    ignore_labels=ignore_labels,
                ),
                darsia.ScalingModel(),
                darsia.ClipModel(min_value=0.0, max_value=1.0),
            ]
        )

        # Define general config options
        config = {
            "diff option": "plain",
            "restoration -> model": False,
        }

        # Define general ConcentrationAnalysis.
        super().__init__(
            base=baseline if relative else None,
            restoration=restoration,
            labels=labels,
            model=model,
            **config,
        )

        # Cache
        self.global_color_path: darsia.ColorPath = darsia.ColorPath(
            colors=[
                0.0 * np.ones(3),
                0.5 * np.ones(3),
                1.0 * np.ones(3),
            ],
            values=[0.0, 0.5, 1.0],
            base_color=np.zeros(3),
            mode="rgb",
        )
        self.color_paths = [self.global_color_path]
        self.color_path_associations = np.zeros(
            np.unique(self.labels.img).size, dtype=int
        )

    def update_color_path(self, label: int, color_path: darsia.ColorPath) -> None:
        """Update the color path for a specific label in the model.

        Args:
            label (int): The label for which to update the color path.
            color_path (darsia.ColorPath): The new color path to set for the label.

        """
        # Update the model with the new color path
        self.model[0][label].color_path = color_path  # copy.copy(color_path)
        self.color_paths.append(color_path)
        self.color_path_associations[label] = len(self.color_paths) - 1

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
        self, images: darsia.Image | list[darsia.Image], mask: darsia.Image
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

        image_idx = 0

        # Interactive tuning of values for each color path (chosen by user)
        done_picking_new_labels = False
        while not done_picking_new_labels:
            # Pick image
            image = images[image_idx]

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

            # Pick label
            concentration = self(image)
            assistant = darsia.RectangleSelectionAssistant(
                concentration, labels=self.labels
            )

            # Identify the label of interest
            label_box: Tuple[slice, slice] = assistant()
            label = np.argmax(np.bincount(self.labels.img[label_box].ravel()))

            # Associate the new color path with the label
            color_path_id = self.color_path_associations[label]

            done_tuning_values = False
            while not done_tuning_values:

                def show_tuner(idx):
                    nonlocal done_tuning_values, done_picking_new_labels
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
                    btn_update = Button(ax_update, "Update values")
                    ax_close = plt.axes([0.68, 0.025, 0.1, 0.04])
                    btn_close = Button(ax_close, "Next layer")
                    ax_next_image = plt.axes([0.56, 0.025, 0.1, 0.04])
                    btn_next_image = Button(ax_next_image, "Switch image")
                    ax_finish = plt.axes([0.44, 0.025, 0.1, 0.04])
                    btn_finish = Button(ax_finish, "Finish")

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

                show_tuner(color_path_id)

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
                "values": [v for v in color_path.values],
                "labels": [],
            }
        # Invert color_path associations
        for label in np.unique(self.labels.img):
            color_path_id = self.color_path_associations[label]
            # self.labels has dtype uint8. Cast
            color_paths[color_path_id]["labels"].append(int(label))

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
