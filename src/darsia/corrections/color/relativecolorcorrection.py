"""Module for relative color balance correction based on global analysis."""

from typing import Optional, Union
from warnings import warn

import numpy as np
import scipy.optimize

import darsia


class RelativeColorCorrection(darsia.BaseCorrection):
    """Relative color correction based on global analysis."""

    def __init__(
        self,
        baseline: Optional[darsia.Image] = None,
        images: Optional[Union[darsia.Image, list[darsia.Image]]] = None,
        config: Optional[dict] = None,
    ) -> None:

        self.baseline = baseline
        """Baseline image."""
        self.calibration_images = images
        """Calibration images."""
        self.config = config if config is not None else {}
        """Configuration dictionary."""

        # Define color correction as linear approximation - the coefficients
        # of the linear approximation will be calibrated through optimization
        self.correction = self.define_correction()
        """Color correction as linear approximation."""

        # Allocate space for the samples and reference colors
        self.data: list[tuple[darsia.VoxelArray, np.ndarray]] = []
        """Samples and colors for calibration."""
        self.reference_data: list[np.ndarray] = []
        """Reference colors for calibration."""

        if self.calibration_images is not None:
            # User-defined calibration
            print("Welcome to the Relative Color Correction Assistant!")
            mode = self.config.get("mode", "custom")
            if mode == "tensorial":
                self.define_similar_and_reference_colors_tensorial()
            elif mode == "custom":
                while True:
                    print("Do you want to choose a set of similar colors? (y/n)")
                    choice = input()
                    if choice == "y":
                        print("Please select a set of similar colors.")
                        self.define_similar_colors()
                        print("Please select a reference color.")
                        self.define_reference_color()
                    elif choice == "n":
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")

            # Find the color correction
            self.calibrate()

            # Setup the color correction
            self.setup()

    # ! ---- I/O ----

    def save(self, path: str) -> None:
        """Save the linear approximation incl. the coefficients and meta information."""
        assert hasattr(self.correction, "coefficients"), "No coefficients to save."
        np.savez(
            path,
            coefficients=self.correction.coefficients,
            config=self.config,
        )

    def load(self, path: str) -> None:
        """Load the linear approximation incl. the coefficients and meta information."""
        data = np.load(path, allow_pickle=True)
        self.config = data["config"].item()
        self.correction = self.define_correction()
        self.correction.coefficients = data["coefficients"]
        self.setup()

    # ! ---- CORRECTION ----

    def correct_array(self, img: np.ndarray) -> np.ndarray:
        """Rescale an array using heterogeneous color correction.

        Args:
            img (np.ndarray): input image

        Returns:
            np.ndarray: corrected image

        """
        # Pixel-by-pixel matrix-vector multiplication
        return np.einsum("ijkl,ijl->ijk", self.evaluated_correction, img)

    # ! ---- SETUP ----

    def define_correction(self) -> darsia.LinearApproximation:
        """Set the correction method.
        
        Returns:
            darsia.LinearApproximation: Linear approximation for color correction.
        
        """
        ansatz = self.config.get("method", "polynomial")
        if ansatz == "polynomial":
            degree = self.config.get("degree", 2)
            space = darsia.PolynomialApproximationSpace(degree)
        else:
            raise ValueError(f"Anstatz '{ansatz}' is not supported.")
        return darsia.LinearApproximation(space, (3, 3))

    def define_similar_colors(self)
        """Define similar colors for calibration.
        
        Use an interactive assistant to select similar colors
        for each calibration image.

        """
        width = self.config.get("sample_size", 50)
        voxels = []
        colors = []

        for img in self.calibration_images:
            box_selection_assistant = darsia.BoxSelectionAssistant(img, width=width)
            samples = box_selection_assistant()

            # Extract the coordinates of all samples
            mid = lambda x: int(0.5 * (x.start + x.stop))
            sample_centers = [[mid(sample[0]), mid(sample[1])] for sample in samples]
            voxels.append(np.array(sample_centers))

            # Extract one characteristic color per sample
            filter: callable = lambda x: x
            show_plot = False
            colors.append(
                darsia.extract_characteristic_data(
                    signal=img.img,
                    samples=samples,
                    filter=filter,
                    show_plot=show_plot,
                )
            )

        # Concatenate the coordinates and colors
        voxels = darsia.VoxelArray(np.concatenate(voxels, axis=0))
        colors = np.concatenate(colors, axis=0)
        self.data.append((voxels, colors))

    def define_reference_color(self):
        """Define reference color for calibration.
        
        Use an interactive assistant to select a reference color.
        Only the first image is considered for now.
        
        """
        width = self.config.get("sample_size", 50)

        for img in self.calibration_images[:1]:
            box_selection_assistant = darsia.BoxSelectionAssistant(img, width=width)
            samples = box_selection_assistant()

            # Check whether samples is provided in the correct format
            assert len(samples) > 0, "No samples selected."
            if len(samples) != 1:
                warn("Only a single sample is allowed.")

            # Extract one characteristic color per sample
            filter: callable = lambda x: x
            show_plot = False
            colors = darsia.extract_characteristic_data(
                signal=img.img,
                samples=samples,
                filter=filter,
                show_plot=show_plot,
            )
        self.reference_data.append(colors)

    def define_similar_and_reference_colors_tensorial(self):
        """Define similar and reference colors for calibration.

        Use an interactive assistant to select similar colors and a reference color.
        A 2-stage procedure is used to select similar colors. First, a grid of distinct
        colors is selected. Second, a set of grid of the same colors through the entire
        image is selected. Tensorial fill-in is used to extract all similar colors.
        As reference colors, the selected colors from stage 1 are used.

        """

        # Define samples
        width = self.config.get("sample_size", 50)
        voxels = []
        colors = []
        # assert len(self.calibration_images) == 1, "Need to adapt code"
        for img_counter, img in enumerate(self.calibration_images):
            if img_counter == 0:
                # Keep the same samples for all images
                box_selection_assistant = darsia.BoxSelectionAssistant(img, width=width)
                main_samples = box_selection_assistant()
            box_selection_assistant2 = darsia.BoxSelectionAssistant(img, width=width)
            pre_top_left_samples = box_selection_assistant2()
            top_left_samples = [
                darsia.subtract_slice_pairs(sample, pre_top_left_samples[0])
                for sample in pre_top_left_samples
            ]

            for sample in main_samples:
                samples = [
                    darsia.add_slice_pairs(sample, top_left_sample)
                    for top_left_sample in top_left_samples
                ]
                # Extract the coordinates of all samples
                mid = lambda x: int(0.5 * (x.start + x.stop))
                sample_centers = [
                    [mid(sample[0]), mid(sample[1])] for sample in samples
                ]
                voxels = np.array(sample_centers)

                # Extract one characteristic color per sample
                filter: callable = lambda x: x
                show_plot = False
                colors = darsia.extract_characteristic_data(
                    signal=img.img,
                    samples=samples,
                    filter=filter,
                    show_plot=show_plot,
                )
                self.data.append((voxels, colors))
                if img_counter == 0:
                    self.reference_data.append(colors[-1])
            if img_counter > 0:
                # Copy the reference data while keeping the main samples
                self.reference_data = (
                    self.reference_data + self.reference_data[: len(main_samples)]
                )

    def calibrate(self):
        """Calibrate the color correction based on the samples and reference colors.
        
        A least-squares problem is solved to find the optimal coefficients.

        """

        # Make sure there exists one reference color for each set of similar colors
        assert len(self.data) == len(self.reference_data)

        # Define the number of samples and color components
        self.stacked_voxels = darsia.VoxelArray(
            np.vstack([data[0] for data in self.data])
        )
        self.stacked_colors = np.vstack([data[1] for data in self.data])

        # Need to repeat the reference colors for each set of similar colors
        num_samples = [data[0].shape[0] for data in self.data]
        self.stacked_reference_colors = np.vstack(
            np.repeat(self.reference_data, num_samples, axis=0)
        )

        def _reshape_coefficients(coefficients):
            return np.reshape(coefficients, self.correction.shape)

        # Define LS objective function
        def objective_function(scaling):
            """Objective function for least-squares problem."""
            self.correction.coefficients = _reshape_coefficients(scaling)
            flat_correction = self.correction.evaluate(self.stacked_voxels).reshape(
                (-1, 3, 3), order="F"
            )
            flat_corrected_colors = np.einsum(
                "ikl,il->ik", flat_correction, self.stacked_colors
            )
            return np.sum((flat_corrected_colors - self.stacked_reference_colors) ** 2)

        # Define initial guess
        initial_guess = np.zeros(self.correction.size)

        # Solve least-squares problem
        opt_result = scipy.optimize.minimize(
            objective_function,
            initial_guess,
            method="Powell",
            tol=1e-16,
            options={"maxiter": 1000, "disp": True},
        )

        # Store final result
        self.correction.coefficients = _reshape_coefficients(opt_result.x)

    def setup(self):
        """Setup the color correction based on the baseline image."""
        assert self.baseline is not None, "Baseline image is missing."
        self.evaluated_correction = self.correction.evaluate(
            self.baseline.coordinatesystem
        )
