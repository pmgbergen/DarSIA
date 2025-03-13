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
        return darsia.LinearApproximation(space, (3, 3), domain="coordinates")

    def define_similar_colors(self):
        """Define similar colors for calibration.

        Use an interactive assistant to select similar colors
        for each calibration image.

        """
        width = self.config.get("sample_size", 50)
        debug = self.config.get("debug", False)
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
            colors.append(
                darsia.extract_characteristic_data(
                    signal=img.img,
                    samples=samples,
                    show_plot=debug,
                )
            )

        # Concatenate the coordinates and colors
        coordinates = self.baseline.coordinatesystem.coordinate(
            darsia.VoxelArray(np.concatenate(voxels, axis=0))
        )
        colors = np.concatenate(colors, axis=0)
        self.data.append((coordinates, colors))

    def define_reference_color(self):
        """Define reference color for calibration.

        Use an interactive assistant to select a reference color.
        Only the first image is considered for now.

        """
        width = self.config.get("sample_size", 50)
        debug = self.config.get("debug", False)

        for img in self.calibration_images[:1]:
            box_selection_assistant = darsia.BoxSelectionAssistant(img, width=width)
            samples = box_selection_assistant()

            # Check whether samples is provided in the correct format
            assert len(samples) > 0, "No samples selected."
            if len(samples) != 1:
                warn("Only a single sample is allowed.")

            # Extract one characteristic color per sample
            colors = darsia.extract_characteristic_data(
                signal=img.img,
                samples=samples,
                show_plot=debug,
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
        debug = self.config.get("debug", False)

        # First stage of the user-interaction: Select a grid of distinct colors
        # And define reference colors
        print("""Step 1. Select reference colors within a single color checker.""")
        reference_img = self.calibration_images[0]
        box_selection_assistant = darsia.BoxSelectionAssistant(
            reference_img, width=width
        )
        reference_samples = box_selection_assistant()

        # Extract one characteristic color per sample
        reference_colors = darsia.extract_characteristic_data(
            signal=reference_img.img, samples=reference_samples, show_plot=debug
        )
        # Define the relative samples to be used as translations in the tensor approach
        relative_reference_samples = [
            darsia.subtract_slice_pairs(sample, reference_samples[0])
            for sample in reference_samples
        ]

        # Second stage of the user-interaction: Select a grid of the same colors
        print(
            """Step 2. Select the same color as the first from the first stage to """
            """define a tensor-grid."""
        )
        for img in self.calibration_images:
            # Define outer tensor
            box_selection_assistant = darsia.BoxSelectionAssistant(img, width=width)
            top_left_samples = box_selection_assistant()

            # Data collection using a tensor approach
            for i, sample in enumerate(relative_reference_samples):
                # User a tensorial fill-in to extract all similar colors
                samples = [
                    darsia.add_slice_pairs(sample, top_left_sample)
                    for top_left_sample in top_left_samples
                ]

                # Extract the coordinates of all samples
                mid = lambda x: int(0.5 * (x.start + x.stop))
                sample_centers = [
                    [mid(sample[0]), mid(sample[1])] for sample in samples
                ]
                voxels = darsia.VoxelArray(sample_centers)
                coordinates = self.baseline.coordinatesystem.coordinate(voxels)

                # Extract one characteristic color per sample
                colors = darsia.extract_characteristic_data(
                    signal=img.img, samples=samples, show_plot=debug
                )

                # Store the samples and (similar) colors
                self.data.append((coordinates, colors))

                # Store reference colors taking into account the tensor approach
                self.reference_data.append(
                    np.vstack([reference_colors[i] for _ in range(len(samples))])
                )

    def calibrate(self):
        """Calibrate the color correction based on the samples and reference colors.

        A least-squares problem is solved to find the optimal coefficients.

        """

        # logging.info("Calibrating the relative color correction.")

        # Make sure there exists one reference color for each set of similar colors
        assert len(self.data) == len(self.reference_data), (
            f"Data mismatch: {len(self.data)} vs. {len(self.reference_data)}"
        )

        # Define the number of samples and color components
        self.stacked_coordinates = darsia.CoordinateArray(
            # self.stacked_coordinates = darsia.VoxelArray(
            np.vstack([data[0] for data in self.data])
        )
        self.stacked_colors = np.vstack([data[1] for data in self.data])

        # Need to repeat the reference colors for each set of similar colors
        self.stacked_reference_colors = np.vstack(self.reference_data)

        def _reshape_coefficients(coefficients):
            """Auxiliary function to reshape the coefficients."""
            return np.reshape(coefficients, self.correction.shape)

        # Define LS objective function
        def objective_function(scaling):
            """Objective function for least-squares problem."""
            self.correction.coefficients = _reshape_coefficients(scaling)
            flat_correction = self.correction.evaluate(
                self.stacked_coordinates
            ).reshape((-1, 3, 3), order="F")
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
