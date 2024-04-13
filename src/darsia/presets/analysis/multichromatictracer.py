"""Preset concentration analysis for labeled media based on multichromatic
color analysis.

"""

from typing import Optional

import numpy as np

import darsia


class MultichromaticTracerAnalysis(darsia.ConcentrationAnalysis):
    """Multichomratic concentration analysis tailored to labeled media.

    Essentially, it offers merely a short-cut definition, accompanied by calibration.

    """

    # ! ---- SETUP ----

    def __init__(
        self,
        baseline: darsia.Image,
        labels: Optional[darsia.Image] = None,
        relative: bool = True,
        show_plot: bool = False,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            baseline (Image): baseline image; relevant for relative analysis as well as
                calibration in a comparative sense (on multiple images)
            labels (Image, optional): labeled image, if not provided, the analysis is
                considered homogeneous
            relative (bool): flag controlling whether the analysis is relative
            show_plot (bool): flag controlling whether intermediate plots are showed
            kwargs: other keyword arguments
                - kernel (Kernel): kernel for interpolation

        """

        # Allow to parse standard objects for initializing a concentration analysis.
        restoration = kwargs.get("restoration")

        # Allow to parse config for concentration analysis.
        config = kwargs.get(
            "config",
            {
                "diff option": "plain",
                "restoration -> model": False,
            },
        )

        # Make sure labels are provided.
        if labels is None:
            labels = darsia.zeros_like(baseline, mode="voxels", dtype=np.uint8)

        # Cache relative flag
        self.relative = relative

        # Define non-calibrated model in a heterogeneous fashion.
        kernel = kwargs.get("kernel", darsia.GaussianKernel(gamma=1))
        model = darsia.CombinedModel(
            [
                darsia.HeterogeneousModel(
                    darsia.KernelInterpolation(kernel),
                    labels,
                )
            ]
        )
        self.characteristic_colors = []
        """Characteristic colors for each label."""
        self.concentrations = []
        """Concentration values for each label."""

        # Define general ConcentrationAnalysis
        super().__init__(
            base=baseline if self.relative else None,
            restoration=restoration,
            labels=labels,
            model=model,
            **config,
        )

        # Cache meta information
        self.show_plot = show_plot

    # ! ---- CALL ----

    def expert_knowledge(self, image: darsia.Image) -> None:
        """Expert knowledge for concentration analysis.

        Args:
            image (Image): image to be analyzed

        """
        ...

    def __call__(self, image: darsia.Image) -> darsia.Image:
        """Perform concentration analysis with additional expert knowledge.

        Args:
            image (Image): image to be analyzed

        Returns:
            Image: concentration map

        """
        concentration = super().__call__(image)
        self.expert_knowledge(concentration)
        return concentration

    # ! ---- SAVE AND LOAD ----
    def save(self, path: darsia.Path) -> None:
        """Save calibration data to a file.

        Args:
            path (Path): path to the file

        """
        np.savez(
            path,
            config={
                "characteristic_colors": self.characteristic_colors,
                "concentrations": self.concentrations,
                "info": "MultichromaticTracerAnalysis calibration data.",
            },
        )
        print(f"Calibration data saved to {path}.")

    def load(self, path: darsia.Path) -> None:
        """Load calibration data from a file.

        Args:
            path (Path): path to the file

        """
        data = np.load(path, allow_pickle=True)["config"].item()
        self.characteristic_colors = data["characteristic_colors"]
        self.concentrations = data["concentrations"]
        self.calibrate(self.characteristic_colors, self.concentrations)

    # ! ---- CALIBRATION ----

    def calibrate(self, colors, concentrations) -> None:
        """Calibrate analysis object from colors and concentrations.

        Update heterogeneous kernel interpolation.

        Args:
            colors (list): list of colors
            concentrations (list): list of concentrations

        """
        for i, _ in enumerate(darsia.Masks(self.labels)):

            self.model[0][i].update(supports=colors[i], values=concentrations[i])

    def calibrate_from_image(
        self,
        calibration_image,
        width: int = 25,
        num_clusters: int = 5,
        reset: bool = False,
    ) -> None:
        """
        Use last caliration image to define support points.

        Use all to fix the support points assignment.

        Args:
            calibration_image (Image): calibration image for extracting colors
            width (int): width of sample boxes returned from assistant - irrelevant if
                boxed defined
            num_clusters (int): number of characteristic clusters extracted
            reset (bool): flag controlling whether the calibration is reset. If False,
                the calibration is appended, allowing multi-step calibration, based on
                different images.

        """
        # TODO include possibility to deactivate untrustful support points

        # ! ---- STEP 0: Deactivate model and restoration

        model_cache = self.model
        restoration_cache = self.restoration
        self.model = None
        self.restoration = None

        # ! ---- STEP 1: Calibrate the support points (x) based on some images

        # Initialize data collections
        if reset:
            self.characteristic_colors = []
            self.concentrations = []

        for i, mask in enumerate(darsia.Masks(self.labels)):

            # Define characteristic points and corresponding data values
            print("Define samples")
            assistant = darsia.BoxSelectionAssistant(
                calibration_image, background=mask, width=width
            )
            samples = assistant()
            print("Define associated concentration values - assumed 1 if empty")
            # Ask for concentration values from user
            concentrations = [
                float(input(f"Concentration for sample {i}: "))
                for i in range(len(samples))
            ]

            # Fetch characteristic colors from samples
            # Apply concentration analysis modulo the model and the restoration
            pre_concentration = self(calibration_image)
            characteristic_colors = darsia.extract_characteristic_data(
                signal=pre_concentration.img,
                samples=samples,
                show_plot=self.show_plot,
                num_clusters=num_clusters,
            )

            # Use baseline image to collect 0-data
            if self.relative:
                # Define zero data
                concentrations_base = len(samples) * [0]

                # Apply concentration analysis modulo the model and the restoration
                pre_concentration_base = self(self.base)

                # Fetch characteristic colors from samples
                characteristic_colors_base = darsia.extract_characteristic_data(
                    signal=pre_concentration_base.img,
                    samples=samples,
                    show_plot=self.show_plot,
                    num_clusters=num_clusters,
                )

                # Collect data
                characteristic_colors = np.vstack(
                    (characteristic_colors_base, characteristic_colors)
                )
                concentrations = np.array(concentrations_base + concentrations)

            # Cache data or append if already existing
            if len(self.characteristic_colors) > i:
                self.characteristic_colors[i] = np.vstack(
                    (characteristic_colors, self.characteristic_colors[i])
                )
                self.concentrations[i] = np.hstack(
                    (concentrations, self.concentrations[i])
                )
            else:
                self.characteristic_colors.append(characteristic_colors)
                self.concentrations.append(concentrations)

        # Reinstall the model and the restoration
        self.model = model_cache
        self.restoration = restoration_cache

        # Update kernel interpolation
        self.calibrate(self.characteristic_colors, self.concentrations)
