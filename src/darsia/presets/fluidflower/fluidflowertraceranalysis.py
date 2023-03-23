"""
Module containing the standardized tracer concentration analysis applicable
for the tracer experiments in the FluidFlower (and other similar assets),
allowing for heterogeneous media.

"""
from pathlib import Path
from typing import Union

import numpy as np

import darsia


class FluidFlowerTracerAnalysis(darsia.TracerAnalysis):
    """
    Class for managing the well test of the FluidFlower benchmark.
    """

    # ! ---- Setup routines

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        results: Union[str, Path],
        update_setup: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Setup of analysis.

        Args:
            baseline (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            results (str or Path): path to results directory
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
            verbosity  (bool): flag controlling whether results of the post-analysis
                are printed to screen; default is False.
        """
        # Assign tracer analysis
        darsia.TracerAnalysis.__init__(self, baseline, config, update_setup)

        # Add labels
        if not hasattr(self, "labels"):
            self.labels = np.ones(self.base.img.shape[:2], dtype=int)

        # Create folder for results if not existent
        self.path_to_results: Path = Path(results)
        self.path_to_results.parents[0].mkdir(parents=True, exist_ok=True)

        # Store verbosity
        self.verbosity = verbosity

    # ! ---- Analysis tools for detecting the tracer concentration

    def define_tracer_analysis(self) -> darsia.ConcentrationAnalysis:
        """
        Identify tracer concentration using a reduction to the grayscale space.

        """
        ########################################################################
        # Define signal reduction
        signal_reduction = darsia.MonochromaticReduction(**self.config["tracer"])

        ########################################################################
        # Balancing
        balancing = darsia.HeterogeneousLinearModel(
            self.labels, key="balancing ", **self.config["tracer"]
        )

        ########################################################################
        # Define restoration object - coarsen, tvd, resize
        original_size = self.base.img.shape[:2]
        restoration = darsia.CombinedModel(
            [
                darsia.Resize(key="restoration ", **self.config["tracer"]),
                darsia.TVD(key="restoration ", **self.config["tracer"]),
                darsia.Resize(dsize=tuple(reversed(original_size))),
            ]
        )

        ########################################################################
        # Linear model for converting signals to data
        model = darsia.CombinedModel(
            [
                darsia.LinearModel(key="model ", **self.config["tracer"]),
                darsia.ClipModel(**{"min value": 0.0, "max value": 1.0}),
            ]
        )

        ########################################################################
        # Final concentration analysis with possibility for calibration
        # of both the balancing and the model
        class TailoredConcentrationAnalysis(
            darsia.ConcentrationAnalysis,
            darsia.ContinuityBasedBalancingCalibrationMixin,
            darsia.InjectionRateModelObjectiveMixin,
        ):
            pass

        verbosity = self.config["tracer"].get("verbosity", 0)

        tracer_analysis = TailoredConcentrationAnalysis(
            self.base,
            signal_reduction,
            balancing,
            restoration,
            model,
            self.labels,
            verbosity=verbosity,
        )

        return tracer_analysis

    # ! ---- Calibration routines

    def calibrate_balancing(
        self, calibration_images: list[Path], options: dict
    ) -> None:
        """
        Calibration routine aiming at decreasing the discontinuity modulus
        across interfaces of the labeling.

        Args:
            calibration_images (list of Path): calibration images.
            options (dict): parameters for calibration.

        """

        # Read and process the images
        print("Calibration: Processing images...")
        images = [self._read(path) for path in calibration_images]

        # Calibrate the overall signal via a simple constant rescaling
        print("Calibration: Balancing...")
        self.tracer_analysis.calibrate_balancing(images, options)

    def calibrate_model(self, calibration_images: list[Path], options: dict) -> None:
        """
        Calibration routine aiming at matching the injection rate

        NOTE: Calling this routine will require the definition of
        a geometry for data integration.

        Args:
            calibration_images (list of Path): calibration images.
            options (dict): parameters for calibration.

        """
        # Read and process the images
        print("Calibration: Processing images...")
        images = [self._read(path) for path in calibration_images]

        # Calibrate the overall signal via a simple constant rescaling
        print("Calibration: Model...")
        self.tracer_analysis.calibrate_model(
            images,
            options=dict(options, **{"model position": 0, "geometry": self.geometry}),
        )

    # ! ----- Analysis tools

    def single_image_analysis(self, img: Path, **kwargs) -> darsia.Image:
        """
        Standard workflow to analyze the tracer concentration.

        Args:
            image (Path): path to single image.
            kwargs: optional keyword arguments, see batch_analysis.

        Returns:
            np.ndarray: tracer concentration map
            dict: dictinary with all stored results from the post-analysis.
        """

        # ! ---- Extract concentration profile

        # Load the current image
        self.load_and_process_image(img)

        # Determine tracer concentration
        tracer = self.determine_tracer()

        return tracer
