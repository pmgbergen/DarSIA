"""
Module containing the standardized tracer concentration analysis applicable
for the well test performed in the large FluidFlower.
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage
from benchmark.rigs.largefluidflower import LargeFluidFlower


class LargeRigTracerAnalysis(LargeFluidFlower, darsia.TracerAnalysis):
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
        verbosity: bool = False,
    ) -> None:
        """
        Constructor for tracer analysis tailored to the benchmark
        geometry in the large FluidFlower.

        Sets up fixed config file required for preprocessing.

        Args:
            baseline (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
            verbosity  (bool): flag controlling whether results of the post-analysis
                are printed to screen; default is False.
        """
        # Segment the geometry
        LargeFluidFlower.__init__(self, baseline, config, update_setup)

        # Determine depht
        LargeFluidFlower._determine_effective_volumes(self)

        # Assign tracer analysis
        darsia.TracerAnalysis.__init__(self, baseline, config, update_setup)

        # Define a geometry to enable integration over the porous domain
        self.define_porous_geometry()

        # Create folder for results if not existent
        self.path_to_results: Path = Path(results)
        self.path_to_results.parents[0].mkdir(parents=True, exist_ok=True)

        # Store verbosity
        self.verbosity = verbosity

        # The above constructors provide access to the config via self.config.
        # Determine the injection start from the config file. Expect format
        # complying with "%y%m%d %H%M%D", e.g., "211127 083412"
        # TODO as part of the calibration, this will be returned.
        self.injection_start: datetime = datetime.strptime(
            self.config["injection_start"], "%y%m%d %H%M%S"
        )

    # TODO move somewhere else - this should be standard.
    def define_porous_geometry(self) -> None:
        shape = self.base.img.shape
        physical_asset = self.config["physical asset"]
        dimensions = {
            "width": self.config["physical asset"]["dimensions"]["width"],
            "height": self.config["physical asset"]["dimensions"]["height"],
            "voxel depth": self.depth,
            "porosity": self.config["physical asset"]["parameters"]["porosity"],
        }
        self.geometry = darsia.PorousGeometry(shape[:2], **dimensions)

    # ! ---- Analysis tools for detecting the tracer concentration

    def define_tracer_analysis(self) -> darsia.ConcentrationAnalysis:
        """
        Identify tracer concentration using a reduction to the garyscale space
        """
        ########################################################################
        # Define signal reduction
        signal_reduction = darsia.MonochromaticReduction(**self.config["tracer"])

        ########################################################################
        # Balancing
        balancing = darsia.HeterogeneousLinearModel(
            self.labels, key="heterogeneous model ", **self.config["tracer"]
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

        tracer_analysis = TailoredConcentrationAnalysis(
            self.base,
            signal_reduction,
            balancing,
            restoration,
            model,
            self.labels,
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

        """

        # Read and process the images
        print("Calibration: Processing images...")
        images = [self._read(path) for path in calibration_images]

        # Calibrate the overall signal via a simple constant rescaling
        print("Calibration: Balancing...")
        self.tracer_analysis.calibrate_balancing(images, options)

    def calibrate_model(self, calibration_images: list[Path]) -> None:
        """
        Calibration routine aiming at matching the injection rate

        Args:
            calibration_images (list of Path): calibration images.

        """
        # Read and process the images
        print("Calibration: Processing images...")
        images = [self._read(path) for path in calibration_images]

        # Calibrate the overall signal via a simple constant rescaling
        print("Calibration: Model...")
        self.tracer_analysis.calibrate_model(
            images,
            options={
                "model position": 0,
                "geometry": self.geometry,
                # TODO make these arguments of calibrate_model
                "injection_rate": 2250,
                "initial_guess": [3.0], #, 0.0],
                "tol": 1e-1,
                "maxiter": 100,
            },
        )

    # ! ----- Analysis tools

    def single_image_analysis(self, img: Path, **kwargs) -> np.ndarray:
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

        # ! ---- Post-analysis

        # Define some general data first:
        # Crop folder and ending from path - required for writing to file.
        img_id = Path(img.name).with_suffix("")

        # Determine the time increment (in terms of hours),
        # referring to injection start, in hours.
        SECONDS_TO_HOURS = 1.0 / 3600
        relative_time = (
            self.img.timestamp - self.injection_start
        ).total_seconds() * SECONDS_TO_HOURS

        # Determine total injected volume
        M3_TO_ML = 1e6
        injected_volume = self.geometry.integrate(tracer.img) * M3_TO_ML

        # ! ---- Export

        if not hasattr(self, "times"):
            self.times = []
        if not hasattr(self, "vols"):
            self.vols = []
        self.times.append(relative_time)
        self.vols.append(injected_volume)

        # Plot and store image with contours
        plot_concentration = kwargs.get("plot_concentration", False)
        write_concentration_to_file = kwargs.get("write_concentration_to_file", False)
        write_data_to_file = kwargs.get("write_data_to_file", False)

        if plot_concentration or write_concentration_to_file or write_data_to_file:

            # Plot
            if plot_concentration:
                plt.figure("Tracer concentration")
                plt.imshow(tracer.img)
                plt.show()

            # Write to file
            if write_concentration_to_file:
                cv2.imwrite(
                    str(self.path_to_results / Path(f"concentration_{img_id}.jpg")),
                    cv2.cvtColor(tracer.img, cv2.COLOR_RGB2BGR),
                )

            # Write array and time to file:
            if write_data_to_file:
                img_array = cv2.resize(tracer.img, (280, 150))
                time = (tracer.timestamp - self.injection_start).total_seconds()
                np.savez(
                    self.path_to_results / Path(f"data_{img_id}.npz"), img_array, time
                )

        return tracer
