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


class TailoredConcentrationAnalysis(darsia.SegmentedConcentrationAnalysis):
    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        labels: np.ndarray,
        color: Union[str, callable] = "gray",
        **kwargs,
    ) -> None:
        super().__init__(base, labels, color)

        self.disk_radius = kwargs.get("median_disk_radius", 1)
        self.labels = labels

    def postprocess_signal(self, signal: np.ndarray, img: np.ndarray) -> np.ndarray:

        # Apply median to fill in whilte preserving edges.
        signal = skimage.filters.rank.median(
            skimage.img_as_ubyte(signal), skimage.morphology.disk(self.disk_radius)
        )

        # Convert back to float
        signal = skimage.img_as_float(signal)

        # TODO hardcode coarsening?

        ## Smoothen the signal
        # signal = skimage.restoration.denoise_tv_bregman(
        #   signal, weight=1e-2, eps=1e-6, max_num_iter=100
        # )

        # Resize
        signal = cv2.resize(signal, (280, 150), interpolation=cv2.INTER_AREA)

        # TVD
        signal = skimage.restoration.denoise_tv_bregman(
            signal, weight=8, eps=1e-4, max_num_iter=100
        )

        return super().postprocess_signal(signal)


class BenchmarkTracerAnalysis(LargeFluidFlower, darsia.SegmentedTracerAnalysis):
    """
    Class for managing the well test of the FluidFlower benchmark.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
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
        LargeFluidFlower.__init__(self, baseline, config, update_setup)
        darsia.SegmentedTracerAnalysis.__init__(
            self, baseline, self.effective_volumes, self.labels, config, update_setup
        )

        # The above constructors provide access to the config via self.config.
        # Determine the injection start from the config file. Expect format
        # complying with "%y%m%d %H%M%D", e.g., "211127 083412"
        # TODO as part of the calibration, this will be returned.
        self.injection_start: datetime = datetime.strptime(
            self.config["injection_start"], "%y%m%d %H%M%S"
        )

        # Initialize results dictionary for post-analysis
        self.results: dict = {}

        # Create folder for results if not existent
        self.path_to_results: Path = Path(self.config.get("results_path", "./results"))
        self.path_to_results.parents[0].mkdir(parents=True, exist_ok=True)

        # Store verbosity
        self.verbosity = verbosity

    # ! ---- Analysis tools for detecting the tracer concentration

    def define_tracer_analysis(self) -> darsia.SegmentedConcentrationAnalysis:
        """
        Identify tracer concentration using a reduction to the garyscale space
        """
        tracer_analysis = TailoredConcentrationAnalysis(
            self.base,
            self.labels,
            color="gray",
            **self.config["tracer"],
        )

        return tracer_analysis

    def determine_tracer(self) -> darsia.Image:
        """Extract tracer from currently loaded image, based on a reference image.
        Add expert knowledge, that there is no tracer in the water.

        Returns:
            darsia.Image: image array of spatial concentration map
        """
        # Extract concentration from the analysis
        tracer_concentration = super().determine_tracer()

        # Add expert knowledge: Turn of any signal in the water zone
        # tracer_concentration.img[self.water] = 0

        return tracer_concentration

    def calibrate(self, images: list[Path], injection_rate: float) -> None:
        """
        Calibration routine, taking into account both discontinuity
        across segment interfaces, as well as signal strength.
        For the latter a known injection rate is matched.

        Args:
            images (list of paths): images used for the calibration.
            injetion_rate (float): injection rate in ml/hrs.
        """
        # Read and process the images
        print("Calibration: Processing images...")
        processed_images = [self._read(img) for img in images]

        # Calibrate the segment-wise scaling
        print("Calibration: Segmentwise scaling...")
        self.tracer_analysis.calibrate_segmentation_scaling(processed_images)

        # Calibrate the overall signal via a simple constant rescaling
        print("Calibration: Global scaling...")
        if "scaling" in self.config["tracer"]:
            self.tracer_analysis.update(scaling=self.config["tracer"]["scaling"])
            print(self.tracer_analysis.scaling)
        else:
            self.tracer_analysis.calibrate(
                injection_rate=injection_rate,
                images=processed_images,
                initial_guess=(6, 9),
            )

    # ! ----- Analysis tools

    def single_image_analysis(self, img: Path, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Standard workflow to analyze the tracer concentration.

        Args:
            image (Path): path to single image.
            kwargs: optional keyword arguments, see batch_analysis.

        Returns:
            np.ndarray: tracer concentration map
            dict: dictinary with all stored results from the post-analysis.
        """
        print("single", img)
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

        return tracer, self.results

    # TODO add to baseanalysis?

    def batch_analysis(self, images: list[Path], **kwargs) -> dict:
        """
        Standard batch analysis for the well test performed for the benchmark.

        Args:
            images (list of Path): paths to batch of images.
            kwargs: optional keyword arguments:
                plot_concentration (bool): flag controlling whether the concentration
                    profile is plotted; default False.
                write_concentration_to_file (bool): flag controlling whether the plot from
                    plot_concentration is written to file; default False.

        Returns:
            dict: dictinary with all stored results from the post-analysis.
        """

        for num, img in enumerate(images):

            tic = time.time()

            # Perform dedicated analysis for the current image
            self.single_image_analysis(img, **kwargs)

            # Information to the user
            if self.verbosity:
                print(f"Elapsed time for {img.name}: {time.time()- tic}.")

        return self.results

    def return_results(self) -> dict:
        """
        Return all results collected throughout any analysis performed.
        """
        # TODO restrict to specific list of keywords.

        return self.results

    def write_results_to_file(self, folder: Path) -> None:
        """
        Write results in separate files.

        Args:
            folder (Path): folder where the results are stored.
        """
        for keys in self.results.keys():
            pass
