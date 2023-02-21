"""
Module containing a standardized FluidFlower analysis.
It will require further information for specific rigs
to be obtained through multiple inheritance.

"""
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia
from darsia.presets.fluidflower.benchmarkco2model import (
    benchmark_binary_cleaning_preset,
    benchmark_concentration_analysis_preset,
)


class FluidFlowerCO2Analysis(darsia.CO2Analysis):
    """
    Class for managing the FluidFlower benchmark analysis.
    Identifiers for CO2 dissolved in water and CO2(g) are
    tailored to the benchmark set of the large rig.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        results: Union[str, Path],
        update_setup: bool = False,
        verbosity: bool = True,
    ) -> None:
        """
        Sets up fixed config file required for preprocessing.

        Args:
            baseline (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            results (str or Path): path to results directory
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
            verbosity  (bool): flag controlling whether results of the
                post-analysis are printed to screen; default is False.
        """
        darsia.CO2Analysis.__init__(self, baseline, config, update_setup)

        # Create folder for results if not existent
        self.path_to_results: Path = Path(results)

        # Store verbosity
        self.verbosity = verbosity

    # ! ---- Segmentation tools for detecting the different CO2 phases

    def define_co2_analysis(self) -> darsia.PriorPosteriorConcentrationAnalysis:
        """
        FluidFlower Benchmark preset for detecting CO2.

        Returns:
            PriorPosteriorConcentrationAnalysis: detector for CO2

        """
        # Extract/define the binary cleaning contribution of the co2 analysis.
        self.co2_binary_cleaning = benchmark_binary_cleaning_preset(
            self.base, self.config["co2"]
        )

        # Define concentration analysis
        return benchmark_concentration_analysis_preset(
            self.base, self.labels, self.config["co2"]
        )

    def define_co2_gas_analysis(self) -> darsia.PriorPosteriorConcentrationAnalysis:
        """
        FluidFlower Benchmark preset for detecting CO2 gas.

        Returns:
            PriorPosteriorConcentrationAnalysis: detector for CO2(g)

        """
        # Extract/define the binary cleaning contribution of the co2(g) analysis.
        self.co2_gas_binary_cleaning = benchmark_binary_cleaning_preset(
            self.base, self.config["co2(g)"]
        )

        # Define concentration analysis
        return benchmark_concentration_analysis_preset(
            self.base, self.labels, self.config["co2(g)"]
        )

    def _expert_knowledge_co2(self) -> np.ndarray:
        """
        Retrieve expert knowledge, i.e., areas with possibility for CO2.

        Returns:
            np.ndarray: mask with possibility for CO2.

        """
        return np.ones(self.base.img.shape[:2], dtype=bool)

    def _expert_knowledge_co2_gas(self, co2: np.ndarray) -> np.ndarray:
        """
        Retrieve expert knowledge, i.e., areas with possibility for CO2(g).

        Args:
            co2 (darsia.Image): mask of CO2.

        Returns:
            np.ndarray: mask with possibility for CO2(g)

        """
        return co2.img

    def determine_co2_mask(self) -> darsia.Image:
        """Determine CO2.

        Returns:
            darsia.Image: boolean image detecting CO2.
        """
        # Apply expert knowledge
        expert_knowledge = self._expert_knowledge_co2()
        self.co2_analysis.update(mask=expert_knowledge)

        # Extract co2 from analysis
        co2 = super().determine_co2()

        # Add expert knowledge, part 2.
        co2.img[~expert_knowledge] = 0
        if np.any(np.logical_not(expert_knowledge)):
            co2.img = self.co2_binary_cleaning(co2.img)

        return co2

    def determine_co2_gas_mask(self, co2: darsia.Image) -> darsia.Image:
        """Determine CO2.

        Args:
            co2 (darsia.Image): boolean image detecting all co2.

        Returns:
            darsia.Image: boolean image detecting CO2(g).
        """
        # Apply expert knowledge.
        expert_knowledge = self._expert_knowledge_co2_gas(co2)
        self.co2_gas_analysis.update(mask=expert_knowledge)

        # Extract co2 from analysis
        co2_gas = super().determine_co2_gas()

        # Add expert knowledge, part 2.
        co2_gas.img[~expert_knowledge] = 0
        if np.any(np.logical_not(expert_knowledge)):
            co2_gas.img = self.co2_gas_binary_cleaning(co2_gas.img)

        return co2_gas

    # ! ---- Segmentation routines

    def single_image_analysis(self, img: Union[Path, darsia.Image], **kwargs) -> None:
        """
        Standard workflow to analyze CO2 phases.

        Args:
            image (Path or Image): path to single image.
            kwargs: optional keyword arguments:
                plot_contours (bool): flag controlling whether the original image
                    is plotted with contours of the two CO2 phases; default False.
                write_contours_to_file (bool): flag controlling whether the plot from
                    plot_contours is written to file; default False.
                write_segmentation_to_file (bool): flag controlling whether the
                    CO2 segmentation is written to file, where water, dissolved CO2
                    and CO2(g) get decoded 0, 1, 2, respectively; default False.
                write_coarse_segmentation_to_file (bool): flag controlling whether
                    a coarse (280 x 150) representation of the CO2 segmentation from
                    write_segmentation_to_file is written to file; default False.
        """
        # ! ----  Pre-processing

        # Load the current image
        if isinstance(img, darsia.Image):
            self.img = img.copy()
        else:
            self.load_and_process_image(img)

        # ! ----  Segmentation

        # Determine binary mask detecting any(!) CO2
        co2 = self.determine_co2_mask()

        # Determine binary mask detecting mobile CO2.
        co2_gas = self.determine_co2_gas_mask(co2)

        # ! ---- Storage of segmentation

        # Define some general data first:
        # Crop folder and ending from path - required for writing to file.
        img_id = Path(img.name).with_suffix("")

        # Plot and store image with contours
        plot_contours = kwargs.pop("plot_contours", True)
        write_contours_to_file = kwargs.pop("write_contours_to_file", True)

        if plot_contours or write_contours_to_file:

            # Start with the original image
            original_img = np.copy(self.img.img)
            original_img = skimage.img_as_ubyte(original_img)

            # Overlay the original image with contours for CO2
            contours_co2, _ = cv2.findContours(
                skimage.img_as_ubyte(co2.img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(original_img, contours_co2, -1, (0, 255, 0), 5)

            # Overlay the original image with contours for CO2(g)
            contours_co2_gas, _ = cv2.findContours(
                skimage.img_as_ubyte(co2_gas.img),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(original_img, contours_co2_gas, -1, (255, 255, 0), 5)

            # Plot
            if plot_contours:
                plt.figure("Image with contours of CO2 segmentation")
                plt.imshow(original_img)
                plt.show()

            # Write corrected image with contours to file
            if write_contours_to_file:
                (self.path_to_results / Path("contour_plots")).mkdir(
                    parents=True, exist_ok=True
                )
                original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(
                        self.path_to_results
                        / Path("contour_plots")
                        / Path(f"{img_id}_with_contours.jpg")
                    ),
                    original_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )

        # Write segmentation to file
        write_segmentation_to_file = kwargs.pop("write_segmentation_to_file", False)
        write_coarse_segmentation_to_file = kwargs.pop(
            "write_coarse_segmentation_to_file", False
        )

        if write_segmentation_to_file or write_coarse_segmentation_to_file:

            # Generate segmentation with codes:
            # 0 - water
            # 1 - dissolved CO2
            # 2 - gaseous CO2
            segmentation = np.zeros(self.img.img.shape[:2], dtype=int)
            segmentation[co2.img] += 1
            segmentation[co2_gas.img] += 1

            # Store fine scale segmentation
            if write_segmentation_to_file:
                (self.path_to_results / Path("npy_segmentation")).mkdir(
                    parents=True, exist_ok=True
                )
                np.save(
                    self.path_to_results
                    / Path("npy_segmentation")
                    / Path(f"{img_id}_segmentation.npy"),
                    segmentation,
                )

            # Store coarse scale segmentation
            if write_coarse_segmentation_to_file:
                coarse_shape = (150, 280)
                coarse_shape_reversed = tuple(reversed(coarse_shape))

                co2_coarse = skimage.img_as_bool(
                    cv2.resize(
                        skimage.img_as_ubyte(
                            co2.img,
                        ),
                        coarse_shape_reversed,
                        interpolation=cv2.INTER_AREA,
                    )
                )
                co2_gas_coarse = skimage.img_as_bool(
                    cv2.resize(
                        skimage.img_as_ubyte(
                            co2_gas.img,
                        ),
                        coarse_shape_reversed,
                        interpolation=cv2.INTER_AREA,
                    )
                )

                coarse_segmentation = np.zeros(coarse_shape, dtype=int)
                coarse_segmentation[co2_coarse] += 1
                coarse_segmentation[co2_gas_coarse] += 1

                # Store segmentation as npy array
                (self.path_to_results / Path("coarse_npy_segmentation")).mkdir(
                    parents=True, exist_ok=True
                )
                np.save(
                    self.path_to_results
                    / Path("coarse_npy_segmentation")
                    / Path(f"{img_id}_coarse_segmentation.npy"),
                    coarse_segmentation,
                )

                ## Store segmentation as csv file
                # (self.path_to_results / Path("coarse_csv_segmentation")).mkdir(
                #    parents=True, exist_ok=True
                # )
                # segmentation_to_csv(
                #    self.path_to_results
                #    / Path("coarse_csv_segmentation")
                #    / Path(f"{img_id}_coarse_segmentation.csv"),
                #    coarse_segmentation,
                #    img.name,
                # )
