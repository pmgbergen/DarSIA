"""
Module providing structures for tracer analysis. The resulting class is
abstract and needs to be tailored to the specific situation by inheritance.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import daria

# TODO make tracer analysis a child of multicomponentanalysis with num_components=1?


class TracerAnalysis(ABC, daria.AnalysisBase):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for CO2Analysis.

        Args:
            baseline (str, Path or list of such): see daria.GeneralAnalysis.
            config_source (str or Path): see daria.GeneralAnalysis.
            update_setup (bool): see daria.GeneralAnalysis.
        """
        super().__init__(baseline, config, update_setup)

        # Define concentration analysis including determining a cleaning filter.
        self.define_concentration_analysis()
        self._setup_concentration_analysis(
            self.concentration_analysis,
            self.config["concentration"]["cleaning_filter"],
            baseline,
            update_setup,
        )

    @abstractmethod
    def define_concentration_analysis(self) -> None:
        """
        The main purpose of this routine is to define self.concentration_analysis,
        which lies at the heart of this class. It is supposed to determine concentrations
        from image differences. Since the choice of a suitable color channel etc.
        may heavily depend on the situation, this method is abstract and has to
        be overwritten in each specific situation.
        """
        pass

    # ! ---- Auxiliary setup routines

    def _setup_concentration_analysis(
        self,
        concentration_analysis: daria.ConcentrationAnalysis,
        cleaning_filter: Union[str, Path],
        baseline_images: list[Union[str, Path]],
        update: bool = False,
    ) -> None:
        """
        Wrapper to find cleaning filter of the concentration analysis.

        Args:
            concentration_analysis (daria.ConcentrationAnalysis): concentration analysis
                to be set up.
            cleaning_filter (str or Path): path to cleaning filter array.
            baseline_images (list of str or Path): paths to baseline images.
            update (bool): flag controlling whether the calibration and setup should
                be updated.
        """
        # Set volume information
        # TODO include; after also including self.determine_effective_volumes (abstractmethod).
        #        concentration_analysis.update_volumes(self.effective_volumes)

        # Fetch or generate cleaning filter
        if not update and Path(cleaning_filter).exists():
            concentration_analysis.read_cleaning_filter_from_file(cleaning_filter)
        else:
            # Process baseline images used for setting up the cleaning filter
            if self.processed_baseline_images is None:
                self.processed_baseline_images = [
                    self._read(path) for path in baseline_images
                ]

            # Construct the concentration analysis specific cleaning filter
            concentration_analysis.find_cleaning_filter(self.processed_baseline_images)

            # Store the cleaning filter to file for later reuse.
            concentration_analysis.write_cleaning_filter_to_file(cleaning_filter)

    def determine_concentration(self) -> daria.Image:
        """Extract tracer from currently loaded image, based on a reference image.

        Returns:
            daria.Image: image array of spatial concentration map
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract concentration map - includes rescaling
        concentration = self.concentration_analysis(img)

        return concentration
