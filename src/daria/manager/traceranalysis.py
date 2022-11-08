"""
Module providing structures for tracer analysis. The resulting class is
abstract and needs to be tailored to the specific situation by inheritance.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

import daria


class TracerAnalysis(ABC, daria.ConcentrationAnalysisBase):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        # TODO volumes
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for TracerAnalysis.

        Args:
            baseline (str, Path or list of such): see daria.GeneralAnalysis.
            config_source (str or Path): see daria.GeneralAnalysis.
            update_setup (bool): see daria.GeneralAnalysis.
        """
        super().__init__(baseline, config, update_setup)

        # Define tracer analysis including determining a cleaning filter.
        self.tracer_analysis = self.define_tracer_analysis()

        # Safety check
        if not isinstance(self.tracer_analysis, daria.ConcentrationAnalysis):
            raise ValueError("tracer_analysis has wrong type.")

        self._setup_concentration_analysis(
            self.tracer_analysis,
            self.config["tracer"]["cleaning_filter"],
            baseline,
            update_setup,
        )

    @abstractmethod
    def define_tracer_analysis(self) -> daria.ConcentrationAnalysis:
        """
        The main purpose of this routine is to define self.tracer_analysis,
        which lies at the heart of this class. It is supposed to determine tracers
        from image differences. Since the choice of a suitable color channel etc.
        may heavily depend on the situation, this method is abstract and has to
        be overwritten in each specific situation.
        """
        pass

    def determine_tracer(self) -> daria.Image:
        """Extract tracer from currently loaded image, based on a reference image.

        Returns:
            daria.Image: image of spatial tracer map
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract tracer map - includes rescaling
        tracer = self.tracer_analysis(img)

        return tracer


class SegmentedTracerAnalysis(ABC, daria.ConcentrationAnalysisBase):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        volumes: np.ndarray,
        labels: np.ndarray,
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for TracerAnalysis.

        Args:
            baseline (str, Path or list of such): see daria.GeneralAnalysis.
            volumes (np.ndarray): effective pixel volumes.
            labels (np.ndarray): labeled image identifying the different segments.
            config_source (str or Path): see daria.GeneralAnalysis.
            update_setup (bool): see daria.GeneralAnalysis.
        """
        super().__init__(baseline, config, update_setup)

        # Cache the labels
        self.labels = labels

        # Define tracer analysis including determining a cleaning filter.
        self.tracer_analysis = self.define_tracer_analysis()

        # Feed tracer analysis with effective volumes
        self.tracer_analysis.update_volumes(volumes)

        # Safety check
        if not isinstance(self.tracer_analysis, daria.ConcentrationAnalysis):
            raise ValueError("tracer_analysis has wrong type.")

        self._setup_concentration_analysis(
            self.tracer_analysis,
            self.config["tracer"]["cleaning_filter"],
            baseline,
            update_setup,
        )

    @abstractmethod
    def define_tracer_analysis(self) -> daria.SegmentedConcentrationAnalysis:
        """
        The main purpose of this routine is to define self.tracer_analysis,
        which lies at the heart of this class. It is supposed to determine tracers
        from image differences. Since the choice of a suitable color channel etc.
        may heavily depend on the situation, this method is abstract and has to
        be overwritten in each specific situation.
        """
        pass

    def determine_tracer(self) -> daria.Image:
        """Extract tracer from currently loaded image, based on a reference image.

        Returns:
            daria.Image: image of spatial tracer map
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract tracer map - includes rescaling
        tracer = self.tracer_analysis(img)

        return tracer
