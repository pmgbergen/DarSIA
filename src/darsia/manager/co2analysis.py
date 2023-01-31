"""
Module providing structure for analyzing images of multi-component multi-phase
experiments, with the goal to identify distinct components/phases. In practice,
it may have to be tailored to the specific scenario by inheritance. For more
general use, a generalized multicomponent analysis manager should be used.
This module is tailored to CO2 experiments in water, strongly motivated by the
International FluidFlower Benchmark study.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import darsia


class CO2Analysis(ABC, darsia.ConcentrationAnalysisBase):
    """
    General setup for an image analysis of time series aiming at analyzing CO2 evolution.
    This class inherits from darsia.TracerAnalysis.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for two-component analysis, where the two specific components
        are identified with CO2(g) and CO2 saturated water. Pure water will be treated
        as neutral (third) phase.

        Args:
            baseline (str, Path or list of such): see darsia.GeneralAnalysis.
            config_source (str or Path): see darsia.GeneralAnalysis.
            update_setup (bool): see darsia.GeneralAnalysis.
        """
        # Call constructor of TracerAnalysis
        super().__init__(baseline, config, update_setup)

        # Define the specific components - use external methods to allow for further tailoring.
        # The main philosophy is to first separate water from the rest, which here will be
        # simply called CO2. In the CO2 region, additional separation will be required to
        # separate CO(g) from CO2 saturated water.
        self.co2_analysis = self.define_co2_analysis()
        self.co2_gas_analysis = self.define_co2_gas_analysis()

        # Safety check
        if not isinstance(self.co2_analysis, darsia.ConcentrationAnalysis):
            raise ValueError("co2_analysis has wrong type.")

        if not isinstance(self.co2_gas_analysis, darsia.ConcentrationAnalysis):
            raise ValueError("co2_gas_analysis has wrong type.")

        # Setup standard data including the cleaning filter
        co2_config = self.config.get("co2", {})
        co2_cleaning_filter = co2_config.get(
            "cleaning_filter", "cache/cleaning_filter_co2.npy"
        )
        self._setup_concentration_analysis(
            self.co2_analysis,
            co2_cleaning_filter,
            baseline,
            update_setup,
        )

        co2_gas_config = self.config.get("co2(g)", {})
        co2_gas_cleaning_filter = co2_gas_config.get(
            "cleaning_filter", "cache/cleaning_filter_co2_gas.npy"
        )
        self._setup_concentration_analysis(
            self.co2_gas_analysis,
            co2_gas_cleaning_filter,
            baseline,
            update_setup,
        )

    @abstractmethod
    def define_co2_analysis(self) -> darsia.BinaryConcentrationAnalysis:
        """
        Empty method which should define self.co2_analysis of type
        darsia.BinaryConcentrationAnalysis.

        Example:
        self.co2_analysis = darsia.BinaryConcentrationAnalysis(
            self.base, color="red", **self.config["co2"]
        )
        """
        pass

    @abstractmethod
    def define_co2_gas_analysis(self) -> darsia.BinaryConcentrationAnalysis:
        """
        Empty method which should define self.co2_gas_analysis of type
        darsia.BinaryConcentrationAnalysis.

        Example:
        self.mobile_co2_analysis = darsia.BinaryConcentrationAnalysis(
            self.base, color="blue", **self.config["mobile_co2"]
        )
        """
        pass

    def determine_co2(self) -> darsia.Image:
        """
        Extract CO2 from currently loaded image, based on a reference image.

        Returns:
            darsia.Image: binary image of spatial CO2 distribution.
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract binary concentration
        co2 = self.co2_analysis(img)

        return co2

    def determine_co2_gas(self) -> darsia.Image:
        """
        Extract CO2(g) from currently loaded image, based on a reference image.

        Returns:
            darsia.Image: binary image of spatial CO2(g) distribution.
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract binary concentration
        co2 = self.co2_gas_analysis(img)

        return co2
