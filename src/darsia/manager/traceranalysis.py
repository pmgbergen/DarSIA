"""Module providing structures for tracer analysis.

The resulting class is abstract and needs to be tailored to the specific situation by
inheritance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import darsia


class TracerAnalysis(ABC, darsia.ConcentrationAnalysisBase):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for TracerAnalysis.

        Args:
            baseline (str, Path or list of such): see darsia.AnalysisBase.
            config_source (str or Path): see darsia.AnalysisBase.
            update_setup (bool): see darsia.AnalysisBase.

        """
        # Call constructor of AnalysisBase
        super().__init__(baseline, config, update_setup)

        # Define tracer analysis.
        if "tracer" in self.config.keys():
            self.tracer_analysis = self.define_tracer_analysis()

            # Safety check
            if not isinstance(self.tracer_analysis, darsia.ConcentrationAnalysis):
                raise ValueError("tracer_analysis has wrong type.")

            # Setup standard data including the cleaning filter
            tracer_config = self.config.get("tracer", {})
            tracer_cleaning_filter = tracer_config.get(
                "cleaning_filter", "cache/cleaning_filter_tracer.npy"
            )
            self._setup_concentration_analysis(
                self.tracer_analysis,
                tracer_cleaning_filter,
                baseline,
                update_setup,
            )
        else:
            raise ValueError("Tracer analysis not well defined.")

    @abstractmethod
    def define_tracer_analysis(self) -> darsia.ConcentrationAnalysis:
        """
        The main purpose of this routine is to define self.tracer_analysis,
        which lies at the heart of this class. It is supposed to determine tracers
        from image differences. Since the choice of a suitable color channel etc.
        may heavily depend on the situation, this method is abstract and has to
        be overwritten in each specific situation.
        """
        pass

    def determine_tracer(
        self, return_volume: bool = False
    ) -> Union[darsia.Image, tuple[darsia.Image, float]]:
        """Extract tracer from currently loaded image, based on a reference image.

        Args:
            return_volume (bool): flag controlling whether the volume of the
                fluid in the porous geometry is returned.

        Returns:
            darsia.Image: image array of spatial concentration map
            float, optional: occupied volume by the fluid in porous geometry
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract tracer map - includes rescaling
        tracer = self.tracer_analysis(img)

        # Integrate concentration over porous domain and/or return concentration
        if return_volume:
            volume = self.geometry.integrate(tracer.img)
            return tracer, volume
        else:
            return tracer
