"""
Example script for simple image analysis. By comparison of images
of the same well test, a tracer concentration can be determined.
"""

import os
from pathlib import Path

import darsia

# ! --- Source


class TailoredTracerAnalysis(darsia.TracerAnalysis):
    """Example of a tailored tracer analysis object."""

    def define_tracer_analysis(self) -> darsia.ConcentrationAnalysis:
        """Definition of signal to data conversion."""

        return darsia.ConcentrationAnalysis(
            base=self.base,  # baseline image
            signal_reduction=darsia.MonochromaticReduction(
                color="red"
            ),  # signal reduction
            restoration=darsia.TVD(),  # restoration
            model=darsia.CombinedModel(  # signal to data conversion
                [
                    darsia.LinearModel(scaling=4.0),
                    darsia.ClipModel(**{"min value": 0.0, "max value": 1.0}),
                ]
            ),
        )

    def single_image_analysis(self, img: Path) -> darsia.Image:
        """Dedicated image analysis."""

        self.load_and_process_image(test_image)
        co2 = self.determine_tracer()
        co2.show("CO2", 5)

        return co2


# ! ---- Run script

# Define path to image folder
image_folder = f"{os.path.dirname(__file__)}/images/"

# Define final co2 analysis object
baseline = image_folder + "co2_0.jpg"
config = image_folder + "config_co2.json"
update_setup = True
co2_analysis = TailoredTracerAnalysis(baseline, config, update_setup)

# Analyze test image
test_image = image_folder + "co2_2.jpg"
test_co2 = co2_analysis.single_image_analysis(test_image)

# Store the final results
test_co2.write_array("co2_test")
test_co2.write("co2_test.jpg", quality=20)
