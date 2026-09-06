"""
Tracer concentration from two images
====================================

A minimal concentration analysis: subclass :class:`~darsia.TracerAnalysis`,
describe how a signal becomes a concentration, and apply it to a test image
relative to a baseline.
"""

import os
from pathlib import Path

import darsia


class TailoredTracerAnalysis(darsia.TracerAnalysis):
    """A tracer analysis with an explicit signal-to-concentration pipeline."""

    def define_tracer_analysis(self) -> darsia.ConcentrationAnalysis:
        return darsia.ConcentrationAnalysis(
            base=self.base,
            signal_reduction=darsia.MonochromaticReduction(color="red"),
            restoration=darsia.TVD(),
            model=darsia.CombinedModel(
                [
                    darsia.LinearModel(scaling=4.0),
                    darsia.ClipModel(min_value=0.0, max_value=1.0),
                ]
            ),
        )

    def single_image_analysis(self, img: Path) -> darsia.Image:
        self.load_and_process_image(img)
        co2 = self.determine_tracer()
        co2.show("CO2 concentration")
        return co2


# %%
# Build the analysis around a baseline image and its configuration file.

image_folder = os.path.join("..", "images")
baseline = os.path.join(image_folder, "co2_0.jpg")
config = os.path.join(image_folder, "config_co2.json")
co2_analysis = TailoredTracerAnalysis(baseline, config, True)

# %%
# Run it on a later frame.

test_image = os.path.join(image_folder, "co2_2.jpg")
test_co2 = co2_analysis.single_image_analysis(Path(test_image))
