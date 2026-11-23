"""Registry of the image porosity detection methods.

A method is a function `detect(baseline, labels, options) -> darsia.Image` with values
in [0, 1] and an options dataclass, the sub-table `[image_porosity.<name>]` of the
config (`"none"`, no detection, has no options). A new method needs a detect function
(here a module `method_<name>.py`, or elsewhere as for `"color_statistics"`), an entry
here, and its name and options in `config/image_porosity.py`.
"""

from dataclasses import dataclass
from typing import Any, Callable

import darsia
from darsia.presets.workflows.config.image_porosity import IMAGE_POROSITY_METHODS
from darsia.presets.workflows.processing.color_statistics import color_statistics_map

from . import method_kmeans, method_none


@dataclass(frozen=True)
class ImagePorosityMethod:
    """A detection method of the image porosity."""

    name: str
    """Name, as in `[image_porosity] method`."""
    detect: Callable[[darsia.Image, darsia.Image, Any], darsia.Image]
    """Function `(baseline, labels, options) -> image porosity`."""


METHODS: dict[str, ImagePorosityMethod] = {
    "none": ImagePorosityMethod("none", method_none.detect),
    "kmeans": ImagePorosityMethod("kmeans", method_kmeans.detect),
    "color_statistics": ImagePorosityMethod("color_statistics", color_statistics_map),
}
"""Detection methods by name."""

assert set(METHODS) == set(IMAGE_POROSITY_METHODS), (
    "The image porosity methods of the registry and the config differ: "
    f"{sorted(METHODS)} and {sorted(IMAGE_POROSITY_METHODS)}."
)
