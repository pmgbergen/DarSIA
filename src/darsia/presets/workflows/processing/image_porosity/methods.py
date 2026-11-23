"""Registry of the image porosity detection methods.

A method is a function `detect(baseline, labels, options) -> darsia.Image` with values
in [0, 1] and an options dataclass, the sub-table `[image_porosity.<name>]` of the
config. A new method needs a module `method_<name>.py`, an entry here, and its name
and options in `config/image_porosity.py`.
"""

from dataclasses import dataclass
from typing import Any, Callable

import darsia
from darsia.presets.workflows.config.image_porosity import IMAGE_POROSITY_METHODS

from . import method_kmeans


@dataclass(frozen=True)
class ImagePorosityMethod:
    """A detection method of the image porosity."""

    name: str
    """Name, as in `[image_porosity] method`."""
    detect: Callable[[darsia.Image, darsia.Image, Any], darsia.Image]
    """Function `(baseline, labels, options) -> image porosity`."""


METHODS: dict[str, ImagePorosityMethod] = {
    "kmeans": ImagePorosityMethod("kmeans", method_kmeans.detect),
}
"""Detection methods by name."""

assert set(METHODS) == set(IMAGE_POROSITY_METHODS), (
    "The image porosity methods of the registry and the config differ: "
    f"{sorted(METHODS)} and {sorted(IMAGE_POROSITY_METHODS)}."
)
