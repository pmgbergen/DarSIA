"""Image porosity: a small library of detection methods, storage and reading.

See `porosity.py` for the meaning and the three steps (detection, cutoff, boundary
band), and `methods.py` for the registry of detection methods.
"""

from .methods import METHODS, ImagePorosityMethod
from .porosity import (
    ImagePorosity,
    boolean_image_porosity,
    compute_image_porosity,
    detect_image_porosity,
    load_image_porosity,
    save_image_porosity,
)

__all__ = [
    "METHODS",
    "ImagePorosity",
    "ImagePorosityMethod",
    "boolean_image_porosity",
    "compute_image_porosity",
    "detect_image_porosity",
    "load_image_porosity",
    "save_image_porosity",
]
