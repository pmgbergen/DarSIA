"""Module providing access to standardized Image objects."""

from typing import Literal, Optional

import numpy as np

import darsia

StandardDtype = Literal[np.uint8, np.uint16, np.float32, np.float64, np.bool_]


def zeros_like(
    image: darsia.Image,
    mode: Literal["shape", "voxels"] = "shape",
    dtype: Optional[StandardDtype] = None,
) -> darsia.Image:
    """Create a new image with the same shape and dtype as the input image, filled with zeros."""
    if dtype is None:
        dtype = image.dtype
    if mode == "shape":
        return darsia.Image(np.zeros(image.shape, dtype=dtype), metadata=image.metadata)
    elif mode == "voxels":
        return darsia.Image(
            np.zeros(image.num_voxels, dtype=dtype), metadata=image.metadata
        )
