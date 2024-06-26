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
    """Analogon of np.zeros_like but for darsia.Image objects.

    Args:
        image (darsia.Image): input image
        mode (Literal["shape", "voxels"], optional): mode of the output image. Defaults to
            "shape".
        dtype (Optional[StandardDtype], optional): dtype of the output image. Defaults to None.

    Returns:
        darsia.Image: output image

    """
    if dtype is None:
        dtype = image.dtype
    if mode == "shape":
        ImageType = type(image)
        return ImageType(np.zeros(image.shape, dtype=dtype), metadata=image.metadata())
    elif mode == "voxels":
        return darsia.ScalarImage(
            np.zeros(image.num_voxels, dtype=dtype), metadata=image.metadata()
        )


def ones_like(
    image: darsia.Image,
    mode: Literal["shape", "voxels"] = "shape",
    dtype: Optional[StandardDtype] = None,
) -> darsia.Image:
    """Analogon of np.ones_like but for darsia.Image objects.

    Args:
        image (darsia.Image): input image
        mode (Literal["shape", "voxels"], optional): mode of the output image. Defaults to
            "shape".
        dtype (Optional[StandardDtype], optional): dtype of the output image. Defaults to None.

    Returns:
        darsia.Image: output image

    """
    if dtype is None:
        dtype = image.dtype
    if mode == "shape":
        ImageType = type(image)
        return ImageType(np.ones(image.shape, dtype=dtype), metadata=image.metadata())
    elif mode == "voxels":
        return darsia.ScalarImage(
            np.ones(image.num_voxels, dtype=dtype), metadata=image.metadata()
        )
