"""Module with auxiliary functions for color computations."""

import numpy as np

import darsia


def get_mean_color(
    image: darsia.Image, mask: darsia.Image | np.ndarray | None = None
) -> np.ndarray:
    """Calculate the mean color of an image, optionally masked by a boolean mask.

    Args:
        image (darsia.Image): The image from which to calculate the mean color.
        mask (darsia.Image | np.ndarray | None): Optional mask to apply on the image.
            If `None`, the entire image is used.

    Returns:
        np.ndarray: The mean color of the image, calculated as the average of RGB values.

    """
    if isinstance(mask, darsia.Image):
        subimage = image.img[mask.img]
    elif isinstance(mask, np.ndarray):
        subimage = image.img[mask]
    else:
        subimage = image.img
    return np.mean(subimage.reshape(-1, 3), axis=0)
