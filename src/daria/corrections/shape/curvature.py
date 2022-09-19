from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


# TODO: Add an automatic way (using e.g, gradient decent) to choose the parameters.
# OR determine manual tuning rules.
def simple_curvature_correction(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    Correction of bulge and stretch effects.

    Args:
        img (np.ndarray): image array
        kwargs (optional keyword arguments):
            "horizontal_bulge" (float): parameter for the curvature correction related to the
                horizontal bulge of the image.
            "horizontal_stretch" (float): parameter for the curvature correction related to the
                horizontal stretch of the image
            "horizontal_center_offset" (int): offset in terms of pixel of the image center in
                x-direction, as compared to the numerical center
            vertical_bulge (float): parameter for the curvature correction related to the
                vertical bulge of the image.
            "vertical_stretch" (float): parameter for the curvature correction related to the
                vertical stretch of the image
            "vertical_center_offset" (int): offset in terms of pixel of the image center in
                y-direction, as compared to the numerical center
            "interpolation_order (int)": interpolation order to map back transformed image to
                Cartesian pixel grid

    Returns:
        np.ndarray: corrected image

    # NOTE: The same image size is used, i.e., the aspect ratio of the image
    # is taken the same as the input and it is therefore implicitly assumed
    # that the input image already is warped such that the aspect ratio of
    # the image is correct. Also it i
    """
    # Read in tuning parameters
    horizontal_bulge: float = kwargs.pop("horizontal_bulge", 0.0)
    horizontal_stretch: float = kwargs.pop("horizontal_stretch", 0.0)
    horizontal_center_offset: int = kwargs.pop("horizontal_center_offset", 0)
    vertical_bulge: float = kwargs.pop("vertical_bulge", 0.0)
    vertical_stretch: float = kwargs.pop("vertical_stretch", 0.0)
    vertical_center_offset: int = kwargs.pop("vertical_center_offset", 0)
    interpolation_order: int = kwargs.pop("interpolation_order", 1)

    # Assume a true image in the form of an array is provided
    if not isinstance(img, np.ndarray):
        raise Exception(
            "Invalid image data. Provide either a path to an image or an image array."
        )

    # Read size of image
    Ny, Nx = img.shape[:2]

    # NOTE: Finding the true centre of the image actually depends on many factors
    # including lense properties. Thus, the task is actually quite hard. Here, a
    # simple approach is used, simply choosing the numerical centre of the image
    # corrected by the user.

    # Image center in pixels, but in (col, row) order
    image_center = [
        round(Nx / 2) + horizontal_center_offset,
        round(Ny / 2) + vertical_center_offset,
    ]

    # Define coordinate system relative to image center, in terms of pixels
    x = np.arange(Nx) - image_center[0]
    y = np.arange(Ny) - image_center[1]

    # Construct associated meshgrid with Cartesian indexing
    X, Y = np.meshgrid(x, y)

    # Warp the coordinate system nonlinearly, correcting for bulge and stretch effects.
    Xmod = (
        X
        + horizontal_bulge * np.multiply(X, (np.max(Y) - Y) * (Y - np.min(Y)))
        + horizontal_stretch * X * (np.max(X) - X) * (X - np.min(X))
    )
    Ymod = (
        Y
        + vertical_bulge * np.multiply(Y, (np.max(X) - X) * (X - np.min(X)))
        + vertical_stretch * Y * (np.max(Y) - Y) * (Y - np.min(Y))
    )

    # Map corrected grid back to positional arguments, i.e. invert the definition
    # of the local coordinate system
    Xmod += image_center[0]
    Ymod += image_center[1]

    # Create out grid as the corrected grid, use (row,col) format
    out_grid = np.array([Ymod.ravel(), Xmod.ravel()])

    # Define the shape corrected image.
    img_mod = np.zeros_like(img, dtype=img.dtype)

    # Do interpolate original image on the new grid
    for i in range(img.shape[2]):
        in_data = img[:, :, i]
        im_array = map_coordinates(in_data, out_grid, order=interpolation_order)
        img_mod[:, :, i] = im_array.reshape(img.shape[:2]).astype(img.dtype)

    return img_mod
