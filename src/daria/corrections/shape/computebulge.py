from __future__ import annotations


def compute_bulge(Nx: int, Ny: int, **kwargs):
    """
    Compute the bulge parameters depending on the maximum number of pixels
    that the image has been displaced on each side.

    Arguments:
        Nx (int): Number of pixels in horizontal direction.
        Ny (int): Number of pixels in vertical direction.
        kwargs (optional keyword arguments):
            "left" (int): the maximum number of pixels that the image
                          has been displaced on the left side
            "right" (int): the maximum number of pixels that the image
                          has been displaced on the right side
            "top" (int): the maximum number of pixels that the image
                          has been displaced on the top
            "bottom" (int): the maximum number of pixels that the image
                          has been displaced on the bottom
    """

    left = kwargs.pop("left", 0)
    right = kwargs.pop("right", 0)
    top = kwargs.pop("top", 0)
    bottom = kwargs.pop("bottom", 0)

    # Determine the center of the image
    image_center = [
        int(Nx * (left + 1e-6) / (left + right + 2e-6)),
        int(Ny * (top + 1e-6) / (top + bottom + 2e-6)),
    ]

    # Determine the offset of the numerical center of the image
    horizontal_bulge_center_offset = image_center[0] - int(Nx / 2)
    vertical_bulge_center_offset = image_center[1] - int(Ny / 2)

    # Determine the bulge tuning coefficients as explained in the daria notes
    # Assume here that the maximum impressions are applied at the image center
    horizontal_bulge = left / (
        (left - image_center[0]) * image_center[1] * (Ny - image_center[1])
    )
    vertical_bulge = top / (
        (top - image_center[1]) * image_center[0] * (Nx - image_center[0])
    )

    return (
        horizontal_bulge,
        horizontal_bulge_center_offset,
        vertical_bulge,
        vertical_bulge_center_offset,
    )
