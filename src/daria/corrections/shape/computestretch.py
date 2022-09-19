from __future__ import annotations

import numpy as np



def compute_stretch(Nx: int, Ny: int, **kwargs):
    """
    Compute the stretch parameters depending on the stretch center,
    and a known translation.

    Arguments:
        Nx (int): Number of pixels in horizontal direction.
        Ny (int): Number of pixels in vertical direction.
        kwargs (optional keyword arguments):
            "point_source" (list): point that has been translated.
            "point_destination" (list): the ought to be position.
            "stretch_center" (list): the stretch center.
    """



    pt_src = kwargs.pop("point_source", [Ny, Nx])
    pt_dst = kwargs.pop("point_destination", [Ny, Nx])
    stretch_center = kwargs.pop("stretch_center", [int(Ny/2), int(Nx/2)])


    # Update the offset to the center
    horizontal_stretch_center_offset = stretch_center[0] - int(Nx / 2)
    vertical_stretch_center_offset = stretch_center[1] - int(Ny / 2)

    # Compute the tuning parameter as explained in the notes
    horizontal_stretch = -(pt_dst[0] - pt_src[0]) / (
        (pt_src[0] - stretch_center[0]) * pt_src[0] * (Nx - pt_src[0])
    )
    vertical_stretch = -(pt_dst[1] - pt_src[1]) / (
        (pt_src[1] - stretch_center[1]) * pt_src[1] * (Ny - pt_src[1])
    )

    return horizontal_stretch, horizontal_stretch_center_offset, vertical_stretch, vertical_stretch_center_offset