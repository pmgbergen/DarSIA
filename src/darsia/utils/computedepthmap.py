"""Create function for depth map computation."""

import numpy as np
from scipy.interpolate import RBFInterpolator

import darsia as da


def compute_depth_map(
    base: da.Image, depth_measurements: tuple[np.ndarray, ...]
) -> np.ndarray:
    """
    Compute depth map, based on the reported measurements.

    Arguments:
        base (darsia.Image): base image
        depth_measurements (Optional[tuple[np.ndarray, ...]]): tuple of depth
                measurements. Should be provided as horizontal coordinates,
                vertical coordinates, and depth cooridinates corresponding
                to each horizontal and vertical entry.
    Returns:
        (np.ndarray): depth map
    """

    # Determine number of voxels in each dimension - assume 2d image
    Ny, Nx = base.img.shape[:2]
    x = np.arange(Nx)
    y = np.arange(Ny)
    X_pixel, Y_pixel = np.meshgrid(x, y)
    pixel_vector = np.transpose(np.vstack((np.ravel(Y_pixel), np.ravel(X_pixel))))
    coords_vector = base.coordinatesystem.pixelToCoordinate(pixel_vector)
    # Fetch physical dimensions
    # Depth of the rig, measured in discrete points, taking into account expansion.
    # Values taken from the benchmark description.
    # Coordinates at which depth measurements have been taken.
    # Note that the y-coordinate differs depending on the x-coordinate,
    # which dissallows use of np.meshgrid. Instead, the meshgrid is
    # constructed by hand.
    depth_interpolator = RBFInterpolator(
        np.transpose(
            np.vstack(
                (
                    depth_measurements[0],
                    depth_measurements[1],
                )
            )
        ),
        depth_measurements[2],
    )
    # Evaluate depth function to determine depth map
    depth_vector = depth_interpolator(coords_vector)
    return depth_vector.reshape((Ny, Nx))
