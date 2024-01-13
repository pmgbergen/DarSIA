"""
Module containing interpolation functionality.

"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator

import darsia


def interpolate_measurements(
    measurements: tuple[np.ndarray, ...],
    shape: tuple[int],
    coordinate_system: darsia.CoordinateSystem,
) -> np.ndarray:
    """
    Determine a voxeled spatial map from measurements through interpolation.

    Arguments:
         measurements (tuple[np.ndarray, ...]): tuple of x, y, and data measurements,
            providing the input for interpolation.
        shape (tuple of int): target shape of the output map.

    Returns:
        np.ndarray: map

    """

    # Create an interpolation object from data.
    interpolator = RBFInterpolator(
        np.transpose(
            np.vstack(
                (
                    measurements[0],
                    measurements[1],
                )
            )
        ),
        measurements[2],
    )

    # Create a mesh of points at which the interpolator shall be evaluated.
    Ny, Nx = shape
    x = np.arange(Nx)
    y = np.arange(Ny)
    X_pixel, Y_pixel = np.meshgrid(x, y)
    pixel_vector = np.transpose(np.vstack((np.ravel(Y_pixel), np.ravel(X_pixel))))
    coords_vector = coordinate_system.pixelToCoordinate(pixel_vector)

    # Evaluate interpolation
    interpolated_data_vector = interpolator(coords_vector)
    return interpolated_data_vector.reshape((Ny, Nx))
