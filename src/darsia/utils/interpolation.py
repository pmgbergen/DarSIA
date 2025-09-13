"""Module containing interpolation functionality for scalar data."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares
from pathlib import Path
import pandas as pd

import darsia


def interpolate_measurements_2d(
    measurements: tuple[np.ndarray, np.ndarray, np.ndarray],
    coordinate_system: darsia.CoordinateSystem,
) -> np.ndarray:
    """Determine a voxeled spatial map from measurements through RBF interpolation.

    Arguments:
         measurements (tuple[np.ndarray, np.ndarray, np.ndarray]): tuple of x, y, and data
            measurements, providing the input for interpolation.
        shape (tuple of int): target shape of the output map.
        coordinate_system (darsia.CoordinateSystem): coordinate system of the
            correspoinding physical image.

    Returns:
        np.ndarray: map

    """
    assert len(measurements) == 3, "Measurements must be a tuple of (x, y, data)."
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
    Ny, Nx = coordinate_system.shape
    coords_vector = coordinate_system.coordinates

    # Evaluate interpolation
    interpolated_data_vector = interpolator(coords_vector)
    return interpolated_data_vector.reshape((Ny, Nx), order="F")


def polynomial_interpolation(
    measurements: tuple[np.ndarray, ...],
    coordinate_system: darsia.CoordinateSystem,
    degree: int = 2,
) -> np.ndarray:
    """Determine a voxeled spatial map from measurements through polynomial interpolation.

    Args:
        measurements (tuple[np.ndarray, ...]): tuple of x, y, and data measurements,
            providing the input for interpolation.
        shape (tuple of int): target shape of the output map.
        coordinate_system (darsia.CoordinateSystem): coordinate system of the
            correspoinding physical image.
        degree (int): degree of the polynomial interpolation.

    Returns:
        np.ndarray: map

    """
    # Deterimine dimension of space of polynomial coefficients
    dimension = sum(
        [1 for i in range(degree + 1) for j in range(degree + 1) if i + j <= degree]
    )

    k = -np.ones((degree + 1, degree + 1), dtype=int)
    counter = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            if i + j <= degree:
                k[i, j] = counter
                counter += 1

    def polynomial_interpolator(
        coords: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Polynomial interpolation function."""
        output = np.zeros_like(coords[:, 0])
        for i in range(degree + 1):
            for j in range(degree + 1):
                if i + j <= degree:
                    output += coefficients[k[i, j]] * np.multiply(
                        coords[:, 0] ** i, coords[:, 1] ** j
                    )
        return output

    # Determine coefficients as Least-squares solution
    measurements_coordinates = np.transpose(
        np.vstack(
            (
                measurements[0],
                measurements[1],
            )
        )
    )
    measurements_data = measurements[2]

    def objective_function(coefficients: np.ndarray) -> np.ndarray:
        """Objective function for least squares optimization."""
        return (
            polynomial_interpolator(measurements_coordinates, coefficients)
            - measurements_data
        )

    opt_result = least_squares(objective_function, np.ones(dimension))
    coefficients = opt_result.x

    # Create a mesh of points at which the interpolator shall be evaluated.
    Ny, Nx = coordinate_system.shape
    coords_vector = coordinate_system.coordinates
    return polynomial_interpolator(coords_vector, coefficients).reshape(
        (Ny, Nx), order="F"
    )


def illumination_interpolation(
    measurements: tuple[np.ndarray, ...],
    coordinate_system: darsia.CoordinateSystem,
) -> np.ndarray:
    """Determine a voxeled spatial map from measurements through polynomial interpolation.

    Args:
        measurements (tuple[np.ndarray, ...]): tuple of x, y, and data measurements,
            providing the input for interpolation.
        shape (tuple of int): target shape of the output map.
        coordinate_system (darsia.CoordinateSystem): coordinate system of the
            correspoinding physical image.
        degree (int): degree of the polynomial interpolation.

    Returns:
        np.ndarray: map

    """

    def interpolator(
        coords: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Polynomial interpolation function."""
        output = np.zeros_like(coords[:, 0])
        coord0 = coefficients[0]
        coord1 = coefficients[1]
        coord2 = coefficients[2]
        p = coefficients[4]  # 2 : classical choice
        dist = (
            np.sqrt(
                (coords[:, 0] - coord0) ** 2
                + (coords[:, 1] - coord1) ** 2
                + coord2**2  # 2d case
            )
            ** p
        )
        i0 = coefficients[3]
        output = i0 / dist
        return output

    # Determine coefficients as Least-squares solution
    measurements_coordinates = np.transpose(
        np.vstack(
            (
                measurements[0],
                measurements[1],
            )
        )
    )
    measurements_data = measurements[2]

    def objective_function(coefficients: np.ndarray) -> np.ndarray:
        """Objective function for least squares optimization."""
        return interpolator(measurements_coordinates, coefficients) - measurements_data

    opt_result = least_squares(objective_function, np.ones(5))
    coefficients = opt_result.x

    # Create a mesh of points at which the interpolator shall be evaluated.
    Ny, Nx = coordinate_system.shape
    coords_vector = coordinate_system.coordinates
    return interpolator(coords_vector, coefficients).reshape((Ny, Nx), order="F")


def interpolate_to_image(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    image: darsia.Image,
    method: Literal[
        "rbf", "polynomial", "linear", "quadratic", "cubic", "quartic"
    ] = "rbf",
) -> darsia.Image:
    """Interpolate data to image.

    Args:
        data (np.ndarray): (x,y,measurements) data to be interpolated.
        image (darsia.Image): Image to which data shall be interpolated.
        method (str): Interpolation method to use. Options are:
            - "rbf": Radial Basis Function interpolation (default).
            - "polynomial": Polynomial interpolation.
            - "linear": Linear interpolation.
            - "quadratic": Quadratic interpolation.
            - "cubic": Cubic interpolation.
            - "quartic": Quartic interpolation.

    Returns:
        darsia.Image: interpolated image.

    """
    # Initialize image
    interpolated_image = image.copy()

    # Convert data to 1D columns if provided in mesh format
    assert len(data) == 3, "Data must be a tuple of (x, y, data)."
    if all([len(d.shape) == 2 for d in data]):
        data = (
            np.ravel(data[0]),
            np.ravel(data[1]),
            np.ravel(data[2]),
        )

    if method.lower() == "rbf":
        # Define array through RBF interpolation
        interpolated_image.img = interpolate_measurements_2d(
            data,
            interpolated_image.coordinatesystem,
        )
    elif method.lower() == "illumination":
        # Define array through polynomial interpolation
        interpolated_image.img = illumination_interpolation(
            data,
            interpolated_image.coordinatesystem,
        )
    elif method.lower() in ["linear", "quadratic", "cubic", "quartic"]:
        degrees = {
            "linear": 1,
            "quadratic": 2,
            "cubic": 3,
            "quartic": 4,
        }
        degree = degrees[method.lower()]
        interpolated_image.img = polynomial_interpolation(
            data,
            interpolated_image.coordinatesystem,
            degree,
        )
    else:
        raise NotImplementedError(
            f"""Interpolation method "{method}" is not supported."""
        )

    return interpolated_image


def interpolate_to_image_from_csv(
    csv_file: Path,
    key: str,
    image: darsia.Image,
    method: Literal[
        "rbf", "polynomial", "linear", "quadratic", "cubic", "quartic"
    ] = "rbf",
) -> darsia.Image:
    """Interpolate data from CSV to image.

    Args:
        csv_file (Path): Path to the CSV file containing the data.
        key (str): Key to identify the data in the CSV file.
        image (darsia.Image): Image to which data shall be interpolated.
        method (str): Interpolation method to use. Options are:
            - "rbf": Radial Basis Function interpolation (default).
            - "polynomial": Polynomial interpolation.
            - "linear": Linear interpolation.
            - "quadratic": Quadratic interpolation.
            - "cubic": Cubic interpolation.
            - "quartic": Quartic interpolation.

    Returns:
        darsia.Image: Interpolated image.

    """
    # Convert data to the format expected by interpolate_to_image
    data = pd.read_csv(csv_file)
    x_key = "x" if "x" in data.columns else "X"
    y_key = "y" if "y" in data.columns else "Y"
    x = data[x_key].to_numpy()
    y = data[y_key].to_numpy()
    mean = data[key].to_numpy()

    return interpolate_to_image(
        (x, y, mean),
        image,
        method=method,
    )
