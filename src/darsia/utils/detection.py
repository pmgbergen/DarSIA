"""Module containing utilities for value, color, points detection."""

import numpy as np

import darsia

from typing import Union, overload
from scipy.spatial.transform import Rotation


def detect_value(
    img: darsia.ScalarImage, value: float, tolerance: float = 0.01
) -> darsia.VoxelArray:
    """Detects a value in an image.

    Args:
        img (darsia.ScalarImage): The image to detect the value in.
        value (float): The value to detect.
        tolerance (float, optional): The tolerance for the value detection. Defaults to 0.1.

    Returns:
        darsia.VoxelArray: Pixels with the detected value.

    """
    # Find pixels with the value within the tolerance
    indices = np.where(np.abs(img.img - value) < tolerance)

    # Convert indices to array of two colums
    indices = np.vstack(indices).T

    # Convert the indices to a VoxelArray
    return darsia.VoxelArray(indices)


def detect_color(
    img: darsia.OpticalImage,
    color: Union[list[float], np.ndarray],
    tolerance: float = 0.01,
) -> darsia.VoxelArray:
    """Detects a color in an image.

    Args:
        img (darsia.OpticalImage): The image to detect the color in. Assumed to be in RGB
            format. With values in the range [0, 1].
        color (Union[list[float], np.ndarray]): The color to detect, assumed to be in
            RGB format. With values in the range [0, 1].
        tolerance (float, optional): The tolerance for the color detection. Defaults to 0.1.

    Returns:
        darsia.VoxelArray: Pixels with the detected color.

    """
    distance = darsia.ScalarImage(np.linalg.norm(img.img - color, axis=2))
    return detect_value(distance, 0, tolerance=tolerance)


def monochromatic_concentration_analysis(
    img, color: Union[list[float], np.ndarray]
) -> darsia.Image:
    """Performs monochromatic concentration analysis on an image."""
    # Find orthogonal colors to the input color
    orthogonal_colors = darsia.orthogonal_colors(color)

    # Define concentration analysis
    concentration_analysis = darsia.ConcentrationAnalysis(
        model=darsia.KernelInterpolation(
            kernel=darsia.LinearKernel(),
            supports=np.vstack((color, orthogonal_colors)),
            values=[1, 0, 0],
        )
    )

    # Perform concentration analysis
    concentration = concentration_analysis(img)
    return concentration


def orthogonal_colors(color: np.ndarray) -> np.ndarray:
    """Returns the orthogonal colors to the input color.

    Args:
        color (Union[list[float], np.ndarray]): The color to find the orthogonal colors to.

    Returns:
        np.ndarray: The orthogonal colors (as rows) to the input color.

    """
    # Define (normalized) plane normal to the input color
    v1 = np.array([1, 0, 0], dtype=float)
    v2 = np.array([0, 1, 0], dtype=float)
    normal = np.cross(v1, v2)
    normal *= np.linalg.norm(color) / np.linalg.norm(normal)

    # Find the 3d rotation matrix to get from normal to color
    rotation = Rotation.align_vectors(color, normal)[0]

    # Define the orthogonal colors
    orthogonal_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    orthogonal_colors = rotation.apply(orthogonal_colors)

    # Normalize the orthogonal colors to be within the cube [0,1]**3
    orthogonal_colors[:, 0] /= np.max(orthogonal_colors[:, 0])
    orthogonal_colors[:, 1] /= np.max(orthogonal_colors[:, 1])

    return orthogonal_colors[:, :2].T


@overload
def detect_closest_point(
    points: darsia.VoxelArray, target: darsia.Voxel
) -> darsia.Voxel: ...


@overload
def detect_closest_point(
    points: darsia.CoordinateArray, target: darsia.Coordinate
) -> darsia.Coordinate: ...


def detect_closest_point(
    points: Union[darsia.VoxelArray, darsia.CoordinateArray],
    target: Union[darsia.Voxel, darsia.Coordinate],
) -> Union[darsia.Voxel, darsia.Coordinate]:
    """Detects the point closest to a target.

    Args:
        points (Union[darsia.VoxelArray, darsia.CoordinateArray]): The points to search for
            the closest one.
        target (Union[darsia.Voxel, darsia.Coordinate]): The target point.

    Returns:
        Union[darsia.Voxel, darsia.Coordinate]: The point in points closest to the target.

    """
    # Find the voxel with the smallest distance to the target
    distances = np.linalg.norm(points - target, axis=1)
    closest_point = points[np.argmin(distances)]

    return closest_point
