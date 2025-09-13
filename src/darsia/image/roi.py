"""Polygonal regions of interest (ROI) for darsia.Image."""

from typing import Tuple

import numpy as np
from shapely.geometry import Point, Polygon

import darsia


class ROI:
    """Polygonal region of interest (ROI) in global (physical) coordinates for
    darsia.Image.

    NOTE: Only 2D polygons are supported at the moment.

    """

    def __init__(
        self, coordinates: list[darsia.Coordinate] | darsia.CoordinateArray
    ) -> None:
        """Initialize the ROI with a list of coordinates.

        Args:
            coordinates (list[darsia.Coordinate] | darsia.CoordinateArray): Coordinates
                defining the polygon. If first and last coordinates are not the same,
                the first coordinate will be appended to the end to close the polygon.

        """
        # Check type of coordinates
        assert all([isinstance(c, darsia.Coordinate) for c in coordinates]), (
            "All coordinates must be of the same type (Voxel or Coordinate)."
        )

        # Test for dimensionality
        dim = coordinates[0].dim
        assert dim == 2, "Only 2D polygons are supported at the moment."

        # Make sure the last coordinate is the same as the first
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        # Create a polygon
        self.polygon = Polygon([(c.x, c.y) for c in coordinates])

    def contains(self, point: Tuple[float, float] | np.ndarray) -> bool:
        """Check if the point is inside the polygon."""
        if isinstance(point, np.ndarray):
            point = tuple(point)
        return self.polygon.contains(Point(point))

    def __repr__(self) -> str:
        return f"ROI({self.polygon})"

    def extract_subregion(self, image: darsia.Image) -> darsia.Image:
        """Apply the ROI to an image, returning a new image with the ROI applied.

        If provided, the exterior_value will be used to fill the area outside the polygon.
        If not provided, the area outside the polygon will be filled with values from the image.

        """

        # Find the bounding box of the polygon
        min_x, min_y, max_x, max_y = self.polygon.bounds

        # Extract subregion
        return image.subregion(
            darsia.make_coordinate(
                [
                    [min_x, min_y],
                    [max_x, max_y],
                ]
            )
        )
