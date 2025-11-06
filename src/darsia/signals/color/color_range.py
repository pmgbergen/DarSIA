"""Module for tracking the range of active colors in an image."""

import darsia
import numpy as np


class ColorRange:
    """Class to track the range of colors in an image or a set of images."""

    def __init__(
        self,
        images: list[darsia.Image],
        baseline: darsia.Image | None = None,
        mask: darsia.Image | None = None,
    ) -> None:
        self.relative = baseline is not None
        """Determine whether absolute or relative color range is considered."""

        # Initialize min and max color arrays
        min_color = np.array([[np.inf, np.inf, np.inf]])
        max_color = np.array([[-np.inf, -np.inf, -np.inf]])

        for _image in images:
            # Consider absolute or relative image
            image = _image.copy()
            if baseline is not None:
                image.img -= baseline.img
            # Deactivate areas outside mask
            if mask is not None:
                image.img[~mask.img] = 0.0

            _min_color = np.min(image.img.reshape((-1, 3)), axis=0)
            _max_color = np.max(image.img.reshape((-1, 3)), axis=0)
            min_color = np.min(np.vstack((min_color, _min_color)), axis=0)
            max_color = np.max(np.vstack((max_color, _max_color)), axis=0)

        self.min_color = min_color.flatten()
        """Minimum color observed."""
        self.max_color = max_color.flatten()
        """Maximum color observed."""
        self.range = (
            (self.min_color[0], self.max_color[0]),
            (self.min_color[1], self.max_color[1]),
            (self.min_color[2], self.max_color[2]),
        )
        """Tuple of color ranges observed."""
        self.center = (self.min_color + self.max_color) / 2
        """Center of the color range."""
        self.extent = self.max_color - self.min_color
        """Color extent observed."""

    def __str__(self) -> str:
        return (
            f"""ColorRange(min_color={self.min_color}, max_color={self.max_color}, """
            f"""range={self.range})"""
        )

    def __repr__(self) -> str:
        return self.__str__()
