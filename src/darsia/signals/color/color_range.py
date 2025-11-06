"""Module for tracking the range of active colors in an image."""

import darsia
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


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

    def save(self, path: Path) -> None:
        """Save color range to a json file.

        Args:
            path (Path): Path to the json file.

        """
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(
                {
                    "relative": self.relative,
                    "min_color": self.min_color.tolist(),
                    "max_color": self.max_color.tolist(),
                },
                f,
                indent=4,
            )
        logger.info("Saved color range to %s", path.with_suffix(".json"))

    @classmethod
    def load(cls, path: Path) -> "ColorRange":
        """Load color range from a json file.

        Args:
            path (Path): Path to the json file.

        Returns:
            ColorRange: Loaded color range.

        """
        with open(path.with_suffix(".json"), "r") as f:
            data = json.load(f)
        color_range = cls.__new__(cls)
        color_range.relative = data["relative"]
        color_range.min_color = np.array(data["min_color"])
        color_range.max_color = np.array(data["max_color"])
        color_range.range = (
            (color_range.min_color[0], color_range.max_color[0]),
            (color_range.min_color[1], color_range.max_color[1]),
            (color_range.min_color[2], color_range.max_color[2]),
        )
        color_range.center = (color_range.min_color + color_range.max_color) / 2
        color_range.extent = color_range.max_color - color_range.min_color
        return color_range
