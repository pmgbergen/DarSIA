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
        min_color: np.ndarray,
        max_color: np.ndarray,
        color_mode: darsia.ColorMode = darsia.ColorMode.ABSOLUTE,
    ) -> None:
        """Initialize ColorRange from min and max color values.

        Args:
            min_color (np.ndarray): Minimum color values (RGB).
            max_color (np.ndarray): Maximum color values (RGB).

        """
        self.min_color = np.asarray(min_color).flatten()
        """Minimum color observed."""
        self.max_color = np.asarray(max_color).flatten()
        """Maximum color observed."""
        self.color_mode = color_mode
        """Whether the color range is relative to a baseline."""

        # Ensure we have 3-channel color data
        if len(self.min_color) != 3 or len(self.max_color) != 3:
            raise ValueError("Color arrays must have exactly 3 channels (RGB)")
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

    @classmethod
    def from_images(
        cls,
        images: list[darsia.Image],
        baseline: darsia.Image | None = None,
        mask: darsia.Image | None = None,
        **kwargs,
    ) -> "ColorRange":
        """Create a ColorRange from a list of images.

        Args:
            images (list[darsia.Image]): List of images to analyze.
            baseline (darsia.Image, optional): Baseline image for relative color range.
            mask (darsia.Image, optional): Mask image to restrict analysis area.

        """
        # Initialize min and max color arrays
        min_color = np.array([[np.inf, np.inf, np.inf]])
        max_color = np.array([[-np.inf, -np.inf, -np.inf]])

        for image in images:
            if baseline is None:
                array = image.img
                color_mode = darsia.ColorMode.ABSOLUTE
            else:
                array = image.img - baseline.img
                color_mode = darsia.ColorMode.RELATIVE
            if mask is not None:
                # Deactivate areas outside mask
                _min_color = np.min(array[mask.img].reshape((-1, 3)), axis=0)
                _max_color = np.max(array[mask.img].reshape((-1, 3)), axis=0)
            else:
                _min_color = np.min(array.reshape((-1, 3)), axis=0)
                _max_color = np.max(array.reshape((-1, 3)), axis=0)
            min_color = np.min(np.vstack((min_color, _min_color)), axis=0)
            max_color = np.max(np.vstack((max_color, _max_color)), axis=0)

        return cls(
            min_color=min_color,
            max_color=max_color,
            color_mode=color_mode,
        )

    def __eq__(self, other: object) -> bool:
        """Check equality between two ColorRange objects."""
        if not isinstance(other, ColorRange):
            return NotImplemented
        return (
            np.allclose(self.min_color, other.min_color)
            and np.allclose(self.max_color, other.max_color)
            and self.color_mode == other.color_mode
        )

    def __str__(self) -> str:
        return (
            f"""ColorRange(min_color={self.min_color}, max_color={self.max_color}, """
            f"""color_mode={self.color_mode})"""
        )

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict[str, object]:
        """Convert color range to a dictionary.

        Returns:
            dict: Dictionary representation of the color range.

        """
        return {
            "min_color": self.min_color.tolist(),
            "max_color": self.max_color.tolist(),
            "color_mode": self.color_mode.name,
        }

    def save(self, path: Path) -> None:
        """Save color range to a json file.

        Args:
            path (Path): Path to the json file.

        """
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=4,
            )
        logger.info("Saved color range to %s", path.with_suffix(".json"))

    @classmethod
    def load_from_dict(cls, data: dict[str, object]) -> "ColorRange":
        """Load color range from a dictionary.

        Args:
            data (dict): Dictionary representation of the color range.

        Returns:
            ColorRange: Loaded color range.

        """
        min_color = np.array(data["min_color"])
        max_color = np.array(data["max_color"])
        color_mode = darsia.ColorMode[data["color_mode"]]
        return cls(min_color=min_color, max_color=max_color, color_mode=color_mode)

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
        return ColorRange.load_from_dict(data)
