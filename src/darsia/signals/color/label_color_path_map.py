"""Module for managing a mapping between integer labels and their corresponding color paths."""

import logging
from pathlib import Path
import darsia
from typing import Literal

logger = logging.getLogger(__name__)


class LabelColorPathMap(dict[int, darsia.ColorPath]):
    """Mapping between integer labels and their corresponding color paths."""

    def __init__(self, color_paths: dict[int, darsia.ColorPath] | None = None):
        """Initialize the LabelColorPathMap.

        Args:
            color_paths (dict[int, darsia.ColorPath], optional): Initial mapping of
                labels to color paths.

        """
        super().__init__(color_paths or {})

    def __repr__(self) -> str:
        str_repr = ""
        for label, color_path in self.items():
            str_repr += f"Label {label}: {color_path}\n"
        return str_repr

    def __str__(self) -> str:
        str_str = ""
        for label, color_path in self.items():
            str_str += f"Label {label}: {str(color_path)}\n"
        return str_str

    def show_cmaps(self) -> None:
        """Show the color paths."""
        for _, color_path in self.items():
            color_path.show_cmap()

    def show_paths(self) -> None:
        """Show the color paths."""
        for _, color_path in self.items():
            color_path.show_path()

    def save(self, directory: Path) -> None:
        """Save color paths to a directory.

        Stores each color path in a separate file named `color_path_{label}.json`.

        Args:
            directory (Path): The directory to save the color path files.

        """
        directory.mkdir(parents=True, exist_ok=True)
        for label, color_path in self.items():
            color_path.save(directory / f"color_path_{label}.json")

    @classmethod
    def load(cls, directory: Path) -> "LabelColorPathMap":
        """Load color paths from a directory.

        Identifies files named `color_path_{label}.json` and loads them.

        Args:
            directory (Path): The directory containing color path files.

        """
        labels = [
            int(f.stem.split("_")[-1]) for f in directory.glob("color_path_*.json")
        ]
        color_path_map = {}
        for label in labels:
            path = directory / f"color_path_{label}.json"
            if path.exists():
                color_path_map[label] = darsia.ColorPath.load(path)
            else:
                logger.warning(f"No color path found for label {label}, skipping.")

        logger.info("Loaded color paths from %s", directory)

        return cls(color_path_map)

    @classmethod
    def refine(
        cls,
        color_path_map: "LabelColorPathMap",
        num_segments: int,
        distance_to_left: float | None = None,
        distance_to_right: float | None = None,
        mode: Literal["relative", "equidistant"] = "relative",
    ) -> "LabelColorPathMap":
        """Refine each color path in the map by increasing the number of segments.

        Args:
            color_path_map (LabelColorPathMap): The original color path map.
            num_segments (int): The number of segments to use for refinement.
            distance_to_left (float, optional): Value to extend the color path to the left (inter).
            distance_to_right (float, optional): Value to extend the color path to the right.

        Returns:
            LabelColorPathMap: The refined color path map.

        """
        refined_map = cls()
        for label, color_path in color_path_map.items():
            refined_map[label] = color_path.refine(
                num_segments=num_segments,
                distance_to_left=distance_to_left,
                distance_to_right=distance_to_right,
                mode=mode,
            )
        return refined_map
