"""Module for managing a mapping between integer labels and their corresponding color paths."""

import logging
from pathlib import Path
import darsia
from collections.abc import ItemsView

logger = logging.getLogger(__name__)


class LabelColorPathMap:
    """Mapping between integer labels and their corresponding color paths."""

    def __init__(self, color_paths: dict[int, darsia.ColorPath] | None = None):
        """Initialize the LabelColorPathMap.

        Args:
            color_paths (dict[int, darsia.ColorPath], optional): Initial mapping of labels to color paths.

        """
        self.color_paths = color_paths if color_paths is not None else {}

    def __getitem__(self, label: int) -> darsia.ColorPath:
        """Get the color path for a label.

        Args:
            label (int): The label to get the color path for.

        Returns:
            darsia.ColorPath: The color path associated with the label.

        """
        return self.color_paths[label]

    def __repr__(self) -> str:
        str_repr = ""
        for label, color_path in self.color_paths.items():
            str_repr += f"Label {label}: {color_path}\n"
        return str_repr

    def __str__(self) -> str:
        str_str = ""
        for label, color_path in self.color_paths.items():
            str_str += f"Label {label}: {str(color_path)}\n"
        return str_str

    def __setitem__(self, label: int, color_path: darsia.ColorPath) -> None:
        """Set the color path for a label.

        Args:
            label (int): The label to set the color path for.
            color_path (darsia.ColorPath): The color path to associate with the label.

        """
        self.color_paths[label] = color_path

    def items(self) -> ItemsView[int, darsia.ColorPath]:
        """Return an iterator over the color path items.

        Returns:
            dict_items: An iterator over (label, color_path) pairs.

        """
        return self.color_paths.items()

    def show(self) -> None:
        """Show the color paths."""
        for label, color_path in self.color_paths.items():
            color_path.show()

    def save(self, directory: Path) -> None:
        """Save color paths to a directory.

        Stores each color path in a separate file named `color_path_{label}.json`.

        Args:
            directory (Path): The directory to save the color path files.

        """
        directory.mkdir(parents=True, exist_ok=True)
        for label, color_path in self.color_paths.items():
            color_path.save(directory / f"color_path_{label}.json")

    def load(self, directory: Path) -> None:
        """Load color paths from a directory.

        Identifies files named `color_path_{label}.json` and loads them into the map.

        Args:
            directory (Path): The directory containing color path files.

        """
        labels = [
            int(f.stem.split("_")[-1]) for f in directory.glob("color_path_*.json")
        ]
        for label in labels:
            path = directory / f"color_path_{label}.json"
            if path.exists():
                self.color_paths[label] = darsia.ColorPath()
                self.color_paths[label].load(path)
            else:
                logger.warning(f"No color path found for label {label}, skipping.")

        logger.info("Loaded color paths from %s", directory)
