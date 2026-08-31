"""Module for managing a mapping between integer labels and their corresponding color paths."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import darsia

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

    def save_csv(self, path: Path) -> None:
        """Save color paths to a CSV file.

        One row per (label, segment_index) with columns:
        label, segment_index, r, g, b, rel_r, rel_g, rel_b, base_r, base_g, base_b

        Args:
            path (Path): The CSV file path to write.
        """
        path = Path(path)
        rows = []
        for label, color_path in sorted(self.items()):
            for segment_idx, color in enumerate(color_path.colors):
                rel_color = (
                    color_path.relative_colors[segment_idx]
                    if segment_idx < len(color_path.relative_colors)
                    else np.array([0.0, 0.0, 0.0])
                )
                rows.append(
                    {
                        "label": int(label),
                        "segment_index": int(segment_idx),
                        "r": float(color[0]),
                        "g": float(color[1]),
                        "b": float(color[2]),
                        "rel_r": float(rel_color[0]),
                        "rel_g": float(rel_color[1]),
                        "rel_b": float(rel_color[2]),
                        "base_r": float(color_path.base_color[0]),
                        "base_g": float(color_path.base_color[1]),
                        "base_b": float(color_path.base_color[2]),
                    }
                )

        df = pd.DataFrame(rows)
        df = df.astype({"label": "int64", "segment_index": "int64"})
        df.to_csv(path, index=False)
        logger.info("Saved color paths to %s", path)

    @classmethod
    def load_csv(cls, path: Path) -> "LabelColorPathMap":
        """Load color paths from a CSV file.

        Expects columns: label, segment_index, r, g, b, rel_r, rel_g, rel_b,
        base_r, base_g, base_b.

        Args:
            path (Path): The CSV file path to read.

        Returns:
            LabelColorPathMap with reconstructed color paths.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("CSV file not found: %s", path)
            return cls()

        df = pd.read_csv(path)
        color_path_map = {}

        for label, group in df.groupby("label"):
            group = group.sort_values("segment_index")
            colors = []
            for _, row in group.iterrows():
                colors.append(np.array([row["r"], row["g"], row["b"]]))

            base_color = np.array(
                [
                    group.iloc[0]["base_r"],
                    group.iloc[0]["base_g"],
                    group.iloc[0]["base_b"],
                ]
            )

            color_path = darsia.ColorPath(
                colors=colors,
                base_color=base_color,
                mode="rgb",
            )
            color_path_map[int(label)] = color_path

        logger.info("Loaded color paths from CSV %s", path)
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
            distance_to_left (float, optional): Value to extend the color path to the left
                (inter).
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
