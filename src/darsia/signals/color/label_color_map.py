import numpy as np
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class LabelColorMap:
    def __init__(self, colors: dict[int, np.ndarray] | None = None) -> None:
        self.colors = colors if colors is not None else {}

    def __getitem__(self, label: int) -> np.ndarray:
        return self.colors[label]

    def mean(self) -> np.ndarray:
        return np.mean(np.array(list(self.colors.values())), axis=0)

    def labels(self) -> list[int]:
        return list(self.colors.keys())

    def __repr__(self) -> str:
        return f"LabelColorMap {self.colors}"

    def load(self, path: Path) -> None:
        """Load base colors from a csv file.

        Args:
            path (Path): The path to the csv file.

        """
        df = pd.read_csv(path)
        # Expect columns: label, r, g, b
        for _, row in df.iterrows():
            label = int(row["label"])
            color = np.array([row["r"], row["g"], row["b"]])
            self.colors[label] = color

    def save(self, path: Path) -> None:
        """Save base colors to a csv file."""
        df = pd.DataFrame.from_dict(
            self.colors, orient="index", columns=["r", "g", "b"]
        )
        df.index.name = "label"
        df.to_csv(path)
        logger.info("Saved base colors to %s", path)
