from pathlib import Path
from typing import Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

import darsia


class PWTransformation:
    """Transformation which cuts off values below 0 and ensures monotonicity."""

    def __init__(
        self,
        supports: Optional[list | np.ndarray] = None,
        values: Optional[list | np.ndarray] = None,
    ) -> None:
        self.supports = supports
        self.values = values
        if supports is not None and values is not None:
            self.update(supports, values)

    def update(
        self,
        supports: Optional[list | np.ndarray] = None,
        values: Optional[list | np.ndarray] = None,
        dofs: Optional[list | np.ndarray] = None,
    ) -> None:
        # Update supports and values
        if supports is not None:
            if dofs is not None:
                self.supports[np.array(dofs)] = supports
            else:
                self.supports = supports
        if values is not None:
            if dofs is not None:
                self.values[np.array(dofs)] = values
            else:
                self.values = values

        # Update interpolator
        if self.supports is None or self.values is None:
            warn("No supports or values provided. Interpolator not updated.")
        else:
            # Sanity checks
            assert len(self.values) == len(self.supports), (
                f"wrong size: {len(values)} vs. {len(self.supports)}"
            )
            values_diff = np.diff(values)
            assert np.all(values_diff > -1e-12), f"monotonicity broken {values_diff}"

            # Interpolator
            self.interpolator = interpolate.interp1d(
                self.supports,
                self.values,
                kind="linear",
            )

    def values_from_diff(self, values_diff: list | np.ndarray) -> list | np.ndarray:
        raise NotImplementedError("Implementation of class has changed.")
        return np.hstack(([0, 0], np.cumsum(values_diff)))

    def __call__(self, img: np.ndarray | darsia.Image) -> np.ndarray | darsia.Image:
        """Apply the transformation to the image."""

        assert hasattr(self, "interpolator"), "Interpolator not set."

        if isinstance(img, np.ndarray):
            return self.interpolator(img)
        elif isinstance(img, darsia.Image):
            result = img.copy()
            result.img = self.interpolator(img.img)
            return result
        else:
            raise ValueError

    def log(self, log: Optional[Path]) -> None:
        """Plot the transformation and store to file."""

        if log:
            plt.figure()
            x_space = np.linspace(0, 1, 1000)
            plt.plot(x_space, self.interpolator(x_space))
            plt.xlabel("Signal")
            plt.ylabel("Converted signal")
            plt.title("PWTransformation")
            plt.savefig(log)
            plt.close()

    def save(self, path: Path) -> None:
        """Save the transformation to file in csv format.

        Args:
            path (Path): Path to the file where the transformation should be saved.

        """
        df = pd.DataFrame({"supports": self.supports, "values": self.values})
        df.to_csv(path.with_suffix(".csv"), index=False)

    def load(self, path: Path) -> None:
        """Load the transformation from file in csv format.

        Args:
            path (Path): Path to the file from which the transformation should be loaded.

        """
        df = pd.read_csv(path)
        self.supports = df["supports"].to_numpy()
        self.values = df["values"].to_numpy()
        self.update(supports=self.supports, values=self.values)
