import darsia
import numpy as np


class DiscreteColorRange:
    def __init__(self, color_range: darsia.ColorRange, resolution: int = 32):
        self.color_range = color_range
        self.resolution = resolution

    @property
    def origin(self) -> tuple[int, int, int]:
        """Get the origin of the color raster in terms of the discrete grid."""
        return self.color_to_index(np.zeros(3))

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the shape of the discrete color raster."""
        return (self.resolution, self.resolution, self.resolution)

    def color_to_index(self, color: np.ndarray) -> np.ndarray:
        """Convert a color to the respective indices in the discrete color raster."""
        print(self.color_range)
        indices = np.round(
            (color - self.color_range.min_color)
            * (self.resolution - 1)
            / (self.color_range.extent)
        ).astype(int)
        assert np.all(indices >= 0) and np.all(indices < self.resolution), (
            f"Indices out of bounds: {indices} for color {color}"
        )
        return indices
