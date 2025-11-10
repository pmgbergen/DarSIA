import darsia
import numpy as np

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class DiscreteColorRange(darsia.ColorRange):
    def __init__(self, color_range: darsia.ColorRange, resolution: int = 32):
        self.resolution = resolution
        super().__init__(
            min_color=color_range.min_color,
            max_color=color_range.max_color,
            color_mode=color_range.color_mode,
        )

    def __repr__(self) -> str:
        return f"DiscreteColorRange(resolution={self.resolution}, {super().__repr__()})"

    @property
    def origin(self) -> tuple[int, int, int]:
        """Get the (relative) origin of the color raster in terms of the discrete grid."""
        # Check if we are inheriting from colorrange or relativecolorrange
        if self.color_mode == darsia.ColorMode.RELATIVE:
            return self.color_to_index(np.zeros(3))
        else:
            raise ValueError(
                "Require base color - call color_to_index with base color."
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the shape of the discrete color raster."""
        return (self.resolution, self.resolution, self.resolution)

    def color_to_index(self, color: np.ndarray) -> np.ndarray:
        """Convert color array to the respective indices in the discrete color raster.

        Args:
            color (np.ndarray): Color array of shape (N,3).

        """
        if len(color.shape) == 1:
            shape = color.shape
            color = color.reshape((1, 3))
        assert len(color.shape) == 2 and color.shape[1] == 3, (
            f"Color array must have shape (N,3), got {color.shape}"
        )
        if NUMBA_AVAILABLE:
            print(type(color), color.dtype, color.shape)
            min_color = self.min_color.astype(np.float64)
            extent = self.color_range.extent.astype(np.float64)
            resolution = self.resolution
            return color_to_index_numba(
                color.astype(np.float64).reshape((-1, 3)),
                min_color,
                extent,
                resolution,
            ).reshape(shape)
        else:
            num_colors = color.shape[0]
            indices = np.zeros((num_colors, 3), dtype=np.int32)
            for i in range(3):
                indices[:, i] = np.clip(
                    np.round(
                        (color[:, i] - self.color_range.min_color[i])
                        * (self.resolution - 1)
                        / (self.color_range.extent[i])
                    ),
                    0,
                    self.resolution - 1,
                ).astype(np.int32)
            return indices

    def flatten_index(self, index: np.ndarray) -> np.ndarray:
        """Convert a color to the respective flat index in the discrete color raster."""
        if NUMBA_AVAILABLE:
            return flatten_index_numba(index, self.resolution)
        else:
            flat_index = (
                index[0] * self.resolution * self.resolution
                + index[1] * self.resolution
                + index[2]
            )
            return flat_index

    def flat_color_index(self, color: np.ndarray) -> np.ndarray:
        """Convert color array to the respective flat indices in the discrete color raster.

        Args:
            color (np.ndarray): Color array of shape (N,3).

        """
        indices = self.color_to_index(color)
        flat_indices = self.flatten_index(indices)
        return flat_indices


@jit(nopython=True, cache=True)
def color_to_index_numba(
    color: np.ndarray,
    min_color: np.ndarray,
    extent: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Convert color array to the respective indices in the discrete color raster.

    Args:
        color (np.ndarray): Color array of shape (N,3).

    """
    assert len(color.shape) == 2 and color.shape[1] == 3, (
        f"Color array must have shape (N,3), got {color.shape}"
    )
    num_colors = color.shape[0]
    indices = np.zeros((num_colors, 3), dtype=np.int32)
    for n in range(num_colors):
        for i in range(3):
            indices[n, i] = int(
                np.round((color[n, i] - min_color[i]) * (resolution - 1) / (extent[i]))
            )
            if indices[n, i] < 0:
                indices[n, i] = 0
            elif indices[n, i] >= resolution:
                indices[n, i] = resolution - 1
    return indices


@jit(nopython=True, cache=True)
def flatten_index_numba(index: np.ndarray, resolution: int) -> np.ndarray:
    """Convert a color to the respective flat index in the discrete color raster."""
    num_indices = index.shape[0]
    flat_indices = np.zeros(num_indices, dtype=np.int32)
    for n in range(num_indices):
        flat_indices[n] = (
            index[n, 0] * resolution * resolution
            + index[n, 1] * resolution
            + index[n, 2]
        )
    return flat_indices
