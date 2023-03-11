"""Module containinig objects for dimension reduction,
in the sense of vertical averaging, but extended to any
axis in space and time."""

from typing import Optional, Union

import numpy as np

import darsia


class AxisAveraging:
    """Object for averaging along a provided axis.

    Attributes:
        axis (int): axis in matrix indexing along which averaging is performed.

    """

    def __init__(self, axis: Union[str, int], dim: Optional[int] = None) -> None:

        # Convert axis to numeric index
        if isinstance(axis, str):
            assert dim is not None and dim in [2, 3]
            axis, _ = darsia.interpret_indexing(axis, "ijk"[:dim])

        self.axis = axis
        """Axis along which averaging is performed."""

    def __call__(self, img: darsia.GeneralImage) -> darsia.GeneralImage:
        """Averaging routine.

        Args:
            img (GeneralImage): 3d image.

        Returns:
            GeneralImage: 2d image.

        """
        assert img.space_dim == 3

        # Reduce dimension
        img_arr = np.sum(img.img, axis=self.axis)
        origin = list(img.origin).copy()
        dimensions = img.dimensions.copy()
        origin.pop(self.axis)
        dimensions.pop(self.axis)

        # Fetch time
        time = img.time

        # Fetch and adapt metadata
        metadata = img.metadata()
        metadata["dim"] = 2
        metadata["indexing"] = "ij"
        metadata["origin"] = origin
        metadata["dimensions"] = dimensions

        return type(img)(img=img_arr, time=time, **metadata)
