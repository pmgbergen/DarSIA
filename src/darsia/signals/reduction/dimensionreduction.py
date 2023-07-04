"""Module containinig objects for dimension reduction,
in the sense of vertical averaging, but extended to any
axis in space and time."""

from typing import Union

import numpy as np

import darsia


class AxisAveraging:
    """Object for averaging along a provided axis.

    Attributes:
        axis (int): axis in matrix indexing along which averaging is performed.

    """

    def __init__(self, axis: Union[str, int], dim: int = 3, mode: str = "sum") -> None:
        """
        Args:
            axis (int or str): numeric axis index (matrix indexing) or Cartesian axis
            dim (int): dimension of the input image

        Raises:
            NotImplementedError: if dim not 3.

        """

        # Convert axis to numeric index
        if isinstance(axis, str):
            assert axis in "xyz"[:dim]
            index, _ = darsia.interpret_indexing(axis, "ijk"[:dim])

        elif isinstance(axis, int):
            assert axis in range(dim)
            index = axis
            index_alpha = "ijk"[:dim][index]
            cartesian_index, _ = darsia.interpret_indexing(index_alpha, "xyz"[:dim])
            axis = "xyz"[cartesian_index]

        self.index: int = index
        """Matrix index along which averaging is performed."""

        self.axis: int = "xyz".find(axis)
        """Cartesian axis along which averaging is performed."""

        self.mode: str = mode
        """Mode."""

    def __call__(self, img: darsia.Image) -> darsia.Image:
        """Averaging routine.

        Args:
            img (Image): nd image.

        Returns:
            Image: (n-1)d image.

        """
        # Manage update of indexing
        original_dim = img.space_dim
        original_axes = "xyz"[:original_dim]
        original_indexing = img.indexing

        # Safety checks
        if not original_indexing == "ijk"[:original_dim]:
            raise NotImplementedError(
                "Only 3d case with standard matrix indexing supported."
            )

        new_dim = original_dim - 1
        new_axes = "xyz"[:new_dim]
        new_indexing = "ijk"[:new_dim]

        interim_indexing = original_indexing.replace(original_indexing[self.index], "")

        # Reduce the data
        img_arr = np.sum(img.img, axis=self.index)

        if self.mode == "sum":
            pass
        elif self.mode == "scaled":
            img_arr /= img.img.shape[self.index]

        # Reduce dimensions
        new_dimensions = img.dimensions.copy()
        new_dimensions.pop(self.index)

        # Fetch effective depth
        # depth = img.dimensions[self.index] # TODO attach geometry and change?

        # Find coordinate of Cartesian 'origin', i.e., [xmin, ymin, zmin]
        min_corner = img.origin.copy()
        for index, matrix_index in enumerate(original_indexing):
            axis, reverse_axis = darsia.interpret_indexing(matrix_index, original_axes)
            if reverse_axis:
                min_corner[axis] -= img.dimensions[index]

        # Reduce to the reduced space
        new_min_corner = min_corner.tolist()
        new_min_corner.pop(self.axis)

        # Determine reduced origin - init with reduced [xmin, ymin, zmin] and add
        # dimensions following the same convention used in the definition of
        # default_origin in Image.
        new_origin = np.array(new_min_corner)
        for new_index, interim_matrix_index in enumerate(interim_indexing):
            # Fetch corresponding character index
            new_matrix_index = new_indexing[new_index]

            # NOTE: The new index is assumed to correspond to new_indexing,
            # uniquely defining the new axis.
            new_cartesian_index, revert_axis = darsia.interpret_indexing(
                new_matrix_index, new_axes
            )

            if revert_axis:
                new_origin[new_cartesian_index] += new_dimensions[new_index]

        # Fetch and adapt metadata
        metadata = img.metadata()
        metadata["dim"] = new_dim
        metadata["indexing"] = new_indexing
        metadata["origin"] = new_origin
        metadata["dimensions"] = new_dimensions

        return type(img)(img=img_arr, **metadata)


def average_over_axis(
    image: darsia.Image, axis: Union[str, int], mode: str = "sum"
) -> darsia.Image:
    """Utility function, essentially wrapping AxisAveraging as a method.

    Args:
        img (Image): nd image.
        axis (int or str): numeric index (corresponding to matrix indexing) or
            Cartesian axis
        dim (int): dimension of the input image
        mode (str): mode used in the averaging

    Returns:
        Image: (n-1)d image.

    """
    dim = image.space_dim
    averaging = AxisAveraging(axis, dim, mode)
    return averaging(image)
