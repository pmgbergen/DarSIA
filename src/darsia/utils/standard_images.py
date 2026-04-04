"""Module providing access to standardized Image objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

import darsia

if TYPE_CHECKING:
    _SingleRoi = darsia.VoxelArray | darsia.CoordinateArray | darsia.ROI

StandardDtype = Literal[np.uint8, np.uint16, np.float32, np.float64, np.bool_]


def zeros_like(
    image: darsia.Image,
    mode: Literal["shape", "voxels"] = "shape",
    dtype: Optional[StandardDtype] = None,
) -> darsia.Image:
    """Analog of np.zeros_like but for darsia.Image objects.

    Args:
        image (darsia.Image): input image
        mode (Literal["shape", "voxels"], optional): mode of the output image. Defaults to
            "shape".
        dtype (Optional[StandardDtype], optional): dtype of the output image. Defaults to None.

    Returns:
        darsia.Image: output image

    """
    if dtype is None:
        dtype = image.dtype
    if mode == "shape":
        ImageType = type(image)
        return ImageType(np.zeros(image.shape, dtype=dtype), **image.metadata())
    elif mode == "voxels":
        return darsia.ScalarImage(
            np.zeros(image.num_voxels, dtype=dtype), **image.metadata()
        )


def ones_like(
    image: darsia.Image,
    mode: Literal["shape", "voxels"] = "shape",
    dtype: Optional[StandardDtype] = None,
) -> darsia.Image:
    """Analog of np.ones_like but for darsia.Image objects.

    Args:
        image (darsia.Image): input image
        mode (Literal["shape", "voxels"], optional): mode of the output image. Defaults to
            "shape".
        dtype (Optional[StandardDtype], optional): dtype of the output image. Defaults to None.

    Returns:
        darsia.Image: output image

    """
    if dtype is None:
        dtype = image.dtype
    if mode == "shape":
        ImageType = type(image)
        return ImageType(np.ones(image.shape, dtype=dtype), **image.metadata())
    elif mode == "voxels":
        return darsia.ScalarImage(
            np.ones(image.num_voxels, dtype=dtype), **image.metadata()
        )


def roi_to_mask(
    roi: _SingleRoi | list[_SingleRoi],
    reference_image: darsia.Image,
    mode: Literal["shape", "voxels"] = "shape",
) -> darsia.Image:
    """Create a full-domain boolean mask image from one or more bounding-box ROIs.

    The returned image has the same spatial extent and coordinate system as
    *reference_image*.  Pixels whose centres fall inside the axis-aligned
    bounding box defined by ``roi`` are set to ``True``; all other
    pixels are ``False``.

    When *roi* is a **list**, the returned mask is the element-wise **union**
    (logical OR) of the individual bounding-box masks.

    Args:
        roi: A single :class:`~darsia.ROI`, :class:`~darsia.CoordinateArray`,
            or :class:`~darsia.VoxelArray` defining the bounding-box corners,
            **or** a list of such objects whose union is returned.  If a pair
            of coordinates is provided, it must be in the form of a 2-D array
            with shape ``(2, 2)``, where the first row is the minimum corner
            and the second row is the maximum corner.
        reference_image: A :class:`~darsia.Image` used to define the output
            shape, metadata and coordinate system.
        mode: Passed through to :func:`zeros_like`; controls whether the
            output has the full image shape (``"shape"``) or only the voxel
            dimensions (``"voxels"``).

    Returns:
        A boolean :class:`~darsia.Image` (same type/shape as
        *reference_image*) with ``True`` inside the bounding box(es) and
        ``False`` everywhere else.

    """
    # Delegate to the single-ROI helper for every element and union the results.
    if isinstance(roi, list):
        mask = darsia.zeros_like(reference_image, mode=mode, dtype=np.bool_)
        for roi_item in roi:
            mask.img |= roi_to_mask(roi_item, reference_image, mode=mode).img
        return mask

    mask = darsia.zeros_like(reference_image, mode=mode, dtype=np.bool_)

    # Convert the two bounding-box corners from physical to voxel coordinates.
    if isinstance(roi, (darsia.ROI, darsia.CoordinateArray)):
        voxels_box = reference_image.coordinatesystem.voxel(roi)
    elif isinstance(roi, darsia.VoxelArray):
        voxels_box = roi
    else:
        raise TypeError(f"Unsupported ROI type: {type(roi)}")

    num_voxels = reference_image.num_voxels
    row_min = int(max(0, np.min(voxels_box[:, 0])))
    row_max = min(int(np.max(voxels_box[:, 0])), num_voxels[0])
    col_min = int(max(0, np.min(voxels_box[:, 1])))
    col_max = min(int(np.max(voxels_box[:, 1])), num_voxels[1])

    # Guard: skip if the bounding box maps to an empty region after clipping.
    # This can happen when the ROI is entirely outside the image domain (e.g.
    # row_max ended up negative before we clamped, but the raw value was
    # negative, so after clipping to [0, num_voxels] we may still get
    # row_max < row_min).
    if row_max > row_min and col_max > col_min:
        mask.img[row_min:row_max, col_min:col_max] = True
    return mask


def full_like(
    image: darsia.Image,
    fill_value: np.ndarray | float | int,
    mode: Literal["shape", "voxels"] = "shape",
    dtype: Optional[StandardDtype] = None,
) -> darsia.Image:
    """Analog of np.full_like but for darsia.Image objects.

    Args:
        image (darsia.Image): input image
        fill_value (np.ndarray | float | int): value to fill the output image with
        mode (Literal["shape", "voxels"], optional): mode of the output image. Defaults to
            "shape".
        dtype (Optional[StandardDtype], optional): dtype of the output image. Defaults to None.

    Returns:
        darsia.Image: output image

    """
    if dtype is None:
        dtype = image.dtype
    if mode == "shape":
        ImageType = type(image)
        return ImageType(np.full_like(image.img, fill_value, dtype), **image.metadata())
    elif mode == "voxels":
        raise NotImplementedError(
            """The 'voxels' mode is not implemented for full_like. """
            """Need to create an Image with correct dimensions based on fill_value."""
        )
