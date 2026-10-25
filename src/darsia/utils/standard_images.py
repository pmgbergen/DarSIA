"""Module providing access to standardized Image objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import cv2
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

    Parameters
    ----------
    image : darsia.Image
        Input image.
    mode : Literal["shape", "voxels"], optional
        Mode of the output image. Defaults to
        "shape".
    dtype : Optional[StandardDtype], optional
        Dtype of the output image. Defaults to None.

    Returns
    -------
    darsia.Image
        Output image.
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

    Parameters
    ----------
    image : darsia.Image
        Input image.
    mode : Literal["shape", "voxels"], optional
        Mode of the output image. Defaults to
        "shape".
    dtype : Optional[StandardDtype], optional
        Dtype of the output image. Defaults to None.

    Returns
    -------
    darsia.Image
        Output image.
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
    (logical OR) of the individual masks.

    Parameters
    ----------
    roi : _SingleRoi | list[_SingleRoi]
        A single :class:`~darsia.ROI`, :class:`~darsia.CoordinateArray`,
        or :class:`~darsia.VoxelArray`, **or** a list of such objects whose
        union is returned. A pair of coordinates (shape ``(2, 2)``, first
        row the minimum corner, second row the maximum corner) is treated
        as an axis-aligned box. More than two, ordered points are treated
        as a polygon and rasterized to their true footprint, not their
        bounding box.
    reference_image : darsia.Image
        A :class:`~darsia.Image` used to define the output
        shape, metadata and coordinate system.
    mode : Literal['shape', 'voxels']
        Passed through to :func:`zeros_like`; controls whether the
        output has the full image shape (``"shape"``) or only the voxel
        dimensions (``"voxels"``).

    Returns
    -------
        A boolean :class:`~darsia.Image` (same type/shape as.
        *reference_image*) with ``True`` inside the bounding box(es) and.
        ``False`` everywhere else.
    """
    # Delegate to the single-ROI helper for every element and union the results.
    if isinstance(roi, list):
        mask = darsia.zeros_like(reference_image, mode=mode, dtype=np.bool_)
        for roi_item in roi:
            mask.img |= roi_to_mask(roi_item, reference_image, mode=mode).img
        return mask

    mask = darsia.zeros_like(reference_image, mode=mode, dtype=np.bool_)

    # Convert from physical to voxel coordinates.
    if isinstance(roi, (darsia.ROI, darsia.CoordinateArray)):
        voxels = reference_image.coordinatesystem.voxel(roi)
    elif isinstance(roi, darsia.VoxelArray):
        voxels = roi
    else:
        raise TypeError(f"Unsupported ROI type: {type(roi)}")

    if len(voxels) > 2:
        # More than 2 ordered points define a polygon, not a 2-corner box -
        # rasterize the true footprint, not its bounding box.
        mask.img[_polygon_mask(voxels, mask.img.shape[:2])] = True
        return mask

    num_voxels = reference_image.num_voxels
    row_min = int(max(0, np.min(voxels[:, 0])))
    row_max = min(int(np.max(voxels[:, 0])), num_voxels[0])
    col_min = int(max(0, np.min(voxels[:, 1])))
    col_max = min(int(np.max(voxels[:, 1])), num_voxels[1])

    # Guard: skip if the bounding box maps to an empty region after clipping.
    # This can happen when the ROI is entirely outside the image domain (e.g.
    # row_max ended up negative before we clamped, but the raw value was
    # negative, so after clipping to [0, num_voxels] we may still get
    # row_max < row_min).
    if row_max > row_min and col_max > col_min:
        mask.img[row_min:row_max, col_min:col_max] = True
    return mask


def _polygon_mask(voxels: "darsia.VoxelArray", shape: tuple[int, int]) -> np.ndarray:
    """Rasterize an ordered polygon of (row, col) voxel points to its true footprint.

    cv2.fillPoly expects (x, y) = (col, row) ordered points, so the row/col
    columns of `voxels` are swapped before filling.
    """
    points_xy = np.asarray(voxels)[:, [1, 0]].astype(np.int32)
    raster = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(raster, [points_xy], color=1)
    return raster.astype(bool)


def full_like(
    image: darsia.Image,
    fill_value: np.ndarray | float | int,
    mode: Literal["shape", "voxels"] = "shape",
    dtype: Optional[StandardDtype] = None,
) -> darsia.Image:
    """Analog of np.full_like but for darsia.Image objects.

    Parameters
    ----------
    image : darsia.Image
        Input image.
    fill_value : np.ndarray | float | int
        Value to fill the output image with.
    mode : Literal["shape", "voxels"], optional
        Mode of the output image. Defaults to
        "shape".
    dtype : Optional[StandardDtype], optional
        Dtype of the output image. Defaults to None.

    Returns
    -------
    darsia.Image
        Output image.
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
