"""Shared active-region ROI visualization helpers for workflow plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.axes
import numpy as np
import skimage.measure

import darsia
from darsia.utils.standard_images import roi_to_mask


@dataclass(frozen=True)
class ActiveRegionRenderData:
    """Container for rendered active-region image data and extracted contours."""

    image: np.ndarray
    mask: np.ndarray
    contours: list[np.ndarray]


def _as_bool_mask(mask: np.ndarray | darsia.Image, shape: tuple[int, int]) -> np.ndarray:
    """Return mask as bool array and validate shape."""

    array = mask if isinstance(mask, np.ndarray) else np.asarray(mask.img)
    if array.shape[:2] != shape:
        raise ValueError(
            f"Mask shape {array.shape[:2]} does not match image shape {shape}."
        )
    return array.astype(bool)


def build_active_mask_from_rois(
    image: darsia.Image,
    rois: (
        darsia.ROI
        | darsia.CoordinateArray
        | darsia.VoxelArray
        | Sequence[darsia.ROI | darsia.CoordinateArray | darsia.VoxelArray]
        | None
    ),
) -> np.ndarray | None:
    """Build a boolean active-region mask from one ROI or a list of ROIs."""

    if rois is None:
        return None
    mask_image = roi_to_mask(rois, image, mode="voxels")
    return np.asarray(mask_image.img).astype(bool)


def render_active_region(
    image: darsia.Image,
    *,
    active_mask: np.ndarray | darsia.Image | None = None,
) -> ActiveRegionRenderData:
    """Render image highlighting active region and extract ROI contours."""

    image_data = np.asarray(image.img).copy()
    shape = image_data.shape[:2]
    if active_mask is None:
        mask = np.ones(shape, dtype=bool)
    else:
        mask = _as_bool_mask(active_mask, shape)

    if image_data.ndim == 2:
        rendered = image_data
    else:
        rendered = image_data.copy()
        gray = np.asarray(image.to_monochromatic("gray").img)
        rendered[~mask] = gray[~mask][:, None]

    contours: list[np.ndarray] = []
    if np.any(mask) and not np.all(mask):
        contours = list(skimage.measure.find_contours(mask.astype(float), level=0.5))

    return ActiveRegionRenderData(image=rendered, mask=mask, contours=contours)


def draw_active_region(
    *,
    ax: matplotlib.axes.Axes,
    image: darsia.Image,
    active_mask: np.ndarray | darsia.Image | None = None,
    title: str | None = None,
    contour_color: str = "white",
    contour_linewidth: float = 2.0,
) -> ActiveRegionRenderData:
    """Draw active-region overlay on matplotlib axes and return render data."""

    render_data = render_active_region(image, active_mask=active_mask)
    ax.imshow(render_data.image)
    for contour in render_data.contours:
        ax.plot(contour[:, 1], contour[:, 0], color=contour_color, linewidth=contour_linewidth)
    if title is not None:
        ax.set_title(title)
    ax.axis("off")
    return render_data
