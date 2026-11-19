from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia
from darsia.image.conversion_kernels import to_ubyte


def as_ubyte(array: np.ndarray) -> np.ndarray:
    """8-bit copy of an image array, as drawn on by `plot_contour_on_image`.

    8-bit data is copied; other data is clipped to [0, 1] and converted as
    ``skimage.img_as_ubyte`` (floats in one pass by `to_ubyte`).

    Parameters
    ----------
    array : np.ndarray
        Image array.

    Returns
    -------
    np.ndarray
        New uint8 array of the same shape.
    """
    if array.dtype == np.uint8:
        return np.clip(array, 0, 255)
    converted = to_ubyte(array)
    if converted is not None:
        return converted
    return skimage.img_as_ubyte(np.clip(array, 0, 1))


def plot_contour_on_image(
    img: darsia.Image | np.ndarray,
    mask: list[darsia.Image],
    contour_smoother: darsia.ContourSmoother | None = None,
    color: Optional[list[tuple]] = None,
    alpha: Optional[list[float]] = None,
    thickness: int = 5,
    path: Path = None,
    show_plot: bool = False,
    return_image: bool = False,
) -> Optional[darsia.Image]:
    """Overlay one or more contour masks on an image, with optional smoothing.

    Parameters
    ----------
    img : darsia.Image or numpy.ndarray
        Background image.
    mask : list of darsia.Image
        Masks whose contours are drawn.
    contour_smoother : darsia.ContourSmoother, optional
        Smoother applied to each contour before drawing.
    color : list of tuple, optional
        Per-mask RGB colours.
    alpha : list of float, optional
        Per-mask opacities.
    thickness : int, optional
        Contour line thickness in pixels. Default 5.
    path : Path, optional
        If given, save the figure there.
    show_plot : bool, optional
        Display the figure. Default False.
    return_image : bool, optional
        Return the composited image instead of None. Default False.

    Returns
    -------
    darsia.Image or None
        The composited image when ``return_image`` is True.
    """
    # Start with the original image
    original_img = as_ubyte(img.img if isinstance(img, darsia.Image) else img)

    if len(original_img.shape) == 2 or (
        len(original_img.shape) == 3 and original_img.shape[-1] == 1
    ):
        tmp_image = np.squeeze(original_img)
        original_img = np.zeros((*tmp_image.shape, 3), dtype=np.uint8)
        original_img[..., 0] = tmp_image.copy()
        original_img[..., 1] = tmp_image.copy()
        original_img[..., 2] = tmp_image.copy()

    # Add default data - color and intensity
    if color is None:
        color = len(mask) * [(255, 0, 0)]
    if alpha is None:
        alpha = len(mask) * [1]

    # Boolean masks of the image size as they are; others via float, resized
    shape = img.num_voxels if isinstance(img, darsia.Image) else img.shape[:2]
    for m, c, a in zip(mask, color, alpha):
        if not (m.img.dtype == bool and tuple(m.img.shape) == tuple(shape)):
            m = m.img_as(np.float32)
            m = darsia.resize(m, shape=shape, interpolation="inter_nearest")
            m = m.img_as(bool)

        # Choose an effective color through weighting with alpha
        c_eff = tuple(int(comp * a) for comp in c)

        # Overlay the original image with contours for mask
        contours, _ = cv2.findContours(
            skimage.img_as_ubyte(m.img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # If contour_smoother provided, smooth contours.
        if contour_smoother:
            contours = [contour_smoother(contour) for contour in contours]

        # Draw.
        cv2.drawContours(original_img, contours, -1, c_eff, thickness)

    # Plot
    if show_plot:
        plt.figure("Image with contours of CO2 segmentation")
        plt.imshow(original_img)
        plt.show()

    # Write corrected image with contours to file
    if path is not None:
        # path.mkdir(parents=True, exist_ok=True)
        bgr_original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(path),
            bgr_original_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )
        print(f"Image with contours saved as: {path}")

    if return_image:
        return darsia.full_like(img, original_img.astype(np.uint8), dtype=np.uint8)


def plot_distribution_on_image(
    img: darsia.Image,
    distribution: darsia.Image,
    mask: darsia.Image,
    path: Path = None,
    show_plot: bool = False,
    return_image: bool = False,
) -> None:
    # Start with the original image
    original_img = np.clip(np.copy(img.img), 0, 1)
    original_img = skimage.img_as_ubyte(original_img)

    distribution_img = skimage.img_as_ubyte(np.clip(distribution.img, 0, 1))
    original_img_0 = original_img[..., 0]
    original_img_1 = original_img[..., 1]
    original_img_2 = original_img[..., 2]

    assert mask.img.dtype == bool, "mask has to be boolean"

    original_img_0[mask.img] = distribution_img[mask.img]
    original_img_1[mask.img] = distribution_img[mask.img]
    original_img_2[mask.img] = distribution_img[mask.img]

    original_img[..., 0] = original_img_0
    original_img[..., 1] = original_img_1
    original_img[..., 2] = original_img_2

    # Plot
    if show_plot:
        plt.figure("Image with tracked distribution")
        plt.imshow(original_img)
        plt.show()

    # Write corrected image with contours to file
    if path is not None:
        bgr_original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(path),
            bgr_original_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )

    if return_image:
        return darsia.full_like(img, original_img)


def plot_image_statistics(
    img: darsia.Image,
    masks: list[darsia.Image],
    fractions: list[float],
    colors: list[tuple],
    path: Optional[Path] = None,
    show_plot: bool = False,
    return_image: bool = False,
) -> None:
    # Start with the original image
    original_img = np.clip(np.copy(img.img), 0, 1)
    original_img = skimage.img_as_ubyte(original_img)

    # Overlay all masks and determine their fraction
    count_mask = np.zeros_like(masks[0].img, dtype=float)
    for m in masks:
        count_mask += m.img.astype(float)

    for fraction, color in zip(fractions, colors):
        mask = count_mask > fraction * len(masks)
        original_img[mask] = color

    # Plot
    if show_plot:
        plt.figure("Image with tracked distribution")
        plt.imshow(original_img)
        plt.show()

    # Write corrected image with contours to file
    if path is not None:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(path),
            original_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )

    if return_image:
        return darsia.full_like(img, original_img)


def stretch_channels(img: darsia.Image) -> darsia.Image:
    """Stretch each color channel to the full range [0, 1] (display only).

    Each channel is mapped affinely from its minimum and maximum over the whole image to
    [0, 1]; a flat channel stays unchanged. This raises the contrast of a background for
    contour plots and evens out color casts; it is not meant for quantitative analysis.

    Parameters
    ----------
    img : darsia.Image
        Image with channels in the last axis, or a scalar image (one channel).

    Returns
    -------
    darsia.Image
        Float image with the same metadata.

    """
    stretched = np.asarray(img.img, dtype=float).copy()
    channels = stretched[..., np.newaxis] if stretched.ndim == 2 else stretched
    for c in range(channels.shape[-1]):
        channel = channels[..., c]
        low, high = np.nanmin(channel), np.nanmax(channel)
        if high > low:
            channels[..., c] = (channel - low) / (high - low)
    result = img.copy()
    result.img = stretched
    return result
