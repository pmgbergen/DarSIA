from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


def plot_contour_on_image(
    img: darsia.Image | np.ndarray,
    mask: list[darsia.Image],
    color: Optional[list[tuple]] = None,
    alpha: Optional[list[float]] = None,
    thickness: int = 5,
    path: Path = None,
    show_plot: bool = False,
    return_image: bool = False,
) -> Optional[darsia.Image]:
    # Start with the original image
    if isinstance(img, darsia.Image):
        if img.img.dtype == np.uint8:
            original_img = np.clip(img.img, 0, 255)
        else:
            original_img = np.clip(img.img, 0, 1)
    else:
        if img.dtype == np.uint8:
            original_img = np.clip(img, 0, 255)
        else:
            original_img = np.clip(img, 0, 1)
    original_img = skimage.img_as_ubyte(original_img)

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

    # Convert mask to float and resize mask to same size as img
    for m, c, a in zip(mask, color, alpha):
        m = m.img_as(np.float32)
        m = darsia.resize(
            m,
            shape=img.num_voxels if isinstance(img, darsia.Image) else img.shape[:2],
            interpolation="inter_nearest",
        )
        m = m.img_as(bool)

        # Choose an effective color through weighting with alpha
        c_eff = tuple(int(comp * a) for comp in c)

        # Overlay the original image with contours for mask
        contours, _ = cv2.findContours(
            skimage.img_as_ubyte(m.img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
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
