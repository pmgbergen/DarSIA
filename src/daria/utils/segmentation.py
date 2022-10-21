"""
Module containing utils for segmentation of layered media.
"""

import cv2
import numpy as np
import skimage
from scipy import ndimage as ndi


def segment(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    Prededfined workflow for segmenting an image based on
    watershed segmentation. In addition, denoising is used.

    Args:
        img (np.ndarray): input image in RGB color space

    Returns:
        np.ndarray: labeled regions
    """

    median_disk_radius = kwargs.pop("median disk radius", 20)
    rescaling_factor = kwargs.pop("rescaling factor", 0.1)
    markers_disk_radius = kwargs.pop("markers disk radius", 10)
    threshold = kwargs.pop("threshold", 20)
    gradient_disk_radius = kwargs.pop("gradient disk radius", 2)
    dilation_size = kwargs.pop("dilation size", 10)
    boundary_size = kwargs.pop("boundary size", 55)

    # Require scalar representation - work with grayscale image. Alternatives exist,
    # but with little difference.
    basis = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Smooth the image to get rid of sand grains
    denoised = skimage.filters.rank.median(
        basis, skimage.morphology.disk(median_disk_radius)
    )

    # Resize image
    denoised = skimage.img_as_ubyte(
        skimage.transform.rescale(denoised, rescaling_factor, anti_aliasing=False)
    )

    # Find continuous region, i.e., areas with low local gradient
    markers_basis = skimage.filters.rank.gradient(
        denoised, skimage.morphology.disk(markers_disk_radius)
    )
    # TODO add smoothing?
    markers = markers_basis < threshold

    # Label the marked regions
    markers = skimage.measure.label(markers)

    # Find edges
    gradient = skimage.filters.rank.gradient(
        denoised, skimage.morphology.disk(gradient_disk_radius)
    )

    # Process the watershed and resize to the original size
    labels_rescaled = skimage.img_as_ubyte(
        skimage.segmentation.watershed(gradient, markers)
    )
    labels = skimage.img_as_ubyte(
        skimage.transform.resize(labels_rescaled, img.shape[:2])
    )

    # Segmentation needs some cleaning, as some areas are just small,
    # tiny lines, etc. Define some auxiliary methods for this.
    # Simplify the segmentation drastically by removing small entities,
    # and correct for boundary effects.
    labels = _reset_labels(labels)
    labels = _dilate_by_size(labels, dilation_size, False)
    labels = _reset_labels(labels)
    labels = _fill_holes(labels)
    labels = _reset_labels(labels)
    labels = _dilate_by_size(labels, dilation_size, True)
    labels = _reset_labels(labels)
    labels = _boundary(labels, boundary_size)

    return labels


# ! ---- Auxiliary functions for segment


def _reset_labels(labels: np.ndarray) -> np.ndarray:
    """
    Rename labels, such that these are consecutive with step size 1,
    starting from 0.

    Args:
        labels (np.ndarray): labeled image

    Returns:
        np.ndarray: new labeled regions
    """
    pre_labels = np.unique(labels)
    for i, label in enumerate(pre_labels):
        mask = labels == label
        labels[mask] = i
    return labels


def _fill_holes(labels: np.ndarray) -> np.ndarray:
    """
    Routine for filling holes in all labeled regions.

    Args:
        labels (np.ndarray): labeled image

    Returns:
        np.ndarray: labels without holes.
    """
    pre_labels = np.unique(labels)
    for label in pre_labels:
        mask = labels == label
        mask = ndi.binary_fill_holes(mask).astype(bool)
        labels[mask] = label
    return labels


def _dilate_by_size(
    labels: np.ndarray, footprint: np.ndarray, decreasing_order: bool
) -> np.ndarray:
    """
    Dilate objects by prescribed size.

    Args:
        labels (np.ndarray): labeled image
        footprint (np.ndarray): foot print for dilation
        descreasing_order (bool): flag controlling whether dilation
            should be performed on objects with decreasing order
            or not (increasing order then).

    Returns:
        np.ndarray: labels after dilation
    """

    # Determine sizes of all marked areas
    pre_labels = np.unique(labels)
    sizes = [np.count_nonzero(labels == label) for label in pre_labels]
    # Sort from small to large
    labels_sorted_sizes = np.argsort(sizes)
    if decreasing_order:
        labels_sorted_sizes = np.flip(labels_sorted_sizes)
    # Erode for each label if still existent
    for label in labels_sorted_sizes:
        mask = labels == label
        mask = skimage.morphology.binary_dilation(
            mask, skimage.morphology.disk(footprint)
        )
        labels[mask] = label
    return labels


def _boundary(labels: np.ndarray, thickness: int) -> np.ndarray:
    """
    Constant extenion in normal direction at the boundary of labeled image.

    Args:
        labels (np.ndarray): labeled image
        thickness (int): thickness of boundary which should be overwritten

    Returns:
        np.ndarray: updated labeled image
    """
    # Top
    labels[:thickness, :] = labels[thickness : 2 * thickness, :]
    # Bottom
    labels[-thickness - 1 : -1, :] = labels[-2 * thickness : -thickness, :]
    # Left
    labels[:, :thickness] = labels[:, thickness : 2 * thickness]
    # Right
    labels[:, -thickness - 1 : -1] = labels[:, -2 * thickness : -thickness]
    return labels
