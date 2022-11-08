"""
Module containing utils for segmentation of layered media.
"""

from typing import Union
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage as ndi

import daria


def segment(
    img: Union[np.ndarray, daria.Image],
    markers_method: str = "gradient_based",
    edges_method: str = "gradient_based",
    verbosity: bool = False,
    **kwargs,
) -> Union[np.ndarray, daria.Image]:
    """
    Prededfined workflow for segmenting an image based on
    watershed segmentation. In addition, denoising is used.

    Args:
        img (np.ndarray, or daria.Image): input image in RGB color space
        markers_method (str): "gradient_based" or "supervised", deciding which algorithm
            is used for detecting markers; the former allows for less input, while the
            latter allows to explicitly address regions of interest.
        edges_method (str): "gradient_based" or "scharr", deciding which algorithm
            is used for determining edges; the former uses gradient filtering while
            the latter uses the Scharr algorithm.
        verbosity (bool): flag controlling whether relevant quantities are plotted
            which is useful in the tuning of the parameters; the default is False.
        keyword arguments (optional): tuning parameters for the watershed algorithm
            "median disk radius" (int): disk radius to be considered to smooth
                the image using rank based median, before the analysis.
            "rescaling factor" (float): factor how the image is scaled before
                the actual watershed segmentation.
            "monochromatic_color" (str): "gray" or "value", identifying the
                monochromatic color space to be used in the analysis; default is gray.

    Returns:
        np.ndarray or daria.Image: labeled regions in the same format as img.
    """

    # ! ---- Preprocessing of input image

    # Extract numpy array from image
    if isinstance(img, np.ndarray):
        basis: np.ndarray = np.copy(img)
    elif isinstance(img, daria.Image):
        basis = np.copy(img.img)
    else:
        raise ValueError(f"img of type {type(img)} not supported.")

    # Require scalar representation - the most natural general choice is either to
    # use a grayscale representation or the value component of the HSV version,
    # when.
    monochromatic = kwargs.pop("monochromatic_color", "gray")
    if monochromatic == "gray":
        basis = cv2.cvtColor(basis, cv2.COLOR_RGB2GRAY)
    elif monochromatic == "value":
        hsv = cv2.cvtColor(basis, cv2.COLOR_RGB2HSV)
        basis = hsv[:, :, 2]
    else:
        raise ValueError(f"Monochromatic color space {monochromatic} not supported.")

    if verbosity:
        plt.figure("Monochromatic input image")
        plt.imshow(basis)

    # In order to surpress any warnings from skimage, reduce to ubyte data type
    basis_ubyte = skimage.img_as_ubyte(basis)

    # Smooth the image to get rid of sand grains
    median_disk_radius = kwargs.pop("median disk radius", 20)
    denoised = skimage.filters.rank.median(
        basis_ubyte, skimage.morphology.disk(median_disk_radius)
    )

    if verbosity:
        plt.figure("Denoised input image")
        plt.imshow(denoised)

    # Resize image
    rescaling_factor = kwargs.pop("rescaling factor", 0.1)
    rescaled = skimage.img_as_ubyte(
        skimage.transform.rescale(denoised, rescaling_factor, anti_aliasing=False)
    )

    if verbosity:
        plt.figure("Rescaled input image")
        plt.imshow(rescaled)

    # ! ---- Determine markers

    if markers_method == "gradient_based":
        # Detect markers by thresholding the gradient and thereby detecting
        # continuous regions.
        labeled_markers = _detect_markers_from_gradient(rescaled, verbosity, **kwargs)

    elif markers_method == "supervised":

        # Read markers from input, provided for the large scale image
        shape = basis.shape[:2]
        labeled_markers = _detect_markers_from_input(shape, **kwargs)

        # Project onto the coarse image
        labeled_markers = cv2.resize(
            labeled_markers,
            tuple(reversed(rescaled.shape[:2])),
            interpolation=cv2.INTER_NEAREST,
        )

    else:

        raise ValueError(
            f"Method {markers_method} for detecting markers not supported."
        )

    # ! ---- Determine edges

    if verbosity:
        if isinstance(img, np.ndarray):
            labeled_markers_large = cv2.resize(
                labeled_markers,
                tuple(reversed(img.shape[:2])),
                interpolation=cv2.INTER_NEAREST,
            )
            img_copy = skimage.img_as_ubyte(img)

        elif isinstance(img, daria.Image):
            labeled_markers_large = cv2.resize(
                labeled_markers,
                tuple(reversed(img.img.shape[:2])),
                interpolation=cv2.INTER_NEAREST,
            )
            img_copy = skimage.img_as_ubyte(img.img)
        img_copy[labeled_markers_large != 0] = [255, 255, 255]
        plt.figure("Original image with markers")
        plt.imshow(img_copy)

    if edges_method == "gradient_based":
        edges = _detect_edges_from_gradient(rescaled, **kwargs)
    elif edges_method == "scharr":
        edges = _detect_edges_from_scharr(rescaled, **kwargs)
    else:
        raise ValueError(f"Method {edges_method} for detecting edges not supported.")

    if verbosity:
        edges_copy = np.copy(edges)
        edges_copy[labeled_markers != 0] = 1.0
        plt.figure("Edges with markers")
        plt.imshow(edges_copy)
        plt.show()

    # ! ---- Actual watershed algorithm

    # Process the watershed algorithm
    labels_rescaled = skimage.img_as_ubyte(
        skimage.segmentation.watershed(edges, labeled_markers)
    )

    # ! ---- Postprocessing of the labels

    labels_rescaled = _dilate_by_size(labels_rescaled, 1, False)
    labels_rescaled = _dilate_by_size(labels_rescaled, 1, True)

    # Resize to oirginal size
    labels = skimage.img_as_ubyte(
        cv2.resize(
            labels_rescaled,
            tuple(reversed(basis.shape[:2])),
            interpolation=cv2.INTER_NEAREST,
        )
    )

    if verbosity:
        plt.figure("Segmentation after watershed algorithm")
        plt.imshow(labels)

    # Segmentation needs some cleaning, as some areas are just small,
    # tiny lines, etc. Define some auxiliary methods for this.
    # Simplify the segmentation drastically by removing small entities,
    # and correct for boundary effects.
    labels = _cleanup(labels, **kwargs)

    if verbosity:
        plt.figure("Final result after clean up")
        plt.imshow(labels)
        plt.show()

    # Return data in the same format as the input data
    if isinstance(img, np.ndarray):
        return labels
    elif isinstance(img, daria.Image):
        return daria.Image(labels, img.metadata)


# ! ---- Auxiliary functions for segment


def _detect_markers_from_gradient(img, verbosity, **kwargs) -> np.ndarray:
    """
    Routine to detect markers as continous regions, based on thresholding gradients.

    Args:
        img (np.ndarray): input image, basis for gradient analysis.
        verbosity (bool): flag controlling whether relevant quantities are plotted
            which is useful in the tuning of the parameters; the default is False.
        keyword arguments: tuning parameters for the watershed algorithm
            "markers disk radius" (int): disk radius used to define continous
                regions via gradients.
            "threshold" (float): threshold value marking regions as either
                continuous or edge.
    """

    # Find continuous region, i.e., areas with low local gradient
    markers_disk_radius = kwargs.pop("markers disk radius")
    markers_basis = skimage.filters.rank.gradient(
        img, skimage.morphology.disk(markers_disk_radius)
    )

    # Apply thresholding - requires fine tuning
    threshold = kwargs.pop("threshold")
    markers = markers_basis < threshold

    # Label the marked regions
    labeled_markers = skimage.measure.label(markers)

    if verbosity:
        plt.figure("Basis for finding continuous regions")
        plt.imshow(markers_basis)
        plt.figure(
            f"Labeled regions after applying thresholding with value {threshold}"
        )
        plt.imshow(labeled_markers)
        plt.show()

    return labeled_markers


def _detect_markers_from_input(shape, **kwargs) -> np.ndarray:
    """
    Routine to transform user-defined points into markers.

    Args:
        shape (tuple): shape of the original image, for which the marker
            coordinates are defined.
        keyword arguments (optional): tuning parameters for the watershed algorithm
            patch (int): size of regions in each dimension to be marked.
            marker_points (np.ndarray): array of coordinates for top left corner of
                each marked region. Each point thereby is a representative point for
                a unique region.

    Returns:
        np.ndarray: labeled markers corresponding to provided image shape.
    """

    # Fetch user-defined coordinates of markers
    patch: int = kwargs.pop("region_size", 1)
    pts: np.ndarray = kwargs.pop("marker_points")

    # Mark squares with points providing the top left corner.
    markers = np.zeros(shape, dtype=bool)
    for pt in pts:
        dx = np.array([patch, patch])
        corners = np.array([pt, pt + dx])
        roi = daria.bounding_box(corners)
        markers[roi] = True

    # Convert markers to labels assuming each point
    labeled_markers = skimage.measure.label(markers).astype(np.uint8)

    return labeled_markers


def _detect_edges_from_gradient(img, **kwargs) -> np.ndarray:
    """
    Routine determining edges via gradient filter from scikit-image.

    Args:
        img (np.ndarray): input image, basis for determining the gradient.
        keyword arguments (optional): tuning parameters for the watershed algorithm
            "gradient disk radius" (int): disk radius to define edges via gradients.
    """
    gradient_disk_radius = kwargs.pop("gradient disk radius", 2)

    # Find edges
    edges = skimage.filters.rank.gradient(
        img, skimage.morphology.disk(gradient_disk_radius)
    )

    return edges


def _detect_edges_from_scharr(img, **kwargs) -> np.ndarray:
    """
    Routine determining edges using the Scharr algorithm from scikit-image.

    Args:
        img (np.ndarray): input image, basis for Scharr.
        keyword arguments (optional): tuning parameters for the watershed algorithm
            mask (np.ndarray): active mask to be considered in the Scharr routine.

    Returns:
            np.ndarray: edge array in terms of intensity.
    """
    # Fetch mask from file
    mask = kwargs.pop("scharr mask", np.ones(img.shape[:2], dtype=bool))

    # Resize mask if necessary
    if mask.shape[:2] != img.shape[:2]:
        mask = skimage.img_as_bool(skimage.transform.resize(mask, img.shape))

    edges = skimage.filters.scharr(img, mask=mask)

    # Take care of the boundary, which is left with values equal to zero
    edges = _boundary(edges, 1)

    return edges


def _cleanup(labels: np.ndarray, **kwargs) -> np.ndarray:
    """
    Cleanup routine, taking care of small marked regions, boundary values etc.

    Args:
        labels (np.ndarray): input labels/segentation.
        keyword arguments (optional): tuning parameters for the watershed algorithm
            "dilation size" (int): amount of pixels, used for dilation in the postprocessing
            "boundary size" (int): amount of pixels normal to the boundary, for which the
                segmentation will be assigned as extension of the nearby interior values.

    Returns:
        np.ndarray: cleaned segmentation.
    """
    # Monitor number of labels prior and after the cleanup.
    num_labels_prior = np.unique(labels).shape[0]

    dilation_size = kwargs.pop("dilation size", 0)
    boundary_size = kwargs.pop("boundary size", 0)
    labels = _reset_labels(labels)
    labels = _dilate_by_size(labels, dilation_size, False)
    labels = _reset_labels(labels)
    labels = _fill_holes(labels)
    labels = _reset_labels(labels)
    labels = _dilate_by_size(labels, dilation_size, True)
    labels = _reset_labels(labels)
    labels = _boundary(labels, boundary_size)

    # Inform the user if labels are removed - in particular, when using
    # markers_method = "supervised", this means that provided markers
    # are ignored.
    num_labels_posterior = np.unique(labels).shape[0]
    if num_labels_prior != num_labels_posterior:
        warn("Cleanup in the segmentation has removed labels.")

    return labels


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
    labels: np.ndarray, footprint: Union[np.ndarray, int], decreasing_order: bool
) -> np.ndarray:
    """
    Dilate objects by prescribed size.

    Args:
        labels (np.ndarray): labeled image
        footprint (np.ndarray or int): foot print for dilation
        descreasing_order (bool): flag controlling whether dilation
            should be performed on objects with decreasing order
            or not (increasing order then).

    Returns:
        np.ndarray: labels after dilation
    """
    if footprint != 0:
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
    if thickness > 0:
        # Top
        labels[:thickness, :] = labels[thickness : thickness + 1, :]
        # Bottom
        labels[-thickness:, :] = labels[-thickness - 1 : -thickness, :]
        # Left
        labels[:, :thickness] = labels[:, thickness : thickness + 1]
        # Right
        labels[:, -thickness:] = labels[:, -thickness - 1 : -thickness]
    return labels
