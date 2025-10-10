"""Module containing utils for segmentation of layered media."""

from __future__ import annotations

from typing import Optional, Union
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage as ndi

import darsia


def segment(
    img: Union[np.ndarray, darsia.Image],
    markers_method: str = "gradient_based",
    edges_method: str = "gradient_based",
    mask: Optional[np.ndarray] = None,
    verbosity: bool = False,
    **kwargs,
) -> Union[np.ndarray, darsia.Image]:
    """Prededfined workflow for segmenting an image based on watershed segmentation.

    In addition, denoising is used.

    Args:
        img (np.ndarray, or darsia.Image): input image in RGB color space
        markers_method (str): "gradient_based" or "supervised", deciding which algorithm
            is used for detecting markers; the former allows for less input, while the
            latter allows to explicitly address regions of interest.
        edges_method (str): "gradient_based" or "scharr", deciding which algorithm
            is used for determining edges; the former uses gradient filtering while
            the latter uses the Scharr algorithm.
        mask (np.ndarray, optional): binary array, only where true, segmentation is
            performed
        verbosity (bool): flag controlling whether relevant quantities are plotted
            which is useful in the tuning of the parameters; the default is False.
        keyword arguments (optional): tuning parameters for the watershed algorithm
            "method" (str): 'median' or 'tvd', while the latter uses a anisotropic
                TVD with fixed settings.
            "median disk radius" (int): disk radius to be considered to smooth
                the image using rank based median, before the analysis.
            "rescaling factor" (float): factor how the image is scaled before
                the actual watershed segmentation.
            "monochromatic_color" (str): "gray", "red", "green", "blue", or "value",
                identifying the monochromatic color space to be used in the analysis;
                default is gray.
            "boundaries" (list of str): containing elements among "top", "bottom",
                "right", "left". These will be omitted in the cleaning routine.

    Returns:
        np.ndarray or darsia.Image: labeled regions in the same format as img.
    """

    # ! ---- Preprocessing of input image

    # Extract numpy array from image
    if isinstance(img, np.ndarray):
        basis: np.ndarray = np.copy(img)
    elif isinstance(img, darsia.Image):
        basis = np.copy(img.img)
    else:
        raise ValueError(f"img of type {type(img)} not supported.")

    # basis = skimage.exposure.adjust_gamma(basis, 1.3)
    basis = skimage.exposure.adjust_log(basis, 1)
    basis = skimage.exposure.equalize_adapthist(basis)

    # Require scalar representation - the most natural general choice is either to
    # use a grayscale representation or the value component of the HSV version,
    # when.
    if len(basis.shape) == 2:
        monochromatic_basis = basis
    else:
        monochromatic = kwargs.get("monochromatic_color", "gray")
        if monochromatic == "gray":
            monochromatic_basis = cv2.cvtColor(
                skimage.img_as_ubyte(basis),  # type: ignore[attr-defined]
                cv2.COLOR_RGB2GRAY,
            )
        elif monochromatic == "red":
            monochromatic_basis = basis[:, :, 0]
        elif monochromatic == "green":
            monochromatic_basis = basis[:, :, 1]
        elif monochromatic == "blue":
            monochromatic_basis = basis[:, :, 2]
        elif monochromatic == "value":
            hsv = cv2.cvtColor(basis, cv2.COLOR_RGB2HSV)
            monochromatic_basis = hsv[:, :, 2]
        else:
            raise ValueError(
                f"Monochromatic color space {monochromatic} not supported."
            )

    if verbosity:
        plt.figure("Monochromatic input image")
        plt.imshow(monochromatic_basis)

    # In order to surpress any warnings from skimage, reduce to ubyte data type
    basis_ubyte = skimage.img_as_ubyte(monochromatic_basis)  # type: ignore[attr-defined]

    # Smooth the image to get rid of sand grains
    smoothing_method = kwargs.get("method", "median")
    assert smoothing_method in ["median", "tvd"]

    if smoothing_method == "median":
        median_disk_radius = kwargs.get("median disk radius", 20)
        denoised = skimage.filters.rank.median(
            basis_ubyte, skimage.morphology.disk(median_disk_radius)
        )
    elif smoothing_method == "tvd":
        denoised = skimage.restoration.denoise_tv_bregman(
            basis_ubyte, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=False
        )

    if verbosity:
        plt.figure("Denoised input image")
        plt.imshow(denoised)

    # Resize image
    rescaling_factor = kwargs.get("rescaling factor", 1.0)
    rescaled = skimage.img_as_ubyte(  # type: ignore[attr-defined]
        cv2.resize(
            denoised,
            None,
            fx=rescaling_factor,
            fy=rescaling_factor,
            interpolation=cv2.INTER_NEAREST,
        )
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
            img_copy = skimage.img_as_ubyte(img)  # type: ignore[attr-defined]

        elif isinstance(img, darsia.Image):
            labeled_markers_large = cv2.resize(
                labeled_markers,
                tuple(reversed(img.img.shape[:2])),
                interpolation=cv2.INTER_NEAREST,
            )
            img_copy = skimage.img_as_ubyte(img.img)  # type: ignore[attr-defined]
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
    if mask is None:
        mask = np.ones(edges.shape[:2], dtype=bool)
    labels_rescaled = skimage.img_as_ubyte(  # type: ignore[attr-defined]
        skimage.segmentation.watershed(edges, labeled_markers, mask=mask)
    )

    # ! ---- Postprocessing of the labels

    labels_rescaled = _dilate_by_size(labels_rescaled, 1, False)
    labels_rescaled = _dilate_by_size(labels_rescaled, 1, True)

    # Resize to original size
    labels = skimage.img_as_ubyte(  # type: ignore[attr-defined]
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
    if kwargs.get("cleanup", True):
        labels = _cleanup(labels, **kwargs)

    if verbosity:
        plt.figure("Final result after clean up")
        plt.imshow(labels)
        plt.figure("Final result after clean up with original image")
        plt.imshow(labels)
        plt.imshow(basis, alpha=0.5)
        plt.show()

    # Return data in the same format as the input data
    if isinstance(img, np.ndarray):
        return labels
    elif isinstance(img, darsia.Image):
        meta = img.metadata()
        return darsia.Image(labels, **meta)


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
    markers_disk_radius = kwargs.get("markers disk radius")
    markers_basis = skimage.filters.rank.gradient(
        img, skimage.morphology.disk(markers_disk_radius)
    )

    # Apply thresholding - requires fine tuning
    threshold = kwargs.get("threshold")
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
    patch: int = kwargs.get("region_size", 1)
    pts = kwargs.get("marker_points")
    assert pts is not None

    # Mark squares with points providing the top left corner.
    markers = np.zeros(shape, dtype=bool)
    for pt in pts:
        dx = np.array([patch, patch])
        corners = np.array([pt, pt + dx])
        roi = darsia.bounding_box(corners)
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
    gradient_disk_radius = kwargs.get("gradient disk radius", 2)

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
    mask = kwargs.get("scharr mask", np.ones(img.shape[:2], dtype=bool))

    # Resize mask if necessary
    if mask.shape[:2] != img.shape[:2]:
        mask = skimage.img_as_bool(  # type: ignore[attr-defined]
            skimage.transform.resize(mask, img.shape)
        )

    edges = skimage.filters.scharr(img, mask=mask)

    # Take care of the boundary, which is left with values equal to zero
    entire_boundary = ["top", "left", "bottom", "right"]
    edges = _boundary(edges, 1, entire_boundary)

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

    dilation_size = kwargs.get("dilation size", 0)
    boundary_size = kwargs.get("boundary size", 0)
    labels = _reset_labels(labels)
    labels = _dilate_by_size(labels, dilation_size, False)
    labels = _reset_labels(labels)
    labels = _fill_holes(labels)
    labels = _reset_labels(labels)
    labels = _dilate_by_size(labels, dilation_size, True)
    labels = _reset_labels(labels)
    boundary: list[str] = kwargs.get("boundary", ["top", "left", "bottom", "right"])
    labels = _boundary(labels, boundary_size, boundary)

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


def _boundary(labels: np.ndarray, thickness: int, boundary: list[str]) -> np.ndarray:
    """
    Constant extenion in normal direction at the boundary of labeled image.

    Args:
        labels (np.ndarray): labeled image
        thickness (int): thickness of boundary which should be overwritten

    Returns:
        np.ndarray: updated labeled image
    """
    if thickness > 0:
        if "top" in boundary:
            # Top
            labels[:thickness, :] = labels[thickness : thickness + 1, :]
        if "bottom" in boundary:
            # Bottom
            labels[-thickness:, :] = labels[-thickness - 1 : -thickness, :]
        if "left" in boundary:
            # Left
            labels[:, :thickness] = labels[:, thickness : thickness + 1]
        if "right" in boundary:
            # Right
            labels[:, -thickness:] = labels[:, -thickness - 1 : -thickness]
    return labels


def label_image(
    img: Union[np.ndarray, darsia.Image],
    map: dict | None = None,
    tol: float = 0.01,
    ensure_connectivity: bool = True,
    expand_labels: bool = True,
    significance: float | None = None,
    max_iter: int = 10,
) -> Union[np.ndarray, darsia.Image]:
    """
    Segment an image based on colors.

    Args:
        img (np.ndarray or darsia.Image): input image in RGB color space
        map (dict, optional): dictionary mapping color names to labels and RGB values;
            default is None, in which case a predefined map is used:
            map = {
                "white": (0, (1, 1, 1)),
                "black": (1, (0, 0, 0)),
                "red": (2, (1, 0, 0)),
                "green": (3, (0, 1, 0)),
                "blue": (4, (0, 0, 1)),
                "cyan": (5, (0, 1, 1)),
                "magenta": (6, (1, 0, 1)),
                "yellow": (7, (1, 1, 0)),
            }
            the keys are the color names (currently not used), the values are tuples
            containing the label (int) and the RGB value (tuple of floats in [0, 1]).
        tol (float): tolerance for color matching in float format; default is 0.1.
        max_iter (int): maximum number of iterations for filling unlabeled pixels;
            default is 10.

    The labeling is done by comparing the RGB values of the input image
    with the RGB values in the map. If the absolute difference is less than
    the tolerance, the corresponding label is assigned to the pixel. After the
    labeling is complete, the connected components are identified and labeled
    using skimage.measure.label.

    Returns:
        np.ndarray or darsia.Image: labeled regions in the same format as img.

    """
    # Work for now only with numpy arrays
    if isinstance(img, darsia.Image):
        return_darsia_image = True
        meta = img.metadata()
        img = img.img
    else:
        return_darsia_image = False

    # Make sure the image is in RGB format
    assert img.ndim == 3 and img.shape[2] == 3, (
        f"Image must be in RGB format, but has shape {img.shape}."
    )

    # Initialize the labeled image with zeros. These are the unlabeled pixels.
    labeled_image = np.zeros(img.shape[:2], dtype=np.uint8)

    # Make sure the image is in float format
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = skimage.img_as_float(img)  # type: ignore[attr-defined]

    # Pre-define colors for segmentation
    if map is None:
        map = {
            "white": (0, (1, 1, 1)),
            "black": (1, (0, 0, 0)),
            "red": (2, (1, 0, 0)),
            "green": (3, (0, 1, 0)),
            "blue": (4, (0, 0, 1)),
            "cyan": (5, (0, 1, 1)),
            "magenta": (6, (1, 0, 1)),
            "yellow": (7, (1, 1, 0)),
        }

    # Ensure that the map does not contain zero labels.
    if any(label == 0 for label, _ in map.values()):
        warn(
            "Map contains zero labels - zeros are also used to mark unlabeled regions."
        )

    for color_name, (label, color_value) in map.items():
        # Create a mask for the current color
        mask = np.all(np.abs(img - color_value) < tol, axis=-1)

        # Label the regions
        labeled_image[mask] = label

    # Mark unlabeled pixels
    unlabeled_mask = np.isclose(labeled_image, 0)

    # Relabel the image based on connectivity. Retain zero values for
    # unlabeled pixels.
    if ensure_connectivity:
        labeled_image = skimage.measure.label(labeled_image, background=0)

    # Ensure unlabeled pixels remain unlabeled
    labeled_image[unlabeled_mask] = 0

    # If significance is provided, remove labels with less than the
    # specified relative number of pixels.
    unsignificant_labels = []
    if significance is not None:
        label_counts = [
            (label, np.count_nonzero(labeled_image == label))
            for label in np.unique(labeled_image)
        ]
        for label, count in label_counts:
            if count < significance * labeled_image.size:
                labeled_image[labeled_image == label] = 0
                unsignificant_labels.append(label)

    # Relabel the image to ensure that labels are consecutive integers
    unique_labels = np.unique(labeled_image)
    current_label = 1
    labels = np.zeros_like(labeled_image, dtype=np.uint8)

    for label in unique_labels:
        if label != 0:  # Skip background
            labels[labeled_image == label] = current_label
            current_label += 1

    # Expand labels and fill unlabeled pixels in labels.
    if expand_labels:
        expanded_labels = skimage.segmentation.expand_labels(labels)
        unlabeled_mask = np.isclose(labels, 0)
        labels[unlabeled_mask] = expanded_labels[unlabeled_mask]

    num_unlabeled = np.count_nonzero(np.isclose(labels, 0))
    if num_unlabeled > 0:
        print(f"Number of unlabeled pixels: {num_unlabeled}")

    # Ensure that the labeled image is in the optimal format
    labels = labels.astype(np.min_scalar_type(labels))

    # Convert labeled image to the same format as input image
    if return_darsia_image:
        return darsia.Image(labels, **meta)
    else:
        return labels


def group_labels(labels: darsia.Image, groups: list[list[int]]) -> None:
    """
    Unite labels in the given groups.

    Args:
        labels (darsia.Image): labeled image with integer labels.
        group (list[list[int]]): list of groups, where each group is a list of labels
            to be united. The first label in each group is the label to which all
            other labels in the group will be united.
    """
    reduced_labels = labels.copy()
    for group in groups:
        for label in group[1:]:
            reduced_labels.img[reduced_labels.img == label] = group[0]

    return reduced_labels


def make_consecutive(labels: darsia.Image) -> darsia.Image:
    """
    Ensure that labels in the image are consecutive integers starting from 0.
    """
    unique_labels = np.unique(labels.img)
    consecutive_labels = darsia.zeros_like(labels)
    for i, label in enumerate(unique_labels):
        consecutive_labels.img[labels.img == label] = i
    return consecutive_labels
