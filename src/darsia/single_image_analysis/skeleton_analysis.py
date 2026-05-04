"""Module containing analysis tools for segmented images - use skeletonization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.ndimage as ndi
import skimage

import darsia

from .contouranalysis import _corners_of_roi

logger = logging.getLogger(__name__)


class SkeletonAnalysis:
    """Skeleton analysis object."""

    def __init__(
        self,
        verbosity: bool = False,
        contour_smoother: darsia.ContourSmoother | None = None,
        reduce_to_main_contour: bool = False,
    ) -> None:
        """Constructor.

        Args:
            verbosity (bool): Verbosity flag.

        """

        self.verbosity = verbosity
        """Vebosity flag."""
        self.contour_smoother = contour_smoother
        """Optional contour smoother for the contours determined from the mask."""
        self.reduce_to_main_contour = reduce_to_main_contour
        """Whether to reduce to main contour."""

    @darsia.timing_decorator
    def load(
        self,
        img: darsia.Image,
        mask: darsia.Image,
        roi: darsia.CoordinateArray | None = None,
        fill_holes: bool = False,
    ) -> None:
        """Read labeled image and restrict to values of interest.

        Args:
            img (Image): image to analyze.
            mask (Image): labeled image.
            roi (array, optional): set of points defining a box.
            values_of_interest (int, list of int, optional): label values of interest.
            fill_holes (bool): flag controlling whether holes in labels are filles.

        """

        # Make copy of image and restrict to region of interest
        mask_roi: darsia.Image = (
            mask.copy() if roi is None else cast(darsia.Image, mask.subregion(roi))
        )

        # Extract boolean mask covering values of interest.
        mask_roi_array = mask_roi.img

        # Fill all holes
        if fill_holes:
            mask_roi_array = ndi.binary_fill_holes(mask_roi_array)

        self.coordinatesystem = mask_roi.coordinatesystem
        """Coordinate system of subimage."""

        self.img = img
        """Image."""

        self.mask = mask_roi_array
        """Mask."""

        self.roi = roi
        """Region of interest."""

    @darsia.timing_decorator
    def skeleton(self, contours: list[np.ndarray] | None) -> list[np.ndarray]:
        """Determine skeleton of loaded labeled image.

        Returns:
            list[np.ndarray]: list of skeletons, where each skeleton is given as an
                array of pixels.

        """
        if contours is None:
            # Extract contours.
            contours, _ = cv2.findContours(
                skimage.img_as_ubyte(self.mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
        if len(contours) == 0:
            self.contour = None
            return None

        # Determine the main contour as the one with the largest area.
        if self.reduce_to_main_contour and len(contours) > 1:
            contour_areas = [cv2.contourArea(contour) for contour in contours]
            main_contour_index = np.argmax(contour_areas)
            contours = [contours[main_contour_index]]

        # Smooth contours if smoother provided.
        if self.contour_smoother:
            contours = [self.contour_smoother(contour) for contour in contours]

        assert (
            len(contours) == 1
        ), "Skeletonization currently only implemented for one contour."
        self.contour = contours[0]

        # Get mask from contour by filling
        contour_mask = np.zeros_like(self.mask, dtype=np.uint8)
        cv2.fillPoly(contour_mask, [self.contour], color=1)

        # Skeletonize the contour mask
        skeleton = skimage.morphology.skeletonize(contour_mask)

        return skeleton

    @darsia.timing_decorator
    def leaves_and_junctions(
        self,
        skeleton: np.ndarray | None,
        max_group_distance: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Determine leaves and junctions of the skeleton.

        Args:
            skeleton (np.ndarray): skeleton for which to determine leaves and junctions.
            max_group_distance (float): maximum distance for grouping pixels in metric units;
                default is 0.01.

        Returns:
            array: pixels of leaves.
            array: pixels of junctions.
            array: pixels of junctions that are part of the top line

        """
        # Convert Euclidean distances to pixel distance
        max_group_pixel_distance = self.coordinatesystem.num_voxels(
            max_group_distance, "x"
        )

        # Special case of no contour
        if skeleton is None or len(skeleton) == 0:
            return (
                np.zeros((0, 1, 2), dtype=int),
                np.zeros((0, 1, 2), dtype=int),
                np.zeros((0, 1, 2), dtype=int),
            )

        # Use networkx to convert the skeleton into a graph.
        skeleton_graph = nx.Graph()

        # Add nodes.
        skeleton_pixels = np.argwhere(skeleton)
        for pixel in skeleton_pixels:
            skeleton_graph.add_node(tuple(pixel))

        # Add edges.
        for pixel in skeleton_graph.nodes:
            x, y = pixel
            potential_neighbors = [
                (x + dx, y + dy)
                for dx, dy in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]
            ]
            neighbors = [n for n in potential_neighbors if n in skeleton_graph.nodes]
            for neighbor in neighbors:
                skeleton_graph.add_edge(tuple(pixel), neighbor)

        # Make graph unidirected by removing self-loops and parallel edges.
        skeleton_graph = skeleton_graph.to_undirected()

        # Identify all pixels that have min [:,0] value, i.e., the pixels that are closest to
        # the top of the image. These will be considered as part of the "top line" and not
        # as leaves or junctions, even if they have degree 1 or >2, respectively.
        top_line = []
        for col in np.unique(skeleton_pixels[:, 1]):
            rows = skeleton_pixels[skeleton_pixels[:, 1] == col][:, 0]
            if len(rows) > 0:
                min_row = np.min(rows)
                top_line.append((min_row, col))

        # Continue with each contour separately.
        leaves_pixels = np.zeros((0, 2), dtype=int)
        junctions_pixels = np.zeros((0, 2), dtype=int)
        base_pixels = np.zeros((0, 2), dtype=int)

        # Determine leaves and junctions based on node degree.
        for node in skeleton_graph.nodes:
            degree = skeleton_graph.degree(node)
            if degree == 1 and node[0] >= np.min(top_line, axis=0)[0]:
                # Only try to exclude the top line, but not boundary fingers.
                leaves_pixels = np.vstack((leaves_pixels, node))
            elif degree > 2:
                if node not in top_line:
                    junctions_pixels = np.vstack((junctions_pixels, node))
                else:
                    base_pixels = np.vstack((base_pixels, node))

        # Clean up - uniquify pixels if they are in a group. Touching junction_pixels
        # should be reduced to a single junction pixel, and the same for leaves_pixels.
        # For this one needs to compute the distance matrix and group pixels that are
        # close to each other, then replace each group by the mean pixel of the group.
        def uniquify_pixels(pixels: np.ndarray) -> np.ndarray:
            if len(pixels) == 0:
                return pixels

            # Strategy: Keep track of all unvisited pixels. Pick an unvisited pixel,
            # compute the distance to all other pixels, and group those that are close.
            # Then mark all pixels in the group as visited and repeat until all are visited.
            unvisited = pixels.copy()
            close_groups = []
            while True:
                if len(unvisited) == 0:
                    break
                pixel = unvisited[0]
                manhatten_distance = np.linalg.norm(unvisited - pixel, ord=1, axis=1)
                close_pixel_indices = np.where(
                    manhatten_distance < max_group_pixel_distance
                )[0]
                close_pixels = unvisited[close_pixel_indices]
                close_groups.append(close_pixels)
                unvisited = np.delete(unvisited, close_pixel_indices, axis=0)

            # Replace each group by one representative.
            unique_pixels = []
            for group in close_groups:
                # Keep the pixel with lowest row index as representative of the group,
                # to be consistent with the top line definition.
                group_representative = group[0]
                for pixel in group:
                    # Check if pixel is in top line and if so, keep it as representative.
                    # Or pick the first in line.
                    if tuple(pixel) in top_line:
                        group_representative = pixel
                        break
                unique_pixels.append(tuple(group_representative))
            assert len(unique_pixels) == len(
                close_groups
            ), "Each group should be represented by exactly one pixel."
            return np.array(unique_pixels)

        # Group them altogether.
        all_pixels = np.vstack((leaves_pixels, junctions_pixels, base_pixels))
        all_pixels = uniquify_pixels(all_pixels)

        # Distribute them into the three categories again, based on the original
        # classification.
        new_base_pixels = []
        new_junctions_pixels = []
        new_leaves_pixels = []
        for pixel in all_pixels:
            if any(np.allclose(pixel, _b, atol=2) for _b in base_pixels):
                new_base_pixels.append(pixel)
            elif any(np.allclose(pixel, _j, atol=2) for _j in junctions_pixels):
                new_junctions_pixels.append(pixel)
            elif any(np.allclose(pixel, _l, atol=2) for _l in leaves_pixels):
                new_leaves_pixels.append(pixel)
            else:
                raise ValueError(
                    f"Pixel {pixel} is not classified as leaf, junction, or base pixel."
                )

        base_pixels = np.array(new_base_pixels)
        junctions_pixels = np.array(new_junctions_pixels)
        leaves_pixels = np.array(new_leaves_pixels)

        # Make sure we have not lost any pixels.
        assert len(base_pixels) + len(junctions_pixels) + len(leaves_pixels) == len(
            all_pixels
        ), "Lost pixels during uniquification."

        # Sort and reshape - allow for possibility that arrays are empty,
        # in which case argsort will fail. In that case, just reshape without sorting.
        if len(leaves_pixels) == 0:
            reshaped_leaves_pixels = np.reshape(leaves_pixels, (-1, 1, 2))
        else:
            arg_sorted_leaves_pixels = np.argsort(leaves_pixels[:, 0], axis=0)
            sorted_leaves_pixels = leaves_pixels[arg_sorted_leaves_pixels]
            reshaped_leaves_pixels = np.reshape(sorted_leaves_pixels, (-1, 1, 2))

        if len(junctions_pixels) == 0:
            reshaped_junctions_pixels = np.reshape(junctions_pixels, (-1, 1, 2))
        else:
            arg_sorted_junctions_pixels = np.argsort(junctions_pixels[:, 0], axis=0)
            sorted_junctions_pixels = junctions_pixels[arg_sorted_junctions_pixels]
            reshaped_junctions_pixels = np.reshape(sorted_junctions_pixels, (-1, 1, 2))

        if len(base_pixels) == 0:
            reshaped_base_pixels = np.reshape(base_pixels, (-1, 1, 2))
        else:
            arg_sorted_base_pixels = np.argsort(base_pixels[:, 0], axis=0)
            sorted_base_pixels = base_pixels[arg_sorted_base_pixels]
            reshaped_base_pixels = np.reshape(sorted_base_pixels, (-1, 1, 2))

        return reshaped_leaves_pixels, reshaped_junctions_pixels, reshaped_base_pixels

    @darsia.timing_decorator
    def plot_skeleton(
        self,
        img: darsia.Image,
        skeleton: np.ndarray,
        leaves: np.ndarray | None = None,
        junctions: np.ndarray | None = None,
        base_junctions: np.ndarray | None = None,
        roi: darsia.CoordinateArray | None = None,
        path: Path | None = None,
        show: bool = True,
        dpi: int = 1000,
        **kwargs,
    ) -> None:
        """Plot skeleton with leaves and junctions on top of the provided image.

        Args:
            img (darsia.Image): image to plot on.
            skeleton (np.ndarray): pixels of the skeleton.
            roi (darsia.CoordinateArray | None): region of interest. If provided, skeleton is
                translated to the top left corner of the ROI; default is None.
            path (Path, optional): path to save the plot; if None, no saving is performed.
            show (bool): flag controlling whether the plot is shown; default is True.
            dpi (int): dots per inch for the saved plot; default is 1000.
            **kwargs: additional keyword arguments for plotting.
                - color (str): color for the skeleton; default is "r".
                - size (int): size for the skeleton; default is 20.

        """
        # Extract the top left pixel of the roi. NOTE: Need to swap for matplotlib,
        # which uses (x, y) convention for pixels, while the image uses (row, column)
        # convention.
        top_left_roi_pixel, bottom_right_roi_pixel = _corners_of_roi(img, roi)
        top_left_roi_index = (top_left_roi_pixel[1], top_left_roi_pixel[0])

        # Start with original image in the background.
        img = img.img

        # Plot the image and the skeleton on top.
        plt.figure("Image with skeleton")
        plt.imshow(img, zorder=0)

        try:
            # Display skeleton on top of original mask and contour mask for debugging.
            skeleton_coordinates = np.argwhere(skeleton)
            if len(skeleton_coordinates) > 0:
                skeleton_coordinates += top_left_roi_index  # Need row/col format

            # Overlay the skeleton with skeleton color just at the skeleton pixels
            skeleton_color = kwargs.get("skeleton_color", "w")
            skeleton_linewidth = kwargs.get("skeleton_linewidth", 3)
            plt.scatter(
                skeleton_coordinates[:, 1],
                skeleton_coordinates[:, 0],
                c=skeleton_color,
                s=skeleton_linewidth,
            )

            # Overlay with the contour
            plt.plot(
                self.contour[:, 0, 0] + top_left_roi_pixel[0],
                self.contour[:, 0, 1] + top_left_roi_pixel[1],
                c=kwargs.get("contour_color", "w"),
                linewidth=kwargs.get("contour_linewidth", 1),
                alpha=kwargs.get("contour_alpha", 0.5),
                zorder=2,
            )

            plt.scatter(
                # Translate pixels to the top left corner of the ROI
                leaves[:, 0, 1] + top_left_roi_pixel[0],
                leaves[:, 0, 0] + top_left_roi_pixel[1],
                c=kwargs.get("leaf_color", "g"),
                s=kwargs.get("leaf_size", 20),
                edgecolors=kwargs.get("leaf_edge_color", "k"),
                zorder=10,
            )

            plt.scatter(
                # Translate pixels to the top left corner of the ROI
                junctions[:, 0, 1] + top_left_roi_pixel[0],
                junctions[:, 0, 0] + top_left_roi_pixel[1],
                c=kwargs.get("junction_color", "b"),
                s=kwargs.get("junction_size", 20),
                edgecolors=kwargs.get("leaf_edge_color", "k"),
                zorder=10,
            )

            plt.scatter(
                # Translate pixels to the top left corner of the ROI
                base_junctions[:, 0, 1] + top_left_roi_pixel[0],
                base_junctions[:, 0, 0] + top_left_roi_pixel[1],
                c=kwargs.get("base_junction_color", "m"),
                s=kwargs.get("base_junction_size", 20),
                edgecolors=kwargs.get("leaf_edge_color", "k"),
                zorder=10,
            )
        except Exception as e:
            logger.error(f"Failed to plot skeleton: {e}")

        if kwargs.get("plot_boundary", False):
            plt.gca().add_patch(
                plt.Rectangle(
                    (top_left_roi_pixel[0], top_left_roi_pixel[1]),
                    bottom_right_roi_pixel[0] - top_left_roi_pixel[0],
                    bottom_right_roi_pixel[1] - top_left_roi_pixel[1],
                    linewidth=kwargs.get("boundary_linewidth", 2),
                    edgecolor=kwargs.get("boundary_color", "y"),
                    facecolor="none",
                )
            )
        if kwargs.get("highlight_roi", False):
            # Add dark overlay to the area outside the ROI
            plt.gca().add_patch(
                plt.Rectangle(
                    (0, 0), img.img.shape[1], img.img.shape[0], color="black", alpha=0.5
                )
            )
            plt.gca().add_patch(
                plt.Rectangle(
                    (top_left_roi_pixel[0], top_left_roi_pixel[1]),
                    bottom_right_roi_pixel[0] - top_left_roi_pixel[0],
                    bottom_right_roi_pixel[1] - top_left_roi_pixel[1],
                    color="white",
                    alpha=0.5,
                )
            )
        if path is not None:
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)

        if show:
            plt.show()
        else:
            plt.close()
