"""
Class for comparing segmented images.

The object contain information about the different segmentations
as well as methods for comparing them and visualizing the result.

"""

from __future__ import annotations

from typing import Optional, Union

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import skimage
from matplotlib.cm import get_cmap

import darsia as da


class SegmentationComparison:
    """
    Class for comparing segmented images.

    Routines for comparing segmentations and creating visualizations of the comparison.

    Attributes:
        number_of_segmented_images (int): Number of segmented images that one compares
        segmentation_names (list[str]): list of names for each of the segmentations.
            Will affect legends in plots.
        components (list): list of values the different (active) components in the
            segmentations. As of now, up to two are allowed for and the default values
            are 1 and 2.
        component_names (list[str]): list of names for each of the components. Will
            be visual in legends.
        gray_colors (np.ndarray): array of base gray colors (in RGB space) that accounts
            for different overlapping segmentations of different components.
        colors (np.ndarray): color values for the different unique segmentations.
            Default is created from a colormap (matplotlib) depending on the amount of
            present segmentations.
        color_dictionary (dict): dictionary relating all of the different colors to
            different overlapping segmentation situations.

    """

    def __init__(self, number_of_segmented_images: int = 2, **kwargs) -> None:
        """
        Constructor of compare segmentations class.

        Args:
            number_of_segmented_images (int): Number of segmentations to be compared
            Optional keyword arguments (kwargs):
                segmentation_names (list): list of names for the different segmented
                    images. So far only used in legends and color dictionary.
                components (list): list of the different components that are
                    considered in the segmented images. So far only two are allowed
                    to be provided.
                component_names (list):  list of names for the different components.
                    So far only used in legends, and color dictionary.
                gray_colors (np.ndarray): array of three different scales of
                    gray (in RGB format), one for each of the different combinations of
                    components in the segmentations.
                colors (np.ndarray): Array of different colors that should
                    indicate unique components in each segmentation.
                light_scaling (float): Indicate how much lighter the second
                    component should be scaled in its unique color.
        """

        self.number_of_segmented_images = number_of_segmented_images
        self.segmentation_names: list = kwargs.pop(
            "segmentation_names",
            [f"Segmentation {i}" for i in range(self.number_of_segmented_images)],
        )

        # Define components
        self.components: list = kwargs.pop("components", [1, 2])
        self.component_names: list = kwargs.pop(
            "component_names", ["Component 0", "Component 1"]
        )

        # Define gray colors
        self.gray_colors: np.ndarray = kwargs.pop(
            "gray_colors",
            np.array([[90, 90, 90], [150, 150, 150], [200, 200, 200]], dtype=np.uint8),
        )

        # Define unique colors
        self.light_scaling: float = kwargs.pop("light_scaling", 1.1)
        # If set of colors are not provided create it with matplotlib colormap.
        if "colors" not in kwargs:
            colormap = get_cmap("Spectral")
            self.colors: np.ndarray = np.zeros(
                (self.number_of_segmented_images, 2, 3), dtype=np.uint8
            )
            for i in range(self.number_of_segmented_images):
                rgba = 255 * np.array(
                    colormap(1 / self.number_of_segmented_images * (i + 0.5))[0:3]
                )
                rgbalight = np.trunc(self.light_scaling * rgba)
                self.colors[i, 0] = rgba.astype(np.uint8)
                self.colors[i, 1] = rgbalight.astype(np.uint8)
        else:
            # Assert that there are a sufficient amount of colors
            colors_pre: np.ndarray = kwargs.pop("colors")
            assert colors_pre.shape[0] == self.number_of_segmented_images
            colors_light: np.ndarray = np.trunc(self.light_scaling * colors_pre).astype(
                np.uint8
            )
            self.colors = np.zeros(
                (self.number_of_segmented_images, 2, 3), dtype=np.uint8
            )
            for i in range(self.number_of_segmented_images):
                self.colors[i, 0] = colors_pre[i]
                self.colors[i, 1] = colors_light[i]
            # self.colors = np.hstack((colors_pre, colors_light))

        # Create dictionary with colors and associated situations. Used for legends.
        self.color_dictionary: dict = {}

        # Adding information about colors that represent unique apperances for each
        # segmented image and each component
        for i in range(self.number_of_segmented_images):
            if abs(self.light_scaling - 1) > 1e-6:
                self.color_dictionary[
                    f"Unique apperance of {self.component_names[0]}"
                    f" in {self.segmentation_names[i]}"
                ] = self.colors[i, 0]
                self.color_dictionary[
                    f"Unique apperance of {self.component_names[1]}"
                    f" in {self.segmentation_names[i]}"
                ] = self.colors[i, 1]
            else:
                self.color_dictionary[
                    f"Unique apperance of {self.segmentation_names[i]}"
                ] = self.colors[i, 0]
                self.color_dictionary[
                    f"Unique apperance of {self.segmentation_names[i]}"
                ] = self.colors[i, 1]

        # Adding information regarding gray colors and overlapping components of
        # different segmentations.
        if np.all(self.gray_colors[0] == self.gray_colors[1]) and np.all(
            self.gray_colors[1] == self.gray_colors[2]
        ):
            self.color_dictionary[f"Segmentations overlap"] = self.gray_colors[0]
            self.color_dictionary[f"Segmentations overlap"] = self.gray_colors[1]
            self.color_dictionary[f"Segmentations overlap"] = self.gray_colors[2]
        else:
            self.color_dictionary[
                f"Overlapping segmentations in {self.component_names[0]}"
            ] = self.gray_colors[0]
            self.color_dictionary[
                f"Overlapping segmentations in {self.component_names[1]}"
            ] = self.gray_colors[1]
            self.color_dictionary[
                f"Segmentations overlap with different components."
            ] = self.gray_colors[2]

    def __call__(
        self,
        *segmentations,
        plot_result: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Comparison of segmentations.

        Args:
            segmentations (asterisk argument): Allows to provide an arbitraty
                number of segmented numpy arrays or da.Images of integers to be compared
            Optional keyword arguments (kwargs):
                plot_result (bool): plots the result with matplotlib if True,
                    default is False.
                roi (Union[tuple, np.ndarray]): roi where the segmentations should be
                    compared, default is the maximal roi that fits in all segmentations.
                    Should be provided in pixel coordinates using matrix indexing, either
                    as a tuple of slices, or an array of corner points.

        """

        # Define number of segmentations
        assert self.number_of_segmented_images == len(segmentations)

        # Checks whether roi is provided and if it is as a tuple (of slices)
        # or an array of corner points
        if "roi" in kwargs:
            roi_input = kwargs["roi"]
            if isinstance(roi_input, tuple):
                roi: tuple = roi_input
            elif isinstance(roi_input, np.ndarray):
                roi = da.bounding_box(roi_input)
            elif isinstance(roi_input, list):
                roi = da.bounding_box(np.array(roi_input))
            else:
                raise Exception(
                    f"{type(roi_input)} is not a valid type for roi. Please provide it as"
                    " a tuple of slices, or an array or list of corner points"
                )
            return_image: np.ndarray = np.zeros(
                (roi[0].stop - roi[0].start, roi[1].stop - roi[1].start) + (3,),
                dtype=np.uint8,
            )

        # If roi is not provided the largest roi that fits all segmentations are chosen.
        else:
            if all([isinstance(seg, np.ndarray) for seg in segmentations]):
                rows = min([seg.shape[0] for seg in segmentations])
                cols = min([seg.shape[1] for seg in segmentations])
            elif all([isinstance(seg, da.Image) or isinstance(seg, da.GeneralImage) for seg in segmentations]):
                rows = min([seg.img.shape[0] for seg in segmentations])
                cols = min([seg.img.shape[1] for seg in segmentations])
            roi = (slice(0, rows), slice(0, cols))
            return_image = np.zeros((rows, cols) + (3,), dtype=np.uint8)

        # Determine whether segmentations are arrays of darsia images.
        # They should all be the same.
        if all([isinstance(seg, np.ndarray) for seg in segmentations]):
            segmentation_arrays: tuple[np.ndarray, ...] = segmentations
        elif all([isinstance(seg, da.Image) or isinstance(seg, da.GeneralImage) for seg in segmentations]):
            segmentation_arrays = tuple([seg.img for seg in segmentations])
        else:
            raise Exception(
                "Segmentation types are not allowed. They should"
                "all be the same, and either arrays, or darsia images."
            )

        # Enter gray everywhere there are ovelaps of different segmentations
        for k in range(self.number_of_segmented_images):
            for i in range(k + 1, self.number_of_segmented_images):
                # Overlap of components
                for c_num, c in enumerate(self.components):
                    return_image[
                        np.logical_and(
                            segmentation_arrays[k][roi] == c,
                            segmentation_arrays[i][roi] == c,
                        )
                    ] = self.gray_colors[c_num]

        # Overlap of different components. Note that it also writes wherever
        # segmentation_arrays[i][roi] is a non active component, but that case
        # should be overwritten when checking for unique apperances.
        for k in range(self.number_of_segmented_images):
            for i in range(k + 1, self.number_of_segmented_images):
                return_image[
                    np.logical_and(
                        np.isin(segmentation_arrays[k][roi], self.components),
                        segmentation_arrays[k][roi] != segmentation_arrays[i][roi],
                    )
                ] = self.gray_colors[2]

        # Determine locations (and make modifications to return image) of unique components
        for c_num, c in enumerate(self.components):
            for k in range(self.number_of_segmented_images):
                unique_apperance: np.ndarray = segmentation_arrays[k][roi] == c
                for j in filter(
                    lambda j: j != k, range(self.number_of_segmented_images)
                ):
                    unique_apperance = np.logical_and(
                        unique_apperance,
                        np.logical_not(
                            np.isin(segmentation_arrays[j][roi], self.components)
                        ),
                    )
                return_image[unique_apperance] = self.colors[k, c_num]

        if plot_result:
            self.plot(return_image, "Comparison")

        return return_image

    def compare_segmentations_binary_array(
        self, *segmentations: tuple[np.ndarray, ...], **kwargs
    ) -> np.ndarray:
        """
        Compares segmentations and returns an an array
        with pixels that containing an array of 1s and 0s depending on
        which segmentations are present there. At the current state it
        does not distinguish between the different kind of components,

        Args:
            *segmentations (tuple[np.ndarray, ...]): The segmentations to be compared.
            **kwargs: Optional keyword arguments.
                roi (Union[tuple, np.ndarray]): roi where the segmentations should be
                    compared, default is the maximal roi that fits in all segmentations.
                    Should be provided in pixel coordinates using matrix indexing, either
                    as a tuple of slices, or an array of corner points.
                components (tuple[int, ...]): The components that should be recognized in
                    the segmentations, default is [1,2].

        """
        # NOTE: At the moment both components are counted as the same.

        # Checks whether roi is provided and if it is as a tuple (of slices)
        # or an array of corner points
        if "roi" in kwargs:
            roi_input = kwargs["roi"]
            if isinstance(roi_input, tuple):
                roi: tuple = roi_input
            elif isinstance(roi_input, np.ndarray):
                roi = da.bounding_box(roi_input)
            elif isinstance(roi_input, list):
                roi = da.bounding_box(np.array(roi_input))
            else:
                raise Exception(
                    f"{type(roi_input)} is not a valid type for roi. Please provide it as"
                    " a tuple of slices, or an array or list of corner points"
                )
            return_image: np.ndarray = np.zeros(
                (roi[0].stop - roi[0].start, roi[1].stop - roi[1].start)
                + (len(segmentations),),
                dtype=np.uint8,
            )

        # If roi is not provided the largest roi that fits all segmentations are chosen.
        else:
            if all([isinstance(seg, np.ndarray) for seg in segmentations]):
                rows = min([seg.shape[0] for seg in segmentations])
                cols = min([seg.shape[1] for seg in segmentations])
            elif all([isinstance(seg, da.Image) or isinstance(seg, da.GeneralImage) for seg in segmentations]):
                rows = min([seg.img.shape[0] for seg in segmentations])
                cols = min([seg.img.shape[1] for seg in segmentations])
            roi = (slice(0, rows), slice(0, cols))
            return_image = np.zeros(
                (rows, cols) + (len(segmentations),), dtype=np.uint8
            )

        # Determine whether segmentations are arrays of darsia images.
        # They should all be the same.
        if all([isinstance(seg, np.ndarray) for seg in segmentations]):
            segmentation_arrays: tuple[np.ndarray, ...] = segmentations
        elif all([isinstance(seg, da.Image) or isinstance(seg, da.GeneralImage) for seg in segmentations]):
            segmentation_arrays = tuple([seg.img for seg in segmentations])
        else:
            raise Exception(
                "Segmentation types are not allowed. They should"
                "all be the same, and either arrays, or darsia images."
            )

        # Extract components
        components = kwargs.pop("components", [1, 2])

        # Go through segmentations and enter 1 in correct position at the
        # pixels where they are present
        for k in range(len(segmentations)):
            k_arr = np.zeros((len(segmentations)), dtype=np.uint8)
            k_arr[k] = 1
            return_image[
                np.logical_or(
                    segmentation_arrays[k][roi] == components[0],
                    segmentation_arrays[k][roi] == components[1],
                )
            ] += k_arr

        return return_image

    def get_combinations(
        self, *segmentation_numbers: tuple[int, ...], num_segmentations: int = 5
    ) -> list[list[int]]:
        """
        Returns a list of all possible combinations of segmentations.

        Args:
            num_segmentations (int, optional): Number of segmentations. Defaults to 5.
            *segmentation_numbers (tuple[int, ...]): The segmentation numbers that
                should be included in the combinations. Defaults to ().

        Returns:
            list[list[int]]: List of all possible combinations of segmentations.
        """

        # Create an empty list of all possible combinations of segmentations
        combinations: list[list[int]] = []

        # Create the base of a full combination
        base: list = np.ones(num_segmentations, dtype=np.ubyte).tolist()

        # Create a list of the segmentations that should be included in the combinations
        segs: list[int] = [
            i for i in range(num_segmentations) if i not in segmentation_numbers
        ]

        # Create a recursive function that loops through all possible combinations
        def loop_rec_bool(n, max=True, tmp=None):
            if n >= 1:
                if max:
                    for i in [0, 1]:
                        tmp = base.copy()
                        tmp[segs[n]] = i
                        loop_rec_bool(n - 1, False, tmp)
                else:
                    for i in [0, 1]:
                        tmp[segs[n]] = i
                        loop_rec_bool(n - 1, False, tmp)
            elif n == 0:
                if max:
                    for i in [0, 1]:
                        tmp = base.copy()
                        tmp[segs[n]] = i
                        combinations.append(tmp.copy())
                else:
                    for i in [0, 1]:
                        tmp[segs[n]] = i
                        combinations.append(tmp.copy())
            else:
                combinations.append(base)

        # Loop through all possible combinations
        loop_rec_bool(len(segs) - 1)
        return combinations

    def plot(
        self,
        image: np.ndarray,
        figure_name: str = "Comparison",
        legend_anchor: tuple = (0.7, 1),
    ) -> None:
        """
        Plots the provided image (should be a comparison of segmentations) with
        matplotlib.pyplot's imshow and prints a legend with colors from the image
        and dictionary

        Args:
            image (np.ndarray): image with comparison of segmentations.
            figure_name (str): Figure name.
            legend_anchor (tuple): tuple of coordinates (x,y) in Euclidean
                style that determines legend anchor.

        """
        plt.figure(figure_name)
        plt.imshow(image)
        unique_colors = self._get_unique_colors(image)
        patches = self._get_legend_patches(unique_colors)
        plt.legend(
            handles=patches, bbox_to_anchor=legend_anchor, loc=2, borderaxespad=0.0
        )
        plt.show()

    def _get_legend_patches(
        self, unique_colors: np.ndarray, custom_legend_text: Optional[list[str]] = None
    ) -> list:
        """
        Function that extracts information from the color dictionary and creates
        legend entries depending on provided colors.

        Args:
            unique_colors (np.ndarray): numpy array of color values whose
                information should be extracted from the color dictionary.
            custom_legend_text (Optional[list[str]]): in case it is desirable
                to customize legend.

        Returns:
            patches (list): patches suitable for legend in matplotlib.pyplot.
        """

        # create a patch (proxy artist) for every color
        if custom_legend_text is None:
            patches: list = [
                mpatches.Patch(
                    color=c / 255, label=self._get_key(c, self.color_dictionary)
                )
                for c in unique_colors
            ]
        else:
            assert len(custom_legend_text) == len(unique_colors)
            patches = [
                mpatches.Patch(color=c / 255, label=custom_legend_text[i])
                for i, c in enumerate(unique_colors)
            ]
        return patches

    def _get_unique_colors(
        self, image: np.ndarray, return_counts=False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Given an image it extracts the unique color values (except background color)
        and returns them as an array.

        Args:
            image (np.ndarray): image array.

        Returns:
            array of unique colors in input image.
        """
        nx, ny, _ = image.shape
        flat_im = image.reshape((nx * ny, 3))
        b = np.ascontiguousarray(flat_im).view(
            np.dtype((np.void, flat_im.dtype.itemsize * flat_im.shape[1]))
        )
        if return_counts:
            unique, counts = np.unique(b, return_counts=True)
            u = unique.view(flat_im.dtype).reshape(-1, flat_im.shape[1])
            return u[1:], counts[1:]
        else:
            u = np.unique(b).view(flat_im.dtype).reshape(-1, flat_im.shape[1])
            return u[1:]

    def _sort_colors(self, colors: np.ndarray) -> np.ndarray:
        """
        Function for sorting an array of colors and setting gray colors
        to the end of the array.

        Args:
            colors (np.ndarray): array of colors values.

        Returns:
            Sorted array of color values.
        """
        sorted_colors = np.zeros_like(colors)
        end_counter = colors.shape[0] - 1
        start_counter = 0
        for c in colors:
            if (c[0] == c[1]) and (c[1] == c[2]):
                sorted_colors[end_counter] = c
                end_counter -= 1
            else:
                sorted_colors[start_counter] = c
                start_counter += 1
        return sorted_colors

    def _post_process_image(
        self,
        image: np.ndarray,
        unique_colors: Optional[np.ndarray] = None,
        opacity: float = 1.0,
        contour_thickness: int = 10,
    ) -> np.ndarray:
        """
        Routine for post processing the image, removing background,
        drawing contours and applying opacity.

        Args:
            image (np.ndarray): image array with comparison of segmentations.
            unique_colors (np.ndarray): array of unique color values found
                in image. If it is not provided it will find them
                (but this takes some extra seconds).
            opacity (float): opacity value for the colors in the image.
            contour_thickness (int): thickness of the contours in the
                return image.

        Returns:
            processed image with drawn contours, opacity in the interior
            of the distinct colors, and a removed background.
        """

        # Gray version of image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute alpha (array that is 0 for all black colors in gray image)
        _, alpha = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY)

        # Scale alpha array with input opacity and return to correct np.uint8 format.
        alpha = opacity * alpha
        alpha = alpha.astype(np.uint8)

        # Create rgba image with background removed and opacity applied elsewhere
        b, g, r = cv2.split(image)
        rgba = [b, g, r, alpha]
        im_new = cv2.merge(rgba, 4)

        # Go over all distinct colors in the image and draw contours
        if unique_colors is None:
            colors = self._get_unique_colors(image)
            colors = self._sort_colors(colors)

        else:
            colors = unique_colors

        for c in colors:
            colored_region = np.all(image == c, axis=2)
            contours, _ = cv2.findContours(
                skimage.img_as_ubyte(colored_region),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cl = c.tolist()
            cl.append(255)
            ct = tuple(cl)
            im_new = cv2.drawContours(
                im_new, contours, -1, ct, thickness=contour_thickness
            )
        return im_new

    def plot_overlay_segmentation(
        self,
        comparison_image: np.ndarray,
        base_image: np.ndarray,
        figure_name: str = "Comparison",
        opacity: float = 0.6,
        legend_anchor: tuple[float, float] = (1.0, 1.0),
        custom_legend: Optional[list[mpatches.Patch]] = None,
        custom_legend_text: Optional[list[str]] = None,
    ) -> None:
        """
        Plots a comparison image overlayed a base image using matplotlib.

        Args:
            comparison_image (np.ndarray): The image containing comparison of segmentations.
            base_image (np.ndarray): The base image that is to be overlayed.
            figure_name (str): Figure name.
            opacity (float): Tha opacity value for the comparison image.
            legend_anchor (tuple): tuple of coordinates (x,y) in euclidean style that
                determines legend anchor.
            custom_legend (Optional[list[mpatches.Patch]]): in case it is desirable to create
                a custom legend.
            custom_legend_text (Optional[list[str]]): in case it is desirable
                to customize legend.
        """

        # Get unique colors and sort them
        unique_colors = self._get_unique_colors(comparison_image)
        unique_colors = self._sort_colors(unique_colors)

        # Process the comparison image
        processed_comparison_image = self._post_process_image(
            comparison_image,
            unique_colors=unique_colors,
            opacity=opacity,
            contour_thickness=10,
        )

        # Scale base image to be of same size with comparison image
        base_image = cv2.resize(
            base_image,
            (processed_comparison_image.shape[1], processed_comparison_image.shape[0]),
        )

        # Create figure with legend
        plt.figure(figure_name)
        plt.imshow(base_image)
        plt.imshow(processed_comparison_image)
        if (custom_legend_text is None) and (custom_legend is None):
            patches = self._get_legend_patches(unique_colors=unique_colors)
        elif custom_legend is not None:
            patches = custom_legend
        else:
            patches = self._get_legend_patches(
                unique_colors=unique_colors, custom_legend_text=custom_legend_text
            )
        plt.legend(
            handles=patches, bbox_to_anchor=legend_anchor, loc=2, borderaxespad=0.0
        )
        plt.show()

    def color_fractions(
        self,
        comparison_image: np.ndarray,
        colors: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> tuple[list[float], list[float], np.ndarray[int], float, np.ndarray]:
        """
        Returns color fractions.

        Arguments:
            comparison_image (np.ndarray): Comparison of segmentations
            colors (np.ndarray): array of color values in the comparison image
            depth_map (np.ndarray, optional): depth map for the image

        Returns:
            (dict): Dictionary relating each color to the fraction of
                the number of pixels that the color occupies and the
                total number of occupied pixels in the image.
        """
        # Create empty list for color fractions
        fractions: list = []

        # Get unique colors
        if colors is None:
            colors = self._get_unique_colors(comparison_image)

        # Get depth map
        if depth_map is None:
            depth_map = np.ones(comparison_image.shape[:2])

        # Check if depth map and comparison image have the same shape
        assert depth_map.shape == comparison_image.shape[:2]

        # Get total number of pixels weighted by depth
        total_image = np.zeros(comparison_image.shape[:2])
        total_image[np.any(comparison_image != [0, 0, 0], axis=2)] = 1
        total_colored = np.sum(depth_map * total_image)

        # get weighted number of pixels for each color
        weighted_colors = []
        for c in colors:
            # Get only the pixels of color c
            only_c = np.zeros(comparison_image.shape[:2])
            only_c[np.all(comparison_image == c, axis=2)] = 1

            # weight the pixels by depth and sum
            weighted_color = np.sum(only_c * depth_map)

            # Append to fractions
            fractions.append(weighted_color / total_colored)
            weighted_colors.append(weighted_color)

        return weighted_colors, fractions, colors, total_colored, depth_map

    def _get_key(self, val, dictionary: dict):
        """
        Returns key from dictionary and provided value.

        Arguments:
            val: value in the dictionary
            dictionary (dict): dictionary where key matching
                to val is searched for

        returns
            key in dictionary


        """
        for key, value in dictionary.items():
            if np.array_equal(val, value):
                return key
        return f"key corresponding to {val} doesn't exist in dictionary"
