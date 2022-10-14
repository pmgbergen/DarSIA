from __future__ import annotations

import numpy as np


def compare_segmentations(*segmentations, **kwargs) -> np.ndarray:
    """
    Comparison of segmentations.

    Args:
        segmentations (asterisk argument): Allows to provide an arbitraty
                number of segmented numpy arrays of integers to be compared
        optional keyword arguments (kwargs):
            components (list): list of the different components that are
                considered in the segmented images. So far only two are allowed
                to be provided.
            non_active_component (int): value of nonactive component in the segmented
                images.
            gray_colors (np.ndarray): array of three different scales of
                gray (in RGB format), one for each of the different combinations of
                components in the segmentations.
            colors (np.ndarray): Array of different colors that should
                indicate uniqe components in each segmentation.
            light_scaling (float): Indicate how much lighter the second
                component should be scaled in its unique color.
    """
    # Define components
    components: list = kwargs.pop("components", [1, 2])
    non_active_component: int = kwargs.pop("non_active_component", 0)

    # Define number of segmentations
    number_of_segmented_images: int = len(segmentations)

    # Define gray colors
    gray_colors: np.ndarray = kwargs.pop(
        "gray_colors",
        np.array([[180, 180, 180], [220, 220, 220], [200, 200, 200]], dtype=np.uint8),
    )
    gray_base: np.ndarray = np.fix(gray_colors / number_of_segmented_images).astype(
        np.uint8
    )

    # Define unique colors
    colors: np.ndarray = kwargs.pop(
        "colors",
        np.array(
            [
                [[205, 0, 0]],
                [[0, 205, 0]],
                [[0, 0, 205]],
                [[205, 205, 0]],
                [[205, 0, 205]],
                [[0, 205, 205]],
            ],
            dtype=np.uint8,
        ),
    )
    light_scaling: float = kwargs.pop("light_scaling", 1.5)
    colors_light: np.ndarray = np.trunc(light_scaling * colors)
    np.clip(colors_light, 0, 255, out=colors_light)
    colors: np.ndarray = np.hstack((colors, colors_light))

    # Assert that there are a sufficient amount of colors
    # and that all of the segmentations are of equal size
    assert len(colors[:, 0, 0]) >= number_of_segmented_images
    for i in range(number_of_segmented_images - 1):
        assert segmentations[0].shape == segmentations[i + 1].shape

    # Create the return array
    return_image: np.ndarray = np.zeros(
        segmentations[0].shape[:2] + (3,), dtype=np.uint8
    )

    # Enter gray everywhere there are ovelaps of different segmentations
    for k in range(number_of_segmented_images):
        for i in range(k + 1, number_of_segmented_images):

            # Overlap of components
            for c_num, c in enumerate(components):
                return_image[
                    np.logical_and(
                        segmentations[k] == c,
                        segmentations[i] == c,
                    )
                ] += gray_base[c_num]

            # Overlap of different components
            return_image[
                np.logical_or(
                    np.logical_and(
                        segmentations[k] == components[0],
                        segmentations[i] == components[1],
                    ),
                    np.logical_and(
                        segmentations[k] == components[1],
                        segmentations[i] == components[0],
                    ),
                )
            ] += gray_base[2]

    # Determine locations (and make modifications to return image) of unique components
    for c_num, c in enumerate(components):
        for k in range(number_of_segmented_images):
            only_tmp: np.ndarray = segmentations[k] == c
            for j in filter(lambda j: j != k, range(number_of_segmented_images)):
                only_tmp = np.logical_and(
                    only_tmp, segmentations[j] == non_active_component
                )

            return_image[only_tmp] = colors[k, c_num]

    return return_image
