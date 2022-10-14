from __future__ import annotations

import numpy as np


def compare_segmentations(*segmentations, **kwargs) -> np.ndarray:
    """
    Comparison of segmentations.

    Args:
        segmentations (asterisk argument): Allows to provide an arbitraty
                number of segmented images to be compared
        optional keyword arguments (kwargs):
            components (list): list of the different components that are
                found in the segmented images. So far only two are allowed
                to be provided.
            gray_colors (np.ndarray): array of three different scales of
                gray, one for each of the different combinations of
                components in the segmentations.
            colors (np.ndarray): Array of different colors that should
                indicate uniqe components in each segmentation.
            light_scaling (float): Indicate how much lighter the second
                component should be scaled in its unique color.
    """
    # Define components
    components: list = kwargs.pop("components", [1, 2])

    # Define number of segmentations
    N: int = len(segmentations)

    # Define gray colors
    gray_colors: np.ndarray = kwargs.pop(
        "gray_colors",
        np.array([[180, 180, 180], [220, 220, 220], [200, 200, 200]], dtype=np.uint8),
    )
    gray_base: np.ndarray = np.fix(gray_colors / N).astype(np.uint8)

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
    assert len(colors[:, 0, 0]) >= N
    for i in range(N - 1):
        assert segmentations[0].shape == segmentations[i + 1].shape

    # Create the return array
    return_image: np.ndarray = np.full(
        shape=segmentations[0].shape + (3,), fill_value=[0, 0, 0], dtype=np.uint8
    )

    # Enter gray everywhere there are ovelaps of different segmentations
    for k in range(N):
        for i in range(k + 1, N):

            # Overlap of first component
            return_image[
                np.where(
                    np.logical_and(
                        segmentations[k] == components[0],
                        segmentations[i] == components[0],
                    )
                )
            ] += gray_base[0]

            # Overlap of second component
            return_image[
                np.where(
                    np.logical_and(
                        segmentations[k] == components[1],
                        segmentations[i] == components[1],
                    )
                )
            ] += gray_base[1]

            # Overlap of different components
            return_image[
                np.where(
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
                )
            ] += gray_base[2]

    # Determine locations (and make modifications to return image) of unique components
    for c in components:
        for k in range(N):
            only_tmp: np.ndarray = segmentations[k] == c
            for j in filter(lambda j: j != k, range(N)):
                only_tmp = np.logical_and(only_tmp, segmentations[j] == 0)

            return_image[np.where(only_tmp)] = colors[k, components.index(c)]

    return return_image
