from math import ceil

import numpy as np


# def compare_segmentations(*segmentations):
#     N = len(segmentations)
#     for i in range(N - 1):
#         assert segmentations[0].shape == segmentations[i + 1].shape

#     return_image = np.full(shape=segmentations[0].shape + (3,), fill_value=[0, 0, 0])

#     gray_base = [round(180 / N), round(180 / N), round(180 / N)]

#     for k in range(N):
#         for i in range(k + 1, N):
#             return_image[
#                 np.where(np.logical_and(segmentations[k] == 1, segmentations[i] == 1))
#             ] += gray_base

#     for k in range(N):
#         only_tmp = segmentations[k] == 1
#         for j in filter(lambda j: j != k, range(N)):
#             only_tmp = np.logical_and(only_tmp, segmentations[j] == 0)
#         if k % 3 == 1:
#             return_image[np.where(only_tmp)] = [
#                 round(255 / (ceil(N / 3)) * ((k - 1) / 3 + 1)),
#                 0,
#                 0,
#             ]
#         if k % 3 == 2:
#             return_image[np.where(only_tmp)] = [
#                 0,
#                 round(255 / (ceil((N - 1) / 3)) * ((k - 2) / 3 + 1)),
#                 0,
#             ]
#         if k % 3 == 0:
#             return_image[np.where(only_tmp)] = [
#                 0,
#                 0,
#                 round(255 / (ceil((N - 2) / 3)) * ((k - 3) / 3 + 1)),
#             ]

#     return return_image


def compare_segmentations(*segmentations, **kwargs) -> np.ndarray:
    """
    Comparison of segmentations

    Args:
        *segmentations (args): 
    """
    # Define components
    components: list = kwargs.pop("components", [1,2])

    # Define number of segmentations
    N: int = len(segmentations)

    # Define gray colors
    gray_colors: np.ndarray = kwargs.pop("gray_colors", np.array([[180, 180, 180], [220, 220, 220], [200, 200, 200]], dtype = np.uint8))
    gray_base: np.ndarray = np.fix(gray_colors/N).astype(np.uint8)

    # Define unique colors
    colors: np.ndarray = kwargs.pop("colors",np.array(
        [
            [[205, 0, 0]],
            [[0, 205, 0]],
            [[0, 0, 205]],
            [[205, 205, 0]],
            [[205, 0, 205]],
            [[0, 205, 205]],
        ]
    , dtype = np.uint8))
    light_scaling: float = kwargs.pop("light_scaling", 1.5)
    colors_light: np.ndarray = np.trunc(light_scaling * colors)
    np.clip(colors_light, 0, 255, out=colors_light)
    colors: np.ndarray = np.hstack((colors, colors_light))

    # Assert that there are a sufficient amount of colors and that all of the segmentations are of equal size
    assert len(colors[:,0,0]) >= N
    for i in range(N - 1):
        assert segmentations[0].shape == segmentations[i + 1].shape

    # Create the return array
    return_image: np.ndarray = np.full(shape=segmentations[0].shape + (3,), fill_value=[0, 0, 0], dtype = np.uint8)


    # Enter gray everywhere there are ovelaps of different segmentations
    for k in range(N):
        for i in range(k + 1, N):

            # Overlap of first component
            return_image[
                np.where(np.logical_and(segmentations[k] == components[0], segmentations[i] == components[0]))
            ] += gray_base[0]

            # Overlap of second component
            return_image[
                np.where(np.logical_and(segmentations[k] == components[1], segmentations[i] == components[1]))
            ] += gray_base[1]

            # Overlap of different components
            return_image[
                np.where(
                    np.logical_or(
                        np.logical_and(segmentations[k] == components[0], segmentations[i] == components[1]),
                        np.logical_and(segmentations[k] == components[1], segmentations[i] == components[0]),
                    )
                )
            ] += gray_base[2]

    # Determine locations (and make modifications to return image) of unique coomponents
    for c in components:
        for k in range(N):
            only_tmp: np.ndarray = segmentations[k] == c
            for j in filter(lambda j: j != k, range(N)):
                only_tmp = np.logical_and(only_tmp, segmentations[j] == 0)

            return_image[np.where(only_tmp)] = colors[k, components.index(c)]

    return return_image
