import numpy as np
from math import ceil


def compare_segmentations(*segmentations):
    N = len(segmentations)
    for i in range(N - 1):
        assert segmentations[0].shape == segmentations[i + 1].shape

    return_image = np.full(shape=segmentations[0].shape + (3,), fill_value=[0, 0, 0])

    gray_base = [round(180 / N), round(180 / N), round(180 / N)]

    for k in range(N):
        for i in range(k + 1, N):
            return_image[
                np.where(np.logical_and(segmentations[k] == 1, segmentations[i] == 1))
            ] += gray_base

    for k in range(N):
        only_tmp = segmentations[k] == 1
        for j in filter(lambda j: j != k, range(N)):
            only_tmp = np.logical_and(only_tmp, segmentations[j] == 0)
        if k % 3 == 1:
            return_image[np.where(only_tmp)] = [
                round(255 / (ceil(N / 3)) * ((k - 1) / 3 + 1)),
                0,
                0,
            ]
        if k % 3 == 2:
            return_image[np.where(only_tmp)] = [
                0,
                round(255 / (ceil((N - 1) / 3)) * ((k - 2) / 3 + 1)),
                0,
            ]
        if k % 3 == 0:
            return_image[np.where(only_tmp)] = [
                0,
                0,
                round(255 / (ceil((N - 2) / 3)) * ((k - 3) / 3 + 1)),
            ]

    return return_image


def compare_segmentations_two_components(*segmentations):
    # unique_colors = {"red": [205,38,38], "light_red": [255,48,48], "blue": [1,1,1], "light_blue": [1,1,1], "green": [1,1,1], "light_green"}
    colors = np.array(
        [
            [[205,0,0]],
            [[0, 205, 0]],
            [[0, 0, 205]],
            [[205, 205, 0]],
            [[205, 0, 205]],
            [[0, 205, 205]],
        ]
    )
    colors_light = np.trunc(1.5*colors)
    np.clip(colors_light, 0, 255, out = colors_light)
    colors = np.hstack((colors,colors_light))
    print(colors)
    NC = len(colors[:, 0, 0])
    N = len(segmentations)
    assert (NC>=N)
    for i in range(N - 1):
        assert segmentations[0].shape == segmentations[i + 1].shape

    return_image = np.full(shape=segmentations[0].shape + (3,), fill_value=[0, 0, 0])

    gray_base_0 = [round(180 / N), round(180 / N), round(180 / N)]
    gray_base_1 = [round(220 / N), round(220 / N), round(220 / N)]
    gray_base_2 = [round(200 / N), round(180 / N), round(190 / N)]

    for k in range(N):
        for i in range(k + 1, N):
            return_image[
                np.where(np.logical_and(segmentations[k] == 1, segmentations[i] == 1))
            ] += gray_base_0
            return_image[
                np.where(np.logical_and(segmentations[k] == 2, segmentations[i] == 2))
            ] += gray_base_1
            return_image[
                np.where(
                    np.logical_or(
                        np.logical_and(segmentations[k] == 1, segmentations[i] == 2),
                        np.logical_and(segmentations[k] == 2, segmentations[i] == 1),
                    )
                )
            ] += gray_base_2

    for c in range(1, 2):
        for k in range(N):
            only_tmp = segmentations[k] == c
            for j in filter(lambda j: j != k, range(N)):
                only_tmp = np.logical_and(only_tmp, segmentations[j] == 0)

            return_image[np.where(only_tmp)] = colors[k, c]

    return return_image
