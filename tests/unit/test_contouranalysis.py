import unittest.mock as mock

import matplotlib.pyplot as plt
import numpy as np

import darsia
from darsia.single_image_analysis.contouranalysis import (
    ContourAnalysis,
    ContourEvolutionAnalysis,
    PathUnit,
)


def test_plot_valleys_respects_range_and_optional_dots():
    img = darsia.Image(img=np.zeros((8, 10, 3), dtype=np.uint8))
    valleys = np.array([[[2, 3]], [[5, 4]]], dtype=int)

    contour_analysis = ContourAnalysis()
    with mock.patch.object(plt, "vlines") as vlines_mock, mock.patch.object(
        plt, "scatter"
    ) as scatter_mock:
        contour_analysis.plot_valleys(
            img,
            valleys,
            show=False,
            y_min=1,
            y_max=7,
            plot_valley_dots=True,
            valley_color="m",
        )

    vlines_mock.assert_called_once()
    x_values, y_min, y_max = vlines_mock.call_args.args[:3]
    assert np.array_equal(x_values, np.array([2, 5]))
    assert y_min == 1
    assert y_max == 7
    assert vlines_mock.call_args.kwargs["colors"] == "m"
    scatter_mock.assert_called_once()


def test_find_valley_paths_tracks_valleys_over_time():
    contour_evolution_analysis = ContourEvolutionAnalysis()
    empty_peaks = np.zeros((0, 1, 2), dtype=int)
    contour_evolution_analysis.add(
        peaks=empty_peaks, valleys=np.array([[[1, 2]], [[6, 2]]], dtype=int), time=0.0
    )
    contour_evolution_analysis.add(
        peaks=empty_peaks, valleys=np.array([[[2, 3]], [[7, 3]]], dtype=int), time=1.0
    )

    contour_evolution_analysis.find_valley_paths()

    assert len(contour_evolution_analysis.valley_paths) == 2
    for valley_path in contour_evolution_analysis.valley_paths:
        assert len(valley_path) == 2
        assert valley_path[0].time == 0
        assert valley_path[1].time == 1


def test_plot_valley_paths_uses_active_and_inactive_alpha():
    img = darsia.Image(img=np.zeros((8, 10, 3), dtype=np.uint8))

    contour_evolution_analysis = ContourEvolutionAnalysis()
    contour_evolution_analysis.valley_paths = [
        [PathUnit(0, 0, np.array([2, 2])), PathUnit(1, 0, np.array([3, 3]))]
    ]

    with mock.patch.object(plt, "plot") as plot_mock:
        contour_evolution_analysis.plot_valley_paths(img=img, show=False, color="k")

    alphas = [call.kwargs.get("alpha") for call in plot_mock.call_args_list]
    assert 1.0 in alphas
    assert 0.5 in alphas
    for call in plot_mock.call_args_list:
        assert call.kwargs.get("color") == "k"
