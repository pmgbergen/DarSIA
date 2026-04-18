import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.presets.workflows.helper.helper_roi_viewer import (
    _build_roi_selection_masks,
    _compute_coarse_shape,
    _preload_coarse_images,
)


def _make_image(shape: tuple[int, int] = (200, 100)) -> darsia.Image:
    img = np.zeros((*shape, 3), dtype=float)
    img[..., 0] = 1.0
    return darsia.OpticalImage(img, dimensions=[2.0, 1.0], color_space="RGB")


def test_compute_coarse_shape_respects_min_rows_and_aspect_ratio() -> None:
    coarse_shape = _compute_coarse_shape((200, 100), min_rows=120, downsampling_factor=4)
    assert coarse_shape == (120, 60)


def test_preload_coarse_images_resizes_once_for_all_images() -> None:
    images = [_make_image((200, 100)), _make_image((200, 100))]
    coarse_images = _preload_coarse_images(
        images, min_rows=50, downsampling_factor=4
    )
    assert len(coarse_images) == 2
    assert coarse_images[0].img.shape[:2] == (50, 25)
    assert coarse_images[1].img.shape[:2] == (50, 25)


def test_preload_coarse_images_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="received no images"):
        _preload_coarse_images([])


def test_build_roi_selection_masks_supports_none_single_and_all_union() -> None:
    image = _make_image((100, 100))
    roi_1 = RoiConfig(
        roi=darsia.CoordinateArray([[0.1, 0.1], [0.5, 0.5]]),
        name="roi_1",
    )
    roi_2 = RoiConfig(
        roi=darsia.CoordinateArray([[0.5, 0.5], [0.9, 0.9]]),
        name="roi_2",
    )
    masks = _build_roi_selection_masks(image, {"roi_1": roi_1, "roi_2": roi_2})

    assert masks["none"] is None
    assert isinstance(masks["roi_1"], np.ndarray)
    assert isinstance(masks["roi_2"], np.ndarray)
    assert isinstance(masks["all"], np.ndarray)
    assert np.array_equal(
        masks["all"],
        np.logical_or(masks["roi_1"], masks["roi_2"]),
    )
