import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.colorchannel_registry import (
    ColorChannelRegistry,
    NamedColorChannelConfig,
)
from darsia.presets.workflows.config.colorrange import (
    ColorRangeConfig,
    NamedColorRangeConfig,
)
from darsia.presets.workflows.mode_resolution import (
    mode_requires_color_to_mass,
    resolve_mode_image,
)
from darsia.presets.workflows.segmentation_contours import SimpleSegmentation


def _optical_image(arr: np.ndarray) -> darsia.OpticalImage:
    return darsia.OpticalImage(img=arr, space_dim=2, indexing="ij", color_space="RGB")


def test_resolve_colorchannel_by_registry_key() -> None:
    arr = np.array(
        [
            [[0, 0, 0], [255, 0, 0]],
            [[127, 10, 10], [64, 0, 0]],
        ],
        dtype=np.uint8,
    )
    img = _optical_image(arr)
    registry = ColorChannelRegistry(
        channels={
            "red_channel": NamedColorChannelConfig(color_space="RGB", channel="r"),
        }
    )
    signal = resolve_mode_image(
        "colorchannel.red_channel",
        img,
        colorchannel_registry=registry,
    )
    assert isinstance(signal, darsia.ScalarImage)
    assert np.isclose(signal.img[0, 0], 0.0)
    assert np.isclose(signal.img[0, 1], 1.0)
    assert np.isclose(signal.img[1, 0], 127.0 / 255.0)


def test_resolve_colorchannel_rejects_legacy_inline_syntax() -> None:
    arr = np.array([[[0, 0, 0], [255, 0, 0]]], dtype=np.uint8)
    img = _optical_image(arr)
    with pytest.raises(ValueError, match="Inline colorchannel modes"):
        resolve_mode_image("colorchannel.rgb.r", img)


def test_resolve_colorrange_hsv_binary_mask() -> None:
    arr = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    img = _optical_image(arr)
    colorrange = ColorRangeConfig(
        ranges={
            "custom_range": NamedColorRangeConfig(
                color_space="HSV",
                ranges=[(0.2, 0.4), (0.5, None), (0.8, None)],
            )
        }
    )
    mask = resolve_mode_image(
        "colorrange.custom_range", img, colorrange_config=colorrange
    )
    assert isinstance(mask, darsia.ScalarImage)
    assert np.array_equal(
        mask.img.astype(np.uint8), np.array([[0, 1, 0]], dtype=np.uint8)
    )


def test_simple_segmentation_supports_colorchannel_mode_without_mass_inputs() -> None:
    arr = np.array([[[0, 0, 0], [255, 0, 0]]], dtype=np.uint8)
    img = _optical_image(arr)
    registry = ColorChannelRegistry(
        channels={
            "red_channel": NamedColorChannelConfig(color_space="RGB", channel="r"),
        }
    )
    segmentation = SimpleSegmentation(mode="colorchannel.red_channel", threshold=0.5)
    mask = segmentation(
        img,
        saturation_g=None,
        concentration_aq=None,
        mass=None,
        mass_analysis_result=None,
        colorrange_config=None,
        colorchannel_registry=registry,
    )
    assert np.array_equal(mask.img.astype(np.uint8), np.array([[0, 1]], dtype=np.uint8))


def test_mode_requires_color_to_mass_for_new_and_legacy_modes() -> None:
    assert mode_requires_color_to_mass("concentration_aq")
    assert not mode_requires_color_to_mass("colorchannel.red_channel")
    assert not mode_requires_color_to_mass("colorrange.custom_range")
