from pathlib import Path

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.color_embedding_registry import (
    ColorEmbeddingRegistry,
)
from darsia.presets.workflows.mode_resolution import (
    mode_requires_color_to_mass,
    resolve_mode_image,
)
from darsia.presets.workflows.segmentation_contours import SimpleSegmentation
from darsia.signals.color import (
    ColorChannelEmbedding,
    ColorEmbeddingBasis,
    ColorEmbeddingRuntime,
    ColorRangeEmbedding,
)


def _optical_image(arr: np.ndarray) -> darsia.OpticalImage:
    return darsia.OpticalImage(
        img=arr, space_dim=2, indexing="ij", color_space="RGB", dtype=int
    )


def _runtime(baseline: darsia.OpticalImage):
    class _Rig:
        def __init__(self, baseline):
            self.baseline = baseline
            self.labels = darsia.ScalarImage(
                img=np.zeros(baseline.img.shape[:2], dtype=int)
            )

    return ColorEmbeddingRuntime(rig=_Rig(baseline))


def test_resolve_color_channel_by_registry_key() -> None:
    arr = (
        np.array(
            [
                [[0, 0, 0], [255, 0, 0]],
                [[127, 10, 10], [64, 0, 0]],
            ],
            dtype=np.uint8,
        )
        / 255.0
    )  # Apply scaling to [0,1] range for RGB mode
    img = _optical_image(arr)
    registry = ColorEmbeddingRegistry(
        embeddings={
            "red_channel": ColorChannelEmbedding(
                embedding_id="red_channel",
                mode=darsia.ColorMode.ABSOLUTE,
                basis=ColorEmbeddingBasis.GLOBAL,
                calibration_root=Path("."),
                color_space="RGB",
                channel="r",
            )
        }
    )
    signal = resolve_mode_image(
        "red_channel",
        img,
        color_embedding_registry=registry,
        color_embedding_runtime=_runtime(img),
    )
    assert isinstance(signal, darsia.ScalarImage)
    assert np.isclose(signal.img[0, 0], 0.0)
    assert np.isclose(signal.img[0, 1], 1.0)
    assert np.isclose(signal.img[1, 0], 127.0 / 255.0)


def test_resolve_color_mode_rejects_invalid_token() -> None:
    arr = np.array([[[0, 0, 0], [255, 0, 0]]], dtype=np.uint8)
    img = _optical_image(arr)
    with pytest.raises(ValueError, match="Unsupported analysis mode"):
        resolve_mode_image("color.rgb.r", img)


def test_resolve_color_range_hsv_binary_mask() -> None:
    arr = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    img = _optical_image(arr)
    registry = ColorEmbeddingRegistry(
        embeddings={
            "custom_range": ColorRangeEmbedding(
                embedding_id="custom_range",
                mode=darsia.ColorMode.ABSOLUTE,
                basis=ColorEmbeddingBasis.GLOBAL,
                calibration_root=Path("."),
                color_space="HSV",
                ranges=[(0.2, 0.4), (0.5, None), (0.8, None)],
            )
        }
    )
    mask = resolve_mode_image(
        "custom_range",
        img,
        color_embedding_registry=registry,
        color_embedding_runtime=_runtime(img),
    )
    assert isinstance(mask, darsia.ScalarImage)
    assert np.array_equal(
        mask.img.astype(np.uint8), np.array([[0, 1, 0]], dtype=np.uint8)
    )


def test_simple_segmentation_supports_color_mode_without_mass_inputs() -> None:
    arr = np.array([[[0, 0, 0], [255, 0, 0]]], dtype=np.uint8)
    img = _optical_image(arr)
    registry = ColorEmbeddingRegistry(
        embeddings={
            "red_channel": ColorChannelEmbedding(
                embedding_id="red_channel",
                mode=darsia.ColorMode.ABSOLUTE,
                basis=ColorEmbeddingBasis.GLOBAL,
                calibration_root=Path("."),
                color_space="RGB",
                channel="r",
            )
        }
    )
    segmentation = SimpleSegmentation(mode="red_channel", threshold=0.5)
    mask = segmentation(
        img,
        saturation_g=None,
        concentration_aq=None,
        mass=None,
        mass_analysis_result=None,
        color_embedding_registry=registry,
        color_embedding_runtime=_runtime(img),
    )
    assert np.array_equal(mask.img.astype(np.uint8), np.array([[0, 1]], dtype=np.uint8))


def test_mode_requires_color_to_mass_for_new_and_legacy_modes() -> None:
    assert mode_requires_color_to_mass("concentration_aq")
    assert not mode_requires_color_to_mass("color.red_channel")
    assert not mode_requires_color_to_mass("color.custom_range")
