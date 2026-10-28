"""Presentation must preserve the scientific signal and its finite channels."""

import numpy as np
import pytest

from darsia.fluidflower.native import display_modes as display


@pytest.fixture
def parameters():
    return dict(
        matrix=np.eye(3).tolist(),
        gains={"base": 1.0, "lift": 2.0},
        luminance_weights=[0.2126, 0.7152, 0.0722],
        base_mode="base",
        modes=["base", "lift"],
        white_multiplier=0.95,
        fill_connectivity=8,
        fill_max_area_px=1,
        fill_radius_px=2,
        fill_max_iterations=1,
        highlight_threshold=0.55,
        highlight_width=0.25,
        chroma_median_size_px=3,
        unfilled_rgb8=40,
        invalid_rgb8=128,
        srgb=dict(
            threshold=0.0031308,
            linear_gain=12.92,
            power_gain=1.055,
            power_denominator=2.4,
            power_offset=0.055,
        ),
        contract_metadata={"contract": "display-only"},
    )


def test_photographic_transfer_quantization_invalid_and_tiling(parameters):
    signal = np.log(np.full((3, 5, 7), 0.25, np.float32))
    signal[:, 1, 2] = np.nan
    saved = signal.copy()
    result = display.photographic_display(signal, tile_rows=2, parameters=parameters)
    other = display.photographic_display(signal, tile_rows=5, parameters=parameters)
    assert np.array_equal(result, other)
    assert result.dtype == np.float32 and result.shape == (5, 7, 3)
    assert np.all(result[1, 2] == np.float32(128 / 255))
    assert np.all(result[0, 0] == np.float32(137 / 255))
    assert np.array_equal(signal, saved, equal_nan=True)
    assert display.photographic_contract(parameters=parameters) == {
        "contract": "display-only"
    }
    assert display.validate_display_mode("base", parameters=parameters) == "base"
    with pytest.raises(ValueError, match="mode"):
        display.validate_display_mode("unknown", parameters=parameters)


def test_modes_keep_large_invalid_regions_neutral_and_input_unchanged(parameters):
    signal = np.log(np.full((3, 9, 9), 0.2, np.float32))
    signal[:, 3:6, 3:6] = np.nan
    saved = signal.copy()
    modes, metadata = display.render_display_modes(
        signal, np.ones((9, 9), bool), tile_rows=2, parameters=parameters
    )
    assert list(modes) == ["base", "lift"]
    assert modes["lift"][0, 0, 0] > modes["base"][0, 0, 0]
    assert all(
        np.all(image[3:6, 3:6] == np.float32(40 / 255)) for image in modes.values()
    )
    assert metadata["invalid_cleanup"]["unfilled_invalid_pixels"] == 9
    assert np.array_equal(signal, saved, equal_nan=True)
    assert metadata["contract"]["contract"] == "display-only"
    assert (
        len(display.display_contract(parameters=parameters)["renderer_code_sha256"])
        == 64
    )


def test_raw_saturation_neutralisation_respects_photo_support(parameters):
    linear = np.array([[[0.2, 0.8, 0.4], [0.2, 0.8, 0.4]]], np.float32)
    sand = np.array([[True, False]])
    metadata = display._neutralise_highlights(
        linear,
        sand,
        np.ones((1, 2), bool),
        np.ones((1, 2), bool),
        parameters=parameters,
    )
    assert np.array_equal(linear[0, 0], np.array([0.2, 0.8, 0.4], np.float32))
    assert np.all(linear[0, 1] == np.float32(0.8))
    assert metadata["pixels_affected"] == 1
