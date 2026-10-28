"""Synthetic proofs for optional exact-CFA clipping evidence, no acquisition IO."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from darsia.fluidflower.native import cfa_clipping_diagnostics as diagnostic


def sensor_for(mosaic, *, black=1008, white=16383):
    """Use exactly the production integer EA arithmetic on a synthetic mosaic."""
    values = np.maximum(mosaic.astype(np.int32) - black, 0).astype(np.uint16)
    rgb = cv2.demosaicing(values, cv2.COLOR_BayerRG2BGR_EA)
    rgb.setflags(write=False)
    sensor = SimpleNamespace(rgb=rgb, source=dict(path="synthetic.ARW", sha256="a" * 64),
        metadata=dict(black_level=black, white_level=white,
                      sensor_saturated_count=int(np.count_nonzero(mosaic >= white))),
        profile_id="fixture", distortion=(0, 0, 0), ca_red=(0, 0, 0), ca_blue=(0, 0, 0))
    parameters = dict(edge_aware=dict(demosaic_code="COLOR_BayerRG2BGR_EA"), common=dict(
        sensor=dict(raw_wh=list(mosaic.shape[::-1]), camera_wh=list(mosaic.shape[::-1]),
                    black_level=black, black_levels=[black] * 4,
                    cfa_pattern=[[0, 1], [3, 2]], color_description="RGBG", tile_rows=13),
        profiles=dict(fixture=dict(coefficients=dict(cu=0., cv=0., tx=0., ty=0., R=1., amp=1.))),
        lens=dict(knots=[0., 1., 1000.], distortion_denominator=16384.,
                  ca_denominator=2097152., raw_offset_xy=[0, 0])))
    return sensor, parameters


@pytest.mark.parametrize("shape", [(12, 16), (37, 51), (200, 301)])
def test_exact_measured_photosites_survive_ea_and_clipping_recovery(shape):
    mosaic = np.random.default_rng(42).integers(0, 18000, shape, dtype=np.uint16)
    sensor, params = sensor_for(mosaic)
    before = sensor.rgb.copy()
    result = diagnostic.recover_cfa_clipping(sensor, params)
    expected = np.maximum(mosaic.astype(np.int32) - 1008, 0).astype(np.uint16)
    np.testing.assert_array_equal(result.mosaic_dn[1:-1, 1:-1], expected[1:-1, 1:-1])
    np.testing.assert_array_equal(result.flags[1:-1, 1:-1], (mosaic[1:-1, 1:-1] >= 16383))
    assert np.all(result.flags[[0, -1], :] == 2) and np.all(result.flags[:, [0, -1]] == 2)
    assert not result.mosaic_dn.flags.writeable and not result.flags.flags.writeable
    np.testing.assert_array_equal(sensor.rgb, before)
    assert json.loads(json.dumps(result.basis))["source"]["sha256"] == "a" * 64


@pytest.mark.parametrize("position", [(16, 16), (16, 17), (17, 16), (17, 17)])
def test_ea_numeric_support_matches_isolated_actual_source_impulses(position):
    mosaic = np.full((37, 51), 1008, np.uint16)
    mosaic[position] = 16383
    sensor, params = sensor_for(mosaic)
    result = diagnostic.recover_cfa_clipping(sensor, params)
    support = diagnostic.ea_clipping_support(result)
    # All other black-subtracted values are zero: every positive EA RGB output
    # therefore has this one original clipped photosite as a numeric contributor.
    np.testing.assert_array_equal((support & 1) != 0, sensor.rgb > 0)
    np.testing.assert_array_equal(support, diagnostic.ea_clipping_support(result, tile_rows=3))


def test_ea_gradient_control_samples_are_not_claimed_as_numeric_contributors():
    mosaic = np.full((37, 51), 1100, np.uint16)
    mosaic[16, 15] = 16383
    # Horizontal gradient is much greater than vertical, so centre R's G uses
    # the two un-clipped vertical samples, despite inspecting the clipped left.
    sensor, params = sensor_for(mosaic)
    support = diagnostic.ea_clipping_support(diagnostic.recover_cfa_clipping(sensor, params))
    assert support[16, 16, 1] == 0
    assert support[16, 15, 1] == 1
    assert support[1, 1, 0] & 2  # This reconstruction needs a lost outer CFA sample.


def test_rgb_border_replication_does_not_invent_original_border_measurements():
    mosaic = np.full((9, 11), 1008, np.uint16)
    mosaic[0, 0] = 16383
    sensor, params = sensor_for(mosaic)
    result = diagnostic.recover_cfa_clipping(sensor, params)
    assert result.basis["original_sensor_clipped_count"] == 1
    assert result.basis["recovered_interior_clipped_count"] == 0
    assert result.flags[0, 0] == 2


@pytest.mark.parametrize("failure", ["version", "code", "cfa", "black", "source", "count"])
def test_unverified_source_contracts_fail_closed(failure, monkeypatch):
    sensor, params = sensor_for(np.full((12, 16), 1008, np.uint16))
    if failure == "version":
        monkeypatch.setattr(cv2, "__version__", "4.99.0")
    elif failure == "code":
        params["edge_aware"]["demosaic_code"] = "COLOR_BayerRG2BGR"
    elif failure == "cfa":
        params["common"]["sensor"]["cfa_pattern"] = [[0, 0], [2, 2]]
    elif failure == "black":
        params["common"]["sensor"]["black_levels"][0] += 1
    elif failure == "source":
        sensor.source["sha256"] = "not-a-source-binding"
    else:
        sensor.metadata["sensor_saturated_count"] = 1000
    with pytest.raises(ValueError):
        diagnostic.recover_cfa_clipping(sensor, params)


@pytest.mark.parametrize("roi", [
    [[10, 10], [180, 10], [180, 280], [10, 280]],
    [[14, 22], [178, 12], [185, 275], [11, 289]],
    [[10, 280], [10, 10], [180, 10], [180, 280]],
])
def test_candidate_footprints_are_identical_to_original_sampler_inputs(roi, monkeypatch):
    import darsia
    image = np.random.default_rng(7).random((200, 301, 3), dtype=np.float32)
    captured = []

    def kmeans(pixels, *args):
        captured.append(pixels.copy())
        return 0., np.zeros((len(pixels), 1), np.int32), np.zeros((5, 3), np.float32)

    monkeypatch.setattr(cv2, "kmeans", kmeans)
    extractor = darsia.ColorCorrection(config=dict(roi=roi, active=True, clip=False))
    darsia.CustomColorChecker(image=extractor._restrict_to_roi(image))
    expected = diagnostic._patch_candidates(image, roi)
    assert len(captured) == 24
    for pixels, patch in zip(captured, [p for row in expected for p in row], strict=True):
        np.testing.assert_array_equal(pixels, patch.reshape(-1, 3))


def test_whole_chart_counts_are_small_json_and_do_not_decode_or_mutate(monkeypatch):
    mosaic = np.full((360, 540), 1008, np.uint16)
    mosaic[120:140, 130:150] = 16383
    sensor, params = sensor_for(mosaic)
    before = sensor.rgb.copy()
    # The provider accepts existing Sensor only; opening rawpy is never needed.
    import sys
    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(imread=lambda *_: pytest.fail("extra decode")))
    result = diagnostic.chart_clipping_diagnostics(sensor, params,
        [[20, 20], [340, 20], [340, 520], [20, 520]])
    counts = np.asarray(result["clipped_source_affected_pixels"])
    assert counts.shape == (4, 6, 3) and counts.max() > 0 and counts.max() <= 2500
    assert len(json.dumps(result)) < 8000
    assert result["unique_cfa_photosite_counts_available"] is False
    assert result["dominant_cluster_support_available"] is False
    assert result["correction_or_fit_policy_changed"] is False
    np.testing.assert_array_equal(sensor.rgb, before)


@pytest.mark.parametrize("roi", [
    [[20, 20], [340, 20], [340, 520], [20, 520]],
    [[27, 28], [331, 21], [337, 514], [24, 509]],
])
def test_patch_support_matches_actual_numeric_impulse_after_lens_ca_and_roi(roi):
    from darsia.fluidflower.native.raw_positioning import sample_sensor
    mosaic = np.full((360, 540), 1008, np.uint16)
    mosaic[120:140, 130:150] = 16383
    sensor, params = sensor_for(mosaic)
    sensor.distortion, sensor.ca_red, sensor.ca_blue = (0, 256, 512), (0, 400, 600), (0, -300, -600)
    params["common"]["profiles"]["fixture"]["coefficients"].update(cu=270., cv=180., R=300.)
    numeric = sample_sensor(sensor, np.eye(3), shape_hw=(360, 540), parameters=params)
    candidates = diagnostic._patch_candidates(np.moveaxis(numeric, 0, -1), roi)
    expected = [[np.count_nonzero(patch > 0, axis=(0, 1)).tolist() for patch in row]
                for row in candidates]
    result = diagnostic.chart_clipping_diagnostics(sensor, params, roi)
    assert np.count_nonzero(expected) > 0
    np.testing.assert_array_equal(result["clipped_source_affected_pixels"], expected)
    assert np.count_nonzero(result["unknown_source_affected_pixels"]) == 0


def test_exact_no_clipping_does_not_become_a_normalized_rgb_cutoff():
    mosaic = np.full((360, 540), 16382, np.uint16)
    sensor, params = sensor_for(mosaic)
    assert (sensor.rgb.astype(float) / (16383 - 1008)).min() > .999
    result = diagnostic.chart_clipping_diagnostics(sensor, params,
        [[20, 20], [340, 20], [340, 520], [20, 520]])
    assert np.count_nonzero(result["clipped_source_affected_pixels"]) == 0


def test_actual_sampler_change_is_not_silently_treated_as_same_footprint(monkeypatch):
    import darsia
    monkeypatch.setattr(darsia.CustomColorChecker, "_extract_from_image", lambda self, img: img)
    with pytest.raises(ValueError, match="sampler changed"):
        diagnostic._patch_candidates(np.zeros((360, 540, 3), np.float32),
            [[20, 20], [340, 20], [340, 520], [20, 520]])
