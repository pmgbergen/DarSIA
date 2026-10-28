"""RAW-only source guards and equivalence to the existing sensor sampler."""

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
import sys

import cv2
import numpy as np
from PIL import Image
import pytest
import tifffile

from darsia.fluidflower.native import raw_positioning as raw
from darsia.fluidflower.native import raw_photograph as photograph
from darsia.fluidflower.native import registration as legacy


@pytest.fixture
def source(tmp_path, monkeypatch):
    mosaic = np.random.default_rng(7).integers(0, 16000, (12, 16), dtype=np.uint16)
    path = tmp_path / "source.ARW"
    tifffile.imwrite(path, mosaic, extratags=[
        (271, "s", 0, "SONY", True), (272, "s", 0, "ZV-1", True)])
    fake = SimpleNamespace(
        raw_image_visible=mosaic, raw_pattern=np.array([[0, 1], [3, 2]]),
        color_desc=b"RGBG", black_level_per_channel=[1008] * 4,
        white_level=16383, camera_whitebalance=[2., 1., 1.5, 1.],
        rgb_xyz_matrix=np.eye(4, 3), sizes=SimpleNamespace(
            crop_left_margin=2, crop_top_margin=2, crop_width=12, crop_height=8))
    calls = []

    class OpenRaw:
        def __enter__(self):
            return fake

        def __exit__(self, *_args):
            return False

    def imread(value):
        calls.append(value)
        return OpenRaw()

    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(imread=imread))
    zeros = (0, 0, 0)
    monkeypatch.setattr(legacy, "read_lens_tables", lambda *a, **k: (zeros, zeros, zeros))
    edge = dict(demosaic_code="COLOR_BayerRG2BGR_EA",
                order_fixture=[[10000, 4000, 10000, 4000, 10000, 4000],
                               [4000, 1000, 4000, 1000, 4000, 1000]] * 3,
                order_check_xy=[2, 2], order_expected_rgb=[10000, 4000, 1000],
                certificate=dict(contract="existing-oracle", white_balance_reason="fixture"))
    parameters = dict(edge_aware=edge, common=dict(
        sensor=dict(raw_wh=[16, 12], camera_wh=[12, 8], content_wh=[9, 6],
                    black_level=1008, black_levels=[1008] * 4, white_maximum=16383,
                    cfa_pattern=[[0, 1], [3, 2]], color_description="RGBG", tile_rows=3,
                    green_mean_weight=.5, plane_stride=2,
                    libraw_crop_xywh=[2, 2, 12, 8]),
        profiles=dict(calibrated=dict(distortion=list(zeros), coefficients=dict(
            cu=5.5, cv=3.5, tx=0., ty=0., R=8., amp=1.))),
        lens=dict(knots=[0., .5, 1.], distortion_denominator=16384.,
                  ca_denominator=2097152., raw_offset_xy=[2, 2])))
    return SimpleNamespace(path=path, digest=sha256(path.read_bytes()).hexdigest(),
                           parameters=parameters, fake=fake, calls=calls, zeros=zeros)


def read(source):
    return raw.read_sensor(source.path, expected_sha256=source.digest,
                           parameters=source.parameters, profile_id="calibrated")


def test_raw_only_read_never_opens_a_companion_or_uses_the_camera_name_registry(source, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("A JPEG/PIL source must not be opened")

    monkeypatch.setattr(Image, "open", refuse)
    sensor = read(source)
    assert sensor.metadata["model"] == "ZV-1"
    assert sensor.source["sha256"] == source.digest
    assert sensor.rgb.dtype == np.uint16 and not sensor.rgb.flags.writeable
    assert raw.camera_u(sensor, parameters=source.parameters).shape == (3, 8, 12)
    assert source.calls == [str(source.path)]
    assert not list(source.path.parent.glob("*.JPG"))


@pytest.mark.parametrize("homography", [
    np.eye(3), np.array([[.98, .01, .25], [-.01, 1.01, .125], [.0001, -.0002, 1.]])])
def test_scientific_samples_match_the_existing_renderer_bit_for_bit(source, monkeypatch, homography):
    sensor = read(source)
    expected = raw.sample_sensor(sensor, homography, shape_hw=(6, 9),
                                 parameters=source.parameters)
    camera, review = source.path.with_suffix(".JPG"), source.path.parent / "review.JPG"
    Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(camera)
    Image.fromarray(np.zeros((6, 9, 3), np.uint8)).save(review)
    monkeypatch.setattr(legacy, "review_to_camera", lambda *a, **k: (homography, {}))
    # This oracle fixes H to compare DN sampling. Independent registration QA
    # requires real-sized scenes and is covered by test_native_registration.
    monkeypatch.setattr(legacy, "raw_review_residual", lambda *a, **k: {})
    original, _ = legacy.registered_edge_aware_sensor_linear(
        source.path, camera, review, expected_source_hashes=dict(
            arw_sha256=source.digest, camera_jpeg_sha256=sha256(camera.read_bytes()).hexdigest(),
            review_native_jpeg_sha256=sha256(review.read_bytes()).hexdigest()),
        content_box=(0, 0, 9, 6), parameters=source.parameters,
        source_id="fixture", profile_id="calibrated")
    assert expected.dtype == original.dtype == np.float32
    np.testing.assert_array_equal(expected.view(np.uint32), original.view(np.uint32))


def test_wrong_sha_stops_before_decoding(source):
    with pytest.raises(ValueError, match="SHA-256 differs"):
        raw.read_sensor(source.path, expected_sha256="0" * 64,
                        parameters=source.parameters, profile_id="calibrated")
    assert not source.calls


def test_wrong_lens_table_stops_before_decoding(source):
    parameters = deepcopy(source.parameters)
    parameters["common"]["profiles"]["calibrated"]["distortion"][0] = 1
    with pytest.raises(ValueError, match="lens table differs"):
        raw.read_sensor(source.path, expected_sha256=source.digest,
                        parameters=parameters, profile_id="calibrated")
    assert not source.calls


@pytest.mark.parametrize("field,value,match", [
    ("crop_width", 11, "explicit sensor profile"),
    ("crop_left_margin", 3, "explicit sensor profile")])
def test_raw_metadata_must_define_the_calibrated_u_grid(source, field, value, match):
    setattr(source.fake.sizes, field, value)
    with pytest.raises(ValueError, match=match):
        read(source)


def test_sensor_version_change_during_decode_is_refused(source, monkeypatch):
    original = cv2.demosaicing

    def mutate(*args, **kwargs):
        source.path.write_bytes(source.path.read_bytes() + b"changed")
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "demosaicing", mutate)
    with pytest.raises(ValueError, match="changed while"):
        read(source)


def test_invalid_sampling_transform_is_refused(source):
    sensor = read(source)
    with pytest.raises(ValueError, match="invertible"):
        raw.sample_sensor(sensor, np.zeros((3, 3)), shape_hw=(6, 9),
                          parameters=source.parameters)


def test_native_review_preserves_raw_dn_on_an_identity_shape_chain(source):
    rgb = raw.camera_u(read(source), parameters=source.parameters)
    # An existing geometric transform is an Image callable, with overwrite.
    def identity(image, **kwargs):
        return image

    review = raw.native_review_rgb(rgb, [identity], white_range_dn=16383 - 1008,
                                  shape_hw=(8, 12), image_metadata=dict(height=1., width=1.))
    assert review.shape == rgb.shape and review.dtype == np.float32
    np.testing.assert_allclose(review, rgb, rtol=1e-7, atol=1e-3)


def test_new_source_mode_refuses_a_processed_photo_before_any_decode(source):
    with pytest.raises(ValueError, match="camera RAW"):
        raw.read_sensor(source.path.with_suffix(".JPG"), expected_sha256=source.digest,
                        parameters=source.parameters, profile_id="calibrated")
    assert not source.calls


def test_lens_origin_is_not_assumed_to_equal_an_unbound_preview_crop(source):
    del source.parameters["common"]["sensor"]["libraw_crop_xywh"]
    source.fake.sizes.crop_left_margin = 0
    sensor = read(source)
    assert sensor.metadata["crop_xywh"][0] == 0
    assert source.parameters["common"]["lens"]["raw_offset_xy"][0] == 2


def test_display_and_science_use_the_shared_u_profile(source):
    profile = source.path.with_suffix(".profile.json")
    profile.write_text(json.dumps(dict(registration=source.parameters)), encoding="utf8")
    binding = dict(profile_path=profile, profile_sha256=sha256(profile.read_bytes()).hexdigest(),
                   profile_id="calibrated", expected_sha256=source.digest)
    yy, xx = np.mgrid[0:12, 0:16]
    display = np.stack([xx * 10, yy * 10, (xx + yy) * 5], axis=-1).astype(np.uint8)
    calls = []
    def postprocess(**kwargs):
        calls.append(kwargs)
        return display
    source.fake.postprocess = postprocess
    rgb = photograph.load_camera_u_rgb8(source.path, **binding)
    np.testing.assert_array_equal(rgb, display[2:10, 2:14])
    linear = photograph.load_camera_u_linear(source.path, **binding)
    np.testing.assert_array_equal(linear, raw.camera_u(read(source), parameters=source.parameters))
    assert photograph.camera_u_size(**{k:v for k,v in binding.items() if k!='expected_sha256'}) == (12, 8)
    assert calls == [dict(use_camera_wb=True, output_bps=8,
                          half_size=False, user_flip=0)]


def test_display_reduction_happens_after_the_u_geometry(source):
    profile = source.path.with_suffix(".profile.json")
    profile.write_text(json.dumps(source.parameters), encoding="utf8")
    yy, xx = np.mgrid[0:12, 0:16]
    display = np.stack([xx * 10, yy * 10, (xx + yy) * 5], axis=-1).astype(np.uint8)
    source.fake.postprocess = lambda **kwargs: display
    rgb = photograph.load_camera_u_rgb8(source.path, profile_path=profile,
        profile_sha256=sha256(profile.read_bytes()).hexdigest(), profile_id='calibrated', factor=2)
    expected = np.asarray(Image.fromarray(display[2:10, 2:14]).resize((6, 4), Image.Resampling.BOX))
    np.testing.assert_array_equal(rgb, expected)
