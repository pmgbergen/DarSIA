"""Camera dispatch, metadata-bound RAW decoding and explicit geometry profiles.

Synthetic tests run everywhere. Tests on the FF_UM (Nikon Z 7II) and FF_BM
(Sony ILCE-7RM3) sample pairs run when the files exist; set DARSIA_TEST_NEF or
DARSIA_TEST_ARW to point at another copy (the companion JPEG must sit next to
it with the same stem).
"""

from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import tifffile

from darsia.fluidflower.native import raw_camera as rc
from darsia.fluidflower.native import registration as reg

EDGE_AWARE = {
    "demosaic_code": "COLOR_BayerRG2BGR_EA",
    "order_fixture": [[10000, 4000] * 3, [4000, 1000] * 3] * 3,
    "order_check_xy": [2, 2],
    "order_expected_rgb": [10000, 4000, 1000],
    "certificate": {"contract": "fixture-edge", "white_balance_reason": "test"},
}
AGREEMENT = dict(
    sigma_small=1.2,
    sigma_large=6.0,
    minimum_template_std=1e-4,
    template_half=16,
    search_half=6,
    grid_step_px=96,
    border_px=96,
    min_correlation=0.5,
    minimum_matches=200,
    radial_bin_edges=[0.0, 0.25, 0.5, 0.75, 1.01],
)
NIKON = rc.CameraIdentity("NIKON CORPORATION", "NIKON Z 7_2", 1)
SONY_7RM3 = rc.CameraIdentity("SONY", "ILCE-7RM3", 1)
ZV1 = rc.CameraIdentity("SONY", "DSC-ZV1", 1)


class FakeRaw:
    """Minimal rawpy object with LibRaw-style metadata."""

    def __init__(
        self,
        mosaic,
        *,
        black=(1008,) * 4,
        white=16383,
        pattern=((0, 1), (3, 2)),
        desc=b"RGBG",
        crop=(2, 2, 8, 6),
    ):
        self.raw_image_visible = mosaic
        self.raw_pattern = np.array(pattern)
        self.color_desc = desc
        self.black_level_per_channel = list(black)
        self.white_level = white
        self.camera_white_level_per_channel = [white - 1000] * 4
        self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
        self.rgb_xyz_matrix = np.eye(4, 3)
        self.sizes = SimpleNamespace(
            flip=0,
            crop_left_margin=crop[0],
            crop_top_margin=crop[1],
            crop_width=crop[2],
            crop_height=crop[3],
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def fake_rawpy(monkeypatch):
    calls, state = [], {}

    def imread(path):
        calls.append(path)
        return state["raw"]

    monkeypatch.setitem(
        sys.modules, "rawpy", SimpleNamespace(imread=imread, libraw_version=(0, 22, 1))
    )
    return calls, state


def mosaic(height=12, width=16, seed=3, low=0, high=16384):
    return (
        np.random.default_rng(seed)
        .integers(low, high, (height, width))
        .astype(np.uint16)
    )


def write_raw(tmp_path, name="DCS00001.NEF"):
    path = tmp_path / name
    path.write_bytes(b"synthetic raw bytes")
    return path


# --- dispatch ------------------------------------------------------------------


def test_zv1_dispatch_returns_the_original_frozen_renderer():
    for identity in (ZV1, rc.CameraIdentity(" sony ", "dsc-zv1")):
        route = rc.select_raw_route(identity)
        assert route.route == rc.FROZEN_ZV1_ROUTE
        assert route.renderer is reg.registered_edge_aware_sensor_linear
        assert route.decoder is None
    assert reg.registered_edge_aware_sensor_linear.__module__ == reg.__name__
    # The frozen module does not depend on the new reader.
    assert "raw_camera" not in Path(reg.__file__).read_text(encoding="utf-8")


def test_new_cameras_select_the_metadata_bound_decoder():
    for identity, key, series in (
        (NIKON, "nikon-z-7-2", "FF_UM"),
        (SONY_7RM3, "sony-ilce-7rm3", "FF_BM"),
    ):
        route = rc.select_raw_route(identity)
        assert route.route == rc.METADATA_ROUTE and route.renderer is None
        assert route.decoder is rc.decode_sensor
        assert route.camera.key == key and route.camera.series == (series,)


def test_unknown_or_unidentified_cameras_are_refused(tmp_path, fake_rawpy):
    calls, _ = fake_rawpy
    with pytest.raises(rc.UnsupportedRawCamera, match="No native RAW route"):
        rc.select_raw_route(rc.CameraIdentity("Canon", "Canon EOS R5"))
    with pytest.raises(rc.UnsupportedRawCamera):
        rc.decode_sensor(
            write_raw(tmp_path),
            parameters={"edge_aware": EDGE_AWARE},
            camera_identity=rc.CameraIdentity("SONY", "ILCE-7RM3A"),
        )
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, np.zeros((4, 4), np.uint16))
    with pytest.raises(rc.UnsupportedRawCamera, match="Make/Model"):
        rc.read_camera_identity(plain)
    assert calls == []


def test_camera_identity_is_read_from_the_tiff_container(tmp_path):
    path = tmp_path / "tagged.tif"
    tifffile.imwrite(
        path,
        np.zeros((4, 4), np.uint16),
        extratags=[
            (271, "s", 0, "NIKON CORPORATION", True),
            (272, "s", 0, "NIKON Z 7_2", True),
        ],
    )
    identity = rc.read_camera_identity(path)
    assert (identity.make, identity.model) == ("NIKON CORPORATION", "NIKON Z 7_2")
    assert rc.select_raw_route(path).camera.key == "nikon-z-7-2"


def test_zv1_files_are_refused_by_the_new_decoder_before_any_read(tmp_path, fake_rawpy):
    calls, _ = fake_rawpy
    missing = (
        tmp_path / "DSC00001.ARW"
    )  # never opened: refusal precedes hashing and decoding
    with pytest.raises(rc.FrozenRouteRequired, match="frozen native route"):
        rc.decode_sensor(
            missing, parameters={"edge_aware": EDGE_AWARE}, camera_identity=ZV1
        )
    assert calls == []


# --- photometric contract --------------------------------------------------------


@pytest.fixture
def frozen_route(tmp_path, monkeypatch):
    """The original edge-aware renderer on a synthetic source with identity geometry."""
    parameters = {
        "common": {
            "sensor": {
                "content_wh": [8, 6],
                "camera_wh": [8, 6],
                "raw_wh": [16, 12],
                "cfa_pattern": [[0, 1], [3, 2]],
                "color_description": "RGBG",
                "black_levels": [800] * 4,
                "black_level": 800,
                "white_maximum": 65535,
                "plane_stride": 2,
                "green_mean_weight": 0.5,
                "tile_rows": 4,
            },
            "lens": {
                "knots": np.linspace(0.0, 1.0, 11).tolist(),
                "distortion_denominator": 16384.0,
                "ca_denominator": float(2**21),
                "raw_offset_xy": [0, 0],
            },
            "profiles": {
                "fixture": {
                    "coefficients": dict(
                        cu=0.0, cv=0.0, tx=0.0, ty=0.0, R=10.0, amp=1.0
                    ),
                    "distortion": [0] * 11,
                }
            },
        },
        "edge_aware": EDGE_AWARE,
    }
    arw, camera, review = (
        tmp_path / "DSC00001.ARW",
        tmp_path / "DSC00001.JPG",
        tmp_path / "review.jpg",
    )
    arw.write_bytes(b"synthetic sensor")
    for path in (camera, review):
        Image.fromarray(np.full((6, 8, 3), 100, np.uint8)).save(path)
    monkeypatch.setattr(
        reg, "review_to_camera", lambda *a, **k: (np.eye(3), {"fixture": True})
    )
    monkeypatch.setattr(reg, "read_lens_tables", lambda *a, **k: ((0,) * 11,) * 3)
    monkeypatch.setattr(reg, "raw_review_residual", lambda *a, **k: {"fixture": True})
    hashes = {
        k: sha256(p.read_bytes()).hexdigest()
        for k, p in [
            ("arw_sha256", arw),
            ("camera_jpeg_sha256", camera),
            ("review_native_jpeg_sha256", review),
        ]
    }
    return parameters, (arw, camera, review), hashes


def test_new_decoder_reproduces_the_frozen_route_arithmetic(frozen_route, fake_rawpy):
    parameters, (arw, camera, review), hashes = frozen_route
    _, state = fake_rawpy
    values = mosaic(low=0, high=16383)
    values[3, 5] = values[7, 9] = 16383  # the only saturated sites
    values[0, 0] = 300  # below black: clamped to zero
    state["raw"] = FakeRaw(values, black=(800,) * 4, white=16383)
    frozen, frozen_cert = reg.registered_edge_aware_sensor_linear(
        arw,
        camera,
        review,
        expected_source_hashes=hashes,
        content_box=(0, 0, 8, 6),
        parameters=parameters,
        source_id="fixture",
        profile_id="fixture",
    )
    decoded = rc.decode_sensor(arw, parameters=parameters, camera_identity=SONY_7RM3)
    assert np.array_equal(decoded.linear_dn()[:, :6, :8], frozen)
    assert decoded.mosaic_dn[0, 0] == 0 and decoded.black_level == 800
    cert = decoded.certificate()
    assert cert["sensor_saturated_count"] == frozen_cert["sensor_saturated_count"] == 2
    assert cert["black_level_subtracted"] == 800
    assert cert["white_level"] == frozen_cert["white_level"]
    for key in ("white_balance", "camera_to_xyz"):
        assert cert[key]["applied"] is frozen_cert[key]["applied"] is False
    assert (
        cert["gamma_tone_auto_bright"]
        == frozen_cert["gamma_tone_auto_bright"]
        == "none"
    )


def test_camera_constants_come_from_metadata(tmp_path, fake_rawpy):
    calls, state = fake_rawpy
    values = mosaic(seed=4, low=900, high=16000)
    state["raw"] = FakeRaw(values, black=(1008,) * 4, white=16383)
    path = write_raw(tmp_path)
    decoded = rc.decode_sensor(
        path, parameters={"edge_aware": EDGE_AWARE}, camera_identity=NIKON
    )
    expected = np.maximum(values.astype(np.int32) - 1008, 0).astype(np.uint16)
    assert np.array_equal(decoded.mosaic_dn, expected)
    meta = decoded.metadata
    assert meta.raw_visible_wh == (16, 12) and meta.cfa_pattern == rc.RGGB_PATTERN
    assert meta.black_level_per_channel == (1008,) * 4 and meta.white_level == 16383
    assert meta.libraw_crop_xywh == (2, 2, 8, 6) and meta.libraw_flip == 0
    assert decoded.white_range_dn == 16383 - 1008
    np.testing.assert_array_equal(
        decoded.linear_normalised(), decoded.linear_dn() / np.float32(16383 - 1008)
    )
    plane, phase = decoded.cfa_plane("G")
    np.testing.assert_array_equal(
        plane,
        0.5
        * (
            expected[0::2, 1::2].astype(np.float32)
            + expected[1::2, 0::2].astype(np.float32)
        ),
    )
    assert phase == (0.5, 0.5)
    assert decoded.cfa_plane("R")[1] == (0.0, 0.0) and decoded.cfa_plane("B")[1] == (
        1.0,
        1.0,
    )
    cert = decoded.certificate()
    assert cert["source"] == {
        "path": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }
    assert cert["normalisation"]["applied"] is False
    assert cert["sensor_constants_source"] == "file metadata (LibRaw)"
    assert calls == [str(path)]


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(pattern=((1, 0), (2, 3))), "RGGB"),
        (dict(desc=b"RGBE"), "RGGB"),
        (dict(black=(512, 512, 513, 512)), "one black level"),
        (dict(white=900), "black/white"),
        (dict(black=(0,) * 4, white=16384), "14-bit headroom"),
    ],
)
def test_decoder_refuses_what_the_contract_does_not_cover(
    tmp_path, fake_rawpy, change, message
):
    _, state = fake_rawpy
    state["raw"] = FakeRaw(mosaic(), **change)
    with pytest.raises(ValueError, match=message):
        rc.decode_sensor(
            write_raw(tmp_path),
            parameters={"edge_aware": EDGE_AWARE},
            camera_identity=NIKON,
        )


def test_demosaic_order_control_is_the_frozen_one(tmp_path, fake_rawpy):
    calls, state = fake_rawpy
    state["raw"] = FakeRaw(mosaic())
    swapped = dict(EDGE_AWARE, order_expected_rgb=[1000, 4000, 10000])
    with pytest.raises(ValueError, match="output order"):
        rc.decode_sensor(
            write_raw(tmp_path),
            parameters={"edge_aware": swapped},
            camera_identity=NIKON,
        )
    assert calls == []


def test_source_hash_and_frozen_sensor_profile_are_enforced(tmp_path, fake_rawpy):
    calls, state = fake_rawpy
    state["raw"] = FakeRaw(mosaic(low=1000, high=16384))
    path = write_raw(tmp_path)
    with pytest.raises(ValueError, match="hash"):
        rc.decode_sensor(
            path,
            parameters={"edge_aware": EDGE_AWARE},
            camera_identity=NIKON,
            expected_sha256="0" * 64,
        )
    assert calls == []
    decoded = rc.decode_sensor(
        path,
        parameters={"edge_aware": EDGE_AWARE},
        camera_identity=NIKON,
        expected_sha256=sha256(path.read_bytes()).hexdigest(),
    )
    frozen = rc.SensorProfile.from_metadata(
        decoded.camera, decoded.metadata, saturation_dn=15000
    )
    checked = rc.decode_sensor(
        path,
        parameters={"edge_aware": EDGE_AWARE},
        camera_identity=NIKON,
        sensor_profile=frozen,
    )
    cert = checked.certificate()
    assert cert["sensor_profile_digest"] is not None
    assert cert["at_or_above_profile_saturation_count"] == int(
        (state["raw"].raw_image_visible >= 15000).sum()
    )
    with pytest.raises(ValueError, match="white_level differs"):
        rc.decode_sensor(
            path,
            parameters={"edge_aware": EDGE_AWARE},
            camera_identity=NIKON,
            sensor_profile=replace(frozen, white_level=16380),
        )
    with pytest.raises(ValueError, match="Sensor profile is for"):
        rc.decode_sensor(
            path,
            parameters={"edge_aware": EDGE_AWARE},
            camera_identity=NIKON,
            sensor_profile=replace(frozen, camera_key="sony-ilce-7rm3"),
        )


# --- geometry profiles ---------------------------------------------------------------


def synthetic_decoded(identity=NIKON, height=12, width=16, black=1008, seed=5):
    camera = rc.lookup_camera(identity)
    values = mosaic(height, width, seed=seed, low=black, high=16000)
    dn = (values - black).astype(np.uint16)
    demosaiced = (
        np.random.default_rng(seed + 1)
        .integers(0, 15000, (height, width, 3))
        .astype(np.uint16)
    )
    meta = rc.RawSensorMetadata(
        make=identity.make,
        model=identity.model,
        raw_visible_wh=(width, height),
        cfa_pattern=rc.RGGB_PATTERN,
        color_description="RGBG",
        black_level_per_channel=(black,) * 4,
        white_level=16383,
        camera_white_level_per_channel=(15000,) * 4,
        camera_whitebalance=(2.0, 1.0, 1.5, 1.0),
        rgb_xyz_matrix=((1.0, 0.0, 0.0),) * 3,
        libraw_flip=0,
        exif_orientation=1,
        libraw_crop_xywh=(2, 2, 8, 6),
        libraw_version="0.22.1",
    )
    return rc.DecodedSensor(
        camera=camera,
        metadata=meta,
        source={"path": "/data/DSC00001.ARW", "sha256": "0" * 64},
        black_level=black,
        mosaic_dn=dn,
        demosaiced_dn=demosaiced,
        saturated=np.zeros((height, width), bool),
        demosaic_code="COLOR_BayerRG2BGR_EA",
    )


def crop_only(decoded, offset=(2, 2), wh=(8, 6), **changes):
    width, height = wh
    profile = rc.GeometryProfile(
        camera_key=decoded.camera.key,
        jpeg_wh=wh,
        raw_visible_wh=decoded.metadata.raw_visible_wh,
        raw_offset_xy=offset,
        centre_uv=((width - 1) / 2, (height - 1) / 2),
        radius_px=math.hypot(width / 2, height / 2),
        knots=(0.0, 1.0),
        distortion=(0.0, 0.0),
        amplitude=0.0,
        distortion_denominator=1.0,
        ca=None,
        ca_denominator=1.0,
    )
    return replace(profile, **changes)


def passed(profile, channels="RGB"):
    stats = dict(p95_px=0.1, mean_dx_px=0.0, mean_dy_px=0.0)
    evidence = {
        "profile_digest": profile.digest,
        "channels": {c: stats for c in channels},
    }
    return profile.with_validation(
        evidence, channels=tuple(channels), tolerance_p95_px=1.0, tolerance_mean_px=0.1
    )


def test_profile_coordinates_are_the_frozen_rule():
    u, v = np.mgrid[0:5, 0:7].astype(np.float32)
    profile = rc.GeometryProfile(
        camera_key="sony-ilce-7rm3",
        jpeg_wh=(7, 5),
        raw_visible_wh=(20, 20),
        raw_offset_xy=(8.0, 4.0),
        centre_uv=(3.0, 2.0),
        radius_px=4.0,
        knots=(0.0, 0.5, 1.0),
        distortion=(0.0, -100.0, -300.0),
        amplitude=1.0,
        distortion_denominator=2.0**14,
        ca={"R": (0.0, 50.0, 90.0), "B": (0.0, -40.0, -80.0)},
        ca_denominator=2.0**21,
        shift_xy=(0.25, -0.5),
    )
    for channel in "RGB":
        x, y = profile.coordinates(u, v, channel)
        ex, ey = reg.raw_coordinates(
            u,
            v,
            np.eye(3),
            profile.coefficients(),
            profile.distortion,
            None if channel == "G" else profile.ca[channel],
            parameters=profile.lens_parameters(),
        )
        assert np.array_equal(x, ex) and np.array_equal(y, ey)
    x, y = crop_only(synthetic_decoded()).coordinates(u, v, "R")
    assert np.array_equal(x, u + 2.0) and np.array_equal(y, v + 2.0)


def test_profile_documents_round_trip_through_a_bound_file(tmp_path):
    decoded = synthetic_decoded()
    geometry = passed(crop_only(decoded, provenance={"derivation": "test"}))
    sensor = rc.SensorProfile.from_metadata(
        decoded.camera, decoded.metadata, saturation_dn=15000
    )
    document = rc.CameraProfile(decoded.camera.key, sensor, geometry).to_document()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    digest = sha256(path.read_bytes()).hexdigest()
    loaded = rc.load_camera_profile(path, digest)
    assert loaded.geometry == geometry and loaded.sensor == sensor
    assert loaded.geometry.digest == geometry.digest and loaded.geometry.covers("B")
    with pytest.raises(ValueError, match="changed"):
        rc.load_camera_profile(path, "0" * 64)
    with pytest.raises(ValueError, match="format"):
        rc.CameraProfile.from_document(dict(document, format="other"))
    with pytest.raises(rc.UnsupportedRawCamera):
        rc.CameraProfile.from_document(dict(document, camera_key="canon"))
    broken = json.loads(json.dumps(document))
    broken["geometry"]["distortion"] = [0.0]
    with pytest.raises(ValueError, match="knots"):
        rc.CameraProfile.from_document(broken)


def test_rendering_refuses_missing_unvalidated_or_uncovered_profiles():
    decoded = synthetic_decoded()
    with pytest.raises(rc.GeometryProfileUnavailable, match="never guessed"):
        rc.render_on_jpeg_grid(decoded, None)
    profile = crop_only(decoded)
    with pytest.raises(rc.GeometryProfileUnavailable, match="no passed validation"):
        rc.render_on_jpeg_grid(decoded, profile)
    output, cert = rc.render_on_jpeg_grid(decoded, profile, allow_unvalidated=True)
    assert (
        cert["geometry_profile_validated"] is False
        and cert["geometry_validation"] is None
    )
    green_only = passed(profile, channels="G")
    with pytest.raises(rc.GeometryProfileUnavailable, match="R, B"):
        rc.render_on_jpeg_grid(decoded, green_only)
    green, cert = rc.render_on_jpeg_grid(decoded, green_only, channels="G")
    assert cert["geometry_profile_validated"] is True
    np.testing.assert_array_equal(green[0], output[1])
    with pytest.raises(ValueError, match="Geometry profile is for"):
        rc.render_on_jpeg_grid(synthetic_decoded(SONY_7RM3), passed(profile))
    with pytest.raises(ValueError, match="RAW grid"):
        rc.render_on_jpeg_grid(
            decoded, passed(replace(profile, raw_visible_wh=(18, 12)))
        )


def test_rendering_samples_the_demosaic_on_the_profile_grid():
    decoded = synthetic_decoded()
    profile = passed(crop_only(decoded, offset=(2, 2), wh=(8, 6)))
    output, cert = rc.render_on_jpeg_grid(decoded, profile, tile_rows=4)
    expected = np.moveaxis(decoded.demosaiced_dn[2:8, 2:10], -1, 0).astype(np.float32)
    assert output.dtype == np.float32 and output.shape == (3, 6, 8)
    np.testing.assert_array_equal(output, expected)
    assert (
        cert["nonfinite_fraction"] == 0
        and cert["decoding"]["route"] == rc.METADATA_ROUTE
    )
    window, _ = rc.render_on_jpeg_grid(decoded, profile, window_xywh=(3, 1, 4, 2))
    np.testing.assert_array_equal(window, expected[:, 1:3, 3:7])
    outside = passed(crop_only(decoded, offset=(10, 8), wh=(8, 6)))
    clipped, cert = rc.render_on_jpeg_grid(decoded, outside)
    # RAW x = u + 10 and y = v + 8 on a 16 x 12 sensor: NaN beyond it.
    assert np.isnan(clipped[:, 4:, :]).all() and np.isnan(clipped[:, :, 6:]).all()
    np.testing.assert_array_equal(
        clipped[:, :3, :5],
        np.moveaxis(decoded.demosaiced_dn[8:11, 10:15], -1, 0).astype(np.float32),
    )
    assert 0 < cert["nonfinite_fraction"] < 1
    with pytest.raises(ValueError, match="window"):
        rc.render_on_jpeg_grid(decoded, profile, window_xywh=(6, 0, 4, 2))


def test_validation_evidence_must_belong_to_the_profile_and_pass():
    profile = crop_only(synthetic_decoded())
    good = {
        "profile_digest": profile.digest,
        "channels": {
            "G": dict(p95_px=0.5, mean_dx_px=0.02, mean_dy_px=0.0),
            "R": dict(p95_px=1.7, mean_dx_px=0.1, mean_dy_px=0.1),
        },
    }
    assert profile.with_validation(
        good, channels=("G",), tolerance_p95_px=1.0, tolerance_mean_px=0.1
    ).covers("G")
    with pytest.raises(rc.GeometryProfileUnavailable, match="R: p95 1.7"):
        profile.with_validation(
            good, channels=("G", "R"), tolerance_p95_px=1.0, tolerance_mean_px=0.25
        )
    with pytest.raises(rc.GeometryProfileUnavailable, match="B: not measured"):
        profile.with_validation(
            good, channels=("B",), tolerance_p95_px=1.0, tolerance_mean_px=0.1
        )
    with pytest.raises(ValueError, match="another geometry profile"):
        replace(profile, raw_offset_xy=(4.0, 2.0)).with_validation(
            good, channels=("G",), tolerance_p95_px=1.0, tolerance_mean_px=0.1
        )


def makernote(settings, header=b""):
    entries = b"".join(
        struct.pack("<HHII", tag, 4, 1, value) for tag, value in settings.items()
    )
    return header + struct.pack("<H", len(settings)) + entries


@pytest.fixture
def sony_tags(monkeypatch):
    class Tags(dict):
        def get(self, code, default=None):
            return super().get(code, default)

    tag = lambda value: SimpleNamespace(value=value)  # noqa: E731
    state = {"settings": {0x2011: 2, 0x2012: 2, 0x2013: 0}, "header": b""}

    class Tiff:
        byteorder = "<"

        def __init__(self, _path):
            sub = SimpleNamespace(
                tags=Tags(
                    {
                        0x7037: tag((3, 0, -10, -20)),
                        0x7035: tag((6, 0, 30, 60, 0, -30, -60)),
                        50719: tag((2, 2)),
                        50720: tag((12, 8)),
                        0x74C7: tag((0, 2)),
                        0x74C8: tag((12, 6)),
                    }
                ),
                pages=None,
            )
            note = makernote(state["settings"], state["header"])
            self.pages = [
                SimpleNamespace(
                    tags=Tags({34665: tag({"MakerNote": note})}), pages=[sub]
                )
            ]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(rc.tifffile, "TiffFile", Tiff)
    return state


def test_sony_profile_reads_tables_and_applies_only_switched_on_corrections(sony_tags):
    decoded = synthetic_decoded(SONY_7RM3, height=12, width=16, black=512)
    profile = rc.derive_embedded_profile(decoded, jpeg_wh=(12, 6), raw_offset_xy=(2, 4))
    assert profile.validation is None and not profile.covers("G")
    assert profile.knots == (0.0, 0.5, 1.0) and profile.distortion == (
        0.0,
        -10.0,
        -20.0,
    )
    assert profile.amplitude == 0.0  # distortion correction Off in camera
    assert profile.ca == {"R": (0.0, 30.0, 60.0), "B": (0.0, -30.0, -60.0)}
    assert profile.radius_px == pytest.approx(math.hypot(6, 4))
    assert profile.centre_uv == (2 + 5.5 - 2, 2 + 3.5 - 4)
    assert (
        profile.distortion_denominator == 2.0**14 and profile.ca_denominator == 2.0**21
    )
    candidates = profile.provenance["crop_origin_candidates_from_metadata"]
    assert candidates["default_crop_x_sony_crop_y"] == [2, 2]
    assert candidates["libraw_crop_origin"] == [2, 2]
    sony_tags["settings"] = {0x2011: 0, 0x2012: 0, 0x2013: 2}
    sony_tags["header"] = b"SONY DSC \x00\x00\x00"
    switched = rc.derive_embedded_profile(
        decoded, jpeg_wh=(12, 6), raw_offset_xy=(2, 4)
    )
    assert switched.amplitude == 1.0 and switched.ca is None
    sony_tags["settings"] = {0x2012: 2}
    with pytest.raises(
        rc.GeometryProfileUnavailable, match="distortion correction setting is None"
    ):
        rc.derive_embedded_profile(decoded, jpeg_wh=(12, 6), raw_offset_xy=(2, 4))
    sony_tags["settings"] = {0x2011: 2, 0x2012: 2, 0x2013: 0xFFFFFFFF}
    with pytest.raises(rc.GeometryProfileUnavailable, match="'n/a'"):
        rc.derive_embedded_profile(decoded, jpeg_wh=(12, 6), raw_offset_xy=(2, 4))
    with pytest.raises(rc.GeometryProfileUnavailable, match="differs from the crop"):
        rc.derive_embedded_profile(decoded, jpeg_wh=(12, 8), raw_offset_xy=(2, 4))


def test_nikon_has_no_embedded_profile():
    with pytest.raises(
        rc.GeometryProfileUnavailable, match="explicit, validated geometry profile"
    ):
        rc.derive_embedded_profile(
            synthetic_decoded(NIKON), jpeg_wh=(8, 6), raw_offset_xy=(2, 2)
        )


# --- real FF_UM / FF_BM samples --------------------------------------------------------

NEF = Path(
    os.environ.get(
        "DARSIA_TEST_NEF",
        "/mnt/user-data/uploads/Experiments/Run6_220912/"
        "220927_CO2_injection/DCS01054.NEF",
    )
)
ARW = Path(
    os.environ.get(
        "DARSIA_TEST_ARW",
        "/mnt/user-data/uploads/Original data/FF_BM_cycle_1/"
        "C1_images/211124_c1_injection_20s/DSC00001.ARW",
    )
)


def companion(path):
    for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg"):
        if path.with_suffix(suffix).is_file():
            return path.with_suffix(suffix)
    pytest.skip("Companion JPEG missing for " + str(path))


def decode_sample(path):
    if not path.is_file():
        pytest.skip("RAW sample not available: " + str(path))
    pytest.importorskip("rawpy")
    route = rc.select_raw_route(path)
    assert route.route == rc.METADATA_ROUTE
    return route.decoder(path, parameters={"edge_aware": EDGE_AWARE})


@pytest.fixture(scope="module")
def arw():
    return decode_sample(ARW)


@pytest.fixture(scope="module")
def nef():
    return decode_sample(NEF)


def worst_radial(metrics):
    return max(
        abs(b["median_px"])
        for b in metrics["radial_median_px"]
        if b["median_px"] is not None
    )


def test_real_arw_constants_come_from_its_own_metadata(arw):
    meta = arw.metadata
    assert arw.camera.key == "sony-ilce-7rm3"
    assert (
        arw.demosaiced_dn.shape == (5320, 7968, 3)
        and arw.demosaiced_dn.dtype == np.uint16
    )
    assert meta.cfa_pattern == rc.RGGB_PATTERN and meta.color_description == "RGBG"
    with tifffile.TiffFile(ARW) as tiff:
        sub = tiff.pages[0].pages[0]
        tiff_black = tuple(
            int(v) for v in sub.tags.get(0x7310).value
        )  # Sony BlackLevel
        tiff_white = int(sub.tags.get(50717).value)  # WhiteLevel
        tiff_cfa = bytes(sub.tags.get(33422).value)  # CFAPattern
    assert meta.black_level_per_channel == tiff_black == (512,) * 4
    assert meta.white_level == tiff_white == 16383
    assert tiff_cfa == b"\x00\x01\x01\x02"  # R G / G B
    assert meta.camera_white_level_per_channel == (15360,) * 4
    assert meta.libraw_crop_xywh == (0, 422, 7952, 4472) and meta.libraw_flip == 0
    normalised = arw.linear_normalised()
    assert (
        np.isfinite(normalised).all()
        and 0 <= normalised.min()
        and normalised.max() <= 1
    )
    lens = rc.read_sony_lens_metadata(ARW)
    assert lens["distortion_header"] == 16 and lens["ca_header"] == 32
    assert lens["settings"] == {
        "vignetting_correction": "Auto",
        "lateral_chromatic_aberration": "Auto",
        "distortion_correction": "Off",
    }
    assert lens["default_crop_origin"] == (8, 8) and lens["default_crop_size"] == (
        7952,
        5304,
    )
    assert lens["sony_crop_top_left"] == (0, 422) and lens["sony_crop_size"] == (
        7952,
        4472,
    )


def test_real_arw_derived_profile_reproduces_the_companion_jpeg(arw):
    jpeg = companion(ARW)
    profile = rc.derive_embedded_profile(
        arw, jpeg_wh=(7952, 4472), raw_offset_xy=(8, 422)
    )
    assert (
        profile.amplitude == 0.0
        and set(profile.ca) == {"R", "B"}
        and len(profile.knots) == 16
    )
    agreement = rc.measure_geometry_agreement(
        arw, jpeg, profile, parameters=AGREEMENT, channels=("G", "R", "B")
    )
    g, r, b = (agreement["channels"][c] for c in "GRB")
    assert g["n"] > 3000 and g["median_match_correlation"] > 0.95
    assert g["p95_px"] <= 1.0 and r["p95_px"] <= 1.0 and b["p95_px"] <= 1.5
    for metrics in (g, r, b):
        assert math.hypot(metrics["mean_dx_px"], metrics["mean_dy_px"]) <= 0.1
        assert worst_radial(metrics) <= 0.1
    validated = profile.with_validation(
        agreement, channels=("G", "R", "B"), tolerance_p95_px=1.5, tolerance_mean_px=0.1
    )
    document = json.loads(
        json.dumps(rc.CameraProfile(arw.camera.key, None, validated).to_document())
    )
    reloaded = rc.CameraProfile.from_document(document).geometry
    assert (
        reloaded == validated
        and reloaded.digest == profile.digest
        and reloaded.covers("B")
    )
    output, cert = rc.render_on_jpeg_grid(
        arw, validated, window_xywh=(3900, 2200, 64, 32)
    )
    assert output.shape == (3, 32, 64) and np.isfinite(output).all()
    assert cert["geometry_profile_validated"] is True
    # Without the switched-on CA table, R and B keep a radial offset.
    no_ca = rc.measure_geometry_agreement(
        arw, jpeg, replace(profile, ca=None), parameters=AGREEMENT, channels=("R", "B")
    )
    assert (
        worst_radial(no_ca["channels"]["R"]) > 0.3
        and worst_radial(no_ca["channels"]["B"]) > 0.3
    )


def test_real_arw_distortion_table_must_follow_the_camera_switch(arw):
    jpeg = companion(ARW)
    profile = rc.derive_embedded_profile(
        arw, jpeg_wh=(7952, 4472), raw_offset_xy=(8, 422)
    )
    applied = rc.measure_geometry_agreement(
        arw, jpeg, replace(profile, amplitude=1.0), parameters=AGREEMENT
    )
    assert (
        applied["channels"]["G"]["p95_px"] > 5
        and worst_radial(applied["channels"]["G"]) > 3
    )


def test_real_arw_only_one_metadata_crop_candidate_matches(arw):
    jpeg = companion(ARW)
    profile = rc.derive_embedded_profile(
        arw, jpeg_wh=(7952, 4472), raw_offset_xy=(8, 422)
    )
    sparse = dict(AGREEMENT, grid_step_px=192, minimum_matches=100)
    shifts = {}
    for name, origin in profile.provenance[
        "crop_origin_candidates_from_metadata"
    ].items():
        candidate = rc.derive_embedded_profile(
            arw, jpeg_wh=(7952, 4472), raw_offset_xy=origin
        )
        try:
            g = rc.measure_geometry_agreement(arw, jpeg, candidate, parameters=sparse)[
                "channels"
            ]["G"]
        except ValueError as error:  # far outside the search window: no texture matches
            assert "Insufficient" in str(error)
            shifts[name] = math.inf
            continue
        shifts[name] = math.hypot(g["mean_dx_px"], g["mean_dy_px"])
    assert profile.provenance["crop_origin_candidates_from_metadata"][
        "default_crop_x_sony_crop_y"
    ] == [8, 422]
    assert shifts.pop("default_crop_x_sony_crop_y") < 0.1
    assert min(shifts.values()) > 5


def test_real_nef_constants_and_the_geometry_gap(nef):
    jpeg = companion(NEF)
    meta = nef.metadata
    assert nef.camera.key == "nikon-z-7-2"
    assert nef.demosaiced_dn.shape == (5520, 8288, 3)
    assert meta.cfa_pattern == rc.RGGB_PATTERN and meta.color_description == "RGBG"
    assert meta.black_level_per_channel == (1008,) * 4 and meta.white_level == 16383
    assert meta.camera_white_level_per_channel == (15311,) * 4
    with Image.open(jpeg) as image:
        jpeg_wh = image.size
    assert meta.libraw_crop_xywh == (16, 8, *jpeg_wh) == (16, 8, 8256, 5504)
    assert np.isfinite(nef.linear_normalised()[:, ::16, ::16]).all()
    with pytest.raises(rc.GeometryProfileUnavailable):
        rc.derive_embedded_profile(nef, jpeg_wh=jpeg_wh, raw_offset_xy=(16, 8))
    # An explicit crop-only candidate from LibRaw's crop: G matches, R/B do not.
    candidate = crop_only(nef, offset=(16, 8), wh=jpeg_wh)
    with pytest.raises(rc.GeometryProfileUnavailable):
        rc.render_on_jpeg_grid(nef, candidate)
    agreement = rc.measure_geometry_agreement(
        nef, jpeg, candidate, parameters=AGREEMENT, channels=("G", "R", "B")
    )
    g, r = agreement["channels"]["G"], agreement["channels"]["R"]
    assert g["p95_px"] <= 1.0 and math.hypot(g["mean_dx_px"], g["mean_dy_px"]) <= 0.1
    assert worst_radial(g) <= 0.1
    assert (
        r["radial_median_px"][-1]["median_px"] > 0.5
    )  # in-camera lateral CA, no table in the NEF
    with pytest.raises(rc.GeometryProfileUnavailable, match="R: p95"):
        candidate.with_validation(
            agreement,
            channels=("G", "R", "B"),
            tolerance_p95_px=1.0,
            tolerance_mean_px=0.25,
        )
    green = candidate.with_validation(
        agreement, channels=("G",), tolerance_p95_px=1.0, tolerance_mean_px=0.1
    )
    with pytest.raises(rc.GeometryProfileUnavailable, match="R, B"):
        rc.render_on_jpeg_grid(nef, green, window_xywh=(4000, 2700, 32, 16))
    output, _ = rc.render_on_jpeg_grid(
        nef, green, channels="G", window_xywh=(4000, 2700, 32, 16)
    )
    assert np.isfinite(output).all()
