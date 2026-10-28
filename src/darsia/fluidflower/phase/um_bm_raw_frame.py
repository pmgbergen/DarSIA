"""Uncropped, lens-corrected RAW inputs for FF_UM/FF_BM phase retraining.

The existing lens sampler and full-field coordinate map are reused unchanged.
Undefined border samples stay NaN and have an explicit mask; they must never
be supplied as black training or validation pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import cv2

from ..native import raw_positioning, registration
from ..native.io import read_bound_json, sha256_file
from ..native.raw_camera import SensorProfile, decode_sensor
from ..native.raw_full_display import (
    RawDisplayMapping,
    full_raw_display_extent,
    mapping_from_parameters,
)


_CAMERAS = {"FF_UM": "nikon-z-7-2", "FF_BM": "sony-ilce-7rm3"}


@dataclass(frozen=True)
class FullRawPhaseFrame:
    linear_rgb_chw: np.ndarray
    valid_support: np.ndarray
    mapping: RawDisplayMapping
    certificate: dict


def rectify_sensor(sensor: raw_positioning.Sensor, *, parameters: Mapping,
                   series: str, extent=None) -> FullRawPhaseFrame:
    """Sample the complete sensor field on the supplied corrected lens plane.

    ``extent`` may fix a common canvas across frames, but may not clip any
    sensor boundary. It is (left, top, width, height) in corrected U pixels.
    No rig crop, alignment, resize, photometric fit or model fit is performed.
    """
    policy = parameters["common"]["sensor"]
    if series not in _CAMERAS or policy["camera_key"] != _CAMERAS[series]:
        raise ValueError("This phase-transfer reader is restricted to FF_UM and FF_BM")
    mapping = mapping_from_parameters(parameters, sensor.profile_id)
    if (sensor.rgb.shape != (*mapping.sensor_wh[::-1], 3)
            or sensor.rgb.dtype != np.uint16
            or tuple(sensor.distortion) != mapping.distortion):
        raise ValueError("The decoded sensor differs from the bound lens/sensor profile")
    if extent is None:
        extent = full_raw_display_extent([mapping])
    if (len(extent) != 4 or any(not isinstance(v, (int, np.integer)) for v in extent)
            or extent[2] <= 0 or extent[3] <= 0):
        raise ValueError("The full-frame extent must have an integer origin and positive size")
    left, top, width, height = map(int, extent)
    if width * height > 4 * int(np.prod(mapping.sensor_wh)):
        raise ValueError("The requested canvas exceeds the full-sensor allocation bound")
    mapping = replace(mapping, origin_u_xy=(left, top))
    boundary = mapping.camera_u_to_display_xy(
        mapping.sensor_to_camera_u_xy(mapping.sensor_boundary_xy()))
    if (np.any(boundary < 0) or np.any(boundary[:, 0] >= width)
            or np.any(boundary[:, 1] >= height)):
        raise ValueError("The requested canvas crops the RAW sensor field")
    to_u = np.array([[1., 0., left], [0., 1., top], [0., 0., 1.]])
    rows = int(policy["tile_rows"])
    if rows <= 0:
        raise ValueError("A positive sampling tile height is required")
    linear = np.empty((3, height, width), np.float32)
    for channel, ca in enumerate((sensor.ca_red, None, sensor.ca_blue)):
        plane = sensor.rgb[..., channel].astype(np.float32)
        for y0 in range(0, height, rows):
            y1 = min(y0 + rows, height)
            yy, xx = np.mgrid[y0:y1, :width].astype(np.float32)
            rx, ry = registration.raw_coordinates(
                xx, yy, to_u, mapping.coefficients, sensor.distortion, ca,
                parameters=mapping.lens)
            valid = ((rx >= 0) & (rx <= mapping.sensor_wh[0] - 1)
                     & (ry >= 0) & (ry <= mapping.sensor_wh[1] - 1))
            # A NaN border would poison even exact last-row/column samples
            # through zero-weight interpolation terms. Mask coordinates instead.
            tile = cv2.remap(plane, rx.astype(np.float32), ry.astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            tile[~valid] = np.nan
            linear[channel, y0:y1] = tile
    support = np.isfinite(linear).all(axis=0)
    if not support.any():
        raise ValueError("The corrected frame has no common RGB sensor support")
    certificate = {
        "series": series,
        "source_raw": dict(sensor.source),
        "profile_id": sensor.profile_id,
        "sensor_wh": list(mapping.sensor_wh),
        "extent_u_xywh": [left, top, width, height],
        "lens_coefficients": dict(mapping.coefficients),
        "full_raw_sensor_field": True,
        "rig_crop_applied": False,
        "valid_rgb_pixels": int(support.sum()),
        "white_range_dn": sensor.metadata["white_level"] - sensor.metadata["black_level"],
        "units": "black-subtracted camera DN",
    }
    return FullRawPhaseFrame(linear, support, mapping, certificate)


def read_full_raw_phase_frame(path, *, series: str, profile_path,
                             profile_sha256: str, profile_id: str,
                             expected_sha256: str, extent=None) -> FullRawPhaseFrame:
    """Read a source-bound UM/BM RAW with an explicit, separately bound lens model.

    The lens coefficients are not inferred from a JPEG crop or camera switch.
    Preparing/validating the actual lens calibration is the caller's concern.
    """
    path = Path(path)
    if (len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("An exact lowercase RAW SHA-256 is required")
    document = read_bound_json(Path(profile_path), profile_sha256)
    parameters = document.get("registration", document)
    common = parameters["common"]
    policy = common["sensor"]
    if series not in _CAMERAS or policy["camera_key"] != _CAMERAS[series]:
        raise ValueError("This phase-transfer reader is restricted to FF_UM and FF_BM")
    profile = common["profiles"][profile_id]
    lens = common["lens"]
    if lens.get("tables", "file") == "file":
        distortion, red, blue = registration.read_lens_tables(path, parameters=lens)
    elif lens["tables"] == "profile":
        distortion = tuple(profile["distortion"])
        red = tuple(profile.get("ca_red", [0.] * len(distortion)))
        blue = tuple(profile.get("ca_blue", [0.] * len(distortion)))
    else:
        raise ValueError("Unknown source of lens tables")
    if tuple(distortion) != tuple(profile["distortion"]):
        raise ValueError("RAW lens table differs from the bound calibration")
    if (len(red) != len(distortion) or len(blue) != len(distortion)
            or not np.isfinite([*distortion, *red, *blue]).all()):
        raise ValueError("Lens channel tables must be finite and share the declared knots")
    before = raw_positioning._version(path)
    sensor_profile = SensorProfile(
        camera_key=policy["camera_key"], raw_visible_wh=tuple(policy["raw_wh"]),
        cfa_pattern=tuple(tuple(row) for row in policy["cfa_pattern"]),
        color_description=policy["color_description"],
        black_level_per_channel=tuple(policy["black_levels"]),
        white_level=policy["white_maximum"],
    )
    decoded = decode_sensor(path, parameters=parameters, sensor_profile=sensor_profile,
                            expected_sha256=expected_sha256)
    if ("libraw_crop_xywh" in policy
            and decoded.metadata.libraw_crop_xywh != tuple(policy["libraw_crop_xywh"])):
        raise ValueError("RAW crop metadata differs from its source binding")
    metadata = {"white_level": decoded.metadata.white_level, "black_level": decoded.black_level}
    sensor = raw_positioning.Sensor(decoded.demosaiced_dn, decoded.source, metadata,
                                    distortion, red, blue, profile_id)
    frame = rectify_sensor(sensor, parameters=parameters, series=series, extent=extent)
    if raw_positioning._version(path) != before or sha256_file(path) != expected_sha256:
        raise ValueError("RAW source changed during full-frame rendering")
    frame.certificate["profile_document"] = {
        "path": str(profile_path), "sha256": profile_sha256,
    }
    return frame
