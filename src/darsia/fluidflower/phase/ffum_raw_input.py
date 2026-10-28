"""Full sensor FF_UM input with the measured relative camera-JPG profile.

This is a relative red-channel alignment, not an optical lens calibration.
The green coordinate anchor and the original full sensor field are preserved.
The same bound reader is used by UM training and inference.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..native import raw_camera


EDGE_AWARE = {
    "demosaic_code": "COLOR_BayerRG2BGR_EA",
    "order_fixture": [[10000, 4000] * 3, [4000, 1000] * 3] * 3,
    "order_check_xy": [2, 2], "order_expected_rgb": [10000, 4000, 1000],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_json(path, expected_sha256):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"UM input binding changed: {path}")
    return json.loads(payload)


def validate_um_camera_identity(identity):
    camera = raw_camera.lookup_camera(identity)
    if camera.key != "nikon-z-7-2" or camera.series != ("FF_UM",):
        raise ValueError("UM relative profile requires the bound Nikon Z7II camera")
    if identity.orientation not in (None, 1):
        raise ValueError("UM full sensor mapping requires the recorded unrotated orientation")
    return camera


@dataclass(frozen=True)
class UmRelativeMapping:
    profile: dict
    origin_jpg_xy: tuple[int, int] = (-19, -11)
    shape_hw: tuple[int, int] = (5526, 8294)
    sensor_wh: tuple[int, int] = (8288, 5520)

    def __post_init__(self):
        if (self.profile.get("candidate") != "R_only"
                or self.profile.get("raw_offset_xy") != [16, 8]
                or self.profile.get("center_jpg_xy") != [4127.5, 2751.5]
                or self.profile.get("coefficients_px", {}).get("G") != [0.0, 0.0]
                or self.profile.get("coefficients_px", {}).get("B") != [0.0, 0.0]
                or not self.profile.get("green_preserved")
                or tuple(self.origin_jpg_xy) != (-19, -11)
                or tuple(self.shape_hw) != (5526, 8294)
                or tuple(self.sensor_wh) != (8288, 5520)):
            raise ValueError("UM relative profile/canvas differs from the validated Nikon setup")
        coefficients = np.asarray(self.profile["coefficients_px"]["R"], dtype=float)
        if (coefficients.shape != (2,) or not np.isfinite(coefficients).all()
                or not np.isfinite(self.profile["radius_px"]) or self.profile["radius_px"] <= 0):
            raise ValueError("Malformed UM red-channel profile")

    def displacement(self, jpg_xy, channel):
        points = np.asarray(jpg_xy, dtype=np.float64)
        vector = points - self.profile["center_jpg_xy"]
        radius = np.linalg.norm(vector, axis=-1)
        relative = radius / self.profile["radius_px"]
        a, b = self.profile["coefficients_px"][channel]
        return ((a * relative + b * relative ** 3) / np.maximum(radius, 1e-12))[..., None] * vector

    def display_to_sensor(self, display_xy, channel="G"):
        jpg = np.asarray(display_xy, dtype=np.float64) + self.origin_jpg_xy
        return jpg + self.profile["raw_offset_xy"] + self.displacement(jpg, channel)

    def sensor_to_display(self, sensor_xy, channel="G"):
        target = np.asarray(sensor_xy, dtype=np.float64) - self.profile["raw_offset_xy"]
        jpg = target.copy()
        for _ in range(8):
            jpg = target - self.displacement(jpg, channel)
        return jpg - self.origin_jpg_xy

    def validate_boundary(self):
        w, h = self.sensor_wh
        x, y = np.arange(w), np.arange(h)
        boundary = np.concatenate((np.column_stack((x, x * 0)), np.column_stack((x, x * 0 + h - 1)),
                                   np.column_stack((y * 0, y)), np.column_stack((y * 0 + w - 1, y))))
        mapped = np.concatenate([self.sensor_to_display(boundary, c) for c in "RGB"])
        lo, hi = mapped.min(0), mapped.max(0)
        if (not np.isfinite(mapped).all() or np.any(lo < 0)
                or np.any(hi > np.asarray(self.shape_hw[::-1]) - 1)):
            raise ValueError("UM canvas would cut the corrected full sensor boundary")
        return {"minimum_display_xy": lo.tolist(), "maximum_display_xy": hi.tolist(),
                "sensor_boundary_points_per_channel": len(boundary)}


@dataclass(frozen=True)
class UmRawFrame:
    rgb: np.ndarray
    valid_support: np.ndarray
    mapping: UmRelativeMapping
    metadata: dict


def read_um_full_frame(raw_path, *, raw_sha256, profile_path, profile_sha256,
                       photometry_path, photometry_sha256, chunk_rows=64):
    """Render one complete RAW field in the existing fixed camera table encoding.

    Output RGB is uint8, as for the approved table render. Features must explicitly
    divide sampled values by 255. Baseline/current need their own geometry maps.
    """
    # This exact fixed-table implementation is captured in the model runtime.
    # Current generic display modes are not a substitute for this contract.
    from ..native.camera_u_display import display_parameters, encode_display

    profile = bound_json(profile_path, profile_sha256)
    mapping = UmRelativeMapping(profile)
    boundary = mapping.validate_boundary()
    document = bound_json(photometry_path, photometry_sha256)
    parameters = display_parameters(document, "camera-u")
    if parameters.get("transfer") != "table":
        raise ValueError("UM requires its bound fixed period camera table")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("Positive integer chunk size required")
    identity = raw_camera.read_camera_identity(raw_path)
    validate_um_camera_identity(identity)
    decoded = raw_camera.decode_sensor(raw_path, parameters={"edge_aware": EDGE_AWARE},
        camera_identity=identity, expected_sha256=raw_sha256)
    if tuple(decoded.metadata.raw_visible_wh) != mapping.sensor_wh:
        raise ValueError("RAW dimensions differ from the bound UM sensor/canvas")
    source = decoded.linear_dn()
    if source.shape != (3, *mapping.sensor_wh[::-1]):
        raise ValueError("UM demosaiced RAW array differs from the bound full sensor")
    height, width = mapping.shape_hw
    rgb = np.empty((height, width, 3), np.uint8)
    support = np.ones((height, width), bool)
    for start in range(0, height, chunk_rows):
        stop = min(start + chunk_rows, height)
        y, x = np.mgrid[start:stop, :width]
        points = np.stack((x, y), axis=-1)
        values = np.empty((3, stop - start, width), np.float32)
        for ci, channel in enumerate("RGB"):
            sensor = mapping.display_to_sensor(points, channel)
            valid = ((sensor[..., 0] >= 0) & (sensor[..., 0] <= mapping.sensor_wh[0] - 1)
                     & (sensor[..., 1] >= 0) & (sensor[..., 1] <= mapping.sensor_wh[1] - 1))
            support[start:stop] &= valid
            values[ci] = cv2.remap(source[ci], sensor[..., 0].astype(np.float32),
                sensor[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rgb[start:stop] = encode_display(values, white_range_dn=float(decoded.white_range_dn),
                                         parameters=parameters)
        rgb[start:stop][~support[start:stop]] = 0
    metadata = dict(format="ffum-full-sensor-relative-jpg-input-v1", raw_path=str(raw_path),
        raw_sha256=raw_sha256, profile_path=str(profile_path), profile_sha256=profile_sha256,
        photometry_path=str(photometry_path), photometry_sha256=photometry_sha256,
        sensor_wh=list(mapping.sensor_wh), canvas_shape_hw=list(mapping.shape_hw),
        canvas_origin_jpg_xy=list(mapping.origin_jpg_xy), sensor_boundary=boundary,
        white_range_dn=float(decoded.white_range_dn), valid_pixels=int(support.sum()),
        camera_identity=dict(make=identity.make, model=identity.model, orientation=identity.orientation),
        color_encoding="fixed camera table uint8; feature unit normalization is /255",
        geometry="relative RAW-to-camera-JPG R-only alignment; not absolute optical calibration",
        unsupported="outside JPG crop and r/R>0.972902 lack measured reference support",
        crop_applied=False, rig_geometry_applied=False, automatic_photometry=False)
    return UmRawFrame(rgb, support, mapping, metadata)
