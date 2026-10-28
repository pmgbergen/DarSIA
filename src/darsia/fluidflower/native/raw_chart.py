"""RAW-only chart views with the existing source-only chart matcher."""

from pathlib import Path
from typing import Mapping

import numpy as np

from . import registration, raw_positioning


def gray_view(sensor: raw_positioning.Sensor, camera_rgb: np.ndarray) -> np.ndarray:
    """Fixed log-DN uint8 registration view; never a photometric input."""
    values = np.asarray(camera_rgb)
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("Chart positioning requires camera-U linear RGB")
    if not np.isfinite(values).all():
        raise ValueError("Chart positioning view has unsupported sensor pixels")
    scale = np.float32(sensor.metadata["white_level"] - sensor.metadata["black_level"])
    if scale <= 0:
        raise ValueError("Chart positioning has no positive RAW DN range")
    return np.rint(np.clip(
        np.log1p(np.maximum(values[1], 0)) * np.float32(255) / np.log1p(scale),
        0, 255)).astype(np.uint8)


def register(*, baseline_sensor: raw_positioning.Sensor,
             frame_sensor: raw_positioning.Sensor,
             baseline_camera_u: np.ndarray, frame_camera_u: np.ndarray,
             baseline_chart_xy8: np.ndarray, baseline_chart_log_rgb: np.ndarray,
             baseline_native_to_u_h: np.ndarray, frame_native_to_u_h: np.ndarray,
             frame_native_linear_rgb: np.ndarray, parameters: Mapping):
    """Reuse chart matching/saturation math without reading a JPEG or globals.

    The legacy helper's internal photograph argument names are adapted at this
    boundary. Its actual bound files are RAW and its ROI reader is explicit.
    """
    pairs = [(baseline_sensor, baseline_camera_u), (frame_sensor, frame_camera_u)]
    views = {}
    expected_wh = tuple(parameters["common"]["sensor"]["camera_wh"])
    for sensor, rgb in pairs:
        view = gray_view(sensor, rgb)
        if view.shape[::-1] != expected_wh:
            raise ValueError("RAW chart view differs from its calibrated U grid")
        views[str(Path(sensor.source["path"]).resolve()).casefold()] = view

    def reader(path, box, *, parameters):
        key = str(Path(path).resolve()).casefold()
        if key not in views:
            raise ValueError("Unbound RAW chart source")
        left, top, right, bottom = box
        return views[key][top:bottom, left:right].copy()

    xy, usable, old = registration.register_chart(
        baseline_camera_jpeg=baseline_sensor.source["path"],
        baseline_camera_sha256=baseline_sensor.source["sha256"],
        frame_camera_jpeg=frame_sensor.source["path"],
        frame_camera_sha256=frame_sensor.source["sha256"],
        baseline_chart_xy8=baseline_chart_xy8,
        baseline_chart_log_rgb=baseline_chart_log_rgb,
        baseline_review_to_camera_h=baseline_native_to_u_h,
        frame_review_to_camera_h=frame_native_to_u_h,
        frame_native_linear_rgb=frame_native_linear_rgb,
        parameters=parameters, _gray_reader=reader)
    certificate = {k: v for k, v in old.items() if k not in {
        "baseline_camera_jpeg", "baseline_camera_sha256",
        "frame_camera_jpeg", "frame_camera_sha256"}}
    certificate.update(
        contract="darsia-raw-source-chart-registration-v1",
        baseline_raw=dict(baseline_sensor.source),
        frame_raw=dict(frame_sensor.source),
        positioning_gray="log1p black-subtracted RAW G, fixed white-range encoding",
        photometric_input="unchanged camera-native DN", jpeg_inputs=False)
    return xy, usable, certificate
