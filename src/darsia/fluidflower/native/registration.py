"""Source-bound sensor registration with explicit calibration and policy data."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import tifffile
from PIL import Image

from .io import sha256_file


def read_lens_tables(arw_path: Path, *, parameters: Mapping) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    'Read and validate the configured sensor distortion and channel tables.'
    with tifffile.TiffFile(arw_path) as tiff:
        sub = tiff.pages[parameters["main_page"]].pages[parameters["sub_page"]]
        d = tuple(int(v) for v in sub.tags[parameters["distortion_tag"]].value)
        ca = tuple(int(v) for v in sub.tags[parameters["ca_tag"]].value)
    if d[0] != parameters["distortion_header"] or len(d) < parameters["distortion_min_length"] or ca[0] != parameters["ca_header"] or len(ca) < parameters["ca_min_length"]:
        raise ValueError("Unexpected sensor lens-table structure")
    red, blue = ca[slice(*parameters["red_slice"])], ca[slice(*parameters["blue_slice"])]
    if red != tuple(parameters["ca_red"]) or blue != tuple(parameters["ca_blue"]):
        raise ValueError("Sensor channel table differs from frozen calibration")
    return d[slice(*parameters["distortion_slice"])], red, blue


def bandpass(image: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    'Subtract two Gaussian scales while retaining the input arithmetic.'
    return cv2.GaussianBlur(image, (0, 0), parameters["sigma_small"]) - cv2.GaussianBlur(image, (0, 0), parameters["sigma_large"])


def match_rows(a: np.ndarray, b: np.ndarray, points,
                  *, template_half: int, search_half: int,
                  guess, min_correlation: float, parameters: Mapping) -> np.ndarray:
    'Match textured patches and retain only interior concave correlation peaks.'
    rows = []
    h, s = template_half, search_half
    for x, y in points:
        x, y = int(x), int(y)
        if x-h < 0 or y-h < 0 or x+h+1 > a.shape[1] or y+h+1 > a.shape[0]:
            continue
        qx, qy = guess(x, y)
        qx, qy = int(round(qx)), int(round(qy))
        if qx-h-s < 0 or qy-h-s < 0 or qx+h+s+1 > b.shape[1] or qy+h+s+1 > b.shape[0]:
            continue
        template = a[y-h:y+h+1, x-h:x+h+1]
        search = b[qy-h-s:qy+h+s+1, qx-h-s:qx+h+s+1]
        if float(template.std()) < parameters["minimum_template_std"]:
            continue
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (px, py) = cv2.minMaxLoc(response)
        if score < min_correlation or px in (0, response.shape[1]-1) or py in (0, response.shape[0]-1):
            continue
        cx = response[py, px-1] + response[py, px+1] - 2*score
        cy = response[py-1, px] + response[py+1, px] - 2*score
        if cx >= 0 or cy >= 0:
            continue
        dx = .5*(response[py, px-1]-response[py, px+1])/cx
        dy = .5*(response[py-1, px]-response[py+1, px])/cy
        rows.append((x, y, qx-s+px+dx, qy-s+py+dy, score))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def match_metrics(rows: np.ndarray, scale: float) -> dict:
    'Summarize displacement and correlation for accepted source matches.'
    if len(rows) == 0:
        return {"n": 0, "p50_px": None, "p95_px": None, "max_px": None}
    error = scale*np.linalg.norm(rows[:, 2:4]-rows[:, :2], axis=1)
    return {"n": len(error), "p50_px": float(np.median(error)),
            "p95_px": float(np.percentile(error, 95)),
            "max_px": float(error.max()),
            "median_match_correlation": float(np.median(rows[:, 4]))}


def review_to_camera(review_g: np.ndarray, camera_g: np.ndarray, *, parameters: Mapping) -> tuple[np.ndarray, dict]:
    'Fit a source-only homography and check orientation and held-out regions.'
    policy = parameters["review"]
    a = bandpass(np.log(review_g + 1.0), parameters=parameters["matching"])
    b = bandpass(np.log(camera_g + 1.0), parameters=parameters["matching"])
    # Fitting only on the middle leaves the chart and right-lower sand independent.
    grid = ((x, y) for y in range(*policy["fit_y_range"]) for x in range(*policy["fit_x_range"]))
    rows = match_rows(a, b, grid, template_half=policy["template_half"], search_half=policy["search_half"],
                         guess=lambda x, y: (x, y+policy["guess_offset_xy"][1]), min_correlation=policy["min_correlation"], parameters=parameters["matching"])
    if len(rows) < policy["minimum_matches"]:
        raise ValueError(f"Insufficient review-to-camera matches: {len(rows)}")
    h, inliers = cv2.findHomography(rows[:, :2].astype(np.float32),
                                   rows[:, 2:4].astype(np.float32), cv2.RANSAC, policy["ransac_threshold"])
    if h is None or inliers is None or inliers.mean() < policy["minimum_inlier_fraction"]:
        raise ValueError("Review-to-camera homography failed source checks")
    if not np.all(np.isfinite(h)):
        raise ValueError("Nonfinite homography")
    # Sampled local orientation must not reverse or fold the native content.
    pts = np.array([(x, y) for y in policy["orientation_y"]
                    for x in policy["orientation_x"]], dtype=np.float32)
    q = cv2.perspectiveTransform(pts[None], h)[0]
    qx = cv2.perspectiveTransform((pts + [1, 0])[None], h)[0]
    qy = cv2.perspectiveTransform((pts + [0, 1])[None], h)[0]
    jac = (qx[:, 0]-q[:, 0])*(qy[:, 1]-q[:, 1]) - (qx[:, 1]-q[:, 1])*(qy[:, 0]-q[:, 0])
    if np.any(jac <= 0):
        raise ValueError("Review-to-camera homography folds native content")

    def predicted(x, y):
        point = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), h)[0, 0]
        return float(point[0]), float(point[1])

    zones = {}
    for name, box, step in policy["heldout_zones"]:
        x0, y0, x1, y1 = box
        checks = match_rows(a, b,
                               ((x, y) for y in range(y0, y1, step)
                                for x in range(x0, x1, step)),
                               template_half=policy["heldout_template_half"], search_half=policy["heldout_search_half"],
                               guess=predicted, min_correlation=policy["heldout_min_correlation"], parameters=parameters["matching"])
        if len(checks) < policy["heldout_minimum_matches"]:
            raise ValueError(f"Insufficient held-out review-camera matches in {name}")
        pred = cv2.perspectiveTransform(checks[:, None, :2].astype(np.float32), h)[:, 0]
        residual = np.linalg.norm(pred-checks[:, 2:4], axis=1)
        zones[name] = {"n": len(checks), "p50_px": float(np.median(residual)),
                       "p95_px": float(np.percentile(residual, 95)),
                       "max_px": float(residual.max()),
                       "median_match_correlation": float(np.median(checks[:, 4]))}
    return h, {"fit_n": len(rows), "fit_inlier_fraction": float(inliers.mean()),
               "sampled_jacobian_min": float(jac.min()), "heldout": zones}


def raw_coordinates(u: np.ndarray, v: np.ndarray, h: np.ndarray, params: dict,
            distortion: Sequence[int], ca: Sequence[int] | None = None, *, parameters: Mapping):
    'Map review coordinates through the supplied camera and radial lens model.'
    uv = np.column_stack((u.ravel(), v.ravel())).astype(np.float32)
    xy = cv2.perspectiveTransform(uv[None], h)[0]
    du, dv = xy[:, 0]-params["cu"], xy[:, 1]-params["cv"]
    radius = np.hypot(du, dv)
    s = 1 + params["amp"]*np.interp(radius/params["R"], parameters["knots"], distortion)/parameters["distortion_denominator"]
    if ca is not None:
        s *= 1 + np.interp(radius/params["R"], parameters["knots"], ca)/parameters["ca_denominator"]
    x = params["cu"]+parameters["raw_offset_xy"][0]+params["tx"]+s*du
    y = params["cv"]+parameters["raw_offset_xy"][1]+params["ty"]+s*dv
    return x.reshape(u.shape), y.reshape(v.shape)


def raw_review_residual(g_plane: np.ndarray, native_green: np.ndarray,
                         h: np.ndarray, params: dict, distortion: Sequence[int], *, parameters: Mapping) -> dict:
    'Measure independent green-sensor residuals on configured review regions.'
    policy = parameters["residual"]
    hh, ww = policy["shape_hw"]
    ay, ax = np.mgrid[0:hh, 0:ww].astype(np.float32)
    u, v = policy["sample_scale"]*ax+policy["sample_offset"], policy["sample_scale"]*ay+policy["sample_offset"]
    rx, ry = raw_coordinates(u, v, h, params, distortion, parameters=parameters["lens"])
    gx = ((rx-policy["sample_offset"])/policy["sample_scale"]).astype(np.float32)
    gy = ((ry-policy["sample_offset"])/policy["sample_scale"]).astype(np.float32)
    raw = cv2.remap(np.log(np.maximum(g_plane, 1.0)), gx, gy, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT)
    jpeg = native_green[:policy["native_crop_rows"]].reshape(hh, policy["sample_scale"], ww, policy["sample_scale"]).mean((1, 3))
    a, b = bandpass(np.log(jpeg+1.0), parameters=parameters["matching"]), bandpass(raw, parameters=parameters["matching"])
    zones = {}
    for name, box, step in policy["zones"]:
        x0, y0, x1, y1 = box
        rows = match_rows(a, b,
                             ((x, y) for y in range(y0, y1, step)
                              for x in range(x0, x1, step)),
                             template_half=policy["template_half"], search_half=policy["search_half"],
                             guess=lambda x, y: (x, y), min_correlation=policy["min_correlation"], parameters=parameters["matching"])
        zones[name] = match_metrics(rows, scale=policy["sample_scale"])
        if zones[name]["n"] < policy["minimum_matches"]:
            raise ValueError(f"Insufficient direct RAW-review matches in {name}")
    return zones


def registered_sensor_linear(
    arw_path: str | Path, camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path, *,
    expected_source_hashes: Mapping[str, str],
    content_box: Sequence[int],
    parameters: Mapping, source_id: str, profile_id: str,
) -> tuple[np.ndarray, dict]:
    'Return source-bound, camera-linear colour planes on the review content grid.'
    import rawpy
    common = parameters["common"]
    sensor = common["sensor"]
    arw_path = Path(arw_path)
    camera_jpeg_path = Path(camera_jpeg_path)
    review_native_jpeg_path = Path(review_native_jpeg_path)
    if arw_path.stem.lower() != camera_jpeg_path.stem.lower():
        raise ValueError("ARW and camera JPEG stem mismatch")
    if len(content_box) != 4:
        raise ValueError("content_box must be (x,y,w,h)")
    bx, by, width, height = map(int, content_box)
    if (width, height) != tuple(sensor["content_wh"]):
        raise ValueError("Unrecognized native review content geometry")
    for key, path in (
        ("arw_sha256", arw_path),
        ("camera_jpeg_sha256", camera_jpeg_path),
        ("review_native_jpeg_sha256", review_native_jpeg_path),
    ):
        expected = expected_source_hashes.get(key)
        if not expected or sha256_file(path) != expected.lower():
            raise ValueError(f"Missing or mismatched exact source hash: {key}")
    with Image.open(camera_jpeg_path) as image:
        if image.size != tuple(sensor["camera_wh"]):
            raise ValueError("Camera JPEG dimensions differ from frozen Sony contract")
        camera_green = np.asarray(image.convert("RGB"))[..., 1].astype(np.float32)
    with Image.open(review_native_jpeg_path) as image:
        if bx < 0 or by < 0 or bx+width > image.width or by+height > image.height:
            raise ValueError("Content box outside native review canvas")
        review_green = np.asarray(image.convert("RGB"))[by:by+height, bx:bx+width, 1].astype(np.float32)
        canvas_wh = image.size
    h, h_cert = review_to_camera(review_green, camera_green, parameters=common)

    distortion, ca_red, ca_blue = read_lens_tables(arw_path, parameters=common["lens"])
    if distortion != tuple(common["profiles"][profile_id]["distortion"]):
        raise ValueError(f"Sony distortion profile mismatch for {source_id}")
    with rawpy.imread(str(arw_path)) as raw:
        if raw.raw_image_visible.shape != (sensor["raw_wh"][1], sensor["raw_wh"][0]):
            raise ValueError("RAW visible dimensions differ from frozen Sony contract")
        if not np.array_equal(raw.raw_pattern, sensor["cfa_pattern"]) or raw.color_desc != sensor["color_description"].encode("ascii"):
            raise ValueError("CFA Bayer pattern differs from frozen Sony contract")
        if tuple(raw.black_level_per_channel) != tuple(sensor["black_levels"]):
            raise ValueError("RAW black level differs from frozen Sony contract")
        mosaic = raw.raw_image_visible.astype(np.float32).copy()
    planes = (
        mosaic[0::sensor["plane_stride"], 0::sensor["plane_stride"]]-sensor["black_level"],
        sensor["green_mean_weight"]*(mosaic[0::sensor["plane_stride"], 1::sensor["plane_stride"]]+mosaic[1::sensor["plane_stride"], 0::sensor["plane_stride"]])-sensor["black_level"],
        mosaic[1::sensor["plane_stride"], 1::sensor["plane_stride"]]-sensor["black_level"],
    )
    params = common["profiles"][profile_id]["coefficients"]
    direct = raw_review_residual(planes[1], review_green, h, params, distortion, parameters=common)
    output = np.empty((3, height, width), dtype=np.float32)
    for y0 in range(0, height, sensor["tile_rows"]):
        y1 = min(height, y0+sensor["tile_rows"])
        ys, xs = np.mgrid[y0:y1, 0:width].astype(np.float32)
        for channel, (plane, ca, phase) in enumerate(zip(
            planes, (ca_red, None, ca_blue), parameters["raw"]["channel_phases"], strict=True
        )):
            rx, ry = raw_coordinates(xs, ys, h, params, distortion, ca, parameters=common["lens"])
            mx = ((rx-phase[0])/sensor["plane_stride"]).astype(np.float32)
            my = ((ry-phase[1])/sensor["plane_stride"]).astype(np.float32)
            output[channel, y0:y1] = cv2.remap(
                plane, mx, my, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
            )
    certificate = {
        "contract": parameters["raw"]["certificate"]["contract"],
        "run": source_id, "profile": profile_id,
        "sources": {key: expected_source_hashes[key] for key in (
            "arw_sha256", "camera_jpeg_sha256", "review_native_jpeg_sha256")},
        "camera_jpeg_wh": tuple(sensor["camera_wh"]), "raw_visible_wh": sensor["raw_wh"],
        "review_canvas_wh": canvas_wh, "content_box_xywh": [bx, by, width, height],
        "review_to_camera_h": h.tolist(), "review_camera_geometry": h_cert,
        "raw_review_direct_residual": direct,
        "sony_distortion_table": list(distortion),
        "sony_ca_red_table": list(ca_red), "sony_ca_blue_table": list(ca_blue),
        "output_shape": list(output.shape),
        "training_eligible": False,
        "eligibility_reason": parameters["raw"]["certificate"]["eligibility_reason"],
    }
    return output, certificate


def check_demosaic_order(*, parameters: Mapping) -> None:
    # OpenCV's BayerRG2BGR_EA returns RGB array for this RGGB fixture. This
    # explicit runtime control guards against a silent R/B swap across builds.
    'Verify the configured Bayer conversion against a known channel-order fixture.'
    policy = parameters["edge_aware"]
    test = cv2.demosaicing(np.array(policy["order_fixture"], dtype=np.uint16), getattr(cv2, policy["demosaic_code"]))
    if not np.array_equal(test[policy["order_check_xy"][1], policy["order_check_xy"][0]], policy["order_expected_rgb"]):
        raise ValueError("OpenCV Bayer output order differs from frozen RGGB test")


def registered_edge_aware_sensor_linear(
    arw_path: str | Path, camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path, *,
    expected_source_hashes: Mapping[str, str],
    content_box: Sequence[int],
    parameters: Mapping, source_id: str, profile_id: str,
) -> tuple[np.ndarray, dict]:
    'Demosaic integer sensor values and remap each channel without radiometric changes.'
    try:
        import rawpy
    except ImportError as error:
        raise ImportError("Edge-aware sensor rendering requires rawpy (CPU LibRaw)") from error
    common = parameters["common"]
    sensor = common["sensor"]
    check_demosaic_order(parameters=parameters)
    arw = Path(arw_path)
    camera = Path(camera_jpeg_path)
    native = Path(review_native_jpeg_path)
    if arw.stem.casefold() != camera.stem.casefold():
        raise ValueError("ARW/JPEG basename mismatch")
    for key, path in (("arw_sha256", arw), ("camera_jpeg_sha256", camera),
                      ("review_native_jpeg_sha256", native)):
        if not expected_source_hashes.get(key) or sha256_file(path) != expected_source_hashes[key].lower():
            raise ValueError(f"Missing or mismatched SHA: {key}")
    x0, y0, width, height = map(int, content_box)
    if (width, height) != tuple(sensor["content_wh"]):
        raise ValueError("Native review content shape mismatch")
    with Image.open(camera) as image:
        if image.size != tuple(sensor["camera_wh"]):
            raise ValueError("Unexpected Sony camera JPEG shape")
        camera_green = np.asarray(image.convert("RGB"))[..., 1].astype(np.float32)
    with Image.open(native) as image:
        if x0 < 0 or y0 < 0 or x0+width > image.width or y0+height > image.height:
            raise ValueError("Content box outside native canvas")
        review_green = np.asarray(image.convert("RGB"))[y0:y0+height, x0:x0+width, 1].astype(np.float32)
        canvas_wh = image.size
    homography, h_qa = review_to_camera(review_green, camera_green, parameters=common)
    distortion, ca_red, ca_blue = read_lens_tables(arw, parameters=common["lens"])
    if distortion != tuple(common["profiles"][profile_id]["distortion"]):
        raise ValueError("Unexpected Sony 0x7037 lens profile")
    with rawpy.imread(str(arw)) as raw:
        if raw.raw_image_visible.shape != (sensor["raw_wh"][1], sensor["raw_wh"][0]):
            raise ValueError("Unexpected RAW visible shape")
        if not np.array_equal(raw.raw_pattern, sensor["cfa_pattern"]) or raw.color_desc != sensor["color_description"].encode("ascii"):
            raise ValueError("Unexpected CFA pattern")
        black_levels = tuple(int(x) for x in raw.black_level_per_channel)
        if black_levels != tuple(sensor["black_levels"]):
            raise ValueError("Unexpected black levels")
        white_level = int(raw.white_level)
        if white_level <= sensor["black_level"] or white_level > sensor["white_maximum"]:
            raise ValueError("Unexpected RAW white level")
        original = raw.raw_image_visible.copy()
        camera_wb = [float(x) for x in raw.camera_whitebalance]
        camera_to_xyz = np.asarray(raw.rgb_xyz_matrix, np.float64).tolist()
    sensor_saturation = original >= white_level
    # Subtract black *before* demosaic; no 0..1 scaling or automatic exposure.
    # All sampled ZV-1 inputs are unsigned <=16-bit. uint16 addition/interpolation
    # cannot overflow as sensor white is <16384 after black subtraction.
    black_subtracted = np.maximum(original.astype(np.int32)-sensor["black_level"], 0).astype(np.uint16)
    demosaiced = cv2.demosaicing(black_subtracted, getattr(cv2, parameters["edge_aware"]["demosaic_code"]))
    if demosaiced.shape != (sensor["raw_wh"][1], sensor["raw_wh"][0], 3):
        raise ValueError("Full-resolution demosaic shape mismatch")
    demosaiced_float = [demosaiced[..., channel].astype(np.float32) for channel in range(3)]
    # Independent green CFA control checks RAW↔native geometry without using
    # the demosaiced RGB itself to fit the homography.
    gg = sensor["green_mean_weight"]*(black_subtracted[0::sensor["plane_stride"], 1::sensor["plane_stride"]].astype(np.float32)
             + black_subtracted[1::sensor["plane_stride"], 0::sensor["plane_stride"]].astype(np.float32))
    direct = raw_review_residual(gg, review_green, homography,
                                       common["profiles"][profile_id]["coefficients"], distortion, parameters=common)
    output = np.empty((3, height, width), dtype=np.float32)
    params = common["profiles"][profile_id]["coefficients"]
    # Sony CA is a channel-specific change of radial scale, applied in the
    # camera-JPEG-centered inverse map for each full-res demosaiced channel.
    for yy0 in range(0, height, sensor["tile_rows"]):
        yy1 = min(height, yy0+sensor["tile_rows"])
        ys, xs = np.mgrid[yy0:yy1, 0:width].astype(np.float32)
        for channel, ca in enumerate((ca_red, None, ca_blue)):
            rx, ry = raw_coordinates(xs, ys, homography, params, distortion, ca, parameters=common["lens"])
            output[channel, yy0:yy1] = cv2.remap(
                demosaiced_float[channel],
                rx.astype(np.float32), ry.astype(np.float32), cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
            )
    finite = np.isfinite(output)
    # Interpolation should remain convex, but still report violations rather
    # than silently clipping a scientific signal.
    overshoot = output > (white_level-sensor["black_level"])
    negative = output < 0
    certificate = {
        "contract": parameters["edge_aware"]["certificate"]["contract"],
        "run": source_id, "profile": profile_id,
        "sources": {key: expected_source_hashes[key] for key in (
            "arw_sha256", "camera_jpeg_sha256", "review_native_jpeg_sha256")},
        "camera_jpeg_wh": tuple(sensor["camera_wh"]), "raw_visible_wh": sensor["raw_wh"],
        "review_canvas_wh": canvas_wh, "content_box_xywh": [x0,y0,width,height],
        "review_to_camera_h": homography.tolist(),
        "review_camera_geometry": h_qa,
        "raw_review_direct_residual": direct,
        "sony_distortion_table": list(distortion),
        "sony_ca_red_table": list(ca_red),
        "sony_ca_blue_table": list(ca_blue),
        "demosaic": "OpenCV edge-aware RGGB, full sensor resolution, 16-bit integer before geometric warp",
        "native_warp_interpolation": "per-channel bilinear float32; NaN outside sensor",
        "channel_basis": "camera-native black-subtracted linear RGB, RAW units",
        "white_balance": {"applied": False, "camera_wb_metadata": camera_wb,
                          "reason": parameters["edge_aware"]["certificate"]["white_balance_reason"]},
        "camera_to_xyz": {"applied": False, "rgb_xyz_metadata": camera_to_xyz,
                          "reason": "Phase calibration is in camera channel basis"},
        "gamma_tone_auto_bright": "none",
        "black_level_per_channel": list(black_levels), "white_level": white_level,
        "sensor_saturated_fraction": float(sensor_saturation.mean()),
        "sensor_saturated_count": int(sensor_saturation.sum()),
        "native_nonfinite_fraction": float(1-finite.mean()),
        "native_negative_overshoot_fraction": float(negative.mean()),
        "native_above_white_overshoot_fraction": float(overshoot.mean()),
        "native_linear_rgb_shape": list(output.shape),
        "training_eligible": False,
        "eligibility_reason": "All-frame, exact-label and photometric parity gates are separate from this geometry/render certificate.",
    }
    return output, certificate


def gray_roi(path: Path, box: tuple[int, int, int, int], *, parameters: Mapping) -> np.ndarray:
    'Read a copied grayscale camera region after its configured size check.'
    left, top, right, bottom = box
    with Image.open(path) as image:
        if image.size != tuple(parameters["camera_wh"]):
            raise ValueError("Original camera JPEG has unexpected dimensions")
        return np.asarray(image.convert("L"))[top:bottom, left:right].copy()


def register_chart(
    *,
    baseline_camera_jpeg: str | Path,
    baseline_camera_sha256: str,
    frame_camera_jpeg: str | Path,
    frame_camera_sha256: str,
    baseline_chart_xy8: np.ndarray,
    baseline_chart_log_rgb: np.ndarray,
    baseline_review_to_camera_h: np.ndarray,
    frame_review_to_camera_h: np.ndarray,
    frame_native_linear_rgb: np.ndarray,
    parameters: Mapping,
    _gray_reader=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    'Register frozen chart swatches from source photographs and reject invalid support.'
    policy = parameters["chart"]
    sensor = parameters["common"]["sensor"]
    baseline = Path(baseline_camera_jpeg)
    frame = Path(frame_camera_jpeg)
    if sha256_file(baseline) != baseline_camera_sha256 or sha256_file(frame) != frame_camera_sha256:
        raise ValueError("Camera JPEG changed since frozen source binding")
    xy = np.asarray(baseline_chart_xy8, np.float32)
    cb = np.asarray(baseline_chart_log_rgb, np.float32)
    bh = np.asarray(baseline_review_to_camera_h, np.float64)
    fh = np.asarray(frame_review_to_camera_h, np.float64)
    native = np.asarray(frame_native_linear_rgb)
    if xy.shape != (policy["swatches"], 2) or cb.shape != (3, policy["swatches"]):
        raise ValueError("The baseline chart must have 24 source-bound swatches")
    if bh.shape != (3, 3) or fh.shape != (3, 3):
        raise ValueError("Chart registration requires two camera homographies")
    if native.shape != (3, sensor["content_wh"][1], sensor["content_wh"][0]) or native.dtype != np.dtype("float32"):
        raise ValueError("Chart saturation control requires full-native EA camera-linear RGB")
    baseline_native = xy * policy["grid_scale"] + policy["center_offset"]
    baseline_camera = cv2.perspectiveTransform(baseline_native[None], bh)[0]
    left = max(0, int(np.floor(np.min(baseline_camera[:, 0]) - policy["template_margin"])))
    top = max(0, int(np.floor(np.min(baseline_camera[:, 1]) - policy["template_margin"])))
    right = min(sensor["camera_wh"][0], int(np.ceil(np.max(baseline_camera[:, 0]) + policy["template_margin"])))
    bottom = min(sensor["camera_wh"][1], int(np.ceil(np.max(baseline_camera[:, 1]) + policy["template_margin"])))
    template_box = (left, top, right, bottom)
    if right-left < policy["minimum_template_size"] or bottom-top < policy["minimum_template_size"]:
        raise ValueError("Baseline checker bounds are incomplete")
    search_box = (max(0, left-policy["search_margin"]), max(0, top-policy["search_margin"]),
                  min(sensor["camera_wh"][0], right+policy["search_margin"]), min(sensor["camera_wh"][1], bottom+policy["search_margin"]))
    cv2.setRNGSeed(policy["rng_seed"])
    sift = cv2.SIFT_create(nfeatures=policy["sift_features"])
    read_gray = gray_roi if _gray_reader is None else _gray_reader
    template_kp, template_desc = sift.detectAndCompute(read_gray(baseline, template_box, parameters=sensor), None)
    frame_kp, frame_desc = sift.detectAndCompute(read_gray(frame, search_box, parameters=sensor), None)
    if template_desc is None or frame_desc is None:
        raise ValueError("Checker registration has no source-only descriptors")
    matches = cv2.BFMatcher().knnMatch(template_desc, frame_desc, k=2)
    good = [m for m, n in matches if m.distance < policy["ratio_threshold"] * n.distance]
    if len(good) < policy["minimum_ratio_matches"]:
        raise ValueError(f"Checker registration has only {len(good)} ratio matches")
    src = np.float32([template_kp[m.queryIdx].pt for m in good])
    dst = np.float32([frame_kp[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=policy["ransac_threshold"],
    )
    if affine is None or inliers is None or int(inliers.sum()) < policy["minimum_inliers"]:
        raise ValueError("Checker registration lacks 50 geometric inliers")
    keep = inliers.ravel().astype(bool)
    residual = np.linalg.norm(cv2.transform(src[None], affine)[0]-dst, axis=1)[keep]
    p95 = float(np.percentile(residual, 95))
    if p95 > policy["maximum_p95"]:
        raise ValueError(f"Checker registration p95 residual exceeds 2 camera px: {p95}")
    origin_b = np.asarray(template_box[:2], np.float32)
    origin_f = np.asarray(search_box[:2], np.float32)
    frame_camera = cv2.transform((baseline_camera-origin_b)[None], affine)[0] + origin_f
    frame_native = cv2.perspectiveTransform(frame_camera[None], np.linalg.inv(fh))[0]
    frame_xy = ((frame_native-policy["center_offset"])/policy["grid_scale"]).astype(np.float32)
    usable = np.all(np.isfinite(cb), axis=0) & np.all(np.isfinite(frame_xy), axis=1)
    for index, (x, y) in enumerate(frame_xy):
        ix, iy = int(round(float(x*policy["grid_scale"]+policy["center_offset"]))), int(round(float(y*policy["grid_scale"]+policy["center_offset"])))
        # The light fitter samples 3×3 grid centres at offsets ±8 native px.
        # Include the 8×8 support around each centre for the RAW saturation gate.
        if ix-policy["patch_lower"] < 0 or iy-policy["patch_lower"] < 0 or ix+policy["patch_upper"] > sensor["content_wh"][0] or iy+policy["patch_upper"] > sensor["content_wh"][1]:
            usable[index] = False
            continue
        patch = native[:, iy-policy["patch_lower"]:iy+policy["patch_upper"], ix-policy["patch_lower"]:ix+policy["patch_upper"]]
        if not np.all(np.isfinite(patch)) or np.any(patch >= policy["saturation_threshold"]):
            usable[index] = False
    if int(np.count_nonzero(usable)) < policy["minimum_usable"]:
        raise ValueError("Fewer than six source-valid ColorChecker patches")
    centre_before = np.mean(baseline_camera, axis=0)
    centre_after = np.mean(frame_camera, axis=0)
    certificate = dict(
        contract=policy["certificate"]["contract"],
        baseline_camera_jpeg=str(baseline), baseline_camera_sha256=baseline_camera_sha256,
        frame_camera_jpeg=str(frame), frame_camera_sha256=frame_camera_sha256,
        baseline_review_to_camera_h=bh.tolist(), frame_review_to_camera_h=fh.tolist(),
        baseline_chart_xy8=xy.tolist(), frame_chart_xy8=frame_xy.tolist(),
        frame_chart_usable=usable.tolist(), template_box_camera_xyxy=list(template_box),
        search_box_camera_xyxy=list(search_box), ratio_matches=len(good),
        ransac_inliers=int(np.count_nonzero(keep)), inlier_residual_p95_camera_px=p95,
        centre_shift_camera_xy=(centre_after-centre_before).tolist(),
        centre_shift_camera_px=float(np.linalg.norm(centre_after-centre_before)),
        clipped_or_nonfinite_patches=int(np.count_nonzero(~usable)),
        usable_patches=int(np.count_nonzero(usable)),
        raw_saturation_threshold=policy["certificate"]["saturation_description"],
        registration_affine_template_to_search=affine.tolist(),
    )
    return frame_xy, usable, certificate


def registered_sensor_grid_log(arw_path, review_to_camera_h, profile_id,
                               geometry_parameters, *, parameters: Mapping):
    """Decode blurred sensor planes onto a configured grid before log encoding."""
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("RAW source processing requires rawpy") from exc
    common = parameters["common"]
    sensor = common["sensor"]
    policy = parameters["raw_grid"]
    arw_path = Path(arw_path)
    H = np.asarray(review_to_camera_h, np.float64)
    if H.shape != (3, 3):
        raise ValueError("Expected 3x3 review-to-camera homography")
    if profile_id not in common["profiles"]:
        raise ValueError(f"Unknown geometry profile: {profile_id}")
    with rawpy.imread(str(arw_path)) as raw:
        if raw.raw_image_visible.shape != (sensor["raw_wh"][1], sensor["raw_wh"][0]):
            raise ValueError(f"Unexpected RAW visible size: {raw.raw_image_visible.shape}")
        if not np.array_equal(raw.raw_pattern, sensor["cfa_pattern"]):
            raise ValueError(f"Unexpected RAW CFA pattern: {raw.raw_pattern}")
        v = raw.raw_image_visible.astype(np.float32).copy()
    planes = {
        "R": v[0::sensor["plane_stride"], 0::sensor["plane_stride"]] - policy["black_level"],
        "G": sensor["green_mean_weight"] * (v[0::sensor["plane_stride"], 1::sensor["plane_stride"]] + v[1::sensor["plane_stride"], 0::sensor["plane_stride"]]) - policy["black_level"],
        "B": v[1::sensor["plane_stride"], 1::sensor["plane_stride"]] - policy["black_level"],
    }
    planes = {c: cv2.GaussianBlur(p, (0, 0), policy["blur_sigma"]) for c, p in planes.items()}
    h, w = policy["shape_hw"]
    gy, gx = np.mgrid[:h, :w]
    points = np.column_stack([
        (policy["grid_scale"] * gx + policy["center_offset"]).ravel(), (policy["grid_scale"] * gy + policy["center_offset"]).ravel()
    ]).astype(np.float32)
    cam = cv2.perspectiveTransform(points[None], H)[0].astype(np.float64)
    u, vv = cam[:, 0], cam[:, 1]
    p = geometry_parameters[policy["geometry_coefficients_key"]][profile_id]
    table = geometry_parameters[policy["geometry_distortion_key"]][profile_id]
    ca = geometry_parameters[policy["geometry_ca_key"]]
    du, dv = u - p["cu"], vv - p["cv"]
    radius = np.hypot(du, dv)
    radial = np.interp(radius / p["R"], np.asarray(common["lens"]["knots"]), table)
    scale = 1 + p["amp"] * radial / common["lens"]["distortion_denominator"]
    phase = policy["channel_phases"]
    channels = []
    for colour in "RGB":
        ca_factor = (
            np.interp(radius / p["R"], np.asarray(common["lens"]["knots"]), ca[colour])
            / policy["ca_denominator"] if colour in ca else 0
        )
        x = p["cu"] + common["lens"]["raw_offset_xy"][0] + p["tx"] + scale * (1 + ca_factor) * du
        y = p["cv"] + common["lens"]["raw_offset_xy"][1] + p["ty"] + scale * (1 + ca_factor) * dv
        oy, ox = phase[colour]
        mx = ((x - ox) / sensor["plane_stride"]).reshape(h, w).astype(np.float32)
        my = ((y - oy) / sensor["plane_stride"]).reshape(h, w).astype(np.float32)
        channels.append(cv2.remap(
            planes[colour], mx, my, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
        ))
    linear = np.stack(channels).astype(np.float32)
    linear = np.where(
        np.isfinite(linear) & (linear > policy["signal_minimum"]) & (linear < policy["signal_maximum"]),
        linear,
        np.nan,
    )
    return np.log(linear).astype(np.float32)
