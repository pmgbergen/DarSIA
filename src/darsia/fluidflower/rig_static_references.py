"""User-selected static rig patches and reviewable planar geometry proposals.

Camera-U coordinates use x right, y down, including negative coordinates.
Measured rig coordinates use x right, y up from the rig bottom in millimetres.
Only an explicit save writes; proposals never adopt geometry or modify maps.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np


FORMAT = "rig-static-references-v1"
MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
MAX_REGIONS = 48
MAX_RUNS = 1000
MAX_TEMPLATE_SIDE = 512
MAX_TEMPLATE_BYTES = 1024 * 1024
MAX_MEASUREMENT_UNCERTAINTY_MM = 100.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_ACCEPTED = {"recognized", "manual"}


def empty_document():
    return {"format": FORMAT, "regions": [], "runs": {}, "measurement_uncertainty_mm": 0.}


def _measurement_uncertainty(value):
    if (isinstance(value, (bool, str)) or not np.isscalar(value)
            or not isinstance(value, (int, float, np.number))):
        raise ValueError("Measurement uncertainty must be a finite millimetre value")
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= MAX_MEASUREMENT_UNCERTAINTY_MM:
        raise ValueError(f"Measurement uncertainty must be between 0 and {MAX_MEASUREMENT_UNCERTAINTY_MM:g} mm")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite numbers") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have shape {shape} and finite coordinates")
    return result


def _identifier(value, label):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{label} needs 1-96 letters, numbers, dots, hyphens or underscores")
    return value


def _project(h, points):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    value = np.column_stack((points, np.ones(len(points)))) @ np.asarray(h).T
    if np.any(np.abs(value[:, 2]) < 1e-10):
        raise ValueError("The coordinate transform crosses a projective horizon")
    result = value[:, :2] / value[:, 2, None]
    if not np.isfinite(result).all():
        raise ValueError("The coordinate transform produced nonfinite points")
    return result


def _rect_corners(rect):
    x, y, w, h = rect
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], np.float64)


def _file_binding(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"The packet needs a bound {label} path and SHA256")
    path, digest = value.get("path"), value.get("sha256")
    if not isinstance(path, str) or not path or len(path) > 4096:
        raise ValueError(f"The {label} path is missing or exceeds its bound")
    if not isinstance(digest, str) or not _HEX.fullmatch(digest.lower()):
        raise ValueError(f"The {label} SHA256 is missing or invalid")
    return dict(path=path, sha256=digest.lower())


def _packet(packet):
    if packet.get("full_raw_sensor_field") is not True:
        raise ValueError("Static references require the complete original RAW display packet")
    image = np.asarray(packet.get("image_rgb"))
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or min(image.shape[:2]) < 8 or image.shape[0]*image.shape[1] > 120_000_000):
        raise ValueError("The complete RAW display must be a bounded RGB uint8 image")
    h = packet.get("camera_u_to_photo_h")
    if h is None and packet.get("mapping") is not None:
        h = packet["mapping"].camera_u_to_photo_h
    if h is None and packet.get("extent_u_xywh") is not None:
        x, y, w, height = _array(packet["extent_u_xywh"], (4,), "Display extent")
        if w <= 0 or height <= 0:
            raise ValueError("The display extent needs positive dimensions")
        h = [[image.shape[1]/w, 0, -x*image.shape[1]/w],
             [0, image.shape[0]/height, -y*image.shape[0]/height], [0, 0, 1]]
    h = _array(h, (3, 3), "Camera-U to photo transform")
    if abs(np.linalg.det(h)) < 1e-12 or np.linalg.cond(h) > 1e12:
        raise ValueError("The camera-U to photo transform is singular or ill-conditioned")
    h = h / (h[2, 2] if abs(h[2, 2]) > 1e-12 else np.linalg.norm(h))
    support = np.asarray(packet.get("valid_support", np.ones(image.shape[:2], bool)))
    if support.shape != image.shape[:2]:
        raise ValueError("RAW support mask and display image have different shapes")
    raw = _file_binding(packet.get("source_raw"), "source RAW")
    profile = _file_binding(packet.get("profile"), "lens profile")
    profile_id = packet["profile"].get("profile_id")
    if not isinstance(profile_id, str) or not profile_id or len(profile_id) > 256:
        raise ValueError("The packet needs its exact lens profile ID")
    profile["profile_id"] = profile_id
    photo = dict(source_raw=raw, profile=profile, image_shape=list(image.shape),
                 camera_u_to_photo_h=h.tolist(), coordinate_system="camera_u_xy",
                 image_encoding="rgb_uint8", full_raw_sensor_field=True)
    for key in ("sensor_wh", "camera_u_wh", "extent_u_xywh"):
        if key in packet:
            size = 4 if key == "extent_u_xywh" else 2
            photo[key] = _array(packet[key], (size,), key).tolist()
    photo["identity"] = _digest(photo)
    return image, support.astype(bool, copy=False), h, photo


def _binding_problem(photo):
    """Recheck a bound source when available; missing originals are explicit."""
    for key in ("source_raw", "profile"):
        binding = photo.get(key, {})
        try:
            path = Path(binding["path"])
            if not path.is_file():
                return f"Bound {key} file is unavailable: {path}"
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            if digest.hexdigest() != binding["sha256"]:
                return f"Bound {key} SHA256 changed: {path}"
        except (OSError, KeyError, ValueError) as error:
            return f"Cannot verify bound {key}: {error}"
    return None


def photo_binding(packet):
    """Return the exact source/display binding used for manual observations."""
    return _packet(packet)[3]


def _encode(array):
    ok, encoded = cv2.imencode(".png", array)
    if not ok or len(encoded) > MAX_TEMPLATE_BYTES:
        raise ValueError("The reference template exceeds its encoded image bound")
    return base64.b64encode(encoded).decode("ascii")


def _decode(value, shape):
    if not isinstance(value, str) or len(value) > 4*MAX_TEMPLATE_BYTES//3+8:
        raise ValueError("The encoded reference template exceeds its bound")
    try:
        encoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("The reference template is not valid base64") from error
    if len(encoded) > MAX_TEMPLATE_BYTES:
        raise ValueError("The reference template exceeds its byte bound")
    # Inspect PNG IHDR before decoding; compressed payloads cannot allocate huge rasters.
    if (len(encoded) < 24 or encoded[:8] != b"\x89PNG\r\n\x1a\n"
            or tuple(int.from_bytes(encoded[i:i+4], "big") for i in (20, 16)) != tuple(shape)):
        raise ValueError("The reference PNG dimensions differ from its bound shape")
    result = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_GRAYSCALE)
    if result is None or result.shape != tuple(shape):
        raise ValueError("The reference template PNG could not be decoded")
    return result


def _region_identity(region):
    return _digest({key: region[key] for key in
                    ("id", "name", "rect_u_xywh", "anchor_u_xy", "metric_xy_mm",
                     "plane_id", "source", "template")})


def capture_region(packet, *, region_id, name, rect_u_xywh, anchor_u_xy,
                   metric_xy_mm, plane_id="rig"):
    """Capture actual static appearance, never just a remembered pixel position."""
    image, support, h, photo = _packet(packet)
    _identifier(region_id, "Region ID")
    _identifier(plane_id, "Depth plane ID")
    if not isinstance(name, str) or not name.strip() or len(name) > 256:
        raise ValueError("The static region needs a name of at most 256 characters")
    rect = _array(rect_u_xywh, (4,), "Reference rectangle")
    anchor = _array(anchor_u_xy, (2,), "Reference anchor")
    metric = _array(metric_xy_mm, (2,), "Measured millimetre coordinate")
    if np.any(rect[2:] <= 0) or np.any(anchor < rect[:2]) or np.any(anchor > rect[:2]+rect[2:]):
        raise ValueError("The positive reference rectangle must contain its anchor")
    quad = _project(h, _rect_corners(rect))
    size = np.array([np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[3]-quad[0])])
    size *= min(1., MAX_TEMPLATE_SIDE/max(size))
    width, height = map(int, np.maximum(8, np.rint(size).astype(int)))
    corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
    template_to_photo = cv2.getPerspectiveTransform(corners, quad.astype(np.float32))
    inverse = np.linalg.inv(template_to_photo)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    template = cv2.warpPerspective(gray, inverse, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(support.astype(np.uint8), inverse, (width, height), flags=cv2.INTER_NEAREST)
    anchor_t = _project(inverse @ h, [anchor])[0]
    if np.mean(mask != 0) < .98:
        raise ValueError("The selected reference extends beyond available original RAW pixels")
    if np.std(template[mask != 0]) < 5:
        raise ValueError("The selected reference has insufficient visible texture")
    region = dict(id=region_id, name=name.strip(), rect_u_xywh=rect.tolist(),
                  anchor_u_xy=anchor.tolist(), metric_xy_mm=metric.tolist(), plane_id=plane_id,
                  source=photo, template=dict(encoding="png_gray8_base64", shape_hw=[height, width],
                    image_png_base64=_encode(template), mask_png_base64=_encode(mask*255),
                    template_to_source_photo_h=template_to_photo.tolist(),
                    anchor_template_xy=anchor_t.tolist()))
    region["identity"] = _region_identity(region)
    return region


def _validate_region(region):
    _identifier(region.get("id"), "Region ID")
    _identifier(region.get("plane_id"), "Depth plane ID")
    rect = _array(region.get("rect_u_xywh"), (4,), "Reference rectangle")
    anchor = _array(region.get("anchor_u_xy"), (2,), "Reference anchor")
    _array(region.get("metric_xy_mm"), (2,), "Measured millimetre coordinate")
    if np.any(rect[2:] <= 0) or np.any(anchor < rect[:2]) or np.any(anchor > rect[:2]+rect[2:]):
        raise ValueError("The reference rectangle must contain its anchor")
    template = region.get("template", {})
    shape = template.get("shape_hw", [])
    if (len(shape) != 2 or any(type(n) is not int or not 8 <= n <= MAX_TEMPLATE_SIDE for n in shape)):
        raise ValueError("The reference template shape is invalid")
    image = _decode(template.get("image_png_base64"), shape)
    mask = _decode(template.get("mask_png_base64"), shape)
    _array(template.get("anchor_template_xy"), (2,), "Template anchor")
    _array(template.get("template_to_source_photo_h"), (3, 3), "Template transform")
    if template.get("encoding") != "png_gray8_base64":
        raise ValueError("The reference template encoding is unsupported")
    if region.get("identity") != _region_identity(region):
        raise ValueError("The reference identity changed; recapture the edited region")
    photo = region["source"]
    if photo.get("identity") != _digest({k: v for k, v in photo.items() if k != "identity"}):
        raise ValueError("The bound source display coordinate identity changed")
    return image, mask


def update_region(region, *, name=None, metric_xy_mm=None, plane_id=None):
    """Edit a physical target without recapturing it from a different run."""
    _validate_region(region)
    result = deepcopy(region)
    if name is not None:
        if not isinstance(name, str) or not name.strip() or len(name) > 256:
            raise ValueError("The static region needs a name of at most 256 characters")
        result["name"] = name.strip()
    if metric_xy_mm is not None:
        result["metric_xy_mm"] = _array(metric_xy_mm, (2,), "Measured coordinate").tolist()
    if plane_id is not None:
        result["plane_id"] = _identifier(plane_id, "Depth plane ID")
    result["identity"] = _region_identity(result)
    return result


def _normalization(points):
    centre = np.mean(points, axis=0)
    radius = np.sqrt(np.mean(np.sum((points-centre)**2, axis=1)))
    if radius < 1e-8:
        raise ValueError("Reference anchors are coincident")
    s = np.sqrt(2.)/radius
    return np.array([[s, 0, -s*centre[0]], [0, s, -s*centre[1]], [0, 0, 1.]])


def _spread(points):
    singular = np.linalg.svd(points-np.mean(points, axis=0), compute_uv=False)
    return len(singular) == 2 and singular[0] > 1e-8 and singular[1]/singular[0] >= .025


def _positive_transform(h, points, orientation=1):
    samples = np.vstack((points, np.mean(points, axis=0)))
    denominator = np.column_stack((samples, np.ones(len(samples)))) @ h[2]
    if np.any(np.abs(denominator) < 1e-8) or np.min(denominator)*np.max(denominator) <= 0:
        return False
    determinant = np.linalg.det(h) / denominator**3
    return bool(np.isfinite(determinant).all() and np.all(determinant*orientation > 1e-10))


def fit_reference_model(regions, observations, *, plane_id="rig", measurement_uncertainty_mm=0.):
    """Fit one depth plane with explicit physical and image measurement budgets.

    The supplied radial millimetre uncertainty is projected through the largest
    local singular value of the fitted homography Jacobian at each anchor. That
    allowance is added to the existing five camera-U pixel image measurement
    budget. Residuals remain visible in both coordinate systems.
    """
    _identifier(plane_id, "Depth plane ID")
    uncertainty = _measurement_uncertainty(measurement_uncertainty_mm)
    output = dict(status="insufficient", plane_id=plane_id, H_metric_mm_to_camera_u=None,
                  measurement_uncertainty_mm=uncertainty, image_measurement_budget_u_px=5.,
                  metric_coordinate_system="rig_x_right_y_up_mm", camera_coordinate_system="camera_u_x_right_y_down",
                  residuals=[], used_region_ids=[], reason="At least four recognized or manually corrected anchors are required")
    selected, metric, camera = [], [], []
    seen = set()
    for region in regions:
        if region.get('purpose') == 'measurement_endpoint':
            continue
        key = _identifier(region.get("id"), "Region ID")
        if key in seen:
            raise ValueError(f"Duplicate physical reference ID: {key}")
        seen.add(key)
        observation = observations.get(key, {})
        if region.get("plane_id", "rig") != plane_id or observation.get("status") not in _ACCEPTED:
            continue
        if observation.get("region_identity") not in (None, region.get("identity")):
            output.update(status="stale", reason=f"Observation for {key} belongs to a different region shape or measurement")
            return output
        selected.append(key)
        metric.append(_array(region.get("metric_xy_mm"), (2,), "Measured coordinate"))
        camera.append(_array(observation.get("anchor_u_xy"), (2,), "Observed camera-U anchor"))
    output["used_region_ids"] = selected
    if len(selected) < 4:
        return output
    metric, camera = np.asarray(metric), np.asarray(camera)
    if (len(np.unique(np.round(metric, 8), axis=0)) < 4
            or len(np.unique(np.round(camera, 8), axis=0)) < 4):
        output.update(reason='At least four distinct physical and image anchors are required; shared line endpoints count once')
        return output
    if not _spread(metric) or not _spread(camera):
        output.update(status="degenerate", reason="Reference anchors must be spread in two dimensions and noncollinear")
        return output
    a, b = _normalization(metric), _normalization(camera)
    normalized, _ = cv2.findHomography(_project(a, metric), _project(b, camera), 0)
    if normalized is None or not np.isfinite(normalized).all() or np.linalg.cond(normalized) > 1e6:
        output.update(status="degenerate", reason="The reference solution is ill-conditioned")
        return output
    h = np.linalg.inv(b) @ normalized @ a
    if abs(h[2, 2]) < 1e-10 or not _positive_transform(h, metric, orientation=-1):
        output.update(status="invalid_orientation", reason="The reference solution is mirrored or crosses a projective horizon")
        return output
    h /= h[2, 2]
    predicted = _project(h, metric)
    errors = np.linalg.norm(predicted-camera, axis=1)
    metric_errors = np.linalg.norm(_project(np.linalg.inv(h), camera)-metric, axis=1)
    denominator = np.column_stack((metric, np.ones(len(metric)))) @ h[2]
    jacobians = (h[None, :2, :2]-predicted[:, :, None]*h[None, 2, :2])/denominator[:, None, None]
    local_scales = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
    allowed = 5. + uncertainty*local_scales
    output.update(residuals=[dict(region_id=key, error_u_px=float(error), error_mm=float(metric_error),
                        local_scale_u_px_per_mm=float(local_scale), allowed_u_px=float(budget),
                        measurement_uncertainty_mm=uncertainty)
                  for key, error, metric_error, local_scale, budget in
                  zip(selected, errors, metric_errors, local_scales, allowed)],
                  rms_u_px=float(np.sqrt(np.mean(errors**2))), max_u_px=float(max(errors)),
                  rms_mm=float(np.sqrt(np.mean(metric_errors**2))), max_mm=float(max(metric_errors)),
                  normalized_condition=float(np.linalg.cond(normalized)))
    if np.any(errors > allowed):
        output.update(status="inconsistent", reason=("Reference residual exceeds the 5 camera-U pixel image budget "
            f"plus the locally projected {uncertainty:g} mm measurement uncertainty; review the anchors or depth plane"))
        return output
    output.update(status="valid", H_metric_mm_to_camera_u=h.tolist(),
                  reason="Measured geometry proposal")
    return output


def _resize(gray, maximum):
    scale = min(1., maximum/max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def _appearance(template, mask, gray, support, h):
    size = (template.shape[1], template.shape[0])
    available = cv2.warpPerspective(support.astype(np.uint8), np.linalg.inv(h), size,
                                    flags=cv2.INTER_NEAREST) != 0
    available &= mask != 0
    coverage = float(available.sum()/max(1, np.count_nonzero(mask)))
    if coverage < .3:
        return 0., coverage
    observed = cv2.warpPerspective(gray, np.linalg.inv(h), size, flags=cv2.INTER_LINEAR)
    x, y = template[available].astype(float), observed[available].astype(float)
    x -= x.mean()
    y -= y.mean()
    norm = np.linalg.norm(x)*np.linalg.norm(y)
    return (float(np.dot(x, y)/norm) if norm > 1e-8 else 0.), coverage


def _feature_candidate(template, mask, gray, support, detector, target, excluded=None):
    points, descriptors = detector.detectAndCompute(template, mask)
    target_points, target_descriptors, scale = target
    if descriptors is None or target_descriptors is None or len(points) < 6:
        return None
    indexes = list(range(len(target_points)))
    if excluded is not None:
        x0, y0 = excluded.min(axis=0)*scale
        x1, y1 = excluded.max(axis=0)*scale
        indexes = [i for i, point in enumerate(target_points)
                   if not (x0 <= point.pt[0] <= x1 and y0 <= point.pt[1] <= y1)]
    if len(indexes) < 8:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2 if hasattr(cv2, "SIFT_create") else cv2.NORM_HAMMING)
    matches = matcher.knnMatch(descriptors, target_descriptors[indexes], k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .72*pair[1].distance]
    if len(good) < 6:
        return None
    # One target feature cannot count as evidence for several source features.
    unique = {}
    for match in good:
        if match.trainIdx not in unique or match.distance < unique[match.trainIdx].distance:
            unique[match.trainIdx] = match
    good = list(unique.values())
    if len(good) < 6:
        return None
    source_xy = np.float64([points[m.queryIdx].pt for m in good])
    target_xy = np.float64([target_points[indexes[m.trainIdx]].pt for m in good])/scale
    h, inliers = cv2.findHomography(source_xy, target_xy, cv2.RANSAC, 3.5/scale)
    if h is None or inliers is None:
        return None
    keep = inliers.ravel() != 0
    if keep.sum() < 6 or keep.mean() < .55 or not _spread(source_xy[keep]):
        return None
    if cv2.contourArea(cv2.convexHull(source_xy[keep].astype(np.float32))) < template.size*.06:
        return None
    corners = _rect_corners([0, 0, template.shape[1]-1, template.shape[0]-1])
    if not _positive_transform(h, corners):
        return None
    quad = _project(h, corners)
    if not cv2.isContourConvex(quad.astype(np.float32)):
        return None
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
    if min(lengths) < 8 or max(lengths) > 2*max(gray.shape):
        return None
    score, coverage = _appearance(template, mask, gray, support, h)
    error = np.linalg.norm(_project(h, source_xy[keep])-target_xy[keep], axis=1)
    if score < .55 or np.median(error) > 3.5/scale:
        return None
    return dict(h=h, score=score, coverage=coverage, quad=quad,
                inliers=int(keep.sum()), matches=len(good), method="sift_homography" if hasattr(cv2, "SIFT_create") else "orb_homography")


def _ncc_candidates(template, mask, gray, support, source_photo):
    """Finite coarse full-frame fallback, using only selected static appearance."""
    small, scale = _resize(gray, 1400)
    template_h = np.asarray(source_photo["template_to_source_photo_h"])
    corners = _project(template_h, _rect_corners([0, 0, template.shape[1]-1, template.shape[0]-1]))
    native = np.mean([np.linalg.norm(corners[1]-corners[0])/template.shape[1],
                      np.linalg.norm(corners[3]-corners[0])/template.shape[0]])
    candidates = []
    for factor in (.6, .75, .9, 1., 1.1, 1.25, 1.5, 1.8):
        ratio = native*factor*scale
        w, h = np.rint(np.array(template.shape[::-1])*ratio).astype(int)
        if min(w, h) < 10 or w > small.shape[1] or h > small.shape[0]:
            continue
        needle = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
        needle_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        response = cv2.matchTemplate(small, needle, cv2.TM_CCOEFF_NORMED, mask=needle_mask)
        response[~np.isfinite(response)] = -1
        for _ in range(2):
            _, score, _, location = cv2.minMaxLoc(response)
            if score < .78:
                break
            x, y = location
            transform = np.array([[w/template.shape[1]/scale, 0, x/scale],
                                  [0, h/template.shape[0]/scale, y/scale], [0, 0, 1.]])
            appearance, coverage = _appearance(template, mask, gray, support, transform)
            if appearance >= .75:
                quad = _project(transform, _rect_corners([0, 0, template.shape[1]-1, template.shape[0]-1]))
                candidates.append(dict(h=transform, score=appearance, coverage=coverage, quad=quad,
                                       method="multiscale_ncc", inliers=0, matches=0))
            response[max(0, y-h//2):y+h//2+1, max(0, x-w//2):x+w//2+1] = -1
    candidates.sort(key=lambda item: item["score"], reverse=True)
    unique = []
    for candidate in candidates:
        centre = candidate["quad"].mean(axis=0)
        if not any(np.linalg.norm(centre-old["quad"].mean(axis=0)) < .35*min(template.shape)*native for old in unique):
            unique.append(candidate)
    return unique


def recognize_regions(document, packet):
    """Recognize selected static areas across corrected full-RAW photographs."""
    document = _validate_document(document)
    image, support, camera_to_photo, photo = _packet(packet)
    run_id = str(packet.get("run_id") or Path(photo["source_raw"]["path"]).stem)
    _identifier(run_id, "Run ID")
    result = dict(run_id=run_id, photo=photo, observations={}, model=None,
                  proposal_only=True, status="unresolved")
    target_problem = _binding_problem(photo)
    if target_problem:
        result["status"] = "stale"
        for region in document["regions"]:
            result["observations"][region["id"]] = dict(status="stale", score=0., method="none",
                anchor_u_xy=None, reason=target_problem, region_identity=region.get("identity"),
                photo_identity=photo["identity"])
        if document.get('measurement_lines'):
            from .rig_measurement_lines import recognize_measurement_lines
            result['measurement_lines'] = recognize_measurement_lines(document,packet)
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    small, scale = _resize(gray, 3200)
    small_support = cv2.resize(support.astype(np.uint8)*255, small.shape[::-1], interpolation=cv2.INTER_NEAREST)
    detector = cv2.SIFT_create(nfeatures=16000, contrastThreshold=.025) if hasattr(cv2, "SIFT_create") else cv2.ORB_create(nfeatures=16000)
    target_points, target_descriptors = detector.detectAndCompute(small, small_support)
    target = (target_points, target_descriptors, scale)
    checked_sources = {}
    for region in document["regions"]:
        observation = dict(status="missing", score=0., method="none", anchor_u_xy=None,
                           region_identity=region.get("identity"), photo_identity=photo["identity"],
                           reason="No consistent static appearance found in the available RAW pixels")
        result["observations"][region["id"]] = observation
        try:
            template, mask = _validate_region(region)
            source = region["source"]
            key = source["identity"]
            if key not in checked_sources:
                checked_sources[key] = _binding_problem(source)
            problem = checked_sources[key]
            # The same physical path changing under a saved template is stale.
            if source["source_raw"]["path"] == photo["source_raw"]["path"] and source["source_raw"]["sha256"] != photo["source_raw"]["sha256"]:
                problem = "The target RAW changed at the saved template source path"
            if problem:
                observation.update(status="stale", reason=problem)
                continue
            candidate = _feature_candidate(template, mask, gray, support, detector, target)
            alternatives = []
            if candidate is not None:
                alternative = _feature_candidate(template, mask, gray, support, detector, target, candidate["quad"])
                if alternative is not None:
                    alternatives.append(alternative)
            else:
                alternatives = _ncc_candidates(template, mask, gray, support, region["template"])
                if alternatives:
                    candidate, alternatives = alternatives[0], alternatives[1:]
            if candidate is None:
                continue
            if any(other["score"] >= candidate["score"]-.06 for other in alternatives):
                observation.update(status="ambiguous", score=float(candidate["score"]), method=candidate["method"],
                                   reason="Several distinct static patches match this reference; choose a more distinctive area")
                continue
            photo_anchor = _project(candidate["h"], [region["template"]["anchor_template_xy"]])[0]
            anchor = _project(np.linalg.inv(camera_to_photo), [photo_anchor])[0]
            x, y = np.rint(photo_anchor).astype(int)
            available = 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and support[y, x]
            partial = candidate["coverage"] < .97 or not available
            observation.update(status="partial" if partial else "recognized", score=float(candidate["score"]),
                               method=candidate["method"], anchor_u_xy=anchor.tolist() if available else None,
                               coverage=float(candidate["coverage"]), inliers=candidate["inliers"],
                               matches=candidate["matches"], quad_photo_xy=candidate["quad"].tolist(),
                               reason="Reference is partly outside available RAW support; manual review is required" if partial else "Static template appearance and geometric consistency verified")
        except (ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as error:
            observation.update(status="invalid", reason=str(error))
    from .rig_measurement_lines import recognize_measurement_lines, fit_measurement_reference_model
    if document.get('measurement_lines'):
        result['measurement_lines'] = recognize_measurement_lines(document,packet)
    planes = sorted({region["plane_id"] for region in document["regions"]
                     if region.get('purpose') != 'measurement_endpoint'} |
                    {line['plane_id'] for line in document.get('measurement_lines', [])})
    models = {plane: fit_measurement_reference_model(document, result['observations'],
                result.get('measurement_lines', {}), plane_id=plane) for plane in planes}
    result["plane_models"] = models
    valid = [model for model in models.values() if model["status"] == "valid"]
    if len(valid) == 1:
        result["model"] = valid[0]
    result["status"] = "proposal" if valid else "unresolved"
    return result


def _validate_document(document):
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ValueError("Unsupported static-reference document format")
    _measurement_uncertainty(document.get("measurement_uncertainty_mm", 0.))
    if 'rig_origin_xy_mm' in document:
        origin = _array(document['rig_origin_xy_mm'], (2,), 'Rig origin x/y')
        if np.any(np.abs(origin) > 1e6):
            raise ValueError('Rig origin coordinates must be within 1000000 mm')
    if 'model_plane_id' in document:
        _identifier(document['model_plane_id'], 'Model depth plane ID')
    if 'measurement_lines' in document:
        from .rig_measurement_lines import validate_measurement_lines
        validate_measurement_lines(document['measurement_lines'])
    if 'guides' in document:
        from .rig_guide_lines import validate_guides
        validate_guides(document['guides'])
    if 'guides_visible' in document and type(document['guides_visible']) is not bool:
        raise ValueError('Support-line visibility must be a boolean')
    regions, runs = document.get("regions"), document.get("runs")
    if not isinstance(regions, list) or len(regions) > MAX_REGIONS:
        raise ValueError(f"Static references are limited to {MAX_REGIONS} regions")
    if not isinstance(runs, dict) or len(runs) > MAX_RUNS:
        raise ValueError(f"Static references are limited to {MAX_RUNS} saved runs")
    seen = set()
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("Each static reference must be an object")
        key = _identifier(region.get("id"), "Region ID")
        if key in seen:
            raise ValueError(f"Duplicate physical reference ID: {key}")
        seen.add(key)
    for key, run in runs.items():
        _identifier(key, "Run ID")
        if not isinstance(run, dict) or not isinstance(run.get("observations"), dict):
            raise ValueError(f"Saved run {key} needs an observation object")
        if len(run["observations"]) > MAX_REGIONS:
            raise ValueError(f"Saved run {key} exceeds its observation bound")
        if 'measurement_lines' in run:
            from .rig_measurement_lines import MAX_LINES
            if (not isinstance(run['measurement_lines'], dict) or len(run['measurement_lines']) > MAX_LINES
                    or not all(isinstance(item, dict) for item in run['measurement_lines'].values())):
                raise ValueError(f'Saved run {key} needs bounded measurement-line observations')
    try:
        encoded = json.dumps(document, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise ValueError("Static references must contain finite JSON values") from error
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise ValueError("The static-reference document exceeds its byte bound")
    return deepcopy(document)


def _invalidate(document):
    problems = {}
    checked = {}
    for region in document["regions"]:
        problem = None
        try:
            _validate_region(region)
            identity = region["source"]["identity"]
            if identity not in checked:
                checked[identity] = _binding_problem(region["source"])
            problem = checked[identity]
        except (ValueError, KeyError, cv2.error) as error:
            problem = str(error)
        if problem:
            problems[region["id"]] = problem
    current = {region["id"]: region.get("identity") for region in document["regions"]}
    lines = {line['id']: line for line in document.get('measurement_lines', [])}
    line_problems = {}
    for key, line in lines.items():
        identity = line['source']['identity']
        if identity not in checked:
            checked[identity] = _binding_problem(line['source'])
        if checked[identity]:
            line_problems[key] = checked[identity]
    for run in document["runs"].values():
        invalid = False
        photo = run.get("photo", {})
        photo_identity = photo.get("identity")
        photo_problem = None
        if not photo_identity or photo_identity != _digest({k: v for k, v in photo.items() if k != "identity"}):
            photo_problem = "Saved run display identity changed"
        else:
            if photo_identity not in checked:
                checked[photo_identity] = _binding_problem(photo)
            photo_problem = checked[photo_identity]
        for key, observation in run["observations"].items():
            reason = photo_problem or problems.get(key)
            if key not in current or observation.get("region_identity") != current.get(key):
                reason = "Saved observation belongs to a removed or edited physical reference"
            if observation.get("photo_identity") != photo_identity:
                reason = "Saved observation belongs to a different source display"
            if reason:
                observation.update(status="stale", anchor_u_xy=None, reason=reason)
                invalid = True
        for key, observation in run.get('measurement_lines', {}).items():
            line = lines.get(key)
            reason = photo_problem or line_problems.get(key)
            if line is None or observation.get('tracking_identity') != line['tracking_identity']:
                reason = 'Saved endpoints belong to a removed or redrawn measurement line'
            if observation.get('photo_identity') != photo_identity:
                reason = 'Saved endpoints belong to a different source display'
            if reason:
                observation.update(status='stale', points_u_xy=None, reason=reason)
                invalid = True
        from .rig_measurement_lines import _geometry_identity, measurement_metric_points
        candidates = list(run.get('plane_models', {}).values())
        if run.get('model') is not None:
            candidates.append(run['model'])
        for model in candidates:
            plane = model.get('plane_id', 'rig')
            expected = model.get('geometry_reference_identity')
            has_placed_lines = any(line['plane_id'] == plane and measurement_metric_points(line) is not None
                                   for line in lines.values())
            if ((expected is not None and expected != _geometry_identity(document, plane))
                    or (has_placed_lines and expected is None)):
                invalid = True
        if invalid or photo_problem:
            run.update(model=None, plane_models={}, status="stale")
    if problems:
        document["validation"] = dict(status="stale", region_errors=problems)
    else:
        document["validation"] = dict(status="valid", region_errors={})
    if 'measurement_lines' in document:
        document['validation']['line_errors'] = line_problems
        if line_problems:
            document['validation']['status'] = 'stale'
    return document


def _path(project_root):
    return Path(project_root).resolve() / "annotations" / "rig_static_references" / "references.json"


def load_references(project_root):
    path = _path(project_root)
    if not path.exists():
        return empty_document()
    with path.open("rb") as stream:
        encoded = stream.read(MAX_DOCUMENT_BYTES+1)
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise ValueError("The static-reference document exceeds its byte bound")
    try:
        document = json.loads(encoded)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"The static-reference JSON is invalid: {path}") from error
    return _invalidate(_validate_document(document))


def save_references(project_root, document, *, actor, reason="Save static rig references"):
    """Atomically save a bounded annotation document after an explicit action.

    ``actor`` is who saves; the application takes it from the series
    configuration (series_decisions.series_actor).
    """
    if not isinstance(actor, str) or not actor.strip() or len(actor) > 256:
        raise ValueError("The explicit save needs an actor of at most 256 characters")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2048:
        raise ValueError("The explicit save needs a reason of at most 2048 characters")
    value = _validate_document(document)
    identities = {region["id"]: region.get("identity") for region in value["regions"]}
    for run in value["runs"].values():
        for key, observation in run["observations"].items():
            if observation.get("status") == "manual" and key in identities:
                observation.setdefault("region_identity", identities[key])
                observation.setdefault("photo_identity", run.get("photo", {}).get("identity"))
    value = _invalidate(value)
    history = value.get("history", [])
    if not isinstance(history, list):
        raise ValueError("Static-reference history must be a list")
    value["history"] = history[-99:] + [dict(actor=actor.strip(), reason=reason.strip(),
        saved_at=datetime.now(timezone.utc).isoformat())]
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise ValueError("The static-reference document exceeds its byte bound")
    path = _path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".references-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return value
