"""FF_UM/FF_BM review regions on a separately bound full-frame RAW grid.

Historical canvases are not sensor grids. The caller must supply the complete
new-display -> old-content coordinate map, including the old photograph grid,
resize, per-frame drift and curvature. No missing transform defaults to identity.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..phase_training.review_regions import annotation_mask

DATA_ROOT_ENV = "DARSIA_FLUIDFLOWER_DATA_ROOT"


def shared_data_root():
    """The configured shared data folder that holds the series folders.

    ``DARSIA_FLUIDFLOWER_DATA_ROOT`` names it (a drive or share path); no
    storage host is assumed and a missing setting is refused.
    """
    value = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not value:
        raise RuntimeError(f"Set {DATA_ROOT_ENV} to the shared data folder that holds the series folders.")
    return Path(value)


def shared_drive_path(path):
    """Map a path recorded on the shared data drive ('Z:\\...') onto the configured folder.

    Other paths are returned unchanged.
    """
    text = str(path)
    if not text.casefold().startswith("z:\\"):
        return Path(text)
    return shared_data_root() / text[3:]


def _series(value):
    value = str(value).casefold().replace("_", "")
    if value not in {"ffum", "ffbm"}:
        raise ValueError("This review adapter is only for FF_UM and FF_BM")
    return value


def _dimensions(value, length):
    if len(value) != length or any(type(v) is not int or v <= 0 for v in value):
        raise ValueError("Review dimensions must be positive integers")
    return tuple(value)


@dataclass(frozen=True)
class ReviewFrame:
    series_id: str
    run: str
    frame_index: int
    source: dict
    manifest: dict
    manifest_path: str
    manifest_sha256: str

    @property
    def shape_hw(self):
        return tuple(self.manifest["analysis_image_shape_hw"])


@dataclass(frozen=True)
class ReviewRegion:
    """Original labels, IDs and any explicit split remain in record, unchanged."""
    record: dict
    frame: ReviewFrame
    extraction_record: dict | None = None


@dataclass(frozen=True)
class TransferredRegion:
    region: ReviewRegion
    mask: np.ndarray


def bind_review_frame(manifest_bytes, *, expected_sha256, manifest_path,
                      series_id, run, frame_index, source_name):
    """Bind a historical frame to the exact manifest bytes used in extraction."""
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Historical review manifest checksum differs: {manifest_path}")
    manifest = json.loads(manifest_bytes)
    series_id = _series(series_id)
    if _series(manifest["dataset"]) != series_id or manifest["run"] != run:
        raise ValueError("Historical review series/run differs from extraction")
    width, height = _dimensions(manifest["canvas_wh"], 2)
    rows, columns = _dimensions(manifest["analysis_image_shape_hw"], 2)
    box = manifest["content_box_xywh"]
    if (len(box) != 4 or any(type(v) is not int for v in box)
            or box[:2] != [max(0, box[0]), max(0, box[1])]
            or box[2:] != [columns, rows] or box[0] + columns > width
            or box[1] + rows > height or manifest.get("full_analysis_resolution") is not True
            or manifest.get("additional_spatial_resampling") is not False):
        raise ValueError("Review content needs its exact unresampled canvas placement")
    sources = [row for row in manifest["frame_sources"] if row["frame_index"] == frame_index]
    if len(sources) != 1 or sources[0]["source_name"] != source_name:
        raise ValueError("Historical source photograph/frame differs from extraction")
    return ReviewFrame(series_id, run, frame_index, deepcopy(sources[0]), manifest,
                       str(manifest_path), digest)


def bind_review_region(record, frame, *, extraction_record=None):
    """Accept the actual evidence payload, or a validation annotation directly.

    The extraction frame is authoritative after historical timeline alignment;
    original_frame_index and any split/fold fields are retained, never rewritten.
    """
    _series(frame.series_id)
    if "dataset" in record and _series(record["dataset"]) != frame.series_id:
        raise ValueError("Region and review frame belong to different series")
    if "run" in record and record["run"] != frame.run:
        raise ValueError("Region and review frame belong to different runs")
    payload = record.get("payload", record)
    if not (payload.get("id") or payload.get("proposal_id")):
        raise ValueError("The original annotation ID is required")
    if payload.get("coordinate_space") != "normalized_native_review_canvas_16x9":
        raise ValueError("Unsupported historical annotation coordinate space")
    points = payload.get("points") or []
    if points:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3 or not np.isfinite(values).all():
            raise ValueError("A review polygon needs at least three finite xy points")
    else:
        if payload.get("shape") not in {"box", "rectangle"}:
            raise ValueError("A freehand review region must retain its polygon")
        values = np.asarray([payload[key] for key in ("x", "y", "width", "height")], dtype=float)
        if not np.isfinite(values).all() or np.any(values[2:] <= 0):
            raise ValueError("A review box needs finite coordinates and positive size")
    return ReviewRegion(deepcopy(record), frame, deepcopy(extraction_record))


def load_model_regions(model_root, *, series_id, resolve_path=Path):
    """Read the captured 7e879bea/2c8a15b8 evidence formats without changing them.

    Resolution of moved manifest paths is explicit and still SHA-checked.
    Included and unsupervised regions are both retained with their saved status.
    No train/validation partition is invented when none was recorded.
    """
    series_id = _series(series_id)
    root = Path(model_root)
    protocol = json.loads((root / "regional_training_protocol.json").read_bytes())
    documents = {}
    for name in ("evidence", "extraction"):
        raw = (root / f"{name}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != protocol[f"{name}_sha256"]:
            raise ValueError(f"Captured {name} differs from the saved training protocol")
        documents[name] = json.loads(raw)
    manifests, bindings = {}, {}
    for extracted in documents["extraction"]["frames"]:
        path = extracted["native_manifest"]
        if path not in manifests:
            manifests[path] = Path(resolve_path(path)).read_bytes()
        frame = bind_review_frame(manifests[path], expected_sha256=extracted["inputs_sha256"][path],
            manifest_path=path, series_id=series_id, run=extracted["run"],
            frame_index=extracted["frame_index"], source_name=extracted["source_name"])
        for record in extracted["records"]:
            identity = record["region_id"]
            if identity in bindings:
                raise ValueError(f"Ambiguous historical frame for region {identity}")
            bindings[identity] = frame, record
    regions, seen = [], set()
    for record in documents["evidence"]["regions"]:
        identity = record["region_id"]
        if identity in seen or identity not in bindings:
            raise ValueError(f"Region needs one historical extraction binding: {identity}")
        seen.add(identity)
        frame, extracted = bindings[identity]
        regions.append(bind_review_region(record, frame, extraction_record=extracted))
    return regions


def transfer_region(region, *, new_to_content_xy, valid_support, chunk_rows=128):
    """Pull back the exact old raster region; never replace it with a bounding box.

    ``new_to_content_xy`` accepts Nx2 display pixel centres and returns Nx2
    historical content pixel centres (canvas offset already removed). NaN marks
    unavailable inverse geometry. Pixel-cell membership uses nearest neighbour;
    the original polygon rasterizer and the target's valid support are retained.
    """
    _series(region.frame.series_id)
    support = np.asarray(valid_support)
    if support.ndim != 2 or support.dtype != np.dtype(bool) or not all(support.shape):
        raise ValueError("The target full-frame valid support must be a nonempty boolean image")
    if type(chunk_rows) is not int or chunk_rows <= 0 or not callable(new_to_content_xy):
        raise ValueError("An explicit complete coordinate map and positive chunk size are required")
    manifest = region.frame.manifest
    old_mask = annotation_mask(region.record.get("payload", region.record),
        canvas_wh=manifest["canvas_wh"], content_box=manifest["content_box_xywh"])
    result = np.zeros(support.shape, dtype=bool)
    old_height, old_width = old_mask.shape
    for start in range(0, support.shape[0], chunk_rows):
        stop = min(start + chunk_rows, support.shape[0])
        y, x = np.nonzero(support[start:stop])
        if not len(x):
            continue
        points = np.asarray(new_to_content_xy(np.column_stack((x, y + start))), dtype=float)
        if points.shape != (len(x), 2):
            raise ValueError("The coordinate map must return one xy point per target pixel")
        valid = (np.isfinite(points).all(axis=1) & (points[:, 0] >= -.5)
            & (points[:, 0] < old_width - .5) & (points[:, 1] >= -.5)
            & (points[:, 1] < old_height - .5))
        pixels = np.floor(points[valid] + .5).astype(np.int64)
        result[y[valid] + start, x[valid]] = old_mask[pixels[:, 1], pixels[:, 0]]
    return TransferredRegion(region, result)


def capture_legacy_drift(correction, image):
    """Capture the matrix actually used by an existing isolated legacy call.

    The caller loads the original correction/image and establishes its original
    RNG/runtime first. This does not fit another registration method, write any
    file, or mutate a shared class. The instance hook is restored even on failure.
    """
    if correction.active is False:
        return correction.correct_array(image), np.eye(3)[:2]
    estimator = correction.translation_estimator
    original = estimator.find_effective_translation
    local = estimator.__dict__.get("find_effective_translation")
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if result[1]:
            captured.append(np.array(result[0], dtype=float, copy=True))
        return result

    estimator.find_effective_translation = capture
    try:
        registered = correction.correct_array(image)
    finally:
        if local is None:
            del estimator.find_effective_translation
        else:
            estimator.find_effective_translation = local
    if len(captured) != 1 or captured[0].shape != (2, 3) or not np.isfinite(captured[0]).all():
        raise ValueError("The legacy drift call did not provide exactly one finite affine matrix")
    return registered, captured[0]


def legacy_content_to_photo_xy(points, *, curvature_cache, drift_matrix,
                               photo_wh, resized_wh):
    """Compose saved curvature RC coordinates, inverse drift and inverse resize.

    Integer content centres use the exact stored grid; fractional coordinates
    bilinearly interpolate that grid. Dimensions must describe the old JPEG/TIFF,
    not an assumed RAW sensor. This alone is not the sensor/full-frame mapping.
    """
    photo_wh, resized_wh = _dimensions(photo_wh, 2), _dimensions(resized_wh, 2)
    shape = _dimensions(curvature_cache["shape"], 2)
    grid = np.asarray(curvature_cache["grid"])
    if grid.shape != (2, shape[0] * shape[1]):
        raise ValueError("Saved curvature grid does not match its declared content shape")
    matrix = np.asarray(drift_matrix, dtype=float)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("The actual frame drift matrix is required")
    inverse = np.linalg.inv(np.vstack((matrix, [0, 0, 1])))
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Content coordinates must be Nx2 xy points")
    valid = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0)
        & (points[:, 0] <= shape[1] - 1) & (points[:, 1] >= 0) & (points[:, 1] <= shape[0] - 1))
    output = np.full(points.shape, np.nan)
    q = points[valid]
    x0, y0 = np.floor(q).astype(int).T
    x1, y1 = np.minimum(x0 + 1, shape[1] - 1), np.minimum(y0 + 1, shape[0] - 1)
    dx, dy = (q - np.column_stack((x0, y0))).T
    registered = ((1-dx)*(1-dy)*grid[:, y0*shape[1]+x0]
        + dx*(1-dy)*grid[:, y0*shape[1]+x1]
        + (1-dx)*dy*grid[:, y1*shape[1]+x0] + dx*dy*grid[:, y1*shape[1]+x1])
    registered_xy = registered[::-1].T
    resized = registered_xy @ inverse[:2, :2].T + inverse[:2, 2]
    output[valid] = (resized + .5) * (np.array(photo_wh) / resized_wh) - .5
    return output
