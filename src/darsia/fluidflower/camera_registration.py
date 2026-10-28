"""Register photographs of one rig taken from slightly different camera positions.

Between campaigns a camera on its stand may move by some tens of pixels and turn
by a fraction of a degree, and the sand may settle. Marks made on the raw
photograph of one run (the rig frame, the outlines of the sand layers) are then
carried over to another run's photograph with a map fitted to matched SIFT
features (RANSAC). Two fits serve different marks:

* :func:`register_photographs` matches the whole photographs. Most features lie
  in the sand, so the homography follows the sand: the camera move together
  with any settling of the packing. It maps layer outlines.
* :func:`register_frame_edges` matches strips along the rig frame at full
  resolution. The affine map describes the camera move alone and maps the
  frame's corners.

Both maps take (row, col) pixels of the source photograph to the target
photograph.

Status: checked on FF_RE's baselines on 12 Oct 2026 (results and scripts in
X:\\FF_reuse\\DarSIA\\results\\camera_registration); no tests yet. Not yet
written: copying a run's painting session and rig frame to another run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FRAME_SIDES = ("top", "bottom", "left", "right")
# The corners each side of the frame runs between.
_SIDE_CORNERS = {
    "top": ("top_left", "top_right"),
    "bottom": ("bottom_left", "bottom_right"),
    "left": ("top_left", "bottom_left"),
    "right": ("top_right", "bottom_right"),
}


@dataclass(frozen=True)
class CameraRegistration:
    """A map from a source photograph's pixels to a target photograph's.

    ``matrix`` is a 3x3 projective matrix acting on (x, y, 1) = (col, row, 1);
    an affine fit has (0, 0, 1) as its last row. ``kind`` is ``"homography"``
    or ``"affine"``.
    """

    matrix: np.ndarray
    kind: str
    matches: int
    inliers: int
    rms_px: float
    p95_px: float
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    side_inliers: dict[str, int] = field(default_factory=dict)

    def map_points(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Source (row, col) points in the target photograph, as (row, col)."""

        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.column_stack(
            [array[:, 1], array[:, 0], np.ones(len(array), dtype=np.float64)]
        )
        mapped = homogeneous @ np.asarray(self.matrix, dtype=np.float64).T
        mapped = mapped[:, :2] / mapped[:, 2:3]
        return np.column_stack([mapped[:, 1], mapped[:, 0]])

    def shift_px(self, points: Sequence[Sequence[float]]) -> float:
        """Largest displacement of the given (row, col) points, in pixels."""

        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not len(array):
            return 0.0
        return float(np.max(np.linalg.norm(self.map_points(array) - array, axis=1)))

    def scale(self) -> float:
        """Mean linear scale of the map (1 for a pure move)."""

        linear = np.asarray(self.matrix, dtype=np.float64)[:2, :2]
        return float(np.sqrt(abs(np.linalg.det(linear))))

    def rotation_deg(self) -> float:
        linear = np.asarray(self.matrix, dtype=np.float64)[:2, :2]
        return float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "matrix": [[float(v) for v in row] for row in np.asarray(self.matrix)],
            "matches": int(self.matches),
            "inliers": int(self.inliers),
            "rms_px": round(float(self.rms_px), 3),
            "p95_px": round(float(self.p95_px), 3),
            "rotation_deg": round(self.rotation_deg(), 4),
            "scale": round(self.scale(), 6),
            "source_size": list(self.source_size),
            "target_size": list(self.target_size),
            "side_inliers": dict(self.side_inliers),
        }


def _read_gray(path: str | Path) -> np.ndarray:
    import cv2

    # np.fromfile + imdecode also reads network and non-ASCII paths on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read the photograph {path}")
    return image


def _equalize(image: np.ndarray, tile: int = 8) -> np.ndarray:
    import cv2

    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile, tile)).apply(
        np.ascontiguousarray(image)
    )


def _features(image: np.ndarray, max_features: int, contrast: float = 0.04):
    import cv2

    sift = cv2.SIFT_create(nfeatures=int(max_features), contrastThreshold=contrast)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    points = np.float32([point.pt for point in keypoints]).reshape(-1, 2)
    return points, descriptors


def _match(source_descriptors, target_descriptors, ratio: float) -> np.ndarray:
    """Index pairs (source, target) that pass Lowe's ratio test."""

    import cv2

    if (
        source_descriptors is None
        or target_descriptors is None
        or len(source_descriptors) < 2
        or len(target_descriptors) < 2
    ):
        return np.zeros((0, 2), dtype=np.int64)
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    pairs = []
    for candidates in matcher.knnMatch(source_descriptors, target_descriptors, k=2):
        if len(candidates) == 2 and candidates[0].distance < ratio * candidates[1].distance:
            pairs.append((candidates[0].queryIdx, candidates[0].trainIdx))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _errors(matrix: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([source, np.ones(len(source))])
    mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return np.linalg.norm(mapped - target, axis=1)


def register_photographs(
    source: str | Path,
    target: str | Path,
    *,
    scale: float = 0.5,
    max_features: int = 40000,
    ratio: float = 0.75,
    threshold_px: float = 3.0,
    min_inliers: int = 50,
) -> CameraRegistration:
    """Homography from the source photograph to the target, from all features.

    Features are detected at ``scale`` of the full resolution after local
    contrast equalisation; the homography is fitted in full-resolution pixels.
    """

    import cv2

    source_image = _read_gray(source)
    target_image = _read_gray(target)
    sizes = (
        (int(source_image.shape[1]), int(source_image.shape[0])),
        (int(target_image.shape[1]), int(target_image.shape[0])),
    )
    if scale != 1.0:
        source_image = cv2.resize(
            source_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        target_image = cv2.resize(
            target_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    source_points, source_descriptors = _features(_equalize(source_image), max_features)
    target_points, target_descriptors = _features(_equalize(target_image), max_features)
    del source_image, target_image
    pairs = _match(source_descriptors, target_descriptors, ratio)
    if len(pairs) < max(4, min_inliers):
        raise ValueError(
            f"Only {len(pairs)} features of the two photographs match; they may not "
            "show the same rig."
        )
    from_points = source_points[pairs[:, 0]] / scale
    to_points = target_points[pairs[:, 1]] / scale
    matrix, mask = cv2.findHomography(
        from_points,
        to_points,
        cv2.RANSAC,
        threshold_px,
        maxIters=20000,
        confidence=0.999,
    )
    if matrix is None or mask is None:
        raise ValueError("No consistent camera map was found for the two photographs.")
    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < min_inliers:
        raise ValueError(
            f"Only {int(inliers.sum())} features agree on one camera map; the "
            "photographs may not show the same rig."
        )
    errors = _errors(matrix, from_points[inliers], to_points[inliers])
    return CameraRegistration(
        matrix=np.asarray(matrix, dtype=np.float64) / float(matrix[2, 2]),
        kind="homography",
        matches=int(len(pairs)),
        inliers=int(inliers.sum()),
        rms_px=float(np.sqrt(np.mean(errors**2))),
        p95_px=float(np.percentile(errors, 95)),
        source_size=sizes[0],
        target_size=sizes[1],
    )


def _side_boxes(
    corners: Mapping[str, Sequence[float]],
    shape: tuple[int, int],
    margin: float,
) -> dict[str, tuple[slice, slice]]:
    height, width = shape
    boxes = {}
    for side in FRAME_SIDES:
        first, second = (corners[name] for name in _SIDE_CORNERS[side])
        rows = (min(first[0], second[0]) - margin, max(first[0], second[0]) + margin)
        cols = (min(first[1], second[1]) - margin, max(first[1], second[1]) + margin)
        row_slice = slice(max(0, int(rows[0])), min(height, int(np.ceil(rows[1]))))
        col_slice = slice(max(0, int(cols[0])), min(width, int(np.ceil(cols[1]))))
        if row_slice.stop - row_slice.start >= 16 and col_slice.stop - col_slice.start >= 16:
            boxes[side] = (row_slice, col_slice)
    return boxes


def register_frame_edges(
    source: str | Path,
    target: str | Path,
    corners: Mapping[str, Sequence[float]],
    *,
    margin_px: float = 110.0,
    search_px: float = 80.0,
    max_features: int = 20000,
    ratio: float = 0.75,
    threshold_px: float = 2.0,
    min_side_inliers: int = 8,
    min_sides: int = 3,
) -> CameraRegistration:
    """Affine camera move from the rig frame, matched side by side.

    ``corners`` are the frame's (row, col) corners in the source photograph
    (``top_left``, ``top_right``, ``bottom_right``, ``bottom_left``). Features
    are matched only within the strip along each side, ``margin_px`` wide in
    the source and ``margin_px + search_px`` in the target, at full resolution.
    The fit needs ``min_side_inliers`` agreeing features on at least
    ``min_sides`` sides; otherwise a ValueError says so.
    """

    import cv2

    source_image = _read_gray(source)
    target_image = _read_gray(target)
    sizes = (
        (int(source_image.shape[1]), int(source_image.shape[0])),
        (int(target_image.shape[1]), int(target_image.shape[0])),
    )
    source_boxes = _side_boxes(corners, source_image.shape, margin_px)
    target_boxes = _side_boxes(corners, target_image.shape, margin_px + search_px)
    from_points, to_points, sides = [], [], []
    for side, (rows, cols) in source_boxes.items():
        if side not in target_boxes:
            continue
        target_rows, target_cols = target_boxes[side]
        points_a, descriptors_a = _features(
            _equalize(source_image[rows, cols]), max_features, contrast=0.02
        )
        points_b, descriptors_b = _features(
            _equalize(target_image[target_rows, target_cols]),
            max_features,
            contrast=0.02,
        )
        pairs = _match(descriptors_a, descriptors_b, ratio)
        if not len(pairs):
            continue
        from_points.append(points_a[pairs[:, 0]] + (cols.start, rows.start))
        to_points.append(points_b[pairs[:, 1]] + (target_cols.start, target_rows.start))
        sides.extend([side] * len(pairs))
    del source_image, target_image
    if not sides:
        raise ValueError("No features of the rig frame match between the photographs.")
    origin = np.concatenate(from_points).astype(np.float32)
    destination = np.concatenate(to_points).astype(np.float32)
    side_array = np.asarray(sides)
    affine, mask = cv2.estimateAffine2D(
        origin,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold_px,
        maxIters=20000,
        confidence=0.999,
    )
    if affine is None or mask is None:
        raise ValueError("The rig frame gives no consistent camera move.")
    inliers = mask.ravel().astype(bool)
    side_inliers = {
        side: int(np.count_nonzero(inliers & (side_array == side))) for side in FRAME_SIDES
    }
    supported = sum(count >= min_side_inliers for count in side_inliers.values())
    if supported < min_sides:
        raise ValueError(
            "Too few features of the rig frame agree on one camera move "
            f"({', '.join(f'{side} {count}' for side, count in side_inliers.items())})."
        )
    matrix = np.vstack([np.asarray(affine, dtype=np.float64), [0.0, 0.0, 1.0]])
    errors = _errors(matrix, origin[inliers], destination[inliers])
    return CameraRegistration(
        matrix=matrix,
        kind="affine",
        matches=int(len(sides)),
        inliers=int(inliers.sum()),
        rms_px=float(np.sqrt(np.mean(errors**2))),
        p95_px=float(np.percentile(errors, 95)),
        source_size=sizes[0],
        target_size=sizes[1],
        side_inliers=side_inliers,
    )


__all__ = [
    "FRAME_SIDES",
    "CameraRegistration",
    "register_frame_edges",
    "register_photographs",
]
