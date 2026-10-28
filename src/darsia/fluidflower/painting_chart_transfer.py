"""Local chart occlusion over a continuous painted material map.

The drawing, camera geometry and detected patch centres are inputs. No layer,
camera registration or material property is fitted here. A proposal becomes
usable only after the human confirms its exact displayed source and pixels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json

import numpy as np

LEGACY_FORMAT = "darsia-chart-transfer-proposal-v1"
LEGACY_APPROVAL_FORMAT = "darsia-chart-transfer-approval-v1"
FORMAT = "darsia-chart-transfer-proposal-v2"
APPROVAL_FORMAT = "darsia-chart-transfer-approval-v2"
DETECTED_FORMAT = "darsia-chart-transfer-proposal-v3"
DETECTED_APPROVAL_FORMAT = "darsia-chart-transfer-approval-v3"
KEY = "chart_transfers"
MOVED_RUNS = frozenset(("ac23", "ac25", "ac26", "ac28", "ac29", "ac30", "ac31",
    "ac32", "ac33", "ac34", "ac35", "ac38", "ac39", "ac40"))
RUNS = ("ac17", "ac18", "ac19", "ac21", "ac22", "ac23", "ac25", "ac26", "ac28",
    "ac29", "ac30", "ac31", "ac32", "ac33", "ac34", "ac35", "ac38", "ac39", "ac40",
    "ac41", "ac42", "ac43", "ac44", "ac45", "ac46", "ac47", "ac48", "ac49", "ac50",
    "ac52", "ac53", "ac54", "ac55", "ac56", "ac57", "ac58", "ac59", "ac60", "ac61")
MARGIN_MM = 5.0


def fresh_transfer_policy():
    """Explicit policy written only to new immutable review records."""
    return dict(format="darsia-full-photo-detected-chart-transfer-v1",
        material_display="full-source-to-full-camera-photo-v1",
        chart_motion="ordered-centres-rigid-unit-scale-all-runs-v1", margin_mm=MARGIN_MM)


def review_proposal_format(review):
    """An absent policy retains the frozen v2 experiment-cohort semantics."""
    policy = review.get('transfer_policy')
    if policy is None:
        return FORMAT
    if policy != fresh_transfer_policy():
        raise ValueError("The saved transfer review has an unsupported explicit policy")
    return DETECTED_FORMAT


def _json(value):
    from .painted_review_comparison import _json as literal
    return json.dumps(literal(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf8")


def _identity(array):
    from .geometry_raster_sources import _carried_array_identity
    return _carried_array_identity(array)


def _matrix(value):
    from .map_placement import _matrix as original
    return original(value, "ColorChecker transfer")


def _project(points, matrix):
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 2 or
            not np.issubdtype(points.dtype, np.floating) or not np.isfinite(points).all()):
        raise ValueError("ColorChecker coordinates must be finite floating point xy pairs")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ _matrix(matrix).T
    if (not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] == 0) or
            (len(points) and homogeneous[:, 2].min() < 0 < homogeneous[:, 2].max())):
        raise ValueError("The ColorChecker projects outside the finite measured grid")
    result = homogeneous[:, :2] / homogeneous[:, 2, None]
    if not np.isfinite(result).all():
        raise ValueError("The ColorChecker projection is not finite")
    return result


def _centres(value):
    value = np.asarray(value)
    if value.shape != (24, 2) or value.dtype != np.float64 or not np.isfinite(value).all():
        raise ValueError("The original ordered 24 ColorChecker centres must be float64 xy")
    return value


def _translation_motion(reference_centres_mm, target_centres_mm):
    """Original v1 centroid-only rule, retained exactly for saved approvals."""
    reference, target = map(_centres, (reference_centres_mm, target_centres_mm))
    before, after = reference.mean(0), target.mean(0)
    delta = after - before
    if not np.isfinite(delta).all():
        raise ValueError("The measured ColorChecker displacement is not finite")
    return dict(dx_mm=float(delta[0]), dy_up_mm=float(delta[1]),
        source_centroid_mm=before.tolist(), target_centroid_mm=after.tolist(),
        coordinate_frame="current measured metric plane; x right, y up; millimetres",
        rotation_applied=False)


def chart_motion(reference_centres_mm, target_centres_mm):
    """Least-squares proper rigid motion of ordered centres, with no scale fit."""
    reference, target = map(_centres, (reference_centres_mm, target_centres_mm))
    result = _translation_motion(reference, target)
    before, after = reference-reference.mean(0), target-target.mean(0)
    if np.linalg.matrix_rank(before) != 2 or np.linalg.matrix_rank(after) != 2:
        raise ValueError("Rigid ColorChecker motion needs non-collinear ordered centres")
    u, _, vt = np.linalg.svd(before.T @ after)
    proper = np.diag([1., 1. if np.linalg.det(vt.T @ u.T) > 0 else -1.])
    rotation = vt.T @ proper @ u.T
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    if abs(angle) < 1e-12:
        angle, rotation = 0., np.eye(2)
    residual = before @ rotation.T - after
    return dict(result, rotation_applied=bool(angle), rotation_degrees=angle,
        rotation_matrix=rotation.tolist(), fit_rms_mm=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        motion_model="proper rigid rotation about source centroid, then centroid translation; scale fixed at one")


def proposal_format_for_approval(approval):
    """Choose explicit historical replay semantics; never upgrade an approval."""
    formats = {LEGACY_APPROVAL_FORMAT: LEGACY_FORMAT, APPROVAL_FORMAT: FORMAT,
               DETECTED_APPROVAL_FORMAT: DETECTED_FORMAT}
    if not isinstance(approval, dict) or approval.get('format') not in formats:
        raise ValueError("The ColorChecker approval has an unsupported replay format")
    return formats[approval['format']]


def _labels(value, name, *, original_integer=False):
    value = np.asarray(value)
    valid_dtype = (np.issubdtype(value.dtype, np.integer) if original_integer
                   else value.dtype == np.uint8)
    if (value.ndim != 2 or not valid_dtype or not all(value.shape) or
            np.any(value < 0) or np.any(value > 255)):
        raise ValueError(f"{name} must be its original integer layer grid with IDs 0..255")
    return value


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.dtype != bool or value.shape != shape:
        raise ValueError(f"{name} must be a bool mask on its original grid")
    return value


def _roles(known_ids, chart_id):
    if (type(chart_id) is not int or not 0 < chart_id < 256 or
            not isinstance(known_ids, (list, tuple)) or not known_ids or
            any(type(value) is not int or not 0 < value < 256 for value in known_ids) or
            len(set(known_ids)) != len(known_ids) or chart_id not in known_ids):
        raise ValueError("Original explicit layer identities are required")
    return set(known_ids)


def restore_chart_material(source_labels, drawn_underlay, whole_chart_mask, *,
                           chart_id, known_ids):
    """Restore the source drawing once, before the unchanged per-run carry.

    Explicit non-chart drawings win. Otherwise each complete horizontal chart
    span takes the original pixel immediately to its LEFT, on the same AC17
    source row. Zero stays unassigned; there is no vertical or nearest-sand
    search, classifier, new boundary fit or material-property inference.
    """
    labels = _labels(source_labels, "Source labels")
    drawn = _labels(drawn_underlay, "Drawn underlay")
    if drawn.shape != labels.shape:
        raise ValueError("Source drawing and underlay must share their grid")
    whole = _mask(whole_chart_mask, labels.shape, "Whole manual ColorChecker")
    ids = _roles(known_ids, chart_id)
    if (not whole.any() or np.any((labels == chart_id) & ~whole) or
            np.any(~np.isin(labels, [0, *ids])) or
            np.any(~np.isin(drawn, [0, *ids])) or np.any(drawn == chart_id)):
        raise ValueError("The full original ColorChecker and non-ColorChecker drawn roles are required")
    result = labels.copy()
    explicit = whole & (drawn != 0)
    continued = np.zeros(labels.shape, bool)
    result[whole] = 0
    result[explicit] = drawn[explicit]
    for y in np.flatnonzero(whole.any(1)):
        boundaries = np.diff(np.r_[False, whole[y], False].astype(np.int8))
        for left, right in zip(np.flatnonzero(boundaries == 1),
                               np.flatnonzero(boundaries == -1)):
            donor = int(labels[y, left-1]) if left else 0
            if donor == chart_id:
                raise ValueError("A ColorChecker pixel cannot donate its hidden material")
            if donor:
                select = ~explicit[y, left:right]
                result[y, left:right][select] = donor
                continued[y, left:right][select] = True
    unknown = whole & (result == 0)
    if np.any(result == chart_id) or not np.array_equal(result[~whole], labels[~whole]):
        raise ValueError("Restoring the ColorChecker changed an unrelated material pixel")
    return dict(material_labels=result, drawn_mask=explicit, left_continued_mask=continued,
                unassigned_mask=unknown, whole_chart_mask=whole.copy())


def _metric_footprint(source_mask, source_to_metric):
    """Union of exact painted source pixel CELLS; project their straight edges."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    strips = []
    for y in np.flatnonzero(source_mask.any(1)):
        boundaries = np.diff(np.r_[False, source_mask[y], False].astype(np.int8))
        for left, right in zip(np.flatnonzero(boundaries == 1),
                               np.flatnonzero(boundaries == -1)):
            strips.append(box(left-.5, y-.5, right-.5, y+.5))
    polygon = unary_union(strips)
    parts = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)
    transformed = []
    for part in parts:
        rings = [_project(np.asarray(ring.coords, np.float64), source_to_metric)
                 for ring in [part.exterior, *part.interiors]]
        transformed.append(Polygon(rings[0], rings[1:]))
    result = unary_union(transformed)
    if result.is_empty or not result.is_valid:
        raise ValueError("The whole ColorChecker does not project to a valid footprint")
    return result


def _translated_mask(source, source_to_native, native_to_metric, motion, shape):
    """Pixel-centre distance <= 5 mm from rigidly moved whole painted cells.

    GEOS measures distances to actual metric polygon edges. No constant-pixel
    dilation or circular-arc polygon approximation is used. Work is bounded
    to 32 native rows at a time, including image-border intersections.
    """
    from shapely import dwithin, points
    polygon = _metric_footprint(source, native_to_metric @ source_to_native)
    polygon = _move_metric_footprint(polygon, motion)
    xmin, ymin, xmax, ymax = polygon.bounds
    result = np.zeros(shape, bool)
    x = np.arange(shape[1], dtype=np.float64)
    for start in range(0, shape[0], 32):
        y = np.arange(start, min(start+32, shape[0]), dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        metric = _project(np.column_stack((xx.ravel(), yy.ravel())), native_to_metric)
        near = ((metric[:, 0] >= xmin-MARGIN_MM) & (metric[:, 0] <= xmax+MARGIN_MM) &
                (metric[:, 1] >= ymin-MARGIN_MM) & (metric[:, 1] <= ymax+MARGIN_MM))
        selected = np.zeros(len(metric), bool)
        selected[near] = dwithin(polygon, points(metric[near, 0], metric[near, 1]), MARGIN_MM)
        result[start:start+len(y)] = selected.reshape(xx.shape)
    return result


def _move_metric_footprint(polygon, motion):
    """The same unit-scale metric transform for analysis and display geometry."""
    from shapely.affinity import affine_transform, translate
    if not motion.get('rotation_applied', False):
        return translate(polygon, xoff=motion['dx_mm'], yoff=motion['dy_up_mm'])
    rotation = np.asarray(motion['rotation_matrix'], np.float64)
    centre = np.asarray(motion['source_centroid_mm'], np.float64)
    offset = centre + [motion['dx_mm'], motion['dy_up_mm']] - rotation @ centre
    return affine_transform(polygon, [rotation[0, 0], rotation[0, 1], rotation[1, 0],
        rotation[1, 1], offset[0], offset[1]])


def _display_outlines(source, source_to_native, native_to_metric, motion, margin):
    """Whole painted card in native coordinates, independent of analysis clipping."""
    original = _metric_footprint(source, native_to_metric @ source_to_native)
    current = _move_metric_footprint(original, motion)
    if margin:
        current = current.buffer(margin)
    inverse = np.linalg.inv(native_to_metric)

    def rings(polygon):
        parts = [polygon] if polygon.geom_type == 'Polygon' else list(polygon.geoms)
        return [_project(np.asarray(ring.coords, np.float64), inverse)
                for part in parts for ring in (part.exterior, *part.interiors)]

    return {'original_chart_mask': rings(original), 'chart_mask': rings(current)}


def propose_chart_transfer(run_id, base_labels, inside, original_chart_mask,
                           source_material, whole_chart_mask, source_to_native_h,
                           native_to_metric_h, reference_centres_mm, target_centres_mm,
                           source_binding, *, chart_id, known_ids, correction_mm=None,
                           proposal_format=FORMAT):
    """Propose an in-memory correction; never write/adopt/start a job.

    The reader supplies authenticated pixels/models and both sets of centres
    in the SAME metric plane. The 25 unchanged masks are copied bit-for-bit.
    Frozen v1/v2 keep their original 14-run cohort. Explicit v3 reviews use
    every run's detected rigid motion and the fixed 5 mm margin. Every
    unrelated target material pixel stays unchanged.
    """
    import cv2
    if proposal_format not in {LEGACY_FORMAT, FORMAT, DETECTED_FORMAT}:
        raise ValueError("Unsupported ColorChecker proposal replay format")

    # The original per-run writer stores int32. Preserve that output dtype;
    # only the original source-map carrier works on its unchanged uint8 ABI.
    labels = _labels(base_labels, "Target labels", original_integer=True)
    domain = _mask(inside, labels.shape, "Own frame")
    old = _mask(original_chart_mask, labels.shape, "Original ColorChecker exclusion")
    material = _labels(source_material, "Continuous source material")
    whole = _mask(whole_chart_mask, material.shape, "Whole source ColorChecker")
    ids = _roles(known_ids, chart_id)
    if not whole.any():
        raise ValueError("The original whole ColorChecker footprint is empty")
    if np.any(~np.isin(labels, [0, *ids])):
        raise ValueError("The target labels contain a layer outside the saved layer identities")
    if np.any(~np.isin(material, [0, *(ids-{chart_id})])):
        raise ValueError("The continuous material contains an unknown layer or the ColorChecker")
    if not np.array_equal(old, labels == chart_id):
        raise ValueError("The original ColorChecker exclusion differs from the saved target labels")
    # The exact saved exclusion may cross the measured rig boundary. Keep its
    # full pixels for unchanged maps and continuous material restoration;
    # inside remains the separate analysis domain and clips shifted masks.
    if (not isinstance(run_id, str) or not run_id.strip() or
            not isinstance(source_binding, dict) or not source_binding):
        raise ValueError("The original target exclusion, material and source binding are required")
    carrier, metric = _matrix(source_to_native_h), _matrix(native_to_metric_h)
    motion = (_translation_motion(reference_centres_mm, target_centres_mm)
              if proposal_format == LEGACY_FORMAT else chart_motion(reference_centres_mm, target_centres_mm))
    correction = [0.0, 0.0] if correction_mm is None else correction_mm
    if (not isinstance(correction, (list, tuple)) or len(correction) != 2 or
            any(type(value) not in (int, float) or not np.isfinite(value) for value in correction)):
        raise ValueError("A human ColorChecker adjustment needs finite dx/dy millimetres")
    correction = [float(value) for value in correction]
    automatic = True if proposal_format == DETECTED_FORMAT else run_id in MOVED_RUNS
    applied = dict(motion, dx_mm=(motion["dx_mm"] if automatic else 0.)+correction[0],
                   dy_up_mm=(motion["dy_up_mm"] if automatic else 0.)+correction[1])
    if proposal_format != LEGACY_FORMAT and not automatic:
        applied.update(rotation_applied=False, rotation_degrees=0., rotation_matrix=np.eye(2).tolist())
    if not np.isfinite([applied["dx_mm"], applied["dy_up_mm"]]).all():
        raise ValueError("The adjusted ColorChecker displacement is not finite")
    carried_material = cv2.warpPerspective(material, carrier, (labels.shape[1], labels.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    continuous = labels.copy()
    continuous[old] = carried_material[old]
    if np.any(continuous == chart_id):
        raise ValueError("A ColorChecker cannot remain a material in the continuous layer map")
    moved = automatic or any(correction)
    footprint, placement = ((whole, carrier) if automatic or proposal_format != LEGACY_FORMAT
                            else (old, np.eye(3)))
    mask = (_translated_mask(footprint, placement, metric, applied, labels.shape) & domain
            if moved else old.copy())
    effective = continuous.copy()
    effective[mask] = chart_id
    if not moved and not np.array_equal(effective, labels):
        raise ValueError("An unchanged run's original analysis map changed")
    result = dict(material_labels=continuous, chart_mask=mask, effective_labels=effective,
        original_chart_mask=old.copy(), vacated_mask=old & ~mask,
        unassigned_mask=(continuous == 0) & domain & ~mask)
    metadata = dict(format=proposal_format, run_id=run_id, chart_id=chart_id,
        moved=moved, automatic=automatic, margin_mm=MARGIN_MM if moved else 0.0,
        motion=motion, correction_mm=correction, applied_motion=applied,
        source_binding=deepcopy(source_binding),
        source_arrays=dict(material_labels=_identity(material), whole_chart_mask=_identity(whole)),
        base_arrays=dict(labels=_identity(labels), inside=_identity(domain), chart_mask=_identity(old)),
        source_to_native_h=carrier.tolist(), native_to_metric_h=metric.tolist(),
        reference_centres_mm=_centres(reference_centres_mm).tolist(),
        target_centres_mm=_centres(target_centres_mm).tolist(),
        raster_rule=("native pixel centres within 5 mm of translated whole painted source pixel cells"
            if proposal_format == LEGACY_FORMAT else
            "native pixel centres within 5 mm of rigidly moved whole painted source pixel cells; scale fixed at one"),
        result={key: _identity(value) for key, value in result.items()})
    outlines = _display_outlines(whole, carrier, metric, applied, metadata['margin_mm'])
    return ChartTransferProposal(_json(metadata), result, outlines)


@dataclass(frozen=True)
class ChartTransferProposal:
    """Exact shown source/result, not Save/Build/scientific eligibility."""
    payload: bytes
    result: dict
    # Derived display paths do not alter the bound analysis arrays or approvals.
    display_outlines: dict | None = field(default=None, compare=False, repr=False)

    def metadata(self):
        return json.loads(self.payload)

    def validate(self):
        value = self.metadata()
        if (value.get("format") not in {LEGACY_FORMAT, FORMAT, DETECTED_FORMAT} or _json(value) != self.payload or
                value["result"] != {key: _identity(array) for key, array in self.result.items()}):
            raise ValueError("A ColorChecker proposal's bound source or result changed")
        return value


def confirm_chart_transfer(proposal, *, actor, reason):
    """Confirm displayed pixels, without Save, adoption or a worker."""
    if not isinstance(proposal, ChartTransferProposal):
        raise ValueError("Choose an actual ColorChecker transfer proposal")
    value = proposal.validate()
    if any(type(item) is not str or not item.strip() for item in (actor, reason)):
        raise ValueError("ColorChecker confirmation needs a human actor and plain reason")
    approval_format = {LEGACY_FORMAT: LEGACY_APPROVAL_FORMAT, FORMAT: APPROVAL_FORMAT,
                       DETECTED_FORMAT: DETECTED_APPROVAL_FORMAT}[value['format']]
    return dict(format=approval_format, run_id=value["run_id"],
        proposal_sha256=hashlib.sha256(proposal.payload).hexdigest(),
        correction_mm=deepcopy(value["correction_mm"]),
        actor=actor.strip(), reason=reason.strip(), approved_utc=datetime.now(timezone.utc).isoformat())


def apply_chart_transfer(proposal, approval):
    """Corrected inputs BEFORE fresh original facies/area/mass derivation.

    Material is continuous behind the card. The chart mask excludes occluded
    cells. Effective labels use the original chart role for compatibility
    with the original excluded-region table/Build path. Never retain physical
    arrays calculated from the old effective labels.
    """
    if not isinstance(proposal, ChartTransferProposal):
        raise ValueError("Choose an actual ColorChecker transfer proposal")
    value = proposal.validate()
    _validate_approval(value, proposal.payload, approval)
    return {key: array.copy() for key, array in proposal.result.items()}


def _validate_approval(value, payload, approval):
    """Validate the same human action for changed and unchanged chart masks."""
    if (not isinstance(approval, dict) or set(approval) != {
            "format", "run_id", "proposal_sha256", "actor", "reason", "approved_utc", "correction_mm"} or
            proposal_format_for_approval(approval) != value.get("format") or approval["run_id"] != value["run_id"] or
            approval["proposal_sha256"] != hashlib.sha256(payload).hexdigest() or
            _json(approval["correction_mm"]) != _json(value["correction_mm"]) or
            any(type(approval[key]) is not str or not approval[key].strip()
                for key in ("actor", "reason", "approved_utc"))):
        raise ValueError("The confirmation does not bind this exact source and result")
    stamp = datetime.fromisoformat(approval["approved_utc"])
    if stamp.tzinfo is None:
        raise ValueError("The confirmation needs an explicit UTC offset")
