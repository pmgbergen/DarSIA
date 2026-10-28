"""Advisory metric position from explicitly bound camera-registration models."""

from copy import deepcopy
import json
import math

import numpy as np


METHOD = "camera registration (map placement)"
SCOPE = "local-projective-x-tangent"


def _number(value, label):
    """Accept only finite literal numbers, without interpreting booleans as units."""
    if type(value) not in (int, float):
        raise ValueError(label + " must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(label + " must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(label + " must be a finite number")
    return result


def _literal(value):
    """Copy finite literal JSON metadata without coercing types or object keys."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_literal(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _literal(item) for key, item in value.items()}
    raise ValueError("Model metadata must be finite literal JSON")


def _metadata(value, label):
    """Require a nonempty metadata value and retain its exact JSON identity."""
    try:
        snapshot = _literal(value)
    except (RecursionError, ValueError) as error:
        raise ValueError(label + " must be finite literal JSON") from error
    if snapshot is None or (type(snapshot) in (str, list, dict) and not snapshot):
        raise ValueError(label + " must be nonempty literal JSON")
    return snapshot, json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _model(models, run_id):
    """Validate one declared metric model without reading or modifying its sources."""
    if not isinstance(models, dict) or not isinstance(models.get(run_id), dict):
        raise ValueError(f"Model for {run_id!r} is missing or not an object")
    row = models[run_id]
    if row.get("metric_units") != "mm":
        raise ValueError(f"Model for {run_id!r} must explicitly declare metric_units='mm'")
    raw = row.get("metric_to_camera_u")
    if type(raw) is not list or len(raw) != 3 or any(type(values) is not list or len(values) != 3 for values in raw):
        raise ValueError(f"Model for {run_id!r} needs a 3x3 nested-list metric_to_camera_u")
    matrix = np.array([[_number(value, "metric_to_camera_u entry") for value in values] for values in raw],
        dtype=np.float64, copy=True)
    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        raise ValueError(f"Model for {run_id!r} has a singular metric_to_camera_u")
    matrix /= scale
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"Model for {run_id!r} has a singular metric_to_camera_u") from error
    if not np.isfinite(inverse).all():
        raise ValueError(f"Model for {run_id!r} has a nonfinite inverse")
    profile, profile_key = _metadata(row.get("source_profile"), "source_profile")
    context, context_key = _metadata(row.get("camera_u_context"), "camera_u_context")
    provenance = row.get("provenance")
    if type(provenance) is not dict or not provenance:
        raise ValueError(f"Model for {run_id!r} needs nonempty provenance")
    provenance, _ = _metadata(provenance, "provenance")
    matrix.flags.writeable = inverse.flags.writeable = False
    return dict(matrix=matrix, inverse=inverse, source_profile=profile, profile_key=profile_key,
        camera_u_context=context, context_key=context_key, provenance=provenance)


def _comparison(models, run_id, comparison_id, anchor, shift_tolerance, angle_tolerance):
    """Evaluate one direct relative transform and its local x-direction tangent."""
    target, comparison = _model(models, run_id), _model(models, comparison_id)
    if target["profile_key"] != comparison["profile_key"]:
        raise ValueError("Compared models have different source_profile bindings")
    if target["context_key"] != comparison["context_key"]:
        raise ValueError("Compared models have different camera_u_context bindings")
    if run_id == comparison_id:
        dx = dy = shift = rotation = 0.0
    else:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            matrix = comparison["inverse"] @ target["matrix"]
            if not np.isfinite(matrix).all():
                raise ValueError("The relative metric transform is nonfinite")
            x, y = anchor
            denominator = float(matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2])
            if denominator == 0:
                raise ValueError("The relative transform reaches a projective horizon at anchor_metric_mm")
            mapped_x = float((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator)
            mapped_y = float((matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator)
            dx, dy = mapped_x - x, mapped_y - y
            # Derivative of the projective quotient with respect to metric x.
            tangent_x = float((matrix[0, 0] - mapped_x * matrix[2, 0]) / denominator)
            tangent_y = float((matrix[1, 0] - mapped_y * matrix[2, 0]) / denominator)
            if not all(math.isfinite(value) for value in (dx, dy, tangent_x, tangent_y)) or (tangent_x == 0 and tangent_y == 0):
                raise ValueError("The local projective displacement or tangent is undefined")
            shift, rotation = math.hypot(dx, dy), math.degrees(math.atan2(tangent_y, tangent_x))
            if not math.isfinite(shift):
                raise ValueError("The local projective displacement is nonfinite")
    shift_warning, rotation_warning = shift > shift_tolerance, abs(rotation) > angle_tolerance
    return dict(target_run_id=run_id, comparison_run_id=comparison_id, anchor_metric_mm=list(anchor), dx_mm=dx, dy_mm=dy,
        shift_mm=shift, rotation_degrees=rotation, rotation_scope=SCOPE,
        warning=shift_warning or rotation_warning, shift_warning=shift_warning,
        rotation_warning=rotation_warning, source_profile=deepcopy(target["source_profile"]),
        camera_u_context=deepcopy(target["camera_u_context"]),
        provenance=dict(target=deepcopy(target["provenance"]), comparison=deepcopy(comparison["provenance"])))


def position_readout(models, run_id, *, reference_run, declared_order, anchor_metric_mm,
                     shift_tolerance_mm=2.0, angle_tolerance_degrees=.5):
    """Report direct reference/previous comparisons as advisory registration metadata.

    ``metric_to_camera_u`` must map declared millimetres into the same explicitly
    bound camera-u context and source profile. Rotation is the local projective
    x-tangent angle at the supplied anchor, not a global or physical rotation.
    No photograph qualification, fitting, source discovery or production gate is
    implied by a result or its tolerance warnings.
    """
    if any(type(value) is not str or not value.strip() for value in (run_id, reference_run)):
        raise ValueError("run_id and reference_run must be nonempty literal identifiers")
    if (type(declared_order) not in (list, tuple) or not declared_order
            or any(type(value) is not str or not value.strip() for value in declared_order)
            or len(set(declared_order)) != len(declared_order) or run_id not in declared_order):
        raise ValueError("declared_order must contain distinct literal run identifiers including run_id")
    if type(anchor_metric_mm) not in (list, tuple) or len(anchor_metric_mm) != 2:
        raise ValueError("anchor_metric_mm must contain two finite metric coordinates")
    anchor = [_number(value, "anchor_metric_mm coordinate") for value in anchor_metric_mm]
    shift_tolerance = _number(shift_tolerance_mm, "shift_tolerance_mm")
    angle_tolerance = _number(angle_tolerance_degrees, "angle_tolerance_degrees")
    if shift_tolerance < 0 or angle_tolerance < 0:
        raise ValueError("Position tolerances must be nonnegative")
    index = declared_order.index(run_id)
    previous_id = declared_order[index - 1] if index else None
    result = dict(method=METHOD, scope=SCOPE, advisory_only=True, run_id=run_id,
        reference_run_id=reference_run, previous_run_id=previous_id, anchor_metric_mm=anchor,
        shift_tolerance_mm=shift_tolerance, angle_tolerance_degrees=angle_tolerance,
        reference=None, previous=None, unavailable_reasons={})
    for label, comparison_id in (("reference", reference_run), ("previous", previous_id)):
        if comparison_id is None:
            result["unavailable_reasons"][label] = "The first declared run has no previous run."
            continue
        try:
            result[label] = _comparison(models, run_id, comparison_id, anchor, shift_tolerance, angle_tolerance)
        except (ValueError, ArithmeticError) as error:
            result["unavailable_reasons"][label] = str(error)
    return result
