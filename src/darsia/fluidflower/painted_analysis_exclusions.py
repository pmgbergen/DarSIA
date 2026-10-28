"""Explicit human review of unmapped analysis pixels, without physical relabelling.

The old outside-frame proof retains its physical meaning.  This optional
contract instead keeps approved gaps unquantified on their unchanged native
grid.  No absent declaration, failed preparation, or diagnostic census grants
approval.  Every new consumer replays the original sources and exact masks.
"""
from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

REVIEW_FORMAT = "darsia-painted-gap-review-v1"
APPROVAL_FORMAT = "darsia-painted-gap-exclusion-approval-v1"
EXCLUSION_FORMAT = "darsia-painted-analysis-exclusions-v1"
APPROVAL_FORMAT_V2 = "darsia-painted-gap-exclusion-approval-v2"
EXCLUSION_FORMAT_V2 = "darsia-painted-analysis-exclusions-v2"
KEY = "analysis_exclusions"
MEANING = "outside-analysis-unquantified"
CLASSIFICATIONS = ("water_above_sand", "air", "rig_or_frame", "other")
CLASSIFICATION_LABELS = {
    "water_above_sand": "Water above the sand",
    "air": "Air",
    "rig_or_frame": "Rig or frame",
    "other": "Other",
}
_LEGACY_CLASSIFICATIONS = {
    "Vann over sanden": "water_above_sand",
    "Luft": "air",
    "Rigg eller ramme": "rig_or_frame",
    "Annet": "other",
}
RESOLUTIONS_FORMAT = "darsia-painted-findings-resolutions-v1"
MAX_REVIEW_BYTES = 128 * 1024 * 1024


def classification_key(value):
    """Resolve an exact stored label or key without granting approval."""
    if type(value) is str:
        if value in CLASSIFICATIONS:
            return value
        if value in _LEGACY_CLASSIFICATIONS:
            return _LEGACY_CLASSIFICATIONS[value]
    raise ValueError("Choose an explicit gap-area classification")


def classification_label(value):
    """Return the English display label for an exact supported classification."""
    return CLASSIFICATION_LABELS[classification_key(value)]


class GapReviewRequired(ValueError):
    """Retained products need human review; this is not a completed preparation."""
    def __init__(self, review, reason=None):
        self.review = deepcopy(review)
        message = "zero labels are INSIDE its frozen own frame; review these unmapped pixels in Use the map in the series; nothing was adopted."
        super().__init__(message if reason is None else f"{message} {reason}")


class DisplayFindingsChanged(ValueError):
    """Retain original source-bound findings when optional display input fails."""
    def __init__(self, original_issues, reason):
        self.original_issues = deepcopy(original_issues)
        super().__init__(reason)


def _json(value):
    from .painted_adoption_table import _json as decode
    return decode(json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf8"))


def _same(left, right):
    return _json(left) == _json(right) and json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _identity(value):
    from .geometry_raster_sources import _carried_array_identity
    return _carried_array_identity(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf8")).hexdigest()


def _bound(path):
    from .painted_adoption_table import _bound
    return _bound(Path(path))


def _checked(value, *, metadata=False):
    from .painted_adoption_table import _checked
    return _checked(value, metadata=metadata)


def _arrays(bound):
    actual = _checked(bound)
    with np.load(_preset_io.source(actual["path"]), allow_pickle=False) as archive:
        fields = {key: np.array(archive[key], copy=True) for key in
                  ("array", "filled", "support", "inside", "continued", "effective_coverage", "gaps")}
    shape = fields["array"].shape
    if fields["array"].ndim != 2 or not np.issubdtype(fields["array"].dtype, np.integer):
        raise ValueError("A gap review needs its original integer native labels")
    if any(value.dtype != bool or value.shape != shape for key, value in fields.items() if key != "array"):
        raise ValueError("Gap review masks must be exact native boolean fields")
    if (not np.array_equal(fields["gaps"], fields["inside"] & ~fields["effective_coverage"])
            or np.any(fields["gaps"] & (fields["array"] != 0))):
        raise ValueError("The reviewed gap must contain only actual unmapped native pixels")
    _checked(bound)
    return fields


def _source(preparation):
    """Stable scientific anchors; fresh output folders are not new authority."""
    from .photo_edge_continuation import POLICY
    if preparation.get("format") != "darsia-fluidflower-painted-map-adoption-preparation-v2":
        raise ValueError("Analysis gap review requires an explicit native V2 preparation")
    return dict(painting_facies=deepcopy(preparation["painting_facies"]),
                reference_run=preparation["reference_run"],
                geometry_index=deepcopy(preparation["geometry_index"]), policy=POLICY)


def target_paint_science(preparation, entry, contract, arrays, prior=()):
    """Stable exact DATA for a target edit; no fresh NPZ container identity."""
    model = _checked(entry["target_model"], metadata=True)
    certificate = _checked(model["baseline_native_grid"]["renderer"], metadata=True)
    photo = certificate["corrected_baseline"]
    snapshot = preparation["painting_facies"]["facies_snapshot"]
    facies = dict(snapshot["facies"])
    allowed = sorted(layer["layer_id"] for layer in snapshot["layers"]
        if facies[layer["facies"]]["role"] == "porous_medium" and 0 < layer["layer_id"] < 255)
    return dict(source=_source(preparation), reference_model=deepcopy(entry["reference_model"]),
        target_model=deepcopy(entry["target_model"]), contract_sha256=_digest(contract),
        native_photo=deepcopy(photo["archive"]), native_photo_pixels=deepcopy(photo["pixels"]),
        scientific_arrays={key: _identity(value) for key, value in arrays.items() if key in
            {"array", "filled", "support", "inside", "continued", "effective_coverage", "gaps"}},
        prior_target_corrections=_json(list(prior)), allowed_layer_ids=allowed)


def _row_digest(row):
    return _digest({key: value for key, value in row.items() if key != "review_row_sha256"})


def _frame_comparison(binding, contract):
    """Measured frame disagreement only; this does not decide which frame is right."""
    from .photo_edge_continuation import validate_contract
    from .painted_adoption_table import _frame
    from .layer_painting import frame_polygon
    from .painting_preview_frame_domain import _project
    source_h, _, _, _, frame, _ = validate_contract(contract, verify_files=True)
    target_h = np.asarray(contract["target_model"]["metric_to_grid"], float)
    source_xy = frame_polygon(_frame(binding["frame"]), (1., 1.))
    target_xy = frame_polygon(frame, (1., 1.))
    source_mm = _project(source_xy, np.linalg.inv(source_h))
    # Preserve a literal identity path, rather than introducing floating point
    # differences through two equivalent inverse/project computations.
    target_mm = source_mm.copy() if np.array_equal(source_xy, target_xy) and np.array_equal(source_h, target_h) else _project(target_xy, np.linalg.inv(target_h))
    edges = []
    def distance(points, first, second):
        delta = second - first
        along = np.clip(((points - first) @ delta) / float(delta @ delta), 0., 1.)
        return float(np.linalg.norm(points - first - along[:, None] * delta, axis=1).max())
    for index, name in enumerate(("top", "right", "bottom", "left")):
        a, b = source_mm[index], source_mm[(index + 1) % 4]
        c, d = target_mm[index], target_mm[(index + 1) % 4]
        first, second = b - a, d - c
        angle = float(np.degrees(np.arctan2(abs(first[0] * second[1] - first[1] * second[0]), abs(float(first @ second)))))
        maximum = max(distance(np.stack((a, b)), c, d), distance(np.stack((c, d)), a, b))
        edges.append(dict(edge=name, max_edge_distance_mm=maximum, angle_degrees=angle))
    maximum = max(row["max_edge_distance_mm"] for row in edges)
    # Equivalent image homographies need not round-trip to bit-identical
    # metric corners. This bound covers floating-point arithmetic only; it is
    # not a tolerance for a physical displacement of the frame.
    roundoff = 64 * np.finfo(np.float64).eps * max(
        1., float(np.abs(source_mm).max()), float(np.abs(target_mm).max()))
    mismatch = not np.allclose(source_mm, target_mm, rtol=0., atol=roundoff)
    return dict(max_edge_distance_mm=maximum, edges=edges, mismatch=mismatch,
                meaning="measured-frame-comparison-not-physical-correctness", threshold=None,
                numerical_roundoff_mm=roundoff)


def _review_metadata(source):
    """Validate already-loaded literal metadata; GUI action creation does no I/O."""
    review = _json(source)
    if review.get("format") != REVIEW_FORMAT or review.get("meaning") != MEANING:
        raise ValueError("An exact analysis-gap review manifest is required")
    from .painting_chart_transfer import review_proposal_format
    review_proposal_format(review)
    bound = review.get("review_binding")
    if bound is None:
        raise ValueError("The retained review manifest needs its original path/SHA/size")
    rows = review.get("runs")
    if not isinstance(rows, list) or len({row.get("run_id") for row in rows}) != len(rows):
        raise ValueError("A gap review needs unique ordered run rows")
    for row in rows:
        if (row.get("review_row_sha256") != _row_digest(row)
                or type(row.get("gap", {}).get("pixels")) is not int
                or row["gap"]["pixels"] < 0):
            raise ValueError("Gap review row identity or count changed")
        components = row.get("components")
        if (not isinstance(components, list)
                or len({item.get("component_id") for item in components}) != len(components)
                or sum(item.get("pixels", -1) for item in components) != row["gap"]["pixels"]):
            raise ValueError("Gap review components changed")
        context = _digest({key: value for key, value in row.items() if key not in {"components", "review_row_sha256"}})
        for item in components:
            if (item.get("component_id") != item.get("identity", {}).get("sha256")
                    or type(item.get("pixels")) is not int or item["pixels"] < 1
                    or item.get("review_row_sha256") != _digest(dict(context=context,
                        component={key: value for key, value in item.items() if key != "review_row_sha256"}))):
                raise ValueError("A connected gap-area identity changed")
    if not _same(review.get("issues"), [issue for row in rows for issue in row.get("findings", [])]):
        raise ValueError("The original frame findings changed")
    return review


def _review_size(path):
    """Reject oversized review files before their descriptor is hashed."""
    if _preset_io.current_scope() is None:
        from .storage_paths import resolve_storage_path
        path = resolve_storage_path(path)
    if _preset_io.file_stat(path).st_size > MAX_REVIEW_BYTES:
        raise ValueError("Analysis-gap review exceeds its explicit read limit")


def _checked_gap_review(binding):
    """Read only the typed cohort review with its separate bounded budget."""
    from .painted_adoption_table import _read, _json as decode
    if (isinstance(binding, dict) and type(binding.get("size_bytes")) is int
            and binding["size_bytes"] > MAX_REVIEW_BYTES):
        raise ValueError("Analysis-gap review exceeds its explicit read limit")
    if (isinstance(binding, dict) and isinstance(binding.get("path"), str)
            and Path(binding["path"]).is_absolute()):
        _review_size(binding["path"])
    actual = _checked(binding)
    raw = _read(actual["path"], limit=MAX_REVIEW_BYTES)
    if hashlib.sha256(raw).hexdigest() != actual["sha256"] or len(raw) != actual["size_bytes"]:
        raise ValueError("Analysis-gap review changed while reading")
    return decode(raw)


def _read_scientific_gap_review(source):
    """Authenticate original review bytes before any optional display stage."""
    if isinstance(source, (str, Path)):
        _review_size(source)
        bound = _bound(source)
        review = _checked_gap_review(bound)
        review["review_binding"] = bound
    else:
        review = source
    review = _review_metadata(review)
    retained = _checked_gap_review(review["review_binding"])
    if not _same(retained, {key: value for key, value in review.items() if key != "review_binding"}):
        raise ValueError("The in-memory gap review differs from its retained manifest")
    return review


def read_gap_review(source):
    """Read a formal review and fully validate its optional display context."""
    review = _read_scientific_gap_review(source)
    try:
        if review.get("display_context") is not None:
            _display_context(review["display_context"], review)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        raise DisplayFindingsChanged(review["issues"], str(error)) from error
    return review


def _display_context(value, review=None):
    if value is None:
        return None
    from .painted_review_comparison import validate_display_context
    return _json(validate_display_context(_json(value), review))


def _same_display_source(first, second):
    """Preferences and new capture locators never invalidate an area choice."""
    if first is None or second is None:
        return first is None and second is None
    from .painted_review_comparison import context_source_sha256
    return context_source_sha256(first) == context_source_sha256(second)


def _components(mask):
    """Eight-connected display/review areas; never a facies classification."""
    import cv2
    from darsia._opencv_runtime import scientific_thread, require_ipp_off
    scientific_thread("painted-gap-review-components:before")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    require_ipp_off("painted-gap-review-components:after")
    for number in range(1, count):
        x, y, width, height, pixels = [int(value) for value in stats[number]]
        value = labels == number
        identity = _identity(value)
        yield dict(component_id=identity["sha256"], pixels=pixels, identity=identity,
                   bounding_box_row_col_half_open=[[y, x], [y + height, x + width]]), value


def _preview_display_rgb(image, display_rgb=None):
    """Choose declared display pixels after the caller authenticates native DATA."""
    if display_rgb is not None:
        if not (image.dtype == np.uint8 or
                (np.issubdtype(image.dtype, np.floating) and np.isfinite(image).all())):
            raise ValueError("The original RGB display encoding is not declared by this preview route")
        if (not isinstance(display_rgb, np.ndarray) or display_rgb.dtype != np.uint8
                or display_rgb.shape != image.shape):
            raise ValueError("The explicit display RGB must be uint8 on this exact native photograph grid")
        return display_rgb, "rgb_uint8"
    if image.dtype == np.uint8:
        return image, "rgb_uint8"
    if np.issubdtype(image.dtype, np.floating) and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1:
        return image, "rgb_float01"
    raise ValueError("The original RGB display encoding is not declared by this preview route")


def read_gap_preview(review, run_id, component_id=None, *, display_rgb=None):
    """Read only one original native RGB and its exact gap, for display."""
    review = read_gap_review(review)
    return _read_admitted_gap_preview(review, run_id, component_id, display_rgb=display_rgb)


def _read_admitted_gap_preview(review, run_id, component_id=None, *, display_rgb=None,
                               include_native_rgb=True):
    """Read selected products immediately after the caller admitted this review.

    This private step retains every selected product and native-pixel check;
    the public reader still admits the whole review before calling it.
    """
    rows = [row for row in review["runs"] if row["run_id"] == run_id]
    if len(rows) != 1:
        raise ValueError("The selected run is not in the reviewed cohort")
    row = rows[0]
    arrays = _arrays(row["product"])
    if not _same(_identity(arrays["gaps"]), row["gap"]["identity"]):
        raise ValueError("The selected review gap changed")
    mask = arrays["gaps"]
    if component_id is not None:
        selected = [(item, value) for item, value in _components(mask) if item["component_id"] == component_id]
        if len(selected) != 1 or not any(item["component_id"] == component_id for item in row["components"]):
            raise ValueError("The selected connected gap area changed")
        mask = selected[0][1]
    from .geometry_facies import _load_array
    image = np.asarray(_load_array(Path(_checked(row["native_photo"])["path"])))
    identity = _identity(image)
    expected = row["native_photo_pixels"]
    if any(identity.get(key) != expected.get(key) for key in ("dtype", "shape", "sha256")):
        raise ValueError("The reviewed photograph differs from its saved native RGB pixels")
    if image.shape != (*arrays["gaps"].shape, 3):
        raise ValueError("The original photograph is not on this exact native gap grid")
    # A full-RAW display still authenticates native DATA but does not interpret
    # normalized linear camera DN as gamma-encoded display RGB in [0, 1].
    appearance = {}
    if include_native_rgb:
        image, encoding = _preview_display_rgb(image, display_rgb)
        appearance = dict(image_rgb=np.array(image, copy=True), image_encoding=encoding)
    elif not (image.dtype == np.uint8 or
              np.issubdtype(image.dtype, np.floating) and np.isfinite(image).all()):
        raise ValueError("The original native RGB encoding is not supported")
    from .map_placement import _edges
    from .painted_adoption_table import _frame
    from .layer_painting import frame_polygon
    return dict(run_id=run_id, component_id=component_id, **appearance,
        gap_mask=mask, layer_boundary_mask=_edges(arrays["array"]), native_to_photo_h=np.eye(3).tolist(),
        own_frame_native_xy=deepcopy(row["own_frame_native_xy"]),
        transported_frame_native_xy=deepcopy(row["transported_frame_native_xy"]),
        native_to_painting_h=deepcopy(row["native_to_painting_h"]),
        painting_shape_hw=deepcopy(row["contract"]["source_model"]["shape_hw"]),
        painting_frame_xy=frame_polygon(_frame(review["source"]["painting_facies"]["frame"]), (1., 1.)).tolist(),
        painting_run_id=review["source"]["reference_run"],
        painting_photograph=deepcopy(review["painting_photograph"]), map_version=deepcopy(review["map_version"]))


def read_target_paint(review, run_id, component_id, *, preview_reader=None):
    """A genuine target-native editable packet, restricted to one reviewed area."""
    review = read_gap_review(review)
    packet = (preview_reader or read_gap_preview)(review, run_id, component_id)
    row = next(row for row in review["runs"] if row["run_id"] == run_id)
    component = next((item for item in row["components"] if item["component_id"] == component_id), None)
    if component is None:
        raise ValueError("Target painting requires one exact unresolved connected area")
    from .painted_adoption_table import verify_version
    session, _, _ = verify_version(review["source"]["painting_facies"])
    packet.update(target_labels=_arrays(row["product"])["array"].astype(np.uint8),
        editable_mask=packet["gap_mask"].copy(), allowed_layers=[dict(layer_id=layer.layer_id,
            name=layer.name, facies=layer.facies, role=session.facies_of(layer.layer_id).role) for layer in session.layers
            if session.facies_of(layer.layer_id).role == "porous_medium" and 0 < layer.layer_id < 255],
        target_correction_basis=dict(run_id=run_id, review=deepcopy(review["review_binding"]),
            review_row_sha256=component["review_row_sha256"], component_id=component_id,
            science=deepcopy(row["target_correction_science"])))
    return packet


def make_gap_approval(review, component_keys, *, actor, approved_utc, reason, decision_ref,
                      classification, other_text="", previous=None):
    """Merge only explicit human actions; retain unselected reasons and times."""
    review = _review_metadata(review)
    for value in (actor, approved_utc, reason, decision_ref):
        if type(value) is not str or not value.strip():
            raise ValueError("Each explicit gap approval needs actor, time, reason and decision reference")
    classification = classification_key(classification)
    if type(other_text) is not str or (classification == "other" and not other_text.strip()):
        raise ValueError("Choose an explicit gap-area class and describe Other")
    selected = [(item["run_id"], item["component_id"]) for item in _json(list(component_keys))]
    rows = {(row["run_id"], item["component_id"]): item for row in review["runs"] for item in row["components"]}
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(rows):
        raise ValueError("Choose one or more unique connected areas from this exact gap review")
    result = dict(format=APPROVAL_FORMAT, meaning=MEANING, review=deepcopy(review["review_binding"]), runs=[])
    actions = {}
    if previous is not None:
        old = _approval(previous, review)
        actions.update({(row["run_id"], row["component_id"]): deepcopy(row) for row in old["runs"]})
    for key in selected:
        actions[key] = dict(run_id=key[0], component_id=key[1], review_row_sha256=rows[key]["review_row_sha256"],
            actor=actor, approved_utc=approved_utc, reason=reason, decision_ref=decision_ref,
            classification=classification, other_text=other_text)
    result["runs"] = [actions[key] for key in rows if key in actions]
    return _approval(result, review)


def _approval(value, review=None):
    value = _json(value)
    if (set(value) != {"format", "meaning", "review", "runs"}
            or value["format"] != APPROVAL_FORMAT or value["meaning"] != MEANING):
        raise ValueError("An explicit analysis-gap approval record is required")
    review = read_gap_review(value["review"]["path"]) if review is None else _review_metadata(review)
    from .storage_paths import same_storage_descriptor
    if not same_storage_descriptor(value["review"], review["review_binding"]):
        raise ValueError("The approval belongs to another gap review")
    known = {(row["run_id"], item["component_id"]): item for row in review["runs"] for item in row["components"]}
    seen = set()
    for row in value["runs"]:
        key = (row.get("run_id"), row.get("component_id"))
        if (set(row) != {"run_id", "component_id", "review_row_sha256", "actor", "approved_utc", "reason", "decision_ref", "classification", "other_text"}
                or key not in known or key in seen
                or row["review_row_sha256"] != known[key]["review_row_sha256"]
                or type(row["other_text"]) is not str
                or (classification_key(row["classification"]) == "other" and not row["other_text"].strip())
                or any(type(row[key]) is not str or not row[key].strip() for key in
                       ("actor", "approved_utc", "reason", "decision_ref"))):
            raise ValueError("The explicit per-area gap approval changed")
        seen.add(key)
    return value


def approval_records(value, review=None):
    """Validate the declared version without translating stored human actions."""
    if isinstance(value, dict) and value.get("format") == APPROVAL_FORMAT_V2:
        from .painted_gap_regions import validate_gap_region_approval
        return validate_gap_region_approval(value, review)
    return _approval(value, review)


def build_gap_review(preparation, folder, *, destination=None, display_context=None):
    """Retain a review of completed run products, never a fake successful index."""
    from .photo_edge_continuation import read_contracts, validate_contract
    from .layer_painting import frame_polygon
    from .painting_preview_frame_domain import _project
    folder = Path(folder)
    review_display_context = _display_context(preparation.get("display_context") if display_context is None else display_context)
    if preparation.get("display_context") is not None and not _same_display_source(review_display_context, preparation["display_context"]):
        raise ValueError("The display context changed its actual source closure")
    from .painted_adoption_table import verify_version_summary
    verify_version_summary(preparation["painting_facies"], preparation["map_version"])
    source = _source(preparation)
    contracts = read_contracts(folder / "continuation.json")["runs"]
    source_map = _bound(folder / "source_map.npz")
    with np.load(_preset_io.source(source_map["path"]), allow_pickle=False) as archive:
        source_layers = np.array(archive["array"], copy=True)
    _checked(source_map)
    rows = []
    for entry in preparation["runs"]:
        if entry.get("status") != "prepared":
            continue
        run = entry["run_id"]
        base_product = dict(entry["labels"], path=str(folder / entry["labels"]["path"]))
        from .painted_target_integration import effective_product
        arrays, product = effective_product(preparation, folder, entry)
        contract = contracts[run]
        _, _, _, _, frame, own = validate_contract(contract, verify_files=True)
        model = _checked(entry["target_model"], metadata=True)
        certificate = _checked(model["baseline_native_grid"]["renderer"], metadata=True)
        photo = certificate["corrected_baseline"]
        if photo["grid"]["num_voxels"] != list(arrays["gaps"].shape):
            raise ValueError("The reviewed native RGB grid differs from the run product")
        composition = np.asarray(entry["composition_reference_u_to_native"], dtype=float)
        from .painted_adoption_table import _frame
        source_frame = _frame(preparation["painting_facies"]["frame"])
        transported = _project(frame_polygon(source_frame, (1., 1.)), composition)
        row = dict(run_id=run, product=product, base_product=base_product,
            base_scientific_arrays={key: _identity(value) for key, value in _arrays(base_product).items()},
            scientific_arrays={key: _identity(value) for key, value in arrays.items()},
            reference_model=deepcopy(entry["reference_model"]), target_model=deepcopy(entry["target_model"]),
            contract=deepcopy(contract), native_photo=deepcopy(photo["archive"]), native_photo_pixels=deepcopy(photo["pixels"]),
            own_frame_native_xy=np.asarray(own).tolist(), transported_frame_native_xy=transported.tolist(),
            native_to_painting_h=np.linalg.inv(composition).tolist(),
            gap=dict(artifact=product, field="gaps", shape_hw=list(arrays["gaps"].shape),
                     pixels=int(arrays["gaps"].sum()), identity=_identity(arrays["gaps"])))
        row["frame_comparison"] = _frame_comparison(preparation["painting_facies"], contract)
        prior = ((preparation.get("target_corrections") or {}).get("runs", {}).get(run) or {}).get("bindings", [])
        row["target_correction_science"] = target_paint_science(preparation, entry, contract, arrays, prior)
        row["target_correction_bindings"] = deepcopy(prior)
        row["findings"] = ([] if not row["frame_comparison"]["mismatch"] else [dict(
            severity="warning", code="painted_map_frame_disagreement", run_id=run,
            message="The transported source frame and this run's own frame differ; the measurement does not decide which frame is physically right.",
            measurement=deepcopy(row["frame_comparison"]),
            source_binding_sha256=_digest(source), continuation_sha256=_digest(contract))])
        context = _digest(row)
        row["components"] = []
        for component, _ in _components(arrays["gaps"]):
            component["review_row_sha256"] = _digest(dict(context=context, component=component))
            row["components"].append(component)
        row["review_row_sha256"] = _row_digest(row)
        rows.append(row)
    result = dict(format=REVIEW_FORMAT, meaning=MEANING, source=source,
        source_map=source_map, source_map_pixels=_identity(source_layers), geometry_index=deepcopy(preparation["geometry_index"]),
        map_version=deepcopy(preparation["map_version"]),
        painting_photograph=deepcopy(preparation["painting_facies"]["frame_basis"]["photograph"]), runs=rows)
    from .painting_chart_transfer import fresh_transfer_policy
    result["transfer_policy"] = fresh_transfer_policy()
    if "chart_review" in preparation:
        # These findings describe derived masks. Reopening the chart editor
        # must use the immutable painting source, not feed its own output back
        # through the historical source-map replay.
        result["chart_source_review"] = deepcopy(preparation["painting_geometry"]["review"])
        from .painted_transfer_replay import chart_replay_recipe
        result["chart_preparation"] = chart_replay_recipe(preparation)
    result["issues"] = [deepcopy(issue) for row in rows for issue in row["findings"]]
    if review_display_context is not None:
        result["display_context"] = _display_context(review_display_context, result)
    if "source_recapture" in preparation:
        result["source_recapture"] = deepcopy(preparation["source_recapture"])
    from .painted_transfer_replay import source_context_from_preparation
    result["preparation_context"] = source_context_from_preparation(preparation, folder)
    path = (folder if destination is None else Path(destination)) / "gap_review.json"
    with path.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return read_gap_review(path)


def build_transfer_gap_review(folder, layers, *, painting_binding, geometry_index,
                              map_version, contracts, fixture_underlay=None, progress=None, display_context=None,
                              run_progress=None):
    """Original native kernels beside unchanged camera-U placement figures.

    These are native review products, never a preparation/index or an adoption.
    Existing models/certificates are consumed; no RAW decode or basis build runs.
    """
    from .map_placement import read_run_geometry
    from .map_adoption import native_labels, native_with_continuation
    from .photo_edge_continuation import save_contracts, verify_source, verify_capture, verify_placement_inputs, verify_product
    from .painted_adoption_table import checked_fixture_analysis
    folder = Path(folder) / "native_gap_review"
    folder.mkdir(exist_ok=False)
    models = read_run_geometry(geometry_index)
    reference = painting_binding["record"]["run_id"].casefold()
    source = np.ascontiguousarray(layers, dtype=np.uint8)
    snapshot = dict(format="darsia-fluidflower-painted-map-adoption-preparation-v2",
        reference_run=reference, painting_facies=deepcopy(painting_binding),
        geometry_index=_bound(geometry_index), map_version=_json(map_version), runs=[])
    if display_context is not None:
        snapshot["display_context"] = _display_context(display_context)
    if fixture_underlay is not None:
        from .layer_painting_fixtures import save_underlay
        checked, derivation = checked_fixture_analysis(painting_binding, fixture_underlay)
        source = np.ascontiguousarray(checked.analysis_layers)
        saved = save_underlay(folder / "source_fixture_underlay.npz", checked)
        snapshot.update(source_fixture_underlay=dict(saved, path="source_fixture_underlay.npz"),
                        painting_analysis_derivation=derivation, fixture_frame=deepcopy(painting_binding["frame"]))
    verify_placement_inputs(contracts, models, geometry_index, reference, list(contracts), "native")
    np.savez_compressed(folder / "source_map.npz", array=source)
    save_contracts(folder / "continuation.json", contracts)
    verify_source(painting_binding, source, contracts, preparation=snapshot, folder=folder)
    for number, run in enumerate(contracts, 1):
        if run_progress is not None:
            run_progress(dict(phase="native_review", completed=number - 1, total=len(contracts), run_id=run))
        if progress is not None:
            progress(f"Checking native own-frame coverage in {run.upper()} ({number} of {len(contracts)})…")
        labels, filled, support, composition = native_labels(source, models[reference], models[run])
        classified, replayed = native_with_continuation(source, models[reference], models[run], contract=contracts[run])
        if replayed != composition:
            raise ValueError("Native review changed the original preparation composition")
        labels, filled, support = (classified[key] for key in ('labels', 'filled', 'support'))
        target = folder / run
        target.mkdir()
        product = target / "labels.npz"
        np.savez_compressed(product, array=labels.astype(np.int32), metadata={}, filled=filled, support=support,
            **{key: classified[key] for key in ("inside", "continued", "effective_coverage", "gaps")})
        verify_product(product, source, contracts[run], counts=classified["counts"])
        underlay_fields = {}
        if fixture_underlay is not None:
            from .layer_painting_fixtures import project_underlay, save_underlay
            native_underlay = project_underlay(checked, models[reference], models[run], labels,
                frame_boundary_samples=classified.get('frame_boundary_samples'))
            bound_underlay = save_underlay(target / "fixture_underlay.npz", native_underlay)
            bound_underlay["path"] = f"{run}/fixture_underlay.npz"
            underlay_fields = dict(fixture_underlay=bound_underlay,
                fixture_underlay_layer_ids=[int(v) for v in np.unique(native_underlay.drawn_layers) if v])
        snapshot["runs"].append(dict(**underlay_fields, run_id=run, status="prepared", labels=dict(_bound(product), path=f"{run}/labels.npz"),
            reference_model=_bound(models[reference].source), target_model=_bound(models[run].source),
            composition_reference_u_to_native=composition))
        if run_progress is not None:
            run_progress(dict(phase="native_review", completed=number, total=len(contracts), run_id=run))
    if progress is not None:
        progress('Saving and checking the transferred maps and run reviews…')
    verify_capture(contracts)
    verify_source(painting_binding, source, contracts, preparation=snapshot, folder=folder)
    return build_gap_review(snapshot, folder)


def verify_analysis_exclusions(preparation, folder, approval, *, run_id=None):
    """Replay an explicit approval against scientifically identical fresh products."""
    approval = approval_records(approval)
    review = read_gap_review(approval["review"]["path"])
    if not _same(review["source"], _source(preparation)):
        raise ValueError("Gap approval belongs to another source, model index, frame or policy")
    if not _same_display_source(review.get("display_context"), preparation.get("display_context")):
        raise ValueError("The reviewed image roles, series order or actual source closure changed")
    old = {row["run_id"]: row for row in review["runs"]}
    actions = {(row["run_id"], row["component_id"]): row for row in approval["runs"]}
    result = {}
    for path in (review["source_map"], _bound(Path(folder) / "source_map.npz")):
        with np.load(_preset_io.source(_checked(path)["path"]), allow_pickle=False) as archive:
            if not _same(_identity(archive["array"]), review["source_map_pixels"]):
                raise ValueError("The reviewed source map pixels changed")
    from .photo_edge_continuation import read_contracts
    contracts = read_contracts(Path(folder) / "continuation.json")["runs"]
    for entry in preparation["runs"]:
        if entry.get("status") != "prepared":
            continue
        run = entry["run_id"]
        if run_id is not None and run != run_id:
            continue
        from .painted_target_integration import effective_product
        arrays, _ = effective_product(preparation, folder, entry)
        chart_changed = run in (preparation.get("chart_transfers") or {})
        if not arrays["gaps"].any() and not (chart_changed and old.get(run, {}).get('components')):
            continue
        if run not in old:
            raise ValueError(f"{run}: unmapped pixels still need explicit human approval")
        reviewed = old[run]
        original = _arrays(reviewed["product"])
        reviewed_arrays = arrays
        if chart_changed and review.get("chart_source_review") is not None:
            # This explicit approval classifies the current chart product;
            # the full scientific-array and source checks below still apply.
            if not _same(review["chart_source_review"], preparation["painting_geometry"]["review"]):
                raise ValueError(f"{run}: current chart gap choices lost their original source review")
        elif chart_changed:
            # Existing area decisions keep their original gap pixels. A chart
            # may cover part of an approved gap, but cannot approve a new one.
            from .painted_target_integration import _effective_without_chart
            reviewed_arrays, _ = _effective_without_chart(preparation, folder, entry)
            if np.any(arrays["gaps"] & ~reviewed_arrays["gaps"]):
                raise ValueError(f"{run}: newly exposed unmapped pixels need human review")
        target_rows = (preparation.get("target_corrections") or {}).get("runs") or {}
        current_target = target_rows.get(run) or {}
        if (not _same({key: _identity(value) for key, value in original.items()}, reviewed["scientific_arrays"])
                or not _same(reviewed["scientific_arrays"], {key: _identity(value) for key, value in reviewed_arrays.items()})
                or not _same(reviewed["reference_model"], entry["reference_model"])
                or not _same(reviewed["target_model"], entry["target_model"])
                or not _same(reviewed["contract"], contracts[run])
                or not _same(reviewed["target_correction_bindings"], current_target.get("bindings", []))):
            raise ValueError(f"{run}: reviewed scientific pixels or exact source bindings changed")
        actual_components = list(_components(reviewed_arrays["gaps"]))
        if any((run, component["component_id"]) not in actions for component, _ in actual_components):
            raise ValueError(f"{run}: each connected unmapped area needs its own explicit human approval")
        if {item["component_id"] for item in reviewed["components"]} != {item["component_id"] for item, _ in actual_components}:
            raise ValueError(f"{run}: connected gap areas changed")
        if approval["format"] == APPROVAL_FORMAT_V2:
            from .painted_gap_regions import read_gap_region_partition
            for component, mask in actual_components:
                read_gap_region_partition(actions[(run, component["component_id"])], mask)
        result[run] = arrays["gaps"]
    expected_runs = {entry["run_id"] for entry in preparation["runs"] if entry.get("status") == "prepared"}
    if {key[0] for key in actions} - expected_runs:
        raise ValueError("Approval includes a run outside this preparation")
    if run_id is None and {key[0] for key in actions} != set(result):
        raise ValueError("Approval includes a run without current gaps")
    if run_id is not None and any(key[0] == run_id for key in actions) and run_id not in result:
        raise ValueError("Approval includes a run without current gaps")
    return result


def prepare_exclusions(preparation, folder, approval):
    """Persist the explicit approval and exact separate masks in a new folder."""
    masks = verify_analysis_exclusions(preparation, folder, approval)
    folder = Path(folder)
    path = folder / "analysis_gap_approval.json"
    with path.open("x", encoding="utf8") as stream:
        json.dump(approval_records(approval), stream, ensure_ascii=False, allow_nan=False, indent=2)
    rows = {}
    for run, mask in masks.items():
        target = folder / run / "analysis_exclusion.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, approved_analysis_gap=mask)
        rows[run] = dict(artifact=_bound(target), field="approved_analysis_gap", identity=_identity(mask))
    return dict(format=EXCLUSION_FORMAT_V2 if approval["format"] == APPROVAL_FORMAT_V2 else EXCLUSION_FORMAT,
                meaning=MEANING, approval=_bound(path), runs=rows)


def prepared_exclusion(preparation, folder, run):
    """Verify one stored mask through its approval and complete current run replay."""
    contract = preparation.get(KEY)
    if contract is None:
        return None
    if contract.get("format") not in (EXCLUSION_FORMAT, EXCLUSION_FORMAT_V2) or contract.get("meaning") != MEANING:
        raise ValueError("Unknown analysis-exclusion contract")
    approval = _checked(contract["approval"], metadata=True)
    if (isinstance(approval, dict) and
            (contract["format"] == EXCLUSION_FORMAT_V2) != (approval.get("format") == APPROVAL_FORMAT_V2)):
        raise ValueError("The analysis exclusion and approval versions differ")
    masks = verify_analysis_exclusions(preparation, folder, approval, run_id=run)
    if set(contract["runs"]) != {row["run_id"] for row in approval["runs"]}:
        raise ValueError("Stored analysis exclusions differ from their human approval")
    for key, expected in masks.items():
        row = contract["runs"][key]
        with np.load(_preset_io.source(_checked(row["artifact"])["path"]), allow_pickle=False) as archive:
            actual = np.asarray(archive[row["field"]])
        if actual.dtype != bool or not _same(_identity(actual), row["identity"]) or not np.array_equal(actual, expected):
            raise ValueError("The stored approved-analysis-gap mask changed")
    return masks.get(run)


def configuration_exclusion(config, run_id=None, *, labels=None):
    """Resolve only the exact optional mask of a genuine painted preparation."""
    contract = config.get(KEY)
    table = config.get("painted_adoption_table") or {}
    if contract is None:
        if table.get("preparation"):
            preparation = _checked(table["preparation"], metadata=True)
            if KEY in preparation:
                raise ValueError("The adopted configuration lost its approved analysis exclusions")
        return None
    if not table.get("preparation"):
        raise ValueError("Analysis exclusions require their genuine painted preparation")
    preparation = _checked(table["preparation"], metadata=True)
    if not _same(contract, preparation.get(KEY)):
        raise ValueError("The configured analysis exclusions differ from their preparation")
    run = run_id or (config.get("frame_binding") or {}).get("display_reference_run")
    mask = prepared_exclusion(preparation, Path(table["preparation"]["path"]).parent, run)
    if mask is not None and labels is not None and (mask.shape != labels.shape or np.any(mask & (labels != 0))):
        raise ValueError("Approved analysis gaps cannot mask mapped labels or another native grid")
    return mask


def exclusion_provenance(config, run_id):
    """Retain the approval and source mask without inventing physical values."""
    mask = configuration_exclusion(config, run_id)
    if mask is None:
        return None
    row = config[KEY]["runs"][run_id]
    return dict(approval=deepcopy(config[KEY]["approval"]), run_id=run_id,
                native_mask=deepcopy(row), meaning=MEANING,
                physical_frame_unchanged=True, labels_unchanged=True)


def native_exclusion_contract(config, run_id, *, exported_labels):
    """Bind the optional unquantified mask for the actual native Build caller."""
    mask = configuration_exclusion(config, run_id)
    if mask is None:
        return None
    table = config["painted_adoption_table"]
    preparation = _checked(table["preparation"], metadata=True)
    folder = Path(table["preparation"]["path"]).parent
    entry = next(row for row in preparation["runs"] if row["run_id"] == run_id)
    from .geometry_facies import _load_array
    labels = np.asarray(_load_array(Path(_checked(exported_labels)["path"])))
    configuration_exclusion(config, run_id, labels=labels)
    from .painted_target_integration import effective_product
    original, effective_labels = effective_product(preparation, folder, entry)
    if labels.dtype != original["array"].dtype or not np.array_equal(labels, original["array"]):
        raise ValueError("The exported native labels differ from the approved gap's source grid")
    model = _checked(entry["target_model"], metadata=True)
    row = config[KEY]["runs"][run_id]
    return dict(format="darsia-approved-analysis-gap-native-v1", meaning=MEANING, run_id=run_id,
        preparation=deepcopy(table["preparation"]),
        approval=deepcopy(config[KEY]["approval"]), mask=deepcopy(row["artifact"]), field=row["field"],
        identity=deepcopy(row["identity"]), source_labels=deepcopy(effective_labels),
        exported_labels=deepcopy(exported_labels), source_model=deepcopy(entry["target_model"]),
        native_grid=deepcopy(model["baseline_native_grid"]), source_to_export_native_h=np.eye(3).tolist(),
        provenance=exclusion_provenance(config, run_id))


def adopted_run_analysis_exclusions(revision, run_id, *, project_root, registered_sources=None):
    """Metadata-only projection shared by planning and export; not acceptance."""
    from .painted_native_product import adopted_run_analysis_exclusions as project
    return project(revision, run_id, project_root=project_root, registered_sources=registered_sources)


def adopted_run_native_product(revision, run_id, *, project_root, registered_sources=None):
    from .painted_native_product import adopted_run_native_product as project
    return project(revision, run_id, project_root=project_root, registered_sources=registered_sources)


def native_analysis_exclusion_dependencies(contract, reader=None):
    from .painted_native_product import native_analysis_exclusion_dependencies as dependencies
    return dependencies(contract, reader)


def read_native_product(contract, *, actual_labels, run_id, reader=None):
    from .painted_native_product import read_native_product as read
    return read(contract, actual_labels=actual_labels, run_id=run_id, reader=reader)


def read_native_analysis_exclusions(contract, *, actual_labels, run_id, reader=None):
    """Read the exact optional native contract; the caller separately proves Build H/grid."""
    from .painted_target_integration import effective_product
    contract = _json(contract)
    required = {"format", "meaning", "run_id", "preparation", "approval", "mask", "field", "identity",
        "source_labels", "exported_labels", "source_model", "native_grid", "source_to_export_native_h", "provenance"}
    if (set(contract) != required or contract["format"] != "darsia-approved-analysis-gap-native-v1"
            or contract["meaning"] != MEANING or contract["run_id"] != run_id
            or contract["field"] != "approved_analysis_gap"
            or not _same(contract["source_to_export_native_h"], np.eye(3).tolist())):
        raise ValueError("The native analysis exclusion contract changed")
    dependencies = native_analysis_exclusion_dependencies(contract, reader)
    preparation = _checked(contract["preparation"], metadata=True)
    folder = Path(contract["preparation"]["path"]).parent
    entry = next(row for row in preparation["runs"] if row["run_id"] == run_id and row.get("status") == "prepared")
    from .painted_native_product import verify_selected_native
    verify_selected_native(preparation, folder, entry)
    arrays, labels = effective_product(preparation, folder, entry)
    if (not isinstance(actual_labels, np.ndarray) or actual_labels.dtype != arrays["array"].dtype
            or not np.array_equal(actual_labels, arrays["array"])
            or not _same(labels, contract["source_labels"])):
        raise ValueError("Native analysis exclusions require their exact effective labels")
    model = _checked(entry["target_model"], metadata=True)
    declared = preparation[KEY]
    row = declared["runs"][run_id]
    if (not _same(contract["source_model"], entry["target_model"])
            or not _same(contract["native_grid"], model["baseline_native_grid"])
            or not _same(contract["approval"], declared["approval"])
            or not _same(contract["mask"], row["artifact"])
            or not _same(contract["identity"], row["identity"])):
        raise ValueError("Native analysis exclusions changed their approved source/model/mask")
    exported = _checked(contract["exported_labels"])
    from .geometry_facies import _load_array
    if not np.array_equal(_load_array(Path(exported["path"])), arrays["array"]):
        raise ValueError("Native analysis exclusions changed their exported effective labels")
    mask = prepared_exclusion(preparation, folder, run_id)
    if mask is None or mask.shape != actual_labels.shape or np.any(mask & (actual_labels != 0)):
        raise ValueError("The native approved-analysis-gap mask changed")
    provenance = dict(approval=deepcopy(declared["approval"]), run_id=run_id,
        native_mask=deepcopy(row), meaning=MEANING, physical_frame_unchanged=True, labels_unchanged=True)
    if not _same(provenance, contract["provenance"]):
        raise ValueError("The native exclusion provenance changed")
    return dict(mask=mask.copy(), provenance=provenance, contract=contract,
                native_grid=deepcopy(contract["native_grid"]), dependencies=dependencies)


def build_retained_gap_review(folder, *, painting_binding, geometry_index, map_version, runs, recapture, destination,
                             display_context=None):
    """Bind original failed-run artifacts; never rerun placement or fake an index.

    The caller supplies the genuine frozen painting closure from the original
    invocation.  Its retained artifact and model identities are checked here;
    a later fresh Prepare independently performs the original scientific replay.
    """
    from .map_placement import read_run_geometry
    from .photo_edge_continuation import read_contracts, verify_source
    from .painted_adoption_table import _history, checked_fixture_analysis
    folder = Path(folder)
    binding = _json(painting_binding)
    if Path(destination).resolve().is_relative_to(folder.resolve()):
        raise ValueError("A retained failed preparation stays read-only; use a fresh review destination")
    captured = _checked(recapture, metadata=True)
    if (captured.get("origin") != "CURRENT_VERIFIED_RECAPTURE"
            or captured.get("original_serialized_binding_claimed") is not False
            or captured.get("source_map_exact") is not True or captured.get("source_frame_inside_exact") is not True
            or not _same(captured.get("painting_binding"), binding)
            or Path(captured.get("preparation", "")).resolve() != folder.resolve()):
        raise ValueError("The retained review requires its explicit current verified recapture")
    for descriptor in [*captured["verified_inputs"], *captured["bound_anchors"]]:
        _checked(descriptor)
    actual = _checked(binding["version"], metadata=True)
    if not _same(actual, binding["record"]) or not _same(binding["approval"], _checked(_bound(folder / "approval.json"), metadata=True)):
        raise ValueError("The retained invocation needs its genuine saved version and literal approval")
    _history(Path(binding["version"]["path"]).parent, binding["approval"])
    for leaf, bound in binding["files"].items():
        if actual["files"].get(leaf) != bound["sha256"]:
            raise ValueError("The retained painting source closure changed")
        _checked(bound)
    index = _bound(geometry_index)
    geometries = read_run_geometry(geometry_index)
    contracts = read_contracts(folder / "continuation.json")["runs"]
    rows = []
    for run in runs:
        if run not in contracts or run not in geometries:
            raise ValueError("Every reviewed run needs its original model and continuation")
        contract = contracts[run]
        if not _same(contract["geometry_index"], index):
            raise ValueError("The retained run belongs to another geometry index")
        model = geometries[run]
        target = _bound(model.source)
        reference = _bound(geometries[contract["source_run"]].source)
        if (not _same(target, contract["target_model"]["descriptor"])
                or not _same(reference, contract["source_model"]["descriptor"])):
            raise ValueError("The retained model differs from the original continuation")
        product = _bound(folder / run / "labels.npz")
        arrays = _arrays(product)
        # Use the same declared model matrices as native_labels, not a new fit.
        source_geometry = geometries[contract["source_run"]]
        target_h = np.asarray(model.model["baseline_native_grid"]["H_baseline_camera_u_to_rig_corrected_native"], float) @ model.metric_to_grid
        composition = target_h @ np.linalg.inv(source_geometry.metric_to_grid)
        rows.append(dict(run_id=run, status="prepared", labels=dict(product, path=f"{run}/labels.npz"),
                         reference_model=reference, target_model=target,
                         composition_reference_u_to_native=composition.tolist()))
    snapshot = dict(format="darsia-fluidflower-painted-map-adoption-preparation-v2",
        reference_run=binding["record"]["run_id"].casefold(), painting_facies=binding,
        geometry_index=index, map_version=_json(map_version), runs=rows)
    if display_context is not None:
        snapshot["display_context"] = _display_context(display_context)
    source_path = _bound(folder / "source_map.npz")
    with np.load(_preset_io.source(source_path["path"]), allow_pickle=False) as archive:
        source_layers = np.array(archive["array"], copy=True)
    if (folder / "source_fixture_underlay.npz").is_file():
        from .layer_painting_fixtures import read_underlay
        underlay_bound = _bound(folder / "source_fixture_underlay.npz")
        _, derivation = checked_fixture_analysis(binding, read_underlay(underlay_bound))
        if not _same(derivation, captured.get("fixture_analysis_derivation")):
            raise ValueError("The retained fixture analysis differs from the verified recapture")
        snapshot.update(source_fixture_underlay=dict(underlay_bound, path="source_fixture_underlay.npz"),
                        painting_analysis_derivation=derivation, fixture_frame=deepcopy(binding["frame"]))
    verify_source(binding, source_layers, contracts, preparation=snapshot, folder=folder)
    snapshot["source_recapture"] = deepcopy(recapture)
    Path(destination).mkdir(parents=True, exist_ok=False)
    return build_gap_review(snapshot, folder, destination=destination)


def transport_build_exclusion(context, reader, matrix, origin, shape, geometry_parameters):
    """Original nearest-neighbour mask transport on the measured Build transform."""
    if "approved_analysis_gap" not in context.provenance:
        return None, None
    adopted = reader.json(context.provenance["adopted_revision"])
    config = adopted["configuration"]
    source = configuration_exclusion(config, context.run, labels=context.labels)
    if source is None:
        return None, None
    from .mass.build_frame_fields import _canvas
    from .geometry.context import homography
    import cv2
    from darsia._opencv_runtime import scientific_thread, require_ipp_off
    scientific_thread("approved-analysis-gap-transport:before")
    h = _canvas(matrix, origin, geometry_parameters)
    h = homography(h, parameters=geometry_parameters)
    result = cv2.warpPerspective(source.astype(np.uint8), h, tuple(map(int, shape[::-1])),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) != 0
    require_ipp_off("approved-analysis-gap-transport:after")
    provenance = exclusion_provenance(config, context.run)
    provenance.update(mask_identity=_identity(result), transport=dict(
        adopted_revision=deepcopy(context.provenance["adopted_revision"]),
        build_to_frame_native_h=np.asarray(matrix, float).tolist(),
        native_origin_xy=list(origin), shape_hw=list(shape), source_to_frame_h=h.tolist(),
        interpolation="original-nearest-neighbour-constant-zero-mask"))
    return result, provenance


def restrict_phase_domain(frame, gap, provenance):
    """Exclude approved gaps from phase/fill/history without changing the frame."""
    from dataclasses import replace
    if gap is None:
        return frame
    context = frame["context"]
    if gap.dtype != bool or gap.shape != context.rig_domain.shape:
        raise ValueError("Approved analysis-gap transport has another phase grid")
    if np.any(gap & context.hardware_excluded):
        raise ValueError("Approved analysis gaps overlap an existing physical fixture exclusion")
    result = dict(frame)
    # The original photographic analysis domain may already omit water/air.
    # Keep the explicitly approved physical locations even in that case;
    # do not turn their phase-255 pixels into a zero-mass conclusion.
    result["physical_rig_domain"] = context.rig_domain | gap
    result["approved_analysis_gap"] = gap.copy()
    result["approved_analysis_gap_provenance"] = deepcopy(provenance)
    keep = ~gap
    result["context"] = replace(context, rig_domain=context.rig_domain & keep,
        modelled=context.modelled & keep, filled_observed=context.filled_observed & keep,
        filled_unobserved=context.filled_unobserved & keep,
        parameters=dict(context.parameters, approved_analysis_gap=deepcopy(provenance)))
    return result


def _scientific_review_projection(review):
    """Keep live scientific pins beside already authenticated historical DATA.

    Native continuation models are still pinned by their descriptor and photo.
    Their embedded document and self-checked frame snapshots are consumed as
    frozen DATA, including after owned Apply changes a current configuration.
    Original proof readers separately verify the retained configuration copies.
    Unknown and legacy fields remain live dependencies, wherever they occur
    outside those typed DATA subtrees. This does not validate a review or proof.
    """
    from .photo_edge_continuation import FORMAT, POLICY
    result = {key: value for key, value in review.items() if key != "display_context"}
    if review.get("format") != REVIEW_FORMAT or not isinstance(review.get("runs"), list):
        return result
    rows = []
    for row in review["runs"]:
        contract = row.get("contract") if isinstance(row, dict) else None
        if (not isinstance(contract, dict) or contract.get("format") != FORMAT
                or contract.get("policy") != POLICY or contract.get("target_grid") != "native"):
            rows.append(row)
            continue
        projected = dict(contract)
        for name in ("source_model", "target_model"):
            model = contract.get(name)
            if (isinstance(model, dict) and isinstance(model.get("document"), dict)
                    and {"descriptor", "document", "run_id", "shape_hw", "photograph", "metric_to_grid"} <= set(model)):
                projected[name] = {key: value for key, value in model.items() if key != "document"}
        authority = contract.get("frame_authority")
        if isinstance(authority, dict):
            frozen = {"saved rig frame": ("project",),
                      "series configuration": ("project", "series", "common", "run")}.get(authority.get("kind"), ())
            authority = dict(authority)
            for name in frozen:
                snapshot = authority.get(name)
                if (isinstance(snapshot, dict) and isinstance(snapshot.get("text"), str)
                        and {"path", "sha256", "size_bytes", "text"} <= set(snapshot)):
                    authority[name] = {key: value for key, value in snapshot.items()
                                       if key not in {"path", "sha256", "size_bytes", "text", "bom"}}
            projected["frame_authority"] = authority
        rows.append(dict(row, contract=projected))
    result["runs"] = rows
    return result


def exclusion_sources(preparation):
    """The same consumed approval/review/source bindings for late revalidation."""
    contract = preparation.get(KEY)
    if contract is None:
        return []
    approval = _checked(contract["approval"], metadata=True)
    review = read_gap_review(approval["review"]["path"])
    result = {}
    def collect(value):
        if isinstance(value, dict):
            if {"path", "sha256", "size_bytes"} <= set(value):
                bound = {key: value[key] for key in ("path", "sha256", "size_bytes")}
                _checked(bound)
                result[(bound["path"], bound["sha256"])] = bound
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(contract)
    if approval.get("format") == APPROVAL_FORMAT_V2:
        from .painted_gap_regions import gap_region_dependencies
        for bound in gap_region_dependencies(approval):
            collect(bound)
    # Frozen review DATA keeps its manifest/model/proof authority; current
    # files are checked only when the original scientific readers consume them.
    projection = _scientific_review_projection(review)
    if projection.get("chart_preparation") is not None:
        from .painted_adoption_table import _local
        chart = deepcopy(projection["chart_preparation"])
        folder = Path(review["source_map"]["path"]).parent
        # Only the typed recipe's local artifacts use preparation-relative
        # locators; late dependency checks need their exact absolute bindings.
        for records in (chart.get("chart_transfers") or {}, (chart.get("target_corrections") or {}).get("runs") or {}):
            for record in records.values():
                record["artifact"] = _local(folder, record["artifact"])
        projection["chart_preparation"] = chart
    collect(projection)
    return list(result.values())


def _review_for_configuration(config):
    """Authenticate the scientific join, without importing display providers."""
    table = config.get("painted_adoption_table") or {}
    if not table.get("preparation"):
        return None, None
    preparation = _checked(table["preparation"], metadata=True)
    descriptor = preparation.get("gap_review")
    if descriptor is None and KEY in preparation:
        approval = _checked(preparation[KEY]["approval"], metadata=True)
        descriptor = approval["review"]
    if descriptor is None:
        if preparation.get("display_context") is not None:
            raise ValueError("The display context lost its genuine bound native findings review")
        return preparation, None
    review = _read_scientific_gap_review(descriptor["path"])
    if not _same(review["review_binding"], descriptor) or not _same(review["source"], _source(preparation)):
        raise ValueError("The original findings belong to another approved source or geometry")
    return preparation, review


def check_review_findings(config, *, display_attachments=None):
    """Reconstruct current display advisories without decoding images or RAW."""
    preparation, review = _review_for_configuration(config)
    if review is None:
        if display_attachments is not None:
            raise ValueError("Display actions/preferences require their genuine bound source context")
        return dict(issues=[])
    try:
        if not _same_display_source(review.get("display_context"), preparation.get("display_context")):
            raise ValueError("The preparation and review have different actual display source closures")
        if preparation.get("display_context") is not None:
            _display_context(preparation["display_context"], review)
        from .painted_findings_catalogue import review_findings_data
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        raise DisplayFindingsChanged(review["issues"], str(error)) from error
    return review_findings_data(review, display_attachments=display_attachments)


def check_review_issues(config, *, display_attachments=None):
    """Retain original early issues plus current source-bound display advisories."""
    return check_review_findings(config, display_attachments=display_attachments)["issues"]


def findings_resolutions_required(config):
    """New explicit reviews require closure even when every gap is painted.

    Historical preparations without a review manifest retain their original
    warning-permitting Save path. The marker belongs to the immutable bound
    preparation, so omitting optional mask/edit keys cannot bypass the new gate.
    """
    if KEY in config or "target_corrections" in config:
        return True
    from .manual_map_approval import approved_candidate
    if approved_candidate(config):
        # Exact per-run human map approvals already cover its displayed
        # placement. Preserve diagnostic warnings without another approval.
        return False
    table = config.get("painted_adoption_table") or {}
    if not table.get("preparation"):
        return False
    preparation = _checked(table["preparation"], metadata=True)
    descriptor = preparation.get("gap_review")
    if descriptor is None:
        return False
    review = read_gap_review(descriptor["path"])
    if (not _same(descriptor, review["review_binding"])
            or not _same(review["source"], _source(preparation))):
        raise ValueError("The mandatory findings review differs from its bound preparation")
    return True


def issue_identity(issue):
    """Identity of the complete original issue, including its evidence."""
    if not isinstance(issue, dict):
        raise ValueError("A genuine Check issue must be literal metadata")
    return _digest(_json(issue))


def _action(value, *, state=False):
    keys = {"actor", "approved_utc", "reason", "decision_ref"}
    if state:
        keys.add("acquisition_state")
    if not isinstance(value, dict) or set(value) != keys or any(type(value[key]) is not str or not value[key].strip() for key in keys):
        raise ValueError("A finding resolution requires an explicit human action and reason")
    return value


def make_facies_source_acquisition(preparation, state, *, actor, approved_utc, reason, decision_ref):
    """Bind one GUI choice to the actual saved painting photograph, off the GUI thread."""
    from .geometry_facies import RUN_GEOMETRY_ACQUISITION_STATES
    from .painted_adoption_table import checked_facies_source_acquisition
    if state not in RUN_GEOMETRY_ACQUISITION_STATES or type(state) is not str:
        raise ValueError("Choose an explicit facies-source acquisition state")
    decision = _action(dict(actor=actor, approved_utc=approved_utc, reason=reason, decision_ref=decision_ref))
    folder = Path(preparation["folder"]) if isinstance(preparation, dict) else Path(preparation)
    bound = _bound(folder / "index.json")
    actual = _checked(bound, metadata=True)
    if isinstance(preparation, dict) and not _same(actual, {key: value for key, value in preparation.items() if key not in {"folder", "gui_adoption_cohort"}}):
        raise ValueError("The GUI state choice belongs to another preparation")
    source = actual.get("painting_facies")
    if not isinstance(source, dict):
        raise ValueError("A GUI acquisition choice needs its approved painting photograph")
    for descriptor in [source["version"], source["frame_basis"]["photograph"], *source["files"].values()]:
        _checked(descriptor)
    declaration = dict(acquisition_state=state, preparation=bound,
        photograph=deepcopy(source["frame_basis"]["photograph"]), version=deepcopy(source["version"]),
        approval=deepcopy(source["approval"]), source_map=_bound(folder / actual["source_map"]["path"]),
        decision=dict(decision, origin="explicit-step5-gui"))
    return checked_facies_source_acquisition(actual, folder, declaration)


def seal_findings_resolutions(candidate, original_report, records, *, display_attachments=None):
    """Resolve exact current findings without removing or downgrading any issue."""
    from .geometry_facies import geometry_config_digest
    candidate, report, records = _json(candidate), _json(original_report), _json(records)
    digest = geometry_config_digest(candidate)
    if report.get("config_sha256") != digest:
        raise ValueError("Finding resolutions require the exact current candidate Check")
    issues = report.get("issues")
    if not isinstance(issues, list) or any(issue.get("severity") == "error" for issue in issues):
        raise ValueError("Errors cannot be approved as findings")
    display = check_review_findings(candidate, display_attachments=display_attachments)
    if "display_findings_sha256" in display:
        if (report.get("display_findings_sha256") != display["display_findings_sha256"]
                or not _same(report.get("display_attachments"), display["display_attachments"])
                or (display["issues"] and not _same(issues[-len(display["issues"]):], display["issues"]))):
            raise ValueError("Display source, actions, preferences or current advisories changed since Check")
    elif "display_findings_sha256" in report or "display_attachments" in report:
        raise ValueError("The Check claims display findings without their actual bound source")
    if set(records) != {"warnings", "acquisition", KEY} or not isinstance(records["warnings"], list) or not isinstance(records["acquisition"], list):
        raise ValueError("Complete explicit findings records are required")
    expected = {index: issue_identity(issue) for index, issue in enumerate(issues) if issue.get("severity") == "warning"}
    seen = set()
    for action in records["warnings"]:
        if not isinstance(action, dict) or set(action) != {"issue_index", "issue_sha256", "actor", "approved_utc", "reason", "decision_ref"}:
            raise ValueError("Each warning resolution needs its original issue index and identity")
        index = action["issue_index"]
        _action({key: value for key, value in action.items() if key not in {"issue_index", "issue_sha256"}})
        if type(index) is not int or index in seen or expected.get(index) != action["issue_sha256"]:
            raise ValueError("A warning resolution belongs to a changed or different issue")
        seen.add(index)
    if seen != set(expected):
        raise ValueError("Every original warning requires its own explicit human resolution")
    declaration = (candidate.get("provenance") or {}).get("facies_source_acquisition")
    painting = candidate.get("painted_adoption_table")
    if painting:
        actions = records["acquisition"]
        if len(actions) != 1 or not isinstance(declaration, dict):
            raise ValueError("The painting source state requires an explicit GUI choice")
        action = _action(actions[0], state=True)
        decision = declaration.get("decision") or {}
        if (declaration.get("acquisition_state") != action["acquisition_state"]
                or not _same(decision, dict({key: value for key, value in action.items() if key != "acquisition_state"}, origin="explicit-step5-gui"))
                or any(row.get("acquisition_state") != action["acquisition_state"] for row in (candidate.get("run_geometry") or {}).get("runs", []) if row.get("enabled", True))):
            raise ValueError("The GUI acquisition choice differs from the bound source or transported rows")
    elif records["acquisition"]:
        raise ValueError("An acquisition action needs its actual painted source")
    if KEY in candidate:
        approval = _checked(candidate[KEY]["approval"], metadata=True)
        if not _same(records[KEY], approval):
            raise ValueError("Finding resolutions lost their exact approved connected areas")
        approval_records(approval)
    elif records[KEY] is not None:
        raise ValueError("An area resolution needs its actual analysis exclusion contract")
    sealed = dict(format=RESOLUTIONS_FORMAT, candidate_sha256=digest,
                  issues_sha256=_digest(issues), records=records)
    if "display_findings_sha256" in display:
        sealed.update(display_attachments=display["display_attachments"],
                      display_findings_sha256=display["display_findings_sha256"])
    return sealed


def verify_findings_resolutions(candidate, fresh_report, sealed):
    """Fail closed on changed candidates, errors, issues, source state or areas."""
    sealed = _json(sealed)
    original_keys = {"format", "candidate_sha256", "issues_sha256", "records"}
    display_keys = {"display_attachments", "display_findings_sha256"}
    if (not isinstance(sealed, dict) or set(sealed) not in (original_keys, original_keys | display_keys)
            or sealed["format"] != RESOLUTIONS_FORMAT
            or not _same(sealed, seal_findings_resolutions(candidate, fresh_report, sealed["records"],
                display_attachments=sealed.get("display_attachments")))):
        raise ValueError("The resolved findings differ from the exact current Check")
    return sealed
