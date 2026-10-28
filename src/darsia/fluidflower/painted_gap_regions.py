"""Explicit human region classes on one unchanged reviewed native gap.

Class IDs are partition DATA, never facies or physical properties. Sand remains
an editable preview choice and cannot enter an outside-analysis approval. A
worker saves only an explicit confirmation against the original current review;
the existing consumer still replays the complete scientific closure separately.
"""
from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

APPROVAL_FORMAT = "darsia-painted-gap-exclusion-approval-v2"
MEANING = "outside-analysis-unquantified"
CLASSIFICATIONS = ("air", "free_water", "sand", "rig_or_frame", "other")
CLASSIFICATION_LABELS = {
    "air": "Air", "free_water": "Free water", "sand": "Sand",
    "rig_or_frame": "Rig or frame", "other": "Other",
}
REGION_CLASSIFICATIONS = CLASSIFICATIONS
REGION_CLASSIFICATION_LABELS = CLASSIFICATION_LABELS
DECISION_TYPE = "outside_analysis_non_sand"


def _gap():
    """Use the original review, byte verification and identity implementation."""
    from . import painted_analysis_exclusions
    return painted_analysis_exclusions


def _descriptor(value):
    """Validate an exact literal file descriptor without touching its path."""
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or type(value["path"]) is not str or not Path(value["path"]).is_absolute()
            or type(value["size_bytes"]) is not int or value["size_bytes"] < 0
            or type(value["sha256"]) is not str
            or re.fullmatch("[0-9a-f]{64}", value["sha256"]) is None):
        raise ValueError("A literal absolute path/SHA/size descriptor is required")
    return deepcopy(value)


def _frozen(array):
    """Detach pixels into immutable bytes, including their native shape/dtype."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _selected(review, run_id, component_id):
    """Find one original component without fitting, merging or rebasing it."""
    rows = [row for row in review["runs"] if row["run_id"] == run_id]
    components = ([] if len(rows) != 1 else
                  [item for item in rows[0]["components"] if item["component_id"] == component_id])
    if len(components) != 1:
        raise ValueError("Choose one original connected area from this exact run review")
    return rows[0], components[0]


def _context_from_review(review, run_id, component_id):
    """Select tiny context from already-verified review DATA without cloning it.

    The worker authenticates the complete retained review before saving. This
    GUI boundary keeps only the selected original identity and raw binding.
    """
    if (not isinstance(review, dict) or review.get("format") != _gap().REVIEW_FORMAT
            or review.get("meaning") != MEANING or not isinstance(review.get("runs"), list)):
        raise ValueError("An already-verified original gap review is required")
    row, component = _selected(review, run_id, component_id)
    context = dict(review=_descriptor(review.get("review_binding")), run_id=run_id,
        component_id=component_id, review_row_sha256=component.get("review_row_sha256"),
        component_identity=deepcopy(component.get("identity")), shape_hw=deepcopy(row.get("gap", {}).get("shape_hw")))
    return _context_metadata(context)


def _context_metadata(context):
    """Validate one tiny native context without file or full-review traversal."""
    if (not isinstance(context, dict) or set(context) != {"review", "run_id", "component_id",
            "review_row_sha256", "component_identity", "shape_hw"}
            or type(context["run_id"]) is not str or not context["run_id"]
            or any(type(context[key]) is not str or re.fullmatch("[0-9a-f]{64}", context[key]) is None
                   for key in ("component_id", "review_row_sha256"))):
        raise ValueError("The selected native component context changed")
    _descriptor(context["review"])
    shape, identity = context["shape_hw"], context["component_identity"]
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(value) is not int or value < 1 for value in shape)
            or not isinstance(identity, dict) or set(identity) != {"dtype", "shape", "sha256"}
            or identity["dtype"] != np.dtype(bool).str or identity["shape"] != shape
            or identity["sha256"] != context["component_id"]):
        raise ValueError("The selected component must retain its exact native boolean identity")
    return deepcopy(context)


class GapRegionDraft:
    """Native class strokes on one component; snapshots carry no approval."""

    def __init__(self, *, review, run_id, component_id, editable_mask):
        """Accept only a supplied review and its exact native component pixels."""
        context = _context_from_review(review, run_id, component_id)
        if (not isinstance(editable_mask, np.ndarray) or editable_mask.dtype != bool
                or editable_mask.ndim != 2 or not editable_mask.any()
                or not _gap()._same(_gap()._identity(editable_mask), context["component_identity"])):
            raise ValueError("The verified packet must contain the exact native gap component")
        self._context_json = json.dumps(context, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self._editable_mask = _frozen(editable_mask)
        self._events = []
        self._sealed = False
        self._seed = _frozen(np.zeros(editable_mask.shape, np.uint8))
        self._codes = self._seed
        self._preview_context_json = None

    @property
    def selected_context(self):
        """Return detached tiny original context, without the cohort review body."""
        return json.loads(self._context_json)

    @property
    def review_binding(self):
        """Return the unchanged original raw review descriptor."""
        return self.selected_context["review"]

    @property
    def run_id(self):
        """Return this run's literal identity."""
        return self.selected_context["run_id"]

    @property
    def component_id(self):
        """Return the original connected-component identity."""
        return self.selected_context["component_id"]

    @property
    def review_row_sha256(self):
        """Return the exact original per-component review context hash."""
        return self.selected_context["review_row_sha256"]

    @property
    def editable_mask(self):
        """Expose immutable native component pixels, never a display transform."""
        return self._editable_mask

    @property
    def class_masks(self):
        """Expose immutable disjoint class masks without scientific layer IDs."""
        return MappingProxyType({name: _frozen(self._codes == index)
                                 for index, name in enumerate(CLASSIFICATIONS, 1)})

    @property
    def preview_source_context(self):
        """Return detached origin metadata for a seed; it grants no approval."""
        return None if self._preview_context_json is None else json.loads(self._preview_context_json)

    def set_preview_class_masks(self, class_masks, *, source_context):
        """Seed a fresh target draft with an explicitly nonapproved preview.

        The caller verifies the coordinate transport. Origin context may name
        another run, while this draft retains its own original target basis.
        """
        if self._sealed or self._events or self._preview_context_json is not None:
            raise ValueError("Preview classes may seed only a fresh editable target draft")
        if not isinstance(class_masks, Mapping) or set(class_masks) != set(CLASSIFICATIONS):
            raise ValueError("Supply all five explicit native preview class masks")
        if (not isinstance(source_context, dict) or set(source_context) != {
                "review", "run_id", "component_id", "review_row_sha256"}
                or type(source_context["run_id"]) is not str or not source_context["run_id"]
                or any(type(source_context[key]) is not str or re.fullmatch("[0-9a-f]{64}", source_context[key]) is None
                       for key in ("component_id", "review_row_sha256"))):
            raise ValueError("A preview needs its literal origin context, without approval claims")
        _descriptor(source_context["review"])
        codes = np.zeros(self.editable_mask.shape, np.uint8)
        for index, name in enumerate(CLASSIFICATIONS, 1):
            mask = class_masks[name]
            if (not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != codes.shape
                    or np.any(mask & ~self.editable_mask) or np.any(mask & (codes != 0))):
                raise ValueError("Preview classes must be disjoint native bool masks within this component")
            codes[mask] = index
        self._seed = _frozen(codes)
        self._codes = self._seed
        self._preview_context_json = json.dumps(source_context, ensure_ascii=False, sort_keys=True, allow_nan=False)

    @property
    def events(self):
        """Return detached complete stroke/undo/redo history."""
        return deepcopy(self._events)

    def _event_points(self, event):
        """Validate native stroke metadata without rasterizing any image."""
        kind, name, radius = event["kind"], event["class_name"], event["radius_px"]
        points = np.asarray(event["points_xy"], dtype=np.float64)
        h, w = self.editable_mask.shape
        if (type(kind) is not str or kind not in {"polygon", "brush"}
                or type(name) is not str or name not in CLASSIFICATIONS
                or points.ndim != 2 or points.shape[1:] != (2,)
                or not np.isfinite(points).all() or len(points) < (3 if kind == "polygon" else 1)
                or np.any(points < 0) or np.any(points[:, 0] > w - 1) or np.any(points[:, 1] > h - 1)
                or type(radius) not in {int, float} or not np.isfinite(radius)
                or not 0 < radius <= max(h, w)):
            raise ValueError("Use an explicit class and finite native-pixel strokes inside this grid")
        return np.rint(points).astype(np.int32), max(1, int(round(radius)))

    def _event_mask(self, event):
        """Rasterize validated native points with the original shared kernel."""
        from .painted_target_corrections import _native_paint_mask
        pixels, radius = self._event_points(event)
        return _native_paint_mask(event["kind"], pixels, radius, self.editable_mask.shape) & self.editable_mask

    def _state(self):
        """Replay the original latest-stroke undo/redo ordering."""
        active, redo, operations = [], [], {}
        for index, event in enumerate(self._events):
            if (not isinstance(event, dict) or type(event.get("event_id")) is not int
                    or event["event_id"] != index + 1):
                raise ValueError("Region strokes must retain their complete original order")
            kind = event.get("kind")
            if type(kind) is not str:
                raise ValueError("Unknown region paint event")
            if kind in {"polygon", "brush"}:
                if set(event) != {"event_id", "kind", "points_xy", "class_name", "radius_px"}:
                    raise ValueError("Unknown region stroke fields")
                self._event_points(event)
                operations[event["event_id"]] = event
                active.append(event["event_id"])
                redo.clear()
            elif kind in {"undo", "redo"}:
                if set(event) != {"event_id", "kind", "target_event_id"} or type(event["target_event_id"]) is not int:
                    raise ValueError("Unknown region undo/redo fields")
                source, target = (active, redo) if kind == "undo" else (redo, active)
                if not source or event["target_event_id"] != source[-1]:
                    raise ValueError("Region undo/redo must name the latest corresponding stroke")
                target.append(source.pop())
            else:
                raise ValueError("Unknown region paint event")
        return active, redo, operations

    def _replayed_codes(self):
        """Apply last-stroke class replacement on the same original component."""
        active, _, operations = self._state()
        codes = np.array(self._seed, copy=True)
        for key in active:
            event = operations[key]
            mask = self._event_mask(event)
            codes[mask] = CLASSIFICATIONS.index(event["class_name"]) + 1
        return codes

    def _rebuild(self):
        """Replace draft masks with detached immutable replay results."""
        self._codes = _frozen(self._replayed_codes())

    @property
    def can_undo(self):
        """Whether the draft has an active stroke."""
        return bool(self._state()[0])

    @property
    def can_redo(self):
        """Whether the draft has an undone stroke."""
        return bool(self._state()[1])

    def paint(self, kind, points_xy_native, class_name, radius_px=1):
        """Paint one explicit class; return False if no editable pixel is hit."""
        if self._sealed:
            raise ValueError("A detached worker snapshot cannot be edited")
        event = dict(event_id=len(self._events) + 1, kind=kind,
            points_xy=_gap()._json(points_xy_native), class_name=class_name, radius_px=radius_px)
        if not self._event_mask(event).any():
            return False
        self._events.append(event)
        self._rebuild()
        return True

    def undo(self):
        """Undo the latest active stroke, retaining its complete history."""
        return self._history_action("undo")

    def redo(self):
        """Redo the latest undone stroke without changing its coordinates."""
        return self._history_action("redo")

    def _history_action(self, kind):
        """Keep worker snapshots sealed and original undo/redo semantics."""
        if self._sealed:
            raise ValueError("A detached worker snapshot cannot be edited")
        state = self._state()[0 if kind == "undo" else 1]
        if not state:
            return False
        self._events.append(dict(event_id=len(self._events) + 1, kind=kind, target_event_id=state[-1]))
        self._rebuild()
        return True

    def snapshot(self):
        """Capture a detached sealed draft before the explicit worker confirmation."""
        result = object.__new__(GapRegionDraft)
        result._context_json = self._context_json
        result._editable_mask = self._editable_mask
        result._seed = self._seed
        result._preview_context_json = self._preview_context_json
        result._events = self.events
        result._codes = _frozen(self._codes)
        result._sealed = True
        return result


def region_draft_factory(review, verified_packet):
    """Use one already-verified native preview packet without file or display I/O."""
    if not isinstance(verified_packet, dict):
        raise ValueError("An original verified native gap packet is required")
    return GapRegionDraft(review=review, run_id=verified_packet.get("run_id"),
        component_id=verified_packet.get("component_id"), editable_mask=verified_packet.get("gap_mask"))


def validate_gap_region_snapshot(snapshot, review=None):
    """Validate detached snapshot/event/mask coherence without source I/O."""
    if not isinstance(snapshot, GapRegionDraft) or snapshot._sealed is not True:
        raise ValueError("Capture a detached region snapshot before explicit confirmation")
    context = _context_metadata(snapshot.selected_context)
    if review is not None and not _gap()._same(context,
            _context_from_review(review, snapshot.run_id, snapshot.component_id)):
        raise ValueError("The region snapshot belongs to another review or post-edit basis")
    if (snapshot.editable_mask.dtype != bool
            or not _gap()._same(_gap()._identity(snapshot.editable_mask), context["component_identity"])
            or snapshot._seed.dtype != np.uint8 or snapshot._seed.shape != snapshot.editable_mask.shape
            or np.any(snapshot._seed > len(CLASSIFICATIONS))
            or np.any(snapshot._seed[~snapshot.editable_mask] != 0)
            or snapshot._codes.dtype != np.uint8 or snapshot._codes.shape != snapshot.editable_mask.shape
            or not np.array_equal(snapshot._codes, snapshot._replayed_codes())):
        raise ValueError("Region masks differ from their exact native human strokes")
    return snapshot


def _partition_metadata(action):
    """Validate one literal region action without reading its partition bytes."""
    if (not isinstance(action, dict) or set(action) != {
            "run_id", "component_id", "review_row_sha256", "actor", "approved_utc",
            "reason", "decision_ref", "partition", "regions"}
            or any(type(action.get(key)) is not str or not action[key].strip() for key in
                   ("run_id", "component_id", "actor", "approved_utc", "reason", "decision_ref"))
            or type(action["review_row_sha256"]) is not str
            or re.fullmatch("[0-9a-f]{64}", action["review_row_sha256"]) is None):
        raise ValueError("An exact explicit per-component region approval is required")
    partition = action["partition"]
    if not isinstance(partition, dict) or set(partition) != {"artifact", "field", "identity"} or partition["field"] != "region_ids":
        raise ValueError("The native int32 region partition field is required")
    _descriptor(partition["artifact"])
    identity = partition["identity"]
    if (not isinstance(identity, dict) or set(identity) != {"dtype", "shape", "sha256"}
            or identity["dtype"] != np.dtype(np.int32).str
            or not isinstance(identity["shape"], list) or len(identity["shape"]) != 2
            or any(type(value) is not int or value < 1 for value in identity["shape"])
            or type(identity["sha256"]) is not str
            or re.fullmatch("[0-9a-f]{64}", identity["sha256"]) is None):
        raise ValueError("The region partition needs its exact native int32 identity")
    regions = action["regions"]
    if not isinstance(regions, list) or not regions:
        raise ValueError("At least one explicitly painted nonsand region is required")
    ids = set()
    for region in regions:
        if (not isinstance(region, dict) or set(region) != {"region_id", "classification", "decision_type", "other_text"}
                or type(region["region_id"]) is not int or not 0 < region["region_id"] <= np.iinfo(np.int32).max
                or region["region_id"] in ids
                or type(region["classification"]) is not str
                or region["classification"] not in CLASSIFICATIONS or region["classification"] == "sand"
                or region["decision_type"] != DECISION_TYPE
                or type(region["other_text"]) is not str
                or (region["classification"] == "other" and not region["other_text"].strip())):
            raise ValueError("Only explicitly confirmed nonsand classes may enter the partition table")
        ids.add(region["region_id"])
    return action


def _approval_metadata(value):
    """Validate literal schema and descriptors without opening review or rasters."""
    value = _gap()._json(value)
    if (not isinstance(value, dict) or set(value) != {"format", "meaning", "review", "runs"}
            or value["format"] != APPROVAL_FORMAT or value["meaning"] != MEANING
            or not isinstance(value["runs"], list) or not value["runs"]):
        raise ValueError("An explicit typed region approval is required")
    _descriptor(value["review"])
    seen = set()
    for action in value["runs"]:
        _partition_metadata(action)
        key = (action["run_id"], action["component_id"])
        if key in seen:
            raise ValueError("A component may have only one exact region partition")
        seen.add(key)
    return value


def _approval_context(value, review):
    """Match strict literal actions to supplied verified DATA without cloning it."""
    value = _approval_metadata(value)
    from .storage_paths import same_storage_descriptor
    if not same_storage_descriptor(value["review"], _descriptor(review.get("review_binding"))):
        raise ValueError("The region approval belongs to another original review")
    for action in value["runs"]:
        context = _context_from_review(review, action["run_id"], action["component_id"])
        if (action["review_row_sha256"] != context["review_row_sha256"]
                or action["partition"]["identity"]["shape"] != context["shape_hw"]):
            raise ValueError("The approved native component or post-edit context changed")
    return value


def _read_bound_review(binding):
    """Check raw review SHA/size before interpreting its retained metadata."""
    binding = _descriptor(binding)
    review = _gap()._checked_gap_review(binding)
    review["review_binding"] = binding
    return _gap().read_gap_review(review)


def validate_gap_region_approval(value, review=None):
    """Return literal valid metadata; omitted review reads only its bound review.

    Supplying already-loaded review DATA keeps this path free of I/O. Partition
    pixels are always checked separately after the original scientific replay.
    """
    value = _approval_metadata(value)
    review = _read_bound_review(value["review"]) if review is None else _gap()._review_metadata(review)
    return _approval_context(value, review)


def _snapshot_partition(snapshot, *, other_text):
    """Derive canonical int32 region DATA once for save/result comparisons."""
    validate_gap_region_snapshot(snapshot)
    if type(other_text) is not str:
        raise ValueError("The explicit Other description must remain text")
    codes = snapshot._codes
    if np.any(codes == CLASSIFICATIONS.index("sand") + 1):
        raise ValueError("Sand requires the original target correction and a new residual review")
    if np.any(codes[snapshot.editable_mask] == 0):
        raise ValueError("Every actual component pixel needs an explicit nonsand region before confirmation")
    ids = np.zeros(snapshot.editable_mask.shape, np.int32)
    regions = []
    for index, name in enumerate(CLASSIFICATIONS, 1):
        if name == "sand":
            continue
        mask = codes == index
        if not mask.any():
            continue
        if name == "other" and not other_text.strip():
            raise ValueError("Describe the explicitly painted Other region")
        number = len(regions) + 1
        ids[mask] = number
        regions.append(dict(region_id=number, classification=name,
            decision_type=DECISION_TYPE, other_text=other_text if name == "other" else ""))
    return ids, regions


def validate_gap_region_result(value, review, snapshot, *, other_text="", previous=None):
    """Purely match a worker result to this exact snapshot and previous actions.

    Original human actor/time/reason checks remain with the explicit caller.
    No additional action, reclassification or previous-action replacement is
    inferred from a returned partition descriptor or fresh output namespace.
    """
    validate_gap_region_snapshot(snapshot, review)
    value = _approval_context(value, review)
    ids, regions = _snapshot_partition(snapshot, other_text=other_text)
    selected = (snapshot.run_id, snapshot.component_id)
    actual = {(row["run_id"], row["component_id"]): row for row in value["runs"]}
    old = {} if previous is None else {
        (row["run_id"], row["component_id"]): row
        for row in _approval_context(previous, review)["runs"]}
    if (set(actual) != set(old) | {selected}
            or any(not _gap()._same(actual[key], row) for key, row in old.items() if key != selected)):
        raise ValueError("A result may change only this explicitly confirmed component")
    action = actual[selected]
    if (action["review_row_sha256"] != snapshot.review_row_sha256
            or not _gap()._same(action["partition"]["identity"], _gap()._identity(ids))
            or not _gap()._same(action["regions"], regions)):
        raise ValueError("The result differs from the exact captured native region classes")
    return value


def gap_region_dependencies(approval, *, run_id=None):
    """Project literal metadata dependencies; never open a partition or infer approval.

    The original caller validates review membership separately. A selected-run
    preclaim includes no unrelated run's partition raster.
    """
    approval = _approval_metadata(approval)
    if run_id is not None and (type(run_id) is not str or not run_id):
        raise ValueError("Select a literal run identity for dependency projection")
    rows = [row for row in approval["runs"] if run_id is None or row["run_id"] == run_id]
    result = [deepcopy(approval["review"])]
    for row in rows:
        descriptor = row["partition"]["artifact"]
        if not any(_gap()._same(descriptor, old) for old in result):
            result.append(deepcopy(descriptor))
    return result


def _read_partition(action, component_mask):
    """Read one exact pinned partition, retaining literal class IDs and pixels."""
    action = _partition_metadata(_gap()._json(action))
    if (not isinstance(component_mask, np.ndarray) or component_mask.dtype != bool
            or component_mask.ndim != 2 or not component_mask.any()
            or list(component_mask.shape) != action["partition"]["identity"]["shape"]
            or _gap()._identity(component_mask)["sha256"] != action["component_id"]):
        raise ValueError("The actual replayed native component must match this action")
    descriptor = action["partition"]["artifact"]
    actual = _gap()._checked(descriptor)
    with np.load(_preset_io.source(actual["path"]), allow_pickle=False) as archive:
        if archive.files != ["region_ids"]:
            raise ValueError("Only the exact native region_ids field is supported")
        ids = np.array(archive["region_ids"], copy=True)
    if (ids.dtype != np.int32 or ids.shape != component_mask.shape
            or not _gap()._same(_gap()._identity(ids), action["partition"]["identity"])
            or np.any(ids < 0) or np.any(ids[~component_mask] != 0)
            or np.any(ids[component_mask] == 0)
            or set(int(value) for value in np.unique(ids[component_mask])) !=
               {row["region_id"] for row in action["regions"]}):
        raise ValueError("The confirmed regions must exactly partition the actual native component")
    _gap()._checked(descriptor)
    return ids


def read_gap_region_partition(action, component_mask):
    """Return the exact native bool union after the original scientific replay.

    The caller supplies the actual replayed component. No partition or class
    changes physical labels or grants independent scientific acceptance.
    """
    return _read_partition(action, component_mask) > 0


def read_confirmed_gap_regions(review, approval_binding, *, run_id, component_id):
    """Authenticate retained human classes for display on this run's native grid.

    This is a source-bound read, not a replacement for the original adoption
    consumer's full scientific replay against its current preparation.
    """
    review = _gap().read_gap_review(review)
    approval_binding = _descriptor(approval_binding)
    literal = _gap()._checked(approval_binding, metadata=True)
    approval = validate_gap_region_approval(literal, review)
    actions = [row for row in approval["runs"]
               if row["run_id"] == run_id and row["component_id"] == component_id]
    if len(actions) != 1:
        raise ValueError("This run/component has no exact confirmed region action")
    action = actions[0]
    packet = _gap().read_gap_preview(review, run_id, component_id)
    ids = _read_partition(action, packet["gap_mask"])
    masks = {name: np.zeros(ids.shape, bool) for name in CLASSIFICATIONS}
    for region in action["regions"]:
        masks[region["classification"]] |= ids == region["region_id"]
    row, _ = _selected(review, run_id, component_id)
    _gap()._checked(approval_binding)
    _read_bound_review(review["review_binding"])
    return dict(run_id=run_id, component_id=component_id,
        review_row_sha256=action["review_row_sha256"], component_mask=_frozen(packet["gap_mask"]),
        class_masks=MappingProxyType({name: _frozen(mask) for name, mask in masks.items()}),
        regions=deepcopy(action["regions"]),
        human_action={key: action[key] for key in ("actor", "approved_utc", "reason", "decision_ref")},
        bindings=dict(review=deepcopy(approval["review"]), approval=approval_binding,
                      partition=deepcopy(action["partition"]["artifact"])),
        target_correction_bindings=deepcopy(row["target_correction_bindings"]))


def _current_snapshot(snapshot):
    """Recheck original review bytes, native masks and photo before publication."""
    validate_gap_region_snapshot(snapshot)
    review = _read_bound_review(snapshot.review_binding)
    validate_gap_region_snapshot(snapshot, review)
    packet = _gap().read_gap_preview(review, snapshot.run_id, snapshot.component_id)
    if not np.array_equal(packet["gap_mask"], snapshot.editable_mask):
        raise ValueError("The current native component differs from the captured draft")
    return review


def _destination(destination, review):
    """Reserve one fresh directory outside immutable reviews and images."""
    from .rig_raw_write import _no_links
    root = Path(destination).expanduser()
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("An explicit fresh absolute region-approval destination is required")
    _no_links(root)
    resolved = root.resolve()
    if resolved.is_relative_to(Path(review["review_binding"]["path"]).resolve().parent):
        raise ValueError("Keep region approvals outside the immutable reviewed products")
    def protect_images(value):
        if isinstance(value, dict):
            raw = value.get("path")
            if isinstance(raw, str) and Path(raw).suffix.lower() in {
                    ".arw", ".raw", ".dng", ".nef", ".cr2", ".cr3", ".jpg", ".jpeg"}:
                if resolved.is_relative_to(Path(raw).resolve().parent):
                    raise ValueError("Region approvals cannot be saved in an original image folder")
            for item in value.values():
                protect_images(item)
        elif isinstance(value, list):
            for item in value:
                protect_images(item)
    protect_images(review)
    root.mkdir(parents=True, exist_ok=False)
    return root


def save_gap_region_approval(snapshot, destination, *, actor, approved_utc, reason,
                             decision_ref, other_text="", previous=None):
    """Publish only an explicitly confirmed complete nonsand partition.

    Sand must first follow the original target correction route. Its resulting
    residual review then needs its own new native region confirmation. A failed
    save retains every partial output and grants no approval.
    """
    review = _current_snapshot(snapshot)
    if (any(type(value) is not str or not value.strip() for value in
            (actor, approved_utc, reason, decision_ref)) or type(other_text) is not str):
        raise ValueError("Explicit confirmation requires its original actor, time, reason and decision reference")
    ids, regions = _snapshot_partition(snapshot, other_text=other_text)
    approval = dict(format=APPROVAL_FORMAT, meaning=MEANING,
        review=deepcopy(review["review_binding"]), runs=[])
    actions = {}
    if previous is not None:
        old = validate_gap_region_approval(previous, review)
        actions.update({(row["run_id"], row["component_id"]): deepcopy(row) for row in old["runs"]})
        for row in old["runs"]:
            _gap()._checked(row["partition"]["artifact"])
    folder = _destination(destination, review)
    partition_path = folder / "region_partition.npz"
    with partition_path.open("xb") as stream:
        np.savez_compressed(stream, region_ids=ids)
    partition = dict(artifact=_gap()._bound(partition_path), field="region_ids", identity=_gap()._identity(ids))
    actions[(snapshot.run_id, snapshot.component_id)] = dict(run_id=snapshot.run_id,
        component_id=snapshot.component_id, review_row_sha256=snapshot.review_row_sha256,
        actor=actor, approved_utc=approved_utc, reason=reason, decision_ref=decision_ref,
        partition=partition, regions=regions)
    approval["runs"] = [actions[(row["run_id"], component["component_id"])]
        for row in review["runs"] for component in row["components"]
        if (row["run_id"], component["component_id"]) in actions]
    approval = validate_gap_region_result(approval, review, snapshot, other_text=other_text, previous=previous)
    action = next(row for row in approval["runs"]
                  if row["run_id"] == snapshot.run_id and row["component_id"] == snapshot.component_id)
    read_gap_region_partition(action, snapshot.editable_mask)
    _current_snapshot(snapshot)
    path = folder / "approval.json"
    with path.open("x", encoding="utf8") as stream:
        json.dump(approval, stream, ensure_ascii=False, allow_nan=False, indent=2)
    _current_snapshot(snapshot)
    for row in approval["runs"]:
        _gap()._checked(row["partition"]["artifact"])
    if not _gap()._same(_gap()._checked(_gap()._bound(path), metadata=True), approval):
        raise ValueError("The saved literal region approval changed")
    return deepcopy(approval)
