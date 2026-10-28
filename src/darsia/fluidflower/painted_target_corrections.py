"""Explicit human target-pixel edits beside an unchanged approved carrier.

The caller independently verifies the review/component and current scientific
closure. These helpers neither authenticate a person nor infer a decision.
Only a GUI save-and-approve action may publish an actual correction version.
"""
from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

import numpy as np

VERSION_FORMAT = "darsia-painted-target-correction-v1"
APPROVAL_FORMAT = "darsia-painted-target-correction-approval-v1"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape),
                sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())


def _pin(path):
    path = Path(path).resolve()
    before = _preset_io.file_stat(path)
    digest = hashlib.sha256()
    with _preset_io.open_read(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    after = _preset_io.file_stat(path)
    stamp = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if stamp(before) != stamp(after):
        raise ValueError("Target correction input changed during read")
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def _checked(binding, *, metadata=False):
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256", "size_bytes"}
            or not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute()
            or type(binding["size_bytes"]) is not int or binding["size_bytes"] < 0
            or re.fullmatch("[0-9a-f]{64}", str(binding["sha256"])) is None):
        raise ValueError("An exact original target-correction descriptor is required")
    path = Path(binding["path"])
    before = _preset_io.file_stat(path)
    raw = _preset_io.read_bytes(path)
    after = _preset_io.file_stat(path)
    stamp = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if (stamp(before) != stamp(after) or len(raw) != binding["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != binding["sha256"]):
        raise ValueError("A bound target-correction input changed")
    return json.loads(raw) if metadata else path


def _basis(value):
    if (not isinstance(value, dict) or set(value) != {
            "run_id", "review", "review_row_sha256", "component_id", "science"}
            or not isinstance(value["run_id"], str)
            or re.fullmatch(r"[\w-]+", value["run_id"]) is None
            or not isinstance(value["component_id"], str) or not value["component_id"]
            or re.fullmatch("[0-9a-f]{64}", str(value["review_row_sha256"])) is None
            or not isinstance(value["science"], dict) or not value["science"]):
        raise ValueError("An independently verified target/component scientific basis is required")
    result = json.loads(_json(value))
    _checked(result["review"])
    return result


def _native_paint_mask(kind, pixels, radius, shape_hw):
    """Rasterize already-validated native strokes with the original paint kernel."""
    import cv2
    from darsia._opencv_runtime import scientific_frame

    r = radius
    h, w = shape_hw
    result = np.zeros((h, w), np.uint8)
    with scientific_frame("explicit-target-pixel-paint"):
        if kind == "polygon":
            cv2.fillPoly(result, [pixels], 1)
        elif kind == "brush":
            centers = {tuple(point) for point in pixels}
            for start, end in zip(pixels[:-1], pixels[1:]):
                delta = end - start
                steps = max(1, int(np.ceil(2 * max(abs(float(delta[0])), abs(float(delta[1]))) / r)))
                for fraction in np.linspace(0., 1., steps + 1):
                    centers.add(tuple(np.rint(start + fraction * delta).astype(int)))
            for center in sorted(centers):
                cv2.ellipse(result, center, (r, r), 0, 0, 360, 1, -1, cv2.LINE_8)
        else:
            raise ValueError("Only polygon and brush target edits are supported")
    return result != 0


class TargetCorrectionDraft:
    """In-memory pixel-coordinate edits; no frame, scale or source-map mutation."""

    def __init__(self, *, labels, editable_mask, allowed_ids, basis):
        if (not isinstance(labels, np.ndarray) or labels.dtype != np.uint8
                or labels.ndim != 2 or min(labels.shape) < 1
                or not isinstance(editable_mask, np.ndarray) or editable_mask.dtype != bool
                or editable_mask.shape != labels.shape or not editable_mask.any()
                or np.any(editable_mask & (labels != 0))):
            raise ValueError("The exact uint8 target and its still-unpainted gap component are required")
        ids = list(allowed_ids)
        if (not ids or len(ids) != len(set(ids))
                or any(type(value) is not int or not 0 < value < 255 for value in ids)):
            raise ValueError("Only the existing positive approved layer IDs may be painted")
        self.basis = _basis(basis)
        if _json(self.basis["science"].get("allowed_layer_ids")) != _json(sorted(ids)):
            raise ValueError("Paintable layers must match the independently bound scientific basis")
        self.base_labels = labels.copy()
        self.editable_mask = editable_mask.copy()
        self.base_labels.setflags(write=False)
        self.editable_mask.setflags(write=False)
        self.allowed_ids = tuple(sorted(ids))
        self.events = []
        self._labels = self.base_labels.copy()
        self._sealed = False

    @property
    def labels(self):
        return self._labels.copy()

    @property
    def painted_mask(self):
        return self._labels != self.base_labels

    @property
    def dirty(self):
        return bool(np.any(self.painted_mask))

    def _state(self):
        active, redo, operations = [], [], {}
        for index, event in enumerate(self.events):
            if type(event.get("event_id")) is not int or event["event_id"] != index + 1:
                raise ValueError("Target paint events must retain their complete original order")
            kind = event.get("kind")
            if kind in {"polygon", "brush"}:
                if set(event) != {"event_id", "kind", "points_xy", "label_id", "radius_px"}:
                    raise ValueError("Unknown target paint event fields")
                self._event_points(event)
                operations[event["event_id"]] = event
                active.append(event["event_id"])
                redo.clear()
            elif kind == "undo":
                if (set(event) != {"event_id", "kind", "target_event_id"}
                        or type(event["target_event_id"]) is not int
                        or not active or event["target_event_id"] != active[-1]):
                    raise ValueError("Undo must name the latest active target stroke")
                redo.append(active.pop())
            elif kind == "redo":
                if (set(event) != {"event_id", "kind", "target_event_id"}
                        or type(event["target_event_id"]) is not int
                        or not redo or event["target_event_id"] != redo[-1]):
                    raise ValueError("Redo must name the latest undone target stroke")
                active.append(redo.pop())
            else:
                raise ValueError("Unknown target paint event")
        return active, redo, operations

    @property
    def can_undo(self):
        return bool(self._state()[0])

    @property
    def can_redo(self):
        return bool(self._state()[1])

    def _event_points(self, event):
        kind = event["kind"]
        label = event["label_id"]
        radius = event["radius_px"]
        points = np.asarray(event["points_xy"], dtype=np.float64)
        h, w = self.base_labels.shape
        if (type(label) is not int or label not in self.allowed_ids
                or points.ndim != 2 or points.shape[1:] != (2,)
                or not np.isfinite(points).all() or len(points) < (3 if kind == "polygon" else 1)
                or np.any(points < 0) or np.any(points[:, 0] > w - 1) or np.any(points[:, 1] > h - 1)
                or type(radius) not in {int, float} or not np.isfinite(radius)
                or not 0 < radius <= max(h, w)):
            raise ValueError("Use existing layers and finite native-pixel strokes inside the actual image")
        return np.rint(points).astype(np.int32), max(1, int(round(radius)))

    def _event_mask(self, event):
        pixels, r = self._event_points(event)
        return _native_paint_mask(event["kind"], pixels, r, self.base_labels.shape) & self.editable_mask

    def _rebuild(self):
        active, _redo, operations = self._state()
        labels = self.base_labels.copy()
        for key in active:
            event = operations[key]
            labels[self._event_mask(event)] = event["label_id"]
        self._labels = labels

    def paint(self, kind, points_xy, label_id, radius_px=1):
        if self._sealed:
            raise ValueError("A saved worker snapshot cannot be edited")
        event = dict(event_id=len(self.events) + 1, kind=kind,
            points_xy=json.loads(_json(points_xy)), label_id=label_id, radius_px=radius_px)
        mask = self._event_mask(event)
        if not mask.any():
            return False
        self.events.append(event)
        self._rebuild()
        return True

    def undo(self):
        if self._sealed:
            raise ValueError("A saved worker snapshot cannot be edited")
        active = self._state()[0]
        if not active:
            return False
        self.events.append(dict(event_id=len(self.events) + 1, kind="undo", target_event_id=active[-1]))
        self._rebuild()
        return True

    def redo(self):
        if self._sealed:
            raise ValueError("A saved worker snapshot cannot be edited")
        redo = self._state()[1]
        if not redo:
            return False
        self.events.append(dict(event_id=len(self.events) + 1, kind="redo", target_event_id=redo[-1]))
        self._rebuild()
        return True

    def snapshot(self):
        """Capture on the GUI thread before handing the detached save to a worker."""
        frozen = TargetCorrectionDraft(labels=self.base_labels, editable_mask=self.editable_mask,
            allowed_ids=self.allowed_ids, basis=self.basis)
        frozen.events = deepcopy(self.events)
        frozen._rebuild()
        frozen._sealed = True
        return frozen

    def save_and_approve(self, destination, *, actor, approved_utc, reason):
        """Publish only on the explicit human GUI save-and-approve action."""
        if any(not isinstance(value, str) or not value.strip() for value in (actor, approved_utc, reason)):
            raise ValueError("The human action needs its actor, time and short reason")
        if datetime.fromisoformat(approved_utc.replace("Z", "+00:00")).utcoffset() is None:
            raise ValueError("The explicit approval time needs a timezone")
        self._rebuild()
        if not self.dirty:
            raise ValueError("There are no target pixels to save")
        basis = _basis(self.basis)
        root = Path(destination).expanduser()
        if not root.is_absolute():
            raise ValueError("An explicit absolute correction-version destination is required")
        from .rig_raw_write import _no_links
        _no_links(root)
        if root.resolve().is_relative_to(Path(basis["review"]["path"]).resolve().parent):
            raise ValueError("Keep corrections outside the immutable reviewed products")
        def protect_raw(value):
            if isinstance(value, dict):
                raw = value.get("path")
                if isinstance(raw, str) and Path(raw).suffix.lower() in {
                        ".arw", ".raw", ".dng", ".nef", ".cr2", ".cr3", ".jpg", ".jpeg"}:
                    if root.resolve().is_relative_to(Path(raw).resolve().parent):
                        raise ValueError("Corrections cannot be saved in an original image folder")
                for item in value.values(): protect_raw(item)
            elif isinstance(value, list):
                for item in value: protect_raw(item)
        protect_raw(basis)
        root.mkdir(parents=True, exist_ok=True)
        folder = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12])
        folder.mkdir()
        arrays = folder / "target_labels.npz"
        with arrays.open("xb") as stream:
            np.savez_compressed(stream, labels=self._labels, painted_mask=self.painted_mask)
        version = dict(format=VERSION_FORMAT, basis=basis, base_identity=_identity(self.base_labels),
            editable_mask_identity=_identity(self.editable_mask), allowed_ids=list(self.allowed_ids),
            events=deepcopy(self.events), edited_pixels=int(np.count_nonzero(self.painted_mask)),
            product=_pin(arrays), source_carrier_unchanged=True,
            physical_frame_unchanged=True, scale_unchanged=True)
        path = folder / "version.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(_json(version) + "\n")
        version_binding = _pin(path)
        approval = dict(format=APPROVAL_FORMAT, version=version_binding, run_id=basis["run_id"],
            actor=actor, approved_utc=approved_utc, reason=reason,
            action="explicit-human-target-paint-save-and-approve")
        path = folder / "approval.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(_json(approval) + "\n")
        _basis(basis)
        return dict(version=version_binding, approval=_pin(path))


def verify_and_apply_target_correction(binding, *, base_labels, editable_mask, current_science):
    """Replay the independently bound human edits over the current exact base.

    The canonical caller also reads the bound original review, obtains this
    component mask from it, and independently constructs current_science. A
    new NPZ namespace alone is not a scientific change; its array identity is.
    """
    if not isinstance(binding, dict) or set(binding) != {"version", "approval"}:
        raise ValueError("A saved target version and its separate explicit approval are required")
    version = _checked(binding["version"], metadata=True)
    approval = _checked(binding["approval"], metadata=True)
    if (set(version) != {"format", "basis", "base_identity", "editable_mask_identity", "allowed_ids",
                         "events", "edited_pixels", "product", "source_carrier_unchanged",
                         "physical_frame_unchanged", "scale_unchanged"}
            or version["format"] != VERSION_FORMAT
            or any(version[key] is not True for key in ("source_carrier_unchanged", "physical_frame_unchanged", "scale_unchanged"))
            or not isinstance(version["events"], list)
            or type(version["edited_pixels"]) is not int):
        raise ValueError("The target version does not retain its explicit unchanged carrier/frame/scale")
    basis = _basis(version["basis"])
    if _json(basis["science"]) != _json(current_science):
        raise ValueError("The target correction belongs to changed scientific inputs")
    if (set(approval) != {"format", "version", "run_id", "actor", "approved_utc", "reason", "action"}
            or approval["format"] != APPROVAL_FORMAT
            or _json(approval["version"]) != _json(binding["version"])
            or approval["run_id"] != basis["run_id"]
            or approval["action"] != "explicit-human-target-paint-save-and-approve"
            or any(not isinstance(approval[key], str) or not approval[key].strip()
                   for key in ("actor", "approved_utc", "reason"))
            or datetime.fromisoformat(approval["approved_utc"].replace("Z", "+00:00")).utcoffset() is None):
        raise ValueError("The target correction lacks its exact explicit human approval")
    draft = TargetCorrectionDraft(labels=base_labels, editable_mask=editable_mask,
        allowed_ids=version["allowed_ids"], basis=basis)
    if (_json(_identity(draft.base_labels)) != _json(version["base_identity"])
            or _json(_identity(draft.editable_mask)) != _json(version["editable_mask_identity"])):
        raise ValueError("The target base or reviewed editable component changed")
    draft.events = deepcopy(version["events"])
    draft._rebuild()
    product = _checked(version["product"])
    with np.load(_preset_io.source(product), allow_pickle=False) as arrays:
        if set(arrays.files) != {"labels", "painted_mask"}:
            raise ValueError("Unknown target correction product fields")
        labels, mask = arrays["labels"], arrays["painted_mask"]
    if (labels.dtype != np.uint8 or mask.dtype != bool
            or not np.array_equal(labels, draft.labels) or not np.array_equal(mask, draft.painted_mask)
            or not draft.dirty or version["edited_pixels"] != int(np.count_nonzero(mask))):
        raise ValueError("Saved target pixels differ from the exact human event replay")
    for descriptor in (binding["version"], binding["approval"], version["product"], basis["review"]):
        _checked(descriptor)
    return dict(labels=labels.copy(), painted_mask=mask.copy(), provenance=dict(
        version=deepcopy(binding["version"]), approval=deepcopy(binding["approval"]),
        basis=basis, edited_pixels=version["edited_pixels"],
        base_array_unchanged_outside_explicit_edits=True,
        original_carrier_proof_applies_to_base_only=True,
        physical_frame_unchanged=True, scale_unchanged=True))
