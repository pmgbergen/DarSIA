"""Synthetic region drafts and literal partitions; no scientific approval claim."""

from copy import deepcopy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from darsia.fluidflower import painted_analysis_exclusions as exclusions
from darsia.fluidflower import painted_gap_regions as regions
from darsia.fluidflower.painted_target_corrections import TargetCorrectionDraft


def _case(tmp_path, *, include_foreign=False):
    """Bind a synthetic verified-packet boundary using original review identities."""
    mask = np.zeros((9, 11), dtype=bool)
    mask[1:8, 1:10] = True
    mask[3:5, 4:6] = False
    rows = []
    for run_id in (("synthetic", "foreign") if include_foreign else ("synthetic",)):
        component, actual_mask = next(iter(exclusions._components(mask)))
        np.testing.assert_array_equal(actual_mask, mask)
        row = {"run_id": run_id,
               "gap": {"pixels": int(mask.sum()), "shape_hw": list(mask.shape)}, "findings": []}
        component["review_row_sha256"] = exclusions._digest({
            "context": exclusions._digest(row), "component": component,
        })
        row["components"] = [component]
        row["review_row_sha256"] = exclusions._row_digest(row)
        rows.append(row)
    review = {
        "format": exclusions.REVIEW_FORMAT, "meaning": exclusions.MEANING,
        "source": {"synthetic_verified_packet_boundary": True},
        "runs": rows, "issues": [],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review, allow_nan=False), encoding="utf-8")
    review["review_binding"] = exclusions._bound(path)
    packet = {"run_id": "synthetic", "component_id": component["component_id"],
              "gap_mask": mask.copy()}
    return regions.region_draft_factory(review, packet), review, packet


def _approval(tmp_path, *, values=None, include_foreign=False):
    """Retain a literal synthetic partition for strict metadata and array tests."""
    _, review, packet = _case(tmp_path, include_foreign=include_foreign)
    mask = packet["gap_mask"]
    if values is None:
        values = mask.astype(np.int32)
    artifact = tmp_path / "region_partition.npz"
    with artifact.open("xb") as stream:
        np.savez_compressed(stream, region_ids=values)
    action = {
        "run_id": "synthetic", "component_id": packet["component_id"],
        "review_row_sha256": review["runs"][0]["components"][0]["review_row_sha256"],
        "actor": "synthetic reviewer", "approved_utc": "2026-01-01T00:00:00Z",
        "reason": "Explicit synthetic non-sand decision", "decision_ref": "synthetic choice",
        "partition": {"artifact": exclusions._bound(artifact), "field": "region_ids",
                      "identity": exclusions._identity(values)},
        "regions": [{"region_id": 1, "classification": "air",
                     "decision_type": "outside_analysis_non_sand", "other_text": ""}],
    }
    return {"format": exclusions.APPROVAL_FORMAT_V2, "meaning": exclusions.MEANING,
            "review": review["review_binding"], "runs": [action]}, review, mask


@pytest.mark.parametrize("kind,points,radius", [
    ("polygon", [[0, 0], [10, 0], [10, 8], [0, 8]], 1),
    ("brush", [[1, 2], [9, 6]], 2),
])
def test_region_strokes_match_original_target_native_rasterization(tmp_path, kind, points, radius):
    """Reuse the real original rasterizer as an independent categorical-mask oracle."""
    draft, review, packet = _case(tmp_path)
    original_mask = packet["gap_mask"].copy()
    labels = np.full(original_mask.shape, 2, dtype=np.uint8)
    labels[original_mask] = 0
    oracle = TargetCorrectionDraft(labels=labels, editable_mask=original_mask,
        allowed_ids=[3], basis={
            "run_id": "synthetic", "review": review["review_binding"],
            "review_row_sha256": review["runs"][0]["components"][0]["review_row_sha256"],
            "component_id": packet["component_id"],
            "science": {"allowed_layer_ids": [3], "synthetic_only": True},
        })
    assert draft.paint(kind, points, "air", radius_px=radius)
    assert oracle.paint(kind, points, 3, radius_px=radius)
    np.testing.assert_array_equal(draft.class_masks["air"], oracle.painted_mask)
    assert not any(mask.any() for name, mask in draft.class_masks.items() if name != "air")
    assert set(draft.class_masks) == {"air", "free_water", "sand", "rig_or_frame", "other"}
    assert not np.any(draft.class_masks["air"] & ~original_mask)
    np.testing.assert_array_equal(packet["gap_mask"], original_mask)
    assert not draft.editable_mask.flags.writeable
    with pytest.raises(ValueError):
        draft.editable_mask.setflags(write=True)
    with pytest.raises(TypeError):
        draft.class_masks["air"] = original_mask
    for mask in draft.class_masks.values():
        assert mask.dtype == bool and not mask.flags.writeable
        with pytest.raises(ValueError):
            mask.setflags(write=True)


def test_region_history_replaces_classes_and_snapshot_is_detached_and_sealed(tmp_path, monkeypatch):
    """Undo/redo preserve replacement order; a worker snapshot cannot be edited."""
    draft, review, packet = _case(tmp_path)
    assert not draft.can_undo and not draft.can_redo
    assert not draft.undo() and not draft.redo()
    assert draft.paint("brush", [[2, 2]], "air", radius_px=1)
    first = draft.class_masks["air"].copy()
    assert draft.paint("brush", [[2, 2]], "free_water", radius_px=1)
    assert not draft.class_masks["air"].any()
    np.testing.assert_array_equal(draft.class_masks["free_water"], first)

    def forbidden(*args, **kwargs):
        """Availability checks must not allocate or rasterize another native frame."""
        pytest.fail("Undo availability rasterized a native frame")

    with monkeypatch.context() as availability:
        availability.setattr("darsia.fluidflower.painted_target_corrections._native_paint_mask", forbidden)
        assert draft.can_undo and not draft.can_redo
    assert draft.undo() and draft.can_redo
    np.testing.assert_array_equal(draft.class_masks["air"], first)
    assert not draft.class_masks["free_water"].any()
    assert draft.redo()
    frozen = draft.snapshot()
    frozen_masks = {key: value.tobytes() for key, value in frozen.class_masks.items()}
    frozen_events = frozen.events
    frozen_context = frozen.selected_context
    frozen_binding = frozen.review_binding
    expected_context = deepcopy(frozen_context)
    assert regions.validate_gap_region_snapshot(frozen, review) is frozen
    frozen_events.clear()
    frozen_context["review"]["sha256"] = "0" * 64
    frozen_binding["sha256"] = "0" * 64
    packet["gap_mask"][:] = False
    review["source"].clear()
    assert frozen.events and frozen.selected_context == expected_context
    assert frozen.review_binding == expected_context["review"]
    assert draft.undo()
    assert draft.paint("brush", [[8, 6]], "rig_or_frame", radius_px=1)
    assert not draft.can_redo and not draft.redo()
    assert {key: value.tobytes() for key, value in frozen.class_masks.items()} == frozen_masks
    assert regions.validate_gap_region_snapshot(frozen) is frozen
    for operation in (
        lambda: frozen.paint("brush", [[2, 2]], "other", radius_px=1),
        frozen.undo, frozen.redo,
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("kind,points,class_name,radius", [
    ("line", [[2, 2]], "air", 1),
    ("brush", [[2, 2]], "Air", 1),
    ("brush", [[2, 2]], "water_above_sand", 1),
    ("brush", [[2, 2]], True, 1),
    ("brush", [[-1, 2]], "air", 1),
    ("brush", [[11, 2]], "air", 1),
    ("brush", [[2, float("nan")]], "air", 1),
    ("brush", [[2, 2]], "air", 0),
    ("brush", [[2, 2]], "air", float("inf")),
    ("polygon", [[2, 2], [3, 2]], "air", 1),
])
def test_invalid_region_strokes_leave_the_entire_draft_unchanged(
        tmp_path, kind, points, class_name, radius):
    """No class alias, clipping, nonfinite coordinate or implicit stroke is admitted."""
    draft, _, _ = _case(tmp_path)
    before = {key: value.tobytes() for key, value in draft.class_masks.items()}
    with pytest.raises((ValueError, TypeError)):
        draft.paint(kind, points, class_name, radius_px=radius)
    assert draft.events == [] and not draft.can_undo and not draft.can_redo
    assert {key: value.tobytes() for key, value in draft.class_masks.items()} == before


def test_foreign_review_and_component_cannot_rebind_a_snapshot(tmp_path):
    """Original row identities remain part of the per-run editing context."""
    draft, review, packet = _case(tmp_path)
    assert draft.paint("brush", [[2, 2]], "air", radius_px=1)
    frozen = draft.snapshot()
    for field, value in (("run_id", "foreign"), ("component_id", "0" * 64)):
        changed = dict(packet, **{field: value})
        with pytest.raises(ValueError):
            regions.region_draft_factory(review, changed)
    changed = deepcopy(review)
    changed["review_binding"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        regions.validate_gap_region_snapshot(frozen, changed)
    changed = deepcopy(review)
    changed["runs"][0]["components"][0]["review_row_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        regions.validate_gap_region_snapshot(frozen, changed)


def test_literal_region_approval_validation_is_strict_without_payload_reads(tmp_path, monkeypatch):
    """Supplied review DATA validates the exact schema without opening partitions."""
    approval, review, _ = _approval(tmp_path)
    before = deepcopy(approval)

    def forbidden(*args, **kwargs):
        """Poison any file access during explicit-review metadata validation."""
        pytest.fail("Pure region metadata validation accessed a file")

    with monkeypatch.context() as pure:
        pure.setattr(Path, "open", forbidden)
        pure.setattr(Path, "stat", forbidden)
        pure.setattr(np, "load", forbidden)
        assert regions.validate_gap_region_approval(approval, review) == before
    assert approval == before
    mutations = []
    bad = deepcopy(approval)
    bad["unexpected"] = True
    mutations.append(bad)
    bad = deepcopy(approval)
    bad["runs"][0]["unexpected"] = True
    mutations.append(bad)
    bad = deepcopy(approval)
    bad["runs"][0]["actor"] = True
    mutations.append(bad)
    bad = deepcopy(approval)
    bad["runs"][0]["regions"][0]["region_id"] = True
    mutations.append(bad)
    bad = deepcopy(approval)
    bad["runs"][0]["regions"].append(deepcopy(bad["runs"][0]["regions"][0]))
    mutations.append(bad)
    for classification in ("sand", "water_above_sand", "Air"):
        bad = deepcopy(approval)
        bad["runs"][0]["regions"][0]["classification"] = classification
        mutations.append(bad)
    bad = deepcopy(approval)
    bad["runs"][0]["regions"][0]["decision_type"] = "air"
    mutations.append(bad)
    bad = deepcopy(approval)
    bad["runs"][0]["regions"][0]["classification"] = "other"
    mutations.append(bad)
    bad = deepcopy(approval)
    bad["runs"][0]["partition"]["field"] = "array"
    mutations.append(bad)
    bad = deepcopy(approval)
    bad["runs"][0]["review_row_sha256"] = "0" * 64
    mutations.append(bad)
    for bad in mutations:
        with pytest.raises(ValueError):
            regions.validate_gap_region_approval(bad, review)
    assert approval == before


def test_partition_reader_returns_only_the_exact_boolean_component(tmp_path):
    """Free water is an explicit decision label and introduces no physical field."""
    approval, review, mask = _approval(tmp_path)
    action = approval["runs"][0]
    action["regions"][0]["classification"] = "free_water"
    assert regions.validate_gap_region_approval(approval, review) == approval
    actual = regions.read_gap_region_partition(action, mask)
    assert actual.dtype == bool and actual.shape == mask.shape
    np.testing.assert_array_equal(actual, mask)
    assert set(action["regions"][0]) == {"region_id", "classification", "decision_type", "other_text"}
    assert action["regions"][0]["decision_type"] == "outside_analysis_non_sand"


@pytest.mark.parametrize("change", ["unmarked", "foreign_id", "outside", "dtype", "duplicate_field"])
def test_partition_reader_refuses_uncovered_or_foreign_native_pixels(tmp_path, change):
    """The complete component and int32 table are required, without mask borrowing."""
    draft, _, packet = _case(tmp_path)
    mask = packet["gap_mask"]
    values = mask.astype(np.int32)
    if change == "unmarked":
        values[1, 1] = 0
    elif change == "foreign_id":
        values[1, 1] = 2
    elif change == "outside":
        values[0, 0] = 1
    elif change == "dtype":
        values = values.astype(np.int64)
    # The artifact has genuine matching pins; array semantics must still refuse.
    artifact = tmp_path / "invalid_partition.npz"
    with artifact.open("xb") as stream:
        np.savez_compressed(stream, region_ids=values)
    if change == "duplicate_field":
        with zipfile.ZipFile(artifact) as archive:
            payload = archive.read("region_ids.npy")
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(artifact, "a") as archive:
                archive.writestr("region_ids.npy", payload)
    action = {
        "run_id": draft.run_id, "component_id": draft.component_id,
        "review_row_sha256": draft.review_row_sha256,
        "actor": "synthetic reviewer", "approved_utc": "2026-01-01T00:00:00Z",
        "reason": "Explicit synthetic decision", "decision_ref": "synthetic choice",
        "partition": {"artifact": exclusions._bound(artifact), "field": "region_ids",
                      "identity": exclusions._identity(values)},
        "regions": [{"region_id": 1, "classification": "air",
                     "decision_type": "outside_analysis_non_sand", "other_text": ""}],
    }
    before = mask.tobytes()
    with pytest.raises(ValueError):
        regions.read_gap_region_partition(action, mask)
    assert mask.tobytes() == before


def test_partition_hash_tampering_refuses_before_numpy_decode(tmp_path, monkeypatch):
    """The original descriptor is authoritative before any array is interpreted."""
    approval, _, mask = _approval(tmp_path)
    action = approval["runs"][0]
    artifact = Path(action["partition"]["artifact"]["path"])
    raw = artifact.read_bytes()
    artifact.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    decoded = []

    def forbidden(*args, **kwargs):
        """Record an attempted decoder call outside the expected refusal oracle."""
        decoded.append(True)
        pytest.fail("Changed partition reached NumPy decoding")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError):
        regions.read_gap_region_partition(action, mask)
    assert decoded == []


@pytest.mark.parametrize("change", ["codes", "event"])
def test_snapshot_validation_replays_events_instead_of_trusting_captured_masks(tmp_path, change):
    """A forged sealed object cannot alter either replayed classes or stroke history."""
    draft, _, _ = _case(tmp_path)
    assert draft.paint("brush", [[2, 2]], "air", radius_px=1)
    frozen = draft.snapshot()
    if change == "codes":
        codes = frozen._codes.copy()
        codes[frozen.class_masks["air"]] = 2
        frozen._codes = codes
    else:
        frozen._events[0]["class_name"] = "other"
    with pytest.raises(ValueError, match="masks differ"):
        regions.validate_gap_region_snapshot(frozen)


def test_selected_partition_dependencies_are_detached_metadata_only(tmp_path, monkeypatch):
    """Planning declares the selected partition without reading any raster or review."""
    approval, _, _ = _approval(tmp_path)
    foreign = deepcopy(approval["runs"][0])
    foreign["run_id"] = "foreign"
    foreign["partition"]["artifact"]["path"] = str(tmp_path / "foreign.npz")
    foreign["partition"]["artifact"]["sha256"] = "f" * 64
    approval["runs"].append(foreign)
    before = deepcopy(approval)

    def forbidden(*args, **kwargs):
        """Reject payload or review access during literal dependency projection."""
        pytest.fail("Dependency projection accessed a file")

    with monkeypatch.context() as pure:
        pure.setattr(Path, "open", forbidden)
        pure.setattr(Path, "stat", forbidden)
        pure.setattr(np, "load", forbidden)
        selected = regions.gap_region_dependencies(approval, run_id="synthetic")
        all_runs = regions.gap_region_dependencies(approval)
    assert selected == [approval["review"], approval["runs"][0]["partition"]["artifact"]]
    assert all_runs == selected + [foreign["partition"]["artifact"]]
    selected[0]["sha256"] = "0" * 64
    assert approval == before


def test_preview_seed_is_detached_partial_and_replayed_without_copying_approval(tmp_path):
    """Synthetic preview DATA tests no coordinate transport or source approval."""
    draft, review, _ = _case(tmp_path)
    masks = {name: np.zeros(draft.editable_mask.shape, bool) for name in draft.class_masks}
    masks["air"][1, 1] = True
    masks["sand"][7, 9] = True
    origin = {"review": deepcopy(review["review_binding"]), "run_id": "origin",
              "component_id": draft.component_id, "review_row_sha256": "a" * 64}
    expected_origin = deepcopy(origin)
    draft.set_preview_class_masks(masks, source_context=origin)
    seeded = {name: value.tobytes() for name, value in draft.class_masks.items()}
    assert draft.preview_source_context == expected_origin
    assert draft.run_id == "synthetic" and draft.events == []
    assert not draft.can_undo and not draft.can_redo
    assert sum(int(mask.sum()) for mask in draft.class_masks.values()) < int(draft.editable_mask.sum())
    masks["air"][:] = False
    origin["run_id"] = "changed"
    detached = draft.preview_source_context
    detached["run_id"] = "changed-again"
    assert draft.preview_source_context == expected_origin
    assert {name: value.tobytes() for name, value in draft.class_masks.items()} == seeded
    with pytest.raises(ValueError):
        draft.set_preview_class_masks(masks, source_context=expected_origin)
    assert draft.paint("polygon", [[9, 7], [9, 7], [9, 7]], "free_water")
    assert draft.class_masks["free_water"][7, 9] and not draft.class_masks["sand"].any()
    assert draft.undo()
    assert {name: value.tobytes() for name, value in draft.class_masks.items()} == seeded
    frozen = draft.snapshot()
    assert regions.validate_gap_region_snapshot(frozen, review) is frozen
    assert frozen.preview_source_context == expected_origin
    assert draft.redo() and frozen.class_masks["sand"][7, 9]
    for change in ("overlap", "dtype", "outside", "unknown"):
        fresh = regions.region_draft_factory(review, {
            "run_id": "synthetic", "component_id": draft.component_id,
            "gap_mask": draft.editable_mask,
        })
        bad = {name: np.zeros(draft.editable_mask.shape, bool) for name in draft.class_masks}
        bad["air"][1, 1] = True
        if change == "overlap":
            bad["sand"][1, 1] = True
        elif change == "dtype":
            bad["air"] = bad["air"].astype(np.uint8)
        elif change == "outside":
            bad["air"][0, 0] = True
        else:
            bad["unknown"] = bad.pop("other")
        with pytest.raises(ValueError):
            fresh.set_preview_class_masks(bad, source_context=expected_origin)
        assert fresh.preview_source_context is None
        assert not any(mask.any() for mask in fresh.class_masks.values())


def test_worker_result_matches_snapshot_and_preserves_unselected_previous_actions(tmp_path, monkeypatch):
    """Pure result checking binds classes and context, without claiming actor authentication."""
    result, review, mask = _approval(tmp_path, include_foreign=True)
    draft = regions.region_draft_factory(review, {
        "run_id": "synthetic", "component_id": result["runs"][0]["component_id"], "gap_mask": mask,
    })
    assert draft.paint("polygon", [[0, 0], [10, 0], [10, 8], [0, 8]], "air")
    frozen = draft.snapshot()
    foreign = deepcopy(result["runs"][0])
    foreign["run_id"] = "foreign"
    foreign["review_row_sha256"] = review["runs"][1]["components"][0]["review_row_sha256"]
    previous = dict(deepcopy(result), runs=[foreign])
    result["runs"].append(deepcopy(foreign))
    before = deepcopy(result)

    def forbidden(*args, **kwargs):
        """The submitted-result check is a pure metadata/array comparison."""
        pytest.fail("Result validation accessed a file")

    with monkeypatch.context() as pure:
        pure.setattr(Path, "open", forbidden)
        pure.setattr(Path, "stat", forbidden)
        pure.setattr(np, "load", forbidden)
        assert regions.validate_gap_region_result(result, review, frozen, previous=previous) == before
    mutations = []
    for field, value in (("partition", None), ("regions", None), ("review_row_sha256", "0" * 64)):
        bad = deepcopy(result)
        if field == "partition":
            bad["runs"][0][field]["identity"]["sha256"] = "0" * 64
        elif field == "regions":
            bad["runs"][0][field][0]["classification"] = "free_water"
        else:
            bad["runs"][0][field] = value
        mutations.append(bad)
    bad = deepcopy(result)
    bad["runs"][1]["reason"] = "Changed unselected human action"
    mutations.append(bad)
    bad = deepcopy(result)
    bad["runs"].pop()
    mutations.append(bad)
    for bad in mutations:
        with pytest.raises(ValueError):
            regions.validate_gap_region_result(bad, review, frozen, previous=previous)
    assert result == before
