"""Human target edits retain original carrier/frame and replay only their gap."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from darsia.fluidflower.painted_target_corrections import (
    TargetCorrectionDraft, verify_and_apply_target_correction,
)


def _pin(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _case(tmp_path):
    review = tmp_path / "review" / "gap_review.json"
    review.parent.mkdir()
    review.write_text('{"fixture":"independently checked review"}', encoding="utf-8")
    base = np.full((9, 11), 2, np.uint8)
    base[:3] = 0
    editable = np.zeros(base.shape, bool)
    editable[:3] = True
    science = dict(source="frozen approved V9 fixture", run="target", frame_unchanged=True,
        allowed_layer_ids=[2, 3],
        numeric_grid=dict(shape=list(base.shape), transform=[[1,0,0],[0,1,0],[0,0,1]]),
        prior_target_corrections=[])
    basis = dict(run_id="target", review=_pin(review), review_row_sha256="a" * 64,
        component_id="component-one", science=science)
    draft = TargetCorrectionDraft(labels=base, editable_mask=editable, allowed_ids=[2, 3], basis=basis)
    return draft, base, editable, science


def _save(draft, tmp_path):
    return draft.snapshot().save_and_approve(tmp_path / "target_corrections", actor="Fixture reviewer",
        approved_utc="2026-10-22T08:00:00+00:00", reason="Paint visible sand in the selected fixture gap")


@pytest.mark.parametrize("kind,points,radius", [
    ("polygon", [[0,0],[10,0],[10,5],[0,5]], 1),
    ("brush", [[1,1],[9,1]], 2),
])
def test_human_gap_paint_replays_over_unchanged_carrier_and_never_changes_known_pixels(tmp_path, kind, points, radius):
    draft, base, editable, science = _case(tmp_path)
    before = base.copy()
    assert draft.paint(kind, points, 3, radius)
    binding = _save(draft, tmp_path)
    result = verify_and_apply_target_correction(binding, base_labels=base,
        editable_mask=editable, current_science=science)
    np.testing.assert_array_equal(base, before)
    np.testing.assert_array_equal(result["labels"][~editable], before[~editable])
    assert result["painted_mask"].any() and not np.any(result["painted_mask"] & ~editable)
    assert np.all(result["labels"][result["painted_mask"]] == 3)
    assert result["provenance"]["original_carrier_proof_applies_to_base_only"]
    assert result["provenance"]["physical_frame_unchanged"]
    assert result["provenance"]["scale_unchanged"]
    assert "acquisition_state" not in result["provenance"]


def test_undo_redo_and_new_stroke_branch_are_replayed_from_the_complete_events(tmp_path):
    draft, base, editable, science = _case(tmp_path)
    assert not draft.can_undo and not draft.can_redo
    draft.paint("brush", [[1,1]], 2, 1)
    first = draft.labels
    draft.paint("brush", [[8,1]], 3, 1)
    second = draft.labels
    assert draft.undo() and draft.can_redo
    np.testing.assert_array_equal(draft.labels, first)
    assert draft.redo()
    np.testing.assert_array_equal(draft.labels, second)
    assert draft.undo()
    draft.paint("brush", [[5,1]], 3, 1)
    assert not draft.can_redo and not draft.redo()
    binding = _save(draft, tmp_path)
    result = verify_and_apply_target_correction(binding, base_labels=base,
        editable_mask=editable, current_science=science)
    np.testing.assert_array_equal(result["labels"], draft.labels)
    assert result["labels"][1,8] == 0


def test_background_save_uses_detached_frozen_gui_snapshot(tmp_path):
    draft, base, editable, science = _case(tmp_path)
    draft.paint("brush", [[1,1]], 2)
    frozen = draft.snapshot()
    expected = frozen.labels
    draft.paint("brush", [[8,1]], 3)
    with pytest.raises(ValueError, match="snapshot"):
        frozen.paint("brush", [[4,1]], 3)
    binding = frozen.save_and_approve(tmp_path / "target_corrections", actor="Fixture reviewer",
        approved_utc="2026-10-22T08:00:00Z", reason="Explicit fixture approval")
    result = verify_and_apply_target_correction(binding, base_labels=base,
        editable_mask=editable, current_science=science)
    np.testing.assert_array_equal(result["labels"], expected)
    assert result["labels"][1,8] == 0


@pytest.mark.parametrize("change", ["base", "component", "science", "boolean-number", "review"])
def test_changed_original_basis_or_science_refuses_before_effective_labels_are_returned(tmp_path, change):
    draft, base, editable, science = _case(tmp_path)
    draft.paint("brush", [[1,1]], 3)
    binding = _save(draft, tmp_path)
    if change == "base": base[8,10] = 3
    elif change == "component": editable[2,10] = False
    elif change == "science": science["source"] = "changed"
    elif change == "boolean-number": science["frame_unchanged"] = 1
    else:
        path = draft.basis["review"]["path"]
        from pathlib import Path
        Path(path).write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError):
        verify_and_apply_target_correction(binding, base_labels=base,
            editable_mask=editable, current_science=science)


def test_tampered_saved_pixels_with_new_file_pins_still_fail_human_event_replay(tmp_path):
    draft, base, editable, science = _case(tmp_path)
    draft.paint("brush", [[1,1]], 2)
    binding = _save(draft, tmp_path)
    from pathlib import Path
    version_path = Path(binding["version"]["path"])
    version = json.loads(version_path.read_bytes())
    product = Path(version["product"]["path"])
    with np.load(product, allow_pickle=False) as arrays:
        labels, mask = arrays["labels"], arrays["painted_mask"]
    labels[1,1] = 3
    np.savez_compressed(product, labels=labels, painted_mask=mask)
    version["product"] = _pin(product)
    version_path.write_text(json.dumps(version), encoding="utf-8")
    binding["version"] = _pin(version_path)
    approval_path = Path(binding["approval"]["path"])
    approval = json.loads(approval_path.read_bytes())
    approval["version"] = binding["version"]
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    binding["approval"] = _pin(approval_path)
    with pytest.raises(ValueError, match="event replay"):
        verify_and_apply_target_correction(binding, base_labels=base,
            editable_mask=editable, current_science=science)


def test_saved_version_cannot_expand_the_current_approved_paintable_layers(tmp_path):
    draft, base, editable, science = _case(tmp_path)
    draft.paint("brush", [[1,1]], 2)
    binding = _save(draft, tmp_path)
    from pathlib import Path
    version_path = Path(binding["version"]["path"])
    version = json.loads(version_path.read_bytes())
    version["allowed_ids"].append(4)
    version_path.write_text(json.dumps(version), encoding="utf-8")
    binding["version"] = _pin(version_path)
    approval_path = Path(binding["approval"]["path"])
    approval = json.loads(approval_path.read_bytes())
    approval["version"] = binding["version"]
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    binding["approval"] = _pin(approval_path)
    with pytest.raises(ValueError, match="Paintable layers"):
        verify_and_apply_target_correction(binding, base_labels=base,
            editable_mask=editable, current_science=science)


@pytest.mark.parametrize("label,points,radius", [
    (0, [[1,1]], 1), (255, [[1,1]], 1), (True, [[1,1]], 1),
    (4, [[1,1]], 1), (2, [[-1,1]], 1), (2, [[11,1]], 1),
    (2, [[1,float("nan")]], 1), (2, [[1,1]], 0), (2, [[1,1]], float("inf")),
])
def test_no_inferred_layer_clamping_or_invalid_pixel_brush(tmp_path, label, points, radius):
    draft, _base, _editable, _science = _case(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        draft.paint("brush", points, label, radius)
    assert not draft.dirty and not draft.events


def test_no_empty_or_implicit_approval_and_no_write_into_review_or_raw_folder(tmp_path):
    draft, _base, _editable, _science = _case(tmp_path)
    with pytest.raises(ValueError, match="no target pixels"):
        _save(draft, tmp_path)
    draft.paint("brush", [[1,1]], 2)
    with pytest.raises(ValueError, match="actor"):
        draft.save_and_approve(tmp_path / "output", actor="", approved_utc="2026-10-22T08:00Z", reason="Fixture")
    with pytest.raises(ValueError, match="immutable"):
        draft.save_and_approve(tmp_path / "review" / "nested", actor="Fixture",
            approved_utc="2026-10-22T08:00Z", reason="Fixture")
    image = tmp_path / "raw" / "source.ARW"
    image.parent.mkdir()
    image.write_bytes(b"immutable fixture acquisition")
    changed = deepcopy(draft.basis)
    changed["science"]["raw"] = _pin(image)
    other = TargetCorrectionDraft(labels=draft.base_labels, editable_mask=draft.editable_mask,
        allowed_ids=draft.allowed_ids, basis=changed)
    other.paint("brush", [[1,1]], 2)
    with pytest.raises(ValueError, match="original image folder"):
        other.save_and_approve(image.parent / "nested", actor="Fixture",
            approved_utc="2026-10-22T08:00Z", reason="Fixture")
    assert not (tmp_path / "review" / "nested").exists()
    assert not (image.parent / "nested").exists()
