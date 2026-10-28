"""Actual typed region results at the presentation boundary; synthetic sources."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import painted_gap_regions as regions
from darsia.fluidflower.painting_guided_review_bindings import (
    RegionPreviewSession, validate_region_result,
)
from darsia.fluidflower.painted_transfer_replay import (
    replay_transfer_review, build_residual_transfer_review,
)

from .test_painted_gap_regions_integration import (
    CONFIRM, _fill, both_gap_case, gap_case, imported, registered_setup, synthetic_payload_admission,
)
from .test_painted_transfer_replay import _context, _partial_target


def _snapshot(review, row, component, classification):
    packet = gaps.read_gap_preview(review, row["run_id"], component["component_id"])
    draft = regions.region_draft_factory(review, packet)
    _fill(draft, classification)
    return draft.snapshot()


def _request(review, snapshot, *, previous=None, other_text=""):
    return dict(review_binding=deepcopy(review["review_binding"]),
        run_id=snapshot.run_id, component_id=snapshot.component_id,
        review_row_sha256=snapshot.review_row_sha256, snapshot=snapshot,
        previous=deepcopy(previous), other_text=other_text, **CONFIRM)


@pytest.mark.parametrize("mismatch", ["classification", "actor", "other_description"])
def test_an_actual_saved_result_cannot_replace_the_submitted_pixels_or_human_action(gap_case, mismatch):
    """A genuine valid result for the same row/time is insufficient when its choice differs."""
    review = gap_case.review
    row = next(row for row in review["runs"] if row["components"])
    component = row["components"][0]
    classification = "other" if mismatch == "other_description" else "air"
    snapshot = _snapshot(review, row, component, classification)
    description = "A fixture visible in this synthetic image" if classification == "other" else ""
    request = _request(review, snapshot, other_text=description)
    selected = (_snapshot(review, row, component, "free_water")
                if mismatch == "classification" else snapshot)
    result = regions.save_gap_region_approval(selected, gap_case.root / "typed-results",
        other_text=description, **CONFIRM)
    original = deepcopy(result)
    if mismatch == "actor":
        request["actor"] = "another synthetic reviewer"
    elif mismatch == "other_description":
        request["other_text"] = "A different description"
    with pytest.raises(ValueError):
        validate_region_result(result, review, request)
    assert result == original


def test_exact_submitted_snapshot_and_previous_records_survive_the_real_writer(both_gap_case):
    """A second component retains the first literal action through the actual C save API."""
    case, approval = both_gap_case, None
    selected = [(row, component) for row in case.review["runs"] for component in row["components"]]
    assert len(selected) >= 2
    for index, (row, component) in enumerate(selected[:2]):
        snapshot = _snapshot(case.review, row, component, "air")
        request = _request(case.review, snapshot, previous=approval)
        result = regions.save_gap_region_approval(snapshot,
            case.root / f"typed-confirmation-{index}", previous=approval, **CONFIRM)
        assert validate_region_result(result, case.review, request) == result
        if approval is not None:
            assert result["runs"][0] == approval["runs"][0]
            earlier = approval["runs"][0]
            wrong_question = dict(request, **{key: earlier[key] for key in (
                "run_id", "component_id", "review_row_sha256")})
            with pytest.raises(ValueError, match="different classification question"):
                validate_region_result(result, case.review, wrong_question)
            missing_previous = dict(request, previous=None)
            with pytest.raises(ValueError, match="only this explicitly confirmed"):
                validate_region_result(result, case.review, missing_previous)
        approval = result


def test_old_local_markings_seed_a_genuine_residual_review_without_rebasing_approval(gap_case):
    """An original old binding remains only the origin of a new descriptive draft."""
    case, review = gap_case, gap_case.review
    row = next(row for row in review["runs"] if row["components"])
    component = max(row["components"], key=lambda item: item["pixels"])
    snapshot = _snapshot(review, row, component, "air")
    request = _request(review, snapshot)
    old_destination = case.root / "session-old-confirmation"
    approval = regions.save_gap_region_approval(snapshot, old_destination, **CONFIRM)
    assert validate_region_result(approval, review, request) == approval
    # This original finite reader deliberately returns the raw saved JSON,
    # without the in-memory review_binding attached by read_gap_review.
    assert "review_binding" not in gaps._checked_gap_review(snapshot.review_binding)
    retained = {path: Path(path).read_bytes() for path in (
        review["review_binding"]["path"], str(old_destination / "approval.json"),
        approval["runs"][0]["partition"]["artifact"]["path"])}
    session = RegionPreviewSession()
    session.remember(request, review=review)

    target_row, _packet, binding, _labels, _painted = _partial_target(case)
    assert target_row["run_id"] == row["run_id"]
    replay = replay_transfer_review(review, source_context=_context(case), target_bindings=[binding])
    fresh = build_residual_transfer_review(replay, case.root / "session-fresh-residual")
    current = next(item for item in fresh["runs"] if item["run_id"] == row["run_id"])
    assert fresh["review_binding"] != review["review_binding"]
    assert current["gap"]["pixels"] == row["gap"]["pixels"] - 1
    assert current["target_correction_bindings"] == [binding]
    for key in ("source", "source_map_pixels", "geometry_index", "map_version", "painting_photograph"):
        assert fresh[key] == review[key]
    for key in ("reference_model", "target_model", "native_photo", "native_photo_pixels",
                "native_to_painting_h", "contract"):
        assert current[key] == row[key]

    packet = session.reader(gaps.read_gap_preview)(fresh, row["run_id"], current["components"][0]["component_id"])
    draft = packet["region_draft"]
    regions.validate_gap_region_snapshot(draft.snapshot(), review=fresh)
    assert draft.review_binding == fresh["review_binding"] and not draft.events
    assert draft.preview_source_context == dict(review=review["review_binding"],
        run_id=snapshot.run_id, component_id=snapshot.component_id,
        review_row_sha256=snapshot.review_row_sha256)
    np.testing.assert_array_equal(draft.class_masks["air"], packet["gap_mask"])
    assert not any(mask.any() for name, mask in draft.class_masks.items() if name != "air")
    assert "Confirm or correct" in packet["region_transfer_note"]
    assert "approval" not in packet and "analysis_exclusions" not in fresh
    with pytest.raises(ValueError):
        regions.validate_gap_region_approval(approval, fresh)
    assert not (Path(fresh["review_binding"]["path"]).parent / "index.json").exists()
    for path, payload in retained.items():
        assert Path(path).read_bytes() == payload
