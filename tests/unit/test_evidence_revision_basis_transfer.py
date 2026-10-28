"""A re-review keeps its original after the original is moved to a new camera basis."""
from copy import deepcopy

import pytest

from darsia.fluidflower.phase_evidence_audit import payload_hash, superseded_evidence_ids
from darsia.fluidflower.phase_training.annotation_basis_transfer import FORMAT, _digest

ORIGINAL = dict(id="label", category="confirmed_free_gas", reviewer_id="reviewer", shape="rectangle",
                frame_index=2, elapsed_h=2., x=.3, y=.3, width=.2, height=.1,
                photo_identity=dict(dataset="series", run="run1", source_sha256="e" * 64))


def transferred(original, **changes):
    """The original as the basis transfer saves it: new geometry, the reviewed source kept."""
    result = deepcopy(original)
    result.update(shape="polygon", points=[[.1, .2], [.3, .2], [.3, .4]], x=.1, y=.2, width=.2, height=.2,
                  frame_index=0, elapsed_h=2., coordinate_space="normalized_native_review_canvas_16x9",
                  photo_identity=dict(dataset="series", run="run1", source_sha256="c" * 64),
                  basis_transfer=dict(format=FORMAT, source_annotation=deepcopy(original),
                                      source_annotation_sha256=_digest(original)))
    result.update(changes)
    return result


def revision(original):
    return dict(id="revision", category="aqueous_present_region", reviewer="reviewer",
                updated_utc="2026-10-20T10:00:00+00:00",
                evidence_audit=dict(verdict="reclassified", original_region_id="run1:manual:label",
                                    original_annotation_sha256=payload_hash(original)))


def test_revision_of_a_transferred_original_still_supersedes_it():
    records = [("run1:manual:label", transferred(ORIGINAL)), ("run1:manual:revision", revision(ORIGINAL))]
    assert superseded_evidence_ids(records) == {"run1:manual:label"}


@pytest.mark.parametrize("change", [dict(category="gas_present_region"), dict(reviewer_id="other")])
def test_a_changed_human_field_is_not_hidden_by_the_transfer(change):
    records = [("run1:manual:label", transferred(ORIGINAL, **change)), ("run1:manual:revision", revision(ORIGINAL))]
    with pytest.raises(ValueError, match="Original evidence changed before re-review"):
        superseded_evidence_ids(records)


def test_a_transfer_whose_source_does_not_match_its_digest_is_not_trusted():
    moved = transferred(ORIGINAL)
    moved["basis_transfer"]["source_annotation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Original evidence changed before re-review"):
        superseded_evidence_ids([("run1:manual:label", moved), ("run1:manual:revision", revision(ORIGINAL))])


def test_an_original_moved_twice_keeps_its_first_fingerprint():
    twice = transferred(transferred(ORIGINAL))
    records = [("run1:manual:label", twice), ("run1:manual:revision", revision(ORIGINAL))]
    assert superseded_evidence_ids(records) == {"run1:manual:label"}
