"""Literal class compatibility on private, byte-bound review metadata.

These fixtures assert metadata identity and explicit human-choice handling.
They do not claim a real photograph, scientific mask, saved campaign approval,
or physical qualification. Full local Store/export joins use the existing
gap fixture in test_painted_analysis_exclusions.
"""

from copy import deepcopy
import hashlib
import json

import pytest

from darsia.fluidflower import painted_analysis_exclusions as subject

from .test_gap_review_read_budget import _manifest, _retain


CLASSES = (
    ("water_above_sand", "Vann over sanden", "Water above the sand"),
    ("air", "Luft", "Air"),
    ("rig_or_frame", "Rigg eller ramme", "Rig or frame"),
    ("other", "Annet", "Other"),
)
ACTION = dict(actor="fixture human", approved_utc="2026-01-01T00:00:00Z",
              reason="Explicit private area choice", decision_ref="private-choice")


@pytest.fixture
def review(tmp_path):
    """Retain a tiny original typed review with two independently bound run rows."""
    value = _manifest()
    second = deepcopy(value["runs"][0])
    second["run_id"] = "second"
    component = second["components"][0]
    component.pop("review_row_sha256")
    context = {key: item for key, item in second.items() if key not in {"components", "review_row_sha256"}}
    component["review_row_sha256"] = subject._digest(dict(context=subject._digest(context), component=component))
    second["review_row_sha256"] = subject._row_digest(second)
    value["runs"].append(second)
    return _retain(tmp_path / "review.json", value)


def _keys(review):
    """Choose each explicit component identity without discovering any pixels."""
    return [dict(run_id=row["run_id"], component_id=component["component_id"])
            for row in review["runs"] for component in row["components"]]


def _legacy(review, classification):
    """Construct old-schema literal DATA without using the new action writer."""
    return dict(format=subject.APPROVAL_FORMAT, meaning=subject.MEANING,
        review=deepcopy(review["review_binding"]), runs=[dict(
            run_id=row["run_id"], component_id=component["component_id"],
            review_row_sha256=component["review_row_sha256"], **ACTION,
            classification=classification, other_text="Retained unquantified remainder",
        ) for row in review["runs"] for component in row["components"]])


def _retain_approval(path, value):
    """Write exact legacy UTF-8 bytes and bind them with the original reader."""
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=1) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw, subject._bound(path)


def test_stable_english_keys_and_labels_have_one_explicit_order():
    """New action identifiers are stable values, separate from display labels."""
    assert subject.CLASSIFICATIONS == tuple(key for key, _, _ in CLASSES)
    assert subject.CLASSIFICATION_LABELS == {key: label for key, _, label in CLASSES}


@pytest.mark.parametrize("key,legacy,label", CLASSES)
def test_exact_key_resolution_supports_both_literal_namespaces(key, legacy, label):
    """Resolving a single explicit argument never changes an enclosing record."""
    assert subject.classification_key(key) == key
    assert subject.classification_key(legacy) == key
    assert subject.classification_label(key) == label
    assert subject.classification_label(legacy) == label


@pytest.mark.parametrize("key,legacy,label", CLASSES)
@pytest.mark.parametrize("argument_namespace", ["english", "legacy"])
def test_new_actions_emit_only_the_resolved_english_key(review, key, legacy, label, argument_namespace):
    """An explicit new action can accept an old spelling but stores the English key."""
    selected = key if argument_namespace == "english" else legacy
    result = subject.make_gap_approval(review, _keys(review), **ACTION,
        classification=selected, other_text="Explicit unquantified remainder")
    assert [row["classification"] for row in result["runs"]] == [key, key]
    assert subject._approval(result, review) == result


@pytest.mark.parametrize("key,legacy,label", CLASSES)
def test_retained_legacy_approval_bytes_proof_and_human_choice_remain_exact(review, tmp_path, key, legacy, label):
    """The original bound read validates legacy records without translating values."""
    old = _legacy(review, legacy)
    path = tmp_path / "retained_approval.json"
    raw, binding = _retain_approval(path, old)
    expected = deepcopy(old)
    original_digest = subject._digest(old)
    parsed = subject._checked(binding, metadata=True)
    validated = subject._approval(parsed, review)
    assert validated == parsed == expected
    assert validated is not parsed
    assert subject._digest(validated) == original_digest
    assert [row["classification"] for row in validated["runs"]] == [legacy, legacy]
    assert all({name: row[name] for name in ACTION} == ACTION for row in validated["runs"])
    assert path.read_bytes() == raw
    assert binding["sha256"] == hashlib.sha256(raw).hexdigest()
    assert binding["size_bytes"] == len(raw)
    assert subject._bound(path) == binding
    assert old == expected


def test_merging_one_new_action_preserves_unselected_legacy_record_and_raw_pin(review, tmp_path):
    """Only the explicitly selected new action may resolve its class argument."""
    previous = _legacy(review, "Annet")
    expected = deepcopy(previous)
    path = tmp_path / "previous_approval.json"
    raw, binding = _retain_approval(path, previous)
    selected = _keys(review)[1:]
    result = subject.make_gap_approval(review, selected, **dict(ACTION,
        approved_utc="2026-01-02T00:00:00Z", reason="A newly selected explicit choice"),
        classification="Luft", previous=previous)
    assert result["runs"][0] == expected["runs"][0]
    assert result["runs"][0]["classification"] == "Annet"
    assert result["runs"][1]["classification"] == "air"
    assert result["runs"][1]["reason"] == "A newly selected explicit choice"
    assert result["runs"][1]["approved_utc"] == "2026-01-02T00:00:00Z"
    assert previous == expected
    assert path.read_bytes() == raw
    assert subject._bound(path) == binding
    assert subject._approval(result, review) == result


def test_equivalent_class_translation_cannot_bypass_original_sha_before_replay(review, tmp_path, monkeypatch):
    """An accepted equivalent English value still cannot re-author retained bytes."""
    old = _legacy(review, "Luft")
    path = tmp_path / "retained_approval.json"
    raw, binding = _retain_approval(path, old)
    altered = raw.replace(b'"Luft"', b'"air"', 1) + b" "
    assert len(altered) == len(raw)
    assert hashlib.sha256(altered).hexdigest() != binding["sha256"]
    assert subject._approval(json.loads(altered), review)["runs"][0]["classification"] == "air"
    path.write_bytes(altered)

    def forbidden(*args, **kwargs):
        """Reject any class normalization or scientific replay before bound validation."""
        pytest.fail("Changed retained bytes reached class resolution or replay")

    monkeypatch.setattr(subject, "classification_key", forbidden)
    monkeypatch.setattr(subject, "verify_analysis_exclusions", forbidden)
    preparation = {subject.KEY: dict(format=subject.EXCLUSION_FORMAT, meaning=subject.MEANING,
                                    approval=binding, runs={})}
    with pytest.raises(ValueError):
        subject.prepared_exclusion(preparation, tmp_path, review["runs"][0]["run_id"])
    assert path.read_bytes() == altered
    assert old["runs"][0]["classification"] == "Luft"


@pytest.mark.parametrize("invalid", [None, True, 1, 1.0, [], {}, "", "unknown", "Air", "AIR", "air ", " air", "Luft "])
def test_unknown_or_malformed_classes_refuse_without_normalizing(review, invalid):
    """Only the two exact declared namespaces are accepted; labels are not keys."""
    with pytest.raises(ValueError):
        subject.classification_key(invalid)
    with pytest.raises(ValueError):
        subject.classification_label(invalid)
    with pytest.raises(ValueError):
        subject.make_gap_approval(review, _keys(review), **ACTION,
                                 classification=invalid, other_text="Explicit choice")
    malformed = _legacy(review, "Luft")
    malformed["runs"][0]["classification"] = deepcopy(invalid)
    with pytest.raises(ValueError):
        subject._approval(malformed, review)


@pytest.mark.parametrize("classification", ["other", "Annet"])
@pytest.mark.parametrize("explanation", ["", "   ", None])
def test_both_other_spellings_retain_the_existing_description_requirement(review, classification, explanation):
    """Legacy reading and explicit new actions share the original nonblank rule."""
    with pytest.raises(ValueError):
        subject.make_gap_approval(review, _keys(review), **ACTION,
                                 classification=classification, other_text=explanation)
    malformed = _legacy(review, classification)
    malformed["runs"][0]["other_text"] = explanation
    with pytest.raises(ValueError):
        subject._approval(malformed, review)
