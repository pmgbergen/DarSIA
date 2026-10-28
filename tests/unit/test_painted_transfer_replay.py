"""Real original transfer/gap/target replay; no replacement scientific guards."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower.painted_target_corrections import TargetCorrectionDraft
from darsia.fluidflower.painted_transfer_replay import (
    replay_transfer_review, build_residual_transfer_review, materialize_transfer_findings,
)

from .test_painted_analysis_exclusions import (
    ACTION, gap_case, registered_setup, imported, synthetic_payload_admission, _paint_component,
)  # noqa: F401 -- original fixture dependency chain remains unchanged


def _context(case):
    # The original producer must bind this before publishing its review. Tests
    # never edit/rebind a review to manufacture a replacement source authority.
    return deepcopy(case.review["preparation_context"])


def _retained(case):
    descriptors = [case.review["review_binding"], case.review["source_map"],
                   _context(case)["continuation"]]
    for row in case.review["runs"]:
        descriptors.extend([row["base_product"], row["product"], row["target_model"]])
    return {str(Path(bound["path"])): Path(bound["path"]).read_bytes() for bound in descriptors}


def _unchanged(before):
    for name, payload in before.items():
        assert Path(name).read_bytes() == payload


def _identities(arrays):
    return {name: gaps._identity(value) for name, value in arrays.items()}


def _partial_target(case):
    row, component = max(((row, item) for row in case.review["runs"] for item in row["components"]),
                         key=lambda pair: pair[1]["pixels"])
    packet = gaps.read_target_paint(case.review, row["run_id"], component["component_id"])
    assert packet["editable_mask"].sum() > 1, "The real tiny fixture must leave a residual area"
    draft = TargetCorrectionDraft(labels=packet["target_labels"], editable_mask=packet["editable_mask"],
        allowed_ids=[item["layer_id"] for item in packet["allowed_layers"]],
        basis=packet["target_correction_basis"])
    y, x = np.argwhere(packet["editable_mask"])[0]
    # Three equal vertices are a legal one-pixel polygon in the ORIGINAL
    # categorical rasterizer. No brush/guard is mocked to force a partial edit.
    point = [int(x), int(y)]
    assert draft.paint("polygon", [point, point, point], packet["allowed_layers"][0]["layer_id"])
    assert draft.painted_mask.sum() == 1
    binding = draft.snapshot().save_and_approve(case.root / "partial_target", actor=ACTION["actor"],
        approved_utc=ACTION["approved_utc"], reason="Explicit synthetic one-pixel target correction")
    return row, packet, binding, draft.labels, draft.painted_mask


def test_original_transfer_replay_preserves_all_base_fields_and_publishes_nothing(gap_case):
    before = _retained(gap_case)
    context = _context(gap_case)
    result = replay_transfer_review(gap_case.review["review_binding"], source_context=context)
    assert context == _context(gap_case)
    assert result["painting_binding"] == gap_case.review["source"]["painting_facies"]
    assert result["fixture_source"] is None and result["fixture_drawn_ids"] == set()
    assert result["target_bindings"] == []
    with np.load(gap_case.review["source_map"]["path"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(result["approved_analysis"], archive["array"])
    assert [row["run_id"] for row in result["runs"]] == [row["run_id"] for row in gap_case.review["runs"]]
    for replayed, original in zip(result["runs"], gap_case.review["runs"]):
        assert _identities(replayed["base_arrays"]) == original["base_scientific_arrays"]
        assert _identities(replayed["reviewed_arrays"]) == original["scientific_arrays"]
        assert _identities(replayed["effective_arrays"]) == original["scientific_arrays"]
        np.testing.assert_array_equal(replayed["residual_mask"], replayed["base_arrays"]["gaps"])
        assert [item for item, _mask in replayed["components"]] == [
            {key: value for key, value in item.items() if key != "review_row_sha256"}
            for item in original["components"]]
        assert replayed["target_correction_bindings"] == [] and replayed["native_drawn_ids"] == set()
        for arrays in (replayed["base_arrays"], replayed["reviewed_arrays"], replayed["effective_arrays"]):
            assert all(not value.flags.writeable for value in arrays.values())
        assert all(not mask.flags.writeable for _component, mask in replayed["components"])
        with pytest.raises(ValueError):
            replayed["effective_arrays"]["array"].flat[0] = 99
    assert result["source_bindings"]["review"] == gap_case.review["review_binding"]
    pinned = {item["path"] for item in result["source_bindings"]["files"]}
    assert gap_case.review["review_binding"]["path"] in pinned
    assert gap_case.review["source_map"]["path"] in pinned and context["continuation"]["path"] in pinned
    assert not (Path(result["folder"]) / "index.json").exists()
    assert "analysis_exclusions" not in result["preparation"]
    _unchanged(before)


@pytest.mark.parametrize("reuse", [False, True])
def test_fixture_replay_retains_bound_geometry_for_reviewed_point_transport(
        registered_setup, tmp_path, monkeypatch, reuse):
    from darsia.fluidflower import painted_adoption_table as table
    from .test_painted_adoption_table import prepared_painting

    prepared, *_ = prepared_painting(registered_setup, tmp_path, fixture=True, continuation=True)
    review = gaps.build_gap_review(prepared, prepared["folder"])
    before = _retained(SimpleNamespace(review=review))
    actual = table._checked_native_underlay
    checked_runs = []

    def checked(folder, row, source, labels, preparation=None):
        # A native fixture is placed with the index's approved point matrix.
        # Dropping the index reverts to the historical model matrix, even when
        # that happens to produce the same categorical pixels at this scale.
        assert preparation is not None
        assert preparation["geometry_index"] == review["geometry_index"]
        result = actual(folder, row, source, labels, preparation)
        checked_runs.append(row["run_id"])
        return result

    monkeypatch.setattr(table, "_checked_native_underlay", checked)
    replay = replay_transfer_review(review["review_binding"],
        source_context=review["preparation_context"], reuse_products=reuse)
    assert checked_runs == [row["run_id"] for row in review["runs"]]
    assert all(row["native_drawn_ids"] for row in replay["runs"])
    _unchanged(before)


def test_equal_bytes_at_another_continuation_locator_do_not_replace_bound_source_context(gap_case):
    context = _context(gap_case)
    original = _retained(gap_case)
    copy = gap_case.root / "same_bytes_another_continuation.json"
    copy.write_bytes(Path(context["continuation"]["path"]).read_bytes())
    replacement = gaps._bound(copy)
    assert replacement["sha256"] == context["continuation"]["sha256"]
    context["continuation"] = replacement
    with pytest.raises(ValueError):
        replay_transfer_review(gap_case.review, source_context=context)
    _unchanged(original)


@pytest.mark.parametrize("artifact", ["source_map", "continuation", "target_model"])
def test_original_bound_source_tampering_still_refuses_replay(gap_case, artifact):
    context = _context(gap_case)
    before = _retained(gap_case)
    if artifact == "source_map": binding = gap_case.review["source_map"]
    elif artifact == "continuation": binding = context["continuation"]
    else: binding = gap_case.review["runs"][0]["target_model"]
    path = Path(binding["path"])
    changed = path.read_bytes() + b"\noriginal guard must reject this changed bound source"
    path.write_bytes(changed)
    before[str(path)] = changed
    with pytest.raises(ValueError):
        replay_transfer_review(gap_case.review, source_context=context)
    _unchanged(before)


def test_partial_real_target_paint_leaves_a_real_residual_and_fresh_review(gap_case):
    before = _retained(gap_case)
    row, packet, binding, labels, painted = _partial_target(gap_case)
    replay = replay_transfer_review(gap_case.review, source_context=_context(gap_case), target_bindings=[binding])
    selected = next(item for item in replay["runs"] if item["run_id"] == row["run_id"])
    np.testing.assert_array_equal(selected["effective_arrays"]["array"], labels)
    np.testing.assert_array_equal(selected["painted_mask"], painted)
    assert selected["residual_mask"].any()
    assert selected["residual_mask"].sum() == selected["base_arrays"]["gaps"].sum() - 1
    for name in ("filled", "support", "inside", "continued"):
        np.testing.assert_array_equal(selected["effective_arrays"][name], selected["base_arrays"][name])
    assert selected["target_correction_bindings"] == [binding]
    version = gaps._checked(binding["version"], metadata=True)
    pinned = {(item["path"], item["sha256"], item["size_bytes"])
              for item in replay["source_bindings"]["files"]}
    for descriptor in (binding["version"], binding["approval"], version["product"], version["basis"]["review"]):
        assert (descriptor["path"], descriptor["sha256"], descriptor["size_bytes"]) in pinned
    fresh = build_residual_transfer_review(replay, gap_case.root / "partial_residual_review")
    fresh_row = next(item for item in fresh["runs"] if item["run_id"] == row["run_id"])
    assert fresh_row["gap"]["pixels"] == int(selected["residual_mask"].sum())
    assert fresh_row["target_correction_bindings"] == [binding]
    assert fresh_row["components"] and all(item["component_id"] != packet["component_id"] for item in fresh_row["components"])
    cold = replay_transfer_review(fresh, source_context=fresh["preparation_context"])
    effective = next(item for item in cold["runs"] if item["run_id"] == row["run_id"])
    assert _identities(effective["effective_arrays"]) == _identities(selected["effective_arrays"])
    # A second genuine approval over the ORIGINAL base is valid by itself;
    # selecting it must not silently replace the partial edit retained above.
    # This tiny fixture has only one run with gaps, so exercise an independent
    # same-run replacement rather than inventing another run's component.
    _original_row, _original_packet, replacement = _paint_component(gap_case, run_id=row["run_id"])
    with pytest.raises(ValueError, match="exact ordered prior edits"):
        replay_transfer_review(fresh, source_context=fresh["preparation_context"], target_bindings=[replacement])
    assert effective["target_correction_bindings"] == [binding]
    assert not (Path(fresh["review_binding"]["path"]).parent / "index.json").exists()
    _unchanged(before)


def test_full_real_target_paint_retires_only_its_gap_and_duplicate_bindings_still_refuse(gap_case):
    before = _retained(gap_case)
    row, packet, binding = _paint_component(gap_case)
    replay = replay_transfer_review(gap_case.review, source_context=_context(gap_case), target_bindings=[binding])
    selected = next(item for item in replay["runs"] if item["run_id"] == row["run_id"])
    assert not np.any(selected["residual_mask"] & packet["editable_mask"])
    assert not np.any(selected["painted_mask"] & ~packet["editable_mask"])
    np.testing.assert_array_equal(selected["effective_arrays"]["array"][~packet["editable_mask"]],
                                  selected["base_arrays"]["array"][~packet["editable_mask"]])
    for other in replay["runs"]:
        if other["run_id"] != row["run_id"]:
            assert _identities(other["effective_arrays"]) == _identities(other["base_arrays"])
    with pytest.raises(ValueError):
        replay_transfer_review(gap_case.review, source_context=_context(gap_case), target_bindings=[binding, binding])
    _unchanged(before)


def test_fresh_residual_publication_copies_original_bytes_without_index_or_exclusion(gap_case):
    before = _retained(gap_case)
    replay = replay_transfer_review(gap_case.review, source_context=_context(gap_case))
    destination = gap_case.root / "fresh_residual_review"
    fresh = build_residual_transfer_review(replay, destination)
    assert Path(fresh["review_binding"]["path"]).parent == destination
    assert Path(fresh["source_map"]["path"]).read_bytes() == Path(gap_case.review["source_map"]["path"]).read_bytes()
    assert (destination / "continuation.json").read_bytes() == Path(_context(gap_case)["continuation"]["path"]).read_bytes()
    assert not (destination / "index.json").exists()
    assert "analysis_exclusions" not in fresh and "target_corrections" not in fresh
    for old, new in zip(gap_case.review["runs"], fresh["runs"]):
        assert Path(new["base_product"]["path"]).read_bytes() == Path(old["base_product"]["path"]).read_bytes()
        assert new["scientific_arrays"] == old["scientific_arrays"]
        assert new["gap"]["identity"] == old["gap"]["identity"]
    cold = replay_transfer_review(fresh["review_binding"], source_context=fresh["preparation_context"])
    assert [item["run_id"] for item in cold["runs"]] == [item["run_id"] for item in replay["runs"]]
    _unchanged(before)


def test_residual_writer_never_overwrites_retained_or_existing_namespace(gap_case):
    before = _retained(gap_case)
    replay = replay_transfer_review(gap_case.review, source_context=_context(gap_case))
    retained = Path(replay["folder"])
    existing = gap_case.root / "already_exists"
    existing.mkdir()
    sentinel = existing / "keep.bin"
    sentinel.write_bytes(b"existing unrelated output; never overwrite")
    for destination in (retained, retained / "child", existing):
        with pytest.raises((ValueError, FileExistsError)):
            build_residual_transfer_review(replay, destination)
    assert sentinel.read_bytes() == b"existing unrelated output; never overwrite"
    assert not (retained / "child").exists() and not (existing / "gap_review.json").exists()
    _unchanged(before)


def test_materialization_requires_real_human_area_choice_and_keeps_its_original_review(gap_case):
    """Pair the original own-frame refusal with its actual v1 human action.

    This checks compatibility only. It does not fabricate a v2 partition or
    convert the residual into an implicit Other/outside-frame decision.
    """
    before = _retained(gap_case)
    replay = replay_transfer_review(gap_case.review, source_context=_context(gap_case))
    assert any(row["residual_mask"].any() for row in replay["runs"])
    refused = gap_case.root / "unresolved_materialization"
    with pytest.raises(ValueError, match="zero labels are INSIDE"):
        materialize_transfer_findings(replay, refused)
    assert not (refused / "index.json").exists()
    keys = [dict(run_id=row["run_id"], component_id=component["component_id"])
            for row in gap_case.review["runs"] for component in row["components"]]
    approval = gaps.make_gap_approval(gap_case.review, keys, classification="air", **ACTION)
    result = materialize_transfer_findings(replay, gap_case.root / "explicit_air_materialization",
                                          analysis_exclusions=approval)
    assert result["review"]["review_binding"] == gap_case.review["review_binding"]
    assert gaps._checked(result["analysis_exclusions"]["approval"], metadata=True) == approval
    assert approval["review"] == gap_case.review["review_binding"]
    assert all(row["classification"] == "air" for row in approval["runs"])
    assert result["outside_frame_proof"] is not None and result["outside_proof_bindings"]
    assert result["preparation"]["outside_frame_proof"] == result["outside_frame_proof"]
    for row in replay["runs"]:
        mask = gaps.prepared_exclusion(result["preparation"], result["folder"], row["run_id"])
        if row["residual_mask"].any():
            np.testing.assert_array_equal(mask, row["residual_mask"])
        else:
            assert mask is None
    for binding in result["outside_proof_bindings"]:
        gaps._checked(binding)
    assert not (Path(result["folder"]) / "index.json").exists()
    assert not (Path(result["folder"]) / "gap_review.json").exists()
    _unchanged(before)
