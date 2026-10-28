"""Tiny original Prepare/Store/export replay with explicit region confirmations."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import corrected_setup as export
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_analysis_exclusions as exclusions
from darsia.fluidflower import painted_gap_regions as regions
from darsia.fluidflower import painted_target_integration as targets
from darsia.fluidflower.painted_target_corrections import TargetCorrectionDraft
from .test_painted_analysis_exclusions import (
    ACTION, _make_gap_case, _resolved, gap_case, imported, registered_setup, synthetic_payload_admission,
)


CONFIRM = {
    "actor": "synthetic reviewer", "approved_utc": "2026-01-01T00:00:00Z",
    "reason": "Explicit synthetic non-sand region choice", "decision_ref": "synthetic confirmation",
}


@pytest.fixture
def both_gap_case(registered_setup, tmp_path, monkeypatch):
    """Reuse the original explicit own-frame DATA seam so both tiny runs have gaps."""
    def full_own_frames(frames):
        """Use the same native corners as the existing two-run TARGET fixture."""
        for frame in frames.values():
            frame["corners"] = dict(top_left=[0.0, 0.0], top_right=[0.0, 2.0],
                                    bottom_right=[1.0, 2.0], bottom_left=[1.0, 0.0])
    return _make_gap_case(registered_setup, tmp_path, monkeypatch, own_frame_change=full_own_frames)


def _fill(draft, classification):
    """Use original polygon semantics, clipped only by the exact component mask."""
    height, width = draft.editable_mask.shape
    assert draft.paint("polygon", [[0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]], classification, radius_px=1)


def _confirm(review, root, *, seed_later=False, captures=None):
    """Make each actual run/component confirmation through the public save API."""
    approval = None
    last_path = None
    position = 0
    for row in review["runs"]:
        for component in row["components"]:
            packet = exclusions.read_gap_preview(review, row["run_id"], component["component_id"])
            draft = regions.region_draft_factory(review, packet)
            decision = CONFIRM
            if seed_later and approval is not None:
                origin = approval["runs"][0]
                source_context = {key: origin[key] for key in
                                  ("run_id", "component_id", "review_row_sha256")}
                source_context["review"] = approval["review"]
                # These are explicit synthetic target-native preview inputs;
                # no coordinate transport is performed or authenticated here.
                masks = {name: np.zeros(draft.editable_mask.shape, bool) for name in draft.class_masks}
                masks["air"] = draft.editable_mask.copy()
                draft.set_preview_class_masks(masks, source_context=source_context)
                assert not draft.events
                decision = dict(CONFIRM, reason="Explicit new target confirmation",
                                decision_ref="new target confirmation")
            else:
                _fill(draft, "air")
            destination = root / ("region-" + str(position))
            frozen = draft.snapshot()
            previous = deepcopy(approval)
            approval = regions.save_gap_region_approval(frozen, destination,
                previous=previous, **decision)
            if captures is not None:
                captures.append((frozen, previous, deepcopy(approval)))
            last_path = destination / "approval.json"
            position += 1
    assert approval is not None and last_path is not None
    return approval, exclusions._bound(last_path)


def _prepare(case, approval, name, *, target_bindings=None):
    """Keep the original scientific preparation and all fixture source guards live."""
    kwargs = dict(case.captured["kwargs"], analysis_exclusions=approval, review_findings=True)
    if target_bindings is not None:
        kwargs["target_corrections"] = target_bindings
    return case.actual(case.root / name, *case.captured["args"][1:], **kwargs)


def _adopt(case, prepared, approval):
    """Use actual Check, exact warning seal and original immutable Store publication."""
    declaration = exclusions.make_facies_source_acquisition(prepared, "indicator", **ACTION)
    candidate, _ = adoption.adoption_candidate(case.data.revision, prepared["folder"],
        facies_source_acquisition=declaration)
    report = geometry.validate_geometry_config(candidate)
    assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    sealed = _resolved(candidate, report, approval)
    saved = geometry.GeometryRevisionStore(case.data.project).save_revision(candidate,
        actor="synthetic reviewer", reason="Explicit synthetic region confirmation",
        validation=report, review_resolutions=sealed, preserve_failed_artifacts=True)
    return candidate, report, sealed, saved


def _export_runs(case, saved):
    """Honor the original reference-first export guard before any selected child."""
    display = saved["configuration"]["frame_binding"]["display_reference_run"]
    outputs = {display: export.export_corrected_setup(case.data.project,
        reference=SimpleNamespace(run_id=display), actor="synthetic export")}
    for row in case.review["runs"]:
        run = row["run_id"]
        if run not in outputs:
            outputs[run] = export.export_corrected_setup(case.data.project,
                reference=SimpleNamespace(run_id=run), actor="synthetic export")
    return outputs


def test_confirmed_partition_keeps_original_buffers_and_roundtrips_native_store_export(both_gap_case, monkeypatch):
    """An explicit v2 partition retains the old native contract and physical pixels."""
    case = both_gap_case
    assert {row["run_id"] for row in case.review["runs"] if row["components"]} == {"one", "two"}
    captures = []
    approval, approval_binding = _confirm(case.review, case.root / "confirmed",
        seed_later=True, captures=captures)
    assert approval["format"] == exclusions.APPROVAL_FORMAT_V2
    target_snapshot, previous, submitted = captures[-1]
    assert target_snapshot.preview_source_context["run_id"] != target_snapshot.run_id
    assert regions.validate_gap_region_result(submitted, case.review, target_snapshot,
                                              previous=previous) == submitted
    selected = next(item for item in approval["runs"] if (item["run_id"], item["component_id"]) ==
                    (target_snapshot.run_id, target_snapshot.component_id))
    assert selected["reason"] == "Explicit new target confirmation"
    assert selected["decision_ref"] == "new target confirmation"
    assert approval["runs"][0] == previous["runs"][0]
    approved_bytes = Path(approval_binding["path"]).read_bytes()
    action = approval["runs"][0]
    restored = regions.read_confirmed_gap_regions(case.review, approval_binding,
        run_id=action["run_id"], component_id=action["component_id"])
    assert set(restored) == {"run_id", "component_id", "review_row_sha256", "component_mask",
                            "class_masks", "regions", "human_action", "bindings",
                            "target_correction_bindings"}
    assert restored["human_action"] == CONFIRM
    assert restored["regions"] == action["regions"]
    assert restored["bindings"] == {"review": approval["review"], "approval": approval_binding,
                                    "partition": action["partition"]["artifact"]}
    assert not restored["class_masks"]["sand"].any()
    np.testing.assert_array_equal(restored["class_masks"]["air"], restored["component_mask"])
    assert not restored["component_mask"].flags.writeable
    for mask in restored["class_masks"].values():
        assert mask.dtype == bool and not mask.flags.writeable
    prepared = _prepare(case, approval, "confirmed_preparation")
    assert prepared["analysis_exclusions"]["format"] == exclusions.EXCLUSION_FORMAT_V2
    for row in case.review["runs"]:
        entry = next(item for item in prepared["runs"] if item["run_id"] == row["run_id"])
        actual = exclusions._arrays(dict(entry["labels"],
            path=str(Path(prepared["folder"]) / entry["labels"]["path"])))
        original = exclusions._arrays(row["product"])
        assert {key: exclusions._identity(value) for key, value in actual.items()} == {
            key: exclusions._identity(value) for key, value in original.items()}
    candidate, report, sealed, saved = _adopt(case, prepared, approval)
    assert saved["review_resolutions"]["input"] == sealed
    assert saved["review_resolutions"]["materialized"]["records"]["analysis_exclusions"] == approval

    # Equal array/ZIP bytes do not authorize rewriting an already sealed action.
    rebound = deepcopy(sealed)
    old = rebound["records"]["analysis_exclusions"]["runs"][0]["partition"]["artifact"]
    copied = case.root / "same-partition-bytes.npz"
    copied.write_bytes(Path(old["path"]).read_bytes())
    new = exclusions._bound(copied)
    assert (new["sha256"], new["size_bytes"]) == (old["sha256"], old["size_bytes"])
    rebound["records"]["analysis_exclusions"]["runs"][0]["partition"]["artifact"] = new
    with pytest.raises(ValueError, match="exact approved connected areas"):
        exclusions.verify_findings_resolutions(candidate, report, rebound)
    fresh, _ = _confirm(case.review, case.root / "fresh_explicit_confirmation")
    assert regions.validate_gap_region_approval(fresh, case.review) == fresh
    assert fresh["runs"][0]["partition"]["artifact"] != old

    for run, output in _export_runs(case, saved).items():
        if run not in prepared["analysis_exclusions"]["runs"]:
            continue
        contract = output.receipt["binding"]["analysis_exclusions"]
        assert contract["format"] == "darsia-approved-analysis-gap-native-v1"
        labels = geometry._load_array(output.destination / "native_labels.npz")
        admitted, decoded = set(), []
        partitions = {item["partition"]["artifact"]["path"] for item in approval["runs"]
                      if item["run_id"] == run}
        foreign = {item["partition"]["artifact"]["path"] for item in approval["runs"]
                   if item["run_id"] != run}
        original_load = np.load

        def admit(bound):
            """Use the existing metadata reader admission callback literally."""
            admitted.add(bound["path"])
            return Path(bound["path"])

        def load(path, *args, **kwargs):
            """Observe genuine selected partition decode after dependency admission."""
            if isinstance(path, (str, Path)):
                assert str(path) not in foreign
                if str(path) in partitions:
                    assert str(path) in admitted
                    decoded.append(str(path))
            return original_load(path, *args, **kwargs)

        with monkeypatch.context() as selected:
            selected.setattr(np, "load", load)
            native = exclusions.read_native_analysis_exclusions(contract, actual_labels=labels,
                run_id=run, reader=SimpleNamespace(path=admit))
        assert set(decoded) == partitions
        expected = exclusions.configuration_exclusion(candidate, run)
        np.testing.assert_array_equal(native["mask"], expected)
        assert native["mask"].dtype == bool
        assert exclusions.adopted_run_analysis_exclusions(saved, run,
            project_root=case.data.project.root) == contract
    assert Path(approval_binding["path"]).read_bytes() == approved_bytes
    assert exclusions._bound(approval_binding["path"]) == approval_binding


def test_confirm_requires_complete_nonsand_regions_and_other_explanation(gap_case):
    """An unfinished or Sand preview never becomes C, even on an actual review."""
    case = gap_case
    row = next(row for row in case.review["runs"] if row["components"])
    component = row["components"][0]
    packet = exclusions.read_gap_preview(case.review, row["run_id"], component["component_id"])
    draft = regions.region_draft_factory(case.review, packet)
    unmarked = case.root / "unmarked_refused"
    with pytest.raises(ValueError):
        regions.save_gap_region_approval(draft.snapshot(), unmarked, **CONFIRM)
    assert not unmarked.exists()
    _fill(draft, "other")
    unexplained = case.root / "other_refused"
    with pytest.raises(ValueError):
        regions.save_gap_region_approval(draft.snapshot(), unexplained, **CONFIRM)
    assert not unexplained.exists()
    _fill(draft, "sand")
    sand = case.root / "sand_refused"
    with pytest.raises(ValueError):
        regions.save_gap_region_approval(draft.snapshot(), sand, **CONFIRM)
    assert not sand.exists()


def test_mixed_sand_uses_original_target_then_new_actual_residual_confirmation(gap_case):
    """The post-TARGET scientific review is new; old preview choices do not approve it."""
    case = gap_case
    row = next(row for row in case.review["runs"] if row["components"])
    component = row["components"][0]
    packet = exclusions.read_target_paint(case.review, row["run_id"], component["component_id"])
    assert int(packet["editable_mask"].sum()) >= 2
    draft = regions.region_draft_factory(case.review, packet)
    _fill(draft, "air")
    y, x = np.argwhere(packet["editable_mask"])[0]
    point = [int(x), int(y)]
    # The original polygon kernel admits a one-pixel polygon on this tiny grid.
    polygon = [point, point, point]
    assert draft.paint("polygon", polygon, "sand", radius_px=1)
    sand_mask = draft.class_masks["sand"].copy()
    assert int(sand_mask.sum()) == 1
    old_snapshot = draft.snapshot()
    with pytest.raises(ValueError, match="Sand"):
        regions.save_gap_region_approval(old_snapshot, case.root / "mixed_refused", **CONFIRM)
    target = TargetCorrectionDraft(labels=packet["target_labels"],
        editable_mask=packet["editable_mask"],
        allowed_ids=[item["layer_id"] for item in packet["allowed_layers"]],
        basis=packet["target_correction_basis"])
    label = packet["allowed_layers"][0]["layer_id"]
    assert target.paint("polygon", polygon, label, radius_px=1)
    target_binding = target.snapshot().save_and_approve(case.root / "target_confirmation",
        actor=CONFIRM["actor"], approved_utc=CONFIRM["approved_utc"], reason=CONFIRM["reason"])
    kwargs = dict(case.captured["kwargs"], target_corrections=[target_binding], review_findings=True)
    with pytest.raises(exclusions.GapReviewRequired) as pending:
        case.actual(case.root / "actual_post_target_review", *case.captured["args"][1:], **kwargs)
    post_review = pending.value.review
    assert post_review["review_binding"] != case.review["review_binding"]
    post_row = next(item for item in post_review["runs"] if item["run_id"] == row["run_id"])
    assert post_row["target_correction_bindings"] == [target_binding]
    assert post_row["gap"]["pixels"] == row["gap"]["pixels"] - 1
    assert post_row["components"] and component["component_id"] not in {
        item["component_id"] for item in post_row["components"]}
    with pytest.raises(ValueError):
        regions.validate_gap_region_snapshot(old_snapshot, post_review)
    approval, _ = _confirm(post_review, case.root / "post_target_confirmation")
    prepared = _prepare(case, approval, "confirmed_post_target", target_bindings=[target_binding])
    entry = next(item for item in prepared["runs"] if item["run_id"] == row["run_id"])
    effective, _ = targets.effective_product(prepared, prepared["folder"], entry)
    assert np.all(effective["array"][sand_mask] == label)
    expected = packet["editable_mask"] & ~sand_mask
    np.testing.assert_array_equal(effective["gaps"], expected)
    mask = exclusions.prepared_exclusion(prepared, prepared["folder"], row["run_id"])
    np.testing.assert_array_equal(mask, expected)
    assert not np.any(mask & sand_mask)
    candidate, _, _, saved = _adopt(case, prepared, approval)
    output = _export_runs(case, saved)[row["run_id"]]
    labels = geometry._load_array(output.destination / "native_labels.npz")
    np.testing.assert_array_equal(labels, effective["array"])
    contract = output.receipt["binding"]["native_product"]
    native = exclusions.read_native_product(contract, actual_labels=labels, run_id=row["run_id"])
    np.testing.assert_array_equal(native["mask"], expected)
    assert native["contract"]["target_corrections"] == [target_binding]
    assert exclusions.configuration_exclusion(candidate, row["run_id"]).tobytes() == expected.tobytes()
