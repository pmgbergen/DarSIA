"""Source-authored regression coverage; deliberately not run in the B2 source phase."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import phase_review_contradictions as subject
from darsia.fluidflower.phase_evidence_audit import disagreement_reasons, payload_hash
from darsia.fluidflower.phase_rendering import AQUEOUS_RGBA, GAS_RGBA
from darsia.fluidflower.phase_training.logistic import ReviewedRegion
from darsia.fluidflower.project import sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def native():
    return dict(run="ac01", canvas_wh=[4, 4], content_box_xywh=[0, 0, 4, 4],
        frame_count=1, frame_sources=[dict(frame_index=0, elapsed_h=0., source_name="camera.raw",
                                        source_sha256="a" * 64)])


def annotation():
    return dict(id="original", frame_index=0, elapsed_h=0., category="confirmed_only_free_gas",
        shape="rectangle", points=[], x=0., y=0., width=1., height=1., confidence="high")


def region(label="gas"):
    return ReviewedRegion(region_id="ac01:manual:original", run="ac01", dataset="ffac",
        frame_index=0, elapsed_h=0., phase_class=label, confidence="high", source="manual",
        reviewer="Human reviewer", payload=annotation())


def rgba(color=AQUEOUS_RGBA):
    return np.broadcast_to(np.asarray(color, np.uint8), (4, 4, 4)).copy()


def test_fresh_actual_pixels_disagree_without_replacing_original_algorithms():
    source = region()
    before = deepcopy(source.payload)
    result = subject.evaluate_region(source, native(), rgba(), np.ones((4, 4), bool))
    assert result["evaluated_pixels"] == 16
    assert result["new"] == dict(gas_fraction=0., aqueous_fraction=1., co2_fraction=1.)
    assert disagreement_reasons(result) == [
        ("model_label_disagreement", 1., "Evaluated phase-area fraction conflicts with the regional label")]
    assert source.payload == before


def test_presence_label_does_not_become_a_full_box_requirement():
    image = np.zeros((4, 4, 4), np.uint8)
    image[0, 0] = GAS_RGBA
    measured = subject.evaluate_region(region("gas_present_region"), native(), image, np.ones((4, 4), bool))
    assert measured["new"]["gas_fraction"] == 1 / 16
    assert disagreement_reasons(measured) == []


def test_original_amber_normalization_and_domain_are_used():
    image = rgba((255, 191, 71, 255))
    image[:, 2:] = GAS_RGBA
    domain = np.zeros((4, 4), bool)
    domain[:, :2] = True
    measured = subject.evaluate_region(region(), native(), image, domain)
    assert measured["evaluated_pixels"] == 8
    assert measured["aqueous_fraction"] == 1.


@pytest.mark.parametrize("image,domain,message", [
    (np.zeros((8, 8, 4), np.uint8), np.ones((4, 4), bool), "geometry differ"),
    (rgba(), np.ones((4, 4), np.uint8), "geometry differ"),
    (rgba(), np.zeros((4, 4), bool), "no_registered_domain_pixels"),
    (rgba((30, 210, 255, 72)), np.ones((4, 4), bool), "unsupported colours"),
])
def test_no_resizing_or_synthetic_values_on_unsupported_pixels(image, domain, message):
    with pytest.raises(ValueError, match=message):
        subject.evaluate_region(region(), native(), image, domain)


def test_camera_and_geometry_must_match_even_at_same_time():
    changed = native()
    changed["frame_sources"][0]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="different camera acquisitions"):
        subject._check_frame(region(), native(), changed)
    changed = native()
    changed["geometry_revision_id"] = "new revision"
    with pytest.raises(ValueError, match="different native geometry"):
        subject._check_frame(region(), native(), changed)


def test_role_exclusion_and_durable_provenance_extension_are_preserved():
    source = annotation()
    role = subject.evidence_role(annotation=source, decision={}, document={}, manifest={},
        session=dict(include_in_training=False, independent_holdout=True),
        source_jobs=[dict(training_scope="heldout acquisition")])
    result = subject.preserve_correction_role(dict(id="correction", evidence_audit={"verdict": "confirmed"}), role)
    assert result["evidence_audit"]["source_evidence_role"] == role
    assert result["evidence_audit"]["include_in_training"] is False
    assert result["evidence_audit"]["post_reveal_edits_are_blind"] is False
    assert role["source_scopes"]["source_job_0"]["training_scope"] == "heldout acquisition"
    assert source == annotation()
    with pytest.raises(ValueError, match="training-excluded"):
        subject.require_correction_session_role(role, dict(include_in_training=True))
    subject.require_correction_session_role(role, dict(include_in_training=False))


def source_fixture(tmp_path, monkeypatch, *, missing=False, raw=False):
    from darsia.fluidflower import phase_evidence_audit, phase_runtime, phase_validation, phase_blind_review_policy
    from darsia.fluidflower.phase_training import trainer
    model, review, source, maps = [tmp_path / name for name in ("model", "review", "source", "maps")]
    model.mkdir()
    manifest = native()
    if raw:
        manifest.update(source_kind="raw")
        from darsia.fluidflower import phase_review_raw_contradictions
        def unsupported_raw(*args, **kwargs):
            raise phase_review_raw_contradictions.UnsupportedSavedPhaseFrame("RAW fixture has no saved support receipt")
        monkeypatch.setattr(phase_review_raw_contradictions, "load_saved_phase_frame", unsupported_raw)
    write(source / "native_frames/ac01/manifest.json", manifest)
    job = dict(run="ac01", dataset="ffac", analysis_source_review_root=str(source),
        review_output_dir=str(source), phase_model_only=True,
        raw_phase_map_output_dir=str(maps), optical_model_sha256="b" * 64)
    source_manifest = write(review / "postflight_manifest.json", dict(model_id="model", jobs=[job]))
    decisions = write(tmp_path / "evidence/decisions.json", dict(decisions={"ac01": dict(
        reviewer="Human reviewer", annotations=[annotation()])}))
    session = write(decisions.parent / "review_session.json", dict(include_in_training=False, independent_holdout=True))
    if not missing:
        path = maps / "native_phase_maps/ac01" / ("A_" + "b" * 16) / "frame_000.png"
        path.parent.mkdir(parents=True)
        Image.fromarray(rgba(), "RGBA").save(path)
    binding = dict(combined_root=str(review), model_id="model", model_sha256="model identity")
    monkeypatch.setattr(phase_evidence_audit, "verified_review_binding", lambda *a, **kw: deepcopy(binding))
    monkeypatch.setattr(phase_validation, "_model_identity", lambda root: ("model", "model identity"))
    monkeypatch.setattr(phase_runtime, "load_physics_profile", lambda root: None)
    monkeypatch.setattr(phase_blind_review_policy, "require_predictions_visible", lambda *a: None)
    def no_inference(*args, **kwargs):
        raise AssertionError("Fresh contradiction evaluation must not invoke inference or fitting")
    monkeypatch.setattr(trainer, "train_campaign", no_inference)
    inputs = [dict(decisions=str(decisions), manifest=str(source_manifest), session=str(session))]
    return model, review, inputs, decisions


def test_new_queue_uses_original_loader_rasterizer_and_queue_builder(tmp_path, monkeypatch):
    model, review, inputs, decisions = source_fixture(tmp_path, monkeypatch)
    before = decisions.read_bytes()
    result = subject.build_contradiction_queue(model, review, inputs, tmp_path / "new")
    queue = subject.read(result["queue_path"])
    assert result["coverage"]["complete"] is True
    assert result["entries"] == 1
    entry = queue["entries"][0]
    assert entry["original_annotation"] == annotation()
    assert entry["old_gas_fraction"] == 0.
    assert entry["aqueous_fraction"] == 1.
    assert entry["score"] == 1.
    assert subject.correction_role(queue, entry)["training_selection"] == "excluded_or_requires_explicit_selection"
    assert decisions.read_bytes() == before
    report = subject.read(result["evaluation_path"])
    assert report["training_performed"] is False and report["inference_performed"] is False
    assert report["scientific_acceptance"] is False


def test_original_supersession_applies_human_revision_without_erasing_original(tmp_path, monkeypatch):
    model, review, inputs, decisions = source_fixture(tmp_path, monkeypatch)
    document = subject.read(decisions)
    correction = dict(annotation(), id="correction", category="confirmed_only_aqueous_co2",
        reviewer="Human reviewer", updated_utc="2026-10-23T12:00:00Z", evidence_audit=dict(
            original_region_id="ac01:manual:original", original_annotation_sha256=payload_hash(annotation()),
            verdict="reclassified", queue_sha256="original queue", source_model_identity="model identity"))
    document["decisions"]["ac01"]["annotations"].append(correction)
    write(decisions, document)
    before = decisions.read_bytes()
    result = subject.build_contradiction_queue(model, review, inputs, tmp_path / "new")
    assert result["coverage"]["total_effective_regions"] == 1
    assert result["coverage"]["scored_regions"] == 1
    assert result["entries"] == 0
    assert subject.read(result["evaluation_path"])["regions"][0]["region_id"] == "ac01:manual:correction"
    assert decisions.read_bytes() == before


@pytest.mark.parametrize("missing,raw", [(True, False), (False, True)])
def test_missing_map_and_raw_domain_are_incomplete_coverage_never_pass(tmp_path, monkeypatch, missing, raw):
    model, review, inputs, _ = source_fixture(tmp_path, monkeypatch, missing=missing, raw=raw)
    result = subject.build_contradiction_queue(model, review, inputs, tmp_path / "new")
    assert result["coverage"]["complete"] is False
    assert result["coverage"]["scored_regions"] == 0
    assert result["coverage"]["skipped_regions"] == 1
    assert result["entries"] == 0


def test_changed_evaluation_cannot_supply_a_correction_role(tmp_path):
    report = write(tmp_path / "evaluation.json", dict(region_evidence_roles={}))
    queue = dict(evaluation_path=str(report), evaluation_sha256=sha256_file(report))
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation changed"):
        subject.correction_role(queue, {})


def test_seal_preserves_canonical_docs_and_first_snapshot(tmp_path):
    decisions = write(tmp_path / "decisions.json", dict(decisions={"ac01": dict(annotations=[annotation()])}))
    session = write(tmp_path / "review_session.json", dict(format="fluidflower-local-review-session-v1",
        target_model_id="model", review_session_id="session-one", reviewer="Human reviewer", include_in_training=False))
    originals = decisions.read_bytes(), session.read_bytes()
    marker = subject.seal_blind_evidence(decisions, session, model_id="model", reviewer="Human reviewer")
    snapshot = subject.read(marker["frozen_blind_snapshot"])
    assert snapshot["decisions"] == subject.read(decisions)["decisions"]
    assert snapshot["session_metadata"]["include_in_training"] is False
    assert (decisions.read_bytes(), session.read_bytes()) == originals
    assert marker == subject.seal_blind_evidence(decisions, session, model_id="model", reviewer="Human reviewer")
    Path(marker["frozen_blind_snapshot"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        subject.read_reveal_marker(tmp_path)


def test_disclosure_marker_never_releases_original_model_hold(tmp_path, monkeypatch):
    from darsia.fluidflower import phase_blind_review_policy
    decisions = write(tmp_path / "decisions.json", dict(decisions={}))
    session = write(tmp_path / "review_session.json", dict(format="fluidflower-local-review-session-v1",
        target_model_id="model", review_session_id="session-one", reviewer="Human reviewer"))
    subject.seal_blind_evidence(decisions, session, model_id="model", reviewer="Human reviewer")
    def held(*args):
        raise PermissionError("Independent candidate publication hold")
    monkeypatch.setattr(phase_blind_review_policy, "require_predictions_visible", held)
    with pytest.raises(PermissionError, match="publication hold"):
        subject.require_disclosure(tmp_path, tmp_path / "model", ["ac01"])


@pytest.mark.parametrize("field,value", [("target_model_id", "other-model"),
    ("review_session_id", "other-session"), ("reviewer", "Other human")])
def test_reopen_refuses_marker_from_another_model_session_or_owner(tmp_path, field, value):
    decisions = write(tmp_path / "decisions.json", dict(decisions={}))
    session = write(tmp_path / "review_session.json", dict(format="fluidflower-local-review-session-v1",
        target_model_id="model", review_session_id="session-one", reviewer="Human reviewer"))
    subject.seal_blind_evidence(decisions, session, model_id="model", reviewer="Human reviewer")
    changed = subject.read(session)
    changed[field] = value
    write(session, changed)
    with pytest.raises(ValueError, match="another model, review session or owner"):
        subject.read_reveal_marker(tmp_path)
