"""Unrun source-phase tests for saved RAW labels through the original queue route."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import phase_review_contradictions as subject
from darsia.fluidflower.phase_evidence_audit import disagreement_reasons, payload_hash
from darsia.fluidflower.phase_review_basis import SAVED_NATIVE_VERSION
from darsia.fluidflower.phase_review_raw_contradictions import SavedPhaseFrame, UnsupportedSavedPhaseFrame
from darsia.fluidflower.phase_training.logistic import ReviewedRegion
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def raw_native():
    """Original RAW schema uses frames, with no historical frame_sources alias."""
    return dict(format=RAW_REVIEW_FORMAT, source_kind="raw", run="ac01", frame_count=1,
        canvas_wh=[6, 6], content_box_xywh=[1, 1, 4, 4], native_shape_hw=[4, 4],
        phase_shape_hw=[6, 6], native_origin_xy=[-1, -1], geometry_revision_id="raw-native-one",
        native_configuration=dict(path="configuration.json", sha256="c" * 64),
        raw_binding=dict(path="raw_binding.json", sha256="d" * 64), reference={},
        frames=[dict(frame_index=0, elapsed_h=1.5, case_id="ac01_f0",
            acquisition=dict(path="camera.ARW", sha256="a" * 64),
            native_geometry=dict(path="native_geometry.json", sha256="e" * 64),
            rig_geometry=dict(path="rig_geometry.json", sha256="f" * 64))])


def annotation(**changes):
    return dict(dict(id="original", frame_index=0, elapsed_h=1.5,
        category="confirmed_only_free_gas", shape="rectangle", points=[],
        x=1 / 6, y=1 / 6, width=4 / 6, height=4 / 6, confidence="high"), **changes)


def region(label="gas", **changes):
    payload = annotation(**changes)
    return ReviewedRegion(region_id="ac01:manual:" + payload["id"], run="ac01", dataset="ffac",
        frame_index=payload["frame_index"], elapsed_h=payload["elapsed_h"], phase_class=label,
        confidence="high", source="manual", reviewer="Human reviewer", payload=payload)


def test_saved_class_fractions_use_native_crop_and_do_not_mutate_inputs():
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]], dtype=np.uint8)
    domain = np.ones((4, 4), bool)
    support = np.ones((4, 4), bool)
    source, native = region(), raw_native()
    before = (labels.copy(), domain.copy(), support.copy(), deepcopy(source.payload), deepcopy(native))
    row = subject.evaluate_saved_labels(source, native, labels, domain, support)
    assert row["new"] == dict(gas_fraction=.5, aqueous_fraction=.25, co2_fraction=.75)
    assert row["evaluated_pixels"] == row["annotation_pixels"] == 16
    assert row["scientific_support_verified"] is True
    assert row["phase_encoding"] == "0=water,1=aqueous,2=free-gas"
    for actual, original in zip((labels, domain, support), before[:3]):
        np.testing.assert_array_equal(actual, original)
    assert source.payload == before[3] and native == before[4]


def test_presence_constraint_is_not_promoted_to_full_box():
    labels = np.zeros((4, 4), np.uint8)
    labels[0, 0] = 2
    row = subject.evaluate_saved_labels(region("gas_present_region"), raw_native(), labels,
                                        np.ones((4, 4), bool), np.ones((4, 4), bool))
    assert row["new_gas_fraction"] == 1 / 16
    assert disagreement_reasons(row) == []


def test_support_hole_is_unscored_instead_of_counted_as_water():
    support = np.ones((4, 4), bool)
    support[1, 2] = False
    with pytest.raises(subject.UnsupportedReviewSource, match="without verified saved scientific support"):
        subject.evaluate_saved_labels(region(), raw_native(), np.zeros((4, 4), np.uint8),
                                      np.ones((4, 4), bool), support)


def test_outside_domain_classes_and_support_do_not_change_fraction():
    labels = np.full((4, 4), 255, np.uint8)
    labels[:, :2] = 1
    domain = np.zeros((4, 4), bool)
    domain[:, :2] = True
    row = subject.evaluate_saved_labels(region(), raw_native(), labels, domain, domain.copy())
    assert row["evaluated_pixels"] == 8
    assert row["annotation_pixels"] == 16
    assert row["new"] == dict(gas_fraction=0., aqueous_fraction=1., co2_fraction=1.)


@pytest.mark.parametrize("invalid", [3, 254, 255])
def test_unresolved_saved_class_inside_region_is_not_agreement(invalid):
    labels = np.zeros((4, 4), np.uint8)
    labels[1, 1] = invalid
    with pytest.raises(subject.UnsupportedReviewSource, match="unsupported classes"):
        subject.evaluate_saved_labels(region(), raw_native(), labels,
                                      np.ones((4, 4), bool), np.ones((4, 4), bool))


@pytest.mark.parametrize("labels,domain,support", [
    (np.zeros((6, 6), np.uint8), np.ones((4, 4), bool), np.ones((4, 4), bool)),
    (np.zeros((4, 4), np.int32), np.ones((4, 4), bool), np.ones((4, 4), bool)),
    (np.zeros((4, 4), np.uint8), np.ones((4, 4), np.uint8), np.ones((4, 4), bool)),
    (np.zeros((4, 4), np.uint8), np.ones((4, 3), bool), np.ones((4, 4), bool)),
    (np.zeros((4, 4), np.uint8), np.ones((4, 4), bool), np.ones((4, 4), np.uint8)),
    (np.zeros((4, 4), np.uint8), np.ones((4, 4), bool), np.ones((3, 4), bool)),
])
def test_saved_arrays_require_exact_native_shape_and_dtypes(labels, domain, support):
    with pytest.raises(ValueError, match="geometry differ"):
        subject.evaluate_saved_labels(region(), raw_native(), labels, domain, support)


@pytest.mark.parametrize("box", [[-1, 1, 4, 4], [3, 1, 4, 4], [1, 3, 4, 4], [1, 1, 0, 4]])
def test_saved_content_box_must_fit_original_canvas(box):
    native = raw_native()
    native["content_box_xywh"] = box
    with pytest.raises(ValueError, match="geometry differ"):
        subject.evaluate_saved_labels(region(), native, np.zeros((4, 4), np.uint8),
                                      np.ones((4, 4), bool), np.ones((4, 4), bool))


@pytest.mark.parametrize("changes,message", [
    ({"width": 0.}, "finite positive"), ({"height": -1.}, "finite positive"),
    ({"x": float("nan")}, "finite positive"), ({"x": 1.}, "outside"),
])
def test_invalid_or_external_annotation_geometry_is_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        subject.evaluate_saved_labels(region(**changes), raw_native(), np.zeros((4, 4), np.uint8),
                                      np.ones((4, 4), bool), np.ones((4, 4), bool))


def test_empty_registered_domain_is_explicitly_unscored():
    with pytest.raises(subject.UnsupportedReviewSource, match="no_registered_domain_pixels"):
        subject.evaluate_saved_labels(region(), raw_native(), np.zeros((4, 4), np.uint8),
                                      np.zeros((4, 4), bool), np.ones((4, 4), bool))


def test_check_frame_accepts_original_raw_frames_without_frame_sources():
    native, source = raw_native(), region()
    before = deepcopy(native)
    assert "frame_sources" not in native
    assert subject._check_frame(source, native, deepcopy(native)) == source
    assert native == before


@pytest.mark.parametrize("key", ["native_origin_xy", "native_shape_hw", "phase_shape_hw",
                                  "canvas_wh", "content_box_xywh", "geometry_revision_id"])
def test_check_frame_refuses_changed_raw_geometry(key):
    original, selected = raw_native(), raw_native()
    selected[key] = "another geometry" if key == "geometry_revision_id" else [99, 98]
    with pytest.raises(ValueError, match="different native geometry"):
        subject._check_frame(region(), original, selected)


@pytest.mark.parametrize("key", ["native_geometry", "rig_geometry"])
@pytest.mark.parametrize("missing", [False, True])
def test_check_frame_requires_both_recorded_geometry_hashes(key, missing):
    original, selected = raw_native(), raw_native()
    if missing:
        original["frames"][0][key].pop("sha256")
    else:
        selected["frames"][0][key]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different recorded geometry"):
        subject._check_frame(region(), original, selected)


def test_check_frame_refuses_another_raw_acquisition_at_the_same_time():
    original, selected = raw_native(), raw_native()
    selected["frames"][0]["acquisition"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different camera acquisitions"):
        subject._check_frame(region(), original, selected)


@pytest.mark.parametrize("side", ["original", "selected", "both"])
def test_raw_acquisition_sha_is_required_even_when_source_names_match(side):
    original, selected = raw_native(), raw_native()
    for name, native in (("original", original), ("selected", selected)):
        native["frames"][0]["source_name"] = "camera.ARW"
        if side in {name, "both"}:
            native["frames"][0]["acquisition"].pop("sha256")
    with pytest.raises(ValueError, match="acquisition"):
        subject._check_frame(region(), original, selected)


@pytest.mark.parametrize("side", ["original", "selected"])
@pytest.mark.parametrize("elapsed_h", [9., float("nan"), float("inf")])
def test_raw_annotation_time_must_match_both_original_and_selected_timeline(side, elapsed_h):
    original, selected = raw_native(), raw_native()
    (original if side == "original" else selected)["frames"][0]["elapsed_h"] = elapsed_h
    with pytest.raises(ValueError, match="time|timeline"):
        subject._check_frame(region(), original, selected)


@pytest.mark.parametrize("change", ["frame_index", "frame_count", "run", "raw_binding"])
def test_original_raw_timeline_structure_is_checked(change):
    original, selected = raw_native(), raw_native()
    if change == "frame_index":
        selected["frames"][0]["frame_index"] = 7
    elif change == "frame_count":
        selected["frame_count"] = 2
    elif change == "run":
        selected["run"] = "ac02"
    else:
        selected.pop("raw_binding")
    with pytest.raises(ValueError):
        subject._check_frame(region(), original, selected)


@pytest.fixture
def raw_queue_source(tmp_path, monkeypatch):
    from darsia.fluidflower import phase_evidence_audit, phase_runtime, phase_validation
    from darsia.fluidflower import phase_blind_review_policy, phase_review_basis, phase_review_raw_contradictions
    from darsia.fluidflower.phase_training import trainer

    model, review, source = (tmp_path / name for name in ("model", "review", "source"))
    model.mkdir()
    native = raw_native()
    acquisition = source / "camera.ARW"
    acquisition.parent.mkdir()
    acquisition.write_bytes(b"original retained camera acquisition")
    native["frames"][0]["acquisition"] = dict(path=str(acquisition), sha256=sha256_file(acquisition))
    native_path = write(source / "native_frames/ac01/manifest.json", native)
    job = dict(run="ac01", dataset="ffac", review_output_dir=str(source),
               quality_flags=dict(raw_production_review=True))
    manifest_path = write(review / "postflight_manifest.json", dict(model_id="model", jobs=[job]))
    decisions = write(tmp_path / "evidence/decisions.json", dict(decisions={"ac01": dict(
        reviewer="Human reviewer", annotations=[annotation()])}))
    session = write(decisions.parent / "review_session.json", dict(include_in_training=False,
        independent_holdout=True, evidence_role="blind_transfer_validation"))
    labels = np.ones((4, 4), np.uint8)
    domain, support = np.ones((4, 4), bool), np.ones((4, 4), bool)
    label_path = source / "saved-labels.png"
    Image.fromarray(labels).save(label_path)
    support_path = source / "saved-support.npz"
    np.savez(support_path, domain=domain, support=support)
    pins = {str(path.resolve()): sha256_file(path) for path in (native_path, label_path, support_path, acquisition)}
    provenance = dict(prediction=dict(path=str(label_path.resolve()), sha256=pins[str(label_path.resolve())]),
                      source_kind="original_saved_raw", scientific_support="retained_modelled_pixels")
    frame = SavedPhaseFrame(labels, deepcopy(native), domain, support, pins, provenance)
    binding = dict(combined_root=str(review), model_id="model", model_sha256="model identity")
    monkeypatch.setattr(phase_evidence_audit, "verified_review_binding", lambda *a, **kw: deepcopy(binding))
    monkeypatch.setattr(phase_validation, "_model_identity", lambda root: ("model", "model identity"))
    monkeypatch.setattr(phase_blind_review_policy, "require_predictions_visible", lambda *a: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("RAW saved review must not fall back to optical maps, domains, inference or fitting")

    monkeypatch.setattr(subject, "_domain", forbidden)
    monkeypatch.setattr(phase_review_basis, "uses_optical", forbidden)
    monkeypatch.setattr(phase_review_basis, "optical_path", forbidden)
    monkeypatch.setattr(phase_runtime, "load_physics_profile", forbidden)
    monkeypatch.setattr(trainer, "train_campaign", forbidden)
    calls, queue_calls = [], []

    def saved_frame(model_root, review_root, selected_job, frame_index, *, expected_native_sha256):
        calls.append((model_root, review_root, deepcopy(selected_job), frame_index, expected_native_sha256))
        return frame

    monkeypatch.setattr(phase_review_raw_contradictions, "load_saved_phase_frame", saved_frame)
    original_builder = phase_evidence_audit.build_queue

    def original_queue(*args, **kwargs):
        queue_calls.append((args, kwargs))
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(phase_evidence_audit, "build_queue", original_queue)
    inputs = [dict(decisions=str(decisions), manifest=str(manifest_path), session=str(session))]
    files = (decisions, session, native_path, manifest_path, label_path, support_path, acquisition)
    return SimpleNamespace(model=model, review=review, source=source, native=native, native_path=native_path,
        job=job, decisions=decisions, inputs=inputs, frame=frame, calls=calls, queue_calls=queue_calls,
        files=files, before={str(path): path.read_bytes() for path in files})


def assert_sources_unchanged(data):
    assert {str(path): path.read_bytes() for path in data.files} == data.before


def test_raw_saved_labels_reach_original_queue_builder_without_optical_fallback(tmp_path, raw_queue_source):
    data = raw_queue_source
    before_frame = deepcopy(data.frame)
    result = subject.build_contradiction_queue(data.model, data.review, data.inputs, tmp_path / "fresh")
    queue = subject.read(result["queue_path"])
    report = subject.read(result["evaluation_path"])
    assert "frame_sources" not in data.native
    assert len(data.queue_calls) == len(data.calls) == 1
    assert data.calls[0][3:] == (0, sha256_file(data.native_path))
    assert result["coverage"]["complete"] is True
    assert result["coverage"]["scored_regions"] == result["entries"] == 1
    assert queue["phase_review_basis"] == report["phase_review_basis"] == SAVED_NATIVE_VERSION
    entry = queue["entries"][0]
    assert entry["source_name"] == "camera.ARW"
    assert entry["frame_index"] == 0 and entry["elapsed_h"] == 1.5
    assert entry["original_annotation"] == annotation()
    assert entry["original_annotation_sha256"] == payload_hash(annotation())
    assert entry["old_gas_fraction"] == entry["new_gas_fraction"] == 0.
    assert entry["aqueous_fraction"] == entry["co2_fraction"] == entry["score"] == 1.
    assert entry["reasons"] == ["model_label_disagreement"]
    assert subject.correction_role(queue, entry)["training_selection"] == "excluded_or_requires_explicit_selection"
    assert report["regions"][0]["scientific_support_verified"] is True
    assert report["regions"][0]["saved_phase_source"] == data.frame.provenance
    assert all(report["prediction_sources"][path] == digest for path, digest in data.frame.pins.items())
    assert report["training_performed"] is report["inference_performed"] is report["scientific_acceptance"] is False
    assert data.frame.native == before_frame.native and data.frame.provenance == before_frame.provenance
    for key in ("labels", "domain", "support"):
        np.testing.assert_array_equal(getattr(data.frame, key), getattr(before_frame, key))
    assert_sources_unchanged(data)


def test_raw_support_hole_preserves_unscored_coverage_and_other_valid_boxes(tmp_path, raw_queue_source):
    data = raw_queue_source
    document = subject.read(data.decisions)
    document["decisions"]["ac01"]["annotations"].append(annotation(id="supported", y=3 / 6, width=1 / 6, height=1 / 6))
    write(data.decisions, document)
    data.before[str(data.decisions)] = data.decisions.read_bytes()
    data.frame.support[0, 0] = False
    support_path = data.source / "saved-support.npz"
    np.savez(support_path, domain=data.frame.domain, support=data.frame.support)
    data.frame.pins[str(support_path.resolve())] = sha256_file(support_path)
    data.before[str(support_path)] = support_path.read_bytes()
    result = subject.build_contradiction_queue(data.model, data.review, data.inputs, tmp_path / "partial")
    queue, report = subject.read(result["queue_path"]), subject.read(result["evaluation_path"])
    assert len(data.calls) == 1  # Same original frame is reused, not re-evaluated by inference.
    assert result["coverage"]["complete"] is False
    assert result["coverage"]["total_effective_regions"] == 2
    assert result["coverage"]["scored_regions"] == result["coverage"]["skipped_regions"] == 1
    assert report["regions"][0]["evaluated_pixels"] == 4
    assert "scientific support" in report["skipped"][0]["reason"]
    assert [entry["id"] for entry in queue["entries"]] == ["ac01:manual:supported"]
    assert queue["phase_review_basis"] == SAVED_NATIVE_VERSION
    assert_sources_unchanged(data)


def test_unavailable_saved_raw_support_never_becomes_agreement_or_optical_fallback(tmp_path, raw_queue_source, monkeypatch):
    from darsia.fluidflower import phase_review_raw_contradictions

    data = raw_queue_source

    def unsupported(*args, **kwargs):
        raise UnsupportedSavedPhaseFrame("original saved scientific support unavailable")

    monkeypatch.setattr(phase_review_raw_contradictions, "load_saved_phase_frame", unsupported)
    result = subject.build_contradiction_queue(data.model, data.review, data.inputs, tmp_path / "unsupported")
    queue = subject.read(result["queue_path"])
    report = subject.read(result["evaluation_path"])
    assert result["entries"] == result["coverage"]["scored_regions"] == 0
    assert result["coverage"]["complete"] is False and result["coverage"]["skipped_regions"] == 1
    assert report["skipped"][0]["reason"] == "original saved scientific support unavailable"
    assert queue["entries"] == [] and queue["coverage"] == result["coverage"]
    assert len(data.queue_calls) == 1
    assert_sources_unchanged(data)


def test_saved_frame_native_mismatch_refuses_publication(tmp_path, raw_queue_source, monkeypatch):
    from darsia.fluidflower import phase_review_raw_contradictions

    data = raw_queue_source
    wrong = deepcopy(data.native)
    wrong["native_origin_xy"] = [0, 0]
    monkeypatch.setattr(phase_review_raw_contradictions, "load_saved_phase_frame",
                        lambda *a, **kw: replace(data.frame, native=wrong))
    output = tmp_path / "wrong-native"
    with pytest.raises(ValueError, match="another native annotation source"):
        subject.build_contradiction_queue(data.model, data.review, data.inputs, output)
    assert not (output / "queue.json").exists()
    assert not data.queue_calls
    assert_sources_unchanged(data)
