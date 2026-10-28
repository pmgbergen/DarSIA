"""Requested proposals use original saved readers/producers and preserve evidence.

Finite synthetic fixtures only; no experiment data is read or changed.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Event, Thread

import numpy as np
import pytest

from darsia.fluidflower import phase_review_proposal_generation as generation
from darsia.fluidflower.phase_saved_assets import asset
from darsia.fluidflower.wasserstein import MASS_ANALYSIS_RESULT_FORMAT


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return asset(path)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    # The original pairwise producer resolves its configured series on import.
    # Bind only this finite fixture, independently of the user's registry.
    series = tmp_path / "configured_series"
    series.mkdir()
    (series / "series.toml").write_text(
        'schema_version = 1\nseries_id = "ffac"\n[paths]\n', encoding="utf8",
    )
    monkeypatch.setenv("DARSIA_FFAC_PROJECT_ROOT", str(series))
    run, candidate, fingerprint, model = "ac01", "saved-candidate", "a" * 64, "saved-model"
    native_root, campaign = tmp_path / "review", tmp_path / "campaign"
    native_path = native_root / "native_frames" / run / "manifest.json"
    classes = tmp_path / "classes.npz"
    np.savez_compressed(classes, rig_domain=np.ones((160, 160), bool), facies=np.ones((160, 160), np.int16))
    frames = []
    for index in range(3):
        fields = {}
        for name, value in zip(generation.PHASE_FIELDS, (.2, .5, .3)):
            path = tmp_path / f"{index}-{name}.npy"
            np.save(path, np.full((160, 160), value, np.float32), allow_pickle=False)
            fields[name] = asset(path)
        frames.append(dict(frame_index=index, elapsed_h=float(index), source_name=f"source-{index}.tif",
            source_sha256=str(index + 1) * 64, rig_domain_revision=dict(classes=asset(classes)),
            phase_probability_fields=dict(model_id=model, candidate_id=candidate, params_sha256=fingerprint,
                source_sha256=str(index + 1) * 64, shape_hw=[160, 160],
                coordinate_space="native_review_content", fields=fields)))
    native = dict(run=run, canvas_wh=[160, 160], content_box_xywh=[0, 0, 160, 160], frame_sources=frames)
    write_json(native_path, native)
    manifest = dict(jobs=[dict(run=run, candidate_id=candidate, params_sha256=fingerprint,
                               review_output_dir=str(native_root))])
    write_json(campaign / "postflight_manifest.json", manifest)
    return SimpleNamespace(root=tmp_path, run=run, candidate=candidate, fingerprint=fingerprint,
        model=model, native=native, native_path=native_path, campaign=campaign, output=tmp_path / "session")


def phase_source(saved):
    return generation.prepare_saved_phase_source(saved.campaign, saved.run, candidate_id=saved.candidate,
        params_sha256=saved.fingerprint, model_id=saved.model)


def test_revocation_after_computation_prevents_replacing_original_pool(saved):
    source = phase_source(saved)
    first = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    path = Path(first["path"])
    before = path.read_bytes()
    guard = generation.ProposalPublicationGuard()
    checked = []

    def revoke_before_commit():
        checked.append(True)
        guard.invalidate()

    with pytest.raises(generation.ProposalRequestCancelled):
        generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer",
            existing_proposals=first["payload"]["proposals"], publication_guard=guard,
            validate_request=revoke_before_commit)
    assert checked == [True]
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


def test_revocation_does_not_wait_for_slow_publication_validation(tmp_path):
    path = tmp_path / "pool.json"
    path.write_bytes(b"original")
    entered, release, revoked = Event(), Event(), Event()
    guard = generation.ProposalPublicationGuard()
    errors = []

    def validate():
        entered.set()
        assert release.wait(5)

    def publish():
        try:
            generation._atomic_pool(path, {"new": True}, b"original",
                publication_guard=guard, validate=validate)
        except BaseException as error:
            errors.append(error)

    def invalidate():
        guard.invalidate()
        revoked.set()

    worker = Thread(target=publish)
    worker.start()
    canceller = None
    try:
        assert entered.wait(5)
        canceller = Thread(target=invalidate)
        canceller.start()
        assert revoked.wait(2), "Cancellation waited for saved-source validation"
    finally:
        release.set()
        worker.join(5)
        if canceller is not None:
            canceller.join(5)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], generation.ProposalRequestCancelled)
    assert path.read_bytes() == b"original"


def test_request_guard_is_checked_under_original_writer_lock(tmp_path):
    path = tmp_path / "pool.json"
    path.write_bytes(b"original")

    def retired():
        with pytest.raises(RuntimeError, match="Another review is saving"):
            with generation._pool_write_lock(path):
                pytest.fail("The publication callback escaped the original writer lock")
        raise PermissionError("The saved review reservation was retired")

    with pytest.raises(PermissionError, match="reservation was retired"):
        generation._atomic_pool(path, {"new": True}, b"original", validate=retired)
    assert path.read_bytes() == b"original"


def test_atomic_pool_rechecks_bytes_after_final_request_check(tmp_path):
    path = tmp_path / "pool.json"
    path.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="Another review changed"):
        generation._atomic_pool(path, {"new": True}, b"original",
            validate=lambda: path.write_bytes(b"independent change"))
    assert path.read_bytes() == b"independent change"


def test_saved_field_changed_during_selection_is_rejected_before_publication(saved, monkeypatch):
    from darsia.fluidflower import phase_validation
    source = phase_source(saved)
    original = phase_validation._canvas_proposals
    descriptor = source["frames"][0]["fields"][generation.PHASE_FIELDS[0]]

    def mutate_after_read(predictions, **kwargs):
        rows = original(predictions, **kwargs)
        # Windows cannot truncate a file while the reader has it memory-mapped.
        # Change its existing payload instead so this exercises the SHA guard.
        with Path(descriptor["path"]).open("r+b") as stream:
            stream.seek(-4, 2)
            stream.write(b"\x00" * 4)
        return rows

    monkeypatch.setattr(phase_validation, "_canvas_proposals", mutate_after_read)
    with pytest.raises(ValueError, match="SHA differs"):
        generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    assert not generation.pool_path(saved.output, saved.run, saved.candidate, saved.fingerprint).exists()




def test_phase_generation_calls_original_producer_without_replacing_old_pool(saved, monkeypatch):
    from darsia.fluidflower import phase_validation
    source = phase_source(saved)
    calls = []
    original = phase_validation._canvas_proposals
    def observed(predictions, **kwargs):
        calls.append((predictions, deepcopy(kwargs)))
        return original(predictions, **kwargs)
    monkeypatch.setattr(phase_validation, "_canvas_proposals", observed)
    answer_file = saved.output / "pairwise_review_decisions.json"
    write_json(answer_file, dict(decisions=dict(ac01=dict(annotations=[dict(id="manual-original")]))))
    before_answers = answer_file.read_bytes()
    result = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    assert result["added_count"] > 0
    assert len(calls) == 1
    assert set(calls[0][0]) == set(generation.PHASE_FIELDS)
    assert calls[0][1]["content_box"] == (0, 0, 160, 160)
    rows = deepcopy(result["payload"]["proposals"])
    expected = original(calls[0][0], **calls[0][1])
    for actual, numerical in zip(rows, expected, strict=True):
        assert {key: actual[key] for key in numerical} == numerical
    next_result = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer", existing_proposals=rows)
    assert next_result["payload"]["proposals"][:len(rows)] == rows
    assert next_result["payload"]["processed_frames"] == [0]
    assert calls[1][1]["existing_regions"]
    assert next_result["payload"]["history"][-1]["person"] == "Reviewer"
    assert answer_file.read_bytes() == before_answers
    assert not (saved.campaign / "generated_phase_quiz").exists()


def test_mass_same_times_and_candidate_are_not_relabelled_as_current_native_source(saved):
    native_before = saved.native_path.read_bytes()
    mass_result = write_json(saved.root / "mass_analysis_result.json", dict(
        format=MASS_ANALYSIS_RESULT_FORMAT, complete=True,
        run_results=[dict(run_id=saved.run, phase_candidate_id=saved.candidate, complete=True,
            target_times_hours=[0., 1., 2.], source_time_indices=[0, 1, 2],
            source_sha256="f" * 64, geometry=dict(shape_rc=[160, 160], source_shape_rc=[160, 160]))]))
    with pytest.raises(generation.ProposalSourceUnavailable, match="no verified bridge"):
        generation.prepare_saved_mass_source(saved.campaign, saved.run, candidate_id=saved.candidate,
            params_sha256=saved.fingerprint, mass_result=mass_result)
    assert saved.native_path.read_bytes() == native_before and not saved.output.exists()


def test_two_gas_scores_cannot_be_substituted_for_three_phase_probabilities(saved):
    del saved.native["frame_sources"][0]["phase_probability_fields"]
    saved.native["frame_sources"][0]["phase_probabilities"] = dict(format="darsia-decision-probabilities-v3")
    write_json(saved.native_path, saved.native)
    with pytest.raises(generation.ProposalSourceUnavailable, match="two gas-decision scores"):
        phase_source(saved)
    assert not saved.output.exists()


def test_handbuilt_mass_source_cannot_bypass_provenance_gate(saved):
    source = phase_source(saved)
    source["format"] = generation.SOURCE_FORMAT
    source.pop("source_sha256")
    source["source_sha256"] = generation._digest(source)
    with pytest.raises(generation.ProposalSourceUnavailable, match="no verified original provenance bridge"):
        generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    assert not saved.output.exists()


def test_changed_field_cannot_overwrite_a_previously_generated_pool(saved):
    source = phase_source(saved)
    result = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    pool = Path(result["path"])
    before = pool.read_bytes()
    changed = Path(source["frames"][0]["fields"][generation.PHASE_FIELDS[0]]["path"])
    np.save(changed, np.zeros((160, 160), np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="SHA differs"):
        generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    assert pool.read_bytes() == before


def test_cancelled_request_preserves_answers_and_previous_pool(saved):
    source = phase_source(saved)
    result = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    path, before = Path(result["path"]), Path(result["path"]).read_bytes()
    with pytest.raises(generation.ProposalRequestCancelled):
        generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer", cancelled=lambda: True)
    assert path.read_bytes() == before


def test_exhaustion_is_reported_without_duplicate_boxes_or_fake_success(saved):
    source = phase_source(saved)
    result = generation.generate_more(source, saved.output, requested_count=1000, actor="Reviewer")
    assert result["exhausted"] is True
    before = Path(result["path"]).read_bytes()
    result = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    assert result["added_count"] == 0 and result["exhausted"] is True
    ids = [row["proposal_id"] for row in result["payload"]["proposals"]]
    assert len(ids) == len(set(ids))
    assert Path(result["path"]).read_bytes() == before


def test_original_loader_overlay_keeps_no_overlay_and_rejects_tamper(saved):
    candidate = SimpleNamespace(run=saved.run, candidate_id=saved.candidate, params_sha256=saved.fingerprint)
    assert generation.load_generated_pool(saved.output, candidate) is None
    source = phase_source(saved)
    result = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    assert generation.load_generated_pool(saved.output, candidate)["proposals"] == result["payload"]["proposals"]
    changed = deepcopy(result["payload"])
    changed["proposals"][0]["x"] = .9
    write_json(Path(result["path"]), changed)
    with pytest.raises(ValueError, match="generation history changed"):
        generation.load_generated_pool(saved.output, candidate)


def test_second_writer_does_not_replace_newer_pool(saved):
    path = saved.output / "proposals.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"newer external pool")
    with pytest.raises(RuntimeError, match="Another review changed"):
        generation._atomic_pool(path, {"test": "requested writer"}, b"older pool")
    assert path.read_bytes() == b"newer external pool"
    assert sorted(p.name for p in path.parent.iterdir()) == [".proposals.json.write.lock", "proposals.json"]


def test_more_boxes_on_same_frame_excludes_original_sixteen_without_new_ids(saved):
    from darsia.fluidflower import phase_validation, phase_learning
    shape, canvas, box = (640, 640), [768, 720], [32, 40, 640, 640]
    frame = saved.native["frame_sources"][0]
    arrays = {}
    for name, value in zip(generation.PHASE_FIELDS, (.2, .5, .3)):
        arrays[name] = np.full(shape, value, np.float32)
        ref = frame["phase_probability_fields"]["fields"][name]
        np.save(ref["path"], arrays[name], allow_pickle=False)
        ref.update(asset(ref["path"]))
    classes = Path(frame["rig_domain_revision"]["classes"]["path"])
    np.savez_compressed(classes, rig_domain=np.ones((720, 768), bool), facies=np.ones((720, 768), np.int16))
    frame["rig_domain_revision"]["classes"] = asset(classes)
    frame["phase_probability_fields"]["shape_hw"] = list(shape)
    saved.native.update(canvas_wh=canvas, content_box_xywh=box, frame_sources=[frame])
    write_json(saved.native_path, saved.native)
    kwargs = dict(run_id=saved.run, frame_index=0, elapsed_h=0., model_id=saved.model,
        candidate_id=saved.candidate, fingerprint=saved.fingerprint, canvas_shape=(720, 768),
        content_box=tuple(box), eligible_mask=np.ones(shape, bool), facies_labels=np.ones(shape, np.int16))
    old = phase_validation._canvas_proposals(arrays, **kwargs)
    assert len(old) == 16
    # Explicit new defaults preserve the former fixed-count result exactly.
    assert phase_validation._canvas_proposals(arrays, count=16, existing_regions=(), **kwargs) == old
    source = phase_source(saved)
    result = generation.generate_more(source, saved.output, requested_count=2, actor="Reviewer", existing_proposals=old)
    rows = result["payload"]["proposals"]
    assert rows[:16] == old and result["added_count"] == 2
    assert all(row["frame_index"] == 0 for row in rows)
    local = generation._local_existing_regions(old, 0, canvas, box)
    expected = phase_learning.propose_uncertain_phase_regions(arrays, run_id=saved.run,
        frame_index=0, elapsed_h=0., model_id=saved.model, count=2,
        eligible_mask=kwargs["eligible_mask"], facies_labels=kwargs["facies_labels"], existing_regions=local)
    assert [row["proposal_id"] for row in rows[16:]] == [row["proposal_id"] for row in expected]
    assert not set(row["proposal_id"] for row in old) & set(row["proposal_id"] for row in rows[16:])
    for actual, previous in zip(local, old, strict=True):
        pixels = previous["bbox_xywh_px"]
        assert list(actual.values()) == [value / 640 for value in pixels]


def test_exclusions_reject_inconsistent_canvas_and_original_pixel_geometry():
    row = dict(frame_index=2, coordinate_space="normalized_native_review_canvas_16x9",
               x=.2, y=.2, width=.3, height=.3, bbox_xywh_px=[0, 0, 30, 30])
    with pytest.raises(generation.ProposalSourceUnavailable, match="differs from its original"):
        generation._local_existing_regions([row], 2, [100, 100], [0, 0, 100, 100])


def test_cancel_just_before_publication_does_not_write_a_pool(saved, monkeypatch):
    source = phase_source(saved)
    original = generation._digest
    cancelled = False
    def observe(value):
        nonlocal cancelled
        digest = original(value)
        if isinstance(value, dict) and value.get("format") == generation.POOL_FORMAT:
            cancelled = True
        return digest
    monkeypatch.setattr(generation, "_digest", observe)
    with pytest.raises(generation.ProposalRequestCancelled):
        generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer", cancelled=lambda: cancelled)
    assert not saved.output.exists()


def test_historical_pool_survives_producer_update_but_append_is_rejected(saved, monkeypatch):
    source = phase_source(saved)
    generated = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    before = Path(generated["path"]).read_bytes()
    candidate = SimpleNamespace(run=saved.run, candidate_id=saved.candidate, params_sha256=saved.fingerprint)
    current_file = generation._file
    def changed_producer(path):
        ref = current_file(path)
        if str(path).endswith("phase_validation.py"):
            ref = dict(ref, sha256="f" * 64)
        return ref
    monkeypatch.setattr(generation, "_file", changed_producer)
    assert generation.load_generated_pool(saved.output, candidate)["proposals"] == generated["payload"]["proposals"]
    with pytest.raises(generation.ProposalSourceUnavailable, match="producer changed"):
        generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    assert Path(generated["path"]).read_bytes() == before


def test_historical_pool_does_not_read_old_scientific_assets_or_code(saved, monkeypatch):
    source = phase_source(saved)
    generated = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    candidate = SimpleNamespace(run=saved.run, candidate_id=saved.candidate, params_sha256=saved.fingerprint)
    def forbidden():
        raise AssertionError("Historical answer loading must not open a scientific source reader")
    monkeypatch.setattr(generation, "_reader", forbidden)
    assert generation.load_generated_pool(saved.output, candidate)["content_sha256"] == generated["payload"]["content_sha256"]


def test_original_loader_recognizes_generated_ids_and_keeps_its_publication_gate(saved, monkeypatch):
    from darsia.fluidflower.tools import calibration_pairwise_review as original
    source = phase_source(saved)
    generated = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    candidate = SimpleNamespace(run=saved.run, candidate_id=saved.candidate, params_sha256=saved.fingerprint)
    monkeypatch.setattr(original, "_candidate_prediction_withheld", lambda *args: False)
    loaded = original._load_phase_quiz_payload(saved.output, candidate, "B")
    assert loaded["proposals"] == [dict(row, candidate="B") for row in generated["payload"]["proposals"]]
    monkeypatch.setattr(original, "_candidate_prediction_withheld", lambda *args: True)
    assert original._load_phase_quiz_payload(saved.output, candidate, "B") == {"proposals": [], "initial_proposal_count": 0}


def test_source_owned_publication_hold_blocks_preparation_and_append(saved, monkeypatch):
    from darsia.fluidflower import phase_blind_review_policy as policy
    source = phase_source(saved)
    generated = generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    before = Path(generated["path"]).read_bytes()
    def withheld(*args):
        raise PermissionError("Exact source candidate is withheld")
    monkeypatch.setattr(policy, "require_predictions_visible", withheld)
    with pytest.raises(PermissionError, match="withheld"):
        phase_source(saved)
    with pytest.raises(PermissionError, match="withheld"):
        generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    assert Path(generated["path"]).read_bytes() == before


def test_original_explicit_targeted_pool_keeps_precedence_over_generated_overlay(saved, monkeypatch):
    from darsia.fluidflower.tools import calibration_pairwise_review as original, phase_targeted_quiz
    source = phase_source(saved)
    generation.generate_more(source, saved.output, requested_count=1, actor="Reviewer")
    candidate = SimpleNamespace(run=saved.run, candidate_id=saved.candidate, params_sha256=saved.fingerprint,
                                phase_quiz_proposals_path=Path("explicit-historical-target.json"))
    targeted = {"proposals": [{"proposal_id": "targeted"}], "initial_proposal_count": 1}
    monkeypatch.setattr(original, "_candidate_prediction_withheld", lambda *args: False)
    monkeypatch.setattr(phase_targeted_quiz, "load_proposals", lambda *args: targeted)
    assert original._load_phase_quiz_payload(saved.output, candidate, "A") is targeted


def test_persistent_unowned_lock_file_allows_next_publication(tmp_path):
    path = tmp_path / "proposals.json"
    lock = path.with_name(".proposals.json.write.lock")
    lock.write_bytes(b"0")  # A prior process left metadata, not OS ownership.
    generation._atomic_pool(path, {"batch": 1}, None)
    previous = path.read_bytes()
    generation._atomic_pool(path, {"batch": 2}, previous)
    assert json.loads(path.read_text()) == {"batch": 2}
    assert lock.exists()


def test_live_os_lock_rejects_competing_writer_without_mutating_pool(tmp_path):
    path = tmp_path / "proposals.json"
    path.write_bytes(b"original")
    with generation._pool_write_lock(path):
        with pytest.raises(RuntimeError, match="Another review is saving"):
            generation._atomic_pool(path, {"batch": 2}, b"original")
    assert path.read_bytes() == b"original"
    generation._atomic_pool(path, {"batch": 3}, b"original")
    assert json.loads(path.read_text()) == {"batch": 3}
