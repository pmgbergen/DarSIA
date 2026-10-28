"""Saved categorical pool mechanics on finite local bytes, authored unrun.

Only current model admission is replaced: the fixture delegates real historical
source validation and exact asset hashing, but cannot prove production model
admission. The real saved-frame decoder, selector, pool readers, OS locks and
atomic publication remain in use. No other test module supplies fixtures.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import phase_review_categorical_proposals as selector
from darsia.fluidflower import phase_review_proposal_generation as generation
from darsia.fluidflower import phase_review_saved_categories as categories
from darsia.fluidflower.phase_saved_assets import VerifiedAssetReader, asset


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return asset(path)


def _seal(value, key):
    value.pop(key, None)
    value[key] = generation._digest(value)
    return value


def _row(identity, *, x, y):
    # Original-schema retained geometry; no selector or prior model execution is
    # claimed by this manually written historical probability-pool fixture.
    nx, ny, nw, nh = (8 + x) / 32, (1 + y) / 18, 2 / 32, 2 / 18
    return dict(proposal_id=identity, run_id="ac01", frame_index=0, elapsed_h=0.,
                model_id="saved-model", candidate_id="saved-candidate", candidate="A",
                params_sha256="a" * 64, coordinate_space=selector.COORDINATE_SPACE,
                shape="box", bbox_xywh_px=[x, y, 2, 2], x=nx, y=ny, width=nw, height=nh,
                points=[[nx, ny], [nx + nw, ny], [nx + nw, ny + nh], [nx, ny + nh]],
                model_suggestion_hidden_until_answer=True)


@pytest.fixture
def saved_pool(tmp_path, monkeypatch):
    series = tmp_path / "configured_series"
    series.mkdir()
    (series / "series.toml").write_text(
        'schema_version = 1\nseries_id = "ffac"\n[paths]\n', encoding="utf-8")
    monkeypatch.setenv("DARSIA_FFAC_PROJECT_ROOT", str(series))
    package = tmp_path / "saved"
    package.mkdir()
    document = _json(package / "fixture_metadata.json", {"finite_pool_fixture": True})
    frames, probability_frames = [], []
    for index in range(3):
        labels = np.full((18, 32), 255, np.uint8)
        labels[1:17, 8:16] = 0
        labels[1:17, 16:24] = 1 if index != 2 else 2
        domain = labels != 255
        label_path = package / f"labels-{index}.png"
        Image.fromarray(labels).save(label_path)
        classes_path = package / f"classes-{index}.npz"
        np.savez_compressed(classes_path, rig_domain=domain, modelled=domain,
                            filled_observed=np.zeros_like(domain),
                            filled_unobserved=np.zeros_like(domain),
                            facies=np.ones(labels.shape, np.int16))
        source_sha = f"{index + 1:064x}"
        rule_assets = {}
        for name in categories.LEGACY_MASKS:
            changed = np.zeros_like(domain)
            if name == "floor_connectivity_masks":
                changed[2, 9] = True
            path = package / f"{index}-{name}.npy"
            np.save(path, np.packbits(changed, bitorder="big"), allow_pickle=False)
            rule_assets[name] = dict(asset(path), storage="packbits-npy",
                shape_hw=[18, 32], frame_index=index, source_sha256=source_sha)
        frames.append(dict(frame_index=index, elapsed_h=float(index),
            source_sha256=source_sha, source_name=f"finite-frame-{index}.png",
            phase_shape_hw=[18, 32], phase_receipt=document, labels=asset(label_path),
            classes=asset(classes_path), displayed_map=asset(label_path),
            rule_storage="legacy-big-packbits-npy", rule_assets=rule_assets,
            rule_coverage=categories._coverage(categories.LEGACY_MASKS, (), "recorded_final3_stage_masks")))
        fields = {}
        for name, probability in zip(generation.PHASE_FIELDS, (.2, .5, .3)):
            path = package / f"{index}-{name}.npy"
            np.save(path, np.full((16, 16), probability, np.float32), allow_pickle=False)
            fields[name] = asset(path)
        probability_frames.append(dict(frame_index=index, elapsed_h=float(index),
            source_sha256=source_sha, source_name=f"finite-frame-{index}.png",
            classes=asset(classes_path), fields=fields))
    module_root = Path(categories.__file__).parent
    source = dict(format=categories.SOURCE_FORMAT, adapter_kind="p1_final3", run="ac01",
        candidate_id="saved-candidate", params_sha256="a" * 64, model_id="saved-model",
        model_identity=["saved-model", "b" * 64], model_root=str(package),
        campaign=document, native_manifest=document, registration=document, proof_assets=[],
        canvas_wh=[32, 18], content_box_xywh=[8, 1, 16, 16], frames=frames,
        publication_roots=[str(package)], scientific_maps_recomputed=False,
        source_arrays_substituted=False, producer=dict(
            module="darsia.fluidflower.phase_review_saved_categories", function="load_frame", schema=1,
            code={name: asset(module_root / name) for name in categories._CODE_FILES}))
    _seal(source, "source_sha256")
    probability_source = dict(format=generation.PHASE_SOURCE_FORMAT, run=source["run"],
        candidate_id=source["candidate_id"], params_sha256=source["params_sha256"],
        model_id=source["model_id"], campaign=document, native_manifest=document,
        canvas_wh=source["canvas_wh"], content_box_xywh=source["content_box_xywh"],
        frames=probability_frames, publication_roots=[str(package)],
        scientific_maps_recomputed=False, source_arrays_substituted=False,
        producer=dict(module="darsia.fluidflower.phase_validation", function="_canvas_proposals",
            proposal_schema=2, source=asset(module_root / "phase_validation.py"),
            algorithm_source=asset(module_root / "phase_learning.py")))
    _seal(probability_source, "source_sha256")
    original_validation = categories.validate_source
    admission_calls = []

    def pool_only_admission(value, reader=None, *, require_current_assets=True):
        # Deliberate narrow seam: original saved model/candidate admission has
        # its own tests. Keep actual recorded-shape/digest/role checks and byte
        # freshness here rather than replacing decoder or selector results.
        original_validation(value, reader=reader, require_current_assets=False)
        if require_current_assets:
            admission_calls.append(value["source_sha256"])
            fresh = VerifiedAssetReader()
            for descriptor in categories._references(value):
                fresh.path(descriptor)

    monkeypatch.setattr(categories, "validate_source", pool_only_admission)
    output = tmp_path / "review_session"
    output.mkdir()
    answers = output / "decisions.json"
    answers.write_bytes(b'{"human_answers": ["unchanged"]}\n')
    candidate = SimpleNamespace(run=source["run"], candidate_id=source["candidate_id"],
                                params_sha256=source["params_sha256"])
    return SimpleNamespace(source=source, probability_source=probability_source, output=output,
        candidate=candidate, answers=answers, admission_calls=admission_calls)


def _prior(saved):
    source = saved.probability_source
    rows = [_row("phase-active-original", x=0, y=0), _row("phase-active-earlier-added", x=14, y=14)]
    payload = dict(format=generation.POOL_FORMAT, run=source["run"], candidate_id=source["candidate_id"],
        params_sha256=source["params_sha256"], native_manifest_sha256=source["native_manifest"]["sha256"],
        source_sha256=source["source_sha256"], source=deepcopy(source), proposals=rows,
        initial_proposal_count=1, processed_frames=[0], exhausted_frames=[], exhausted=False,
        history=[dict(action="requested more blind validation boxes", person="Reviewer", added_count=1)],
        scientific_maps_recomputed=False, source_arrays_substituted=False)
    _seal(payload, "content_sha256")
    path = generation.pool_path(saved.output, source["run"], source["candidate_id"], source["params_sha256"])
    _json(path, payload)
    return path, payload, path.read_bytes()


def _more(saved, **changes):
    arguments = dict(requested_count=1, actor="Reviewer")
    arguments.update(changes)
    return generation.generate_more(saved.source, saved.output, **arguments)


def _categorical_path(saved):
    source = saved.source
    return generation.categorical_pool_path(saved.output, source["run"], source["candidate_id"], source["params_sha256"])


def test_probability_migration_keeps_exact_prefix_initial_count_and_original_bytes(saved_pool):
    saved = saved_pool
    prior_path, prior, prior_bytes = _prior(saved)
    answer_bytes = saved.answers.read_bytes()
    first = _more(saved, existing_proposals=prior["proposals"], initial_proposal_count=1)
    payload = first["payload"]
    assert first["added_count"] == 1
    assert Path(first["path"]) != prior_path
    assert payload["proposals"][:2] == prior["proposals"]
    assert payload["retained_prefix_count"] == 2
    assert payload["retained_prefix_sha256"] == generation._digest(prior["proposals"])
    assert payload["initial_proposal_count"] == 1
    assert payload["prior_probability_pool"] == dict(
        sha256=hashlib.sha256(prior_bytes).hexdigest(), source_sha256=prior["source_sha256"])
    new = payload["proposals"][2]
    assert new["bbox_xywh_px"] == [0, 0, 16, 16]
    assert [new[key] for key in ("x", "y", "width", "height")] == [8 / 32, 1 / 18, 16 / 32, 16 / 18]
    assert new["selection_basis"] == "saved_final_phase_boundaries_and_rule_changes"
    assert prior_path.read_bytes() == prior_bytes
    assert saved.answers.read_bytes() == answer_bytes
    second = _more(saved, existing_proposals=payload["proposals"], initial_proposal_count=1)
    assert second["added_count"] == 1
    assert second["payload"]["proposals"][:len(payload["proposals"])] == payload["proposals"]
    assert second["payload"]["initial_proposal_count"] == 1
    assert second["payload"]["retained_prefix_count"] == 2
    assert prior_path.read_bytes() == prior_bytes
    assert saved.answers.read_bytes() == answer_bytes


@pytest.mark.parametrize("bad", ["prefix", "initial_count"])
def test_migration_refuses_changed_live_prefix_or_initial_offering(saved_pool, bad):
    prior_path, prior, before = _prior(saved_pool)
    rows = deepcopy(prior["proposals"])
    initial = 1
    if bad == "prefix": rows[0]["proposal_id"] = "changed-live-original"
    else: initial = 2
    with pytest.raises(ValueError, match="prefix|initial offered"):
        _more(saved_pool, existing_proposals=rows, initial_proposal_count=initial)
    assert prior_path.read_bytes() == before
    assert not _categorical_path(saved_pool).exists()


def test_historical_reload_needs_no_scientific_arrays_or_current_producer_code(saved_pool, monkeypatch):
    saved = saved_pool
    _path, prior, _bytes = _prior(saved)
    result = _more(saved, existing_proposals=prior["proposals"], initial_proposal_count=1)
    Path(saved.source["frames"][0]["labels"]["path"]).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical pool loading must not access scientific arrays or current code")

    monkeypatch.setattr(generation, "_reader", forbidden)
    monkeypatch.setattr(generation, "_file", forbidden)
    monkeypatch.setattr(generation, "_phase_producer", forbidden)
    monkeypatch.setattr(categories, "_decode_frame", forbidden)
    assert generation.load_generated_pool(saved.output, saved.candidate) == result["payload"]


def test_retained_prefix_tamper_is_refused_even_with_recomputed_outer_digest(saved_pool):
    _path, prior, _bytes = _prior(saved_pool)
    result = _more(saved_pool, existing_proposals=prior["proposals"], initial_proposal_count=1)
    path = Path(result["path"])
    modified = deepcopy(result["payload"])
    modified["proposals"][0]["proposal_id"] = "replaced-original"
    _json(path, _seal(modified, "content_sha256"))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unchanged original proposal prefix"):
        generation.load_generated_pool(saved_pool.output, saved_pool.candidate)
    assert path.read_bytes() == before


def test_append_rejects_changed_embedded_source_even_with_recomputed_pool_digest(saved_pool):
    _path, prior, _bytes = _prior(saved_pool)
    result = _more(saved_pool, existing_proposals=prior["proposals"], initial_proposal_count=1)
    path = Path(result["path"])
    modified = deepcopy(result["payload"])
    modified["source"]["model_id"] = "changed-embedded-model"
    _json(path, _seal(modified, "content_sha256"))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        _more(saved_pool, existing_proposals=result["payload"]["proposals"], initial_proposal_count=1)
    assert path.read_bytes() == before


@pytest.mark.parametrize("invalid_count", [-1, True, "1", 3])
def test_migration_refuses_invalid_recorded_initial_count_when_caller_omits_it(saved_pool, invalid_count):
    prior_path, prior, _bytes = _prior(saved_pool)
    prior["initial_proposal_count"] = invalid_count
    _json(prior_path, _seal(prior, "content_sha256"))
    before = prior_path.read_bytes()
    with pytest.raises(ValueError):
        _more(saved_pool, existing_proposals=prior["proposals"])
    assert prior_path.read_bytes() == before
    assert not _categorical_path(saved_pool).exists()


def test_probability_history_change_conflicts_without_replacing_categorical_pool(saved_pool):
    prior_path, prior, _bytes = _prior(saved_pool)
    result = _more(saved_pool, existing_proposals=prior["proposals"], initial_proposal_count=1)
    path = Path(result["path"])
    before = path.read_bytes()
    prior["history"].append(dict(action="requested more blind validation boxes", person="Reviewer", added_count=0))
    _json(prior_path, _seal(prior, "content_sha256"))
    changed_prior = prior_path.read_bytes()
    with pytest.raises(ValueError, match="original probability pool changed"):
        generation.load_generated_pool(saved_pool.output, saved_pool.candidate)
    with pytest.raises(ValueError, match="original probability pool changed"):
        _more(saved_pool, existing_proposals=result["payload"]["proposals"], initial_proposal_count=1)
    assert path.read_bytes() == before
    assert prior_path.read_bytes() == changed_prior


@pytest.mark.parametrize("namespace", ["categorical", "probability"])
def test_either_live_os_writer_lock_blocks_categorical_publication(saved_pool, namespace):
    prior_path, prior, before = _prior(saved_pool)
    destination = _categorical_path(saved_pool)
    held = destination if namespace == "categorical" else prior_path
    with generation._pool_write_lock(held):
        with pytest.raises(RuntimeError, match="Another review is saving"):
            _more(saved_pool, existing_proposals=prior["proposals"], initial_proposal_count=1)
    assert prior_path.read_bytes() == before
    assert not destination.exists()
    assert not list(destination.parent.glob(".*.tmp"))


def test_both_namespace_locks_are_held_during_last_validation_and_revocation(saved_pool):
    prior_path, prior, before = _prior(saved_pool)
    destination = _categorical_path(saved_pool)
    guard = generation.ProposalPublicationGuard()
    checked = []

    def retire():
        for path in (prior_path, destination):
            with pytest.raises(RuntimeError, match="Another review is saving"):
                with generation._pool_write_lock(path):
                    pytest.fail("Final validation escaped a required pool writer lock")
            checked.append(path)
        guard.invalidate()

    with pytest.raises(generation.ProposalRequestCancelled):
        _more(saved_pool, existing_proposals=prior["proposals"], initial_proposal_count=1,
              publication_guard=guard, validate_request=retire)
    assert checked == [prior_path, destination]
    assert prior_path.read_bytes() == before
    assert not destination.exists()
    assert not list(destination.parent.glob(".*.tmp"))


def test_one_current_source_pass_occurs_after_selection_under_both_writer_locks(saved_pool, monkeypatch):
    prior_path, prior, before = _prior(saved_pool)
    destination = _categorical_path(saved_pool)
    original_validation = categories.validate_source
    original_selection = selector.propose_saved_category_regions
    selected = []
    validations = []

    def select(labels, **kwargs):
        result = original_selection(labels, **kwargs)
        selected.append(kwargs["frame_index"])
        return result

    def validate(source, reader=None, *, require_current_assets=True):
        if require_current_assets:
            assert selected, "Current source admission ran as a redundant pre-selection pass"
            for path in (prior_path, destination):
                with pytest.raises(RuntimeError, match="Another review is saving"):
                    with generation._pool_write_lock(path):
                        pytest.fail("Current source check escaped the publication locks")
            validations.append(source["source_sha256"])
        return original_validation(source, reader=reader, require_current_assets=require_current_assets)

    monkeypatch.setattr(selector, "propose_saved_category_regions", select)
    monkeypatch.setattr(categories, "validate_source", validate)
    result = _more(saved_pool, existing_proposals=prior["proposals"], initial_proposal_count=1)
    assert result["added_count"] == 1 and selected == [0]
    assert validations == [saved_pool.source["source_sha256"]]
    assert saved_pool.admission_calls == validations
    assert prior_path.read_bytes() == before


def test_old_probability_pool_changed_during_generation_cannot_be_published_as_prefix(saved_pool):
    prior_path, prior, _bytes = _prior(saved_pool)
    first = _more(saved_pool, existing_proposals=prior["proposals"], initial_proposal_count=1)
    destination = Path(first["path"])
    before = destination.read_bytes()
    changed = dict(prior, history=prior["history"] + [dict(action="another retained request", person="Reviewer")])
    changed_bytes = (json.dumps(_seal(changed, "content_sha256"), indent=2) + "\n").encode("utf-8")

    def change_dependency():
        # Simulates an uncooperative byte change after computation. Compliant
        # probability writers are separately excluded by the real lock tests.
        prior_path.write_bytes(changed_bytes)

    with pytest.raises(RuntimeError, match="probability pool changed during generation"):
        _more(saved_pool, existing_proposals=first["payload"]["proposals"],
              initial_proposal_count=1, validate_request=change_dependency)
    assert destination.read_bytes() == before
    assert prior_path.read_bytes() == changed_bytes


def test_actual_saved_map_mutation_after_selection_prevents_publication(saved_pool, monkeypatch):
    _prior_path, prior, _bytes = _prior(saved_pool)
    first = _more(saved_pool, existing_proposals=prior["proposals"], initial_proposal_count=1)
    destination = Path(first["path"])
    before = destination.read_bytes()
    original = selector.propose_saved_category_regions
    called = []

    def changed_after_selection(labels, **kwargs):
        result = original(labels, **kwargs)
        if result and not called:
            frame = saved_pool.source["frames"][kwargs["frame_index"]]
            Path(frame["labels"]["path"]).write_bytes(b"changed actual saved PNG")
            called.append(frame["frame_index"])
        return result

    monkeypatch.setattr(selector, "propose_saved_category_regions", changed_after_selection)
    with pytest.raises(ValueError, match="SHA differs"):
        _more(saved_pool, existing_proposals=first["payload"]["proposals"], initial_proposal_count=1)
    assert called
    assert destination.read_bytes() == before


def test_stale_source_digest_fails_before_decode_or_pool_publication(saved_pool, monkeypatch):
    source = deepcopy(saved_pool.source)
    source["model_id"] = "changed-model"

    def forbidden(*args, **kwargs):
        raise AssertionError("Stale source reached saved array decoding")

    monkeypatch.setattr(categories, "_decode_frame", forbidden)
    with pytest.raises(generation.ProposalSourceUnavailable, match="identity changed"):
        generation.generate_more(source, saved_pool.output, requested_count=1, actor="Reviewer")
    assert not _categorical_path(saved_pool).exists()


def test_exhaustion_keeps_history_and_never_falls_back_to_probabilities(saved_pool, monkeypatch):
    prior_path, prior, prior_bytes = _prior(saved_pool)

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved categorical generation attempted a probability or mass fallback")

    for name in ("_phase_producer", "_probability_rows", "prepare_saved_phase_source", "prepare_saved_mass_source"):
        monkeypatch.setattr(generation, name, forbidden)
    result = _more(saved_pool, requested_count=100, existing_proposals=prior["proposals"], initial_proposal_count=1)
    assert result["added_count"] == 3
    assert result["exhausted"] is True
    assert result["payload"]["exhausted_frames"] == [0, 1, 2]
    assert result["payload"]["initial_proposal_count"] == 1
    path = Path(result["path"])
    before = path.read_bytes()
    monkeypatch.setattr(categories, "_decode_frame", forbidden)
    again = _more(saved_pool, existing_proposals=result["payload"]["proposals"], initial_proposal_count=1)
    assert again["added_count"] == 0 and again["exhausted"] is True
    assert again["payload"] == result["payload"]
    assert path.read_bytes() == before
    assert prior_path.read_bytes() == prior_bytes


def test_first_new_pool_uses_original_initial_offering_limit_and_keeps_it(saved_pool):
    original_limit = generation._producer().PHASE_QUIZ_INITIAL_PROPOSALS_PER_CANDIDATE
    first = _more(saved_pool, requested_count=1)
    assert first["payload"]["initial_proposal_count"] == min(1, original_limit)
    second = _more(saved_pool, requested_count=100, existing_proposals=first["payload"]["proposals"])
    assert second["payload"]["initial_proposal_count"] == first["payload"]["initial_proposal_count"]
    assert second["payload"]["proposals"][:1] == first["payload"]["proposals"]
