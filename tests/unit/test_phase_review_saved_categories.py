"""Finite saved-category reader contracts; authored UNRUN.

The fixtures contain synthetic pixels and receipts. They exercise the actual
decoder, historical source validation and SHA reader. Tests that stub _prepare
or model identity say so explicitly: they do not establish successful original
43-run P1-final3 or selected-camera package admission.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import phase_review_saved_categories as saved
from darsia.fluidflower import phase_review_proposal_generation as generation
from darsia.fluidflower.phase_saved_assets import VerifiedAssetReader, asset


def _seal(source):
    source.pop("source_sha256", None)
    source["source_sha256"] = generation._digest(source)
    return source


@pytest.fixture(params=["p1_final3", "selected_camera"])
def frame_source(tmp_path, request, monkeypatch):
    # Publication-policy behavior has separate original-policy tests. This
    # fixture isolates exact byte decoding and does not claim current admission.
    monkeypatch.setattr(saved, "_publication", lambda roots, run: None)
    labels = np.full((5, 7), 255, np.uint8)
    labels[1:4, 1:6] = np.array([[0, 0, 1, 2, 2]] * 3, np.uint8)
    domain = labels != 255
    changed = np.zeros_like(domain)
    changed[2, 2] = True
    png = tmp_path / "final.png"
    Image.fromarray(labels).save(png)
    classes = tmp_path / "classes.npz"
    np.savez_compressed(classes, rig_domain=domain, modelled=domain,
        filled_observed=np.zeros_like(domain), filled_unobserved=np.zeros_like(domain))
    document = tmp_path / "receipt.json"
    document.write_text(json.dumps({"synthetic_fixture": True}), encoding="utf8")
    reference = asset(document)
    frame = dict(frame_index=0, elapsed_h=2.5, source_sha256="d" * 64,
        source_name="synthetic-camera-frame", phase_shape_hw=[5, 7],
        phase_receipt=reference, labels=asset(png), classes=asset(classes), displayed_map=asset(png))
    kind = request.param
    if kind == "p1_final3":
        mask = tmp_path / "rule.npy"
        np.save(mask, np.packbits(changed, bitorder="big"), allow_pickle=False)
        names = ["floor_connectivity_masks"]
        frame.update(rule_storage="legacy-big-packbits-npy", rule_assets={names[0]: dict(
            asset(mask), storage="packbits-npy", shape_hw=[5, 7], frame_index=0, source_sha256="d" * 64)},
            rule_coverage=saved._coverage(names, [name for name in saved.LEGACY_MASKS if name not in names],
                                         "recorded_final3_stage_masks"))
    else:
        mask = tmp_path / "rules.npz"
        names = ["post_B", "seam_affected_gas_graph"]
        np.savez_compressed(mask, shape_hw=np.array([5, 7], np.int64),
            post_B=np.packbits(changed, bitorder="little"),
            seam_affected_gas_graph=np.packbits(domain, bitorder="little"))
        frame.update(source_work_index=17, phase_job_sha256="e" * 64, rule_storage="selected-little-packbits-npz",
            operator_masks=asset(mask), operator_mask_names=names, rule_names=["post_B"],
            rule_coverage=saved._coverage(["post_B"], [name for name in saved.SELECTED_CHANGES if name != "post_B"],
                                         "recorded_selected_stage_masks"))
    source = dict(format=saved.SOURCE_FORMAT, adapter_kind=kind, run="ac01",
        candidate_id="saved-candidate", params_sha256="a" * 64, model_id="saved-model",
        model_identity=["saved-model", "b" * 64], model_root=str(tmp_path), candidate_job={},
        campaign=reference, registration=reference, native_manifest=reference, proof_assets=[reference],
        canvas_wh=[7, 4], content_box_xywh=[1, 1, 5, 3], frames=[frame], publication_roots=[str(tmp_path)],
        scientific_maps_recomputed=False, source_arrays_substituted=False,
        producer=dict(module="darsia.fluidflower.phase_review_saved_categories", function="load_frame", schema=1,
            code={name: asset(Path(saved.__file__).parent / name) for name in saved._CODE_FILES}))
    return _seal(source), labels, changed


def test_original_pack_order_exact_crop_and_immutable_final_categories(frame_source):
    source, labels, changed = frame_source
    saved.validate_source(source, require_current_assets=False)
    frame = source["frames"][0]
    result = saved.load_frame(source, frame)
    assert np.array_equal(result["labels"], labels[1:4, 1:6])
    assert result["labels"].dtype == np.uint8 and not result["labels"].flags.writeable
    assert result["domain"].all() and not result["domain"].flags.writeable
    assert len(result["rule_masks"]) == 1
    assert np.array_equal(next(iter(result["rule_masks"].values())), changed[1:4, 1:6])
    assert not next(iter(result["rule_masks"].values())).flags.writeable
    assert result["facies"] is None
    assert result["rule_coverage"]["available"] is True
    assert result["rule_coverage"]["complete"] is False
    assert result["rule_coverage"]["net_final_change_claimed"] is False


def test_genuinely_absent_rule_archive_retains_boundaries_with_explicit_coverage(frame_source):
    source, _labels, _changed = frame_source
    frame = source["frames"][0]
    if source["adapter_kind"] == "p1_final3":
        frame["rule_assets"] = {}
        frame["rule_coverage"] = saved._coverage([], saved.LEGACY_MASKS, "recorded_final3_stage_masks")
    else:
        frame.update(operator_masks=None, operator_mask_names=[], rule_names=[])
        frame["rule_coverage"] = saved._coverage([], saved.SELECTED_CHANGES, "recorded_selected_stage_masks")
    _seal(source)
    result = saved.load_frame(source, frame)
    assert result["rule_masks"] == {} and result["domain"].all()
    assert result["rule_coverage"]["available"] is False and result["rule_coverage"]["missing"]


@pytest.mark.parametrize("change", ["rgba", "unknown_code", "wrong_shape"])
def test_final_labels_cannot_be_reconstructed_from_display_or_another_grid(frame_source, change):
    source, labels, _changed = frame_source
    frame = source["frames"][0]
    path = Path(frame["labels"]["path"])
    if change == "rgba":
        Image.fromarray(np.zeros((*labels.shape, 4), np.uint8)).save(path)
    elif change == "unknown_code":
        labels[1, 1] = 3
        Image.fromarray(labels).save(path)
    else:
        Image.fromarray(labels[:-1]).save(path)
    frame["labels"] = asset(path)
    with pytest.raises(generation.ProposalSourceUnavailable, match="L-mode|encoding"):
        saved._decode_frame(source, frame, VerifiedAssetReader())


@pytest.mark.parametrize("change", ["integer_domain", "overlapping_classes", "different_domain", "missing_class"])
def test_final_class_partition_is_exact_not_an_inferred_validity_mask(frame_source, change):
    source, labels, _changed = frame_source
    frame = source["frames"][0]
    domain = labels != 255
    fields = dict(rig_domain=domain, modelled=domain.copy(),
        filled_observed=np.zeros_like(domain), filled_unobserved=np.zeros_like(domain))
    if change == "integer_domain": fields["rig_domain"] = domain.astype(np.uint8)
    elif change == "overlapping_classes": fields["filled_observed"][1, 1] = True
    elif change == "different_domain":
        fields["rig_domain"] = domain.copy()
        fields["rig_domain"][1, 1] = False
        fields["modelled"][1, 1] = False
    else: fields.pop("filled_unobserved")
    path = Path(frame["classes"]["path"])
    np.savez_compressed(path, **fields)
    frame["classes"] = asset(path)
    with pytest.raises(generation.ProposalSourceUnavailable, match="partition|bool arrays"):
        saved._decode_frame(source, frame, VerifiedAssetReader())


@pytest.mark.parametrize("change", ["padding", "dtype", "length"])
def test_declared_packed_masks_refuse_corrupt_storage(frame_source, change):
    source, _labels, changed = frame_source
    frame = source["frames"][0]
    order = "big" if source["adapter_kind"] == "p1_final3" else "little"
    packed = np.packbits(changed, bitorder=order)
    if change == "padding": packed[-1] |= np.uint8(1 if order == "big" else 128)
    elif change == "dtype": packed = packed.astype(np.int16)
    else: packed = packed[:-1]
    if order == "big":
        reference = next(iter(frame["rule_assets"].values()))
        path = Path(reference["path"])
        np.save(path, packed, allow_pickle=False)
        reference.update(asset(path))
    else:
        path = Path(frame["operator_masks"]["path"])
        np.savez_compressed(path, shape_hw=np.array([5, 7], np.int64), post_B=packed,
            seam_affected_gas_graph=np.packbits(changed, bitorder="little"))
        frame["operator_masks"] = asset(path)
    with pytest.raises(generation.ProposalSourceUnavailable, match="packed"):
        saved._decode_frame(source, frame, VerifiedAssetReader())


@pytest.mark.parametrize("change", ["extra_name", "missing_name", "shape_dtype", "shape_value", "unused_bad_mask"])
def test_selected_archive_declared_names_and_full_grid_are_strict(frame_source, tmp_path, change):
    source, _labels, changed = frame_source
    frame = deepcopy(source["frames"][0])
    fields = dict(shape_hw=np.array([5, 7], np.int64), post_B=np.packbits(changed, bitorder="little"),
                  seam_affected_gas_graph=np.packbits(changed, bitorder="little"))
    if change == "extra_name": fields["undeclared"] = fields["post_B"]
    elif change == "missing_name": fields.pop("post_B")
    elif change == "shape_dtype": fields["shape_hw"] = fields["shape_hw"].astype(np.float64)
    elif change == "shape_value": fields["shape_hw"] = np.array([7, 5], np.int64)
    else: fields["seam_affected_gas_graph"] = np.ones(5, np.int16)
    path = tmp_path / "malformed-selected-archive.npz"
    np.savez_compressed(path, **fields)
    frame.update(rule_storage="selected-little-packbits-npz", operator_masks=asset(path),
                 operator_mask_names=["post_B", "seam_affected_gas_graph"], rule_names=["post_B"])
    with pytest.raises(generation.ProposalSourceUnavailable, match="mask names|shape|packed"):
        saved._decode_frame(source, frame, VerifiedAssetReader())


def test_frame_membership_is_exact_before_any_array_is_decoded(frame_source, monkeypatch):
    source, _labels, _changed = frame_source
    frame = deepcopy(source["frames"][0])
    frame["elapsed_h"] += 1
    monkeypatch.setattr(saved, "_decode_frame", lambda *args: pytest.fail("Foreign frame was decoded"))
    with pytest.raises(generation.ProposalSourceUnavailable, match="not part"):
        saved.load_frame(source, frame)


def test_declared_rule_file_disappearance_is_corruption_not_missing_optional_evidence(frame_source):
    source, _labels, _changed = frame_source
    frame = source["frames"][0]
    reference = (next(iter(frame["rule_assets"].values())) if source["adapter_kind"] == "p1_final3"
                 else frame["operator_masks"])
    Path(reference["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        saved.load_frame(source, frame)


def test_fresh_hash_after_decode_rejects_same_source_asset_mutation(frame_source, monkeypatch):
    source, _labels, _changed = frame_source
    frame = source["frames"][0]
    original = saved._decode_frame
    def mutated(*args):
        result = original(*args)
        Path(frame["labels"]["path"]).write_bytes(b"changed after decoding")
        return result
    monkeypatch.setattr(saved, "_decode_frame", mutated)
    with pytest.raises(ValueError, match="SHA differs"):
        saved.load_frame(source, frame)


def test_final_validation_rehashes_actual_assets_even_when_rebuilt_metadata_matches(frame_source, monkeypatch):
    source, _labels, _changed = frame_source
    # Isolate the final byte check. Original admission is deliberately not
    # claimed here; a full original package fixture remains Root-owned QA.
    monkeypatch.setattr(saved, "prepare_saved_categorical_source", lambda *args, **kwargs: deepcopy(source))
    Path(source["frames"][0]["labels"]["path"]).write_bytes(b"metadata unchanged, final PNG changed")
    with pytest.raises(ValueError, match="SHA differs"):
        saved.validate_source(source)


@pytest.mark.parametrize("change", ["registration_bytes", "registration_location", "model_identity"])
def test_attachment_identity_rejects_changes_without_decoding_arrays(frame_source, monkeypatch, change):
    from darsia.fluidflower import phase_saved_review_metadata as metadata
    source, _labels, _changed = frame_source
    registration = Path(source["registration"]["path"])
    monkeypatch.setattr(metadata, "configured_registration_path", lambda model: registration)
    monkeypatch.setattr(metadata, "configured_model_identity", lambda root: tuple(source["model_identity"]))
    monkeypatch.setattr(saved, "_decode_frame", lambda *args: pytest.fail("Attachment read pixels"))
    saved.validate_current_identity(source)
    if change == "registration_bytes": registration.write_text("{}", encoding="utf8")
    elif change == "registration_location":
        monkeypatch.setattr(metadata, "configured_registration_path", lambda model: registration.with_name("other.json"))
    else:
        monkeypatch.setattr(metadata, "configured_model_identity", lambda root: (source["model_id"], "c" * 64))
    with pytest.raises(ValueError, match="SHA differs|registration location|model identity"):
        saved.validate_current_identity(source)


@pytest.mark.parametrize("change", ["source_sha256", "elapsed_h", "frame_index", "run", "pipeline"])
def test_original_receipt_identity_cannot_be_joined_by_shape_or_presentation_index(change):
    record = dict(source_sha256="a" * 64, elapsed_h=2.5)
    receipt = dict(complete=True, run="ac01", frame_index=17, source_sha256="a" * 64,
                   elapsed_h=2.5, binding=dict(pipeline_sha256="b" * 64))
    saved._frame_identity(receipt, record, run="ac01", original_index=17, pipeline="b" * 64)
    if change == "pipeline": receipt["binding"]["pipeline_sha256"] = "c" * 64
    elif change == "source_sha256": receipt[change] = "c" * 64
    elif change == "elapsed_h": receipt[change] = 0.
    elif change == "frame_index": receipt[change] = 0
    else: receipt[change] = "ac02"
    with pytest.raises(generation.ProposalSourceUnavailable, match="another run"):
        saved._frame_identity(receipt, record, run="ac01", original_index=17, pipeline="b" * 64)


def test_exact_original_negative_origin_allows_only_declared_bottom_padding():
    revision = dict(phase_shape_hw=[5, 7], canvas_wh=[7, 4], native_origin_xy=[-1, -1])
    assert saved._shape({}, revision, [7, 4], [1, 1, 5, 3]) == [5, 7]
    revision["native_origin_xy"] = [1, 1]
    with pytest.raises(generation.ProposalSourceUnavailable, match="native origin"):
        saved._shape({}, revision, [7, 4], [1, 1, 5, 3])


def test_selected_canvas_can_be_exact_padded_grid_with_smaller_photographic_canvas():
    revision = dict(phase_shape_hw=[5, 7], canvas_wh=[7, 4], native_origin_xy=[-1, -1])
    assert saved._shape({"phase_shape_hw": [5, 7]}, revision, [7, 5], [1, 1, 5, 3]) == [5, 7]
    revision["canvas_wh"] = [7, 3]
    with pytest.raises(generation.ProposalSourceUnavailable, match="photographic canvas"):
        saved._shape({"phase_shape_hw": [5, 7]}, revision, [7, 5], [1, 1, 5, 3])


@pytest.mark.parametrize("code", ["phase_review_categorical_proposals.py", "phase_review_proposal_generation.py"])
def test_selector_and_publication_source_are_part_of_recorded_reader_identity(frame_source, code):
    source, _labels, _changed = frame_source
    assert code in source["producer"]["code"]
    source["producer"]["code"].pop(code)
    _seal(source)
    with pytest.raises(generation.ProposalSourceUnavailable, match="reader declaration"):
        saved.validate_source(source, require_current_assets=False)


def _selected_receipts(root, source):
    """Actual small producer-schema documents, without pretending admission."""
    def document(name, value):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf8")
        return asset(path)
    original = source["frames"][0]
    binding = dict(run="ac01", source_sha256="d" * 64, elapsed_h=2.5, work_index=17)
    scientific = dict(complete=True, run="ac01", source_sha256="d" * 64, elapsed_h=2.5,
                      full_extended_scientific=source["campaign"], baseline_extended_scientific=source["campaign"])
    scientific_ref = document("scientific-receipt.json", scientific)
    receipt = dict(contract=saved.SELECTED_CONTRACT, complete=True, run="ac01", frame_index=17,
        source_sha256="d" * 64, elapsed_h=2.5,
        binding=dict(source=binding, pipeline_sha256="a" * 64, job_sha256="e" * 64),
        labels_map=original["labels"], maps=dict(combined=original["displayed_map"]),
        rig_domain_revision=dict(phase_shape_hw=[5, 7], canvas_wh=[7, 4], native_origin_xy=[-1, -1],
                                 classes=original["classes"]),
        full_extended_scientific=source["campaign"], baseline_extended_scientific=source["campaign"])
    receipt_ref = document("selected-phase-receipt.json", receipt)
    plan_ref = document("phase-plan.json", dict(pipeline_sha256="a" * 64, job_sha256="f" * 64, heads={}))
    entry = dict(run="ac01", frame_index=17, elapsed_h=2.5, source_sha256="d" * 64,
                 source_binding=binding, receipt=receipt_ref, scientific_receipt=scientific_ref)
    gate_ref = document("accepted-gate.json", dict(phase_plan=plan_ref, pipeline_sha256="a" * 64,
                                                  entries=[entry], baselines=[]))
    record = dict(frame_index=0, source_work_index=17, source_sha256="d" * 64, elapsed_h=2.5,
        source_binding=binding, phase_receipt=receipt_ref, scientific_receipt=scientific_ref,
        labels_map=receipt["labels_map"], maps=receipt["maps"], rig_domain_revision=receipt["rig_domain_revision"])
    state = dict(native=dict(canvas_wh=[7, 5], content_box_xywh=[1, 1, 5, 3], phase_shape_hw=[5, 7],
                             frame_sources=[record]),
        inputs=dict(phase_plan=plan_ref, accepted_phase_gate=gate_ref), input_path=root / "inputs.json")
    return state, receipt, entry


def test_selected_receipt_gate_join_retains_original_job_and_work_index(frame_source, tmp_path):
    source, _labels, _changed = frame_source
    state, _receipt, _entry = _selected_receipts(tmp_path, source)
    frames = saved._selected_frames(tmp_path, "ac01", state, saved._Pins(), "a" * 64, None)
    assert len(frames) == 1 and frames[0]["frame_index"] == 0
    assert frames[0]["source_work_index"] == 17 and frames[0]["phase_job_sha256"] == "e" * 64
    assert frames[0]["labels"] == source["frames"][0]["labels"]
    assert frames[0]["operator_masks"] is None and frames[0]["rule_coverage"]["available"] is False


@pytest.mark.parametrize("change", ["gate_index", "gate_time", "gate_source", "gate_receipt",
    "native_labels", "native_operator_masks", "partial_operator_declaration", "job_sha"])
def test_selected_receipt_join_rejects_other_evidence_before_any_pixels(frame_source, tmp_path, monkeypatch, change):
    source, _labels, _changed = frame_source
    state, receipt, entry = _selected_receipts(tmp_path, source)
    record = state["native"]["frame_sources"][0]
    if change == "gate_index": entry["frame_index"] = 0
    elif change == "gate_time": entry["elapsed_h"] = 0.
    elif change == "gate_source": entry["source_binding"] = dict(entry["source_binding"], work_index=0)
    elif change == "gate_receipt": entry["receipt"] = source["registration"]
    elif change == "native_labels": record["labels_map"] = source["registration"]
    elif change == "native_operator_masks": record["operator_masks"] = source["registration"]
    else:
        if change == "job_sha": receipt["binding"]["job_sha256"] = "missing"
        else: receipt["operator_mask_names"] = []
        path = Path(record["phase_receipt"]["path"])
        path.write_text(json.dumps(receipt), encoding="utf8")
        record["phase_receipt"] = entry["receipt"] = asset(path)
    gate = state["inputs"]["accepted_phase_gate"]
    path = Path(gate["path"])
    value = json.loads(path.read_text(encoding="utf8"))
    value["entries"] = [entry]
    path.write_text(json.dumps(value), encoding="utf8")
    state["inputs"]["accepted_phase_gate"] = asset(path)
    monkeypatch.setattr(saved, "_decode_frame", lambda *args: pytest.fail("Receipt admission decoded pixels"))
    with pytest.raises(generation.ProposalSourceUnavailable):
        saved._selected_frames(tmp_path, "ac01", state, saved._Pins(), "a" * 64, None)
