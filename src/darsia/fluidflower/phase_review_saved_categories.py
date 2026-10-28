"""Read final saved categories and recorded rule changes for blind box selection.

This is a review-only reader, not inference or a scientific-support evaluator.
The ordinary P1-final3 receipt and selected-camera receipt are distinct formats.
Original admission runs before either is read. No historical producer is imported.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import math
import re

SOURCE_FORMAT = "saved-final-phase-categories-blind-proposal-source-v1"
LEGACY_CONTRACT = "ffac-g3-t4c-p1-final3-candidate-v1"
SELECTED_CONTRACT = "ffac-selected-p1-final3-phase-v1"
CLASS_FIELDS = ("rig_domain", "modelled", "filled_observed", "filled_unobserved")
LEGACY_MASKS = ("below_sand_floor_masks", "floor_connectivity_masks",
                "floor_contact_fill_v2", "gas_envelope_masks", "contact_fill_b_reverted_masks")
SELECTED_CHANGES = ("seam_changed_labels", "seam_changed_co2", "seam_changed_gas",
                    "post_removed_top", "post_A", "post_B", "post_C", "post_D",
                    "post_contact_reason_a", "post_contact_reason_b")
SELECTED_REQUIRED = SELECTED_CHANGES[3:]
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CODE_FILES = ("phase_review_saved_categories.py", "phase_review_categorical_proposals.py",
               "phase_review_proposal_generation.py", "phase_saved_assets.py",
               "phase_saved_review_metadata.py", "phase_saved_contact_review.py",
               "phase_saved_categorical_review.py", "phase_saved_boundary_review.py",
               "phase_saved_selected_registration.py", "phase_selected_policy_review.py",
               "phase_selected_review.py", "phase_blind_review_policy.py",
               "tools/calibration_pairwise_review.py")


def _core():
    # The generation core dispatches here lazily; do not create a module cycle.
    from . import phase_review_proposal_generation
    return phase_review_proposal_generation


def _require(condition, message):
    if not condition:
        raise _core().ProposalSourceUnavailable(message)


def _descriptor(value, base=None):
    result = _core()._descriptor(value, base=base)
    for name in ("size_bytes", "storage", "shape_hw", "frame_index", "source_sha256",
                 "contract", "array_key"):
        if name in value:
            result[name] = deepcopy(value[name])
    return result


class _Pins:
    """Record small proof documents read here; arrays are verified at frame load."""
    def __init__(self, reader=None):
        self.reader = reader or _core()._reader()
        self.proofs = {}

    def path(self, descriptor):
        descriptor = _descriptor(descriptor)
        path = self.reader.path(descriptor)
        if "size_bytes" in descriptor:
            _require(type(descriptor["size_bytes"]) is int
                     and descriptor["size_bytes"] == path.stat().st_size,
                     "A saved categorical proof has changed size.")
        prior = self.proofs.get(descriptor["path"])
        _require(prior is None or prior["sha256"] == descriptor["sha256"],
                 "A saved categorical proof has conflicting SHA bindings.")
        self.proofs[descriptor["path"]] = descriptor
        return path

    def json(self, descriptor):
        descriptor = _descriptor(descriptor)
        self.path(descriptor)
        result = self.reader.json(descriptor)
        self.path(descriptor)
        return result

    def document(self, path):
        descriptor = _core()._file(path)
        return descriptor, self.json(descriptor)


def _resolve(path, base):
    from .review_storage import resolve_review_path
    path = Path(path)
    return resolve_review_path(path if path.is_absolute() else Path(base) / path).resolve()


def _publication(roots, run):
    _core()._require_publication_visible(roots, run)


def _registration_proofs(pins, root, registration):
    from .phase_saved_review_metadata import configured_registration_path
    reference, actual = pins.document(configured_registration_path(root.name))
    _require(actual == registration, "The selected registration changed during categorical preparation.")
    for relative, digest in registration["metadata_sha256"].items():
        path = (root / relative).resolve()
        _require(path.is_relative_to(root), "Registered metadata leaves its declared candidate.")
        pins.path(dict(path=str(path), sha256=digest))
    return reference


def _shape(native, revision, canvas, box):
    shape = revision.get("phase_shape_hw", native.get("phase_shape_hw", canvas[::-1]))
    _require(isinstance(shape, (list, tuple)) and len(shape) == 2
             and all(type(v) is int and v > 0 for v in shape),
             "Saved categories have no exact phase raster.")
    photographic = revision.get("canvas_wh")
    _require(isinstance(photographic, list) and len(photographic) == 2
             and all(type(value) is int and value > 0 for value in photographic)
             and photographic[0] == canvas[0] and box[1] + box[3] <= photographic[1] <= shape[0]
             and shape[1] == canvas[0] and shape[0] >= canvas[1]
             and (photographic == canvas or canvas == list(shape)[::-1])
             and revision.get("native_origin_xy") == [-box[0], -box[1]],
             "Saved final phase raster differs from its photographic canvas or native origin.")
    if native.get("phase_shape_hw") is not None:
        _require(list(shape) == native["phase_shape_hw"], "Native and final phase rasters disagree.")
    return list(shape)


def _frame_identity(receipt, record, *, run, original_index, pipeline):
    _require(receipt.get("complete") is True
             and receipt.get("run") == run
             and type(receipt.get("frame_index")) is int
             and receipt["frame_index"] == original_index
             and receipt.get("source_sha256") == record.get("source_sha256")
             and receipt.get("elapsed_h") == record.get("elapsed_h")
             and receipt.get("binding", {}).get("pipeline_sha256") == pipeline,
             "Final phase receipt belongs to another run, source, time, frame or pipeline.")


def _packed_descriptor(value, record, shape, base):
    result = _descriptor(value, base)
    _require(result.get("storage") == "packbits-npy"
             and result.get("shape_hw") == shape
             and result.get("frame_index") == record["frame_index"]
             and result.get("source_sha256") == record["source_sha256"],
             "Saved rule mask lost its exact frame/source/grid binding.")
    return result


def _legacy_frames(root, run, state, pins, pipeline, candidate, campaign, cancelled):
    from .phase_saved_contact_review import PLAN, SEAL
    from .tools import calibration_pairwise_review as review
    _plan_ref, plan = pins.document(root / PLAN)
    _seal_ref, seal = pins.document(root / SEAL)
    inventory = [item for item in plan["source_inventory"] if item["run"] == run]
    receipts = [item for item in seal["frames"] if item["run"] == run]
    records = state["native"]["frame_sources"]
    _require(len(inventory) == len(receipts) == len(records)
             and {item["frame_index"] for item in inventory} == set(range(len(records)))
             and {item["frame_index"] for item in receipts} == set(range(len(records))),
             "Final3 seal does not bind this complete native timeline uniquely.")
    by_source = {item["frame_index"]: item for item in inventory}
    by_receipt = {item["frame_index"]: item for item in receipts}
    frames = []
    for index, record in enumerate(records):
        _core()._cancelled(cancelled)
        source = by_source[index]
        _require(record.get("frame_index") == index
                 and all(source[key] == record.get(key) for key in ("source_sha256", "elapsed_h")),
                 "Final3 plan and native frame identify different acquisitions.")
        receipt_ref = _descriptor(by_receipt[index]["receipt"], root)
        receipt = pins.json(receipt_ref)
        _frame_identity(receipt, record, run=run, original_index=index, pipeline=pipeline)
        _require(receipt.get("contract") == LEGACY_CONTRACT
                 and receipt.get("placement") == "P1-final3"
                 and receipt.get("qa", {}).get("all_pass") is True
                 and receipt.get("binding", {}).get("parent_receipt") == source["parent_receipt"]
                 and all(receipt["binding"]["source"].get(k) == source[k]
                         for k in ("run", "frame_index", "source_sha256", "elapsed_h"))
                 and receipt.get("rig_domain_revision") == record.get("rig_domain_revision"),
                 "Final3 frame lost its original parent, final classes or source binding.")
        parent = pins.json(_descriptor(source["parent_receipt"], root))
        _require(parent.get("labels_map") == receipt.get("before_labels")
                 and parent.get("before_labels") == receipt.get("p1_final_before_labels")
                 and parent.get("binding", {}).get("source") == receipt["binding"]["source"],
                 "Final3 before-label evidence is not its exact parent frame.")
        frame = _base_frame(record, receipt, receipt_ref, state["native"], root)
        expected = review._native_phase_map_path(campaign, candidate, "A", index)
        _require(_resolve(receipt["maps"]["combined"]["path"], root) == expected.resolve(),
                 "Final3 labels are not bound to the displayed candidate map.")
        frame["rule_storage"] = "legacy-big-packbits-npy"
        available = [name for name in LEGACY_MASKS if name in receipt]
        _require(all((name in record) == (name in receipt) for name in LEGACY_MASKS),
                 "Final3 native timeline and rule receipts disagree about available evidence.")
        frame["rule_assets"] = {
            name: _packed_descriptor(receipt[name], record, frame["phase_shape_hw"], root)
            for name in available
        }
        _require(all(record[name] == receipt[name] for name in available),
                 "Final3 native timeline and rule receipts disagree.")
        frame["rule_coverage"] = _coverage(available, [name for name in LEGACY_MASKS if name not in available],
                                           "recorded_final3_stage_masks")
        frames.append(frame)
    return frames


def _coverage(available, missing, basis):
    return dict(basis=basis, available=bool(available), available_names=list(available), missing=list(missing),
                complete=False, net_final_change_claimed=False,
                scientific_support_claimed=False,
                note="Recorded stage-change locations only; complete rule coverage and net-final differences are not asserted.")


def _base_frame(record, receipt, receipt_ref, native, base):
    canvas, box = _core()._geometry(native)
    revision = receipt["rig_domain_revision"]
    shape = _shape(native, revision, canvas, box)
    _require(_SHA.fullmatch(str(record.get("source_sha256")))
             and type(record.get("elapsed_h")) in (int, float)
             and math.isfinite(record["elapsed_h"]), "Final category frame lacks a finite source/time identity.")
    return dict(frame_index=record["frame_index"], elapsed_h=record["elapsed_h"],
                source_sha256=record["source_sha256"],
                source_name=record.get("source_name") or record.get("source_path"),
                phase_shape_hw=shape, phase_receipt=receipt_ref,
                labels=_descriptor(receipt["labels_map"], base),
                classes=_descriptor(revision["classes"], base),
                displayed_map=_descriptor(receipt["maps"]["combined"], base))


def _selected_frames(root, run, state, pins, pipeline, cancelled):
    # The original package stores both the accepted gate and the phase plan in
    # its source inputs. Bind the exact gate entry, including accepted imports;
    # do not relabel an older producer job as a later execution job.
    from .phase_saved_boundary_review import _head_assets
    plan_ref = _descriptor(state["inputs"]["phase_plan"], state["input_path"].parent)
    plan = pins.json(plan_ref)
    gate = pins.json(_descriptor(state["inputs"]["accepted_phase_gate"], state["input_path"].parent))
    _require(_descriptor(gate["phase_plan"], root) == plan_ref
             and plan.get("pipeline_sha256") == gate.get("pipeline_sha256") == pipeline,
             "Selected final categories lost their accepted phase-plan identity.")
    for descriptor in _head_assets(plan["heads"]):
        pins.path(_descriptor(descriptor, Path(plan_ref["path"]).parent))
    accepted = [item for item in [*gate["entries"], *gate["baselines"]] if item["run"] == run]
    _require(len(accepted) == len(state["native"]["frame_sources"]),
             "Selected accepted gate and native timeline have different frame counts.")
    frames = []
    for index, record in enumerate(state["native"]["frame_sources"]):
        _core()._cancelled(cancelled)
        original_index = record.get("source_work_index")
        _require(record.get("frame_index") == index and type(original_index) is int and original_index >= 0,
                 "Selected timeline has no exact original work index.")
        receipt_ref = _descriptor(record["phase_receipt"], root)
        receipt = pins.json(receipt_ref)
        _frame_identity(receipt, record, run=run, original_index=original_index, pipeline=pipeline)
        entries = [item for item in accepted if item["source_sha256"] == record["source_sha256"]]
        _require(len(entries) == 1 and entries[0].get("frame_index") == original_index
                 and entries[0].get("elapsed_h") == record["elapsed_h"]
                 and entries[0].get("source_binding") == record["source_binding"]
                 and _descriptor(entries[0]["receipt"], root) == receipt_ref
                 and entries[0].get("scientific_receipt") == record["scientific_receipt"],
                 "Selected phase receipt is not the exact accepted source/work-index entry.")
        _require(receipt.get("contract") == SELECTED_CONTRACT
                 and _SHA.fullmatch(str(receipt.get("binding", {}).get("job_sha256")))
                 and receipt.get("binding", {}).get("source") == record.get("source_binding")
                 and receipt.get("labels_map") == record.get("labels_map")
                 and receipt.get("operator_masks") == record.get("operator_masks")
                 and receipt.get("maps") == record.get("maps")
                 and receipt.get("rig_domain_revision") == record.get("rig_domain_revision"),
                 "Selected final labels lost their exact source, display map or class binding.")
        scientific = pins.json(_descriptor(record["scientific_receipt"], root))
        _require(scientific.get("complete") is True and scientific.get("run") == run
                 and scientific.get("source_sha256") == record["source_sha256"]
                 and scientific.get("elapsed_h") == record["elapsed_h"]
                 and scientific.get("full_extended_scientific") == receipt.get("full_extended_scientific")
                 and scientific.get("baseline_extended_scientific") == receipt.get("baseline_extended_scientific"),
                 "Selected final phase and scientific receipts belong to different acquisitions.")
        frame = _base_frame(record, receipt, receipt_ref, state["native"], root)
        has_archive = "operator_masks" in receipt
        _require(has_archive == ("operator_mask_names" in receipt),
                 "Selected phase receipt declares incomplete operator-mask evidence.")
        names = receipt.get("operator_mask_names", [])
        _require(isinstance(names, list) and all(isinstance(name, str) for name in names)
                 and names == sorted(set(names)), "Selected phase receipt has invalid operator-mask names.")
        available = [name for name in SELECTED_CHANGES if name in names]
        frame.update(source_work_index=original_index,
                     phase_job_sha256=receipt["binding"]["job_sha256"],
                     rule_storage="selected-little-packbits-npz",
                     operator_masks=_descriptor(receipt["operator_masks"], root) if has_archive else None,
                     operator_mask_names=names, rule_names=available,
                     rule_coverage=_coverage(available, [name for name in SELECTED_CHANGES if name not in names],
                                             "recorded_selected_stage_masks"))
        frames.append(frame)
    return frames


def _legacy_native_proofs(state, pins):
    """Only the metadata certificates used by the original saved native reader."""
    pins.document(state["input_path"])
    for key in ("parent_native_manifest", "camera_timeline_run"):
        pins.json(_descriptor(state["inputs"][key], state["input_path"].parent))
    pairs = [(state["inputs"]["reference"], state["native"]["reference"])]
    pairs.extend(zip(state["inputs"]["frames"], state["native"]["frame_sources"]))
    for row, record in pairs:
        pins.json(_descriptor(row["geometry_receipt"], state["input_path"].parent))
        pins.json(_descriptor(record["full_native_scientific"]["receipt"], state["input_path"].parent))


def _prepare(campaign_root, run_id, *, candidate_id, params_sha256, model_id,
             model_root, reader=None, cancelled=None):
    from .tools import calibration_pairwise_review as review
    root, campaign = _resolve(model_root, Path.cwd()), _resolve(campaign_root, Path.cwd())
    run = str(run_id).strip().casefold()
    _require(bool(run) and bool(candidate_id) and bool(model_id) and _SHA.fullmatch(str(params_sha256)),
             "Exact model, candidate, run and pipeline identities are required.")
    _core()._cancelled(cancelled)
    _publication([str(root), str(campaign)], run)
    pins = _Pins(reader)
    campaign_ref, manifest = pins.document(campaign / "postflight_manifest.json")
    jobs = [row for row in manifest.get("jobs", ()) if isinstance(row, Mapping)
            and str(row.get("run", "")).casefold() == run and row.get("candidate_id") == candidate_id
            and row.get("params_sha256") == params_sha256]
    _require(len(jobs) == 1, "This campaign does not identify one exact saved phase candidate.")
    job = jobs[0]
    _require(_resolve(job.get("candidate_folder", ""), campaign) == root,
             "The selected job belongs to another model folder.")
    publication_roots = list(dict.fromkeys([str(root), str(campaign),
        *[str(_resolve(job[key], campaign)) for key in ("review_output_dir", "phase_map_output_dir") if job.get(key)]]))
    _publication(publication_roots, run)
    flags = job.get("quality_flags") or {}
    if flags.get("selected_camera_review") is True:
        from . import phase_saved_selected_registration as registration
        from . import phase_selected_policy_review as original
        checked = registration.selected_saved_review(root)
        candidate = review._selected_policy_a_review_candidate(campaign, job, root, checked)
        state = original.load_native_source(candidate.review_output_dir, run, reader=pins, verify_geometry=True)
        kind = "selected_camera"
    elif flags.get("approved_g3_downstream") is True and (root / "p1_final3_plan.json").is_file():
        from . import phase_saved_contact_review as registration
        from . import phase_saved_review_metadata as original
        checked = registration.p1_final3_saved_review(root)
        candidate = review._admitted_saved_review_candidate(campaign, job, root, checked,
                                                            revision="P1-final3", kind=registration.KIND)
        state = original.load_native_source(candidate.review_output_dir, run, verify_geometry=True)
        kind = "p1_final3"
        _legacy_native_proofs(state, pins)
    else:
        _require(False, "No saved categorical adapter is available for this candidate; supported routes are admitted P1-final3 and selected-camera review.")
    _require(checked["model_identity"][0] == model_id and checked["pipeline_sha256"] == params_sha256,
             "Original saved-review admission identifies another model or pipeline.")
    _require(not review._candidate_prediction_withheld(campaign, candidate),
             "Original publication policy withholds this candidate's predictions.")
    registration_ref = _registration_proofs(pins, root, checked["registration"])
    # Bind the exact original model identity files as well as the admitted heads.
    # These are read as bytes only, never evaluated or used for prediction.
    from .phase_saved_review_metadata import configured_parameters
    from .phase_saved_boundary_review import _head_assets
    model_parameters = configured_parameters("floor")["model_identity"]
    for relative in dict.fromkeys([model_parameters["summary"], *model_parameters["files"]]):
        pins.path(_core()._file(root / relative))
    if kind == "p1_final3":
        from .phase_saved_contact_review import PLAN
        _plan_reference, model_plan = pins.document(root / PLAN)
        for descriptor in _head_assets(model_plan["binding"]["heads"]):
            pins.path(_descriptor(descriptor, root))
    native_ref, native = pins.document(state["native_path"])
    _require(native == state["native"], "Native timeline changed during categorical source admission.")
    canvas, box = _core()._geometry(native)
    frames = (_selected_frames(root, run, state, pins, params_sha256, cancelled) if kind == "selected_camera"
              else _legacy_frames(root, run, state, pins, params_sha256, candidate, campaign, cancelled))
    _require(bool(frames), "Saved categorical source contains no frames.")
    code = {name: _core()._file(Path(__file__).parent / name) for name in _CODE_FILES}
    for descriptor in code.values():
        pins.path(descriptor)
    _publication(publication_roots, run)
    _core()._cancelled(cancelled)
    source = dict(format=SOURCE_FORMAT, adapter_kind=kind, run=run, candidate_id=candidate_id,
                  params_sha256=params_sha256, model_id=model_id, model_root=str(root),
                  model_identity=list(checked["model_identity"]),
                  campaign=campaign_ref, candidate_job=deepcopy(job), registration=registration_ref,
                  native_manifest=native_ref, canvas_wh=canvas, content_box_xywh=box, frames=frames,
                  publication_roots=publication_roots,
                  producer=dict(module="darsia.fluidflower.phase_review_saved_categories",
                                function="load_frame", schema=1, code=code),
                  proof_assets=[pins.proofs[key] for key in sorted(pins.proofs)],
                  scientific_maps_recomputed=False, source_arrays_substituted=False)
    source["source_sha256"] = _core()._digest(source)
    return source


def prepare_saved_categorical_source(campaign_root, run_id, *, candidate_id, params_sha256,
                                     model_id, model_root, reader=None, cancelled=None):
    source = _prepare(campaign_root, run_id, candidate_id=candidate_id, params_sha256=params_sha256,
                      model_id=model_id, model_root=model_root, reader=reader, cancelled=cancelled)
    # A fresh reader checks every metadata/code proof after admission, avoiding
    # accidental acceptance of a cached document changed during preparation.
    fresh = _core()._reader()
    for descriptor in source["proof_assets"]:
        _core()._cancelled(cancelled)
        fresh.path(descriptor)
    _publication(source["publication_roots"], source["run"])
    return source


def _references(source):
    yield from (source["campaign"], source["native_manifest"], source["registration"])
    yield from source["proof_assets"]
    yield from source["producer"]["code"].values()
    for frame in source["frames"]:
        yield from (frame["phase_receipt"], frame["labels"], frame["classes"], frame["displayed_map"])
        if frame["rule_storage"] == "legacy-big-packbits-npy":
            yield from frame["rule_assets"].values()
        elif frame["operator_masks"] is not None:
            yield frame["operator_masks"]


def validate_source(source, reader=None, *, require_current_assets=True):
    """Historical reading needs stored identities only; appending re-admits source."""
    _require(isinstance(source, Mapping), "Saved categorical source must be a record.")
    content = dict(source)
    digest = content.pop("source_sha256", None)
    _require(source.get("format") == SOURCE_FORMAT and _SHA.fullmatch(str(digest))
             and _core()._digest(content) == digest
             and source.get("adapter_kind") in {"p1_final3", "selected_camera"}
             and source.get("scientific_maps_recomputed") is False
             and source.get("source_arrays_substituted") is False,
             "Saved categorical source content or identity changed.")
    try:
        canvas, box = _core()._geometry(source)
        _require(isinstance(source["model_identity"], list) and len(source["model_identity"]) == 2
                 and source["model_identity"][0] == source["model_id"]
                 and _SHA.fullmatch(str(source["model_identity"][1])), "Saved model identity is incomplete.")
        _require(source["producer"]["module"] == "darsia.fluidflower.phase_review_saved_categories"
                 and source["producer"]["function"] == "load_frame" and source["producer"]["schema"] == 1
                 and set(source["producer"]["code"]) == set(_CODE_FILES),
                 "Saved categorical reader declaration is unsupported.")
        _require(isinstance(source["frames"], list) and bool(source["frames"]), "Saved categorical frames are absent.")
        for index, frame in enumerate(source["frames"]):
            _require(type(frame["frame_index"]) is int and frame["frame_index"] == index
                     and _SHA.fullmatch(str(frame["source_sha256"]))
                     and type(frame["elapsed_h"]) in (int, float) and math.isfinite(frame["elapsed_h"]),
                     "Recorded categorical frame identity is invalid.")
            shape = frame["phase_shape_hw"]
            _require(isinstance(shape, list) and len(shape) == 2
                     and all(type(value) is int and value > 0 for value in shape)
                     and shape[1] == canvas[0] and shape[0] >= canvas[1],
                     "Recorded categorical raster is invalid.")
            if source["adapter_kind"] == "p1_final3":
                _require(frame["rule_storage"] == "legacy-big-packbits-npy"
                         and isinstance(frame["rule_assets"], Mapping)
                         and set(frame["rule_assets"]) <= set(LEGACY_MASKS), "Recorded final3 rule evidence is invalid.")
                available = [name for name in LEGACY_MASKS if name in frame["rule_assets"]]
                expected_coverage = _coverage(available, [name for name in LEGACY_MASKS if name not in available],
                                              "recorded_final3_stage_masks")
                for descriptor in frame["rule_assets"].values():
                    _require(descriptor.get("storage") == "packbits-npy"
                             and descriptor.get("shape_hw") == shape
                             and descriptor.get("frame_index") == index
                             and descriptor.get("source_sha256") == frame["source_sha256"],
                             "Recorded final3 mask belongs to another frame.")
            else:
                names = frame["operator_mask_names"]
                _require(frame["rule_storage"] == "selected-little-packbits-npz"
                         and type(frame["source_work_index"]) is int and frame["source_work_index"] >= 0
                         and _SHA.fullmatch(str(frame["phase_job_sha256"]))
                         and isinstance(names, list) and all(isinstance(name, str) for name in names)
                         and names == sorted(set(names))
                         and (frame["operator_masks"] is not None or not names)
                         and frame["rule_names"] == [name for name in SELECTED_CHANGES if name in names],
                         "Recorded selected rule evidence is invalid.")
                expected_coverage = _coverage(frame["rule_names"], [name for name in SELECTED_CHANGES if name not in names],
                                              "recorded_selected_stage_masks")
            _require(frame["rule_coverage"] == expected_coverage, "Recorded rule coverage differs from its evidence.")
        for descriptor in _references(source):
            _require(isinstance(descriptor, Mapping) and isinstance(descriptor.get("path"), str)
                     and bool(descriptor["path"]) and _SHA.fullmatch(str(descriptor.get("sha256"))),
                     "Recorded categorical asset lacks its exact path and SHA.")
        _require(isinstance(source["publication_roots"], list) and bool(source["publication_roots"])
                 and all(isinstance(path, str) and path for path in source["publication_roots"]),
                 "Recorded categorical publication roots are absent.")
    except (KeyError, TypeError) as error:
        raise _core().ProposalSourceUnavailable("Recorded categorical source is incomplete.") from error
    if not require_current_assets:
        return
    rebuilt = prepare_saved_categorical_source(Path(source["campaign"]["path"]).parent, source["run"],
        candidate_id=source["candidate_id"], params_sha256=source["params_sha256"], model_id=source["model_id"],
        model_root=source["model_root"], reader=reader)
    _require(rebuilt == source, "The saved final-category source, admission or reader code changed; retain old boxes and answers.")
    # Use a new reader: a receipt SHA alone does not prove the final arrays are
    # still the bytes selected earlier in this request. No arrays are decoded.
    fresh = _core()._reader()
    for descriptor in _references(source):
        fresh.path(descriptor)
    _publication(source["publication_roots"], source["run"])


def validate_current_identity(source):
    """Recheck the bound registration/model and metadata before live attachment.

    The worker already verified final arrays under the publication locks. This
    GUI-side check reads only identity/code/receipt files, never image arrays.
    """
    validate_source(source, require_current_assets=False)
    from .phase_saved_review_metadata import configured_model_identity, configured_registration_path
    root = _resolve(source["model_root"], Path.cwd())
    _require(_resolve(configured_registration_path(root.name), root) == Path(source["registration"]["path"]),
             "Current registration location changed before attaching boxes.")
    fresh = _core()._reader()
    for descriptor in source["proof_assets"]:
        fresh.path(descriptor)
    for descriptor in (source["registration"], source["campaign"], source["native_manifest"]):
        fresh.path(descriptor)
    _require(list(configured_model_identity(root)) == source["model_identity"],
             "The selected model identity changed before attaching boxes.")
    _publication(source["publication_roots"], source["run"])


def _decode_frame(source, frame, reader):
    """Decode authenticated saved arrays; no inference, resizing or rule replay."""
    import numpy as np
    from PIL import Image
    shape = tuple(frame["phase_shape_hw"])
    with Image.open(reader.path(frame["labels"])) as image:
        _require(image.mode == "L", "Final saved labels must be the original L-mode categorical PNG.")
        labels = np.array(image)
    _require(labels.dtype == np.uint8 and labels.shape == shape and np.isin(labels, (0, 1, 2, 255)).all(),
             "Final saved labels have another shape, dtype or category encoding.")
    with np.load(reader.path(frame["classes"]), allow_pickle=False) as archive:
        _require(set(CLASS_FIELDS) <= set(archive.files), "Saved final class partition is incomplete.")
        classes = {name: archive[name].copy() for name in CLASS_FIELDS}
    _require(all(array.dtype == np.bool_ and array.shape == shape for array in classes.values()),
             "Final saved classes must be bool arrays on the exact phase grid.")
    membership = sum(classes[name].astype(np.uint8) for name in CLASS_FIELDS[1:])
    _require(not np.any(membership > 1) and np.array_equal(membership == 1, classes["rig_domain"])
             and np.array_equal(labels != 255, classes["rig_domain"]),
             "Final categories and their modelled/filled/domain partition disagree.")
    count = shape[0] * shape[1]
    def unpack(array, order):
        _require(array.dtype == np.uint8 and array.ndim == 1 and array.size == (count + 7) // 8,
                 "Recorded rule mask has invalid packed storage.")
        bits = np.unpackbits(array, bitorder=order)
        _require(not bits[count:].any(), "Recorded rule mask has nonzero packed padding.")
        return bits[:count].reshape(shape).astype(bool)
    if frame["rule_storage"] == "legacy-big-packbits-npy":
        masks = {name: unpack(np.load(reader.path(descriptor), allow_pickle=False), "big")
                 for name, descriptor in frame["rule_assets"].items()}
    elif frame["operator_masks"] is not None:
        with np.load(reader.path(frame["operator_masks"]), allow_pickle=False) as archive:
            _require(set(archive.files) == {"shape_hw", *frame["operator_mask_names"]},
                     "Selected operator archive differs from its declared mask names.")
            saved_shape = archive["shape_hw"]
            _require(saved_shape.dtype == np.int64 and saved_shape.shape == (2,)
                     and saved_shape.tolist() == list(shape), "Selected operator archive has another shape.")
            # Check every declared mask, even masks not used as change seeds.
            masks = {}
            for name in frame["operator_mask_names"]:
                decoded = unpack(archive[name], "little")
                if name in frame["rule_names"]:
                    masks[name] = decoded
    else:
        masks = {}
    x, y, w, h = source["content_box_xywh"]
    def cropped(array):
        result = array[y:y+h, x:x+w].copy()
        _require(result.shape == (h, w), "Saved categorical native-content crop leaves its phase raster.")
        result.flags.writeable = False
        return result
    fields = {name: cropped(array) for name, array in classes.items()}
    return dict(labels=cropped(labels), domain=fields["rig_domain"], classes=fields,
                rule_masks={name: cropped(array) for name, array in masks.items()},
                facies=None, rule_coverage=deepcopy(frame["rule_coverage"]))


def load_frame(source, frame, reader=None):
    # The generation owner performs full current admission at both ends of the
    # request. Per-frame reads need only exact membership and authenticated bytes.
    validate_source(source, require_current_assets=False)
    _require(isinstance(frame, Mapping) and type(frame.get("frame_index")) is int
             and 0 <= frame["frame_index"] < len(source["frames"])
             and source["frames"][frame["frame_index"]] == frame,
             "Requested final-category frame is not part of this exact source.")
    _publication(source["publication_roots"], source["run"])
    result = _decode_frame(source, frame, reader or _core()._reader())
    # Hash the files actually decoded with a fresh reader before publishing boxes.
    fresh = _core()._reader()
    descriptors = [frame["labels"], frame["classes"], frame["displayed_map"]]
    descriptors.extend(frame["rule_assets"].values() if source["adapter_kind"] == "p1_final3"
                       else ([frame["operator_masks"]] if frame["operator_masks"] is not None else []))
    for descriptor in descriptors:
        fresh.path(descriptor)
    _publication(source["publication_roots"], source["run"])
    return result
