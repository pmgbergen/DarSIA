"""Reuse reviewed local maps without repeating their original RAW preparation.

This ordinary local route copies authentic original map bytes into a new
preparation. The saved review, source painting, geometry and every product stay
bound. It grants no approval and never changes the retained original files.
"""
from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
from threading import RLock
import json
from pathlib import Path
import time

PROOF_FORMAT = "darsia-local-reviewed-map-reuse-v1"
_REVIEWS = OrderedDict()
_REVIEW_LOCK = RLock()

CHANGED_CHART_RUNS = frozenset(("ac23", "ac25", "ac26", "ac28", "ac29", "ac30",
    "ac31", "ac32", "ac33", "ac34", "ac35", "ac38", "ac39", "ac40"))


def _read_local_review(descriptor):
    """Keep an authenticated immutable local review parsed once per process."""
    from .painted_adoption_table import _checked
    from .painted_transfer_replay import _read_review
    actual = _checked(descriptor)
    key = (actual["path"], actual["sha256"], actual["size_bytes"])
    with _REVIEW_LOCK:
        result = _REVIEWS.get(key)
        if result is None:
            result = _read_review(actual)
            _REVIEWS[key] = result
            while len(_REVIEWS) > 2:
                _REVIEWS.popitem(last=False)
        else:
            _REVIEWS.move_to_end(key)
        return result


def _same(a, b):
    from .painted_adoption_table import _same as original_same
    return original_same(a, b)


def verify_local_geometry(binding, proof, expected_models, folder):
    """Check the original reviewed source/H/products, without a RAW decoder."""
    from .painted_adoption_table import _checked
    from .painted_transfer_replay import _read_review, _context
    from .photo_edge_continuation import read_contracts, validate_contract
    if (not isinstance(proof, dict)
            or set(proof) != {"format", "review", "source_context", "models", "products", "files"}
            or proof["format"] != PROOF_FORMAT):
        raise ValueError("An exact local reviewed-map reuse proof is required")
    review = _read_local_review(proof["review"])
    if not _same(binding, review["source"]["painting_facies"]):
        raise ValueError("Local map reuse belongs to another approved painting")
    original = Path(review["source_map"]["path"]).resolve().parent
    context, _ = _context(review, proof["source_context"], original)
    contracts = read_contracts(context["continuation"]["path"])["runs"]
    expected_runs = [row["run_id"] for row in review["runs"]]
    if set(contracts) != set(expected_runs) or len(expected_runs) != len(set(expected_runs)):
        raise ValueError("Local map reuse must retain the exact reviewed cohort")
    models, products = {}, {}
    guarded = [review["review_binding"], review["source_map"], context["continuation"], review["geometry_index"]]
    for row in review["runs"]:
        run, contract = row["run_id"], contracts[row["run_id"]]
        if not _same(contract, row["contract"]):
            raise ValueError("Local map reuse changed an original H/frame contract")
        # Structural original frame/H rules remain; each actual model document
        # is authenticated below. No RAW replay or camera re-fit is performed.
        validate_contract(contract, verify_files=False)
        for key, name in ((contract["source_run"], "source_model"),
                          (run, "target_model")):
            descriptor = row["reference_model" if name == "source_model" else "target_model"]
            if not _same(descriptor, contract[name]["descriptor"]):
                raise ValueError("Local map reuse changed its model association")
            actual = _checked(descriptor, metadata=True)
            if not _same(actual, contract[name]["document"]):
                raise ValueError("Local map reuse changed its exact model/H bytes")
            if key in models and not _same(models[key], descriptor):
                raise ValueError("Local map reuse repeats one model with different bytes")
            models[key] = deepcopy(descriptor)
            guarded.append(descriptor)
        products[run] = deepcopy(row["base_product"])
        guarded.append(row["base_product"])
    if (not _same(models, proof["models"]) or not _same(models, expected_models)
            or not _same(products, proof["products"])):
        raise ValueError("Local map reuse lost its exact source/model/product closure")
    if not isinstance(proof["files"], list):
        raise ValueError("Local map reuse needs the original consumed file list")
    required = {(x["path"], x["sha256"], x["size_bytes"]) for x in guarded}
    listed = {(x["path"], x["sha256"], x["size_bytes"]) for x in proof["files"]}
    if not required <= listed:
        raise ValueError("Local map reuse lost a result-critical original source")
    for bound in proof["files"]:
        _checked(bound)
    return deepcopy(proof["files"])


def verify_local_preparation(preparation, folder):
    """Verify unchanged original and copied bytes; do not recompute the maps."""
    from .setup_source_reads import reuse_local_preparation_proof
    return reuse_local_preparation_proof(preparation, folder, reader=_verify_local_preparation)


def _verify_local_preparation(preparation, folder):
    """Replay the complete original reviewed-map closure and copied products."""
    from .painted_adoption_table import _checked, _local
    from .painted_transfer_replay import _read_review
    proof = preparation.get("painting_geometry")
    review = _read_local_review(proof["review"])
    models = {}
    for row in preparation.get("runs", []):
        models[preparation["reference_run"]] = row["reference_model"]
        models[row["run_id"]] = row["target_model"]
    guarded = verify_local_geometry(preparation["painting_facies"], proof, models, folder)
    if (preparation.get("reference_run") != review["source"]["reference_run"]
            or not _same(preparation.get("geometry_index"), review["geometry_index"])
            or not _same(preparation.get("map_version"), review["map_version"])
            or not _same(preparation.get("approved_source"), review["source"]["painting_facies"])):
        raise ValueError("Local map reuse changed its approved source/reference")
    from .painted_adoption_table import _bound
    approval = _checked(_bound(Path(folder) / "approval.json"), metadata=True)
    if not _same(approval, preparation["painting_facies"]["approval"]):
        raise ValueError("Local map reuse changed the approved painting action")
    source = _local(folder, preparation["source_map"])
    if any(source[k] != review["source_map"][k] for k in ("sha256", "size_bytes")):
        raise ValueError("Local map reuse changed the source map")
    continuation = preparation["continuation"]
    copied = _local(folder, dict(path=continuation["file"],
        **{k: continuation[k] for k in ("sha256", "size_bytes")}))
    if any(copied[k] != proof["source_context"]["continuation"][k] for k in ("sha256", "size_bytes")):
        raise ValueError("Local map reuse changed the continuation bytes")
    rows = preparation.get("runs")
    if ([row["run_id"] for row in rows] != [row["run_id"] for row in review["runs"]]
            or any(row.get("status") != "prepared" for row in rows)):
        raise ValueError("Local map reuse changed its reviewed run order")
    for row, original in zip(rows, review["runs"]):
        copied = _local(folder, row["labels"])
        if (any(copied[k] != original["base_product"][k] for k in ("sha256", "size_bytes"))
                or row.get("shape_hw") != original["base_scientific_arrays"]["array"]["shape"]
                or not _same(row["reference_model"], original["reference_model"])
                or not _same(row["target_model"], original["target_model"])):
            raise ValueError("Local map reuse changed a reviewed map/grid")
        guarded.append(copied)
    transfers = preparation.get("chart_transfers", {})
    if not isinstance(transfers, dict) or set(transfers) - {row['run_id'] for row in rows}:
        raise ValueError("Only explicitly reviewed adoption runs may change their ColorChecker")
    if 'chart_review' in preparation or transfers:
        from .painted_target_integration import effective_product
        for row in rows:
            _, bound = effective_product(preparation, folder, row)
            guarded.append(bound)
    return guarded


def prepare_local_adoption(review_source, *, source_context, destination,
                           analysis_exclusions=None, chart_transfers=None, detection_manifest=None):
    """Publish a new ordinary preparation from existing reviewed map bytes."""
    from .painted_transfer_replay import replay_transfer_review, _write_recipe
    from .painted_adoption_table import _bound
    from .painted_analysis_exclusions import prepare_exclusions
    from .map_adoption import PREPARATION_FORMAT_V2
    replay = replay_transfer_review(review_source, source_context=source_context,
        reuse_products=True, retain_arrays=False)
    preparation, folder = _write_recipe(replay, destination)
    session, painting = replay["session"], replay["painting_binding"]
    models = {}
    for row in preparation["runs"]:
        models[preparation["reference_run"]] = deepcopy(row["reference_model"])
        models[row["run_id"]] = deepcopy(row["target_model"])
    proof = dict(format=PROOF_FORMAT, review=deepcopy(replay["review"]["review_binding"]),
        source_context=deepcopy(source_context), models=models,
        products={row["run_id"]: deepcopy(row["base_product"]) for row in replay["review"]["runs"]},
        files=deepcopy(replay["source_bindings"]["files"]))
    preparation.update(format=PREPARATION_FORMAT_V2,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        approval=deepcopy(painting["approval"]), approved_source=deepcopy(painting),
        painting_geometry=proof, blockers=[], adopted=False,
        composition="Reuse exact reviewed native arrays; retain original H, support, inside and gap masks",
        regions=[dict(layer_id=int(layer.layer_id), name=layer.name,
            role=session.facies[layer.facies].role, facies=layer.facies, facies_id=None)
            for layer in session.layers])
    with (folder / "approval.json").open("x", encoding="utf-8") as stream:
        json.dump(painting["approval"], stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if chart_transfers:
        from .painting_chart_products import prepare_chart_products
        preparation["chart_transfers"] = prepare_chart_products(preparation, folder,
            chart_transfers, detection_manifest=detection_manifest)
    if chart_transfers:
        # Keep the historical source review in painting_geometry. The current
        # findings must describe the newly approved effective chart products.
        from .painted_analysis_exclusions import build_gap_review
        current_review = build_gap_review(preparation, folder)
        preparation["gap_review"] = deepcopy(current_review["review_binding"])
    else:
        current_review = replay["review"]
        preparation["gap_review"] = deepcopy(replay["review"]["review_binding"])
    from .painted_analysis_exclusions import GapReviewRequired
    if analysis_exclusions is not None:
        try:
            preparation["analysis_exclusions"] = prepare_exclusions(preparation, folder, analysis_exclusions)
        except ValueError as error:
            raise GapReviewRequired(current_review, reason=str(error)) from error
    elif any(row["gap"]["pixels"] for row in current_review["runs"]):
        # Retain the actual products and open their ordinary findings review.
        # An outside-frame proof cannot decide what an interior gap means.
        raise GapReviewRequired(current_review)
    verify_local_preparation(preparation, folder)
    from .painted_adoption_table import prepare_outside_proof
    outside = prepare_outside_proof(folder, painting, preparation["runs"], preparation=preparation)
    if outside is not None:
        preparation["outside_frame_proof"] = outside
    path = folder / "index.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(preparation, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(preparation, folder=str(folder))


def check_local_adoption(current, folder, *, facies_source_acquisition,
                         display_attachments=None):
    """Build the ordinary candidate and execute its one real geometry Check."""
    from .map_adoption import adoption_candidate
    from .local_geometry_check import check_geometry_candidate
    candidate, summary = adoption_candidate(current, folder,
        facies_source_acquisition=facies_source_acquisition)
    checked = check_geometry_candidate(candidate, display_attachments=display_attachments)
    return candidate, summary, checked


def save_checked_local_adoption(store, candidate, checked, *, actor, reason,
                                review_resolutions=None):
    """Save the actual unchanged Check through the ordinary revision Store."""
    return store.save_revision(candidate, actor=actor, reason=reason,
        checked_candidate=checked, review_resolutions=review_resolutions,
        preserve_failed_artifacts=True)


def verify_local_selected(preparation, folder, entry):
    """Verify one consumed map and its exact saved source/H, without other runs."""
    from .painted_adoption_table import _checked, _local
    from .photo_edge_continuation import validate_contract
    proof = preparation["painting_geometry"]
    if proof.get("format") != PROOF_FORMAT:
        raise ValueError("A selected reused map needs its genuine local proof")
    review = _read_local_review(proof["review"])
    selected = [row for row in review["runs"] if row["run_id"] == entry["run_id"]]
    if len(selected) != 1:
        raise ValueError("The selected map is not unique in its original review")
    original = selected[0]
    if (not _same(preparation["painting_facies"], review["source"]["painting_facies"])
            or not _same(preparation["map_version"], review["map_version"])
            or not _same(preparation["geometry_index"], review["geometry_index"])):
        raise ValueError("The selected reused map changed its source")
    contract = original["contract"]
    validate_contract(contract, verify_files=False)
    guarded = [review["review_binding"], review["geometry_index"],
               preparation["painting_facies"]["version"],
               *preparation["painting_facies"]["files"].values()]
    for name, side in (("reference_model", "source_model"), ("target_model", "target_model")):
        bound = entry[name]
        if (not _same(bound, original[name])
                or not _same(_checked(bound, metadata=True), contract[side]["document"])):
            raise ValueError("The selected reused map changed its actual model/H")
        guarded.append(bound)
    copied = _local(folder, entry["labels"])
    if (any(copied[k] != original["base_product"][k] for k in ("sha256", "size_bytes"))
            or entry["shape_hw"] != original["base_scientific_arrays"]["array"]["shape"]):
        raise ValueError("The selected reused map changed its grid/bytes")
    source = _local(folder, preparation["source_map"])
    if any(source[k] != review["source_map"][k] for k in ("sha256", "size_bytes")):
        raise ValueError("The selected reused map changed its source pixels")
    guarded.extend([copied, source])
    for bound in guarded:
        _checked(bound)
    return guarded


def validate_local_registered_run(config, payload, row, preparation, folder):
    """Original registered-run checks against the already reviewed map bytes."""
    import numpy as np
    from .painted_adoption_table import _local, _checked, _bound
    from .painted_target_integration import effective_product
    from .geometry_facies import _load_array, _darsia_metadata
    from .geometry_raster_sources import _same_grid_metadata
    entry = next(item for item in preparation["runs"] if item["run_id"] == row["run_id"])
    verify_local_selected(preparation, folder, entry)
    effective, bound = effective_product(preparation, folder, entry)
    record = row["registration"]
    table = config.get("painted_adoption_table") or {}
    if (not _same(record["measurement"], table.get("preparation"))
            or not _same(_checked(record["authority"], metadata=True),
                         preparation["painting_facies"]["approval"])
            or not _same(record["source_map"], _local(folder, preparation["source_map"]))
            or not _same(record["reference_model"], entry["reference_model"])
            or not _same(record["target_model"], entry["target_model"])):
        raise ValueError("Registered reused map lost its original source/approval/model join")
    run = row["run_id"]
    fields = ((((config.get("frame_binding") or {}).get("runs") or {}).get(run) or {}).get("fields") or {})
    photograph = fields.get("shape_corrected_baseline") or {}
    if (any(record["source_image"][k] != photograph.get(k) for k in ("path", "sha256", "size_bytes"))
            or Path(record["source_image"]["path"]).resolve() != Path(row["source_image_path"]).resolve()):
        raise ValueError("Registered reused map belongs to another bound photograph")
    actual = _checked(record["labels"])
    if (Path(actual["path"]).resolve() != Path(row["labels_path"]).resolve()
            or any(actual[k] != bound[k] for k in ("sha256", "size_bytes"))):
        raise ValueError("Registered labels differ from the exact reused effective product")
    labels = _load_array(Path(actual["path"]))
    if (labels.shape != payload["labels"].shape
            or labels.dtype != np.dtype("int32")
            or not np.array_equal(labels, effective["array"])):
        raise ValueError("Registered reused labels changed their native values/grid")
    base = _local(folder, entry["labels"])
    if not _same_grid_metadata(_darsia_metadata(Path(actual["path"])),
                               _darsia_metadata(Path(base["path"]))):
        raise ValueError("Registered reused labels changed their original grid metadata")
    if "analysis_exclusions" in config:
        from .painted_analysis_exclusions import configuration_exclusion
        configuration_exclusion(config, row["run_id"], labels=labels)
