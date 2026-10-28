"""Positive reader for genuine N1 mass-only storage, never native-phase labels.

Only committed coarse products, scalar results and copied phase-one metadata are
opened. The old native-phase reader is not broadened. No collection, inference,
camera decoder or source-path fallback executes here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import math
import re

import numpy as np

from . import storage_outputs as output
from . import storage_saved_products as saved
from .storage_n1_outputs import FRAME_FORMAT, PRODUCER_KIND
from .storage_resolution import LABEL_LEGEND
from .pipeline import staged_data as staged


def _same(first, second):
    return staged._same_json(first, second)


def _bytes(bound, resolve, loaded):
    return saved._read_bytes(bound, resolve, loaded)


def _copy_identity(copy, original):
    saved._descriptor(copy)
    saved._descriptor(original)
    if any(copy[key] != original[key] for key in ("sha256", "size_bytes")):
        raise ValueError("Copied phase-one evidence differs from its original producer")


def _manifest_code(manifest):
    """Check the original N1 commit-ASCII identity, not a code-closure digest."""
    code = manifest.get("code")
    if (not isinstance(code, dict)
            or code.get("identity_algorithm") != "sha256-git-commit-ascii"
            or not isinstance(code.get("commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", code["commit"]) is None
            or manifest.get("code_sha256") != hashlib.sha256(code["commit"].encode("ascii")).hexdigest()):
        raise ValueError("Genuine N1 manifest code identity differs")
    return code["commit"]


def _phase1(document, manifest, spec, *, resolve, loaded):
    """Verify the copied exact original metadata, without following old paths."""
    evidence = document["phase1"]
    binding = manifest["storage"]["phase1"]
    if not _same(evidence["binding"], binding):
        raise ValueError("N1 saved evidence differs from the manifest's sealed phase one")
    _copy_identity(evidence["core_copy"], binding["completion"])
    core = saved._json(_bytes(evidence["core_copy"], resolve, loaded))
    order = manifest["storage"]["ordered_case_ids"]
    if (core.get("format") != staged.COMPLETION_FORMAT or core.get("complete") is not True
            or core.get("history_complete") is not True or core.get("run") != manifest["run"]
            or not _same(core.get("configuration"), binding["configuration"])
            or core.get("code_commit") != binding["code_commit"]
            or core.get("precision") != document["precision"]
            or core.get("ordered_case_ids") != order or set(binding["frames"]) != set(order)
            or not isinstance(core.get("frames"), list) or len(core["frames"]) != len(order)):
        raise ValueError("N1 copied phase-one run completion/order differs")
    case = document["case_id"]
    original = binding["frames"][case]
    if not _same(core["frames"][order.index(case)], original["commit"]):
        raise ValueError("N1 copied phase-one core does not commit this selected case")
    _copy_identity(evidence["commit_copy"], original["commit"])
    commit = saved._json(_bytes(evidence["commit_copy"], resolve, loaded))
    if (commit.get("format") != staged.FRAME_FORMAT or commit.get("complete") is not True
            or commit.get("case_id") != case or commit.get("run") != document["run"]
            or not _same(commit.get("configuration"), binding["configuration"])
            or commit.get("precision") != document["precision"]
            or not _same(commit.get("source"), document["source"])
            or not _same(commit.get("source", {}).get("source_binding"), spec["source_binding"])
            or commit.get("source", {}).get("source_sha256") != spec["source_binding"]["source_sha256"]
            or not _same(commit.get("frame_index"), spec["frame_index"])
            or not _same(commit.get("elapsed_h"), spec["elapsed_h"])
            or not _same(commit.get("native", {}).get("full"), spec["current"])
            or commit.get("native", {}).get("complete") is not True
            or commit.get("phase", {}).get("complete") is not True
            or any(commit.get(part, {}).get("source_sha256") != spec["source_binding"]["source_sha256"]
                   for part in ("native", "phase"))
            or any(not _same(commit.get("phase", {}).get(key), spec[key]) for key in ("labels", "classes", "area"))):
        raise ValueError("N1 copied phase-one case/source/final-phase identity differs")
    for key in ("statistics", "before_rules", "photograph"):
        if not _same(commit.get(key), original.get(key)):
            raise ValueError("N1 copied saved evidence differs from its sealed case")
    for key in ("labels", "classes", "area"):
        staged._owned(commit["phase"][key], commit["files"], where="copied phase-one " + key, array=key == "area")
    staged._owned(commit["before_rules"]["receipt"], commit["files"], where="copied BEFORE receipt")
    _copy_identity(evidence["before_copy"], commit["before_rules"]["receipt"])
    before = saved._json(_bytes(evidence["before_copy"], resolve, loaded))
    if (before.get("format") != staged.BEFORE_FORMAT or before.get("complete") is not True
            or not _same(before.get("source"), commit["source"])
            or not _same(before.get("before_rules"), document["before_rules"])
            or not _same(before.get("png"), commit["before_rules"].get("png"))
            or before["before_rules"]["state"] != commit["before_rules"]["state"]):
        raise ValueError("N1 copied BEFORE source/metadata differs")
    staged._before_metadata(before["before_rules"], list(document["coarse"]["grid"]["native_shape"]))
    return commit


def read_n1_stored_frame(record, *, manifest, manifest_asset, resolve_asset, load_before=False):
    """Read one genuine N1 commit and optional storage without native arrays."""
    _manifest_code(manifest)
    loaded, case = [], record["frame_id"]
    if not isinstance(case, str) or re.fullmatch(r"[A-Za-z0-9_-]+", case) is None:
        raise ValueError("Explicit N1 frame ID required")
    commit = saved._json(_bytes(saved._record_asset(record, "commit.json"), resolve_asset, loaded))
    if ({name: value for name, value in record["files"].items() if name != "commit.json"} != commit.get("files")
            or commit.get("complete") is not True or commit.get("frame_id") != case
            or commit.get("job_id") != manifest["job_id"]
            or commit.get("manifest_sha256") != manifest_asset["sha256"]
            or record.get("manifest_sha256") != manifest_asset["sha256"]):
        raise ValueError("N1 original committed frame inventory/manifest differs")
    result_bound = saved._record_asset(record, "result.json")
    result = saved._json(_bytes(result_bound, resolve_asset, loaded))
    receipt = saved._record_asset(record, "storage_completed.json")
    document = saved._json(_bytes(receipt, resolve_asset, loaded))
    spec = manifest["frames"][case]
    if (document.get("format") != FRAME_FORMAT or document.get("producer_kind") != PRODUCER_KIND
            or document.get("complete") is not True or document.get("run") != manifest["run"]
            or document.get("case_id") != case or document.get("job_id") != manifest["job_id"]
            or not _same(document.get("configuration"), manifest["configuration"])
            or document.get("manifest_sha256") != manifest_asset["sha256"]
            or not _same(document.get("source_binding"), spec["source_binding"])
            or document.get("existing_native_products_unchanged") is not True
            or any(document["result"][key] != result_bound[key] for key in ("path", "sha256", "size_bytes"))
            or result.get("complete") is not True or result.get("case_id") != case or result.get("run") != manifest["run"]
            or not _same(result.get("source_binding"), spec["source_binding"])
            or not _same(result.get("frame_index"), spec["frame_index"])
            or not _same(result.get("elapsed_h"), spec["elapsed_h"])):
        raise ValueError("Genuine N1 saved producer/result/source identity differs")
    backends = (manifest["settings"].get("backend"), commit.get("settings", {}).get("backend"),
                commit.get("products", {}).get("backend"), result.get("backend"))
    if backends[0] not in {"cpu", "cuda"} or any(value != backends[0] for value in backends):
        raise ValueError("N1 saved backend differs across manifest/commit/result")
    precision = document["precision"]
    code = document["code_commit"]
    if (precision not in {"float32", "float64"} or re.fullmatch(r"[0-9a-f]{40}", code) is None
            or code != manifest["code"]["commit"]
            or any(v != precision for v in (manifest["settings"]["precision"], result["precision"],
                result["working_pixel_dtype"], commit["products"]["precision"], commit["settings"]["precision"]))
            or result["bindings"]["code_sha256"] != hashlib.sha256(code.encode("ascii")).hexdigest()
            or result["bindings"]["source_sha256"] != spec["source_binding"]["source_sha256"]
            or result["bindings"]["phase_sha256"] != spec["labels"]["sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", document["calibration_sha256"]) is None
            or document["calibration_sha256"] != result["bindings"]["calibration_sha256"]
            or not _same(document["request"], manifest["storage"]["request"])):
        raise ValueError("N1 saved configuration/freeze/code/precision differs")
    saved._request(document)
    if document.get("extensive", {}).get("qualification") == "genuine-n15-n16-source-producer":
        from .storage_extensive_io import verify_producer_document
        binding = document.get("n16_extensive_producer")
        verify_producer_document(binding, document)
        if (not _same(binding, result.get("n16_extensive_producer"))
                or not _same(binding["phase"], spec["labels"])
                or not _same(binding["classes"], spec["classes"])
                or not _same(binding["area"], spec["area"])
                or not _same(binding["source_geometry"], spec["pixel_domain_geometry"])
                or not _same(binding["frozen_response_basis"], spec["frozen_response_basis"])
                or not _same(binding["native_products"], result["products"])
                or not _same(binding["native_geometry"], result["static"])
                or binding["chronological_position"] != result["chronological_position"]
                or binding["frame_index"] != result["frame_index"]
                or binding["elapsed_h"] != result["elapsed_h"]
                or binding["n16"]["U_total_g"] != result.get("quantified_U_g", result["U_g"])
                or binding["n16"]["M_edge_g"] != result["M_edge_g"]
                or binding["n16"]["M_measured_g"] != result["M_measured_g"]
                or not _same(document["native_transport_weight_l_identity"], binding["native_geometry"]["arrays"]["transport_weight_l"])
                or not _same(document["native_mass_density_identities"],
                    {key: binding["provider_map_identities"][key] for key in output.DENSITIES.values()})):
            raise ValueError("N15/N16 actual producer/configuration/geometry/chronology differs")
    by_name = {}
    for bound in document["files"]:
        from pathlib import PureWindowsPath
        name = PureWindowsPath(bound["path"]).name
        if name in by_name or not name.startswith("storage_") or name not in record["files"]:
            raise ValueError("N1 saved storage inventory has an unknown/repeated file")
        if not _same(bound, saved._record_asset(record, name)):
            raise ValueError("N1 saved storage file is outside its actual commit")
        by_name[name] = bound
    expected_names = {"storage_phase1_completed.json", "storage_phase1_commit.json",
                      "storage_phase1_before.json", "storage_stored_frame.npz"}
    added = {}
    if document.get("extensive") is not None:
        from .storage_extensive_io import declared_files
        added = declared_files(document["extensive"])
    if any(not _same(by_name.get(name), row) for name, row in added.items()):
        raise ValueError("Extensive I/O file is outside its committed N1 entry")
    expected_names |= added.keys()
    if document["before_rules"]["state"] == "complete":
        expected_names.add("storage_gas_probability_before_rules.png")
    if document["photograph"]["state"] == "complete":
        expected_names |= {"storage_phase1_photo.json", "storage_original_photo.json", "storage_original_photo.png"}
    if set(by_name) != expected_names:
        raise ValueError("N1 storage has an incomplete or unexplained flat file inventory")
    if {name for name in record["files"] if name.startswith("storage_")} != expected_names | {"storage_completed.json"}:
        raise ValueError("N1 committed storage inventory differs from its declared flat files")
    for key, name in (("core_copy", "storage_phase1_completed.json"),
                      ("commit_copy", "storage_phase1_commit.json"),
                      ("before_copy", "storage_phase1_before.json")):
        if not _same(document["phase1"][key], by_name[name]):
            raise ValueError("N1 evidence locator is outside the committed storage inventory")
    phase1 = _phase1(document, manifest, spec, resolve=resolve_asset, loaded=loaded)
    before = document["before_rules"]
    if (before["state"] == "complete") != ("storage_gas_probability_before_rules.png" in by_name):
        raise ValueError("N1 saved BEFORE PNG state differs")
    if before["state"] == "complete":
        staged._owned(phase1["before_rules"]["png"], phase1["files"], where="copied BEFORE PNG")
        _copy_identity(by_name["storage_gas_probability_before_rules.png"], phase1["before_rules"]["png"])
    after = document["after_rules"]
    if (not _same(after["source"], spec["labels"]) or after.get("label_legend") != LABEL_LEGEND
            or after.get("support_mask") is not None or after.get("values") != [0, 1]):
        raise ValueError("N1 AFTER does not bind the unchanged final phase labels")
    arrays, grid, shape = saved._archive(_bytes(by_name["storage_stored_frame.npz"], resolve_asset, loaded), document)
    counts = document["final_label_counts"]
    if (set(counts) != set(LABEL_LEGEND) or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values()) != math.prod(shape) or after["gas_pixels"] != counts["2"]):
        raise ValueError("N1 saved final-label count closure differs")
    totals = {}
    document_totals = output.checked_mass_totals(document, result, arrays=arrays, public=True)
    source_totals = output.quantified_totals(result)
    for component, name in output.TOTALS.items():
        target = saved._finite(source_totals[component], component + " N1 scalar mass")
        cells = arrays["cell_mass_" + component + "_g"]
        actual = float(cells.sum(dtype=np.float64))
        check = document["native_pixel_mass_checks"][component]
        if (not np.isfinite(cells).all() or np.any(cells < 0)
                or document_totals[component] != target
                or abs(actual - target) > 1e-6 * abs(target)
                or document["coarse"]["native_mass_checks"][component] != dict(native_total_g=target,
                    stored_cell_sum_g=actual, absolute_difference_g=abs(actual-target), relative_limit=1e-6, passed=True)
                or check.get("native_total_g") != target or check.get("passed") is not True
                or check.get("relative_limit") != 1e-6
                or abs(saved._finite(check["pixel_sum_g"], "pixel mass") - target) > 1e-6 * abs(target)):
            raise ValueError("N1 saved/native " + component + " mass closure differs")
        totals[component] = actual
    native_before = None
    if load_before and before["state"] == "complete":
        from PIL import Image
        with Image.open(io.BytesIO(_bytes(by_name["storage_gas_probability_before_rules.png"], resolve_asset, loaded))) as image:
            if image.mode != "LA":
                raise ValueError("N1 saved BEFORE must remain literal LA")
            la = np.asarray(image).copy()
        if la.dtype != np.uint8 or la.shape != (*shape, 2) or not np.isin(la[..., 1], (0, 255)).all():
            raise ValueError("N1 saved BEFORE support/grid differs")
        native_before = la[..., 0].astype(precision) / np.dtype(precision).type(255)
        native_before[la[..., 1] == 0] = np.nan
        native_before.setflags(write=False)
    photo = document["photograph"]
    if photo["state"] == "complete":
        original = phase1["photograph"]
        if original.get("state") != "complete" or any(not _same(photo[key], by_name[name])
                for key, name in (("original_receipt_copy", "storage_phase1_photo.json"),
                                  ("receipt", "storage_original_photo.json"), ("png", "storage_original_photo.png"))):
            raise ValueError("N1 photograph escapes its committed storage inventory")
        _copy_identity(photo["original_receipt_copy"], original["receipt"])
        staged._owned(original["receipt"], phase1["files"], where="phase-one original-photo receipt")
        staged._owned(original["png"], phase1["files"], where="phase-one original-photo PNG")
        _copy_identity(photo["png"], original["png"])
    elif (photo.get("state") != "pending" or not photo.get("reason")
          or phase1.get("photograph", {}).get("state") == "complete"):
        raise ValueError("N1 saved original photograph state differs")
    for array in arrays.values():
        array.setflags(write=False)
    value = dict(format="darsia-read-coarse-stored-frame-v1", producer_kind=PRODUCER_KIND,
        run=document["run"], case_id=case, precision=precision, grid=deepcopy(grid), imshow_origin="lower",
        arrays=arrays, before_rules=deepcopy(before), before_native=native_before,
        photograph=deepcopy(photo), phase1_photograph=deepcopy(phase1.get("photograph")),
        native_scalar_reference=deepcopy(result), mass_totals_g=totals,
        source=deepcopy(document["source"]), loaded_assets=loaded,
        storage_receipt=deepcopy(receipt), storage_document=deepcopy(document))
    return output.read_mass_scope(value, result, totals)


def read_n1_original_photo(stored, *, resolve_asset, geometry_producer_sha256):
    """Photo availability is independent of scalar/mass reads and checked here."""
    photo = stored["photograph"]
    if photo["state"] != "complete":
        raise FileNotFoundError("N1 saved original photograph is explicitly pending")
    original = stored["phase1_photograph"]
    loaded = []
    _copy_identity(photo["original_receipt_copy"], original["receipt"])
    original_photo = saved._json(_bytes(photo["original_receipt_copy"], resolve_asset, loaded))
    copied_photo = saved._json(_bytes(photo["receipt"], resolve_asset, loaded))
    expected = deepcopy(original_photo)
    expected["photograph"] = {key: photo["png"][key] for key in ("path", "sha256", "size_bytes")}
    expected["phase1_original_receipt"] = original["receipt"]
    if not _same(expected, copied_photo):
        raise ValueError("N1 original-photo receipt changed more than its copied output locator")
    from .storage_saved_export import _photograph
    rgba, evidence = _photograph(photo["receipt"], stored, resolve_asset=resolve_asset,
                                 geometry_producer_sha256=geometry_producer_sha256)
    evidence["loaded_phase1_receipt_assets"] = loaded
    return rgba, evidence


def read_n1_stored_run(completed_metadata, *, resolve_asset, load_before=False, stream=False):
    """An explicit N1 mass-only reader; the native-phase API stays unchanged."""
    if type(stream) is not bool or type(load_before) is not bool:
        raise ValueError("N1 saved read options must be explicit booleans")
    value, loaded = deepcopy(completed_metadata), []
    records, order = value["records"], value["ordered_case_ids"]
    if (value.get("complete") is not True or "pipeline" in value
            or type(value["frames"]) is not int or value["frames"] != len(records)
            or value["case_count"] != len(records) or not records
            or order != [r["frame_id"] for r in records] or len(set(order)) != len(order)):
        raise ValueError("An original complete N1 mass-only run/order is required")
    manifest = saved._json(_bytes(value["manifest"], resolve_asset, loaded))
    if (manifest.get("format") != "darsia-n1-run-manifest-v1"
            or manifest.get("run") != value["run"]
            or not _same(manifest.get("configuration"), value["configuration"])
            or manifest.get("settings", {}).get("precision") != value["precision"]
            or set(manifest["frames"]) != set(order)
            or manifest["storage"].get("ordered_case_ids") != order
            or manifest.get("storage", {}).get("producer_kind") != PRODUCER_KIND):
        raise ValueError("Genuine N1 storage manifest/configuration/order required")
    _manifest_code(manifest)
    if (value.get("backend") not in {"cpu", "cuda"}
            or manifest["settings"].get("backend") != value["backend"]):
        raise ValueError("N1 saved completion backend differs from its manifest")
    def frames():
        for record in records:
            item = read_n1_stored_frame(record, manifest=manifest, manifest_asset=value["manifest"],
                                       resolve_asset=resolve_asset, load_before=load_before)
            if item["storage_document"]["calibration_sha256"] != value["calibration"]["sha256"]:
                raise ValueError("N1 saved freeze differs from the genuine completion")
            yield item
            del item
    return dict(format="darsia-read-coarse-stored-run-v1", producer_kind=PRODUCER_KIND,
        run=value["run"], precision=value["precision"], ordered_case_ids=list(order),
        frames=frames() if stream else list(frames()), loaded_manifest_assets=loaded,
        original_completion=value)


def read_indexed_n1_stored_run(index, *, current_machine_id, collections=(), source_paths=None,
                              load_before=False, stream=False):
    """Resolve only indexed/collected original identities, never foreign paths."""
    from .storage_indexed_products import indexed_asset_resolver
    from .pipeline import products_publication
    resolve = indexed_asset_resolver(index, current_machine_id=current_machine_id,
                                     collections=collections, source_paths=source_paths)
    completion = saved._json(products_publication.read_published_metadata(index))
    return read_n1_stored_run(completion, resolve_asset=resolve, load_before=load_before, stream=stream)
