"""Metadata-only planning and explicit runtime replay of adopted native inputs.

Planning pins the genuine immutable Store record and declared artifacts. It
does not decode NPZs, accept scientific masks, or infer policies from old TOML.
Only the runtime reader replays the original base and optional human products.
"""
from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from copy import deepcopy
from pathlib import Path

import numpy as np

FORMAT = "darsia-painted-native-product-v1"


def _declared_local(folder, value):
    """Resolve an immutable publication locator without reading its binary."""
    from .storage_paths import resolve_storage_path
    folder = resolve_storage_path(folder).resolve()
    name = value.get("path")
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("A declared native dependency needs its relative local path")
    target = (folder / name).resolve()
    if not target.is_relative_to(folder) or target == folder:
        raise ValueError("A declared native dependency leaves its preparation")
    return dict(path=str(target), sha256=value["sha256"], size_bytes=value["size_bytes"])


def _context(revision, run_id, project_root, registered_sources=None):
    from .painted_analysis_exclusions import _bound, _checked, _same
    from .geometry_facies import resolve_run_geometry_artifacts
    config = revision["configuration"]
    if ("analysis_exclusions" not in config and "target_corrections" not in config
            and "local_map_reuse" not in (config.get("provenance") or {})):
        return None
    root = Path(project_root).resolve()
    revision_id = revision["revision_id"]
    immutable = _bound(root / "results" / "geometry" / revision_id / "revision.json")
    if not _same(_checked(immutable, metadata=True), revision):
        raise ValueError("Native product planning requires the exact immutable adopted revision")
    table = config["painted_adoption_table"]
    preparation = _checked(table["preparation"], metadata=True)
    folder = Path(table["preparation"]["path"]).parent
    if (preparation.get("format") != "darsia-fluidflower-painted-map-adoption-preparation-v2"
            or not _same(preparation.get("analysis_exclusions"), config.get("analysis_exclusions"))
            or not _same(preparation.get("target_corrections"), config.get("target_corrections"))):
        raise ValueError("Native product planning lost the exact adopted V2 contracts")
    entry = next(row for row in preparation["runs"] if row["run_id"] == run_id and row.get("status") == "prepared")
    artifacts = resolve_run_geometry_artifacts(revision, run_id)
    labels_path = root / "results" / "geometry" / revision_id / "runs" / run_id / "labels.npz"
    from .storage_paths import same_storage_path, same_storage_descriptor
    if artifacts.get("scope") != "run_override" or not same_storage_path(artifacts["labels_path"], labels_path):
        raise ValueError("A native product requires its genuine selected-run Store artifact")
    target = ((preparation.get("target_corrections") or {}).get("runs") or {}).get(run_id)
    chart = (preparation.get("chart_transfers") or {}).get(run_id)
    effective = (chart["artifact"] if chart is not None else
                 entry["labels"] if target is None else target["artifact"])
    source = _declared_local(folder, effective)
    # Store publication already proved this exact byte copy. Planning projects
    # its declared identity; runtime/export independently check the actual file.
    labels = dict(source, path=str(labels_path))
    if registered_sources is not None and not same_storage_descriptor(registered_sources["files"]["labels"], labels):
        raise ValueError("The selected native source differs from the actual Store artifact")
    model = _checked(entry["target_model"], metadata=True)
    return dict(config=config, revision=immutable, preparation=preparation, preparation_binding=table["preparation"],
                folder=folder, entry=entry, source_labels=source, labels=labels, model=model, target=target)


def _gap_contract(context, run_id):
    if context is None:
        return None
    config, preparation = context["config"], context["preparation"]
    declared = preparation.get("analysis_exclusions")
    row = ((declared or {}).get("runs") or {}).get(run_id)
    if row is None:
        return None
    from .painted_analysis_exclusions import (_checked, approval_records, MEANING,
        APPROVAL_FORMAT_V2, EXCLUSION_FORMAT_V2)
    # Metadata action closure only. Mask/scientific replay is intentionally
    # deferred to read_native_analysis_exclusions inside the owned operation.
    approval = approval_records(_checked(declared["approval"], metadata=True))
    if ((declared.get("format") == EXCLUSION_FORMAT_V2) != (approval["format"] == APPROVAL_FORMAT_V2)):
        raise ValueError("The analysis exclusion and approval versions differ")
    provenance = dict(approval=deepcopy(declared["approval"]), run_id=run_id,
        native_mask=deepcopy(row), meaning=MEANING, physical_frame_unchanged=True, labels_unchanged=True)
    return dict(format="darsia-approved-analysis-gap-native-v1", meaning=MEANING, run_id=run_id,
        preparation=deepcopy(context["preparation_binding"]), approval=deepcopy(declared["approval"]),
        mask=deepcopy(row["artifact"]), field=row["field"], identity=deepcopy(row["identity"]),
        source_labels=deepcopy(context["source_labels"]), exported_labels=deepcopy(context["labels"]),
        source_model=deepcopy(context["entry"]["target_model"]), native_grid=deepcopy(context["model"]["baseline_native_grid"]),
        source_to_export_native_h=np.eye(3).tolist(), provenance=provenance)


def adopted_run_analysis_exclusions(revision, run_id, *, project_root, registered_sources=None):
    """The same literal metadata projection before Apply and during export."""
    return _gap_contract(_context(revision, run_id, project_root, registered_sources), run_id)


def adopted_run_native_product(revision, run_id, *, project_root, registered_sources=None):
    """An explicit new native-input policy, including fully painted/no-gap runs."""
    context = _context(revision, run_id, project_root, registered_sources)
    if context is None:
        return None
    return dict(format=FORMAT, run_id=run_id, adopted_revision=deepcopy(context["revision"]),
        preparation=deepcopy(context["preparation_binding"]), source_labels=deepcopy(context["source_labels"]),
        exported_labels=deepcopy(context["labels"]), source_model=deepcopy(context["entry"]["target_model"]),
        native_grid=deepcopy(context["model"]["baseline_native_grid"]), source_to_export_native_h=np.eye(3).tolist(),
        analysis_exclusions=_gap_contract(context, run_id),
        target_corrections=[] if context["target"] is None else deepcopy(context["target"]["bindings"]))


def native_analysis_exclusion_dependencies(contract, reader=None):
    """Declare the selected reader closure; never hash/decode binary ancestry.

    Only the preparation, selected continuation/model/renderer, and selected
    approval/review/version JSON are expanded. Binary descriptors are returned
    literally for the owned runtime reader. A preclaim ``reader.path`` callback
    must provide admission only, without imposing a new binary SHA sweep.
    This catalogue is not scientific acceptance or a whole-cohort replay.
    """
    from .painted_analysis_exclusions import _checked, _checked_gap_review, _json, APPROVAL_FORMAT_V2
    if contract is None:
        return []
    contract = _json(contract)
    result, documents = {}, {}
    def declare(value):
        bound = {key: value[key] for key in ("path", "sha256", "size_bytes")}
        if (type(bound["path"]) is not str or not Path(bound["path"]).is_absolute()
                or type(bound["sha256"]) is not str or len(bound["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in bound["sha256"])
                or type(bound["size_bytes"]) is not int or bound["size_bytes"] < 0):
            raise ValueError("A declared native dependency needs its exact literal descriptor")
        if reader is not None:
            reader.path(bound)
        result[(bound["path"], bound["sha256"])] = bound
        if len(result) > 8192:
            raise ValueError("Native product metadata exceeds the existing bound-file count")
        return bound
    def document(bound, *, gap_review=False):
        bound = declare(bound)
        if Path(bound["path"]).suffix.casefold() != ".json":
            raise ValueError("Only selected JSON metadata may be expanded by this catalogue")
        key = (bound["path"], bound["sha256"], gap_review)
        if key not in documents:
            documents[key] = _checked_gap_review(bound) if gap_review else _checked(bound, metadata=True)
        return documents[key]
    def literals(value):
        if isinstance(value, dict):
            if {"path", "sha256", "size_bytes"} <= set(value) and Path(value["path"]).is_absolute():
                declare(value)
            for item in value.values():
                literals(item)
        elif isinstance(value, list):
            for item in value:
                literals(item)
    def grid_inputs(grid):
        # Compatibility and inherited configuration fields are frozen DATA,
        # not current file reads. Do not turn them into restore-time SHA gates.
        for key in ("renderer", "raw_shape_basis_authority", "parent_workspace_revision"):
            literals((grid or {}).get(key))
    def painting_inputs(binding):
        literals(binding.get("version"))
        literals(binding.get("files"))
        basis = binding.get("frame_basis") or {}
        for key in ("workspace_revision", "photograph"):
            literals(basis.get(key))
        literals((basis.get("camera_profile") or {}).get("parameters"))
    def contract_inputs(value):
        literals(value.get("geometry_index"))
        for key in ("source_model", "target_model"):
            model = value.get(key) or {}
            literals(model.get("descriptor"))
            literals(model.get("photograph"))
        # frame_authority uses _document's self-checked embedded snapshots;
        # their historical project/TOML paths are not reopened by this replay.
    for key in ("adopted_revision", "preparation", "source_labels", "exported_labels", "source_model", "approval", "mask"):
        literals(contract.get(key))
    grid_inputs(contract.get("native_grid"))
    literals(contract.get("target_corrections"))
    preparation = document(contract["preparation"])
    folder = Path(contract["preparation"]["path"]).parent
    run = contract["run_id"]
    entry = next(row for row in preparation["runs"] if row["run_id"] == run)
    def local(value):
        return _declared_local(folder, value)
    for value in (preparation["source_map"], entry["labels"]):
        declare(local(value))
    if preparation.get("source_fixture_underlay") is not None:
        declare(local(preparation["source_fixture_underlay"]))
    painting_inputs(preparation["painting_facies"])
    literals(preparation["geometry_index"])
    continuation = preparation["continuation"]
    sidecar = local(dict(path=continuation["file"], sha256=continuation["sha256"], size_bytes=continuation["size_bytes"]))
    document(sidecar)  # Admit and bind bytes before the original pooled reader.
    from .photo_edge_continuation import read_contracts
    selected_contract = read_contracts(sidecar["path"])["runs"][run]
    contract_inputs(selected_contract)
    for name in ("reference_model", "target_model"):
        model = document(entry[name])
        literals(model.get("reference_image"))
        for key in ("source_model", "current_edges", "model_photograph", "baseline_to_model_registration"):
            literals(model.get(key))
        grid = model.get("baseline_native_grid") or {}
        grid_inputs(grid)
        if grid.get("renderer") is not None:
            renderer = document(grid["renderer"])
            literals((renderer.get("corrected_baseline") or {}).get("archive"))
            saved = renderer.get("renderer") or {}
            for key in ("source_npz", "raw_acquisition", "shape_corrections"):
                literals(saved.get(key))
            literals((saved.get("saved_shape_chain_replay") or {}).get("correction_assets"))
            inputs = renderer.get("inputs") or {}
            for key in ("raw", "profile", "operation_request"):
                literals(inputs.get(key))
            owner = renderer.get("output_ownership") or {}
            for key in ("request", "owner", "setup_started", "save_started", "save_completed"):
                literals(owner.get(key))
    def review(bound):
        value = document(bound, gap_review=True)
        painting_inputs(value["source"]["painting_facies"])
        literals(value["source"]["geometry_index"])
        literals(value["source_map"])
        row = next(item for item in value["runs"] if item["run_id"] == run)
        for key in ("product", "base_product", "reference_model", "target_model", "native_photo", "target_correction_bindings"):
            literals(row.get(key))
        contract_inputs(row["contract"])
    gap = (preparation.get("analysis_exclusions") or {}).get("runs", {}).get(run)
    if gap is not None:
        literals(gap)
        approval_bound = preparation["analysis_exclusions"]["approval"]
        approval = document(approval_bound)
        review(approval["review"])
        if approval.get("format") == APPROVAL_FORMAT_V2:
            from .painted_gap_regions import gap_region_dependencies
            for bound in gap_region_dependencies(approval, run_id=run):
                declare(bound)
    target = ((preparation.get("target_corrections") or {}).get("runs") or {}).get(run)
    if target is not None:
        literals(target["artifact"] if Path(target["artifact"]["path"]).is_absolute() else local(target["artifact"]))
        for binding in target["bindings"]:
            version = document(binding["version"])
            declare(binding["approval"])
            literals(version["product"])
            review(version["basis"]["review"])
    chart_review = ((preparation.get("chart_review") or {}).get("runs") or {}).get(run)
    if chart_review is not None:
        literals(chart_review["proposal"]["source_binding"])
    chart = (preparation.get("chart_transfers") or {}).get(run)
    if chart is not None:
        declare(local(chart["artifact"]))
    return list(result.values())


def verify_selected_native(preparation, folder, entry):
    """Replay the original source/base guards inside an actual owned operation."""
    from .painted_adoption_table import _local
    from .photo_edge_continuation import load_contract, verify_source, verify_product
    from .local_map_adoption import PROOF_FORMAT, verify_local_selected
    if (preparation.get("painting_geometry") or {}).get("format") == PROOF_FORMAT:
        return verify_local_selected(preparation, folder, entry)
    source = _local(folder, preparation["source_map"])
    with np.load(_preset_io.source(source["path"]), allow_pickle=False) as archive:
        layers = np.array(archive["array"], copy=True)
    run = entry["run_id"]
    contract = load_contract(Path(folder) / "continuation.json", run)
    verify_source(preparation["approved_source"], layers, {run: contract}, preparation=preparation, folder=folder)
    verify_product(_local(folder, entry["labels"])["path"], layers, contract, counts=entry.get("continuation"))
    # Whole-cohort native proof belongs to the immutable Store publication.
    # The native Rig caller independently proves this run's actual Build basis,
    # model/grid/H; repeating all other runs here would be a new cohort sweep.


def read_native_product(contract, *, actual_labels, run_id, reader=None):
    """Numerical replay for Rig; its caller independently proves actual Build H/grid."""
    from .painted_analysis_exclusions import _checked, _same, _json, read_native_analysis_exclusions
    from .painted_target_integration import effective_product
    contract = _json(contract)
    required = {"format", "run_id", "adopted_revision", "preparation", "source_labels", "exported_labels",
        "source_model", "native_grid", "source_to_export_native_h", "analysis_exclusions", "target_corrections"}
    if set(contract) != required or contract["format"] != FORMAT or contract["run_id"] != run_id:
        raise ValueError("The explicit native product contract changed")
    dependencies = native_analysis_exclusion_dependencies(contract, reader)
    revision = _checked(contract["adopted_revision"], metadata=True)
    root = Path(contract["adopted_revision"]["path"]).parent.parent.parent.parent
    expected = adopted_run_native_product(revision, run_id, project_root=root)
    if not _same(expected, contract):
        raise ValueError("The native product differs from its actual adopted metadata")
    preparation = _checked(contract["preparation"], metadata=True)
    folder = Path(contract["preparation"]["path"]).parent
    entry = next(row for row in preparation["runs"] if row["run_id"] == run_id)
    verify_selected_native(preparation, folder, entry)
    arrays, _ = effective_product(preparation, folder, entry)
    if (not isinstance(actual_labels, np.ndarray) or actual_labels.dtype != arrays["array"].dtype
            or not np.array_equal(actual_labels, arrays["array"])):
        raise ValueError("The actual Rig labels differ from the approved effective native product")
    mask, provenance = None, None
    if contract["analysis_exclusions"] is not None:
        result = read_native_analysis_exclusions(contract["analysis_exclusions"], actual_labels=actual_labels, run_id=run_id, reader=reader)
        mask, provenance = result["mask"], result["provenance"]
    return dict(mask=mask, provenance=provenance, contract=contract, native_grid=deepcopy(contract["native_grid"]),
                dependencies=dependencies)
