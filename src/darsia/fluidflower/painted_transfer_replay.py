"""Original source/product replay for an early transfer findings list.

Returned arrays and recipes are checked DATA.  They are not a completed
preparation, a Store candidate, a physical classification or an approval.
The optional writer creates a separate residual review through the original
target-correction and gap-review writers, preserving every original artifact.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np


_DESCRIPTOR = {"path", "sha256", "size_bytes"}
_FIXTURE = {"source_fixture_underlay", "painting_analysis_derivation", "fixture_frame"}
_CHART_RECIPE = {"painting_geometry", "chart_review", "chart_transfers", "target_corrections"}


def _read_review(source):
    from .painted_analysis_exclusions import _checked, _same, read_gap_review
    if isinstance(source, dict) and set(source) == _DESCRIPTOR:
        _checked(source)
        result = read_gap_review(source["path"])
        if not _same(result["review_binding"], source):
            raise ValueError("The transfer review differs from its supplied descriptor")
        return result
    return read_gap_review(source)


def source_context_from_preparation(preparation, folder):
    """Retain original producer locators without creating publication authority."""
    from .painted_analysis_exclusions import _bound
    from .painted_adoption_table import _local
    folder = Path(folder)
    result = dict(continuation=_bound(folder / "continuation.json"), run_entries={})
    if preparation.get("source_fixture_underlay") is not None:
        result.update(source_fixture_underlay=_local(folder, preparation["source_fixture_underlay"]),
            painting_analysis_derivation=deepcopy(preparation.get("painting_analysis_derivation")),
            fixture_frame=deepcopy(preparation.get("fixture_frame")))
    for row in preparation["runs"]:
        if row.get("status") == "prepared" and row.get("fixture_underlay") is not None:
            result["run_entries"][row["run_id"]] = dict(
                fixture_underlay=_local(folder, row["fixture_underlay"]),
                fixture_underlay_layer_ids=deepcopy(row.get("fixture_underlay_layer_ids")))
    if preparation.get("source_recapture") is not None:
        result["recapture"] = deepcopy(preparation["source_recapture"])
    return result


def _local_descriptor(folder, descriptor):
    """Re-express one explicit bound artifact through the original local reader."""
    from .painted_adoption_table import _checked, _local
    if not isinstance(descriptor, dict) or set(descriptor) != _DESCRIPTOR:
        raise ValueError("An exact source artifact descriptor is required")
    actual = _checked(descriptor)
    root, path = Path(folder).resolve(), Path(actual["path"]).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("The transfer source artifact leaves its original preparation")
    relative = dict(actual, path=path.relative_to(root).as_posix())
    _local(root, relative)
    return relative


def _context(review, source_context, folder):
    from .painted_analysis_exclusions import _checked, _json, _same
    context = _json(source_context)
    if (not isinstance(context, dict) or "continuation" not in context
            or set(context) - ({"continuation", "run_entries", "recapture"} | _FIXTURE)):
        raise ValueError("The transfer needs its explicit continuation and fixture source context")
    present = set(context) & _FIXTURE
    if present and present != _FIXTURE:
        raise ValueError("A fixture source needs its underlay, derivation and exact frame")
    recapture = context.get("recapture")
    recapture_sources = []
    if recapture is not None:
        if not _same(recapture, review.get("source_recapture")):
            raise ValueError("The current source recapture is not bound by this review")
        captured = _checked(recapture, metadata=True)
        if (captured.get("origin") != "CURRENT_VERIFIED_RECAPTURE"
                or captured.get("original_serialized_binding_claimed") is not False
                or captured.get("source_map_exact") is not True
                or captured.get("source_frame_inside_exact") is not True
                or Path(captured.get("preparation", "")).resolve() != Path(folder).resolve()
                or not _same(captured.get("painting_binding"), review["source"]["painting_facies"])):
            raise ValueError("The retained transfer needs its exact current verified recapture")
        for descriptor in [*captured["verified_inputs"], *captured["bound_anchors"]]:
            _checked(descriptor)
            recapture_sources.append(descriptor)
        if captured.get("fixture_analysis_derivation") is not None:
            if (present != _FIXTURE
                    or not any(_same(context["source_fixture_underlay"], item)
                               for item in captured["verified_inputs"])
                    or not _same(context["painting_analysis_derivation"], captured["fixture_analysis_derivation"])
                    or not _same(context["fixture_frame"], captured["painting_binding"]["frame"])):
                raise ValueError("The recovered transfer lost its explicitly bound fixture source")
    elif not _same(context, review.get("preparation_context")):
        raise ValueError("The transfer source context is not bound by the original review")
    continuation = _checked(context["continuation"])
    if Path(continuation["path"]).resolve() != (Path(folder) / "continuation.json").resolve():
        raise ValueError("The transfer continuation is not its original preparation sidecar")
    run_entries = context.setdefault("run_entries", {})
    if (not isinstance(run_entries, dict)
            or set(run_entries) - {row["run_id"] for row in review["runs"]}):
        raise ValueError("The transfer fixture context names a run outside its review")
    for row in run_entries.values():
        if not isinstance(row, dict) or set(row) != {"fixture_underlay", "fixture_underlay_layer_ids"}:
            raise ValueError("A native fixture context needs its exact artifact and original layer IDs")
    return context, recapture_sources


def _target_groups(review, bindings):
    from .painted_analysis_exclusions import _checked, _json, _same
    from .painted_target_integration import _selected_bindings
    selected = _json(list(bindings))
    if not selected:
        selected = [deepcopy(binding) for row in review["runs"]
                    for binding in row["target_correction_bindings"]]
    if not selected:
        return [], {}
    groups = {}
    for binding in selected:
        run = _checked(binding["version"], metadata=True)["basis"]["run_id"]
        if run not in {row["run_id"] for row in review["runs"]}:
            raise ValueError("A target correction belongs to a run outside this transfer")
        groups.setdefault(run, dict(bindings=[]))["bindings"].append(binding)
    contract = dict(bindings=selected, runs=groups)
    for run in groups:
        _selected_bindings(contract, run)
    for row in review["runs"]:
        retained = row["target_correction_bindings"]
        current = groups.get(row["run_id"], {}).get("bindings", [])
        if not _same(current[:len(retained)], retained):
            raise ValueError("Current target corrections must retain every reviewed run's exact ordered prior edits")
    return selected, {run: row["bindings"] for run, row in groups.items()}


def chart_replay_recipe(preparation):
    """Retain the exact chart recipe, including before an index can be written."""
    return {key: deepcopy(preparation[key]) for key in _CHART_RECIPE if key in preparation}


def _checked_chart_recipe(review, folder):
    """Authenticate current chart DATA against its immutable original source."""
    from .painted_analysis_exclusions import _bound, _checked, _same
    from .local_map_adoption import PROOF_FORMAT, verify_local_geometry

    source = review.get("chart_source_review")
    if source is None:
        if "chart_preparation" in review:
            raise ValueError("A chart replay recipe needs its exact original source review")
        return {}, []
    original = _read_review(source)
    if (original.get("chart_source_review") is not None
            or any(not _same(original[key], review[key]) for key in ("source", "map_version", "geometry_index"))
            or [row["run_id"] for row in original["runs"]] != [row["run_id"] for row in review["runs"]]
            or any(original["source_map"][key] != review["source_map"][key] for key in ("sha256", "size_bytes"))):
        raise ValueError("Current chart findings lost their original approved source, geometry or cohort")
    guards = [original["review_binding"]]
    recipe = review.get("chart_preparation")
    if recipe is None:
        # Published reviews predating the embedded recipe retain it in their
        # exact same-folder preparation. Never fall back to the old masks.
        index = _bound(Path(folder) / "index.json")
        preparation = _checked(index, metadata=True)
        if not _same(preparation.get("gap_review"), review["review_binding"]):
            raise ValueError("The chart preparation does not bind this exact current review")
        recipe = chart_replay_recipe(preparation)
        guards.append(index)
    if (not isinstance(recipe, dict) or set(recipe) - _CHART_RECIPE
            or not {"painting_geometry", "chart_review", "chart_transfers"} <= set(recipe)
            or (recipe["painting_geometry"] or {}).get("format") != PROOF_FORMAT
            or not _same(recipe["painting_geometry"].get("review"), source)
            or not _same((recipe["chart_review"] or {}).get("review"), source)):
        raise ValueError("Current chart findings need their exact approved local chart recipe")
    models = {}
    for row, old in zip(review["runs"], original["runs"]):
        if (any(row["base_product"][key] != old["base_product"][key] for key in ("sha256", "size_bytes"))
                or any(not _same(row[key], old[key]) for key in
                    ("base_scientific_arrays", "contract", "reference_model", "target_model", "target_correction_bindings"))):
            raise ValueError("Current chart findings changed their original native base or target history")
        models[review["source"]["reference_run"]] = row["reference_model"]
        models[row["run_id"]] = row["target_model"]
    guards.extend(verify_local_geometry(review["source"]["painting_facies"],
        recipe["painting_geometry"], models, folder))
    approval = _bound(Path(folder) / "approval.json")
    if not _same(_checked(approval, metadata=True), review["source"]["painting_facies"]["approval"]):
        raise ValueError("Current chart findings changed the approved painting action")
    guards.append(approval)

    def collect(value):
        if isinstance(value, dict):
            if _DESCRIPTOR <= set(value) and Path(value["path"]).is_absolute():
                guards.append(_checked({key: value[key] for key in _DESCRIPTOR}))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(recipe["chart_review"])
    return deepcopy(recipe), guards


def replay_transfer_review(review_source, *, source_context, target_bindings=(),
                           reuse_products=False, retain_arrays=True, retain_source_arrays=True):
    """Replay every existing guard; optionally omit unused source arrays from DATA."""
    from .painted_analysis_exclusions import (_arrays, _checked, _components, _identity,
        _same, _source, target_paint_science)
    from .painted_adoption_table import (checked_fixture_analysis, verify_version_summary,
        _checked_native_underlay)
    from .photo_edge_continuation import (POLICY, read_contracts, verify_source, verify_product,
        verify_placement_inputs)
    from .map_placement import read_run_geometry
    from .painted_target_integration import FORMAT, _replay, target_sources

    if any(type(value) is not bool for value in (reuse_products, retain_arrays, retain_source_arrays)):
        raise ValueError("Transfer reuse options must be explicit booleans")
    review = _read_review(review_source)
    source_map = _checked(review["source_map"])
    folder = Path(source_map["path"]).resolve().parent
    context, recapture_sources = _context(review, source_context, folder)
    chart_recipe, chart_guards = _checked_chart_recipe(review, folder)
    contracts = read_contracts(context["continuation"]["path"])["runs"]
    runs = [row["run_id"] for row in review["runs"]]
    if set(contracts) != set(runs):
        raise ValueError("The transfer and original continuation have different runs")
    with np.load(source_map["path"], allow_pickle=False) as archive:
        source = np.array(archive["array"], copy=True)
    if not _same(_identity(source), review["source_map_pixels"]):
        raise ValueError("The transfer source pixels differ from its retained review")
    painting = review["source"]["painting_facies"]
    reference = review["source"]["reference_run"]
    preparation = dict(format="darsia-fluidflower-painted-map-adoption-preparation-v2",
        reference_run=reference, painting_facies=deepcopy(painting),
        geometry_index=deepcopy(review["geometry_index"]), map_version=deepcopy(review["map_version"]),
        source_map=_local_descriptor(folder, source_map), runs=[], approved_source=deepcopy(painting),
        continuation=dict(policy=POLICY, file="continuation.json", adopted=False,
            **{name: context["continuation"][name] for name in ("sha256", "size_bytes")}))
    preparation.update(chart_recipe)
    if not _same(_source(preparation), review["source"]):
        raise ValueError("The transfer source, reference, index or policy changed")
    if review.get("display_context") is not None:
        preparation["display_context"] = deepcopy(review["display_context"])
    if context.get("recapture") is not None:
        preparation["source_recapture"] = deepcopy(context["recapture"])
    fixture_source, fixture_ids = None, set()
    if context.get("source_fixture_underlay") is not None:
        from .layer_painting_fixtures import read_underlay
        preparation.update(source_fixture_underlay=_local_descriptor(folder, context["source_fixture_underlay"]),
            painting_analysis_derivation=deepcopy(context["painting_analysis_derivation"]),
            fixture_frame=deepcopy(context["fixture_frame"]))
        fixture_source, derivation = checked_fixture_analysis(painting, read_underlay(context["source_fixture_underlay"]))
        if (not _same(derivation, context["painting_analysis_derivation"])
                or not _same(context["fixture_frame"], painting["frame"])):
            raise ValueError("The transfer fixture analysis or frame changed")
        fixture_ids = {int(value) for value in np.unique(fixture_source.drawn_layers) if value}
    verify_version_summary(painting, preparation["map_version"])
    session, approved_analysis, _ = verify_source(painting, source, contracts,
        preparation=preparation, folder=folder)
    _checked(review["geometry_index"])
    models = read_run_geometry(review["geometry_index"]["path"])
    verify_placement_inputs(contracts, models, review["geometry_index"]["path"], reference, runs, "native")
    selected, groups = _target_groups(review, target_bindings)
    result = []
    guards = [review["review_binding"], source_map, context["continuation"], review["geometry_index"],
              painting["version"], *painting["files"].values(), *recapture_sources,
              painting["frame_basis"]["workspace_revision"], painting["frame_basis"]["photograph"], *chart_guards]
    if painting["frame_basis"]["camera_profile"]["kind"] == "camera_u":
        guards.append(painting["frame_basis"]["camera_profile"]["parameters"])
    if context.get("source_fixture_underlay") is not None:
        guards.append(context["source_fixture_underlay"])
    if context.get("recapture") is not None:
        guards.append(context["recapture"])
    for row in review["runs"]:
        run = row["run_id"]
        if not _same(row["contract"], contracts[run]):
            raise ValueError("The transfer changed its original continuation contract")
        if (not _same(row["reference_model"], contracts[run]["source_model"]["descriptor"])
                or not _same(row["target_model"], contracts[run]["target_model"]["descriptor"])):
            raise ValueError("The transfer product names another original model")
        target_h = (np.asarray(models[run].model["baseline_native_grid"]["H_baseline_camera_u_to_rig_corrected_native"], float)
                    @ models[run].metric_to_grid)
        composition = target_h @ np.linalg.inv(models[reference].metric_to_grid)
        if not _same(np.linalg.inv(composition).tolist(), row["native_to_painting_h"]):
            raise ValueError("The transfer changed its original model composition")
        base = _checked(row["base_product"])
        base_arrays = _arrays(base)
        if not _same({name: _identity(value) for name, value in base_arrays.items()}, row["base_scientific_arrays"]):
            raise ValueError("The transfer base scientific arrays changed")
        if reuse_products and run not in groups:
            # These exact archives and all scientific array identities are
            # already retained by the genuine original review. Reusing them
            # needs no second categorical carrier/RAW-basis computation.
            # TARGET replay independently needs the full original counts,
            # including any frame-boundary samples, from verify_product.
            replay = dict(counts=dict(
                continued_pixels=int(base_arrays["continued"].sum()),
                inside_gap_pixels=int(base_arrays["gaps"].sum()),
                inside_pixels=int(base_arrays["inside"].sum()),
                original_support_pixels=int(base_arrays["support"].sum()),
                effective_coverage_pixels=int(base_arrays["effective_coverage"].sum())))
        else:
            replay = verify_product(base["path"], source, contracts[run])
        reviewed = base_arrays if _same(row["product"], base) else _arrays(row["product"])
        if not _same({name: _identity(value) for name, value in reviewed.items()}, row["scientific_arrays"]):
            raise ValueError("The original reviewed effective arrays changed")
        if not _same(_identity(reviewed["gaps"]), row["gap"]["identity"]):
            raise ValueError("The original reviewed gap identity changed")
        old_components = ([component for component, _ in _components(reviewed["gaps"])]
                          if not reuse_products else None)
        declared_components = [{name: value for name, value in component.items() if name != "review_row_sha256"}
                               for component in row["components"]]
        if old_components is not None and not _same(old_components, declared_components):
            raise ValueError("The original reviewed connected components changed")
        entry = dict(run_id=run, status="prepared", labels=_local_descriptor(folder, base),
            reference_model=deepcopy(row["reference_model"]), target_model=deepcopy(row["target_model"]),
            composition_reference_u_to_native=composition.tolist(),
            continuation=dict(replay["counts"], contract_run=run, authority=contracts[run]["frame_authority"]["kind"]),
            shape_hw=list(base_arrays["array"].shape),
            label_values=[int(value) for value in np.unique(base_arrays["array"])])
        native_drawn = set()
        if fixture_source is not None:
            native = context["run_entries"].get(run)
            if native is None:
                raise ValueError("The transfer is missing this run's explicitly bound native fixture underlay")
            entry.update(fixture_underlay=_local_descriptor(folder, native["fixture_underlay"]),
                fixture_underlay_layer_ids=deepcopy(native["fixture_underlay_layer_ids"]))
            native_drawn, bound = _checked_native_underlay(
                folder, entry, fixture_source, base_arrays["array"], preparation)
            guards.append(bound)
        preparation["runs"].append(entry)
        if run in groups:
            effective, painted, prior, provenance = _replay(preparation, folder, groups[run], run)
        else:
            effective = base_arrays
            painted = np.zeros(effective["array"].shape, bool)
            prior, provenance = [], []
        if chart_recipe:
            from .painting_chart_products import chart_product
            from .painted_adoption_table import _local
            before_chart = (_local(folder, preparation["target_corrections"]["runs"][run]["artifact"])
                            if prior else base)
            # The ordinary chart reader checks every approval's cohort even
            # while this replay is still assembling individual run entries.
            effective, chart_bound = chart_product(dict(preparation, runs=review["runs"]),
                folder, entry, effective, before_chart)
            if not _same(chart_bound, row["product"]):
                raise ValueError("Current chart findings differ from their exact approved effective product")
            guards.append(chart_bound)
        if _same(prior, row["target_correction_bindings"]):
            if not _same({name: _identity(value) for name, value in effective.items()}, row["scientific_arrays"]):
                raise ValueError("The effective transfer differs from its original target replay")
            science = target_paint_science(preparation, entry, contracts[run], effective, prior)
            if not _same(science, row["target_correction_science"]):
                raise ValueError("The transfer target science changed")
            if chart_recipe:
                effective = reviewed  # All fields just matched the authenticated chart product.
        components = list(_components(effective["gaps"])) if retain_arrays else []
        # Most runs have no target edit. Keep one verified native-array set,
        # rather than three 160-MiB copies per production run. Consumers only
        # receive read-only DATA; edits belong to their separate typed draft.
        for arrays in (base_arrays, reviewed, effective):
            for value in arrays.values():
                value.setflags(write=False)
        for _component, mask in components:
            mask.setflags(write=False)
        painted.setflags(write=False)
        result.append(dict(run_id=run, entry=entry, base_product=base,
            base_arrays=base_arrays if retain_arrays and retain_source_arrays else None,
            reviewed_arrays=reviewed if retain_arrays and retain_source_arrays else None,
            effective_arrays=effective if retain_arrays else None,
            residual_mask=effective["gaps"] if retain_arrays else None, components=components,
            target_correction_bindings=deepcopy(prior), native_drawn_ids=set(native_drawn),
            painted_mask=painted if retain_arrays else None, target_provenance=provenance))
        guards.extend([base, row["product"], row["reference_model"], row["target_model"]])
        guards.append(models[run].model["baseline_native_grid"]["renderer"])
    if selected:
        target_preparation = dict(preparation, target_corrections=dict(format=FORMAT,
            bindings=selected, runs={run: dict(bindings=bindings) for run, bindings in groups.items()}))
        guards.extend(target_sources(target_preparation))
    unique = {}
    for descriptor in guards:
        checked = _checked(descriptor)
        unique[(checked["path"], checked["sha256"], checked["size_bytes"])] = checked
    return dict(review=review, preparation=preparation, folder=folder,
        painting_binding=deepcopy(painting), session=session, approved_analysis=approved_analysis,
        fixture_source=fixture_source, fixture_drawn_ids=fixture_ids,
        source_bindings=dict(review=deepcopy(review["review_binding"]), files=list(unique.values())),
        runs=result, target_bindings=selected,
        source_context=context)


def _copy_bound(descriptor, destination):
    from .painted_analysis_exclusions import _checked, _bound
    checked = _checked(descriptor)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(checked["path"]).open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    _checked(descriptor)
    copied = _bound(destination)
    if any(copied[key] != checked[key] for key in ("sha256", "size_bytes")):
        raise ValueError("A separate residual review copy differs from its bound original bytes")
    return copied


def _write_recipe(replay, destination, *, target_bindings=None):
    """Copy original bytes into an exclusive new folder for original writers."""
    from .painted_analysis_exclusions import _checked, _json, _same
    from .painted_target_integration import prepare_target_corrections
    from .rig_raw_write import _no_links
    from .raw_shape_basis_reader import guard_output_root
    root = Path(destination).expanduser()
    if not root.is_absolute():
        raise ValueError("A fresh absolute residual review destination is required")
    original = Path(replay["folder"]).resolve()
    if root.resolve().is_relative_to(original) or original.is_relative_to(root.resolve()):
        raise ValueError("Keep the fresh residual review separate from its retained original")
    bindings = replay["target_bindings"] if target_bindings is None else _json(list(target_bindings))
    if not _same(bindings, replay["target_bindings"]):
        raise ValueError("Replay the exact current target bindings before writing their residual review")
    _no_links(root)
    models = {}
    for row in replay["review"]["runs"]:
        for name in ("reference_model", "target_model"):
            descriptor = row[name]
            models[(descriptor["path"], descriptor["sha256"])] = _checked(descriptor, metadata=True)
    guard_output_root(root, models.values())
    root.mkdir(parents=True, exist_ok=False)
    preparation = deepcopy(replay["preparation"])
    preparation.pop("source_recapture", None)
    preparation["source_map"] = dict(_copy_bound(replay["review"]["source_map"], root / "source_map.npz"), path="source_map.npz")
    continuation = _copy_bound(replay["source_context"]["continuation"], root / "continuation.json")
    preparation["continuation"].update({name: continuation[name] for name in ("sha256", "size_bytes")})
    if replay["source_context"].get("source_fixture_underlay") is not None:
        preparation["source_fixture_underlay"] = dict(_copy_bound(
            replay["source_context"]["source_fixture_underlay"], root / "source_fixture_underlay.npz"),
            path="source_fixture_underlay.npz")
    by_run = {row["run_id"]: row for row in replay["runs"]}
    for entry in preparation["runs"]:
        run = entry["run_id"]
        entry["labels"] = dict(_copy_bound(by_run[run]["base_product"], root / run / "labels.npz"), path=f"{run}/labels.npz")
        old_native = replay["source_context"]["run_entries"].get(run)
        if old_native is not None:
            entry["fixture_underlay"] = dict(_copy_bound(old_native["fixture_underlay"], root / run / "fixture_underlay.npz"),
                path=f"{run}/fixture_underlay.npz")
    if bindings:
        preparation["target_corrections"] = prepare_target_corrections(preparation, root, bindings)
    if "chart_review" in preparation:
        from .painted_adoption_table import _local, _bound
        # The ordinary local outside-proof reader also authenticates this
        # unchanged source action; copying it creates no new human approval.
        _copy_bound(_bound(original / "approval.json"), root / "approval.json")
        for run, record in preparation["chart_transfers"].items():
            record["artifact"] = dict(_copy_bound(_local(original, record["artifact"]),
                root / run / "chart_labels.npz"), path=f"{run}/chart_labels.npz")
    return preparation, root


def build_residual_transfer_review(replay, destination, *, target_bindings=None):
    """Publish a genuine post-TARGET review without rebasing any human choice."""
    from .painted_analysis_exclusions import build_gap_review
    preparation, root = _write_recipe(replay, destination, target_bindings=target_bindings)
    return build_gap_review(preparation, root)


def materialize_transfer_findings(replay, destination, *, analysis_exclusions=None):
    """Run original exclusion/outside proof writers after actual human choices.

    This does not replace or rewrite the human's retained review.  Original
    exclusion replay accepts only scientifically identical fresh containers.
    No index, candidate, source-state action or adoption record is created.
    """
    from .painted_analysis_exclusions import prepare_exclusions, _bound
    from .painted_adoption_table import (verify_version, prepare_outside_proof,
        verify_outside_proof)
    preparation, root = _write_recipe(replay, destination)
    if analysis_exclusions is not None:
        preparation["analysis_exclusions"] = prepare_exclusions(preparation, root, analysis_exclusions)
    proof = prepare_outside_proof(root, preparation["painting_facies"], preparation["runs"], preparation=preparation)
    guarded = []
    if "chart_review" in preparation:
        guarded.append(_bound(root / "approval.json"))
    if proof is not None:
        preparation["outside_frame_proof"] = proof
        _, approved, frame = verify_version(preparation["painting_facies"])
        guarded.extend(verify_outside_proof(preparation, root, approved, frame))
    return dict(review=replay["review"], preparation=preparation, folder=root,
        analysis_exclusions=deepcopy(preparation.get("analysis_exclusions")),
        outside_frame_proof=deepcopy(proof), outside_proof_bindings=guarded)
