"""Explicit target-edit replay beside byte-identical original native products.

The original continuation proof always consumes labels.npz. Only consumers
requesting these verified effective fields may use target_labels.npz. No edit
is attributed to the carrier, original support, continuation, or optical data.
"""
from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from copy import deepcopy
from pathlib import Path

import numpy as np

KEY = "target_corrections"
FORMAT = "darsia-painted-target-correction-products-v1"


def _replay(preparation, folder, bindings, run):
    from .painted_analysis_exclusions import (_arrays, _checked, _same, read_gap_review,
        _components, target_paint_science, _json, _digest)
    from .photo_edge_continuation import read_contracts, verify_product
    from .painted_target_corrections import verify_and_apply_target_correction
    from .painted_adoption_table import _local, verify_version
    entry = next(row for row in preparation["runs"] if row["run_id"] == run and row.get("status") == "prepared")
    base = _local(folder, entry["labels"])
    arrays = _arrays(base)
    original = {key: value.copy() for key, value in arrays.items()}
    contract = read_contracts(Path(folder) / "continuation.json")["runs"][run]
    with np.load(_preset_io.source(_local(folder, preparation["source_map"])["path"]), allow_pickle=False) as archive:
        source = np.array(archive["array"], copy=True)
    verify_product(base["path"], source, contract, counts=entry.get("continuation"))
    session, _, _ = verify_version(preparation["painting_facies"])
    supported = sorted(layer.layer_id for layer in session.layers
        if session.facies_of(layer.layer_id).role == "porous_medium" and 0 < layer.layer_id < 255)
    prior, provenance = [], []
    painted = np.zeros(arrays["array"].shape, bool)
    for binding in _json(bindings):
        version = _checked(binding["version"], metadata=True)
        basis = version["basis"]
        if basis["run_id"] != run:
            raise ValueError("The selected target correction belongs to another run")
        review = read_gap_review(basis["review"]["path"])
        if not _same(review["review_binding"], basis["review"]):
            raise ValueError("The target edit belongs to another retained review")
        reviewed = next(row for row in review["runs"] if row["run_id"] == run)
        component = next(item for item in reviewed["components"] if item["component_id"] == basis["component_id"])
        if basis["review_row_sha256"] != component["review_row_sha256"]:
            raise ValueError("The target edit belongs to another connected area")
        old = _arrays(reviewed["product"])
        masks = [mask for item, mask in _components(old["gaps"]) if item["component_id"] == basis["component_id"]]
        if len(masks) != 1 or np.any(masks[0] & ~arrays["gaps"]):
            raise ValueError("The target edit's original unresolved area changed")
        science = target_paint_science(preparation, entry, contract, arrays, prior)
        if (not _same(reviewed["target_correction_science"], science)
                or version.get("allowed_ids") != supported):
            raise ValueError("The target edit changed its source, prior edits or supported layer IDs")
        result = verify_and_apply_target_correction(binding, base_labels=arrays["array"].astype(np.uint8),
            editable_mask=masks[0], current_science=science)
        labels, edit = result["labels"], result["painted_mask"]
        if (labels.dtype != np.uint8 or labels.shape != arrays["array"].shape
                or edit.dtype != bool or edit.shape != labels.shape or not edit.any()
                or np.any(edit & ~masks[0]) or np.any(edit & ~arrays["gaps"])
                or not np.array_equal(labels[~edit], arrays["array"][~edit])
                or np.any(edit & ~np.isin(labels, supported))):
            raise ValueError("A target correction changed mapped pixels, support or an unapproved area")
        arrays["array"] = labels.astype(np.int32)
        arrays["effective_coverage"] = arrays["effective_coverage"] | edit
        arrays["gaps"] = arrays["inside"] & ~arrays["effective_coverage"]
        painted |= edit
        prior.append(deepcopy(binding))
        compact = {key: deepcopy(value) for key, value in result["provenance"].items() if key != "basis"}
        compact["basis_sha256"] = _digest(result["provenance"]["basis"])
        provenance.append(compact)
    if not prior:
        raise ValueError("The target product has no explicit correction for this run")
    for key in ("filled", "support", "inside", "continued"):
        if not np.array_equal(arrays[key], original[key]):
            raise ValueError("Target painting changed original carrier/continuation evidence")
    return arrays, painted, prior, provenance


def prepare_target_corrections(preparation, folder, bindings):
    """Write only separate corrected products; keep every original byte intact."""
    from .painted_analysis_exclusions import _json, _checked, _identity, _bound
    bindings = _json(list(bindings))
    if not bindings or len({(item["version"]["path"], item["version"]["sha256"]) for item in bindings}) != len(bindings):
        raise ValueError("Target corrections require unique explicit saved approvals")
    runs = {}
    for binding in bindings:
        run = _checked(binding["version"], metadata=True)["basis"]["run_id"]
        runs.setdefault(run, []).append(binding)
    rows = {}
    for run, selected in runs.items():
        arrays, painted, prior, provenance = _replay(preparation, folder, selected, run)
        original = Path(folder) / run / "labels.npz"
        with np.load(_preset_io.source(original), allow_pickle=True) as archive:
            metadata = deepcopy(archive["metadata"].item())
        target = Path(folder) / run / "target_labels.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, **arrays, metadata=metadata, painted_mask=painted)
        rows[run] = dict(artifact=dict(_bound(target), path=f"{run}/target_labels.npz"),
            scientific_arrays={key: _identity(value) for key, value in arrays.items()},
            painted_mask=_identity(painted), bindings=prior, provenance=provenance,
            label_values=[int(value) for value in np.unique(arrays["array"])])
    return dict(format=FORMAT, bindings=bindings, runs=rows)


def _selected_bindings(contract, run):
    """Join immutable literal row/global ordering without opening foreign versions.

    Prepare and Store publication replay every run. A selected restore replays
    only its row, retaining exact membership and ordering in the full contract.
    """
    from .painted_analysis_exclusions import _digest
    bindings, rows = contract["bindings"], contract["runs"]
    if not isinstance(bindings, list) or not bindings or not isinstance(rows, dict) or not rows:
        raise ValueError("The target-correction contract lacks its published rows")
    global_ids = [_digest(binding) for binding in bindings]
    if len(set(global_ids)) != len(global_ids):
        raise ValueError("Target-correction bindings must be unique")
    assigned = set()
    for record in rows.values():
        selected = record.get("bindings")
        if not isinstance(selected, list) or not selected:
            raise ValueError("A published target row lacks its exact bindings")
        ids = [_digest(binding) for binding in selected]
        if (len(set(ids)) != len(ids) or assigned.intersection(ids)
                or ids != [key for key in global_ids if key in set(ids)]):
            raise ValueError("The target row changed its published binding membership/order")
        assigned.update(ids)
    if assigned != set(global_ids):
        raise ValueError("Target-correction product runs differ from their saved approvals")
    return rows[run]["bindings"] if run in rows else None


def _effective_without_chart(preparation, folder, row):
    """Independently verify an effective product; original proofs retain their base."""
    from .painted_analysis_exclusions import _arrays, _identity, _same
    from .painted_adoption_table import _local
    contract = preparation.get(KEY)
    def base_product():
        bound = _local(folder, row["labels"])
        if preparation.get("format") == "darsia-fluidflower-painted-map-adoption-preparation-v2":
            return _arrays(bound), bound
        # Historical V1 archives have only their original three fields. The
        # optional target branch must not impose V2 masks on that old route.
        with np.load(_preset_io.source(bound["path"]), allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in ("array", "filled", "support")}
        return arrays, bound
    if contract is None:
        return base_product()
    if contract.get("format") != FORMAT or set(contract) != {"format", "bindings", "runs"}:
        raise ValueError("Unknown target-correction product contract")
    selected = _selected_bindings(contract, row["run_id"])
    if selected is None:
        return base_product()
    run = row["run_id"]
    expected, painted, bindings, provenance = _replay(preparation, folder, selected, run)
    record = contract["runs"][run]
    bound = _local(folder, record["artifact"])
    actual = _arrays(bound)
    with np.load(_preset_io.source(bound["path"]), allow_pickle=False) as archive:
        if set(archive.files) != set(actual) | {"metadata", "painted_mask"}:
            raise ValueError("The effective target product has unknown fields")
        stored_paint = np.array(archive["painted_mask"], copy=True)
    if (any(not np.array_equal(actual[key], expected[key]) for key in expected)
            or not _same(record["scientific_arrays"], {key: _identity(value) for key, value in expected.items()})
            or stored_paint.dtype != bool or not np.array_equal(stored_paint, painted)
            or not _same(record["painted_mask"], _identity(painted))
            or not _same(record["bindings"], bindings) or not _same(record["provenance"], provenance)
            or record["label_values"] != [int(value) for value in np.unique(actual["array"]) ]):
        raise ValueError("The effective target product differs from its exact approved event replay")
    return actual, bound


def target_sources(preparation):
    """All exact metadata and pixel artifacts consumed by target edit replay."""
    from .painted_analysis_exclusions import _checked, _scientific_review_projection, read_gap_review
    contract = preparation.get(KEY)
    if contract is None:
        return []
    result = {}
    def collect(value):
        if isinstance(value, dict):
            if {"path", "sha256", "size_bytes"} <= set(value) and Path(value["path"]).is_absolute():
                bound = {key: value[key] for key in ("path", "sha256", "size_bytes")}
                _checked(bound)
                result[(bound["path"], bound["sha256"])] = bound
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(contract)
    for binding in contract["bindings"]:
        version = _checked(binding["version"], metadata=True)
        collect(version)
        collect(_checked(binding["approval"], metadata=True))
        review = read_gap_review(version["basis"]["review"]["path"])
        # Use the same typed frozen-DATA boundary as gap approvals, while
        # retaining every model/photo descriptor and unknown scientific pin.
        collect(_scientific_review_projection(review))
    return list(result.values())


def effective_product(preparation, folder, row):
    """Resolve approved chart occlusion before the ordinary physical-field readers."""
    arrays, bound = _effective_without_chart(preparation, folder, row)
    from .painting_chart_products import chart_product
    return chart_product(preparation, folder, row, arrays, bound)
