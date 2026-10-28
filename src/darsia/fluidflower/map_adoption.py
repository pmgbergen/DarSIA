"""The provenance and the preparation of a painted map's adoption.

A placement review folder (see :mod:`map_placement`) holds, per run, the
approved map placed on the run's camera grid. Before such a placement can
be adopted as a geometry revision, its provenance is bound: the approved
version and its approval, the placement index, the geometry index, and each
placed run's layer map, all by SHA-256. :func:`adoption_blockers` names what
an adoption still lacks.

:func:`prepare_adoption` carries the approved map to each run's
rig-corrected native label grid, the grid that Build run model and the
jobs read, through the run's measured geometry and its baseline native
grid: ``H_referenceU_to_targetNative = H_targetU_to_targetNative @
H_target_mm_to_U @ H_referenceU_to_mm``, in the existing nearest-neighbour
categorical carrier, with the fill and support flags kept. It writes a new
preparation folder; the facies ids of the regions wait for the facies table
of the map, and none is invented.

:func:`adoption_candidate` turns a preparation and the facies table of the
map (each label's facies id in the series' full facies table) into a
geometry revision candidate on the current corrected workspace revision: a
canonical edit of its imported parent whose every run carries a
``painted-map-placement-v1`` registration, replayed exactly by the Store.
Saving it is a separate, explicit step.

Nothing here writes a revision, a map or a run configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np

PROVENANCE_FORMAT = "darsia-fluidflower-painted-placement-provenance-v1"
PROVENANCE_FORMAT_V2 = "darsia-fluidflower-painted-placement-provenance-v2"
PREPARATION_FORMAT = "darsia-fluidflower-painted-map-adoption-preparation-v1"
PREPARATION_FORMAT_V2 = "darsia-fluidflower-painted-map-adoption-preparation-v2"
FACIES_IDS_PENDING = "waits for the facies table of the map"


def _bound(path: Path) -> dict[str, Any]:
    """A file bound by its resolved path, SHA-256 and size."""

    from darsia.fluidflower.geometry_raster_sources import descriptor

    return descriptor(path)


def placement_provenance(folder: Path | str) -> dict[str, Any]:
    """The provenance record of a placement review folder.

    Every placed run's ``layers.npz`` is bound by SHA-256 and size, as is the
    folder's ``index.json``; a placed run whose layer map is missing is
    refused. Runs the placement could not place are listed as incomplete.
    """

    from darsia.fluidflower.map_placement import read_placement

    folder = Path(folder)
    placement = read_placement(folder)
    runs, incomplete = [], []
    for row in placement.get("runs") or []:
        run = str(row.get("run_id") or "")
        if row.get("status") != "placed":
            incomplete.append({"run_id": run, "status": str(row.get("status") or "")})
            continue
        layers = folder / str(row["layers"])
        if not layers.is_file():
            raise FileNotFoundError(f"{run}: the placed layer map is missing: {layers}")
        runs.append(
            {
                "run_id": run,
                "layers": _bound(layers),
                "shape_hw": list(row.get("shape_hw") or []),
                "geometry_model": row.get("geometry_model"),
                "offset_px": row.get("offset_px"),
            }
        )
    version = dict(placement.get("map_version") or {})
    result = {
        "format": PROVENANCE_FORMAT,
        "placement": {"folder": str(folder), "index": _bound(folder / "index.json")},
        "reference_run": placement.get("reference_run"),
        "map_version": version,
        "approval": version.get("approval"),
        "geometry_index": placement.get("geometry_index"),
        "layers": placement.get("layers"),
        "runs": runs,
        "incomplete_runs": incomplete,
    }
    if placement.get("continuation") is not None:
        result["format"] = PROVENANCE_FORMAT_V2
        result["continuation"] = _bound(folder / "continuation.json")
        result["approved_source"] = deepcopy(placement["approved_source"])
    return result


def map_facies(provenance: Mapping[str, Any]) -> list[str]:
    """The facies names of the placed map's layers, from its placement index."""

    return sorted(
        {
            str(row.get("facies") or "")
            for row in (provenance.get("layers") or {}).values()
            if row.get("facies")
        }
    )


def adoption_blockers(
    provenance: Mapping[str, Any],
    *,
    transports: Mapping[str, Any] | None = None,
    facies_ids: Mapping[str, int] | None = None,
    facies: Iterable[str] = (),
) -> list[str]:
    """What an adoption of this placement still lacks, one line each; empty
    when nothing does.

    ``transports`` maps a run id to its transport from the camera grid to
    the run's label grid; ``facies_ids`` maps each facies name of the map to
    its id in the series' facies table; ``facies`` names the map's facies.
    """

    blockers = []
    approval = provenance.get("approval")
    version = dict(provenance.get("map_version") or {})
    if not approval or approval.get("sha256") != version.get("sha256"):
        blockers.append("The placed map version has no approval of its own SHA-256.")
    incomplete = [row["run_id"].upper() for row in provenance.get("incomplete_runs") or []]
    if incomplete:
        blockers.append(f"Not placed: {', '.join(incomplete)}.")
    placed = [row["run_id"] for row in provenance.get("runs") or []]
    missing = [run.upper() for run in placed if run not in (transports or {})]
    if missing:
        blockers.append(
            "No transport from the camera grid to the run's label grid: "
            f"{', '.join(missing)}."
        )
    names = sorted(set(facies))
    without = [name for name in names if name not in (facies_ids or {})]
    if without:
        blockers.append(f"No facies id for: {', '.join(without)}.")
    return blockers


def native_transport(geometry: Any) -> tuple[np.ndarray, tuple[int, int]]:
    """A run's homography from its baseline camera grid to its rig-corrected
    native grid, and the native grid's shape, from the model's
    ``baseline_native_grid``."""

    from darsia.fluidflower.map_placement import _matrix

    grid = dict(geometry.model.get("baseline_native_grid") or {})
    value = grid.get("H_baseline_camera_u_to_rig_corrected_native")
    shape = grid.get("shape_hw")
    if value is None or not shape:
        raise ValueError(f"{geometry.run_id}: the geometry has no baseline native grid")
    matrix = _matrix(value, f"{geometry.run_id} H_baseline_camera_u_to_rig_corrected_native")
    return matrix, (int(shape[0]), int(shape[1]))


def native_labels(
    layers: np.ndarray, reference: Any, target: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[float]]]:
    """The reference run's layer map on another run's rig-corrected native grid.

    Returns the layer ids, the pixels the map does not reach (filled from its
    edge), the pixels that come from inside the map (its non-zero layers)
    and the composed matrix, reference camera grid to target native grid.
    """

    from darsia.fluidflower.geometry_raster_sources import carry_label_reference

    to_native, shape = native_transport(target)
    layers = np.ascontiguousarray(layers, dtype=np.uint8)
    target_matrix = to_native @ target.metric_to_grid
    carried = carry_label_reference(
        layers,
        # The carrier takes explicit nested lists of numbers.
        reference_matrix=reference.metric_to_grid.tolist(),
        target_matrix=target_matrix.tolist(),
        target_shape_hw=[shape[0], shape[1]],
        reference_filled=np.zeros(layers.shape, dtype=bool),
        source_fill_status="KNOWN",
        reference_support=layers != 0,
    )
    composition = target_matrix @ np.linalg.inv(reference.metric_to_grid)
    return (
        np.asarray(carried["labels"], dtype=np.uint8),
        np.asarray(carried["geometry_filled"], dtype=bool),
        np.asarray(carried["support"], dtype=bool),
        composition.tolist(),
    )


def native_with_continuation(layers, reference, target, *, contract):
    """Keep the original native carrier, adding separately replayed coverage."""
    from .photo_edge_continuation import apply_policy
    if (contract.get("source_run") != reference.run_id or contract.get("target_run") != target.run_id
            or contract.get("target_grid") != "native"):
        raise ValueError("The continuation contract belongs to another native placement.")
    labels, filled, support, composition = native_labels(layers, reference, target)
    return apply_policy(layers, contract, labels=labels, filled=filled, support=support), composition


def prepare_adoption(
    destination_root: Path | str,
    layers: np.ndarray,
    *,
    reference_run: str,
    geometry_index: Path | str,
    runs: Sequence[str],
    roles: Mapping[int, str],
    version: Mapping[str, Any],
    names: Mapping[int, str] | None = None,
    facies: Mapping[int, str] | None = None,
    facies_ids: Mapping[str, int] | None = None,
    current_labels: Callable[[str], Mapping[str, Any] | None] | None = None,
    progress: Callable[[str], None] | None = None,
    fixture_underlay: Any | None = None,
    fixture_frame: Any | None = None,
    painting_version: Mapping[str, Any] | None = None,
    painting_frame: Any = None,
    painting_project: Any = None,
    continuation: Mapping[str, Any] | None = None,
    approved_source: Mapping[str, Any] | None = None,
    analysis_exclusions: Mapping[str, Any] | None = None,
    review_findings: bool = False,
    target_corrections: Sequence[Mapping[str, Any]] | None = None,
    display_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a new adoption preparation folder for every run; adopt nothing.

    ``layers`` is the approved map's layer ids on the reference run's camera
    grid. ``current_labels(run)`` gives the run's current corrected label
    field as ``{"metadata", "shape_hw", "source"}``; the prepared map takes
    its DarSIA metadata, and a run whose current field has another shape is
    refused. Per run, ``<run>/labels.npz`` holds ``array`` (the layer ids on
    the native grid, int32), ``metadata``, ``filled`` and ``support``.
    ``index.json`` (PREPARATION_FORMAT) binds the approved version, the
    geometry index and each run's composition and output by SHA-256, lists
    the regions (layer id, name, role, facies, facies id) and what an
    adoption still lacks. Facies without an id in ``facies_ids`` stay
    without one.

    Explicit ``painting_version`` preparations instead bind the actual approved
    immutable session and map, separately freezing the provided current frame
    and its source context. Their later Check derives one facies row
    per used painting facies, carrying porous extra cells by name and leaving
    nonporous extra cells blank with provenance; it makes no
    manual label-to-old-facies choices. Legacy preparations are unchanged.
    """

    from darsia.fluidflower.map_placement import read_run_geometry
    from darsia.fluidflower.project import sha256_file

    painting_binding = None
    analysis_derivation = None
    exclusion_approval = None if analysis_exclusions is None else deepcopy(dict(analysis_exclusions))
    target_approvals = None if target_corrections is None else deepcopy(list(target_corrections))
    if exclusion_approval is not None and (painting_version is None or continuation is None):
        raise ValueError("Analysis gap approval requires its exact saved painting and V2 native continuation")
    if target_approvals is not None and (painting_version is None or continuation is None):
        raise ValueError("Target corrections require their exact saved painting and V2 native continuation")
    context = None
    if display_context is not None:
        if painting_version is None or continuation is None:
            raise ValueError("Display findings require the genuine saved V2 native painting review")
        from .painted_analysis_exclusions import _display_context
        context = _display_context(display_context)
    if painting_version is not None:
        from .raw_shape_basis_reader import require_basis_index
        require_basis_index(geometry_index, [reference_run, *runs])
        from .painted_adoption_table import capture_version, verify_version_summary
        painting_binding = capture_version(painting_version, version.get("approval"), painting_frame, layers,
                                          project=painting_project)
        verify_version_summary(painting_binding, dict(version))
    index_path = Path(geometry_index)
    frozen_index = _bound(index_path) if painting_binding is not None else None
    geometries = read_run_geometry(index_path)
    if frozen_index is not None:
        from .painted_adoption_table import _checked
        _checked(frozen_index)
    reference_key = str(reference_run).casefold()
    if reference_key not in geometries:
        raise ValueError(f"The geometry index has no model for the reference run {reference_run}")
    reference = geometries[reference_key]
    geometry_proof = None
    basis_checks = {}
    if painting_binding is not None:
        from .painted_adoption_geometry import make_proof
        from .painted_adoption_table import _checked, _same
        selected = {reference_key, *(str(run).casefold() for run in runs)}
        model_bindings = {key: _bound(Path(geometries[key].source))
                          for key in sorted(selected) if key in geometries}
        geometry_proof = make_proof(painting_binding, model_bindings, basis_checks=basis_checks)
        if any(not _same(geometries[key].model, _checked(bound, metadata=True))
               for key, bound in model_bindings.items()):
            raise ValueError("A measured model changed between selection and source admission")
    layers = np.ascontiguousarray(layers, dtype=np.uint8)
    if reference.shape_hw is not None and layers.shape != reference.shape_hw:
        raise ValueError("The map is not on the reference run's grid")
    original_grid = None
    if continuation is not None:
        from .photo_edge_continuation import verify_source, verify_placement_inputs
        verify_source(approved_source, layers, continuation)
        verify_placement_inputs(continuation, geometries, index_path, reference_key,
            [str(run).casefold() for run in runs], "native")
        if set(continuation) != {str(run).casefold() for run in runs}:
            raise ValueError("Every newly prepared run needs its own continuation contract.")
        if painting_binding is not None:
            from .painted_adoption_table import _same
            if not _same(painting_binding, approved_source):
                raise ValueError("The continuation must bind the same exact current painting preparation.")
        else:
            # Retain the qualified explicit V2 route for historical callers.
            from .painted_adoption_geometry import make_proof
            model_bindings = {reference_key: _bound(Path(reference.source))}
            model_bindings.update({str(run).casefold(): _bound(Path(geometries[str(run).casefold()].source)) for run in runs})
            geometry_proof = make_proof(approved_source, model_bindings, basis_checks=basis_checks)
            from .geometry_facies import _darsia_metadata, _load_array
            revision = json.loads(Path(approved_source["frame_basis"]["workspace_revision"]["path"]).read_bytes())
            original = revision["configuration"]["canonical_import_parent"]["imported_configuration"]["labels_path"]
            original_grid = dict(source=_bound(Path(original)), metadata=_darsia_metadata(Path(original)),
                                 shape_hw=list(_load_array(Path(original)).shape))
    if fixture_underlay is not None:
        if painting_binding is not None:
            from .painted_adoption_table import checked_fixture_analysis
            fixture_underlay, analysis_derivation = checked_fixture_analysis(painting_binding, fixture_underlay)
            if fixture_frame is None or fixture_frame.to_dict() != painting_binding["frame"]:
                raise ValueError("The fixture and saved painting preparation frames differ")
            layers = fixture_underlay.analysis_layers
        else:
            from .layer_painting_fixtures import checked_for_adoption
            fixture_underlay = checked_for_adoption(fixture_underlay, layers, version, frame=fixture_frame)
        frozen_roles = fixture_underlay.provenance["roles"]
        if any(str(roles.get(int(key))) != value for key, value in frozen_roles.items()):
            raise ValueError("The fixture underlay's frozen layer roles differ from the adoption roles")
    elif np.isin(layers, [key for key, role in roles.items() if role == "colorchecker"]).any():
        raise ValueError("A ColorChecker map needs its frozen version's drawn-only underlay")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    folder = Path(destination_root) / f"{stamp}-{uuid.uuid4().hex[:8]}"
    if geometry_proof is not None and "raw_shape_basis" in geometry_proof:
        from .raw_shape_basis_reader import guard_output_root
        guard_output_root(folder, [geometries[key].model for key in model_bindings])
    folder.mkdir(parents=True, exist_ok=False)
    if geometry_proof is not None:
        from .painted_adoption_geometry import write_configuration_copies, basis_label_fields
        write_configuration_copies(folder, geometry_proof, basis_checks=basis_checks)
        basis_labels = basis_label_fields(painting_binding or approved_source, geometry_proof, None, folder)
    else:
        basis_labels = None
    if continuation is not None:
        from .photo_edge_continuation import save_contracts
        save_contracts(folder / "continuation.json", continuation)
    # The map itself and its approval, bound for the Store's replay.
    np.savez_compressed(folder / "source_map.npz", array=layers)
    if fixture_underlay is not None:
        from .layer_painting_fixtures import save_underlay
        source_underlay = save_underlay(folder / "source_fixture_underlay.npz", fixture_underlay)
        source_underlay["path"] = "source_fixture_underlay.npz"
    with (folder / "approval.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(version.get("approval") or {}), stream, indent=2, ensure_ascii=False)
    reference_bound = _bound(Path(reference.source))
    keys = [str(run).casefold() for run in runs]
    rows = []
    for number, run in enumerate(keys, 1):
        if progress is not None:
            progress(f"Preparing {run.upper()} on its label grid ({number} of {len(keys)})…")
        target = geometries.get(run)
        if target is None:
            rows.append({"run_id": run, "status": "no geometry"})
            continue
        try:
            labels, filled, support, composition = native_labels(layers, reference, target)
            classified = None
            if continuation is not None:
                classified, replayed = native_with_continuation(layers, reference, target,
                    contract=continuation[run])
                if replayed != composition:
                    raise ValueError("The new native placement changed the original composition.")
                labels, filled, support = (classified[key] for key in ('labels', 'filled', 'support'))
        except ValueError as error:
            rows.append({"run_id": run, "status": "no native grid", "error": str(error)})
            continue
        if basis_labels is not None:
            current = dict(basis_labels[run])
        else:
            current = dict(current_labels(run) or {}) if current_labels is not None else {}
        if original_grid is not None:
            from .geometry_raster_sources import _same_grid_metadata
            if (current.get("source") != original_grid["source"]
                    or current.get("shape_hw") != original_grid["shape_hw"]
                    or not _same_grid_metadata(current.get("metadata"), original_grid["metadata"])):
                raise ValueError("The V2 native labels need the exact imported parent's grid metadata and source.")
        if current.get("shape_hw") and tuple(current["shape_hw"]) != labels.shape:
            rows.append(
                {
                    "run_id": run,
                    "status": "grid differs",
                    "error": f"current label field {list(current['shape_hw'])}, "
                    f"native grid {list(labels.shape)}",
                }
            )
            continue
        run_folder = folder / run
        run_folder.mkdir()
        path = run_folder / "labels.npz"
        fields = dict(array=labels.astype(np.int32), metadata=dict(current.get("metadata") or {}),
                      filled=filled, support=support)
        if classified is not None:
            fields.update({key: classified[key] for key in ("inside", "continued", "effective_coverage", "gaps")})
        np.savez_compressed(path, **fields)
        underlay_fields = {}
        if fixture_underlay is not None:
            from .layer_painting_fixtures import project_underlay, save_underlay
            native_underlay = project_underlay(fixture_underlay, reference, target, labels,
                frame_boundary_samples=None if classified is None else classified.get('frame_boundary_samples'))
            bound_underlay = save_underlay(run_folder / "fixture_underlay.npz", native_underlay)
            bound_underlay["path"] = f"{run}/fixture_underlay.npz"
            underlay_fields = dict(fixture_underlay=bound_underlay,
                fixture_underlay_layer_ids=[int(v) for v in np.unique(native_underlay.drawn_layers) if v])
        rows.append(
            {
                **underlay_fields,
                "run_id": run,
                "status": "prepared",
                "labels": {
                    "path": f"{run}/labels.npz",
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                },
                "shape_hw": list(labels.shape),
                "label_values": [int(value) for value in np.unique(labels)],
                "reference_model": reference_bound,
                "target_model": _bound(Path(target.source)),
                "composition_reference_u_to_native": composition,
                "filled_pixels": int(np.count_nonzero(filled)),
                "outside_support_pixels": int(np.count_nonzero(~support)),
                "current_labels": current.get("source"),
            }
        )
        if classified is not None:
            rows[-1]["continuation"] = dict(classified["counts"], contract_run=run,
                authority=continuation[run]["frame_authority"]["kind"])
    regions = [
        {
            "layer_id": int(layer),
            "name": str((names or {}).get(layer, "")),
            "role": str(role),
            "facies": str((facies or {}).get(layer, "")),
            "facies_id": (facies_ids or {}).get(str((facies or {}).get(layer, ""))),
        }
        for layer, role in sorted(roles.items())
    ]
    blockers = []
    not_prepared = [row["run_id"].upper() for row in rows if row["status"] != "prepared"]
    if not_prepared:
        blockers.append(f"Not prepared: {', '.join(not_prepared)}.")
    without = sorted({row["facies"] for row in regions if row["facies"] and row["facies_id"] is None})
    if without and painting_binding is None:
        blockers.append(f"No facies id for: {', '.join(without)} ({FACIES_IDS_PENDING}).")
    if continuation is not None:
        from .photo_edge_continuation import verify_capture
        verify_capture(continuation)
    index = {
        "format": PREPARATION_FORMAT if continuation is None else PREPARATION_FORMAT_V2,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_run": reference_key,
        "map_version": dict(version),
        "geometry_index": frozen_index if frozen_index is not None else {"path": str(index_path.resolve()), "sha256": sha256_file(index_path)},
        "source_map": {"path": "source_map.npz", **{
            key: value for key, value in _bound(folder / "source_map.npz").items() if key != "path"
        }},
        "approval": dict(version.get("approval") or {}),
        "composition": "H_targetU_to_targetNative @ H_target_mm_to_U @ H_referenceU_to_mm, "
        "nearest neighbour; filled and support flags kept",
        "regions": regions,
        "runs": rows,
        "blockers": blockers,
        "adopted": False,
    }
    if fixture_underlay is not None:
        index["source_fixture_underlay"] = source_underlay
        index["fixture_frame"] = fixture_frame.to_dict() if fixture_frame is not None else None
        if analysis_derivation is not None:
            index["painting_analysis_derivation"] = analysis_derivation
    if painting_binding is not None:
        index["painting_facies"] = painting_binding
        index["painting_geometry"] = geometry_proof
    if context is not None:
        index["display_context"] = context
    if continuation is not None:
        from .photo_edge_continuation import POLICY
        index["continuation"] = dict(policy=POLICY, file="continuation.json",
            **{key: value for key, value in _bound(folder / "continuation.json").items() if key != "path"}, adopted=False)
        index["approved_source"] = deepcopy(dict(approved_source))
        index["painting_geometry"] = geometry_proof
        from .photo_edge_continuation import verify_native_proof
        verify_source(approved_source, layers, continuation,
            preparation=index if painting_binding is not None else None, folder=folder)
        verify_native_proof(index, folder)
    if target_approvals is not None:
        from .painted_target_integration import KEY as TARGET_KEY, prepare_target_corrections
        index[TARGET_KEY] = prepare_target_corrections(index, folder, target_approvals)
    if painting_binding is not None:
        if continuation is not None:
            from .painted_analysis_exclusions import (
                KEY as EXCLUSIONS_KEY, GapReviewRequired, build_gap_review, prepare_exclusions)
            from .raw_shape_basis_reader import has_shape_basis
            reviewable = all(has_shape_basis(geometries[row["run_id"]].model) for row in rows if row.get("status") == "prepared")
            review = build_gap_review(index, folder) if (review_findings or context is not None) and reviewable else None
            if review is not None:
                index["gap_review"] = deepcopy(review["review_binding"])
            if exclusion_approval is not None:
                try:
                    index[EXCLUSIONS_KEY] = prepare_exclusions(index, folder, exclusion_approval)
                except ValueError as error:
                    if reviewable:
                        raise GapReviewRequired(review or build_gap_review(index, folder), reason=str(error)) from error
                    raise
            else:
                # Completed run artifacts are retained for an explicit review;
                # no successful preparation index or adoption is claimed.
                has_gaps = False
                for row in rows:
                    if row.get("status") == "prepared":
                        from .painted_target_integration import effective_product
                        arrays, _ = effective_product(index, folder, row)
                        has_gaps |= bool(np.any(arrays["gaps"]))
                if has_gaps:
                    if reviewable:
                        raise GapReviewRequired(review or build_gap_review(index, folder))
        elif exclusion_approval is not None:
            raise ValueError("Analysis gap approval requires the exact V2 native continuation")
        from .painted_adoption_table import prepare_outside_proof
        proof = prepare_outside_proof(folder, painting_binding, rows, preparation=index)
        if proof is not None:
            index["outside_frame_proof"] = proof
    temporary = folder / f".index.json.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if painting_binding is not None:
        from .painted_adoption_table import verify_preparation_publication
        verify_preparation_publication(index, folder)
    os.replace(temporary, folder / "index.json")
    return dict(index, folder=str(folder))


def read_preparation(folder: Path | str) -> dict[str, Any]:
    """An adoption preparation folder's index, with ``folder`` added."""

    from .adoption_check_progress import report_check_progress

    report_check_progress("Reading and verifying the prepared map…")
    folder = Path(folder)
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    if index.get("format") not in {PREPARATION_FORMAT, PREPARATION_FORMAT_V2}:
        raise ValueError(f"{folder} is not an adoption preparation folder")
    from .local_map_adoption import PROOF_FORMAT, verify_local_preparation
    if (index.get("painting_geometry") or {}).get("format") == PROOF_FORMAT:
        if index.get("format") != PREPARATION_FORMAT_V2:
            raise ValueError("Local reviewed maps require their V2 native product closure")
        verify_local_preparation(index, folder)
        return dict(index, folder=str(folder))
    if index.get("format") == PREPARATION_FORMAT_V2:
        from .photo_edge_continuation import POLICY, read_contracts, validate_contract, verify_product, verify_source, verify_native_proof
        from .geometry_raster_sources import _integer_map
        continuation = index.get("continuation") or {}
        sidecar = folder / "continuation.json"
        actual = _bound(sidecar)
        if (continuation.get("policy") != POLICY or continuation.get("file") != sidecar.name
                or continuation.get("adopted") is not False
                or any(continuation.get(key) != actual[key] for key in ("sha256", "size_bytes"))):
            raise ValueError("The native continuation sidecar changed.")
        source = folder / "source_map.npz"
        source_bound = index.get("source_map") or {}
        actual_source = _bound(source)
        if source_bound.get("path") != source.name or any(source_bound.get(k) != actual_source[k] for k in ("sha256", "size_bytes")):
            raise ValueError("The V2 preparation source map changed.")
        original = _integer_map(source)
        captured = read_contracts(sidecar)["runs"]
        contracts = {}
        prepared_count = sum(row.get("status") == "prepared" for row in index.get("runs") or [])
        for row in index.get("runs") or []:
            if row.get("status") != "prepared":
                continue
            run = row["run_id"]
            report_check_progress(f"Verifying preparation for {str(run).upper()} ({len(contracts) + 1} of {prepared_count} runs)…")
            contract = captured.get(run)
            validate_contract(contract, verify_files=True)
            if contract["target_grid"] != "native":
                raise ValueError("The V2 preparation is not on the native label grid.")
            summary = row.get("continuation") or {}
            if (summary.get("contract_run") != run or
                    summary.get("authority") != contract["frame_authority"]["kind"]):
                raise ValueError("The V2 native continuation has another run or frame authority.")
            path = folder / run / "labels.npz"
            bound = row.get("labels") or {}
            actual_labels = _bound(path)
            if bound.get("path") != f"{run}/labels.npz" or any(bound.get(k) != actual_labels[k] for k in ("sha256", "size_bytes")):
                raise ValueError("The V2 prepared labels changed.")
            verify_product(path, original, contract, counts=row.get("continuation"))
            contracts[run] = contract
        report_check_progress("Verifying the approved source and native placement proof…")
        verify_source(index.get("approved_source"), original, contracts,
            preparation=index if "painting_facies" in index else None, folder=folder)
        if (index.get("approval") != index["approved_source"]["approval"]
                or (index.get("map_version") or {}).get("sha256") != index["approval"].get("sha256")
                or (index.get("map_version") or {}).get("approval") != index["approval"]):
            raise ValueError("The V2 preparation approval differs from its approved-version closure.")
        verify_native_proof(index, folder)
        if "target_corrections" in index:
            from .painted_target_integration import effective_product
            for row in index["runs"]:
                if row.get("status") == "prepared":
                    effective_product(index, folder, row)
    return dict(index, folder=str(folder))


def facies_table(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The rows of a configuration's full facies properties table (id, name,
    porosity, permeability), in id order."""

    from darsia.fluidflower.geometry_facies import _load_properties

    path = str(config.get("facies_properties_source_path") or "")
    if not path:
        raise ValueError("The geometry revision names no facies properties table")
    return [row for _key, row in sorted(_load_properties(Path(path)).items())]


def label_values(preparation: Mapping[str, Any]) -> list[int]:
    """Every label value in the prepared runs' maps (0 is outside the map)."""

    values = sorted(
        {
            int(value)
            for row in preparation.get("runs") or []
            if row.get("status") == "prepared"
            for value in row.get("label_values") or []
        }
    )
    if preparation.get("source_fixture_underlay") is not None:
        values = sorted(set(values) | {int(value) for row in preparation.get("runs") or []
                                      for value in row.get("fixture_underlay_layer_ids") or []})
    return values


def adoption_run_data(run_id, frame_row, labels_path, *, acquisition_state):
    """Describe a run without creating any registration or adoption authority."""
    labels = Path(labels_path)
    photograph = (frame_row.get("fields") or {}).get("shape_corrected_baseline") or {}
    return {
        "run_id": run_id,
        "enabled": True,
        "geometry_choice": "run_override",
        "acquisition_state": acquisition_state,
        "labels_path": str(labels.resolve()),
        "source_image_path": str(photograph.get("path") or ""),
    }


def adoption_candidate(
    current: Mapping[str, Any],
    folder: Path | str,
    facies_ids: Mapping[int, int] | None = None,
    *,
    facies_source_acquisition: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A geometry revision candidate adopting a prepared painted map.

    ``current`` is the current corrected workspace revision, a canonical
    edit of its imported parent; ``facies_ids`` maps each label value of the
    prepared maps to a facies id of the series' full facies table (the
    facies table of the map). The candidate keeps the parent, the frame and
    the photographs; its labels are the display run's prepared map, its
    regions take each facies' name, porosity and permeability from the
    table and the role of its layer (excluded outside the map), and every
    prepared run of the frame binding gets a ``painted-map-placement-v1``
    registration. Returns the candidate and a summary; nothing is saved.

    An optional explicit ``facies_source_acquisition`` statement binds the one
    approved painting photograph/map and decision, independently of each
    target's review photograph. Without it every new row stays unknown.
    """

    from darsia.fluidflower import geometry_raster_sources as sources
    from .adoption_check_progress import report_check_progress

    # A caller may change its input while this worker reads the preparation.
    # Keep its explicit statement separate from old target/run metadata.
    declaration = (None if facies_source_acquisition is None else deepcopy(dict(facies_source_acquisition)))
    folder = Path(folder)
    preparation = read_preparation(folder)
    config = deepcopy(dict(current.get("configuration") or {}))
    if config.get("source_mode") != sources.CANONICAL_EDIT:
        raise ValueError(
            "The adoption builds on a corrected workspace revision that is a canonical edit; "
            "the current revision is not."
        )
    parent = config["canonical_import_parent"]
    binding = dict((config.get("frame_binding") or {}).get("runs") or {})
    display = str(config["frame_binding"]["display_reference_run"])
    prepared = {
        row["run_id"]: row for row in preparation.get("runs") or [] if row.get("status") == "prepared"
    }
    if display not in prepared:
        raise ValueError(f"The display reference run {display.upper()} is not prepared")
    values = label_values(preparation)
    if "painting_facies" in preparation:
        from .painted_adoption_table import candidate_table
        report_check_progress("Verifying the automatic facies table and its saved sources…")
        contract, regions, unit, table_result = candidate_table(current, preparation, folder)
        automatic = {int(row["label_id"]): int(row["facies_id"]) for row in regions}
        if facies_ids is not None and (dict(facies_ids) != automatic
                or any(type(key) is not int or type(value) is not int for key, value in facies_ids.items())):
            raise ValueError("Approved painting facies IDs are automatic; manual remapping is not allowed")
        facies_ids = automatic
        config["painted_adoption_table"] = contract
        config["permeability_unit"] = unit
    else:
        regions = None
    missing = [value for value in values if value not in (facies_ids or {})] if regions is None else []
    if missing:
        raise ValueError(
            f"No facies id for label(s) {', '.join(str(value) for value in missing)} "
            f"({FACIES_IDS_PENDING})"
        )
    table = {int(row["facies_id"]): row for row in facies_table(config)} if regions is None else {}
    layers = {int(row["layer_id"]): row for row in preparation.get("regions") or []}
    automatic_regions = regions
    regions = [] if regions is None else regions
    for value in (values if automatic_regions is None else ()):
        facies_id = int(facies_ids[value])
        if facies_id not in table:
            raise ValueError(f"Facies id {facies_id} is not in the series' facies table")
        row = table[facies_id]
        role = str((layers.get(value) or {}).get("role") or "excluded")
        regions.append(
            {
                "label_id": value,
                "facies_id": facies_id,
                "name": row["name"],
                "analysis_role": role,
                "porosity": 0.0 if role == "colorchecker" else row["porosity"],
                "permeability": None if role == "colorchecker" else row["permeability"],
                "enabled": False if role == "colorchecker" else True,
            }
        )
    if any(row["analysis_role"] == "colorchecker" for row in regions):
        for facies_id in {row["facies_id"] for row in regions}:
            roles = {row["analysis_role"] for row in regions if row["facies_id"] == facies_id}
            if "colorchecker" in roles and len(roles) != 1:
                raise ValueError("A ColorChecker and another role cannot share one facies id")
            if preparation.get("source_fixture_underlay") is not None and len(roles) != 1:
                raise ValueError("Different drawn roles cannot share one facies id")
    source_acquisition = None
    if declaration is not None:
        from .painted_adoption_table import checked_facies_source_acquisition
        source_acquisition = checked_facies_source_acquisition(preparation, folder, declaration)
    measurement = _bound(folder / "index.json")
    authority = _bound(folder / "approval.json")
    source_map = _bound(folder / "source_map.npz")
    rows, not_prepared = [], []
    prepared_count = sum(run in prepared for run in binding)
    for run, row in binding.items():
        entry = prepared.get(run)
        if entry is None:
            not_prepared.append(run)
            continue
        from .painted_target_integration import effective_product
        report_check_progress(f"Building registration for {str(run).upper()} ({len(rows) + 1} of {prepared_count} runs)…")
        _, effective = effective_product(preparation, folder, entry)
        labels = Path(effective["path"])
        photograph = (row.get("fields") or {}).get("shape_corrected_baseline") or {}
        rows.append(
            {
                **adoption_run_data(run, row, labels,
                    acquisition_state=(source_acquisition["acquisition_state"]
                                       if source_acquisition is not None else "unknown")),
                "registration": {
                    "kind": sources.PAINTED if preparation["format"] == PREPARATION_FORMAT else sources.PAINTED_V2,
                    "parent_revision_id": parent["revision_id"],
                    "status": "painted",
                    "authority": authority,
                    "measurement": measurement,
                    "labels": _bound(labels),
                    "source_image": {
                        key: photograph.get(key) for key in ("path", "sha256", "size_bytes")
                    },
                    "source_map": source_map,
                    "reference_model": entry["reference_model"],
                    "target_model": entry["target_model"],
                },
            }
        )
        if preparation["format"] == PREPARATION_FORMAT_V2:
            rows[-1]["registration"]["continuation"] = _bound(folder / "continuation.json")
    config["run_geometry"] = dict(config.get("run_geometry") or {}, runs=rows)
    from .painted_target_integration import effective_product, KEY as TARGET_KEY
    _, display_labels = effective_product(preparation, folder, prepared[display])
    config["labels_path"] = display_labels["path"]
    config["regions"] = regions
    if preparation.get("source_fixture_underlay") is not None:
        def bound_underlay(value):
            return dict(value, path=str((folder / value["path"]).resolve()))
        config["painted_fixture_underlays"] = {
            run: dict(artifact=bound_underlay(entry["fixture_underlay"]),
                      source=bound_underlay(preparation["source_fixture_underlay"]),
                      preparation=measurement,
                      layer_to_facies={str(key): int(value) for key, value in facies_ids.items()})
            for run, entry in prepared.items()
        }
    else:
        # A later ordinary map must not inherit a preceding fixture underlay.
        config.pop("painted_fixture_underlays", None)
    config["provenance"] = {
        "origin": "approved painted map placed in every run",
        "parent_revision_id": parent["revision_id"],
        "previous_revision_id": current.get("revision_id"),
        "preparation": measurement,
        "map_version": {
            key: (preparation.get("map_version") or {}).get(key) for key in ("name", "sha256")
        },
        "registered_geometry_authorized": False,
    }
    from .local_map_adoption import PROOF_FORMAT
    if (preparation.get("painting_geometry") or {}).get("format") == PROOF_FORMAT:
        config["provenance"]["local_map_reuse"] = dict(format=PROOF_FORMAT,
            review=deepcopy(preparation["painting_geometry"]["review"]))
    if source_acquisition is not None:
        config["provenance"]["facies_source_acquisition"] = source_acquisition
    from .painted_analysis_exclusions import KEY as EXCLUSIONS_KEY
    config.pop(EXCLUSIONS_KEY, None)
    if EXCLUSIONS_KEY in preparation:
        config[EXCLUSIONS_KEY] = deepcopy(preparation[EXCLUSIONS_KEY])
        config["provenance"]["approved_analysis_gap"] = deepcopy(preparation[EXCLUSIONS_KEY])
    config.pop(TARGET_KEY, None)
    if TARGET_KEY in preparation:
        config[TARGET_KEY] = deepcopy(preparation[TARGET_KEY])
        config["provenance"][TARGET_KEY] = dict(preparation=measurement,
            bindings=deepcopy(preparation[TARGET_KEY]["bindings"]))
    summary = {
        "runs": [row["run_id"] for row in rows],
        "not_prepared": not_prepared,
        "regions": regions,
        "display_reference_run": display,
    }
    if automatic_regions is not None:
        from .painted_adoption_table import frame_basis_text
        summary["facies_table"] = dict(columns=list(table_result.columns),
            rows=[dict(row) for row in table_result.rows], sha256=table_result.sha256)
        summary["frame_basis"] = frame_basis_text(preparation["painting_facies"])
    return config, summary


__all__ = [
    "FACIES_IDS_PENDING",
    "PREPARATION_FORMAT",
    "PREPARATION_FORMAT_V2",
    "PROVENANCE_FORMAT",
    "PROVENANCE_FORMAT_V2",
    "adoption_blockers",
    "adoption_candidate",
    "adoption_run_data",
    "facies_table",
    "label_values",
    "map_facies",
    "native_labels",
    "native_with_continuation",
    "native_transport",
    "placement_provenance",
    "prepare_adoption",
    "read_preparation",
]
