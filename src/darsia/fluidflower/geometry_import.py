"""Adopt an existing raw-grid palette map through the existing geometry schema.

The palette workflow supplies the authoritative native label raster. Editable
polygons are a separate initial draft; they never prove production equivalence.
No depth, crop, registration or physical property is guessed or resampled here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping
import uuid

import numpy as np

from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower.coloured_map_import import import_coloured_map
from darsia.fluidflower.geometry_facies import (
    GeometryRevisionStore,
    _load_properties,
    default_geometry_config,
)
from darsia.fluidflower.series_geometry import (
    SeriesRunInputs,
    native_series_link,
    protocol_injection_points,
    read_series_manifest,
    rig_frame_from_inputs,
    save_injection_points,
    save_rig_frame,
    set_reference_run,
)
from darsia.fluidflower.storage_paths import resolve_storage_path


@dataclass(frozen=True)
class GeometryAdoption:
    revision_id: str
    revision_name: str
    revision: Mapping[str, Any]
    editor_session: Path
    receipt: Path


def _descriptor(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _bound_path(value: Any, series_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("An explicit existing configuration path is required")
    path = resolve_storage_path(value)
    return (path if path.is_absolute() else series_root / path).resolve()


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(label + " must be an explicitly recorded positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(label + " must be an explicitly recorded positive number") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(label + " must be an explicitly recorded positive number")
    return result


def _region_rows(config, properties, native_ids, facies_roles):
    """Join existing numeric IDs; names are not grouping keys or role guesses."""
    mapping = {}
    for raw_id, row in config.get("facies", {}).items():
        if raw_id in {"props", "path"}:
            continue
        if not isinstance(row, Mapping) or not re.fullmatch(r"\d+", str(raw_id)):
            raise ValueError("Facies mappings must use recorded non-negative integer IDs")
        facies_id = int(raw_id)
        labels = row.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"Facies {facies_id} has no explicit label list")
        for label_id in labels:
            if not isinstance(label_id, int) or isinstance(label_id, bool) or label_id < 0:
                raise ValueError("Facies labels must be non-negative integers")
            if label_id in mapping:
                raise ValueError(f"Label {label_id} has more than one facies assignment")
            mapping[label_id] = facies_id
    regions = []
    for label_id in native_ids:
        if label_id not in mapping:
            raise ValueError(f"Original palette label {label_id} has no configured facies")
        facies_id = mapping[label_id]
        if facies_id not in properties:
            raise ValueError(f"Facies {facies_id} is absent from the original properties table")
        role = facies_roles.get(facies_id)
        if role not in painting.LAYER_ROLES:
            raise ValueError(f"Facies {facies_id} needs an explicit existing analysis role")
        prop = properties[facies_id]
        # Keep the existing table loader's actual values or missing values for
        # every role. Role semantics must not invent physical source data; the
        # unmodified Store validator decides whether this candidate is usable.
        porosity = prop.get("porosity")
        permeability = prop.get("permeability")
        regions.append({"label_id": label_id, "facies_id": facies_id,
                        "name": str(prop.get("name") or f"Facies {facies_id}"),
                        "analysis_role": role, "porosity": 0.0 if role == "colorchecker" else porosity,
                        "permeability": None if role == "colorchecker" else permeability,
                        "enabled": False if role == "colorchecker" else role != "excluded"})
    return regions


def adopt_pipeline_map(
    project,
    *,
    series_root: str | Path,
    source_map: str | Path,
    source_image: str | Path,
    source_run: str,
    display_reference: SeriesRunInputs,
    configuration: str | Path,
    facies_table: str | Path,
    workspace_depth_path: str | Path,
    facies_roles: Mapping[int, str],
    permeability_unit: str,
    actor: str,
    revision_name: str = "adopted-v1",
) -> GeometryAdoption:
    """Explicitly save one adoption; never modify any pipeline input or rig.

    ``project`` is a new package created by ``create_series_geometry_project``.
    ``workspace_depth_path`` must be an actual existing same-grid depth input;
    the unmodified Store validator rejects a mismatch. No 0.01-m placeholder,
    inverse warp, interpolation or corrected-rig import is provided here.

    A historical source_run may be outside the series. The project reference is
    still the explicitly supplied registered display run. Neither identity is
    silently rewritten into the other. No export or rig rebuild is triggered.
    """
    root = Path(series_root).resolve()
    paths = {name: Path(value).resolve() for name, value in {
        "source_map": source_map, "source_image": source_image,
        "configuration": configuration, "facies_table": facies_table,
        "workspace_depth": workspace_depth_path,
    }.items()}
    if not source_run or not actor.strip() or not revision_name.strip():
        raise ValueError("Explicit source run, actor and human revision name are required")
    if permeability_unit not in (set(painting.PERMEABILITY_UNITS) | {"source_unit"}):
        raise ValueError("Use the existing source's explicit Darcy, mD or m2 unit")
    manifest = read_series_manifest(root)
    registered = (manifest.get("runs") or {}).get("ids") or []
    if display_reference.run_id not in registered or display_reference.baseline_image is None:
        raise ValueError("The display reference needs a registered run and its actual configured photograph")
    link = native_series_link(project)
    if Path(str(link.get("series_root") or "")).resolve() != root:
        raise ValueError("The geometry project belongs to another series root")
    store = GeometryRevisionStore(project)
    if store.current_path.exists() or store.revisions():
        raise FileExistsError("Adoption only creates the initial revision; existing workspace revisions are preserved")
    for path in [*paths.values(), Path(display_reference.baseline_image)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_config = paths["configuration"].read_bytes()
    if len(raw_config) > 1024 * 1024:
        raise ValueError("The explicit series configuration exceeds 1 MiB")
    config = tomllib.loads(raw_config.decode("utf-8-sig"))
    if _bound_path((config.get("labeling") or {}).get("colored_image"), root) != paths["source_map"]:
        raise ValueError("The selected source map is not the configuration's colored_image")
    if _bound_path((config.get("facies") or {}).get("props"), root) != paths["facies_table"]:
        raise ValueError("The selected properties table is not the configuration's facies.props")
    rig = config.get("rig") or {}
    width = _finite_positive(rig.get("width"), "Recorded rig width")
    height = _finite_positive(rig.get("height"), "Recorded rig height")
    crop = deepcopy((((config.get("corrections") or {}).get("curvature") or {}).get("crop")))
    if not isinstance(crop, dict) or display_reference.crop != crop:
        raise ValueError("The display reference must bind this configuration's recorded crop")
    source_reference = replace(display_reference, run_id=source_run,
                               baseline_image=paths["source_image"],
                               image_folders=(paths["source_image"].parent,),
                               injection_protocol=None, rig_folder=None,
                               config_path=paths["configuration"], crop=crop,
                               width_m=width, height_m=height)
    source_frame = rig_frame_from_inputs(source_reference)
    display_frame = rig_frame_from_inputs(display_reference)
    if source_frame is None or display_frame is None:
        raise ValueError("Four existing configured crop corners are required")
    protocol = display_reference.injection_protocol
    if protocol is None or not Path(protocol).is_file():
        raise ValueError("The display reference needs its actual injection protocol")
    points = protocol_injection_points(Path(protocol))
    if not points:
        raise ValueError("The existing injection protocol contains no finite recorded point")
    paths["display_image"] = Path(display_reference.baseline_image).resolve()
    paths["display_injection_protocol"] = Path(protocol).resolve()
    sources = {name: _descriptor(path) for name, path in paths.items()}
    session_target = painting.session_path(project, source_run)
    if session_target.exists() or painting.proposal_path(project, source_run).exists():
        raise FileExistsError("The source run already has an editor draft; it will not be overwritten")
    work = Path(project.root) / "annotations" / "geometry_import" / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=False)
    frozen_table = work / ("source_facies" + paths["facies_table"].suffix)
    with frozen_table.open("xb") as stream:
        stream.write(paths["facies_table"].read_bytes())
    frozen_config = work / "source_configuration.toml"
    with frozen_config.open("xb") as stream:
        stream.write(raw_config)
    imported = import_coloured_map(paths["source_map"], source_reference, config,
                                  input_grid="raw", work_root=work, frame=source_frame)
    with np.load(imported.native_labels, allow_pickle=False) as archive:
        labels = archive["array"]
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("The original palette workflow did not yield its native 2-D integer raster")
        native_ids = sorted(int(value) for value in np.unique(labels))
    properties = _load_properties(paths["facies_table"])
    regions = _region_rows(config, properties, native_ids, facies_roles)
    by_label = {row["label_id"]: row for row in regions}
    session = imported.session
    session.facies.clear()
    for layer in session.layers:
        match = re.fullmatch(r"Imported layer (\d+)", layer.name)
        if match is None or int(match.group(1)) not in by_label:
            raise ValueError("The original import's editor-to-native label binding is unavailable")
        region = by_label[int(match.group(1))]
        key = f"Facies {region['facies_id']}: {region['name']}"
        layer.facies = key
        session.facies[key] = painting.FaciesEntry(region["analysis_role"], region["porosity"], region["permeability"])
    session.permeability_unit = permeability_unit
    imported_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "origin": "adopted from pipeline-era map", "revision_name": revision_name,
        "source_image": str(paths["source_image"]), "source_image_sha256": sources["source_image"]["sha256"],
        "source_map": str(paths["source_map"]), "source_map_sha256": sources["source_map"]["sha256"],
        "facies_table_sha256": sources["facies_table"]["sha256"], "configuration_sha256": sources["configuration"]["sha256"],
        "reference_run": source_run, "source_run_registered": source_run in registered,
        "display_reference_run": display_reference.run_id, "imported_utc": imported_at,
        "explicit_adoption_request": {"permeability_unit": permeability_unit,
                                      "facies_roles": dict(facies_roles)},
        "sources": sources,
        "original_facies_table_preserved": True,
        "frozen_facies_table": _descriptor(frozen_table),
        "frozen_configuration": _descriptor(frozen_config),
        "canonical_properties_note": "Store CSV is derived; the full original table and measured extra columns remain source-bound",
        "editor_note": "Initial exact-outline draft only; its layer IDs are separate from the authoritative palette raster",
        "lossless_export_gate": "NOT RUN; label IDs, topology, facies and corrected-grid resampling must be measured against the old rig",
    }
    candidate = default_geometry_config(project)
    candidate.update({
        "source_rig_path": "", "baseline_path": str(paths["source_image"]),
        "depth_path": str(paths["workspace_depth"]), "labels_path": str(imported.native_labels),
        "facies_source_path": "", "facies_properties_source_path": str(frozen_table),
        "protocol_source_path": str(paths["display_injection_protocol"]),
        "display_image_path": str(paths["display_image"]), "draft_source_image_path": str(paths["source_image"]),
        "width_m": width, "height_m": height, "grouping_mode": "shared_facies",
        "permeability_unit": permeability_unit, "regions": regions,
        "injectors": [{"injector_id": row["location_id"], "protocol_location_id": row["location_id"],
                       "name": row["name"] or row["location_id"],
                       "x_m": row["x_m"], "y_m": row["y_m"], "uncertainty_m": None,
                       "coordinate_source": "injection protocol"}
                      for row in points],
        "provenance": provenance,
    })
    if {name: _descriptor(path) for name, path in paths.items()} != sources:
        raise RuntimeError("An original adoption source changed before revision save")
    # The Store performs its genuine shape/physical validation and creates its
    # own UUID/artifacts. No supplied validation result or ID override is used.
    revision = store.save_revision(candidate, actor=actor, reason=f"{revision_name}: adopted from pipeline-era map")
    revision_id = str(revision.get("revision_id") or "")
    if re.fullmatch(r"[0-9a-f]{32}", revision_id) is None:
        raise ValueError("The original Store did not return its genuine UUID revision ID")
    session_file = painting.save_painting_session(project, session, actor=actor)
    set_reference_run(project, display_reference.run_id, actor=actor)
    save_rig_frame(project, display_frame, actor=actor)
    save_injection_points(project, points, actor=actor)
    if {name: _descriptor(path) for name, path in paths.items()} != sources:
        raise RuntimeError("An original adoption source changed during save; preserved revision requires inspection")
    receipt = work / "adoption.json"
    payload = {
        "format": "darsia-geometry-pipeline-map-adoption-v1", "created_utc": imported_at,
        "revision_id": revision_id, "revision_name": revision_name, "sources": sources,
        "palette_import": _descriptor(imported.receipt), "authoritative_native_labels": _descriptor(imported.native_labels),
        "editor_session": _descriptor(session_file), "revision": _descriptor(store.artifact_root / revision_id / "revision.json"),
        "source_run": source_run, "display_reference_run": display_reference.run_id,
        "editor_proposal_saved": False, "classifier_fit_run": False,
        "workspace_depth_generated": False, "original_files_rewritten": False,
        "native_setup_export_run": False, "rig_rebuild_run": False,
        "lossless_export_gate": provenance["lossless_export_gate"],
    }
    with receipt.open("xb") as stream:
        stream.write((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode())
    return GeometryAdoption(revision_id, revision_name, revision, session_file, receipt)



def prepare_corrected_adoption(project, *, series_root, configuration, facies_table,
                               source_map, source_image, frame_binding, facies_roles,
                               permeability_unit, actor, revision_name="adopted-v1",
                               source_mode="region-derived", raster_diagnosis=None):
    """Prepare exact reference copies and genuine Store configuration; do not apply.

    Every run keeps all four original source descriptors. The reference copies
    can be materialised only into the planned fresh owned directory and saved
    with the original GeometryRevisionStore. No ID override, fabricated report,
    palette segmentation, painting proposal, export or Build is performed here.
    """
    from .geometry_workspace_frame import frame_kind, selected_fields, FRAME_BINDING_FORMAT
    root = Path(series_root).resolve()
    if Path(str(native_series_link(project).get("series_root") or "")).resolve() != root:
        raise ValueError("The geometry project belongs to another series root")
    manifest = read_series_manifest(root)
    registered = (manifest.get("runs") or {}).get("ids") or []
    if (not registered or len(registered) != len(set(registered))
            or frame_binding.get("format") != FRAME_BINDING_FORMAT
            or set(frame_binding.get("runs", {})) != set(registered)):
        raise ValueError("Bind the four old maps of every actual registered run")
    if not actor.strip() or not revision_name.strip() or permeability_unit not in (set(painting.PERMEABILITY_UNITS) | {"source_unit"}):
        raise ValueError("Explicit actor, revision name and source permeability unit are required")
    display = frame_binding.get("display_reference_run")
    if display != registered[0]:
        raise ValueError("The initial display reference must be the earliest registered run")
    bound_config = {"frame": "rig-corrected", "frame_binding": deepcopy(frame_binding)}
    for run_id in registered:
        own_fields = selected_fields(bound_config, run_id, check_files=True)
        row = frame_binding["runs"][run_id]
        config_source = row.get("configuration")
        if not isinstance(config_source, Mapping) or not config_source.get("path"):
            raise ValueError("Each run must retain its exact original configuration: " + run_id)
        with Path(config_source["path"]).open("rb") as stream:
            source_bytes = stream.read(1024 * 1024 + 1)
        if (len(source_bytes) > 1024 * 1024 or len(source_bytes) != config_source.get("size_bytes")
                or hashlib.sha256(source_bytes).hexdigest() != config_source.get("sha256")):
            raise ValueError("The original run configuration changed: " + run_id)
        run_config = tomllib.loads(source_bytes.decode("utf-8-sig"))
        rig_root = _bound_path((run_config.get("rig") or {}).get("path"), root)
        for field in ("shape_corrected_baseline", "labels", "facies", "depth"):
            if Path(own_fields[field]["path"]).resolve() != rig_root / (field + ".npz"):
                raise ValueError("A corrected field belongs to another original rig: " + run_id + "/" + field)
        original_depth = _bound_path((run_config.get("depth") or {}).get("depth_map"), root)
        if Path(own_fields["configured_depth_map"]["path"]).resolve() != original_depth:
            raise ValueError("The configured production depth pointer changed: " + run_id)
    if GeometryRevisionStore(project).current_path.exists() or GeometryRevisionStore(project).revisions():
        raise FileExistsError("Existing workspace revisions must be preserved")
    with Path(configuration).open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("The explicit source configuration exceeds 1 MiB")
    original = tomllib.loads(raw.decode("utf-8-sig"))
    paths = {"configuration": Path(configuration).resolve(), "facies_table": Path(facies_table).resolve(),
             "source_map": Path(source_map).resolve(), "source_image": Path(source_image).resolve()}
    if _bound_path((original.get("labeling") or {}).get("colored_image"), root) != paths["source_map"]:
        raise ValueError("The original source map differs from labeling.colored_image")
    if _bound_path((original.get("facies") or {}).get("props"), root) != paths["facies_table"]:
        raise ValueError("The original facies table differs from facies.props")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    properties = _load_properties(paths["facies_table"])
    ids = sorted({value for key, row in (original.get("facies") or {}).items()
                  if key not in {"props", "path"} for value in row.get("labels", [])})
    regions = _region_rows(original, properties, ids, facies_roles)
    fields = selected_fields(bound_config, display)
    own = Path(project.root) / "annotations" / "geometry_import" / uuid.uuid4().hex
    if own.exists():
        raise FileExistsError("Use a fresh import directory")
    copies = []
    for field in ("shape_corrected_baseline", "labels", "facies", "depth"):
        source = fields[field]
        copies.append({"source": deepcopy(source), "target": str(own / (field + ".npz"))})
    sources = {name: _descriptor(path) for name, path in paths.items()}
    copied = {field: str(own / (field + ".npz")) for field in
              ("shape_corrected_baseline", "labels", "facies", "depth")}
    candidate = default_geometry_config(project)
    candidate.update({"frame": "rig-corrected", "frame_binding": deepcopy(frame_binding),
        "baseline_path": copied["shape_corrected_baseline"], "labels_path": copied["labels"],
        "facies_source_path": copied["facies"], "depth_path": copied["depth"],
        "source_rig_path": str(Path(fields["labels"]["path"]).parent),
        "facies_properties_source_path": str(paths["facies_table"]),
        "width_m": _finite_positive((original.get("rig") or {}).get("width"), "Recorded rig width"),
        "height_m": _finite_positive((original.get("rig") or {}).get("height"), "Recorded rig height"),
        "regions": regions, "permeability_unit": permeability_unit,
        "provenance": {"origin": "adopted from pipeline-era map", "revision_name": revision_name,
            "reference_run": frame_binding["source_reference_run"], "display_reference_run": display,
            "source_run_registered": frame_binding["source_reference_run"] in registered,
            "sources": sources, "original_rig_fields": deepcopy(frame_binding["runs"]),
            "original_configured_depth_pointer": deepcopy(fields["configured_depth_map"]),
            "original_files_rewritten": False, "workspace_depth_generated": False,
            "lossless_export_gate": "NOT RUN; old-map copies are identity by construction, Store-derived facies and any export require actual pixel comparison",
            "Build_run_model": "NOT QUALIFIED; corrected input dispatch awaits the explicit downstream contract",
            "canonical_properties_note": "Store CSV is derived; the full original table remains source-bound"}})
    if source_mode == "imported-rasters":
        from .geometry_raster_sources import bind_imported_rasters
        candidate = bind_imported_rasters(candidate, sources={
            "labels": deepcopy(fields["labels"]), "facies": deepcopy(fields["facies"]),
            "facies_properties": deepcopy(sources["facies_table"])}, diagnosis=raster_diagnosis)
        candidate["provenance"]["canonical_properties_note"] = (
            "Independent original labels/facies and the FULL original CSV are authoritative; "
            "Save refuses silent recanonicalisation. See the SHA-bound diagnosis.")
    elif source_mode != "region-derived":
        raise ValueError("Choose an explicit supported geometry source mode.")
    from .geometry_workspace_frame import adopted_content_digest
    candidate["provenance"]["adopted_content_sha256"] = adopted_content_digest(candidate)
    if frame_kind(candidate) != "rig-corrected":
        raise ValueError("Incorrect prepared frame")
    return {"format": "darsia-corrected-geometry-adoption-preparation-v1", "candidate": candidate,
        "reference_copy_plan": copies, "original_source_bindings": sources,
        "source_run": frame_binding["source_reference_run"], "display_reference_run": display,
        "revision_name": revision_name, "actual_Store_UUID": None, "applied": False,
        "copied": False, "Store_called": False, "exported": False, "Build_qualified": False}


def _verify_corrected_adoption_source(row):
    """Verify a declared source by full bytes; never decode a raster."""
    if (not isinstance(row, Mapping) or not isinstance(row.get("path"), str)
            or not Path(row["path"]).is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))) is None
            or type(row.get("size_bytes")) is not int or row["size_bytes"] < 0):
        raise ValueError("Adoption sources require absolute path, full SHA and size")
    path = Path(row["path"])
    before = path.stat()
    if (before.st_size != row["size_bytes"]
            or ("mtime_ns" in row and before.st_mtime_ns != row["mtime_ns"])):
        raise ValueError("An original adoption source changed: " + str(path))
    actual = _descriptor(path)
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or actual["sha256"] != row["sha256"]
            or actual["size_bytes"] != row["size_bytes"]):
        raise ValueError("An original adoption source changed: " + str(path))
    return actual


def _write_corrected_adoption_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _require_initial_corrected_store(store):
    """An initial adoption never replaces any saved or orphaned Store work."""
    import os
    if store.current_path.exists() or store.revisions_path.exists():
        raise FileExistsError("Existing workspace revisions must be preserved")
    if store.artifact_root.exists():
        with os.scandir(store.artifact_root) as entries:
            if next(entries, None) is not None:
                raise FileExistsError("Existing Store artifacts must be preserved")


def apply_prepared_corrected_adoption(project, *, preparation, preparation_sha256, actor):
    """Explicitly apply one SHA-reviewed plan with one genuine Store call.

    The four reference archives are copied verbatim into the plan's fresh owned
    directory. Every run binding is retained. The real Store validates and
    derives its own facies/UUID/current metadata; no validation report, revision
    ID, pixel transform or configured production-depth substitute is supplied.
    All own partial files and opt-in Store partials survive failure. A retained
    initial-adoption claim prevents retry; inspection/new authorization is then
    required. This claim serializes this entry, not arbitrary external writers.
    Store-derived facies equality and a rebuilt-rig gate are NOT proved here.
    """
    from .geometry_workspace_frame import adopted_content_digest, frame_kind, selected_fields
    import traceback

    path = Path(preparation).resolve()
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if (len(raw) > 1024 * 1024
            or re.fullmatch(r"[0-9a-f]{64}", str(preparation_sha256)) is None
            or hashlib.sha256(raw).hexdigest() != preparation_sha256):
        raise ValueError("The explicit prepared adoption document changed")
    prepared = json.loads(raw.decode("utf-8-sig"))
    if (not isinstance(prepared, dict)
            or prepared.get("format") != "darsia-corrected-geometry-adoption-preparation-v1"
            or any(prepared.get(key) is not False for key in
                   ("applied", "copied", "Store_called", "exported", "Build_qualified"))
            or prepared.get("actual_Store_UUID") is not None):
        raise ValueError("Only an unexecuted original corrected adoption plan is accepted")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("An explicit adoption actor is required")
    candidate = deepcopy(prepared["candidate"])
    provenance = candidate.get("provenance") or {}
    binding = candidate.get("frame_binding") or {}
    if (frame_kind(candidate) != "rig-corrected"
            or provenance.get("adopted_content_sha256") != adopted_content_digest(candidate)
            or provenance.get("sources") != prepared.get("original_source_bindings")
            or provenance.get("original_rig_fields") != binding.get("runs")
            or provenance.get("reference_run") != prepared.get("source_run")
            or provenance.get("display_reference_run") != prepared.get("display_reference_run")
            or provenance.get("revision_name") != prepared.get("revision_name")):
        raise ValueError("The prepared corrected adoption identity is inconsistent")
    link = native_series_link(project)
    root = Path(str(link.get("series_root") or "")).resolve()
    if not link.get("series_root"):
        raise ValueError("The geometry workspace requires its original series link")
    manifest_path = root / "series.toml"
    manifest_source = _descriptor(manifest_path)
    manifest = read_series_manifest(root)
    registered = (manifest.get("runs") or {}).get("ids") or []
    display = prepared["display_reference_run"]
    if (not registered or len(registered) != len(set(registered))
            or set(binding.get("runs", {})) != set(registered)
            or display != registered[0] or binding.get("display_reference_run") != display
            or binding.get("source_reference_run") != prepared["source_run"]
            or link.get("reference_run") != display):
        raise ValueError("The prepared run list or initial display reference changed")

    sources = dict(prepared["original_source_bindings"])
    sources["series_manifest"] = manifest_source
    for run_id in registered:
        fields = selected_fields(candidate, run_id, check_files=True)
        row = binding["runs"][run_id]
        for name in ("shape_corrected_baseline", "labels", "facies", "depth"):
            sources[run_id + "/" + name] = fields[name]
        for name in ("configuration", "facies_properties", "configured_depth_map"):
            sources[run_id + "/" + name] = row.get(name)
    for index, correction in enumerate(binding.get("reference_shape_corrections") or []):
        sources["reference_shape_correction/" + str(index)] = correction
    # Verify all declared original inputs once before any materialization.
    verified = {key: _verify_corrected_adoption_source(value) for key, value in sources.items()}
    reference = selected_fields(candidate, display, check_files=True)
    if provenance.get("original_configured_depth_pointer") != reference["configured_depth_map"]:
        raise ValueError("The original configured depth pointer must remain separate")
    copies = prepared.get("reference_copy_plan")
    fields = ("shape_corrected_baseline", "labels", "facies", "depth")
    config_paths = ("baseline_path", "labels_path", "facies_source_path", "depth_path")
    if not isinstance(copies, list) or len(copies) != 4:
        raise ValueError("Exactly the four declared reference copies are required")
    work = Path(str(copies[0].get("target") or "")).resolve().parent
    owner_root = (Path(project.root) / "annotations" / "geometry_import").resolve()
    if (work.parent != owner_root or re.fullmatch(r"[0-9a-f]{32}", work.name) is None
            or work.exists()):
        raise FileExistsError("Use the prepared fresh owned import directory")
    for copy, field, config_path in zip(copies, fields, config_paths):
        target = work / (field + ".npz")
        if (copy.get("source") != reference[field]
                or Path(str(copy.get("target") or "")).resolve() != target
                or Path(str(candidate.get(config_path) or "")).resolve() != target):
            raise ValueError("The prepared reference-copy plan changed")

    store = GeometryRevisionStore(project)
    _require_initial_corrected_store(store)
    owner_root.mkdir(parents=True, exist_ok=True)
    claim = owner_root / "initial_corrected_adoption_claim.json"
    _write_corrected_adoption_json(claim, {
        "format": "darsia-corrected-geometry-initial-adoption-claim-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "actor": actor,
        "preparation": {"path": str(path), "sha256": preparation_sha256, "size_bytes": len(raw)},
        "owned_directory": str(work), "automatic_retry": False,
    })
    # Never remove the claim, own copy directory, or any partial Store output.
    copied = {}
    store_called = False
    work_created = False
    phase = "owned directory"
    try:
        work.mkdir(exist_ok=False)
        work_created = True
        with (work / "preparation.json").open("xb") as stream:
            stream.write(raw)
        phase = "reference copies"
        for copy, field in zip(copies, fields):
            source = Path(copy["source"]["path"])
            target = Path(copy["target"])
            before = source.stat()
            digest = hashlib.sha256()
            count = 0
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                while block := incoming.read(1024 * 1024):
                    outgoing.write(block)
                    digest.update(block)
                    count += len(block)
            after = source.stat()
            if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                    or digest.hexdigest() != copy["source"]["sha256"]
                    or count != copy["source"]["size_bytes"]):
                raise ValueError("An original reference changed during its exact copy")
            copied[field] = _descriptor(target)
            if (copied[field]["sha256"] != copy["source"]["sha256"]
                    or copied[field]["size_bytes"] != copy["source"]["size_bytes"]):
                raise ValueError("A materialized reference copy changed bytes")
        phase = "pre-Store source check"
        if {key: _verify_corrected_adoption_source(value) for key, value in sources.items()} != verified:
            raise RuntimeError("An original input changed before the Store call")
        _require_initial_corrected_store(store)
        phase = "original Store"
        store_called = True
        revision = store.save_revision(
            candidate, actor=actor,
            reason=f"{prepared['revision_name']}: adopted from pipeline-era corrected maps",
            preserve_failed_artifacts=True,
        )
        revision_id = str(revision.get("revision_id") or "")
        if re.fullmatch(r"[0-9a-f]{32}", revision_id) is None:
            raise ValueError("The original Store did not return its genuine UUID")
        stored = revision.get("configuration") or {}
        if (stored.get("frame_binding") != binding
                or stored.get("provenance") != provenance
                or adopted_content_digest(stored) != provenance["adopted_content_sha256"]):
            raise ValueError("The Store changed the prepared non-depth configuration")
        revision_path = store.artifact_root / revision_id / "revision.json"
        if json.loads(revision_path.read_bytes()) != revision or json.loads(store.current_path.read_bytes()) != revision:
            raise ValueError("The genuine Store revision/current publication is inconsistent")
        phase = "post-Store source check"
        if {key: _verify_corrected_adoption_source(value) for key, value in sources.items()} != verified:
            raise RuntimeError("An original input changed during Store save; preserved revision needs inspection")
        receipt = work / "adoption.json"
        payload = {
            "format": "darsia-corrected-geometry-adoption-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "actor": actor,
            "revision_id": revision_id, "revision_name": prepared["revision_name"],
            "preparation": _descriptor(work / "preparation.json"), "claim": _descriptor(claim),
            "source_run": prepared["source_run"], "display_reference_run": display,
            "sources": verified, "reference_copies": copied,
            "revision": _descriptor(revision_path), "current": _descriptor(store.current_path),
            "Store_artifacts": deepcopy(revision.get("artifacts") or {}),
            "original_per_run_bindings_unchanged": stored["frame_binding"] == binding,
            "original_configured_depth_pointer": deepcopy(reference["configured_depth_map"]),
            "Store_called_once": True, "Store_validation_overridden": False,
            "Store_UUID_overridden": False, "all_original_sources_unchanged": True,
            "original_files_rewritten": False, "workspace_depth_generated": False,
            "editor_proposal_saved": False, "native_setup_export_run": False,
            "rig_rebuild_run": False,
            "lossless_export_gate": "NOT RUN; actual Store-derived facies equality is required",
            "rebuilt_rig_gate": "NOT RUN; original labels/facies/porosity and configured depth must be compared",
        }
        _write_corrected_adoption_json(receipt, payload)
        return {"revision_id": revision_id, "revision_name": prepared["revision_name"],
                "revision": revision, "receipt": receipt}
    except Exception as error:
        partials = []
        import os
        if store.artifact_root.exists():
            with os.scandir(store.artifact_root) as entries:
                partials = [entry.path for entry in entries]
        failure = {
            "format": "darsia-corrected-geometry-adoption-failure-v1",
            "observed_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "Store_called": store_called,
            "owned_directory": str(work), "claim": str(claim),
            "completed_reference_copies": copied, "Store_partial_paths": partials,
            "actual_current_exists": store.current_path.exists(),
            "automatic_rollback": False, "automatic_retry": False, "deletion_requested": False,
        }
        try:
            failure_path = (work / "failure.json" if work_created else
                            owner_root / ("initial_adoption_failure_" + work.name + ".json"))
            _write_corrected_adoption_json(failure_path, failure)
        except Exception as receipt_error:
            error.add_note("Could not write preserved adoption failure receipt: " + str(receipt_error))
        raise


def save_original_raw_reference_geometry(project, *, series_root, run_id, source_bindings, actor):
    """Save only genuine raw-photo corners and metric protocol points.

    This is an explicit metadata import, not part of Store save or Build. The
    caller supplies full-SHA descriptors for the actual manifest, common/run
    configuration, raw baseline and injection protocol. Missing real corners or
    points return missing/no-save; changed identities raise. The corrected
    layer canvas keeps its own saved-rig frame and is never rewritten here.
    """
    from .series_geometry import series_run_inputs

    root = Path(series_root).resolve()
    link = native_series_link(project)
    if (Path(str(link.get("series_root") or "")).resolve() != root
            or link.get("reference_run") != run_id):
        raise ValueError("Raw geometry import needs this workspace's exact registered reference")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("An explicit raw geometry import actor is required")
    if not isinstance(source_bindings, Mapping):
        raise ValueError("Explicit raw geometry source bindings are required")
    required = {"manifest", "run_configuration", "raw_baseline", "injection_protocol"}
    if not required <= set(source_bindings) or set(source_bindings) - (required | {"common_configuration"}):
        raise ValueError("Bind only the actual raw-frame manifest/configuration/photo/protocol inputs")

    result = {"format": "darsia-original-raw-reference-geometry-import-v1",
              "run_id": run_id, "frame_scope": "raw_photograph",
              "injection_coordinate_scope": "metric_rig_coordinates_from_protocol",
              "raw_frame_saved": False, "protocol_points_saved": False,
              "existing_metadata_reused": False, "missing": [],
              "raw_metadata_complete": False,
              "corrected_canvas_frame_changed": False,
              "revision_or_current_written": False, "raster_transformed": False}

    def missing(message):
        result["missing"].append(message)
        return result

    def bound_path(key):
        row = source_bindings[key]
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError("An explicit raw geometry source descriptor is required: " + key)
        return Path(row["path"]).resolve()

    # Verify metadata before the original reader consumes it. Missing inputs
    # are distinct from a SHA/path mismatch, which must remain a hard error.
    manifest_path = root / "series.toml"
    if bound_path("manifest") != manifest_path:
        raise ValueError("Raw-frame manifest binding belongs to another series")
    if source_bindings["manifest"].get("size_bytes", 0) > 1024 * 1024:
        raise ValueError("Raw-frame manifest exceeds the 1 MiB metadata limit")
    try:
        verified = {"manifest": _verify_corrected_adoption_source(source_bindings["manifest"])}
    except FileNotFoundError:
        return missing("The original series manifest is unavailable")
    manifest = read_series_manifest(root)
    if run_id not in ((manifest.get("runs") or {}).get("ids") or []):
        raise ValueError("The raw-frame source run is not registered in this series")
    paths = manifest.get("paths") or {}
    common = _bound_path(paths.get("common_config", "config/common.toml"), root)
    config_files = (manifest.get("runs") or {}).get("config_files") or {}
    partial = _bound_path(config_files.get(run_id) or
                          f"{paths.get('run_configs', 'config/runs')}/{run_id}.toml", root)
    expected_metadata = {"run_configuration": partial}
    if common.is_file():
        expected_metadata["common_configuration"] = common
    elif "common_configuration" in source_bindings:
        return missing("The original declared common configuration is unavailable")
    if set(source_bindings) != required | set(expected_metadata):
        raise ValueError("The original reader's common/run configuration scope is not bound")
    for key, path in expected_metadata.items():
        if bound_path(key) != path:
            raise ValueError("Raw-frame source configuration binding changed: " + key)
        if source_bindings[key].get("size_bytes", 0) > 1024 * 1024:
            raise ValueError("Raw-frame configuration exceeds the 1 MiB metadata limit")
        try:
            verified[key] = _verify_corrected_adoption_source(source_bindings[key])
        except FileNotFoundError:
            return missing("Original raw-frame configuration is unavailable: " + str(path))

    references = series_run_inputs(root, runs=[run_id], check_files=True)
    if len(references) != 1 or references[0].run_id != run_id:
        raise ValueError("The original series reader did not resolve exactly this registered run")
    reference = references[0]
    if reference.issues:
        return missing("; ".join(reference.issues))
    for key, path in (("raw_baseline", reference.baseline_image),
                      ("injection_protocol", reference.injection_protocol)):
        if path is None or not Path(path).is_file():
            return missing("The original " + key.replace("_", " ") + " is unavailable")
        if bound_path(key) != Path(path).resolve():
            raise ValueError("The bound raw-frame source is not the original reader's " + key)
        verified[key] = _verify_corrected_adoption_source(source_bindings[key])
    try:
        frame = rig_frame_from_inputs(reference)
    except (TypeError, ValueError, KeyError) as error:
        return missing("The original raw-photograph crop is incomplete: " + str(error))
    names = ("top_left", "bottom_left", "bottom_right", "top_right")
    if (frame is None or not frame.complete()
            or any(not math.isfinite(value) for name in names for value in frame.corners[name])):
        return missing("Four finite original raw-photograph crop corners are missing")
    try:
        _finite_positive(frame.width_m, "Recorded raw rig width")
        _finite_positive(frame.height_m, "Recorded raw rig height")
    except ValueError as error:
        return missing(str(error))
    points = protocol_injection_points(Path(reference.injection_protocol))
    if not points:
        return missing("The original injection protocol contains no finite recorded metric point")
    # These are exactly the existing parser's IDs/values. Never fill pixel
    # coordinates, invent ports, or assign raw pixels to the corrected canvas.
    old_frame = ((link.get("rig_frames") or {}).get(run_id))
    if old_frame:
        old_frame = {key: value for key, value in old_frame.items() if key != "updated_utc"}
        if old_frame != frame.to_dict():
            return missing("An existing raw rig frame differs; it was preserved for explicit review")
    old_points = link.get("injection_points")
    if old_points and old_points != points:
        return missing("Existing injection points differ; they were preserved for explicit review")
    if {key: _verify_corrected_adoption_source(row) for key, row in source_bindings.items()} != verified:
        raise RuntimeError("An original raw-frame source changed before metadata save")
    if not old_frame:
        save_rig_frame(project, frame, actor=actor)
        result["raw_frame_saved"] = True
    if not old_points:
        save_injection_points(project, points, actor=actor)
        result["protocol_points_saved"] = True
    if {key: _verify_corrected_adoption_source(row) for key, row in source_bindings.items()} != verified:
        raise RuntimeError("An original raw-frame source changed during metadata save; inspect preserved metadata")
    result.update(sources=verified, raw_frame=frame.to_dict(), protocol_points=points,
                  existing_metadata_reused=bool(old_frame or old_points),
                  raw_metadata_complete=True)
    return result
