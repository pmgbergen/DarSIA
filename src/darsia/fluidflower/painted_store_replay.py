"""Replay already materialized painted Store evidence for one Apply run.

Candidate Check/Save still derives and validates the complete painting. This
reader requires the genuine committed revision and its successful exact Check;
it consumes that revision's CSV and the requested run's categorical products.
"""
from copy import deepcopy
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

FORMAT = "darsia-committed-painted-store-run-v1"


def _metadata(current, revision_pin):
    from .painted_adoption_table import _checked, _same
    from .geometry_facies import GEOMETRY_REVISION_FORMAT, geometry_config_digest
    from .storage_paths import resolve_storage_path, same_storage_path
    revision = _checked(revision_pin, metadata=True)
    if (not _same(revision, current) or current.get("format") != GEOMETRY_REVISION_FORMAT
            or current.get("schema") != 1 or not current.get("recorded_by") or not current.get("reason")):
        raise ValueError("Painted Store replay needs its actual immutable revision and decision")
    path = Path(resolve_storage_path(revision_pin["path"]))
    if (re.fullmatch(r"[0-9a-f]{32}", str(current.get("revision_id"))) is None
            or path.name != "revision.json" or path.parent.name != current["revision_id"]
            or not same_storage_path(current["artifacts"]["facies_properties_path"], path.parent / "facies_properties.csv")):
        raise ValueError("The committed painted table is outside its genuine Store UUID")
    report = current.get("validation") or {}
    if (report.get("format") != "darsia-fluidflower-geometry-validation-v1"
            or report.get("schema") != 1
            or report.get("config_sha256") != geometry_config_digest(current["configuration"])
            or type(report.get("summary", {}).get("error_count")) is not int
            or report["summary"]["error_count"] != 0):
        raise ValueError("Painted Store replay needs the successful Check for its exact configuration")
    return revision


def _table(current):
    """Reconstruct the original immutable table type from its published bytes."""
    from . import adoption_facies_table as typed
    from .painted_adoption_table import _checked, _read, _same, table_region_data, FORMAT as TABLE_FORMAT
    from .geometry_raster_sources import _full_csv
    config = current["configuration"]
    contract = config.get("painted_adoption_table") or {}
    if contract.get("format") != TABLE_FORMAT or config.get("grouping_mode") != "shared_facies":
        raise ValueError("The committed revision has no explicit painted facies-table authority")
    spec = contract["table"]
    pin = current["artifacts"]["source_mode_files"]["facies_properties"]
    actual = _checked(pin)
    if any(actual[key] != spec[key] for key in ("sha256", "size_bytes")):
        raise ValueError("The materialized painted facies table differs from its committed recipe")
    raw = _read(actual["path"], typed.MAX_PARENT_BYTES)
    columns, rows, ids = _full_csv(None, csv_bytes=raw, strict=True, max_rows=typed.MAX_PARENT_ROWS)
    if columns != spec["columns"] or spec.get("name_policy") != typed.NAME_POLICY:
        raise ValueError("The materialized painted table changed its original columns or name policy")
    try:
        provenance = tuple(typed.FaciesRowProvenance(name=row["name"], facies_id=row["facies_id"],
            layer_ids=tuple(row["layer_ids"]), parent_ids=tuple(row["parent_ids"]),
            fields=tuple(tuple(item) for item in row["fields"])) for row in spec["provenance"])
        mapping = {int(key): value for key, value in spec["layer_to_facies"].items()}
        if (any(str(key) not in spec["layer_to_facies"] or type(key) is not int or key < 0 for key in mapping)
                or any(type(value) is not int or value < 0 for value in mapping.values())):
            raise ValueError("Invalid materialized layer-to-facies identities")
        facies_ids = {row.name: row.facies_id for row in provenance}
        if len(facies_ids) != len(provenance) or set(facies_ids.values()) != set(ids):
            raise ValueError("Materialized CSV identities differ from the approved table provenance")
        table = typed.AdoptionFaciesTable(tuple(columns), tuple(MappingProxyType(row) for row in rows),
            MappingProxyType(mapping), MappingProxyType(facies_ids), raw,
            contract["source"]["csv"]["sha256"], actual["sha256"], provenance,
            name_policy=spec["name_policy"], source_max_id=spec["source_max_id"],
            retired_source_ids=tuple(spec["retired_source_ids"]), changed_source_ids=tuple(spec["changed_source_ids"]),
            aliases=tuple(typed.FaciesAlias(**row) for row in spec["aliases"]))
        rebuilt, regions = table_region_data(table, {row["label_id"] for row in config["regions"]})
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The committed painted table has invalid typed row provenance") from error
    if not _same(rebuilt, spec) or not _same(regions, config["regions"]):
        raise ValueError("The committed table mapping, roles or physical properties differ from Check")
    return table, actual


def _approval(current, row):
    """Keep the original saved version and literal human approval decision."""
    from .painted_adoption_table import (_checked, _same, _snapshot, _history,
        verify_version_summary)
    from .storage_paths import same_storage_descriptor
    config = current["configuration"]
    contract = config["painted_adoption_table"]
    preparation = _checked(contract["preparation"], metadata=True)
    binding = preparation["painting_facies"]
    if not _same(_checked(binding["version"], metadata=True), binding["record"]):
        raise ValueError("The committed painting's saved version changed")
    verify_version_summary(binding, preparation["map_version"])
    if (not _same(preparation["approval"], binding["approval"])
            or not _same(_checked(row["registration"]["authority"], metadata=True), binding["approval"])
            or not same_storage_descriptor(row["registration"]["measurement"], contract["preparation"])
            or not same_storage_descriptor(config["provenance"]["preparation"], contract["preparation"])):
        raise ValueError("The committed table lost its exact preparation and painting approval")
    session = _checked(binding["files"]["session.json"], metadata=True)
    if not _same(_snapshot(session), binding["facies_snapshot"]):
        raise ValueError("The committed table's approved facies metadata changed")
    version_path = _checked(binding["version"])["path"]
    _history(Path(version_path).parent, binding["approval"])
    prepared = [item for item in preparation["runs"] if item["run_id"] == row["run_id"]]
    if len(prepared) != 1 or prepared[0].get("status") != "prepared":
        raise ValueError("The selected run is absent from the committed painting preparation")
    for key in ("reference_model", "target_model"):
        if not same_storage_descriptor(row["registration"][key], prepared[0][key]):
            raise ValueError("The selected registration changed its original measured model identity")
    if not _same(config.get("target_corrections"), preparation.get("target_corrections")):
        raise ValueError("The committed target-correction authority changed")
    if not _same(config.get("analysis_exclusions"), preparation.get("analysis_exclusions")):
        raise ValueError("The committed analysis-exclusion authority changed")
    return preparation, binding


def _selected_products(current, row, files):
    """Keep original selected-run numeric/grid checks without foreign products."""
    from .geometry_facies import _canonical_facies, _darsia_metadata
    from .geometry_raster_sources import _integer_map, _same_grid_metadata
    from .painted_adoption_table import _checked
    from .storage_paths import same_storage_path
    original = _checked(row["registration"]["labels"])
    if not same_storage_path(row["registration"]["labels"]["path"], row["labels_path"]):
        raise ValueError("The selected registered label path changed")
    expected_files = current["artifacts"]["run_geometry"][row["run_id"]]
    if any(not same_storage_path(pin["path"], expected_files[name + "_path"]) for name, pin in files.items()):
        raise ValueError("The painted Store receipt points to another run's products")
    labels, expected = _integer_map(files["labels"]["path"]), _integer_map(original["path"])
    if labels.dtype != np.dtype("int32") or not np.array_equal(labels, expected):
        raise ValueError("Stored registered labels differ from the approved selected run")
    if not _same_grid_metadata(_darsia_metadata(Path(files["labels"]["path"])), _darsia_metadata(Path(original["path"]))):
        raise ValueError("Stored registered labels changed their grid metadata")
    run_config = deepcopy(current["configuration"])
    run_config["labels_path"] = files["labels"]["path"]
    canonical, _properties = _canonical_facies(run_config, run_id=row["run_id"])
    facies = _integer_map(files["facies"]["path"])
    if facies.dtype != np.dtype("int32") or not np.array_equal(facies, canonical):
        raise ValueError("Stored registered facies differ from the committed layer-to-facies mapping")
    if not _same_grid_metadata(_darsia_metadata(Path(files["facies"]["path"])), _darsia_metadata(Path(files["labels"]["path"]))):
        raise ValueError("Stored registered facies changed their grid metadata")
    return list(labels.shape)


def registered_run(current, run, *, current_pin, revision_pin):
    """Return the original registered-run binding with explicit Store evidence."""
    from .geometry_facies import resolve_run_geometry_artifacts, _effective_run_geometry_choice
    from .geometry_raster_sources import descriptor, PAINTED, PAINTED_V2
    from .storage_paths import resolve_storage_path
    _metadata(current, revision_pin)
    table, csv_pin = _table(current)
    config = current["configuration"]
    selected = [row for row in config["run_geometry"]["runs"] if row["run_id"].casefold() == run]
    if len(selected) != 1 or _effective_run_geometry_choice(selected[0]) != "run_override":
        raise ValueError("No committed painted override exists for the selected run")
    row = selected[0]
    preparation, painting = _approval(current, row)
    registration = row["registration"]
    if registration.get("kind") not in {PAINTED, PAINTED_V2}:
        raise ValueError("The committed run is not an approved painted placement")
    artifacts = resolve_run_geometry_artifacts(current, run)
    if artifacts.get("scope") != "run_override" or artifacts.get("run_id") != run:
        raise ValueError("No genuine Store run artifact exists; series fallback is forbidden")
    directory = Path(resolve_storage_path(revision_pin["path"])).parent / "runs" / run
    files = {}
    for name, leaf in (("labels", "labels.npz"), ("facies", "facies.npz"),
                       ("facies_properties", "facies_properties.csv")):
        path = Path(resolve_storage_path(artifacts[name + "_path"]))
        if path.resolve() != (directory / leaf).resolve():
            raise ValueError("Committed run products belong to another Store UUID or run")
        files[name] = descriptor(path)
    if any(files["facies_properties"][key] != csv_pin[key] for key in ("sha256", "size_bytes")):
        raise ValueError("The selected run's properties table differs from the committed full CSV")
    prepared_row = next(item for item in preparation["runs"] if item["run_id"] == run)
    guarded = [current_pin, revision_pin, csv_pin, config["painted_adoption_table"]["preparation"],
        registration["authority"], registration["labels"], registration["source_map"],
        registration["source_image"], registration["reference_model"], registration["target_model"],
        painting["version"], *painting["files"].values()]
    receipt = dict(format=FORMAT, current=current_pin, revision=revision_pin, run_id=run,
        configuration_sha256=current["validation"]["config_sha256"], table=csv_pin,
        preparation=config["painted_adoption_table"]["preparation"])
    result = dict(revision_id=current["revision_id"], run_id=run, current=current_pin,
        revision=revision_pin, files=files, labels_shape=list(prepared_row["shape_hw"]),
        registration=deepcopy(registration), guarded_sources=guarded, committed_painted_table=receipt,
        label_serialization="Original Store int32; exact numeric equality to the approved selected labels",
        provenance="Existing committed Store table and exact selected-run raster evidence; no new approval")
    verify(result)
    return result


def verify(binding):
    """Verify the positive typed receipt without replaying any other run."""
    from .setup_source_reads import reuse_painted_store_proof
    return reuse_painted_store_proof(binding, reader=_verify)


def _verify(binding):
    """Original numeric proof; operation reuse never retains its raster arrays."""
    from .painted_adoption_table import _checked, _same
    from .storage_paths import same_storage_descriptor, resolve_storage_path
    receipt = binding.get("committed_painted_table")
    if not isinstance(receipt, dict) or receipt.get("format") != FORMAT:
        raise ValueError("An explicit committed painted Store receipt is required")
    for name in ("current", "revision"):
        if not same_storage_descriptor(receipt[name], binding[name]):
            raise ValueError("The painted Store receipt identifies another revision")
    current = _checked(binding["current"], metadata=True)
    _metadata(current, binding["revision"])
    current_path = Path(resolve_storage_path(binding["current"]["path"]))
    revision_path = Path(resolve_storage_path(binding["revision"]["path"]))
    if (current_path.name != "current.json" or current_path.parent.name != "geometry_facies"
            or current_path.parent.parent.name != "overrides"
            or revision_path.resolve() != (current_path.parents[2] / "results" / "geometry"
                                           / current["revision_id"] / "revision.json").resolve()):
        raise ValueError("The painted Store receipt needs its genuine current/immutable UUID pair")
    _table(current)
    config = current["configuration"]
    run = binding["run_id"]
    rows = [row for row in config["run_geometry"]["runs"] if row["run_id"] == run]
    if (len(rows) != 1 or receipt["run_id"] != run or current["revision_id"] != binding["revision_id"]
            or receipt["configuration_sha256"] != current["validation"]["config_sha256"]
            or not _same(rows[0]["registration"], binding["registration"])
            or not same_storage_descriptor(receipt["preparation"], config["painted_adoption_table"]["preparation"])
            or not same_storage_descriptor(receipt["table"], current["artifacts"]["source_mode_files"]["facies_properties"])):
        raise ValueError("The painted Store receipt differs from the committed run/table authority")
    _approval(current, rows[0])
    for pin in [*binding["guarded_sources"], *binding["files"].values()]:
        _checked(pin)
    if _selected_products(current, rows[0], binding["files"]) != binding["labels_shape"]:
        raise ValueError("The painted Store receipt changed its selected native grid")
