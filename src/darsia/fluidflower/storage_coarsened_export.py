"""Export inert coarser arrays from an explicitly verified stored 1 mm run.

This standalone command writes NPZ arrays and their unmodified coarsening
metadata. It selects no simulator format, runs no native replay, and grants no
analysis-schema or production admission. Every destination must be fresh.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from . import storage_coarsening as coarsening
from . import storage_extensive_io as extensive_io
from . import storage_indexed_products as indexed
from . import storage_outputs as output
from . import storage_saved_products as saved
from .pipeline import products_publication as publication

FORMAT = "darsia-coarsened-extensive-export-v1"
FRAME_FORMAT = "darsia-coarsened-extensive-export-frame-v1"
_FALSE_FLAGS = dict(adopted=False, physical_producer_admitted=False, physical_binding_admitted=False,
    producer_identity_admitted=False, production_permission_granted=False,
    pipeline_schema_admitted=False, simulator_format_selected=False,
    native_fields_reconstructed=False)


def _literal(path):
    value = publication._absolute_literal(str(path))
    return (PureWindowsPath(value) if PureWindowsPath(value).drive
            else PurePosixPath(value))


def _inside(destination, folder):
    destination, folder = _literal(destination), _literal(folder)
    return type(destination) is type(folder) and destination.is_relative_to(folder)


def _canonical(path):
    """Canonicalize only a path explicitly accessible on this machine."""
    return Path(path).resolve(strict=False)


def _accessible_folder(folder, folders):
    path = publication._no_links(Path(folder))
    folders.add(_literal(path))
    folders.add(_literal(_canonical(path)))


def _protect_product(row, folders):
    """Original paths are passive identities, including on another machine."""
    path = _literal(row["local_path"])
    folders.add(path.parent)
    parts = PurePosixPath(row["product_id"]).parts
    if len(parts) == 3 and parts[0] in {"frames", "phase1"}:
        branch = "frames" if parts[0] == "frames" else "preparation"
        # These are the established producer layouts, not a common-ancestor
        # guess over unrelated index entries or accessible copy locations.
        if path.parent.name == parts[1] and path.parent.parent.name == branch:
            folders.add(path.parent.parent.parent)
    if row["product_id"] in {"completed.json", "manifest.json"}:
        folders.add(path.parent)


def _destination(path, folders):
    path = publication._no_links(path)
    if path.exists():
        raise FileExistsError("Coarsened exports require one fresh output directory")
    canonical = _canonical(path)
    if any(_inside(candidate, folder) for candidate in (path, canonical) for folder in folders):
        raise ValueError("Export destination is inside an index, collection or producer folder")
    return path


def _inputs(index, *, current_machine_id, collections, source_paths):
    root, document, rows = publication._read_index(index)
    resolve = indexed.indexed_asset_resolver(index,
        current_machine_id=current_machine_id, collections=collections,
        source_paths=source_paths)
    folders = set()
    _accessible_folder(root, folders)
    current_paths = set()
    for row in rows.values():
        original_folders = set()
        _protect_product(row, original_folders)
        folders.update(original_folders)
        if row["machine_id"] == current_machine_id:
            current_paths.add(row["local_path"])
            for folder in original_folders:
                _accessible_folder(str(folder), folders)
    # Protect every explicit alias and collection, including assets used only
    # by a late frame. Never stat or resolve foreign original producer paths.
    for alias in source_paths.values():
        alias = publication._no_links(Path(alias))
        _accessible_folder(alias.parent, folders)
    for collection in collections:
        bound = publication._bound(collection)
        path = publication._no_links(Path(bound["path"]))
        _accessible_folder(path.parent, folders)
    completion = saved._json(publication.read_published_metadata(index))
    if completion.get("run") != document["run"]:
        raise ValueError("Published completion differs from the indexed run")
    for record in completion["records"]:
        folders.add(_literal(record["nas_directory"]))
        # Native storage has an explicit producer attempt, which may differ
        # from the final frame's producer after a resumed run.
        bound = saved._record_asset(record, "result.json")
        result = saved._json(saved._read_bytes(bound, resolve, []))
        attempt = (result.get("prediction", {}).get("pipeline", {})
                   .get("storage", {}).get("attempt_directory"))
        if attempt is not None:
            folders.add(_literal(attempt))
            if bound["path"] in current_paths:
                _accessible_folder(attempt, folders)
    for producer in completion.get("frame_producers", []):
        if "attempt_directory" in producer:
            folders.add(_literal(producer["attempt_directory"]))
    return folders, document


def _fingerprint(product, *, case, run):
    metadata = product["metadata"]
    identity = metadata["source_binding"].get("original_frame_identity")
    if not isinstance(identity, dict) or identity.get("case_id") != case or identity.get("run") != run:
        raise ValueError("Verified extensive frame differs from the exact indexed run/order")
    raw = json.dumps(dict(source_metadata=metadata["source_metadata"],
        source_binding=metadata["source_binding"]), ensure_ascii=False,
        allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(raw).hexdigest()


def _stream(index, *, producer_kind, qualification, current_machine_id,
            collections, source_paths):
    return extensive_io.read_indexed_extensive_run(index,
        producer_kind=producer_kind, qualification=qualification,
        current_machine_id=current_machine_id, collections=collections,
        source_paths=source_paths)


def _products(run, *, indexed_run, target_resolution_mm):
    """Keep only one yielded product; retain no run-sized array collection."""
    order = run["ordered_case_ids"]
    number = 0
    for payload in run["frames"]:
        if number >= len(order):
            raise ValueError("Verified frame stream exceeds its exact case order")
        product = coarsening.coarsen_verified_extensive(payload,
            target_resolution_mm=target_resolution_mm)
        fingerprint = _fingerprint(product, case=order[number], run=indexed_run)
        yield number, order[number], product, fingerprint
        del product, payload
        number += 1
    if number != len(order):
        raise ValueError("Verified frame stream ended before its exact case order")


def export_coarsened_extensive(index, destination, *, producer_kind,
        qualification, current_machine_id, target_resolution_mm,
        collections=(), source_paths=None):
    """Export one pinned stored run, preserving partial files on any failure.

    A complete read-only first pass validates all frames and target grids before
    creating the destination. A second bounded stream rechecks the index and
    each original frame envelope/metadata. Only small fingerprints and output
    descriptors accumulate; source and coarsened arrays are never accumulated.
    """
    target = coarsening._target(target_resolution_mm)
    if producer_kind not in {"native-phase-n1", "n1-mass-only"}:
        raise ValueError("An explicit original producer kind is required")
    if qualification not in {extensive_io.QUALIFICATION, extensive_io.GENUINE_QUALIFICATION}:
        raise ValueError("An explicit original extensive qualification is required")
    index = deepcopy(index)
    collections = tuple(deepcopy(collections))
    source_paths = deepcopy(dict(source_paths or {}))
    destination = Path(destination).absolute()
    folders, document = _inputs(index, current_machine_id=current_machine_id,
        collections=collections, source_paths=source_paths)
    destination = _destination(destination, folders)
    options = dict(producer_kind=producer_kind, qualification=qualification,
        current_machine_id=current_machine_id, collections=collections,
        source_paths=source_paths)
    first = _stream(index, **options)
    order = deepcopy(first["ordered_case_ids"])
    fingerprints = []
    for _number, _case, product, fingerprint in _products(first,
            indexed_run=document["run"], target_resolution_mm=target):
        # Also retain the original optional-extension source folders. These
        # descriptors remain logical identities; no foreign filesystem probe.
        for bound in product["metadata"]["source_binding"].get("loaded_assets", []):
            folders.add(_literal(bound["path"]).parent)
        fingerprints.append(fingerprint)
        del product
    del first
    _destination(destination, folders)
    # Reopen the pinned index and bindings before the first write. This also
    # catches changed collection receipts without creating an output folder.
    second_folders, second_document = _inputs(index,
        current_machine_id=current_machine_id, collections=collections,
        source_paths=source_paths)
    if second_document != document or second_folders - folders:
        raise ValueError("The indexed producer closure changed during export preflight")
    second = _stream(index, **options)
    if second["ordered_case_ids"] != order:
        raise ValueError("The original case order changed during export preflight")
    _destination(destination, folders)
    result = dict(format=FORMAT, complete=False, state="streaming",
        products_index=index, run=document["run"], original_producer_kind=producer_kind,
        original_qualification=qualification, target_resolution_mm=target,
        source_access=dict(current_machine_id=current_machine_id,
            collections=list(collections), source_paths=source_paths),
        ordered_case_ids=order, frames=[], active_frame=None,
        source_frame_fingerprints=fingerprints,
        native_to_1mm_reexport_gate="NOT_RUN", **_FALSE_FLAGS)
    publication._fresh_root(destination)
    try:
        output._write_json(publication._target(destination, "export_started.json"), result)
        for number, case, product, fingerprint in _products(second,
                indexed_run=document["run"], target_resolution_mm=target):
            relative = "frames/frame_%06d/" % number
            result["active_frame"] = dict(number=number, case_id=case, folder=relative)
            if fingerprint != fingerprints[number]:
                raise ValueError("The original frame envelope or arrays changed between export passes")
            arrays_path = publication._target(destination, relative + "arrays.npz")
            output._write_npz(arrays_path, product["arrays"])
            metadata_path = publication._target(destination, relative + "metadata.json")
            # Do not wrap, amend or re-admit the accepted coarsener's metadata.
            output._write_json(metadata_path, product["metadata"])
            frame = dict(format=FRAME_FORMAT, complete=True, number=number,
                case_id=case, run=document["run"], products_index=index,
                original_producer_kind=producer_kind, original_qualification=qualification,
                source_access=result["source_access"],
                source_frame_fingerprint=fingerprint,
                arrays=output._asset(arrays_path), metadata=output._asset(metadata_path),
                data_format=coarsening.FORMAT, native_to_1mm_reexport_gate="NOT_RUN",
                **_FALSE_FLAGS)
            receipt_path = publication._target(destination, relative + "frame.json")
            output._write_json(receipt_path, frame)
            result["frames"].append(dict(number=number, case_id=case,
                arrays=frame["arrays"], metadata=frame["metadata"],
                receipt=output._asset(receipt_path)))
            result["active_frame"] = None
            del product, frame
        # Reverify the pinned source index before declaring the export closed.
        publication._read_index(index)
        result.update(complete=True, state="coarsened_data_export_complete")
        receipt_path = publication._target(destination, "export.json")
        output._write_json(receipt_path, result)
        return dict(result, receipt=output._asset(receipt_path))
    except Exception as error:
        result.update(complete=False, state="incomplete_failed",
            failure=dict(type=type(error).__name__, message=str(error)),
            partial_files_preserved=True)
        try:
            failure_path = publication._target(destination, "export_failure.json")
            output._write_json(failure_path, result)
            error.add_note("Partial export and failure receipt preserved at " + str(failure_path))
        except Exception as receipt_error:
            error.add_note("Partial export preserved; failure receipt unavailable: " + str(receipt_error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--index-size-bytes", required=True, type=int)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--producer-kind", required=True, choices=("native-phase-n1", "n1-mass-only"))
    parser.add_argument("--qualification", required=True,
        choices=(extensive_io.QUALIFICATION, extensive_io.GENUINE_QUALIFICATION))
    parser.add_argument("--target-resolution-mm", required=True, type=int, choices=(1, 2, 5, 10))
    parser.add_argument("--bindings", help="Explicit JSON collections/source_paths; never a search root")
    args = parser.parse_args(argv)
    bindings = {}
    if args.bindings:
        with Path(args.bindings).open("rb") as stream:
            raw = stream.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("Explicit export bindings exceed the metadata limit")
        bindings = saved._json(raw)
        if not isinstance(bindings, dict) or not bindings.keys() <= {"collections", "source_paths"}:
            raise ValueError("Unknown explicit export binding")
    result = export_coarsened_extensive(dict(path=args.index, sha256=args.index_sha256,
        size_bytes=args.index_size_bytes), args.output,
        current_machine_id=args.machine, producer_kind=args.producer_kind,
        qualification=args.qualification, target_resolution_mm=args.target_resolution_mm,
        **bindings)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
