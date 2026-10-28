"""Additional export maps from the original verified extensive-data readers.

Expected results keep their original stored resolution. Coarser attachments
use only the existing extensive-cell coarsener, never native replay, sampled
display pixels, interpolation or invented physical denominators.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import hashlib
import io
import json

import numpy as np

from .archives import Member
from .job_reader import JobError


def _json(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _resolver(job, run):
    # This is the same FinishedJob product access used by the original export.
    # Match the entire descriptor, never a basename or an unbound candidate.
    products = {}
    for product in job.products[run].values():
        matches = products.setdefault((product.local_path, product.sha256, product.size_bytes), [])
        if any(other.machine_id != product.machine_id for other in matches):
            raise JobError("One original export asset has conflicting producer machines")
        matches.append(product)

    def resolve(bound):
        matches = products.get((bound["path"], bound["sha256"], bound["size_bytes"]))
        if not matches:
            raise JobError("Export map asset is outside the finished job's product closure")
        return job.product_source(matches[0])
    return resolve


def _portable_metadata(product):
    original = product["metadata"]
    # Retain the numerical metadata verbatim; bind the full source metadata by
    # digest instead of copying machine paths into a published record.
    return dict(format="darsia-export-mass-map-v1",
                **{key: deepcopy(original[key]) for key in (
                    "grid", "source_grid", "source_precision", "output_float_dtype", "reduction_dtype",
                    "operation", "arrays", "fields", "facies_ids", "conservation", "summation_scope",
                    "native_to_1mm_reexport_gate", "native_fields_reconstructed",
                    "physical_binding_admitted", "producer_identity_admitted", "production_permission_granted")},
                channel_states={key: row["status"] for key, row in original["channels"].items()},
                original_metadata_sha256=hashlib.sha256(_json(original)).hexdigest())


def _encoded(product):
    with io.BytesIO() as stream:
        np.savez_compressed(stream, **product["arrays"])
        return stream.getvalue()


def _regenerate(record, *, producer_kind, resolve, manifest, manifest_asset, resolution):
    from ..storage_extensive_io import read_committed_payload
    from ..storage_coarsening import coarsen_verified_extensive

    source = read_committed_payload(record, producer_kind=producer_kind, resolve_asset=resolve,
                                    manifest=manifest, manifest_asset=manifest_asset)
    product = coarsen_verified_extensive(source, target_resolution_mm=resolution)
    return _encoded(product)


def stored_export_run(job, run):
    """Open the original saved run with FinishedJob's exact product closure."""
    from .. import storage_saved_products as saved
    from ..storage_n1_saved_products import read_n1_stored_run

    resolve = _resolver(job, run)
    completed = job.completed[run]
    if completed.get("pipeline") == "native-phase-n1":
        source = saved.read_stored_run(completed, resolve_asset=resolve, stream=True)
        kind = "native-phase-n1"
    elif "pipeline" not in completed and "manifest" in completed:
        source = read_n1_stored_run(completed, resolve_asset=resolve, stream=True)
        kind = "n1-mass-only"
    else:
        raise ValueError("Export resolution needs the original committed native or N1 stored run")
    if source["ordered_case_ids"] != job.frames(run):
        raise ValueError("Export mass-map order differs from the finished job's saved selection")
    return source, resolve, kind


def iter_export_products(job, run, resolution):
    """Stream coarsened fields for media through the same readers as data export."""
    from .. import storage_extensive_io as extensive
    from ..storage_coarsening import coarsen_verified_extensive

    source, resolve, _kind = stored_export_run(job, run)
    for stored in source["frames"]:
        payload = extensive._read_payload(stored, resolve)
        if payload.get("state") == "not_requested":
            raise ValueError("Coarser export needs the saved extensive fields; they were not stored")
        yield stored["case_id"], coarsen_verified_extensive(payload, target_resolution_mm=resolution)
        del stored, payload


def mass_map_members(job, resolution, folder):
    """Check one frame at a time and return bounded, regenerable archive members.

    Every generated member is checked again by the existing archive writer
    against its preflight bytes. Missing original extensive channels/geometry
    are reported by their owning reader, with no approximate fallback.
    """
    from .. import storage_extensive_io as extensive
    from .. import storage_saved_products as saved
    from ..storage_coarsening import _target, coarsen_verified_extensive

    target = _target(resolution)
    members, sections = {}, {}
    for run in job.runs:
        source, resolve, kind = stored_export_run(job, run)
        completed = job.completed[run]
        manifest_asset = deepcopy(completed["manifest"])
        manifest = saved._json(saved._read_bytes(manifest_asset, resolve, []))
        members[run], sections[run] = [], {}
        for number, stored in enumerate(source["frames"]):
            case = source["ordered_case_ids"][number]
            payload = extensive._read_payload(stored, resolve)
            if payload.get("state") == "not_requested":
                raise ValueError(f"Run {run}, {case}: coarser export needs the saved extensive fields; they were not stored")
            product = coarsen_verified_extensive(payload, target_resolution_mm=target)
            archive = _encoded(product)
            metadata = _json(_portable_metadata(product))
            base = f"{folder}/{run}/{case}/export_{target}mm"
            digest = hashlib.sha256(archive).hexdigest()
            generate = partial(_regenerate, deepcopy(completed["records"][number]),
                producer_kind=kind, resolve=resolve, manifest=manifest,
                manifest_asset=manifest_asset, resolution=target)
            members[run].extend([
                Member(base + ".npz", sha256=digest, size_bytes=len(archive), generate=generate),
                Member(base + ".json", data=metadata)])
            sections[run][case] = dict(resolution_mm=target, source_state=payload["state"],
                arrays=dict(path=base + ".npz", sha256=digest, size_bytes=len(archive)),
                metadata=dict(path=base + ".json", sha256=hashlib.sha256(metadata).hexdigest(),
                              size_bytes=len(metadata)))
            del archive, metadata, product, payload, stored
    return members, dict(resolution_mm=target, runs=sections,
                         reduction="original extensive-cell coarsener",
                         native_to_1mm_reexport_gate="NOT_RUN", production_permission_granted=False)
