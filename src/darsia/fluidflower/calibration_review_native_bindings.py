"""Admit existing native review bindings without discovery or collection.

Only the original done publication and explicitly supplied immutable receipt
descriptors are considered. N1 photograph geometry comes from its admitted
per-frame storage metadata. Native-phase retains the original exporter's
explicit photo-receipt mapping and geometry SHA. Neither branch infers a
photograph or geometry binding from current source files.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from . import storage_saved_products as saved
from .pipeline import products_publication as publication
from .pipeline.job_queue import _own_asset, name
from .storage_indexed_products import indexed_asset_resolver
from .storage_n1_outputs import FRAME_FORMAT, PRODUCER_KIND


def _normalized(bound):
    original = publication._bound(bound)
    return dict(path=original["path"], sha256=original["sha256"], size_bytes=original["bytes"])


def _owned(job_root, reference):
    path = _own_asset(job_root, reference)
    return dict(path=str(path), sha256=reference["sha256"], size_bytes=reference["bytes"])


def collect_bindings(index, job_root, done, *, current_machine_id, collections=()):
    """Return admitted descriptors of existing publications/collections only.

    This name denotes gathering bindings, not copying products. The original
    done record must bind this exact index/run/attempt. Its optional explicit
    payload publication is combined with caller-supplied full receipt bindings.
    Original resolver validation reads receipt metadata, not the payloads. It
    preserves producer identities and does not promise that every payload is
    present; the selected-frame reader checks actual bytes when requested.
    """
    if not isinstance(done, Mapping) or not isinstance(collections, (tuple, list)):
        raise ValueError("Original done metadata and explicit collection descriptors are required")
    name(current_machine_id)
    normalized_index = _normalized(index)
    _root, document, _rows = publication._read_index(normalized_index)
    if (done.get("format") != "darsia-alias-done-v1"
            or done.get("run") != document["run"]
            or done.get("attempt_id") != document["attempt_id"]):
        raise ValueError("Native review bindings differ from the original done run/attempt")
    original_index = _owned(job_root, done["products_index"])
    if publication._identity(original_index) != publication._identity(normalized_index):
        raise ValueError("Native review bindings differ from the original done products index")
    bound = []
    if "payload_publication" in done:
        payload = _owned(job_root, done["payload_publication"])
        receipt = publication._bound(payload)
        bundle = publication._json(publication._verified_bytes(
            receipt["path"], receipt, limit=publication.MAX_INDEX_BYTES))
        if bundle.get("format") != publication.BUNDLE_FORMAT:
            raise ValueError("Original done payload must be its existing publication receipt")
        bound.append(payload)
    bound.extend(_normalized(item) for item in collections)
    # A refreshed attachment may repeat its original done descriptor among its
    # explicit bindings. Only exactly identical full descriptors coalesce;
    # different receipts with overlapping product copies remain an error.
    unique, seen = [], set()
    for item in bound:
        key = item["path"], item["sha256"], item["size_bytes"]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    # Reuse every original index/run/attempt/copy/producer-identity check. The
    # returned resolver is deliberately not invoked: no payload stat/read here.
    indexed_asset_resolver(normalized_index, current_machine_id=current_machine_id,
                           collections=unique)
    return tuple(deepcopy(unique))


def photo_geometry(stored, *, expected=None, pipeline="n1"):
    """Retain each original photograph protocol's geometry requirement.

    storage_n1_outputs.write_n1_frame records this literal digest only after
    checking the saved PNG against the phase-one photograph request. The caller
    must first retain its usual frame/receipt/run/freeze guards, then pass this
    digest to read_n1_original_photo, whose original receipt/source/ROI/pixel
    validation still runs. Native-phase has a separate case-to-photo receipt
    binding and requires the export API's explicit original producer SHA;
    neither value is inferred from N1 metadata. No file or array access occurs.
    """
    if pipeline == "native-phase-n1":
        # The native-phase exporter takes a separate explicit geometry SHA;
        # its storage frame has no mass-only producer kind or photograph row.
        if (not isinstance(stored, Mapping)
                or stored.get("format") != "darsia-read-coarse-stored-frame-v1"
                or "producer_kind" in stored):
            raise ValueError("Photograph geometry requires an original native-phase stored frame")
        document = stored.get("storage_document")
        if (not isinstance(document, Mapping) or document.get("format") != saved.output.FRAME_FORMAT
                or "producer_kind" in document or document.get("complete") is not True
                or document.get("existing_native_products_unchanged") is not True
                or stored.get("grid") != document.get("coarse", {}).get("grid")
                or any(stored.get(field) != document.get(field)
                       for field in ("run", "case_id", "precision", "source"))):
            raise ValueError("Photograph geometry differs from the admitted native-phase frame identity")
        saved._descriptor(stored.get("storage_receipt"))
        publication._sha(document.get("calibration_sha256"))
        if expected is None:
            raise ValueError("Native-phase photograph requires its explicit original geometry producer SHA")
        return publication._sha(expected)
    if pipeline != "n1":
        raise ValueError("Unsupported original photograph protocol")
    if (not isinstance(stored, Mapping)
            or stored.get("format") != "darsia-read-coarse-stored-frame-v1"
            or stored.get("producer_kind") != PRODUCER_KIND):
        raise ValueError("Photograph geometry requires an admitted original N1 stored frame")
    document = stored.get("storage_document")
    if (not isinstance(document, Mapping) or document.get("format") != FRAME_FORMAT
            or document.get("producer_kind") != PRODUCER_KIND
            or document.get("complete") is not True
            or document.get("existing_native_products_unchanged") is not True
            or any(stored.get(field) != document.get(field) for field in ("run", "case_id", "precision", "source"))):
        raise ValueError("Photograph geometry differs from the admitted N1 frame identity")
    saved._descriptor(stored.get("storage_receipt"))
    publication._sha(document.get("calibration_sha256"))
    photo = document.get("photograph")
    if not isinstance(photo, Mapping) or stored.get("photograph") != photo:
        raise ValueError("Photograph geometry differs from its original stored metadata")
    if photo.get("state") == "pending":
        raise FileNotFoundError("N1 saved original photograph is explicitly pending")
    if photo.get("state") != "complete":
        raise ValueError("Original N1 photograph state is not complete")
    digest = publication._sha(photo.get("geometry_producer_sha256"))
    if expected is not None and publication._sha(expected) != digest:
        raise ValueError("Selected photograph geometry differs from its original producer request")
    return digest
