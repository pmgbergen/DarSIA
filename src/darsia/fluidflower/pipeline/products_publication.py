"""Immutable, byte-preserving publication of explicitly listed run products.

This is job-layer I/O only. The caller supplies the complete final-product list
from the original producer's descriptors, including resumed frame producers.
Nothing is discovered by walking an output/preparation directory. Indexing a
large product does not open it; an explicit publish/collect does.

``file_writers_closed=True`` is an explicit caller contract for the selected
files. It means their writers have closed and those bytes are final. It is NOT
an assertion that the worker/service OS processes have retired. Parent-written
retirement/control receipts are separate, later immutable publications.

The original Main completed.json is copied as bytes, without changing any
embedded producer paths, precision, provenance, or scientific fields. Recovery
must request a published metadata descriptor through this module; it must not
fall back to an embedded path on another machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

from .job_queue import name, rename, write_immutable
from .io_ops import retry_io

INDEX_FORMAT = "darsia-products-index-v1"
BUNDLE_FORMAT = "darsia-product-publication-v1"
COLLECTION_FORMAT = "darsia-product-collection-v1"
DEFAULT_PUBLISH = ("film", "lossless_export")
# This new byte-publication layer must admit the original full run manifest
# (the actual 39-frame AC52 manifest is 7,141,781 bytes). It changes no existing
# series/status reader cap. The explicit metadata bound matches J2 InputCatalog.
MAX_METADATA_BYTES = 128 * 1024 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
_SMALL_KINDS = frozenset({
    "completion", "manifest", "scalar_table", "frame_scalar", "commit_receipt",
    "operational_receipt", "row_status",
})


class ProductUnavailable(FileNotFoundError):
    """No explicitly bound, currently accessible copy is available here."""


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("A full lowercase SHA-256 is required")
    return value


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("A nonnegative integer byte count is required")
    return value


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Product paths must be nonempty portable relative paths")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive:
        raise ValueError("An absolute/drive product path is forbidden")
    parts = value.split("/")
    for part in parts:
        if (part in {"", ".", ".."} or part.endswith((".", " "))
                or any(ord(c) < 32 or c in ':*?"<>|' for c in part)):
            raise ValueError("Unsafe product path: " + value)
        name(part) if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", part) else None
        if part.casefold().split(".")[0] in {
            "con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
            *(f"lpt{i}" for i in range(1, 10)),
        }:
            raise ValueError("Reserved product path: " + value)
    return value


def _absolute_literal(value):
    if (not isinstance(value, str) or not value
            or any(ord(c) < 32 for c in value)
            or value.replace("/", "\\").startswith(("\\\\?\\", "\\\\.\\"))):
        raise ValueError("An ordinary absolute producer path is required")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if not (windows.is_absolute() or posix.is_absolute()):
        raise ValueError("An absolute producer path is required")
    if ".." in windows.parts or ".." in posix.parts:
        raise ValueError("Parent traversal is forbidden in producer paths")
    return value


def _identity(bound):
    if not isinstance(bound, dict):
        raise ValueError("A file descriptor is required")
    return {"sha256": _sha(bound.get("sha256")),
            "bytes": _count(bound.get("bytes", bound.get("size_bytes")))}


def _bound(bound):
    result = _identity(bound)
    result["path"] = _absolute_literal(bound.get("path"))
    return result


def product(product_id, *, machine_id, local_path, sha256, bytes, kind):
    """Normalize one producer descriptor without touching its file.

    The machine is the original producer, which may be a resumed predecessor.
    ``kind`` is a caller-declared final-product role, never inferred by suffix.
    """
    name(machine_id)
    if not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", kind):
        raise ValueError("A declared lowercase product kind is required")
    return dict(product_id=_relative(product_id), machine_id=machine_id,
                local_path=_absolute_literal(str(local_path)), sha256=_sha(sha256),
                bytes=_count(bytes), kind=kind)


def _products(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("Final products must be an explicit list or tuple")
    rows, seen = [], set()
    for value in values:
        row = product(**{key: value[key] for key in (
            "product_id", "machine_id", "local_path", "sha256", "bytes", "kind")})
        key = row["product_id"].casefold()
        if key in seen or any(key.startswith(old + "/") or old.startswith(key + "/")
                              for old in seen):
            raise ValueError("Conflicting product identities: " + row["product_id"])
        seen.add(key)
        rows.append(row)
    return rows


@retry_io
def _no_links(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("The actual accessible path must be absolute")
    if ".." in path.parts:
        raise ValueError("Parent traversal is forbidden")
    for candidate in reversed((path, *path.parents)):
        try:
            info = candidate.lstat()
        except FileNotFoundError as error:
            from .job_queue import transient_os_error
            if transient_os_error(error):
                raise
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse paths are forbidden: " + str(candidate))
    return path


@retry_io
def _file(path):
    path = _no_links(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("A regular immutable file is required: " + str(path))
    return path


def _stamp(info):
    return info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino


@retry_io
def _verified_bytes(path, expected, *, limit):
    path = _file(path)
    before = path.stat()
    if before.st_size > limit or before.st_size != expected["bytes"]:
        raise ValueError("Metadata size differs or exceeds the bounded read: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.stat()
    if (_stamp(before) != _stamp(after) or len(raw) != expected["bytes"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]):
        raise ValueError("Metadata bytes differ: " + str(path))
    return raw


@retry_io
def _verify_file(path, expected, *, limit):
    """Verify a requested shared descriptor without buffering its metadata."""
    path = _file(path)
    before = path.stat()
    if before.st_size > limit or before.st_size != expected["bytes"]:
        raise ValueError("Metadata size differs or exceeds the bounded read: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after_open = os.fstat(stream.fileno())
    after = path.stat()
    if (_stamp(before) != _stamp(opened) or _stamp(opened) != _stamp(after_open)
            or _stamp(after_open) != _stamp(after)
            or digest.hexdigest() != expected["sha256"]):
        raise ValueError("Metadata bytes differ: " + str(path))


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key: " + key)
            value[key] = item
        return value

    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)

    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs,
                       parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("A JSON object is required")
    return value


def _fresh_root(path):
    path = _no_links(path)
    from .job_queue import _create_directory
    _create_directory(path, parents=True)
    return _no_links(path)


def _target(root, relative):
    root = _no_links(root)
    destination = root.joinpath(*_relative(relative).split("/"))
    _no_links(destination)
    from .job_queue import retry_transient
    from .series_setup_sink import current_writer
    writer = current_writer()
    retry_transient(lambda: destination.parent.mkdir(parents=True, exist_ok=True)
                    if writer is None else writer.mkdir(destination.parent, parents=True, exist_ok=True))
    _no_links(destination.parent)
    if not destination.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("Publication target escapes its fresh root")
    if destination.exists():
        raise FileExistsError(destination)
    return destination


@retry_io
def _copy_pending(source, expected, destination):
    """Stream/hash, verify the closed temporary bytes, then publish exclusively."""
    source = _file(source)
    before = source.stat()
    if before.st_size != expected["bytes"]:
        raise ValueError("Product byte count differs: " + str(source))
    temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".partial")
    digest = hashlib.sha256()
    from .series_setup_sink import current_writer
    sink = current_writer()
    output = temporary.open("xb") if sink is None else sink.open_exclusive(temporary)
    with source.open("rb") as reader, output as writer:
        opened = os.fstat(reader.fileno())
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
            writer.write(block)
        writer.flush()
        if sink is None:
            os.fsync(writer.fileno())
        after = os.fstat(reader.fileno())
    if (_stamp(before) != _stamp(opened) or _stamp(opened) != _stamp(after)
            or _stamp(source.stat()) != _stamp(after)
            or digest.hexdigest() != expected["sha256"]):
        raise ValueError("Source changed/differs; partial preserved: " + str(temporary))
    check = hashlib.sha256()
    with temporary.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check.update(block)
    if temporary.stat().st_size != expected["bytes"] or check.hexdigest() != expected["sha256"]:
        raise ValueError("Copied bytes differ; partial preserved: " + str(temporary))
    _no_links(destination.parent)
    return temporary


def _copy(source, expected, destination):
    """Retry copying and committing as separate IO steps, never scientific work."""
    temporary = _copy_pending(source, expected, destination)
    if os.name == "nt":
        rename(temporary, destination)
    else:
        from .series_setup_sink import current_writer
        sink = current_writer()
        if sink is None:
            os.link(temporary, destination)  # Never replace an existing target.
            temporary.unlink()  # Only this successful extra link is removed.
        else:
            def publish_link(source, target):
                os.link(source, target)
                source.unlink()
            sink.rename(temporary, destination, rename_operation=publish_link)
    return dict(relative_path=destination.name, **expected)


def _write_index(root, filename, document):
    projected = (json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(projected) > MAX_INDEX_BYTES:
        raise ValueError("Publication index exceeds its independent bounded metadata size")
    destination = _target(root, filename)
    write_immutable(destination, document, create_parents=False)
    from .job_queue import retry_transient
    raw = retry_transient(destination.read_bytes)
    return dict(path=str(destination), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def _closed(value):
    if value is not True:
        raise ValueError("Explicit file_writers_closed=True is required before copying")


def _source(row, current_machine_id, source_paths):
    """Use an explicit accessible alias or this machine's own producer path."""
    if row["product_id"] in source_paths:
        return _file(source_paths[row["product_id"]])
    if name(row["machine_id"]) != name(current_machine_id):
        raise ProductUnavailable(
            "Product is local to another machine; supply an explicit accessible copy: "
            + row["product_id"] + " (" + row["machine_id"] + ")")
    return _file(row["local_path"])


def publish_run_metadata(publication_root, *, job, run, attempt_id, machine_id,
                         completed, products, metadata_product_ids,
                         file_writers_closed, all_final_products_listed,
                         source_paths=None):
    """Publish one immutable per-attempt metadata bundle and final product index.

    ``publication_root`` must be fresh. ``completed`` is its original full
    descriptor. ``products`` explicitly lists *all* final products, not inputs
    or cache files. Only completion and ``metadata_product_ids`` are opened.
    The caller must enumerate small frame scalars, manifests, scalar tables,
    immutable commit/registration and row-status receipts required by readers.
    Each such item is bounded to MAX_METADATA_BYTES. Never pass mutable logs,
    progress.json, active leases, or not-yet-written retirement.json here.

    The index is written LAST. A preserved partial directory without that index
    is not an admitted publication. Transient file steps are retried; scientific
    work is never rerun. Partial files are retained on final failure.
    """
    _closed(file_writers_closed)
    if all_final_products_listed is not True:
        raise ValueError("The caller must explicitly confirm its final-product list")
    name(machine_id)
    name(run)
    if not isinstance(attempt_id, str) or not re.fullmatch(r"[0-9a-f]{32}", attempt_id):
        raise ValueError("An original attempt token is required")
    job = _bound(job)
    completed = _bound(completed)
    raw = _verified_bytes(completed["path"], completed, limit=MAX_METADATA_BYTES)
    result = _json(raw)
    if (result.get("complete") is not True or result.get("run") != run
            or result.get("worker_references_closed") is not True):
        raise ValueError("An actual closed-reference Main completion is required")
    completion = product("completed.json", machine_id=machine_id,
                         local_path=completed["path"], **_identity(completed), kind="completion")
    if not isinstance(products, (list, tuple)) or not isinstance(metadata_product_ids, (list, tuple)):
        raise ValueError("Explicit final-product and metadata-product lists are required")
    rows = _products([completion, *products])
    selected = set(metadata_product_ids)
    selected.add("completed.json")
    by_id = {row["product_id"]: row for row in rows}
    if not selected <= by_id.keys():
        raise ValueError("Requested metadata is not an indexed final product")
    for key in selected:
        row = by_id[key]
        if row["kind"] not in _SMALL_KINDS or row["bytes"] > MAX_METADATA_BYTES:
            raise ValueError("Large/nonmetadata product requires explicit product publication: " + key)
    source_paths = dict(source_paths or {})
    if not source_paths.keys() <= by_id.keys():
        raise ValueError("Accessible source aliases must name explicit indexed products")
    # Validate sources before reserving the fresh bundle, without reading maps.
    accessible = {key: _source(by_id[key], machine_id, source_paths) for key in selected}
    root = _fresh_root(publication_root)
    published = []
    for row in rows:
        entry = dict(row, identity_basis="original_producer_descriptor")
        if row["product_id"] in selected:
            relative = ("completed.json" if row["product_id"] == "completed.json"
                        else "metadata/" + row["product_id"])
            target = _target(root, relative)
            _copy(accessible[row["product_id"]], _identity(row), target)
            entry["metadata_copy"] = dict(relative_path=relative, **_identity(row))
        published.append(entry)
    return _write_index(root, "products_index.json", dict(
        format=INDEX_FORMAT, job=job, run=run, attempt_id=attempt_id,
        publishing_machine_id=machine_id, products=published,
        file_closure=dict(caller_confirms_final_files_closed=True,
                          worker_references_closed=True, os_retirement_asserted=False),
        all_final_products_listed_by_caller=True,
        payload_identity_scope="Only copied files hash-verified during this publication",
    ))


def _read_index(bound):
    bound = _bound(bound)
    raw = _verified_bytes(bound["path"], bound, limit=MAX_INDEX_BYTES)
    value = _json(raw)
    if (value.get("format") != INDEX_FORMAT
            or value.get("all_final_products_listed_by_caller") is not True
            or value.get("file_closure") != dict(caller_confirms_final_files_closed=True,
                                                worker_references_closed=True,
                                                os_retirement_asserted=False)):
        raise ValueError("An immutable admitted products index is required")
    _bound(value["job"])
    name(value["run"])
    name(value["publishing_machine_id"])
    if not re.fullmatch(r"[0-9a-f]{32}", value.get("attempt_id", "")):
        raise ValueError("Invalid original attempt identity in products index")
    rows = _products(value["products"])
    if not any(row["product_id"] == "completed.json" and row["kind"] == "completion" for row in rows):
        raise ValueError("Products index has no original completion")
    root = Path(bound["path"]).parent
    for original, row in zip(value["products"], rows):
        if "metadata_copy" in original:
            expected = ("completed.json" if row["product_id"] == "completed.json"
                        else "metadata/" + row["product_id"])
            copy = original["metadata_copy"]
            if (row["kind"] not in _SMALL_KINDS or copy.get("relative_path") != expected
                    or _identity(copy) != _identity(row)):
                raise ValueError("Metadata copy is not its original indexed product")
            row["metadata_copy"] = dict(copy)
    return root, value, {row["product_id"]: row for row in rows}


def published_metadata_descriptor(index, product_id="completed.json"):
    """Return only a shared byte-copy descriptor; never a local producer path."""
    root, _document, rows = _read_index(index)
    row = rows[_relative(product_id)]
    copy = row.get("metadata_copy")
    if copy is None:
        raise ProductUnavailable("No published metadata copy: " + product_id)
    path = root.joinpath(*copy["relative_path"].split("/"))
    _verify_file(path, copy, limit=MAX_METADATA_BYTES)
    return dict(path=str(path), **_identity(copy))


def read_published_metadata(index, product_id="completed.json"):
    """Read the admitted shared copy, retaining original embedded paths verbatim."""
    copy = published_metadata_descriptor(index, product_id)
    return _verified_bytes(copy["path"], copy, limit=MAX_METADATA_BYTES)


def _bundle_copies(index, publications):
    copies = {}
    for bound in publications:
        normalized = _bound(bound)
        bundle = _json(_verified_bytes(normalized["path"], normalized, limit=MAX_INDEX_BYTES))
        # The original publication index path is passive producer provenance.
        # A lease may have atomically moved to publishing/reclaimed; bind the
        # currently supplied shared index by its exact bytes, never reopen the
        # old path or reject an unchanged index merely because its lease moved.
        if (bundle.get("format") != BUNDLE_FORMAT
                or _identity(_bound(bundle["products_index"])) != _identity(_bound(index))
                or bundle.get("file_writers_closed") is not True):
            raise ValueError("Publication bundle differs from its original products index")
        for copy in bundle["copies"]:
            key = _relative(copy["product_id"])
            relative = _relative(copy["relative_path"])
            if relative != "products/" + key or key in copies:
                raise ValueError("Conflicting/escaping explicit payload publication")
            copies[key] = dict(path=str(Path(normalized["path"]).parent.joinpath(*relative.split("/"))),
                               **_identity(copy))
    return copies


def _collect(index, product_ids, target_root, *, current_machine_id,
             source_paths, payload_publications, file_writers_closed,
             document_format):
    _closed(file_writers_closed)
    name(current_machine_id)
    index_root, document, rows = _read_index(index)
    if not isinstance(product_ids, (list, tuple)):
        raise ValueError("Collect requires an explicit product-id list")
    keys = list(product_ids)
    if len(keys) != len(set(keys)) or not set(keys) <= rows.keys():
        raise ValueError("Collect only explicitly selected, unique indexed products")
    source_paths = dict(source_paths or {})
    if not source_paths.keys() <= set(keys):
        raise ValueError("Accessible source aliases must name selected indexed products")
    bundles = _bundle_copies(index, payload_publications)
    sources = {}
    for key in keys:
        row = rows[key]
        if key in bundles:
            copy = bundles[key]
            if _identity(copy) != _identity(row):
                raise ValueError("Published payload differs from its indexed original")
            sources[key] = _file(copy["path"])
        elif "metadata_copy" in row:
            sources[key] = _file(index_root.joinpath(*row["metadata_copy"]["relative_path"].split("/")))
        else:
            sources[key] = _source(row, current_machine_id, source_paths)
    root = _fresh_root(target_root)
    copies = []
    for key in keys:
        row = rows[key]
        relative = "products/" + key
        _copy(sources[key], _identity(row), _target(root, relative))
        copies.append(dict(product_id=key, relative_path=relative, machine_id=row["machine_id"],
                           local_path=row["local_path"], **_identity(row)))
    return _write_index(root, "publication.json" if document_format == BUNDLE_FORMAT else "collection.json", dict(
        format=document_format, products_index=_bound(index),
        run=document["run"], original_attempt_id=document["attempt_id"],
        copying_machine_id=current_machine_id, file_writers_closed=True, copies=copies,
        verification="Every explicitly copied product SHA-256 and byte count verified",
    ))


def publish_indexed_products(index, target_root, *, current_machine_id,
                             publish=DEFAULT_PUBLISH, source_paths=None,
                             file_writers_closed):
    """Publish selected job ``publish`` kinds to a fresh shared-results bundle.

    Default kinds are film and lossless_export. An explicit empty list copies
    none. Maps are opened only if the job explicitly includes their kind.
    This creates a separate immutable publication.json; it never edits index.
    """
    _root, _document, rows = _read_index(index)
    if not isinstance(publish, (list, tuple)):
        raise ValueError("The job publish list must be an explicit list or tuple")
    kinds = tuple(publish)
    if any(not isinstance(item, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", item) for item in kinds):
        raise ValueError("The job publish list must contain declared product kinds")
    keys = [key for key, row in rows.items() if row["kind"] in kinds]
    return _collect(index, keys, target_root, current_machine_id=current_machine_id,
                    source_paths=source_paths, payload_publications=(),
                    file_writers_closed=file_writers_closed, document_format=BUNDLE_FORMAT)


def collect_products(index, product_ids, target_root, *, current_machine_id,
                     source_paths=None, payload_publications=(), file_writers_closed):
    """Hash-verified collection into a fresh target, without any remote command.

    Source order: explicitly bound publication bundles, shared metadata copies,
    then an explicit accessible source alias or this machine's producer path.
    A foreign machine's local path is NEVER probed as an automatic fallback.
    No SSH, staging, mounts, globbing or queue/claim mutation occurs here.
    """
    return _collect(index, product_ids, target_root, current_machine_id=current_machine_id,
                    source_paths=source_paths, payload_publications=payload_publications,
                    file_writers_closed=file_writers_closed, document_format=COLLECTION_FORMAT)


def publish_late_receipts(index, receipts, target_root, *, current_machine_id,
                          file_writers_closed, source_paths=None):
    """Publish explicitly listed, now-closed parent receipts after completion.

    Receipts are not appended to or used to rewrite products_index. Their
    original bytes supply any actual OS-retirement/control claims. The helper
    neither invents such claims nor treats completed.json as OS retirement.
    """
    _closed(file_writers_closed)
    _read_index(index)
    name(current_machine_id)
    rows = _products(receipts)
    if any(row["kind"] != "operational_receipt" or row["bytes"] > MAX_METADATA_BYTES for row in rows):
        raise ValueError("Only explicit small operational receipts belong in this bundle")
    paths = dict(source_paths or {})
    if not paths.keys() <= {row["product_id"] for row in rows}:
        raise ValueError("Receipt aliases must name explicit receipts")
    sources = {row["product_id"]: _source(row, current_machine_id, paths) for row in rows}
    root = _fresh_root(target_root)
    copies = []
    for row in rows:
        relative = "receipts/" + row["product_id"]
        _copy(sources[row["product_id"]], _identity(row), _target(root, relative))
        copies.append(dict(row, relative_path=relative))
    return _write_index(root, "receipts_index.json", dict(
        format="darsia-late-receipts-publication-v1", products_index=_bound(index),
        publishing_machine_id=current_machine_id, file_writers_closed=True, receipts=copies,
        retirement_claim_scope="Only the unchanged receipt bytes; not inferred by publication",
    ))
