"""Opt-in reuse of exact source hashes within one bounded Setup operation."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import RLock

from .bound_verification_cache import BoundVerificationCache


_CACHE = ContextVar("setup-source-reads", default=None)
_PROOFS = ContextVar("setup-local-preparation-proofs", default=None)
_REGISTERED = ContextVar("setup-registered-bindings", default=None)
_PAINTED_STORE = ContextVar("setup-painted-store-proofs", default=None)
_STAMPS = ContextVar("setup-source-stamps", default=None)
_MAXIMUM_PROOFS = 4
_MAXIMUM_PROOF_METADATA_BYTES = 16 * 1024**2
_MAXIMUM_REGISTERED_BINDINGS = 64
_MAXIMUM_REGISTERED_METADATA_BYTES = 16 * 1024**2
_MAXIMUM_PAINTED_STORE_PROOFS = 64
_MAXIMUM_PAINTED_STORE_METADATA_BYTES = 16 * 1024**2


@contextmanager
def reuse_setup_source_stamps(*, maximum_files=8192, maximum_metadata_bytes=8 * 1024**2):
    """Share stat tuples in one pure-read group and validate all of them at exit.

    Callers must finish this group before using its result for any write. Every
    retained locator is re-statted through its original reader on success. A
    full snapshot is validated and then continues with original raw stats; an
    exception from the body restores the prior scope immediately.
    """
    if (type(maximum_files) is not int or maximum_files < 1
            or type(maximum_metadata_bytes) is not int or maximum_metadata_bytes < 1):
        raise ValueError("Setup stamp reuse requires positive metadata limits.")
    snapshot = dict(files={}, maximum_files=maximum_files,
                    maximum_metadata_bytes=maximum_metadata_bytes, cost=0,
                    disabled=False, lock=RLock())
    token = _STAMPS.set(snapshot)
    try:
        yield
        with snapshot["lock"]:
            _validate_stamp_snapshot(snapshot)
    finally:
        _STAMPS.reset(token)
        with snapshot["lock"]:
            snapshot["files"].clear()


def _validate_stamp_snapshot(snapshot):
    """Re-stat every retained locator before returning or falling back."""
    failure = None
    for path, reader, expected in snapshot["files"].values():
        try:
            actual = reader(path)
        except OSError as error:
            if failure is None:
                failure = (path, error)
        else:
            if actual != expected and failure is None:
                failure = (path, None)
    if failure is not None:
        path, cause = failure
        raise ValueError("A Setup source changed during read-only verification: " + path) from cause


def _disable_stamp_reuse(snapshot):
    """Keep large valid groups on the original raw-stat path without a new gate."""
    _validate_stamp_snapshot(snapshot)
    snapshot["files"].clear()
    snapshot["cost"] = 0
    snapshot["disabled"] = True


def _source_stamp(path, *, reader):
    """Use the unchanged raw reader unless an explicit pure-read group is active."""
    snapshot = _STAMPS.get()
    if snapshot is None:
        return reader(path)
    locator = os.path.abspath(path)
    key = os.path.normcase(locator)
    with snapshot["lock"]:
        if snapshot["disabled"]:
            return reader(path)
        cached = snapshot["files"].get(key)
        if cached is not None:
            return cached[2]
        if len(snapshot["files"]) >= snapshot["maximum_files"]:
            _disable_stamp_reuse(snapshot)
            return reader(path)
        actual = reader(path)
        row = (locator, reader, actual)
        cost = _metadata_size(row) + sys.getsizeof(key)
        if snapshot["cost"] + cost > snapshot["maximum_metadata_bytes"]:
            _disable_stamp_reuse(snapshot)
            return actual
        snapshot["files"][key] = row
        snapshot["cost"] += cost
        return actual


@contextmanager
def setup_source_read_group():
    """Batch only a Setup pure reader, reusing its enclosing read transaction."""
    if _CACHE.get() is None or _STAMPS.get() is not None:
        yield
    else:
        with reuse_setup_source_stamps():
            yield


@contextmanager
def reuse_setup_source_reads(*, maximum_files=8192):
    """Hash each unchanged source once, with a new cache on every entry.

    A completion reverify must enter its own context so its first source read
    hashes the actual bytes again. Nested scopes restore the outer operation.
    Only file descriptors are retained; arrays and source bytes are not cached.
    """
    if type(maximum_files) is not int or maximum_files < 1:
        raise ValueError("Setup source reuse requires a positive file limit.")
    cache = BoundVerificationCache(
        maximum_files=maximum_files, maximum_metadata_bytes=0
    )
    token = _CACHE.set(cache)
    proofs = dict(entries=OrderedDict(), lock=RLock())
    proof_token = _PROOFS.set(proofs)
    registered = dict(entries=OrderedDict(), lock=RLock())
    registered_token = _REGISTERED.set(registered)
    painted_store = dict(entries=OrderedDict(), lock=RLock())
    painted_store_token = _PAINTED_STORE.set(painted_store)
    try:
        yield cache
    finally:
        _PAINTED_STORE.reset(painted_store_token)
        _REGISTERED.reset(registered_token)
        _PROOFS.reset(proof_token)
        _CACHE.reset(token)
        with proofs["lock"]:
            proofs["entries"].clear()
        with registered["lock"]:
            registered["entries"].clear()
        with painted_store["lock"]:
            painted_store["entries"].clear()
        cache.clear()


def reuse_painted_store_proof(binding, *, reader):
    """Decode one committed run's exact raster proof once per Setup operation.

    The original verifier still runs first. Subsequent consumers must retain
    its identical receipt and every consumed file's complete version/SHA.
    Only small proof metadata is retained; no raster or numerical result is
    cached. Outside this operation, or at a fresh boundary, readers are unchanged.
    """
    cache, retained = _CACHE.get(), _PAINTED_STORE.get()
    if cache is None or retained is None:
        return reader(binding)
    from .painted_adoption_table import _FILES
    try:
        identity = json.dumps(binding, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return reader(binding)
    key = hashlib.sha256(identity.encode("utf8")).hexdigest()
    with retained["lock"]:
        entries = retained["entries"]
        fresh = cache.requires_fresh() or _FILES.requires_fresh()
        cached = None if fresh else entries.get(key)
        if cached is not None:
            try:
                with setup_source_read_group():
                    unchanged = cache.unchanged(cached["dependencies"])
            except BaseException:
                entries.pop(key, None)
                raise
            if unchanged:
                entries.move_to_end(key)
                return
        entries.pop(key, None)
        # Both descriptor owners participate: a warm global descriptor need
        # not call the operation reader again, and the inverse is also true.
        from contextlib import ExitStack
        with ExitStack() as scopes:
            if fresh:
                scopes.enter_context(cache.fresh())
                scopes.enter_context(_FILES.fresh())
            scopes.enter_context(setup_source_read_group())
            source_dependencies = scopes.enter_context(cache.capture())
            file_dependencies = scopes.enter_context(_FILES.capture())
            result = reader(binding)
        dependencies = dict(stable=source_dependencies["stable"] and file_dependencies["stable"], files={})
        for capture in (source_dependencies, file_dependencies):
            for path, row in capture["files"].items():
                previous = dependencies["files"].get(path)
                if previous is not None and previous != row:
                    dependencies["stable"] = False
                dependencies["files"][path] = row
        # This verifier returns only success/failure. Unexpected return values
        # or untracked inputs keep their original uncached behavior.
        if fresh or result is not None or not dependencies["stable"] or not dependencies["files"]:
            return result
        entry = dict(dependencies=dependencies, cost=0)
        entry["cost"] = _metadata_size(entry) + sys.getsizeof(key)
        if entry["cost"] <= _MAXIMUM_PAINTED_STORE_METADATA_BYTES:
            with setup_source_read_group():
                if cache.unchanged(dependencies):
                    entries[key] = entry
                    while (len(entries) > _MAXIMUM_PAINTED_STORE_PROOFS
                           or sum(row["cost"] for row in entries.values()) > _MAXIMUM_PAINTED_STORE_METADATA_BYTES):
                        entries.popitem(last=False)
        return result


def source_descriptor(path, *, reader):
    """Run the original hash reader unless this operation has checked the file."""
    cache = _CACHE.get()
    if cache is None:
        from .adoption_check_progress import report_check_progress
        report_check_progress("Verifying saved source: " + path.name)
        return reader(path)
    actual = cache.descriptor(path, reader=reader)
    # Each reader keeps its original locator spelling, even when another
    # reader seeded the shared descriptor using an equivalent spelling.
    actual["path"] = str(path)
    return actual


def _metadata_size(value, seen=None):
    """Measure retained finite metadata without counting shared strings twice."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    cost = sys.getsizeof(value)
    if isinstance(value, dict):
        return cost + sum(_metadata_size(key, seen) + _metadata_size(item, seen)
                          for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return cost + sum(_metadata_size(item, seen) for item in value)
    return cost


def reuse_registered_binding(project_root, current, config, run, *, reader):
    """Reuse a proved Store binding while its original consumed files still verify."""
    cache, retained = _CACHE.get(), _REGISTERED.get()
    if (cache is None or retained is None or "manual_label_edit" in config
            or config.get("source_mode") != "canonical-from-import"):
        return reader(project_root, current, config, run)
    identity = json.dumps(dict(project_root=str(Path(project_root).resolve()),
        current=current, configuration=config, run=run), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    from .painted_adoption_table import _FILES
    from .geometry_raster_sources import verify_registered_sources
    with retained["lock"]:
        entries = retained["entries"]
        fresh = cache.requires_fresh() or _FILES.requires_fresh()
        cached = None if fresh else entries.get(key)
        if cached is not None:
            binding = deepcopy(cached["binding"])
            try:
                with setup_source_read_group():
                    verify_registered_sources(binding)
            except BaseException:
                entries.pop(key, None)
                raise
            entries.move_to_end(key)
            return binding
        entries.pop(key, None)
        with cache.fresh() if fresh else nullcontext():
            binding = reader(project_root, current, config, run)
        if binding is None or fresh:
            return binding
        # Unsupported or oversized metadata stays on the original reader path.
        try:
            json.dumps(binding, allow_nan=False)
        except (TypeError, ValueError):
            return binding
        cost = _metadata_size(binding) + sys.getsizeof(key) + sys.getsizeof(0)
        if cost <= _MAXIMUM_REGISTERED_METADATA_BYTES:
            entries[key] = dict(binding=deepcopy(binding), cost=cost)
            while (len(entries) > _MAXIMUM_REGISTERED_BINDINGS
                   or sum(row["cost"] for row in entries.values()) > _MAXIMUM_REGISTERED_METADATA_BYTES):
                entries.popitem(last=False)
        return binding


def reuse_local_preparation_proof(preparation, folder, *, reader):
    """Reuse one complete local proof only within an explicit Setup operation."""
    proofs = _PROOFS.get()
    if proofs is None:
        return reader(preparation, folder)
    from .painted_adoption_table import _FILES, _checked
    identity = json.dumps(dict(preparation=preparation, folder=str(Path(folder).resolve())),
                          sort_keys=True, separators=(",", ":"), allow_nan=False)
    key = hashlib.sha256(identity.encode("utf8")).hexdigest()
    with proofs["lock"]:
        entries = proofs["entries"]
        fresh = _FILES.requires_fresh()
        cached = None if fresh else entries.get(key)
        if cached is not None and _FILES.unchanged(cached["dependencies"]):
            entries.move_to_end(key)
            return deepcopy(cached["guarded"])
        entries.pop(key, None)
        # A forced original proof must also bypass the operation's descriptor
        # reuse, including when its filesystem reports an unchanged stamp.
        source_reads = _CACHE.get()
        with source_reads.fresh() if fresh else nullcontext():
            with _FILES.capture() as dependencies:
                guarded = reader(preparation, folder)
                for bound in guarded:
                    _checked(bound)
        entry = dict(guarded=deepcopy(guarded), dependencies=deepcopy(dependencies), cost=0)
        # Reject arrays or other non-finite/non-JSON objects before retention.
        json.dumps(entry, allow_nan=False)
        entry["cost"] = _metadata_size(entry) + sys.getsizeof(key) + sys.getsizeof(0)
        if (not fresh and entry["cost"] <= _MAXIMUM_PROOF_METADATA_BYTES
                and _FILES.unchanged(dependencies)):
            entries[key] = entry
            while (len(entries) > _MAXIMUM_PROOFS
                   or sum(row["cost"] for row in entries.values()) > _MAXIMUM_PROOF_METADATA_BYTES):
                entries.popitem(last=False)
        return guarded
