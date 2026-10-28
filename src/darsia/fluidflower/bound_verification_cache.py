"""Bounded process-local file verification with fresh publication checks.

Only explicit callers opt in. Cache entries bind a physical spelling, SHA and
complete observed file stamp. Raster arrays and parsed image objects are never
retained. A changed stamp causes the original reader to run again.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from threading import RLock

from .pipeline.job_queue import retry_transient


class BoundVerificationCache:
    """Reuse checked bytes only while their file identity/version stays equal."""

    def __init__(self, *, maximum_files=8192, maximum_metadata_bytes=16 * 1024**2):
        """Bound descriptor count and retained small-file bytes independently."""
        self.maximum_files = maximum_files
        self.maximum_metadata_bytes = maximum_metadata_bytes
        self._pid = os.getpid()
        self._descriptors, self._heads, self._metadata = OrderedDict(), {}, OrderedDict()
        self._metadata_bytes = 0
        self._lock = RLock()
        self._captures = ContextVar("bound-file-captures", default=())
        self._fresh = ContextVar("bound-file-fresh", default=False)

    def _process(self):
        """Discard inherited cache entries after a process fork."""
        if self._pid != os.getpid():
            self._descriptors.clear()
            self._heads.clear()
            self._metadata.clear()
            self._metadata_bytes = 0
            self._pid = os.getpid()

    @staticmethod
    def _key(path):
        """Keep the actual consumed spelling without inventing a source alias."""
        return os.path.normcase(os.path.abspath(path))

    @staticmethod
    def stamp(path):
        """Retry a stat and retain device, file, length and both time identities."""
        from .setup_source_reads import _source_stamp
        return _source_stamp(path, reader=BoundVerificationCache._raw_stamp)

    @staticmethod
    def _raw_stamp(path):
        """Read the complete original version tuple without Setup stamp reuse."""
        value = retry_transient(Path(path).stat)
        return tuple(getattr(value, key) for key in
                     ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))

    def _remember(self, path, stamp, digest):
        """Record every consumed locator for active derivation dependencies."""
        key = self._key(path)
        for captured in self._captures.get():
            prior = captured["files"].get(key)
            if prior is not None and (prior["stamp"] != stamp or prior["sha256"] != digest):
                captured["stable"] = False
            captured["files"][key] = dict(path=str(path), stamp=stamp, sha256=digest)

    def descriptor(self, path, *, expected=None, reader=None, force=False):
        """Reuse an original descriptor or rehash through its existing reader."""
        from .preset_wasserstein_inputs import current_scope
        scope = current_scope()
        if scope is not None:
            return scope.descriptor(path, expected)
        custom_reader = reader is not None
        if reader is None:
            from .geometry_raster_sources import descriptor as reader
        from .storage_paths import resolve_storage_path
        path = resolve_storage_path(path)
        with self._lock:
            self._process()
            before = self.stamp(path)
            key = self._key(path)
            head = self._heads.get(key)
            digest = (expected["sha256"] if expected is not None
                      else head[1] if head is not None and head[2] == before else None)
            cache_key = (key, digest, before)
            actual = self._descriptors.get(cache_key)
            if actual is None or force or self._fresh.get():
                if custom_reader:
                    from .adoption_check_progress import report_check_progress
                    report_check_progress("Verifying saved source: " + path.name)
                actual = retry_transient(lambda: reader(path))
                if before != self.stamp(path):
                    raise ValueError("A bound source changed during verification: " + str(path))
                cache_key = (key, actual["sha256"], before)
                if head is not None and head != cache_key:
                    self._descriptors.pop(head, None)
                    for previous in list(self._metadata):
                        if previous[0] == key:
                            self._metadata_bytes -= len(self._metadata.pop(previous))
                self._descriptors[cache_key] = deepcopy(actual)
                self._heads[key] = cache_key
                while len(self._descriptors) > self.maximum_files:
                    old, _ = self._descriptors.popitem(last=False)
                    if self._heads.get(old[0]) == old:
                        self._heads.pop(old[0])
            self._descriptors.move_to_end(cache_key)
            self._remember(path, before, actual["sha256"])
            # The existing caller still compares this actual descriptor with
            # expected. A conflicting expected SHA is never adopted as fact.
            return deepcopy(actual)

    def read(self, path, *, limit, force=False):
        """Retry one bounded read and cache only unchanged small-file bytes."""
        from .preset_wasserstein_inputs import current_scope
        scope = current_scope()
        if scope is not None:
            if scope.descriptor(path)["size_bytes"] > limit:
                raise ValueError("Bound metadata exceeds its explicit read limit")
            return scope.read(path)
        from .storage_paths import resolve_storage_path
        path = resolve_storage_path(path)
        with self._lock:
            self._process()
            before = self.stamp(path)
            key = (self._key(path), before)
            if before[2] > limit:
                raise ValueError("Bound metadata exceeds its explicit read limit")
            raw = self._metadata.get(key)
            if raw is None or force or self._fresh.get():
                def read_once():
                    """Restart a bounded read from byte zero after a transient failure."""
                    with path.open("rb") as stream:
                        return stream.read(limit + 1)
                raw = retry_transient(read_once)
                if len(raw) > limit:
                    raise ValueError("Bound metadata exceeds its explicit read limit")
                if before != self.stamp(path):
                    raise ValueError("Bound metadata changed during its read: " + str(path))
                if len(raw) <= self.maximum_metadata_bytes:
                    previous = self._metadata.pop(key, None)
                    if previous is not None:
                        self._metadata_bytes -= len(previous)
                    self._metadata[key] = raw
                    self._metadata_bytes += len(raw)
                    while (self._metadata_bytes > self.maximum_metadata_bytes
                           or len(self._metadata) > self.maximum_files):
                        _, old = self._metadata.popitem(last=False)
                        self._metadata_bytes -= len(old)
            digest = hashlib.sha256(raw).hexdigest()
            head = self._heads.get(key[0])
            if head is not None and (head[1] != digest or head[2] != before):
                # A real read can discover a new hash even when a filesystem
                # reports the same full stamp. Never retain the old success.
                self._descriptors.pop(head, None)
                self._heads.pop(key[0], None)
                for previous in list(self._metadata):
                    if previous[0] == key[0] and previous != key:
                        self._metadata_bytes -= len(self._metadata.pop(previous))
            if key in self._metadata:
                self._metadata.move_to_end(key)
            self._remember(path, before, digest)
            return raw

    @contextmanager
    def capture(self):
        """Collect finite consumed-file stamps without retaining array values."""
        from .preset_wasserstein_inputs import current_scope
        if current_scope() is not None:
            # Worker reads must not seed a reusable local metadata success.
            yield dict(stable=False, files={})
            return
        captured = dict(stable=True, files={})
        token = self._captures.set((*self._captures.get(), captured))
        try:
            yield captured
        finally:
            self._captures.reset(token)

    def unchanged(self, captured):
        """Check all dependencies before reusing an all-run metadata result."""
        from .preset_wasserstein_inputs import current_scope
        if current_scope() is not None:
            return False
        if not captured["stable"]:
            return False
        for row in captured["files"].values():
            if self.stamp(row["path"]) != row["stamp"]:
                return False
            actual = self.descriptor(row["path"], expected={"sha256": row["sha256"]})
            if actual["sha256"] != row["sha256"]:
                return False
        return True

    @contextmanager
    def fresh(self):
        """Force original source/output rehash and reread at NEW publication."""
        token = self._fresh.set(True)
        try:
            yield
        finally:
            self._fresh.reset(token)

    def requires_fresh(self):
        """Let a metadata memo honor the caller's forced publication scope."""
        from .preset_wasserstein_inputs import current_scope
        return self._fresh.get() or current_scope() is not None

    def clear(self):
        """Drop process-local entries without touching any original file."""
        with self._lock:
            self._descriptors.clear()
            self._heads.clear()
            self._metadata.clear()
            self._metadata_bytes = 0
