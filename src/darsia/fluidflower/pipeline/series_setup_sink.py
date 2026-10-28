"""Explicit Setup-only metadata writes under the admitted whole-operation cap.

Original export, Apply and publication writers keep their file selection and
serializers. This context only wraps their actual output streams and rename
boundaries. All created bytes, including backups and retained partials, consume
the same allowance; replacing or moving a file never releases that allowance.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import io
import os
from pathlib import Path

_active = ContextVar("setup_metadata_output", default=None)
_claim_sink = ContextVar("series_setup_claim_controls", default=None)


def current_sink():
    """Return only this explicitly active operation's bounded output sink."""
    return _active.get()


def current_writer():
    """Select the finite claim writer before the actual bound operation sink."""
    return _claim_sink.get() or current_sink()


class SetupMetadataSink:
    """Bound metadata/scratch writes to actual active volume reservations."""

    def __init__(self, admission):
        from .series_setup_admission import SetupAdmission
        if not isinstance(admission, SetupAdmission):
            raise TypeError("Setup metadata requires its actual typed admission")
        self.admission = admission
        limits = admission.metadata_limits
        self.maximum = limits["maximum_output_bytes"]
        self.maximum_file = limits["maximum_file_bytes"]
        self.written_bytes = 0
        self.files = {}
        self.open_writers = 0
        self.closed = False

    @contextmanager
    def activate(self):
        """Use one unchanged allowance across the operation's original writers."""
        self._require_open()
        previous = current_sink()
        if previous is not None and previous is not self:
            raise ValueError("A different Setup metadata operation is already active")
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    def _require_open(self):
        if self.closed:
            raise ValueError("The Setup metadata writer is closed")

    def close(self):
        """Freeze the writer only after every original stream has closed."""
        evidence = self.evidence()
        self.closed = True
        return evidence

    def _allow(self, path, growth, size):
        self._require_open()
        self.admission.require_write(Path(path))
        if (size > self.maximum_file
                or self.written_bytes + growth > self.maximum):
            raise ValueError("Setup metadata byte ceiling exceeded before write; partials retained")

    @contextmanager
    def open_exclusive(self, path):
        """Supply the original serializer an exclusive capped binary stream."""
        path = Path(path)
        self._allow(path, 0, 0)
        stream = path.open("xb")
        writer = _MetadataStream(stream, path, self)
        self.open_writers += 1
        self.files[str(path)] = 0
        try:
            yield writer
            writer.flush()
            os.fsync(stream.fileno())
        finally:
            stream.close()
            self.open_writers -= 1

    def write_bytes(self, path, raw):
        """Refuse an oversized known buffer before creating its output file."""
        raw = memoryview(raw)
        self._allow(path, raw.nbytes, raw.nbytes)
        with self.open_exclusive(path) as stream:
            stream.write(raw)

    def copy_file(self, source, destination):
        """Retain the original byte copy, with every output write capped."""
        source, destination = Path(source), Path(destination)
        size = source.stat().st_size
        self._allow(destination, size, size)
        with source.open("rb") as incoming, self.open_exclusive(destination) as outgoing:
            while block := incoming.read(1024 * 1024):
                outgoing.write(block)

    def mkdir(self, path, *, parents=False, exist_ok=False):
        """Check the actual output domain before the original directory create."""
        path = Path(path)
        self._require_open()
        self.admission.require_write(path)
        path.mkdir(parents=parents, exist_ok=exist_ok)

    def rename(self, source, destination, *, replace=False, rename_operation=None):
        """Keep retained byte accounting through the original atomic boundary."""
        source, destination = Path(source), Path(destination)
        self._require_open()
        self.admission.require_write(source)
        self.admission.require_write(destination)
        if rename_operation is not None:
            if replace:
                raise ValueError("An immutable queue rename cannot replace a peer")
            rename_operation(source, destination)
        else:
            (os.replace if replace else os.rename)(source, destination)
        moved = {}
        for key, size in list(self.files.items()):
            path = Path(key)
            if path == source or path.is_relative_to(source):
                target = destination / path.relative_to(source)
                moved[str(target)] = size
                del self.files[key]
        self.files.update(moved)

    def evidence(self):
        """Report actual retained writes, without claiming task retirement."""
        if self.open_writers:
            raise ValueError("Setup metadata writer streams have not closed")
        return dict(maximum_output_bytes=self.maximum,
                    maximum_file_bytes=self.maximum_file,
                    actual_created_bytes=self.written_bytes,
                    files=[dict(path=path, size_bytes=size)
                           for path, size in sorted(self.files.items())],
                    frame_progress_releases_bytes=False,
                    task_retirement_claimed=False)


class _MetadataStream:
    """Preserve encoder seeks while checking each possible file growth first."""

    def __init__(self, stream, path, owner):
        self._stream, self.path, self._owner = stream, path, owner
        observed = os.fstat(stream.fileno())
        self._identity = observed.st_dev, observed.st_ino
        self._size = 0

    def write(self, raw):
        data = memoryview(raw)
        current = self.path.stat()
        position = self._stream.tell()
        if ((current.st_dev, current.st_ino) != self._identity
                or current.st_size != self._size):
            raise ValueError("The owned Setup output changed outside its writer")
        size = max(self._size, position + data.nbytes)
        growth = size - self._size
        self._owner._allow(self.path, growth, size)
        before = self._size
        try:
            count = self._stream.write(data)
            self._stream.flush()
        finally:
            observed = os.fstat(self._stream.fileno()).st_size
            self._owner.written_bytes += max(0, observed - before)
            self._size = observed
            self._owner.files[str(self.path)] = observed
        if count != data.nbytes:
            raise OSError("The bounded Setup output write was incomplete; partial retained")
        return count

    def fileno(self):
        """Require serializers to use checked writes, never native-fd bypass."""
        raise io.UnsupportedOperation("Setup metadata requires bounded writes")

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        if name in {"tell", "seek", "seekable", "readable", "writable", "closed", "name", "mode"}:
            return getattr(self._stream, name)
        raise AttributeError(name)
