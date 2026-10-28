"""Bounded retries of file steps, using the queue's shared N111 decision.

These helpers never call a scientific stage. A retry serializes the same
already computed values or reads the same content-bound input. Interrupted
temporary files are retained; an existing peer's file is never replaced.
"""

from functools import wraps
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
from secrets import token_hex

from .job_queue import rename, retry_transient

_RETRYING = ContextVar("darsia_file_step_retrying", default=False)
_METADATA_CEILING = ContextVar("darsia_metadata_ceiling", default=None)
# The positive lean resource contract binds this exact publication guard.
# Full/default publication never enters the context and keeps its old ABI.
LEAN_METADATA_BYTES = 16 * 1024 * 1024


@contextmanager
def metadata_ceiling(maximum_bytes):
    """Limit metadata publication in this context, without changing its bytes.

    Only explicitly scoped lean output uses this guard. Nested contexts cannot
    weaken an outer ceiling. A refusal retains the existing owned partial and
    never retries a scientific stage or removes a file.
    """
    if maximum_bytes is not None and (type(maximum_bytes) is not int or maximum_bytes < 1):
        raise ValueError("Metadata ceiling must be an explicit positive byte count")
    previous = _METADATA_CEILING.get()
    limit = previous if maximum_bytes is None else maximum_bytes
    if previous is not None and limit is not None:
        limit = min(previous, limit)
    token = _METADATA_CEILING.set(limit)
    try:
        yield
    finally:
        _METADATA_CEILING.reset(token)


def metadata_limit(path):
    """The active ceiling for an actual JSON/statistics metadata filename."""
    limit = _METADATA_CEILING.get()
    if limit is None:
        return None
    filename = Path(path).name.casefold()
    return limit if filename.endswith((".json", ".json.gz")) or filename == "mass_table.csv" else None


def check_metadata_bytes(path, data):
    """Refuse a known oversized record before any directory/file write."""
    limit = metadata_limit(path)
    if limit is not None and len(data) > limit:
        raise ValueError("Lean metadata exceeds its publication byte ceiling; existing files retained")


class _CeilingStream:
    """The existing binary stream with an upper bound checked before writes."""
    def __init__(self, stream, maximum_bytes):
        self._stream, self._maximum_bytes = stream, maximum_bytes

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def write(self, data):
        if self._stream.tell() + len(data) > self._maximum_bytes:
            raise ValueError("Lean metadata exceeds its publication byte ceiling; partial retained")
        return self._stream.write(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self._stream.tell() + offset
        elif whence == 2:
            current = self._stream.tell()
            end = self._stream.seek(0, 2)
            self._stream.seek(current)
            position = end + offset
        else:
            raise ValueError("Invalid metadata stream seek origin")
        if position < 0 or position > self._maximum_bytes:
            raise ValueError("Lean metadata seek exceeds its publication byte ceiling")
        return self._stream.seek(position)

    def truncate(self, size=None):
        size = self._stream.tell() if size is None else size
        if size < 0 or size > self._maximum_bytes:
            raise ValueError("Lean metadata truncate exceeds its publication byte ceiling")
        return self._stream.truncate(size)


def retry_io(function):
    """Retry one read/hash/copy operation, retaining its original checks."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _RETRYING.get():
            return function(*args, **kwargs)
        def action():
            token = _RETRYING.set(True)
            try:
                return function(*args, **kwargs)
            finally:
                _RETRYING.reset(token)
        return retry_transient(action)
    return wrapped


@retry_io
def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def publish_file(path, writer, *, reuse_identical=False, expected_sha256=None):
    """Serialize immutable values to a fresh temporary, then commit once.

    ``writer(stream)`` performs serialization only. Each retry owns another
    temporary; partial files remain available. A completed interrupted commit
    is accepted only after its actual bytes match the newly serialized bytes.
    """
    path = Path(path)
    from .series_setup_sink import current_writer
    setup_writer = current_writer()
    retry_transient(lambda: path.parent.mkdir(parents=True, exist_ok=True)
        if setup_writer is None else setup_writer.mkdir(path.parent, parents=True, exist_ok=True))
    try:
        retry_transient(path.stat)
    except FileNotFoundError as error:
        from .job_queue import transient_os_error
        if transient_os_error(error):
            raise
    else:
        if not reuse_identical:
            raise FileExistsError(path)

    @retry_io
    def serialize():
        from darsia_cache.publication_paths import partial_name
        temporary = path.with_name(partial_name(path.name, token_hex(16)))
        if setup_writer is None:
            with temporary.open("xb") as stream:
                limit = metadata_limit(path)
                writer(stream if limit is None else _CeilingStream(stream, limit))
                stream.flush()
                os.fsync(stream.fileno())
        else:
            register = getattr(setup_writer, "register_temporary", None)
            if register is not None:
                register(path, temporary)
            with setup_writer.open_exclusive(temporary) as stream:
                limit = metadata_limit(path)
                writer(stream if limit is None else _CeilingStream(stream, limit))
        return temporary

    temporary = serialize()
    digest = _digest(temporary)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Copied bytes differ from their frozen source; partial retained")
    try:
        rename(temporary, path)
    except FileExistsError:
        if not reuse_identical:
            raise
        if _digest(path) != digest:
            raise ValueError("Unreceipted output differs from its frozen bytes")
    return digest
