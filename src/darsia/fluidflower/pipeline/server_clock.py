"""Lease time from the admitted share's file mtimes, without client wall time.

The worker startup must admit the shared filesystem before constructing this
operational helper. This does not establish transport, ownership or progress.
Each clock has one private, unique append-only file and only its owner reads it.
Clock samples never serve as a task's actual-progress evidence.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
import threading
import uuid


class ServerTimeError(RuntimeError):
    """The admitted filesystem did not provide usable lease-time evidence."""


def file_mtime_ns(path):
    """Return filesystem metadata only; embedded client timestamps are ignored."""
    from darsia.fluidflower.pipeline.job_queue import retry_transient

    info = retry_transient(Path(path).stat)
    if not stat.S_ISREG(info.st_mode):
        raise ServerTimeError("Lease-time evidence must be a regular file: " + str(path))
    value = info.st_mtime_ns
    if type(value) is not int or value <= 0:
        raise ServerTimeError("The shared filesystem returned an invalid file mtime")
    return value


class ServerFileClock:
    """One writer and reader per incarnation; no shared mutable clock document.

    A byte append updates this owner's small clock file on the admitted SMB/NFS
    server. Flush and close precede stat, so the sample comes from that file's
    mtime. The file is kept after retirement. No other worker needs its path.
    """

    def __init__(self, job_root, machine_id):
        from darsia.fluidflower.pipeline.job_queue import name

        self.machine_id = name(machine_id)
        from darsia.fluidflower.pipeline.job_queue import retry_transient

        directory = Path(job_root) / "clock_samples" / self.machine_id
        self._lock = threading.Lock()
        self._last_ns = None

        def create():
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / (uuid.uuid4().hex + ".clock")  # New on each try.
            with self.path.open("xb") as stream:
                stream.write(b"DarSIA private server-time samples\n")
                stream.flush()
                os.fsync(stream.fileno())

        retry_transient(create)  # N111: a transient SMB error waits and retries.

    def now_ns(self):
        """Sample server mtime; reject backward movement, never use time.time."""
        from darsia.fluidflower.pipeline.job_queue import retry_transient

        def sample():
            with self.path.open("ab") as stream:
                stream.write(b".")
                stream.flush()
                os.fsync(stream.fileno())
            return file_mtime_ns(self.path)

        with self._lock:
            value = retry_transient(sample)  # N111; an extra sample byte is harmless.
            if self._last_ns is not None and value < self._last_ns:
                raise ServerTimeError("The shared filesystem's clock moved backwards")
            self._last_ns = value
            return value

    def stale(self, actual_progress_path, timeout_seconds=600):
        """Compare only server file times; no synthetic heartbeat extends a lease.

        The caller supplies its actual claim marker or last shared immutable
        I/O/frame progress event. It must not pass this clock's own sample file.
        """
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("A positive integer lease duration is required")
        progress = Path(actual_progress_path)
        if progress == self.path:
            raise ServerTimeError("A clock sample is not actual task progress")
        last_ns = file_mtime_ns(progress)
        current_ns = self.now_ns()
        if current_ns < last_ns:
            raise ServerTimeError("Actual-progress mtime is ahead of the server clock sample")
        return current_ns - last_ns >= timeout_seconds * 1_000_000_000
