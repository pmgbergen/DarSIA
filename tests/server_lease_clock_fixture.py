"""Explicit simulated server mtime for ordinary queue lease tests only."""
from pathlib import Path
import os

import pytest

from darsia.fluidflower.pipeline import server_clock as clocks


@pytest.fixture
def server_lease_clock(tmp_path, monkeypatch):
    """Use the real stale comparator with controlled sample/progress file times.

    No real server/transport qualification is claimed. Only this private clock's
    sample mtime is supplied; actual queue progress files retain real stat reads
    after the fixture explicitly stamps the closed files at event time.
    """
    server = clocks.ServerFileClock(tmp_path, "synthetic-test-server")
    current_ns = [1_000_000_000_000]
    original_mtime = clocks.file_mtime_ns
    monkeypatch.setattr(clocks, "file_mtime_ns", lambda path:
        current_ns[0] if Path(path) == server.path else original_mtime(path))

    class ControlledServer:
        clock = server

        def advance(self, seconds):
            current_ns[0] += int(seconds * 1_000_000_000)

        def stamp_progress(self, claim):
            paths = [claim.lease / "owner.json", claim.lease / "claimed.json",
                     *claim.lease.glob("progress_*.json"),
                     *claim.output.glob("progress_*.json")]
            for path in paths:
                if path.is_file():
                    os.utime(path, ns=(current_ns[0], current_ns[0]))

    return ControlledServer()
