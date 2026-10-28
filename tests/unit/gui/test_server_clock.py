"""Operational lease-clock tests with explicit simulated server timestamps.

Temporary local files exercise append/ownership behavior only. They are never
claimed to prove SMB/NFS admission or real server-clock behavior.
"""
from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from darsia.fluidflower.pipeline import server_clock as clocks


def test_client_wall_clock_skew_cannot_expire_a_lease(tmp_path, monkeypatch):
    clock = clocks.ServerFileClock(tmp_path, "main")
    progress = tmp_path / "actual_progress.json"
    progress.write_bytes(b'{"written_unix_ns": 9999999999999999999}')
    server_now = [600_000_000_000]
    monkeypatch.setattr(clocks, "file_mtime_ns", lambda path: (
        1_000_000_000 if path == progress else server_now[0]))
    for client_time in (-1e10, 0, 1e10):
        monkeypatch.setattr(time, "time", lambda: client_time)
        assert not clock.stale(progress)
    server_now[0] = 601_000_000_000
    assert clock.stale(progress)


def test_samples_do_not_extend_actual_task_progress(tmp_path, monkeypatch):
    clock = clocks.ServerFileClock(tmp_path, "main")
    progress = tmp_path / "frame_committed.json"
    progress.write_bytes(b"actual committed frame")
    monkeypatch.setattr(clocks, "file_mtime_ns", lambda path: (
        1_000_000_000 if path == progress else 601_000_000_000))
    before = progress.read_bytes()
    for _ in range(10):
        clock.now_ns()
    assert clock.stale(progress)
    assert progress.read_bytes() == before
    with pytest.raises(clocks.ServerTimeError, match="not actual task progress"):
        clock.stale(clock.path)


def test_private_clock_files_are_distinct_and_append_only(tmp_path):
    first = clocks.ServerFileClock(tmp_path, "main")
    second = clocks.ServerFileClock(tmp_path, "main")
    assert first.path != second.path
    initial = first.path.read_bytes()
    second_before = second.path.read_bytes()
    first.now_ns()
    assert first.path.read_bytes() == initial + b"."
    assert second.path.read_bytes() == second_before


def test_one_clock_serializes_its_own_writer_threads(tmp_path, monkeypatch):
    clock = clocks.ServerFileClock(tmp_path, "main")
    initial = clock.path.read_bytes()
    monkeypatch.setattr(clocks, "file_mtime_ns", lambda path: 10_000_000_000)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: clock.now_ns(), range(16))) == [10_000_000_000] * 16
    assert clock.path.read_bytes() == initial + b"." * 16


def test_backward_server_clock_refuses_reclaim(tmp_path, monkeypatch):
    clock = clocks.ServerFileClock(tmp_path, "main")
    values = iter((20_000_000_000, 19_000_000_000))
    monkeypatch.setattr(clocks, "file_mtime_ns", lambda path: next(values))
    assert clock.now_ns() == 20_000_000_000
    with pytest.raises(clocks.ServerTimeError, match="moved backwards"):
        clock.now_ns()


def test_future_progress_timestamp_refuses_reclaim(tmp_path, monkeypatch):
    clock = clocks.ServerFileClock(tmp_path, "main")
    progress = tmp_path / "progress.json"
    progress.write_bytes(b"actual progress")
    monkeypatch.setattr(clocks, "file_mtime_ns", lambda path: (
        2_000_000_000 if path == progress else 1_000_000_000))
    with pytest.raises(clocks.ServerTimeError, match="ahead"):
        clock.stale(progress)


def test_non_file_progress_is_refused(tmp_path):
    with pytest.raises(clocks.ServerTimeError, match="regular file"):
        clocks.file_mtime_ns(tmp_path)
