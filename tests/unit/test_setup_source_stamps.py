"""Bounded stat reuse validates complete versions at each pure-read boundary."""
import hashlib
import os
from pathlib import Path

import pytest

from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import setup_source_reads as setup
from darsia.fluidflower.bound_verification_cache import BoundVerificationCache


def count_actual_stats(monkeypatch, paths):
    counts = {os.path.normcase(os.path.abspath(path)): 0 for path in paths}
    original = Path.stat

    def stat(path, *args, **kwargs):
        key = os.path.normcase(os.path.abspath(path))
        if key in counts:
            counts[key] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    return counts


def count_for(counts, path):
    return counts[os.path.normcase(os.path.abspath(path))]


def test_setup_read_group_is_opt_in_and_nested_readers_share_one_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"original")
    counts = count_actual_stats(monkeypatch, [path])
    with setup.setup_source_read_group():
        BoundVerificationCache.stamp(path)
        BoundVerificationCache.stamp(path)
        assert count_for(counts, path) == 2
    with setup.reuse_setup_source_reads():
        with setup.setup_source_read_group():
            BoundVerificationCache.stamp(path)
            with setup.setup_source_read_group():
                BoundVerificationCache.stamp(path)
            assert count_for(counts, path) == 3
        assert count_for(counts, path) == 4


def test_repeated_requests_make_one_real_stat_then_one_exit_stat(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("first", "second")]
    for path in paths:
        path.write_bytes(b"original")
    expected = [BoundVerificationCache.stamp(path) for path in paths]
    counts = count_actual_stats(monkeypatch, paths)
    with setup.reuse_setup_source_stamps():
        for _ in range(5):
            for path, stamp in zip(paths, expected):
                assert BoundVerificationCache.stamp(path) == stamp
        assert list(counts.values()) == [1, 1]
        snapshot = setup._STAMPS.get()
    assert list(counts.values()) == [2, 2]
    assert not snapshot["files"] and setup._STAMPS.get() is None
    for _ in range(2):
        BoundVerificationCache.stamp(paths[0])
    assert count_for(counts, paths[0]) == 4


@pytest.mark.parametrize("change", ["write", "replacement", "removal"])
def test_exit_rejects_changed_versions_and_still_restats_every_consumed_locator(
        tmp_path, monkeypatch, change):
    path, other = tmp_path / "source", tmp_path / "other"
    path.write_bytes(b"original")
    other.write_bytes(b"unchanged")
    before = path.stat()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"modified")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    counts = count_actual_stats(monkeypatch, [path, other])
    with pytest.raises(ValueError, match="changed during read-only verification") as error:
        with setup.reuse_setup_source_stamps():
            first = BoundVerificationCache.stamp(path)
            BoundVerificationCache.stamp(other)
            if change == "write":
                path.write_bytes(b"changed source")
            elif change == "replacement":
                os.replace(replacement, path)
            else:
                path.unlink()
            assert BoundVerificationCache.stamp(path) == first
            assert list(counts.values()) == [1, 1]
    assert list(counts.values()) == [2, 2]
    if change == "removal":
        assert isinstance(error.value.__cause__, FileNotFoundError)
    assert setup._STAMPS.get() is None


@pytest.mark.parametrize("inner_failure", [False, True])
def test_nested_groups_are_isolated_restore_outer_and_skip_failed_body_exit_checks(
        tmp_path, monkeypatch, inner_failure):
    path = tmp_path / "source"
    path.write_bytes(b"original")
    counts = count_actual_stats(monkeypatch, [path])
    with setup.reuse_setup_source_stamps():
        first = BoundVerificationCache.stamp(path)
        outer = setup._STAMPS.get()
        try:
            with setup.reuse_setup_source_stamps():
                assert setup._STAMPS.get() is not outer
                assert BoundVerificationCache.stamp(path) == first
                assert BoundVerificationCache.stamp(path) == first
                if inner_failure:
                    raise RuntimeError("original verification failed")
        except RuntimeError as error:
            assert str(error) == "original verification failed"
        assert setup._STAMPS.get() is outer
        assert BoundVerificationCache.stamp(path) == first
        assert count_for(counts, path) == (2 if inner_failure else 3)
    assert count_for(counts, path) == (3 if inner_failure else 4)
    BoundVerificationCache.stamp(path)
    assert count_for(counts, path) == (4 if inner_failure else 5)


def test_original_exception_restores_default_without_any_extra_stat(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"original")
    counts = count_actual_stats(monkeypatch, [path])
    with pytest.raises(RuntimeError, match="original failure"):
        with setup.reuse_setup_source_stamps():
            BoundVerificationCache.stamp(path)
            raise RuntimeError("original failure")
    assert count_for(counts, path) == 1
    assert setup._STAMPS.get() is None
    BoundVerificationCache.stamp(path)
    assert count_for(counts, path) == 2


@pytest.mark.parametrize("forced", ["argument", "fresh_scope"])
def test_forced_hashes_and_original_reader_before_after_stats_still_run(
        tmp_path, monkeypatch, forced):
    path = tmp_path / "source"
    raw = b"exact bytes" * 10000
    path.write_bytes(raw)
    counts = count_actual_stats(monkeypatch, [path])
    original_open, opened = Path.open, []

    def open_file(value, *args, **kwargs):
        if value == path and args == ("rb",):
            opened.append(value)
        return original_open(value, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    cache = BoundVerificationCache()
    with setup.reuse_setup_source_stamps():
        first = cache.descriptor(path, reader=sources._read_descriptor)
        if forced == "argument":
            second = cache.descriptor(path, reader=sources._read_descriptor, force=True)
        else:
            with cache.fresh():
                second = cache.descriptor(path, reader=sources._read_descriptor)
        assert first == second == dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        assert opened == [path, path]
        # Each original hash still performs both of its own actual stats.
        assert count_for(counts, path) == 5
    assert count_for(counts, path) == 6


@pytest.mark.parametrize("ceiling", ["files", "metadata"])
def test_snapshot_ceiling_validates_and_falls_back_without_rejecting_valid_data(
        tmp_path, monkeypatch, ceiling):
    paths = [tmp_path / str(index) for index in range(3)]
    for path in paths:
        path.write_bytes(b"unchanged")
    counts = count_actual_stats(monkeypatch, paths)
    limits = dict(maximum_files=1) if ceiling == "files" else dict(maximum_metadata_bytes=1)
    with setup.reuse_setup_source_stamps(**limits):
        BoundVerificationCache.stamp(paths[0])
        BoundVerificationCache.stamp(paths[1])
        snapshot = setup._STAMPS.get()
        assert snapshot["disabled"] and not snapshot["files"] and snapshot["cost"] == 0
        BoundVerificationCache.stamp(paths[0])
        BoundVerificationCache.stamp(paths[2])
        inside = list(counts.values())
    assert list(counts.values()) == inside
    assert inside == ([3, 1, 1] if ceiling == "files" else [2, 1, 1])


def test_ceiling_fallback_still_rejects_a_changed_retained_source(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"original")
    second.write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="changed during read-only verification"):
        with setup.reuse_setup_source_stamps(maximum_files=1):
            BoundVerificationCache.stamp(first)
            first.write_bytes(b"changed source")
            BoundVerificationCache.stamp(second)
    assert setup._STAMPS.get() is None
