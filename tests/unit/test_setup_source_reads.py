"""Exact source reads are shared only inside an explicit Setup operation."""
import hashlib
import os
from pathlib import Path

import pytest

from darsia.fluidflower import corrected_setup as corrected
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower.setup_source_reads import reuse_setup_source_reads


def binding(path, raw):
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw))


def count_source_opens(monkeypatch, paths):
    """Count the actual binary file opens used by the original hash readers."""
    observed = {str(path.resolve()): 0 for path in paths}
    original = Path.open

    def opened(path, *args, **kwargs):
        key = str(path.resolve())
        if args == ("rb",) and key in observed:
            observed[key] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    return observed


def test_default_readers_still_hash_every_time(tmp_path, monkeypatch):
    path = tmp_path / "source.npz"
    expected = binding(path, b"original bytes")
    observed = count_source_opens(monkeypatch, [path])
    for _ in range(2):
        assert corrected._verified(expected) == expected
        assert sources._verify(expected) == expected
    assert observed[str(path)] == 4


def test_all_run_guards_share_each_actual_source_read(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("labels.npz", "facies.npz", "table.csv")]
    expected = [binding(path, (path.name.encode() + b"x") * 100000) for path in paths]
    observed = count_source_opens(monkeypatch, paths)
    with reuse_setup_source_reads() as cache:
        for _ in range(5):
            for row in expected:
                assert corrected._verified(row) == row
                assert sources._verify(row) == row
        # Returned descriptors cannot alter the retained exact identity.
        corrected._descriptor(paths[0])["sha256"] = "0" * 64
        assert sources._verify(expected[0]) == expected[0]
        assert len(cache._descriptors) == 3
        assert not cache._metadata
    assert list(observed.values()) == [1, 1, 1]
    assert not cache._descriptors


def test_wrong_expected_bytes_reject_before_and_after_reuse(tmp_path, monkeypatch):
    path = tmp_path / "source.npz"
    expected = binding(path, b"exact original")
    wrong_hash = dict(expected, sha256="0" * 64)
    wrong_size = dict(expected, size_bytes=expected["size_bytes"] + 1)
    observed = count_source_opens(monkeypatch, [path])
    with reuse_setup_source_reads():
        with pytest.raises(ValueError, match="identity mismatch"):
            corrected._verified(wrong_hash)
        assert sources._verify(expected) == expected
        with pytest.raises(ValueError, match="changed"):
            sources._verify(wrong_hash)
        with pytest.raises(ValueError, match="identity mismatch"):
            corrected._verified(wrong_size)
    assert observed[str(path)] == 1


def test_same_size_replacement_with_restored_mtime_is_rehashed_and_rejected(tmp_path, monkeypatch):
    path = tmp_path / "source.npz"
    expected = binding(path, b"original")
    observed = count_source_opens(monkeypatch, [path])
    with reuse_setup_source_reads() as cache:
        assert corrected._verified(expected) == expected
        before = path.stat()
        old_stamp = cache.stamp(path)
        replacement = tmp_path / "replacement.npz"
        replacement.write_bytes(b"modified")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, path)
        assert path.stat().st_size == expected["size_bytes"]
        assert cache.stamp(path) != old_stamp
        with pytest.raises(ValueError, match="identity mismatch"):
            corrected._verified(expected)
        with pytest.raises(ValueError, match="changed"):
            sources._verify(expected)
    assert observed[str(path)] == 2


def test_a_source_changed_during_a_hash_is_not_cached(tmp_path, monkeypatch):
    path = tmp_path / "source.npz"
    binding(path, b"original")
    original = corrected._read_descriptor

    def changed_after_read(path):
        actual = original(path)
        path.write_bytes(b"a changed source")
        return actual

    monkeypatch.setattr(corrected, "_read_descriptor", changed_after_read)
    with reuse_setup_source_reads() as cache:
        with pytest.raises(ValueError, match="changed during verification"):
            corrected._descriptor(path)
        assert not cache._descriptors


def test_completion_pass_hashes_again_even_when_a_file_stamp_collides(tmp_path, monkeypatch):
    path = tmp_path / "source.npz"
    expected = binding(path, b"original")
    observed = count_source_opens(monkeypatch, [path])
    with reuse_setup_source_reads() as operation:
        assert corrected._verified(expected) == expected
        assert sources._verify(expected) == expected
        old_stamp = operation.stamp(path)
    path.write_bytes(b"modified")
    with reuse_setup_source_reads() as completion:
        monkeypatch.setattr(completion, "stamp", lambda path: old_stamp)
        with pytest.raises(ValueError, match="changed"):
            sources._verify(expected)
        with pytest.raises(ValueError, match="identity mismatch"):
            corrected._verified(expected)
    assert observed[str(path)] == 2


def test_nested_scope_and_failure_restore_the_outer_and_default_readers(tmp_path, monkeypatch):
    path = tmp_path / "source.npz"
    expected = binding(path, b"original")
    observed = count_source_opens(monkeypatch, [path])
    with reuse_setup_source_reads():
        assert corrected._verified(expected) == expected
        with pytest.raises(RuntimeError, match="stop"):
            with reuse_setup_source_reads():
                assert sources._verify(expected) == expected
                raise RuntimeError("stop")
        assert sources._verify(expected) == expected
        assert observed[str(path)] == 2
    assert corrected._verified(expected) == expected
    assert sources._verify(expected) == expected
    assert observed[str(path)] == 4


def test_descriptor_retention_is_bounded_and_evicted_sources_rehash(tmp_path, monkeypatch):
    paths = [tmp_path / str(index) for index in range(3)]
    expected = [binding(path, b"source") for path in paths]
    observed = count_source_opens(monkeypatch, paths)
    with reuse_setup_source_reads(maximum_files=2) as cache:
        for row in expected:
            assert corrected._verified(row) == row
            assert len(cache._descriptors) <= 2
            assert not cache._metadata
        assert sources._verify(expected[0]) == expected[0]
    assert list(observed.values()) == [2, 1, 1]


def test_shared_hash_preserves_each_readers_locator_spelling(tmp_path, monkeypatch):
    path = tmp_path / "source.npz"
    expected = binding(path, b"original")
    observed = count_source_opens(monkeypatch, [path])
    monkeypatch.chdir(tmp_path)
    with reuse_setup_source_reads():
        assert sources._verify(expected) == expected
        assert corrected._descriptor("source.npz") == dict(expected, path="source.npz")
    assert observed[str(path)] == 1
