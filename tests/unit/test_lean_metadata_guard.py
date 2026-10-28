"""The lean-only publication ceiling is an actual bound, with unchanged bytes."""
import gzip
import hashlib
from pathlib import Path

import pytest

from darsia.fluidflower.mass import n1_pipeline as producer
from darsia.fluidflower.pipeline import io_ops, job_queue, lean_capacity


def test_capacity_and_existing_publishers_share_exact_ceiling(tmp_path):
    """Capacity and existing publishers share exact ceiling."""
    limit = io_ops.LEAN_METADATA_BYTES
    assert limit == 16 * 1024 * 1024
    assert lean_capacity.LEAN_METADATA_BYTES == limit
    # A JSON string contributes exactly its two quote bytes in this writer.
    text = "a" * (limit - 2)
    expected = ('"' + text + '"').encode("ascii")
    target = tmp_path / "boundary.json"
    with io_ops.metadata_ceiling(limit):
        descriptor = producer._write_json(target, text)
    assert descriptor["size_bytes"] == limit
    assert descriptor["sha256"] == hashlib.sha256(expected).hexdigest()
    assert target.read_bytes() == expected
    with io_ops.metadata_ceiling(limit), pytest.raises(ValueError, match="ceiling"):
        producer._write_json(tmp_path / "too-large.json", text + "a")
    assert not (tmp_path / "too-large.json").exists()
    assert list(tmp_path.glob(".too-large.json.*.partial")) == []


def test_known_oversize_refuses_before_creating_a_directory_and_context_restores(tmp_path):
    """Known oversize refuses before creating a directory and context restores."""
    target = tmp_path / "never-created" / "result.json"
    with io_ops.metadata_ceiling(8), pytest.raises(ValueError, match="ceiling"):
        job_queue._publish_bytes(target, b"123456789", create_parents=True)
    assert not target.parent.exists()
    # Existing unscoped publication stays unchanged, including its exact bytes.
    job_queue._publish_bytes(target, b"123456789", create_parents=True)
    assert target.read_bytes() == b"123456789"


def test_streaming_refusal_retains_only_bounded_partial_before_crossing_limit(tmp_path):
    """Streaming refusal retains only bounded partial before crossing limit."""
    target = tmp_path / "native_statistics.json.gz"
    def writer(stream):
        stream.write(b"12345678")
        stream.write(b"123456789")
    with io_ops.metadata_ceiling(16), pytest.raises(ValueError, match="ceiling"):
        io_ops.publish_file(target, writer)
    assert not target.exists()
    partials = list(tmp_path.glob(".native_statistics.json.gz.*.partial"))
    assert len(partials) == 1 and partials[0].read_bytes() == b"12345678"
    # A refusal is a ValueError, not a transient I/O retry or a science replay.


def test_compressed_statistics_are_bounded_on_actual_published_bytes(tmp_path):
    """Compressed statistics are bounded on actual published bytes."""
    target = tmp_path / "native_statistics.json.gz"
    content = b'{"values":[1,2,3]}'
    def writer(stream):
        with gzip.GzipFile(filename="native_statistics.json", mode="wb", fileobj=stream,
                           compresslevel=3, mtime=0) as compressed:
            compressed.write(content)
    with io_ops.metadata_ceiling(256):
        io_ops.publish_file(target, writer)
    assert gzip.decompress(target.read_bytes()) == content
    assert target.stat().st_size <= 256


def test_no_seek_or_truncate_can_cross_a_scoped_metadata_limit(tmp_path):
    """No seek or truncate can cross a scoped metadata limit."""
    for action in (lambda stream: stream.seek(17), lambda stream: stream.truncate(17)):
        target = tmp_path / ("seek.json" if not list(tmp_path.iterdir()) else "truncate.json")
        with io_ops.metadata_ceiling(16), pytest.raises(ValueError, match="ceiling"):
            io_ops.publish_file(target, action)
        assert not target.exists()
        assert all(path.stat().st_size <= 16 for path in tmp_path.iterdir())


def test_nested_scope_cannot_weaken_limit_and_array_output_is_unchanged(tmp_path):
    """Nested scope cannot weaken limit and array output is unchanged."""
    with io_ops.metadata_ceiling(8), io_ops.metadata_ceiling(16):
        with pytest.raises(ValueError, match="ceiling"):
            job_queue._publish_bytes(tmp_path / "result.json", b"123456789", create_parents=False)
        io_ops.publish_file(tmp_path / "arrays.npz", lambda stream: stream.write(b"123456789"))
    assert (tmp_path / "arrays.npz").read_bytes() == b"123456789"


@pytest.mark.parametrize("profile, expected", [(None, None), ("full", None), ("lean", io_ops.LEAN_METADATA_BYTES)])
def test_n1_run_scope_uses_only_explicit_lean_and_forwards_original_arguments(profile, expected, monkeypatch):
    """N1 run scope uses only explicit lean and forwards original arguments."""
    task = dict(run="sample")
    if profile is not None:
        task["storage_profile"] = profile
    observed = []
    def runner(*args, **kwargs):
        observed.append((args, kwargs, io_ops.metadata_limit("result.json")))
        return dict(complete=True)
    monkeypatch.setattr(producer, "_run_n1_content", runner)
    runtime, fields, progress = {"threads": 1}, {"field": "unchanged"}, lambda event: None
    assert producer._run_n1(task, Path("unused"), "cpu", "float32", progress,
        thread_runtime=runtime, geometry_fields=fields) == dict(complete=True)
    args, kwargs, actual = observed[0]
    assert actual == expected
    assert args == (task, Path("unused"), "cpu", "float32", progress)
    assert kwargs == dict(thread_runtime=runtime, geometry_fields=fields)
    assert io_ops.metadata_limit("result.json") is None


@pytest.mark.parametrize("lean", [None, object()])
def test_direct_frame_publisher_scopes_lean_only_and_restores_after_failure(lean, monkeypatch):
    """Direct frame publisher scopes lean only and restores after failure."""
    def writer(*args, **kwargs):
        assert io_ops.metadata_limit("result.json") == (None if lean is None else io_ops.LEAN_METADATA_BYTES)
        assert kwargs["lean"] is lean and kwargs["storage_context"] == "same-context"
        raise ValueError("Synthetic publication refusal")
    monkeypatch.setattr(producer, "_publish_frame_content", writer)
    with pytest.raises(ValueError, match="Synthetic"):
        producer._publish_frame(None, None, None, None, None, None, None, None, None,
            lean=lean, storage_context="same-context")
    assert io_ops.metadata_limit("result.json") is None


def test_optional_storage_receipts_share_ceiling_and_keep_exact_bytes(tmp_path):
    """Optional storage receipts share ceiling and keep exact bytes."""
    from darsia.fluidflower import storage_outputs
    value = dict(z="a", a=[1, 2])
    target = tmp_path / "storage_completed.json"
    with io_ops.metadata_ceiling(64):
        storage_outputs._write_json(target, value)
    assert target.read_bytes() == b'{"a":[1,2],"z":"a"}\n'
    refused = tmp_path / "storage_extensive_frame.json"
    with io_ops.metadata_ceiling(8), pytest.raises(ValueError, match="ceiling"):
        storage_outputs._write_json(refused, value)
    assert not refused.exists()
    assert list(tmp_path.glob(".storage_extensive_frame.json.*.partial")) == []


def test_pixel_domain_scalar_table_is_bounded_in_the_same_lean_scope(tmp_path):
    """Pixel domain scalar table is bounded in the same lean scope."""
    path = tmp_path / "mass_table.csv"
    with io_ops.metadata_ceiling(8), pytest.raises(ValueError, match="ceiling"):
        io_ops.publish_file(path, lambda stream: stream.write(b"123456789"))
    assert not path.exists()
    assert io_ops.metadata_limit(path) is None
