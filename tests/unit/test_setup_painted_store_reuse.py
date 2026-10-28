"""Finite complete-proof reuse within one immutable Setup operation only."""
import hashlib
from pathlib import Path

import pytest

from darsia.fluidflower import setup_source_reads as reads
from darsia.fluidflower.painted_adoption_table import _FILES, _checked


def fixture(tmp_path):
    path = tmp_path / "labels.npz"
    raw = b"synthetic exact payload"
    path.write_bytes(raw)
    bound = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    calls = []

    def reader(binding):
        _checked(binding["source"])
        calls.append(binding["run"])
        # Stand in for expensive decompression after the original bound read.
        assert Path(binding["source"]["path"]).read_bytes() == raw

    return dict(source=bound, run="one"), reader, calls


def test_default_and_new_operations_still_run_original_proof(tmp_path):
    bound, reader, calls = fixture(tmp_path)
    for _ in range(2):
        reads.reuse_painted_store_proof(bound, reader=reader)
    assert calls == ["one", "one"]
    with reads.reuse_setup_source_reads():
        for _ in range(5):
            reads.reuse_painted_store_proof(bound, reader=reader)
        with reads.reuse_setup_source_reads():
            reads.reuse_painted_store_proof(bound, reader=reader)
        reads.reuse_painted_store_proof(bound, reader=reader)
    assert calls == ["one"] * 4
    with reads.reuse_setup_source_reads():
        reads.reuse_painted_store_proof(bound, reader=reader)
    assert calls == ["one"] * 5


def test_changed_receipt_identity_cannot_reuse_another_run(tmp_path):
    bound, reader, calls = fixture(tmp_path)
    with reads.reuse_setup_source_reads():
        reads.reuse_painted_store_proof(bound, reader=reader)
        reads.reuse_painted_store_proof(dict(bound, run="two"), reader=reader)
    assert calls == ["one", "two"]


def test_fresh_publication_rehashes_even_when_complete_stamp_collides(tmp_path, monkeypatch):
    bound, reader, calls = fixture(tmp_path)
    with reads.reuse_setup_source_reads() as cache:
        reads.reuse_painted_store_proof(bound, reader=reader)
        stamp = cache.stamp(bound["source"]["path"])
        Path(bound["source"]["path"]).write_bytes(b"modified bytes payload")
        monkeypatch.setattr(cache, "stamp", lambda path: stamp)
        monkeypatch.setattr(_FILES, "stamp", lambda path: stamp)
        with _FILES.fresh(), pytest.raises(ValueError, match="changed"):
            reads.reuse_painted_store_proof(bound, reader=reader)
        assert not reads._PAINTED_STORE.get()["entries"]
    assert calls == ["one"]


@pytest.mark.parametrize("limit", ["entries", "bytes"])
def test_retention_is_bounded_without_raster_values(tmp_path, monkeypatch, limit):
    bound, reader, calls = fixture(tmp_path)
    monkeypatch.setattr(reads, "_MAXIMUM_PAINTED_STORE_PROOFS", 1)
    if limit == "bytes":
        monkeypatch.setattr(reads, "_MAXIMUM_PAINTED_STORE_METADATA_BYTES", 1)
    with reads.reuse_setup_source_reads():
        reads.reuse_painted_store_proof(bound, reader=reader)
        reads.reuse_painted_store_proof(dict(bound, run="two"), reader=reader)
        reads.reuse_painted_store_proof(bound, reader=reader)
        entries = reads._PAINTED_STORE.get()["entries"]
        assert len(entries) == (0 if limit == "bytes" else 1)
        assert all(set(value) == {"dependencies", "cost"} for value in entries.values())
    assert calls == ["one", "two", "one"]
