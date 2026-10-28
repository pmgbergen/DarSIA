"""Finite, strict shared-metadata snapshots using private byte fixtures only."""
from collections import Counter
import os
from pathlib import Path

import pytest

from darsia.fluidflower import published_jobs_snapshot as snapshots
from darsia.fluidflower.pipeline import products_publication as publication

from .test_alias_published_snapshot import published_fixture, snapshot
from .test_products_publication import fixture, publish, as_product, write


def context(index):
    root, _document, rows = publication._read_index(index)
    return snapshots._SnapshotMetadata(index, root, rows), rows


def copy_path(index, rows, key):
    return Path(index["path"]).parent.joinpath(*rows[key]["metadata_copy"]["relative_path"].split("/"))


def alter_same_size_and_mtime(path):
    before = path.stat()
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_actual_terminal_parses_one_index_and_verifies_each_distinct_shared_copy(tmp_path, monkeypatch):
    data = published_fixture(tmp_path, monkeypatch, prefix=True)
    index_calls, byte_reads, final_checks = [], Counter(), Counter()
    original_index = snapshots._read_index
    original_bytes, original_verify = snapshots._verified_bytes, snapshots._verify_file
    original_open = Path.open

    def index_reader(bound):
        index_calls.append(bound["path"])
        return original_index(bound)

    def read_bytes(path, expected, **kwargs):
        byte_reads[str(path)] += 1
        return original_bytes(path, expected, **kwargs)

    def verify(path, expected, **kwargs):
        final_checks[str(path)] += 1
        return original_verify(path, expected, **kwargs)

    def no_original(path, *args, **kwargs):
        assert all(not path.is_relative_to(claim.output) for claim in data.claims)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(snapshots, "_read_index", index_reader)
    monkeypatch.setattr(snapshots, "_verified_bytes", read_bytes)
    monkeypatch.setattr(snapshots, "_verify_file", verify)
    monkeypatch.setattr(Path, "open", no_original)
    value = snapshot(data)
    assert value["state"] == "Complete" and value["committed"] == 2
    assert len(index_calls) == 1
    metadata_reads = {path: count for path, count in byte_reads.items() if path != index_calls[0]}
    assert metadata_reads and set(metadata_reads.values()) == {1}
    assert final_checks == Counter({path: 1 for path in metadata_reads})
    assert byte_reads[index_calls[0]] == 1, "Final index SHA is checked without reparsing it"


def test_cache_is_bounded_and_eviction_reverifies_the_same_physical_copy(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    index = publish(tmp_path, data)
    reader, rows = context(index)
    keys = ["frames/f1/result.json", "manifest.json"]
    budget = max(rows[key]["bytes"] for key in keys)
    monkeypatch.setattr(snapshots, "MAX_METADATA_BYTES", budget)
    reads = Counter()
    original = snapshots._verified_bytes

    def observed(path, expected, **kwargs):
        reads[str(path)] += 1
        return original(path, expected, **kwargs)

    monkeypatch.setattr(snapshots, "_verified_bytes", observed)
    first = reader.read(keys[0])[1]
    assert reader.read(keys[0])[1] == first
    reader.read(keys[1])
    assert reader.read(keys[0])[1] == first
    assert reads[str(copy_path(index, rows, keys[0]))] == 2
    assert reads[str(copy_path(index, rows, keys[1]))] == 1
    reader.finish()


def test_final_check_detects_same_size_same_mtime_metadata_tampering(tmp_path):
    index = publish(tmp_path, fixture(tmp_path))
    reader, rows = context(index)
    reader.read("manifest.json")
    alter_same_size_and_mtime(copy_path(index, rows, "manifest.json"))
    with pytest.raises(ValueError, match="Metadata bytes differ"):
        reader.finish()


def test_new_snapshot_never_reuses_a_previous_refreshs_metadata_bytes(tmp_path):
    index = publish(tmp_path, fixture(tmp_path))
    first, rows = context(index)
    first.read("manifest.json")
    first.finish()
    alter_same_size_and_mtime(copy_path(index, rows, "manifest.json"))
    second, _rows = context(index)
    with pytest.raises(ValueError, match="Metadata bytes differ"):
        second.read("manifest.json")


def test_every_duplicate_original_path_requires_its_own_valid_shared_copy(tmp_path):
    data = fixture(tmp_path)
    original = next(row for row in data["products"] if row["product_id"] == "manifest.json")
    asset = dict(path=original["local_path"], sha256=original["sha256"], size_bytes=original["bytes"])
    data["products"].append(as_product("duplicate/manifest.json", asset, kind="manifest"))
    index = publish(tmp_path, data, metadata_product_ids=[
        "frames/f1/result.json", "manifest.json", "mass_table.csv", "duplicate/manifest.json"])
    reader, rows = context(index)
    alter_same_size_and_mtime(copy_path(index, rows, "duplicate/manifest.json"))
    with pytest.raises(ValueError, match="Metadata bytes differ"):
        snapshots._original_metadata(index, rows, asset, Path(asset["path"]).parent,
                                     original["machine_id"], metadata=reader)


def test_index_identity_is_rechecked_before_snapshot_return(tmp_path):
    index = publish(tmp_path, fixture(tmp_path))
    reader, _rows = context(index)
    reader.read("manifest.json")
    alter_same_size_and_mtime(Path(index["path"]))
    with pytest.raises(ValueError, match="Metadata bytes differ"):
        reader.finish()


def test_link_replacement_of_a_previously_verified_copy_is_refused(tmp_path):
    index = publish(tmp_path, fixture(tmp_path))
    reader, rows = context(index)
    reader.read("manifest.json")
    path = copy_path(index, rows, "manifest.json")
    retained = path.with_name("retained-manifest.json")
    path.rename(retained)
    try:
        path.symlink_to(retained)
    except OSError:
        pytest.skip("The private test account cannot create symlinks")
    with pytest.raises(ValueError, match="Symlink/reparse"):
        reader.finish()


def test_shared_json_keeps_the_original_duplicate_key_rejection(tmp_path):
    data = fixture(tmp_path)
    row = next(value for value in data["products"] if value["product_id"] == "manifest.json")
    asset = write(Path(row["local_path"]), b'{"precision":"float32","precision":"float64"}\n')
    data["products"] = [as_product("manifest.json", asset, kind="manifest")
                        if value["product_id"] == "manifest.json" else value
                        for value in data["products"]]
    index = publish(tmp_path, data)
    reader, rows = context(index)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        snapshots._original_metadata(index, rows, asset, Path(asset["path"]).parent,
                                     row["machine_id"], metadata=reader)
