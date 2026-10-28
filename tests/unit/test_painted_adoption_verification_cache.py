"""Private-file verification, drift and bounded-cache publication regressions."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import pytest

from darsia.fluidflower import adoption_facies_table as facies
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower.bound_verification_cache import BoundVerificationCache
from darsia.fluidflower.pipeline import job_queue


@pytest.fixture
def cache(monkeypatch):
    value = BoundVerificationCache()
    monkeypatch.setattr(table, "_FILES", value)
    # The production memo is an ordered bounded container.
    from collections import OrderedDict
    monkeypatch.setattr(table, "_DERIVED", OrderedDict())
    return value


def bound_file(tmp_path, raw=b'{"value":1}'):
    path = tmp_path / "source.json"
    path.write_bytes(raw)
    return path, sources.descriptor(path)


def count_hashes(monkeypatch):
    original, observed = sources.descriptor, []
    def counted(path):
        observed.append(str(path))
        return original(path)
    monkeypatch.setattr(sources, "descriptor", counted)
    return observed


def test_unchanged_bound_bytes_are_hashed_once_and_metadata_results_are_isolated(tmp_path, monkeypatch, cache):
    path, binding = bound_file(tmp_path)
    observed = count_hashes(monkeypatch)
    first = table._checked(binding, metadata=True)
    first["value"] = 99
    assert table._checked(binding, metadata=True) == {"value": 1}
    assert table._bound(path) == binding
    assert observed == [str(path)]
    assert path.read_bytes() == b'{"value":1}'


def test_same_size_atomic_replacement_with_restored_mtime_invalidates_bound_cache(tmp_path, cache):
    path, binding = bound_file(tmp_path)
    table._checked(binding)
    old = path.stat()
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"value":2}')
    os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
    os.replace(replacement, path)
    assert path.stat().st_size == binding["size_bytes"]
    with pytest.raises(ValueError, match="changed"):
        table._checked(binding)
    assert path.read_bytes() == b'{"value":2}'


def test_new_publication_force_hashes_even_if_a_stamp_is_reported_unchanged(tmp_path, monkeypatch, cache):
    path, binding = bound_file(tmp_path)
    observed = count_hashes(monkeypatch)
    fixed = cache.stamp(path)
    monkeypatch.setattr(cache, "stamp", lambda path: fixed)
    table._checked(binding)
    path.write_bytes(b'{"value":2}')
    monkeypatch.setattr(table, "_verify_preparation_publication", lambda preparation, folder: table._checked(binding))
    with pytest.raises(ValueError, match="changed"):
        table.verify_preparation_publication({}, tmp_path)
    assert observed == [str(path), str(path)]
    # A known new hash must never resurrect an older success at the same stamp.
    with pytest.raises(ValueError, match="changed"):
        table._checked(binding)
    assert path.read_bytes() == b'{"value":2}'


def test_transient_hash_failure_retries_whole_original_reader_then_reuses_success(tmp_path, monkeypatch, cache):
    path, binding = bound_file(tmp_path)
    original, calls = sources.descriptor, []
    def flaky(path):
        calls.append(str(path))
        if len(calls) == 1:
            raise OSError("temporary SMB read failure")
        return original(path)
    monkeypatch.setattr(job_queue.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(sources, "descriptor", flaky)
    assert table._checked(binding) == binding
    assert table._checked(binding) == binding
    assert calls == [str(path), str(path)]


def test_transient_metadata_open_retries_without_caching_the_failed_read(tmp_path, monkeypatch, cache):
    path, _ = bound_file(tmp_path)
    original, calls = Path.open, []
    def flaky(self, *args, **kwargs):
        if self == path and args == ("rb",):
            calls.append(str(self))
            if len(calls) == 1:
                raise OSError("temporary SMB open failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(job_queue.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(Path, "open", flaky)
    assert cache.read(path, limit=32) == b'{"value":1}'
    assert cache.read(path, limit=32) == b'{"value":1}'
    assert calls == [str(path), str(path)]


def test_missing_file_is_not_retried_as_a_transient_error(tmp_path, monkeypatch, cache):
    waited = []
    monkeypatch.setattr(job_queue.time, "sleep", waited.append)
    with pytest.raises(FileNotFoundError):
        cache.read(tmp_path / "absent.json", limit=32)
    assert waited == []


def test_expected_identity_is_validated_before_using_an_optional_reader(tmp_path):
    path, binding = bound_file(tmp_path)
    called = []
    with pytest.raises(ValueError, match="explicit path"):
        sources._verify(dict(binding, sha256="bad"), descriptor_reader=called.append)
    assert called == []
    assert sources._verify(binding) == binding


def test_descriptor_and_metadata_cache_storage_is_bounded(tmp_path):
    cache = BoundVerificationCache(maximum_files=2, maximum_metadata_bytes=4)
    for number in range(4):
        path = tmp_path / str(number)
        path.write_bytes(b"large")
        cache.descriptor(path)
        assert cache.read(path, limit=5) == b"large"
    assert len(cache._descriptors) <= 2
    assert cache._metadata_bytes == 0
    assert not cache._metadata


def test_changed_process_discards_inherited_verification(tmp_path, monkeypatch):
    from darsia.fluidflower import bound_verification_cache as module
    cache = BoundVerificationCache()
    path, binding = bound_file(tmp_path)
    observed = count_hashes(monkeypatch)
    cache.descriptor(path, expected=binding)
    other_pid = cache._pid + 1
    monkeypatch.setattr(module.os, "getpid", lambda: other_pid)
    cache.descriptor(path, expected=binding)
    assert observed == [str(path), str(path)]


def simple_table():
    session = painting.PaintingSession("fixture", permeability_unit="mD")
    session.facies = {"Sand": painting.FaciesEntry("porous_medium", .4, 10.)}
    session.layers = [painting.PaintedLayer(1, "sand", "#aaaaaa", "Sand")]
    raw = b"id,name,extra\n4,Sand,original\n"
    return facies.build_adoption_facies_table(session, raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(), used_layer_ids=(1,))


def test_all_run_metadata_result_is_reused_but_a_changed_source_replays_original_checks(tmp_path, monkeypatch, cache):
    path, binding = bound_file(tmp_path)
    built, calls = simple_table(), []
    def derive_once(preparation, folder, source):
        calls.append(1)
        value = table._checked(binding, metadata=True)
        return built, {"actual": value}, [{"facies_id": 4}], "mD", [deepcopy(binding)]
    monkeypatch.setattr(table, "_derive_uncached", derive_once)
    first = table.derive({}, tmp_path, {})
    first[1]["actual"]["value"] = 99
    assert table.derive({}, tmp_path, {})[1] == {"actual": {"value": 1}}
    assert len(calls) == 1
    path.write_bytes(b'{"value":2}')
    with pytest.raises(ValueError, match="changed"):
        table.derive({}, tmp_path, {})
    assert len(calls) == 2


def test_changed_unbound_approval_history_invalidates_derived_metadata_without_revoking_history(tmp_path, monkeypatch, cache):
    history = tmp_path / "approvals.jsonl"
    history.write_bytes(b'{"approval":"original"}\n')
    built, calls = simple_table(), []
    def derive_once(preparation, folder, source):
        calls.append(1)
        raw = table._read(history)
        assert b'"original"' in raw
        return built, {"rows": len(raw.splitlines())}, [], "mD", []
    monkeypatch.setattr(table, "_derive_uncached", derive_once)
    assert table.derive({}, tmp_path, {})[1] == {"rows": 1}
    assert table.derive({}, tmp_path, {})[1] == {"rows": 1}
    with history.open("ab") as stream:
        stream.write(b'{"approval":"later"}\n')
    assert table.derive({}, tmp_path, {})[1] == {"rows": 2}
    assert len(calls) == 2


def test_forced_scope_bypasses_the_all_run_metadata_memo(tmp_path, monkeypatch, cache):
    _, binding = bound_file(tmp_path)
    built, calls = simple_table(), []
    def derive_once(preparation, folder, source):
        calls.append(1)
        table._checked(binding)
        return built, {}, [], "mD", [binding]
    monkeypatch.setattr(table, "_derive_uncached", derive_once)
    table.derive({}, tmp_path, {})
    table.derive({}, tmp_path, {})
    assert len(calls) == 1
    with cache.fresh():
        table.derive({}, tmp_path, {})
    assert len(calls) == 2


def test_known_new_hash_invalidates_derived_metadata_even_with_a_colliding_stamp(tmp_path, monkeypatch, cache):
    path, binding = bound_file(tmp_path)
    fixed = cache.stamp(path)
    monkeypatch.setattr(cache, "stamp", lambda path: fixed)
    built, calls = simple_table(), []
    def derive_once(preparation, folder, source):
        calls.append(1)
        table._checked(binding)
        return built, {}, [], "mD", [binding]
    monkeypatch.setattr(table, "_derive_uncached", derive_once)
    table.derive({}, tmp_path, {})
    path.write_bytes(b'{"value":2}')
    with cache.fresh():
        cache.descriptor(path, expected=binding)
    with pytest.raises(ValueError, match="changed"):
        table.derive({}, tmp_path, {})
    assert len(calls) == 2


def test_outside_row_is_above_old_and_new_ids_and_never_carries_parent_extra_cells():
    session = painting.PaintingSession("fixture", permeability_unit="mD")
    session.facies = {"Free water": painting.FaciesEntry("free_water"),
                      "Sand": painting.FaciesEntry("porous_medium", .4, 10.),
                      "Fixture": painting.FaciesEntry("excluded")}
    session.layers = [painting.PaintedLayer(i, name, "#aaaaaa", name)
                      for i, name in enumerate(session.facies, 1)]
    raw = ("id,name,extra\n0,water,old-water\n2,Sand,keep\n"
           f"50,{painting.OUTSIDE_FACIES},never-carry\n").encode()
    original = facies.build_adoption_facies_table(session, raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(), used_layer_ids=(1, 2, 3))
    result = table._with_frame_row(original, raw)
    outside = result.rows[-1]
    assert original.facies_ids["Fixture"] == 51
    assert result.facies_ids[painting.OUTSIDE_FACIES] == 52
    assert outside["extra"] == ""
    assert dict(result.provenance[-1].fields)["extra"] == "not-applicable:proven-nonphysical-frame"
    assert result.provenance[-1].parent_ids == (50,)
    assert result.source_max_id == original.source_max_id == 50
    assert result.retired_source_ids == original.retired_source_ids
    assert result.changed_source_ids == original.changed_source_ids == ()
    assert result.aliases == original.aliases
    assert result.layer_to_facies[1] == 0 and result.layer_to_facies[0] == 52


def test_forced_metadata_read_invalidates_old_descriptor_at_the_same_full_stamp(tmp_path, monkeypatch, cache):
    path, binding = bound_file(tmp_path)
    observed = count_hashes(monkeypatch)
    fixed = cache.stamp(path)
    monkeypatch.setattr(cache, "stamp", lambda path: fixed)
    assert table._checked(binding, metadata=False) == binding
    path.write_bytes(b'{"value":2}')
    assert cache.read(path, limit=32, force=True) == b'{"value":2}'
    # The read itself observed the new bytes. An old metadata=False binding
    # must now be rehashed/refused, even though every reported stamp collides.
    with pytest.raises(ValueError, match="changed"):
        table._checked(binding, metadata=False)
    assert observed == [str(path), str(path)]
    with pytest.raises(ValueError, match="changed"):
        table._checked(binding, metadata=False)
    assert path.read_bytes() == b'{"value":2}'
