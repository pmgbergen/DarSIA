"""Existing whole-cache accounting includes retained driver files and their hashes."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia_cache import bootstrap, common_qualification as common
from darsia_cache.environment import derive_cache_environment, first_qualification_environment
from darsia_cache.policy import CachePolicyError


def cache_measurement(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    root.mkdir()
    identity = dict(synthetic="cache root")
    class Namespace:
        def __init__(self, path, expected):
            assert Path(path) == root and expected == identity
            self.handle = root
            self.api = self
        def names(self, handle):
            return [path.name for path in Path(handle).iterdir()]
        def child(self, handle, name, **kwargs):
            return Path(handle) / name, object()
        def close(self, *args):
            pass
    monkeypatch.setattr(bootstrap, "WindowsNamespace", Namespace)
    monkeypatch.setattr(bootstrap, "volume_info", lambda path: dict(free_bytes=10**9))
    state = SimpleNamespace(policy=SimpleNamespace(root=str(root)), admission=dict(cache_identity=identity),
        entry_limit=1000, ledger=SimpleNamespace(control=SimpleNamespace(names=lambda: [])))
    def measure(**kwargs):
        return bootstrap.ImportLease.measure(state, str(root), **kwargs)
    return root, identity, measure, Namespace


def test_existing_scanner_counts_driver_cache_before_and_after_without_cleanup(tmp_path, monkeypatch):
    root, identity, measure, _ = cache_measurement(tmp_path, monkeypatch)
    token = "a" * 32
    service = root / "jobs" / token
    document = dict(cache=dict(root=str(root), quota_bytes=500 * 1024**3, profiles=dict(jobs={})))
    environment = first_qualification_environment(derive_cache_environment(document,
        purpose="jobs", token=token, scope_root=str(service), parent_environment={}), scope_root=str(service))
    driver = Path(environment["APPDATA"]) / "driver" / "ComputeCache"
    driver.mkdir(parents=True)
    existing = driver / "index.bin"
    existing.write_bytes(b"retained index")
    before = measure(include_file_stats=True, hash_files=True)
    generated = driver / "compiled.bin"
    generated.write_bytes(b"new compiler payload")
    after = measure(include_file_stats=True, hash_files=True)
    assert before["logical_bytes"] == len(b"retained index")
    assert after["logical_bytes"] == len(b"retained indexnew compiler payload")
    files = {Path(row["relative_path"]).name: row for row in after["files"]}
    assert set(files) == {"index.bin", "compiled.bin"}
    assert files["compiled.bin"]["sha256"] == hashlib.sha256(generated.read_bytes()).hexdigest()
    assert after["inventory_elapsed_seconds"] >= 0 and existing.is_file() and generated.is_file()
    namespace = dict(path=str(tmp_path / "namespace.json"), sha256="a" * 64, size_bytes=1)
    state = SimpleNamespace(root_provider=dict(namespace_admission=namespace))
    evidence = dict(service_namespace=str(service), registered_owner_environments=[dict(actual_writer_environment=environment)])
    measured = dict(retained_cache_before=before, retained_cache_after=after,
        retained_cache_files_deleted_by_this_code=False, retained_cache_scope_is_exhaustive_OS_write_proof=False)
    read = lambda pin: dict(cache_identity=identity)
    assert common._checked_cache_retention(state, evidence, measured, read) == after
    incorrect = deepcopy(measured)
    incorrect["retained_cache_after"]["logical_bytes"] -= 1
    with pytest.raises(CachePolicyError, match="total omits"):
        common._checked_cache_retention(state, evidence, incorrect, read)
    changed_environment = deepcopy(evidence)
    changed_environment["registered_owner_environments"][0]["actual_writer_environment"]["APPDATA"] = str(root / "other")
    with pytest.raises(CachePolicyError, match="driver-cache locations"):
        common._checked_cache_retention(state, changed_environment, measured, read)


def test_default_live_accounting_keeps_its_original_keys_and_reads_no_payload(tmp_path, monkeypatch):
    root, _, measure, _ = cache_measurement(tmp_path, monkeypatch)
    (root / "retained.bin").write_bytes(b"preserved")
    original = Path.open
    def refuse_payload(path, *args, **kwargs):
        if Path(path) == root / "retained.bin":
            pytest.fail("Default live accounting must not read active cache bytes")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", refuse_payload)
    measured = measure()
    assert set(measured) == {"root", "fresh", "root_identity", "no_reparse_traversal", "active_scope_bytes",
        "logical_bytes", "free_bytes", "measured_utc", "eligible_reclaimable_bytes"}
    assert measured["logical_bytes"] == len(b"preserved")


def test_changed_directory_inventory_refuses_instead_of_claiming_complete_files(tmp_path, monkeypatch):
    root, _, measure, namespace = cache_measurement(tmp_path, monkeypatch)
    (root / "old.bin").write_bytes(b"old")
    names = namespace.names
    def changing_names(self, handle):
        result = names(self, handle)
        if Path(handle) == root:
            (root / "late.bin").write_bytes(b"late")
            stamp = root.stat()
            os.utime(root, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
        return result
    monkeypatch.setattr(namespace, "names", changing_names)
    with pytest.raises(CachePolicyError, match="directory changed"):
        measure(include_file_stats=True, hash_files=True)
    assert (root / "late.bin").is_file()
