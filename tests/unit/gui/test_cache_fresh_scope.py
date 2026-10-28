"""Bounded TESTONLY ownership transitions, not production retirement profiles.

The in-memory backend exercises partial target decisions without WinAPI calls.
Two isolated Windows tests additionally exercise actual nofollow/disposition on
pytest-owned NTFS scopes. No old cache, real service, quota or publication is
adopted; actual complete child-retirement coverage remains externally required.
"""
from contextlib import contextmanager
from copy import deepcopy
import base64
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia_cache import fresh_scope, windows
from darsia_cache.control import _encoded
from darsia_cache.environment import (derive_cache_environment,
    WRITER_ENVIRONMENT_KEYS, WRITER_SUBDIRECTORIES)
from darsia_cache.policy import CachePolicy, CachePolicyError


TOKEN = "1234567812344abc8abc123456789abc"
UTC = "2026-01-01T00:00:00+00:00"


class MemoryAPI:
    """Explicit synthetic namespace/handle model, never a WinAPI substitute."""
    def __init__(self, nodes):
        self.nodes = {"": dict(directory=True, bytes=0, id=1), **deepcopy(nodes)}
        for index, row in enumerate(self.nodes.values(), 1):
            row.setdefault("id", index)
            row.setdefault("bytes", 0)
        self.disposed, self.closed, self.opens = set(), [], []
        self.live, self.peak_open = {""}, 1
        self.reverse_each_enumeration = False
        self.enumerations = 0

    def identity(self, handle):
        row = self.nodes[handle]
        return windows.ObjectIdentity(7, f"{row['id']:032x}", row["directory"],
                                      row["bytes"], 100, 90, 100)

    def names(self, handle, *, with_types=False):
        self.enumerations += 1
        prefix = handle + "/" if handle else ""
        result = []
        for path, row in sorted(self.nodes.items()):
            if path != handle and path.startswith(prefix) and "/" not in path[len(prefix):]:
                result.append(dict(name=path[len(prefix):], directory=row["directory"],
                                   reparse=row.get("unsafe") == "reparse"))
        if not with_types:
            if any(row["reparse"] for row in result):
                raise CachePolicyError("TESTONLY reparse entry")
            return [row["name"] for row in result]
        return list(reversed(result)) if self.reverse_each_enumeration and self.enumerations % 2 == 0 else result

    def child(self, parent, name, *, directory, mutable=False):
        assert mutable is True
        path = parent + "/" + name if parent else name
        self.opens.append(path)
        row = self.nodes[path]
        if row.get("unsafe") in {"open", "hardlink"}:
            raise windows.CacheObjectIncomplete("TESTONLY exclusive/single-link refusal",
                dict(handle_returned=False, handle_close_confirmed=False))
        if row.get("unsafe") == "uncertain_close":
            raise windows.CacheObjectIncomplete("TESTONLY child close unconfirmed",
                dict(handle_returned=True, handle_close_confirmed=False,
                     close_failures=[dict(handle=73, winerror=6)]))
        assert row.get("unsafe") != "reparse", "Reparse entry must never be opened"
        assert row["directory"] is directory
        self.live.add(path)
        self.peak_open = max(self.peak_open, len(self.live))
        return path, self.identity(path)

    def dispose(self, handle):
        row = self.nodes[handle]
        if row.get("fail_dispose"):
            raise OSError("TESTONLY disposition refused")
        assert not row.get("unsafe")
        if row["directory"]:
            assert not self.names(handle, with_types=True)
        self.disposed.add(handle)

    def close(self, handle):
        self.closed.append(handle)
        if self.nodes[handle].get("fail_close"):
            raise windows.CacheHandleCloseIncomplete([dict(handle=handle, winerror=6)])
        self.live.remove(handle)
        if handle in self.disposed:
            del self.nodes[handle]


class MemoryTree:
    def __init__(self, nodes):
        self.api, self.handles, self.closed = MemoryAPI(nodes), {"": ""}, False

    def close(self):
        if not self.closed:
            while self.handles:
                _, handle = self.handles.popitem()
                self.api.close(handle)
            self.closed = True


def tree_for(nodes):
    return MemoryTree(nodes)


def context_for(tmp_path):
    policy = CachePolicy(str(tmp_path / "cache"), 4096, 48, {})
    owner = dict(machine="TESTONLY", pid=11, process_create_time="1")
    reservation = dict(token=TOKEN, pool="TESTONLY", sha256="a" * 64)
    root_identity = windows.ObjectIdentity(7, "1".zfill(32), True, 0, 100, 90, 100).document()
    root = policy.purpose("jobs") / TOKEN
    fresh = dict(format="darsia-fresh-exclusive-empty-service-scope-v1", token=TOKEN,
        owner=owner, purpose="jobs", root_identity=root_identity,
        creation_disposition="NtCreateFile.FILE_CREATE", exclusive_root_handle=True,
        initial_names=[], initial_empty_observed_utc=UTC,
        observation_precedes_writer_subdirectories_and_imports=True,
        writer_subdirectories=list(WRITER_SUBDIRECTORIES),
        writer_environment_keys=list(WRITER_ENVIRONMENT_KEYS))
    manifest = dict(owner=owner, reservation=reservation, token=TOKEN, root=str(root),
                    root_identity=root_identity, fresh_scope_provenance=fresh)
    return SimpleNamespace(policy=policy, purpose="jobs", reservation=reservation,
        manifest=manifest, root=root, maximum=4096, entry_limit=128,
        admission=dict(purposes=dict(jobs=dict(path=str(root.parent), identity=root_identity))),
        receipt=dict(path=str(tmp_path / "receipt.json"), sha256="b" * 64, size_bytes=10),
        context=dict(host="TESTONLY", common_root=str(tmp_path / "control")),
        owner=dict(machine="TESTONLY", pid=22, process_create_time="2"))


def test_typed_bound_scope_requires_the_same_expanded_writer_environment_as_qualification(tmp_path):
    from darsia_cache.environment import FIRST_WRITER_ENVIRONMENT_KEYS, FIRST_WRITER_SUBDIRECTORIES
    context = context_for(tmp_path)
    original = deepcopy(context.manifest["fresh_scope_provenance"])
    # The original direct path still accepts precisely its old writer list.
    assert fresh_scope._fresh_provenance(context) == original
    context.profile = dict(format="darsia-cache-volume-bound-import-profile-v1")
    with pytest.raises(CachePolicyError, match="fresh-exclusive-empty"):
        fresh_scope._fresh_provenance(context)
    context.manifest["fresh_scope_provenance"].update(
        writer_subdirectories=list(FIRST_WRITER_SUBDIRECTORIES),
        writer_environment_keys=list(FIRST_WRITER_ENVIRONMENT_KEYS))
    assert fresh_scope._fresh_provenance(context)["writer_environment_keys"] == list(FIRST_WRITER_ENVIRONMENT_KEYS)
    context.manifest["fresh_scope_provenance"]["initial_names"] = ["old-cache"]
    with pytest.raises(CachePolicyError): fresh_scope._fresh_provenance(context)


def test_fresh_safe_unknown_names_are_not_a_predeclared_inventory():
    tree = tree_for({"third-party": dict(directory=True),
                     "third-party/unpredicted.cache": dict(directory=False, bytes=23)})
    result = fresh_scope._cleanup_current_fresh_tree(tree, entry_limit=32)
    assert result["complete"] is True
    assert result["removed_files"] == ["third-party/unpredicted.cache"]
    assert result["removed_logical_bytes"] == 23
    assert "" in result["removed_directories"]
    assert not tree.api.nodes


@pytest.mark.parametrize("unsafe", ["reparse", "hardlink", "open"])
def test_unsafe_entry_retained_but_independent_safe_branch_is_removed(unsafe):
    tree = tree_for({"unsafe": dict(directory=False, bytes=11, unsafe=unsafe),
                     "safe": dict(directory=True),
                     "safe/unannounced.bin": dict(directory=False, bytes=29)})
    result = fresh_scope._cleanup_current_fresh_tree(tree, entry_limit=32)
    assert result["complete"] is False
    assert result["reservation_must_remain_charged"] is True
    assert result["removed_files"] == ["safe/unannounced.bin"]
    assert "safe" in result["removed_directories"]
    assert set(tree.api.nodes) == {"", "unsafe"}
    assert "unsafe" not in tree.api.disposed
    assert any(row["path"] == "unsafe" for row in result["retained"])
    if unsafe == "reparse":
        assert "unsafe" not in tree.api.opens


@pytest.mark.parametrize("failure", ["fail_dispose", "fail_close"])
def test_disposition_or_close_failure_never_becomes_complete(failure):
    tree = tree_for({"safe.bin": dict(directory=False, bytes=5, **{failure: True})})
    result = fresh_scope._cleanup_current_fresh_tree(tree, entry_limit=32)
    assert result["complete"] is False
    assert result["removed_files"] == []
    if failure == "fail_close":
        assert result["close_failures"]
        assert result["mutations"][0]["disposition_set"] is True
        assert result["mutations"][0]["handle_close_confirmed"] is False
    else:
        assert result["mutations"] == []
        assert "safe.bin" not in tree.api.disposed
    assert tree.api.closed.count("safe.bin") == 1


def test_child_handle_close_unconfirmed_prevents_every_disposition():
    tree = tree_for({"a-safe.bin": dict(directory=False),
                     "z-uncertain.bin": dict(directory=False, unsafe="uncertain_close")})
    result = fresh_scope._cleanup_current_fresh_tree(tree, entry_limit=32)
    assert result["complete"] is False
    assert result["mutations"] == []
    assert not tree.api.disposed
    assert result["close_failures"] == [dict(handle=73, winerror=6)]
    assert result["preflight_refusal"]["created_evidence"]["handle_returned"] is True


def test_recursive_children_do_not_spend_parent_sibling_capacity_twice():
    tree = tree_for({"A": dict(directory=True), "A/a.bin": dict(directory=False),
                     "A/b.bin": dict(directory=False), "B.bin": dict(directory=False)})
    result = fresh_scope._cleanup_current_fresh_tree(tree, entry_limit=4)
    assert tree.api.peak_open <= 4
    assert "B.bin" not in tree.api.opens
    assert "B.bin" in tree.api.nodes
    assert set(result["removed_files"]) == {"A/a.bin", "A/b.bin"}
    assert result["reservation_must_remain_charged"] is True



def test_many_unsafe_names_do_not_hide_an_independent_safe_sibling():
    tree = tree_for({"reparse-a": dict(directory=True, unsafe="reparse"),
                     "reparse-b": dict(directory=True, unsafe="reparse"),
                     "reparse-c": dict(directory=True, unsafe="reparse"),
                     "safe.cache": dict(directory=False, bytes=13)})
    result = fresh_scope._cleanup_current_fresh_tree(tree, entry_limit=2)
    assert tree.api.peak_open <= 2
    assert result["removed_files"] == ["safe.cache"]
    assert result["reservation_must_remain_charged"] is True
    assert len(result["retained"]) == 3
    assert tree.api.opens == ["safe.cache"]


def test_directory_enumeration_order_is_not_a_scope_identity():
    tree = tree_for({"a.cache": dict(directory=False), "b.cache": dict(directory=False)})
    tree.api.reverse_each_enumeration = True
    result = fresh_scope._cleanup_current_fresh_tree(tree, entry_limit=32)
    assert result["complete"] is True


@pytest.mark.parametrize("mutation", ["missing", "nonempty", "wrong_identity"])
def test_old_or_unproven_scope_refused_before_finalization(tmp_path, mutation):
    context = context_for(tmp_path)
    if mutation == "missing":
        del context.manifest["fresh_scope_provenance"]
    elif mutation == "nonempty":
        context.manifest["fresh_scope_provenance"]["initial_names"] = ["old.cache"]
    else:
        context.manifest["fresh_scope_provenance"]["root_identity"] = {}
    with pytest.raises(CachePolicyError, match="no old-scope adoption"):
        fresh_scope._fresh_provenance(context)


def test_actual_retirement_refusal_precedes_any_scope_open(tmp_path, monkeypatch):
    context = context_for(tmp_path)
    monkeypatch.setattr(fresh_scope, "resolve_original_service_context", lambda *a, **k: context)

    class RetentionBoundary:
        def __init__(self, context):
            pass

        def retention_eligibility(self, **arguments):
            assert arguments["verify_retirement"]({}, arguments["retirement"]) is False
            raise CachePolicyError("TESTONLY actual retirement refused")

    monkeypatch.setattr(fresh_scope, "ServiceLifecycle", RetentionBoundary)
    monkeypatch.setattr(fresh_scope, "WindowsNamespace", lambda *a, **k: pytest.fail("No scope open"))
    monkeypatch.setattr(fresh_scope, "CacheBudgetLedger", lambda *a, **k: pytest.fail("No allocation ledger"))
    with pytest.raises(CachePolicyError, match="actual retirement refused"):
        fresh_scope.finalize_original_fresh_scope(context.receipt, package_file=__file__,
            now_utc=UTC, retirement={}, verify_retirement=lambda *args: False)


@pytest.mark.parametrize("scenario", ["safe", "unsafe", "oversized_audit", "budget_audit_failure"])
def test_only_complete_scope_cleanup_closes_original_service_budget(tmp_path, monkeypatch, scenario):
    """TESTONLY callback/ledger boundaries; no production profile is accepted."""
    context = context_for(tmp_path)
    unsafe = scenario == "unsafe"
    tree = tree_for({"unknown.cache": dict(directory=False, bytes=7),
        **({"unsafe.cache": dict(directory=False, bytes=11, unsafe="hardlink")} if unsafe else {})})
    if scenario == "oversized_audit":
        context.entry_limit = 8192
        for index in range(3000):
            name = f"cache-{index:05d}-" + "x" * 190 + ".bin"
            tree.api.nodes[name] = dict(directory=False, bytes=1, id=index + 10)
    binding = dict(TESTONLY_original_service="exact")
    value = dict(retention_due=True, original_retirement_verifier_accepted=True,
                 observer=context.owner, TESTONLY=True)
    key = hashlib.sha256(_encoded(value)).hexdigest()
    name = "cache-l-" + TOKEN + "-eligibility-" + key + ".json"
    row = dict(format="darsia-cache-service-lifecycle-v1", kind="eligibility",
               key=key, binding=binding, value=value, actor=context.owner)
    raw = _encoded(row)
    eligibility = dict(value, receipt=dict(path=str(tmp_path / "control" / name),
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))

    class Control:
        def __init__(self):
            self.rows = {name: raw, "cache-s-" + TOKEN + ".json": _encoded(context.manifest)}

        def names(self):
            return list(self.rows)

        def read_once(self, name):
            return self.rows[name], {}

        def write_once(self, name, raw, maximum=1024**2):
            assert name not in self.rows
            assert len(raw) <= maximum
            if scenario == "budget_audit_failure" and name.endswith("-budget-closed.json"):
                raise OSError("TESTONLY audit publication failed after durable budget closure")
            self.rows[name] = raw

    control = Control()
    closed = []

    class Ledger:
        def __init__(self, policy, actual_context, **arguments):
            assert policy == context.policy
            assert actual_context.host == "TESTONLY"
            self.control, self.bound = control, dict(path=str(tmp_path / "control"))

        @contextmanager
        def locked(self, *, create_binding):
            assert create_binding is False
            yield self

        def require_reservation(self, reservation, owner, maximum):
            assert reservation == context.reservation
            assert owner == context.manifest["owner"]
            assert maximum == context.maximum

        def close(self, reservation, *, owner, verify_no_further_writes):
            assert not unsafe, "Unsafe retained data must not release the budget"
            assert verify_no_further_writes(dict(owner=context.manifest["owner"])) is True
            closed.append(reservation)
            control.write_once("cache-c-" + TOKEN + ".json", _encoded(dict(owner=owner)))

    class RetentionBoundary:
        def __init__(self, original):
            assert original is context
            self.binding = binding

        def retention_eligibility(self, **arguments):
            assert arguments["verify_retirement"]({}, arguments["retirement"]) is True
            assert arguments["verify_publication"]({}, {}) is True
            return deepcopy(eligibility)

    class Namespace:
        def __init__(self, path, identity):
            assert path == str(context.root.parent)

        def scope(self, token, *, expected):
            assert token == TOKEN and expected == context.manifest["root_identity"]
            return tree

        def close(self):
            pass

    monkeypatch.setattr(fresh_scope, "resolve_original_service_context", lambda *a, **k: context)
    monkeypatch.setattr(fresh_scope, "ServiceLifecycle", RetentionBoundary)
    monkeypatch.setattr(fresh_scope, "CacheBudgetLedger", Ledger)
    monkeypatch.setattr(fresh_scope, "WindowsNamespace", Namespace)
    def finalize():
        return fresh_scope.finalize_original_fresh_scope(context.receipt, package_file=__file__,
            now_utc=UTC, retirement={}, verify_retirement=lambda *args: True,
            verify_publication=lambda *args: True)

    if scenario == "budget_audit_failure":
        with pytest.raises(fresh_scope.FreshScopeAuditIncomplete) as caught:
            finalize()
        assert caught.value.finalization_state["service_budget_closed"] is True
        assert caught.value.finalization_state["budget_closure"]
        assert "cache-c-" + TOKEN + ".json" in control.rows
        result = caught.value.finalization_state["cleanup_result"]
    else:
        result = finalize()
    assert result["complete"] is (not unsafe)
    assert result["service_budget_closed"] is (not unsafe)
    assert len(closed) == (0 if unsafe else 1)
    assert "unknown.cache" not in tree.api.nodes
    receipt_name = "cache-f-" + TOKEN + "-result.json"
    receipt = json.loads(control.rows[receipt_name])
    payload = b"".join(base64.b64decode(json.loads(control.rows[Path(part["path"]).name])["payload_base64"])
                       for part in receipt["chunks"])
    assert len(payload) == receipt["logical_result_bytes"]
    assert hashlib.sha256(payload).hexdigest() == receipt["logical_result_sha256"]
    durable_result = json.loads(payload)
    assert durable_result["removed_files"] == result["removed_files"]
    assert all(len(raw) <= 1024**2 for raw in control.rows.values())
    if scenario == "oversized_audit":
        assert len(payload) > 1024**2
        assert len(receipt["chunks"]) > 1
        assert len(durable_result["removed_files"]) == 3001
        with pytest.raises(CachePolicyError, match="one MiB"):
            _encoded(durable_result)  # Regression: the old single write loses this outcome.
        assert result["service_budget_closed"] is True
        assert "cache-f-" + TOKEN + "-budget-closed.json" in control.rows
    if unsafe:
        assert "unsafe.cache" in tree.api.nodes
        assert result["reservation_must_remain_charged"] is True


def test_identified_writer_environment_includes_cupy_without_mutating_parent():
    stores = dict(cache=dict(root="C:/synthetic-owned-cache", quota_bytes=1,
                            profiles=dict(tests={})))
    parent = dict(TEMP="old-temp", CUPY_CACHE_DIR="old-cupy", HOME="durable-user-home")
    child = derive_cache_environment(stores, purpose="tests", token=TOKEN,
        scope_root="C:/synthetic-owned-cache/tests/current/" + TOKEN,
        parent_environment=parent)
    assert child["CUPY_CACHE_DIR"].endswith(TOKEN + "\\cupy")
    assert all(key in child for key in WRITER_ENVIRONMENT_KEYS)
    assert child["HOME"] == parent["HOME"]
    assert parent["TEMP"] == "old-temp"
    assert parent["CUPY_CACHE_DIR"] == "old-cupy"


@pytest.mark.skipif(os.name != "nt" or C.sizeof(C.c_void_p) != 8,
                    reason="Actual Windows local-NTFS capability is unqualified on this platform")
@pytest.mark.parametrize("extra_hardlink", [False, True])
def test_actual_ntfs_unknown_cache_and_unsafe_link_are_separated(tmp_path, extra_hardlink):
    parent_path = tmp_path / "pytest-owned-purpose"
    parent_path.mkdir()
    api = windows._API()
    handle = api.k.CreateFileW(str(parent_path), 0x80 | 1 | 0x20 | 0x100000,
                              1, None, 3, 0x02000000 | 0x00200000, None)
    if handle in (None, C.c_void_p(-1).value):
        raise OSError(C.get_last_error(), "Cannot admit pytest-owned directory")
    try:
        identity = api.identity(handle).document()
    finally:
        api.close(handle)
    namespace = windows.WindowsNamespace(parent_path, identity)
    try:
        tree = namespace.scope(TOKEN, create=True)
        try:
            root_identity = tree.directories[""]
            assert tree.api.names(tree.handles[""]) == []
        finally:
            tree.close()
    finally:
        namespace.close()
    scope = parent_path / TOKEN
    (scope / "unpredicted-library.cache").write_bytes(b"synthetic cache")
    if extra_hardlink:
        (scope / "unsafe.cache").write_bytes(b"preserve both links")
        os.link(scope / "unsafe.cache", tmp_path / "outside-link.cache")
    namespace = windows.WindowsNamespace(parent_path, identity)
    try:
        tree = namespace.scope(TOKEN, expected=root_identity)
        try:
            result = fresh_scope._cleanup_current_fresh_tree(tree, entry_limit=32)
        finally:
            tree.close()
    finally:
        namespace.close()
    assert not (scope / "unpredicted-library.cache").exists()
    assert result["complete"] is (not extra_hardlink)
    if extra_hardlink:
        assert (scope / "unsafe.cache").read_bytes() == b"preserve both links"
        assert (tmp_path / "outside-link.cache").read_bytes() == b"preserve both links"
        assert result["reservation_must_remain_charged"] is True
    else:
        assert not scope.exists()
