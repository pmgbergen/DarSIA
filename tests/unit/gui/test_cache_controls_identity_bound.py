"""Canonical Windows cache ownership/lifecycle tests; only pytest-owned NTFS paths.

The host-ledger constructor bypass is explicitly synthetic. It exercises actual
WindowsNamespace, immutable control I/O, mutex, reservation and cleanup methods,
not production CommonAppData installation, ACL/account sharing or real writers.
No DarSIA application import, handoff path, request environment or cache adoption.
A capability skip is unqualified, never a passing race/reparse guarantee.
"""
import ctypes as C
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback
from types import SimpleNamespace

import psutil
import pytest

from darsia_cache import control, windows
from darsia_cache.policy import CachePolicy, CachePolicyError
from ._cache_owned_retirement import OwnedRetirements
from darsia_cache.windows import (
    _API, WindowsNamespace, CacheCleanupIncomplete, CacheHandleCloseIncomplete,
    CacheWriteIncomplete,
)

pytestmark = pytest.mark.skipif(os.name != "nt" or C.sizeof(C.c_void_p) != 8,
    reason="Windows x64 local-NTFS ownership capability remains unqualified here")


def _evidence(root, name, document):
    folder = root / "test-evidence"
    folder.mkdir(exist_ok=True)
    with (folder / name).open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)


@pytest.fixture
def owned_retirements(tmp_path):
    """Only this test's spawned owners; close every duplicated handle on failure."""
    witnesses = OwnedRetirements()
    try:
        yield witnesses
    finally:
        try:
            witnesses.close()
        finally:
            _evidence(tmp_path, "native-owner-retirement.json", witnesses.document())


def _root_identity(path):
    # Test admission only: pytest created this exact isolated directory.
    api = _API()
    handle = api.k.CreateFileW(str(path), 0x80 | 0x1 | 0x20 | 0x100000,
                              0x1, None, 3, 0x02000000 | 0x00200000, None)
    if handle in (None, C.c_void_p(-1).value):
        raise OSError(C.get_last_error(), "Cannot admit isolated test directory")
    try:
        return api.identity(handle).document()
    finally:
        api.close(handle)


@pytest.fixture
def owned(tmp_path):
    root = tmp_path / "cache-purpose"
    root.mkdir()
    namespace = WindowsNamespace(root, _root_identity(root))
    tree = namespace.scope("a" * 32, create=True)
    tree.write_once("nested/own.bin", b"declared owned bytes", 64)
    frozen = tree.frozen()
    tree.close()
    reopened = namespace.scope("a" * 32, expected=frozen["root"])
    try:
        yield root, namespace, reopened, frozen
    finally:
        reopened.close()
        namespace.close()


def _synthetic_ledger(base, quota=4096):
    """Test-only constructor bypass; actual NTFS controls/mutex, no host install."""
    base = Path(base)
    for relative in ("cache/jobs", "cache/tests/current", "cache/clones/current",
                     "host-ledger", "receipts", "deliverables"):
        (base / relative).mkdir(parents=True, exist_ok=True)
    admitted = dict(host="synthetic-test-host", authorized_sids=["synthetic-test-account"],
        cache_identity=_root_identity(base / "cache"),
        host_ledger=dict(path=str(base / "host-ledger"), identity=_root_identity(base / "host-ledger")),
        receipts=dict(path=str(base / "receipts"), identity=_root_identity(base / "receipts")),
        purposes={p: dict(path=str(base / folder), identity=_root_identity(base / folder))
                  for p, folder in {"jobs": "cache/jobs", "tests": "cache/tests/current"}.items()})
    ledger = control.CacheBudgetLedger.__new__(control.CacheBudgetLedger)
    ledger.policy = CachePolicy(str(base / "cache"), quota, 48, {})
    ledger.capacity = SimpleNamespace(host=admitted["host"], account="synthetic-test-account",
                                      common_root=base / "host-ledger")
    ledger.admission, ledger.bound = admitted, admitted["host_ledger"]
    ledger.key = hashlib.sha256((str(base.absolute()).casefold() + "\0fullsuite-test-only").encode()).hexdigest()
    ledger.control, ledger.lock_active = None, False

    def measure(root):
        total, active, entries = 0, {}, 0
        for parent, dirs, names in os.walk(root, followlinks=False):
            for name in dirs + names:
                entries += 1
                assert entries <= 2048
                st = (Path(parent) / name).lstat()
                if getattr(st, "st_file_attributes", 0) & 1024:
                    raise CachePolicyError("Synthetic accounting never follows reparse entries")
            for name in names:
                total += (Path(parent) / name).lstat().st_size
        for path in (base / "host-ledger").glob("cache-r-*.json"):
            row = json.loads(path.read_bytes())
            token, written = row["token"], 0
            for scope in (base / "cache/jobs" / token, base / "cache/tests/current" / token):
                if scope.exists():
                    for parent, dirs, names in os.walk(scope, followlinks=False):
                        written += sum((Path(parent) / name).lstat().st_size for name in names)
            active[token] = written
        return dict(root=root, fresh=True, root_identity=admitted["cache_identity"],
            no_reparse_traversal=True, logical_bytes=total, active_scope_bytes=active,
            free_bytes=shutil.disk_usage(root).free, measured_utc=datetime.now(timezone.utc).isoformat())
    ledger.measure = measure
    return ledger


def _owner_now():
    return dict(pid=os.getpid(), process_create_time=str(psutil.Process().create_time()))


def _epoch_absent(owner):
    try:
        observed = Decimal(str(psutil.Process(owner["pid"]).create_time()))
        expected = Decimal(owner["process_create_time"])
        return observed.is_finite() and expected.is_finite() and abs(observed - expected) > Decimal("0.000002")
    except psutil.NoSuchProcess:
        return True
    except (psutil.AccessDenied, InvalidOperation, TypeError, ValueError, KeyError):
        return False  # Unknown is not absence, including an ambiguous epoch.


def _lifecycle_actor(base, token, outcome, result):
    owner, scratch = _owner_now(), None
    try:
        ledger = _synthetic_ledger(base)
        with ledger.locked():
            reservation = ledger.reserve(token, owner, 128)
            scratch = control.OwnedScratch(ledger.policy, "tests", token, owner, "synthetic-same-gate",
                {"diagnostic.txt": 64}, namespace_admission=ledger.admission,
                ledger=ledger, reservation=reservation)
        value = b"synthetic diagnostic only"
        scratch.write_bytes("diagnostic.txt", value)
        deliverable = Path(base) / "deliverables" / (token + ".txt")
        with deliverable.open("xb") as stream:
            stream.write(value)
        publication = dict(path=str(deliverable), sha256=hashlib.sha256(value).hexdigest(), size_bytes=len(value))
        utc = datetime.now(timezone.utc).isoformat()
        scratch.finish(outcome, utc=utc, publication=publication,
            traceback_text="literal traceback\nValueError: synthetic controlled failure" if outcome == "failed" else None)
        scratch.close_handles()
        result.put(dict(owner=owner, reservation=reservation, token=token, utc=utc,
                        publication=publication, outcome=outcome, complete=True))
    except BaseException:
        result.put(dict(owner=owner, error=traceback.format_exc()))
        raise
    finally:
        if scratch is not None:
            scratch.close_handles()


def _completed_actor(base, token, outcome, retirements):
    context = mp.get_context("spawn")
    result = context.Queue()
    actor = context.Process(target=_lifecycle_actor, args=(str(base), token, outcome, result))
    actor.start()
    try:
        witness = retirements.capture(actor)  # Own the exact kernel object now.
        row = result.get(timeout=45)
        assert witness.bind(row["owner"])
        actor.join(45)
        exitcode = actor.exitcode
        assert not actor.is_alive() and exitcode == 0 and "error" not in row
        actor.close()
        assert retirements.verify(row["owner"])
        return row
    finally:
        result.close()
        result.join_thread()
        if not actor._closed:
            actor.join(45)  # Failure cleanup of this fixture's child only.
            if not actor.is_alive():
                actor.close()


def _valid_publication(manifest, publication):
    # Only this tests/current/token fixture's exact external deliverable is read.
    assert manifest["purpose"] == "tests"
    expected = Path(manifest["root"]).parents[3] / "deliverables" / (manifest["token"] + ".txt")
    path = Path(publication["path"])
    if path != expected:
        return False
    raw = path.read_bytes()
    return len(raw) == publication["size_bytes"] and hashlib.sha256(raw).hexdigest() == publication["sha256"]


def _close_reservation(ledger, row, retirements):
    with ledger.locked():
        ledger.close(row["reservation"], owner=row["owner"],
            verify_no_further_writes=lambda original: original["owner"] == row["owner"]
                and retirements.verify(row["owner"]))


def _reopen(ledger, row):
    return control.OwnedScratch.reopen(ledger.policy, row["token"],
        namespace_admission=ledger.admission, ledger=ledger)


def _reserve_actor(base, token, barrier, result):
    owner = _owner_now()
    try:
        ledger = _synthetic_ledger(base, quota=128)
        barrier.wait(20)
        try:
            with ledger.locked():
                reservation = ledger.reserve(token, owner, 80)
            result.put(dict(admitted=True, reservation=reservation, owner=owner))
        except CachePolicyError as error:
            result.put(dict(admitted=False, refusal=str(error), owner=owner))
    except BaseException:
        result.put(dict(error=traceback.format_exc(), owner=owner))
        raise


def _retirement_gate_actor(gate, result):
    result.put(_owner_now())
    if not gate.wait(45):
        raise RuntimeError("Test parent did not release its own gated child")


def test_owned_native_retirement_refuses_live_unknown_changed_and_closed_identity(
        owned_retirements, monkeypatch):
    context = mp.get_context("spawn")
    gate, result = context.Event(), context.Queue()
    actor = context.Process(target=_retirement_gate_actor, args=(gate, result))
    actor.start()
    try:
        witness = owned_retirements.capture(actor)
        owner = result.get(timeout=45)
        assert owner["pid"] == actor.pid and witness.bind(owner)
        assert owned_retirements.verify(owner) is False
        assert witness.events[-1]["native"]["wait_zero_result"] == 258
        assert _epoch_absent(owner) is False  # A live exact epoch is not absent.
        assert owned_retirements.verify(None) is False
        assert witness.retired({**owner, "pid": owner["pid"] + 1}) is False
        assert witness.retired({**owner, "process_create_time": str(
            Decimal(owner["process_create_time"]) + 1)}) is False

        gate.set()  # Natural exit, no terminate, sleep, or absence polling.
        actor.join(45)
        exitcode = actor.exitcode
        assert not actor.is_alive() and exitcode == 0
        actor.close()
        assert owned_retirements.verify(owner) is True
        native = witness.events[-1]["native"]
        assert native["wait_zero_result"] == 0 and native["exitcode"] == 0
        assert native["creation_filetime_100ns"] == witness.birth["creation_filetime_100ns"]
        assert native["exit_filetime_100ns"] >= native["creation_filetime_100ns"] > 0
        assert owned_retirements.verified_retirement({"owner": owner}, {"owner": owner}) is True
        assert owned_retirements.verified_retirement(
            {"owner": {**owner, "pid": owner["pid"] + 1}}, {"owner": owner}) is False

        actual_snapshot = witness.api.snapshot
        def changed_creation(handle):
            state = actual_snapshot(handle)
            state["creation_filetime_100ns"] += 1
            return state
        with monkeypatch.context() as patch:
            patch.setattr(witness.api, "snapshot", changed_creation)
            assert owned_retirements.verify(owner) is False  # Exact native pin, even 100ns.
        def denied(handle):
            raise PermissionError("Synthetic unknown native status")
        with monkeypatch.context() as patch:
            patch.setattr(witness.api, "snapshot", denied)
            assert owned_retirements.verify(owner) is False
        witness.close()
        assert owned_retirements.verify(owner) is False
    finally:
        gate.set()
        if not actor._closed:
            actor.join(45)
            if not actor.is_alive():
                actor.close()
        result.close()
        result.join_thread()


@pytest.mark.parametrize("observation, absent", [
    ("same", False), ("access-denied", False), ("invalid", False),
    ("absent", True), ("different", True),
])
def test_strict_epoch_absence_keeps_same_epoch_and_unknown_refused(monkeypatch, observation, absent):
    owner = dict(pid=123, process_create_time="100.0")
    def observed_process(pid):
        assert pid == owner["pid"]
        if observation == "access-denied":
            raise psutil.AccessDenied(pid)
        if observation == "absent":
            raise psutil.NoSuchProcess(pid)
        value = {"same": 100.0, "different": 101.0, "invalid": float("nan")}[observation]
        return SimpleNamespace(create_time=lambda: value)
    monkeypatch.setattr(psutil, "Process", observed_process)
    assert _epoch_absent(owner) is absent


def test_only_declared_owned_objects_removed(owned):
    root, namespace, tree, frozen = owned
    result = tree.cleanup(frozen)
    assert result["complete"] is True
    assert result["removed_logical_bytes"] == len(b"declared owned bytes")
    assert result["physical_reclaimed_bytes"] is None
    assert root.is_dir()
    assert not (root / ("a" * 32)).exists()


def test_same_file_id_modified_content_is_preserved(owned):
    root, namespace, tree, frozen = owned
    target = root / ("a" * 32) / "nested/own.bin"
    target.write_bytes(b"foreign changed content and length")
    with pytest.raises(CacheCleanupIncomplete):
        tree.cleanup(frozen)
    assert target.read_bytes() == b"foreign changed content and length"


def test_foreign_hardlink_preserved_before_first_removal(owned, tmp_path):
    root, namespace, tree, frozen = owned
    target = root / ("a" * 32) / "nested/own.bin"
    foreign = tmp_path / "foreign-link.bin"
    # Existing-foreign-object fixture must precede exclusive namespace admission.
    tree.close()
    namespace.close()
    os.link(target, foreign)
    namespace = WindowsNamespace(root, _root_identity(root))
    tree = namespace.scope("a" * 32, expected=frozen["root"])
    try:
        with pytest.raises(CacheCleanupIncomplete) as refusal:
            tree.cleanup(frozen)
        assert refusal.value.removed_owned == ()
        assert foreign.read_bytes() == b"declared owned bytes"
        assert target.exists()
    finally:
        tree.close()
        namespace.close()


def test_parent_rename_attempt_after_preflight_cannot_redirect(owned, tmp_path, monkeypatch):
    root, namespace, tree, frozen = owned
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    sentinel = foreign / "own.bin"
    sentinel.write_bytes(b"must survive")
    scope = root / ("a" * 32)
    ready, attempted = threading.Event(), threading.Event()
    outcomes = []

    def attacker():
        assert ready.wait(10)
        try:
            os.rename(scope / "nested", scope / "moved")
            # A successful unexpected rename must not grant authority to a
            # replacement; the original relative handles remain authoritative.
            os.symlink(foreign, scope / "nested", target_is_directory=True)
            outcomes.append("renamed")
        except OSError:
            outcomes.append("refused")
        finally:
            attempted.set()

    original = tree.api.dispose

    def pause_before_disposition(handle):
        ready.set()
        assert attempted.wait(10)
        return original(handle)

    monkeypatch.setattr(tree.api, "dispose", pause_before_disposition)
    thread = threading.Thread(target=attacker)
    thread.start()
    try:
        try:
            tree.cleanup(frozen)
        except CacheCleanupIncomplete:
            pass
    finally:
        thread.join(10)
    assert not thread.is_alive()
    assert outcomes == ["refused"]  # deny-delete directory sharing is required
    assert sentinel.read_bytes() == b"must survive"


def test_new_foreign_entry_after_preflight_stays_and_reports_incomplete(owned, monkeypatch):
    root, namespace, tree, frozen = owned
    foreign = root / ("a" * 32) / "foreign.bin"
    original = tree.api.dispose
    inserted, refused = [], []

    def insert_before_first_disposition(handle):
        if not inserted:
            try:
                foreign.write_bytes(b"not owned")
                inserted.append(True)
            except OSError:
                refused.append(True)
        return original(handle)

    monkeypatch.setattr(tree.api, "dispose", insert_before_first_disposition)
    problem = None
    try:
        tree.cleanup(frozen)
    except CacheCleanupIncomplete as error:
        problem = error
    if inserted:
        assert problem is not None
        assert foreign.read_bytes() == b"not owned"
        assert foreign.parent.exists()
    else:
        assert refused and not foreign.exists()
        assert problem is None


@pytest.mark.parametrize("kind", ["symlink", "junction"])
def test_replaced_directory_reparse_never_touches_foreign_target(owned, tmp_path, kind):
    root, namespace, tree, frozen = owned
    scope = root / ("a" * 32)
    # Prepare a pre-existing foreign reparse before any exclusive cleanup handle.
    tree.close()
    namespace.close()
    os.rename(scope / "nested", scope / "original-nested")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    sentinel = foreign / "own.bin"
    sentinel.write_bytes(b"foreign sentinel")
    if kind == "symlink":
        try:
            os.symlink(foreign, scope / "nested", target_is_directory=True)
        except OSError as error:
            pytest.skip("Symlink capability unqualified: " + str(error))
    else:
        result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J",
                                 str(scope / "nested"), str(foreign)], capture_output=True, timeout=10)
        if result.returncode:
            pytest.skip("Junction fixture capability unqualified")
    namespace = WindowsNamespace(root, _root_identity(root))
    tree = namespace.scope("a" * 32, expected=frozen["root"])
    try:
        with pytest.raises(CacheCleanupIncomplete) as refusal:
            tree.cleanup(frozen)
        assert refusal.value.removed_owned == ()
        assert sentinel.read_bytes() == b"foreign sentinel"
        assert (scope / "original-nested/own.bin").read_bytes() == b"declared owned bytes"
    finally:
        tree.close()
        namespace.close()


def test_failed_close_cannot_claim_file_removed(owned, monkeypatch):
    root, namespace, tree, frozen = owned
    original = tree.api.close
    first = []

    def reported_failure_after_close(handle):
        original(handle)
        if not first:
            first.append(True)
            raise CacheHandleCloseIncomplete([dict(handle=None, winerror="injected reported failure")])

    # Install only at the first disposition; preflight read-close behavior is
    # unchanged. The assertion concerns truthfulness, never a claimed OS pass.
    dispose = tree.api.dispose

    def arm_close_failure(handle):
        result = dispose(handle)
        monkeypatch.setattr(tree.api, "close", reported_failure_after_close)
        return result

    monkeypatch.setattr(tree.api, "dispose", arm_close_failure)
    with pytest.raises(CacheCleanupIncomplete) as incomplete:
        tree.cleanup(frozen)
    assert incomplete.value.removed_owned == ()
    assert incomplete.value.close_failures
    assert incomplete.value.mutations[0]["handle_close_confirmed"] is False


def test_partial_cleanup_records_already_removed_directories(owned, monkeypatch):
    root, namespace, tree, frozen = owned
    original = tree.api.dispose
    dispositions = []

    def stop_before_root(handle):
        dispositions.append(True)
        if len(dispositions) == 3:  # own.bin, nested, then scope root
            raise OSError("injected root-disposition refusal")
        return original(handle)

    monkeypatch.setattr(tree.api, "dispose", stop_before_root)
    with pytest.raises(CacheCleanupIncomplete) as incomplete:
        tree.cleanup(frozen)
    assert any(m["directory"] and m["path"] == "nested" and m["handle_close_confirmed"]
               for m in incomplete.value.mutations)


def test_create_succeeded_but_identity_failed_is_not_unrecorded(owned, monkeypatch):
    root, namespace, reopened, frozen = owned
    reopened.close()
    tree = namespace.scope("c" * 32, create=True)
    original = tree.api.identity

    def fail_new_file_identity(handle):
        identity = original(handle)
        if not identity.directory:
            raise OSError("injected post-create identity failure")
        return identity

    monkeypatch.setattr(tree.api, "identity", fail_new_file_identity)
    try:
        with pytest.raises(CacheWriteIncomplete) as incomplete:
            tree.write_once("new.bin", b"bytes", 64)
        assert incomplete.value.created_evidence["new_object_created"] is True
        assert "new.bin" in tree.frozen()["unresolved_creates"]
        assert "new.bin" not in tree.files
    finally:
        tree.close()


def test_windows_x64_abi_layout_and_handle_roundtrip(tmp_path):
    expected = {"_UNICODE_STRING": 16, "_OBJECT_ATTRIBUTES": 48, "_IO_STATUS_BLOCK": 16,
                "_FILE_ID_INFO": 24, "_FILE_ATTRIBUTE_TAG_INFO": 8,
                "_FILE_STANDARD_INFO": 24, "_FILE_BASIC_INFO": 40,
                "_FILE_ID_EXTD_DIR_INFO": 96}
    actual = {name: C.sizeof(getattr(windows, name)) for name in expected}
    assert actual == expected
    assert windows._FILE_ID_EXTD_DIR_INFO.FileName.offset == 88
    assert windows._OBJECT_ATTRIBUTES.RootDirectory.offset == 8
    assert C.sizeof(windows.W.WCHAR) == 2
    identity = _root_identity(tmp_path)
    assert identity["directory"] and len(identity["file_id_128"]) == 32
    namespace = WindowsNamespace(tmp_path, identity)
    try:
        assert namespace.identity.document() == identity
        assert namespace.names() == []
    finally:
        namespace.close()
    _evidence(tmp_path,"REAL_WINDOWS_ABI.json", dict(structure_sizes=actual, file_name_offset=88,
        object_root_handle_offset=8, windows_version=tuple(sys.getwindowsversion()),
        real_handle_identity=identity, real_ntfs_open_and_close=True))


def test_real_immutable_control_write_read_and_collision(tmp_path):
    namespace = WindowsNamespace(tmp_path, _root_identity(tmp_path))
    try:
        identity = namespace.write_once("immutable.json", b'{"synthetic":true}\n', 128)
        raw, read_identity = namespace.read_once("immutable.json", 128)
        assert raw == b'{"synthetic":true}\n' and read_identity == identity
        with pytest.raises(windows.CacheObjectIncomplete):
            namespace.write_once("immutable.json", b"replacement", 128)
        assert namespace.read_once("immutable.json", 128)[0] == raw
    finally:
        namespace.close()


def test_real_exclusive_directory_compatibility_is_observed(tmp_path):
    namespace = WindowsNamespace(tmp_path, _root_identity(tmp_path))
    tree = namespace.scope("d" * 32, create=True)
    target = tmp_path / ("d" * 32)
    observations = {}
    try:
        tree.write_once("own.bin", b"small owned bytes", 64)
        assert tree.api.names(tree.handles[""]) == ["own.bin"]
        for name, action in {
            "Path.iterdir": lambda: [p.name for p in target.iterdir()],
            "os.scandir": lambda: [p.name for p in os.scandir(target)],
            "Path.read_bytes_child": lambda: (target / "own.bin").read_bytes().decode("ascii"),
        }.items():
            try:
                observations[name] = dict(allowed=True, observed=action())
            except OSError as error:
                observations[name] = dict(allowed=False, winerror=error.winerror, error=str(error))
        api = tree.api
        handle = api.k.CreateFileW(str(target), 0x80 | 0x1 | 0x20 | 0x100000,
                                  0x1 | 0x2 | 0x4, None, 3, 0x02000000 | 0x00200000, None)
        if handle not in (None, C.c_void_p(-1).value):
            api.close(handle)
            pytest.fail("Mutable directory was independently opened despite exclusive sharing")
        observations["CreateFileW_independent_directory_read"] = dict(allowed=False, winerror=C.get_last_error())
        _evidence(tmp_path,"REAL_EXCLUSIVE_DIRECTORY_COMPATIBILITY.json", dict(observations=observations,
            general_external_writer_compatibility_qualified=False,
            internal_relative_handle_enumeration=True, no_sharing_relaxation=True))
    finally:
        tree.close()
        namespace.close()


def test_same_parent_regular_file_replacement_preserves_both_objects(owned):
    root, namespace, tree, frozen = owned
    folder = root / ("a" * 32) / "nested"
    original = folder / "own.bin"
    retained = folder / "original-retained.bin"
    original.rename(retained)
    original.write_bytes(b"declared owned bytes")
    parent, _ = tree.api.child(tree.handles[""], "nested", directory=True)
    try:
        handle, replacement = tree.api.child(parent, "own.bin", directory=False)
        tree.api.close(handle)
    finally:
        tree.api.close(parent)
    assert replacement.file_id_128 != frozen["files"]["nested/own.bin"]["file_id_128"]
    with pytest.raises(CacheCleanupIncomplete) as incomplete:
        tree.cleanup(frozen)
    assert incomplete.value.removed_owned == ()
    assert original.read_bytes() == retained.read_bytes() == b"declared owned bytes"


def test_real_two_process_quota_reservations_count_remaining_once(tmp_path, owned_retirements):
    base = tmp_path / "quota"
    ledger = _synthetic_ledger(base, quota=128)
    context = mp.get_context("spawn")
    barrier, result = context.Barrier(2), context.Queue()
    actors = [context.Process(target=_reserve_actor, args=(str(base), token, barrier, result))
              for token in ("1" * 32, "2" * 32)]
    witnesses = {}
    try:
        for actor in actors:
            actor.start()
            witnesses[actor.pid] = owned_retirements.capture(actor)
        records = [result.get(timeout=45) for _ in actors]
        for actor in actors:
            row = next(row for row in records if row["owner"]["pid"] == actor.pid)
            assert witnesses[actor.pid].bind(row["owner"])
            actor.join(45)
            exitcode = actor.exitcode
            assert not actor.is_alive() and exitcode == 0
            actor.close()
        assert all("error" not in row and owned_retirements.verify(row["owner"])
                   for row in records)
        assert sum(row["admitted"] for row in records) == 1
        winner = next(row for row in records if row["admitted"])
        with ledger.locked():
            assert ledger.remaining({winner["reservation"]["token"]: 0}) == 80
        _close_reservation(ledger, winner, owned_retirements)
        with ledger.locked():
            assert ledger.remaining({}) == 0
    finally:
        result.close()
        result.join_thread()
        for actor in actors:
            if actor._popen is not None and not actor._closed:
                actor.join(45)
                if not actor.is_alive():
                    actor.close()


def test_success_requires_publication_and_retirement_preserves_durable_outputs_and_receipts(tmp_path, owned_retirements):
    base = tmp_path / "success"
    row = _completed_actor(base, "3" * 32, "success", owned_retirements)
    ledger = _synthetic_ledger(base)
    _close_reservation(ledger, row, owned_retirements)
    item = _reopen(ledger, row)
    output = Path(row["publication"]["path"])
    expected = output.read_bytes()
    try:
        with pytest.raises(CachePolicyError, match="retired"):
            item.cleanup(now_utc=row["utc"], retirement=row,
                verify_retirement=lambda manifest, proof: False, verify_publication=_valid_publication)
        with pytest.raises(CachePolicyError, match="publication"):
            item.cleanup(now_utc=row["utc"], retirement=row,
                verify_retirement=owned_retirements.verified_retirement, verify_publication=lambda manifest, proof: False)
        assert item.root.exists() and output.read_bytes() == expected
        cleaned = item.cleanup(now_utc=row["utc"], retirement=row,
            verify_retirement=owned_retirements.verified_retirement, verify_publication=_valid_publication)
        assert cleaned["complete"] and not item.root.exists()
        assert cleaned["removed_logical_bytes"] == len(b"synthetic diagnostic only")
    finally:
        item.close_handles()
    assert output.read_bytes() == expected
    for stage in ("declared", "frozen", "outcome", "cleanup"):
        assert (base / "receipts" / (row["token"] + "." + stage + ".json")).is_file()


@pytest.mark.parametrize("trigger", ["48h", "verified-success"])
def test_failed_scope_retention_and_verified_same_gate_success(tmp_path, trigger, owned_retirements):
    base = tmp_path / "retention"
    row = _completed_actor(base, "4" * 32, "failed", owned_retirements)
    ledger = _synthetic_ledger(base)
    _close_reservation(ledger, row, owned_retirements)
    when = datetime.fromisoformat(row["utc"])
    item = _reopen(ledger, row)
    try:
        with pytest.raises(CachePolicyError, match="retained"):
            item.cleanup(now_utc=(when + timedelta(hours=47)).isoformat(), retirement=row,
                verify_retirement=owned_retirements.verified_retirement, verify_publication=_valid_publication)
        assert (item.root / "diagnostic.txt").read_bytes() == b"synthetic diagnostic only"
    finally:
        item.close_handles()
    outcome_path = base / "receipts" / (row["token"] + ".outcome.json")
    outcome_before = outcome_path.read_bytes()
    assert "literal traceback" in json.loads(outcome_before)["traceback"]
    if trigger == "48h":
        item = _reopen(ledger, row)
        try:
            with pytest.raises(CachePolicyError, match="retired"):
                item.cleanup(now_utc=(when + timedelta(hours=48)).isoformat(), retirement=row,
                    verify_retirement=lambda manifest, proof: False, verify_publication=_valid_publication)
            result = item.cleanup(now_utc=(when + timedelta(hours=48)).isoformat(), retirement=row,
                verify_retirement=owned_retirements.verified_retirement, verify_publication=_valid_publication)
        finally:
            item.close_handles()
    else:
        success = _completed_actor(base, "5" * 32, "success", owned_retirements)
        _close_reservation(ledger, success, owned_retirements)
        now = max(datetime.fromisoformat(success["utc"]), when) + timedelta(seconds=1)
        proof = dict(complete=True, logical_gate="synthetic-same-gate", utc=success["utc"], actual=success)
        item = _reopen(ledger, row)
        try:
            with pytest.raises(CachePolicyError, match="verified"):
                item.cleanup(now_utc=now.isoformat(), retirement=row,
                    verify_retirement=owned_retirements.verified_retirement, verify_publication=_valid_publication,
                    same_gate_success=proof)
            with pytest.raises(CachePolicyError):
                item.cleanup(now_utc=now.isoformat(), retirement=row,
                    verify_retirement=owned_retirements.verified_retirement, verify_publication=_valid_publication,
                    same_gate_success={**proof, "logical_gate": "unrelated"},
                    verify_same_gate_success=lambda manifest, proposed: True)
            result = item.cleanup(now_utc=now.isoformat(), retirement=row,
                verify_retirement=owned_retirements.verified_retirement, verify_publication=_valid_publication,
                same_gate_success=proof, verify_same_gate_success=lambda manifest, proposed:
                    proposed["actual"] == success and owned_retirements.verify(success["owner"])
                    and _valid_publication({**manifest, "token": success["token"],
                        "root": str(base / "cache/tests/current" / success["token"])}, success["publication"]))
        finally:
            item.close_handles()
    assert result["complete"] and not item.root.exists()
    assert outcome_path.read_bytes() == outcome_before
    assert Path(row["publication"]["path"]).read_bytes() == b"synthetic diagnostic only"


def test_foreign_file_refuses_before_any_owned_removal(tmp_path, owned_retirements):
    base = tmp_path / "foreign"
    row = _completed_actor(base, "6" * 32, "success", owned_retirements)
    ledger = _synthetic_ledger(base)
    _close_reservation(ledger, row, owned_retirements)
    scope = base / "cache/tests/current" / row["token"]
    foreign = scope / "foreign.bin"
    foreign.write_bytes(b"preserve foreign")
    item = _reopen(ledger, row)
    try:
        with pytest.raises(CacheCleanupIncomplete) as refusal:
            item.cleanup(now_utc=row["utc"], retirement=row,
                verify_retirement=owned_retirements.verified_retirement, verify_publication=_valid_publication)
        assert refusal.value.removed_owned == ()
        assert foreign.read_bytes() == b"preserve foreign"
        assert (scope / "diagnostic.txt").read_bytes() == b"synthetic diagnostic only"
    finally:
        item.close_handles()


def test_undeclared_and_oversized_writes_refuse_before_file_creation(tmp_path):
    ledger = _synthetic_ledger(tmp_path / "writes")
    owner, token = _owner_now(), "7" * 32
    with ledger.locked():
        reservation = ledger.reserve(token, owner, 10)
        item = control.OwnedScratch(ledger.policy, "tests", token, owner, "bounded-writer",
            {"scratch/a": 10}, namespace_admission=ledger.admission, ledger=ledger, reservation=reservation)
    try:
        for name, payload in (("foreign", b"x"), ("scratch/a", b"x" * 11)):
            with pytest.raises(CachePolicyError):
                item.write_bytes(name, payload)
            assert not (item.root / name).exists()
        assert item.tree.files == {}
    finally:
        item.close_handles()  # No success/retirement/cleanup is fabricated.


def test_sequential_gates_reuse_test_parent_with_fresh_token_children(tmp_path, owned_retirements):
    base = tmp_path / "reuse"
    ledger = _synthetic_ledger(base)
    parent = base / "cache/tests/current"
    before = _root_identity(parent)
    paths = []
    for token in ("8" * 32, "9" * 32):
        row = _completed_actor(base, token, "success", owned_retirements)
        _close_reservation(ledger, row, owned_retirements)
        item = _reopen(ledger, row)
        try:
            assert item.root == parent / token and item.root.parent == parent
            assert all(not old.exists() for old in paths)
            result = item.cleanup(now_utc=row["utc"], retirement=row,
                verify_retirement=owned_retirements.verified_retirement, verify_publication=_valid_publication)
            assert result["complete"] and not item.root.exists()
            paths.append(item.root)
        finally:
            item.close_handles()
        assert Path(row["publication"]["path"]).read_bytes() == b"synthetic diagnostic only"
    after = _root_identity(parent)
    assert before["file_id_128"] == after["file_id_128"] and before["volume_serial"] == after["volume_serial"]
    assert paths[0] != paths[1] and parent.is_dir() and list(parent.iterdir()) == []
    assert len(list((base / "receipts").glob("*.cleanup.json"))) == 2


def test_old_store_policy_and_alternate_host_authority_refuse_without_allocation(tmp_path):
    from darsia_cache.policy import cache_settings
    with pytest.raises(CachePolicyError):
        cache_settings({"stores": {}}, required=True)
    ledger = _synthetic_ledger(tmp_path / "split")
    before = list((tmp_path / "split/host-ledger").iterdir())
    with pytest.raises(CachePolicyError):
        control.CacheBudgetLedger(ledger.policy, ledger.capacity, measurement_provider=ledger.measure,
            namespace_admission=ledger.admission, verify_host_admission=lambda admitted, actual: True)
    assert list((tmp_path / "split/host-ledger").iterdir()) == before == []

