"""Fault injection and actual Windows handles on tiny private directories only."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import job_queue as jobs
from darsia.fluidflower.pipeline import publication_handles as boundary
from tests.server_lease_clock_fixture import server_lease_clock


def sharing_error():
    error = PermissionError("fixture sharing denial")
    error.winerror = 32
    return error


@pytest.fixture
def unavailable(monkeypatch):
    value = dict(status="unavailable", pids=None, error="fixture native query unavailable",
                 diagnostic_handle_closed=None, proves_no_handles=False)
    monkeypatch.setattr(boundary, "directory_handle_holders", lambda path: dict(value, path=str(path)))


def test_five_physical_attempts_have_four_two_second_backoffs(tmp_path, monkeypatch, unavailable):
    source = tmp_path / "lease"; source.mkdir()
    target = tmp_path / "publication"
    calls, waits, records = [], [], []
    def move(a, b):
        calls.append((a, b))
        if len(calls) < 5:
            raise sharing_error()
        os.rename(a, b)
    monkeypatch.setattr(boundary.time, "sleep", waits.append)
    boundary.ordinary_publication_move(source, target, move=move, assert_owner=lambda: None,
        record=lambda *row: records.append(row))
    assert len(calls) == 5 and waits == [2, 2, 2, 2]
    assert [(i, phase) for i, phase, _ in records] == [
        (i, phase) for i in range(1, 6) for phase in ("before", "after")]
    assert records[-1][2]["result"] == "moved" and target.is_dir()


def test_fifth_failure_remains_primary_without_sixth_move(tmp_path, monkeypatch, unavailable):
    errors, waits, records = [], [], []
    def move(*args):
        error = sharing_error(); errors.append(error); raise error
    monkeypatch.setattr(boundary.time, "sleep", waits.append)
    with pytest.raises(PermissionError) as caught:
        boundary.ordinary_publication_move(tmp_path / "lease", tmp_path / "target",
            move=move, assert_owner=lambda: None, record=lambda *row: records.append(row))
    assert caught.value is errors[4] and len(errors) == 5 and waits == [2] * 4
    assert records[-1][:2] == (5, "after") and records[-1][2]["result"] == "failed"


def test_existing_destination_refuses_without_query_move_or_sleep(tmp_path, monkeypatch):
    target = tmp_path / "target"; target.write_bytes(b"original")
    def unexpected(*args): raise AssertionError("existing destination must refuse first")
    monkeypatch.setattr(boundary, "directory_handle_holders", unexpected)
    monkeypatch.setattr(boundary.time, "sleep", unexpected)
    with pytest.raises(FileExistsError):
        boundary.ordinary_publication_move(tmp_path / "lease", target, move=unexpected,
            assert_owner=lambda: None, record=unexpected)
    assert target.read_bytes() == b"original"


def test_lost_owner_stops_before_another_physical_attempt(tmp_path, monkeypatch, unavailable):
    owners, calls, waits = [], [], []
    def owner():
        owners.append(1)
        if len(owners) == 3: raise jobs.LeaseLost("new owner")
    def move(*args): calls.append(1); raise sharing_error()
    monkeypatch.setattr(boundary.time, "sleep", waits.append)
    with pytest.raises(jobs.LeaseLost):
        boundary.ordinary_publication_move(tmp_path / "lease", tmp_path / "target",
            move=move, assert_owner=owner, record=lambda *row: None)
    assert calls == [1] and waits == [2]


def test_nonsharing_error_has_no_backoff(tmp_path, monkeypatch, unavailable):
    calls, waits = [], []
    error = OSError("unsupported move")
    def move(*args): calls.append(1); raise error
    monkeypatch.setattr(boundary.time, "sleep", waits.append)
    with pytest.raises(OSError) as caught:
        boundary.ordinary_publication_move(tmp_path / "lease", tmp_path / "target",
            move=move, assert_owner=lambda: None, record=lambda *row: None)
    assert caught.value is error and calls == [1] and waits == []


def test_failed_diagnostic_log_cannot_mask_original_move_error(tmp_path, monkeypatch, unavailable):
    error = sharing_error(); calls = []
    def move(*args): calls.append(1); raise error
    def record(attempt, phase, value):
        if phase == "after": raise OSError("diagnostic storage failed")
    with pytest.raises(PermissionError) as caught:
        boundary.ordinary_publication_move(tmp_path / "lease", tmp_path / "target",
            move=move, assert_owner=lambda: None, record=record)
    assert caught.value is error and calls == [1]
    assert "diagnostic storage failed" in " ".join(error.__notes__)


def test_unavailable_query_is_literal_not_empty_holder_proof(tmp_path, unavailable):
    records = []
    boundary.ordinary_publication_move(tmp_path / "lease", tmp_path / "target",
        move=lambda *args: None, assert_owner=lambda: None, record=lambda *row: records.append(row))
    value = records[0][2]["handle_holders"]
    assert value["status"] == "unavailable" and value["pids"] is None
    assert value["error"] and value["proves_no_handles"] is False


def test_after_log_failure_reports_that_move_succeeded(tmp_path, monkeypatch, unavailable):
    calls = []
    error = OSError("after-log storage failed")
    def move(*args):
        calls.append(1)
        if len(calls) == 1: raise sharing_error()
    def record(attempt, phase, value):
        if value.get("result") == "moved": raise error
    monkeypatch.setattr(boundary.time, "sleep", lambda seconds: None)
    with pytest.raises(OSError) as caught:
        boundary.ordinary_publication_move(tmp_path / "lease", tmp_path / "target",
            move=move, assert_owner=lambda: None, record=record)
    assert caught.value is error and calls == [1, 1]
    assert "physical publication move succeeded" in " ".join(error.__notes__)


def test_owned_query_handle_close_failure_refuses_move(tmp_path, monkeypatch):
    monkeypatch.setattr(boundary, "directory_handle_holders", lambda p:
        dict(status="unavailable", pids=None, diagnostic_handle_closed=False, close_error="real close error"))
    calls = []
    with pytest.raises(RuntimeError, match="did not close"):
        boundary.ordinary_publication_move(tmp_path / "lease", tmp_path / "target",
            move=lambda *args: calls.append(1), assert_owner=lambda: None, record=lambda *row: None)
    assert calls == []


def test_prepared_success_is_not_rewritten_or_moved_by_failure(tmp_path, monkeypatch, server_lease_clock):
    shared = jobs.JobQueue(tmp_path, server_clock=server_lease_clock.clock)
    shared.enqueue(dict(run="sample", commit="1" * 40, calibration_sha256="2" * 64,
        configuration=dict(path=r"\\server\share\config.json", sha256="3" * 64),
        precision="float32", backend="any"))
    claim = shared.claim("machine", "cpu", dict(pid=123, process_create_time="1.0"))
    decisions = []
    def blocked(*args, **kwargs): decisions.append(1); raise sharing_error()
    monkeypatch.setattr(jobs, "decide", blocked)  # The ownership decision fails.
    with pytest.raises(PermissionError):
        shared.complete(claim, dict(complete=True, run="sample", worker_references_closed=True))
    publication = claim.lease / "publication.json"
    original = (publication.read_bytes(), publication.stat().st_mtime_ns)
    value = shared.fail(claim, "Original PermissionError: publication decision failed", {})
    assert value["complete"] is False and decisions == [1]
    assert (publication.read_bytes(), publication.stat().st_mtime_ns) == original
    assert jobs.read_json(claim.output / "failed.json")["traceback"].startswith("Original PermissionError")
    assert not (tmp_path / "failed/sample.json").exists() and not (tmp_path / "done/sample.json").exists()
    assert claim.lease.is_dir() and not (tmp_path / "publishing/sample").exists()
    assert not list((tmp_path / "decisions").iterdir())
    log = next((tmp_path / "logs").glob("*failure_after_prepared_success.json"))
    assert jobs.read_json(log)["no_publication_rewrite"] is True


@pytest.fixture
def native_directory_reader(tmp_path):
    if os.name != "nt": pytest.skip("Actual Windows directory handles require Windows")
    import ctypes
    from ctypes import wintypes
    path = tmp_path / "lease"; path.mkdir()
    # The venv executable is a redirector on this runtime. This stdlib-only
    # child uses the same actual base interpreter, so Popen owns the real PID.
    executable = Path(sys._base_executable).resolve(strict=True)
    assert executable.parent == Path(sys.base_prefix).resolve(strict=True)
    script = r'''
import ctypes,json,os,sys
from ctypes import wintypes
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
k.CreateFileW.restype=wintypes.HANDLE
k.CloseHandle.argtypes=[wintypes.HANDLE];k.CloseHandle.restype=wintypes.BOOL
k.GetCurrentProcess.restype=wintypes.HANDLE
k.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
k.GetProcessTimes.restype=wintypes.BOOL
times=[wintypes.FILETIME() for _ in range(4)]
if not k.GetProcessTimes(k.GetCurrentProcess(),*[ctypes.byref(value) for value in times]):raise ctypes.WinError(ctypes.get_last_error())
created=(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
# Attribute-only access is exempt from sharing; retain real directory read access.
FILE_LIST_DIRECTORY,FILE_SHARE_READ_WRITE=0x1,0x3
h=k.CreateFileW(sys.argv[1],FILE_LIST_DIRECTORY,FILE_SHARE_READ_WRITE,None,3,0x02000000,None)
if h==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
try:
 print(json.dumps(dict(pid=os.getpid(),path=sys.argv[1],executable=sys.executable,
  creation_filetime_100ns=str(created),python_version=list(sys.version_info[:3]),
  requested_access=FILE_LIST_DIRECTORY,share_mode=FILE_SHARE_READ_WRITE)),flush=True)
 sys.stdin.readline()
finally:
 if not k.CloseHandle(h):raise ctypes.WinError(ctypes.get_last_error())
'''
    proc = subprocess.Popen([str(executable), "-X", "utf8", "-B", "-c", script, str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ready = queue.Queue()
    threading.Thread(target=lambda: ready.put(proc.stdout.readline()), daemon=True).start()
    def close():
        if proc.poll() is None:
            _out, error = proc.communicate("release\n", timeout=10)
            assert proc.returncode == 0, error
        assert proc.poll() == 0  # Exact Popen-owned process retired successfully.
    try:
        owner = json.loads(ready.get(timeout=15))
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        kernel.GetProcessTimes.restype=wintypes.BOOL
        observed=[wintypes.FILETIME() for _ in range(4)]
        assert kernel.GetProcessTimes(proc._handle,*[ctypes.byref(value) for value in observed])
        created=(observed[0].dwHighDateTime<<32)|observed[0].dwLowDateTime
        assert Path(owner.pop("executable")).resolve(strict=True) == executable
        assert owner == dict(pid=proc.pid,path=str(path),
            creation_filetime_100ns=str(created),python_version=list(sys.version_info[:3]),
            requested_access=0x1,share_mode=0x3)
        yield SimpleNamespace(path=path,pid=proc.pid,creation_filetime_100ns=str(created),close=close)
    finally:
        try: close()
        finally:
            if proc.poll() is None: proc.kill(); proc.communicate(timeout=10)
            for stream in (proc.stdin,proc.stdout,proc.stderr):
                if not stream.closed: stream.close()


def test_native_query_reports_actual_directory_reader_pid_or_literal_unavailable(native_directory_reader):
    reader = native_directory_reader
    value = boundary.directory_handle_holders(reader.path)
    assert value["path"] == str(reader.path.resolve())
    if value["status"] != "observed":
        assert value["pids"] is None and value["error"]
        pytest.skip("Native capability unavailable: " + value["error"])
    assert reader.pid in value["pids"] and value["diagnostic_handle_closed"] is True
    reader.close()
    after = boundary.directory_handle_holders(reader.path)
    assert after["status"] == "observed" and reader.pid not in after["pids"]
    assert after["diagnostic_handle_closed"] is True and not after["proves_no_handles"]


def test_actual_native_reader_closes_between_physical_attempts(native_directory_reader, monkeypatch):
    reader = native_directory_reader; attempts, waits, records = [], [], []
    def move(a, b): attempts.append(1); jobs._windows_rename(a, b)
    def wait(seconds): waits.append(seconds); reader.close()
    monkeypatch.setattr(boundary.time, "sleep", wait)
    destination = reader.path.with_name("publication")
    boundary.ordinary_publication_move(reader.path, destination, move=move,
        assert_owner=lambda: None, record=lambda *row: records.append(row))
    assert attempts == [1, 1] and waits == [2]
    assert destination.is_dir() and not reader.path.exists()
    assert [row[2]["result"] for row in records if row[1] == "after"] == ["failed", "moved"]
