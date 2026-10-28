"""Bounded ordinary publication diagnostics; never a recovery rename policy."""
from __future__ import annotations

import os
import time
from pathlib import Path


def directory_handle_holders(path: Path) -> dict:
    """Query actual Windows directory-holder PIDs, or retain an unavailable error.

    FileProcessIdsUsingFileInformation (47) is reserved/system dependent. This
    is a local, non-atomic observation, not proof about a remote SMB server or
    a complete absence of handles. Its own query handle is always closed before
    returning; the querying PID can itself appear in the returned list.
    """
    value = dict(path=os.path.abspath(path), query="NtQueryInformationFile:47",
                 status="unavailable", pids=None, query_pid=os.getpid(),
                 local_snapshot_only=True, proves_no_handles=False,
                 diagnostic_handle_closed=None)
    if os.name != "nt":
        return dict(value, error="Windows native directory-handle query unavailable on " + os.name)
    import ctypes
    from ctypes import wintypes

    class IoStatus(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

    class ProcessIds(ctypes.Structure):
        _fields_ = [("Count", wintypes.ULONG), ("Ids", ctypes.c_size_t * 1)]

    handle = None
    kernel = None
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateFileW(value["path"], 0x80, 0x7, None, 3, 0x02000000, None)
        if handle == ctypes.c_void_p(-1).value:
            handle = None
            raise ctypes.WinError(ctypes.get_last_error())
        query = ctypes.WinDLL("ntdll").NtQueryInformationFile
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(IoStatus), ctypes.c_void_p,
                          wintypes.ULONG, ctypes.c_int]
        query.restype = wintypes.LONG
        for length in (8192, 32768, 131072, 524288, 1048576):
            buffer = ctypes.create_string_buffer(length)
            status = IoStatus()
            code = query(handle, ctypes.byref(status), buffer, length, 47) & 0xffffffff
            value["ntstatus"] = f"0x{code:08X}"
            if code in {0xC0000004, 0x80000005, 0xC0000023}:
                continue
            if code != 0:
                raise OSError("NtQueryInformationFile(47) returned " + value["ntstatus"])
            count = ProcessIds.from_buffer(buffer).Count
            offset = ProcessIds.Ids.offset
            used = int(status.Information)
            if not offset <= used <= length or count > (used - offset) // ctypes.sizeof(ctypes.c_size_t):
                raise OSError("NtQueryInformationFile(47) returned an invalid/truncated PID buffer")
            ids = (ctypes.c_size_t * count).from_buffer(buffer, offset)
            value.update(status="observed", pids=sorted(set(int(pid) for pid in ids)))
            break
        else:
            raise OSError("NtQueryInformationFile(47) PID buffer exceeds the 1 MiB diagnostic bound")
    except (OSError, AttributeError) as error:
        value.update(status="unavailable", pids=None,
                     error=type(error).__name__ + ": " + str(error))
    finally:
        if handle is not None:
            if kernel.CloseHandle(handle):
                value["diagnostic_handle_closed"] = True
            else:
                value.update(status="unavailable", pids=None, diagnostic_handle_closed=False,
                             close_error=str(ctypes.WinError(ctypes.get_last_error())))
    return value


def ordinary_publication_move(source, destination, *, move, assert_owner, record):
    """At most five physical sharing-error attempts, with four 2-second waits.

    ``move`` is the single original OS primitive, never the generic retrying
    rename helper. Every observation/log stream must be closed on return.
    Ownership or an existing destination refuses immediately. A diagnostic/log
    failure after a move error preserves that move error as the primary error.
    """
    source, destination = Path(source), Path(destination)
    previous = None

    def emit(attempt, phase, value):
        try:
            record(attempt, phase, value)
        except Exception as error:
            if previous is not None:
                previous.add_note("Publication diagnostic log also failed: " + repr(error))
                raise previous from error
            raise

    for attempt in range(1, 6):
        assert_owner()
        if destination.exists():
            raise FileExistsError("Publication destination already exists: " + str(destination))
        try:
            holders = directory_handle_holders(source)
        except Exception as error:
            holders = dict(path=str(source), status="unavailable", pids=None,
                           error=type(error).__name__ + ": " + str(error), proves_no_handles=False)
        emit(attempt, "before", dict(source=str(source), destination=str(destination),
             handle_holders=holders, maximum_physical_attempts=5, backoff_seconds=2))
        if holders.get("diagnostic_handle_closed") is False:
            error = RuntimeError("The owned publication diagnostic handle did not close")
            if previous is not None:
                previous.add_note(str(error))
                raise previous from error
            raise error
        assert_owner()
        if destination.exists():
            raise FileExistsError("Publication destination already exists: " + str(destination))
        try:
            move(source, destination)
        except OSError as error:
            previous = error
            emit(attempt, "after", dict(result="failed", error=type(error).__name__ + ": " + str(error),
                 winerror=getattr(error, "winerror", None)))
            if not isinstance(error, PermissionError) or getattr(error, "winerror", None) not in {5, 32}:
                raise
            if attempt == 5:
                raise
            time.sleep(2)
        else:
            previous = None
            try:
                emit(attempt, "after", dict(result="moved"))
            except Exception as error:
                error.add_note("The physical publication move succeeded; its after-log failed")
                raise
            return
