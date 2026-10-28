"""Kernel retirement witnesses for this file's pytest-owned child processes.

This is a fixture implementation of the existing retirement callback contract,
not a production process observer. It never equates retirement with PID absence.
"""
import ctypes as C
from decimal import Decimal, InvalidOperation


class _FileTime(C.Structure):
    _fields_ = [("low", C.c_uint32), ("high", C.c_uint32)]


class _Native:
    def __init__(self):
        self.k = C.WinDLL("kernel32", use_last_error=True)
        handle = C.c_void_p
        signatures = {
            "GetCurrentProcess": ([], handle),
            "DuplicateHandle": ([handle, handle, handle, C.POINTER(handle),
                                  C.c_uint32, C.c_int, C.c_uint32], C.c_int),
            "GetProcessId": ([handle], C.c_uint32),
            "GetProcessTimes": ([handle, *([C.POINTER(_FileTime)] * 4)], C.c_int),
            "WaitForSingleObject": ([handle, C.c_uint32], C.c_uint32),
            "GetExitCodeProcess": ([handle, C.POINTER(C.c_uint32)], C.c_int),
            "CloseHandle": ([handle], C.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = arguments, result

    def duplicate(self, borrowed):
        process, owned = self.k.GetCurrentProcess(), C.c_void_p()
        if not self.k.DuplicateHandle(process, borrowed, process, C.byref(owned),
                                      0, False, 2):  # DUPLICATE_SAME_ACCESS
            raise C.WinError(C.get_last_error())
        return owned.value

    def snapshot(self, handle):
        pid = self.k.GetProcessId(handle)
        if not pid:
            raise C.WinError(C.get_last_error())
        created, exited, kernel, user = (_FileTime() for _ in range(4))
        if not self.k.GetProcessTimes(handle, C.byref(created), C.byref(exited),
                                      C.byref(kernel), C.byref(user)):
            raise C.WinError(C.get_last_error())
        wait = self.k.WaitForSingleObject(handle, 0)
        if wait not in (0, 258):  # WAIT_OBJECT_0 or WAIT_TIMEOUT; never wait/retry.
            raise C.WinError(C.get_last_error())
        exitcode = C.c_uint32()
        if not self.k.GetExitCodeProcess(handle, C.byref(exitcode)):
            raise C.WinError(C.get_last_error())
        ticks = lambda value: (int(value.high) << 32) | int(value.low)
        return dict(pid=pid, creation_filetime_100ns=ticks(created),
                    exit_filetime_100ns=ticks(exited), wait_zero_result=wait,
                    exitcode=exitcode.value)

    def close(self, handle):
        if not self.k.CloseHandle(handle):
            raise C.WinError(C.get_last_error())


def _epoch_matches(owner, snapshot):
    """Same existing 2us precision allowance, plus exact native object pinning."""
    try:
        if (set(owner) != {"pid", "process_create_time"}
                or type(owner["pid"]) is not int or owner["pid"] <= 0
                or not isinstance(owner["process_create_time"], str)
                or owner["pid"] != snapshot["pid"]):
            return False
        expected = Decimal(owner["process_create_time"])
        actual = ((Decimal(snapshot["creation_filetime_100ns"])
                   - Decimal(116444736000000000)) / Decimal(10000000))
        return (expected.is_finite() and actual.is_finite()
                and abs(actual - expected) <= Decimal("0.000002"))
    except (InvalidOperation, TypeError, ValueError, KeyError):
        return False


class OwnedProcessWitness:
    """One duplicated spawn handle; PID lookup can never select a recycled PID."""
    def __init__(self, api, actor):
        self.api, self.handle, self.owner = api, None, None
        self.events = []
        self.handle = api.duplicate(actor.sentinel)
        try:
            self.birth = api.snapshot(self.handle)
            if self.birth["pid"] != actor.pid:
                raise AssertionError("Duplicated handle is not this pytest child")
        except BaseException:
            self.close()
            raise

    def bind(self, owner):
        if self.handle is None:
            return False
        if self.owner is not None:
            return self.owner == owner
        if not _epoch_matches(owner, self.birth):
            return False
        self.owner = dict(owner)
        return True

    def retired(self, owner):
        event = dict(requested_owner=dict(owner) if isinstance(owner, dict) else repr(owner),
                     closed=self.handle is None,
                     accepted=False)
        self.events.append(event)
        if self.handle is None or self.owner is None or owner != self.owner:
            event["reason"] = "closed, unbound, or different owner"
            return False
        try:
            current = self.api.snapshot(self.handle)
            event["native"] = current
            event["accepted"] = bool(
                current["pid"] == self.birth["pid"]
                and current["creation_filetime_100ns"] == self.birth["creation_filetime_100ns"]
                and _epoch_matches(owner, current)
                and current["wait_zero_result"] == 0
                and current["exitcode"] == 0
                and current["exit_filetime_100ns"] >= current["creation_filetime_100ns"] > 0)
            return event["accepted"]
        except Exception as error:
            event["error"] = dict(type=type(error).__name__, literal=repr(error))
            return False  # Unknown is never retirement.

    def close(self):
        if self.handle is not None:
            self.api.close(self.handle)
            self.handle = None

    def document(self):
        return dict(owner=self.owner, birth=self.birth, closed=self.handle is None,
                    events=self.events)


class OwnedRetirements:
    def __init__(self):
        self.api, self.witnesses = _Native(), []

    def capture(self, actor):
        witness = OwnedProcessWitness(self.api, actor)
        self.witnesses.append(witness)
        return witness

    def verify(self, owner):
        if not isinstance(owner, dict):
            return False
        matches = [witness for witness in self.witnesses if witness.owner == owner]
        return len(matches) == 1 and matches[0].retired(owner)

    def verified_retirement(self, manifest, proof):
        try:
            return manifest["owner"] == proof["owner"] and self.verify(proof["owner"])
        except (KeyError, TypeError, ValueError):
            return False

    def close(self):
        failures = []
        for witness in self.witnesses:
            try:
                witness.close()
            except Exception as error:
                failures.append(error)
        if failures:
            raise failures[0]

    def document(self):
        return dict(format="pytest-owned-native-retirement-v1",
                    is_pid_absence_proof=False,
                    witnesses=[witness.document() for witness in self.witnesses])
