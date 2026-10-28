"""Fixed Windows worker supervision; no scientific import or host installation.

Only a source-qualified explicit worker profile enables this entry. The initial
public worker invocation is the external observer, not a cache owner. It starts
the same public argv inside one native Job before its first instruction. The
actual worker owns its original lease, and only that owner seals its outcome.

No PID absence, atexit, sampled descendant list, or notification is accepted as
whole-family retirement. There is no kill-on-close, breakaway, retry, quota,
historical adoption, or result-reservation operation in this module.
"""
from __future__ import annotations

from copy import deepcopy
import ctypes as C
from ctypes import wintypes as W
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from .policy import CachePolicyError, _descriptor, _module_source_descriptor, nonnegative
from .control import _encoded


CONTEXT_ENV = "DARSIA_CACHE_NATIVE_LAUNCH"
CHANNEL_ENV = "DARSIA_CACHE_NATIVE_BOOTSTRAP_CHANNEL"
FORMAT = "darsia-native-owned-worker-launch-v1"
_ACTIVE_CHILD = None
_MAX_CONTROL = 1024 ** 2
_FORBIDDEN_JOB_FLAGS = 0x800 | 0x1000 | 0x2000


def _utc():
    return datetime.now(timezone.utc).isoformat()


class _STARTUPINFO(C.Structure):
    _fields_ = [("cb", W.DWORD), ("reserved", W.LPWSTR), ("desktop", W.LPWSTR),
        ("title", W.LPWSTR), ("x", W.DWORD), ("y", W.DWORD),
        ("xsize", W.DWORD), ("ysize", W.DWORD), ("xchars", W.DWORD),
        ("ychars", W.DWORD), ("fill", W.DWORD), ("flags", W.DWORD),
        ("show", W.WORD), ("reserved_size", W.WORD), ("reserved_bytes", W.LPBYTE),
        ("stdin", W.HANDLE), ("stdout", W.HANDLE), ("stderr", W.HANDLE)]


class _STARTUPINFOEX(C.Structure):
    _fields_ = [("startup", _STARTUPINFO), ("attributes", W.LPVOID)]


class _PROCESSINFO(C.Structure):
    _fields_ = [("process", W.HANDLE), ("thread", W.HANDLE),
               ("pid", W.DWORD), ("tid", W.DWORD)]


class _SECURITY(C.Structure):
    _fields_ = [("length", W.DWORD), ("descriptor", W.LPVOID), ("inherit", W.BOOL)]


class _BASIC_LIMIT(C.Structure):
    _fields_ = [("process_time", C.c_int64), ("job_time", C.c_int64),
        ("flags", W.DWORD), ("minimum_ws", C.c_size_t), ("maximum_ws", C.c_size_t),
        ("process_limit", W.DWORD), ("affinity", C.c_size_t),
        ("priority", W.DWORD), ("scheduling", W.DWORD)]


class _IO(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class _EXTENDED_LIMIT(C.Structure):
    _fields_ = [("basic", _BASIC_LIMIT), ("io", _IO),
        ("process_memory", C.c_size_t), ("job_memory", C.c_size_t),
        ("peak_process", C.c_size_t), ("peak_job", C.c_size_t)]


class _ACCOUNTING(C.Structure):
    _fields_ = [("user_time", C.c_int64), ("kernel_time", C.c_int64),
        ("period_user", C.c_int64), ("period_kernel", C.c_int64),
        ("page_faults", W.DWORD), ("total_processes", W.DWORD),
        ("active_processes", W.DWORD), ("terminated_processes", W.DWORD)]


class NativeCloseIncomplete(CachePolicyError):
    def __init__(self, failures, *, created_process=None):
        self.close_failures = deepcopy(failures)
        self.created_process = created_process
        super().__init__("Native supervision handle close incomplete: " + repr(failures))


def _close_handles(api, entries):
    failures = []
    for name, handle in entries:
        if handle is None:
            continue
        try:
            api.close(handle)
        except BaseException as error:
            failures.append(dict(kind=name, error_type=type(error).__name__, message=str(error)))
    if failures:
        raise NativeCloseIncomplete(failures)


class NativeAPI:
    """One explicit Windows x64 ABI. Construction performs no process launch."""
    def __init__(self):
        if os.name != "nt" or C.sizeof(C.c_void_p) != 8:
            raise CachePolicyError("Native worker supervision requires Windows x64")
        if (C.sizeof(_STARTUPINFO) != 104 or C.sizeof(_STARTUPINFOEX) != 112
                or C.sizeof(_PROCESSINFO) != 24 or C.sizeof(_BASIC_LIMIT) != 64
                or C.sizeof(_EXTENDED_LIMIT) != 144 or C.sizeof(_ACCOUNTING) != 48):
            raise CachePolicyError("Native worker ABI differs; no launch fallback")
        self.k = C.WinDLL("kernel32", use_last_error=True)
        declarations = {
            "GetCurrentProcess": ([], W.HANDLE),
            "GetCurrentProcessId": ([], W.DWORD),
            "GetProcessId": ([W.HANDLE], W.DWORD),
            "OpenProcess": ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
            "GetProcessTimes": ([W.HANDLE, *([C.POINTER(W.FILETIME)] * 4)], W.BOOL),
            "CreateJobObjectW": ([W.LPVOID, W.LPCWSTR], W.HANDLE),
            "OpenJobObjectW": ([W.DWORD, W.BOOL, W.LPCWSTR], W.HANDLE),
            "SetInformationJobObject": ([W.HANDLE, C.c_int, W.LPVOID, W.DWORD], W.BOOL),
            "QueryInformationJobObject": ([W.HANDLE, C.c_int, W.LPVOID, W.DWORD, W.LPVOID], W.BOOL),
            "IsProcessInJob": ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
            "AssignProcessToJobObject": ([W.HANDLE, W.HANDLE], W.BOOL),
            "CreateProcessW": ([W.LPCWSTR, W.LPWSTR, W.LPVOID, W.LPVOID, W.BOOL,
                W.DWORD, W.LPVOID, W.LPCWSTR, W.LPVOID, C.POINTER(_PROCESSINFO)], W.BOOL),
            "InitializeProcThreadAttributeList": ([W.LPVOID, W.DWORD, W.DWORD,
                C.POINTER(C.c_size_t)], W.BOOL),
            "UpdateProcThreadAttribute": ([W.LPVOID, W.DWORD, C.c_size_t, W.LPVOID,
                C.c_size_t, W.LPVOID, W.LPVOID], W.BOOL),
            "DeleteProcThreadAttributeList": ([W.LPVOID], None),
            "CreatePipe": ([C.POINTER(W.HANDLE), C.POINTER(W.HANDLE),
                C.POINTER(_SECURITY), W.DWORD], W.BOOL),
            "SetHandleInformation": ([W.HANDLE, W.DWORD, W.DWORD], W.BOOL),
            "GetStdHandle": ([W.DWORD], W.HANDLE),
            "DuplicateHandle": ([W.HANDLE, W.HANDLE, W.HANDLE, C.POINTER(W.HANDLE),
                W.DWORD, W.BOOL, W.DWORD], W.BOOL),
            "ResumeThread": ([W.HANDLE], W.DWORD),
            "TerminateProcess": ([W.HANDLE, W.UINT], W.BOOL),
            "WaitForSingleObject": ([W.HANDLE, W.DWORD], W.DWORD),
            "GetExitCodeProcess": ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL),
            "ReadFile": ([W.HANDLE, W.LPVOID, W.DWORD, C.POINTER(W.DWORD), W.LPVOID], W.BOOL),
            "WriteFile": ([W.HANDLE, W.LPCVOID, W.DWORD, C.POINTER(W.DWORD), W.LPVOID], W.BOOL),
            "CloseHandle": ([W.HANDLE], W.BOOL),
        }
        for name, (args, result) in declarations.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = args, result

    def check(self, okay, operation):
        if not okay:
            raise OSError(C.get_last_error(), operation)

    def close(self, handle):
        self.check(self.k.CloseHandle(handle), "CloseHandle; native state is unconfirmed")

    def identity(self, handle):
        pid = int(self.k.GetProcessId(handle))
        self.check(pid, "GetProcessId")
        times = [W.FILETIME() for _ in range(4)]
        self.check(self.k.GetProcessTimes(handle, *[C.byref(t) for t in times]), "GetProcessTimes")
        creation = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        if not creation:
            raise CachePolicyError("Native process creation FILETIME unavailable")
        return dict(pid=pid, creation_filetime_100ns=str(creation))

    def current_in_job(self):
        result = W.BOOL()
        self.check(self.k.IsProcessInJob(self.k.GetCurrentProcess(), None, C.byref(result)),
                   "Cannot determine nested parent Job")
        return bool(result.value)

    def inherited_job_membership(self, process):
        """Record inherited membership; kernel assignment decides valid nesting.

        This does not open a foreign Job or promise independence from its
        lifetime. Our one fresh own Job still forbids all breakaways.
        """
        return self.member(process, None)

    def member(self, process, job):
        result = W.BOOL()
        self.check(self.k.IsProcessInJob(process, job, C.byref(result)), "Cannot verify original Job membership")
        return bool(result.value)

    def open_owner_while_live(self, owner, job):
        """Bind the actual Python lease owner, distinct from a venv launcher.

        The returned handle is retained through finalization. A fast process
        that has already exited is not reconstructed from a reported PID.
        """
        if (not isinstance(owner, dict) or type(owner.get("pid")) is not int
                or owner["pid"] <= 0
                or not re.fullmatch(r"[1-9][0-9]*", str(owner.get("creation_filetime_100ns", "")))):
            raise CachePolicyError("Reported actual native service owner is unbound")
        handle = self.k.OpenProcess(0x1000 | 0x100000, False, owner["pid"])
        self.check(handle, "Actual service owner exited/unavailable before live handle binding")
        try:
            expected = {key: owner[key] for key in ("pid", "creation_filetime_100ns")}
            if self.identity(handle) != expected:
                raise CachePolicyError("Actual native owner PID/FILETIME differs; no late binding")
            if self.signaled(handle):
                raise CachePolicyError("Actual service owner exited before live handle binding")
            if not self.member(handle, job):
                raise CachePolicyError("Actual service owner is outside the original retained Job")
            return handle
        except BaseException as primary:
            try:
                self.close(handle)
            except BaseException as error:
                failure = dict(kind="temporary_actual_owner_process", handle=handle,
                    error_type=type(error).__name__, message=str(error),
                    primary_error_type=type(primary).__name__, primary_message=str(primary),
                    close_confirmed=False, no_automatic_close_retry=True)
                incomplete = NativeCloseIncomplete([failure])
                incomplete.original_owner_error = dict(
                    error_type=type(primary).__name__, message=str(primary))
                raise incomplete from primary
            raise

    def new_job(self, name):
        C.set_last_error(0)
        handle = self.k.CreateJobObjectW(None, name)
        error = C.get_last_error()
        if not handle:
            raise OSError(error, "CreateJobObjectW")
        if error == 183:
            self.close(handle)
            raise CachePolicyError("Fresh native Job name already exists")
        try:
            limits = _EXTENDED_LIMIT()  # No memory cap, breakaway or kill-on-close.
            self.check(self.k.SetInformationJobObject(handle, 9, C.byref(limits), C.sizeof(limits)),
                       "Cannot set the explicit non-breakaway Job contract")
            self.require_job_flags(handle)
            return handle
        except BaseException:
            self.close(handle)
            raise

    def require_job_flags(self, handle):
        limits = _EXTENDED_LIMIT()
        self.check(self.k.QueryInformationJobObject(handle, 9, C.byref(limits),
            C.sizeof(limits), None), "Cannot revalidate native Job flags")
        if limits.basic.flags != 0:
            raise CachePolicyError("Original own Job flags changed; no retirement qualification")
        return int(limits.basic.flags)

    def accounting(self, handle):
        row = _ACCOUNTING()
        self.check(self.k.QueryInformationJobObject(handle, 1, C.byref(row), C.sizeof(row), None),
                   "Native whole-Job accounting unavailable")
        # TotalTerminatedProcesses counts limit violations, not every natural
        # exit. Never use it as a reconstructed per-process retirement inventory.
        return dict(total_processes=int(row.total_processes),
            active_processes=int(row.active_processes),
            terminated_by_limit_processes=int(row.terminated_processes))

    def lifetime_memory(self, handle):
        """Kernel-maintained Job commit peaks; not RSS or sampled cache bytes."""
        row = _EXTENDED_LIMIT()
        self.check(self.k.QueryInformationJobObject(handle, 9, C.byref(row), C.sizeof(row), None),
                   "Native lifetime Job memory unavailable")
        if row.basic.flags != 0:
            raise CachePolicyError("Native Job flags changed during lifetime observation")
        return dict(peak_job_committed_bytes=int(row.peak_job),
            peak_single_process_committed_bytes=int(row.peak_process),
            semantics="kernel PeakJobMemoryUsed/PeakProcessMemoryUsed; not RSS or filesystem allocation")

    def pipe(self):
        read, write = W.HANDLE(), W.HANDLE()
        security = _SECURITY(C.sizeof(_SECURITY), None, True)
        self.check(self.k.CreatePipe(C.byref(read), C.byref(write), C.byref(security), 0),
                   "Cannot create bounded native bootstrap channel")
        try:
            self.check(self.k.SetHandleInformation(read, 1, 0), "Cannot make observer channel noninheritable")
        except BaseException:
            _close_handles(self, [("pipe_read", read), ("pipe_write", write)])
            raise
        return read.value, write.value

    def create_suspended(self, command, environment, cwd, channel, *, process_information=None):
        """Only exact stdio duplicates + the one bootstrap handle are inherited."""
        startup = _STARTUPINFOEX()
        startup.startup.cb = C.sizeof(startup)
        startup.startup.flags = 0x100  # STARTF_USESTDHANDLES; preserves absent handles.
        duplicates, inherited = [], [channel]
        current = self.k.GetCurrentProcess()
        attributes = None
        attributes_initialized = False
        created = None
        info = process_information if process_information is not None else _PROCESSINFO()
        if not isinstance(info, _PROCESSINFO) or info.process or info.thread or info.pid or info.tid:
            raise CachePolicyError("Original caller-owned PROCESS_INFORMATION must be empty before launch")
        primary_error = None
        try:
            for number, field in ((-10, "stdin"), (-11, "stdout"), (-12, "stderr")):
                original = self.k.GetStdHandle(number & 0xffffffff)
                if original not in (None, 0, C.c_void_p(-1).value):
                    duplicate = W.HANDLE()
                    self.check(self.k.DuplicateHandle(current, original, current,
                        C.byref(duplicate), 0, True, 2), "Cannot duplicate original standard handle")
                    duplicates.append(duplicate.value); inherited.append(duplicate.value)
                    setattr(startup.startup, field, duplicate.value)
            size = C.c_size_t()
            C.set_last_error(0)
            okay = self.k.InitializeProcThreadAttributeList(None, 1, 0, C.byref(size))
            if okay or C.get_last_error() != 122 or not 0 < size.value <= 65536:
                raise CachePolicyError("Unexpected native attribute-list size contract")
            storage = C.create_string_buffer(size.value)
            attributes = C.cast(storage, W.LPVOID)
            self.check(self.k.InitializeProcThreadAttributeList(attributes, 1, 0, C.byref(size)),
                       "Cannot initialize exact inherited-handle list")
            attributes_initialized = True
            handles = (W.HANDLE * len(inherited))(*inherited)
            self.check(self.k.UpdateProcThreadAttribute(attributes, 0, 0x00020002,
                C.cast(handles, W.LPVOID), C.sizeof(handles), None, None),
                "Cannot pin exact inherited-handle list")
            startup.attributes = attributes
            if any(not isinstance(k, str) or not isinstance(v, str) or "\0" in k + v
                   or ("=" in k and re.fullmatch(r"=[A-Za-z]:", k) is None)
                   for k, v in environment.items()):
                raise CachePolicyError("Invalid original child environment")
            ordered = sorted(environment, key=str.casefold)
            if len({k.casefold() for k in ordered}) != len(ordered):
                raise CachePolicyError("Ambiguous Windows environment keys")
            block = C.create_unicode_buffer("\0".join(k + "=" + environment[k] for k in ordered) + "\0\0")
            command_line = C.create_unicode_buffer(subprocess.list2cmdline(command))
            self.check(self.k.CreateProcessW(command[0], command_line, None, None, True,
                0x4 | 0x400 | 0x80000, C.cast(block, W.LPVOID), str(cwd),
                C.byref(startup), C.byref(info)), "Exact worker CreateProcessW failed before resume")
            created = dict(process=info.process, thread=info.thread, pid=int(info.pid))
            return created
        except BaseException as error:
            # PROCESS_INFORMATION is allocated before the native call. Even a
            # KeyboardInterrupt between its return and the Python dict must
            # return the actual handles to the one original controller.
            primary_error = error
            if info.process or info.thread:
                created = dict(process=info.process, thread=info.thread, pid=int(info.pid))
                error.created_process = created
                error.native_create_flags = 0x4 | 0x400 | 0x80000
                error.resume_not_attempted_by_create = True
            raise
        finally:
            # Recover directly from the retained native output structure, not
            # from successful completion of a later Python assignment.
            if info.process or info.thread:
                created = dict(process=info.process, thread=info.thread, pid=int(info.pid))
            failures = []
            if attributes_initialized:
                try:
                    self.k.DeleteProcThreadAttributeList(attributes)
                except BaseException as error:
                    failures.append(dict(kind="attribute_list", error_type=type(error).__name__, message=str(error)))
            for handle in duplicates:
                try: self.close(handle)
                except BaseException as error:
                    failures.append(dict(kind="stdio_duplicate", error_type=type(error).__name__, message=str(error)))
            if failures:
                error = NativeCloseIncomplete(failures, created_process=created)
                if primary_error is not None:
                    error.original_create_error = dict(error_type=type(primary_error).__name__, message=str(primary_error))
                raise error from primary_error

    def assign(self, job, process):
        self.check(self.k.AssignProcessToJobObject(job, process),
                   "AssignProcessToJobObject refused before first instruction")
        if not self.member(process, job):
            raise CachePolicyError("Controlled suspended launch is not in its original Job")

    def resume(self, thread):
        previous = int(self.k.ResumeThread(thread))
        if previous == 0xffffffff:
            raise OSError(C.get_last_error(), "ResumeThread state unconfirmed; no post-attempt kill")
        if previous != 1:
            raise CachePolicyError("Unexpected ResumeThread result; launch state held: " + str(previous))

    def signaled(self, process):
        result = int(self.k.WaitForSingleObject(process, 0))
        if result not in (0, 258):
            raise OSError(C.get_last_error(), "Native process wait state unavailable")
        return result == 0

    def exit_code(self, process):
        if not self.signaled(process):
            raise CachePolicyError("A live process has no final native exit code")
        result = W.DWORD()
        self.check(self.k.GetExitCodeProcess(process, C.byref(result)), "GetExitCodeProcess")
        return int(result.value)

    def terminate_before_resume(self, process):
        self.check(self.k.TerminateProcess(process, 1), "Unstarted owned launch termination incomplete")
        if int(self.k.WaitForSingleObject(process, 30000)) != 0:
            raise CachePolicyError("Unstarted original process did not retire; preserve exact handles")

    def read_message(self, handle):
        result = bytearray()
        while len(result) <= _MAX_CONTROL:
            buffer, count = C.create_string_buffer(4096), W.DWORD()
            okay = self.k.ReadFile(handle, buffer, 4096, C.byref(count), None)
            if not okay:
                error = C.get_last_error()
                if error == 109:
                    break
                raise OSError(error, "Native bootstrap channel read failed")
            if not count.value:
                break
            result.extend(buffer.raw[:count.value])
            if b"\n" in result:
                if result[-1:] != b"\n" or result.count(b"\n") != 1:
                    raise CachePolicyError("Native bootstrap channel has extra messages")
                if len(result) > _MAX_CONTROL:
                    break
                return json.loads(result)
        raise CachePolicyError("Native child did not publish one bounded actual service receipt")

    def write_message(self, handle, document):
        _encoded(document)  # Preserve the existing metadata bound/finite JSON contract.
        raw = (json.dumps(document, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
        if len(raw) > _MAX_CONTROL:
            raise CachePolicyError("Native bootstrap channel message exceeds its bound")
        offset = 0
        while offset < len(raw):
            block = raw[offset:offset + 4096]
            buffer, count = C.create_string_buffer(block), W.DWORD()
            self.check(self.k.WriteFile(handle, buffer, len(block), C.byref(count), None),
                       "Native child service receipt channel write incomplete")
            if not count.value:
                raise CachePolicyError("Native bootstrap channel short write")
            offset += count.value


def _checked_contract(lease):
    if getattr(lease, "first_qualification", None) is not None:
        from .first_qualification import validate_startup
        return validate_startup(lease)
    block = lease.profile.get("native_supervision")
    keys = {"format", "completion", "startup", "maximum_family_processes", "publication_contract"}
    if (not isinstance(block, dict) or set(block) != keys
            or block["format"] != "darsia-native-worker-supervision-profile-v1"
            or block["publication_contract"] != "alias-done-small-metadata-v1"
            or not nonnegative(block["maximum_family_processes"], "Measured native family count")):
        raise CachePolicyError("Explicit qualified native worker-supervision profile is required")
    from .bootstrap import _exact
    completion = _exact(block["completion"], lease.package_file)
    if completion.get("complete") is not True or completion.get("passed") is not True:
        raise CachePolicyError("Native supervision profile has no passed original qualification")
    startup = _exact(block["startup"], lease.package_file)
    if (startup.get("format") != "darsia-qualified-worker-startup-source-v1"
            or startup.get("python_executable") != lease.profile["python_executable"]
            or startup.get("module_closure") != lease.profile["module_closure"]
            or startup.get("include_system_site_packages") is not False
            or not startup.get("source_files") or not sys.flags.dont_write_bytecode):
        raise CachePolicyError("Exact qualified pre-import startup closure and interpreter -B required")
    actual_path = [str(Path(value or os.getcwd()).resolve()) for value in sys.path]
    if actual_path != startup.get("sys_path"):
        raise CachePolicyError("Current pre-import search path differs from its qualified source closure")
    for pin in startup["source_files"]:
        _exact_source(pin)
    actual = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
              if getattr(module, "__file__", None) and not name.startswith("darsia.")}
    for name, path in actual.items():
        if name not in startup.get("allowed_loaded_modules", {}) or startup["allowed_loaded_modules"][name] != path:
            raise CachePolicyError("Unqualified import precedes worker cache admission: " + name)
    if any(name.split(".")[0] in {"numba", "matplotlib", "cupy", "cv2", "numpy", "PIL"}
           for name in sys.modules):
        raise CachePolicyError("A third-party producer was imported before the actual managed child")
    return block


def _exact_source(binding):
    pin = _module_source_descriptor(binding)
    path = Path(pin["path"])
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
            or len(raw) != pin["size_bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]):
        raise CachePolicyError("Qualified native/startup source changed")


def _control(lease, name):
    from .windows import WindowsNamespace
    bound = lease.admission["receipts"]
    if Path(name).name != name:
        raise CachePolicyError("Native control must be one original receipt component")
    namespace = WindowsNamespace(bound["path"], bound["identity"])
    try:
        raw, _ = namespace.read_once(name)
    finally:
        namespace.close()
    return json.loads(raw), dict(path=str(Path(bound["path"]) / name),
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _bound_control(lease, binding):
    pin = _descriptor(binding)
    if Path(pin["path"]).parent != Path(lease.admission["receipts"]["path"]):
        raise CachePolicyError("Native evidence escaped its original receipt namespace")
    document, actual = _control(lease, Path(pin["path"]).name)
    if actual != pin:
        raise CachePolicyError("Native control bytes differ from original binding")
    return document


def _public_command(lease):
    original = list(getattr(sys, "orig_argv", ()))
    from .bootstrap import _python_module_invocation
    entry = _python_module_invocation()
    if (not original or not entry or entry[0] != "darsia.fluidflower.worker"
            or not sys.flags.dont_write_bytecode):
        raise CachePolicyError("Native supervision keeps the exact public worker command with -B")
    selected = Path(sys.executable).resolve()
    selected_binding = _descriptor(lease.profile["python_executable"])
    if Path(selected_binding["path"]).resolve() != selected:
        raise CachePolicyError("Selected public venv executable differs from qualified runtime")
    _exact_source(selected_binding)
    observed = Path(original[0]).resolve()
    if observed != selected:
        base = getattr(sys, "_base_executable", None)
        if not base or observed != Path(base).resolve():
            raise CachePolicyError("Original argv0 is neither selected venv nor its actual backing executable")
        from .bootstrap import _exact
        startup = _exact(lease.profile["native_supervision"]["startup"], lease.package_file)
        bindings = [pin for pin in startup["source_files"]
                    if Path(_module_source_descriptor(pin)["path"]).resolve() == observed]
        if len(bindings) != 1:
            raise CachePolicyError("Actual backing executable has no unique qualified runtime source binding")
        _exact_source(bindings[0])
    # uv exposes its backing interpreter in orig_argv[0]. Re-enter through the
    # exact selected venv, preserving every original flag/module/worker argument.
    command = [sys.executable, *original[1:]]
    if lease.profile.get("format") == "darsia-setup-service-import-profile-v1":
        from .setup_qualification import require_public_command
        require_public_command(lease.profile, command)
    if lease.profile.get("format") == "darsia-preset-wasserstein-bounded-profile-v1":
        from .preset_qualification import require_public_command
        require_public_command(lease.profile, command)
    return command


def native_child_context(lease, *, api=None):
    """Validate original control + actual membership; no lease is allocated here."""
    global _ACTIVE_CHILD
    api = api or NativeAPI()
    context = json.loads(os.environ[CONTEXT_ENV])
    if set(context) != {"intent", "observed_name"}:
        raise CachePolicyError("Native launch context is not an exact original binding")
    intent = _bound_control(lease, context["intent"])
    observed, pin = _control(lease, context["observed_name"])
    if (intent.get("format") != FORMAT or observed.get("format") != FORMAT
            or observed.get("intent") != context["intent"]
            or observed.get("job_name") != intent.get("job_name")
            or observed.get("assignment_before_resume") is not True
            or observed.get("own_job_assignment_verified_before_resume") is not True
            or type(observed.get("inherited_job_membership_observed")) is not bool
            or observed.get("own_job_flags_before_resume") != 0
            or observed.get("own_job_accounting_before_resume") != dict(
                total_processes=1, active_processes=1, terminated_by_limit_processes=0)
            or observed.get("assigned_child_native_identity") != observed.get("actual_launch_wrapper")
            or observed.get("policy") != lease.source
            or observed.get("profile") != lease.profile_binding
            or intent.get("controller") == lease.owner):
        raise CachePolicyError("Managed child differs from its original native launch")
    name = intent["job_name"]
    if re.fullmatch(r"Local\\DarSIA-cache-[0-9a-f]{32}", name) is None:
        raise CachePolicyError("Unexpected native Job name")
    handle = api.k.OpenJobObjectW(0x4, False, name)
    api.check(handle, "Original native Job unavailable before child admission")
    try:
        api.require_job_flags(handle)
        if not api.member(api.k.GetCurrentProcess(), handle):
            raise CachePolicyError("Worker is outside its original native Job")
    finally:
        api.close(handle)
    _ACTIVE_CHILD = dict(binding=pin, job_name=name, current_native=api.identity(api.k.GetCurrentProcess()))
    return pin


def prepare_native_entry(lease, *, inherited_receipt, api=None):
    """Called only for an explicitly qualified worker profile before lease.start."""
    if lease.entry_kind != "worker":
        raise CachePolicyError("Native worker supervision cannot classify runtime/tests")
    _checked_contract(lease)
    if CONTEXT_ENV in os.environ:
        return native_child_context(lease, api=api)
    if inherited_receipt:
        raise CachePolicyError("Native profile receipt has no actual original Job context")
    raise SystemExit(supervise_original_worker(lease, api=api))


def report_native_service(lease, *, api=None):
    if _ACTIVE_CHILD is None:
        return
    if lease.manifest.get("native_launch_context") != _ACTIVE_CHILD["binding"]:
        raise CachePolicyError("Actual service receipt lost its original native launch binding")
    if CHANNEL_ENV not in os.environ:
        return  # Already reported by primary; ordinary multiprocessing attachment.
    api = api or NativeAPI()
    value = os.environ.pop(CHANNEL_ENV)
    if not value.isdecimal() or not int(value):
        raise CachePolicyError("Original native bootstrap channel is missing")
    handle = int(value)
    try:
        api.write_message(handle, dict(format="darsia-native-service-bootstrap-v1",
            service_receipt=lease.receipt, native_launch=_ACTIVE_CHILD["binding"],
            owner=deepcopy(lease.owner)))
    finally:
        api.close(handle)


def current_native_membership(*, api=None):
    if _ACTIVE_CHILD is None:
        raise CachePolicyError("Current owner has no original native Job context")
    api = api or NativeAPI()
    handle = api.k.OpenJobObjectW(0x4, False, _ACTIVE_CHILD["job_name"])
    api.check(handle, "Original native Job unavailable during owner registration")
    try:
        api.require_job_flags(handle)
        if (not api.member(api.k.GetCurrentProcess(), handle)
                or api.identity(api.k.GetCurrentProcess()) != _ACTIVE_CHILD["current_native"]):
            raise CachePolicyError("Current owner native FILETIME/Job membership differs")
        return deepcopy(_ACTIVE_CHILD["binding"])
    finally:
        api.close(handle)


def native_launch_identity(process, *, sentinel=None, api=None):
    """Enrich an OWNED existing observation; never accept foreign ancestor PIDs."""
    if _ACTIVE_CHILD is None:
        return None
    api = api or NativeAPI()
    temporary = None
    handle = sentinel
    if handle is None:
        handle = api.k.OpenProcess(0x1000 | 0x100000, False, process.pid)
        api.check(handle, "Native original process identity unavailable")
        temporary = handle
    job = None
    try:
        job = api.k.OpenJobObjectW(0x4, False, _ACTIVE_CHILD["job_name"])
        api.check(job, "Original native Job no longer available")
        identity = api.identity(handle)
        if identity["pid"] != process.pid or not api.member(handle, job):
            raise CachePolicyError("Observed child is outside the original controlled Job")
        if sentinel is None:
            # The exact native FILETIME is authoritative. This comparison only
            # refuses a psutil PID-reuse race within its documented float
            # representation envelope; it is never a scientific tolerance.
            native_epoch = (Decimal(identity["creation_filetime_100ns"]) -
                            Decimal(116444736000000000)) / Decimal(10000000)
            if abs(native_epoch - Decimal(str(process.create_time()))) > Decimal("0.000002"):
                raise CachePolicyError("Observed psutil/native process incarnations differ")
        return dict(identity, native_launch_context=_ACTIVE_CHILD["binding"])
    finally:
        _close_handles(api, [("identity_job", job or None), ("identity_process", temporary)])


def bind_retained_native_observation(native, observation):
    """Never mix a retained HANDLE epoch with a newer psutil PID incarnation."""
    epoch = (Decimal(native["creation_filetime_100ns"]) -
             Decimal(116444736000000000)) / Decimal(10000000)
    valid = (isinstance(observation, dict)
             and observation.get("pid") == native["pid"]
             and observation.get("creation_filetime_100ns") == native["creation_filetime_100ns"]
             and observation.get("native_launch_context") == native.get("native_launch_context"))
    if valid:
        result = deepcopy(observation)
    else:
        result = dict(pid=native["pid"], process_create_time=str(epoch), command_line=None,
            psutil_epoch_and_argv_not_bound=True,
            discarded_psutil_observation=observation is not None)
    result.update(native)
    return result


class OwnedWorker:
    """One original launch, no retries. Process/Job handles outlive child exit."""
    def __init__(self, lease, *, api=None):
        self.lease, self.api = lease, api or NativeAPI()
        self.job = self.info = self.read = self.write = None
        self.owner_handle = self.primary_binding = None
        self.resumed = False
        self.resume_attempted = False
        self.close_failures = []
        self.retirement_binding = self.service_binding = None
        self.observed_binding = None
        # This storage belongs to the outer controller before the native call.
        # It survives API-finally and return-to-assignment interruptions.
        self.creation_output = _PROCESSINFO()

    def _recover_creation_output(self):
        if self.info is None and (self.creation_output.process or self.creation_output.thread):
            self.info = dict(process=self.creation_output.process,
                thread=self.creation_output.thread, pid=int(self.creation_output.pid))
        return self.info

    def _close(self, name, handle):
        if handle is None:
            return
        try:
            self.api.close(handle)
        except BaseException as error:
            self.close_failures.append(dict(kind=name, error_type=type(error).__name__, message=str(error)))

    def start(self, command, environment, cwd):
        token = uuid.uuid4().hex
        name = "Local\\DarSIA-cache-" + token
        from .bootstrap import _owner
        controller, _ = _owner()
        self.job = self.api.new_job(name)
        if self.api.accounting(self.job) != dict(
                total_processes=0, active_processes=0, terminated_by_limit_processes=0):
            raise CachePolicyError("Fresh original Job is not empty before controlled launch")
        self.read, self.write = self.api.pipe()
        intent = dict(format=FORMAT, job_name=name, controller=controller,
            command=list(command), cwd=str(cwd), policy=self.lease.source,
            observed_interpreter_argv=list(getattr(sys, "orig_argv", ())),
            profile=self.lease.profile_binding, utc=_utc(), producer_owner_not_allocated=True)
        bound = self.lease._record("native-" + token + "-intent.json", intent)
        observed_name = "native-" + token + "-observed.json"
        child_environment = dict(environment)
        child_environment[CONTEXT_ENV] = json.dumps(dict(intent=bound, observed_name=observed_name),
                                                   sort_keys=True, separators=(",", ":"))
        child_environment[CHANNEL_ENV] = str(self.write)
        try:
            try:
                self.info = self.api.create_suspended(command, child_environment, cwd, self.write,
                    process_information=self.creation_output)
            except BaseException as error:
                self.info = getattr(error, "created_process", None)
                raise
            native = self.api.identity(self.info["process"])
            if native["pid"] != self.info["pid"]:
                raise CachePolicyError("Original CreateProcessW PID/handle identity differs")
            inherited = self.api.inherited_job_membership(self.info["process"])
            self.api.assign(self.job, self.info["process"])
            assigned_native = self.api.identity(self.info["process"])
            flags = self.api.require_job_flags(self.job)
            accounting = self.api.accounting(self.job)
            if (assigned_native != native or not self.api.member(self.info["process"], self.job)
                    or flags != 0 or accounting != dict(
                        total_processes=1, active_processes=1, terminated_by_limit_processes=0)):
                raise CachePolicyError("Original suspended child own-Job identity/flags/accounting differs before resume")
            self.observed_binding = self.lease._record(observed_name, dict(format=FORMAT,
                intent=bound, job_name=name, controller=controller, actual_launch_wrapper=native,
                policy=self.lease.source, profile=self.lease.profile_binding,
                assignment_before_resume=True, before_first_instruction=True,
                inherited_job_membership_observed=inherited,
                own_job_assignment_verified_before_resume=True,
                assigned_child_native_identity=assigned_native,
                own_job_flags_before_resume=flags,
                own_job_accounting_before_resume=accounting,
                independence_or_survival_of_outer_job_not_claimed=True, utc=_utc()))
            self.resume_attempted = True
            self.api.resume(self.info["thread"])
            self.resumed = True
            thread = self.info.pop("thread")
            self._close("thread_after_resume", thread)
            self._close("parent_channel_write", self.write); self.write = None
            if self.close_failures:
                raise NativeCloseIncomplete(self.close_failures)
            message = self.api.read_message(self.read)
            self._close("parent_channel_read", self.read); self.read = None
            if (message.get("format") != "darsia-native-service-bootstrap-v1"
                    or message.get("native_launch") != self.observed_binding):
                raise CachePolicyError("Native bootstrap channel did not bind this original launch")
            self.service_binding = _descriptor(message["service_receipt"])
            self.native_primary = deepcopy(message["owner"])
            self.owner_handle = self.api.open_owner_while_live(self.native_primary, self.job)
            self.primary_binding = self.lease._record(
                "native-" + token + "-actual-primary-bound.json", dict(format=FORMAT,
                    original_native_launch=self.observed_binding,
                    actual_service_receipt=self.service_binding,
                    actual_primary=self.native_primary,
                    bound_native_identity=self.api.identity(self.owner_handle),
                    same_original_job_membership_observed=True,
                    held_process_handle_observed_while_live=True, utc=_utc()))
            self.intent = intent
            return self.service_binding
        except BaseException as error:
            self._recover_creation_output()
            self.close_failures.extend(getattr(error, "close_failures", []))
            if self.info is not None and self.info.get("process") and not self.resume_attempted:
                # Exact process handle returned suspended by our CreateProcessW;
                # no instruction has been admitted, never a PID-based kill.
                self.api.terminate_before_resume(self.info["process"])
            raise

    def await_family(self, *, observer=None):
        while True:
            try:
                accounting = self.api.accounting(self.job)
                flags = self.api.require_job_flags(self.job)
                if observer is not None:
                    observer(self, accounting)
                if (accounting["active_processes"] == 0
                        and self.api.signaled(self.info["process"])
                        and (self.owner_handle is None or self.api.signaled(self.owner_handle))):
                    if accounting["total_processes"] < 1:
                        raise CachePolicyError("Native family has no original controlled launch")
                    return dict(accounting=accounting, job_flags=flags,
                        wrapper_exit_code=self.api.exit_code(self.info["process"]), utc=_utc())
                time.sleep(.2)
            except KeyboardInterrupt:
                # Console Ctrl-C also reaches unchanged worker behavior. Do not
                # turn observer cancellation into a kill or false retirement.
                continue

    def record_closed_family(self, terminal):
        if (terminal["accounting"]["active_processes"] != 0 or self.close_failures
                or self.owner_handle is None or self.primary_binding is None
                or not self.api.signaled(self.owner_handle)
                or self.api.identity(self.owner_handle) != {
                    key: self.native_primary[key] for key in ("pid", "creation_filetime_100ns")}):
            raise CachePolicyError("Native family retirement or control handles remain unconfirmed")
        self.terminal = deepcopy(terminal)
        document = dict(format="darsia-native-owned-worker-retirement-v1", utc=_utc(),
            native_launch=self.observed_binding, service_receipt=self.service_binding,
            primary=self.native_primary, controller=self.intent["controller"],
            actual_primary_binding=self.primary_binding,
            job_name=self.intent["job_name"], terminal=terminal,
            whole_family_kernel_observation=True,
            transient_process_individual_epochs_not_fabricated=True,
            controller_is_not_producer_owner=True, no_process_kill_after_resume=True)
        self.retirement_binding = self.lease._record(
            Path(self.observed_binding["path"]).stem + "-retirement.json", document)
        return self.retirement_binding

    def verify_retirement(self, snapshot, binding):
        if (binding != self.retirement_binding or not self.retirement_binding
                or self.owner_handle is None or self.primary_binding is None
                or snapshot["binding"]["manifest"].get("native_launch_context") != self.observed_binding
                or snapshot["binding"]["manifest"].get("owner") != self.native_primary
                or snapshot["binding"].get("policy") != self.lease.source
                or snapshot["binding"].get("profile") != self.lease.profile_binding
                or self.api.accounting(self.job)["active_processes"] != 0
                or not self.api.signaled(self.info["process"])
                or not self.api.signaled(self.owner_handle)
                or self.api.identity(self.owner_handle) != {
                    key: self.native_primary[key] for key in ("pid", "creation_filetime_100ns")}):
            return False
        recorded = _bound_control(self.lease, binding)
        primary = _bound_control(self.lease, self.primary_binding)
        if (recorded.get("service_receipt") != self.service_binding
                or recorded.get("native_launch") != self.observed_binding
                or recorded.get("primary") != self.native_primary
                or recorded.get("actual_primary_binding") != self.primary_binding
                or primary.get("actual_service_receipt") != self.service_binding
                or primary.get("original_native_launch") != self.observed_binding
                or primary.get("actual_primary") != self.native_primary
                or primary.get("held_process_handle_observed_while_live") is not True
                or primary.get("same_original_job_membership_observed") is not True
                or recorded.get("terminal") != self.terminal):
            return False
        self.api.require_job_flags(self.job)
        if self.close_failures:
            return False
        for registered in snapshot["owners"]:
            value = registered["document"]["value"]
            owner = value["owner"]
            if (not str(owner.get("creation_filetime_100ns", "")).isdecimal()
                    or value.get("native_job_membership") != self.observed_binding):
                return False
        for launch in snapshot["launches"]:
            native = launch["observed"]["document"]["value"].get("original_native_identity", {})
            if (not str(native.get("creation_filetime_100ns", "")).isdecimal()
                    or native.get("native_launch_context") != self.observed_binding):
                return False
        return not snapshot["legacy_launches"]

    def close(self):
        self._recover_creation_output()
        owner = self.owner_handle
        self.owner_handle = None
        self._close("actual_primary_process", owner)
        if self.info:
            for name in ("thread", "process"):
                self._close(name, self.info.pop(name, None))
        for name in ("read", "write", "job"):
            handle = getattr(self, name)
            setattr(self, name, None)
            self._close(name, handle)
        if self.close_failures:
            raise NativeCloseIncomplete(self.close_failures)


def _budget_state(result, error, attempted):
    state = getattr(error, "finalization_state", None)
    if not isinstance(state, dict):
        state = result
    if isinstance(state, dict) and "service_budget_closed" in state:
        return state["service_budget_closed"]
    return None if attempted else False


def supervise_original_worker(lease, *, api=None):
    """External controller; never calls lease.start/attach/reserve/register/seal."""
    if getattr(lease, "first_qualification", None) is not None:
        from .first_qualification import supervise_first_worker
        return supervise_first_worker(lease, api=api)
    contract = _checked_contract(lease)
    command = _public_command(lease)
    owned = OwnedWorker(lease, api=api)
    outcome = None
    result = None
    finalization_attempted = False
    from .first_qualification import BOUND_IMPORT_PROFILE, BoundServiceReserveObservations
    bound_watch = (BoundServiceReserveObservations(lease, command)
        if lease.profile.get("format") in {BOUND_IMPORT_PROFILE, "darsia-common-phase-import-profile-v1",
                                          "darsia-setup-service-import-profile-v1",
                                          "darsia-preset-wasserstein-bounded-profile-v1"} else None)
    terminal = None
    try:
        if bound_watch is not None:
            bound_watch.admit()
        owned.start(command, dict(os.environ), Path.cwd())
        terminal = (owned.await_family(observer=bound_watch.observe)
            if bound_watch is not None else owned.await_family())
        if bound_watch is not None:
            bound_watch.observe(owned, terminal["accounting"], force=True)
        outcome = owned.record_closed_family(terminal)
        if bound_watch is not None:
            bound_watch.finish(owned, terminal, outcome)
            bound_watch.require_no_stop()
        if terminal["accounting"]["total_processes"] > contract["maximum_family_processes"]:
            raise CachePolicyError("Actual native process count exceeds qualified profile; cache remains charged")
        from .fresh_scope import finalize_original_fresh_scope
        from .adapters import verify_worker_publication
        # Real native handles stay open through this callback and finalizer.
        finalization_attempted = True
        result = finalize_original_fresh_scope(owned.service_binding,
            package_file=lease.package_file, now_utc=_utc(), retirement=outcome,
            verify_retirement=owned.verify_retirement,
            verify_publication=verify_worker_publication)
        lease._record(Path(owned.observed_binding["path"]).stem + "-finalization.json",
            dict(original_service=owned.service_binding, retirement=outcome,
                 result=result, actual_public_command_exit_code=terminal["wrapper_exit_code"]))
        return terminal["wrapper_exit_code"]
    except BaseException as error:
        # Failure retention/unsealed crash stays charged. No positive proof is
        # reconstructed from PIDs; no original scientific output is modified.
        try:
            if bound_watch is not None and bound_watch.closed is None:
                try:
                    bound_watch.finish(owned, terminal, outcome)
                except BaseException as diagnostic_error:
                    error.add_note("Bound reserve diagnostics incomplete: " + repr(diagnostic_error))
            lease._record("native-observer-held-" + uuid.uuid4().hex + ".json", dict(
                utc=_utc(), exception_type=type(error).__name__, message=str(error),
                native_launch=owned.observed_binding, service_receipt=owned.service_binding,
                retirement=outcome, no_automatic_retry=True,
                actual_partial_launch_handles=deepcopy(owned.info),
                resume_attempted=owned.resume_attempted,
                service_budget_closed=_budget_state(result, error, finalization_attempted),
                structured_finalization_state=getattr(error, "finalization_state", None)))
        except BaseException as diagnostic_error:
            if bound_watch is None:
                raise
            error.add_note("Bound observer held record failed: " + repr(diagnostic_error))
        finally:
            # A started live family must finish naturally even if metadata failed.
            if owned.resume_attempted and owned.info is not None:
                owned.await_family()
        raise
    finally:
        try:
            owned.close()
        except BaseException as close_error:
            lease._record("native-observer-close-incomplete-" + uuid.uuid4().hex + ".json",
                dict(utc=_utc(), native_launch=owned.observed_binding,
                    service_receipt=owned.service_binding, retirement=outcome,
                    close_failures=deepcopy(owned.close_failures),
                    service_budget_closed=_budget_state(result, close_error, finalization_attempted),
                    no_automatic_close_retry=True))
            raise
