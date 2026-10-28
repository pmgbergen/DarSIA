"""Exact existing operational helper closure; no scientific imports."""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import re
import shutil
from .policy import CachePolicyError
RunUnavailable = CachePolicyError
MAX_METADATA_BYTES = 128 * 1024 * 1024

def _json_bytes(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("Duplicate metadata key: " + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite metadata constant: " + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Input metadata must be a JSON object")
    return value

def _integer(value, where, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(where + " must be an explicit nonnegative integer")
    return value

def _bound(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or not isinstance(value["path"], str) or not value["path"]
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None):
        raise ValueError("Exact capacity metadata path/SHA/size required")
    _integer(value["size_bytes"], "Capacity metadata bytes")
    return value

def read_bound(value):
    """Read only an explicitly supplied, bounded JSON metadata file."""
    _bound(value)
    if value["size_bytes"] > MAX_METADATA_BYTES:
        raise ValueError("Capacity metadata exceeds the existing metadata limit")
    path = Path(value["path"])
    before = path.stat()
    if before.st_size != value["size_bytes"]:
        raise ValueError("Capacity metadata SHA/size/version changed before read: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(value["size_bytes"] + 1)
    after = path.stat()
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if (version(before) != version(after) or len(raw) != value["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != value["sha256"]):
        raise ValueError("Capacity metadata SHA/size/version changed: " + str(path))
    return _json_bytes(raw), raw

def volume_info(root):
    """Fresh space and actual containing-volume metadata; no write probe."""
    path = Path(root).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    path = path.resolve(strict=True)
    stat = path.stat()
    anchor = str(path.anchor)
    if os.name == "nt":
        buffer = ctypes.create_unicode_buffer(32768)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        from ctypes import wintypes
        kernel.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        kernel.GetVolumePathNameW.restype = wintypes.BOOL
        if not kernel.GetVolumePathNameW(str(path), buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
        anchor = buffer.value
    if not stat.st_dev:
        raise RunUnavailable("OS did not identify the actual results volume")
    usage = shutil.disk_usage(path)
    return dict(volume_id=str(stat.st_dev), volume_anchor=anchor,
                free_bytes=int(usage.free), checked_container=str(path))

class _LocalVolumeLock:
    """Same OS mutex/flock primitive as ServiceLock, never a NAS lock."""
    def __init__(self, directory, key):
        self.directory, self.key = Path(directory), key
        self.handle = self.file = None

    def __enter__(self):
        if os.name == "nt":
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            kernel.CreateMutexW.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            self.handle = kernel.CreateMutexW(None, False, "Global\\DarSIACapacity-" + self.key)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            status = kernel.WaitForSingleObject(self.handle, 30000)
            if status not in (0, 0x80):
                kernel.CloseHandle(self.handle)
                self.handle = None
                raise RunUnavailable("Local capacity mutex is busy/unavailable; no claim made")
        else:
            import fcntl
            self.file = (self.directory / ("mutex-" + self.key)).open("a")
            fcntl.flock(self.file, fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.ReleaseMutex(self.handle)
            kernel.CloseHandle(self.handle)
        if self.file is not None:
            self.file.close()

def _capacity_host_context():
    """Actual OS machine-common folder/account; never process environment paths."""
    if os.name != "nt":
        raise RunUnavailable("Host-wide capacity binding is not implemented on this OS; no per-user fallback")
    from ctypes import wintypes
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    shell=ctypes.WinDLL("shell32",use_last_error=True)
    advapi=ctypes.WinDLL("advapi32",use_last_error=True)
    common=ctypes.create_unicode_buffer(32768)
    # CSIDL_COMMON_APPDATA, SHGFP_TYPE_CURRENT: OS KnownFolder compatibility API.
    shell.SHGetFolderPathW.argtypes=[wintypes.HWND,ctypes.c_int,wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR]
    shell.SHGetFolderPathW.restype=ctypes.c_long
    if shell.SHGetFolderPathW(None,0x23,None,0,common)!=0:
        raise RunUnavailable("OS CommonAppData is unavailable; no binding or claim")
    host=ctypes.create_unicode_buffer(256);length=wintypes.DWORD(len(host))
    kernel.GetComputerNameW.argtypes=[wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetComputerNameW.restype=wintypes.BOOL
    if not kernel.GetComputerNameW(host,ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.LocalFree.argtypes=[ctypes.c_void_p]
    kernel.LocalFree.restype=ctypes.c_void_p
    advapi.OpenProcessToken.argtypes=[wintypes.HANDLE,wintypes.DWORD,ctypes.POINTER(wintypes.HANDLE)]
    advapi.OpenProcessToken.restype=wintypes.BOOL
    advapi.GetTokenInformation.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
    advapi.GetTokenInformation.restype=wintypes.BOOL
    advapi.ConvertSidToStringSidW.argtypes=[ctypes.c_void_p,ctypes.POINTER(wintypes.LPWSTR)]
    advapi.ConvertSidToStringSidW.restype=wintypes.BOOL
    token=wintypes.HANDLE();sid_text=wintypes.LPWSTR()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(),0x0008,ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size=wintypes.DWORD()
        advapi.GetTokenInformation(token,1,None,0,ctypes.byref(size))  # TokenUser
        if ctypes.get_last_error()!=122 or not 0<size.value<=65536:
            raise RunUnavailable("Actual service-account token metadata is unavailable")
        data=ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token,1,data,size.value,ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        # TOKEN_USER begins with SID_AND_ATTRIBUTES; its first field is PSID.
        sid=ctypes.cast(data,ctypes.POINTER(ctypes.c_void_p)).contents.value
        if not advapi.ConvertSidToStringSidW(sid,ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        account=sid_text.value
    finally:
        if sid_text:kernel.LocalFree(ctypes.cast(sid_text,ctypes.c_void_p))
        kernel.CloseHandle(token)
    return dict(host=host.value.casefold(),service_account_sid=account,
                common_root=str(Path(common.value)/"DarSIA"/"capacity-bindings"))
