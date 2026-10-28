"""Read-only shared-folder transport admission; no rename/clock/lease experiment.

Only an explicit startup call queries the OS. Import and pure classification do
nothing. This establishes SMB/NFS transport, not J3 server-time correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys


@dataclass(frozen=True)
class FilesystemFacts:
    protocol: str
    evidence: str
    offline_cache: bool = False
    synchronized: bool = False


@dataclass(frozen=True)
class SharedFolderAdmission:
    allowed: bool
    protocol: str
    reason: str
    evidence: str
    server_mtime_verified: bool = False
    atomic_rename_tested: bool = False


@dataclass(frozen=True)
class LocalFilesystemFacts:
    filesystem: str
    local: bool | None
    evidence: str


@dataclass(frozen=True)
class LocalFileAdmission:
    allowed: bool
    filesystem: str
    reason: str
    evidence: str
    checked_container: str


def _under(path, root):
    windows = PureWindowsPath(str(path)).drive or PureWindowsPath(str(root)).drive
    left, right = str(path).replace("\\", "/").rstrip("/"), str(root).replace("\\", "/").rstrip("/")
    if windows:
        left, right = left.casefold(), right.casefold()
    return left == right or left.startswith(right + "/")


def known_sync_roots(*, environment=None, dropbox_info_paths=()):
    """Explicit bounded local client metadata; no process/cloud/network discovery."""
    env = os.environ if environment is None else environment
    roots = {env[key] for key in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial") if env.get(key)}
    for path in dropbox_info_paths:
        path = Path(path)
        try:
            with path.open("rb") as stream:
                raw = stream.read(64 * 1024 + 1)
        except FileNotFoundError:
            continue
        if len(raw) > 64 * 1024:
            raise ValueError("Dropbox local client metadata exceeds64KiB")
        doc = json.loads(raw)
        if not isinstance(doc, dict):
            raise ValueError("Invalid Dropbox local client metadata")
        for value in doc.values():
            if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"]:
                roots.add(value["path"])
    return tuple(sorted(roots))


def _known_sync_path(path, sync_roots):
    components = str(path).replace("\\", "/").casefold().split("/")
    known_component = any(re.fullmatch(r"onedrive(?: - .+)?|dropbox|google drive|googledrive|box sync|icloud drive|nextcloud|syncthing", item) for item in components)
    return known_component or any(_under(path, root) for root in sync_roots)


def classify_shared_folder(path, facts, *, sync_roots=()):
    """Pure classification; a UNC spelling or a recent mtime proves nothing."""
    if facts.synchronized or _known_sync_path(path, sync_roots):
        return SharedFolderAdmission(False, facts.protocol, "A synchronization-client folder cannot host the shared queue", facts.evidence)
    if facts.offline_cache:
        return SharedFolderAdmission(False, facts.protocol, "An offline/client-cached namespace cannot host the shared queue", facts.evidence)
    if facts.protocol.casefold() not in {"smb", "nfs"}:
        return SharedFolderAdmission(False, facts.protocol, "A positively identified SMB or NFS filesystem is required", facts.evidence)
    return SharedFolderAdmission(True, facts.protocol.casefold(), "OS-reported SMB/NFS transport; queue atomicity and server-time lease semantics are not tested by J1", facts.evidence)


def _windows_facts(path):
    # Microsoft FILE_REMOTE_PROTOCOL_INFO and FileRemoteProtocolInfo(0x0D).
    # No file is created: OPEN_EXISTING directory handle, FILE_READ_ATTRIBUTES.
    import ctypes
    from ctypes import wintypes
    class RemoteInfo(ctypes.Structure):
        _fields_ = [("version", wintypes.WORD), ("size", wintypes.WORD),
            ("protocol", wintypes.DWORD), ("major", wintypes.WORD),
            ("minor", wintypes.WORD), ("revision", wintypes.WORD),
            ("reserved", wintypes.WORD), ("flags", wintypes.DWORD),
            ("generic", wintypes.DWORD * 8), ("specific_reserved", wintypes.DWORD * 16),
            ("specific", wintypes.DWORD * 16)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    close = kernel.CloseHandle
    close.argtypes, close.restype = [wintypes.HANDLE], wintypes.BOOL
    handle = create(str(path), 0x80, 0x7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = RemoteInfo()
        info.version, info.size = 2, ctypes.sizeof(info)
        if not query(handle, 0x0D, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        protocol = {0x00020000: "smb", 0x00420000: "nfs",
                    0x00070000: "nfs", 0x00320000: "nfs"}.get(info.protocol, "unknown")
        return FilesystemFacts(protocol, f"GetFileInformationByHandleEx(FileRemoteProtocolInfo): protocol=0x{info.protocol:08x}, version={info.major}.{info.minor}",
                               offline_cache=bool(info.flags & 0x2))
    finally:
        close(handle)


def _mount_unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def linux_mount_facts(path, mountinfo):
    """Pure parser for an explicit /proc/self/mountinfo snapshot."""
    target = str(PurePosixPath(path))
    if not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError("Mount classification requires a resolved absolute path")
    candidates = []
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        fields, extra = left.split(), right.split()
        if not separator or len(fields) < 6 or len(extra) < 3:
            raise ValueError("Malformed mountinfo entry")
        mount = _mount_unescape(fields[4])
        if _under(target, mount):
            candidates.append((len(mount), mount, extra[0]))
    if not candidates:
        return FilesystemFacts("unknown", "No containing mountinfo entry")
    _, mount, fs = max(candidates)
    protocol = {"cifs": "smb", "smb3": "smb", "nfs": "nfs", "nfs4": "nfs"}.get(fs, "unknown")
    return FilesystemFacts(protocol, f"/proc/self/mountinfo: mount={mount}; fstype={fs}", synchronized=fs.startswith(("fuse.rclone", "fuse.dropbox", "fuse.onedrive")))


def inspect_shared_folder(path, *, filesystem_provider=None, sync_roots=None):
    """Explicit startup-only query. No files/claims/mtime probes are written.

    Invoke outside the UI event thread. OS network metadata calls may wait for
    an unavailable server; this function must not be used during navigation.
    """
    original_path = Path(path).expanduser()
    if sync_roots is None:
        infos = []
        for variable in ("APPDATA", "LOCALAPPDATA"):
            if os.environ.get(variable):
                infos.append(Path(os.environ[variable]) / "Dropbox/info.json")
        if os.name != "nt":
            infos.append(Path.home() / ".dropbox/info.json")
        sync_roots = known_sync_roots(dropbox_info_paths=infos)
    if _known_sync_path(original_path, sync_roots):
        return classify_shared_folder(original_path, FilesystemFacts("unknown", "Explicit selected path and known local sync-client locations"), sync_roots=sync_roots)
    path = original_path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Shared job root must be an existing directory")
    if filesystem_provider is not None:
        facts = filesystem_provider(path)
    elif os.name == "nt":
        facts = _windows_facts(path)
    elif sys.platform.startswith("linux"):
        with Path("/proc/self/mountinfo").open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Mount metadata exceeds1MiB")
        facts = linux_mount_facts(path, raw.decode("utf-8"))
    else:
        facts = FilesystemFacts("unknown", "No verified SMB/NFS metadata provider on this platform")
    return classify_shared_folder(path, facts, sync_roots=sync_roots)


def _read_mountinfo():
    with Path("/proc/self/mountinfo").open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Mount metadata exceeds 1 MiB")
    return raw.decode("utf-8")


def linux_local_facts(path, mountinfo):
    """Positive, conservative local-FS classification of an explicit snapshot."""
    target = str(PurePosixPath(path))
    if not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError("Locality requires a resolved absolute path")
    candidates = []
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        fields, extra = left.split(), right.split()
        if not separator or len(fields) < 6 or len(extra) < 3:
            raise ValueError("Malformed mountinfo entry")
        mount = _mount_unescape(fields[4])
        if _under(target, mount):
            candidates.append((len(PurePosixPath(mount).parts), mount, extra[0]))
    if not candidates:
        return LocalFilesystemFacts("unknown", None, "No containing mountinfo entry")
    depth = max(row[0] for row in candidates)
    deepest = {row[1:] for row in candidates if row[0] == depth}
    if len(deepest) != 1:
        return LocalFilesystemFacts("unknown", None, "Ambiguous deepest mountinfo entries")
    mount, filesystem = next(iter(deepest))
    evidence = f"/proc/self/mountinfo: mount={mount}; fstype={filesystem}"
    if filesystem in {"cifs", "smb3", "nfs", "nfs4", "9p", "ceph", "afs", "davfs", "fuse.sshfs"}:
        return LocalFilesystemFacts(filesystem, False, evidence)
    # Unknown/stacked/FUSE filesystems remain unknown: a local-looking path is
    # not evidence of its backing transport. More providers need explicit proof.
    known_local = {"ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs",
                   "tmpfs", "ramfs", "vfat", "exfat", "ntfs", "ntfs3", "ufs"}
    return LocalFilesystemFacts(filesystem, True if filesystem in known_local else None, evidence)


def windows_mapped_drive_target(drive):
    """Read one current-token drive mapping; never connect, enumerate or write."""
    if not isinstance(drive, str) or re.fullmatch(r"[A-Z]:", drive) is None:
        raise ValueError("One explicit uppercase drive letter and colon required")
    import ctypes
    from ctypes import wintypes
    try:
        mpr = ctypes.WinDLL("mpr", use_last_error=True)
    except (AttributeError, OSError) as error:
        raise OSError("Current-token Windows drive mapping query unavailable") from error
    query = mpr.WNetGetConnectionW
    query.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    length = wintypes.DWORD(len(buffer))
    status = int(query(drive, buffer, ctypes.byref(length)))
    if status != 0:
        raise OSError(status, "Current-token drive mapping unavailable: " + drive)
    if not 0 < length.value <= len(buffer) or not buffer.value:
        raise OSError("Current-token drive mapping returned no bounded target")
    return buffer.value


def _windows_local_facts(container):
    """Query the containing volume, including mounted volumes and mapped drives."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    volume_path = kernel.GetVolumePathNameW
    volume_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    volume_path.restype = wintypes.BOOL
    drive_type = kernel.GetDriveTypeW
    drive_type.argtypes, drive_type.restype = [wintypes.LPCWSTR], wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    if not volume_path(str(container), buffer, len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error())
    kind = int(drive_type(buffer.value))
    names = {0: "unknown", 1: "no-root", 2: "removable", 3: "fixed",
             4: "remote", 5: "cdrom", 6: "ramdisk"}
    return LocalFilesystemFacts(names.get(kind, "unknown"),
        True if kind in {2, 3, 5, 6} else False if kind == 4 else None,
        f"GetVolumePathNameW/GetDriveTypeW: volume={buffer.value}; drive_type={kind}")


def inspect_local_file(path, *, locality_provider=None, sync_roots=None):
    """Explicit local-config admission; never writes or probes at import time.

    A missing final file is allowed for explicit creation only when an existing
    ancestor has positive local-FS evidence. The caller still handles file CAS.
    OS/network queries run outside the UI event thread.
    """
    original = Path(path).expanduser()
    if sync_roots is None:
        infos = [Path(os.environ[key]) / "Dropbox/info.json"
                 for key in ("APPDATA", "LOCALAPPDATA") if os.environ.get(key)]
        if os.name != "nt":
            infos.append(Path.home() / ".dropbox/info.json")
        sync_roots = known_sync_roots(dropbox_info_paths=infos)
    if _known_sync_path(original, sync_roots):
        return LocalFileAdmission(False, "synchronized", "A per-machine store map cannot be a synchronization-client file", "Selected path and known local sync-client locations", str(original.parent))
    actual = original.resolve()
    if _known_sync_path(actual, sync_roots):
        return LocalFileAdmission(False, "synchronized", "The store map resolves into a synchronization-client folder", "Resolved path and known local sync-client locations", str(actual.parent))
    container = actual.parent
    while not container.exists() and container != container.parent:
        container = container.parent
    container = container.resolve(strict=True)
    if not container.is_dir():
        raise ValueError("The store map has no existing parent directory")
    if locality_provider is not None:
        facts = locality_provider(container)
    elif os.name == "nt":
        facts = _windows_local_facts(container)
    elif sys.platform.startswith("linux"):
        facts = linux_local_facts(container, _read_mountinfo())
    else:
        facts = LocalFilesystemFacts("unknown", None, "No verified local-filesystem provider on this platform")
    if facts.local is not True:
        reason = "The store map must be on a positively identified local filesystem"
        return LocalFileAdmission(False, facts.filesystem, reason, facts.evidence, str(container))
    return LocalFileAdmission(True, facts.filesystem, "OS-reported local filesystem; no write/space guarantee claimed", facts.evidence, str(container))


@dataclass(frozen=True)
class ResultsRootAdmission:
    """Output-only OS evidence; a remote result root is never called local."""

    allowed: bool
    protocol: str
    local: bool
    reason: str
    evidence: str
    checked_container: str
    results_root: str
    source_admission: object


def inspect_results_root(root, *, locality_provider=None, filesystem_provider=None,
                         sync_roots=None):
    """Inspect one explicit output root without creating any directory or file.

    Local settings and input rules still use their original separate guards.
    An explicit remote result needs positive SMB/NFS transport evidence; a UNC
    spelling by itself is not admission. No lease or throughput guarantee is made.
    """
    selected = Path(root).expanduser().resolve()
    container = selected
    while not container.exists() and container != container.parent:
        container = container.parent
    container = container.resolve(strict=True)
    if not container.is_dir() or not selected.is_relative_to(container):
        raise ValueError("Results root has no containing existing directory")
    local = inspect_local_file(selected / ".darsia-location-probe",
                               locality_provider=locality_provider, sync_roots=sync_roots)
    if local.allowed and not PureWindowsPath(str(selected)).drive.startswith("\\\\"):
        return ResultsRootAdmission(True, local.filesystem, True,
            "OS-reported local results filesystem", local.evidence,
            local.checked_container, str(selected), local)
    remote = inspect_shared_folder(container, filesystem_provider=filesystem_provider,
                                   sync_roots=sync_roots)
    return ResultsRootAdmission(remote.allowed, remote.protocol, False,
        "Explicit remote results filesystem: " + remote.reason, remote.evidence,
        str(container), str(selected), remote)


def valid_results_admission(admission, root):
    """Accept positive output proofs only, retaining the old local-only proof."""
    try:
        if (isinstance(admission, LocalFileAdmission) or
                isinstance(admission, ResultsRootAdmission) and admission.local is True):
            if PureWindowsPath(str(root)).drive.startswith("\\\\"):
                return False
        actual_root = Path(root).resolve()
        checked = Path(admission.checked_container).resolve(strict=True)
        if not checked.is_dir() or not actual_root.is_relative_to(checked):
            return False
        invalid = {"smb", "nfs", "remote", "unknown", "synchronized", "offline", "cloud"}
        if isinstance(admission, LocalFileAdmission):
            return (admission.allowed is True and isinstance(admission.filesystem, str)
                    and bool(admission.filesystem) and admission.filesystem.casefold() not in invalid
                    and not PureWindowsPath(str(root)).drive.startswith("\\\\"))
        if (not isinstance(admission, ResultsRootAdmission) or admission.allowed is not True
                or type(admission.local) is not bool or not isinstance(admission.protocol, str)
                or not admission.protocol or Path(admission.results_root).resolve() != actual_root):
            return False
        source = admission.source_admission
        if admission.local:
            return (isinstance(source, LocalFileAdmission) and source.allowed is True
                    and source.filesystem == admission.protocol
                    and source.filesystem.casefold() not in invalid
                    and Path(source.checked_container).resolve(strict=True) == checked)
        return (isinstance(source, SharedFolderAdmission) and source.allowed is True
                and source.protocol == admission.protocol and source.protocol in {"smb", "nfs"})
    except (AttributeError, OSError, ValueError, TypeError):
        return False
