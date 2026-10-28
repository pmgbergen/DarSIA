"""Prepared Windows local-NTFS cache primitive. Not runtime-qualified.

No path unlink/rmdir, reparse following, historical-scope adoption, ACL changes,
privilege enablement or POSIX fallback. All mutation below an admitted root is
relative to pinned directory handles. Callers still own quota/publication and
actual process/reader retirement. This module is not a filesystem hard quota.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
import os
from pathlib import PureWindowsPath
import re
import uuid

from .policy import CachePolicyError


class CacheNamespaceUnsupported(CachePolicyError):
    pass


class CacheCleanupIncomplete(CachePolicyError):
    def __init__(self, message, removed, mutations=(), close_failures=()):
        super().__init__(message)
        self.removed_owned = tuple(removed)
        self.mutations = tuple(mutations)
        self.close_failures = tuple(close_failures)


class CacheObjectIncomplete(CachePolicyError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.created_evidence = dict(evidence)


class CacheHandleCloseIncomplete(CachePolicyError):
    def __init__(self, failures):
        super().__init__("Handle close failed; completion/retirement is not established")
        self.close_failures = tuple(failures)


class CacheWriteIncomplete(CacheObjectIncomplete):
    def __init__(self, message, identity, evidence):
        super().__init__(message, evidence)
        self.partial_identity = identity


def _component(name):
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or any(x in name for x in "\\/:\x00") or name[-1] in " ."
            or any(ord(x) < 32 for x in name)
            or name.split(".", 1)[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *["COM"+str(i) for i in range(1, 10)],
             *["LPT"+str(i) for i in range(1, 10)]}):
        raise CachePolicyError("One non-reserved relative name component required")
    if len(name.encode("utf-16-le")) > 510:
        raise CachePolicyError("Cache name exceeds one NTFS component")
    return name


def _parts(relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise CachePolicyError("Portable relative cache name required")
    return tuple(_component(x) for x in relative.split("/"))


class _UNICODE_STRING(C.Structure):
    _fields_ = [("Length", W.USHORT), ("MaximumLength", W.USHORT), ("Buffer", W.LPWSTR)]


class _OBJECT_ATTRIBUTES(C.Structure):
    _fields_ = [("Length", W.ULONG), ("RootDirectory", W.HANDLE),
                ("ObjectName", C.POINTER(_UNICODE_STRING)), ("Attributes", W.ULONG),
                ("SecurityDescriptor", W.LPVOID), ("SecurityQualityOfService", W.LPVOID)]


class _IO_UNION(C.Union):
    _fields_ = [("Status", W.LONG), ("Pointer", W.LPVOID)]


class _IO_STATUS_BLOCK(C.Structure):
    _fields_ = [("result", _IO_UNION), ("Information", C.c_size_t)]


class _FILE_ID_INFO(C.Structure):
    _fields_ = [("VolumeSerialNumber", C.c_uint64), ("FileId", C.c_ubyte * 16)]


class _FILE_ATTRIBUTE_TAG_INFO(C.Structure):
    _fields_ = [("FileAttributes", W.DWORD), ("ReparseTag", W.DWORD)]


class _FILE_STANDARD_INFO(C.Structure):
    _fields_ = [("AllocationSize", C.c_int64), ("EndOfFile", C.c_int64),
                ("NumberOfLinks", W.DWORD), ("DeletePending", C.c_ubyte),
                ("Directory", C.c_ubyte)]


class _FILE_BASIC_INFO(C.Structure):
    _fields_ = [("CreationTime", C.c_int64), ("LastAccessTime", C.c_int64),
                ("LastWriteTime", C.c_int64), ("ChangeTime", C.c_int64),
                ("FileAttributes", W.DWORD)]


class _FILE_ID_EXTD_DIR_INFO(C.Structure):
    _fields_ = [("NextEntryOffset", W.ULONG), ("FileIndex", W.ULONG),
                ("CreationTime", C.c_int64), ("LastAccessTime", C.c_int64),
                ("LastWriteTime", C.c_int64), ("ChangeTime", C.c_int64),
                ("EndOfFile", C.c_int64), ("AllocationSize", C.c_int64),
                ("FileAttributes", W.ULONG), ("FileNameLength", W.ULONG),
                ("EaSize", W.ULONG), ("ReparsePointTag", W.ULONG),
                ("FileId", C.c_ubyte * 16), ("FileName", W.WCHAR * 1)]


@dataclass(frozen=True)
class ObjectIdentity:
    volume_serial: int
    file_id_128: str
    directory: bool
    bytes: int
    last_write_100ns: int
    creation_100ns: int
    change_100ns: int

    def document(self):
        return dict(volume_serial=self.volume_serial, file_id_128=self.file_id_128,
                    directory=self.directory, bytes=self.bytes,
                    last_write_100ns=self.last_write_100ns,
                    creation_100ns=self.creation_100ns, change_100ns=self.change_100ns)

    def same_object(self, expected):
        return (self.volume_serial == expected["volume_serial"]
                and self.file_id_128 == expected["file_id_128"]
                and self.directory is expected["directory"]
                and self.creation_100ns == expected["creation_100ns"])


class _API:
    def __init__(self):
        if os.name != "nt" or C.sizeof(C.c_void_p) != 8:
            raise CacheNamespaceUnsupported("Only reviewed Windows x64 local NTFS is supported")
        self.k = C.WinDLL("kernel32", use_last_error=True)
        self.n = C.WinDLL("ntdll", use_last_error=True)
        self.k.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, W.LPVOID,
                                       W.DWORD, W.DWORD, W.HANDLE]
        self.k.CreateFileW.restype = W.HANDLE
        self.k.CloseHandle.argtypes = [W.HANDLE]
        self.k.CloseHandle.restype = W.BOOL
        self.k.GetFileInformationByHandleEx.argtypes = [W.HANDLE, C.c_int, W.LPVOID, W.DWORD]
        self.k.GetFileInformationByHandleEx.restype = W.BOOL
        self.k.SetFileInformationByHandle.argtypes = [W.HANDLE, C.c_int, W.LPVOID, W.DWORD]
        self.k.SetFileInformationByHandle.restype = W.BOOL
        self.k.GetVolumeNameForVolumeMountPointW.argtypes = [W.LPCWSTR, W.LPWSTR, W.DWORD]
        self.k.GetVolumeNameForVolumeMountPointW.restype = W.BOOL
        self.k.GetDriveTypeW.argtypes = [W.LPCWSTR]
        self.k.GetDriveTypeW.restype = W.UINT
        self.k.GetVolumeInformationByHandleW.argtypes = [W.HANDLE, W.LPWSTR, W.DWORD,
            C.POINTER(W.DWORD), C.POINTER(W.DWORD), C.POINTER(W.DWORD), W.LPWSTR, W.DWORD]
        self.k.GetVolumeInformationByHandleW.restype = W.BOOL
        self.k.WriteFile.argtypes = [W.HANDLE, C.c_void_p, W.DWORD, C.POINTER(W.DWORD), W.LPVOID]
        self.k.WriteFile.restype = W.BOOL
        self.k.ReadFile.argtypes = [W.HANDLE, W.LPVOID, W.DWORD, C.POINTER(W.DWORD), W.LPVOID]
        self.k.ReadFile.restype = W.BOOL
        self.k.FlushFileBuffers.argtypes = [W.HANDLE]
        self.k.FlushFileBuffers.restype = W.BOOL
        self.n.NtCreateFile.argtypes = [C.POINTER(W.HANDLE), W.DWORD,
            C.POINTER(_OBJECT_ATTRIBUTES), C.POINTER(_IO_STATUS_BLOCK), W.LPVOID,
            W.ULONG, W.ULONG, W.ULONG, W.ULONG, W.LPVOID, W.ULONG]
        self.n.NtCreateFile.restype = W.LONG

    def check(self, ok, operation):
        if not ok:
            raise OSError(C.get_last_error(), operation)

    def close(self, handle):
        if not self.k.CloseHandle(handle):
            value = handle if isinstance(handle, int) else handle.value
            raise CacheHandleCloseIncomplete([dict(handle=value, winerror=C.get_last_error())])

    def info(self, handle, kind, structure):
        value = structure()
        self.check(self.k.GetFileInformationByHandleEx(handle, kind, C.byref(value), C.sizeof(value)),
                   "Cache handle information unavailable; no fallback")
        return value

    def identity(self, handle):
        tag = self.info(handle, 9, _FILE_ATTRIBUTE_TAG_INFO)
        standard = self.info(handle, 1, _FILE_STANDARD_INFO)
        fid = self.info(handle, 18, _FILE_ID_INFO)
        basic = self.info(handle, 0, _FILE_BASIC_INFO)
        if tag.FileAttributes & 0x400 or tag.ReparseTag or standard.DeletePending:
            raise CachePolicyError("Reparse or delete-pending object preserved")
        if not standard.Directory and standard.NumberOfLinks != 1:
            raise CachePolicyError("Foreign/ambiguous hardlink object preserved")
        return ObjectIdentity(fid.VolumeSerialNumber, bytes(fid.FileId).hex(),
                              bool(standard.Directory), standard.EndOfFile,
                              basic.LastWriteTime, basic.CreationTime, basic.ChangeTime)

    def child(self, parent, name, *, directory, create=False, mutable=False, write=False):
        _component(name)
        parent_identity = self.identity(parent).document()
        buffer = C.create_unicode_buffer(name)
        encoded = len(name.encode("utf-16-le"))
        string = _UNICODE_STRING(encoded, encoded + 2, C.cast(buffer, W.LPWSTR))
        attrs = _OBJECT_ATTRIBUTES(C.sizeof(_OBJECT_ATTRIBUTES), parent, C.pointer(string),
                                   0x40 | 0x1000, None, None)  # CASE_INSENSITIVE, DONT_REPARSE
        result, iosb = W.HANDLE(), _IO_STATUS_BLOCK()
        access = 0x80 | 0x100000  # READ_ATTRIBUTES, SYNCHRONIZE
        access |= (0x1 | 0x20) if directory else 0
        if mutable:
            access |= 0x10000  # DELETE, including directories opened for later disposition
        if write:
            access |= 0x2
        elif not directory:
            access |= 0x1  # READ_DATA for bounded metadata reads
        options = 0x20 | 0x200000 | (0x1 if directory else 0x40)
        # Every mutable target, including directories, is exclusive. Read-only
        # namespace ancestors permit readers but no WRITE/DELETE sharing.
        status = self.n.NtCreateFile(C.byref(result), access, C.byref(attrs), C.byref(iosb),
                                    None, 0x80, 0x1 if directory and not mutable else 0,
                                    2 if create else 1, options, None, 0)
        evidence = dict(parent_identity=parent_identity, component=name, directory=directory,
                        create_requested=create, handle_returned=bool(result.value), ntstatus=status & 0xffffffff,
                        nt_success=status == 0, new_object_created=bool(create) if status == 0 else None,
                        io_information=int(iosb.Information) if status == 0 else None,
                        handle_close_confirmed=False)
        if status != 0:
            if result.value:
                try:
                    self.close(result)
                    evidence["handle_close_confirmed"] = True
                except CacheHandleCloseIncomplete as error:
                    evidence["close_failures"] = list(error.close_failures)
            raise CacheObjectIncomplete("Relative NtCreateFile refused; no fallback", evidence)
        try:
            identity = self.identity(result)
            if identity.directory is not directory:
                raise CachePolicyError("Frozen cache object type differs")
            return result, identity
        except BaseException as error:
            try:
                self.close(result)
                evidence["handle_close_confirmed"] = True
            except CacheHandleCloseIncomplete as close_error:
                evidence["close_failures"] = list(close_error.close_failures)
            raise CacheObjectIncomplete(str(error), evidence) from error

    def names(self, handle, *, with_types=False):
        names, folded, restart = [], set(), True
        offset_name = _FILE_ID_EXTD_DIR_INFO.FileName.offset
        while True:
            buffer = C.create_string_buffer(65536)
            ok = self.k.GetFileInformationByHandleEx(handle, 20 if restart else 19,
                                                    buffer, len(buffer))
            restart = False
            if not ok:
                error = C.get_last_error()
                if error == 18:  # ERROR_NO_MORE_FILES
                    return names
                raise OSError(error, "Pinned directory enumeration unavailable; no fallback")
            offset = 0
            while True:
                if offset + C.sizeof(_FILE_ID_EXTD_DIR_INFO) > len(buffer):
                    raise CachePolicyError("Truncated native directory record")
                row = _FILE_ID_EXTD_DIR_INFO.from_buffer_copy(buffer.raw[offset:offset+C.sizeof(_FILE_ID_EXTD_DIR_INFO)])
                length = int(row.FileNameLength)
                if length % 2 or offset + offset_name + length > len(buffer):
                    raise CachePolicyError("Invalid native filename length")
                name = buffer.raw[offset+offset_name:offset+offset_name+length].decode("utf-16-le", "strict")
                if name not in {".", ".."}:
                    _component(name)
                    if name.casefold() in folded or len(names) >= 65536:
                        raise CachePolicyError("Duplicate/oversized frozen directory inventory")
                    folded.add(name.casefold())
                    reparse = bool(row.FileAttributes & 0x400 or row.ReparsePointTag)
                    if reparse and not with_types:
                        raise CachePolicyError("Unowned reparse entry preserved")
                    names.append(dict(name=name, directory=bool(row.FileAttributes & 0x10),
                                      reparse=reparse) if with_types else name)
                step = int(row.NextEntryOffset)
                if not step:
                    break
                if step % 8 or step < offset_name + length or offset + step >= len(buffer):
                    raise CachePolicyError("Invalid native directory record offset")
                offset += step

    def dispose(self, handle):
        pending = C.c_ubyte(1)  # FILE_DISPOSITION_INFO.Boolean, no POSIX/ignore-readonly flags
        self.check(self.k.SetFileInformationByHandle(handle, 4, C.byref(pending), 1),
                   "Identity-bound disposition refused; no path fallback")



class _CAP_GUID(C.Structure):
    _fields_ = [("Data1", C.c_uint32), ("Data2", C.c_uint16), ("Data3", C.c_uint16), ("Data4", C.c_ubyte*8)]


class _CAP_STORAGE_TYPE(C.Structure):
    _fields_ = [("DeviceId", C.c_uint32), ("VendorId", _CAP_GUID)]


class _CAP_OPEN_V2(C.Structure):
    _fields_ = [("GetInfoOnly", C.c_int32), ("ReadOnly", C.c_int32), ("ResiliencyGuid", _CAP_GUID)]


class _CAP_OPEN_UNION(C.Union):
    _fields_ = [("Version1RWDepth", C.c_uint32), ("Version2", _CAP_OPEN_V2)]


class _CAP_OPEN(C.Structure):
    _fields_ = [("Version", C.c_uint32), ("value", _CAP_OPEN_UNION)]


class _CAP_SIZE(C.Structure):
    _fields_ = [("VirtualSize", C.c_uint64), ("PhysicalSize", C.c_uint64),
                ("BlockSize", C.c_uint32), ("SectorSize", C.c_uint32)]


class _CAP_INFO_UNION(C.Union):
    _fields_ = [("Size", _CAP_SIZE), ("Identifier", _CAP_GUID),
                ("Storage", _CAP_STORAGE_TYPE), ("UInt32", C.c_uint32)]


class _CAP_INFO(C.Structure):
    _fields_ = [("Version", C.c_uint32), ("value", _CAP_INFO_UNION)]


class _CAP_EXTENT(C.Structure):
    _fields_ = [("DiskNumber", C.c_uint32), ("StartingOffset", C.c_int64), ("ExtentLength", C.c_int64)]


class _CAP_EXTENTS_ONE(C.Structure):
    _fields_ = [("NumberOfDiskExtents", C.c_uint32), ("Extent", _CAP_EXTENT)]


def _match_volume_cap(binding, observed):
    """Pure comparison of the fixed native-query result, not an admission API."""
    if (observed["volume_guid"].casefold() != binding.volume_guid.casefold()
            or observed["virtual_disk_identifier"] != binding.virtual_disk_identifier
            or observed["virtual_size_bytes"] != binding.maximum_size_bytes
            or observed["storage_device_id"] != 3 or observed["provider_subtype"] != 2
            or observed["is_loaded"] is not True or observed["filesystem"] != "NTFS"
            or observed["extent_count"] != 1
            or observed["volume_disk_number"] != observed["vhdx_disk_number"]
            or not 0 <= observed["extent_start"] < binding.maximum_size_bytes
            or not 0 < observed["volume_total_bytes"] <= observed["extent_bytes"]
            or observed["extent_start"] + observed["extent_bytes"] > binding.maximum_size_bytes):
        raise CachePolicyError("Attached cache volume/backing fixed VHDX/cap differs; no fallback")
    return dict(observed, format="darsia-readonly-fixed-vhdx-cap-observation-v1",
                binding=binding.document(), verified=True,
                semantics="kernel fixed-volume capacity, not a per-user quota or complete writer-coverage proof")


class _VolumeCapQueries:
    """Only getters. No Create/Attach/Detach/Expand/Resize or privilege API.

    This deliberately supports one fixed VHDX and one nonspanning NTFS volume.
    Unsupported access/ABI/extent/identity is refusal, never another query route.
    """
    def __init__(self, namespace):
        self.ns, self.api = namespace, namespace.api
        if (C.sizeof(_CAP_GUID), C.sizeof(_CAP_OPEN), _CAP_INFO.value.offset,
                C.sizeof(_CAP_INFO), _CAP_EXTENTS_ONE.Extent.offset, C.sizeof(_CAP_EXTENTS_ONE)) != (16,28,8,32,8,32):
            raise CacheNamespaceUnsupported("Reviewed x64 VirtDisk/extents ABI required")
        self.v = C.WinDLL("virtdisk", use_last_error=True)
        self.v.OpenVirtualDisk.argtypes = [C.POINTER(_CAP_STORAGE_TYPE), W.LPCWSTR,
            C.c_uint32, C.c_uint32, C.POINTER(_CAP_OPEN), C.POINTER(W.HANDLE)]
        self.v.OpenVirtualDisk.restype = C.c_uint32
        self.v.GetVirtualDiskInformation.argtypes = [W.HANDLE, C.POINTER(C.c_uint32),
            C.POINTER(_CAP_INFO), C.POINTER(C.c_uint32)]
        self.v.GetVirtualDiskInformation.restype = C.c_uint32
        self.v.GetVirtualDiskPhysicalPath.argtypes = [W.HANDLE, C.POINTER(C.c_uint32), W.LPWSTR]
        self.v.GetVirtualDiskPhysicalPath.restype = C.c_uint32
        self.api.k.DeviceIoControl.argtypes = [W.HANDLE, C.c_uint32, W.LPVOID,
            C.c_uint32, W.LPVOID, C.c_uint32, C.POINTER(C.c_uint32), W.LPVOID]
        self.api.k.DeviceIoControl.restype = W.BOOL
        self.api.k.GetDiskFreeSpaceExW.argtypes = [W.LPCWSTR, *([C.POINTER(C.c_uint64)]*3)]
        self.api.k.GetDiskFreeSpaceExW.restype = W.BOOL

    @staticmethod
    def status(value, label):
        if value:
            raise OSError(int(value), label + "; no fallback or privilege adjustment")

    def info(self, handle, version):
        info, size = _CAP_INFO(), C.c_uint32(C.sizeof(_CAP_INFO))
        info.Version = version
        self.status(self.v.GetVirtualDiskInformation(handle,C.byref(size),C.byref(info),None), "Read VHDX information")
        if info.Version != version or not 8 <= size.value <= C.sizeof(info):
            raise CacheNamespaceUnsupported("Unexpected VirtDisk information shape")
        return info.value

    def pin_image_attributes(self, parent, name):
        # An attached provider legitimately writes its VHDX. READ_ATTRIBUTES only;
        # allow READ/WRITE sharing, forbid DELETE, and never read/hash the payload.
        _component(name)
        buf=C.create_unicode_buffer(name); n=len(name.encode("utf-16-le"))
        string=_UNICODE_STRING(n,n+2,C.cast(buf,W.LPWSTR))
        attrs=_OBJECT_ATTRIBUTES(C.sizeof(_OBJECT_ATTRIBUTES),parent.handle,C.pointer(string),0x1040,None,None)
        result,iosb=W.HANDLE(),_IO_STATUS_BLOCK()
        status=self.api.n.NtCreateFile(C.byref(result),0x80|0x100000,C.byref(attrs),C.byref(iosb),
            None,0x80,3,1,0x20|0x200000|0x40,None,0)
        if status != 0:
            if result.value:
                self.api.close(result)
            raise OSError(int(status)&0xffffffff,"Read-only no-follow VHDX identity open refused")
        return result

    def observe(self, binding):
        parent=None; image=W.HANDLE(); disk=W.HANDLE(); volume=W.HANDLE()
        error=None; result=None; close_failures=[]
        try:
            path=PureWindowsPath(binding.image_path)
            parent=WindowsNamespace(str(path.parent),binding.image_parent_identity)
            image=self.pin_image_attributes(parent,path.name)
            before=self.api.identity(image)
            tag=self.api.info(image,9,_FILE_ATTRIBUTE_TAG_INFO)
            if (not before.same_object(binding.image_identity) or tag.FileAttributes & (0x200|0x800|0x4000)):
                raise CachePolicyError("VHDX identity/sparse/compression/encryption differs")
            storage=_CAP_STORAGE_TYPE()  # UNKNOWN selects by .vhdx, then actual format must be3.
            params=_CAP_OPEN();params.Version=2
            params.value.Version2.GetInfoOnly=1;params.value.Version2.ReadOnly=1
            self.status(self.v.OpenVirtualDisk(C.byref(storage),str(path),0,1,C.byref(params),C.byref(disk)),
                        "Open VHDX GetInfoOnly/ReadOnly V2 ACCESS_NONE NO_PARENTS")
            initial=(str(uuid.UUID(bytes_le=bytes(self.info(disk,2).Identifier))),
                     int(self.info(disk,1).Size.VirtualSize),int(self.info(disk,6).Storage.DeviceId),
                     int(self.info(disk,7).UInt32),int(self.info(disk,13).UInt32))
            physical=C.create_unicode_buffer(128); length=C.c_uint32(C.sizeof(physical))
            self.status(self.v.GetVirtualDiskPhysicalPath(disk,C.byref(length),physical),"Read attached VHDX physical path")
            matched=re.fullmatch(r"\\\\\.\\PhysicalDrive([0-9]{1,10})",physical.value)
            if not matched or length.value > C.sizeof(physical):
                raise CacheNamespaceUnsupported("Unexpected attached VHDX device path")
            # Open the exact pinned GUID device for this one read-only IOCTL.
            # No drive-number opening, mount changes or raw disk data access.
            volume=W.HANDLE(self.api.k.CreateFileW(self.ns.volume_guid[:-1],0,3,None,3,0x00200000,None))
            if volume.value in (None,C.c_void_p(-1).value):
                raise OSError(C.get_last_error(),"Cannot query exact attached VolumeGUID device")
            ext,returned=_CAP_EXTENTS_ONE(),C.c_uint32()
            self.api.check(self.api.k.DeviceIoControl(volume,0x00560000,None,0,C.byref(ext),C.sizeof(ext),C.byref(returned),None),
                           "One-volume disk extent query refused; no spanning fallback")
            if returned.value != C.sizeof(ext) or ext.NumberOfDiskExtents != 1:
                raise CacheNamespaceUnsupported("Exactly one complete NTFS volume extent required")
            free,total,total_free=C.c_uint64(),C.c_uint64(),C.c_uint64()
            self.api.check(self.api.k.GetDiskFreeSpaceExW(self.ns.volume_guid,C.byref(free),C.byref(total),C.byref(total_free)),
                           "Read actual capped filesystem size")
            final=(str(uuid.UUID(bytes_le=bytes(self.info(disk,2).Identifier))),
                   int(self.info(disk,1).Size.VirtualSize),int(self.info(disk,6).Storage.DeviceId),
                   int(self.info(disk,7).UInt32),int(self.info(disk,13).UInt32))
            if initial != final or not self.api.identity(image).same_object(binding.image_identity):
                raise CachePolicyError("VHDX identity/cap changed while querying")
            result=dict(volume_guid=self.ns.volume_guid,virtual_disk_identifier=initial[0],virtual_size_bytes=initial[1],
                storage_device_id=initial[2],provider_subtype=initial[3],is_loaded=initial[4]==1,filesystem="NTFS",
                extent_count=int(ext.NumberOfDiskExtents),volume_disk_number=int(ext.Extent.DiskNumber),
                vhdx_disk_number=int(matched.group(1)),extent_start=int(ext.Extent.StartingOffset),
                extent_bytes=int(ext.Extent.ExtentLength),volume_total_bytes=int(total.value))
            result=_match_volume_cap(binding,result)
        except BaseException as caught:
            error=caught
        finally:
            for handle in (volume,disk,image):
                if handle.value not in (None,C.c_void_p(-1).value):
                    try:self.api.close(handle)
                    except CacheHandleCloseIncomplete as failed:close_failures.extend(failed.close_failures)
            if parent is not None:
                try:parent.close()
                except CacheHandleCloseIncomplete as failed:close_failures.extend(failed.close_failures)
        if close_failures:
            raise CacheHandleCloseIncomplete(close_failures) from error
        if error is not None:raise error
        return result


class WindowsNamespace:
    """Pinned existing root. Root identity must come from accepted host admission.

    The trusted root is never deleted. Ancestors are opened one component at a
    time from an actual fixed local NTFS volume GUID root; every handle remains
    open. A scope is either freshly created or reopened against its frozen ID.
    """
    def __init__(self, root, expected_root):
        self.api, self.ancestors, self.closed = _API(), [], False
        path = PureWindowsPath(root)
        if (not path.is_absolute() or not re.fullmatch(r"[A-Za-z]:", path.drive)
                or len(path.parts) < 2 or not isinstance(expected_root, dict)):
            raise CacheNamespaceUnsupported("Admitted drive-root-relative directory and kernel identity required")
        try:
            if self.api.k.GetDriveTypeW(path.anchor) != 3:
                raise CacheNamespaceUnsupported("Only fixed local volumes are supported")
            volume = C.create_unicode_buffer(64)
            self.api.check(self.api.k.GetVolumeNameForVolumeMountPointW(path.anchor, volume, len(volume)),
                           "Actual volume GUID unavailable")
            handle = self.api.k.CreateFileW(volume.value, 0x80 | 0x1 | 0x20 | 0x100000,
                                            0x1, None, 3, 0x02000000 | 0x00200000, None)
            if handle in (None, C.c_void_p(-1).value):
                raise OSError(C.get_last_error(), "Cannot pin volume root")
            self.ancestors.append(handle)
            fs = C.create_unicode_buffer(32)
            serial, maximum, flags = W.DWORD(), W.DWORD(), W.DWORD()
            self.api.check(self.api.k.GetVolumeInformationByHandleW(handle, None, 0,
                C.byref(serial), C.byref(maximum), C.byref(flags), fs, len(fs)), "Cannot verify NTFS")
            if fs.value != "NTFS":
                raise CacheNamespaceUnsupported("Only local NTFS is currently implemented")
            anchor_id = self.api.identity(handle)
            for part in path.parts[1:]:
                handle, identity = self.api.child(handle, part, directory=True)
                self.ancestors.append(handle)
                if identity.volume_serial != anchor_id.volume_serial:
                    raise CachePolicyError("Cache namespace changed physical volume")
            if not identity.same_object(expected_root):
                raise CachePolicyError("Configured cache/control root kernel identity changed")
            self.handle, self.identity = handle, identity
            self.volume_guid, self.volume_serial32 = volume.value, int(serial.value)
        except BaseException:
            self.close()
            raise

    def verify_volume_cap(self, binding):
        if (self.closed or not self.identity.same_object(binding.cache_identity)
                or self.volume_guid.casefold() != binding.volume_guid.casefold()):
            raise CachePolicyError("Capped root differs from the accepted cache identity")
        return _VolumeCapQueries(self).observe(binding)

    def close(self):
        if not self.closed:
            failures = []
            while self.ancestors:
                handle = self.ancestors.pop()
                try:
                    self.api.close(handle)
                except CacheHandleCloseIncomplete as error:
                    failures.extend(error.close_failures)
            self.closed = True
            if failures:
                raise CacheHandleCloseIncomplete(failures)

    def scope(self, name, *, create=False, expected=None):
        if self.closed:
            raise CachePolicyError("Closed namespace cannot allocate")
        handle, identity = self.api.child(self.handle, name, directory=True,
                                          create=create, mutable=True)
        if not create and (not expected or not identity.same_object(expected)):
            self.api.close(handle)
            raise CachePolicyError("Owned scope identity changed or ownership is absent")
        return OwnedTree(self, name, handle, identity)

    def names(self):
        if self.closed:
            raise CachePolicyError("Closed namespace")
        return self.api.names(self.handle)

    def read_once(self, name, maximum=1024**2):
        """Read one flat control file through the pinned parent, exclusively."""
        if self.closed or type(maximum) is not int or not 0 < maximum <= 16*1024**2:
            raise CachePolicyError("Bounded open namespace read required")
        handle, before = self.api.child(self.handle, name, directory=False)
        try:
            if not 0 <= before.bytes <= maximum:
                raise CachePolicyError("Control file exceeds its declared bound")
            data = C.create_string_buffer(before.bytes + 1)
            count = W.DWORD()
            self.api.check(self.api.k.ReadFile(handle, data, len(data), C.byref(count), None),
                           "Pinned control read failed")
            if count.value != before.bytes or self.api.identity(handle) != before:
                raise CachePolicyError("Control file changed while open")
            return data.raw[:count.value], before.document()
        finally:
            self.api.close(handle)

    def write_once(self, name, data, maximum=1024**2):
        """Immutable flat control write; a partial write remains as a refusal."""
        if self.closed:
            raise CachePolicyError("Closed namespace")
        return _write_once(self.api, self.handle, name, data, maximum)


def _write_once(api, parent, name, value, maximum):
    if type(maximum) is not int or maximum < 0 or not isinstance(value, bytes) or len(value) > maximum:
        raise CachePolicyError("Bounded bytes required before creating owned file")
    try:
        handle, original = api.child(parent, name, directory=False, create=True, mutable=True, write=True)
    except CacheObjectIncomplete as error:
        raise CacheWriteIncomplete(str(error), None, error.created_evidence) from error
    evidence = dict(component=name, new_object_created=True, original_identity=original.document(),
                    writer_close_confirmed=False, reader_close_confirmed=False)
    error = None
    try:
        for offset in range(0, len(value), 1024**2):
            block = value[offset:offset+1024**2]
            buffer, written = C.create_string_buffer(block), W.DWORD()
            api.check(api.k.WriteFile(handle, buffer, len(block), C.byref(written), None), "Owned cache write failed")
            if written.value != len(block):
                raise OSError("Short write; retain partial object and full reservation")
        api.check(api.k.FlushFileBuffers(handle), "Owned cache flush failed")
    except BaseException as caught:
        error = caught
    try:
        api.close(handle)
        evidence["writer_close_confirmed"] = True
    except CacheHandleCloseIncomplete as close_error:
        evidence["close_failures"] = list(close_error.close_failures)
        evidence["write_error"] = str(error) if error is not None else None
        raise CacheWriteIncomplete(str(close_error), None, evidence) from close_error
    # Last-write metadata may be finalized on the writer's close. Reopen against
    # the pinned parent, bind the original ID, and record the post-close state.
    try:
        reader, final = api.child(parent, name, directory=False)
    except BaseException as reopen_error:
        evidence["write_error"] = str(error) if error is not None else None
        evidence["reopen_error"] = str(reopen_error)
        raise CacheWriteIncomplete(str(reopen_error), None, evidence) from reopen_error
    problem = error
    final_identity = None
    try:
        if (not final.same_object(original.document()) or final.bytes > maximum
                or (error is None and final.bytes != len(value))):
            raise CachePolicyError("Owned object was replaced/changed after write")
        final_identity = final.document()
    except BaseException as identity_error:
        problem = identity_error
    try:
        api.close(reader)
        evidence["reader_close_confirmed"] = True
    except CacheHandleCloseIncomplete as close_error:
        evidence["close_failures"] = list(close_error.close_failures)
        problem = close_error
    if problem is not None:
        evidence["error"] = str(problem)
        raise CacheWriteIncomplete(str(problem), final_identity, evidence) from problem
    return final_identity


class OwnedTree:
    def __init__(self, namespace, name, handle, identity):
        self.namespace, self.api, self.name = namespace, namespace.api, name
        self.handles, self.directories, self.files = {"": handle}, {"": identity.document()}, {}
        self.unresolved_creates = {}
        self.closed = False

    def _parent(self, parts):
        handle, prefix = self.handles[""], []
        for part in parts:
            prefix.append(part)
            relative = "/".join(prefix)
            if relative not in self.handles:
                try:
                    child, identity = self.api.child(handle, part, directory=True, create=True, mutable=True)
                except CacheObjectIncomplete as error:
                    self.unresolved_creates[relative] = error.created_evidence
                    raise
                self.handles[relative], self.directories[relative] = child, identity.document()
            handle = self.handles[relative]
        return handle

    def write_once(self, relative, value, maximum):
        parts = _parts(relative)
        if (self.closed or type(maximum) is not int or maximum < 0 or not isinstance(value, bytes)
                or len(value) > maximum or relative in self.files):
            raise CachePolicyError("Closed, repeated, unbudgeted or oversized cache write")
        parent = self._parent(parts[:-1])
        try:
            self.files[relative] = _write_once(self.api, parent, parts[-1], value, maximum)
        except CacheWriteIncomplete as error:
            if error.partial_identity is not None:
                self.files[relative] = error.partial_identity
            else:
                self.unresolved_creates[relative] = error.created_evidence
            raise  # Known partial ownership is recorded, never called success.
        return dict(self.files[relative])

    def frozen(self):
        return dict(format="darsia-windows-owned-tree-v2", scope=self.name,
                    root=self.directories[""], directories=dict(self.directories), files=dict(self.files),
                    unresolved_creates=dict(self.unresolved_creates))

    def close(self):
        if not self.closed:
            failures = []
            for relative in sorted(self.handles, key=lambda x: (x.count("/"), len(x)), reverse=True):
                handle = self.handles.pop(relative)
                try:
                    self.api.close(handle)
                except CacheHandleCloseIncomplete as error:
                    failures.extend(error.close_failures)
            self.closed = True
            if failures:
                raise CacheHandleCloseIncomplete(failures)

    def cleanup(self, frozen):
        """Open and bind the entire exact inventory before first disposition.

        Unexpected entries refuse preflight. A later insertion can produce a
        partial cleanup of declared objects only; that is reported as incomplete.
        Leaves are disposed and closed before parents are checked/removed.
        """
        if self.closed or frozen.get("format") != "darsia-windows-owned-tree-v2" or frozen.get("scope") != self.name:
            raise CachePolicyError("Actual frozen v2 owned inventory required")
        if set(self.handles) != {""}:
            raise CachePolicyError("Close allocator handles; reopen frozen scope before cleanup")
        opened, removed, removed_dirs, mutations, close_failures = {}, [], [], [], []
        problem = None
        try:
            if frozen.get("unresolved_creates"):
                raise CachePolicyError("Unresolved created objects remain; no automatic cleanup")
            expected_dirs, expected_files = frozen["directories"], frozen["files"]
            if (not isinstance(expected_dirs, dict) or not isinstance(expected_files, dict)
                    or len(expected_dirs) + len(expected_files) > 8192
                    or set(expected_dirs) & set(expected_files) or "" not in expected_dirs
                    or frozen.get("root") != expected_dirs[""]):
                raise CachePolicyError("Invalid frozen file/directory inventory")
            paths = (set(expected_dirs) | set(expected_files)) - {""}
            if len({x.casefold() for x in paths}) != len(paths):
                raise CachePolicyError("Case-ambiguous frozen inventory preserved")
            if not self.api.identity(self.handles[""]).same_object(expected_dirs[""]):
                raise CachePolicyError("Frozen root was replaced")
            for relative in sorted((set(expected_dirs) | set(expected_files)) - {""}, key=lambda x: (x.count("/"), x)):
                parts = _parts(relative)
                parent_key = "/".join(parts[:-1])
                if parent_key not in self.handles:
                    raise CachePolicyError("Frozen inventory omits a parent")
                directory = relative in expected_dirs
                handle, identity = self.api.child(self.handles[parent_key], parts[-1],
                                                  directory=directory, mutable=True)
                expected = (expected_dirs if directory else expected_files)[relative]
                if not identity.same_object(expected) or (not directory and identity.document() != expected):
                    self.api.close(handle)
                    raise CachePolicyError("Foreign/changed frozen object preserved")
                if directory:
                    self.handles[relative] = handle
                else:
                    opened[relative] = handle
            for relative, handle in self.handles.items():
                prefix = relative + "/" if relative else ""
                declared = {name[len(prefix):] for name in set(expected_dirs) | set(expected_files)
                            if name.startswith(prefix) and name != relative and "/" not in name[len(prefix):]}
                if set(self.api.names(handle)) != declared:
                    raise CachePolicyError("Foreign/unrecorded namespace entry preserved before cleanup")
            for relative in sorted(opened):
                handle = opened.pop(relative)
                try:
                    self.api.dispose(handle)
                except BaseException:
                    try:
                        self.api.close(handle)
                    except CacheHandleCloseIncomplete as error:
                        close_failures.extend(error.close_failures)
                    raise
                mutation = dict(path=relative, directory=False, disposition_set=True,
                                handle_close_confirmed=False)
                mutations.append(mutation)
                # Pop before close and never retry an uncertain close.
                self.api.close(handle)
                mutation["handle_close_confirmed"] = True
                removed.append(relative)
            for relative in sorted(self.handles, key=lambda x: (x.count("/"), len(x)), reverse=True):
                handle = self.handles[relative]
                if self.api.names(handle):
                    raise CachePolicyError("New foreign entry remains; owned cleanup is incomplete")
                self.api.dispose(handle)
                del self.handles[relative]
                mutation = dict(path=relative, directory=True, disposition_set=True,
                                handle_close_confirmed=False)
                mutations.append(mutation)
                self.api.close(handle)
                mutation["handle_close_confirmed"] = True
                removed_dirs.append(relative)
            self.closed = True
        except BaseException as error:
            problem = error
            if isinstance(error, CacheHandleCloseIncomplete):
                close_failures.extend(error.close_failures)
            elif isinstance(error, CacheObjectIncomplete):
                close_failures.extend(error.created_evidence.get("close_failures", []))
        finally:
            while opened:
                _, handle = opened.popitem()
                try:
                    self.api.close(handle)
                except CacheHandleCloseIncomplete as error:
                    close_failures.extend(error.close_failures)
            try:
                self.close()
            except CacheHandleCloseIncomplete as error:
                close_failures.extend(error.close_failures)
        if problem is not None or close_failures:
            raise CacheCleanupIncomplete(str(problem or "Handle closure incomplete"), removed,
                                         mutations, close_failures) from problem
        return dict(complete=True, removed_files=removed, removed_directories=removed_dirs,
                    mutations=mutations, close_failures=[],
                    removed_logical_bytes=sum(expected_files[x]["bytes"] for x in removed),
                    physical_reclaimed_bytes=None)
