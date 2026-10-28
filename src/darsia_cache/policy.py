"""Pure prospective host cache policy; no probing, mkdir or legacy migration.

Cache settings do not make results, models, reviews or user state disposable.
Admission accounting is not an OS filesystem quota. A numerical quota must be
configured from a separately accepted measurement; there is no default size.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import uuid

PURPOSES = frozenset({"jobs", "gates", "tests", "clones", "display", "logs"})
PHYSICAL_RESERVE_BYTES = 40 * 1024**3
SETUP_CONTROL_MAX_BYTES = 128 * 1024**2


class CachePolicyError(ValueError):
    pass


def nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise CachePolicyError(label + " must be an explicit nonnegative integer")
    return value


def _absolute(value, *, allow_volume_root=False):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CachePolicyError("An explicit cache root is required")
    win = PureWindowsPath(value)
    path = win if win.drive or "\\" in value else PurePosixPath(value)
    if (not path.is_absolute() or ".." in path.parts
            or str(path).startswith(("\\\\", "//"))
            or (not allow_volume_root and path == type(path)(path.anchor))):
        raise CachePolicyError("Cache root must be a normalized local non-volume-root path")
    return path


def _descriptor(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))
            or not isinstance(value["path"], str) or not value["path"]
            or not 0 < nonnegative(value["size_bytes"], "Profile bytes") <= 16 * 1024**2):
        raise CachePolicyError("Cache profile requires exact bounded metadata path/SHA/size")
    return dict(value)


def _module_source_descriptor(value):
    """Allow canonical empty Python modules only in exact source closures."""
    if (isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"}
            and isinstance(value["path"], str) and value["path"]
            and PureWindowsPath(value["path"]).suffix.casefold() == ".py"
            and type(value["size_bytes"]) is int and value["size_bytes"] == 0
            and value["sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"):
        return dict(value)
    # Native module bytes are source inputs, not small JSON control documents.
    # Retain the exact SHA/length; a Windows extension can exceed the metadata
    # reader's 16 MiB limit (which still applies to every profile/control file).
    if (isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"}
            and isinstance(value["path"], str) and value["path"]
            and PureWindowsPath(value["path"]).suffix.casefold() in {".pyd", ".dll", ".so"}
            and type(value["size_bytes"]) is int and 0 < value["size_bytes"] < 2**63
            and re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"]))):
        return dict(value)
    return _descriptor(value)


def setup_control_descriptor(value):
    """The existing worker catalogue bound, solely for typed Setup controls."""
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or not isinstance(value["path"], str) or not value["path"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))
            or type(value["size_bytes"]) is not int
            or not 0 < value["size_bytes"] <= SETUP_CONTROL_MAX_BYTES):
        raise CachePolicyError("Exact bounded Setup control path/SHA/size required")
    return dict(value)



def _kernel_object(value, *, directory):
    keys = {"volume_serial", "file_id_128", "directory", "creation_100ns"}
    if (not isinstance(value, dict) or set(value) != keys
            or type(value["volume_serial"]) is not int or not 0 < value["volume_serial"] < 2**64
            or not re.fullmatch(r"[0-9a-f]{32}", str(value["file_id_128"]))
            or value["file_id_128"] == "0"*32 or value["directory"] is not directory
            or type(value["creation_100ns"]) is not int or value["creation_100ns"] <= 0):
        raise CachePolicyError("Exact original kernel object identity required")
    return dict(value)


@dataclass(frozen=True)
class VolumeCapBinding:
    """Accepted immutable identities, never a self-reported successful query.

    VHDX bytes, mtime and change time are deliberately NOT immutable: the
    attached filesystem provider writes that file. FileID/creation, actual
    VHD identifier, format, subtype and VirtualSize are checked separately.
    """
    volume_guid: str
    virtual_disk_identifier: str
    maximum_size_bytes: int
    image_path: str
    image_parent_identity: dict
    image_identity: dict
    cache_identity: dict

    def document(self):
        return dict(format="darsia-fixed-vhdx-cap-binding-v1", **self.__dict__)


def _volume_cap(value, root, quota):
    if value is None:
        return None  # Explicitly logical-only internal API; managed ImportLease refuses it.
    keys = {"format", "volume_guid", "virtual_disk_identifier", "maximum_size_bytes",
            "image_path", "image_parent_identity", "image_identity", "cache_identity"}
    if not isinstance(value, dict) or set(value) != keys or value["format"] != "darsia-fixed-vhdx-cap-binding-v1":
        raise CachePolicyError("Explicit fixed-VHDX volume cap binding required")
    guid = value["volume_guid"]
    # Exactly the Win32 volume GUID namespace; no UNC, drive-letter or mount alias.
    if not isinstance(guid, str) or not re.fullmatch(r"\\\\\?\\Volume\{[0-9a-fA-F-]{36}\}\\", guid):
        raise CachePolicyError("Exact attached VolumeGUID required")
    try:
        volume_uuid = uuid.UUID(guid[11:-2])
        disk_uuid = uuid.UUID(value["virtual_disk_identifier"])
    except (ValueError, TypeError, AttributeError) as error:
        raise CachePolicyError("Canonical nonzero volume/disk UUID required") from error
    if not volume_uuid.int or not disk_uuid.int or str(disk_uuid) != value["virtual_disk_identifier"]:
        raise CachePolicyError("Canonical nonzero volume/disk UUID required")
    maximum = nonnegative(value["maximum_size_bytes"], "Virtual disk maximum")
    if maximum != quota or not maximum or maximum % 512:
        raise CachePolicyError("Quota must equal the exact positive sector-aligned VHDX cap")
    image = _absolute(value["image_path"])
    if not isinstance(root, PureWindowsPath) or not isinstance(image, PureWindowsPath) or image.suffix.casefold() != ".vhdx":
        raise CachePolicyError("Reviewed Windows fixed VHDX image required")
    if image.drive.casefold() == root.drive.casefold():
        raise CachePolicyError("VHDX backing image must be outside its own capped volume")
    parent_id = _kernel_object(value["image_parent_identity"], directory=True)
    image_id = _kernel_object(value["image_identity"], directory=False)
    cache_id = _kernel_object(value["cache_identity"], directory=True)
    if parent_id["volume_serial"] != image_id["volume_serial"] or image_id["volume_serial"] == cache_id["volume_serial"]:
        raise CachePolicyError("Backing and capped volume identities must be distinct")
    return VolumeCapBinding(guid, str(disk_uuid), maximum, str(image), parent_id, image_id, cache_id)


@dataclass(frozen=True)
class CachePolicy:
    root: str
    quota_bytes: int
    failed_retention_hours: int
    profiles: dict
    volume_cap: VolumeCapBinding | None = None

    def purpose(self, purpose):
        if purpose not in PURPOSES:
            raise CachePolicyError("Unknown named cache purpose: " + str(purpose))
        return Path(self.root) / purpose


def cache_settings(document, *, required=False):
    """Validate metadata only. Actual filesystem locality is admitted later."""
    value = document.get("cache")
    if value is None:
        if required:
            raise CachePolicyError("Configure one measured [cache] policy before managed admission")
        return None
    if (not isinstance(value, dict) or set(value) - {
            "root", "quota_bytes", "failed_retention_hours", "profiles", "volume_cap"}
            or not {"root", "quota_bytes", "profiles"} <= set(value)):
        raise CachePolicyError("Cache needs root, quota_bytes and measured profiles")
    root = _absolute(value["root"])
    quota = nonnegative(value["quota_bytes"], "Cache quota bytes")
    if quota == 0:
        raise CachePolicyError("Measure and configure the cache quota; zero is not admission")
    hours = value.get("failed_retention_hours", 48)
    if type(hours) is not int or not 0 < hours <= 48:
        raise CachePolicyError("Failed retention must be at most 48 hours")
    profiles = value["profiles"]
    if (not isinstance(profiles, dict) or not profiles or set(profiles) - PURPOSES):
        raise CachePolicyError("Bound measured profiles by named cache purpose")
    for blocks in document.get("stores", {}).values():
        for location in blocks.values():
            try:
                other = _absolute(location, allow_volume_root=True)
            except CachePolicyError:
                continue  # UNC/invalid names need actual filesystem/alias admission later.
            if type(root) is type(other) and (root.is_relative_to(other) or other.is_relative_to(root)):
                raise CachePolicyError("Managed cache must be disjoint from durable/input store roots")
    return CachePolicy(str(root), quota, hours,
                       {key: _descriptor(row) for key, row in profiles.items()},
                       _volume_cap(value.get("volume_cap"), root, quota))


def budget_report(policy, *, used_bytes, reserved_unwritten_bytes, request_bytes,
                  physical_free_bytes, required_noncache_bytes=0,
                  eligible_reclaimable_bytes=0):
    """Pure conservative accounting; reclaimable is never subtracted as free."""
    used = nonnegative(used_bytes, "Observed cache logical bytes")
    reserved = nonnegative(reserved_unwritten_bytes, "Active unwritten cache bytes")
    request = nonnegative(request_bytes, "Requested cache maximum bytes")
    noncache = nonnegative(required_noncache_bytes, "Separate noncache allocation bytes")
    free = nonnegative(physical_free_bytes, "Fresh physical free bytes")
    reclaimable = nonnegative(eligible_reclaimable_bytes, "Eligible logical bytes")
    headroom = policy.quota_bytes - used - reserved
    # Keep the measured free-space observation separate from logical quota.
    # Declared pending lengths supply a conservative allocation estimate, not
    # a proved bound on physical allocation, metadata, slack or other writers.
    needed = request + reserved + noncache + PHYSICAL_RESERVE_BYTES
    return dict(required_cache_bytes=request, current_cache_logical_bytes=used,
                active_cache_remaining_bytes=reserved, quota_bytes=policy.quota_bytes,
                quota_headroom_bytes=headroom, post_admission_quota_headroom_bytes=headroom-request,
                eligible_reclaimable_logical_bytes=reclaimable,
                required_noncache_bytes=noncache, reserve_bytes=PHYSICAL_RESERVE_BYTES,
                needed_physical_bytes=needed, free_physical_bytes=free,
                needed_physical_bytes_semantics="estimated pending allocation plus reserve; not a physical upper bound",
                expected_cache_allocation_bytes=request+reserved,
                physical_reserve_guard_satisfied=free >= needed,
                future_cache_physical_allocation_bound=None,
                sufficient=headroom >= request and free >= needed,
                size_semantics="logical file lengths; not physical reclaimed allocation",
                enforcement="admission reservation; not a filesystem hard cap",
                hard_filesystem_quota_verified=False)


def cache_status_line(machine, observation, previous=None):
    """Format an existing measurement; never invent a fresh total or delta."""
    total = nonnegative(observation["logical_bytes"], "Measured cache bytes")
    when = observation["measured_utc"]
    comparable = (previous is not None and previous.get("root") == observation.get("root")
                  and previous.get("semantics") == observation.get("semantics")
                  and previous.get("measured_utc") != when)
    delta = total - previous["logical_bytes"] if comparable else "UNMEASURED"
    largest = sorted(observation.get("purposes", []),
                     key=lambda row: nonnegative(row["logical_bytes"], "Purpose bytes"), reverse=True)[:3]
    return dict(machine=machine, root=observation.get("root"), logical_bytes=total,
                measured_utc=when, delta_bytes=delta, largest_three=largest,
                current_unmeasured_delta=not comparable, hard_filesystem_quota_verified=False)
