"""Machine-local store bindings and inert, portable job input locators.

No file is written, no payload is opened, and no root is probed on import.
The execution/queue adapters remain unchanged until J2/J3 are adopted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import tempfile
import ntpath
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tomllib
from typing import Mapping

from .pipeline.job_queue import name
from .series_paths import series_key
from .job_filesystem import inspect_local_file

LIMIT = 1024 * 1024
MAX_ENTRIES = 128
ROLES = {"image", "configuration", "rig_model", "model", "calibration", "protocol"}


class StoreMapError(ValueError):
    pass


def _alias(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise StoreMapError("Root alias must be a lowercase identifier: " + repr(value))
    return value


def _path(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise StoreMapError("An explicit nonempty storage path is required")
    if value.replace("/", "\\").startswith(("\\\\?\\", "\\\\.\\")):
        raise StoreMapError("Device paths are not store roots")
    windows = PureWindowsPath(value)
    result = windows if windows.drive or "\\" in value else PurePosixPath(value)
    if not result.is_absolute():
        raise StoreMapError("An absolute store path is required: " + value)
    if ".." in result.parts:
        raise StoreMapError("Store paths must be normalized without parent traversal")
    return result


def _key(value):
    path = _path(str(value))
    return str(path).casefold() if isinstance(path, PureWindowsPath) else str(path)


def _relative(path, root):
    if type(path) is not type(root):
        return None
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _relative_input(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or "\x00" in value or ":" in value):
        raise StoreMapError("Input relative_path must use nonempty portable slash-separated segments")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts) or PurePosixPath(value).is_absolute():
        raise StoreMapError("Input relative_path cannot escape its declared root")
    if PureWindowsPath(value).drive:
        raise StoreMapError("Input relative_path cannot contain a drive or UNC path")
    for part in parts:
        if part.endswith((" ", ".")) or re.search(r'[<>"|?*\x00-\x1f]', part):
            raise StoreMapError("Input path contains a nonportable file name")
        if part.split(".")[0].casefold() in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}:
            raise StoreMapError("Input path contains a reserved file name")
    return parts


def default_store_map_path(*, environment=None, platform=None):
    env = os.environ if environment is None else environment
    platform = os.name if platform is None else platform
    if platform == "nt":
        base = env.get("LOCALAPPDATA")
        if not base:
            raise StoreMapError("LOCALAPPDATA is not configured; choose the local stores.toml explicitly")
        primary = Path(base) / "DarSIA" / "stores.toml"
        if primary.is_file():
            # Windows may virtualize AppData for a packaged launcher. Forward
            # the actual file location to workers outside that package too.
            return primary.resolve(strict=True)
        packages = Path(base) / "Packages"
        candidates = set()
        if packages.is_dir():
            for package in packages.iterdir():
                candidate = package / "LocalCache" / "Local" / "DarSIA" / "stores.toml"
                if candidate.is_file():
                    candidates.add(candidate.resolve(strict=True))
        if len(candidates) == 1:
            return candidates.pop()
        if candidates:
            raise StoreMapError(
                "Several packaged DarSIA store maps exist; choose this machine's stores.toml explicitly")
        return primary
    return Path(env.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "darsia" / "stores.toml"


def _external_file(path, forbidden_roots=()):
    pure = _path(str(path))
    if isinstance(pure, PureWindowsPath) and pure.drive.startswith("\\\\"):
        raise StoreMapError("The machine store map must be local, not a shared UNC file")
    if (os.name == "nt") != isinstance(pure, PureWindowsPath):
        raise StoreMapError("The local store-map filename must use this machine's path syntax")
    native = Path(path).expanduser().resolve()
    actual = _path(str(native))
    if isinstance(actual, PureWindowsPath) and actual.drive.startswith("\\\\"):
        raise StoreMapError("The machine store map resolves to a shared UNC file")
    for root in forbidden_roots:
        if native.is_relative_to(Path(root).expanduser().resolve()):
            raise StoreMapError("The store map must be outside repositories and series folders")
    for parent in (native.parent, *native.parent.parents):
        if (parent / ".git").exists() or (parent / "series.toml").is_file():
            raise StoreMapError("The store map must be outside repositories and series folders")
    return native


@dataclass(frozen=True)
class StoreMap:
    machine_id: str
    stores: Mapping
    source: Mapping
    document: Mapping

    def roots(self, series_id):
        key = series_key(series_id)
        if key not in self.stores:
            raise StoreMapError("No machine store block is configured for series " + str(series_id))
        return dict(self.stores[key])


def read_store_map(path=None, *, forbidden_roots=(), locality_provider=None, sync_roots=None):
    """Explicit metadata read; positive local-FS proof precedes opening the file.

    Run outside the GUI event thread. No root/input or execution admission is
    implied by accepting this machine-local configuration document.
    """
    path = _external_file(default_store_map_path() if path is None else path, forbidden_roots)
    locality = inspect_local_file(path, locality_provider=locality_provider, sync_roots=sync_roots)
    if not locality.allowed:
        raise StoreMapError(locality.reason + ": " + locality.evidence)
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    return parse_store_map(raw, source={"path": str(path), "locality": asdict(locality)})


def capacity_settings(document):
    """Pure optional machine-local expected-space settings, not root aliases."""
    value = document.get("capacity")
    if value is None:
        return None
    if (not isinstance(value, dict) or not {"state_directory", "profiles"} <= set(value)
            or set(value) - {"state_directory", "profiles", "reserve_bytes", "remote_owner"}):
        raise StoreMapError("Capacity needs state_directory, profiles and optional reserve_bytes/remote_owner")
    state = _path(value["state_directory"])
    if isinstance(state, PureWindowsPath) and state.drive.startswith("\\\\"):
        raise StoreMapError("Capacity coordination state must be local")
    reserve = value.get("reserve_bytes", 40 * 1024**3)
    if type(reserve) is not int or reserve < 0:
        raise StoreMapError("Capacity reserve_bytes must be a nonnegative integer")
    profiles = value["profiles"]
    if not isinstance(profiles, dict) or not profiles or len(profiles) > MAX_ENTRIES:
        raise StoreMapError("Capacity needs a bounded table of explicit series profiles")
    result = {}
    for key, item in profiles.items():
        identity = series_key(key)
        if identity in result:
            raise StoreMapError("Duplicate canonical capacity-profile series")
        if (not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}
                or not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                or type(item["size_bytes"]) is not int or not 0 < item["size_bytes"] <= 128 * 1024 * 1024):
            raise StoreMapError("Capacity profile needs an exact metadata path/SHA/size")
        _path(item["path"])
        result[identity] = dict(item)
    settings = dict(state_directory=str(state), reserve_bytes=reserve, profiles=result)
    if "remote_owner" in value:
        owner = value["remote_owner"]
        if (not isinstance(owner, dict) or set(owner) != {"path", "sha256", "size_bytes"}
                or not isinstance(owner["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", owner["sha256"]) is None
                or type(owner["size_bytes"]) is not int or not 0 < owner["size_bytes"] <= 16 * 1024):
            raise StoreMapError("NAS owner needs one exact path/SHA/size descriptor no larger than 16 KiB")
        selected = _path(owner["path"])
        if not isinstance(selected, PureWindowsPath) or not selected.drive.startswith("\\\\"):
            raise StoreMapError("The operator-owned NAS authority document must use its canonical UNC path")
        settings["remote_owner"] = dict(owner)
    return settings


def parse_store_map(raw, *, source=None):
    """Pure settings validation; unknown fields and exact text remain caller-owned."""
    if not isinstance(raw, bytes) or len(raw) > LIMIT:
        raise StoreMapError("Store map must be UTF-8 bytes no larger than 1 MiB")
    doc = tomllib.loads(raw.decode("utf-8-sig"))
    machine = doc.get("machine")
    if not isinstance(machine, dict) or not isinstance(machine.get("id"), str):
        raise StoreMapError("Store map requires [machine].id")
    # Same identifier contract as the existing whole-run worker, without changing it.
    name(machine["id"])
    blocks = doc.get("stores")
    if not isinstance(blocks, dict) or not blocks or len(blocks) > MAX_ENTRIES:
        raise StoreMapError("Store map needs bounded per-series store blocks")
    stores = {}
    for identity, values in blocks.items():
        key = series_key(identity)
        if key in stores:
            raise StoreMapError("Duplicate canonical series in store map")
        if not isinstance(values, dict) or len(values) > MAX_ENTRIES:
            raise StoreMapError("Store aliases must be a bounded table")
        roots = {_alias(alias): str(_path(value)) for alias, value in values.items()}
        if not {"series", "results"}.issubset(roots):
            raise StoreMapError("Each store needs explicit series and results roots")
        stores[key] = roots
    capacity_settings(doc)
    from darsia_cache.policy import cache_settings
    cache_settings(doc)  # Pure optional parse; missing managed policy refuses at claim.
    return StoreMap(machine["id"], stores,
        {**dict(source or {}), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}, doc)


def manifest_root_proposal(document):
    """Data-converter proposal from actual legacy fields; never saves a manifest."""
    legacy = document.get("legacy")
    if not isinstance(legacy, dict):
        raise StoreMapError("No explicit legacy root data is recorded")
    result = {"series": "."}
    for alias, key in (("raw", "raw_inputs"), ("legacy_results", "previous_results")):
        value = legacy.get(key)
        if isinstance(value, str) and value.strip():
            result[alias] = value
    return result


def manifest_roots(series_root, document):
    """Validate explicit producer-site roots, independent of the local store map."""
    base = _path(str(series_root))
    if document.get("schema_version") != 1:
        raise StoreMapError("A schema-1 series manifest is required")
    series_key(document.get("series_id", ""))
    roots = document.get("roots")
    if not isinstance(roots, dict) or not roots or len(roots) > MAX_ENTRIES:
        raise StoreMapError("series.toml needs an explicit bounded [roots] table")
    if roots.get("series") != ".":
        raise StoreMapError("[roots].series must be '.' relative to this manifest")
    resolved = {"series": base}
    for alias, value in roots.items():
        _alias(alias)
        if alias == "series":
            continue
        if not isinstance(value, str) or not value:
            raise StoreMapError("Each declared root needs a path")
        # Explicit relative root declarations are allowed; input locators are
        # stricter and may never contain '..'. Normalize root declarations once.
        try:
            result = _path(value)
        except StoreMapError:
            if PureWindowsPath(value).drive or PurePosixPath(value).is_absolute() or "\x00" in value:
                raise
            if isinstance(base, PureWindowsPath):
                result = _path(ntpath.normpath(str(base / PureWindowsPath(value))))
            else:
                result = _path(os.path.normpath(str(base / PurePosixPath(value))))
        resolved[alias] = result
    return resolved


def input_reference(path, roots, *, sha256, size_bytes, role):
    """J2-facing inert locator; unknown/ambiguous producer roots are refused."""
    absolute = _path(str(path))
    candidates = []
    for alias, root in roots.items():
        relative = _relative(absolute, _path(str(root)))
        if relative is not None and relative.parts:
            candidates.append((len(_path(str(root)).parts), _alias(alias), relative.as_posix()))
    if not candidates:
        raise StoreMapError("No declared root contains input path: " + str(path))
    longest = max(row[0] for row in candidates)
    matches = [row for row in candidates if row[0] == longest]
    if len(matches) != 1:
        raise StoreMapError("Input path has ambiguous equally specific declared roots: " + str(path))
    value = {"root_alias": matches[0][1], "relative_path": matches[0][2],
             "sha256": sha256, "bytes": size_bytes, "role": role}
    validate_input_reference(value)
    return value


def validate_input_reference(value):
    if not isinstance(value, dict) or set(value) != {"root_alias", "relative_path", "sha256", "bytes", "role"}:
        raise StoreMapError("Input reference requires root_alias/relative_path/sha256/bytes/role")
    _alias(value["root_alias"])
    _relative_input(value["relative_path"])
    if not isinstance(value["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None:
        raise StoreMapError("Input reference needs its original SHA-256")
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise StoreMapError("Input reference bytes must be nonnegative integer")
    if not isinstance(value["role"], str) or value["role"] not in ROLES:
        raise StoreMapError("Input role is unsupported")


def resolve_input_reference(reference, stores, series_id):
    """Lexical locator only; neither existence nor SHA verification is claimed."""
    validate_input_reference(reference)
    roots = stores.roots(series_id)
    alias = reference["root_alias"]
    if alias not in roots:
        raise StoreMapError("This machine has no configured root alias: " + alias)
    root = _path(roots[alias])
    original = root.joinpath(*_relative_input(reference["relative_path"]))
    from .storage_paths import resolve_storage_path
    located = str(resolve_storage_path(str(original), series_roots=(roots["series"],)))
    _input_container(reference, roots, located)
    return located


def _input_container(reference, roots, located):
    """Contain a remapped locator in its original root or explicit result/project roots."""
    root = _path(roots[reference["root_alias"]])
    original = root.joinpath(*_relative_input(reference["relative_path"]))
    path = _path(located)
    allowed = [root]
    if _key(str(path)) != _key(str(original)):
        allowed += [_path(roots[key]) for key in ("results", "project") if key in roots]
    matches = [candidate for candidate in allowed if _relative(path, candidate) is not None]
    if not matches:
        raise StoreMapError("Remapped input escapes the configured input, results and project roots")
    return max(matches, key=lambda candidate: len(candidate.parts))


def require_input_containment(reference, stores, series_id):
    """Check real containment using only the explicit root allowed by the locator."""
    validate_input_reference(reference)
    roots = stores.roots(series_id)
    located = resolve_input_reference(reference, stores, series_id)
    return require_real_containment(_input_container(reference, roots, located), located)


def root_availability(stores, series_id, *, is_directory=None):
    """Explicit read-only availability check; never creates output directories.

    The UI must run this off its event thread. Directory existence is not a
    write/space/protocol guarantee and is not per-input hash verification.
    """
    inspect = Path.is_dir if is_directory is None else is_directory
    rows = []
    for alias, value in stores.roots(series_id).items():
        try:
            if is_directory is None and (os.name == "nt") != isinstance(_path(value), PureWindowsPath):
                available, error = False, "Configured path uses another operating system's syntax"
            else:
                available = bool(inspect(Path(value)))
                error = "" if available else "Directory is unavailable or not yet created"
        except OSError as exc:
            available, error = False, str(exc)
        rows.append({"alias": alias, "path": value, "available": available, "reason": error,
                     "input_hashes_verified": False, "directory_created": False,
                     "local_output_filesystem_verified": False})
    return {"series_id": series_id, "machine_id": stores.machine_id, "roots": rows}


def require_real_containment(root, path):
    """Explicit pre-use path check, catching symlink/junction escape without payload reads."""
    actual_root, actual_path = Path(root).resolve(strict=True), Path(path).resolve(strict=True)
    if not actual_path.is_relative_to(actual_root):
        raise StoreMapError("Resolved input escapes its configured root through a link")
    return actual_path


def save_store_map_text(path, raw, *, expected_bytes, forbidden_roots=(),
                        locality_provider=None, sync_roots=None):
    """Explicit Settings save only, preserving exact edited text and old bytes.

    No job or series file is edited. The displayed source must still match;
    replacement is atomic, with a content-addressed previous copy retained.
    This does not supply queue/command concurrency or worker admission.
    """
    parsed = parse_store_map(raw)
    path = _external_file(path, forbidden_roots)
    locality = inspect_local_file(path, locality_provider=locality_provider, sync_roots=sync_roots)
    if not locality.allowed:
        raise StoreMapError(locality.reason + ": " + locality.evidence)
    def previous():
        try:
            with path.open("rb") as stream:
                value = stream.read(LIMIT + 1)
        except FileNotFoundError:
            return None
        if len(value) > LIMIT:
            raise StoreMapError("Existing store map exceeds 1 MiB")
        return value
    if previous() != expected_bytes:
        raise StoreMapError("Store map changed since it was displayed; reload before saving")
    path.parent.mkdir(parents=True, exist_ok=True)
    checked_parent = Path(locality.checked_container).resolve(strict=True)
    if not path.parent.resolve(strict=True).is_relative_to(checked_parent):
        raise StoreMapError("Store-map parent changed while saving")
    backup = None
    if expected_bytes is not None:
        folder = path.parent / "_previous"
        folder.mkdir(exist_ok=True)
        if folder.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise StoreMapError("Previous-store copy would escape the local settings folder")
        backup = folder / ("stores-" + hashlib.sha256(expected_bytes).hexdigest() + ".toml")
        try:
            with backup.open("xb") as stream:
                stream.write(expected_bytes)
        except FileExistsError:
            if backup.read_bytes() != expected_bytes:
                raise StoreMapError("Existing previous-store copy has different bytes")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".stores-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if previous() != expected_bytes:
            raise StoreMapError("Store map changed during save; replacement was refused")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "machine_id": parsed.machine_id,
            "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
            "previous_sha256": hashlib.sha256(expected_bytes).hexdigest() if expected_bytes is not None else None,
            "previous_copy": str(backup) if backup else None, "locality": asdict(locality)}


@dataclass(frozen=True)
class RootsMigrationProposal:
    before: bytes
    after: bytes
    receipt: Mapping


def propose_manifest_roots(series_root, raw):
    """Prepare a logged append-only data migration; never writes a series file."""
    import difflib
    if not isinstance(raw, bytes) or len(raw) > LIMIT:
        raise StoreMapError("Series manifest exceeds 1 MiB")
    document = tomllib.loads(raw.decode("utf-8-sig"))
    if document.get("schema_version") != 1:
        raise StoreMapError("A schema-1 series manifest is required")
    series_key(document.get("series_id", ""))
    before_sha = hashlib.sha256(raw).hexdigest()
    if "roots" in document:
        manifest_roots(series_root, document)
        return RootsMigrationProposal(raw, raw, {
            "format": "darsia-series-roots-migration-proposal-v1", "state": "already_declared",
            "series_id": document["series_id"], "before_sha256": before_sha,
            "after_sha256": before_sha, "proposed_roots": dict(document["roots"]),
            "source_fields": "Existing [roots] remains authoritative; not replaced from legacy fields",
            "unified_diff": "", "applied": False,
        })
    roots = manifest_root_proposal(document)
    wanted = dict(document)
    wanted["roots"] = roots
    manifest_roots(series_root, wanted)
    newline = "\r\n" if b"\r\n" in raw else "\n"
    addition = newline + "[roots]" + newline
    addition += "".join(alias + " = " + json.dumps(value, ensure_ascii=False) + newline
                        for alias, value in roots.items())
    after = raw + (b"" if raw.endswith(b"\n") else newline.encode()) + addition.encode("utf-8")
    if len(after) > LIMIT:
        raise StoreMapError("Proposed series manifest exceeds 1 MiB")
    if tomllib.loads(after.decode("utf-8-sig")) != wanted:
        raise StoreMapError("Roots proposal changed unrelated manifest values")
    receipt = {
        "format": "darsia-series-roots-migration-proposal-v1", "state": "proposal_only",
        "series_id": document["series_id"], "manifest": str(Path(series_root) / "series.toml"),
        "before_sha256": before_sha, "after_sha256": hashlib.sha256(after).hexdigest(),
        "proposed_roots": roots,
        "source_fields": {"series": "manifest location", "raw": "legacy.raw_inputs when present",
                          "legacy_results": "legacy.previous_results when present"},
        "backup_required": "Copy exact previous series.toml into config/_previous/<migration-id>/ before applying",
        "receipt_required": "Write config/_migrations/<migration-id>/receipt.json with actual UTC, before/after SHA and exact diff",
        "unified_diff": "".join(difflib.unified_diff(raw.decode("utf-8-sig").splitlines(keepends=True),
            after.decode("utf-8-sig").splitlines(keepends=True), fromfile="a/series.toml", tofile="b/series.toml")),
        "unknown_or_unrelated_fields_preserved": True, "applied": False,
    }
    return RootsMigrationProposal(raw, after, receipt)
