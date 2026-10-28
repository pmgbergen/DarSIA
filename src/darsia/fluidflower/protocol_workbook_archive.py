"""Archive superseded protocol folders after shared workbooks are active."""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import stat
import tomllib
import uuid

from .protocol_workbooks import _descriptor, _exact_replace
from .series_method import _manifest_bytes, _write_manifest
from .series_locations import _path
from .storage_paths import machine_stored_text, refresh_storage_paths, resolve_storage_path


def _plain_directory(path):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or path.is_symlink()
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
        raise ValueError("Protocol folder moves cannot follow links or junctions: " + str(path))
    return path.resolve(strict=True)


def archive_protocol_sources(plan, package, *, destination=None, writer=None):
    """Rename only old first-level protocol folders and register shared aliases.

    Root-level files, the active workbook package, configuration files and sources
    outside the selected protocol folder are never moved. Directory rename has
    no copy fallback; a different filesystem therefore fails without copying.
    """
    if not package.get("activated"):
        raise ValueError("Activate the shared workbooks before archiving source folders")
    if package.get("archived"):
        raise ValueError("The source protocol folders are already archived")
    root = Path(plan.root).absolute()
    manifest_path = root / "series.toml"
    original = manifest_path.read_bytes()
    manifest = tomllib.loads(original.decode("utf-8-sig"))
    paths = manifest.get("paths", {})
    protocols = _path(root, paths.get("protocols", "protocols"))
    protocols_real = _plain_directory(protocols)
    if root.resolve().is_relative_to(protocols_real):
        raise ValueError("The protocol folder cannot contain the DarSIA configuration root")
    protected = {root.resolve()}
    for _run, path, raw in getattr(plan, "configurations", ()):
        protected.add(Path(path).resolve())
        data = tomllib.loads(raw.decode("utf-8-sig")).get("data", {})
        for value in (data.get("folder"), *(data.get("folders") or [])):
            if value:
                protected.add(_path(root, value).resolve())
    for key, value in (manifest.get("roots") or {}).items():
        if key in ("raw", "images") or key.startswith("images_"):
            protected.add(_path(root, value).resolve())
    workbook_folder = _plain_directory(package["folder"])
    excluded = (workbook_folder.relative_to(protocols_real).parts[0]
                if workbook_folder.is_relative_to(protocols_real) and workbook_folder != protocols_real else None)
    sources = []
    for path in sorted(protocols.iterdir()):
        if path.name.startswith(".") or path.name == excluded:
            continue
        # Do not classify a link by following it; links are rejected even when
        # their destination lies outside the selected protocol root.
        info = path.lstat()
        if path.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Protocol folder moves cannot follow links or junctions: " + str(path))
        if not stat.S_ISDIR(info.st_mode):
            continue
        actual = _plain_directory(path)
        if actual.parent != protocols_real:
            raise ValueError("The source protocol folder is outside its selected root")
        if any(value.is_relative_to(actual) for value in protected):
            raise ValueError("A protocol folder also contains configuration or original image data: " + str(path))
        sources.append(path)
    if not sources:
        raise ValueError("There are no old source protocol folders to archive")
    project = _plain_directory(_path(root, paths.get("project", "..")))
    archive_parent = project / "Archive"
    if archive_parent.exists():
        _plain_directory(archive_parent)
    else:
        archive_parent.mkdir()
    archive_parent = archive_parent.resolve(strict=True)
    target = (Path(destination).absolute() if destination is not None else archive_parent /
              ("ProtocolMigration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]))
    if (target.parent.resolve(strict=True) != archive_parent or not target.name.startswith("ProtocolMigration_")
            or target.is_relative_to(protocols_real) or protocols_real.is_relative_to(target)):
        raise ValueError("Choose a fresh ProtocolMigration folder inside the project's Archive folder")
    if target.exists() or target.is_symlink():
        raise FileExistsError("The protocol archive destination already exists: " + str(target))
    updated = deepcopy(manifest)
    aliases = updated.setdefault("storage", {}).setdefault("path_aliases", {})
    moves = [(source, target / source.name) for source in sources]
    for source, archived in moves:
        key, value = machine_stored_text(source), machine_stored_text(archived)
        if key in aliases and aliases[key] != value:
            raise ValueError("A different storage alias already owns this protocol folder: " + str(source))
        aliases[key] = value
    written, parsed = _manifest_bytes(original, updated)
    migration_path = Path(package["folder"]) / "protocol_workbook_migration.json"
    migration_bytes = migration_path.read_bytes()
    migration = json.loads(migration_bytes)
    if migration.get("status") != "active":
        raise ValueError("The workbook package has no completed activation record")
    target.mkdir()
    completed = []
    save = writer or _write_manifest
    try:
        for source, archived in moves:
            # A first-level rename preserves its contents without traversing
            # anything inside it, including any unused historical protocol rows.
            if archived.exists() or archived.is_symlink():
                raise FileExistsError("The protocol archive target is occupied: " + str(archived))
            source.rename(archived)
            completed.append((source, archived))
        save(manifest_path, original, parsed)
        if manifest_path.read_bytes() != written:
            raise ValueError("The series writer returned unexpected bytes")
        refresh_storage_paths(root)
        for source, archived in moves:
            actual = resolve_storage_path(source, series_roots=(root,)).resolve()
            if actual != archived.resolve():
                raise ValueError("The saved storage aliases do not resolve this archived protocol folder: " + str(source))
        migration["archive"] = dict(status="complete", folder=machine_stored_text(target),
            moves=[dict(source=machine_stored_text(source), destination=machine_stored_text(archived))
                   for source, archived in moves],
            manifest_before={**_descriptor(manifest_path, original), "bytes_base64": base64.b64encode(original).decode("ascii")},
            manifest_after=_descriptor(manifest_path, written))
        record = (json.dumps(migration, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        _exact_replace(migration_path, migration_bytes, record)
    except BaseException as archive_error:
        rollback_errors = []
        current = manifest_path.read_bytes()
        if current != original:
            try:
                _exact_replace(manifest_path, written, original)
            except Exception as error:
                rollback_errors.append(str(error))
        for source, archived in reversed(completed):
            try:
                if source.exists() or source.is_symlink():
                    raise FileExistsError("A new folder now occupies the original protocol location: " + str(source))
                _plain_directory(archived)
                archived.rename(source)
            except Exception as error:
                rollback_errors.append(str(error))
        refresh_storage_paths(root)
        try:
            target.rmdir()  # Only the now-empty directory owned by this operation.
        except OSError:
            pass
        if rollback_errors:
            raise RuntimeError("Source archive failed; active workbooks are unchanged. "
                               "Some concurrent folder/manifest changes prevented rollback: " + "; ".join(rollback_errors)) from archive_error
        raise RuntimeError("Source archive failed and was rolled back; active workbooks are unchanged: " + str(archive_error)) from archive_error
    package["archived"] = True
    return dict(folder=target, archived=True, moves=moves, active_workbooks_preserved=True)
