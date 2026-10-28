"""Non-destructive project storage for FluidFlower experiment series."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from darsia.fluidflower.image_formats import (
    DEFAULT_IMAGE_FILE_POLICY,
    normalize_image_file_policy,
)
from darsia.fluidflower.modules import (
    MODULE_BY_KEY,
    VALID_MODULE_STATUSES,
    dependencies_satisfied,
    initial_module_states,
    invalidate_downstream,
    normalize_module_states,
)
from darsia.fluidflower.templates import RIG_TYPES, template_by_id

PROJECT_FORMAT = "darsia-fluidflower-project-v1"
SOURCE_MANIFEST_FORMAT = "darsia-fluidflower-source-manifest-v1"
REGISTRY_FORMAT = "darsia-fluidflower-project-registry-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, fallback: str = "fluidflower-series") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().casefold()).strip("-")
    return slug or fallback


def default_project_root() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys_platform() == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "DarSIA" / "FluidFlower Projects"


def sys_platform() -> str:
    # Isolated for deterministic tests without importing GUI modules.
    import sys

    return sys.platform


def default_registry_path() -> Path:
    return default_project_root().parent / "fluidflower_projects.json"


def resolve_project_package_root(selection: Path) -> Path:
    """Resolve either a project folder or a package containing one project."""

    from .storage_paths import resolve_storage_path
    root = resolve_storage_path(selection).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if (root / "series.json").is_file():
        return root

    candidates = sorted(
        child.resolve()
        for child in root.iterdir()
        if child.is_dir() and (child / "series.json").is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "No DarSIA FluidFlower project was found. Select either the project "
            "folder itself or a package with one project folder directly inside it."
        )
    raise ValueError(
        "The selected package contains multiple DarSIA FluidFlower projects. "
        "Select the specific project folder to open."
    )


def _atomic_json(path: Path, payload: Mapping[str, Any], *, copy_payload: bool = False, expected_raw: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload) if copy_payload else payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        if expected_raw is not None and path.read_bytes() != expected_raw:
            raise ValueError("Project registry changed; refresh Start before removing it")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any], *, copy_payload: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(payload) if copy_payload else payload, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file_uncached(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    from darsia.fluidflower.phase_source_hash_cache import active_source_hash_cache
    cache = active_source_hash_cache()
    return (_sha256_file_uncached(Path(path)) if cache is None else
            cache.hash(Path(path), _sha256_file_uncached))


def source_record(kind: str, path: Path, *, run_id: str = "") -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    record: dict[str, Any] = {
        "source_id": uuid.uuid4().hex,
        "kind": str(kind),
        "run_id": str(run_id).strip().casefold(),
        "path": str(resolved),
        "registered_utc": utc_now(),
        "exists_at_registration": resolved.exists(),
        "is_directory": resolved.is_dir(),
    }
    if resolved.is_file():
        record.update(
            {
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return record


def _path_identity(value: str | Path) -> str:
    """Return a stable comparison key without changing the referenced source."""

    return os.path.normcase(str(Path(value).expanduser().resolve()))


def _mapping_source_specs(
    mapping: Mapping[str, Any],
) -> list[tuple[str, Path, str]]:
    """List the independently traceable sources used by one run mapping."""

    specs: list[tuple[str, Path, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, raw_path: Any, scan_policy: str = "content") -> None:
        value = str(raw_path or "").strip()
        if not value:
            return
        path = Path(value).expanduser().resolve()
        key = (kind, _path_identity(path))
        if key in seen:
            return
        seen.add(key)
        specs.append((kind, path, scan_policy))

    def covered_by(raw_path: Any, raw_folders: Sequence[Any]) -> bool:
        value = str(raw_path or "").strip()
        if not value:
            return False
        path = Path(value).expanduser().resolve()
        for raw_folder in raw_folders:
            folder_value = str(raw_folder or "").strip()
            if not folder_value:
                continue
            folder = Path(folder_value).expanduser().resolve()
            try:
                path.relative_to(folder)
                return True
            except ValueError:
                continue
        return False

    add("run_folder", mapping.get("experiment_folder"), "reference_only")
    experiment_folders = list(mapping.get("experiment_folders") or [])
    for path in experiment_folders:
        add("co2_image_folder", path)
    add(
        "baseline_image",
        mapping.get("baseline_candidate"),
        (
            "reference_only"
            if covered_by(mapping.get("baseline_candidate"), experiment_folders)
            else "content"
        ),
    )
    clearwater_folders = list(mapping.get("clearwater_folders") or [])
    add(
        "clearwater_image",
        mapping.get("clearwater_candidate"),
        (
            "reference_only"
            if covered_by(mapping.get("clearwater_candidate"), clearwater_folders)
            else "content"
        ),
    )
    for path in clearwater_folders:
        add("clearwater_image_folder", path)
    for path in mapping.get("indicator_reference_folders") or []:
        add("indicator_reference_image_folder", path)
    for path in mapping.get("alternate_color_reference_folders") or []:
        add("alternate_color_reference_image_folder", path)
    protocol_tables = list(mapping.get("protocol_tables") or [])
    if mapping.get("protocol_table"):
        protocol_tables.append(mapping.get("protocol_table"))
    for path in protocol_tables:
        add("protocol", path)
    overflow_folders = list(mapping.get("overflow_image_folders") or [])
    for path in overflow_folders:
        add("overflow_image_folder", path)
    overflow_tables = list(mapping.get("overflow_tables") or [])
    if mapping.get("overflow_table"):
        overflow_tables.append(mapping.get("overflow_table"))
    for path in overflow_tables:
        add(
            "overflow_table",
            path,
            "reference_only" if covered_by(path, overflow_folders) else "content",
        )
    return specs


def _new_run_workflow(*, actor: str, recorded_utc: str) -> dict[str, dict[str, Any]]:
    states = initial_module_states()
    states["setup"].update(
        {
            "status": "approved",
            "updated_utc": recorded_utc,
            "updated_by": actor,
            "reason": "Experiment added to an established series.",
        }
    )
    states["data_integrity"].update(
        {
            "status": "ready",
            "updated_utc": recorded_utc,
            "updated_by": actor,
            "reason": "New source mapping requires run-specific integrity review.",
        }
    )
    return states


def _mapping_signature(mapping: Mapping[str, Any]) -> str:
    comparable = deepcopy(dict(mapping))
    for key in ("source_record_ids", "source_revision_id"):
        comparable.pop(key, None)
    return json.dumps(comparable, sort_keys=True, ensure_ascii=False, default=str)


class FluidFlowerProject:
    """One experimental-series project with immutable source references."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        from .storage_paths import resolve_storage_path
        self.root = resolve_storage_path(root).expanduser().resolve()
        self.manifest_path = self.root / "series.json"
        self.source_manifest_path = self.root / "source_manifest.json"
        self.audit_path = self.root / "history" / "audit.jsonl"
        self.manifest = deepcopy(dict(manifest))
        self.manifest["modules"] = normalize_module_states(self.manifest.get("modules"))
        self.validate()

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        name: str,
        series_id: str,
        template_id: str,
        rig_type: str,
        protocol_path: Path | None = None,
        image_sources: Mapping[str, Path | None] | None = None,
        overflow_mode: str = "none",
        overflow_path: Path | None = None,
        created_by: str = "",
    ) -> "FluidFlowerProject":
        root = Path(root).expanduser().resolve()
        if (root / "series.json").exists():
            raise FileExistsError(f"A FluidFlower project already exists in {root}")
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Project folder is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        for folder in (
            "annotations",
            "cache",
            "history",
            "overrides",
            "results",
            "runs",
            "timeline",
        ):
            (root / folder).mkdir(exist_ok=True)

        template = template_by_id(template_id)
        if rig_type not in RIG_TYPES:
            raise ValueError(f"Unknown FluidFlower rig type: {rig_type!r}")
        now = utc_now()
        states = initial_module_states()
        states["setup"].update(
            {
                "status": "approved",
                "updated_utc": now,
                "updated_by": created_by.strip(),
                "reason": "Created with the FluidFlower series setup wizard.",
            }
        )
        states["data_integrity"]["status"] = "ready"
        manifest: dict[str, Any] = {
            "schema": 1,
            "format": PROJECT_FORMAT,
            "project_id": uuid.uuid4().hex,
            "series_id": slugify(series_id, fallback="series"),
            "name": str(name).strip(),
            "template": {
                "id": template.template_id,
                "name": template.name,
                "protocol_template": template.protocol_template,
            },
            "created_utc": now,
            "updated_utc": now,
            "created_by": created_by.strip(),
            "storage": {
                "schema": 1,
                "default_result_root": {
                    "scope": "project_relative",
                    "path": "results",
                },
            },
            "rig": {
                "type": rig_type,
                "type_label": RIG_TYPES[rig_type],
                "geometry": {},
                "depth": {},
                "injectors": [],
                "facies": [],
            },
            "chemistry": {
                "indicator_system": template.indicator_system,
                "temperature": None,
                "pressure": None,
                "initial_ph": None,
                "indicator_response_reference": {
                    "schema": 1,
                    "format": "darsia-fluidflower-indicator-reference-v1",
                    "indicator_system": template.indicator_system,
                    "scientific_role": "chemical_response_reference",
                    "radiometric_normalization_allowed": False,
                    "status": "not_provided",
                    "entry_count": 0,
                    "distinct_ph_values": [],
                    "ph_span": None,
                    "issues": [],
                    "entries": [],
                },
                "notes": "",
            },
            "protocol": {
                "schema": "fluidflower-protocol-v1",
                "template": template.protocol_template,
                "source_id": None,
            },
            "overflow": {
                "mode": str(overflow_mode),
                "source_id": None,
                "optional": True,
            },
            "runs": [],
            "modules": states,
            "tools": {},
        }
        project = cls(root, manifest)

        sources: list[dict[str, Any]] = []
        if protocol_path is not None and str(protocol_path):
            record = source_record("protocol", Path(protocol_path))
            sources.append(record)
            project.manifest["protocol"]["source_id"] = record["source_id"]
        for kind, path in (image_sources or {}).items():
            if path is None or not str(path):
                continue
            sources.append(source_record(str(kind), Path(path)))
        if overflow_path is not None and str(overflow_path):
            record = source_record("overflow", Path(overflow_path))
            sources.append(record)
            project.manifest["overflow"]["source_id"] = record["source_id"]

        source_manifest = {
            "schema": 1,
            "format": SOURCE_MANIFEST_FORMAT,
            "project_id": project.manifest["project_id"],
            "created_utc": now,
            "policy": "source-files-are-read-only-and-never-copied-or-modified",
            "sources": sources,
        }
        project._write_manifest()
        _atomic_json(project.source_manifest_path, source_manifest)
        project._audit(
            "project_created",
            created_by,
            {
                "template_id": template.template_id,
                "rig_type": rig_type,
                "source_count": len(sources),
            },
        )
        return project

    @classmethod
    def load(cls, root: Path) -> "FluidFlowerProject":
        from .storage_paths import resolve_storage_path
        root = resolve_storage_path(root).expanduser().resolve()
        path = root / "series.json"
        if not path.is_file():
            raise FileNotFoundError(f"No FluidFlower series.json found in {root}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid FluidFlower project manifest: {path}")
        return cls(root, payload)

    def validate(self) -> None:
        if self.manifest.get("format") != PROJECT_FORMAT:
            raise ValueError("Unsupported FluidFlower project format")
        if int(self.manifest.get("schema") or 0) != 1:
            raise ValueError("Unsupported FluidFlower project schema")
        if not str(self.manifest.get("project_id") or ""):
            raise ValueError("FluidFlower project has no project_id")
        if str((self.manifest.get("rig") or {}).get("type") or "") not in RIG_TYPES:
            raise ValueError("FluidFlower project has an invalid rig type")

    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or self.manifest.get("series_id") or "")

    @property
    def project_id(self) -> str:
        return str(self.manifest["project_id"])

    @property
    def module_states(self) -> dict[str, dict[str, Any]]:
        return self.manifest["modules"]

    def default_result_root(self) -> Path:
        """Return the portable default root for generated analysis products."""

        storage = self.manifest.get("storage") or {}
        specification = (
            storage.get("default_result_root") if isinstance(storage, Mapping) else None
        )
        if not isinstance(specification, Mapping):
            return self.root / "results"

        scope = str(specification.get("scope") or "project_relative")
        raw_path = str(specification.get("path") or "results").strip()
        if scope == "project_relative":
            resolved = (self.root / raw_path).resolve()
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(
                    "Project-relative result storage must stay inside the project"
                ) from exc
            return resolved
        if scope == "absolute":
            return Path(raw_path).expanduser().resolve()
        raise ValueError(f"Unsupported result-storage scope: {scope!r}")

    def source_manifest(self) -> dict[str, Any]:
        if not self.source_manifest_path.is_file():
            return {
                "schema": 1,
                "format": SOURCE_MANIFEST_FORMAT,
                "project_id": self.project_id,
                "sources": [],
            }
        payload = json.loads(self.source_manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("Invalid source manifest")
        return payload

    def image_file_policy(self) -> str:
        """Return the series-wide policy for paired processed/RAW images."""

        tools = self.manifest.get("tools") or {}
        source_mapping = (
            tools.get("source_mapping") if isinstance(tools, Mapping) else {}
        )
        value = (
            source_mapping.get("image_file_policy")
            if isinstance(source_mapping, Mapping)
            else DEFAULT_IMAGE_FILE_POLICY
        )
        return normalize_image_file_policy(value)

    def current_source_records(self) -> list[dict[str, Any]]:
        """Return sources referenced by the current project configuration.

        Source records are immutable and therefore include superseded run mappings.
        Once runs have been mapped, the original series image root is retained as a
        discovery reference rather than scanned recursively. This keeps excluded
        sibling runs outside integrity scans while preserving source provenance.
        """

        payload = self.source_manifest()
        active_run_source_ids = {
            str(source_id)
            for row in self.manifest.get("runs") or []
            if isinstance(row, Mapping)
            and isinstance((mapping := row.get("source_mapping")), Mapping)
            for source_id in mapping.get("source_record_ids") or []
            if str(source_id)
        }
        has_mapped_runs = bool(active_run_source_ids)
        image_file_policy = self.image_file_policy()
        records: list[dict[str, Any]] = []
        for raw_source in payload.get("sources") or []:
            if not isinstance(raw_source, Mapping):
                continue
            source = deepcopy(dict(raw_source))
            source_id = str(source.get("source_id") or "")
            run_id = str(source.get("run_id") or "")
            if run_id and source_id not in active_run_source_ids:
                continue
            if (
                not run_id
                and has_mapped_runs
                and str(source.get("kind") or "") == "experiment_images"
            ):
                source["scan_policy"] = "reference_only"
                source["effective_role"] = "series_container"
            source.setdefault("image_file_policy", image_file_policy)
            records.append(source)
        return records

    def set_module_status(
        self,
        module_key: str,
        status: str,
        *,
        updated_by: str = "",
        reason: str = "",
    ) -> tuple[str, ...]:
        if module_key not in MODULE_BY_KEY:
            raise ValueError(f"Unknown FluidFlower module: {module_key!r}")
        if status not in VALID_MODULE_STATUSES:
            raise ValueError(f"Invalid module status: {status!r}")
        if status in {"approved", "complete"} and not dependencies_satisfied(
            module_key, self.module_states
        ):
            unmet = [
                MODULE_BY_KEY[key].title
                for key in MODULE_BY_KEY[module_key].dependencies
                if str(self.module_states.get(key, {}).get("status") or "")
                not in {"approved", "complete"}
            ]
            raise ValueError(
                "Upstream quality gates must be approved first: " + ", ".join(unmet)
            )
        now = utc_now()
        previous = str(self.module_states[module_key].get("status") or "")
        self.module_states[module_key].update(
            {
                "status": status,
                "updated_utc": now,
                "updated_by": updated_by.strip(),
                "reason": str(reason).strip(),
            }
        )
        invalidated: tuple[str, ...] = ()
        if previous in {"approved", "complete"} and status not in {
            "approved",
            "complete",
        }:
            invalidated = invalidate_downstream(
                self.module_states,
                module_key,
                updated_utc=now,
                updated_by=updated_by.strip(),
                reason=f"Upstream module {module_key} changed: {reason}".strip(),
            )
        self.manifest["updated_utc"] = now
        self._write_manifest()
        self._audit(
            "module_status_changed",
            updated_by,
            {
                "module": module_key,
                "previous_status": previous,
                "status": status,
                "reason": reason,
                "invalidated_modules": list(invalidated),
            },
        )
        return invalidated

    def update_metadata(
        self,
        section: str,
        values: Mapping[str, Any],
        *,
        updated_by: str = "",
        reason: str = "",
    ) -> None:
        if section not in {"rig", "chemistry", "protocol", "overflow", "tools"}:
            raise ValueError(f"Unsupported project metadata section: {section!r}")
        current = self.manifest.setdefault(section, {})
        if not isinstance(current, dict):
            raise ValueError(f"Project section {section!r} is not editable")
        before = deepcopy(current)
        current.update(deepcopy(dict(values)))
        self.manifest["updated_utc"] = utc_now()
        self._write_manifest()
        self._audit(
            "project_metadata_updated",
            updated_by,
            {
                "section": section,
                "before": before,
                "after": current,
                "reason": reason,
            },
        )

    def update_rig_type(
        self,
        rig_type: str,
        *,
        updated_by: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Version the apparatus class and preserve established rig metadata."""

        requested = str(rig_type).strip()
        if requested not in RIG_TYPES:
            raise ValueError(f"Unknown FluidFlower rig type: {requested!r}")
        current_rig = self.manifest.get("rig") or {}
        if not isinstance(current_rig, Mapping):
            raise ValueError("Project rig metadata is not editable")
        previous_type = str(current_rig.get("type") or "")
        if requested == previous_type:
            raise ValueError("The apparatus class has not changed.")

        actor = updated_by.strip()
        now = utc_now()
        new_manifest = deepcopy(self.manifest)
        rig = new_manifest.setdefault("rig", {})
        if not isinstance(rig, dict):
            raise ValueError("Project rig metadata is not editable")
        history = list(rig.get("class_history") or [])
        history.append(
            {
                "class_revision": int(rig.get("class_revision") or 1),
                "type": previous_type,
                "type_label": str(rig.get("type_label") or ""),
                "replaced_utc": now,
                "replaced_by": actor,
            }
        )
        rig.update(
            {
                "type": requested,
                "type_label": RIG_TYPES[requested],
                "class_revision": int(rig.get("class_revision") or 1) + 1,
                "class_history": history,
                "class_updated_utc": now,
                "class_updated_by": actor,
            }
        )

        reason_text = str(reason).strip() or "Apparatus class changed."
        invalidated_global = self._invalidate_geometry_after_rig_change(
            new_manifest["modules"],
            updated_utc=now,
            updated_by=actor,
            reason=reason_text,
        )
        invalidated_by_run: dict[str, list[str]] = {}
        for row in new_manifest.get("runs") or []:
            if not isinstance(row, dict):
                continue
            workflow = normalize_module_states(
                row.get("workflow")
                if isinstance(row.get("workflow"), Mapping)
                else new_manifest["modules"]
            )
            changed = self._invalidate_geometry_after_rig_change(
                workflow,
                updated_utc=now,
                updated_by=actor,
                reason=reason_text,
            )
            row["workflow"] = workflow
            if changed:
                invalidated_by_run[str(row.get("run_id") or "")] = list(changed)

        new_manifest["updated_utc"] = now
        _atomic_json(self.manifest_path, new_manifest)
        self.manifest = new_manifest
        report = {
            "previous_rig_type": previous_type,
            "rig_type": requested,
            "class_revision": rig["class_revision"],
            "global_modules_invalidated": list(invalidated_global),
            "run_modules_invalidated": invalidated_by_run,
        }
        self._audit(
            "project_rig_type_updated",
            actor,
            {**report, "reason": reason_text},
        )
        return report

    @staticmethod
    def _invalidate_geometry_after_rig_change(
        states: dict[str, dict[str, Any]],
        *,
        updated_utc: str,
        updated_by: str,
        reason: str,
    ) -> tuple[str, ...]:
        """Require renewed geometry review only when geometry work already exists."""

        geometry = states["geometry_facies"]
        status = str(geometry.get("status") or "not_started")
        if status not in {
            "ready",
            "in_progress",
            "review_required",
            "approved",
            "complete",
        }:
            return ()
        geometry.update(
            {
                "status": "review_required",
                "updated_utc": updated_utc,
                "updated_by": updated_by,
                "reason": reason,
            }
        )
        downstream = invalidate_downstream(
            states,
            "geometry_facies",
            updated_utc=updated_utc,
            updated_by=updated_by,
            reason=reason,
        )
        return ("geometry_facies", *downstream)

    @staticmethod
    def _reopen_data_integrity_after_source_change(
        manifest: dict[str, Any],
        *,
        updated_utc: str,
        updated_by: str,
        reason: str,
    ) -> tuple[str, ...]:
        """Reopen the series gate when its active source set changes."""

        states = normalize_module_states(
            manifest.get("modules")
            if isinstance(manifest.get("modules"), Mapping)
            else None
        )
        previous = str(states["data_integrity"].get("status") or "")
        states["data_integrity"].update(
            {
                "status": "ready",
                "updated_utc": updated_utc,
                "updated_by": updated_by,
                "reason": reason,
            }
        )
        downstream: tuple[str, ...] = ()
        if previous in {"approved", "complete"}:
            downstream = invalidate_downstream(
                states,
                "data_integrity",
                updated_utc=updated_utc,
                updated_by=updated_by,
                reason=reason,
            )
        manifest["modules"] = states
        return ("data_integrity", *downstream)

    def set_run_module_status(
        self,
        run_ids: Sequence[str],
        module_key: str,
        status: str,
        *,
        updated_by: str = "",
        reason: str = "",
    ) -> dict[str, tuple[str, ...]]:
        """Update a quality gate for selected runs without changing other runs."""

        if module_key not in MODULE_BY_KEY:
            raise ValueError(f"Unknown FluidFlower module: {module_key!r}")
        if status not in VALID_MODULE_STATUSES:
            raise ValueError(f"Invalid module status: {status!r}")
        selected = tuple(
            dict.fromkeys(slugify(value, fallback="") for value in run_ids)
        )
        if not selected or any(not value for value in selected):
            raise ValueError("At least one valid run ID is required")
        rows = [
            deepcopy(dict(row))
            for row in self.manifest.get("runs") or []
            if isinstance(row, Mapping)
        ]
        available = {str(row.get("run_id") or "") for row in rows}
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError("Unknown run IDs: " + ", ".join(missing))

        actor = updated_by.strip()
        now = utc_now()
        invalidated_by_run: dict[str, tuple[str, ...]] = {}
        for row in rows:
            run_id = str(row.get("run_id") or "")
            if run_id not in selected:
                continue
            workflow = normalize_module_states(
                row.get("workflow")
                if isinstance(row.get("workflow"), Mapping)
                else self.module_states
            )
            if status in {"approved", "complete"} and not dependencies_satisfied(
                module_key, workflow
            ):
                unmet = [
                    MODULE_BY_KEY[key].title
                    for key in MODULE_BY_KEY[module_key].dependencies
                    if str(workflow.get(key, {}).get("status") or "")
                    not in {"approved", "complete"}
                ]
                raise ValueError(
                    f"{run_id.upper()} has unmet run-specific quality gates: "
                    + ", ".join(unmet)
                )
            previous = str(workflow[module_key].get("status") or "")
            workflow[module_key].update(
                {
                    "status": status,
                    "updated_utc": now,
                    "updated_by": actor,
                    "reason": str(reason).strip(),
                }
            )
            invalidated: tuple[str, ...] = ()
            if previous in {"approved", "complete"} and status not in {
                "approved",
                "complete",
            }:
                invalidated = invalidate_downstream(
                    workflow,
                    module_key,
                    updated_utc=now,
                    updated_by=actor,
                    reason=(
                        f"Run-specific module {module_key} changed: {reason}"
                    ).strip(),
                )
            row["workflow"] = workflow
            invalidated_by_run[run_id] = invalidated
        self.manifest["runs"] = rows
        self.manifest["updated_utc"] = now
        self._write_manifest()
        projection_errors = self._write_run_projections(
            selected, self.source_manifest()
        )
        self._audit(
            "run_module_status_changed",
            actor,
            {
                "run_ids": list(selected),
                "module": module_key,
                "status": status,
                "reason": reason,
                "invalidated_modules": {
                    run_id: list(keys) for run_id, keys in invalidated_by_run.items()
                },
                "global_module_states_changed": False,
                "projection_errors": projection_errors,
            },
        )
        return invalidated_by_run

    @staticmethod
    def _normalized_run_row(raw: Mapping[str, Any]) -> dict[str, Any]:
        row = deepcopy(dict(raw))
        run_id = slugify(str(row.get("run_id") or ""), fallback="")
        if not run_id:
            raise ValueError("Run IDs must be non-empty")
        row["run_id"] = run_id
        mapping = row.get("source_mapping")
        if isinstance(mapping, Mapping):
            normalized_mapping = deepcopy(dict(mapping))
            normalized_mapping["run_id"] = run_id
            row["source_mapping"] = normalized_mapping
        return row

    @staticmethod
    def _mapping_issues(mapping: Mapping[str, Any]) -> list[str]:
        run_id = slugify(str(mapping.get("run_id") or ""), fallback="")
        issues: list[str] = []
        if not run_id:
            issues.append("Run ID is missing.")
        if str(mapping.get("candidate_status") or "") not in {
            "approved",
            "user_confirmed",
        }:
            issues.append(f"{run_id or 'Run'} source mapping has not been reviewed.")
        run_root = str(mapping.get("experiment_folder") or "").strip()
        if not run_root or not Path(run_root).expanduser().is_dir():
            issues.append(f"{run_id or 'Run'} run folder is unavailable.")
        co2_folders = [
            str(value).strip()
            for value in mapping.get("experiment_folders") or []
            if str(value).strip()
        ]
        if not co2_folders:
            issues.append(f"{run_id or 'Run'} has no CO2 image folder.")
        elif any(not Path(value).expanduser().is_dir() for value in co2_folders):
            issues.append(f"{run_id or 'Run'} has an unavailable CO2 image folder.")
        if int(mapping.get("image_count") or 0) < 1:
            issues.append(f"{run_id or 'Run'} has no discovered CO2 images.")
        return issues

    def validate_run_additions(
        self, mappings: Sequence[Mapping[str, Any]]
    ) -> tuple[str, ...]:
        """Validate an incremental import without changing project state."""

        issues: list[str] = []
        existing_rows = [
            dict(row)
            for row in self.manifest.get("runs") or []
            if isinstance(row, Mapping)
        ]
        existing_ids = {
            slugify(str(row.get("run_id") or ""), fallback="") for row in existing_rows
        }
        existing_roots = {
            _path_identity(str(mapping.get("experiment_folder")))
            for row in existing_rows
            if isinstance((mapping := row.get("source_mapping")), Mapping)
            and str(mapping.get("experiment_folder") or "").strip()
        }
        proposed_ids: set[str] = set()
        proposed_roots: set[str] = set()
        if not mappings:
            issues.append("No new experiments were selected.")
        for raw_mapping in mappings:
            mapping = dict(raw_mapping)
            run_id = slugify(str(mapping.get("run_id") or ""), fallback="")
            issues.extend(self._mapping_issues(mapping))
            if run_id in existing_ids:
                issues.append(f"Run ID already exists in this project: {run_id}")
            if run_id in proposed_ids:
                issues.append(f"Run ID occurs more than once in the import: {run_id}")
            proposed_ids.add(run_id)
            run_root = str(mapping.get("experiment_folder") or "").strip()
            if not run_root:
                continue
            identity = _path_identity(run_root)
            if identity in existing_roots:
                issues.append(
                    f"Run folder is already registered in this project: {run_root}"
                )
            if identity in proposed_roots:
                issues.append(f"Run folder occurs more than once: {run_root}")
            proposed_roots.add(identity)
        return tuple(dict.fromkeys(issues))

    @staticmethod
    def _register_mapping_sources(
        source_payload: dict[str, Any],
        mapping: Mapping[str, Any],
        *,
        run_id: str,
        revision_id: str,
    ) -> dict[str, Any]:
        normalized = deepcopy(dict(mapping))
        normalized["run_id"] = run_id
        source_ids: list[str] = []
        sources = source_payload.setdefault("sources", [])
        for kind, path, scan_policy in _mapping_source_specs(normalized):
            record = source_record(kind, path, run_id=run_id)
            record["source_revision_id"] = revision_id
            record["scan_policy"] = scan_policy
            record["image_file_policy"] = normalize_image_file_policy(
                normalized.get("image_file_policy") or DEFAULT_IMAGE_FILE_POLICY
            )
            sources.append(record)
            source_ids.append(str(record["source_id"]))
        normalized["source_record_ids"] = source_ids
        normalized["source_revision_id"] = revision_id
        return normalized

    @staticmethod
    def _refresh_source_mapping_indexes(manifest: dict[str, Any]) -> None:
        run_rows = [
            dict(row) for row in manifest.get("runs") or [] if isinstance(row, Mapping)
        ]
        mappings = [
            deepcopy(dict(mapping))
            for row in run_rows
            if isinstance((mapping := row.get("source_mapping")), Mapping)
        ]
        tools = manifest.setdefault("tools", {})
        if not isinstance(tools, dict):
            tools = {}
            manifest["tools"] = tools
        source_index = tools.setdefault("source_mapping", {})
        if not isinstance(source_index, dict):
            source_index = {}
            tools["source_mapping"] = source_index
        source_index["rows"] = mappings
        source_index["run_definitions"] = [
            {
                "run_id": str(row.get("run_id") or ""),
                "folder": str(mapping.get("experiment_folder") or ""),
                "included": True,
            }
            for row in run_rows
            if isinstance((mapping := row.get("source_mapping")), Mapping)
        ]
        source_index["baseline_and_clearwater_candidates_approved"] = bool(
            mappings
        ) and all(
            str(mapping.get("candidate_status") or "") in {"approved", "user_confirmed"}
            for mapping in mappings
        )
        source_index["reference_candidates_approved"] = source_index[
            "baseline_and_clearwater_candidates_approved"
        ]

        chemistry = manifest.setdefault("chemistry", {})
        if not isinstance(chemistry, dict):
            chemistry = {}
            manifest["chemistry"] = chemistry
        chemistry["runs_with_indicator_reference_images"] = [
            str(mapping.get("run_id") or "")
            for mapping in mappings
            if mapping.get("indicator_reference_folders")
        ]
        chemistry["runs_with_alternate_color_reference_images"] = [
            str(mapping.get("run_id") or "")
            for mapping in mappings
            if mapping.get("alternate_color_reference_folders")
        ]

        protocol = manifest.setdefault("protocol", {})
        if not isinstance(protocol, dict):
            protocol = {}
            manifest["protocol"] = protocol
        runs_with_protocols = [
            str(mapping.get("run_id") or "")
            for mapping in mappings
            if mapping.get("protocol_tables") or mapping.get("protocol_table")
        ]
        protocol.update(
            {
                "per_run_source_mapping": bool(runs_with_protocols),
                "run_count": len(run_rows),
                "runs_with_protocol_workbooks": runs_with_protocols,
            }
        )

        overflow = manifest.setdefault("overflow", {})
        if not isinstance(overflow, dict):
            overflow = {}
            manifest["overflow"] = overflow
        runs_with_images = [
            str(mapping.get("run_id") or "")
            for mapping in mappings
            if mapping.get("overflow_image_folders")
        ]
        runs_with_tables = [
            str(mapping.get("run_id") or "")
            for mapping in mappings
            if mapping.get("overflow_tables") or mapping.get("overflow_table")
        ]
        overflow.update(
            {
                "per_run_source_mapping": bool(runs_with_images or runs_with_tables),
                "run_count": len(run_rows),
                "runs_with_scale_images": runs_with_images,
                "runs_with_data_tables": runs_with_tables,
            }
        )
        if (runs_with_images or runs_with_tables) and str(
            overflow.get("mode") or "none"
        ) == "none":
            overflow["mode"] = "per_run"

    def _write_run_projections(
        self,
        run_ids: Sequence[str],
        source_payload: Mapping[str, Any],
    ) -> list[str]:
        """Write generated, read-only-friendly per-run indexes for navigation."""

        errors: list[str] = []
        sources_by_id = {
            str(row.get("source_id") or ""): dict(row)
            for row in source_payload.get("sources") or []
            if isinstance(row, Mapping)
        }
        rows_by_id = {
            str(row.get("run_id") or ""): dict(row)
            for row in self.manifest.get("runs") or []
            if isinstance(row, Mapping)
        }
        for run_id in run_ids:
            row = rows_by_id.get(run_id)
            if row is None:
                continue
            try:
                target = self.root / "runs" / run_id
                target.mkdir(parents=True, exist_ok=True)
                _atomic_json(
                    target / "run.json",
                    {
                        "schema": 1,
                        "generated": True,
                        "authoritative_file": "../../series.json",
                        "run": row,
                    },
                )
                mapping = row.get("source_mapping") or {}
                source_ids = list(mapping.get("source_record_ids") or [])
                _atomic_json(
                    target / "sources.json",
                    {
                        "schema": 1,
                        "generated": True,
                        "authoritative_file": "../../source_manifest.json",
                        "run_id": run_id,
                        "sources": [
                            sources_by_id[source_id]
                            for source_id in source_ids
                            if source_id in sources_by_id
                        ],
                    },
                )
            except OSError as exc:
                errors.append(f"{run_id}: {exc}")
        return errors

    def _commit_run_change(
        self,
        new_manifest: Mapping[str, Any],
        new_source_payload: Mapping[str, Any],
        *,
        revision_id: str,
        action: str,
        actor: str,
        details: Mapping[str, Any],
        changed_run_ids: Sequence[str],
    ) -> None:
        """Commit both project indexes with snapshots and normal-error rollback."""

        old_manifest = deepcopy(self.manifest)
        old_sources = self.source_manifest()
        transaction = self.root / "history" / "run-source-changes" / revision_id
        transaction.mkdir(parents=True, exist_ok=False)
        _atomic_json(transaction / "series.before.json", old_manifest)
        _atomic_json(transaction / "sources.before.json", old_sources)
        _atomic_json(transaction / "series.after.json", dict(new_manifest))
        _atomic_json(transaction / "sources.after.json", dict(new_source_payload))
        _atomic_json(
            transaction / "transaction.json",
            {
                "schema": 1,
                "revision_id": revision_id,
                "status": "prepared",
                "action": action,
                "recorded_utc": utc_now(),
            },
        )
        try:
            _atomic_json(self.source_manifest_path, dict(new_source_payload))
            _atomic_json(self.manifest_path, dict(new_manifest))
        except Exception:
            _atomic_json(self.source_manifest_path, old_sources)
            _atomic_json(self.manifest_path, old_manifest)
            self.manifest = old_manifest
            _atomic_json(
                transaction / "transaction.json",
                {
                    "schema": 1,
                    "revision_id": revision_id,
                    "status": "rolled_back",
                    "action": action,
                    "recorded_utc": utc_now(),
                },
            )
            raise
        self.manifest = deepcopy(dict(new_manifest))
        projection_errors = self._write_run_projections(
            changed_run_ids, new_source_payload
        )
        _atomic_json(
            transaction / "transaction.json",
            {
                "schema": 1,
                "revision_id": revision_id,
                "status": "committed",
                "action": action,
                "recorded_utc": utc_now(),
                "projection_errors": projection_errors,
            },
        )
        self._audit(
            action,
            actor,
            {
                **deepcopy(dict(details)),
                "revision_id": revision_id,
                "projection_errors": projection_errors,
            },
        )

    def append_runs(
        self,
        mappings: Sequence[Mapping[str, Any]],
        *,
        updated_by: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Append reviewed runs without replacing existing runs or results."""

        issues = self.validate_run_additions(mappings)
        if issues:
            raise ValueError("\n".join(issues))
        actor = updated_by.strip()
        now = utc_now()
        revision_id = f"run-add-{uuid.uuid4().hex}"
        new_manifest = deepcopy(self.manifest)
        new_sources = deepcopy(self.source_manifest())
        new_sources["updated_utc"] = now
        added_rows: list[dict[str, Any]] = []
        for raw_mapping in mappings:
            run_id = slugify(str(raw_mapping.get("run_id") or ""), fallback="")
            mapping = self._register_mapping_sources(
                new_sources,
                raw_mapping,
                run_id=run_id,
                revision_id=revision_id,
            )
            added_rows.append(
                {
                    "run_id": run_id,
                    "label": str(raw_mapping.get("label") or run_id.upper()),
                    "indicator_state": str(
                        raw_mapping.get("indicator_state") or "unknown"
                    ),
                    "added_utc": now,
                    "added_by": actor,
                    "source_revision": 1,
                    "source_updated_utc": now,
                    "source_mapping": mapping,
                    "workflow": _new_run_workflow(actor=actor, recorded_utc=now),
                }
            )
        new_manifest.setdefault("runs", []).extend(added_rows)
        new_manifest["updated_utc"] = now
        global_invalidated = self._reopen_data_integrity_after_source_change(
            new_manifest,
            updated_utc=now,
            updated_by=actor,
            reason="New experiments were added and require integrity review.",
        )
        self._refresh_source_mapping_indexes(new_manifest)
        run_ids = [str(row["run_id"]) for row in added_rows]
        self._commit_run_change(
            new_manifest,
            new_sources,
            revision_id=revision_id,
            action="project_runs_appended",
            actor=actor,
            details={
                "run_ids": run_ids,
                "run_count_before": len(self.manifest.get("runs") or []),
                "run_count_after": len(new_manifest.get("runs") or []),
                "global_module_states_changed": True,
                "global_modules_invalidated": list(global_invalidated),
                "reason": reason,
            },
            changed_run_ids=run_ids,
        )
        return {
            "revision_id": revision_id,
            "run_ids": run_ids,
            "run_count": len(run_ids),
            "global_module_states_changed": True,
            "global_modules_invalidated": list(global_invalidated),
        }

    def update_run_sources(
        self,
        run_id: str,
        mapping: Mapping[str, Any],
        *,
        updated_by: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Version the source mapping of one run and leave other runs untouched."""

        normalized_id = slugify(run_id, fallback="")
        normalized_mapping = deepcopy(dict(mapping))
        normalized_mapping["run_id"] = normalized_id
        issues = self._mapping_issues(normalized_mapping)
        if issues:
            raise ValueError("\n".join(issues))
        new_manifest = deepcopy(self.manifest)
        rows = [
            deepcopy(dict(row))
            for row in new_manifest.get("runs") or []
            if isinstance(row, Mapping)
        ]
        matching = [
            index
            for index, row in enumerate(rows)
            if row.get("run_id") == normalized_id
        ]
        if len(matching) != 1:
            raise ValueError(f"Unknown or ambiguous run ID: {normalized_id}")
        index = matching[0]
        previous = rows[index]
        previous_mapping = dict(previous.get("source_mapping") or {})
        if _mapping_signature(previous_mapping) == _mapping_signature(
            normalized_mapping
        ):
            raise ValueError("The reviewed source mapping has not changed.")

        actor = updated_by.strip()
        now = utc_now()
        revision_id = f"run-source-{uuid.uuid4().hex}"
        new_sources = deepcopy(self.source_manifest())
        new_sources["updated_utc"] = now
        registered_mapping = self._register_mapping_sources(
            new_sources,
            normalized_mapping,
            run_id=normalized_id,
            revision_id=revision_id,
        )
        history = list(previous.get("source_history") or [])
        history.append(
            {
                "source_revision": int(previous.get("source_revision") or 1),
                "replaced_utc": now,
                "replaced_by": actor,
                "source_mapping": previous_mapping,
            }
        )
        workflow = normalize_module_states(
            previous.get("workflow")
            if isinstance(previous.get("workflow"), Mapping)
            else self.module_states
        )
        data_status = str(workflow["data_integrity"].get("status") or "")
        workflow["data_integrity"].update(
            {
                "status": "ready",
                "updated_utc": now,
                "updated_by": actor,
                "reason": "Run source mapping changed and requires integrity review.",
            }
        )
        invalidated: tuple[str, ...] = ()
        if data_status in {"approved", "complete"}:
            invalidated = invalidate_downstream(
                workflow,
                "data_integrity",
                updated_utc=now,
                updated_by=actor,
                reason="Source mapping changed for this run.",
            )
        previous.update(
            {
                "source_mapping": registered_mapping,
                "source_history": history,
                "source_revision": int(previous.get("source_revision") or 1) + 1,
                "source_updated_utc": now,
                "source_updated_by": actor,
                "workflow": workflow,
            }
        )
        rows[index] = previous
        new_manifest["runs"] = rows
        new_manifest["updated_utc"] = now
        global_invalidated = self._reopen_data_integrity_after_source_change(
            new_manifest,
            updated_utc=now,
            updated_by=actor,
            reason="An experiment source mapping changed and requires integrity review.",
        )
        self._refresh_source_mapping_indexes(new_manifest)
        self._commit_run_change(
            new_manifest,
            new_sources,
            revision_id=revision_id,
            action="project_run_sources_updated",
            actor=actor,
            details={
                "run_id": normalized_id,
                "source_revision": previous["source_revision"],
                "run_modules_invalidated": list(invalidated),
                "global_module_states_changed": True,
                "global_modules_invalidated": list(global_invalidated),
                "reason": reason,
            },
            changed_run_ids=[normalized_id],
        )
        return {
            "revision_id": revision_id,
            "run_id": normalized_id,
            "source_revision": previous["source_revision"],
            "run_modules_invalidated": list(invalidated),
            "global_module_states_changed": True,
            "global_modules_invalidated": list(global_invalidated),
        }

    def validate_run_source_updates(
        self, mappings: Sequence[Mapping[str, Any]]
    ) -> tuple[str, ...]:
        """Validate one or more replacements of registered run mappings."""

        issues: list[str] = []
        existing_ids = {
            str(row.get("run_id") or "")
            for row in self.manifest.get("runs") or []
            if isinstance(row, Mapping)
        }
        proposed_ids: set[str] = set()
        if not mappings:
            issues.append("No experiments were selected for source remapping.")
        for raw_mapping in mappings:
            mapping = deepcopy(dict(raw_mapping))
            run_id = slugify(str(mapping.get("run_id") or ""), fallback="")
            mapping["run_id"] = run_id
            issues.extend(self._mapping_issues(mapping))
            if run_id not in existing_ids:
                issues.append(f"Run ID is not registered in this project: {run_id}")
            if run_id in proposed_ids:
                issues.append(f"Run ID occurs more than once: {run_id}")
            proposed_ids.add(run_id)
        return tuple(dict.fromkeys(issues))

    def apply_run_layout_template(
        self,
        mappings: Sequence[Mapping[str, Any]],
        layout_template: Mapping[str, Any],
        *,
        updated_by: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Atomically version a shared layout and selected run source mappings."""

        template = deepcopy(dict(layout_template))
        if not template.get("folders"):
            raise ValueError("The expected folder structure has no assigned folders.")
        image_file_policy = normalize_image_file_policy(
            template.get("image_file_policy") or self.image_file_policy()
        )
        if (
            "image_file_policy" in template
            or image_file_policy != DEFAULT_IMAGE_FILE_POLICY
        ):
            template["image_file_policy"] = image_file_policy
        validation_issues = self.validate_run_source_updates(mappings)
        if validation_issues:
            raise ValueError("\n".join(validation_issues))

        actor = updated_by.strip()
        now = utc_now()
        revision_id = f"run-layout-{uuid.uuid4().hex}"
        new_manifest = deepcopy(self.manifest)
        new_sources = deepcopy(self.source_manifest())
        new_sources["updated_utc"] = now

        tools = new_manifest.setdefault("tools", {})
        if not isinstance(tools, dict):
            tools = {}
            new_manifest["tools"] = tools
        source_index = tools.setdefault("source_mapping", {})
        if not isinstance(source_index, dict):
            source_index = {}
            tools["source_mapping"] = source_index
        previous_template = deepcopy(
            dict(source_index.get("run_layout_template") or {})
        )
        previous_template_revision = int(
            source_index.get("run_layout_template_revision")
            or (1 if previous_template else 0)
        )
        template_changed = _mapping_signature(previous_template) != _mapping_signature(
            template
        )
        template_revision = (
            previous_template_revision + 1
            if template_changed
            else max(1, previous_template_revision)
        )
        if template_changed and previous_template:
            template_history = list(
                source_index.get("run_layout_template_history") or []
            )
            template_history.append(
                {
                    "template_revision": previous_template_revision,
                    "replaced_utc": now,
                    "replaced_by": actor,
                    "template": previous_template,
                }
            )
            source_index["run_layout_template_history"] = template_history
        source_index.update(
            {
                "run_layout_template": template,
                "image_file_policy": image_file_policy,
                "run_layout_template_revision": template_revision,
                "run_layout_template_updated_utc": now,
                "run_layout_template_updated_by": actor,
            }
        )

        normalized_by_id: dict[str, dict[str, Any]] = {}
        for raw_mapping in mappings:
            mapping = deepcopy(dict(raw_mapping))
            run_id = slugify(str(mapping.get("run_id") or ""), fallback="")
            mapping["run_id"] = run_id
            mapping["layout_template_revision"] = template_revision
            normalized_by_id[run_id] = mapping

        changed_run_ids: list[str] = []
        unchanged_run_ids: list[str] = []
        invalidated_by_run: dict[str, list[str]] = {}
        rows = [
            deepcopy(dict(row))
            for row in new_manifest.get("runs") or []
            if isinstance(row, Mapping)
        ]
        for row in rows:
            run_id = str(row.get("run_id") or "")
            mapping = normalized_by_id.get(run_id)
            if mapping is None:
                continue
            previous_mapping = deepcopy(dict(row.get("source_mapping") or {}))
            if _mapping_signature(previous_mapping) == _mapping_signature(mapping):
                unchanged_run_ids.append(run_id)
                continue
            registered_mapping = self._register_mapping_sources(
                new_sources,
                mapping,
                run_id=run_id,
                revision_id=revision_id,
            )
            history = list(row.get("source_history") or [])
            history.append(
                {
                    "source_revision": int(row.get("source_revision") or 1),
                    "replaced_utc": now,
                    "replaced_by": actor,
                    "source_mapping": previous_mapping,
                }
            )
            workflow = normalize_module_states(
                row.get("workflow")
                if isinstance(row.get("workflow"), Mapping)
                else self.module_states
            )
            data_status = str(workflow["data_integrity"].get("status") or "")
            workflow["data_integrity"].update(
                {
                    "status": "ready",
                    "updated_utc": now,
                    "updated_by": actor,
                    "reason": (
                        "Expected folder structure was applied; source mapping "
                        "requires integrity review."
                    ),
                }
            )
            invalidated: tuple[str, ...] = ()
            if data_status in {"approved", "complete"}:
                invalidated = invalidate_downstream(
                    workflow,
                    "data_integrity",
                    updated_utc=now,
                    updated_by=actor,
                    reason="Expected folder structure changed this run's sources.",
                )
            row.update(
                {
                    "source_mapping": registered_mapping,
                    "source_history": history,
                    "source_revision": int(row.get("source_revision") or 1) + 1,
                    "source_updated_utc": now,
                    "source_updated_by": actor,
                    "workflow": workflow,
                }
            )
            changed_run_ids.append(run_id)
            if invalidated:
                invalidated_by_run[run_id] = list(invalidated)

        metadata_changed = template_changed or not previous_template_revision
        if not changed_run_ids and not metadata_changed:
            raise ValueError(
                "The expected folder structure and selected mappings have not changed."
            )
        new_manifest["runs"] = rows
        new_manifest["updated_utc"] = now
        global_invalidated: tuple[str, ...] = ()
        if changed_run_ids:
            global_invalidated = self._reopen_data_integrity_after_source_change(
                new_manifest,
                updated_utc=now,
                updated_by=actor,
                reason=(
                    "The expected folder structure changed active experiment sources; "
                    "a new integrity report is required."
                ),
            )
        self._refresh_source_mapping_indexes(new_manifest)
        reason_text = str(reason).strip() or "Expected folder structure applied."
        self._commit_run_change(
            new_manifest,
            new_sources,
            revision_id=revision_id,
            action="project_run_layout_applied",
            actor=actor,
            details={
                "template_revision": template_revision,
                "template_changed": template_changed,
                "changed_run_ids": changed_run_ids,
                "unchanged_run_ids": unchanged_run_ids,
                "run_modules_invalidated": invalidated_by_run,
                "global_module_states_changed": bool(changed_run_ids),
                "global_modules_invalidated": list(global_invalidated),
                "reason": reason_text,
            },
            changed_run_ids=changed_run_ids,
        )
        return {
            "revision_id": revision_id,
            "template_revision": template_revision,
            "template_changed": template_changed,
            "changed_run_ids": changed_run_ids,
            "unchanged_run_ids": unchanged_run_ids,
            "run_modules_invalidated": invalidated_by_run,
            "global_module_states_changed": bool(changed_run_ids),
            "global_modules_invalidated": list(global_invalidated),
        }

    def set_runs(
        self,
        rows: list[Mapping[str, Any]],
        *,
        updated_by: str = "",
        reason: str = "",
    ) -> None:
        """Replace editable run metadata while retaining an audit snapshot."""

        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in rows:
            row = deepcopy(dict(raw))
            run_id = slugify(str(row.get("run_id") or ""), fallback="")
            if not run_id or run_id in seen:
                raise ValueError("Run IDs must be non-empty and unique")
            seen.add(run_id)
            row["run_id"] = run_id
            normalized.append(row)
        before = deepcopy(list(self.manifest.get("runs") or []))
        actor = updated_by.strip()
        now = utc_now()
        revision_id = f"run-set-{uuid.uuid4().hex}"
        new_manifest = deepcopy(self.manifest)
        new_sources = deepcopy(self.source_manifest())
        changed_ids: list[str] = []
        prepared: list[dict[str, Any]] = []
        for row in normalized:
            run_id = str(row["run_id"])
            changed_ids.append(run_id)
            mapping = row.get("source_mapping")
            if isinstance(mapping, Mapping):
                row["source_mapping"] = self._register_mapping_sources(
                    new_sources,
                    mapping,
                    run_id=run_id,
                    revision_id=revision_id,
                )
                row.setdefault("source_revision", 1)
                row.setdefault("source_updated_utc", now)
            row.setdefault("added_utc", now)
            row.setdefault("added_by", actor)
            row.setdefault("workflow", _new_run_workflow(actor=actor, recorded_utc=now))
            prepared.append(row)
        new_manifest["runs"] = prepared
        new_manifest["updated_utc"] = now
        new_sources["updated_utc"] = now
        self._refresh_source_mapping_indexes(new_manifest)
        self._commit_run_change(
            new_manifest,
            new_sources,
            revision_id=revision_id,
            action="project_runs_updated",
            actor=actor,
            details={"before": before, "after": prepared, "reason": reason},
            changed_run_ids=changed_ids,
        )

    def _write_manifest(self) -> None:
        _atomic_json(self.manifest_path, self.manifest)

    def _audit(self, action: str, actor: str, details: Mapping[str, Any]) -> None:
        _append_jsonl(
            self.audit_path,
            {
                "schema": 1,
                "event_id": uuid.uuid4().hex,
                "recorded_utc": utc_now(),
                "action": action,
                "actor": str(actor).strip(),
                "details": deepcopy(dict(details)),
            },
        )


class ProjectRegistry:
    """Small local index of projects; project folders remain authoritative."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or default_registry_path()).expanduser().resolve()

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, TypeError):
            return []
        if not isinstance(payload, Mapping) or payload.get("format") != REGISTRY_FORMAT:
            return []
        rows = [
            dict(row)
            for row in payload.get("projects") or []
            if isinstance(row, Mapping)
        ]
        return sorted(
            rows, key=lambda row: str(row.get("last_opened_utc") or ""), reverse=True
        )

    def remember(self, project: FluidFlowerProject) -> None:
        rows = self.entries()
        rows = [row for row in rows if str(row.get("project_id")) != project.project_id]
        rows.insert(
            0,
            {
                "project_id": project.project_id,
                "name": project.name,
                "series_id": str(project.manifest.get("series_id") or ""),
                "path": str(project.root),
                "template_id": str(
                    (project.manifest.get("template") or {}).get("id") or ""
                ),
                "run_count": len(project.manifest.get("runs") or []),
                "last_opened_utc": utc_now(),
            },
        )
        _atomic_json(
            self.path,
            {
                "schema": 1,
                "format": REGISTRY_FORMAT,
                "updated_utc": utc_now(),
                "projects": rows[:100],
            },
        )

    def forget_path(self, path: Path | str) -> bool:
        """Remove one local registration, leaving project files and jobs alone."""
        from .storage_paths import same_storage_path
        if not self.path.is_file():
            return False
        original = self.path.read_bytes()
        payload = json.loads(original.decode("utf-8-sig"))
        if not isinstance(payload, dict) or payload.get("format") != REGISTRY_FORMAT:
            raise ValueError("Unsupported local project registry")
        rows = payload.get("projects") or []
        kept = [row for row in rows if not isinstance(row, Mapping) or not same_storage_path(row.get("path", ""), path)]
        if len(kept) == len(rows):
            return False
        payload.update(projects=kept, updated_utc=utc_now())
        _atomic_json(self.path, payload, expected_raw=original)
        return True

    def forget_missing(self) -> int:
        rows = self.entries()
        kept = [
            row
            for row in rows
            if (Path(str(row.get("path") or "")) / "series.json").is_file()
        ]
        removed = len(rows) - len(kept)
        if removed:
            _atomic_json(
                self.path,
                {
                    "schema": 1,
                    "format": REGISTRY_FORMAT,
                    "updated_utc": utc_now(),
                    "projects": kept,
                },
            )
        return removed
