"""UI-only planning and explicit application of exported native crop geometry.

The existing frame-registration and native crop implementations are reused.
Creating a plan reads inputs and may register photographs; it never writes a
series file. Only ``apply_crop_plan`` changes the explicitly planned files.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid
import tomllib

import toml

from darsia.fluidflower.camera_registration import register_frame_edges
from darsia.fluidflower.series_geometry import (
    CROP_CORNERS,
    NATIVE_SETUP_EXPORT_FORMAT,
    RigFrame,
    photo_size,
    series_run_inputs,
    _storage_path,
)
from darsia.fluidflower.setup_crop_area import (
    EXPORT_KEY,
    CropArea,
    CropExtent,
    crop_extent,
)
from darsia.fluidflower.storage_paths import storage_path_remapper


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return _key(first) == _key(second)


def _toml(data: bytes) -> dict[str, Any]:
    return tomllib.loads(data.decode("utf-8-sig"))


@dataclass(frozen=True)
class ReadBinding:
    path: Path
    data: bytes

    def require_unchanged(self) -> None:
        from .storage_paths import resolve_storage_path
        if resolve_storage_path(self.path).read_bytes() != self.data:
            raise RuntimeError(f"File changed after geometry was prepared: {self.path}")


@dataclass(frozen=True)
class PhotoBinding:
    path: Path
    size: int
    mtime_ns: int

    @classmethod
    def read(cls, path: Path) -> "PhotoBinding":
        stat = path.stat()
        return cls(path, stat.st_size, stat.st_mtime_ns)

    def require_unchanged(self) -> None:
        if self != PhotoBinding.read(self.path):
            raise RuntimeError(f"Photograph changed after registration: {self.path}")


@dataclass(frozen=True)
class CropFileChange:
    path: Path
    run_id: str
    before: bytes
    after: bytes


@dataclass(frozen=True)
class CropApplyPlan:
    series_root: Path
    export_json: Path
    revision_id: str
    bindings: tuple[ReadBinding, ...]
    photographs: tuple[PhotoBinding, ...]
    changes: tuple[CropFileChange, ...]
    runs: tuple[dict[str, Any], ...]

    def require_unchanged(self) -> None:
        for binding in self.bindings:
            binding.require_unchanged()
        for photograph in self.photographs:
            photograph.require_unchanged()


class CropApplyError(RuntimeError):
    """A failed application with an honest record of possible partial writes."""

    def __init__(self, message: str, *, applied: tuple[Path, ...], receipt: Path):
        super().__init__(message)
        self.applied = applied
        self.receipt = receipt


def _frame(
    payload: Mapping[str, Any], *, image: Path, size: tuple[int, int]
) -> RigFrame:
    raw = payload.get("corners")
    if not isinstance(raw, Mapping) or any(name not in raw for name in CROP_CORNERS):
        raise ValueError("The exported rig frame needs all four named corners.")
    corners = {}
    for name in CROP_CORNERS:
        point = raw[name]
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Invalid exported rig corner: {name}")
        values = tuple(float(value) for value in point)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite exported rig corner: {name}")
        corners[name] = values
    if len(set(corners.values())) != 4:
        raise ValueError("The exported rig corners are not distinct.")
    recorded_size = payload.get("image_size")
    if recorded_size is not None and tuple(recorded_size) != tuple(size):
        raise ValueError(f"The rig frame belongs to another photograph size: {image}")
    width = float(payload.get("width_m") or 0.0)
    height = float(payload.get("height_m") or 0.0)
    if not all(math.isfinite(value) and value > 0.0 for value in (width, height)):
        raise ValueError("The exported rig frame needs its physical width and height.")
    result = RigFrame(
        corners=corners,
        width_m=width,
        height_m=height,
        run_id=str(payload.get("run_id") or ""),
        source=str(payload.get("source") or "exported geometry"),
        image=str(image),
        image_size=size,
    )
    # Reuse the frame's existing coordinate implementation to reject a singular
    # frame, without changing corner names or introducing a quality threshold.
    result.pixel_to_unit(*corners[CROP_CORNERS[0]])
    return result


def crop_config_bytes(
    original: bytes,
    crop: Mapping[str, Any],
    *,
    inherited: Mapping[str, Any] | None = None,
) -> bytes:
    """Change only native crop keys and necessary existing activation lists.

    A crop without ``origin`` is the reference quadrilateral's crop, so an
    origin left by an earlier separate crop area is removed.
    """
    parsed = _toml(original)
    changed = deepcopy(parsed)
    corrections = changed.setdefault("corrections", {})
    if not isinstance(corrections, dict):
        raise ValueError("[corrections] is not a table.")
    curvature = corrections.setdefault("curvature", {})
    if not isinstance(curvature, dict):
        raise ValueError("[corrections.curvature] is not a table.")
    target = curvature.setdefault("crop", {})
    if not isinstance(target, dict):
        raise ValueError("[corrections.curvature.crop] is not a table.")
    target.update(deepcopy(dict(crop)))
    if "origin" not in crop:
        target.pop("origin", None)
    inherited_corrections = (inherited or {}).get("corrections") or {}
    inherited_curvature = inherited_corrections.get("curvature") or {}
    for section, previous, name in (
        (corrections, inherited_corrections, "curvature"),
        (curvature, inherited_curvature, "crop"),
    ):
        effective_active = section.get("active", previous.get("active"))
        if effective_active is not None:
            active = deepcopy(effective_active)
            if not isinstance(active, list) or any(
                not isinstance(item, str) for item in active
            ):
                raise ValueError("A corrections active list must contain strings.")
            if name not in active:
                active.append(name)
                section["active"] = active
    if changed == parsed:
        return original
    text = toml.dumps(changed)
    if _toml(text.encode("utf-8")) != changed:
        raise ValueError(
            "The native crop update did not preserve TOML semantic values."
        )
    if b"\r\n" in original:
        text = text.replace("\n", "\r\n")
    return (
        b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
    ) + text.encode("utf-8")


def _baseline_from_config(config: Mapping[str, Any], root: Path) -> Path | None:
    data = config.get("data") or {}
    baseline = str(data.get("baseline") or "").strip()
    if not baseline:
        return None
    folders = []
    for value in [data.get("folder"), *(data.get("folders") or [])]:
        folder = _storage_path(value, root)
        if folder is not None and folder not in folders:
            folders.append(folder)
    candidate = _storage_path(baseline, root)
    if Path(baseline).is_absolute():
        return candidate
    choices = [folder / baseline for folder in folders]
    return next(
        (path for path in choices if path.is_file()), choices[0] if choices else None
    )


def _separate_crop_extent(
    export: Mapping[str, Any], source_frame: RigFrame, source: Path
) -> CropExtent | None:
    """The exported separate crop area's rectangle; None for the default crop.

    The area must be marked on the exported rig frame's own photograph, so
    that each run's registration moves the frame and the area together.
    """
    payload = export.get(EXPORT_KEY)
    if payload is None:
        return None
    area = CropArea.from_dict(payload)
    image = Path(storage_path_remapper()(area.image)) if area.image else None
    if image is None or not _same_file(image, source):
        raise ValueError(
            "The separate crop area was not marked on the exported rig frame's photograph."
        )
    return crop_extent(source_frame, area)


def _registered_corners(
    mapped: Any, corners: Mapping[str, tuple[float, float]], run_id: str
) -> dict[str, tuple[float, float]]:
    """Named (row, col) corners moved by one run's registration; refuse invalid ones."""
    points = mapped.map_points([corners[name] for name in CROP_CORNERS])
    result = {
        name: tuple(float(value) for value in point)
        for name, point in zip(CROP_CORNERS, points)
    }
    if len(result) != 4 or any(
        len(point) != 2 or not all(math.isfinite(value) for value in point)
        for point in result.values()
    ):
        raise ValueError(f"{run_id}: registration returned invalid corners.")
    return result


def prepare_crop_plan(
    series_root: str | Path,
    export_json: str | Path,
    *,
    expected_revision_id: str,
    register: Callable[..., Any] = register_frame_edges,
    get_photo_size: Callable[[Path], tuple[int, int]] = photo_size,
) -> CropApplyPlan:
    """Prepare every run crop from one frozen export; do not write anything.

    Target crops are measured from the export's source photograph for each
    run. An old run-config crop is not substituted for a registration. Any
    registration error aborts the whole plan, with the original error intact.

    When the export records a separate crop area, each run is still
    registered through the reference quadrilateral, the area's rectangle
    moves with that registration, and the crop keeps the reference
    quadrilateral's rig coordinates through its origin.
    """
    from .storage_paths import resolve_storage_path
    root, export_path = resolve_storage_path(series_root), resolve_storage_path(export_json)
    bindings: dict[str, ReadBinding] = {}

    def read(path: Path) -> bytes:
        key = _key(path)
        if key not in bindings:
            bindings[key] = ReadBinding(path, path.read_bytes())
        return bindings[key].data

    export = json.loads(read(export_path))
    if export.get("format") != NATIVE_SETUP_EXPORT_FORMAT:
        raise ValueError("Not a native geometry export.")
    if not expected_revision_id or export.get("revision_id") != expected_revision_id:
        raise ValueError("The geometry export is not the current selected revision.")
    frame_data = export.get("rig_frame")
    if not isinstance(frame_data, Mapping):
        raise ValueError("The native geometry export has no rig frame.")
    source_name = str(frame_data.get("image") or "").strip()
    if not source_name:
        raise ValueError("The exported rig frame does not identify its raw photograph.")
    # The export keeps the absolute path it was made with; this machine may
    # reach the same raw photograph through a [storage.path_aliases] prefix.
    source = Path(storage_path_remapper()(source_name))
    if not source.is_absolute():
        raise ValueError("The exported rig frame photograph must be an absolute path.")
    source_photo = PhotoBinding.read(source)
    source_frame = _frame(frame_data, image=source, size=get_photo_size(source))
    extent = _separate_crop_extent(export, source_frame, source)
    manifest = _toml(read(root / "series.toml"))
    paths = manifest.get("paths") or {}
    common = root / str(paths.get("common_config") or "config/common.toml")
    common_data = _toml(read(common)) if common.is_file() else {}
    merged_folder = root / str(paths.get("merged_configs") or "config/merged")
    # Capture every run file before the original resolver reads it, so that a
    # concurrent edit cannot change baseline bindings invisibly between reads.
    run_table = manifest.get("runs") or {}
    run_names = run_table.get("config_files") or {}
    run_folder = str(paths.get("run_configs") or "config/runs")
    for run_id in run_table.get("ids") or []:
        path = _storage_path(
            run_names.get(str(run_id)) or f"{run_folder}/{run_id}.toml", root
        )
        if path is None:
            raise ValueError(f"{run_id}: no run configuration path.")
        read(path)
        merged = merged_folder / f"{run_id}.toml"
        if merged.is_file():
            read(merged)
    rows = series_run_inputs(root, check_files=True)
    if not rows:
        raise ValueError("The selected series has no runs.")
    if (
        not source_frame.run_id
        or export.get("reference_run") != source_frame.run_id
        or source_frame.run_id not in {row.run_id for row in rows}
    ):
        raise ValueError(
            "The exported source frame is not bound to a run of this series."
        )
    photos = {_key(source): source_photo}
    changes: list[CropFileChange] = []
    reports: list[dict[str, Any]] = []
    target_owners: dict[str, str] = {}
    for row in rows:
        if row.issues:
            raise ValueError(f"{row.run_id}: {' '.join(row.issues)}")
        target = row.baseline_image
        if target is None:
            raise ValueError(f"{row.run_id}: no raw baseline photograph.")
        photos.setdefault(_key(target), PhotoBinding.read(target))
        target_size = get_photo_size(target)
        if _same_file(source, target):
            if tuple(target_size) != tuple(source_frame.image_size or ()):
                raise ValueError(f"The source photograph dimensions changed: {target}")
            corners = deepcopy(source_frame.corners)
            area_corners = None if extent is None else dict(extent.corners)
            registration = {"kind": "same_raw_photograph"}
        else:
            mapped = register(source, target, source_frame.corners)
            if tuple(mapped.source_size) != tuple(
                source_frame.image_size or ()
            ) or tuple(mapped.target_size) != tuple(target_size):
                raise ValueError(
                    f"{row.run_id}: registration belongs to another photograph size."
                )
            corners = _registered_corners(mapped, source_frame.corners, row.run_id)
            area_corners = (
                None
                if extent is None
                else _registered_corners(mapped, extent.corners, row.run_id)
            )
            registration = mapped.to_dict()
        frame = RigFrame(
            corners=corners,
            width_m=source_frame.width_m,
            height_m=source_frame.height_m,
            run_id=row.run_id,
            source="exported reference frame registered to raw baseline",
            image=str(target),
            image_size=tuple(target_size),
        )
        crop = frame.native_crop() if extent is None else extent.native_crop(area_corners)
        config_paths = [row.config_path]
        merged = merged_folder / f"{row.run_id}.toml"
        if merged.is_file() and not _same_file(merged, row.config_path):
            merged_baseline = _baseline_from_config(_toml(read(merged)), root)
            if merged_baseline is None or not _same_file(merged_baseline, target):
                raise ValueError(
                    f"{row.run_id}: run and merged configurations identify different raw baselines."
                )
            config_paths.append(merged)
        for path in config_paths:
            key = _key(path)
            if key in target_owners and target_owners[key] != row.run_id:
                raise ValueError(f"Two runs share the same crop configuration: {path}")
            target_owners[key] = row.run_id
            before = read(path)
            after = crop_config_bytes(
                before,
                crop,
                inherited=common_data if _same_file(path, row.config_path) else None,
            )
            if after != before:
                changes.append(CropFileChange(path, row.run_id, before, after))
        reports.append(
            {
                "run_id": row.run_id,
                "baseline": str(target),
                "crop": crop,
                "registration": registration,
                "config_paths": [str(path) for path in config_paths],
            }
        )
    plan = CropApplyPlan(
        root,
        export_path,
        expected_revision_id,
        tuple(bindings.values()),
        tuple(photos.values()),
        tuple(changes),
        tuple(reports),
    )
    plan.require_unchanged()
    return plan


def _exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def apply_crop_plan(plan: CropApplyPlan, *, receipt_directory: str | Path) -> Path:
    """Explicitly apply a prepared plan, preserving original bytes in backups.

    Each target replacement is atomic. NAS cannot promise an atomic operation
    spanning multiple files, so failure records exact partial completion and
    leaves all backups/staged files intact. No automatic retry or rollback.
    """
    plan.require_unchanged()
    attempt = Path(receipt_directory) / f"crop-apply-{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    receipt_path = attempt / "receipt.json"
    backups = attempt / "before"
    backups.mkdir()
    applied: list[Path] = []
    files = []
    staged = []
    receipt: dict[str, Any] = {
        "format": "darsia-native-crop-apply-v1",
        "state": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "series_root": str(plan.series_root),
        "export_json": str(plan.export_json),
        "revision_id": plan.revision_id,
        "runs": list(plan.runs),
        "files": files,
    }
    try:
        for index, change in enumerate(plan.changes):
            backup = backups / f"{index:04d}-{change.path.name}"
            _exclusive(backup, change.before)
            temporary = change.path.with_name(
                f".{change.path.name}.{uuid.uuid4().hex}.tmp"
            )
            _exclusive(temporary, change.after)
            staged.append((change, temporary))
            files.append(
                {
                    "path": str(change.path),
                    "run_id": change.run_id,
                    "backup": str(backup),
                    "before_sha256": _sha(change.before),
                    "after_sha256": _sha(change.after),
                    "state": "staged",
                }
            )
        plan.require_unchanged()
        for index, (change, temporary) in enumerate(staged):
            already_applied = {_key(path) for path in applied}
            for binding in plan.bindings:
                if _key(binding.path) not in already_applied:
                    binding.require_unchanged()
            if change.path.read_bytes() != change.before:
                raise RuntimeError(
                    f"File changed before geometry replacement: {change.path}"
                )
            for photograph in plan.photographs:
                photograph.require_unchanged()
            os.replace(temporary, change.path)
            applied.append(change.path)
            files[index]["state"] = "applied"
        receipt["state"] = "complete"
    except Exception as error:
        receipt["state"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["applied_paths"] = [str(path) for path in applied]
        _exclusive(
            receipt_path,
            (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        raise CropApplyError(
            str(error), applied=tuple(applied), receipt=receipt_path
        ) from error
    _exclusive(
        receipt_path,
        (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return receipt_path
