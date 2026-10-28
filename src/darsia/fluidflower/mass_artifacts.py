"""Immutable streaming artifact formats shared by mass and downstream analyses."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from darsia.fluidflower.execution import verify_file_reference
from darsia.fluidflower.project import sha256_file

FIELD_INDEX_FORMAT = "darsia-fluidflower-field-index-v1"
FIELD_FRAME_FORMAT = "darsia-fluidflower-field-frame-v1"
PREVIEW_INDEX_FORMAT = "darsia-fluidflower-preview-index-v1"

MASS_FIELD_PRODUCTS = ("mass_total", "mass_aqueous", "mass_free_gas")
PHASE_FIELD_PRODUCTS = ("concentration_aqueous", "saturation_free_gas")
TRANSPORT_FIELD_PRODUCTS = ("transport_weight",)
MASS_INPUT_DYNAMIC_PRODUCTS = (
    "optical_acidification",
    "phase_co2_probability",
    "phase_free_gas_probability",
)
MASS_INPUT_STATIC_PRODUCTS = ("depth", "porosity", "permeability")
MASS_INPUT_FIELD_PRODUCTS = (
    *MASS_INPUT_DYNAMIC_PRODUCTS,
    *MASS_INPUT_STATIC_PRODUCTS,
)
INDEXED_FIELD_PRODUCTS = (
    *MASS_FIELD_PRODUCTS,
    *PHASE_FIELD_PRODUCTS,
    *TRANSPORT_FIELD_PRODUCTS,
    *MASS_INPUT_FIELD_PRODUCTS,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except (FileNotFoundError, OSError):
            pass


def _finite_times(values: Sequence[Any]) -> list[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("Artifact times must be finite")
    if result != sorted(set(result)):
        raise ValueError("Artifact times must be unique and strictly increasing")
    return result


def write_field_frame(
    path: Path,
    values: np.ndarray,
    *,
    run_id: str,
    product: str,
    frame_id: str,
    time_hours: float,
    display_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write one numerical frame and return its immutable index reference."""

    if product not in INDEXED_FIELD_PRODUCTS:
        raise ValueError(f"Unsupported indexed field product: {product}")
    array = np.asarray(values)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Field frames require one 2D floating-point values array")
    if not np.all(np.isfinite(array)):
        raise ValueError("Field frames cannot contain NaN or infinite values")
    payload: dict[str, np.ndarray] = {
        "format": np.asarray(FIELD_FRAME_FORMAT),
        "run_id": np.asarray(str(run_id)),
        "product": np.asarray(str(product)),
        "frame_id": np.asarray(str(frame_id)),
        "time_hours": np.asarray(float(time_hours), dtype=np.float64),
        "values": array,
    }
    if display_values is not None:
        display = np.asarray(display_values)
        if display.shape != array.shape or not np.issubdtype(
            display.dtype, np.floating
        ):
            raise ValueError("display_values must match the numerical field shape")
        if not np.all(np.isfinite(display)):
            raise ValueError("display_values cannot contain non-finite values")
        payload["display_values"] = display
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except (FileNotFoundError, OSError):
            pass
    return {
        "frame_id": str(frame_id),
        "time_hours": float(time_hours),
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape_rc": [int(array.shape[0]), int(array.shape[1])],
        "dtype": str(array.dtype),
        "has_display_values": display_values is not None,
    }


def write_field_index(
    path: Path,
    *,
    run_id: str,
    product: str,
    quantity: str,
    unit: str,
    frames: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one transitive field index whose frame files remain independently read."""

    if product not in INDEXED_FIELD_PRODUCTS:
        raise ValueError(f"Unsupported indexed field product: {product}")
    rows = [deepcopy(dict(row)) for row in frames]
    _finite_times([row.get("time_hours") for row in rows])
    if not rows or any(not row.get("path") or not row.get("sha256") for row in rows):
        raise ValueError("Every indexed field frame requires path and SHA-256")
    payload = {
        "schema": 1,
        "format": FIELD_INDEX_FORMAT,
        "run_id": str(run_id),
        "product": str(product),
        "quantity": str(quantity),
        "unit": str(unit),
        "grid": deepcopy(dict(grid)),
        "frames": rows,
    }
    _atomic_json(path, payload)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": FIELD_INDEX_FORMAT,
        "frame_count": len(rows),
    }


def write_preview_index(
    path: Path,
    *,
    run_id: str,
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write corrected-frame references used only as immutable visual backgrounds."""

    rows = [deepcopy(dict(row)) for row in frames]
    _finite_times([row.get("time_hours") for row in rows])
    if not rows or any(not row.get("path") or not row.get("sha256") for row in rows):
        raise ValueError("Every preview frame requires path and SHA-256")
    payload = {
        "schema": 1,
        "format": PREVIEW_INDEX_FORMAT,
        "run_id": str(run_id),
        "frames": rows,
    }
    _atomic_json(path, payload)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": PREVIEW_INDEX_FORMAT,
        "frame_count": len(rows),
    }


def _read_index(
    path_value: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path = verify_file_reference(path_value, expected_sha256, label=label)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return path, payload


@dataclass(frozen=True)
class IndexedFieldProduct:
    """Lazy, hash-verifying access to one run/product field sequence."""

    index_path: Path
    run_id: str
    product: str
    quantity: str
    unit: str
    grid: dict[str, Any]
    frames: tuple[dict[str, Any], ...]

    @property
    def times_hours(self) -> tuple[float, ...]:
        return tuple(float(row["time_hours"]) for row in self.frames)

    @property
    def shape_rc(self) -> tuple[int, int]:
        shape = self.frames[0].get("shape_rc") or []
        if len(shape) != 2:
            raise ValueError(f"Field index has no valid shape: {self.index_path}")
        return int(shape[0]), int(shape[1])

    def load_frame(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        row = self.frames[index]
        path = verify_file_reference(
            str(row["path"]),
            str(row["sha256"]),
            label=f"{self.run_id}/{self.product}/{row['frame_id']}",
            relative_to=self.index_path.parent,
        )
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "format",
                "run_id",
                "product",
                "frame_id",
                "time_hours",
                "values",
            }
            if not required.issubset(archive.files):
                raise ValueError(f"Incomplete field frame: {path}")
            if str(archive["format"].item()) != FIELD_FRAME_FORMAT:
                raise ValueError(f"Unsupported field frame: {path}")
            identity = (
                str(archive["run_id"].item()),
                str(archive["product"].item()),
                str(archive["frame_id"].item()),
            )
            expected = (self.run_id, self.product, str(row["frame_id"]))
            if identity != expected:
                raise ValueError(f"Field frame identity mismatch: {path}")
            if not math.isclose(
                float(archive["time_hours"].item()),
                float(row["time_hours"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"Field frame time mismatch: {path}")
            values = np.asarray(archive["values"])
            display = (
                np.asarray(archive["display_values"])
                if "display_values" in archive.files
                else None
            )
        if values.shape != self.shape_rc or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid field array: {path}")
        if display is not None and (
            display.shape != values.shape or not np.all(np.isfinite(display))
        ):
            raise ValueError(f"Invalid display field: {path}")
        return values, display


def load_field_index(
    path_value: str | Path,
    expected_sha256: str,
    *,
    run_id: str,
    product: str,
    expected_times_hours: Sequence[float] | None = None,
) -> IndexedFieldProduct:
    path, payload = _read_index(
        path_value,
        expected_sha256,
        label=f"{run_id}/{product} field index",
    )
    if payload.get("format") != FIELD_INDEX_FORMAT:
        raise ValueError(f"Unsupported field index format: {path}")
    if (str(payload.get("run_id") or ""), str(payload.get("product") or "")) != (
        run_id,
        product,
    ):
        raise ValueError(f"Field index identity mismatch: {path}")
    rows = [
        dict(row) for row in payload.get("frames") or [] if isinstance(row, Mapping)
    ]
    times = _finite_times([row.get("time_hours") for row in rows])
    if expected_times_hours is not None and times != [
        float(value) for value in expected_times_hours
    ]:
        raise ValueError(f"Field index time grid mismatch: {path}")
    if not rows:
        raise ValueError(f"Field index has no frames: {path}")
    shapes = {tuple(row.get("shape_rc") or []) for row in rows}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(f"Field index frames do not share one 2D shape: {path}")
    return IndexedFieldProduct(
        index_path=path,
        run_id=run_id,
        product=product,
        quantity=str(payload.get("quantity") or ""),
        unit=str(payload.get("unit") or ""),
        grid=deepcopy(dict(payload.get("grid") or {})),
        frames=tuple(rows),
    )


@dataclass(frozen=True)
class PreviewProduct:
    index_path: Path
    run_id: str
    frames: tuple[dict[str, Any], ...]

    @property
    def times_hours(self) -> tuple[float, ...]:
        return tuple(float(row["time_hours"]) for row in self.frames)

    def load_frame(self, index: int) -> np.ndarray:
        row = self.frames[index]
        path = verify_file_reference(
            str(row["path"]),
            str(row["sha256"]),
            label=f"{self.run_id}/preview/{row.get('frame_id', index)}",
            relative_to=self.index_path.parent,
        )
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode preview image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_preview_index(
    path_value: str | Path,
    expected_sha256: str,
    *,
    run_id: str,
    expected_times_hours: Sequence[float] | None = None,
) -> PreviewProduct:
    path, payload = _read_index(
        path_value,
        expected_sha256,
        label=f"{run_id}/preview image index",
    )
    if payload.get("format") != PREVIEW_INDEX_FORMAT:
        raise ValueError(f"Unsupported preview index format: {path}")
    if str(payload.get("run_id") or "") != run_id:
        raise ValueError(f"Preview index identity mismatch: {path}")
    rows = [
        dict(row) for row in payload.get("frames") or [] if isinstance(row, Mapping)
    ]
    times = _finite_times([row.get("time_hours") for row in rows])
    if expected_times_hours is not None and times != [
        float(value) for value in expected_times_hours
    ]:
        raise ValueError(f"Preview index time grid mismatch: {path}")
    if not rows:
        raise ValueError(f"Preview index has no frames: {path}")
    return PreviewProduct(path, run_id, tuple(rows))
