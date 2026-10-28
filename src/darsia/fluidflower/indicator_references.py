"""Versioned metadata for optical indicator-response reference images."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import re
from typing import Any


INDICATOR_REFERENCE_FORMAT = "darsia-fluidflower-indicator-reference-v1"
REFERENCE_KIND_PH_BUFFER = "ph_buffer"
REFERENCE_KIND_ALTERNATE_STATE = "alternate_state"
REFERENCE_KINDS = {REFERENCE_KIND_PH_BUFFER, REFERENCE_KIND_ALTERNATE_STATE}
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def infer_ph(value: str | Path) -> float | None:
    """Infer an explicitly labelled pH value from a path without guessing chemistry."""

    text = str(value).replace(",", ".")
    patterns = (
        r"(?i)(?:^|[^a-z0-9])ph[\s_=-]*([0-9]{1,2}(?:\.[0-9]+)?)",
        r"(?i)([0-9]{1,2}(?:\.[0-9]+)?)[\s_-]*ph(?:$|[^a-z0-9])",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is None:
            continue
        candidate = float(match.group(1))
        if 0.0 <= candidate <= 14.0:
            return candidate
    return None


def discover_indicator_reference_images(
    sources: Iterable[tuple[Path, str]],
    *,
    default_kind: str | None = None,
) -> list[dict[str, Any]]:
    """List reference images from source folders with stable scope metadata."""

    if default_kind is not None and default_kind not in REFERENCE_KINDS:
        raise ValueError(f"Unsupported default reference kind: {default_kind!r}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_root, raw_scope in sources:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(
            (
                item.resolve()
                for item in root.rglob("*")
                if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda item: item.as_posix().casefold(),
        ):
            identity = str(path).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            inferred_ph = infer_ph(path)
            kind = default_kind or (
                REFERENCE_KIND_PH_BUFFER
                if inferred_ph is not None
                else REFERENCE_KIND_ALTERNATE_STATE
            )
            ph = inferred_ph if kind == REFERENCE_KIND_PH_BUFFER else None
            rows.append(
                {
                    "kind": kind,
                    "ph": ph,
                    "label": (
                        ""
                        if kind == REFERENCE_KIND_PH_BUFFER
                        else path.parent.name
                    ),
                    "scope": str(raw_scope).strip().casefold() or "series",
                    "image_path": str(path),
                }
            )
    return rows


def merge_indicator_reference_entries(
    existing: Sequence[Mapping[str, Any]],
    discovered: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge newly discovered paths without replacing manually entered metadata."""

    merged = [deepcopy(dict(row)) for row in existing if isinstance(row, Mapping)]
    known = {
        str(Path(str(row.get("image_path") or "")).expanduser().resolve()).casefold()
        for row in merged
        if str(row.get("image_path") or "").strip()
    }
    for row in discovered:
        path = str(row.get("image_path") or "").strip()
        if not path:
            continue
        identity = str(Path(path).expanduser().resolve()).casefold()
        if identity not in known:
            merged.append(deepcopy(dict(row)))
            known.add(identity)
    return merged


def normalize_indicator_reference_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one UI or manifest row while preserving the source path."""

    kind = str(row.get("kind") or REFERENCE_KIND_PH_BUFFER).strip().casefold()
    if kind not in REFERENCE_KINDS:
        raise ValueError(f"Unsupported indicator-reference kind: {kind!r}")
    path_text = str(row.get("image_path") or "").strip()
    if not path_text:
        raise ValueError("Indicator-reference rows require an image path")
    ph_raw = row.get("ph")
    ph_text = "" if ph_raw is None else str(ph_raw).strip()
    ph = None if not ph_text else float(ph_text.replace(",", "."))
    if ph is not None and not 0.0 <= ph <= 14.0:
        raise ValueError("Indicator-reference pH must lie between 0 and 14")
    if kind == REFERENCE_KIND_PH_BUFFER and ph is None:
        raise ValueError("Every pH-buffer reference requires a measured pH")
    label = str(row.get("label") or "").strip()
    if kind == REFERENCE_KIND_ALTERNATE_STATE and not label:
        raise ValueError("Alternate-state references require a descriptive label")
    return {
        "kind": kind,
        "scientific_role": (
            "indicator_response_at_measured_ph"
            if kind == REFERENCE_KIND_PH_BUFFER
            else "labelled_alternate_optical_state"
        ),
        "ph": ph,
        "label": label,
        "scope": str(row.get("scope") or "series").strip().casefold() or "series",
        "image_path": str(Path(path_text).expanduser().resolve()),
    }


def build_indicator_reference_contract(
    entries: Sequence[Mapping[str, Any]],
    *,
    indicator_system: str,
) -> dict[str, Any]:
    """Build the chemistry-facing reference contract and readiness diagnostics."""

    normalized: list[dict[str, Any]] = []
    issues: list[str] = []
    for index, row in enumerate(entries, start=1):
        try:
            normalized.append(normalize_indicator_reference_entry(row))
        except (TypeError, ValueError) as exc:
            fallback = deepcopy(dict(row))
            fallback["validation_error"] = str(exc)
            normalized.append(fallback)
            issues.append(f"Reference {index}: {exc}")

    ph_values = sorted(
        {
            float(row["ph"])
            for row in normalized
            if row.get("kind") == REFERENCE_KIND_PH_BUFFER
            and row.get("ph") is not None
            and not row.get("validation_error")
        }
    )
    if not normalized:
        status = "not_provided"
    elif issues:
        status = "metadata_required"
    elif any(row.get("kind") == REFERENCE_KIND_PH_BUFFER for row in normalized):
        status = "ready" if len(ph_values) >= 2 else "insufficient_ph_span"
        if len(ph_values) < 2:
            issues.append(
                "A pH-response curve requires at least two distinct measured pH values."
            )
    else:
        status = "reference_states_only"

    return {
        "schema": 1,
        "format": INDICATOR_REFERENCE_FORMAT,
        "indicator_system": str(indicator_system).strip() or "unspecified",
        "scientific_role": "chemical_response_reference",
        "radiometric_normalization_allowed": False,
        "status": status,
        "entry_count": len(normalized),
        "known_ph_buffer_count": sum(
            row.get("kind") == REFERENCE_KIND_PH_BUFFER
            and not row.get("validation_error")
            for row in normalized
        ),
        "alternate_state_count": sum(
            row.get("kind") == REFERENCE_KIND_ALTERNATE_STATE
            and not row.get("validation_error")
            for row in normalized
        ),
        "distinct_ph_values": ph_values,
        "ph_span": (
            [ph_values[0], ph_values[-1]] if len(ph_values) >= 2 else None
        ),
        "issues": issues,
        "entries": normalized,
    }


__all__ = [
    "INDICATOR_REFERENCE_FORMAT",
    "REFERENCE_KIND_ALTERNATE_STATE",
    "REFERENCE_KIND_PH_BUFFER",
    "REFERENCE_KINDS",
    "build_indicator_reference_contract",
    "discover_indicator_reference_images",
    "infer_ph",
    "merge_indicator_reference_entries",
    "normalize_indicator_reference_entry",
]
