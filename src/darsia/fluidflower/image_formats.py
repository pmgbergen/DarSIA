"""Shared image-file selection policy for FluidFlower source folders."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


PROCESSED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
CAMERA_RAW_SUFFIXES = frozenset(
    {
        ".3fr",
        ".arw",
        ".cr2",
        ".cr3",
        ".dcr",
        ".dng",
        ".erf",
        ".kdc",
        ".mef",
        ".mos",
        ".mrw",
        ".nef",
        ".nrw",
        ".orf",
        ".pef",
        ".raf",
        ".raw",
        ".rw2",
        ".sr2",
        ".srf",
        ".srw",
        ".x3f",
    }
)
ALL_IMAGE_SUFFIXES = PROCESSED_IMAGE_SUFFIXES | CAMERA_RAW_SUFFIXES

IMAGE_FILE_POLICY_PROCESSED = "prefer_processed"
IMAGE_FILE_POLICY_RAW = "prefer_raw"
IMAGE_FILE_POLICY_ALL = "all"
# Strict policies: one image family, never the other as a fallback.
IMAGE_FILE_POLICY_RAW_ONLY = "raw_only"
IMAGE_FILE_POLICY_PROCESSED_ONLY = "processed_only"
DEFAULT_IMAGE_FILE_POLICY = IMAGE_FILE_POLICY_PROCESSED
IMAGE_FILE_POLICIES = {
    IMAGE_FILE_POLICY_PROCESSED,
    IMAGE_FILE_POLICY_RAW,
    IMAGE_FILE_POLICY_ALL,
    IMAGE_FILE_POLICY_RAW_ONLY,
    IMAGE_FILE_POLICY_PROCESSED_ONLY,
}
IMAGE_FILE_POLICY_LABELS = {
    IMAGE_FILE_POLICY_PROCESSED: "Prefer JPEG/TIFF/PNG (recommended)",
    IMAGE_FILE_POLICY_RAW: "Prefer camera RAW",
    IMAGE_FILE_POLICY_ALL: "Use both processed and RAW files",
    IMAGE_FILE_POLICY_RAW_ONLY: "Only camera RAW; never JPEG/TIFF/PNG",
    IMAGE_FILE_POLICY_PROCESSED_ONLY: "Only JPEG/TIFF/PNG; never camera RAW",
}
# The family each strict policy requires.
STRICT_IMAGE_FAMILIES = {
    IMAGE_FILE_POLICY_RAW_ONLY: "raw",
    IMAGE_FILE_POLICY_PROCESSED_ONLY: "processed",
}
_FAMILY_NAMES = {"raw": "camera RAW", "processed": "JPEG/TIFF/PNG"}


def normalize_image_file_policy(value: Any) -> str:
    policy = str(value or DEFAULT_IMAGE_FILE_POLICY).strip().casefold()
    if policy not in IMAGE_FILE_POLICIES:
        raise ValueError(f"Unsupported image-file policy: {value!r}")
    return policy


def image_family(path: str | Path) -> str:
    suffix = Path(path).suffix.casefold()
    if suffix in PROCESSED_IMAGE_SUFFIXES:
        return "processed"
    if suffix in CAMERA_RAW_SUFFIXES:
        return "raw"
    return "other"


def photo_dialog_filter(reference: str | Path | None = None) -> str:
    """File-dialog filters for photographs, with the reference photo's kind first.

    Camera RAW comes first when ``reference`` is a RAW file, JPEG/TIFF/PNG
    first otherwise; then the other kind, then all files. A series works with
    one kind of file, so the first filter is the one that fits it.
    """

    raw = "Camera RAW (" + " ".join(f"*{s}" for s in sorted(CAMERA_RAW_SUFFIXES)) + ")"
    processed = (
        "Photographs (" + " ".join(f"*{s}" for s in sorted(PROCESSED_IMAGE_SUFFIXES)) + ")"
    )
    first, second = (
        (raw, processed)
        if reference is not None and image_family(reference) == "raw"
        else (processed, raw)
    )
    return f"{first};;{second};;All files (*)"


def select_preferred_images(
    paths: Iterable[Path],
    policy: str = DEFAULT_IMAGE_FILE_POLICY,
) -> list[Path]:
    """Select one image family per folder.

    The preferring policies fall back to the other family when the preferred
    one is unavailable; the strict policies (``raw_only``, ``processed_only``)
    never do, see ``missing_image_family``.
    """

    normalized = normalize_image_file_policy(policy)
    candidates = [
        Path(path)
        for path in paths
        if Path(path).suffix.casefold() in ALL_IMAGE_SUFFIXES
    ]
    processed = [path for path in candidates if image_family(path) == "processed"]
    raw = [path for path in candidates if image_family(path) == "raw"]
    if normalized == IMAGE_FILE_POLICY_ALL:
        return candidates
    if normalized == IMAGE_FILE_POLICY_RAW_ONLY:
        return raw
    if normalized == IMAGE_FILE_POLICY_PROCESSED_ONLY:
        return processed
    if normalized == IMAGE_FILE_POLICY_RAW:
        return raw or processed
    return processed or raw


def missing_image_family(paths: Iterable[Path], policy: str) -> str:
    """Why one folder's images do not fit a strict policy, or "" when they do.

    A strict policy never falls back to the other family. A folder that holds
    only images of the other family is therefore reported, so that callers
    stop or warn instead of thinning a series or switching its kind of file.
    """

    family = STRICT_IMAGE_FAMILIES.get(normalize_image_file_policy(policy))
    if family is None:
        return ""
    families = [
        image_family(path)
        for path in paths
        if Path(path).suffix.casefold() in ALL_IMAGE_SUFFIXES
    ]
    if not families or family in families:
        return ""
    other = "processed" if family == "raw" else "raw"
    return (
        f"{len(families)} {_FAMILY_NAMES[other]} image(s) but no {_FAMILY_NAMES[family]} "
        f"image; the series uses only {_FAMILY_NAMES[family]} files"
    )


def image_file_counts(paths: Iterable[Path]) -> dict[str, int]:
    counts = {"processed": 0, "raw": 0, "other": 0}
    for path in paths:
        counts[image_family(path)] += 1
    return counts


__all__ = [
    "ALL_IMAGE_SUFFIXES",
    "CAMERA_RAW_SUFFIXES",
    "DEFAULT_IMAGE_FILE_POLICY",
    "IMAGE_FILE_POLICIES",
    "IMAGE_FILE_POLICY_ALL",
    "IMAGE_FILE_POLICY_LABELS",
    "IMAGE_FILE_POLICY_PROCESSED",
    "IMAGE_FILE_POLICY_RAW",
    "PROCESSED_IMAGE_SUFFIXES",
    "image_family",
    "image_file_counts",
    "normalize_image_file_policy",
    "select_preferred_images",
]
