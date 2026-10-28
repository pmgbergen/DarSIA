"""Names, limits and folders of a published dataset record.

The record is flat so that it fits a repository record of fewer than 100
files: the manifest, a README, the checksum list, ``series.zip``,
``expected.zip`` and one ``raw_<run>.zip`` per run. Unpacked, the same names
sit in the dataset root next to the series folder content.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
import re

FORMAT = "darsia-published-dataset-v1"
MANIFEST = "dataset_manifest.json"
README = "README.md"
CHECKSUMS = "CHECKSUMS.sha256"
SERIES_ARCHIVE = "series.zip"
EXPECTED_ARCHIVE = "expected.zip"
ALIASES = "dataset_aliases.toml"
PLAN_FOLDER = "plan"
PROVENANCE_FOLDER = "provenance"
# Pipelines whose prepared tasks the launcher replays with the production code.
REPLAY_PIPELINES = frozenset({"native-phase-n1", "lattice-phase-n1"})
EXPECTED_FOLDER = "expected"
RAW_FOLDER = "raw"

# R: the reproduction record (raw images as acquired). S: simulator products.
RECORD_TYPES = ("R", "S")
# Nested raw-image tiers and the whole-record budget of each, in bytes (10^9 per GB).
TIER_BUDGETS = {"S50": 50 * 10**9, "S100": 100 * 10**9, "S200": 200 * 10**9}
TIER_ORDER = ("S50", "S100", "S200")
# A record must hold fewer than this many files.
FILE_LIMIT = 100


def require_record_type(value: str) -> str:
    """Return ``value`` when it names a record type (R or S), else raise ValueError."""
    if value not in RECORD_TYPES:
        raise ValueError(f"Record type must be one of {', '.join(RECORD_TYPES)}: {value!r}")
    return value


def require_tier(value: str) -> str:
    """Return ``value`` when it names a raw-image tier, else raise ValueError."""
    if value not in TIER_BUDGETS:
        raise ValueError(f"Tier must be one of {', '.join(TIER_ORDER)}: {value!r}")
    return value


def task_path(run: str) -> str:
    """Where a run's prepared task (its input catalogue) lies in the dataset."""
    from ..pipeline.job_queue import name

    name(run)
    return f"{PLAN_FOLDER}/tasks/{run}.json"


def raw_archive_name(run: str) -> str:
    """The record file name of one run's stored raw-image archive."""
    from ..pipeline.job_queue import name

    name(run)
    return f"raw_{run}.zip"


def run_of_raw_archive(file_name: str) -> str | None:
    """The run of a ``raw_<run>.zip`` record file name, or None for other files."""
    match = re.fullmatch(r"raw_([A-Za-z0-9][A-Za-z0-9_.-]{0,127})\.zip", file_name)
    return match.group(1) if match else None


def record_relative(value: str) -> str:
    """Validate a portable, slash-separated path inside a record or dataset folder.

    Absolute paths, drive letters, backslashes and parent traversal are refused,
    so a name read from a manifest can never point outside the dataset root.
    """
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"A portable relative path is required: {value!r}")
    if PureWindowsPath(value).drive or PurePosixPath(value).is_absolute():
        raise ValueError(f"A relative path is required: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"A normalized relative path is required: {value!r}")
    return value


def inside(root: str | Path, relative: str) -> Path:
    """Join a validated relative path to ``root`` and refuse a link that escapes it."""
    base = Path(root)
    path = base.joinpath(*record_relative(relative).split("/"))
    resolved_root = base.resolve()
    if path.exists() and not path.resolve().is_relative_to(resolved_root):
        raise ValueError(f"Path escapes the dataset folder: {relative}")
    return path
