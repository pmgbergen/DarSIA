"""UI-only status of a completed export for the current geometry revision.

The caller supplies geometry_package_root(selected_series_root). No project is constructed and no configuration or artifact is written.
A result proves step 7 exported this current revision; it does not prove the user
has imported the emitted TOML sections into a run configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT
from darsia.fluidflower.series_geometry import NATIVE_SETUP_EXPORT_FORMAT

_NATIVE_SETUP_EXPORT_FILES = (
    "colored_labels.png",
    "native_labels.npz",
    "facies_properties.csv",
    "native_setup.toml",
)


def completed_current_native_setup_export(
    package_root: Path,
) -> dict[str, Any] | None:
    """Return the generator's unchanged export receipt, or None if unproven.

    Read exactly two small metadata files, then stat the generator's four
    required artifacts. Do not scan old exports, choose by mtime, load images,
    hash artifacts, create a project, or write any file. An unreadable, stale,
    malformed or incomplete export leaves the automatic alternative available.
    """
    package = Path(package_root)
    try:
        current = json.loads(
            (package / "overrides" / "geometry_facies" / "current.json").read_bytes()
        )
        if not isinstance(current, dict):
            return None
        if (
            current.get("schema") != 1
            or current.get("format") != GEOMETRY_REVISION_FORMAT
        ):
            return None
        revision = current.get("revision_id")
        # GeometryRevisionStore.save_revision writes uuid.uuid4().hex. Besides
        # checking its actual format, this prevents metadata from escaping the
        # selected series' export directory.
        if (
            not isinstance(revision, str)
            or len(revision) != 32
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            return None
        destination = package / "exports" / "native_setup" / revision
        receipt = json.loads((destination / "export.json").read_bytes())
        if not isinstance(receipt, dict):
            return None
        if (
            receipt.get("schema") != 1
            or receipt.get("format") != NATIVE_SETUP_EXPORT_FORMAT
            or receipt.get("revision_id") != revision
        ):
            return None
        files = receipt.get("files")
        if not isinstance(files, dict):
            return None
        for name in _NATIVE_SETUP_EXPORT_FILES:
            descriptor = files.get(name)
            if not isinstance(descriptor, dict):
                return None
            digest = descriptor.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not (destination / name).is_file()
            ):
                return None
        return receipt
    except (OSError, ValueError, TypeError):
        return None
