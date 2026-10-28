"""Remember the Phase Model Studio reviewer in the selected project folder."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from darsia.fluidflower.series_paths import darsia_source_checkout


_FORMAT = "darsia-phase-model-studio-reviewer-v1"
_RELATIVE_PATH = Path("config") / "phase_model_studio_reviewer.json"


def reviewer_identifier(name: str) -> str:
    """The reviewer_id saved with a reviewer's evidence: a slug of the name."""
    value = re.sub(r"[^a-z0-9]+", "-", str(name).strip().casefold()).strip("-")
    return value or "anonymous"


def _preference_path(project_root: Path | str) -> Path:
    if not str(project_root).strip():
        raise ValueError("Select a project folder before saving a reviewer name")
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    # A project folder may be its own Git repository (configuration and small
    # inputs); only the DarSIA source checkout itself is refused.
    checkout = darsia_source_checkout()
    if checkout is not None and root.is_relative_to(checkout):
        raise ValueError(
            "Reviewer preferences belong in a project folder outside the "
            "DarSIA source checkout"
        )
    return root / _RELATIVE_PATH


def load_reviewer_name(project_root: Path | str) -> str:
    """Return the saved reviewer for one project, or an empty string if unset."""
    path = _preference_path(project_root)
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ValueError(f"Unrecognized reviewer preference file: {path}")
    name = payload.get("reviewer_name")
    if not isinstance(name, str):
        raise ValueError(f"Invalid reviewer name in {path}")
    return name


def save_reviewer_name(project_root: Path | str, name: str) -> Path:
    """Atomically store an editable reviewer name beside project configuration."""
    path = _preference_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {"format": _FORMAT, "reviewer_name": str(name).strip()}
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
