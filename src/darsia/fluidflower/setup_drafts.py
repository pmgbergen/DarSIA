"""Autosaved GUI drafts for incomplete FluidFlower project setup."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any
import uuid

from darsia.fluidflower.project import default_project_root, utc_now


SETUP_DRAFT_FORMAT = "darsia-fluidflower-setup-draft-v1"


def default_setup_draft_root() -> Path:
    return default_project_root().parent / "setup_drafts"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


class SetupDraftStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or default_setup_draft_root()).expanduser().resolve()

    def save(
        self,
        values: Mapping[str, Any],
        *,
        draft_id: str = "",
    ) -> dict[str, Any]:
        identifier = str(draft_id).strip() or uuid.uuid4().hex
        payload = {
            "schema": 1,
            "format": SETUP_DRAFT_FORMAT,
            "draft_id": identifier,
            "updated_utc": utc_now(),
            "values": deepcopy(dict(values)),
        }
        _atomic_json(self.root / f"{identifier}.json", payload)
        return deepcopy(payload)

    def load(self, draft_id: str) -> dict[str, Any]:
        path = self.root / f"{str(draft_id).strip()}.json"
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or payload.get("format") != SETUP_DRAFT_FORMAT:
            raise ValueError(f"Invalid setup draft: {path}")
        return payload

    def drafts(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in self.root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError, TypeError):
                continue
            if (
                isinstance(payload, dict)
                and payload.get("format") == SETUP_DRAFT_FORMAT
            ):
                rows.append(payload)
        return sorted(
            rows, key=lambda row: str(row.get("updated_utc") or ""), reverse=True
        )

    def latest(self) -> dict[str, Any]:
        rows = self.drafts()
        return deepcopy(rows[0]) if rows else {}

    def delete(self, draft_id: str) -> None:
        try:
            (self.root / f"{str(draft_id).strip()}.json").unlink()
        except FileNotFoundError:
            pass

    def discard(self, draft_id: str) -> dict[str, Any]:
        """Remove a draft from the active list while retaining a recovery copy."""

        identifier = str(draft_id).strip()
        payload = self.load(identifier)
        payload["discarded_utc"] = utc_now()
        archive_path = self.root / "discarded" / f"{identifier}.json"
        _atomic_json(archive_path, payload)
        (self.root / f"{identifier}.json").unlink()
        return deepcopy(payload)


__all__ = [
    "SETUP_DRAFT_FORMAT",
    "SetupDraftStore",
    "default_setup_draft_root",
]
