"""Reversible Studio browser organization for immutable concentration results.

The archive changes a small series-owned NAS index only.  Scientific results,
delivery receipts and calibration provenance are never moved or rewritten.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


VERSION = "studio-concentration-browser-archive-v1"
NAME = "browser_archive.json"


def archive_path(results_root):
    return Path(results_root).resolve() / "catalog" / NAME


def family_key(result_path, results_root):
    """Group every run/scope of one fitted method branch in a single choice."""
    root = Path(results_root).resolve()
    path = Path(result_path).resolve()
    try:
        parts = path.relative_to(root).parts
    except ValueError as error:
        # A mapped NAS drive and its UNC path can name the same directory.
        # Match the existing ancestor by file identity, then retain a single
        # series-relative key regardless of how the result was opened.
        parts = None
        for ancestor in path.parents:
            try:
                if ancestor.samefile(root):
                    parts = path.relative_to(ancestor).parts
                    break
            except OSError:
                continue
        if parts is None:
            raise ValueError("Calibration does not belong to this series") from error
    if not parts or parts[-1].lower() != "result.json":
        raise ValueError("Select a saved concentration result")
    # Normal worker paths: jobs/<job>/results/[variant]/<scope>/<run>/result.json.
    # A comparison job's shared and individual results are one model family;
    # independent photometry or gas variants under it remain separate.
    try:
        branch = parts.index("results", 2)
    except ValueError:
        return Path(*parts[:-1]).as_posix().casefold()
    prefix = parts[:branch + 1]
    rest = parts[branch + 1:-1]
    if rest and rest[0].casefold() not in {"shared", "individual", "series"}:
        prefix = (*prefix, rest[0])
    return Path(*prefix).as_posix().casefold()


def load_archive(results_root, dataset):
    path = archive_path(results_root)
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("format") != VERSION or value.get("dataset") != str(dataset).casefold():
        raise ValueError("Saved calibration archive belongs to another series or format")
    entries = value.get("families")
    if not isinstance(entries, dict):
        raise ValueError("Malformed saved calibration archive")
    return entries


def save_archive(results_root, dataset, families):
    path = archive_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(format=VERSION, dataset=str(dataset).casefold(), families=families,
                 updated_utc=datetime.now(timezone.utc).isoformat())
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def set_archived(results_root, dataset, result_path, archived):
    """Move a family into or out of the historical browser, idempotently."""
    key = family_key(result_path, results_root)
    families = load_archive(results_root, dataset)
    if archived:
        families[key] = dict(archived_utc=datetime.now(timezone.utc).isoformat(),
                             selected_result=str(Path(result_path).resolve()))
    else:
        families.pop(key, None)
    save_archive(results_root, dataset, families)
    return key, families
