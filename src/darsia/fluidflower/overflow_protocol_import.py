"""Import saved scale readings into an ordinary overflow protocol table."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath


def load_overflow_sources(list_file, source_root=None, *, runs=None):
    """Resolve explicitly selected review files, retaining their recorded bytes.

    An optional source folder explicitly relocates each run/filename pair. No
    directory is searched and no camera or scale image is opened.
    """
    from .storage_paths import storage_path_remapper

    selected = Path(list_file)
    document = json.loads(selected.read_text(encoding="utf-8-sig"))
    references = document.get("scale_reviews")
    if not isinstance(references, dict):
        raise ValueError("The overflow source list needs a scale_reviews table.")
    wanted = None if runs is None else set(runs)
    remap = storage_path_remapper()
    result = {}
    for run, reference in references.items():
        if wanted is not None and run not in wanted:
            continue
        if not isinstance(run, str) or Path(run).name != run or run in {".", ".."}:
            raise ValueError("Each overflow source needs one run identifier.")
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise ValueError("Each overflow source needs a file path and its saved identity.")
        if source_root:
            name = PureWindowsPath(reference["path"]).name
            path = Path(source_root) / run / name
        else:
            path = Path(remap(reference["path"]))
            if not path.is_absolute():
                path = selected.parent / path
        raw = path.read_bytes()
        if (len(raw) != reference.get("size_bytes")
                or hashlib.sha256(raw).hexdigest() != reference.get("sha256")):
            raise ValueError(f"The selected overflow source differs from its saved readings: {run}")
        result[run] = path
    return result


def read_overflow_measurements(path, run):
    """Keep actual stored readings and review decisions, excluding OCR guesses.

    The original source document retains the OCR history. This measurement
    table records only its weight_g values with their original clock, status,
    image source and injection-time origin. No density or gas model is used.
    """
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8-sig"))
    metadata, records = document.get("metadata"), document.get("records")
    if (not isinstance(metadata, dict) or metadata.get("run") != run
            or not isinstance(records, list)):
        raise ValueError("The scale-review source must belong to the selected run.")
    columns = (
        "image_time", "time_from_injection_h", "weight_g", "status", "image",
        "image_id", "reading_pattern", "reviewer", "verified_at", "notes",
        "injection_start", "weight_data_valid", "weight_data_status", "source_review",
    )
    rows = []
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Each saved overflow reading must be a table.")
        if row.get("weight_g") is None:
            continue
        if (type(row["weight_g"]) not in (int, float)
                or not math.isfinite(row["weight_g"])):
            raise ValueError("A saved overflow weight must be finite grams.")
        values = dict(row, injection_start=metadata.get("injection_start"),
                      weight_data_valid=metadata.get("weight_data_valid"),
                      weight_data_status=metadata.get("weight_data_status"),
                      source_review=str(source))
        rows.append(tuple(None if values.get(name) is None else str(values[name])
                          for name in columns))
    return columns, tuple(rows)
