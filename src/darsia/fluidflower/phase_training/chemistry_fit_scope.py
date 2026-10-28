"""Frozen light-only FF_AC fit scope; excluded reviews remain in the inventory."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


CONTRACT = "ffac-v5-light-only-chemistry-exclusion-v2"
EXCLUDED_RUNS = ("ac20", "ac21", "ac22", "ac24", "ac27", "ac51")
EXPECTED_EXCLUDED_COUNTS = {
    "ac20": 0, "ac21": 404, "ac22": 142,
    "ac24": 0, "ac27": 0, "ac51": 0,
}


def _row_identity(row: Any) -> tuple[str, str]:
    if isinstance(row, Mapping):
        return str(row.get("region_id") or row.get("annotation_id") or ""), str(row["run"]).casefold()
    return str(row.region_id), str(row.run).casefold()


def scope_for_regions(regions: Iterable[Any]) -> dict[str, Any]:
    """Describe the exact excluded IDs without removing any source annotation."""
    seen: set[str] = set()
    excluded_ids: list[str] = []
    counts: Counter[str] = Counter()
    for region in regions:
        identifier, run = _row_identity(region)
        if not identifier or identifier in seen:
            raise ValueError("V5 chemistry fit scope has missing or duplicate review IDs")
        seen.add(identifier)
        if run in EXCLUDED_RUNS:
            counts[run] += 1
            excluded_ids.append(identifier)
    by_run = {run: counts[run] for run in EXCLUDED_RUNS}
    if by_run != EXPECTED_EXCLUDED_COUNTS:
        raise ValueError(f"V5 excluded review cohort changed: {by_run}")
    payload = json.dumps(sorted(excluded_ids), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return {
        "contract": CONTRACT,
        "excluded_runs": list(EXCLUDED_RUNS),
        "excluded_review_ids": len(excluded_ids),
        "excluded_review_ids_sha256": hashlib.sha256(payload).hexdigest(),
        "excluded_reviews_by_run": by_run,
        "chemistry_alignment_applied": False,
        "scientific_chemistry_acceptance": False,
        "excluded_reviews_role": "evaluation-only-out-of-distribution",
    }


def verify_scope(scope: Mapping[str, Any], regions: Iterable[Any]) -> set[str]:
    expected = scope_for_regions(regions)
    if dict(scope) != expected:
        raise ValueError("V5 chemistry fit scope or preserved review IDs changed")
    return set(EXCLUDED_RUNS)
