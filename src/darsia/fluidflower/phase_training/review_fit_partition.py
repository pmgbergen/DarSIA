"""Exact review-ID fit/evaluation partition for Track1 FF_AC validation.

The original review record stays in inventory.  Only an explicit ``fit`` row
may open an optical training source or own sampled pixels.  This is separate
from the source-photo mask, which determines supported pixels inside a fitted
region without changing its saved annotation.
"""

from __future__ import annotations
from ..phase_saved_review_metadata import configured_parameters

def _parameters():
    return configured_parameters("training_review_partition")

def __getattr__(name):
    keys={'CONTRACT': 'contract', 'CHEMISTRY_EVAL_RUNS': 'evaluation_runs'}
    if name not in keys: raise AttributeError(name)
    value=_parameters()[keys[name]]
    return frozenset(value) if name=="CHEMISTRY_EVAL_RUNS" else value

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from darsia.fluidflower.project import sha256_file




def verify_partition_rows(binding: Mapping[str, Any],
                          regions: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Validate all review IDs/roles before any image source is opened."""
    path = Path(binding.get("path") or "")
    expected_sha = binding.get("sha256")
    if (binding.get("contract") != _parameters()["contract"]
            or not isinstance(expected_sha, str) or len(expected_sha) != 64
            or not path.is_file() or sha256_file(path) != expected_sha):
        raise ValueError("Track1 review fit partition is absent or changed")
    partition = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = partition.get("rows")
    if partition.get("contract") != _parameters()["contract"] or not isinstance(rows, list) or len(rows) != _parameters()["total_review_ids"]:
        raise ValueError("Track1 fit partition omits frozen review IDs")
    by_id = {}
    for row in rows:
        identifier = row.get("region_id")
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("Track1 fit partition has duplicate or missing review ID")
        if row.get("role") not in {"fit", "evaluation-only"}:
            raise ValueError("Track1 review has no explicit fit/evaluation role")
        if row["role"] == "evaluation-only" and not row.get("reason"):
            raise ValueError("Track1 evaluation-only review has no reason")
        if (row["role"] == "fit" and
                str(row.get("run") or "").casefold() in frozenset(_parameters()["evaluation_runs"])):
            raise ValueError("Chemistry-shifted review entered Track1 optical fit")
        by_id[identifier] = row
    def identifier(region: Any) -> str:
        return str(region.get("region_id") if isinstance(region, Mapping)
                   else region.region_id)

    def run_name(region: Any) -> str:
        return str(region.get("run") if isinstance(region, Mapping)
                   else region.run).casefold()

    scope = binding.get("region_scope", "all52")
    if scope == "current43":
        materialized = binding.get("materialized_source_manifest_sha256s")
        if (not isinstance(materialized, list) or len(materialized) != _parameters()["current_source_count"]
                or len(set(materialized)) != _parameters()["current_source_count"]
                or any(not isinstance(value, str) or len(value) != 64
                       for value in materialized)):
            raise ValueError("Track1 current43 scope lacks 43 exact source revisions")
        source_set = set(materialized)
        if any(row["role"] == "fit" and
               row.get("source_manifest_sha256") not in source_set
               for row in by_id.values()):
            raise ValueError("Historical review entered current43 optical fit")
        expected_by_id = {key: row for key, row in by_id.items()
                          if row.get("source_manifest_sha256") in source_set}
    elif scope == "all52":
        expected_by_id = by_id
        if binding.get("materialized_source_manifest_sha256s") is not None:
            listed = binding["materialized_source_manifest_sha256s"]
            if (not isinstance(listed, list) or len(listed) != _parameters()["complete_source_count"]
                    or len(set(listed)) != _parameters()["complete_source_count"]
                    or set(partition.get("source_manifest_sha256s") or []) != set(listed)):
                raise ValueError("Mixed52 partition differs from materialized source cohort")
        strict_native_support = len(partition.get("source_manifest_sha256s") or []) == _parameters()["complete_source_count"]
        for row in by_id.values():
            if not strict_native_support:
                continue
            if row["role"] != "fit":
                continue
            raw = row.get("raw_native_pixels")
            photo = row.get("photo_safe_pixels")
            joint = row.get("photo_and_feature_supported_pixels")
            mask_sha = row.get("feature_support_mask_sha256")
            if (not all(isinstance(value, int) for value in (raw, photo, joint))
                    or raw <= 0 or photo < _parameters()["minimum_native_support_fraction"] * raw or joint < _parameters()["minimum_native_support_fraction"] * photo
                    or not isinstance(mask_sha, str) or len(mask_sha) != 64):
                raise ValueError("Track1 fit review lacks measured native source support")
    else:
        raise ValueError("Unknown Track1 review source scope")
    regions_by_id = {identifier(region): region for region in regions}
    if (len(regions_by_id) != len(regions)
            or regions_by_id.keys() != expected_by_id.keys()):
        raise ValueError("Track1 partition differs from exact 4048 scoped source review IDs")
    for identifier, row in expected_by_id.items():
        region = regions_by_id[identifier]
        if str(row.get("run") or "").casefold() != run_name(region):
            raise ValueError("Track1 review fit role belongs to another run")
    fit_ids = {identifier for identifier, row in by_id.items() if row["role"] == "fit"}
    if (not fit_ids or partition.get("fit_review_ids") != len(fit_ids)
            or partition.get("evaluation_only_review_ids") != _parameters()["total_review_ids"] - len(fit_ids)):
        raise ValueError("Track1 fit/evaluation counts disagree with source partition")
    return expected_by_id


def verify_fit_partition(
    binding: Mapping[str, Any], regions: Sequence[Any],
    origin_sources: Mapping[str, Any],
) -> set[str]:
    """Return fit IDs after binding the exact materialized source subset."""
    by_id = verify_partition_rows(binding, regions)
    if origin_sources.keys() != by_id.keys():
        raise ValueError("Track1 partition differs from exact source origins")
    source_shas: dict[Path, str] = {}
    for identifier, row in by_id.items():
        source = origin_sources[identifier]
        manifest_path = Path(source.asset_manifest_path)
        digest = source_shas.get(manifest_path)
        if digest is None:
            native = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if (native.get("v5_track1_photo_geometry") or {}).get("contract") != (
                    "ffac-v5-track1-photo-geometry-cohort-v1"):
                raise ValueError("Track1 review source lacks photo-only geometry")
            digest = (native.get("v5_photometry") or {}).get("source_manifest_sha256")
            source_shas[manifest_path] = digest
        if row.get("source_manifest_sha256") != digest:
            raise ValueError("Track1 review fit role belongs to another source revision")
    fit_ids = {identifier for identifier, row in by_id.items() if row["role"] == "fit"}
    return fit_ids


__all__ = ["CONTRACT", "CHEMISTRY_EVAL_RUNS", "verify_partition_rows",
           "verify_fit_partition"]
