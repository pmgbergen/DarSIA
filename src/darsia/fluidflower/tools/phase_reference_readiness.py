"""Bind deferred Studio references or explicitly prepare their exact maps.

Only the readiness receipt and disposable reference caches are written. Native
evidence, model parameters and the final validation binding remain unchanged.
"""
from __future__ import annotations

from pathlib import Path
import time

from darsia.fluidflower import phase_saved_review_metadata as maps
from darsia.fluidflower.tools.phase_injection_references import injection_end_references
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools._paths import canonical_metadata_paths
from darsia.fluidflower.series_paths import series_key, series_manifest

RECEIPT = "phase_reference_readiness.json"
FORMAT = "phase-reference-readiness-v1"


def _same_bound_asset(recorded, current):
    recorded = canonical_metadata_paths(recorded)
    if recorded == current:
        return True
    # The project share can be opened through its server name or IP address.
    # Keep the frozen receipt intact and accept that spelling difference only
    # when the digest agrees and the two paths identify the same filesystem file.
    if (not isinstance(recorded, dict) or not isinstance(current, dict)
            or set(recorded) != {"path", "sha256"}
            or set(current) != {"path", "sha256"}
            or recorded["sha256"] != current["sha256"]):
        return False
    try:
        return Path(recorded["path"]).samefile(current["path"])
    except (OSError, TypeError, ValueError):
        return False


def require_reference_readiness(validation, *, allow_pending=False):
    """Accept deferred references only where ordinary timeline review permits it."""
    validation = resolve_review_path(validation)
    path = validation / RECEIPT
    if not path.is_file():
        return
    receipt = maps._read(path)
    allowed = {"ready", "preparing", "deferred"} if allow_pending else {"ready"}
    if receipt.get("format") != FORMAT or receipt.get("status") not in allowed:
        raise ValueError("Reference phase maps are not ready: " + str(receipt.get("error") or receipt.get("status")))
    if any(key not in receipt for key in ('binding', 'campaign')):
        raise ValueError('Validation package is not ready: no bound reference receipt')
    for key, filename in (("binding", "validation_binding.json"), ("campaign", "postflight_manifest.json")):
        if not _same_bound_asset(receipt.get(key), maps._asset(validation / filename)):
            raise ValueError("Reference readiness identifies another validation package")


def mark_references_pending(validation):
    maps._write(Path(validation) / RECEIPT, {"format": FORMAT, "status": "preparing"})


def _require_reference_series(jobs):
    """Preserve existing series; other series need their own manifest-listed runs."""
    manifests = {}
    for job in jobs:
        dataset = job["dataset"]
        if dataset in {"ffac", "ffum", "ffbm"}:
            continue
        if not isinstance(dataset, str) or series_key(dataset) != dataset:
            raise ValueError("Reference series must use its canonical registered ID")
        if dataset not in manifests:
            manifests[dataset] = series_manifest(dataset)
        if job["run"] not in manifests[dataset]["runs"]["ids"]:
            raise ValueError("Reference run is not listed in its own series manifest")


def defer_validation_references(validation):
    """Publish ordinary maps without computing optional reference overlays.

    The reference button still resolves the exact acquisition against this
    frozen binding, with chronological replay when an endpoint is off timeline.
    No image, inference, or replay is needed merely to make a run reviewable.
    """
    validation = Path(validation).resolve()
    binding_path = validation / "validation_binding.json"
    campaign_path = validation / "postflight_manifest.json"
    binding, campaign = maps._read(binding_path), maps._read(campaign_path)
    mode = binding.get("output_mode", "review")
    if mode not in ("review", "production") or campaign.get("output_mode", "review") != mode:
        raise ValueError("Reference preparation requires matching phase output modes")
    jobs, selected = campaign["jobs"], binding["selected_candidates"]
    if (not jobs or len(jobs) != len(selected)
            or {job["run"]: job["candidate_id"] for job in jobs} != selected):
        raise ValueError("Reference preparation requires the complete selected validation campaign")
    _require_reference_series(jobs)
    existing_path = validation / RECEIPT
    if existing_path.is_file():
        existing = maps._read(existing_path)
        if existing.get("format") != FORMAT:
            raise ValueError("Unknown reference readiness receipt")
        if existing.get("status") in {"ready", "deferred"}:
            require_reference_readiness(validation, allow_pending=True)
            return existing
        # A previous failed preparation is evidence of an error, not permission
        # to hide it by replacing the receipt with a deferred one.
        if existing.get("status") == "failed":
            require_reference_readiness(validation)
        if existing.get("status") != "preparing":
            raise ValueError("Unknown reference readiness status")
        if "binding" in existing or "campaign" in existing:
            require_reference_readiness(validation, allow_pending=True)
    receipt = {"format": FORMAT, "status": "deferred", "output_mode": mode,
               "binding": maps._asset(binding_path), "campaign": maps._asset(campaign_path),
               "materialization_policy": "on-demand-exact-reference",
               "reference_coverage": "not-requested", "references": [],
               "unavailable_references": []}
    maps._write(existing_path, receipt)
    return receipt
