"""Read one explicitly selected, already loaded Studio timeline on Load only."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tomllib

from darsia.fluidflower.series_paths import series_key


def _same_path(left, right):
    if not left or not right:
        return False
    left, right = Path(left).resolve(), Path(right).resolve()
    if str(left).casefold() == str(right).casefold():
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _loaded_choice(window, metadata):
    """Inspect existing UI objects; never construct/load/select a Studio page."""
    workspace = getattr(window, "phase_model_workspace", None)
    page = getattr(workspace, "studio_page", None)
    studio = getattr(page, "studio", None)
    steps = getattr(page, "native_steps", None)
    existing = getattr(steps, "existing_basis", None)
    if studio is None or existing is None or not existing.isChecked():
        return None
    root = metadata.series_root
    series = str(getattr(workspace, "series_id", "") or "")
    if not series:
        return None
    if any(
        series_key(value or "missing") != series_key(series)
        for value in (
            getattr(page, "series_id", ""),
            getattr(steps, "series_id", ""),
            getattr(studio, "_active_model_series", ""),
        )
    ):
        return None
    roots = getattr(workspace, "series_roots", {})
    if not all(
        _same_path(value, root)
        for value in (
            window._method_series_root(),
            roots.get(series),
            getattr(page, "project_root", None),
            getattr(steps, "series_root", None),
        )
    ):
        return None
    selector = getattr(studio, "validation_model", None)
    selected_root = selector.currentData() if selector is not None else None
    if not selected_root:
        return None
    entry = studio._selected_validation_model()
    if (
        not isinstance(entry, dict)
        or entry.get("validation_binding_loading")
        or entry.get("errors")
        or entry.get("library_available") is False
        or not _same_path(entry.get("root"), selected_root)
    ):
        return None
    binding = entry.get("validation_binding")
    if not isinstance(binding, dict) or not binding.get("combined_root"):
        return None
    return dict(
        series=series_key(series),
        model_root=str(Path(selected_root).resolve()),
        binding=deepcopy(binding),
    )


def loaded_studio_stored_plan(
    window,
    metadata,
    *,
    verify_binding=None,
    analysis_root=None,
    resolve_path=None,
):
    """Return a SHA-bound native JSON descriptor, or no available loaded basis.

    Call only from the image-selection dialog's explicit Load action. Ordinary
    campaigns use the original saved-map verifier. Collections require its full
    payload audit, so remain an explicit Import gap in this metadata-only UI.
    """
    choice = _loaded_choice(window, metadata)
    if choice is None:
        return None
    if resolve_path is None:
        from darsia.fluidflower.review_storage import resolve_review_path

        resolve_path = resolve_review_path
    root = Path(metadata.series_root).resolve()
    series_path = root / "series.toml"
    series_raw = series_path.read_bytes()
    series = tomllib.loads(series_raw.decode("utf-8-sig"))
    if (
        series.get("schema_version") != 1
        or series_key(series["series_id"]) != choice["series"]
    ):
        raise ValueError(
            "The loaded Studio basis belongs to another registered series."
        )
    model_root = resolve_path(choice["model_root"])
    campaign = resolve_path(choice["binding"]["combined_root"])
    postflight_path = campaign / "postflight_manifest.json"
    binding_path = campaign / "validation_binding.json"
    postflight_raw, binding_raw = (
        postflight_path.read_bytes(),
        binding_path.read_bytes(),
    )
    postflight = json.loads(postflight_raw)
    if postflight.get("format") == "fluidflower-phase-review-collection-v1":
        raise ValueError(
            "This loaded basis is a review collection whose original verifier "
            "requires a payload audit. Import a source-bound native manifest explicitly."
        )
    if verify_binding is None:
        from darsia.fluidflower.phase_evidence_audit import verified_review_binding

        verify_binding = verified_review_binding
    binding = verify_binding(model_root, campaign, saved_comparison=True)
    if binding != choice["binding"]:
        raise ValueError(
            "The loaded Studio basis binding changed; reload it before Load."
        )
    run = str(metadata.run_id).casefold()
    selections = [
        (key, value)
        for key, value in binding["selected_candidates"].items()
        if str(key).casefold() == run
    ]
    if len(selections) != 1:
        raise ValueError(f"The loaded basis has no unique bound selection for {run}.")
    jobs = [
        job
        for job in postflight.get("jobs", [])
        if str(job.get("run", "")).casefold() == run
        and job.get("candidate_id") == selections[0][1]
    ]
    if len(jobs) != 1:
        raise ValueError(f"The loaded basis has no unique postflight job for {run}.")
    job = jobs[0]
    if (
        series_key(job.get("dataset", "missing")) != choice["series"]
        or not binding.get("pipeline_sha256")
        or job.get("params_sha256") != binding["pipeline_sha256"]
        or job.get("optical_model_sha256") != binding.get("model_sha256")
        or job.get("physics_runtime") != binding.get("physics_runtime")
    ):
        raise ValueError(f"The postflight job differs from the loaded basis: {run}.")
    digest = job.get("native_manifest_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"The bound job has no native manifest SHA-256: {run}.")
    review = job.get("analysis_source_review_root") or job.get("review_output_dir")
    if not review:
        raise ValueError(f"The bound job has no saved analysis review source: {run}.")
    if analysis_root is None:
        from darsia.fluidflower.phase_training.radiometry import analysis_review_root

        analysis_root = analysis_review_root
    native_root = resolve_path(analysis_root(resolve_path(review), run))
    path = native_root / "native_frames" / run / "manifest.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(
            f"The saved native manifest differs from its postflight SHA: {run}."
        )
    if (
        postflight_path.read_bytes() != postflight_raw
        or binding_path.read_bytes() != binding_raw
        or series_path.read_bytes() != series_raw
        or _loaded_choice(window, metadata) != choice
    ):
        raise ValueError(
            "The selected basis or its metadata changed during Load; retry explicitly."
        )
    # The existing dialog checks native dataset/run/source/time and retains its
    # actual frame count. No pixels or image payload hashes are read here.
    return dict(path=str(path), sha256=digest, size_bytes=len(raw))
