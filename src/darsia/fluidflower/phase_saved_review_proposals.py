"""Read saved model-selected review proposals without starting inference."""
from __future__ import annotations
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from .review_storage import resolve_review_path


def _read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Expected an object in saved review metadata")
    return payload


def ensure_review_proposals(campaign_root: Path, run_id: str, *, parameters: Mapping[str, Any],
                            progress: Callable[[str], None] | None = None) -> Path:
    """Return the bound saved quiz, or report it unavailable; never replay a model."""
    root = Path(campaign_root).expanduser().resolve()
    run = str(run_id).strip().casefold()
    manifest = _read_json(root / parameters["manifest_filename"])
    jobs = [job for job in manifest.get("jobs", []) if str(job.get("run", "")).casefold() == run]
    if len(jobs) != 1:
        raise ValueError(f"Expected one selected-model candidate for {run!r}")
    job = jobs[0]
    fingerprint = str(job.get("params_sha256") or manifest.get("pipeline_sha256") or "")
    if not fingerprint or any(character not in parameters["fingerprint_alphabet"] for character in fingerprint.casefold()):
        raise ValueError("The selected quiz has no valid phase-pipeline fingerprint")
    path = Path(str(job.get("phase_map_output_dir") or parameters["assets_directory"]))
    assets = resolve_review_path(path if path.is_absolute() else root / path)
    target = (assets / parameters["quiz_directory"] / run /
              f"{parameters['candidate_prefix']}{fingerprint[:parameters['fingerprint_prefix_length']]}"
              f"{parameters['schema_separator']}{parameters['proposal_schema']}" / parameters["proposals_filename"])
    if target.is_file():
        payload = _read_json(target)
        if payload.get("candidate_id") != job.get("candidate_id"):
            raise ValueError("The saved quiz belongs to another selected-model candidate")
        if not isinstance(payload.get("proposals"), list):
            raise ValueError("The saved quiz is incomplete; its existing file has been preserved")
        return target
    package = assets.parent
    if package != root and (package / parameters["binding_filename"]).is_file():
        return ensure_review_proposals(package, run, parameters=parameters, progress=progress)
    raise FileNotFoundError(f"Saved review proposals are unavailable: {target}")
