"""Apply the current corrected geometry revision to Setup for all runs.

The existing per-run route does each run's work: ``export_corrected_setup``
(the run's corrected export, byte copies of the revision's fields) and then
``apply_corrected_selected_run`` (only the run configuration's label and
facies references change; the originals are kept beside the export). The
display reference run goes first, because every other run's export is bound
to its export. The first run that cannot be applied stops the sequence, and
one receipt lists every run: applied, stopped (with its error and its own
receipt) or not attempted. Build run model is not run.

:func:`check_rebuilt_runs` is the re-export gate after Build run model:
each run's applied export is checked as Build checks it, and the rig
folder its configuration names must hold exactly the labels and facies
the export adopted. It reads only, and one receipt lists every run as
identical, differs, not built or not applied.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import uuid

import tomllib

import numpy as np

from darsia.fluidflower import corrected_setup as cs
from darsia.fluidflower.geometry_workspace_frame import frame_kind
from darsia.fluidflower.project import utc_now

ALL_RUNS_FORMAT = "darsia-corrected-setup-apply-all-runs-v1"
REBUILT_CHECK_FORMAT = "darsia-corrected-setup-rebuilt-check-v1"


def apply_order(runs: Sequence[str], display: str) -> list[str]:
    """The display reference run first, then the other runs in their order."""

    if display not in runs:
        raise ValueError(f"The display reference run {display} is not among the runs to apply.")
    return [display] + [run for run in runs if run != display]


def current_corrected_revision(project: Any) -> dict[str, Any]:
    """The current corrected-frame revision's id, display run and SHA-256.

    Raises when the current revision is not a saved corrected-frame revision.
    """

    path = cs.GeometryRevisionStore(project).current_path
    raw = cs._small(path)
    current = json.loads(raw)
    config = current.get("configuration") or {}
    if frame_kind(config) != "rig-corrected":
        raise ValueError("Apply to all runs is for a corrected-frame geometry revision.")
    return {
        "revision_id": str(current["revision_id"]),
        "display_reference_run": str(config["frame_binding"]["display_reference_run"]),
        "reason": str(current.get("reason") or ""),
        "recorded_utc": str(current.get("recorded_utc") or ""),
        "current_sha256": hashlib.sha256(raw).hexdigest(),
        "configuration": config,
    }


def apply_geometry_to_all_runs(
    project: Any,
    references: Sequence[Any],
    *,
    actor: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Export and apply the current corrected revision for every run.

    ``references`` name the runs (each has ``run_id``), in series order.
    Returns the all-runs receipt's path and its content; the receipt goes
    beside the display run's export of the revision.
    """

    revision = current_corrected_revision(project)
    by_run = {str(reference.run_id): reference for reference in references}
    order = apply_order(list(by_run), revision["display_reference_run"])
    rows: list[dict[str, Any]] = []
    for number, run in enumerate(order, 1):
        if progress is not None:
            progress(f"Applying the geometry to {run.upper()} ({number} of {len(order)})…")
        try:
            export = cs.export_corrected_setup(project, reference=by_run[run], actor=actor)
            receipt = cs.apply_corrected_selected_run(project, export)
        except Exception as error:  # noqa: BLE001 - recorded, and the sequence stops
            failed = getattr(error, "receipt", None)
            rows.append(
                {
                    "run_id": run,
                    "status": "stopped",
                    "error": f"{type(error).__name__}: {error}",
                    "apply_receipt": str(failed) if failed else None,
                }
            )
            break
        rows.append(
            {
                "run_id": run,
                "status": "applied",
                "export": str(export.destination),
                "export_reused": bool(export.reused),
                "apply_receipt": str(receipt),
            }
        )
    attempted = {row["run_id"] for row in rows}
    not_attempted = [run for run in order if run not in attempted]
    result = {
        "format": ALL_RUNS_FORMAT,
        "created_utc": utc_now(),
        "created_by": str(actor),
        "revision_id": revision["revision_id"],
        "current_sha256": revision["current_sha256"],
        "display_reference_run": revision["display_reference_run"],
        "runs": rows,
        "not_attempted": not_attempted,
        "complete": not not_attempted and all(row["status"] == "applied" for row in rows),
        "Build_run": False,
        "crop_changed": False,
        "depth_pointer_changed": False,
    }
    root = cs._canonical_location(
        project.root, revision["configuration"], revision["revision_id"],
        revision["display_reference_run"],
    )[0]
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = root / f"apply-all-runs-{stamp}-{uuid.uuid4().hex[:8]}.json"
    with path.open("xb") as stream:
        stream.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return path, result


def rebuilt_run_check(project: Any, run: str, *, revision: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Whether a run's built rig holds exactly what its applied export adopted.

    The run's applied corrected export is checked as Build run model checks
    it (on the run's declared merged configuration); then the labels and
    facies in the rig folder that configuration names are compared, array
    for array, with the export's adopted labels and facies. Reads only.
    """

    from darsia.fluidflower.geometry_facies import _load_array
    from darsia.fluidflower.series_geometry import _storage_path

    revision = dict(revision or current_corrected_revision(project))
    try:
        _root, destination, _display, _planned = cs._canonical_location(
            project.root, revision["configuration"], revision["revision_id"], run
        )
        export = json.loads(cs._small(destination / "export.json"))
        merged = Path(export["binding"]["configuration_bundles"][run][-1])
        receipt = cs.require_corrected_export_for_build(merged)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return {"run_id": run, "status": "not applied", "detail": f"{type(error).__name__}: {error}"}
    configuration = tomllib.loads(cs._small(merged).decode("utf-8-sig"))
    rig = _storage_path((configuration.get("rig") or {}).get("path"), Path(receipt["binding"]["series_root"]))
    row = {"run_id": run, "configuration": str(merged), "rig": str(rig) if rig is not None else None}
    if rig is None or not all((rig / f"{name}.npz").is_file() for name in ("labels", "facies")):
        return dict(row, status="not built")
    comparisons = {}
    for name in ("labels", "facies"):
        built = np.squeeze(_load_array(rig / f"{name}.npz"))
        adopted = np.squeeze(_load_array(Path(receipt["binding"]["sources"][name]["path"])))
        same_grid = built.shape == adopted.shape
        comparisons[name] = {
            "identical": bool(same_grid and np.array_equal(built, adopted)),
            "different_pixels": int(np.count_nonzero(built != adopted)) if same_grid else None,
            "built_shape": list(built.shape),
            "adopted_shape": list(adopted.shape),
        }
    status = "identical" if all(item["identical"] for item in comparisons.values()) else "differs"
    return dict(row, status=status, comparisons=comparisons)


def check_rebuilt_runs(
    project: Any,
    references: Sequence[Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """The re-export gate for every run, with one receipt beside the revision's
    export: each run identical, differs, not built or not applied."""

    revision = current_corrected_revision(project)
    order = apply_order([str(reference.run_id) for reference in references], revision["display_reference_run"])
    rows = []
    for number, run in enumerate(order, 1):
        if progress is not None:
            progress(f"Checking {run.upper()} ({number} of {len(order)})…")
        rows.append(rebuilt_run_check(project, run, revision=revision))
    result = {
        "format": REBUILT_CHECK_FORMAT,
        "created_utc": utc_now(),
        "revision_id": revision["revision_id"],
        "current_sha256": revision["current_sha256"],
        "runs": rows,
        "identical": [row["run_id"] for row in rows if row["status"] == "identical"],
        "all_identical": all(row["status"] == "identical" for row in rows),
    }
    root = cs._canonical_location(
        project.root, revision["configuration"], revision["revision_id"],
        revision["display_reference_run"],
    )[0]
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = root / f"rebuilt-check-{stamp}-{uuid.uuid4().hex[:8]}.json"
    with path.open("xb") as stream:
        stream.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return path, result


__all__ = [
    "ALL_RUNS_FORMAT",
    "REBUILT_CHECK_FORMAT",
    "apply_geometry_to_all_runs",
    "apply_order",
    "check_rebuilt_runs",
    "current_corrected_revision",
    "rebuilt_run_check",
]
