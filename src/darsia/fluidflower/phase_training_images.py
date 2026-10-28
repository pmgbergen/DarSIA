"""Which images a phase model was trained on: the frames that carry phase labels.

A phase model is trained on each run's baseline plus the frames that have
reviewed phase labels. The labels sit in the reviewed bundles that the model's
recorded training request lists, possibly through inherited requests. This
module only reads those records; it chooses nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MAX_ANCESTRY = 32


def _read(path, sha256=""):
    from .storage_paths import resolve_storage_path
    raw = Path(resolve_storage_path(Path(path))).read_bytes()
    if sha256 and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"Changed since the model was trained: {path}")
    return json.loads(raw.decode("utf-8-sig"))


def training_request_with_bundles(model_root):
    """Follow a model's recorded training-request ancestry to the request that lists its labelled bundles."""
    pin = dict(path=str(Path(model_root) / "training_request.json"), sha256="")
    for _ in range(MAX_ANCESTRY):
        request = _read(pin["path"], pin.get("sha256", ""))
        if request.get("bundles"):
            return request
        parent = request.get("inherited_training_request")
        if not isinstance(parent, dict) or not parent.get("path"):
            raise ValueError("The model records no labelled training images")
        pin = parent
    raise ValueError("The model's training-request ancestry is too long")


def labelled_training_images(model_root) -> dict:
    """Per run, the labelled images (times in hours) and the number of labels a model was trained on.

    Returns ``dict(runs={run: dict(times_h=[...], labels=n)}, bundles=n, unreadable=[...])``.
    An image is one recorded acquisition time; a bundle that cannot be read is
    named in ``unreadable`` instead of failing the whole list.
    """
    request = training_request_with_bundles(model_root)
    runs, unreadable = {}, []
    for bundle in request["bundles"]:
        decisions = bundle.get("decisions") or {}
        path = decisions.get("snapshot_path") or decisions.get("path") or decisions.get("source_path")
        try:
            document = _read(path, decisions.get("sha256", ""))
        except (OSError, ValueError, TypeError) as error:
            unreadable.append(str(error))
            continue
        for run, value in (document.get("decisions") or {}).items():
            row = runs.setdefault(str(run), dict(times=set(), labels=0))
            for annotation in (value or {}).get("annotations") or []:
                row["labels"] += 1
                if annotation.get("elapsed_h") is not None:
                    row["times"].add(round(float(annotation["elapsed_h"]), 4))
    return dict(runs={run: dict(times_h=sorted(row["times"]), labels=row["labels"])
                      for run, row in sorted(runs.items()) if row["labels"]},
                bundles=len(request["bundles"]), unreadable=unreadable)


def series_training_images(series_root) -> dict:
    """The labelled training images of the phase model the series' method names."""
    from .series_method import read_series_method
    from .pipeline.current_model_basis import _model_root
    method = read_series_method(series_root)
    if not method.model_version:
        raise ValueError("No phase model is named for this series under Calibration Method")
    folder = _model_root(series_root, method.model_version)
    return dict(labelled_training_images(folder), model=Path(folder).name)
