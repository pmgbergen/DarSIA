"""Read original boxes from the loaded native series' exact configuration chain.

Call in the painting controller's metadata worker. Capture the reference's
``config_path`` on the GUI thread first: this module never calls a widget or a
reference getter. It writes nothing and never selects a job or merged file.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from darsia.presets.workflows.config.roi_registry import RoiRegistry

from . import metric_analysis_domains as domains
from . import series_paths
from .series_geometry import _storage_path, native_series_link, read_series_manifest
from .storage_paths import resolve_storage_path


LEGACY_FRAME = "legacy_nominal_0.9x0.49_m"
_ROLES = ("box1", "box2")


def _selection(project, run_id, reference_config_path):
    link = native_series_link(project)
    if not link.get("series_root") or not link.get("series_id"):
        raise ValueError("Original boxes need the project's explicit native-series link")
    root = resolve_storage_path(link["series_root"])
    if not root.is_absolute():
        raise ValueError("The linked native-series root must be an explicit absolute path")
    root = root.resolve()
    dataset = series_paths.series_key(link["series_id"])
    run = str(run_id).strip().casefold()
    bound = domains.descriptor(root / "series.toml")
    manifest = read_series_manifest(root)
    if series_paths.series_key(manifest.get("series_id", "")) != dataset:
        raise ValueError("The linked native-series ID and selected manifest differ")
    ids = (manifest.get("runs") or {}).get("ids", [])
    if (not isinstance(ids, list) or run not in ids
            or sum(str(value).strip().casefold() == run for value in ids) != 1):
        raise ValueError("The selected reference run is not uniquely declared in this series")
    if series_paths.series_root(dataset).resolve() != root:
        raise ValueError("The registered series root differs from the loaded project's link")

    paths = tuple(Path(path).resolve() for path in series_paths.config_files(dataset, run))
    table = manifest.get("runs") or {}
    declared = (table.get("config_files") or {}).get(run)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("The reference run has no explicit declared configuration file")
    settings = manifest.get("paths") or {}
    common = _storage_path(settings.get("common_config", "config/common.toml"), root)
    selected = _storage_path(declared, root)
    expected = _storage_path(reference_config_path, root)
    if common is None or selected is None or expected is None:
        raise ValueError("The common and reference configuration paths must be explicit")
    common, selected, expected = (path.resolve() for path in (common, selected, expected))
    if expected != selected:
        raise ValueError("The captured reference configuration differs from this run's declared file")
    if len(paths) not in (2, 3) or paths[:2] != (common, selected) or len(set(paths)) != len(paths):
        raise ValueError("The native configuration chain is not the declared common/reference sequence")
    config_folder = _storage_path(settings.get("config", "config"), root)
    if len(paths) == 3 and (config_folder is None or paths[2] != (config_folder / "coloron.toml").resolve()):
        raise ValueError("The native configuration chain contains an undeclared overlay")
    state = selected.parent / ".color_state" / f"{run}.txt"
    selector = domains.descriptor(state) if state.is_file() else None
    if domains.descriptor(root / "series.toml") != bound:
        raise ValueError("The series manifest changed while selecting the original boxes")
    return dict(project_root=str(Path(project.root).resolve()), series_root=str(root),
        series_id=dataset, run_id=run, series_manifest=bound, paths=paths,
        reference_config_path=str(expected), color_state=selector)


def load_legacy_box_source(project, run_id, *, reference_config_path):
    """Return original box values and provenance, pending legacy-frame review.

    The existing ``config_files`` and ``RoiRegistry`` readers own selection and
    parsing. Their full order, optional active colour overlay and duplicate
    refusal are retained. Both absent roles return ``None``. A partial pair,
    label restriction or ambiguous/unavailable source raises; no other file or
    job is tried. Existing measured-box conflicts belong to the caller's draft
    and are never resolved or overwritten here.
    """
    selected = _selection(project, run_id, reference_config_path)
    paths = selected["paths"]
    pinned = [domains.descriptor(path) for path in paths]
    source = {key: deepcopy(value) for key, value in selected.items() if key != "paths"}
    source.update(kind="native_series_roi_registry", configurations=deepcopy(pinned),
                  frame=LEGACY_FRAME, decision="pending")
    registry = RoiRegistry().load(list(paths))
    present = set(registry.keys()) & set(_ROLES)
    values = None
    if present:
        if present != set(_ROLES):
            raise ValueError("Both original box1 and box2 must be present; partial input is refused")
        resolved = registry.resolve_rois(list(_ROLES))
        if set(resolved) != set(_ROLES):
            raise ValueError("Original box1 and box2 must be unrestricted ROIs, without a label restriction")
        values = {name: np.asarray(resolved[name].roi).tolist() for name in _ROLES}
        # Reuse the existing rectangle/finite-value validation. It also defines
        # the unchanged legacy frame; no coordinates are rescaled or clipped.
        domains.import_legacy_boxes(values, source=source)

    try:
        unchanged = (_selection(project, run_id, reference_config_path) == selected
                     and [domains.descriptor(path) for path in paths] == pinned)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("The original box source changed while it was being read") from error
    if not unchanged:
        raise ValueError("The original box source changed while it was being read")
    return dict(values=values, source=source) if values is not None else None


__all__ = ["LEGACY_FRAME", "load_legacy_box_source"]
