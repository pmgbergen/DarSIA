"""Read-only placement of an unfinished painting on another run's own photo.

This path writes no review, approval, session or geometry. Logged placement
and adoption remain separate operations with their existing approval gates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np

from . import geometry_workspace_frame as wf
from . import layer_painting as lp
from . import layer_painting_manual as manual
from . import map_placement as placement


@dataclass(frozen=True)
class RunPreview:
    run_id: str
    photo: np.ndarray
    layers: np.ndarray
    uncovered: np.ndarray
    photo_path: str
    source_run: str
    source_name: str
    measurements: dict = field(default_factory=dict)
    native_shape_hw: tuple[int, int] | None = None
    display_shape_hw: tuple[int, int] | None = None
    H_native_to_display: tuple[tuple[float, ...], ...] | None = None
    geometry_binding: dict | None = None
    full_photo: dict | None = field(default=None, repr=False, compare=False)


def read_preview_geometry(index_path):
    """Read measured models with their exact file and transform identities.

    This is a metadata-worker operation. The preliminary index read discovers
    files to pin before the existing geometry reader runs. Both the index and
    every model are rechecked afterwards; no geometry is inferred or changed.
    """
    from .metric_analysis_domains import descriptor

    def bounded_descriptor(path):
        if Path(path).stat().st_size > placement.MAX_JSON_BYTES:
            raise ValueError("The measured geometry metadata exceeds the existing JSON size limit")
        return descriptor(path)

    from .storage_paths import resolve_storage_path
    index_path = resolve_storage_path(index_path).resolve()
    before = bounded_descriptor(index_path)
    index = placement._read_json(index_path)
    if index.get("format") != placement.GEOMETRY_FORMAT:
        raise ValueError("The selected file is not a measured geometry index")
    runs = index.get("runs")
    if isinstance(runs, dict):
        entries = [dict(value, run_id=key) for key, value in runs.items()]
    elif isinstance(runs, list):
        entries = [dict(value) for value in runs]
    else:
        raise ValueError("The geometry index needs explicit measured run entries")
    pinned = {}
    for entry in entries:
        run = str(entry.get("run_id") or entry.get("run") or "").strip().casefold()
        if not run or run in pinned:
            raise ValueError("The geometry index must name each measured run exactly once")
        path = placement._model_path(index_path, run, entry.get("reference_model"))
        pinned[run] = (bounded_descriptor(path), entry)
    models = placement.read_run_geometry(index_path)
    if set(models) != set(pinned) or bounded_descriptor(index_path) != before:
        raise ValueError("The geometry index changed while reading the preview models")
    identity = {}
    for run, model in models.items():
        bound, entry = pinned[run]
        if bounded_descriptor(model.source) != bound or {key:value for key,value in model.entry.items()
                if key != '_reference_point_review'} != entry:
            raise ValueError("A measured model changed while reading the preview geometry")
        identity[run] = dict(model=deepcopy(bound), entry=deepcopy(entry),
            reference_image=deepcopy(model.reference_image),
            shape_hw=None if model.shape_hw is None else list(model.shape_hw),
            metric_to_grid=model.metric_to_grid.tolist())
    if bounded_descriptor(index_path) != before:
        raise ValueError("The geometry index changed while reading the preview models")
    return models, dict(geometry_index=before, models=identity)


def _same_path(left, right):
    from .storage_paths import resolve_storage_path
    return os.path.normcase(str(resolve_storage_path(left).resolve())) == os.path.normcase(str(resolve_storage_path(right).resolve()))


def _photo_geometry(geometry, config, photo_path, shape):
    """Name the exact photo grid; never infer a crop or resize from dimensions."""
    raw = geometry.reference_image
    if raw.get("path") and _same_path(raw["path"], photo_path):
        if geometry.shape_hw is None or tuple(geometry.shape_hw) != tuple(shape):
            raise ValueError("The measured model and the painting have different camera grids")
        return geometry
    if wf.frame_kind(config) == "rig-corrected":
        raise ValueError("The saved corrected baseline has no verified identity link to the measured native renderer. "
                         "Its dimensions alone cannot establish that coordinate transform.")
    raise ValueError("The measured model does not bind the painting's photograph and coordinate frame")


def drawing_layers(session, frame):
    """The current polygons and strokes, with gaps left unassigned."""
    if session.image_size is None:
        raise ValueError("The drawing has no recorded photograph grid")
    width, height = session.image_size
    shape = (height, width)
    layers = lp.rasterize_polygons(session.polygons, shape, (1., 1.))
    strokes = lp.rasterize_strokes(session.strokes, shape, (1., 1.))
    # Erase strokes remove previous strokes, as in the drawing canvas; they
    # do not erase polygons beneath that independent marks layer.
    layers[strokes != 0] = strokes[strokes != 0]
    from .painting_layer_priority import apply_priority

    layers = apply_priority(session, layers, (1., 1.))
    layers[~lp.frame_mask(frame, shape, (1., 1.))] = lp.OUTSIDE
    return layers


def preview_layers(session, proposal, corrections, frame, *, source="map", record=None):
    """Rasterize a snapshot only. Approval is deliberately irrelevant here."""
    if record is not None:
        if record.get("kind") == "drawing":
            # A drawing record is loaded by the controller, which owns its project.
            raise ValueError("Load the drawing snapshot before preparing its preview")
        if manual.is_manual_version(record):
            saved, painted = manual.load_manual_version(record)
            return saved, painted.classes.copy()
        if record.get("proposal_method") == "polygons_only_v1":
            from . import layer_painting_polygons_only as polygons
            saved, painted, saved_corrections = polygons.load_version(record)
            classes, _changed, _unused = polygons.corrected_classes(painted, saved_corrections,
                layer_ids={row.layer_id for row in saved.layers})
            return saved, classes
        saved, painted, saved_corrections = lp.load_version(record)
        return saved, manual.painted_layers(saved, painted, frame=frame, corrections=saved_corrections)
    if source == "manual":
        return session, manual.manual_map(session, session.image_size, frame=frame).classes
    if source == "drawing" or proposal is None:
        return session, drawing_layers(session, frame)
    if hasattr(proposal, "source") and hasattr(proposal, "inside"):
        from . import layer_painting_polygons_only as polygons
        if not polygons.proposal_current(proposal, session, frame):
            raise ValueError("The proposal is out of date. Choose Current drawing or propose again.")
        classes, _changed, _unused = polygons.corrected_classes(proposal, corrections,
            layer_ids={row.layer_id for row in session.layers})
        return session, classes
    # A stale proposal must not silently be described as the current map.
    if proposal.digest != lp.proposal_digest(session, frame):
        raise ValueError("The proposal is out of date. Choose Current drawing or propose again.")
    return session, manual.painted_layers(session, proposal, frame=frame, corrections=corrections)


def prepare_preview(session, layers, *, geometry_index, target_run, config,
                    photo_loader, source_name="Current map", display_limit=1600,
                    measure_offsets=False, on_progress=None, on_preview=None):
    """Apply the same measured homographies as placement, entirely in memory."""
    import cv2
    def progress(message):
        if on_progress is not None:
            on_progress(message)

    progress("Reading measured frame models")
    models, geometry_binding = read_preview_geometry(geometry_index)
    source_run, target_run = session.run_id.casefold(), str(target_run).casefold()
    if source_run not in models or target_run not in models:
        raise ValueError("The geometry index needs exact models for both selected runs")
    layers = np.asarray(layers)
    if layers.dtype != np.uint8 or layers.ndim != 2:
        raise ValueError("The preview requires a native uint8 layer grid")
    source = _photo_geometry(models[source_run], config, session.image_path, layers.shape)
    target = models[target_run]
    if wf.frame_kind(config) == "rig-corrected":
        raise ValueError("The saved corrected baseline has no verified identity link to the measured native renderer. "
                         "Preview on that grid is unavailable; no raw-photo fallback was used.")
    progress("Loading " + target_run + " photograph on its camera grid")
    photo = np.asarray(photo_loader(target_run, target))
    if photo.dtype != np.uint8 or photo.ndim != 3 or photo.shape[2] != 3:
        raise ValueError("The target photograph must be uint8 RGB")
    target_shape = photo.shape[:2]
    path = str(target.reference_image.get("path") or "")
    target = _photo_geometry(target, config, path, target_shape)
    progress("Placing the map on " + target_run)
    placed, uncovered = placement.place_layer_map(layers, source, target, target_shape)
    roles = {row.layer_id: session.facies_of(row.layer_id).role for row in session.layers}
    measurements = {"offset_px": placement.map_offset(layers, source, target),
                    "checks": placement.run_checks(placed, uncovered, roles, set(roles))}
    def display_result(values):
        # Only display copies are reduced; the full native arrays below still
        # feed the unchanged optional boundary measurements.
        ratio = min(1., display_limit / max(target_shape))
        size = (max(1, round(target_shape[1] * ratio)), max(1, round(target_shape[0] * ratio)))
        shown_photo, shown_layers, shown_uncovered = photo, placed, uncovered
        if (photo.shape[1], photo.shape[0]) != size:
            shown_photo = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
        if (placed.shape[1], placed.shape[0]) != size:
            shown_layers = cv2.resize(placed, size, interpolation=cv2.INTER_NEAREST)
            shown_uncovered = cv2.resize(uncovered.astype(np.uint8), size,
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
        sx, sy = size[0] / target_shape[1], size[1] / target_shape[0]
        display_transform = ((sx, 0., (sx-1.)/2.), (0., sy, (sy-1.)/2.), (0., 0., 1.))
        return RunPreview(target_run, shown_photo, shown_layers, shown_uncovered,
                          str(path), source_run, source_name, values,
                          native_shape_hw=tuple(target_shape),
                          display_shape_hw=tuple(shown_photo.shape[:2]),
                          H_native_to_display=display_transform,
                          geometry_binding=deepcopy(geometry_binding))

    if measure_offsets and on_preview is not None:
        # A shallow copy prevents adding the later measurement to an already
        # published packet. Its nested base measurements are read-only here.
        on_preview(display_result(dict(measurements)))
    if measure_offsets:
        progress("Loading " + source_run + " reference photograph for boundary measurements")
        reference_photo = np.asarray(photo_loader(source_run, source))
        if reference_photo.shape != (*layers.shape, 3):
            raise ValueError("The reference photograph no longer matches the painting grid")
        progress("Measuring boundaries on " + source_run)
        reference_offsets = placement.layer_offsets(reference_photo, layers, source, roles)
        progress("Measuring boundaries on " + target_run)
        offsets = placement.layer_offsets(photo, placed, target, roles)
        placement.add_changes_from_reference(offsets, reference_offsets)
        measurements["layer_offsets"] = offsets
    progress("Preview ready")
    return display_result(measurements)
