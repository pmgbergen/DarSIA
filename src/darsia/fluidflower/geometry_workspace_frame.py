"""Explicit saved-rig coordinates for the geometry workspace, never a warp.

The binding records each run independently. Display reduction does not change
any source raster or claim that native Setup is qualified for corrected inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

FRAME_BINDING_FORMAT = "darsia-rig-corrected-workspace-frame-v1"
FIELDS = ("shape_corrected_baseline", "labels", "facies", "depth")
METADATA_LIMIT = 1024 * 1024


def adopted_content_digest(config):
    """Lock the unedited adoption; Store may only materialise its depth copy."""
    content = {key: value for key, value in config.items()
               if key not in {"provenance", "depth_path", "depth_artifact_path", "depth_source_reference"}}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def frame_kind(config):
    value = config.get("frame", "raw")
    if value not in {"raw", "rig-corrected"}:
        raise ValueError("Unknown geometry workspace frame: " + repr(value))
    return value


def current_configuration(project):
    """One bounded current-revision read; no history, source scan or writes."""
    from .geometry_facies import GeometryRevisionStore
    path = GeometryRevisionStore(project).current_path
    try:
        with path.open("rb") as stream:
            raw = stream.read(METADATA_LIMIT + 1)
    except FileNotFoundError:
        return {}
    if len(raw) > METADATA_LIMIT:
        raise ValueError("Workspace revision exceeds the 1 MiB display metadata limit")
    revision = json.loads(raw.decode("utf-8-sig"))
    config = revision.get("configuration", revision)
    if not isinstance(config, dict):
        raise ValueError("Workspace revision has no configuration object")
    kind = frame_kind(config)
    if revision.get("frame", kind) != kind:
        raise ValueError("Revision and configuration name different coordinate frames")
    return config


def _descriptor(value, *, check_file):
    if (not isinstance(value, Mapping) or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0):
        raise ValueError("Corrected fields require explicit absolute path/SHA/byte bindings")
    shape = value.get("shape")
    if (not isinstance(shape, list) or len(shape) not in {2, 3}
            or any(type(v) is not int or v <= 0 for v in shape)):
        raise ValueError("Corrected fields require their recorded grid shape")
    from .storage_paths import resolve_storage_path
    physical = resolve_storage_path(value["path"])
    if check_file:
        info = physical.stat()
        if info.st_size != value["size_bytes"]:
            raise ValueError("Corrected display source size changed: " + value["path"])
        if "mtime_ns" in value and info.st_mtime_ns != value["mtime_ns"]:
            raise ValueError("Corrected display source version changed: " + value["path"])
    # SHA is the import's source identity. Passive display does not re-hash a
    # 195 MiB image or claim that a stat test is a new payload-SHA validation.
    return dict(deepcopy(dict(value)), path=str(physical))


def selected_fields(config, run_id, *, check_files=True):
    """Return this exact run's four old rig fields; never a common-map fallback."""
    if frame_kind(config) != "rig-corrected":
        raise ValueError("No corrected-field selection exists for a raw workspace")
    binding = config.get("frame_binding")
    if (not isinstance(binding, Mapping) or binding.get("format") != FRAME_BINDING_FORMAT
            or not isinstance(binding.get("runs"), Mapping)):
        raise ValueError("Corrected workspace is missing its explicit per-run frame binding")
    if (not isinstance(binding.get("source_reference_run"), str)
            or not binding["source_reference_run"].strip()
            or binding.get("display_reference_run") not in binding["runs"]
            or not isinstance(binding.get("reference_shape_corrections"), list)
            or not binding["reference_shape_corrections"]):
        raise ValueError("Corrected workspace lacks its source/display/correction-chain identity")
    for correction in binding["reference_shape_corrections"]:
        if (not isinstance(correction, Mapping) or not isinstance(correction.get("path"), str)
                or not Path(correction["path"]).is_absolute()
                or re.fullmatch(r"[0-9a-f]{64}", str(correction.get("sha256", ""))) is None
                or type(correction.get("size_bytes")) is not int or correction["size_bytes"] < 0):
            raise ValueError("Reference shape corrections require exact path/SHA/byte bindings")
    row = binding["runs"].get(str(run_id))
    if not isinstance(row, Mapping):
        raise ValueError("No corrected rig maps are bound for selected run " + str(run_id))
    result = {name: _descriptor((row.get("fields") or {}).get(name), check_file=check_files)
              for name in FIELDS}
    shape = result["shape_corrected_baseline"]["shape"][:2]
    if any(result[name]["shape"] != shape for name in ("labels", "facies", "depth")):
        raise ValueError("This run's corrected baseline, labels, facies and depth grids differ")
    if result["shape_corrected_baseline"]["shape"] != [*shape, 3]:
        raise ValueError("Corrected workspace baseline must be an RGB field")
    if not isinstance(row.get("configured_depth_map"), Mapping) or not row["configured_depth_map"].get("path"):
        raise ValueError("The original configured depth pointer is not recorded for " + str(run_id))
    result["configured_depth_map"] = deepcopy(dict(row["configured_depth_map"]))
    return result


def workspace_reference(config, reference, *, check_files=True):
    if frame_kind(config) == "raw" or reference is None:
        return reference
    fields = selected_fields(config, reference.run_id, check_files=check_files)
    return replace(reference, baseline_image=Path(fields["shape_corrected_baseline"]["path"]))


def corrected_rig_frame(config, run_id):
    """The saved metric image's outside edges, not new photograph corners."""
    from .series_geometry import RigFrame
    fields = selected_fields(config, run_id, check_files=False)
    height, width = fields["labels"]["shape"]
    size = []
    for key in ("width_m", "height_m"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Corrected frame requires recorded " + key)
        size.append(float(value))
    # Existing RigFrame uses pixel centres; image edges at +/-0.5 give exactly
    # width_m/width and height_m/height. This does not transform any raster.
    return RigFrame(corners={"top_left": (-.5, -.5), "bottom_left": (height-.5, -.5),
        "bottom_right": (height-.5, width-.5), "top_right": (-.5, width-.5)},
        width_m=size[0], height_m=size[1], run_id=str(run_id), source="saved corrected rig frame",
        image=fields["shape_corrected_baseline"]["path"], image_size=(width, height))


def load_painting_image(config, reference, *, analysis_width=None, display_limit=None):
    """Use the original raw loader or saved corrected RGB; UI reduction only."""
    from . import layer_painting as painting
    if frame_kind(config) == "raw":
        return painting.load_painting_image(Path(reference.baseline_image))
    import cv2
    import numpy as np
    from .geometry_facies import _load_array, _rgb_float
    fields = selected_fields(config, reference.run_id)
    selected = Path(fields["shape_corrected_baseline"]["path"])
    raw = _load_array(selected)
    if list(raw.shape) != fields["shape_corrected_baseline"]["shape"]:
        raise ValueError("Corrected baseline no longer has its bound grid")
    rgb = _rgb_float(raw)
    height, width = rgb.shape[:2]
    limit = painting.DISPLAY_LIMIT_PX if display_limit is None else display_limit
    analysis_limit = painting.ANALYSIS_WIDTH if analysis_width is None else analysis_width
    if type(limit) is not int or limit <= 0 or type(analysis_limit) is not int or analysis_limit <= 0:
        raise ValueError("Positive display limits are required")
    scale = min(1.0, limit/max(width, height))
    target = (max(1, int(round(width*scale))), max(1, int(round(height*scale))))
    display = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
    target_width = min(analysis_limit, display.shape[1])
    if target_width < display.shape[1]:
        target_height = max(8, int(round(height*target_width/width)))
        analysis = cv2.resize(display, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        analysis = display.copy()
    return painting.PaintingImage(selected, (width, height),
        np.ascontiguousarray(display, dtype=np.float32), np.ascontiguousarray(analysis, dtype=np.float32))


def saved_boundaries(config, run_id, display_shape, *, project=None):
    """A transparent view of saved labels; no classification or new labels."""
    import cv2
    import numpy as np
    from .geometry_facies import _load_array, _layer_boundary_mask
    fields = selected_fields(config, run_id)
    registered_selected = False
    if config.get("source_mode") == "canonical-from-import":
        from .geometry_facies import _effective_run_geometry_choice
        registered_selected = any(str(row.get("run_id") or "").strip().casefold() == str(run_id).strip().casefold()
            and _effective_run_geometry_choice(row) == "run_override"
            for row in (config.get("run_geometry") or {}).get("runs") or [])
    if registered_selected:
        if project is None:
            raise ValueError("A registered boundary view needs its actual project/current Store.")
        from .geometry_facies import GeometryRevisionStore
        from .geometry_raster_sources import registered_run_sources, verify_registered_sources
        current = GeometryRevisionStore(project).current()
        if current.get("configuration") != config:
            raise ValueError("The boundary view differs from the genuine saved registered revision.")
        registered = registered_run_sources(project.root, current, run_id)
        fields = deepcopy(fields)
        fields["labels"] = {**registered["files"]["labels"], "shape": registered["labels_shape"]}
        verify_registered_sources(registered)
    labels = _load_array(Path(fields["labels"]["path"]))
    if list(labels.shape) != fields["labels"]["shape"] or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Saved corrected labels no longer match their recorded integer grid")
    height, width = display_shape
    reduced = cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    boundary = _layer_boundary_mask(reduced, set(int(v) for v in np.unique(reduced)))
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[boundary] = (255, 255, 255, 235)
    return rgba
