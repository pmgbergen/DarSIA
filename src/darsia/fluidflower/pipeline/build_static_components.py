"""Coordinate-only DATA transport of an acquisition's own frozen components.

Original static IDs and masks are gathered through one nearest-neighbour index
map. This module does not extend classes, use phase categories, or qualify any
physical domain. Reference-basis component production is outside this route.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np


_FIELDS = ("open_components", "barrier_mask", "reference_domain")


def _warp_components(values, transform, target_shape):
    """Gather every original dtype through one exact nearest source-index map."""
    import cv2
    from darsia._opencv_runtime import scientific_frame
    from ..geometry.photographic_data import homography

    shape = values["open_components"].shape
    if (len(shape) != 2 or len(target_shape) != 2
            or any(type(value) is not int or value < 1 for value in target_shape)):
        raise ValueError("Explicit positive native source and target grids are required")
    count = int(shape[0]) * int(shape[1])
    if not count or count > 2**53 or count > np.iinfo(np.intp).max:
        raise ValueError("Source indices must be exactly representable in float64 and native indexing")
    source_indices = np.arange(count, dtype=np.float64).reshape(shape)
    source_indices += 1
    with scientific_frame("static-components-coordinate-transport") as opencv:
        warped = cv2.warpPerspective(source_indices, homography(transform),
            (target_shape[1], target_shape[0]), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    del source_indices
    if (warped.shape != tuple(target_shape) or warped.dtype != np.float64
            or not np.isfinite(warped).all() or np.any(warped < 0) or np.any(warped > count)):
        raise ValueError("Nearest source indices are nonfinite or outside their original range")
    indices = warped.astype(np.intp)
    if not np.array_equal(warped, indices):
        raise ValueError("Nearest source indices are not exact integers")
    del warped
    valid = indices != 0
    selected = indices[valid] - 1
    del indices
    result = {}
    for name in _FIELDS:
        original = values[name]
        target = np.zeros(tuple(target_shape), dtype=original.dtype)
        target[valid] = original.ravel(order="C")[selected]
        result[name] = target
    return result, opencv


def _read_old_static(binding, case_id, *, run, acquisition, old_geometry,
                     old_native_configuration, old_raw_binding, old_audit,
                     reference_field, parameters, old_shape, reader):
    """Join an original typed run and its own component grid to admitted sources."""
    from .series import RUN_FORMAT, validate_run
    from ..mass.native_optical_source import _bound, _same
    from ..native import raw_rig_source, source

    bound = _bound(binding)
    configuration = reader.json(bound)
    if not isinstance(configuration, dict) or configuration.get("format") != RUN_FORMAT:
        raise ValueError("The original typed native run configuration is required")
    validate_run(configuration)
    if (configuration["run"] != run
            or not _same(configuration["native"]["configuration"], old_native_configuration)):
        raise ValueError("Old run and native configuration differ from the admitted transport")
    matches = [row for row in configuration["frames"] if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("One exact original component case is required")
    frame = matches[0]
    task = frame["native_task"]
    if (frame["source"]["source_sha256"] != acquisition["sha256"]
            or not _same(frame["geometry"], old_geometry)
            or task.get("kind") != "raw_rig"
            or not _same(task["processor"]["raw_binding"], old_raw_binding)):
        raise ValueError("Old case acquisition, geometry or RAW binding differs")
    native_binding = deepcopy(task["geometry"])
    native_frame = reader.json(native_binding)
    h_ref, h_b, _ = raw_rig_source.validate_frame(
        native_frame, reader.json(old_raw_binding), parameters)
    if (native_frame["acquisition"]["sha256"] != frame["source"]["source_sha256"]
            or frame["source"]["source_binding"]["source_sha256"] != frame["source"]["source_sha256"]
            or native_frame["run"] != run):
        raise ValueError("RAW native acquisition differs from the selected source identity")
    if (not source._same_h(h_ref, source.source_homography(old_audit[reference_field], parameters=parameters),
                           parameters=parameters)
            or not source._same_h(h_b, source.source_homography(old_audit["H_v5_B_to_frame_native"],
                                      parameters=parameters), parameters=parameters)):
        raise ValueError("Old native frame differs from its original RAW audit transforms")
    phase_binding = _bound(configuration["phase"]["configuration"])
    phase = reader.json(phase_binding)
    if (phase["parameters"]["geometry"]["native_shape_hw"] != list(old_shape)
            or phase["parameters"]["scientific"]["native_shape_hw"] != list(old_shape)):
        raise ValueError("Old phase configuration does not describe the old native grid")
    static = frame["phase_roles"]["static"]
    if not isinstance(static, dict) or set(static) != {"components", "source_ids", "unchanged_source"}:
        raise ValueError("Original exact static component roles are required")
    if static["unchanged_source"] is not True:
        raise ValueError("Reference-basis components are unsupported; an own old-native bundle is required")
    ids = static["source_ids"]
    if not isinstance(ids, list) or any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError("Literal positive original source component IDs are required")
    component_binding = deepcopy(static["components"])
    return deepcopy(static), dict(old_configuration=bound, old_phase_configuration=phase_binding,
                                  old_components=component_binding, old_native_frame=native_binding)


def _fresh_output(output, consumed_paths):
    """Reserve a fresh local destination outside every consumed source folder."""
    from ..job_filesystem import inspect_local_file
    from ..rig_raw_write import _no_links

    path = Path(output)
    spelling = str(path).replace("/", "\\")
    if (not path.is_absolute() or ".." in path.parts or spelling.startswith("\\\\")
            or path.exists() or not inspect_local_file(path / ".static-data-location").allowed):
        raise ValueError("Static component DATA needs one fresh positively local output folder")
    _no_links(path)
    resolved = path.resolve()
    for consumed in consumed_paths:
        parent = Path(consumed).resolve().parent
        if resolved == parent or resolved.is_relative_to(parent):
            raise ValueError("Static component output overlaps a consumed source folder")
    path.mkdir()
    return path


def prepare_static_components_data(*, geometry, old_configuration, case_id, output, reader=None):
    """Write an own-current-native bundle using the original checked RAW transport.

    File arguments are exact original descriptors. The shared coordinate reader
    replays its original metadata joins rather than trusting a saved matrix.
    Only the old component archive is decoded here. Its own-native source basis
    must be explicit; reference-basis bundles are refused without reinterpretation.
    The three original arrays use one nearest source-index map with outside zero
    or false. Integer ID storage is never converted through floating point.

    The caller supplies a fresh local destination. Original publishers preserve
    partial output on failure. This DATA does not grant physical qualification,
    extend classes, change a source receipt, or create phase labels.
    """
    from .build_geometry_data import _read_geometry_transport
    from ..geometry.barrier import array_bytes_sha256, validate_component_arrays
    from ..mass.n1_pipeline import _write_json, _write_npz, asset

    context = _read_geometry_transport(geometry, reader=reader)
    reader = context["reader"]
    inputs = deepcopy(context["inputs"])
    old_shape = context["old"]["renderer"]["native_shape_hw"]
    target_shape = context["native_shape"]
    static, old_inputs = _read_old_static(old_configuration, case_id,
        run=context["native"]["run"], acquisition=context["native"]["acquisition"],
        old_geometry=inputs["old_geometry"], old_native_configuration=inputs["old_native_configuration"],
        old_raw_binding=inputs["old_raw_binding"], old_audit=context["old"],
        reference_field=context["reference_field"], parameters=context["previous"],
        old_shape=old_shape, reader=reader)
    inputs.update(old_inputs)
    inputs["producer"] = asset(Path(__file__))
    values = validate_component_arrays(reader.archive(static["components"], _FIELDS), old_shape)
    transported, opencv = _warp_components(values, context["transform"], target_shape)
    validate_component_arrays(transported, target_shape)
    identities = {
        name: dict(dtype=values[name].dtype.str, old_shape_hw=list(values[name].shape),
                   new_shape_hw=list(transported[name].shape),
                   old_array_bytes_sha256=array_bytes_sha256(values[name]),
                   new_array_bytes_sha256=array_bytes_sha256(transported[name]))
        for name in _FIELDS
    }
    consumed_paths = [reader.path(binding) for binding in inputs.values()]
    destination = _fresh_output(output, consumed_paths)
    components = _write_npz(destination / "static_components.npz", transported)
    for binding in inputs.values():
        reader.path(binding)
    provenance = dict(format="darsia-static-component-coordinate-data-v1",
        run=context["native"]["run"], case_id=case_id,
        acquisition=deepcopy(context["native"]["acquisition"]), inputs=inputs,
        components=components, source_component_ids=deepcopy(static["source_ids"]),
        old_native_shape_hw=list(old_shape), current_native_shape_hw=list(target_shape),
        old_native_to_current_native_h=context["transform"].tolist(), arrays=identities,
        basis="own current acquisition native grid", interpolation="nearest source-index gather",
        outside_values=dict(open_components=0, barrier_mask=False, reference_domain=False),
        phase_categories_used=False, components_relabelled=False, class_extension_performed=False,
        physical_qualification=False, opencv=opencv,
        temporary_storage_scope="source/warped float64 index maps and native integer gather; not a measured memory bound")
    receipt = _write_json(destination / "static_components.json", provenance)
    for binding in inputs.values():
        reader.path(binding)
    return dict(static=dict(components=components, source_ids=deepcopy(static["source_ids"]),
                            unchanged_source=True), static_transport=receipt)
