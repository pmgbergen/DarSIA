"""Prepare measured geometry DATA from saved camera-plane models.

The legacy model path performs no image decoding or fitting. The explicit
released RAW-basis path reads its saved NPZ and original grid in its caller's
worker; it does not run RAW decoding, registration, adoption or worker launch.
The camera plane is U (column right, row down); physical coordinates are
millimetres from the inner left-wall/bottom intersection, x right and y up.
The optical offset is an explicit area factor, applied once after float32
area construction. It is not folded into the physical coordinate matrix.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

FORMAT = "darsia-measured-run-geometry-v1"


def matrix(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("A finite 3 by 3 coordinate matrix is required")
    if np.linalg.matrix_rank(result) != 3:
        raise ValueError("The coordinate matrix is singular")
    return result


def descriptor(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f"Input changed while being read: {path}")
    return {"path": str(path), "sha256": digest, "size_bytes": after.st_size}


def read_bound_json(bound):
    path = Path(bound["path"])
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError("Saved model changed during its read")
    if hashlib.sha256(content).hexdigest() != bound["sha256"] or len(content) != bound["size_bytes"]:
        raise ValueError("Saved model content differs from its binding")
    return json.loads(content)


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return descriptor(path)


def reference_model(run, frame, *, reference_image, camera_profile, source_model,
                    edge_review, baseline_configuration, model_photograph=None,
                    baseline_to_model=None, baseline_native_renderer=None,
                    raw_shape_basis_authority=None, raw_shape_basis_workspace=None,
                    raw_shape_basis_aliases=(), raw_shape_basis_folder=None):
    """Bind a saved model to the registered run's actual reference photograph."""
    if not run or not reference_image.get("sha256"):
        raise ValueError("A run and bound reference photograph are required")
    saved_h = matrix(frame["H_camera_to_mm"])
    h = saved_h.copy()
    photograph = reference_image if model_photograph is None else model_photograph
    different = photograph["sha256"] != reference_image["sha256"]
    if different and baseline_to_model is None:
        raise ValueError("A different model photograph needs its measured baseline registration")
    if baseline_to_model is not None:
        if raw_shape_basis_authority is None:
            bridge = read_bound_json(baseline_to_model)
        else:
            from .raw_shape_basis_reader import _Pins
            bridge = _Pins(raw_shape_basis_aliases, folder=raw_shape_basis_folder).add(
                baseline_to_model, metadata=True, copyable=True)
        if bridge.get("accepted") is not True:
            raise ValueError("The baseline registration has not passed its displacement gate")
        if different:
            if (bridge["baseline_raw"]["sha256"] != reference_image["sha256"]
                    or bridge["model_raw"]["sha256"] != photograph["sha256"]):
                raise ValueError("The baseline registration binds another acquisition")
        elif bridge.get("exact_same_RAW_sha256") != reference_image["sha256"]:
            raise ValueError("Identity registration needs the exact same RAW acquisition")
        transport = matrix(bridge["H_baseline_u_to_model_u"])
        if (not np.array_equal(transport[:2, :2], np.eye(2))
                or not np.array_equal(transport[2], [0, 0, 1])):
            raise ValueError("The production drift bridge must remain a translation")
        displacement = float(np.linalg.norm(transport[:2, 2]))
        limit = float(bridge.get("displacement_limit_px", 0.5))
        if limit != 0.5 or displacement > limit:
            raise ValueError("The baseline registration exceeds the authorised displacement gate")
        if not math.isclose(displacement, float(bridge["maximum_displacement_px"]), abs_tol=1e-9):
            raise ValueError("The baseline displacement differs from its recorded transform")
        h = saved_h @ transport
    width, height = camera_profile["camera_wh"]
    if min(width, height) < 1:
        raise ValueError("The camera-U grid must have positive dimensions")
    factor = float(frame["sand_area_factor14"])
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("The saved optical area factor must be finite and positive")
    result = {
        "format": FORMAT, "run": run, "run_id": run,
        "reference_image": deepcopy(reference_image),
        "baseline_configuration": deepcopy(baseline_configuration),
        "grid": "camera_u", "shape_hw": [height, width],
        "camera_profile": deepcopy(camera_profile),
        "H_camera_u_to_metric_mm": h.tolist(),
        "H_metric_mm_to_camera_u": np.linalg.inv(h).tolist(),
        "metric_frame": {
            "unit": "mm", "origin": "inner left-wall and bottom intersection",
            "x_axis": "right", "y_axis": "up", "measured_width_mm": 899.0,
            "top_height_independently_measured": False,
            "legacy_dimensions_m_are_separate": [0.49, 0.9],
        },
        "optical_offset_mm": 14.0, "area_factor": factor,
        "area_factor_rule": "saved sand_area_factor14; apply once, after float32 area construction",
        "source_model": deepcopy(source_model), "current_edges": deepcopy(edge_review),
        "saved_model_certificate": deepcopy(frame["explicit_certificate"]),
        "saved_model_state": frame.get("state"),
        "new_fit": False, "adopted": False,
        "frame_rows_complete": False, "area_arrays_complete": False,
    }
    result.update(model_photograph=deepcopy(photograph),
                  H_model_photograph_u_to_metric_mm=saved_h.tolist(),
                  baseline_to_model_registration=deepcopy(baseline_to_model))
    if ((raw_shape_basis_authority is None) != (raw_shape_basis_workspace is None)
            or raw_shape_basis_authority is not None and baseline_native_renderer is None):
        raise ValueError("An explicit RAW basis needs its renderer, external authority and original per-run workspace")
    if baseline_native_renderer is not None:
        checked_basis = None
        if raw_shape_basis_authority is not None:
            from .raw_shape_basis_reader import read_shape_basis
            checked_basis = read_shape_basis(baseline_native_renderer, run_id=run,
                authority=raw_shape_basis_authority, parent_workspace=raw_shape_basis_workspace,
                reference_image=reference_image, camera_profile=camera_profile,
                aliases=raw_shape_basis_aliases, folder=raw_shape_basis_folder)
            native = checked_basis.document
            if not _same_identity(baseline_configuration, checked_basis.compatibility["configuration"]):
                raise ValueError("The RAW basis and measured model have another original run configuration")
        else:
            native = read_bound_json(baseline_native_renderer)
            if native.get("artifact_kind") == "raw_shape_basis":
                raise ValueError("A RAW shape basis requires its explicit external authority; no implicit native-grid admission")
        certificate = native["renderer"]
        if (certificate["raw_acquisition"]["sha256"] != reference_image["sha256"]
                or certificate["camera_u_wh"] != [width, height]
                or certificate["native_shape_hw"] != native["shape_hw"]):
            raise ValueError("The native renderer has another baseline or grid")
        to_u = matrix(native["H_rig_corrected_native_to_baseline_camera_u"])
        if not np.array_equal(to_u, matrix(certificate["native_to_camera_u_h"])):
            raise ValueError("The native grid matrix differs from its production certificate")
        result["baseline_native_grid"] = {
            "renderer": deepcopy(baseline_native_renderer),
            "shape_hw": list(native["shape_hw"]),
            "H_rig_corrected_native_to_baseline_camera_u": to_u.tolist(),
            "H_baseline_camera_u_to_rig_corrected_native": np.linalg.inv(to_u).tolist(),
            "legacy_dimensions_m": [0.49, 0.9],
            "physical_metric_is_separate": True,
        }
        if checked_basis is not None:
            result["baseline_native_grid"].update(
                raw_shape_basis_authority=deepcopy(raw_shape_basis_authority),
                parent_workspace_revision=deepcopy(raw_shape_basis_workspace),
                parent_grid_compatibility=deepcopy(checked_basis.compatibility))
            checked_basis.revalidate()
    return result


def _same_identity(left, right):
    """Compare the existing run configuration's exact original file binding."""
    from .raw_shape_basis_reader import _same
    return _same({key: left.get(key) for key in ("path", "sha256", "size_bytes")},
                 {key: right.get(key) for key in ("path", "sha256", "size_bytes")})


def label_placement(reference, target):
    """Map reference-U pixels to target-U pixels, with the existing order."""
    return (matrix(target["H_metric_mm_to_camera_u"])
            @ matrix(reference["H_camera_u_to_metric_mm"]))


def frame_matrices(model, source_to_reference_u, source_to_frame_u, *, fixed_area_mm2):
    """Compose existing source registrations with the measured reference model.

    ``source`` is the run configuration's existing common registration frame,
    not the saved physical raster's array index. Its origin is kept explicit
    by the existing frame geometry. The worker's reference-to-metric slot is
    in metric pixels; its configured pixel area cancels that unit conversion.
    The mass reference matrix is in metres and is accepted as DATA already.
    """
    fixed = float(fixed_area_mm2)
    if not math.isfinite(fixed) or fixed <= 0:
        raise ValueError("The metric pixel area must be finite and positive")
    source_to_mm = (matrix(model["H_camera_u_to_metric_mm"])
                    @ matrix(source_to_reference_u))
    mm_to_metric_pixel = np.diag([1.0 / math.sqrt(fixed),
                                 -1.0 / math.sqrt(fixed), 1.0])
    source_to_metric_pixel = mm_to_metric_pixel @ source_to_mm
    frame_to_mm = source_to_mm @ np.linalg.inv(matrix(source_to_frame_u))
    source_to_metre = np.diag([1.0e-3, 1.0e-3, 1.0]) @ source_to_mm
    return {
        "H_source_to_metric_pixel": source_to_metric_pixel.tolist(),
        "H_source_to_metric_m": source_to_metre.tolist(),
        "H_frame_native_to_metric_mm": frame_to_mm.tolist(),
        "H_metric_mm_to_frame_native": np.linalg.inv(frame_to_mm).tolist(),
        "fixed_area_mm2": fixed,
    }


def geometry_row(original, reference_to_metric):
    """Change only the existing metric-matrix slot in a copied geometry row.

    Identity mode therefore leaves today's row content exactly unchanged.
    Provenance and optical scaling belong to the separate bound build receipt
    and the run's phase_roles.area, not an invented worker input field.
    """
    result = deepcopy(original)
    result["area_parameters"]["H_ac28_to_metric"] = matrix(reference_to_metric).tolist()
    return result


def frame_area(row, *, metric_matrix, area_factor, geometry_parameters):
    """Use the unchanged worker's double-inversion and float32 scaling recipe."""
    from .geometry.context import rig_plane_area_map
    from .geometry.areas import apply_selected_area_scale

    factor = float(area_factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("The selected area factor must be finite and positive")
    area = row["area_parameters"]
    frame_to_metric = matrix(metric_matrix) @ np.linalg.inv(
        matrix(area["H_ac28_to_frame_native"]))
    original = rig_plane_area_map(
        frame_to_metric, tuple(row["phase_shape_hw"]),
        origin_xy=tuple(row["native_origin_xy"]),
        metric_pixel_area_mm2=float(area["fixed_area_mm2"]),
        parameters=geometry_parameters,
    )
    return apply_selected_area_scale(original, factor)


def build_reference_index(plan: Mapping, *, references: Mapping, profiles: Mapping,
                          output: Path, observations: Mapping | None = None,
                          allow_incomplete=False, raw_shape_basis_authorities=None,
                          raw_shape_basis_workspaces=None):
    """Write real per-run reference files and an early, explicitly partial index.

    ``plan`` binds the saved joint models, the twelve saved replacement-bottom
    models and the current 39-run edge ledger. No solver is called. Every
    reference is checked against the plan's already bound actual acquisition.
    The output directory must be new; incomplete work is preserved on error.
    """
    if raw_shape_basis_authorities is not None or raw_shape_basis_workspaces is not None:
        return _build_basis_reference_index(plan, references=references, profiles=profiles,
            output=output, observations=observations, authorities=raw_shape_basis_authorities,
            workspaces=raw_shape_basis_workspaces)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pins = plan["source_inputs"]
    joint = read_bound_json(pins["actual_joint_result"])
    joint_by_run = {row["run"]: row for row in joint["runs"]}
    order = list(plan["current_analysis_order"])
    if len(order) != 39 or len(set(order)) != 39 or set(references) != set(order):
        raise ValueError("Exactly the current 39 registered reference runs are required")
    planned = {row["run"]: row for row in plan["runs"]}
    rows = {}
    pending = {}
    for run in order:
        acquisition = planned[run]["RAW"]
        actual_reference = descriptor(references[run]["path"])
        observation = None if observations is None else observations.get(run)
        different = any(actual_reference[key] != acquisition[key] for key in ("sha256", "size_bytes"))
        if observations is not None and observation is None and allow_incomplete:
            pending[run] = "Native/baseline registration is not complete; no model substituted"
            continue
        if different and observation is None:
            raise ValueError(f"{run}: saved model and registered reference image differ")
        source = pins.get("q108_model_" + run, pins["actual_joint_result"])
        if "q108_model_" + run in pins:
            saved = read_bound_json(source)
            if saved["run"] != run or saved["new_fit_or_optimizer_calls"] != 0:
                raise ValueError("Saved replacement-bottom model has another run or role")
            frame = saved["conditional_frame"]
        else:
            frame = joint_by_run[run]["conditional_frame"]
        value = reference_model(run, frame, reference_image=actual_reference,
                                camera_profile=profiles[run], source_model=source,
                                edge_review=pins["current_confirmed39_ledger"],
                                baseline_configuration=references[run]["configuration"],
                                model_photograph=acquisition,
                                baseline_to_model=None if observation is None else observation["bridge"],
                                baseline_native_renderer=None if observation is None else observation["native_renderer"])
        bound = write_new_json(output / run / "reference_model.json", value)
        rows[run] = {"run": run, "reference_model": bound}
    index = {
        "format": FORMAT, "state": ("REFERENCE_MODELS_COMPLETE_FRAME_ROWS_PENDING" if not pending
                                     else "PARTIAL_REFERENCE_MODELS_REGISTRATION_PENDING"),
        "complete": False, "reference_models_complete": not pending,
        "frame_rows_complete": False, "area_arrays_complete": False,
        "run_order": order, "runs": rows, "run_count": len(rows),
        "metric_unit": "mm", "grid": "camera_u", "adopted": False,
        "source_joint": pins["actual_joint_result"],
        "source_current_edges": pins["current_confirmed39_ledger"],
        "pending_runs": pending,
    }
    return write_new_json(output / "index.json", index)


def _build_basis_reference_index(plan, *, references, profiles, output, observations, authorities, workspaces):
    """Publish only NEW preflighted basis models; never rewrite an old index."""
    from .raw_shape_basis_reader import read_model_basis
    order = list(plan["current_analysis_order"])
    if (not order or len(order) != len(set(order)) or set(references) != set(order)
            or not isinstance(authorities, Mapping) or set(authorities) != set(order)
            or not isinstance(workspaces, Mapping) or set(workspaces) != set(order)
            or not isinstance(observations, Mapping) or set(observations) != set(order)):
        raise ValueError("Every requested NEW basis model needs its exact reference, authority, workspace and observation")
    pins = plan["source_inputs"]
    joint = read_bound_json(pins["actual_joint_result"])
    saved_by_run = {row["run"]: row for row in joint["runs"]}
    planned = {row["run"]: row for row in plan["runs"]}
    values, checks = {}, {}
    for run in order:
        acquisition = planned[run]["RAW"]
        reference = descriptor(references[run]["path"])
        observation = observations[run]
        source = pins.get("q108_model_" + run, pins["actual_joint_result"])
        if "q108_model_" + run in pins:
            saved = read_bound_json(source)
            if saved["run"] != run or saved["new_fit_or_optimizer_calls"] != 0:
                raise ValueError("Saved replacement-bottom model has another run or role")
            frame = saved["conditional_frame"]
        else:
            frame = saved_by_run[run]["conditional_frame"]
        value = reference_model(run, frame, reference_image=reference, camera_profile=profiles[run],
            source_model=source, edge_review=pins["current_confirmed39_ledger"],
            baseline_configuration=references[run]["configuration"], model_photograph=acquisition,
            baseline_to_model=observation.get("bridge"), baseline_native_renderer=observation["native_renderer"],
            raw_shape_basis_authority=authorities[run], raw_shape_basis_workspace=workspaces[run])
        checked = read_model_basis(value)
        for row in (source, pins["current_confirmed39_ledger"]):
            checked.guards.add(row, metadata=True, copyable=True)
        if observation.get("bridge") is not None:
            checked.guards.add(observation["bridge"], metadata=True, copyable=True)
        checked.guards.add({key: acquisition[key] for key in ("path", "sha256", "size_bytes")})
        values[run], checks[run] = value, checked
    for checked in checks.values():
        checked.revalidate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = {}
    for run in order:
        checks[run].revalidate()
        bound = write_new_json(output / run / "reference_model.json", values[run])
        rows[run] = dict(run=run, reference_model=bound)
    for checked in checks.values():
        checked.revalidate()
    index = dict(format=FORMAT, state="REFERENCE_MODELS_COMPLETE_FRAME_ROWS_PENDING", complete=False,
        reference_models_complete=True, frame_rows_complete=False, area_arrays_complete=False,
        run_order=order, runs=rows, run_count=len(rows), metric_unit="mm", grid="camera_u", adopted=False,
        source_joint=pins["actual_joint_result"], source_current_edges=pins["current_confirmed39_ledger"], pending_runs={},
        raw_shape_basis_authorities=deepcopy(authorities))
    return write_new_json(output / "index.json", index)
