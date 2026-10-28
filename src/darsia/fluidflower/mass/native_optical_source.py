"""Read measured physical archives from an exact completed RAW Setup Build.

This prefix admits saved files and their original grid. It does not construct
FrameData, verify phase/source registration, or qualify a training context.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .n1_pipeline import DataReader
from ..rig_raw_build import CERTIFICATE_NAME, FORMAT, REQUEST_FORMAT, _grid
from ..rig_raw_write import _canonical, _identity, _no_links


def _same(first, second):
    """Compare literal finite JSON DATA, retaining numeric types."""
    encode = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
    return encode(first) == encode(second)


def _bound(value):
    """Require a finite exact producer file descriptor before reading it."""
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or ".." in Path(value["path"]).parts
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value["size_bytes"]) is not int or value["size_bytes"] < 0):
        raise ValueError("An exact physical-source file descriptor is required")
    _no_links(Path(value["path"]))
    return value


def _grid_identity(image):
    """Retain the original spatial axes, origin, dimensions and pixel sizes."""
    if image.space_dim != 2 or image.indexing != "ij" or image.series:
        raise ValueError("Physical archives require an original single ij image")
    result = {key: np.asarray(getattr(image, key)).tolist()
              for key in ("num_voxels", "dimensions", "origin", "voxel_size")}
    if (any(len(result[key]) != 2 or not np.isfinite(result[key]).all() for key in result)
            or min(result["dimensions"]) <= 0 or min(result["voxel_size"]) <= 0):
        raise ValueError("Physical archives require finite positive measured grids")
    return result


@dataclass(frozen=True)
class BuildPhysicalContext:
    """Original saved buffers and source joins, without whole-context eligibility."""
    run: str
    baseline_rgb: np.ndarray
    labels: np.ndarray
    facies: np.ndarray
    depth_m: np.ndarray
    porosity: np.ndarray
    grid: dict
    provenance: dict


def load_build_physical_context(operation_result_binding, *, reader=None, local_reuse=None):
    """Read one bound completed Build and its exact owned physical archives.

    The original Setup plan supplies the adopted revision and native parameters.
    Phase configuration, clear-water, per-frame registration and physical-domain
    eligibility remain the later frame collector's responsibility.
    An explicit local reuse record replaces only the obsolete live manifest
    read; the original completed Build and every numerical source stay exact.
    """
    import darsia
    reader = DataReader() if reader is None else reader
    consumed = []

    def document(bound):
        """Consume an exact JSON identity through the original cached reader."""
        consumed.append(deepcopy(_bound(bound)))
        return reader.json(bound)

    result = document(operation_result_binding)
    if (result.get("format") != "darsia-series-setup-result-v1"
            or result.get("pipeline") != "series-setup" or result.get("operation") != "build"
            or result.get("complete") is not True or result.get("worker_references_closed") is not True):
        raise ValueError("An exact completed Setup Build result is required")
    run = result["run"]
    plan = document(result["setup_plan"])
    builds = [row for row in plan["build_plans"] if row["run_id"] == run]
    if plan.get("format") != "darsia-series-setup-plan-v1" or len(builds) != 1:
        raise ValueError("The completed Build differs from its original Setup plan")
    build = builds[0]
    request = document(result["raw_build_request"])
    template = {key: value for key, value in request.items() if key != "configuration_files"}
    if (request.get("format") != REQUEST_FORMAT or request.get("run_id") != run
            or not _same(template, build["request_template"])
            or not _same(request["configuration_files"], result["configuration_files"])
            or [row["path"] for row in request["configuration_files"]] != build["configuration_paths"]
            or not _same(request["series_context"]["manifest"], plan["series_manifest"])
            or not _same(request["profile"], build["native_parameters"])):
        raise ValueError("The RAW request differs from its completed Build and plan")
    adopted = document(plan["adopted_revision"])
    document(build["native_parameters"])
    if adopted.get("revision_id") is None:
        raise ValueError("The Setup plan lacks its actual adopted revision")
    reuse = None
    manifest_inputs = [request["series_context"]["manifest"]]
    if local_reuse is not None:
        from ..local_build_reuse import verify_local_build_reuse
        reuse = verify_local_build_reuse(local_reuse,
            operation_result=operation_result_binding, request=request, run=run, reader=reader)
        consumed.extend(deepcopy(reuse["inputs"]))
        manifest_inputs = []
    from ..protocol_content import current_source
    # A corrected injection position keeps the Build (protocol_content); the
    # protocol file's current bytes are then what is consumed.
    for bound in [request["baseline"], *manifest_inputs,
                  request["shape_basis_certificate"], *request["configuration_files"],
                  *(current_source(row, check=reader.path) for row in request["protocol_inputs"])]:
        consumed.append(deepcopy(_bound(bound)))
        reader.path(bound)
    certificate = document(result["baseline_renderer"])
    destination = _canonical(request["output"]["destination"])
    root = _canonical(request["output"]["intended_root"])
    if (destination == root or not destination.is_relative_to(root)
            or Path(result["baseline_renderer"]["path"]).resolve() != destination / CERTIFICATE_NAME
            or certificate.get("format") != FORMAT or certificate.get("artifact_kind") != "raw_rig_baseline"
            or certificate.get("run_id") != run or certificate.get("complete_rig_model") is not False
            or certificate.get("production_qualification_claimed") is not False
            or not _same(certificate["inputs"]["operation_request"], result["raw_build_request"])
            or not _same(certificate["inputs"]["configuration_files"], request["configuration_files"])
            or not _same(certificate["inputs"]["series_context"], request["series_context"])
            or not _same(certificate["inputs"]["raw"], request["baseline"])
            or not _same(certificate["inputs"]["profile"], request["profile"])
            or not _same(certificate["inputs"]["protocol_inputs"], request["protocol_inputs"])
            or not _same(certificate["inputs"]["operation_overrides"], request["operation_overrides"])
            or not _same(certificate["outputs"], result["produced_files"])):
        raise ValueError("The saved physical sources differ from the original full RAW Build")
    camera = certificate["inputs"].get("camera_binding")
    if (not isinstance(camera, dict)
            or not _same(camera, build["raw_setup_settings"]["raw_camera"])):
        raise ValueError("The RAW camera binding differs from its saved Build settings")
    outputs = {}
    for bound in certificate["outputs"]:
        path = reader.path(_bound(bound)).resolve()
        if path == destination or not path.is_relative_to(destination) or path in outputs:
            raise ValueError("The Build contains duplicate or unowned saved outputs")
        outputs[path] = bound
        consumed.append(deepcopy(bound))
    ownership = certificate["output_ownership"]
    if (not _same(ownership["request"], result["raw_build_request"])
            or not _same(ownership["authority"], request["output"])
            or ownership["directory_identity"] != list(_identity(destination))
            or ownership["intended_root_identity"] != list(_identity(root))):
        raise ValueError("The original RAW output ownership changed")
    operation_id = ownership["operation_id"]
    if certificate["producer"]["operation_id"] != operation_id:
        raise ValueError("The RAW producer differs from its original output owner")
    for role, filename in (("owner", "raw_build_owner.json"),
                           ("setup_started", "raw_setup_started.json"),
                           ("save_started", "raw_save_started.json"),
                           ("save_completed", "raw_save_completed.json")):
        bound = ownership[role]
        if not _same(outputs.get(destination / filename), bound):
            raise ValueError("A RAW output ownership record differs from its saved output")
        record = document(bound)
        if record.get("operation_id") != operation_id:
            raise ValueError("A RAW output ownership record belongs to another operation")
        if role in {"owner", "save_completed"} and (
                not _same(record.get("request"), result["raw_build_request"])
                or Path(record["destination"]).resolve() != destination
                or record["directory_identity"] != ownership["directory_identity"]):
            raise ValueError("The original RAW ownership/save identity differs")
        if role == "owner" and (record.get("format") != "darsia-owned-raw-output-v1"
                or Path(record["intended_root"]).resolve() != root):
            raise ValueError("The original RAW output owner differs from its intended root")
        if role == "save_completed" and (record.get("format") != "darsia-owned-raw-save-completed-v1"
                or record.get("complete") is not True
                or record.get("no_clobber_save") is not True):
            raise ValueError("The original RAW archive save is not complete")
    images = {}
    for key in ("shape_corrected_baseline", "labels", "facies", "depth", "porosity"):
        bound = outputs.get(destination / (key + ".npz"))
        if bound is None:
            raise ValueError("The full Build lacks the saved physical archive: " + key)
        images[key] = darsia.imread(reader.path(bound))
        reader.path(bound)
    baseline = images.pop("shape_corrected_baseline")
    grid = _grid_identity(baseline)
    if not _same(_grid(baseline), certificate["corrected_baseline"]["grid"]):
        raise ValueError("The saved baseline grid differs from its original renderer")
    if not _same(outputs[destination / "shape_corrected_baseline.npz"],
                 certificate["corrected_baseline"]["archive"]):
        raise ValueError("The saved baseline archive differs from its original renderer")
    for key, image in images.items():
        if (not image.scalar or image.img.ndim != 2
                or not _same(_grid_identity(image), grid)):
            raise ValueError("The saved physical archive uses another original grid: " + key)
        categorical = key in {"labels", "facies"}
        if (categorical and (not np.issubdtype(image.img.dtype, np.integer) or np.any(image.img < 0))
                or not categorical and not np.issubdtype(image.img.dtype, np.floating)):
            raise ValueError("The saved physical archive has another field type: " + key)
    for bound in consumed:
        reader.path(bound)
    arrays = dict(baseline_rgb=baseline.img, labels=images["labels"].img,
                  facies=images["facies"].img, depth_m=images["depth"].img,
                  porosity=images["porosity"].img)
    for value in arrays.values():
        value.setflags(write=False)
    context = BuildPhysicalContext(run=run, **arrays, grid=grid, provenance=deepcopy(dict(
        operation_result=operation_result_binding, setup_plan=result["setup_plan"],
        adopted_revision=plan["adopted_revision"], native_configuration=build["native_parameters"],
        raw_build_request=result["raw_build_request"], baseline_renderer=result["baseline_renderer"],
        camera_binding=camera, raw_acquisition=request["baseline"],
        physical_archives={key: outputs[destination / (key + ".npz")]
                           for key in ("shape_corrected_baseline", "labels", "facies", "depth", "porosity")})))
    if "configuration" in adopted and "analysis_exclusions" in adopted["configuration"]:
        from ..painted_analysis_exclusions import configuration_exclusion, exclusion_provenance, exclusion_sources
        from ..painted_target_integration import target_sources
        configuration_exclusion(adopted["configuration"], run, labels=context.labels)
        provenance = exclusion_provenance(adopted["configuration"], run)
        if provenance is not None:
            context.provenance["approved_analysis_gap"] = provenance
            bound = adopted["configuration"]["painted_adoption_table"]["preparation"]
            preparation = reader.json(bound)
            for dependency in [bound, *exclusion_sources(preparation), *target_sources(preparation)]:
                reader.path(dependency)
    if reuse is not None:
        # Record the current local decision separately; never rewrite ancestry.
        for bound in reuse["inputs"]:
            reader.path(bound)
        context.provenance["local_build_reuse"] = deepcopy(reuse["provenance"])
    return context
