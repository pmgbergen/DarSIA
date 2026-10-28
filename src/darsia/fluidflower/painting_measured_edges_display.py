"""Read a project-scoped display companion and verified saved edge metadata."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

from . import painting_measured_edges as edges


FORMAT = "darsia-layer-painting-display-v1"
RELATIVE_PATH = Path("config") / "layer_painting_display.json"
MAX_CONFIG_BYTES = 64 * 1024


@dataclass(frozen=True)
class MeasuredEdgeDisplay:
    edges: edges.SavedMeasuredEdges
    configuration: dict
    configuration_descriptor: dict
    photograph_version: tuple


def _version(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _read_configuration(project_root, project_id, series_root):
    path = Path(project_root) / RELATIVE_PATH
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_CONFIG_BYTES + 1)
    except FileNotFoundError as error:
        raise ValueError("No measured-edge display metadata is saved for this project.") from error
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("Measured-edge display metadata exceeds the bounded size.")
    try:
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=edges._unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Measured-edge display metadata is not valid JSON.") from error
    if not isinstance(value, Mapping) or value.get("format") != FORMAT:
        raise ValueError("Unrecognized measured-edge display metadata.")
    if (not isinstance(project_id, str) or not project_id
            or value.get("project_id") != project_id):
        raise ValueError("Measured-edge display metadata belongs to another project.")
    saved_root = edges._text(value.get("series_root"), "Display native series root")
    actual_root = edges._text(str(series_root or ""), "Current native series root")
    if edges._path_key(saved_root) != edges._path_key(actual_root):
        raise ValueError("Measured-edge display metadata belongs to another native series root.")
    settings = edges._object(value.get("measured_edges"), "Measured-edge display settings")
    edges._descriptor(settings.get("geometry_index"), "Display geometry index")
    edges._text(settings.get("expected_confirmation"), "Expected confirmation")
    descriptor = dict(path=str(path.absolute()), sha256=hashlib.sha256(raw).hexdigest(),
                      size_bytes=len(raw))
    return deepcopy(dict(value)), descriptor, path.parent


def _profile_assignment(document, run, profile_id, shape, label):
    """Verify the grid and exact run assignment of one SHA-checked document."""
    registration = edges._object(document.get("registration", document), label + " registration")
    common = edges._object(registration.get("common"), label + " registration settings")
    sensor = edges._object(common.get("sensor"), label + " sensor settings")
    declared = edges._object(common.get("profiles"), label + " declared profiles")
    if (profile_id not in declared
            or edges._shape(sensor.get("camera_wh"), label + " camera-U size") != shape[::-1]):
        raise ValueError(label + " does not declare this profile and camera-U grid.")
    source = edges._object(document.get("source"), label + " source")
    assignments = edges._object(source.get("profiles"), label + " run assignments")
    normalized = {}
    for key, value in assignments.items():
        identity = edges._run(key)
        if identity in normalized:
            raise ValueError(label + " has a duplicate run assignment.")
        normalized[identity] = value
    if normalized.get(run) != profile_id:
        raise ValueError(label + " assigns another profile to this run.")


def _read_profile(binding, *, base, run, profile_id, shape, label, cache):
    try:
        document, actual, _ = edges._read_document(binding, base=base, label=label, cache=cache)
    except OSError as error:
        raise ValueError(label + " cannot be verified: " + str(error)) from error
    _profile_assignment(document, run, profile_id, shape, label)
    return actual


def load_measured_edge_display(project_root, *, project_id, series_root, run,
                               photograph_path, shape_hw, camera_u_binding,
                               photograph_stamp, geometry_index_path=""):
    """Read and verify in a metadata worker; never decode or write a photograph.

    The photo hash is checked as raw bytes, with its displayed-session stat
    stamp and before/after version. The current view supplies its actual
    camera-U profile identity; a model's own profile is not treated as proof
    that the view uses it. Missing or mismatched companion data fails closed.
    """
    configuration, config_ref, config_folder = _read_configuration(
        project_root, project_id, series_root)
    settings = configuration["measured_edges"]
    cache = {}
    index, index_ref, index_folder = edges._read_document(settings["geometry_index"],
        base=config_folder, label="Display geometry index", cache=cache)
    if geometry_index_path and edges._path_key(geometry_index_path) != edges._path_key(index_ref["path"]):
        raise ValueError("The chosen frame models differ from the display metadata's pinned geometry index.")
    normalized_run = edges._run(run)
    entry = edges._selected_run(index.get("runs"), normalized_run, "Display geometry index", mapping=True)
    model, _model_ref, model_folder = edges._read_document(entry.get("reference_model"),
        base=index_folder, label="Display reference model", cache=cache)
    if (index.get("format") != edges.GEOMETRY_FORMAT or index.get("grid") != "camera_u"
            or model.get("format") != edges.GEOMETRY_FORMAT or model.get("grid") != "camera_u"):
        raise ValueError("Measured-edge display metadata must bind the camera-U grid.")
    edges._run_fields(model, normalized_run, "Display reference model", required=True)
    shape = edges._shape(shape_hw, "Displayed photograph shape")
    if edges._shape(model.get("shape_hw"), "Display model shape") != shape:
        raise ValueError("The displayed photograph and measured model have different grids.")
    photograph = edges._descriptor(model.get("reference_image"), "Measured photograph")
    if edges._path_key(photograph["path"]) != edges._path_key(photograph_path):
        raise ValueError("The measured model belongs to another displayed photograph.")
    camera = edges._object(camera_u_binding, "Displayed camera-U profile")
    saved_camera = edges._object(model.get("camera_profile"), "Measured camera-U profile")
    for key in ("sha256", "profile_id"):
        if camera.get(key) != saved_camera.get(key):
            raise ValueError("The displayed camera-U profile differs from the measured model.")
    current_path = edges._text(camera.get("path"), "Displayed camera-U path")
    if not Path(current_path).is_absolute():
        raise ValueError("The displayed camera-U profile needs its actual absolute parameter path.")
    if edges._shape(camera.get("camera_wh"), "Displayed camera-U size") != shape[::-1]:
        raise ValueError("The displayed camera-U profile has another grid.")
    if "size_bytes" in camera and camera["size_bytes"] != saved_camera.get("size_bytes"):
        raise ValueError("The displayed camera-U profile has another byte size.")
    # Complete the missing runtime size from the pinned model, then actually
    # hash BOTH locators and validate their grid/profile/run assignments.
    # No host-name equivalence or path-only alias is admitted.
    complete_camera = dict(camera, size_bytes=saved_camera.get("size_bytes"))
    profile_id = edges._text(camera.get("profile_id"), "Displayed camera-U profile id")
    current_profile = _read_profile(complete_camera, base=config_folder, run=normalized_run,
        profile_id=profile_id, shape=shape, label="Displayed camera-U profile", cache=cache)
    measured_profile = _read_profile(saved_camera, base=model_folder, run=normalized_run,
        profile_id=profile_id, shape=shape, label="Measured camera-U profile", cache=cache)
    if any(current_profile[key] != measured_profile[key] for key in ("sha256", "size_bytes")):
        raise ValueError("The displayed and measured camera-U parameter bytes differ.")
    path = Path(photograph_path)
    before = path.stat()
    stamp = f"{before.st_size}:{before.st_mtime_ns}"
    if not isinstance(photograph_stamp, str) or not photograph_stamp or photograph_stamp != stamp:
        raise ValueError("The displayed-session photograph stamp changed or is missing.")
    if before.st_size != photograph["size_bytes"]:
        raise ValueError("The measured photograph changed byte size.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if _version(path.stat()) != _version(before) or digest.hexdigest() != photograph["sha256"]:
        raise ValueError("The measured photograph changed during display verification.")
    saved = edges.read_saved_measured_edges(index_ref, normalized_run,
        photograph_binding=photograph, camera_u_binding=deepcopy(dict(saved_camera)), shape_hw=shape,
        expected_confirmation=settings["expected_confirmation"])
    # The strict reader keeps its original saved-path contract. This display
    # admission separately proves the current locator has those exact bytes.
    for binding, label in ((current_profile, "Displayed camera-U profile"),
                           (measured_profile, "Measured camera-U profile")):
        _read_profile(binding, base=Path.cwd(), run=normalized_run, profile_id=profile_id,
                      shape=shape, label=label, cache={})
    if _version(path.stat()) != _version(before):
        raise ValueError("The measured photograph changed while its edges were loading.")
    _fresh, fresh_ref, _ = _read_configuration(project_root, project_id, series_root)
    if fresh_ref != config_ref:
        raise ValueError("Measured-edge display metadata changed while its edges were loading.")
    evidence = dict(format="darsia-camera-u-profile-locator-identity-v1",
        configured=deepcopy(current_profile), measured=deepcopy(measured_profile),
        run_id=normalized_run, profile_id=profile_id, camera_wh=list(shape[::-1]),
        locators_differ=edges._path_key(current_profile["path"]) != edges._path_key(measured_profile["path"]),
        exact_sha256_and_size=True, both_run_assignments_and_grids_checked=True,
        configuration_changed=False)
    saved = replace(saved, provenance=dict(saved.provenance, camera_profile_locator_binding=evidence))
    return MeasuredEdgeDisplay(saved, configuration, config_ref, _version(before))
