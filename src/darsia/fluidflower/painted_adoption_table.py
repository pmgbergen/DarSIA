"""Explicit approved-painting facies derivation, separate from old CSV authority.

Only the positive ``painted_adoption_table`` contract calls this module. Old
canonical edits keep their original table and physical-property guards. A Check
builds a recipe in memory; the Store materializes its exact CSV on explicit Save.
"""
from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from copy import deepcopy
from dataclasses import asdict
from collections import OrderedDict
import os
from threading import RLock
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from .adoption_facies_table import build_adoption_facies_table, MAX_PARENT_BYTES
from .bound_verification_cache import BoundVerificationCache
from .pipeline.job_queue import retry_transient

FORMAT = "darsia-painted-adoption-facies-table-v1"
VERSION_FORMAT = "darsia-painted-adoption-approved-version-v2"
FRAME_BASIS_FORMAT = "darsia-painted-adoption-preparation-frame-v1"
KEY = "painted_adoption_table"
PREPARATION_KEY = "painting_facies"
OUTSIDE_FORMAT = "darsia-painted-adoption-outside-frame-proof-v1"
OUTSIDE_FORMAT_V2 = "darsia-painted-adoption-outside-frame-proof-v2"
FIXTURE_DERIVATION_FORMAT = "darsia-painted-fixture-analysis-derivation-v1"
_LIMIT = 16 * 1024 * 1024
_FILES = BoundVerificationCache()
_DERIVED = OrderedDict()
_DERIVED_LOCK = RLock()
_DERIVED_PID = os.getpid()
_DERIVED_MAX_BYTES = 16 * 1024 * 1024


def _finite(value):
    """Recognize finite explicit numeric fields without coercing booleans."""
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def _json(raw):
    """Read duplicate-free finite JSON metadata through the original strict rules."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate painted-adoption metadata key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite painted-adoption metadata")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Unreadable painted-adoption metadata") from error


def _same(left, right):
    """Compare serialized identities without conflating numeric and boolean types."""
    return json.dumps(left, sort_keys=True, allow_nan=False, ensure_ascii=False) == json.dumps(
        right, sort_keys=True, allow_nan=False, ensure_ascii=False)


def _read(path, limit=_LIMIT):
    """Retry one bounded metadata read, retaining no raster or image arrays."""
    return _FILES.read(path, limit=limit)


def _bound(path):
    """Capture an actual descriptor without substituting an expected identity."""
    return _FILES.descriptor(Path(path))


def _checked(binding, *, metadata=False):
    """Check frozen bytes via the existing validator and process-local cache."""
    from .geometry_raster_sources import _verify
    actual = _verify(binding, descriptor_reader=lambda path: _FILES.descriptor(path, expected=binding))
    if metadata:
        raw = _read(actual["path"])
        if hashlib.sha256(raw).hexdigest() != actual["sha256"] or len(raw) != actual["size_bytes"]:
            raise ValueError("Painted-adoption metadata changed while reading")
        return _json(raw)
    return actual


def _local(folder, row):
    """Resolve a preparation-relative source and verify its original bound bytes."""
    name = row.get("path") if isinstance(row, dict) else None
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("A prepared table source needs its relative local path")
    root = Path(folder).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("A prepared table source leaves its preparation")
    return _checked(dict(row, path=str(path)))


def _frame(payload):
    """Validate the full explicit preparation frame without inventing corners."""
    from .series_geometry import RigFrame, CROP_CORNERS
    keys = {"corners", "width_m", "height_m", "run_id", "source", "image", "image_size"}
    optional = {"disabled_corners", "reference_geometry"}
    if (not isinstance(payload, dict) or not keys <= set(payload)
            or not set(payload) <= keys | optional):
        raise ValueError("Painted adoption needs its full explicit preparation frame")
    corners = payload["corners"]
    if not isinstance(corners, dict) or set(corners) != set(CROP_CORNERS):
        raise ValueError("Painted adoption needs every explicit preparation frame corner")
    for point in corners.values():
        if (not isinstance(point, list) or len(point) != 2
                or any(not _finite(v) for v in point)):
            raise ValueError("Invalid painted-adoption frame corner")
    for key in ("width_m", "height_m"):
        value = payload[key]
        if not _finite(value) or value <= 0:
            raise ValueError("Painted adoption needs the explicit positive frame size")
    size = payload["image_size"]
    if (not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
            or any(not isinstance(payload[key], str) for key in ("run_id", "source", "image"))):
        raise ValueError("Invalid painted-adoption frame identity")
    if not payload["run_id"] or not Path(payload["image"]).is_absolute():
        raise ValueError("Painted adoption needs the frozen frame's actual run and absolute photograph identity")
    disabled = payload.get("disabled_corners", [])
    if (not isinstance(disabled, list)
            or any(not isinstance(name, str) or name not in CROP_CORNERS for name in disabled)
            or len(set(disabled)) != len(disabled)):
        raise ValueError("Invalid unobserved painted-adoption frame corners")
    reference = payload.get("reference_geometry")
    if "reference_geometry" in payload:
        def finite_json(value):
            if value is None or type(value) in {str, bool} or _finite(value):
                return True
            if isinstance(value, list):
                return all(finite_json(item) for item in value)
            if isinstance(value, dict):
                return all(isinstance(key, str) and finite_json(item) for key, item in value.items())
            return False
        if not isinstance(reference, dict) or not finite_json(reference):
            raise ValueError("Invalid painted-adoption reference geometry record")
    frame = RigFrame(corners={key: tuple(value) for key, value in corners.items()},
        width_m=payload["width_m"], height_m=payload["height_m"], run_id=payload["run_id"],
        source=payload["source"], image=payload["image"], image_size=tuple(size),
        disabled_corners=tuple(disabled), reference_geometry=deepcopy(reference))
    if not frame.complete():
        raise ValueError("Unobserved preparation frame corners need a measured reference model")
    return frame


def _history(folder, approval):
    """Require the literal approval entry in the retained actual history."""
    from .layer_painting import LAYER_APPROVAL_FORMAT
    raw = _read(Path(folder).parent / "approvals.jsonl")
    found = False
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = _json(line)
        if not isinstance(row, dict) or row.get("format") != LAYER_APPROVAL_FORMAT:
            raise ValueError("Unreadable painted-map approval history")
        found |= _same(row, approval)
    if not found:
        raise ValueError("The saved painting approval is absent from its actual history")


def verify_version(binding):
    """Verify immutable files, actual approval, map SHA and exact saved facies."""
    from . import layer_painting as lp
    from . import layer_painting_manual as manual
    from . import layer_painting_polygons_only as polygons

    required = {"format", "record", "version", "files", "approval", "frame", "frame_basis", "facies_snapshot"}
    if not isinstance(binding, dict) or set(binding) != required or binding["format"] != VERSION_FORMAT:
        raise ValueError("Explicit approved-painting version closure required")
    actual = _checked(binding["version"], metadata=True)
    if not _same(actual, binding["record"]):
        raise ValueError("The immutable saved painting version changed")
    if (not isinstance(actual, dict) or actual.get("format") != lp.LAYER_VERSION_FORMAT or type(actual.get("schema")) is not int
            or actual["schema"] != 1 or actual.get("kind") not in {"proposal", "edited map", manual.MANUAL_MAP_KIND}
            or not isinstance(actual.get("run_id"), str) or not actual["run_id"]):
        raise ValueError("Unsupported saved painting version")
    folder = Path(binding["version"]["path"]).resolve().parent
    if Path(binding["version"]["path"]).name != "version.json":
        raise ValueError("The saved version descriptor must name its original version.json")
    manual_map = actual.get("kind") == manual.MANUAL_MAP_KIND
    leaves = set(manual.MANUAL_VERSION_FILES if manual_map else lp.VERSION_FILES)
    if (not isinstance(actual.get("files"), dict) or not isinstance(binding["files"], dict)
            or set(actual["files"]) != leaves or set(binding["files"]) != leaves):
        raise ValueError("The saved painting version has an incomplete file closure")
    for leaf in sorted(leaves):
        bound = binding["files"][leaf]
        if (not isinstance(bound, dict) or not isinstance(bound.get("path"), str)
                or Path(bound["path"]).resolve() != folder / leaf
                or bound.get("sha256") != actual["files"][leaf]):
            raise ValueError("A saved painting file differs from its original version")
        _checked(bound)
    approval = binding["approval"]
    expected = dict(format=lp.LAYER_APPROVAL_FORMAT, run_id=actual.get("run_id"),
        version=folder.name, name=actual.get("name"), kind=actual.get("kind"),
        sha256=actual.get("sha256"), files=actual["files"])
    if (not isinstance(approval, dict) or any(not _same(approval.get(key), value)
            for key, value in expected.items()) or not isinstance(approval.get("approved_utc"), str)
            or not approval["approved_utc"]
            or re.fullmatch("[0-9a-f]{64}", str(actual.get("sha256"))) is None):
        raise ValueError("The saved painting approval does not bind its exact version and files")
    _history(folder, approval)
    raw_session = _checked(binding["files"]["session.json"], metadata=True)
    if (not isinstance(raw_session, dict) or raw_session.get("format") != lp.LAYER_PAINTING_FORMAT
            or raw_session.get("run_id") != actual.get("run_id")
            or not isinstance(raw_session.get("facies"), dict)
            or "permeability_unit" not in raw_session):
        raise ValueError("The saved painting facies need their explicit original session")
    if any(not isinstance(row, dict) or set(row) != {"role", "porosity", "permeability"}
           for row in raw_session["facies"].values()):
        raise ValueError("Saved painting FaciesEntry fields must be explicit, without defaults")
    if any(not isinstance(row["role"], str)
            or any(value is not None and not _finite(value)
                   for value in (row["porosity"], row["permeability"]))
           for row in raw_session["facies"].values()):
        raise ValueError("Saved painting physical values must retain their explicit numeric types")
    if not _same(binding["facies_snapshot"], _snapshot(raw_session)):
        raise ValueError("Preparation facies snapshot differs from the immutable approved session")
    frame = _frame(binding["frame"])
    _verify_frame_basis(binding, raw_session)
    record = dict(actual, folder=str(folder))
    if actual.get("proposal_method") == polygons.METHOD:
        session, proposal, corrections = retry_transient(lambda: polygons.load_version(record))
        layers = polygons.painted_layers(session, proposal, frame=frame, corrections=corrections)
    elif manual_map:
        session, saved = retry_transient(lambda: manual.load_manual_version(record))
        if manual.manual_map_digest(session, saved) != actual["sha256"]:
            raise ValueError("Saved manual-map content SHA differs from its approval")
        if saved.uncovered_pixels():
            raise ValueError("The approved manual map still has gaps")
        inside = lp.frame_mask(frame, saved.classes.shape, (1., 1.))
        if not np.array_equal(inside, saved.inside):
            raise ValueError("The saved manual map's inside raster is incompatible with the explicit preparation frame")
        layers = saved.classes.copy()
    else:
        session, proposal, corrections = retry_transient(lambda: lp.load_version(record))
        if (lp.map_digest(session, proposal, corrections) != actual["sha256"]
                or proposal.digest != lp.proposal_digest(session, frame)):
            raise ValueError("Saved proposal content or frame differs from its approval")
        layers = manual.painted_layers(session, proposal, frame=frame, corrections=corrections)
    if (layers.dtype != np.dtype("uint8") or layers.ndim != 2
            or tuple(session.image_size or ()) != tuple(frame.image_size)
            or layers.shape != (frame.image_size[1], frame.image_size[0])
            or (frame.run_id and frame.run_id.casefold() != session.run_id.casefold())
            or (frame.image and Path(frame.image).resolve() != Path(session.image_path).resolve())):
        raise ValueError("The approved painting has another photograph, frame, run or grid")
    inside = lp.frame_mask(frame, layers.shape, (1., 1.))
    if (not np.any(inside) or np.any((layers == 0) & inside)
            or np.any((layers != 0) & ~inside)):
        raise ValueError("The approved map must cover the explicit preparation frame; interior zeros or painted outside pixels are refused")
    # from_dict may supply defaults; the bound fields must survive it exactly.
    if session.to_dict()["facies"] != raw_session["facies"] or session.permeability_unit != raw_session["permeability_unit"]:
        raise ValueError("The saved facies would require a default or coercion")
    _checked(binding["version"])
    for bound in binding["files"].values():
        _checked(bound)
    return session, layers, frame


def _snapshot(payload):
    """Freeze the saved session facies and layer scalars exactly."""
    if (not isinstance(payload, dict) or not isinstance(payload.get("facies"), dict)
            or not isinstance(payload.get("layers"), list)
            or any(not isinstance(name, str) for name in payload["facies"])
            or any(not isinstance(row, dict) or type(row.get("layer_id")) is not int
                   or not 1 <= row["layer_id"] <= 255 or not isinstance(row.get("name"), str)
                   or not isinstance(row.get("facies"), str) or row["facies"] not in payload["facies"]
                   for row in payload["layers"])
            or len({row["layer_id"] for row in payload["layers"]}) != len(payload["layers"])):
        raise ValueError("Saved painting layer/facies identities must be explicit and unambiguous")
    return dict(facies=[[name, deepcopy(row)] for name, row in (payload.get("facies") or {}).items()],
        layers=[{key: row.get(key) for key in ("layer_id", "name", "facies")}
                for row in payload.get("layers") or []], permeability_unit=payload.get("permeability_unit"))


def _verify_frame_basis(binding, raw_session):
    """Join the frame to its exact saved workspace and photograph basis."""
    from . import layer_painting as lp
    from . import layer_painting_polygons_only as polygons
    from .storage_paths import same_storage_path
    basis = binding["frame_basis"]
    keys = {"format", "basis", "historical_frame_bound", "workspace_revision", "photograph",
            "image_stamp", "camera_profile", "source_identity_basis"}
    if (not isinstance(basis, dict) or set(basis) != keys or basis["format"] != FRAME_BASIS_FORMAT
            or basis["basis"] != "explicit-current-frame-at-preparation"
            or type(basis["historical_frame_bound"]) is not bool
            or basis["historical_frame_bound"] != (binding["record"].get("proposal_method") == polygons.METHOD)
            or basis["source_identity_basis"] != "current-file-at-preparation; saved-stat-stamp-matched"):
        raise ValueError("The map approval and explicit preparation-frame basis must remain separate")
    _checked(basis["workspace_revision"], metadata=True)
    photo = _checked(basis["photograph"])
    image = raw_session.get("image") or {}
    if (not same_storage_path(str(image.get("path") or ""), photo["path"])
            or basis["image_stamp"] != image.get("stamp")
            or basis["image_stamp"] != retry_transient(lambda: lp.image_stamp(photo["path"]))):
        raise ValueError("The approved session does not match the current preparation photograph and saved stat stamp")
    profile = basis["camera_profile"]
    if not isinstance(profile, dict) or profile.get("kind") not in {"plain_pixels", "camera_u"}:
        raise ValueError("The preparation photograph needs its explicit grid/profile basis")
    if profile["kind"] == "plain_pixels":
        if set(profile) != {"kind"}:
            raise ValueError("Plain pixels cannot claim a camera-U profile")
    else:
        if (set(profile) != {"kind", "parameters", "profile_id", "camera_wh"}
                or not isinstance(profile["profile_id"], str) or not profile["profile_id"]
                or not _same(profile["camera_wh"], image.get("size"))):
            raise ValueError("The RAW preparation needs its exact configured camera-U profile")
        _checked(profile["parameters"])
    from .image_formats import image_family
    if (image_family(Path(photo["path"])) == "raw") != (profile["kind"] == "camera_u"):
        raise ValueError("RAW needs a bound camera-U profile; plain-image grids do not acquire one")


def capture_version(record, approval, frame, layers, *, project):
    """Capture only an actual approved version whose exact map was supplied."""
    folder = Path(str(record.get("folder") or ""))
    if not folder.is_absolute() or frame is None:
        raise ValueError("An approved painting needs its absolute saved folder and full frame")
    version = _bound(folder / "version.json")
    actual = _checked(version, metadata=True)
    for key in ("run_id", "name", "kind", "sha256", "files", "proposal_method"):
        if not _same(record.get(key), actual.get(key)):
            raise ValueError("The selected painting version changed before preparation")
    files = actual.get("files") or {}
    if any(leaf not in {"session.json", "proposal.npz", "corrections.json", "manual_map.npz"} for leaf in files):
        raise ValueError("Unexpected saved painting file")
    from . import layer_painting as lp
    from . import layer_painting_polygons_only as polygons
    from .geometry_facies import GeometryRevisionStore
    if project is None:
        raise ValueError("The current preparation frame needs its actual workspace revision and source context")
    raw_session = _json(_read(folder / "session.json"))
    saved_session = lp.PaintingSession.from_dict(raw_session)
    current = GeometryRevisionStore(project).current()
    workspace = _source(current)["revision"]
    profile = lp.photo_camera_u(project, saved_session.run_id, saved_session.image_path)
    profile_basis = dict(kind="plain_pixels")
    if profile is not None:
        parameters = _bound(profile.path)
        if parameters["sha256"] != profile.sha256:
            raise ValueError("The current camera-U parameter document changed")
        profile_basis = dict(kind="camera_u", parameters=parameters,
                             profile_id=profile.profile_id, camera_wh=list(profile.camera_wh))
    basis = dict(format=FRAME_BASIS_FORMAT, basis="explicit-current-frame-at-preparation",
        historical_frame_bound=actual.get("proposal_method") == polygons.METHOD,
        workspace_revision=workspace, photograph=_bound(saved_session.image_path),
        image_stamp=lp.image_stamp(saved_session.image_path), camera_profile=profile_basis,
        source_identity_basis="current-file-at-preparation; saved-stat-stamp-matched")
    result = dict(format=VERSION_FORMAT, record=actual, version=version,
        files={leaf: _bound(folder / leaf) for leaf in files}, approval=deepcopy(approval),
        frame=frame.to_dict(), frame_basis=basis, facies_snapshot=_snapshot(raw_session))
    _, expected, _ = verify_version(result)
    if (not isinstance(layers, np.ndarray) or layers.dtype != expected.dtype
            or not np.array_equal(layers, expected)):
        raise ValueError("Preparation source map is not the exact approved saved painting")
    return result



def checked_fixture_analysis(binding, underlay):
    """Replay an explicit fixture overlay without changing the approved carrier.

    The saved map is verified by the ordinary exact version reader first. Only
    the frozen non-ColorChecker polygon underlay may derive the separate
    analysis map; it cannot change the source zero domain or approval identity.
    """
    from .layer_painting_fixtures import checked_for_adoption, _array_sha

    session, carrier, frame = verify_version(binding)
    record = underlay.provenance.get("version")
    folder = Path(binding["version"]["path"]).resolve().parent
    if (not isinstance(record, dict)
            or Path(str(record.get("folder") or "")).resolve() != folder
            or any(not _same(record.get(key), value) for key, value in binding["record"].items())
            or ("approval" in record and not _same(record["approval"], binding["approval"]))):
        raise ValueError("The fixture derivation does not bind the exact saved painting version")
    checked = checked_for_adoption(underlay, underlay.analysis_layers,
                                  dict(record, approval=binding["approval"]), frame=frame)
    analysis = checked.analysis_layers
    roles = {str(row.layer_id): session.facies[row.facies].role for row in session.layers}
    if (checked.provenance.get("approved_layers_sha256") != _array_sha(carrier)
            or checked.provenance.get("roles") != roles
            or analysis.dtype != carrier.dtype or analysis.shape != carrier.shape
            or not np.array_equal(analysis == 0, carrier == 0)):
        raise ValueError("The fixture derivation changed its approved carrier, roles or zero domain")
    provenance = dict(format=FIXTURE_DERIVATION_FORMAT,
        version=deepcopy(binding["version"]), files=deepcopy(binding["files"]),
        approval=deepcopy(binding["approval"]), frame=deepcopy(binding["frame"]),
        carrier_sha256=_array_sha(carrier), analysis_sha256=_array_sha(analysis),
        underlay_sha256=checked.digest(),
        replay="checked_for_adoption: exact frozen version and explicit frame")
    return checked, provenance


def _checked_native_underlay(folder, row, source, labels, preparation=None):
    """Replay the original nearest-neighbour transport of a bound drawn underlay."""
    from types import SimpleNamespace
    from .layer_painting_fixtures import read_underlay, project_underlay
    from .map_adoption import native_labels
    from .rig_point_alignment import effective_model_matrix

    models = [_checked(row[key], metadata=True) for key in ("reference_model", "target_model")]
    reference, target = [SimpleNamespace(metric_to_grid=effective_model_matrix(model, (preparation or {}).get("geometry_index")),
        model=model, run_id=run) for model, run in zip(models,
        (source.provenance["version"]["run_id"], row["run_id"]))]
    replayed, _, _, _ = native_labels(source.analysis_layers, reference, target)
    samples = None
    if (preparation or {}).get('continuation') is not None:
        from .photo_edge_continuation import load_contract, apply_policy
        contract = load_contract(Path(folder) / preparation['continuation']['file'], row['run_id'])
        if 'frame_raster' in contract:
            classified = apply_policy(source.analysis_layers, contract)
            replayed, samples = classified['labels'], classified['frame_boundary_samples']
    expected = project_underlay(source, reference, target, replayed, frame_boundary_samples=samples)
    bound = _local(folder, row["fixture_underlay"])
    actual = retry_transient(lambda: read_underlay(bound))
    drawn_ids = sorted(int(value) for value in np.unique(actual.drawn_layers) if value)
    if (actual.digest() != expected.digest() or not np.array_equal(actual.analysis_layers, labels)
            or not _same(drawn_ids, row.get("fixture_underlay_layer_ids"))):
        raise ValueError("The native fixture underlay differs from its frozen placement replay")
    return drawn_ids, bound



def verify_version_summary(binding, summary):
    """Join the outward version summary to its literal saved identity and approval."""
    record = binding["record"]
    folder = Path(binding["version"]["path"]).resolve().parent
    required = {"folder", "name", "kind", "sha256", "approval"}
    if (not isinstance(summary, dict) or not required.issubset(summary)
            or not isinstance(summary["folder"], str) or not Path(summary["folder"]).is_absolute()
            or Path(summary["folder"]).resolve() != folder
            or not _same(summary["approval"], binding["approval"])
            or any(not _same(summary.get(key), record.get(key)) for key in ("name", "kind", "sha256"))
            or any(not _same(value, record[key]) for key, value in summary.items() if key in record)):
        raise ValueError("The preparation summary differs from its literal saved version and approval")


def verify_preparation_publication(preparation, folder):
    """Force original and generated file reads immediately before NEW publication."""
    from .local_map_adoption import PROOF_FORMAT, verify_local_preparation
    if (preparation.get("painting_geometry") or {}).get("format") == PROOF_FORMAT:
        return verify_local_preparation(preparation, folder)
    with _FILES.fresh():
        return _verify_preparation_publication(preparation, folder)


def _verify_preparation_publication(preparation, folder):
    """Recheck frozen sources and outputs immediately before publication.

    This repeats source admission, not the later facies-table choice. Missing
    CSV name matches and property conflicts remain the Check/Store gate.
    Historical saved-version and approval readers keep their original guards.
    """
    from .geometry_raster_sources import _integer_map
    from .painted_adoption_geometry import verify_proof, received_source_rows

    binding = preparation[PREPARATION_KEY]
    verify_version_summary(binding, preparation.get("map_version"))
    _, carrier, frame = verify_version(binding)
    guarded = [_checked(preparation["geometry_index"])]
    current = _checked(binding["frame_basis"]["workspace_revision"], metadata=True)
    parent_source = _source(current)
    guarded.extend([parent_source["revision"], parent_source["csv"], binding["version"],
                    *binding["files"].values(), _bound(Path(folder) / "approval.json")])
    if not _same(_checked(guarded[-1], metadata=True), binding["approval"]):
        raise ValueError("The preparation approval changed before publication")
    analysis, fixture_source = carrier, None
    if preparation.get("source_fixture_underlay") is not None:
        from .layer_painting_fixtures import read_underlay
        bound = _local(folder, preparation["source_fixture_underlay"])
        fixture_source, derivation = checked_fixture_analysis(binding, retry_transient(lambda: read_underlay(bound)))
        if (not _same(preparation.get("painting_analysis_derivation"), derivation)
                or not _same(preparation.get("fixture_frame"), binding["frame"])):
            raise ValueError("The fixture derivation changed before publication")
        analysis = fixture_source.analysis_layers
        guarded.append(bound)
    source_map = _local(folder, preparation["source_map"])
    actual_source = retry_transient(lambda: _integer_map(source_map["path"]))
    if actual_source.dtype != analysis.dtype or not np.array_equal(actual_source, analysis):
        raise ValueError("The source map changed before publication")
    guarded.append(source_map)
    proof = preparation["painting_geometry"]
    models = proof.get("models") if isinstance(proof, dict) else None
    if not isinstance(models, dict):
        raise ValueError("The preparation lost its frozen model closure")
    requested = {preparation["reference_run"], *[row["run_id"] for row in preparation["runs"]]}
    if set(models) - requested:
        raise ValueError("The preparation model closure contains an unrequested run")
    for row in preparation["runs"]:
        if row.get("status") != "prepared":
            continue
        for run, name in ((preparation["reference_run"], "reference_model"), (row["run_id"], "target_model")):
            if not _same(row.get(name), models.get(run)):
                raise ValueError("The prepared run changed its frozen model identity")
        labels, _, _, bound = _native_flags(folder, row)
        if (list(labels.shape) != row.get("shape_hw")
                or not _same([int(value) for value in np.unique(labels)], row.get("label_values"))):
            raise ValueError("The native output identity changed before publication")
        guarded.append(bound)
        if fixture_source is not None:
            _, bound = _checked_native_underlay(folder, row, fixture_source, labels, preparation)
            guarded.append(bound)
    guarded.extend(verify_proof(binding, proof, models, folder))
    received, received_checks = received_source_rows(proof, folder=folder)
    # Only a NEW publication requires the original configuration to remain
    # unchanged. Historical verify_proof continues using its retained copies.
    for bound in models.values():
        model = _checked(bound, metadata=True)
        original = model["baseline_configuration"]
        source = {key: original[key] for key in ("path", "sha256", "size_bytes")}
        guarded.append(received.get((source["path"], source["sha256"], source["size_bytes"]), source))
    # Current receipted pools preserve foreign original identity without
    # reopening foreign runtime paths or trusting newly made preparation copies.
    for row in (proof.get("raw_shape_basis") or {}).get("source_copies", []):
        source = row["original"]
        guarded.append(received.get((source["path"], source["sha256"], source["size_bytes"]), source))
    if preparation.get("outside_frame_proof") is not None:
        guarded.extend(verify_outside_proof(preparation, folder, carrier, frame))
    if "analysis_exclusions" in preparation:
        from .painted_analysis_exclusions import prepared_exclusion, exclusion_sources
        for row in preparation["runs"]:
            if row.get("status") == "prepared":
                prepared_exclusion(preparation, folder, row["run_id"])
        guarded.extend(exclusion_sources(preparation))
    if "target_corrections" in preparation:
        from .painted_target_integration import effective_product, target_sources
        for row in preparation["runs"]:
            if row.get("status") == "prepared":
                _, effective = effective_product(preparation, folder, row)
                guarded.append(effective)
        guarded.extend(target_sources(preparation))
    for bound in guarded:
        _checked(bound)
    for checked in received_checks:
        checked.revalidate()


def _source(current):
    """Bind the actual current Store revision and its full original facies CSV."""
    if (not isinstance(current, dict) or not isinstance(current.get("configuration"), dict)
            or not current.get("revision_id")):
        raise ValueError("Automatic adoption needs an actual current Store revision and source context")
    csv_path = Path(str((current.get("artifacts") or {}).get("facies_properties_path") or ""))
    if not csv_path.is_absolute():
        raise ValueError("Automatic adoption needs the actual current Store facies CSV")
    revision = _bound(csv_path.parent / "revision.json")
    if not _same(_checked(revision, metadata=True), current):
        raise ValueError("Current geometry differs from its actual immutable Store revision")
    committed = (current.get("artifacts") or {}).get("source_mode_files", {}).get("facies_properties")
    if not isinstance(committed, dict) or Path(committed["path"]).resolve() != csv_path.resolve():
        raise ValueError("Current table lacks its committed Store artifact identity")
    _checked(committed)
    return dict(revision=revision, revision_id=current.get("revision_id"), csv=_bound(csv_path))


def saved_adoption_revision(current, folder):
    """Recognize an already saved exact preparation without adopting it again.

    The preparation keeps its original parent revision. Reopening its result
    must not derive a second table from the newly saved output revision.
    This is metadata recognition only; Setup retains its original admission.
    """
    configuration = current.get("configuration") or {}
    contract = configuration.get(KEY) or {}
    if contract.get("format") != FORMAT:
        return None
    preparation = _bound(Path(folder) / "index.json")
    if not _same(contract.get("preparation"), preparation):
        return None
    if not _same((configuration.get("provenance") or {}).get("preparation"), preparation):
        raise ValueError("The saved adoption has inconsistent preparation identities")
    source = _source(current)
    table = contract.get("table") or {}
    if any(table.get(key) != source["csv"][key] for key in ("sha256", "size_bytes")):
        raise ValueError("The saved adoption table differs from its committed Store CSV")
    return deepcopy(current)


def frame_basis_text(binding):
    """Describe the independently frozen preparation frame and its recorded limits."""
    basis = binding.get("frame_basis") or {}
    if basis.get("historical_frame_bound") is True:
        return "Current frame frozen at preparation; exact saved polygons frame also checked."
    return ("Current frame frozen at preparation; this legacy version records no exact "
            "historical metric frame. Its approved map/files and facies remain separate.")


def _native_inside(row, inside, preparation=None):
    """Reconstruct the physical frame mask on one measured native grid."""
    from .geometry_raster_sources import painted_map_composition, carry_label_reference
    reference = _checked(row["reference_model"], metadata=True)
    target = _checked(row["target_model"], metadata=True)
    left, right, shape = painted_map_composition(reference, target,
        geometry_index=(preparation or {}).get("geometry_index"))
    return carry_label_reference(inside.astype(np.uint8), reference_matrix=left.tolist(),
        target_matrix=right.tolist(), target_shape_hw=list(shape),
        reference_filled=np.zeros(inside.shape, bool), source_fill_status="KNOWN", reference_support=inside)


def _native_flags(folder, row, preparation=None):
    """Read exact native labels and their retained fill/support evidence."""
    if preparation is not None and any(key in preparation for key in ("target_corrections", "chart_transfers")):
        from .painted_target_integration import effective_product
        arrays, bound = effective_product(preparation, folder, row)
        return arrays["array"], arrays["filled"], arrays["support"], bound
    bound = _local(folder, row["labels"])
    def read_once():
        """Restart the read-only native archive load after a transient failure."""
        with np.load(_preset_io.source(bound["path"]), allow_pickle=False) as archive:
            return archive["array"], archive["filled"], archive["support"]
    labels, filled, support = retry_transient(read_once)
    if (labels.dtype != np.dtype("int32") or labels.ndim != 2
            or filled.dtype != np.dtype("bool") or support.dtype != np.dtype("bool")
            or filled.shape != labels.shape or support.shape != labels.shape):
        raise ValueError("Outside proof needs the exact native labels and bool fill/support fields")
    return labels, filled, support, bound


def _inside_file(folder, binding, expected):
    """Load and verify the retained native inside-frame support array."""
    bound = _local(folder, binding)
    def read_once():
        """Restart the read-only native support load after a transient failure."""
        with np.load(_preset_io.source(bound["path"]), allow_pickle=False) as archive:
            if set(archive.files) != {"inside"}:
                raise ValueError("Unexpected outside-frame mask fields")
            return archive["inside"]
    inside = retry_transient(read_once)
    if inside.dtype != np.dtype("bool") or not np.array_equal(inside, expected):
        raise ValueError("Outside-frame mask is not its frozen frame's exact mask")
    return bound


def _relative(folder, path):
    """Bind a new file using its preparation-relative locator."""
    bound = _bound(path)
    return dict(bound, path=str(Path(path).resolve().relative_to(Path(folder).resolve())).replace("\\", "/"))


def _continuation_outside_context(preparation, folder, binding):
    """Bind the existing V2 policy; a sidecar's mere presence grants nothing."""
    from .geometry_raster_sources import PAINTED_PREPARATION_V2, _integer_map
    from .photo_edge_continuation import POLICY, read_contracts, verify_source
    if preparation is None or preparation.get("format") != PAINTED_PREPARATION_V2:
        return None
    record = preparation.get("continuation")
    if (not isinstance(record, dict)
            or set(record) != {"policy", "file", "sha256", "size_bytes", "adopted"}
            or record["policy"] != POLICY or record["file"] != "continuation.json"
            or record["adopted"] is not False
            or not _same(preparation.get("approved_source"), binding)
            or not _same(preparation.get(PREPARATION_KEY), binding)):
        raise ValueError("V2 outside proof needs its exact approved source and continuation")
    sidecar = _local(folder, {"path": record["file"],
                              **{key: record[key] for key in ("sha256", "size_bytes")}})
    contracts = read_contracts(sidecar["path"])["runs"]
    rows = preparation.get("runs") or []
    ids = [row.get("run_id") for row in rows]
    if (not ids or any(not isinstance(run, str) for run in ids)
            or len(ids) != len(set(ids)) or set(ids) != set(contracts)
            or any(row.get("status") != "prepared" for row in rows)):
        raise ValueError("V2 outside proof needs the exact complete continuation run set")
    source_bound = _local(folder, preparation["source_map"])
    source = retry_transient(lambda: _integer_map(source_bound["path"]))
    verify_source(binding, source, contracts, preparation=preparation, folder=folder)
    guarded = [sidecar, source_bound]
    for row in rows:
        run = row["run_id"]
        contract = contracts[run]
        summary = row.get("continuation") or {}
        if (contract["target_grid"] != "native"
                or contract["source_run"] != preparation.get("reference_run")
                or contract["target_run"] != run
                or not _same(contract["source_model"]["descriptor"], row.get("reference_model"))
                or not _same(contract["target_model"]["descriptor"], row.get("target_model"))
                or not _same(contract["geometry_index"], preparation.get("geometry_index"))
                or summary.get("contract_run") != run
                or summary.get("authority") != contract["frame_authority"]["kind"]):
            raise ValueError("V2 outside proof belongs to other models, frame authority or native grid")
        # These external dependencies must participate in the table's original
        # derivation cache, as well as the policy's independent strict readers.
        guarded.append(_checked(contract["geometry_index"]))
        guarded.extend(_checked(contract[key]["descriptor"]) for key in ("source_model", "target_model"))
    context = dict(contracts=contracts, source=source, guarded=guarded, sidecar=sidecar,
                   preparation=preparation)
    from .local_map_adoption import PROOF_FORMAT, verify_local_preparation
    if (preparation.get("painting_geometry") or {}).get("format") == PROOF_FORMAT:
        # Authenticate the complete closure once for this outside-proof
        # operation, rather than every run rechecking every other run.
        guarded.extend(verify_local_preparation(preparation, folder))
        context["verified_local_preparation"] = deepcopy(preparation)
    return context


def _continuation_outside_mask(folder, row, context):
    """Replay every original V2 field before classifying any zero as outside."""
    from .photo_edge_continuation import verify_product
    product = _local(folder, row["labels"])
    from .local_map_adoption import PROOF_FORMAT, verify_local_preparation
    preparation = context.get("preparation") or {}
    if (preparation.get("painting_geometry") or {}).get("format") == PROOF_FORMAT:
        if not _same(context.get("verified_local_preparation"), preparation):
            verify_local_preparation(preparation, folder)
        with np.load(_preset_io.source(product["path"]), allow_pickle=False) as archive:
            inside = np.array(archive["inside"], copy=True)
        if inside.dtype != bool or list(inside.shape) != row["shape_hw"]:
            raise ValueError("Reused inside mask changed its original native grid")
        replay = dict(inside=inside)
    else:
        replay = verify_product(product["path"], context["source"],
                                context["contracts"][row["run_id"]], counts=row["continuation"])
    from .painted_analysis_exclusions import prepared_exclusion
    exclusion = prepared_exclusion(context["preparation"], folder, row["run_id"]) if "preparation" in context else None
    labels, _, _, _ = _native_flags(folder, row, context.get("preparation"))
    forbidden = (labels == 0) & replay["inside"]
    if exclusion is not None:
        forbidden &= ~exclusion
    if np.any(forbidden):
        raise ValueError(f"{row['run_id']}: zero labels are INSIDE its frozen own frame; no nonphysical classification is permitted")
    return replay["inside"]


def prepare_outside_proof(folder, binding, rows, *, preparation=None):
    """Write only explicit frame proof products inside a new preparation."""
    from . import layer_painting as lp
    _, approved, frame = verify_version(binding)
    inside = lp.frame_mask(frame, approved.shape, (1., 1.))
    if np.any((approved == 0) & inside):
        raise ValueError("Approved painting has zero pixels INSIDE its frozen frame; these cannot be frame/outside")
    zero_rows = []
    from .painted_analysis_exclusions import prepared_exclusion
    for row in rows:
        if row.get("status") == "prepared":
            labels, _, _, _ = _native_flags(folder, row, preparation)
            exclusion = prepared_exclusion(preparation, folder, row["run_id"]) if preparation is not None else None
            physical_zeros = labels == 0
            if exclusion is not None:
                physical_zeros &= ~exclusion
            if np.any(physical_zeros):
                zero_rows.append(row)
    if not zero_rows:
        return None
    context = _continuation_outside_context(preparation, folder, binding)
    path = Path(folder) / "source_frame_inside.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, inside=inside)
    result = dict(format=OUTSIDE_FORMAT, source_inside=_relative(folder, path), runs={})
    if context is not None:
        from .photo_edge_continuation import POLICY
        result.update(format=OUTSIDE_FORMAT_V2, policy=POLICY,
                      continuation=_relative(folder, context["sidecar"]["path"]))
    for row in zero_rows:
        labels, filled, support, _ = _native_flags(folder, row, preparation)
        if context is not None:
            native = _continuation_outside_mask(folder, row, context)
        else:
            carried = _native_inside(row, inside, preparation)
            native = np.asarray(carried["labels"]) != 0
            if (not np.array_equal(filled, carried["geometry_filled"])
                    or not np.array_equal(support, carried["support"])
                    or np.any((labels == 0) & (native | filled))):
                raise ValueError(f"{row['run_id']}: zero labels are interior or unmapped; no nonphysical frame classification is permitted")
        target = Path(folder) / row["run_id"] / "frame_inside.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, inside=native)
        result["runs"][row["run_id"]] = _relative(folder, target)
    return result


def verify_outside_proof(preparation, folder, approved, frame):
    """Recheck every retained native outside-frame support against the exact saved map."""
    from . import layer_painting as lp
    inside = lp.frame_mask(frame, approved.shape, (1., 1.))
    if np.any((approved == 0) & inside):
        raise ValueError("Zero pixels INSIDE the approved frame cannot be classified as outside")
    proof = preparation.get("outside_frame_proof")
    context = _continuation_outside_context(preparation, folder, preparation[PREPARATION_KEY])
    own_frame_proof = isinstance(proof, dict) and proof.get("format") == OUTSIDE_FORMAT_V2
    keys = {"format", "source_inside", "runs"}
    if own_frame_proof:
        keys |= {"policy", "continuation"}
    if (not isinstance(proof, dict) or set(proof) != keys
            or proof["format"] != (OUTSIDE_FORMAT_V2 if own_frame_proof else OUTSIDE_FORMAT)
            or (own_frame_proof and context is None)
            or not isinstance(proof["runs"], dict)):
        raise ValueError("Native zero labels lack their bound frozen-frame inside-mask proof")
    guarded = [_inside_file(folder, proof["source_inside"], inside)]
    if own_frame_proof:
        from .photo_edge_continuation import POLICY
        if proof["policy"] != POLICY or not _same(_local(folder, proof["continuation"]), context["sidecar"]):
            raise ValueError("V2 outside proof changed its exact continuation policy or sidecar")
    if context is not None:
        guarded.extend(context["guarded"])
    expected = set()
    from .painted_analysis_exclusions import prepared_exclusion
    for row in preparation["runs"]:
        if row.get("status") != "prepared":
            continue
        labels, filled, support, _ = _native_flags(folder, row, preparation)
        exclusion = prepared_exclusion(preparation, folder, row["run_id"])
        physical_zeros = labels == 0
        if exclusion is not None:
            physical_zeros &= ~exclusion
        if not np.any(physical_zeros):
            continue
        run = row["run_id"]
        expected.add(run)
        if context is not None:
            native = _continuation_outside_mask(folder, row, context)
        # Historical V2 source-frame proofs retain their original comparison
        # too. Choosing that older format cannot evade the own-frame veto.
        if not own_frame_proof:
            carried = _native_inside(row, inside, preparation)
            native = np.asarray(carried["labels"]) != 0
            if (not np.array_equal(filled, carried["geometry_filled"])
                    or not np.array_equal(support, carried["support"])
                    or np.any((labels == 0) & (native | filled))):
                raise ValueError(f"{run}: zero labels include interior or unmapped pixels, not proven outside frame")
        if run not in proof["runs"]:
            raise ValueError(f"{run}: the native frame-inside proof is missing")
        guarded.append(_inside_file(folder, proof["runs"][run], native))
    if set(proof["runs"]) != expected:
        raise ValueError("Outside-frame proof run identities differ")
    return guarded


def _with_frame_row(table, parent_bytes):
    """Append a nonphysical row above all old/current IDs with blank extra cells."""
    from . import layer_painting as lp
    from .adoption_facies_table import AdoptionFaciesTable, FaciesRowProvenance, _parent_table
    if lp.OUTSIDE_FACIES in table.facies_ids:
        raise ValueError("The physical painting uses the reserved outside-frame facies name")
    _, _, _, _, parents, _ = _parent_table(parent_bytes, table.parent_sha256)
    matched = [(identity, row) for identity, name, row in parents if name == lp.OUTSIDE_FACIES]
    names = {column.strip().casefold(): column for column in table.columns}
    identity = names.get("id") or names["facies_id"]
    facies_id = max(table.source_max_id, *table.facies_ids.values()) + 1
    known = {identity: str(facies_id), names["name"]: lp.OUTSIDE_FACIES,
        names["porosity"]: "0.0", names["permeability"]: "", names["permeability_unit"]: "",
        names["analysis_role"]: "frame", names["enabled"]: "False"}
    row, fields = {}, []
    for column in table.columns:
        if column in known:
            row[column], origin = known[column], "established:nonphysical-frame"
        else:
            row[column], origin = "", "not-applicable:proven-nonphysical-frame"
        fields.append((column, origin))
    rows = (*table.rows, MappingProxyType(row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=table.columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode("utf8")
    provenance = FaciesRowProvenance(lp.OUTSIDE_FACIES, facies_id, (0,),
        tuple(value for value, _ in matched), tuple(fields))
    return AdoptionFaciesTable(table.columns, rows,
        MappingProxyType(dict(table.layer_to_facies) | {0: facies_id}),
        MappingProxyType(dict(table.facies_ids) | {lp.OUTSIDE_FACIES: facies_id}), raw,
        table.parent_sha256, hashlib.sha256(raw).hexdigest(), (*table.provenance, provenance),
        name_policy=table.name_policy, source_max_id=table.source_max_id,
        retired_source_ids=table.retired_source_ids, changed_source_ids=table.changed_source_ids,
        aliases=table.aliases)


def _derived_copy(result):
    """Share the immutable table while isolating each caller's mutable metadata."""
    return (result[0], *deepcopy(result[1:]))


def derive(preparation, folder, source):
    """Reuse one all-run derivation only while every consumed file stamp matches."""
    global _DERIVED_PID
    key = hashlib.sha256(json.dumps(dict(preparation=preparation, source=source,
        folder=str(Path(folder).absolute())), sort_keys=True, allow_nan=False).encode("utf8")).hexdigest()
    with _DERIVED_LOCK:
        if _DERIVED_PID != os.getpid():
            _DERIVED.clear()
            _DERIVED_PID = os.getpid()
        cached = None if _FILES.requires_fresh() else _DERIVED.get(key)
        if cached is not None:
            if _FILES.unchanged(cached["dependencies"]):
                _DERIVED.move_to_end(key)
                return _derived_copy(cached["result"])
            _DERIVED.pop(key)
    with _FILES.capture() as dependencies:
        result = _derive_uncached(preparation, folder, source)
    cost = (len(result[0].csv_bytes)
            + len(json.dumps(result[1:], allow_nan=False).encode("utf8"))
            + len(json.dumps(dependencies, allow_nan=False).encode("utf8")))
    if cost <= _DERIVED_MAX_BYTES and _FILES.unchanged(dependencies):
        with _DERIVED_LOCK:
            _DERIVED[key] = dict(result=_derived_copy(result), dependencies=deepcopy(dependencies), cost=cost)
            while len(_DERIVED) > 4 or sum(row["cost"] for row in _DERIVED.values()) > _DERIVED_MAX_BYTES:
                _DERIVED.popitem(last=False)
    return result


def table_region_data(table, native_ids, native_drawn_ids=()):
    """Describe an already derived table without reading or granting authority."""
    native_ids, native_drawn_ids = set(native_ids), set(native_drawn_ids)
    normalized = {column.strip().casefold(): column for column in table.columns}
    identity = normalized.get("id") or normalized["facies_id"]
    records = {int(row[identity]): row for row in table.rows}
    regions = []
    for layer_id in sorted(native_ids | native_drawn_ids):
        facies_id = table.layer_to_facies[layer_id]
        row = records[facies_id]
        regions.append(dict(label_id=layer_id, facies_id=facies_id,
            name=row[normalized["name"]], analysis_role=row[normalized["analysis_role"]],
            porosity=float(row[normalized["porosity"]]),
            permeability=float(row[normalized["permeability"]]) if row[normalized["permeability"]] else None,
            enabled=row[normalized["enabled"]] == "True"))
    spec = dict(sha256=table.sha256, size_bytes=len(table.csv_bytes), columns=list(table.columns),
        layer_to_facies={str(key): value for key, value in table.layer_to_facies.items()},
        name_policy=table.name_policy, source_max_id=table.source_max_id,
        retired_source_ids=list(table.retired_source_ids),
        changed_source_ids=list(table.changed_source_ids), aliases=[asdict(alias) for alias in table.aliases],
        # Store returns the in-memory record; its JSON file must compare exactly.
        provenance=json.loads(json.dumps([asdict(row) for row in table.provenance], allow_nan=False)))
    return spec, regions


def _derive_uncached(preparation, folder, source):
    """Rebuild the complete automatic mapping from checked sources, read only."""
    from .geometry_raster_sources import _integer_map
    from .adoption_check_progress import report_check_progress
    report_check_progress("Verifying approved painting sources for the facies table…")
    if not isinstance(source, dict) or set(source) != {"revision", "revision_id", "csv"}:
        raise ValueError("Explicit parent Store revision and full CSV source required")
    session, approved, frozen_frame = verify_version(preparation[PREPARATION_KEY])
    verify_version_summary(preparation[PREPARATION_KEY], preparation.get("map_version"))
    if preparation.get("reference_run") != session.run_id.casefold():
        raise ValueError("Prepared map and approved painting have different reference runs")
    if preparation.get("map_version", {}).get("sha256") != preparation[PREPARATION_KEY]["record"]["sha256"]:
        raise ValueError("Prepared map and saved painting have different approved SHAs")
    authority = _bound(Path(folder) / "approval.json")
    if (not _same(_checked(authority, metadata=True), preparation[PREPARATION_KEY]["approval"])
            or not _same(preparation.get("approval"), preparation[PREPARATION_KEY]["approval"])):
        raise ValueError("Preparation approval differs from the actual saved-map approval")
    source_map = _local(folder, preparation["source_map"])
    original = retry_transient(lambda: _integer_map(source_map["path"]))
    fixture_source = None
    fixture_ids = set()
    fixture_guards = []
    analysis = approved
    if preparation.get("source_fixture_underlay") is not None:
        from .layer_painting_fixtures import read_underlay
        bound = _local(folder, preparation["source_fixture_underlay"])
        fixture_source, derivation = checked_fixture_analysis(preparation[PREPARATION_KEY], retry_transient(lambda: read_underlay(bound)))
        if (not _same(preparation.get("painting_analysis_derivation"), derivation)
                or not _same(preparation.get("fixture_frame"), preparation[PREPARATION_KEY]["frame"])):
            raise ValueError("The explicit fixture analysis derivation or frame changed")
        analysis = fixture_source.analysis_layers
        fixture_ids.update(int(value) for value in np.unique(fixture_source.drawn_layers) if value)
        fixture_guards.append(bound)
    elif preparation.get("painting_analysis_derivation") is not None:
        raise ValueError("A fixture analysis derivation needs its bound source underlay")
    if original.dtype != analysis.dtype or not np.array_equal(original, analysis):
        raise ValueError("Prepared source map differs from its exact approved or replayed analysis map")
    native_ids, guarded = set(), [authority, source_map, *fixture_guards]
    if "analysis_exclusions" in preparation:
        from .painted_analysis_exclusions import exclusion_sources
        guarded.extend(exclusion_sources(preparation))
    if "target_corrections" in preparation:
        from .painted_target_integration import target_sources, effective_product
        guarded.extend(target_sources(preparation))
        for row in preparation["runs"]:
            if row.get("status") == "prepared":
                effective_product(preparation, folder, row)
    native_drawn_ids = set()
    models = {}
    chart_review = None
    prepared_count = sum(row.get("status") == "prepared" for row in preparation.get("runs") or [])
    checked_count = 0
    for row in preparation.get("runs") or []:
        if row.get("status") != "prepared":
            continue
        from .painted_target_integration import effective_product
        checked_count += 1
        report_check_progress(f"Verifying native labels for {str(row.get('run_id') or '').upper()} ({checked_count} of {prepared_count} runs)…")
        effective, bound = effective_product(preparation, folder, row)
        labels = effective["array"]
        values = [int(value) for value in np.unique(labels)]
        target_rows = (preparation.get("target_corrections") or {}).get("runs") or {}
        expected_values = (target_rows.get(row["run_id"]) or {}).get("label_values", row.get("label_values"))
        if not _same(values, expected_values) and row["run_id"] in (preparation.get("chart_transfers") or {}):
            # Chart restoration can legitimately add/remove a used layer.
            # Authenticate its exact current reviewed pixels instead of
            # treating the unchanged base map's value list as effective DATA.
            from .painted_analysis_exclusions import _read_scientific_gap_review, _identity
            if chart_review is None:
                chart_review = _read_scientific_gap_review(preparation["gap_review"]["path"])
                if not _same(chart_review["review_binding"], preparation["gap_review"]):
                    raise ValueError("The effective chart labels need their exact current review")
                guarded.append(chart_review["review_binding"])
            current = next(item for item in chart_review["runs"] if item["run_id"] == row["run_id"])
            if not _same(current["product"], bound) or not _same(current["scientific_arrays"]["array"], _identity(labels)):
                raise ValueError("The effective chart labels differ from their exact reviewed pixels")
            expected_values = values
        if not _same(values, expected_values) or list(labels.shape) != row.get("shape_hw"):
            raise ValueError("Prepared label identities or grid metadata changed")
        from .painted_analysis_exclusions import prepared_exclusion
        exclusion = prepared_exclusion(preparation, folder, row["run_id"])
        if exclusion is not None:
            values = [int(value) for value in np.unique(labels[~exclusion])]
        native_ids.update(values)
        guarded.append(bound)
        if fixture_source is not None:
            base_labels = retry_transient(lambda: _integer_map(_local(folder, row["labels"])["path"]))
            drawn, native_bound = _checked_native_underlay(folder, row, fixture_source, base_labels, preparation)
            native_drawn_ids.update(drawn)
            guarded.append(native_bound)
        reference_run = preparation["reference_run"]
        for key, model in ((reference_run, row["reference_model"]), (row["run_id"], row["target_model"])):
            if key in models and not _same(models[key], model):
                raise ValueError("Prepared runs bind different models for one geometry identity")
            models[key] = model
    if not native_ids:
        raise ValueError("No native run map is prepared")
    from .painted_adoption_geometry import verify_proof
    proof = preparation.get("painting_geometry")
    # The preparation may also name measured models for runs subsequently
    # marked unavailable. Their identities are still explicit, never inferred.
    if isinstance(proof, dict) and isinstance(proof.get("models"), dict):
        extra = set(proof["models"]) - set(models)
        requested = {row["run_id"] for row in preparation.get("runs") or []}
        if extra - requested:
            raise ValueError("The transport proof claims an unrequested run model")
        models.update({key: proof["models"][key] for key in extra})
    report_check_progress("Verifying painted-map geometry and outside-frame proof…")
    guarded.extend(verify_proof(preparation[PREPARATION_KEY], proof, models, folder))
    if 0 in native_ids:
        guarded.extend(verify_outside_proof(preparation, folder, approved, frozen_frame))
    approved_ids = set(map(int, np.unique(analysis))) | fixture_ids
    if (set(native_ids) | native_drawn_ids) - approved_ids:
        raise ValueError("A prepared run contains a layer absent from its approved source map")
    metadata = preparation.get("regions") or []
    metadata_by_layer = {row["layer_id"]: row for row in metadata}
    if len(metadata_by_layer) != len(metadata):
        raise ValueError("Prepared layer metadata has duplicate identities")
    for layer in session.layers:
        if layer.layer_id not in approved_ids:
            continue
        row = metadata_by_layer.get(layer.layer_id) or {}
        if (row.get("name") != layer.name or row.get("facies") != layer.facies
                or row.get("role") != session.facies[layer.facies].role):
            raise ValueError("Prepared facies metadata differs from its exact approved session")
    report_check_progress("Verifying the parent facies table and deriving automatic rows…")
    parent = _checked(source["revision"], metadata=True)
    if not _same(source["revision"], preparation[PREPARATION_KEY]["frame_basis"]["workspace_revision"]):
        raise ValueError("The table and explicit preparation frame belong to different workspace revisions")
    if (not isinstance(parent, dict) or parent.get("revision_id") != source.get("revision_id")
            or Path(str((parent.get("artifacts") or {}).get("facies_properties_path"))).resolve() != Path(source["csv"]["path"]).resolve()):
        raise ValueError("The table source is not its actual parent Store CSV")
    committed = (parent.get("artifacts") or {}).get("source_mode_files", {}).get("facies_properties")
    if not isinstance(committed, dict) or not _same(_checked(committed), _checked(source["csv"])):
        raise ValueError("The parent table differs from its committed Store artifact identity")
    _checked(source["csv"])
    parent_bytes = _read(source["csv"]["path"], MAX_PARENT_BYTES)
    used = sorted(approved_ids - {0})
    table = build_adoption_facies_table(session, parent_bytes,
        expected_sha256=source["csv"]["sha256"], used_layer_ids=used)
    if 0 in native_ids:
        table = _with_frame_row(table, parent_bytes)
    spec, regions = table_region_data(table, native_ids, native_drawn_ids)
    guarded.extend([source["revision"], source["csv"], preparation[PREPARATION_KEY]["version"],
                    *preparation[PREPARATION_KEY]["files"].values()])
    return table, spec, regions, session.permeability_unit, guarded


def checked_facies_source_acquisition(preparation, folder, declaration):
    """Bind an explicit statement to the already-verified painting, not targets.

    The ordinary derivation must first replay its original approved source/map
    checks. This metadata handoff neither infers chemistry nor interprets or
    grants the caller's scientific/adoption authority. The literal decision is
    retained for review; no saved map, approval or preparation is rewritten.
    """
    from .geometry_facies import RUN_GEOMETRY_ACQUISITION_STATES

    required = {"acquisition_state", "preparation", "photograph", "version",
                "approval", "source_map", "decision"}
    if not isinstance(declaration, dict) or set(declaration) != required:
        raise ValueError("A facies-source acquisition statement needs its exact source and decision")
    # Keep only literal finite JSON metadata, without coercing an enum or
    # accepting a mutable caller object as the candidate's provenance.
    try:
        result = _json(json.dumps(declaration, ensure_ascii=False, allow_nan=False).encode("utf8"))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("The facies-source acquisition statement is not finite JSON metadata") from error
    if (type(result["acquisition_state"]) is not str
            or result["acquisition_state"] not in RUN_GEOMETRY_ACQUISITION_STATES
            or not isinstance(result["decision"], dict) or not result["decision"]):
        raise ValueError("A facies-source acquisition statement needs an explicit state and decision")
    binding = preparation.get(PREPARATION_KEY)
    if not isinstance(binding, dict):
        raise ValueError("A facies-source acquisition statement requires an approved painting source")
    expected = dict(preparation=_bound(Path(folder) / "index.json"),
        photograph=binding["frame_basis"]["photograph"], version=binding["version"],
        approval=binding["approval"], source_map=_local(folder, preparation["source_map"]))
    for name, value in expected.items():
        if not _same(result[name], value):
            raise ValueError(f"The facies-source acquisition statement binds another {name}")
    return result


def candidate_table(current, preparation, folder):
    """Create the explicit table contract from one verified all-run derivation."""
    source = _source(current)
    table, spec, regions, unit, _ = derive(preparation, folder, source)
    contract = dict(format=FORMAT, source=source, preparation=_bound(Path(folder) / "index.json"), table=spec)
    return contract, regions, unit, table


def canonical_payload(config, parent, payload, *, run_id=None):
    """Validate the explicit additive branch; return derived bytes for Store IO."""
    from . import geometry_raster_sources as sources
    contract = config.get(KEY)
    if (not isinstance(contract, dict) or set(contract) != {"format", "source", "preparation", "table"}
            or contract["format"] != FORMAT or config.get("grouping_mode") != "shared_facies"
            or str(config.get("facies_properties_source_path")) != str(parent["imported_configuration"].get("facies_properties_source_path"))):
        raise ValueError("Explicit painted-adoption table contract and unchanged imported source required")
    preparation = _checked(contract["preparation"], metadata=True)
    if (not isinstance(preparation, dict) or preparation.get("format") not in {sources.PAINTED_PREPARATION, sources.PAINTED_PREPARATION_V2}
            or PREPARATION_KEY not in preparation):
        raise ValueError("The facies table lacks its exact approved-painting preparation")
    folder = Path(contract["preparation"]["path"]).parent
    source_revision = _checked(contract["source"]["revision"], metadata=True)
    if (not isinstance(source_revision, dict)
            or not _same((source_revision.get("configuration") or {}).get("canonical_import_parent"), parent)):
        raise ValueError("The table source belongs to another actual imported parent")
    table, spec, regions, unit, guarded = derive(preparation, folder, contract["source"])
    if (not _same(spec, contract["table"]) or not _same(regions, config.get("regions"))
            or config.get("permeability_unit") != unit):
        raise ValueError("Derived painting table, IDs, roles or physical properties changed")
    labels = retry_transient(lambda: sources._integer_map(config["labels_path"]))
    if labels.shape != payload["labels"].shape:
        raise ValueError("A painted adoption cannot change the imported native grid")
    prepared = {row["run_id"]: row for row in preparation["runs"] if row.get("status") == "prepared"}
    display = config["frame_binding"]["display_reference_run"]
    from .painted_target_integration import effective_product
    effective_display = effective_product(preparation, folder, prepared[display])[1] if display in prepared else None
    if effective_display is None or Path(config["labels_path"]).resolve() != Path(effective_display["path"]):
        raise ValueError("The display labels are not their prepared painted map")
    expected_runs = set(prepared) & set(config["frame_binding"]["runs"])
    rows = (config.get("run_geometry") or {}).get("runs") or []
    if len(rows) != len(expected_runs) or {row.get("run_id") for row in rows} != expected_runs:
        raise ValueError("Painted adoption run rows differ from their bound preparation")
    checked_rows = rows
    if run_id is not None:
        selected = str(run_id).strip().casefold()
        checked_rows = [row for row in rows
                        if str(row.get("run_id") or "").strip().casefold() == selected]
        if len(checked_rows) != 1:
            raise ValueError("The selected run has no explicit registered override.")
    provenance = config.get("provenance") or {}
    if not _same(config.get("target_corrections"), preparation.get("target_corrections")):
        raise ValueError("The adopted target corrections differ from their exact preparation")
    from .painted_analysis_exclusions import KEY as EXCLUSIONS_KEY, prepared_exclusion
    if not _same(config.get(EXCLUSIONS_KEY), preparation.get(EXCLUSIONS_KEY)):
        raise ValueError("The adopted analysis exclusion contract differs from its preparation")
    if EXCLUSIONS_KEY in preparation:
        # The common derivation above admits every prepared run. A selected
        # Store read replays its own exclusion; all-run admission keeps all.
        for row in checked_rows:
            prepared_exclusion(preparation, folder, row["run_id"])
    if "facies_source_acquisition" in provenance:
        # This belongs outside memoized derive: every current candidate must
        # bind its statement and every transported row, even after a warm read.
        acquisition = checked_facies_source_acquisition(
            preparation, folder, provenance["facies_source_acquisition"])
        if any(row.get("acquisition_state") != acquisition["acquisition_state"] for row in rows):
            raise ValueError("Painted run acquisition states differ from their explicit facies source")
    authority = _bound(folder / "approval.json")
    source_map = _local(folder, preparation["source_map"])
    expected_kind = sources.PAINTED_V2 if preparation["format"] == sources.PAINTED_PREPARATION_V2 else sources.PAINTED
    for row in rows:
        record = row.get("registration") or {}
        if (record.get("kind") != expected_kind or not _same(record.get("measurement"), contract["preparation"])
                or not _same(record.get("authority"), authority) or not _same(record.get("source_map"), source_map)):
            raise ValueError("Automatic facies apply only to their exact painted run registration")
    sources._validate_registered_runs(config, payload, parent, run_id=run_id)
    for binding in [contract["preparation"], *guarded]:
        _checked(binding)
    return dict(payload, rows=[dict(row) for row in table.rows], columns=list(table.columns),
        ids=list(table.facies_ids.values()), painted_table_bytes=table.csv_bytes,
        painted_table_guarded_sources=[contract["preparation"], *guarded])
