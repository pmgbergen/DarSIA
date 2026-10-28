"""Bind fresh training to the scientific RAW photographs already reviewed.

The image producer owns crop, registration and photometric corrections. This
module records those existing inputs; it does not reinterpret display images,
copy a previous model's image recipe, or grant model adoption.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from functools import lru_cache
import json
from pathlib import Path

from ..phase_saved_assets import VerifiedAssetReader, asset
from ..raw_photographic_review import RAW_REVIEW_FORMAT, validate_raw_review_manifest
from ..review_storage import resolve_review_path

FAMILY = "reviewed-raw-fresh-v1"
BASIS_FORMAT = "darsia-trained-image-basis-v1"


@lru_cache(maxsize=4096)
def _metadata_pin(path, size, modified_ns, inode):
    result = asset(path)
    after = Path(path).stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino) != (size, modified_ns, inode):
        raise ValueError("Training input metadata changed while its identity was read")
    return result


def _metadata_asset(path):
    path = resolve_review_path(path)
    info = path.stat()
    return dict(_metadata_pin(str(path), info.st_size, info.st_mtime_ns, info.st_ino))


def canonical_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def reviewed_input_basis(bundles, *, reader=None):
    """Return the declared RAW basis, or None for wholly historical inputs.

    Annotated runs determine the training family. Other explicitly prepared RAW
    runs retain their input contracts for inference without acquiring labels.
    Mixed RAW/historical training evidence cannot inherit transforms implicitly.
    """
    reader = reader or VerifiedAssetReader()
    rows, kinds, training_runs, checked = {}, set(), set(), {}
    for decisions_path, combined_path in bundles:
        decisions = reader.json(_metadata_asset(decisions_path))
        combined = reader.json(_metadata_asset(combined_path))
        selected = {str(run).casefold() for run, decision in decisions.get("decisions", {}).items()
                    if any(decision.get(key) for key in
                           ("annotations", "phase_quiz_answers", "frame_choices", "frame_verdicts"))}
        for job in combined.get("jobs", []):
            run = str(job.get("run", "")).casefold()
            annotated = run in selected
            root = resolve_review_path(job.get("analysis_source_review_root") or job["review_output_dir"])
            path = root / "native_frames" / run / "manifest.json"
            digest = job.get("native_manifest_sha256") or (job.get("v5_raw_photometry") or {}).get("materialized_manifest_sha256")
            key = (str(path).casefold(), digest, run)
            if key in checked:
                row = checked[key]
                if annotated:
                    kinds.add(row is not None)
                    if row is not None:
                        training_runs.add(run)
                if row is not None:
                    rows[(run, digest)] = row
                continue
            if not path.is_file():
                if annotated:
                    kinds.add(False)
                continue
            if not digest:
                if path.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError("The training source metadata is too large: " + run)
                if json.loads(path.read_text(encoding="utf-8-sig")).get("format") == RAW_REVIEW_FORMAT:
                    raise ValueError("The selected training source has no native-manifest SHA: " + run)
                checked[key] = None
                if annotated:
                    kinds.add(False)
                continue
            bound = dict(path=str(path), sha256=digest)
            manifest = reader.json(bound)
            raw = manifest.get("format") == RAW_REVIEW_FORMAT
            if annotated:
                kinds.add(raw)
                if raw:
                    training_runs.add(run)
            if not raw:
                checked[key] = None
                continue
            manifest = validate_raw_review_manifest(manifest, run)
            configuration = dict(manifest["native_configuration"])
            binding = dict(manifest["raw_binding"])
            reader.path(configuration)
            reader.path(binding)
            row = dict(run=run, native_shape_hw=list(manifest["native_shape_hw"]),
                       native_configuration=configuration, raw_binding=binding,
                       source_manifest=bound)
            checked[key] = row
            rows[(run, digest)] = row
    if not kinds or kinds == {False}:
        return None
    if kinds != {True}:
        raise ValueError("Choose one prepared image basis: RAW and historical photograph inputs cannot be mixed in fresh training.")
    ordered = [rows[key] for key in sorted(rows)]
    configurations = {row["native_configuration"]["sha256"] for row in ordered}
    if len(configurations) != 1:
        raise ValueError("The reviewed RAW photographs use different image-correction configurations; prepare them with the chosen series settings first.")
    from .log_linear_input import V5_UNIT_DENOMINATOR
    from .raw_frame_context import TEMPORAL_CONTRACT
    configuration = deepcopy(ordered[0]["native_configuration"])
    photometry = dict(format="darsia-reviewed-raw-input-photometry-v1",
        full_native_resolution=True, display_jpeg_is_training_input=False,
        native_configuration_path=configuration["path"], native_configuration_sha256=configuration["sha256"],
        unit_denominator=V5_UNIT_DENOMINATOR,
        scientific_rgb="float32-log-linear-native-centre",
        prepared_sources=[dict(run=row["run"], raw_binding=deepcopy(row["raw_binding"]),
                               native_shape_hw=row["native_shape_hw"], source_manifest=deepcopy(row["source_manifest"]))
                          for row in ordered])
    return dict(format=BASIS_FORMAT, native_configuration=configuration,
        input_photometry=photometry, input_photometry_sha256=canonical_digest(photometry),
        encoding="float32-camera-linear-unit-rgb", temporal_contract=TEMPORAL_CONTRACT,
        runs=ordered, training_runs=sorted(training_runs))


def apply_settings(settings, bundles, *, reader=None):
    """Use actual prepared RAW inputs rather than a selected parent's recipe."""
    basis = reviewed_input_basis(bundles, reader=reader)
    if basis is None:
        return dict(settings)
    from .scope import FROZEN_ANNOTATION_SET_SETTINGS
    result = {key: deepcopy(value) for key, value in settings.items()
              if key not in FROZEN_ANNOTATION_SET_SETTINGS and key not in
              {"resume_completed_heads", "completed_fit_source", "imported_completed_heads", "inherited_heads", "radiometry_profile"}}
    result.update(training_family=FAMILY, v5_raw_photometry=deepcopy(basis["input_photometry"]),
                  raw_image_basis=basis, adapters=["generic"], disable_resume_completed_heads=True)
    return result


def execution_preparation(request_binding):
    """Attach the existing fresh numerical producer to an ordinary Studio job.

    The ordinary request has already selected and snapshotted the evidence.
    This derivative keeps those exact snapshots and the image-producer basis.
    """
    from ..phase_workflow import PHASE_TRAINING_REQUEST_FORMAT
    from . import raw_fresh_request as fresh
    reader = VerifiedAssetReader()
    request_path = reader.path(request_binding)
    request = reader.json(request_binding)
    settings = deepcopy(request.get("settings") or {})
    if request.get("format") != PHASE_TRAINING_REQUEST_FORMAT or settings.get("training_family") != FAMILY:
        raise ValueError("A saved fresh RAW Studio training request is required")
    root = resolve_review_path(request["output_root"])
    if request_path.resolve() != (root / "training_request.json").resolve():
        raise ValueError("The fresh training request must belong to its selected model folder")
    pairs, records = [], []
    for index, row in enumerate(request["bundles"], 1):
        record = dict(bundle_index=index)
        for role in ("decisions", "combined_manifest"):
            bound = dict(path=row[role]["snapshot_path"], sha256=row[role]["sha256"])
            reader.path(bound)
            record[role] = bound
        pairs.append((record["decisions"]["path"], record["combined_manifest"]["path"]))
        records.append(record)
    basis = reviewed_input_basis(pairs, reader=reader)
    if basis != settings.get("raw_image_basis") or basis["input_photometry"] != settings.get(fresh.PHOTOMETRY_KEY):
        raise ValueError("The prepared RAW training image basis changed after the request was saved")
    source_jobs, checked_jobs = [], {}
    for record in records:
        jobs = fresh._verify_raw_jobs(reader.json(record["combined_manifest"]), reader=reader, checked=checked_jobs)
        record["source_jobs"] = jobs
        source_jobs.extend(jobs)
    configuration = basis["native_configuration"]
    # Full physical context, when supplied, is verified by the existing producer.
    # Photographic-only inputs retain that explicit limitation.
    first = source_jobs[0]
    photo = first["photographic_context"]
    context = None
    mode = "photo-only"
    if photo.get("reference_context_path") and photo.get("reference_context_sha256"):
        mode = "full"
        source = dict(first["native_manifest"], binding_key="training_context")
        context = fresh.inspect_physical_context(source, configuration, reader=reader)
    settings.pop("raw_image_basis", None)
    document = dict(format=fresh.FORMAT, artifact_kind="fresh-raw-training-preparation",
        created_by=request.get("created_by", ""), preparation_id=request["model_id"],
        execution_ready=True, execution_family=FAMILY, prepared_model_root=str(root),
        settings=settings, series_scope=settings["series"][0], context_mode=mode,
        native_configuration=configuration, physical_context=context, bundles=records,
        deployment_allowed=False, generalization_validated=False, image_basis=basis,
        provenance=dict(parent_request=dict(request_binding), parent_model_id=request["model_id"],
            photometry_setting=fresh.PHOTOMETRY_KEY,
            photometry_fragment=dict(request_binding), excluded_inherited_settings={},
            cohort_policy="current explicitly selected RAW review evidence"))
    # The exact photometry fragment lives in the ordinary immutable request.
    from ..native_training_photometry import _fragment
    raw = request_path.read_bytes()
    fragment = _fragment(_fragment(raw, "settings"), fresh.PHOTOMETRY_KEY)
    fragment_path = root / "input_snapshot" / "raw_input_photometry.json"
    with fragment_path.open("xb") as stream:
        stream.write(fragment)
    document["provenance"]["photometry_fragment"] = asset(fragment_path)
    document["provenance"]["photometry_fragment_sha256"] = sha256(fragment).hexdigest()
    path = root / "raw_training_preparation.json"
    from .raw_fresh_training import _request_bytes
    with path.open("xb") as stream:
        stream.write(_request_bytes(document, fragment))
    return asset(path)
