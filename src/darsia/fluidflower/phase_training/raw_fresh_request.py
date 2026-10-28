"""Prepare an explicit new RAW review cohort without invoking a frozen trainer.

This preparation preserves the supplied input photometry byte-for-byte while
selecting saved evidence and reviewers independently of historical cohort-ID
partitions. It performs no fitting, source-image decoding or phase inference.

The dedicated ``raw_fresh_training`` producer consumes these selected snapshots
and physical bindings and fits new optical heads with whole-run validation.
The historical frozen-cohort trainer cannot execute this preparation.

Raster completeness and supplied reference QA do not certify adoption provenance
or each run's configuration and transform bindings. Those links still require
verification before fitting; photographic and mass eligibility are unchanged.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import uuid

import numpy as np

from ..native_training_photometry import _fragment
from ..phase_saved_assets import VerifiedAssetReader, asset
from ..raw_photographic_review import RAW_REVIEW_FORMAT, validate_raw_review_manifest
from .scope import (FROZEN_ANNOTATION_SET_SETTINGS, partition_bundle,
                    partition_reviewers, partition_review_events, requested_reviewers, series_id)

FORMAT = "darsia-raw-phase-training-preparation-v1"
PHOTOMETRY_KEY = "v5_raw_photometry"
PHYSICAL_BINDING_KEYS = ("reference_context_path", "reference_context_sha256",
                         "optical_geometry_qa_pass", "physical_scale_qa_pass")


def _descriptor(value):
    """Validate an asset descriptor while preserving its bound metadata."""
    if (not isinstance(value, Mapping) or not isinstance(value.get("path"), str)
            or not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64):
        raise ValueError("An exact input path and SHA-256 are required")
    return dict(value)


def inspect_physical_context(source, native_configuration, *, reader=None):
    """Verify full reference fields and optical geometry QA for non-mass RAW.

    ``source`` identifies a JSON document and the key containing its context
    binding. The existing RAW context keys and configured raster ABI are used;
    no missing mask, coordinate, area or QA flag receives a substitute.
    Scale QA is required for mass-eligible use, not this RAW preparation.
    Its value is preserved when supplied and omitted when absent.
    """
    reader = reader or VerifiedAssetReader()
    source = _descriptor(source)
    key = source.get("binding_key", "binding")
    binding = reader.json(source).get(key)
    if not isinstance(binding, Mapping):
        raise ValueError("The physical-context source has no declared binding: " + str(key))
    path, digest = binding.get("reference_context_path"), binding.get("reference_context_sha256")
    context_asset = _descriptor(dict(path=path, sha256=digest))
    configuration = reader.json(_descriptor(native_configuration))
    try:
        exports = configuration["saved_review_metadata"]["training_frame_context"]["exports"]
        fields = tuple(exports["RASTER_FIELDS"]["value"])
        shape = tuple(exports["SHAPE"]["value"])
        safe_key = exports["REFERENCE_SAFE_KEY"]["value"]
    except (KeyError, TypeError) as error:
        raise ValueError("The supplied configuration has no complete physical-context raster ABI") from error
    if (len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape)
            or not fields or len(set(fields)) != len(fields)):
        raise ValueError("The physical-context configuration has invalid fields or native dimensions")
    masks = {"domain", "physics_supported_domain", "physical_sand_domain", "observable_phase_domain", safe_key}
    physical = {*masks, "facies", "permeability", "x_m_native", "y_m_native", "pixel_area_m2"}
    if not physical <= set(fields):
        raise ValueError("The configured context ABI omits required physical fields: " +
                         ", ".join(sorted(physical - set(fields))))
    with np.load(reader.path(context_asset), allow_pickle=False) as archive:
        missing = sorted((set(fields) | {"x_m", "y_m"}) - set(archive.files))
        if missing:
            raise ValueError("Full physical context is missing fields: " + ", ".join(missing))
        arrays = {name: np.asarray(archive[name]) for name in (*fields, "x_m", "y_m")}
        if (any(arrays[name].shape != shape for name in fields)
                or arrays["x_m"].shape != (shape[1],) or arrays["y_m"].shape != (shape[0],)):
            raise ValueError("Physical-context fields or nominal axes differ from the configured native grid")
        if any(arrays[name].dtype != np.bool_ for name in masks):
            raise ValueError("Physical-context domains and safe interior must be exact boolean masks")
        supported, sand, observable = (arrays[name] for name in
                                       ("physics_supported_domain", "physical_sand_domain", "observable_phase_domain"))
        if (not supported.any() or not observable.all() or not np.array_equal(arrays["domain"], supported)
                or np.any(supported & ~sand) or np.any(sand & ~observable)
                or np.any(arrays[safe_key] & ~sand)):
            raise ValueError("Physical-context supported, sand, observable and safe masks conflict")
        if arrays["facies"].dtype.kind not in "iu":
            raise ValueError("Physical-context facies must be a categorical integer raster")
        for name in ("permeability", "x_m_native", "y_m_native", "pixel_area_m2"):
            if not np.isfinite(arrays[name][supported]).all():
                raise ValueError("Physical-context field is nonfinite inside its supported domain: " + name)
        if (np.any(arrays["pixel_area_m2"][supported] <= 0)
                or not np.isfinite(arrays["x_m"]).all() or not np.isfinite(arrays["y_m"]).all()):
            raise ValueError("Physical-context areas and nominal axes must be finite, with positive areas")
    missing_qa = [name for name in ("optical_geometry_qa_pass",)
                  if binding.get(name) is not True]
    if missing_qa:
        raise ValueError("Full physical context lacks approved QA: " + ", ".join(missing_qa))
    return dict(source=source, archive=context_asset, native_shape_hw=list(shape), fields=list(fields),
                binding={name: binding[name] for name in PHYSICAL_BINDING_KEYS if name in binding})


def _verify_photometry_inputs(value, reader):
    """Check recorded scientific input declarations and their bound source files."""
    if not isinstance(value, Mapping):
        raise ValueError("The supplied parent has no saved input-photometry block")
    if value.get("display_jpeg_is_training_input") is not False or value.get("full_native_resolution") is not True:
        raise ValueError("The saved input-photometry block does not declare full-resolution non-display inputs")
    for key, item in value.items():
        if key.endswith("_path"):
            reader.path(_descriptor(dict(path=item, sha256=value.get(key[:-5] + "_sha256"))))


def _verify_raw_jobs(manifest, *, reader, checked=None):
    """Check RAW manifest identities and the declared photographic support."""
    result = []
    checked = {} if checked is None else checked
    for job in manifest.get("jobs") or ():
        run = str(job.get("run") or "").casefold()
        root = Path(job.get("analysis_source_review_root") or job["review_output_dir"])
        path = root / "native_frames" / run / "manifest.json"
        bound = _descriptor(dict(path=str(path), sha256=job.get("native_manifest_sha256")))
        key = (run, str(path).casefold(), bound["sha256"])
        if key in checked:
            result.append(checked[key])
            continue
        native = reader.json(bound)
        if native.get("format") != RAW_REVIEW_FORMAT:
            raise ValueError("A fresh RAW cohort requires saved RAW photographic review inputs for " + run)
        native = validate_raw_review_manifest(native, run)
        photo = native.get("training_context") or {}
        if photo.get("photo_sand_qa_pass") is not True:
            raise ValueError("The RAW review lacks approved photographic sand support for " + run)
        references = [(key, photo[key]) for key in ("photo_sand_reference_mask_path", "photo_certified_sand_mask_path")
                      if photo.get(key)]
        if not references:
            raise ValueError("The RAW review has no bound photographic sand source for " + run)
        for key, value in references:
            reader.path(_descriptor(dict(path=value, sha256=photo.get(key[:-5] + "_sha256"))))
        record = dict(run=run, native_manifest=bound, native_shape_hw=native["native_shape_hw"],
                      photographic_context=dict(photo))
        checked[key] = record
        result.append(record)
    if not result:
        raise ValueError("The selected review evidence contains no RAW source jobs")
    return result


def prepare_request(*, output_root: Path, parent_request: Mapping, bundles: Sequence[Mapping],
                    series: str, reviewers: Sequence[str] | None, settings: Mapping,
                    native_configuration: Mapping, context_source: Mapping | None = None,
                    context_mode="full", actor=""):
    """Prepare immutable selected evidence for the dedicated fresh RAW trainer."""
    if context_mode not in {"full", "photo-only"}:
        raise ValueError("Choose explicit full or photo-only context")
    if reviewers is not None and not reviewers:
        raise ValueError("Select at least one reviewer for the new cohort")
    selected_series = series_id(series)
    selected_reviewers = requested_reviewers({"reviewers": reviewers})
    reader = VerifiedAssetReader()
    parent_request = _descriptor(parent_request)
    parent_path = reader.path(parent_request)
    parent_raw = parent_path.read_bytes()
    parent = reader.json(parent_request)
    parent_settings = parent.get("settings") or {}
    if parent_settings.get("series") and selected_series not in {
            series_id(value) for value in parent_settings["series"]}:
        raise ValueError("The supplied input-photometry request belongs to another series")
    photometry = parent_settings.get(PHOTOMETRY_KEY)
    _verify_photometry_inputs(photometry, reader)
    fragment = _fragment(_fragment(parent_raw, "settings"), PHOTOMETRY_KEY)
    configured = dict(settings)
    forbidden = set(configured) & set(FROZEN_ANNOTATION_SET_SETTINGS)
    if forbidden or any(configured.get(key) for key in
                        ("resume_completed_heads", "completed_fit_source", "imported_completed_heads", "inherited_heads")):
        raise ValueError("A new RAW cohort cannot inherit frozen annotation partitions or completed fits")
    if PHOTOMETRY_KEY in configured and configured[PHOTOMETRY_KEY] != photometry:
        raise ValueError("The supplied input photometry differs from its bound parent")
    if configured.get("model_scope", "single-series") != "single-series":
        raise ValueError("The prepared RAW cohort must belong to one explicit series")
    if "series" in configured and {series_id(value) for value in configured["series"]} != {selected_series}:
        raise ValueError("The supplied training settings select another series")
    if "reviewers" in configured and requested_reviewers(configured) != selected_reviewers:
        raise ValueError("The supplied training settings select different reviewers")
    configured.update(model_scope="single-series", series=[selected_series],
                      reviewers=None if selected_reviewers is None else list(selected_reviewers),
                      **{PHOTOMETRY_KEY: deepcopy(photometry)})
    context = None
    if context_mode == "full":
        if context_source is None:
            raise ValueError("Supply a complete SHA-bound physical-context source before preparing full-context training")
        context = inspect_physical_context(context_source, native_configuration, reader=reader)
    else:
        reader.path(_descriptor(native_configuration))
    from .. import phase_review_retractions as withdrawals
    project_root = (configured.get("review_project_roots") or {}).get(selected_series)
    if not project_root:
        raise ValueError("Supply the series review-project root so withdrawn evidence can be checked")
    policy = partition_review_events(withdrawals.load_retractions(project_root, series=selected_series),
                                     selected_reviewers)
    configured[withdrawals.DOCUMENT_KEY] = policy
    selected = []
    for bundle in bundles:
        decisions_source, manifest_source = _descriptor(bundle["decisions"]), _descriptor(bundle["combined_manifest"])
        decisions, manifest = partition_bundle(reader.json(decisions_source), reader.json(manifest_source), selected_series)
        decisions = partition_reviewers(decisions, selected_reviewers)
        if not decisions.get("decisions"):
            continue
        selected_runs = {str(run).casefold() for run in decisions["decisions"]}
        manifest = dict(manifest, jobs=[job for job in manifest.get("jobs") or ()
                                       if str(job.get("run") or "").casefold() in selected_runs])
        decisions, _audit = withdrawals.filter_document(decisions, manifest, policy)
        decisions = dict(decisions, decisions={run: decision for run, decision in decisions["decisions"].items()
            if any(decision.get(name) for name in ("annotations", "phase_quiz_answers", "frame_choices"))})
        if not decisions["decisions"]:
            continue
        active_runs = {str(run).casefold() for run in decisions["decisions"]}
        manifest = dict(manifest, jobs=[job for job in manifest["jobs"]
                                       if str(job.get("run") or "").casefold() in active_runs])
        sources = _verify_raw_jobs(manifest, reader=reader)
        if context is not None and any(list(row["native_shape_hw"]) != context["native_shape_hw"] for row in sources):
            raise ValueError("The full physical context and selected RAW reviews use different native grids")
        selected.append((decisions_source, manifest_source, decisions, manifest, sources))
    if not selected:
        raise ValueError("No saved RAW review evidence remains for the selected series and reviewers")
    present = {str(item.get("id")) for entry in selected
               for decision in entry[2]["decisions"].values() for item in decision.get("annotations") or ()}
    if any(str(event["replacement"]["annotation"]["id"]) not in present
           for event in withdrawals.active_replacements(policy)):
        raise ValueError("Include the active replacement annotations in the saved evidence before preparing this cohort")
    root = Path(output_root).resolve()
    protected = [parent_path, reader.path(native_configuration)]
    protected += [reader.path(bound) for entry in selected for bound in entry[:2]]
    if any(path.resolve().is_relative_to(root.resolve()) for path in protected):
        raise ValueError("Choose a fresh preparation folder outside its bound inputs")
    root.mkdir(parents=True, exist_ok=False)
    snapshots = root / "input_snapshot"
    snapshots.mkdir()
    records = []
    for index, (decisions_source, manifest_source, decisions, manifest, sources) in enumerate(selected, start=1):
        record = dict(bundle_index=index, source_jobs=sources)
        for role, document, original in (("decisions", decisions, decisions_source),
                                         ("combined_manifest", manifest, manifest_source)):
            path = snapshots / f"{index:02d}_{role}.json"
            if role == "decisions":
                from .gas_negative_sampling import freeze_document_sources
                document = freeze_document_sources(document, snapshots, Path(original["path"]).parent)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
            record[role] = dict(asset(path), source=original)
        records.append(record)
    fragment_path = root / "input_photometry.json"
    with fragment_path.open("xb") as stream:
        stream.write(fragment)
    request = dict(format=FORMAT, artifact_kind="fresh-raw-training-preparation", created_by=str(actor),
        preparation_id=uuid.uuid4().hex, execution_ready=True,
        execution_entrypoint="darsia.fluidflower.phase_training.raw_fresh_training",
        frozen_trainer_compatible=False, execution_blocker="",
        settings=configured, series_scope=selected_series, context_mode=context_mode,
        native_configuration=_descriptor(native_configuration), physical_context=context,
        bundles=records, deployment_allowed=False, generalization_validated=False,
        provenance=dict(parent_request=parent_request, parent_model_id=parent.get("model_id"),
            photometry_setting=PHOTOMETRY_KEY, photometry_fragment=asset(fragment_path),
            photometry_fragment_sha256=sha256(fragment).hexdigest(),
            excluded_inherited_settings={key: deepcopy(parent_settings[key]) for key in FROZEN_ANNOTATION_SET_SETTINGS
                                         if key in parent_settings},
            cohort_policy="new explicit selected RAW evidence; historical annotation-ID partitions are not reused",
            review_policy_digest=policy["digest"],
            reviewer_scope=None if selected_reviewers is None else list(selected_reviewers)))
    marker = "saved-photometry-" + uuid.uuid4().hex
    serializable = deepcopy(request)
    serializable["settings"][PHOTOMETRY_KEY] = marker
    encoded = json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=False)
    encoded = encoded.replace(json.dumps(marker), fragment.decode("utf-8"), 1)
    if (_fragment(_fragment(encoded.encode("utf-8"), "settings"), PHOTOMETRY_KEY) != fragment
            or json.loads(encoded)["settings"][PHOTOMETRY_KEY] != photometry):
        raise ValueError("The prepared request did not preserve the exact input photometry")
    path = root / "raw_training_preparation.json"
    with path.open("xb") as stream:
        stream.write((encoded + "\n").encode("utf-8"))
    return dict(request, request_path=str(path))


def main(argv=None):
    """Freeze a verified preparation specification and report the saved request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args(argv)
    spec = VerifiedAssetReader().json(dict(path=str(args.spec), sha256=args.spec_sha256))
    result = prepare_request(**spec)
    print(json.dumps(dict(request_path=result["request_path"], execution_ready=result["execution_ready"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
