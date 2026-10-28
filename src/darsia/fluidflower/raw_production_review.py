"""Bind finished RAW photographs and published phase labels for Studio review.

Only one explicitly selected, byte-verified model origin is supported. A
production pipeline assembled from several model folders needs its own full
origin declaration; matching a pair of optical heads is insufficient.
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import io
import json
import math
from pathlib import Path

FORMAT = "darsia-raw-production-review-v1"


def install_readers():
    """Attach RAW production display adapters in a GUI process only.

    The shared reader source files also serve production code. Keep their
    bytes unchanged and install these optional display routes at GUI startup.
    """
    from functools import wraps
    from .tools import calibration_pairwise_review as review
    from .tools import phase_review_collections as collections
    if getattr(review._candidate_from_job, "_raw_production_reader", False):
        return
    candidate_reader = review._candidate_from_job
    native_path = review._native_phase_map_path
    overlay_path = review._preferred_phase_overlay_path
    bound_jobs = collections._bound_jobs

    @wraps(candidate_reader)
    def candidate(selection_dir, job, *, selected_admission=None):
        if (job.get("quality_flags") or {}).get("raw_production_review") is True:
            return review_candidate(selection_dir, job)
        return candidate_reader(selection_dir, job, selected_admission=selected_admission)

    @wraps(native_path)
    def native(output_dir, candidate, label, frame_index):
        if (getattr(candidate, "quality_flags", None) or {}).get("raw_production_review") is True:
            return phase_overlay(candidate, frame_index)
        return native_path(output_dir, candidate, label, frame_index)

    @wraps(overlay_path)
    def overlay(output_dir, candidate, label, frame_index, phase="all"):
        if (getattr(candidate, "quality_flags", None) or {}).get("raw_production_review") is True:
            return phase_overlay(candidate, frame_index, phase), "published_production_labels"
        return overlay_path(output_dir, candidate, label, frame_index, phase)

    @wraps(bound_jobs)
    def jobs(manifest_path, binding):
        values = bound_jobs(manifest_path, binding)
        for job in values:
            if (job.get("quality_flags") or {}).get("raw_production_review") is True:
                validated_job(job)
        return values

    candidate._raw_production_reader = True
    review._candidate_from_job = candidate
    review._native_phase_map_path = native
    review._preferred_phase_overlay_path = overlay
    collections._bound_jobs = jobs


def _file(path):
    """Return the current file's byte descriptor without modifying it."""
    from .raw_review_postrun import _pin
    return _pin(path)


def _read(bound):
    """Read saved JSON only after verifying its complete byte descriptor."""
    from .raw_review_postrun import _read
    return _read(bound)


def _write(path, value):
    """Create new durable JSON metadata through the shared exclusive writer."""
    from .raw_review_postrun import _write
    return _write(path, value)


def _assets(value):
    """Yield declared file descriptors from nested mappings and sequences."""
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield value
        else:
            for item in value.values():
                yield from _assets(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _assets(item)


def _origin(model_root, phase):
    """Require all declared inference model assets to have this exact origin."""
    from .phase_validation import _model_identity
    from .review_storage import resolve_review_path
    model = resolve_review_path(model_root).resolve()
    documents = list(_assets(phase.get("model_bundles")))
    if not documents:
        raise ValueError("The production configuration declares no complete model origin")
    checked = []
    for descriptor in documents:
        path = resolve_review_path(descriptor["path"]).resolve()
        if not path.is_relative_to(model):
            raise ValueError("Production uses multiple model origins; a full explicit pipeline binding is required: " + str(path))
        pinned = _file(path)
        if (pinned["sha256"] != descriptor["sha256"]
                or "size_bytes" in descriptor and pinned["size_bytes"] != descriptor["size_bytes"]):
            raise ValueError("A production model asset differs from the selected model: " + str(path))
        checked.append(pinned)
    required = [model / "models/generic" / name for name in
                ("co2_presence_signal.json", "gas_within_co2_signal.json")]
    declared = {(Path(row["path"]).resolve(), row["sha256"]) for row in checked}
    if any((path, _file(path)["sha256"]) not in declared for path in required):
        raise ValueError("Both original optical heads must belong to the selected production model")
    return model, _model_identity(model), checked


def register_review(completion_path, model_root, *, output_root=None):
    """Register an exact existing post-run result; never render or run inference.

    Callers explicitly choose the model. The registered collection retains the
    production configuration and final categorical predictions as provenance;
    it grants no scientific acceptance, mass use, or training continuation.
    """
    install_readers()
    from . import raw_review_postrun as post
    from .raw_photographic_review import validate_raw_review_manifest
    from .phase_runtime import load_physics_profile, physics_runtime_identity
    from .review_storage import resolve_review_path
    from .tools import phase_review_collections as collections
    completion_bound = _file(completion_path)
    completion = _read(completion_bound)
    if (completion.get("format") != post.COMPLETION_FORMAT or completion.get("complete") is not True
            or completion.get("adopted") is not False):
        raise ValueError("A completed RAW post-run review is required")
    request_bound = completion["request"]
    request = post._request(request_bound)
    job = post._finished(request["job_root"])
    stores = post._stores(request["stores"], request["series_root"])
    if (completion.get("review_root") != request["output_root"]
            or set(request["configurations"]) != set(job.runs)
            or [row.get("run") for row in completion.get("manifests", ())] != list(job.runs)):
        raise ValueError("RAW review completion names another job, root, or run order")
    model = resolve_review_path(model_root).resolve()
    parent = Path(output_root).resolve() if output_root else collections._model_collection_root(model)
    if parent == model or parent.is_relative_to(model) or parent.is_relative_to(job.root.resolve()):
        raise ValueError("Review registration must stay outside the model and original production job")
    identity = collections._identity(model)
    runs, pins = [], [completion_bound, request_bound, request["job"], request["stores"]]
    for run, manifest_bound in zip(job.runs, completion["manifests"], strict=True):
        manifest = validate_raw_review_manifest(_read(manifest_bound), run)
        for entry in (manifest["reference"], *manifest["frames"]):
            raw_parent = Path(entry["acquisition"]["path"]).resolve().parent
            if parent == raw_parent or parent.is_relative_to(raw_parent):
                raise ValueError("Review registration must stay outside original RAW data folders")
        expected_path = Path(request["output_root"]) / "native_frames" / run.casefold() / "manifest.json"
        if Path(manifest_bound["path"]).resolve() != expected_path.resolve():
            raise ValueError("RAW review timeline is outside the completed output root")
        sources = post._Sources(job, run, stores)
        config_bound, config = sources.configuration()
        if config_bound != request["configurations"][run]:
            raise ValueError("The production configuration changed after RAW review")
        phase_bound = sources.input(config["phase"]["configuration"], "phase.configuration", "configuration")
        phase = sources.json(config["phase"]["configuration"])
        origin, model_identity, assets = _origin(model, phase)
        if origin != model or tuple(model_identity) != (identity["source_model_id"], identity["source_model_sha256"]):
            raise ValueError("The selected production model identity changed")
        if len(manifest["frames"]) != len(config["frames"]) or job.frames(run) != [row["case_id"] for row in config["frames"]]:
            raise ValueError("RAW photographs differ from the original production timeline")
        labels = []
        for row, spec in zip(manifest["frames"], config["frames"], strict=True):
            if (row["case_id"] != spec["case_id"] or row["elapsed_h"] != spec["elapsed_h"]
                    or not math.isfinite(float(row["elapsed_h"]))):
                raise ValueError("RAW photographs differ from the original production frame/time")
            result = job.frame_result(run, spec["case_id"])
            stage = result.get("prediction", {}).get("pipeline", {})
            prediction = stage.get("phase", {})
            if (stage.get("complete") is not True or stage.get("configuration_sha256") != config_bound["sha256"]
                    or stage.get("source") != spec["source"]
                    or prediction.get("complete") is not True
                    or prediction.get("source_sha256") != spec["source"]["source_sha256"]
                    or stage.get("files", {}).get("native.json") != row["native_record"]):
                raise ValueError("Published predictions differ from the RAW source or production configuration")
            labels.append(sources.product(prediction["labels"]))
        pins.extend([manifest_bound, config_bound, phase_bound, *assets,
                     *request["completion"][run].values()])
        runs.append(dict(run=run, native=manifest_bound, labels=labels,
                         times=[row["elapsed_h"] for row in manifest["frames"]],
                         shape=list(manifest["phase_shape_hw"]), configuration=config_bound))
    receipt = dict(format=FORMAT, **identity, source_model_root=str(model), series_id=job.series_id,
        completion=completion_bound, pins=pins, runs=runs, adopted=False,
        scientific_acceptance=False, mass_analysis_allowed=False)
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    destination = parent / "production_sources" / digest
    for row in runs:
        raw = _read(row["native"])
        timeline = destination / row["run"] / "timeline.csv"
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(("frame_index", "elapsed_h", "source_sha256", "source_name"))
        writer.writerows((index, repr(frame["elapsed_h"]), frame["acquisition"]["sha256"],
                          Path(frame["acquisition"]["path"]).name)
                         for index, frame in enumerate(raw["frames"]))
        text = buffer.getvalue()
        if timeline.exists():
            if timeline.read_text(encoding="utf-8") != text:
                raise ValueError("Existing production timeline differs; it is preserved")
        else:
            timeline.parent.mkdir(parents=True, exist_ok=True)
            with timeline.open("x", encoding="utf-8", newline="") as stream:
                stream.write(text)
        row["timeline"] = _file(timeline)
    receipt_path = destination / "production_review.json"
    if receipt_path.exists():
        if _read(_file(receipt_path)) != receipt:
            raise ValueError("Existing production review registration differs")
    else:
        _write(receipt_path, receipt)
    jobs = []
    for row in runs:
        jobs.append(dict(run=row["run"], dataset=job.series_id, role="production", family="phase",
            variant="published-production", candidate_id=row["run"] + "__" + digest[:16], grid_order=1,
            candidate_folder=str(model), review_output_dir=request["output_root"], phase_map_output_dir=str(destination),
            optical_model_sha256=identity["source_model_sha256"], physics_runtime=identity["source_physics_runtime"],
            params_sha256=row["configuration"]["sha256"], phase_model_only=True, output_mode="review",
            quality_flags=dict(raw_production_review=True, scientific_acceptance=False, mass_analysis_allowed=False),
            production_review=_file(receipt_path)))
    common = dict(model_id=identity["source_model_id"], model_sha256=identity["source_model_sha256"],
                  physics_runtime=physics_runtime_identity(load_physics_profile(model)), output_mode="review")
    manifest = dict(common, jobs=jobs, models={})
    binding = dict(common, **{key: str(destination) for key in ("ffac_root", "ffum_root", "combined_root")},
        selected_candidates={row["run"]: row["candidate_id"] for row in jobs}, run_count=len(jobs))
    for name, value in (("postflight_manifest.json", manifest), ("validation_binding.json", binding)):
        path = destination / name
        if path.exists():
            if _read(_file(path)) != value:
                raise ValueError("Existing production review metadata differs")
        else:
            _write(path, value)
    for pin in pins:
        if _file(pin["path"]) != {key: pin[key] for key in ("path", "sha256", "size_bytes")}:
            raise ValueError("A production review input changed during registration")
    return collections.build_collection(model, [], output_root=parent, base_root=destination)


def validated_job(job):
    """Verify the metadata witness before a saved production candidate is shown."""
    from .phase_validation import _model_identity
    record = _read(job["production_review"])
    if (record.get("format") != FORMAT or record.get("adopted") is not False
            or record.get("scientific_acceptance") is not False
            or record.get("mass_analysis_allowed") is not False):
        raise ValueError("Production review needs its original read-only binding")
    identity = _model_identity(Path(record["source_model_root"]))
    if (identity != (record["source_model_id"], record["source_model_sha256"])
            or job.get("optical_model_sha256") != identity[1]
            or Path(job["candidate_folder"]).resolve() != Path(record["source_model_root"]).resolve()):
        raise ValueError("Production review identifies another source model")
    rows = [row for row in record["runs"] if row["run"] == job["run"]]
    if (len(rows) != 1 or job.get("dataset") != record["series_id"]
            or job.get("params_sha256") != rows[0]["configuration"]["sha256"]
            or job.get("quality_flags") != dict(raw_production_review=True, scientific_acceptance=False, mass_analysis_allowed=False)):
        raise ValueError("Production candidate differs from its original run binding")
    for pin in record["pins"]:
        actual = _file(pin["path"])
        if any(actual[key] != pin[key] for key in ("path", "sha256", "size_bytes")):
            raise ValueError("A registered production review source changed: " + pin["path"])
    raw = _read(rows[0]["native"])
    if _file(rows[0]["timeline"]["path"]) != rows[0]["timeline"]:
        raise ValueError("Registered production timeline changed")
    if Path(job["review_output_dir"]).resolve() / "native_frames" / job["run"].casefold() / "manifest.json" != Path(rows[0]["native"]["path"]).resolve():
        raise ValueError("Production candidate points to another RAW review")
    maps = Path(job["phase_map_output_dir"]).resolve()
    model = Path(record["source_model_root"]).resolve()
    if (maps != Path(job["production_review"]["path"]).resolve().parent
            or maps == model or maps.is_relative_to(model)):
        raise ValueError("Production display output differs from its separate registered folder")
    return record, rows[0], raw


def review_candidate(selection_dir, job):
    """Build a review candidate from verified RAW and published-label provenance."""
    from .tools.calibration_pairwise_review import Candidate
    record, run, raw = validated_job(job)
    root = Path(record["source_model_root"])
    return Candidate(run=job["run"], role=job["role"], family="phase", variant=job["variant"],
        candidate_id=job["candidate_id"], grid_order=job["grid_order"], params_sha256=job["params_sha256"],
        normalized_mae=None, mass_plot=root / "no_mass_plot", contour_movie=root / "no_movie",
        per_frame_csv=Path(run["timeline"]["path"]), summary_path=root / "summary.json",
        plot_seconds=0.0, image_seconds=0.0, fps=0.0, frame_count=raw["frame_count"],
        dataset=job["dataset"], review_output_dir=Path(job["review_output_dir"]),
        phase_map_output_dir=Path(job["phase_map_output_dir"]), phase_model_only=True,
        optical_model_sha256=record["source_model_sha256"],
        quality_flags=dict(job["quality_flags"], production_review=job["production_review"]))


def phase_overlay(candidate, frame_index, phase="all"):
    """Display exact published categorical labels; no thresholding or inference."""
    import numpy as np
    from PIL import Image
    from .phase_rendering import AQUEOUS_RGBA, GAS_RGBA
    descriptor = candidate.quality_flags["production_review"]
    record = _read(descriptor)
    from .phase_validation import _model_identity
    if (_model_identity(Path(record["source_model_root"])) !=
            (record["source_model_id"], candidate.optical_model_sha256)):
        raise ValueError("Production overlay identifies another source model")
    rows = [row for row in record["runs"] if row["run"] == candidate.run]
    if len(rows) != 1 or type(frame_index) is not int or not 0 <= frame_index < len(rows[0]["labels"]):
        raise ValueError("Production phase frame is outside its published timeline")
    bound = rows[0]["labels"][frame_index]
    if _file(bound["path"])["sha256"] != bound["sha256"]:
        raise ValueError("Published production phase labels changed")
    if phase not in {"all", "aqueous", "gas", "mixed"}:
        raise ValueError("This production result has no requested provenance layer")
    with Image.open(bound["path"]) as image:
        labels = np.asarray(image)
    if labels.dtype != np.uint8 or list(labels.shape) != rows[0]["shape"] or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Published categorical phase labels have another grid or encoding")
    rgba = np.zeros((*labels.shape, 4), np.uint8)
    if phase in {"all", "aqueous"}:
        rgba[labels == 1] = AQUEOUS_RGBA
    if phase in {"all", "gas"}:
        rgba[labels == 2] = GAS_RGBA
    encoded = io.BytesIO()
    Image.fromarray(rgba).save(encoded, format="PNG")
    target = Path(candidate.phase_map_output_dir) / "display" / bound["sha256"] / (phase + ".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != encoded.getvalue():
            raise ValueError("Existing production display copy changed; it is preserved")
    else:
        with target.open("xb") as stream:
            stream.write(encoded.getvalue())
    return target
