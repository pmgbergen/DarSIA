"""New optical paths from original P1 physical frames on the current image basis."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math

from ..mass import pooled_optical_paths as optical
from ..mass.n1_pipeline import N1Pipeline, _write_json, _write_npz
from .local_build import LocalBuildInputs

DEFAULT_SAMPLING = dict(stride_px=8, interior_radius_px=0, min_samples=16, min_runs=2)
DEFAULT_IMAGES_PER_RUN = 15
FRAME_SELECTION_METHOD = "common-elapsed-log-nearest-v1"


def colour_path_image_count(document):
    """Read the independent optical selection, never the mass/production count."""
    value = document.get("image_selection", {}).get("color_paths", {}).get(
        "max_images_per_run", DEFAULT_IMAGES_PER_RUN)
    if type(value) is not int or not 1 <= value <= 1000:
        raise ValueError("Colour paths need between 1 and 1000 representative images per run")
    return value


def optical_frame_selection(selection, max_images_per_run=DEFAULT_IMAGES_PER_RUN):
    """Choose comparable times from saved metadata, retaining every P1 input.

    Shared targets are dense during the early changes and spread across the
    full observed duration. A short run uses targets only while it has images.
    Baseline and clear water are separate references, outside this image count.
    """
    count = colour_path_image_count(dict(image_selection=dict(color_paths=dict(
        max_images_per_run=max_images_per_run))))
    candidates = {}
    for row in selection["runs"]:
        if any(type(frame["elapsed_h"]) not in {float, int} or not math.isfinite(frame["elapsed_h"])
               for frame in row["selected_frames"]):
            raise ValueError("Optical image times must be finite elapsed hours")
        frames = [frame for frame in row["selected_frames"] if frame["elapsed_h"] >= 0]
        if not frames:
            raise ValueError("An included run has no post-injection optical frames")
        candidates[row["run"]] = frames
    times = [float(frame["elapsed_h"]) for frames in candidates.values() for frame in frames]
    if not times:
        raise ValueError("The saved selection has no included optical images")
    positive = [value for value in times if value > 0]
    targets = [0.]
    if count > 1 and positive:
        nominal = [frame.get("nominal_fit_target_h") for frames in candidates.values() for frame in frames
                   if type(frame.get("nominal_fit_target_h")) in {float, int}
                   and math.isfinite(frame["nominal_fit_target_h"]) and frame["nominal_fit_target_h"] > 0]
        first, last = min(nominal or positive), max(positive)
        first = min(first, last)
        targets += ([last] if count == 2 else [
            math.exp(math.log(first) + index / (count - 2) * (math.log(last) - math.log(first)))
            for index in range(count - 1)])
        targets[-1] = last  # Preserve the exact endpoint despite log rounding.
    targets = list(dict.fromkeys(targets))
    chosen = {}
    def source_key(frame):
        return frame.get("raw_sha256") or frame["raw_path"]
    for run, frames in candidates.items():
        last = max(frame["elapsed_h"] for frame in frames)
        seen, selected = set(), []
        for target in targets:
            if target > last:
                continue
            frame = min(frames, key=lambda frame: (abs(frame["elapsed_h"] - target),
                frame["elapsed_h"], frame["current_frame_index"], source_key(frame)))
            key = source_key(frame), frame["current_frame_index"]
            if key not in seen:
                selected.append(dict(frame=deepcopy(frame), target_h=target))
                seen.add(key)
        chosen[run] = sorted(selected, key=lambda row: (row["frame"]["elapsed_h"],
            row["frame"]["current_frame_index"]))
    return dict(method=FRAME_SELECTION_METHOD, max_images_per_run=count,
                target_hours=targets, runs=chosen)


def _member(binding, role, run, reader):
    value = reader.json(binding)
    if value.get("format") == "darsia-pooled-optical-run-inputs-v1":
        if value.get("role") != role or run not in value.get("runs", {}):
            raise ValueError("The optical input bundle lacks its exact run and role")
        return value["runs"][run]
    return binding


def recipe_for_runs(rows, selection, folder, *, sampling=None, max_images_per_run=DEFAULT_IMAGES_PER_RUN):
    """Bind optical paths to every included run, independently of mass-fit roles."""
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    expected = [row["run"] for row in selection["runs"]]
    if [row["run"] for row in rows] != expected or len(set(expected)) != len(expected):
        raise ValueError("Prepared optical inputs differ from the selected run cohort")
    first = rows[0]
    if any(row["basis"]["adopted_revision"] != first["basis"]["adopted_revision"]
           or row["roles"] != first["roles"] or row["flash_bounds"] != first["flash_bounds"] for row in rows):
        raise ValueError("Shared optical paths need one adopted facies table and flash convention")
    # The pooled optical API calls its source cohort fit_runs. Here that is
    # optical fitting across the included series, not the mass objective.
    fit_runs = list(expected)
    if not fit_runs:
        raise ValueError("The saved selection has no included runs for optical paths")
    frame_selection = optical_frame_selection(selection, max_images_per_run)
    def bundle(role):
        values = {row["run"]: row["basis"][role] for row in rows}
        return _write_json(path / (role + ".json"), dict(
            format="darsia-pooled-optical-run-inputs-v1", role=role, runs=values))
    runs = {}
    for row, chosen in zip(rows, selection["runs"]):
        for frame in chosen["selected_frames"]:
            case = row["run"] + "-" + frame["raw_sha256"][:20]
            if row["source_rows"][case]["sha256"] != frame["raw_sha256"]:
                raise ValueError("Prepared optical sources differ from the selected RAW identities")
        frames = [dict(case_id=row["run"] + "-" + chosen_frame["frame"]["raw_sha256"][:20],
            frame_index=chosen_frame["frame"]["current_frame_index"], elapsed_h=chosen_frame["frame"]["elapsed_h"],
            target_h=chosen_frame["target_h"],
            source=deepcopy(row["source_rows"][row["run"] + "-" + chosen_frame["frame"]["raw_sha256"][:20]]))
            for chosen_frame in frame_selection["runs"][row["run"]]]
        if not frames:
            raise ValueError("An included run has no post-injection optical frames")
        runs[row["run"]] = dict(baseline=row["basis"]["baseline"],
            clearwater=row["basis"]["clearwater"], profile=row["basis"]["profile"], frames=frames)
    sampling = deepcopy(sampling or DEFAULT_SAMPLING)
    if len(fit_runs) == 1:
        sampling["min_runs"] = 1
    return optical.recipe_from_document(dict(format=optical.FORMAT, method=optical.METHOD,
        fit_runs=fit_runs, excluded_runs=[run for run in expected if run not in fit_runs],
        roles=first["roles"], runs=runs,
        adopted_revision=first["basis"]["adopted_revision"],
        build_context=bundle("build_context"), phase_configuration=bundle("phase_configuration"),
        native_configuration=bundle("native_configuration"), sampling=sampling,
        flash_bounds=deepcopy(first["flash_bounds"]), path_name_prefix="Current image basis · facies "))


def sample_local_frame(pipeline, spec, recipe):
    """Reuse N1's checked physical load without reading any old optical model."""
    if (not isinstance(pipeline, N1Pipeline) or not pipeline.frame_loading_only
            or not isinstance(pipeline._local_build, LocalBuildInputs)):
        raise TypeError("Current optical sampling requires the original local Build frame loader")
    local = pipeline._local_build
    frame = pipeline.load_frame(spec)[0]
    if pipeline._local_prepared is None or pipeline._local_prepared[0] is not frame:
        raise ValueError("Optical samples lack the original checked local physical frame")
    value = recipe.document
    if frame.run_id not in value["fit_runs"]:
        raise ValueError("Only the included optical runs may contribute optical samples")
    selected = [row for row in value["runs"][frame.run_id]["frames"]
                if row["case_id"] == spec["case_id"]]
    if len(selected) != 1:
        raise ValueError("The optical recipe does not select this source frame")
    phase_spec = local.frames[spec["case_id"]]
    raw = phase_spec["source"]["source_binding"]["raw_acquisition"]
    if raw != selected[0]["source"]:
        raise ValueError("The optical recipe and checked P1 frame name different RAW acquisitions")
    for role, actual in (("build_context", local.operation_result),
                         ("phase_configuration", local.config["phase"]["configuration"]),
                         ("native_configuration", local.config["native"]["configuration"])):
        if _member(value[role], role, frame.run_id, local.reader) != actual:
            raise ValueError("The checked physical frame uses another optical " + role)
    if value["adopted_revision"] != local.context.provenance["adopted_revision"]:
        raise ValueError("The optical recipe uses another adopted geometry")
    # load_frame has just executed the original LocalBuild area, material,
    # domain, phase-QA and exclusion checks. This describes that actual call;
    # it does not promote an unverified collector or add another approval.
    provenance = deepcopy(frame.provenance)
    provenance.update(physical_context_verified=True,
        physical_context_verification="original N1 load_frame with LocalBuildInputs",
        optical_source_contract=optical._source_contract(value, frame.run_id, selected[0]))
    return optical.sample_frame_for_common_paths(replace(frame, provenance=provenance),
                                                recipe, case_id=spec["case_id"])


def save_samples(folder, batch):
    """Persist only sampled RGB triples, never duplicate full RAW/frame arrays."""
    path = Path(folder)
    arrays = _write_npz(path / "samples.npz", {str(key): value for key, value in batch.samples.items()})
    return _write_json(path / "samples.json", dict(format="darsia-current-optical-samples-v1",
        recipe_sha256=batch.recipe_sha256, run=batch.run, case_id=batch.case_id,
        arrays=arrays, identities=batch.identities, audit=batch.audit,
        frame_provenance_json=batch.frame_provenance_json))


def load_samples(binding, reader):
    value = reader.json(binding)
    if value.get("format") != "darsia-current-optical-samples-v1":
        raise ValueError("Expected current optical samples from a completed physical frame")
    samples = {int(key): reader.array(dict(value["arrays"], array_key=key))
               for key in value["identities"]}
    return optical.FrameSamples(value["recipe_sha256"], value["run"], value["case_id"],
        samples, {int(key): val for key, val in value["identities"].items()},
        value["audit"], value["frame_provenance_json"])
