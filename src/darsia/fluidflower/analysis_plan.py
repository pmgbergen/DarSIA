"""Prepare and inspect explicit staged analysis DATA outside the worker.

Preparation creates fresh files only. Adoption and queue creation are separate
explicit actions. Missing frame roles and phase-one completion are never inferred.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import tomllib

from .pipeline.job_queue import descriptor, name, write_immutable


PHASE1 = "native-phase-statistics"
PHASE2 = "n1"


def _same(first, second):
    return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False)


def _same_asset_bytes(first, second):
    """A relocated metadata file keeps its exact cryptographic identity."""
    return (isinstance(first, dict) and isinstance(second, dict)
            and re.fullmatch(r"[0-9a-f]{64}", first.get("sha256", "")) is not None
            and first["sha256"] == second.get("sha256")
            and type(first.get("size_bytes")) is int
            and first["size_bytes"] == second.get("size_bytes"))


def phase1_configuration(original, *, precision, atmosphere=None, frame_data=None,
                         gas_probability=False, method_bindings=None, role_catalog=None):
    """Map complete supplied run roles onto the existing phase-one contract.

    ``frame_data`` is an exact case-id mapping of measured frame records, not a
    reference frame to copy to every acquisition. Native/photographic/domain
    roles remain the supplied originals. The optional pressure table is sampled
    by the existing N1 consumer, retaining the required per-frame atmosphere ABI.
    """
    from .pipeline.series import validate_run
    from .pipeline.staged_data import (
        PHASE1_FORMAT, STATISTICS_FORMAT, validate_phase1_run,
    )
    from .storage_outputs import REQUEST_FORMAT, LA_ENCODING

    validate_run(original)
    if precision != "float32":
        raise ValueError("A new production plan requires explicit float32")
    if type(gas_probability) is not bool:
        raise ValueError("Choose the gas probability explicitly")
    template = original["mass"]["template"]
    required = ("shared", "physical_correction", "recipe", "dimensions_m",
                "boxes", "summary", "bindings", "geometry_fields")
    statistics = {key: deepcopy(template[key]) for key in required}
    statistics["format"] = STATISTICS_FORMAT
    for key in ("atmosphere_series", "dose", "contracts", "provenance", "opencl_acceleration"):
        if key in template:
            statistics[key] = deepcopy(template[key])
    clearwater = statistics["shared"]["clearwater"]
    if "receipt" in clearwater:
        provenance = statistics.setdefault("provenance", {})
        if "original_clearwater_receipt" in provenance:
            raise ValueError("The statistics provenance already records a clear-water receipt")
        provenance["original_clearwater_receipt"] = clearwater.pop("receipt")
    if method_bindings is not None:
        keys = {"round4_spec_sha256", "amendment_sha256"}
        if not isinstance(method_bindings, dict) or set(method_bindings) != keys:
            raise ValueError("Supply the two original statistics method digests explicitly")
        for key, value in method_bindings.items():
            if key in statistics["bindings"] and statistics["bindings"][key] != value:
                raise ValueError("The explicit statistics method binding changed")
            statistics["bindings"][key] = value
    candidate = {key: deepcopy(original[key]) for key in (
        "run", "frames", "native", "phase", "history_policy",
        "selection_index_field", "ordered_case_ids")}
    candidate.update(format=PHASE1_FORMAT, precision=precision, statistics=statistics,
        storage=dict(format=REQUEST_FORMAT, resolution_mm="native",
                     gas_score_encoding=LA_ENCODING))
    # The current Build belongs to these exact selected sources and geometry.
    # Carry it to the P1 consumer; validate_phase1_run below also rejects a
    # frame_data replacement that no longer matches the embedded declaration.
    for key in ("data_aliases", "provenance", "photographs", "current_build_binding"):
        if key in original:
            candidate[key] = deepcopy(original[key])
    alias_metadata = []
    for index, alias in enumerate(candidate.get("data_aliases", ())):
        logical, physical = alias["logical"], alias["physical"]
        if set(logical) == {"path", "sha256", "bytes", "dtype", "shape"}:
            if (type(logical["bytes"]) is not int
                    or logical["bytes"] != physical.get("size_bytes")
                    or logical["sha256"] != physical.get("sha256")):
                raise ValueError("Legacy array alias byte identity differs")
            alias_metadata.append(dict(index=index, path=logical["path"],
                bytes=logical["bytes"], dtype=logical["dtype"], shape=deepcopy(logical["shape"])))
            alias["logical"] = {key: logical[key] for key in ("path", "sha256")}
            alias["logical"]["size_bytes"] = logical["bytes"]
    if alias_metadata:
        provenance = candidate.setdefault("provenance", {})
        if "legacy_alias_array_metadata" in provenance:
            raise ValueError("Legacy alias metadata was already migrated")
        provenance["legacy_alias_array_metadata"] = alias_metadata
    if frame_data is not None:
        wanted = candidate["ordered_case_ids"]
        if not isinstance(frame_data, dict) or set(frame_data) != set(wanted):
            raise ValueError("Measured frame data must cover the complete selection")
        reference = None
        for frame in candidate["frames"]:
            measured = frame_data[frame["case_id"]]
            if (measured.get("format") != "darsia-measured-frame-geometry-v1"
                    or measured.get("run") != candidate["run"]
                    or measured.get("case_id") != frame["case_id"]
                    or not _same_asset_bytes(measured.get("original_geometry"), frame["geometry"])):
                raise ValueError("Measured geometry belongs to another source frame")
            roles = frame["phase_roles"]
            if not _same(measured["source_to_frame_native"], roles["reference_to_current_h"]):
                raise ValueError("Measured geometry changes the supplied native positioning")
            if measured["phase_area"]["role"] != roles["area"]["role"]:
                raise ValueError("Measured geometry changes the supplied area route")
            matrix = measured["mass_shared_reference_matrix"]
            if reference is not None and not _same(reference, matrix):
                raise ValueError("A run needs one common physical reference matrix")
            reference = matrix
            frame["geometry"] = deepcopy(measured["worker_geometry"])
            roles["area"] = deepcopy(measured["phase_area"])
        statistics["shared"]["reference_matrix"] = deepcopy(reference)
    if atmosphere is not None:
        from .accepted_pressure_policy import verify_accepted_pressure
        verify_accepted_pressure(atmosphere)
        if atmosphere.get("run") != candidate["run"]:
            raise ValueError("Running atmosphere belongs to another run")
        from .round5_pressure import bind_running_atmosphere
        from .mass.n1_pipeline import _atmosphere
        bound = bind_running_atmosphere(
            {"mass": {"template": {}}, "frames": candidate["frames"]}, atmosphere)
        statistics["atmosphere_series"] = deepcopy(atmosphere)
        for frame, unbound in zip(candidate["frames"], bound["frames"]):
            frame["mass_spec"]["atmosphere"] = _atmosphere(
                statistics, {"elapsed_h": unbound["elapsed_h"]})
    if gas_probability:
        from .phase.probability_storage import REQUEST_FORMAT, ENCODING
        candidate["gas_probability_storage"] = dict(format=REQUEST_FORMAT, encoding=ENCODING)
    if role_catalog is not None:
        from .pipeline.job_inputs import InputCatalog
        if type(role_catalog) is not InputCatalog:
            raise ValueError("Missing input sizes require the real rooted input catalogue")
    def complete_size(bound, use, label):
        if "size_bytes" not in bound:
            if role_catalog is None:
                raise ValueError(label + " size requires the rooted input catalogue")
            reference = role_catalog.add(bound, "rig_model", use)
            bound["size_bytes"] = reference["bytes"]
    for frame in candidate["frames"]:
        use = "plan." + candidate["run"] + "." + frame["case_id"]
        area = frame["phase_roles"]["area"]
        for field in ("historical", "measured"):
            if field in area:
                complete_size(area[field], use + ".area." + field, "Physical-area input")
                hints = {key: area[field].pop(key) for key in ("dtype", "shape_hw")
                         if key in area[field]}
                if hints:
                    revision = area.setdefault("revision", {})
                    metadata = revision.setdefault("original_array_metadata", {})
                    if field in metadata:
                        raise ValueError("Physical-area metadata was already migrated")
                    metadata[field] = hints
        # Old run DATA admitted the static component's SHA without a byte count.
        # The staged contract requires that count. Obtain it through original
        # root admission and stat, never a guessed archive size or another frame.
        bound = frame["phase_roles"]["static"]["components"]
        complete_size(bound, use + ".static_components", "Static component")
    validate_phase1_run(candidate)
    return candidate


def _job(job, pipeline, runs):
    if (not isinstance(job, dict) or job.get("precision") != "float32"
            or re.fullmatch(r"[0-9a-f]{40}", job.get("commit", "")) is None):
        raise ValueError("The plan needs its actual full commit and float32 precision")
    lock = job.get("environment_lock", {})
    path = PureWindowsPath(lock.get("path", ""))
    if (set(lock) != {"path", "sha256"} or not str(lock["path"])
            or path.drive or path.root or ".." in path.parts
            or re.fullmatch(r"[0-9a-f]{64}", lock.get("sha256", "")) is None):
        raise ValueError("Bind the repository-relative environment lock")
    allowed = {"commit", "environment_lock", "precision", "created_utc", "publish"}
    if pipeline == PHASE2:
        allowed.add("calibration_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", job.get("calibration_sha256", "")) is None:
            raise ValueError("Phase two needs its fitted calibration SHA")
    if set(job) - allowed:
        raise ValueError("Unsupported plan job fields")
    if "publish" in job and (not isinstance(job["publish"], list) or any(
            not isinstance(value, str) or not value or "\\" in value
            or PureWindowsPath(value).drive or value.startswith("/")
            or any(part in {"", ".", ".."} for part in value.split("/"))
            for value in job["publish"])):
        raise ValueError("Publish must name relative products")
    stamp = datetime.fromisoformat(job["created_utc"].replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("The job creation time needs an explicit timezone")
    return dict(deepcopy(job), expected_runs=list(runs))


def _resources(choice):
    if (not isinstance(choice, dict) or set(choice) != {"backend", "threads"}
            or choice["backend"] != "cpu" or type(choice["threads"]) is not int
            or choice["threads"] < 1):
        raise ValueError("This plan builder requires explicit CPU and positive threads")


def _toml_string(value):
    if not isinstance(value, str):
        raise ValueError("Plan text fields must be literal strings")
    return json.dumps(value, ensure_ascii=True)


def _plan_text(*, pipeline, job_id, job, selection, runs):
    q = _toml_string
    lines = ['format = "darsia-series-v1"', 'adopted_for_execution = false',
             "pipeline = " + q(pipeline), "job_id = " + q(job_id), "", "[selection]"]
    for key in ("path", "sha256"):
        lines.append(key + " = " + q(selection[key]))
    lines += ["size_bytes = " + str(selection["size_bytes"]), "", "[job]"]
    for key in ("commit", "precision", "created_utc", "calibration_sha256"):
        if key in job:
            lines.append(key + " = " + q(job[key]))
    if "publish" in job:
        if (not isinstance(job["publish"], list)
                or any(not isinstance(value, str) for value in job["publish"])):
            raise ValueError("Publish must be a list of relative product names")
        lines.append("publish = [" + ", ".join(q(v) for v in job["publish"]) + "]")
    lines += ["", "[job.environment_lock]", "path = " + q(job["environment_lock"]["path"]),
              "sha256 = " + q(job["environment_lock"]["sha256"])]
    for run in runs:
        lines += ["", "[[runs]]", "run = " + q(run["run"]), "", "[runs.configuration]"]
        for key in ("path", "sha256"):
            lines.append(key + " = " + q(run["configuration"][key]))
        lines += ["size_bytes = " + str(run["configuration"]["size_bytes"]),
                  "", "[runs.resources]", "backend = " + q(run["resources"]["backend"]),
                  "threads = " + str(run["resources"]["threads"])]
    return "\n".join(lines) + "\n"


def _write_plan(folder, configurations, *, pipeline, job_id, job, resources):
    from .pipeline.staged_data import validate_phase1_run, validate_phase2_run
    validator = validate_phase1_run if pipeline == PHASE1 else validate_phase2_run
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("Supply the complete ordered run configurations")
    runs = [name(config["run"]) for config in configurations]
    if len(runs) != len(set(runs)):
        raise ValueError("The plan cannot repeat a run")
    name(job_id)
    job = _job(job, pipeline, runs)
    if not isinstance(resources, dict) or set(resources) != set(runs):
        raise ValueError("Supply the resources for every ordered run")
    for config in configurations:
        validator(config)
        atmosphere = (config["statistics"] if pipeline == PHASE1 else config).get("atmosphere_series", {})
        from .accepted_pressure_policy import verify_accepted_pressure
        verify_accepted_pressure(atmosphere)
        if atmosphere.get("provisional_gap_default") is True:
            raise ValueError(config["run"] + ": provisional pressure gaps await an explicit decision")
        choice = resources[config["run"]]
        _resources(choice)
        if config["precision"] != job["precision"]:
            raise ValueError("Run precision differs from the job")
        if pipeline == PHASE2 and config["calibration"]["sha256"] != job["calibration_sha256"]:
            raise ValueError("Phase two cannot mix fitted calibrations")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for config in configurations:
        path = folder / (config["run"] + "_configuration.json")
        write_immutable(path, deepcopy(config))
        rows.append(dict(run=config["run"], configuration=descriptor(path),
                         resources=deepcopy(resources[config["run"]])))
    selection_path = folder / "selection.json"
    write_immutable(selection_path, dict(complete=True, runs=[
        dict(run=config["run"], ordered_case_ids=deepcopy(config["ordered_case_ids"])
             if pipeline == PHASE1 else [f["case_id"] for f in config["frames"]])
        for config in configurations]))
    path = folder / "execution.toml"
    text = _plan_text(pipeline=pipeline, job_id=job_id, job=job,
                      selection=descriptor(selection_path), runs=rows)
    with path.open("xb") as stream:
        stream.write(text.encode("utf-8"))
    return dict(plan=descriptor(path), pipeline=pipeline, runs=rows,
                adopted_for_execution=False, job_created=False, worker_started=False)


def write_phase1_plan(folder, configurations, *, job_id, job, resources):
    """Write complete phase-one inputs into a fresh inert plan folder."""
    return _write_plan(folder, configurations, pipeline=PHASE1,
                       job_id=job_id, job=job, resources=resources)


def write_phase2_plan(folder, sources, *, job_id, job, resources, reader):
    """Seal from genuine phase-one completion, then write an inert N1 plan.

    Every source contains the arguments of the existing completion verifier and
    sealer. This function has no path that manufactures a completion receipt.
    """
    from .pipeline.staged_data import seal_phase2_configuration
    from .mass.n1_pipeline import DataReader
    if type(reader) is not DataReader:
        raise ValueError("Phase two requires the genuine bound DATA reader")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Supply the complete ordered genuine phase-one sources")
    target = Path(folder)
    if target.exists() or target.with_name(target.name + "_sealed").exists():
        raise FileExistsError("Choose a fresh plan and seal namespace")
    sealed = target.with_name(target.name + "_sealed")
    sealed.mkdir(parents=True, exist_ok=False)
    configurations = []
    for index, source in enumerate(sources):
        result = seal_phase2_configuration(sealed / (str(index) + ".json"),
                                          reader=reader, **deepcopy(source))
        configurations.append(reader.json(result["configuration"]))
    return _write_plan(target, configurations, pipeline=PHASE2,
                       job_id=job_id, job=job, resources=resources)


def read_staged_plan(path, *, expected_sha256, require_adopted=True):
    """Read prepared or adopted staged metadata without running a pipeline."""
    from .jobs_metadata import JobSnapshotReader
    from .pipeline.staged_data import validate_phase1_run, validate_phase2_run
    path = Path(path)
    bound = descriptor(path)
    if bound["sha256"] != expected_sha256:
        raise ValueError("Execution plan bytes changed")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Execution plan changed while read")
    value = tomllib.loads(raw.decode("utf-8"))
    from .pipeline.series import _no_executable_configuration
    _no_executable_configuration(value)
    pipeline = value.get("pipeline")
    if value.get("format") != "darsia-series-v1" or pipeline not in {PHASE1, PHASE2}:
        raise ValueError("Choose an explicit staged analysis plan")
    if require_adopted and value.get("adopted_for_execution") is not True:
        raise ValueError("The prepared plan is inert until explicitly adopted")
    rows = value["runs"]
    run_ids = [name(row["run"]) for row in rows]
    if not run_ids or len(run_ids) != len(set(run_ids)):
        raise ValueError("The plan requires distinct ordered runs")
    _job(value["job"], pipeline, run_ids)
    reader = JobSnapshotReader()
    selection = reader.read(Path(value["selection"]["path"]), value["selection"])
    if (selection.get("complete") is not True
            or [row["run"] for row in selection["runs"]] != run_ids):
        raise ValueError("The plan differs from the complete fixed selection")
    configs = []
    for row, chosen in zip(rows, selection["runs"]):
        _resources(row["resources"])
        validator = validate_phase1_run if pipeline == PHASE1 else validate_phase2_run
        config = validator(reader.read(
            Path(row["configuration"]["path"]), row["configuration"]))
        if (config["run"] != row["run"] or config["precision"] != value["job"]["precision"]
                or [f["case_id"] for f in config["frames"]] != chosen["ordered_case_ids"]):
            raise ValueError("Run configuration differs from the fixed plan selection")
        if pipeline == PHASE2 and config["calibration"]["sha256"] != value["job"]["calibration_sha256"]:
            raise ValueError("Phase two cannot mix fitted calibrations")
        configs.append(config)
    return value, configs, bound


def read_phase1_plan(path, *, expected_sha256, require_adopted=True):
    """Positive phase-one reader; the unchanged worker's series reader is unused."""
    result = read_staged_plan(path, expected_sha256=expected_sha256,
                              require_adopted=require_adopted)
    if result[0]["pipeline"] != PHASE1:
        raise ValueError("Choose an explicit native phase-statistics plan")
    return result


def adopt_prepared_plan(source, destination, *, expected_sha256, actor, reason):
    """An explicit review action writes a separate adopted plan; no job starts."""
    if any(not isinstance(v, str) or not v.strip() for v in (actor, reason)):
        raise ValueError("Record the actual approving actor and reason")
    source, destination = Path(source), Path(destination)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("The reviewed prepared plan changed")
    value = tomllib.loads(raw.decode("utf-8"))
    if (value.get("format") != "darsia-series-v1"
            or value.get("pipeline") not in {PHASE1, PHASE2}
            or value.get("adopted_for_execution") is not False):
        raise ValueError("Choose an inert staged analysis plan for review")
    read_staged_plan(source, expected_sha256=expected_sha256, require_adopted=False)
    line = b"adopted_for_execution = false\n"
    if raw.count(line) != 1:
        raise ValueError("Prepared adoption field is not the original literal")
    approved = raw.replace(line, b"adopted_for_execution = true\n", 1)
    receipt = dict(prepared_sha256=expected_sha256, actor=actor, reason=reason,
                   approved_utc=datetime.now(timezone.utc).isoformat())
    approved += ("\n[adoption]\n" + "\n".join(
        key + " = " + _toml_string(value) for key, value in receipt.items()) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(approved)
    return dict(plan=descriptor(destination), approval=receipt,
                adopted_for_execution=True, job_created=False, worker_started=False)
