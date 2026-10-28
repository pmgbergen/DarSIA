"""Pure metadata identities for the explicit existing-queue Setup operation.

An operation token is not an analysis frame or a fitted calibration. These
validators neither open inputs nor authorize Apply, RAW decoding, allocation,
publication, adoption or training. The original queue and admitted consumers
still own those boundaries. No legacy pipeline calls this positive protocol.
"""
from __future__ import annotations

import json
import ntpath
from pathlib import Path, PureWindowsPath
import re

PIPELINE = "series-setup"
FORMAT = "darsia-series-setup-operation-v1"
OPERATIONS = {"apply", "build", "image-preparation"}


def _identity(value, *, kind):
    """Require the named stage and actual worker source/compute metadata."""
    if not isinstance(value, dict) or value.get("pipeline") != PIPELINE:
        raise ValueError("An explicit series-setup " + kind + " is required")
    if value.get("format") != "darsia-alias-" + kind + "-v1":
        raise ValueError("Series Setup uses the existing explicit alias " + kind)
    commit = value.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("The Setup source commit must be a full lowercase SHA")
    if value.get("backend") != "cpu":
        raise ValueError("The original Setup producer needs the explicit CPU worker")
    if not isinstance(value.get("precision"), str) or value["precision"] not in {"float32", "float64"}:
        raise ValueError("One supported explicit Setup working precision is required")


def _absolute(value, where):
    """Check only the literal locator, without stat, resolve or alias discovery."""
    if (not isinstance(value, str) or not value or "\x00" in value
            or not (Path(value).is_absolute() or PureWindowsPath(value).is_absolute())):
        raise ValueError(where + " must be an explicit absolute path")
    return value


def _key(value):
    """Compare literal Windows locators without following the filesystem."""
    return ntpath.normcase(ntpath.normpath(value))


def _descriptor(value, where):
    """Keep an original path/SHA/size identity without reading its source."""
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError(where + " needs its exact path/SHA/size descriptor")
    _absolute(value["path"], where)
    digest = value["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(where + " needs its exact lowercase SHA-256")
    if type(value["size_bytes"]) is not int or value["size_bytes"] < 0:
        raise ValueError(where + " size must be a literal nonnegative integer")


def _caps(value):
    """Read explicit whole/per-file metadata ceilings, never an estimate."""
    if not isinstance(value, dict) or set(value) != {"maximum_output_bytes", "maximum_file_bytes"}:
        raise ValueError("Setup metadata needs its exact whole/per-file output caps")
    if any(type(value[key]) is not int or value[key] <= 0 for key in value):
        raise ValueError("Setup metadata caps must be literal positive integers")
    if value["maximum_file_bytes"] > value["maximum_output_bytes"]:
        raise ValueError("A metadata file ceiling cannot exceed the whole output ceiling")


def _output_scope(value):
    """Bind explicitly selected metadata and local scratch roots, without IO."""
    if (not isinstance(value, dict) or set(value) != {"metadata_roots", "scratch_roots"}
            or not isinstance(value["metadata_roots"], list) or not value["metadata_roots"]
            or not isinstance(value["scratch_roots"], list)):
        raise ValueError("Setup needs its explicit metadata and scratch output domains")
    for roots in value.values():
        if len({_key(_absolute(root, "Setup output root")) for root in roots}) != len(roots):
            raise ValueError("A Setup output domain is duplicated")


def validate_setup_task(task):
    """Validate a single operation's original alias-task identities, read-only."""
    _identity(task, kind="task")
    from .job_queue import name
    from ..store_map import validate_input_reference
    from .job_inputs import validate_catalog
    name(task.get("run"))
    name(task.get("target_machine"))
    operation = task.get("operation")
    if not isinstance(operation, str) or operation not in OPERATIONS or task.get("ordered_case_ids") != [operation]:
        raise ValueError("A Setup task needs one literal apply or build operation token")
    if task.get("format_version") != 1 or task.get("legacy_paths") is not False:
        raise ValueError("Setup requires the original explicit full alias-task format")
    for key in ("configuration", "series", "series_manifest", "setup_plan"):
        validate_input_reference(task.get(key))
    if not isinstance(task.get("input_catalog"), dict):
        raise ValueError("The Setup task needs its original input catalogue")
    catalog = validate_catalog(task["input_catalog"])
    if catalog.get("run") != task["run"] or catalog["configuration"] != task["configuration"]:
        raise ValueError("The Setup task and its original input catalogue differ")
    if sum(row["reference"] == task["setup_plan"] for row in catalog["inputs"]) != 1:
        raise ValueError("The Setup catalogue must consume its original immutable plan")
    return task


def validate_setup_job(job):
    """Validate a typed job without inventing a future Build configuration map."""
    _identity(job, kind="job")
    from .job_queue import name
    from ..store_map import validate_input_reference
    validate_input_reference(job.get("setup_plan"))
    runs = job.get("expected_runs")
    if (not isinstance(runs, list) or not runs
            or len({name(run) for run in runs}) != len(runs)):
        raise ValueError("The Setup job needs distinct explicit expected operation runs")
    operations = job.get("setup_operations")
    if operations is not None:
        if (not isinstance(operations, list)
                or any(not isinstance(row, dict) or set(row) != {"run", "operation"}
                       or row.get("operation") not in OPERATIONS for row in operations)
                or [row["run"] for row in operations] != runs):
            raise ValueError("The Setup job must order one Apply coordinator before its declared Builds")
        stages = [row["operation"] for row in operations]
        geometric = stages[:-1] if stages[-1] == "image-preparation" else stages
        if geometric != ["apply"] + ["build"] * (len(geometric) - 1) or (
                stages[-1] == "image-preparation" and runs[-1] != "setup-image-preparation"):
            raise ValueError("Image preparation follows all declared geometric Builds")
    return job


def validate_setup_operation(config, *, task=None):
    """Validate exact metadata roles, preserving templates until actual Apply.

    Build plans are opaque per-run templates. This function does not promote
    them into consumed RAW requests. Only the later original full-request
    reader can verify actual post-Apply configuration and writer authorities.
    """
    if not isinstance(config, dict) or config.get("format") != FORMAT:
        raise ValueError("An explicit series-setup operation document is required")
    from .job_queue import name
    run = config.get("run")
    name(run)
    operation = config.get("operation")
    common = {"format", "run", "operation", "project_root", "project_manifest",
              "adopted_revision", "series_manifest", "native_parameters", "metadata_output", "setup_plan",
              "target_machine", "output_scope"}
    if operation == "apply":
        expected = common | {"runs", "build_plans"}
    elif operation == "build":
        expected = common | {"configuration_files", "raw_build_request", "apply_completion", "coordinator_completion"}
    elif operation == "image-preparation":
        expected = common | {"runs", "job_root", "choice_authority", "output_root"}
        expected |= {key for key in ("colour_path_scope", "reuse_image_basis") if key in config}
    else:
        raise ValueError("Setup requires the explicit apply or build operation")
    if set(config) != expected:
        raise ValueError("The Setup operation has missing or unsupported metadata roles")
    _absolute(config["project_root"], "The selected project root")
    for key in ("project_manifest", "adopted_revision", "series_manifest", "native_parameters", "setup_plan"):
        _descriptor(config[key], "Setup " + key)
    if _key(config["project_manifest"]["path"]) != _key(str(PureWindowsPath(config["project_root"]) / "series.json")):
        raise ValueError("The project manifest belongs to another selected project root")
    _caps(config["metadata_output"])
    name(config["target_machine"])
    _output_scope(config["output_scope"])
    if operation == "apply":
        runs = config["runs"]
        if (not isinstance(runs, list) or not runs
                or any(not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None for value in runs)
                or len({name(value) for value in runs}) != len(runs)):
            raise ValueError("Apply needs distinct ordered actual lowercase RAW run ids")
        plans = config["build_plans"]
        if (not isinstance(plans, list) or len(plans) != len(runs)
                or any(not isinstance(plan, dict) for plan in plans)
                or [plan.get("run_id") for plan in plans] != runs):
            raise ValueError("Apply needs one build template for each run in its exact order")
        if any("raw_build_request" in plan for plan in plans):
            raise ValueError("Apply templates cannot claim future consumed RAW Build requests")
    elif operation == "image-preparation":
        if (run != "setup-image-preparation" or config["series_manifest"] != config["choice_authority"]
                or not isinstance(config["runs"], list) or not config["runs"]
                or len(set(config["runs"])) != len(config["runs"])):
            raise ValueError("Image preparation needs its exact current choice and geometric runs")
        _absolute(config["job_root"], "The existing Setup job")
        _absolute(config["output_root"], "The job-owned image preparation output")
        _descriptor(config["choice_authority"], "The saved image preparation choice")
        if "colour_path_scope" in config and config["colour_path_scope"] != "per-run":
            raise ValueError("Current Setup colour paths must be scoped separately to each run")
        if "reuse_image_basis" in config:
            if config.get("colour_path_scope") != "per-run":
                raise ValueError("Completed image reuse requires explicit per-run colour paths")
            _descriptor(config["reuse_image_basis"], "The completed image preparation being reused")
    else:
        files = config["configuration_files"]
        if not isinstance(files, list) or not files:
            raise ValueError("Build needs its actual post-Apply configuration files")
        for item in files:
            _descriptor(item, "Setup run configuration")
        paths = [_key(item["path"]) for item in files]
        if _key(config["native_parameters"]["path"]) in paths:
            raise ValueError("The native parameter JSON is distinct from the run TOML configuration")
        for key in ("raw_build_request", "apply_completion", "coordinator_completion"):
            _descriptor(config[key], "Setup " + key)
        if _key(config["coordinator_completion"]["path"]) == _key(config["apply_completion"]["path"]):
            raise ValueError("The coordinator completion is distinct from the selected run's Apply completion")
    if task is not None:
        validate_setup_task(task)
        if (config["run"] != task["run"] or operation != task["operation"]
                or config["target_machine"] != task["target_machine"]):
            raise ValueError("The Setup operation belongs to another task run or stage")
        if (config["setup_plan"]["sha256"] != task["setup_plan"]["sha256"]
                or config["setup_plan"]["size_bytes"] != task["setup_plan"]["bytes"]):
            raise ValueError("The Setup operation names another immutable plan")
    return config


def validate_setup_claim(task, job, *, config=None):
    """Join a claim to the selected immutable plan, without a calibration alias."""
    validate_setup_task(task)
    validate_setup_job(job)
    if (any(task[key] != job[key] for key in ("commit", "backend", "precision", "setup_plan"))
            or task["run"] not in job["expected_runs"]):
        raise ValueError("The Setup claim differs from the job's source, compute scope or immutable plan")
    operations = job.get("setup_operations")
    if operations is not None:
        declared = {row["run"]: row["operation"] for row in operations}
        if task["operation"] != declared[task["run"]]:
            raise ValueError("The Setup claim differs from its declared job operation")
    if config is not None:
        validate_setup_operation(config, task=task)
    return task


PLAN_FORMAT = "darsia-series-setup-plan-v1"
APPLY_COMPLETE_FORMAT = "darsia-series-setup-apply-complete-v1"


def validate_setup_plan(plan):
    """Validate the explicit operation recipe without promoting future inputs.

    Request templates retain every original science field. Only ordered TOML
    descriptors are captured later, from the successful original Apply CAS.
    The original full-request reader remains the authority after that capture.
    """
    from .job_queue import name
    expected = {"format", "project_root", "project_manifest", "adopted_revision",
                "series_manifest", "native_parameters", "runs", "display_reference_run",
                "control_root", "build_plans", "target_machine", "metadata_output", "output_scope"}
    if not isinstance(plan, dict) or set(plan) != expected or plan["format"] != PLAN_FORMAT:
        raise ValueError("An explicit complete series Setup plan is required")
    _absolute(plan["project_root"], "Setup project root")
    _absolute(plan["control_root"], "Setup control root")
    for key in ("project_manifest", "adopted_revision", "series_manifest", "native_parameters"):
        _descriptor(plan[key], "Setup plan " + key)
    if _key(plan["project_manifest"]["path"]) != _key(str(PureWindowsPath(plan["project_root"]) / "series.json")):
        raise ValueError("The Setup plan belongs to another project root")
    _caps(plan["metadata_output"])
    _output_scope(plan["output_scope"])
    name(plan["target_machine"])
    runs = plan["runs"]
    if (not isinstance(runs, list) or not runs
            or any(not isinstance(run, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run) is None for run in runs)
            or len(set(runs)) != len(runs) or plan["display_reference_run"] not in runs):
        raise ValueError("The Setup plan needs distinct actual runs and its display reference")
    builds = plan["build_plans"]
    if (not isinstance(builds, list) or len(builds) != len(runs)
            or any(not isinstance(value, dict) for value in builds)
            or [value.get("run_id") for value in builds] != runs):
        raise ValueError("The Setup build templates differ from its actual run order")
    request_fields = {"format", "run_id", "series_context", "baseline", "profile",
                      "operation_overrides", "protocol_inputs", "output", "shape_basis_certificate"}
    build_fields = {"run_id", "target_machine", "native_parameters", "configuration_paths",
                    "request_template", "metadata_output", "output_scope"}
    for build in builds:
        if set(build) not in (build_fields, build_fields | {"raw_setup_settings"}):
            raise ValueError("A Setup build template has unsupported or missing roles")
        if "raw_setup_settings" in build:
            settings = build["raw_setup_settings"]
            from ..rig_raw_input import camera_binding
            if (not isinstance(settings, dict)
                    or not {"run_id", "raw_camera", "rig_path"} <= set(settings)
                        <= {"run_id", "raw_camera", "rig_path", "analysis_exclusions", "native_product"}
                    or settings["run_id"] != build["run_id"]
                    or camera_binding(settings["raw_camera"])["run_id"] != build["run_id"]):
                raise ValueError("RAW Setup settings need the exact selected run and camera binding")
            _absolute(settings["rig_path"], "The fresh RAW rig destination")
            if _key(settings["rig_path"]) != _key(build["request_template"]["output"]["destination"]):
                raise ValueError("RAW Setup settings differ from the original output authority")
        name(build["target_machine"])
        _descriptor(build["native_parameters"], "Build native parameters")
        _caps(build["metadata_output"])
        _output_scope(build["output_scope"])
        paths = build["configuration_paths"]
        if not isinstance(paths, list) or not paths:
            raise ValueError("A Build needs its actual ordered configuration path selection")
        for path in paths:
            _absolute(path, "Build configuration path")
        request = build["request_template"]
        if (not isinstance(request, dict) or not request_fields <= set(request)
                <= request_fields | {"analysis_exclusions", "native_product"}
                or request["format"] != "darsia-raw-rig-build-request-v1"
                or request["run_id"] != build["run_id"] or request["operation_overrides"] != {}):
            raise ValueError("Only the unchanged original full RAW request may be staged after Apply")
        for field in ("analysis_exclusions", "native_product"):
            if field in request:
                if (not isinstance(request[field], dict)
                        or json.dumps(request[field], sort_keys=True, allow_nan=False)
                            != json.dumps(build.get("raw_setup_settings", {}).get(field), sort_keys=True, allow_nan=False)):
                    raise ValueError("RAW " + field + " differs from the exact Setup settings")
            elif field in build.get("raw_setup_settings", {}):
                raise ValueError("RAW " + field + " is absent from the original request")
        if ("analysis_exclusions" in request and "native_product" in request
                and json.dumps(request["analysis_exclusions"], sort_keys=True, allow_nan=False)
                    != json.dumps(request["native_product"].get("analysis_exclusions"), sort_keys=True, allow_nan=False)):
            raise ValueError("The RAW native product and analysis exclusions differ")
        context = request["series_context"]
        if (not isinstance(context, dict) or set(context) != {"root", "manifest"}
                or context["manifest"] != plan["series_manifest"]
                or _key(context["root"]) != _key(str(PureWindowsPath(plan["series_manifest"]["path"]).parent))):
            raise ValueError("The Build template differs from its original series context")
        for key in ("baseline", "profile", "shape_basis_certificate"):
            _descriptor(request[key], "RAW template " + key)
    return plan
