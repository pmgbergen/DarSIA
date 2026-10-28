"""Retry a retired first Apply failure without losing its unapplied export."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json

from .job_management import _identity, _retirement_binding
from .pipeline.job_queue import descriptor, name, read_json


FORMAT = "darsia-setup-failed-before-apply-v1"
COMPLETED_FORMAT = "darsia-setup-failed-after-completed-apply-v1"


def _metadata_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _version(path):
    _identity(path, directory=False)
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _version_files(folder):
    """Retain named existing outputs by identity/version, without raster reads."""
    result = {}
    _identity(folder, directory=True)
    for path in folder.rglob("*"):
        _identity(path, directory=path.is_dir())
        if path.is_file():
            if len(result) >= 8192:
                raise ValueError("Completed Apply output inventory exceeds its bounded file count")
            result[path.relative_to(folder).as_posix()] = _version(path)
    return result


def completed_apply_snapshot(root, plan):
    """Admit only original committed Apply and exact remaining Build controls."""
    from .pipeline.series_setup_jobs import _control
    from .pipeline.capacity import read_bound
    from .pipeline.series_setup_inputs import _same
    from .corrected_setup import EXPORT_FILES
    from .storage_paths import resolve_storage_path
    from .setup_completed_apply import read_completed_apply
    root = Path(root)
    coordinator_path = root / "setup-controls/apply-complete.json"
    if not coordinator_path.exists():
        return None
    config, _ = _control(root / "setup-controls/apply-operation.json")
    revision = read_bound(plan["adopted_revision"])[0]
    completed = read_completed_apply(config, plan, revision)
    if completed is None or Path(completed["bound"]["path"]).resolve() != coordinator_path.resolve():
        raise ValueError("Completed Apply retry requires its original coordinator")
    metadata = {}
    for pin in completed["metadata"]:
        physical = resolve_storage_path(pin["path"])
        version = _version(physical)
        if version[2] != pin["size_bytes"]:
            raise ValueError("A completed Apply metadata source changed")
        metadata[pin["path"]] = dict(descriptor=deepcopy(pin), physical=str(physical), version=version)
    exports = root / "exports"
    base = exports / "native_setup" / revision["revision_id"]
    allowed = {str(Path(row["physical"]).relative_to(exports)).replace("\\", "/")
               for row in metadata.values() if Path(row["physical"]).is_relative_to(exports)}
    rows = {row["run_id"]: row for row in completed["runs"]}
    if set(rows) != set(plan["runs"]):
        raise ValueError("Completed Apply retry changed its exact run scope")
    for run in plan["runs"]:
        directory = base if run == plan["display_reference_run"] else base / "runs" / run
        if Path(rows[run]["export"]["path"]).resolve() != (directory / "export.json").resolve():
            raise ValueError("Completed Apply export is outside its original job run")
        receipt, _ = _control(directory / "export.json")
        if set(receipt.get("files", {})) != EXPORT_FILES:
            raise ValueError("Completed Apply export has unexpected output roles")
        for filename, pin in receipt["files"].items():
            path = directory / filename
            if (resolve_storage_path(pin["path"]) != path or _version(path)[2] != pin["size_bytes"]):
                raise ValueError("A retained Apply export changed its named output")
            allowed.add(path.relative_to(exports).as_posix())
    saved_exports = _version_files(exports)
    if set(saved_exports) != allowed:
        raise ValueError("Completed Apply exports contain partial, unknown or missing work")
    controls = root / "setup-controls"
    build_controls = {}
    build_root = controls / "build"
    if build_root.exists():
        _identity(build_root, directory=True)
        for directory in build_root.iterdir():
            _identity(directory, directory=True)
            run = directory.name
            if run not in rows:
                raise ValueError("Retained Build controls belong to another run")
            expected = {"raw-build-request.json", "operation.json"}
            actual = {path.name for path in directory.iterdir()}
            if not actual or not actual <= expected or "raw-build-request.json" not in actual:
                raise ValueError("Retained Build controls contain partial or unknown work")
            build = next(value for value in plan["build_plans"] if value["run_id"] == run)
            request = dict(deepcopy(build["request_template"]), configuration_files=deepcopy(rows[run]["configuration_files"]))
            observed, request_pin = _control(directory / "raw-build-request.json")
            if not _same(observed, request):
                raise ValueError("Retained Build request differs from its exact completed Apply template")
            build_controls[request_pin["path"]] = request_pin
            if "operation.json" in actual:
                operation = dict(format=config["format"], run=run, operation="build",
                    **{key: deepcopy(config[key]) for key in ("project_root", "project_manifest", "adopted_revision", "series_manifest", "setup_plan")},
                    **{key: deepcopy(build[key]) for key in ("native_parameters", "target_machine", "metadata_output", "output_scope")},
                    configuration_files=deepcopy(rows[run]["configuration_files"]), raw_build_request=request_pin,
                    apply_completion=deepcopy(rows[run]["apply_completion"]), coordinator_completion=deepcopy(completed["bound"]))
                observed, pin = _control(directory / "operation.json")
                if not _same(observed, operation):
                    raise ValueError("Retained Build operation differs from its completed Apply controls")
                build_controls[pin["path"]] = pin
    return dict(format=COMPLETED_FORMAT, coordinator=deepcopy(completed["bound"]), metadata=metadata,
                exports=saved_exports, build_controls=build_controls,
                guarded_geometry_sha256=_metadata_digest(completed["guarded_geometry"]),
                build_versions=_version_files(build_root) if build_root.exists() else None)


def _completed_operation_result(path, root, job, plan, snapshot):
    """Accept only the original Apply metadata returned before queue publication."""
    from .pipeline.series_setup_inputs import _same
    value = read_json(path)
    children = []
    for build in plan["build_plans"]:
        directory = Path(root) / "setup-controls/build" / build["run_id"]
        pins = snapshot["build_controls"]
        request, operation = (pins.get(str(directory / leaf)) for leaf in
                              ("raw-build-request.json", "operation.json"))
        if request is None or operation is None:
            raise ValueError("Apply result lacks its complete original Build controls")
        children.append(dict(run_id=build["run_id"], configuration=operation,
                             raw_build_request=request, target_machine=build["target_machine"]))
    config = read_json(Path(root) / "setup-controls/apply-operation.json")
    expected = dict(format="darsia-series-setup-result-v1", complete=True, pipeline="series-setup",
        run="setup-apply", operation="apply", backend="cpu", precision=job["precision"],
        setup_plan=config["setup_plan"], worker_references_closed=True,
        coordinator_completion=snapshot["coordinator"], children=children,
        Build_run=False, optical_calibration_run=False)
    if (set(value) != set(expected) | {"guarded_geometry"}
            or not _same({key: value[key] for key in expected}, expected)
            or _metadata_digest(value["guarded_geometry"]) != snapshot["guarded_geometry_sha256"]):
        raise ValueError("Retained Apply operation result differs from its committed controls")


def verify_completed_apply_snapshot(root, snapshot):
    """Recheck the admitted metadata/output versions before a code-only retry."""
    from .storage_paths import resolve_storage_path
    root = Path(root)
    for row in snapshot["metadata"].values():
        physical = resolve_storage_path(row["descriptor"]["path"])
        if str(physical) != row["physical"] or _version(physical) != row["version"]:
            raise ValueError("The completed Apply metadata changed before code update")
    if _version_files(root / "exports") != snapshot["exports"]:
        raise ValueError("The completed Apply outputs changed before code update")
    build = root / "setup-controls/build"
    actual = _version_files(build) if build.exists() else None
    if actual != snapshot["build_versions"]:
        raise ValueError("The retained Build controls changed before code update")


def _completed_apply_progress(lease, token, plan):
    """Metadata stages are admissible only after the coordinator proved Apply."""
    names, notification = set(), 0
    indexed = {"capture_configuration_sources", "reverify_saved_geometry", "prepare_build_requests"}
    plain = {"apply", "reuse_completed_apply"}
    for sequence, path in enumerate(sorted(lease.glob("progress_*.json"))):
        value = read_json(path)
        event = value.get("event", {})
        step = event.get("step")
        keys = {"state", "operation", "run", "step"}
        if step in indexed:
            keys |= {"run_index", "run_count"}
        if "message" in event:
            notification += 1
            keys |= {"message", "check_progress_sequence"}
            if (not isinstance(event["message"], str) or not event["message"].strip()
                    or type(event.get("check_progress_sequence")) is not int
                    or event["check_progress_sequence"] != notification):
                raise ValueError("Completed Apply source notifications changed their original sequence")
        if (path.name != f"progress_{sequence:08d}.json" or value.get("attempt_id") != token
                or type(value.get("sequence")) is not int or value["sequence"] != sequence
                or value.get("run") != "setup-apply" or set(event) != keys
                or event.get("state") != "setup_progress" or event.get("operation") != "apply"
                or event.get("run") not in plan["runs"] or step not in indexed | plain):
            raise ValueError("Completed Apply retry refuses Build execution or unknown progress")
        if step in indexed:
            index = event.get("run_index")
            if (type(index) is not int or not 1 <= index <= len(plan["runs"])
                    or type(event.get("run_count")) is not int or event["run_count"] != len(plan["runs"])
                    or plan["runs"][index - 1] != event["run"]):
                raise ValueError("Completed Apply progress names another indexed run")
        names.add(path.name)
    if not names:
        raise ValueError("Completed Apply retry lacks its original progress")
    return names


def unapplied_export_snapshot(root, plan):
    """Preserve a first export without counting it as Apply or Build progress.

    This is operational retry admission, not scientific acceptance. Only the
    complete named first-run publication is retained; the original exporter
    still verifies every content identity when the GUI starts the next attempt.
    No raster is decoded or rehashed just to update a stopped job's code.
    """
    from .corrected_setup import EXPORT_FILES, FORMAT as EXPORT_FORMAT
    from .pipeline.capacity import read_bound
    root = Path(root)
    base = root / "exports"
    if not base.exists() or not any(base.iterdir()):
        return None
    revision = read_bound(plan["adopted_revision"])[0]
    directory = base / "native_setup" / revision["revision_id"]
    for parent, expected in ((base, {"native_setup"}),
                             (base / "native_setup", {revision["revision_id"]}),
                             (directory, EXPORT_FILES | {"export.json"})):
        _identity(parent, directory=True)
        if {path.name for path in parent.iterdir()} != expected:
            raise ValueError("Failed Setup export has Apply work, partial or unknown files; retained")
    receipt_path = directory / "export.json"
    _identity(receipt_path, directory=False)
    if receipt_path.stat().st_size > 4 * 1024**2:
        raise ValueError("Failed Setup export metadata exceeds its existing bound")
    receipt = read_json(receipt_path)
    expected_scope = dict(format="darsia-job-corrected-export-scope-v1", job_id=root.name,
        project_root=plan["project_root"], setup_plan=descriptor(root / "setup-controls" / "plan.json"))
    if (receipt.get("format") != EXPORT_FORMAT or receipt.get("schema") != 1
            or receipt.get("frame") != "rig-corrected"
            or receipt.get("revision_id") != revision["revision_id"]
            or receipt.get("run_id") != plan["display_reference_run"]
            or receipt.get("reference_run") != plan["display_reference_run"]
            or receipt.get("current_sha256") != plan["adopted_revision"]["sha256"]
            or receipt.get("setup_operation") != expected_scope
            or set(receipt.get("files", {})) != EXPORT_FILES):
        raise ValueError("Failed Setup export belongs to another job, plan or reference revision")
    result = {"export.json": descriptor(receipt_path)}
    for filename, pin in receipt["files"].items():
        path = directory / filename
        _identity(path, directory=False)
        stat = path.stat()
        if Path(pin.get("path", "")).resolve() != path.resolve() or pin.get("size_bytes") != stat.st_size:
            raise ValueError("The preserved first export changed its named output")
        result[filename] = dict(pin, file_version=[stat.st_dev, stat.st_ino, stat.st_size,
                                                  stat.st_mtime_ns, stat.st_ctime_ns])
    return result


def _files(folder):
    result = {}
    if not folder.exists():
        return result
    _identity(folder, directory=True)
    for path in folder.rglob("*"):
        _identity(path, directory=path.is_dir())
        if path.is_file():
            if path.stat().st_size > 4 * 1024**2:
                raise ValueError("Failed Setup recovery only retains bounded operational metadata")
            result[path.relative_to(folder).as_posix()] = descriptor(path)
    return result


def _initial_apply_progress(lease, token, reference):
    """Admit only the first Apply stage's original metadata notifications.

    Nested source checks publish distinct operational events while still in
    this stage. They are not scientific completions; the caller separately
    refuses Apply mutations, checkpoints and queued Build work. A complete
    unapplied first export may remain for the original consumer to verify.
    """
    paths = sorted(lease.glob("progress_*.json"))
    if not paths:
        raise ValueError("Failed Setup lacks its initial Apply progress")
    base = dict(state="setup_progress", operation="apply", run=reference, step="apply")
    names = set()
    for sequence, path in enumerate(paths):
        value = read_json(path)
        event = value.get("event", {})
        keys = set(base) if sequence == 0 else set(base) | {"message", "check_progress_sequence"}
        if (path.name != f"progress_{sequence:08d}.json" or value.get("attempt_id") != token
                or type(value.get("sequence")) is not int or value["sequence"] != sequence
                or value.get("run") != "setup-apply" or set(event) != keys
                or any(event.get(key) != expected for key, expected in base.items())
                or sequence > 0 and (type(event.get("check_progress_sequence")) is not int
                    or event["check_progress_sequence"] != sequence
                    or not isinstance(event.get("message"), str) or not event["message"].strip())):
            raise ValueError("The failed Setup passed its initial Apply stage; its work is retained")
        names.add(path.name)
    return names


def review_failed_setup(root, job, plan, stores_path, *, series_refresh=None):
    """Validate the original published failure, its empty work and actual owner.

    This is not retirement admission. The caller subsequently publishes Pause
    and invokes the original owned-process retirement verifier before moving
    any operational records.
    """
    from .jobs_metadata import JobSnapshotReader
    from .published_jobs_snapshot import _failed_publication
    from .pipeline.series_setup_protocol import validate_setup_claim
    from .pipeline.series_setup_inputs import _same
    from .pipeline.series_setup_jobs import APPLY_RUN
    root = Path(root)
    failed = _files(root / "failed")
    if not failed:
        if (root.joinpath("done", APPLY_RUN + ".json").is_file()
                and any(path.name != APPLY_RUN + ".json" for path in (root / "claimed").glob("*/*.json"))):
            from .setup_failed_build import review
            return review(root, job, plan, stores_path, series_refresh=series_refresh)
        return None
    if set(failed) != {APPLY_RUN + ".json"}:
        from .setup_failed_build import review
        return review(root, job, plan, stores_path, series_refresh=series_refresh)
    for folder in ("queue", "done", "reclaimed"):
        if _files(root / folder):
            raise ValueError("Failed Setup recovery refuses queued Build work or completed/checkpointed work")
    completed_apply = completed_apply_snapshot(root, plan)
    unapplied_export = None if completed_apply is not None else unapplied_export_snapshot(root, plan)
    base = read_json(root / "results" / APPLY_RUN / "task.json")
    validate_setup_claim(base, job)
    reader = JobSnapshotReader()
    failure, counted, times, _ = _failed_publication(root, base, root / "failed" / (APPLY_RUN + ".json"), reader.read)
    if counted or times or failure["task"].get("operation") != "apply":
        raise ValueError("This Setup failure contains scientific progress")
    task, token = failure["task"], failure["attempt_id"]
    service = name(failure["actual_worker"]["machine"])
    claimed = root / "claimed" / service / (APPLY_RUN + ".json")
    if read_json(claimed) != task:
        raise ValueError("The failed Setup claim differs from its original publication")
    lease = root / "leases" / (APPLY_RUN + "." + token)
    progress_names = (_completed_apply_progress(lease, token, plan) if completed_apply is not None
                      else _initial_apply_progress(lease, token, plan["display_reference_run"]))
    expected = {
        "claimed": {service + "/" + APPLY_RUN + ".json"},
        "failed": {APPLY_RUN + ".json"},
        "publishing": {APPLY_RUN},
        "decisions": {APPLY_RUN + "." + token + ".json"},
        "leases": {APPLY_RUN + "." + token + "/" + name for name in (
            {"owner.json", "claimed.json", "publication.json", "failure_artifacts/failed.json"} | progress_names)},
    }
    operational = {}
    for folder, names in expected.items():
        files = _files(root / folder)
        if set(files) != names:
            raise ValueError("Failed Setup recovery found other attempts or saved work in " + folder)
        operational[folder] = files
    binding = _retirement_binding(root, claimed, stores_path)
    output = binding.path
    expected_output = Path(binding.results_root) / job["job_id"] / APPLY_RUN / ".attempts" / token
    if output.resolve() != expected_output.resolve():
        raise ValueError("The failed Setup output is outside its original named attempt")
    external = _files(output)
    result_names = set()
    if "operation_result.json" in external and completed_apply is not None:
        _completed_operation_result(output / "operation_result.json", root, job, plan, completed_apply)
        result_names.add("operation_result.json")
    if set(external) != {"failed.json", "retirement.json"} | progress_names | result_names:
        raise ValueError("The failed Setup has product/checkpoint files and cannot be reset")
    detail = "Original event provenance; frame evidence is published_frame within this shared bundle"
    for filename in progress_names:
        original = read_json(output / filename)
        shared = read_json(lease / filename)
        if not _same(shared, dict(original, detail_scope=detail)):
            raise ValueError("The retained source-check progress differs from its original publication")
    if output.joinpath("failed.json").read_bytes() != (root / "failed" / (APPLY_RUN + ".json")).read_bytes():
        raise ValueError("The retained local failure differs from its original publication")
    return dict(format=COMPLETED_FORMAT if completed_apply is not None else FORMAT,
        attempt_id=token, claim=str(claimed),
        original_output=str(output), output_files=external, operational=operational,
        unapplied_export=unapplied_export, completed_apply=completed_apply)


def verify_failed_setup(root, review, *, expected_pause, stores_path):
    """Reuse the original proof of every owned process having retired."""
    from .pipeline.job_queue import retired_claim_evidence
    confirmation = None
    if any((Path(root) / "ui/launches").glob("*.local-stop.json")):
        from .job_local_stop import verified_local_stop, verified_local_stop_claim
        confirmation = verified_local_stop(root, stores_path=stores_path)
    proofs = []
    for attempt in review.get("attempts", [review]):
        claimed = Path(attempt["claim"])
        if confirmation is not None:
            proofs.append(verified_local_stop_claim(root, claimed, expected_stop=expected_pause,
                confirmation=confirmation, stores_path=stores_path))
        elif attempt.get("completed") is True:
            from .job_lifecycle import require_service_retirement
            from .pipeline.worker import _same_process
            from .job_control import read_control
            require_service_retirement(root)
            control = read_control(root)
            if (control["action"] != "pause" or control["command"] != expected_pause
                    or _same_process(attempt["actual_worker"]) is not None
                    or descriptor(Path(attempt["completion"]["path"])) != attempt["completion"]):
                raise ValueError("The completed Apply worker or its original completion changed")
            proofs.append(dict(completed=attempt["completion"],
                               actual_worker=attempt["actual_worker"], stop=expected_pause))
        else:
            binding = _retirement_binding(root, claimed, stores_path)
            proofs.append(retired_claim_evidence(root, claimed, local_output=binding,
                expected_stop=expected_pause, expected_action="pause"))
        if _files(Path(attempt["original_output"])) != attempt["output_files"]:
            raise ValueError("Failed Setup metadata changed after its retirement review")
    return proofs if "attempts" in review else proofs[0]
