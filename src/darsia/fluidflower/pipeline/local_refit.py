"""Standalone local refit plans and serial execution, without a queue or provider.

Capacity figures are estimates only. Original statistics, rules, numerical
runner and frozen-calibration files remain the scientific authority.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tomllib

from .local_refit_source import source_identity

FORMAT = "darsia-local-refit-job-v1"
RESERVE_BYTES = 1024 ** 3


def _local(path):
    # Execution is local to this machine; its durable results may be shared.
    from ..job_filesystem import inspect_results_root
    admission = inspect_results_root(Path(path).expanduser())
    if not admission.allowed:
        raise ValueError("The selected calibration results folder is unavailable: " + admission.reason)
    return Path(admission.results_root)


def _write_new(path, value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(raw)


def _reader(series_root=None):
    from .run_io import current_binding
    from ..mass.n1_pipeline import DataReader
    if current_binding() is not None:
        raise ValueError("Local refit requires direct descriptor reads outside a worker binding")
    from .current_inputs import CurrentInputReader
    return CurrentInputReader(DataReader(unc_only=False),
                              series_roots=() if series_root is None else (series_root,))


def _task(job, run):
    return dict(pipeline="n1-calibration-refit", run=run, configuration=job["plans"][run],
                commit=job["commit"], calibration_sha256=job["calibration_sha256"],
                backend="cpu", precision="float32", threads=1)


def _overflow_basis(job, plans):
    """Keep the approved overflow cohort while fitting separate run parameters."""
    if "refit" in plans:
        return plans["refit"]
    from .refit_jobs import build_plans
    first = next(iter(plans.values()))
    statistics = {run: rows for plan in plans.values() for run, rows in plan["statistics"].items()}
    return build_plans(statistics=statistics, roles=job["calibration_roles"], rules=first["rules"],
        baseline_calibration=job["baseline"], loro=False,
        chemistry_model=first.get("chemistry_model"))["refit"]


def _budgets(job, plans, reader):
    from .refit_capacity import budget
    values = {}
    for run, plan in plans.items():
        # This shape is solely an argument to the existing estimator. It is
        # never a queued task, input catalogue, provider or qualification.
        estimate_task = dict(_task(job, run), format="darsia-alias-task-v1",
            format_version=1, storage_profile="full",
            ordered_case_ids=[entry["case_id"] for key in sorted(plan["statistics"])
                              for entry in plan["statistics"][key]])
        # The estimator checks a historical 40-hex shape but never uses Git or
        # emits this token. Actual job/frozen provenance keeps the tagged SHA256.
        estimate_task["commit"] = job["source_binding"]["sha256"][:40]
        values[run] = budget(estimate_task, plan,
                            path_for_use=lambda use, pin: reader.path(pin))
    if "overflow" in job:
        from .local_overflow_refit import preparation_budget
        basis = _overflow_basis(job, plans)
        if "refit" in values:
            reference = values["refit"]
        else:
            estimate_task = dict(_task(job, next(iter(plans))), run="refit", format="darsia-alias-task-v1",
                format_version=1, storage_profile="full", commit=job["source_binding"]["sha256"][:40],
                ordered_case_ids=[entry["case_id"] for key in sorted(basis["statistics"])
                                  for entry in basis["statistics"][key]])
            reference = budget(estimate_task, basis, path_for_use=lambda use, pin: reader.path(pin))
        values["overflow_preparation"] = preparation_budget(job, basis, reference, reader=reader)
        if not job.get("overflow_phase_split_policy"):
            if job.get("calibration_scope") == "individual":
                for run in plans:
                    values["reference_without_overflow:" + run] = deepcopy(values[run])
            else:
                values["reference_without_overflow"] = deepcopy(values["refit"])
    return values


def _require_capacity(root, budgets):
    from ..execution import detect_host_resources
    parent = root
    while not parent.exists():
        parent = parent.parent
    disk = sum(row["products_estimated_bytes"] + row["transient_estimated_bytes"]
               for row in budgets.values()) + RESERVE_BYTES
    if shutil.disk_usage(parent).free < disk:
        raise ValueError("Insufficient space in the selected results folder for the refit output estimate and reserve")
    available = (detect_host_resources().get("memory") or {}).get("available_bytes")
    ram = max(row["workspace_estimated_bytes"] for row in budgets.values()) + RESERVE_BYTES
    if available is None:
        raise ValueError("Available RAM could not be read for local refit admission")
    if available < ram:
        raise ValueError("Insufficient available RAM for the original serial refit workspace estimate and reserve")


def create_local_refit_job(series_root, *, roles, rules, baseline, phase_job, jobs_root, job_id,
                          loro=True, job_identity=None, repository=None, chemistry_model=None, overflow=None,
                          calibration_scope="shared"):
    """Create fresh local plans; no fit, publication, queue or worker starts."""
    from . import refit_jobs
    from .job_queue import descriptor, name
    if name(job_id) != job_id:
        raise ValueError("Use the normalized lowercase job identifier")
    root = _local(jobs_root) / job_id
    if root.exists():
        raise FileExistsError("A local refit job requires a fresh folder: " + str(root))
    repository = Path(repository or Path(__file__).resolve().parents[4]).resolve()
    reader = _reader(series_root)
    from ..shared_calibration_cohort import current_mass_cohort
    cohort = current_mass_cohort(series_root, roles)
    reader.json(baseline)
    if overflow is not None:
        from .local_overflow_refit import read_request
        from ..mass.overflow_phase_split import POLICY
        read_request(overflow, roles=roles, reader=reader, cohort=cohort, policy=POLICY)
    identity = ((job_identity or source_identity)(repository, calibration_sha256=baseline["sha256"])
                if overflow is None else source_identity(repository, calibration_sha256=baseline["sha256"], overflow=True))
    if (identity.get("precision") != "float32" or identity.get("calibration_sha256") != baseline["sha256"]
            or re.fullmatch(r"local-source-sha256:[0-9a-f]{64}", str(identity.get("commit", ""))) is None
            or identity["commit"] != "local-source-sha256:" + identity.get("source_binding", {}).get("sha256", "")):
        raise ValueError("Local refit requires the exact CPU/float32 source identity and baseline")
    statistics, sources = {}, {}
    for run in roles["fit"]:
        if name(run) != run:
            raise ValueError("Use normalized fit-run identifiers")
        attempt = refit_jobs.completed_attempt(phase_job, run)
        completion = descriptor(Path(phase_job) / "results" / run / "completed.json")
        completed = reader.json(completion)
        if Path(completed["manifest"]["path"]).parent.resolve() != attempt.resolve():
            raise ValueError("The completed phase-one attempt changed during preparation")
        reader.json(completed["manifest"])
        statistics[run] = refit_jobs.statistics_entries(attempt, include_elapsed=True)
        sources[run] = dict(completion=completion, manifest=completed["manifest"])
    from ..calibration_image_selection import calibration_image_count, calibration_statistics_selection
    document = tomllib.loads((Path(series_root) / "series.toml").read_text(encoding="utf-8-sig"))
    statistics, rules, selection = calibration_statistics_selection(
        statistics, rules, calibration_image_count(document))
    plans = refit_jobs.build_plans(statistics=statistics, roles=roles, rules=rules,
        baseline_calibration=baseline, loro=loro, chemistry_model=chemistry_model,
        calibration_scope=calibration_scope, image_selection=selection)
    for run in plans:
        name(run)
    job = dict(deepcopy(identity), format=FORMAT, pipeline="n1-calibration-refit", job_id=job_id,
        repository=str(repository), series_manifest=descriptor(Path(series_root) / "series.toml"),
        baseline=deepcopy(baseline), phase_sources=sources, expected_runs=list(plans),
        image_selection=selection,
        backend="cpu", threads=1, plans={run: {} for run in plans},
        qualification="local execution; no provider or production qualification")
    if overflow is not None:
        job.update(overflow=deepcopy(overflow), phase_job=str(Path(phase_job).resolve()),
                   overflow_phase_split_policy=POLICY)
        if cohort is not None:
            job["calibration_cohort"] = cohort
    if calibration_scope == "individual":
        job.update(calibration_scope=calibration_scope, calibration_roles=deepcopy(roles))
    estimates = _budgets(job, plans, reader)
    _require_capacity(root, estimates)
    root.mkdir(parents=True, exist_ok=False)
    folder = root / "plans"
    folder.mkdir()
    for run, plan in plans.items():
        path = folder / (run + ".json")
        _write_new(path, plan)
        job["plans"][run] = descriptor(path)
    job["capacity_estimates"] = estimates
    _write_new(root / "job.json", job)
    return dict(complete=True, job=descriptor(root / "job.json"), pipeline=job["pipeline"],
                tasks=[dict(run=run, configuration=job["plans"][run]) for run in plans],
                numerical_calls_started=False)


def _read_job(root):
    from .job_queue import name
    job = json.loads((root / "job.json").read_bytes())
    scope = job.get("calibration_scope", "shared")
    if (job.get("format") != FORMAT or job.get("pipeline") != "n1-calibration-refit"
            or job.get("backend") != "cpu" or job.get("precision") != "float32"
            or type(job.get("threads")) is not int or job["threads"] != 1
            or job.get("job_id") != root.name or not job.get("expected_runs")
            or len(set(job["expected_runs"])) != len(job["expected_runs"])
            or set(job["plans"]) != set(job["expected_runs"])
            or scope not in {"shared", "individual"}):
        raise ValueError("An intact standalone local refit job is required")
    if scope == "individual":
        from .refit_runner import INDIVIDUAL_PREFIX
        if job["expected_runs"] != [INDIVIDUAL_PREFIX + run for run in job["calibration_roles"]["fit"]]:
            raise ValueError("Individual calibration jobs must retain their exact selected fit runs")
    for run in job["expected_runs"]:
        if name(run) != run or Path(job["plans"][run]["path"]).resolve() != root / "plans" / (run + ".json"):
            raise ValueError("Local refit plan escapes its named job")
    return job


def read_local_refit_job(job_root):
    """Read local task state and exact frozen bytes in the existing GUI shape."""
    root = _local(job_root)
    job = _read_job(root)
    reader = _reader(Path(job["series_manifest"]["path"]).parent)
    tasks, results = {}, {}
    launched = (root / "launch.json").exists()
    failed = json.loads((root / "failed.json").read_bytes()) if (root / "failed.json").exists() else None
    stopped = False
    for run in job["expected_runs"]:
        path = root / "results" / run / "completed.json"
        result = json.loads(path.read_bytes()) if path.exists() else None
        if result is not None:
            if result.get("complete") is not True or result.get("run") != run:
                raise ValueError("Local refit completion differs from its task")
            if result.get("local_source_binding") != job["source_binding"]:
                raise ValueError("Local refit result differs from its loaded numerical source binding")
            for bound in result["files"].values():
                if not Path(bound["path"]).resolve().is_relative_to(root / "results" / run):
                    raise ValueError("Local result descriptor escapes its run")
                reader.path(bound)
            results[run] = result
        state = ("completed" if result is not None else "failed" if failed else
                 "stopped" if stopped else "running" if (root / (run + "-started.json")).exists() else "waiting")
        tasks[run] = dict(state=state, step=None, latest_event=None,
            parameters=None if result is None else result.get("parameters"),
            stop_required=None if result is None else result.get("stop_required"))
        stopped = stopped or bool(result and result.get("stop_required"))
    global_result = results.get("refit", {})
    bound = global_result.get("frozen_calibration")
    frozen, error = None, ""
    if bound is not None:
        try:
            if bound != global_result["files"].get("frozen_calibration.json"):
                raise ValueError("Frozen calibration differs from the original result files")
            frozen = (reader.json(bound), deepcopy(bound))
            if frozen[0].get("code_commit") != job["commit"]:
                raise ValueError("Frozen calibration differs from its local numerical source identity")
        except (OSError, ValueError) as problem:
            error = str(problem)
    individual = {}
    if job.get("calibration_scope") == "individual":
        for run, result in results.items():
            pin = result.get("frozen_calibration")
            if pin is not None:
                value = reader.json(pin)
                if (pin != result["files"].get("frozen_calibration.json")
                        or value.get("code_commit") != job["commit"]):
                    raise ValueError("Individual calibration differs from its original result")
                individual[value["calibrated_run"]] = (value, deepcopy(pin))
    status = dict(job_root=str(root), tasks=tasks,
        complete=all(row["state"] == "completed" for row in tasks.values()),
        failed=[run for run, row in tasks.items() if row["state"] == "failed"],
        parameters=global_result.get("parameters"), stop_required=stopped,
        frozen_calibration=bound, frozen_calibration_origin=bound)
    blocker = ("This local job has already been launched; create a new job to run again." if launched else "")
    return dict(status=status, frozen=frozen, frozen_error=error, start=dict(blocker=blocker),
                local_variant_f="overflow" in job, overflow_inputs=deepcopy(job.get("overflow")),
                **(dict(calibration_scope="individual", individual_calibrations=individual) if job.get("calibration_scope") == "individual" else {}))


def run_local_refit_job(job_root, on_progress=None):
    """Run original refit tasks serially once, retaining failure/stop evidence."""
    from . import refit_jobs, refit_runner
    from threadpoolctl import threadpool_limits
    root = _local(job_root)
    job = _read_job(root)
    reader = _reader(Path(job["series_manifest"]["path"]).parent)
    if (root / "launch.json").exists():
        raise FileExistsError("This local refit job has already been launched")
    actual = source_identity(job["repository"], calibration_sha256=job["calibration_sha256"],
                             **({"overflow": True} if "overflow" in job else {}))
    for key in ("commit", "precision", "calibration_sha256", "source_binding"):
        if actual[key] != job[key]:
            raise ValueError("Local refit source identity changed: " + key)
    reader.path(job["series_manifest"])
    reader.json(job["baseline"])
    for source in job["phase_sources"].values():
        reader.json(source["completion"])
        reader.json(source["manifest"])
    plans = {run: reader.json(job["plans"][run]) for run in job["expected_runs"]}
    _require_capacity(root, _budgets(job, plans, reader))
    _write_new(root / "launch.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
        job=refit_jobs._descriptor(root / "job.json"), backend="cpu", precision="float32", threads=1))
    run = None
    try:
        with threadpool_limits(limits=1):
            overflow_plans = None
            reference_records = {}
            if "overflow" in job:
                from . import local_overflow_refit
                basis = dict(plans)
                basis.setdefault("refit", _overflow_basis(job, plans))
                overflow_plans = local_overflow_refit.prepare(job, basis, root,
                    on_progress=on_progress, reader=reader)
                folder = root / "overflow_plans"
                folder.mkdir()
                for target, plan in overflow_plans.items():
                    _write_new(folder / (target + ".json"), plan)
                # Approved F1: retain a distinct ordinary B1 fit/freeze beside F.
                references = ([] if job.get("overflow_phase_split_policy") else
                    list(plans) if job.get("calibration_scope") == "individual" else ["refit"])
                for name in references:
                    directory = root / "reference_without_overflow"
                    if job.get("calibration_scope") == "individual":
                        directory /= name
                    reference = refit_runner.run_local_refit(_task(job, name), directory, on_progress=on_progress)
                    reference_record = directory / "completed.json"
                    _write_new(reference_record, reference)
                    reference_records[name] = reference_record
            for run in job["expected_runs"]:
                output = root / "results" / run
                _write_new(root / (run + "-started.json"), dict(run=run,
                    started_utc=datetime.now(timezone.utc).isoformat()))
                if overflow_plans is None:
                    result = refit_runner.run_local_refit(_task(job, run), output, on_progress=on_progress)
                else:
                    from .job_queue import descriptor
                    task = dict(_task(job, run), configuration=descriptor(root / "overflow_plans" / (run + ".json")))
                    reference = reference_records.get(run, reference_records.get("refit"))
                    if reference is not None:
                        task["reference_without_overflow"] = descriptor(reference)
                    result = local_overflow_refit.run_fit(task, overflow_plans[run], output, on_progress=on_progress)
                # Keep the complete evidence beside every result. The original
                # frozen document points to it with its honest tagged digest.
                result["local_source_binding"] = deepcopy(job["source_binding"])
                _write_new(output / "completed.json", result)
                if result["stop_required"] and job.get("calibration_scope") != "individual":
                    break
    except BaseException as problem:
        _write_new(root / "failed.json", dict(run=run, error=type(problem).__name__ + ": " + str(problem)))
        raise
    return read_local_refit_job(root)
