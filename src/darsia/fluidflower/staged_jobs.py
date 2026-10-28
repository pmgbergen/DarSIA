"""Positive GUI admission for explicit phase-one plans, outside the worker."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import socket
import tomllib

from .analysis_plan import PHASE1, read_phase1_plan
from .pipeline.job_queue import descriptor, write_immutable


def plan_pipeline(plan):
    """Read the bound plan before choosing a metadata route."""
    bound = {key: plan[key] for key in ("path", "sha256", "size_bytes")}
    path = Path(bound["path"])
    if descriptor(path) != bound:
        raise ValueError("Selected execution plan changed")
    value = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    if descriptor(path) != bound:
        raise ValueError("Selected execution plan changed while read")
    return value.get("pipeline", "native-phase-n1")


def read_execution_plan(path, *, expected_sha256):
    """Keep the existing reader for existing plans; select phase one positively."""
    bound = descriptor(Path(path))
    if bound["sha256"] != expected_sha256:
        raise ValueError("Execution plan changed")
    if plan_pipeline(bound) == PHASE1:
        return read_phase1_plan(path, expected_sha256=expected_sha256)
    from .pipeline.series import read_series
    return read_series(path, expected_sha256=expected_sha256)


def phase1_route_blocker():
    """Name the current source admission; never substitute another pipeline."""
    if importlib.util.find_spec("darsia.fluidflower.pipeline.phase1_capacity") is None:
        return "Phase-one capacity admission is not installed; its final resource qualification is pending."
    return ""


def phase1_inputs(plan, *, series_manifest):
    """Build original rooted catalogue metadata without creating a queue."""
    from .pipeline.alias_job import _manifest
    from .pipeline.job_inputs import InputCatalog, validate_catalog
    from .pipeline.phase1_protocol import bind_phase1_identity
    value, configs, bound = read_phase1_plan(plan["path"], expected_sha256=plan["sha256"])
    if bound != {key: plan[key] for key in ("path", "sha256", "size_bytes")}:
        raise ValueError("The plan descriptor changed")
    _, manifest, roots = _manifest(series_manifest)
    admission = InputCatalog(roots)
    parent = admission.add(bound, "configuration", "execution_series")
    manifest_ref = admission.add(series_manifest, "configuration", "series_manifest")
    job = dict(deepcopy(value["job"]), format="darsia-alias-job-v1", pipeline=PHASE1,
               expected_runs=[row["run"] for row in value["runs"]])
    tasks, rooted = [], []
    for row, config in zip(value["runs"], configs):
        from .accepted_pressure_policy import verify_accepted_pressure
        verify_accepted_pressure(config["statistics"].get("atmosphere_series", {}))
        if config["statistics"].get("atmosphere_series", {}).get("provisional_gap_default") is True:
            raise ValueError(config["run"] + ": provisional pressure gaps await an explicit decision")
        task = dict(deepcopy(row["resources"]), format="darsia-alias-task-v1",
            pipeline=PHASE1, run=config["run"], configuration=deepcopy(row["configuration"]),
            commit=job["commit"], precision=job["precision"], job_id=value["job_id"])
        catalog = InputCatalog(roots, config.get("data_aliases", ()))
        catalog.add(bound, "configuration", "run.execution_series")
        catalog.add(series_manifest, "configuration", "run.series_manifest")
        inputs = validate_catalog(catalog.collect(row["configuration"], config))
        inputs["runtime_support"] = "darsia-role-io-runtime-v1"
        rooted.append(dict(deepcopy(task), format_version=1, legacy_paths=False,
            execution_ready=True, execution_blocker="", configuration=inputs["configuration"],
            series=parent, series_manifest=manifest_ref, input_catalog=inputs,
            ordered_case_ids=deepcopy(config["ordered_case_ids"])))
        tasks.append(task)
    # The original identity validator proves the same whole-run stage and order.
    bind_phase1_identity(job, rooted)
    return value, configs, job, tasks, rooted, manifest


def inspect_phase1_resources(plan, *, series_root, stores_path=None):
    """Read the installed route and original qualified per-run capacity profile."""
    blocker = phase1_route_blocker()
    if blocker:
        return dict(ready=False, reason=blocker)
    try:
        from .store_map import read_store_map
        from .pipeline.capacity import load_estimate
        manifest = descriptor(Path(series_root) / "series.toml")
        value, configs, _, _, rooted, registered = phase1_inputs(plan, series_manifest=manifest)
        stores = read_store_map(stores_path, forbidden_roots=(series_root, Path(__file__).resolve().parents[3]))
        estimates = [load_estimate(stores, registered["series_id"], task, config,
            repository=Path(__file__).resolve().parents[3]) for task, config in zip(rooted, configs)]
        return dict(ready=True, reason="The original result profiles admit this phase-one DATA; Start repeats fresh result/cache checks.",
                    runs=[config["run"] for config in configs],
                    products_expected_bytes=sum(row.products_estimated_bytes for row in estimates),
                    store_source=deepcopy(stores.source), execution_plan=value["job_id"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(ready=False, reason=str(error))


def inspect_staged_resources(plan, *, series_root, stores_path=None):
    """Ask the original profile estimator about either explicit staged route."""
    if plan_pipeline(plan) == PHASE1:
        return inspect_phase1_resources(plan, series_root=series_root, stores_path=stores_path)
    try:
        from .pipeline.alias_job import _manifest, read_alias_series
        from .pipeline.job_inputs import InputCatalog, validate_catalog
        from .pipeline.capacity import load_estimate
        from .store_map import read_store_map
        manifest = descriptor(Path(series_root) / "series.toml")
        _, registered, roots = _manifest(manifest)
        value, configs, bound = read_alias_series(plan["path"], expected_sha256=plan["sha256"], series_manifest=manifest)
        if value.get("pipeline") != "n1":
            raise ValueError("Choose an explicit staged mass plan")
        rooted = []
        for row, config in zip(value["runs"], configs):
            catalog = InputCatalog(roots, [*value.get("data_aliases", ()), *config.get("data_aliases", ())])
            parent = catalog.add(bound, "configuration", "run.execution_series")
            registered_ref = catalog.add(manifest, "configuration", "run.series_manifest")
            inputs = validate_catalog(catalog.collect(row["configuration"], config))
            inputs["runtime_support"] = "darsia-role-io-runtime-v1"
            rooted.append(dict(deepcopy(row["resources"]), format="darsia-alias-task-v1", format_version=1,
                pipeline="n1", run=config["run"], commit=value["job"]["commit"], precision=value["job"]["precision"],
                calibration_sha256=value["job"]["calibration_sha256"], job_id=value["job_id"],
                legacy_paths=False, execution_ready=True, execution_blocker="", storage_profile="full",
                configuration=inputs["configuration"], series=parent, series_manifest=registered_ref,
                input_catalog=inputs, ordered_case_ids=[f["case_id"] for f in config["frames"]]))
        stores = read_store_map(stores_path, forbidden_roots=(series_root, Path(__file__).resolve().parents[3]))
        estimates = [load_estimate(stores, registered["series_id"], task, config,
            repository=Path(__file__).resolve().parents[3]) for task, config in zip(rooted, configs)]
        return dict(ready=True, reason="The original result profiles admit this staged mass DATA; Start repeats fresh checks.",
                    products_expected_bytes=sum(row.products_estimated_bytes for row in estimates))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(ready=False, reason=str(error))


def create_analysis_job(plan, job_root, *, stores_path=None, **kwargs):
    """Use the original GUI creator unless this is a positive phase-one plan."""
    pipeline = plan_pipeline(plan)
    if pipeline != PHASE1:
        from .jobs_metadata import create_job_from_plan
        if pipeline == "n1":
            if kwargs.get("storage_profile") not in (None, "full"):
                raise ValueError("This staged mass route requires Full")
            admission = inspect_staged_resources(plan, series_root=kwargs.get("series_root"), stores_path=stores_path)
            if not admission["ready"]:
                raise ValueError("Staged mass creation is blocked: " + admission["reason"])
            kwargs["storage_profile"] = "full"
        return create_job_from_plan(plan, job_root, **kwargs)
    if kwargs.get("storage_profile") not in (None, "full"):
        raise ValueError("Phase one requires Full; Lean is not admitted for this route")
    root = Path(job_root)
    if root.exists():
        raise FileExistsError("Choose a fresh job folder")
    from .jobs_metadata import validate_current_selection
    from .pipeline.alias_job import (
        prepare_phase1_alias_job, portable_selection_binding, job_metadata_reference,
    )
    series_root, series_id = kwargs.get("series_root"), kwargs.get("series_id")
    if not series_root or not series_id:
        raise ValueError("Connect the current registered series first")
    manifest = descriptor(Path(series_root) / "series.toml")
    binding = validate_current_selection(plan, series_root=series_root, series_id=series_id,
        selection_provider=kwargs.get("selection_provider"), admitter=read_phase1_plan)
    resource = inspect_phase1_resources(plan, series_root=series_root, stores_path=stores_path)
    if resource["ready"] is not True:
        raise ValueError("Phase-one job creation is blocked: " + resource["reason"])
    value, configs, job, tasks, _, registered = phase1_inputs(plan, series_manifest=manifest)
    if registered["series_id"] != series_id:
        raise ValueError("The plan belongs to another registered series")
    portable = portable_selection_binding(binding, series_manifest=manifest)
    if descriptor(Path(series_root) / "series.toml") != manifest:
        raise ValueError("Series roots changed during admission")
    result = prepare_phase1_alias_job(job, tasks, root, execution_series=plan,
                                      series_manifest=manifest)
    if result.get("complete") is not True or result.get("numerical_calls_started") is not False:
        raise ValueError("The original phase-one metadata producer did not complete inertly")
    for row in binding["runs"]:
        if descriptor(Path(row["manifest"]["path"])) != row["manifest"] or any(
                descriptor(Path(item["path"])) != item for item in row["source_metadata"]):
            raise ValueError("Job files preserved; current image selection changed during preparation")
    selected = root / "ui/image_selection_binding.json"
    write_immutable(selected, portable)
    master = dict(format="darsia-job-ui-master-v2", job=job_metadata_reference(root, root / "job.json"),
        machine=kwargs.get("machine") or socket.gethostname(), created_utc=datetime.now(timezone.utc).isoformat(),
        name=str(kwargs.get("display_name", "")).strip() or value["job_id"],
        series_label=str(kwargs.get("series_label", "")), execution_plan=deepcopy(result["series"]),
        numerical_calls_started=False, image_selection_binding=job_metadata_reference(root, selected))
    write_immutable(root / "ui/master.json", master)
    return dict(root=str(root), name=master["name"], series_label=master["series_label"], active=True,
        master=descriptor(root / "ui/master.json"), producer_receipt=result, worker_started=False)
