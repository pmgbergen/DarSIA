"""Production profile admission through the original staged resource contracts.

This adapter creates no queue during inspection. Lean's byte bound is separate
from its completed common provider; neither replaces fresh worker/cache guards.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .pipeline.job_queue import descriptor


def _pin(value):
    return {key: value[key] for key in ("path", "sha256", "size_bytes")}


def _unchanged(bound, where):
    if descriptor(Path(bound["path"])) != _pin(bound):
        raise ValueError(where + " changed during production resource admission")


def _lean_provider(stores, series_id):
    """Resolve the same completed provider selected by capacity and the worker."""
    from darsia_cache import bootstrap, common_qualification, first_qualification
    from darsia_cache.policy import cache_settings
    from .pipeline import capacity

    _, _, binding = capacity.settings(stores, series_id)
    policy = cache_settings(stores.document, required=True)
    role = bootstrap.select_entry_role(stores.document, entry_kind="worker")
    if policy.profiles.get("jobs") != binding or role["profile"] != binding:
        raise ValueError("Lean capacity, cache/jobs and worker entry must name the same common provider")
    provider, _ = capacity.read_bound(binding)
    if provider.get("format") != common_qualification.PROVIDER:
        raise ValueError("Lean production requires its completed common provider; no Full-profile fallback")
    profile = first_qualification.resolve_profile(
        binding, "jobs", lambda pin: capacity.read_bound(pin)[0],
        stores_source=_pin(stores.source),
        read_authorities=lambda pin: bootstrap._exact_authority_list(pin, capacity.__file__))
    if profile.get("format") != common_qualification.JOB_PROFILE:
        raise ValueError("Lean production requires a resolved common job profile")
    return binding, profile


def inspect_production_resources(plan, *, series_root, stores_path=None, storage_profile=None):
    """Inspect exact profile/source admission; start still performs fresh guards."""
    from . import staged_jobs, storage_profile as storage

    try:
        pipeline = staged_jobs.plan_pipeline(plan)
        if pipeline != "n1":
            if pipeline == staged_jobs.PHASE1 and storage_profile not in (None, "full"):
                raise ValueError("Phase one requires Full; Lean is not admitted for this route")
            return staged_jobs.inspect_staged_resources(
                plan, series_root=series_root, stores_path=stores_path)

        from .pipeline.alias_job import _manifest, read_alias_series
        from .pipeline.job_inputs import InputCatalog, validate_catalog
        from .pipeline import capacity
        from .store_map import read_store_map

        chosen = storage.new_job_profile(storage_profile)
        expected = _pin(plan)
        repository = Path(__file__).resolve().parents[3]
        manifest = descriptor(Path(series_root) / "series.toml")
        _, registered, roots = _manifest(manifest)
        value, configs, bound = read_alias_series(
            expected["path"], expected_sha256=expected["sha256"], series_manifest=manifest)
        if bound != expected or value.get("pipeline") != "n1":
            raise ValueError("The selected staged production plan changed")
        if not configs or len(configs) != len(value["runs"]):
            raise ValueError("Every production run requires its original configuration")
        rooted, configuration_pins = [], []
        for row, config in zip(value["runs"], configs):
            setting = storage.require_run_choice(chosen, config, pipeline="n1")
            if chosen == "lean" and ("storage_profile" not in config or setting !=
                    dict(profile="lean", colour_intermediates="keep")):
                raise ValueError("Lean production requires an explicit sealed Lean/KEEP configuration")
            catalog = InputCatalog(roots, [*value.get("data_aliases", ()), *config.get("data_aliases", ())])
            parent = catalog.add(bound, "configuration", "run.execution_series")
            registered_ref = catalog.add(manifest, "configuration", "run.series_manifest")
            inputs = validate_catalog(catalog.collect(row["configuration"], config))
            inputs["runtime_support"] = "darsia-role-io-runtime-v1"
            configuration_pins.append(dict(catalog.physical(row["configuration"])))
            rooted.append(dict(deepcopy(row["resources"]), format="darsia-alias-task-v1",
                format_version=storage.FORMAT_VERSIONS[chosen], pipeline="n1", run=config["run"],
                commit=value["job"]["commit"], precision=value["job"]["precision"],
                calibration_sha256=value["job"]["calibration_sha256"], job_id=value["job_id"],
                legacy_paths=False, execution_ready=True, execution_blocker="", storage_profile=chosen,
                configuration=inputs["configuration"], series=parent, series_manifest=registered_ref,
                input_catalog=inputs, ordered_case_ids=[f["case_id"] for f in config["frames"]]))
        stores = read_store_map(stores_path, forbidden_roots=(series_root, repository))
        provider = None
        from darsia_cache.folders import folder_mode
        if chosen == "lean" and not folder_mode(stores, registered["series_id"]):
            from darsia_cache.common_qualification import require_applicable
            provider, profile = _lean_provider(stores, registered["series_id"])
            for task, config in zip(rooted, configs):
                if require_applicable(profile, task, config=config) != "phase2":
                    raise ValueError("The common provider does not admit this phase-two route")
        estimates = [capacity.load_estimate(stores, registered["series_id"], task, config,
            repository=repository) for task, config in zip(rooted, configs)]
        for pin, where in [(expected, "Execution plan"), (manifest, "Series manifest"),
                           (_pin(stores.source), "Local store map"),
                           *((pin, "Run configuration") for pin in configuration_pins)]:
            _unchanged(pin, where)
        if provider is not None:
            _unchanged(provider, "Common provider")
        return dict(ready=True, reason="The original profile and resource contracts admit this production DATA; "
                    "Start repeats source, cache, host and free-space checks.", storage_profile=chosen,
                    series_id=registered["series_id"], store_source=_pin(stores.source),
                    common_provider=deepcopy(provider),
                    products_expected_bytes=sum(row.products_estimated_bytes for row in estimates))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(ready=False, reason=str(error))


def create_profiled_analysis_job(plan, job_root, *, stores_path=None, **kwargs):
    """Admit explicit N1 profile choice, then use the unchanged Jobs producer."""
    from . import staged_jobs, storage_profile as storage

    if staged_jobs.plan_pipeline(plan) != "n1":
        return staged_jobs.create_analysis_job(plan, job_root, stores_path=stores_path, **kwargs)
    chosen = storage.new_job_profile(kwargs.get("storage_profile"))
    admission = inspect_production_resources(plan, series_root=kwargs.get("series_root"),
        stores_path=stores_path, storage_profile=chosen)
    if not admission["ready"]:
        raise ValueError("Production creation is blocked: " + admission["reason"])
    if admission["series_id"] != kwargs.get("series_id"):
        raise ValueError("Production resource admission belongs to another registered series")
    from .jobs_metadata import create_job_from_plan
    kwargs["storage_profile"] = chosen
    return create_job_from_plan(plan, job_root, **kwargs)
