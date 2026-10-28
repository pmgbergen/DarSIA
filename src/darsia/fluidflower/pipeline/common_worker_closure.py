"""Scientific closure inside the original admitted worker before its exit inventory."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from darsia_cache.policy import CachePolicyError
from darsia_cache.common_qualification import (
    PIPELINES, PROFILES, _compute_evidence, _slot_read, _slot_write,
    stage_provider, _digest,
)


def _original_input_rows(binding, resolved_now):
    """Retain original admitted locators; a later alias resolution cannot replace them."""
    rows = binding.verified.rows
    if [(row.reference, str(row.path), row.uses) for row in resolved_now] != [
            (row.reference, str(row.path), row.uses) for row in rows]:
        raise CachePolicyError("The original admitted input resolution changed before scientific closure")
    return rows


def _closed_input_bindings(task, binding, rows):
    """Record original admission versions for the retried post-retirement close."""
    resolved = []
    for row in rows:
        version = binding.verified.versions[str(row.path)]
        resolved.append(dict(reference=deepcopy(row.reference), uses=list(row.uses),
            path=str(row.path), version=list(version)))
    return resolved


def finalize_worker_stage(first, lease, *, claim, binding):
    """Validate actual scientific joins while the existing native family is live."""
    from darsia_cache.bootstrap import current_import_lease
    from darsia_cache.owned_process import current_native_membership
    if lease is not current_import_lease() or getattr(first, "common_mode", False) is not True:
        raise CachePolicyError("Common science closure requires its actual admitted worker lease")
    first.require_task(lease, claim.task)
    native = lease.manifest.get("native_launch_context")
    if not native or current_native_membership() != native:
        raise CachePolicyError("Scientific closure must stay in its original live native family")
    if binding.task != claim.task or binding.output != Path(claim.output):
        raise CachePolicyError("Scientific closure differs from its original admitted RunIO claim")
    from darsia.fluidflower.jobs_metadata import JobSnapshotReader
    from darsia.fluidflower.published_jobs_snapshot import _terminal
    from darsia.fluidflower.pipeline.job_queue import descriptor, _own_asset
    from darsia.fluidflower.pipeline.products_publication import _read_index
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    from darsia.fluidflower.pipeline.staged_data import verify_phase1_completion, validate_phase2_run
    from darsia.fluidflower.pipeline.common_calibration import (
        verify_frozen_refit, heldout_application_anchor, phase1_statistics,
    )
    stage, task = first.stage, first.task
    root = Path(first.request["job_root"])
    metadata = JobSnapshotReader().read
    if descriptor(root / "job.json") != first.request["job"]:
        raise CachePolicyError("The original common job changed before stage close")
    done = metadata(root / "done" / (task["run"] + ".json"))
    completed, _machines, _lineage = _terminal(root, root / "results" / task["run"],
                                              task, first.job, done, metadata)
    if completed.get("task") != task or completed.get("attempt_id") != task["attempt_id"]:
        raise CachePolicyError("The closed common result is from another actual claim")
    completion = descriptor(root / "results" / task["run"] / "completed.json")
    index = descriptor(_own_asset(root, done["products_index"]))
    _index_root, _index_document, products = _read_index(index)
    if stage == "calibration":
        binding.admit_common_completion(completion, job=first.request["job"], index=index)
    configuration = first.request["configuration"]
    from darsia.fluidflower.store_map import read_store_map
    from darsia.fluidflower.pipeline.store_runtime import resolve_catalog
    from darsia.fluidflower.pipeline.capacity import configuration_for_task
    stores = read_store_map(first.stores_source["path"])
    if {key: stores.source[key] for key in ("path", "sha256", "size_bytes")} != first.stores_source:
        raise CachePolicyError("The startup stores changed before scientific closure")
    resolved_now = resolve_catalog(task, stores, first.job["series_id"])
    source_rows = _original_input_rows(binding, resolved_now)
    preliminary = configuration_for_task(stores, first.job["series_id"], task, source_rows)
    configured = [row for row in source_rows if "run.configuration" in row.uses]
    if len(configured) != 1:
        raise CachePolicyError("One original rooted scientific configuration is required")
    actual = dict(path=str(configured[0].path), sha256=configured[0].reference["sha256"],
                  size_bytes=configured[0].reference["bytes"])
    aliases = [*preliminary.get("data_aliases", ()), dict(logical=configuration, physical=actual)]
    reader = DataReader(aliases, unc_only=False)
    config = reader.json(configuration)
    phase1, join, files = None, None, []
    if stage == "phase1":
        verified = verify_phase1_completion(completed["phase1_completion"], configuration,
            expected_commit=task["commit"], reader=reader)
        phase1_statistics(verified)
        phase1 = dict(configuration=deepcopy(configuration), completion=deepcopy(completed["phase1_completion"]))
        for item in verified["records"]:
            files.extend(deepcopy(list(item["record"]["files"].values())))
    else:
        p1, _ = _slot_read(stage_provider(first.root_provider, "phase1"), "scientific_evidence")
        phase1 = deepcopy(p1["phase1"])
        if stage == "calibration":
            from .common_calibration import PHASE1_INPUTS_FORMAT
            if config.get("common_phase1") != dict(format=PHASE1_INPUTS_FORMAT,
                    configuration=phase1["configuration"], completion=phase1["completion"], code_commit=task["commit"]):
                raise CachePolicyError("The actual global plan lacks its exact declared predecessor P1 closure inputs")
        p1config = DataReader(unc_only=False).json(phase1["configuration"])
        reader = DataReader([*p1config.get("data_aliases", ()), *aliases], unc_only=False)
        verified = verify_phase1_completion(phase1["completion"], phase1["configuration"],
            expected_commit=task["commit"], reader=reader)
        if stage == "calibration":
            c_binding = dict(completion=completion, plan=configuration)
            frozen_pin = completed["frozen_calibration"]
            frozen = reader.json(frozen_pin)
            run = verified["configuration"]["run"]
            if run in config["statistics"]:
                anchor = {key: frozen["anchors_by_run"][run][key]
                          for key in ("elapsed_h", "affected_pore_volume_m3")}
            else:
                anchor = heldout_application_anchor(verified, config, reader=reader)["anchor"]
            join = verify_frozen_refit(c_binding, verified_phase1=verified,
                fitted_calibration=frozen_pin, fitted_anchor=anchor,
                expected_commit=task["commit"], reader=reader)
            files = deepcopy(list(completed["files"].values()))
        else:
            validate_phase2_run(config)
            from darsia.fluidflower.storage_profile import resolve
            profile = resolve(config)
            if profile["profile"] != "lean" or profile["colour_intermediates"] != "keep":
                raise CachePolicyError("Actual P2 did not consume Lean/KEEP")
            recorded = config.get("provenance", {}).get("common_frozen_refit")
            if not isinstance(recorded, dict):
                raise CachePolicyError("P2 lacks the original positive C sealer join")
            c, _ = _slot_read(stage_provider(first.root_provider, "calibration"), "scientific_evidence")
            join = verify_frozen_refit(recorded["binding"], verified_phase1=verified,
                fitted_calibration=config["calibration"], fitted_anchor=config["anchor"],
                expected_commit=task["commit"], reader=reader)
            if _digest(join) != _digest(recorded) or _digest(join) != _digest(c["calibration_join"]):
                raise CachePolicyError("Actual P2's frozen C/application binding changed")
            if (config["phase1"]["configuration"] != phase1["configuration"]
                    or config["phase1"]["completion"] != phase1["completion"]
                    or config["phase1"]["code_commit"] != task["commit"]):
                raise CachePolicyError("Actual P2 reads a different P1 completion/source")
            for record in completed["records"]:
                directory = Path(record["nas_directory"])
                for leaf, row in record["files"].items():
                    files.append(dict(path=str(directory / leaf),
                        sha256=row["sha256"], size_bytes=row.get("size_bytes", row.get("bytes"))))
    # The original terminal reader checked the exact product index against its
    # actual claim/producer. Include once-per-run lean stores and all indexed
    # scalar/receipt outputs too, rather than only frame-local NPZ files.
    files = [dict(path=row["local_path"], sha256=row["sha256"], size_bytes=row["bytes"])
             for row in products.values()]
    for row in files:
        reader.path(row)
    reader.json(configuration)
    compute = _compute_evidence(config, stage, task["ordered_case_ids"], files, reader)
    compute["stage_wall_seconds"] = completed.get("wall_seconds")
    compute["stage_images_per_hour"] = completed.get("images_per_hour")
    compute["throughput_scope"] = "original stage runner wall, including device probes and CPU comparisons; excludes publication and finalization"
    compute["memory_scopes"] = "kernel commit peaks, per-process working-set facts and application device buffers are distinct"
    result_space = dict(configuration=deepcopy(configuration), case_count=len(task["ordered_case_ids"]),
        maximum_output_bytes=first.output_entitlement["maximum_output_bytes"], measured_native_layout=None)
    if stage == "phase1":
        from darsia.fluidflower.pipeline.capacity import _native_layout_for_task
        result_space["measured_native_layout"] = _native_layout_for_task(
            stores, first.job["series_id"], task, config, source_rows)
    lean_bound = None
    if stage == "phase2":
        from darsia.fluidflower.pipeline.capacity import _phase2_lean_estimate
        estimate = _phase2_lean_estimate(stores, first.job["series_id"], task, config,
            repository=binding.repository, metadata_rows=source_rows)
        lean_bound = deepcopy(estimate.estimate)
    resolved = _closed_input_bindings(task, binding, source_rows)
    frame_records = [dict(frame_id=row["frame_id"], nas_directory=row["nas_directory"])
                     for row in completed.get("records", ())]
    if stage == "phase2":
        producers = {row["case_id"]: row for row in completed["frame_producers"]}
        for row in frame_records:
            row["producer"] = deepcopy(producers[row["frame_id"]])
    payload = dict(format="darsia-common-worker-scientific-closure-v1",
        request=deepcopy(first.binding), provider=deepcopy(first.provider_binding),
        stores_source=deepcopy(first.stores_source), stage=stage, task=deepcopy(task),
        owner=deepcopy(lease.owner), native_launch=deepcopy(native),
        service_receipt=deepcopy(lease.receipt), configuration=deepcopy(configuration),
        completion=completion, publication=dict(job=deepcopy(first.request["job"]),
            done=descriptor(root / "done" / (task["run"] + ".json")), products_index=index),
        phase1=phase1, calibration_join=join, output_files=files,
        frame_records=frame_records, resolved_inputs=resolved,
        compute_evidence=compute, result_space=result_space, lean_bound=lean_bound,
        complete=True, process_retirement_claimed=False)
    pin = _slot_write(first.provider, "worker_scientific_evidence", payload)
    first.worker_scientific_evidence = deepcopy(pin)
    return pin
