"""Review one failed/stopped image attempt while retaining completed geometry."""
from pathlib import Path

from .pipeline.job_queue import name, read_json


def geometry_operations(root, job, plan):
    """Read original typed completions, whether local or already retained."""
    from .setup_completed_builds import retained_operations, output_record
    from .pipeline.job_queue import descriptor
    from .jobs_metadata import JobSnapshotReader
    root = Path(root)
    kept = retained_operations(root, job)
    expected = {"setup-apply", *plan["runs"]}
    if set(kept) == expected:
        return kept
    snapshot = JobSnapshotReader()(dict(root=str(root)))
    if snapshot.get("errors"):
        raise ValueError("Image retry requires valid original Setup publications")
    for row in snapshot["runs"]:
        run = row["run"]
        if run not in expected or run in kept:
            continue
        result = row.get("result", {})
        operation = "apply" if run == "setup-apply" else "build"
        if row.get("state") not in {"Complete", "Completed"} or result.get("operation") != operation:
            raise ValueError("Complete the original geometry before retrying only images")
        record = {} if operation == "apply" else output_record(run, result, result["raw_build_request"])
        kept[run] = dict(record, run=run, metadata_root=str(root),
            source_job=descriptor(root / "job.json"), result=result, task=result["task"],
            display={key:row[key] for key in ("machine", "worker", "backend", "started_utc", "finished_utc")
                     if key in row})
    if set(kept) != expected:
        raise ValueError("Image retry requires every original geometry completion")
    return kept


def review(root, job, plan, stores_path):
    """Read only original publications and named output versions, never RAW."""
    from .job_management import _retirement_binding
    from .jobs_metadata import JobSnapshotReader
    from .published_jobs_snapshot import _failed_publication, _owner_machine, _task_identity
    from .pipeline.series_setup_protocol import validate_setup_claim
    from .pipeline.setup_image_preparation import IMAGE_RUN
    from .setup_failed_start import _files, completed_apply_snapshot
    from .setup_partial_build import snapshot
    root = Path(root)
    kept = geometry_operations(root, job, plan)
    if set(kept) != {"setup-apply", *plan["runs"]}:
        raise ValueError("Image retry requires all original completed geometry operations")
    failures = _files(root / "failed")
    if set(failures) not in (set(), {IMAGE_RUN + ".json"}):
        raise ValueError("Image retry found a failure outside its typed operation")
    claims = list((root / "claimed").glob("*/" + IMAGE_RUN + ".json"))
    if len(claims) != 1 or claims[0].name != IMAGE_RUN + ".json":
        raise ValueError("Image retry needs its one original operation claim")
    claim = claims[0]
    task = read_json(claim)
    base = read_json(root / "results" / IMAGE_RUN / "task.json")
    validate_setup_claim(task, job)
    if task.get("operation") != "image-preparation" or not _task_identity(base, task):
        raise ValueError("Image retry changed the original claimed task")
    token, service = task["attempt_id"], name(claim.parent.name)
    lease = root / "leases" / (IMAGE_RUN + "." + token)
    before_owner = not (lease / "owner.json").exists()
    if before_owner:
        # The queue rename precedes output admission and owner publication.
        # A stopped service may therefore retain this exact claim and an empty
        # lease even though no scientific attempt was ever started.
        from .store_map import read_store_map
        if service != name(task["target_machine"] + "-" + task["backend"]):
            raise ValueError("Image admission retry belongs to another recorded target")
        stores = read_store_map(stores_path)
        output = (Path(stores.roots(job["series_id"])["results"])
                  / job["job_id"] / IMAGE_RUN / ".attempts" / token)
        pre_owner = dict(run=IMAGE_RUN, attempt_id=token, claim=str(claim),
                         lease=str(lease), original_output=str(output))
        verify_pre_owner(pre_owner)
        if failures:
            raise ValueError("An image claim without an owner cannot have published work")
        output_files = {}
    else:
        owner = read_json(lease / "owner.json")
        _owner_machine(owner, base, token)
        if owner.get("task") != task or owner.get("machine") != service:
            raise ValueError("Image retry differs from its original owner")
        if failures:
            failure, count, times, _ = _failed_publication(root, base,
                root / "failed" / (IMAGE_RUN + ".json"), JobSnapshotReader().read)
            if count or times or failure["task"] != task:
                raise ValueError("Image retry cannot promote failed work to completed frames")
        output = _retirement_binding(root, claim, stores_path).path
        output_files = _files(output)
    progress = set()
    for sequence, path in enumerate(sorted(lease.glob("progress_*.json"))):
        value = read_json(path)
        if (path.name != f"progress_{sequence:08d}.json" or value.get("run") != IMAGE_RUN
                or value.get("attempt_id") != token or value.get("sequence") != sequence):
            raise ValueError("Image progress differs from its original attempt")
        original = read_json(output / path.name)
        if value != dict(original, detail_scope=
                "Original event provenance; frame evidence is published_frame within this shared bundle"):
            raise ValueError("Image progress changed between original and shared publication")
        progress.add(path.name)
    if failures and (output / "failed.json").read_bytes() != (root / "failed" / (IMAGE_RUN + ".json")).read_bytes():
        raise ValueError("Image failure differs from its original output")
    expected = dict(claimed={service + "/" + IMAGE_RUN + ".json"},
        leases={IMAGE_RUN + "." + token + "/" + leaf for leaf in
            (set() if before_owner else {"owner.json", "claimed.json"}) | progress | (
                {"publication.json", "failure_artifacts/failed.json"} if failures else set())},
        decisions={IMAGE_RUN + "." + token + ".json"} if failures else set(),
        publishing={IMAGE_RUN} if failures else set(), failed=set(failures),
        queue=set(), done=set(), reclaimed=set())
    # From-scratch jobs still hold the genuine geometry publications here.
    # Their typed completion reader already authenticated each source, owner
    # and product index; retain those exact metadata bytes in the same history.
    for run, row in kept.items():
        if Path(row["metadata_root"]).resolve() != root.resolve():
            continue
        result = row["result"]
        physical = name(result["actual_worker"]["machine"])
        original_claim = root / "claimed" / physical / (run + ".json")
        if read_json(original_claim) != result["task"]:
            raise ValueError("Completed geometry claim differs from its original result")
        original_lease = run + "." + result["attempt_id"]
        expected["claimed"].add(physical + "/" + run + ".json")
        expected["leases"] |= {original_lease + "/" + leaf for leaf in _files(root / "leases" / original_lease)}
        expected["publishing"].add(run)
        expected["decisions"].add(original_lease + ".json")
        expected["done"].add(run + ".json")
    # The immutable publisher retains byte-identical temporaries after a
    # retry. Preserve only those copies of the verified original completions.
    import re
    for filename in _files(root / "done"):
        if filename in expected["done"]:
            continue
        match = re.fullmatch(r"\.(.+)\.json\.[0-9a-f]{32}\.partial", filename)
        if (match is None or match[1] not in kept
                or (root / "done" / filename).read_bytes()
                    != (root / "done" / (match[1] + ".json")).read_bytes()):
            raise ValueError("Image retry found another completed operation")
        expected["done"].add(filename)
    operational = {}
    for folder, expected_names in expected.items():
        files = _files(root / folder)
        if set(files) != expected_names:
            raise ValueError("Image retry found other operation work in " + folder)
        operational[folder] = files
    from .pipeline.series_setup_jobs import _control
    from .pipeline.series_setup_protocol import validate_setup_operation
    config, configuration = _control(root / "setup-controls/image-operation.json")
    validate_setup_operation(config)
    if (configuration != job["image_preparation"]["configuration"]
            or base["configuration"]["sha256"] != configuration["sha256"]
            or config["choice_authority"] != job["image_preparation"]["choice_authority"]):
        raise ValueError("Image retry configuration differs from its original claimed authority")
    source = Path(config["output_root"]) / token
    partials = []
    if source.exists():
        if before_owner:
            raise ValueError("An image claim without an owner cannot have prepared outputs")
        if not source.resolve().is_relative_to((root / "image-preparation").resolve()):
            raise ValueError("Image retry output escaped its original job")
        partials.append(dict(run=IMAGE_RUN, source=str(source), snapshot=snapshot(source)))
    attempt = dict(run=IMAGE_RUN, attempt_id=token, claim=str(claim),
                   original_output=str(output), output_files=output_files)
    return dict(format="darsia-setup-failed-image-preparation-v1", **attempt,
        attempts=[] if before_owner else [attempt],
        **({"pre_owner_claim": pre_owner} if before_owner else {}),
        operational=operational, queued_runs=[], unapplied_export=None,
        completed_apply=completed_apply_snapshot(root, plan),
        completed_builds=[row for run, row in kept.items() if run != "setup-apply"],
        image_partials=partials, geometry_operations=kept)


def verify_pre_owner(row):
    """An unbound queue rename grants no ownership or scientific completion."""
    from .job_lifecycle import require_service_retirement
    claim, lease = Path(row["claim"]), Path(row["lease"])
    require_service_retirement(claim.parents[2])
    if (not lease.is_dir() or any(lease.iterdir())
            or Path(row["original_output"]).exists()):
        raise ValueError("Image admission retry requires its empty original lease and absent output")


def verify_partial(row, *, directory=None):
    from .setup_partial_build import snapshot
    if snapshot(Path(row["source"] if directory is None else directory)) != row["snapshot"]:
        raise ValueError("Image preparation partial changed during retry")
