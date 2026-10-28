"""Verify immutable frame prefixes and owned-process retirement for resume.

This module reads receipts and payload bytes. It never reads scientific arrays,
calls a numerical operator or changes queue ownership.
"""

from __future__ import annotations

import hashlib
import ntpath
import re
from pathlib import Path

from .job_queue import read_json, retired_lease_directory
from .io_ops import retry_io


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


@retry_io
def _verify_file(path, bound):
    """Verify a declared immutable file, including its identity before/after."""
    path = Path(path)
    if not isinstance(bound, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", bound.get("sha256", "")
    ):
        raise ValueError("Resume requires an original full SHA descriptor")
    if type(bound.get("size_bytes")) is not int:
        raise ValueError("Resume requires the original declared file size")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    stamp = lambda s: (s.st_size, s.st_mtime_ns, s.st_dev, s.st_ino)
    if (
        stamp(before) != stamp(after)
        or after.st_size != bound["size_bytes"]
        or digest.hexdigest() != bound["sha256"]
    ):
        raise ValueError("Resume payload/receipt differs: " + str(path))
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def _bound_json(bound):
    """Read strict metadata while requiring its original content identity."""
    path = Path(bound["path"])
    _verify_file(path, bound)
    value = read_json(path)
    _verify_file(path, bound)
    return value


def run_identity(task, backend, precision, calibration_sha256):
    """Bind the same source inputs and numerical settings across attempts."""
    if task.get("calibration_sha256", calibration_sha256) != calibration_sha256:
        raise ValueError("Task calibration differs from the configured calibration")
    value = dict(
        run=task["run"],
        configuration=task["configuration"],
        commit=task["commit"],
        calibration_sha256=calibration_sha256,
        backend=backend,
        precision=precision,
        threads=task.get("threads", 1),
    )
    for key in ("device_id", "pool_limit_bytes", "environment_lock", "pipeline"):
        if key in task:
            value[key] = task[key]
    return value


def resume_source_for_claim(shared, claim):
    """Select only this claim's genuine reclaimed predecessor, never a scan."""
    if claim.task.get("format") == "darsia-alias-task-v1":
        from .run_io import alias_resume_source
        return alias_resume_source(shared,claim)
    matches = []
    root = shared.root / "reclaimed" / claim.task["run"]
    for record_path in root.glob("*/reclaimed.json"):
        record = read_json(record_path)
        if record.get("next_attempt") == claim.token:
            matches.append((record_path, record))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Resume has more than one reclaimed predecessor")
    record_path, record = matches[0]
    old_token = record["previous_attempt"]
    if record.get("interrupted_claim_before_science") is True:
        archived = read_json(record_path.parent / "task.json")
        left, right = dict(archived), dict(claim.task)
        left.pop("attempt_id", None)
        right.pop("attempt_id", None)
        if left != right:
            raise ValueError("Interrupted pre-science claim task differs")
        return None
    if record_path.parent.name != old_token:
        raise ValueError("Reclaimed predecessor token differs")
    old_owner = read_json(
        retired_lease_directory(shared.root, claim.task["run"], old_token) / "owner.json")
    old_task = dict(old_owner["task"])
    new_task = dict(claim.task)
    old_task.pop("attempt_id", None)
    new_task.pop("attempt_id", None)
    if old_task != new_task or old_owner["token"] != old_token:
        raise ValueError("Reclaimed predecessor task differs")
    old_output = Path(old_owner["output_path"])
    if (
        _key(old_output.parent) != _key(claim.output.parent)
        or old_output.name != old_token
    ):
        raise ValueError("Resume source escapes the canonical run attempt root")
    retirement = old_output / "retirement.json"
    if not retirement.exists():
        raise ValueError("Resume requires genuine owned-child retirement evidence")
    raw = retirement.read_bytes()
    return dict(
        attempt_directory=str(old_output),
        retirement=dict(
            path=str(retirement),
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
        ),
    )


def _original_storage_facts(task, publication):
    """Verify actual storage facts without changing original product checks."""
    if task.get("format") == "darsia-alias-task-v1":
        return (publication.get("direct_NAS_write") is False
                and publication.get("direct_local_write") is True
                and publication.get("local_retained") is True)
    return publication.get("direct_NAS_write") is True


def verify_frame_prefix(
    task,
    config,
    output,
    backend,
    precision,
    source,
    *,
    frame_validator=None,
    calibration_sha256=None,
    _seen=None,
):
    """Return only an exact contiguous prefix with original producer receipts.

    A finalized directory without its original registration/publication receipt
    fails closed. Partial directories stay untouched. Chained resume retains the
    original frame producer; it never rewrites its manifest or receipts.
    """
    output = Path(output)
    if source is None:
        return dict(
            records=[],
            frame_producers=[],
            source=None,
            prefix_case_ids=[],
            predecessor_input=None,
            predecessor_index=None,
        )
    n1 = config.get("format") == "darsia-n1-run-v1"
    if not n1 and (not callable(frame_validator) or calibration_sha256 is None):
        raise ValueError(
            "Series resume requires its literal input/artifact validator and calibration SHA"
        )
    calibration_sha256 = config["calibration"]["sha256"] if n1 else calibration_sha256
    attempt = Path(source["attempt_directory"])
    if (
        _key(attempt.parent) != _key(output.parent)
        or attempt == output
        or attempt.name in {"", ".", ".."}
    ):
        raise ValueError("Resume source must be another attempt of this run")
    seen = set() if _seen is None else set(_seen)
    if _key(attempt) in seen:
        raise ValueError("Resume predecessor cycle")
    seen.add(_key(attempt))
    if task.get("format") == "darsia-alias-task-v1":
        from .run_io import for_task
        for_task(task).require_resume_location(attempt)
    proof = _bound_json(source["retirement"])
    if (
        proof.get("complete") is not True
        or proof.get("all_absent") is not True
        or proof.get("worker_os_references_closed") is not True
        or _key(proof.get("attempt_directory", "")) != _key(attempt)
        or proof.get("attempt_id") != attempt.name
        or proof.get("run") != task["run"]
        or not proof.get("identities")
    ):
        raise ValueError("Resume retirement is incomplete or bound to another attempt")
    manifest_path = attempt / "manifest.json"
    manifest_asset = proof.get("manifest")
    if manifest_asset is None:
        if manifest_path.exists() or any((attempt / "frames").glob("*")):
            raise ValueError("Resume zero-prefix retirement omitted existing output")
        return dict(
            records=[],
            frame_producers=[],
            source=source,
            prefix_case_ids=[],
            predecessor_input=None,
            predecessor_index=None,
        )
    if _key(manifest_asset["path"]) != _key(manifest_path):
        raise ValueError("Resume retirement bound a different manifest")
    manifest = _bound_json(manifest_asset)
    expected = run_identity(task, backend, precision, calibration_sha256)
    if (
        manifest.get("format")
        != ("darsia-n1-run-manifest-v1" if n1 else "darsia-series-run-manifest-v1")
        or manifest.get("task_identity") != expected
        or manifest.get("configuration") != task["configuration"]
        or manifest.get("code", {}).get("commit") != task["commit"]
        or manifest.get("code_sha256")
        != hashlib.sha256(task["commit"].encode("ascii")).hexdigest()
        or manifest.get("settings", {}).get("backend") != backend
        or manifest.get("settings", {}).get("precision") != precision
        or manifest.get("frames") != {f["case_id"]: f for f in config["frames"]}
    ):
        raise ValueError("Resume producer/source/configuration/settings differ")
    records, origins = [], []
    if manifest.get("resume_checkpoint") is not None:
        checkpoint = _bound_json(manifest["resume_checkpoint"])
        previous = verify_frame_prefix(
            task,
            config,
            attempt,
            backend,
            precision,
            checkpoint["source"],
            frame_validator=frame_validator,
            calibration_sha256=calibration_sha256,
            _seen=seen,
        )
        if (
            checkpoint["prefix_case_ids"]
            != [f["case_id"] for f in config["frames"][: len(previous["records"])]]
            or checkpoint["frame_producers"] != previous["frame_producers"]
        ):
            raise ValueError("Resume checkpoint changed its original prefix")
        records.extend(previous["records"])
        origins.extend(previous["frame_producers"])
    missing = False
    for position, spec in enumerate(config["frames"]):
        final = attempt / "frames" / spec["case_id"]
        if position < len(records):
            if final.exists():
                raise ValueError("Resume attempt duplicated a retained frame")
            continue
        if not final.exists():
            missing = True
            continue
        if missing:
            raise ValueError("Resume completed-frame prefix is not contiguous")
        commit_path = final / "commit.json"
        commit_asset = _verify_file(commit_path, _descriptor(commit_path))
        commit = _bound_json(commit_asset)
        if (
            commit.get("complete") is not True
            or commit.get("frame_id") != spec["case_id"]
            or commit.get("manifest_sha256") != manifest_asset["sha256"]
            or commit.get("job_id") != manifest["job_id"]
            or commit.get("settings") != manifest["settings"]
            or commit.get("products") != dict(precision=precision, backend=backend)
            or not re.fullmatch(r"[0-9a-f]{32}", commit.get("commit_id", ""))
        ):
            raise ValueError("Resume frame commit differs from its producer")
        files = commit["files"]
        required = {
            "products.npz",
            "native_geometry.npz",
            "native_statistics.json.gz",
            "result.json",
        }
        if not required.issubset(files) and not _lean_without_geometry(final, files, required):
            raise ValueError("Resume frame omitted a required scientific product")
        if {p.name for p in final.iterdir()} != set(files) | {"commit.json"}:
            raise ValueError("Resume finalized frame inventory differs")
        for filename, bound in files.items():
            if (
                Path(filename).name != filename
                or ntpath.basename(filename) != filename
                or filename in {".", ".."}
            ):
                raise ValueError("Resume frame member escapes its directory")
            _verify_file(final / filename, bound)
        registration_path = (
            attempt / "ops" / ("frame_commit_" + commit["commit_id"] + ".json")
        )
        publication_path = (
            attempt / "ops" / ("nas_copy_" + commit["commit_id"] + ".json")
        )
        receipts = [
            r for r in proof["frame_receipts"] if r["frame_id"] == spec["case_id"]
        ]
        if len(receipts) != 1:
            raise ValueError(
                "Resume retirement omitted or duplicated an original frame receipt"
            )
        registration_asset, publication_asset = (
            receipts[0]["registration"],
            receipts[0]["publication"],
        )
        if _key(registration_asset["path"]) != _key(registration_path) or _key(
            publication_asset["path"]
        ) != _key(publication_path):
            raise ValueError("Resume retirement changed a receipt locator")
        registration, publication = _bound_json(registration_asset), _bound_json(
            publication_asset
        )
        inventory = dict(
            files,
            **{"commit.json": {k: v for k, v in commit_asset.items() if k != "path"}},
        )
        if (
            registration.get("manifest_sha256") != manifest_asset["sha256"]
            or registration.get("frame_id") != spec["case_id"]
            or registration.get("commit") != commit_asset
            or registration.get("files") != files
            or publication.get("frame_id") != spec["case_id"]
            or _key(publication.get("nas_directory", "")) != _key(final)
            or publication.get("manifest_sha256") != manifest_asset["sha256"]
            or publication.get("files") != inventory
            or not _original_storage_facts(task,publication)
        ):
            raise ValueError("Resume original frame registration/publication differs")
        result_asset = dict(files["result.json"], path=str(final / "result.json"))
        result = _bound_json(result_asset)
        if (
            result.get("complete") is not True
            or result.get("run") != task["run"]
            or result.get("case_id") != spec["case_id"]
            or result.get("frame_index") != spec["frame_index"]
            or result.get("chronological_position") != position
            or result.get("backend") != backend
            or result.get("precision") != precision
            or result.get("working_pixel_dtype") != precision
            or result.get("bindings", {}).get("calibration_sha256")
            != calibration_sha256
            or result.get("bindings", {}).get("code_sha256") != manifest["code_sha256"]
        ):
            raise ValueError("Resume result producer/precision/calibration differs")
        if result.get("lean_storage") is not None:
            _verify_lean_frame(result, files)
        if n1:
            bindings = dict(
                source_sha256=spec["source_binding"]["source_sha256"],
                phase_sha256=spec["labels"]["sha256"],
                calibration_sha256=calibration_sha256,
                code_sha256=manifest["code_sha256"],
            )
            if (
                result.get("bindings") != bindings
                or result.get("source_binding") != spec["source_binding"]
                or result.get("physical_correction") != config["physical_correction"]
                or result.get("anchor") != config["anchor"]
            ):
                raise ValueError("Resume result source/science bindings differ")
        else:
            if task.get("format") == "darsia-alias-task-v1":
                # Original artifact checks may read the producer's preparation
                # files. Permit only this exact same-machine candidate while
                # those unchanged checks run; no permanent prefix is admitted.
                from .run_io import for_task
                with for_task(task).prefix_validation(source):
                    frame_validator(spec, result, final, config)
            else:
                frame_validator(spec, result, final, config)
        records.append(
            publication
        )  # Literal original record, never reserialized as a new producer.
        origins.append(
            dict(
                case_id=spec["case_id"],
                attempt_directory=str(attempt),
                manifest=manifest_asset,
                commit=commit_asset,
                frame_registration=registration_asset,
                publication=publication_asset,
                result=result_asset,
            )
        )
    previous_input = None
    if records:
        last = config["frames"][len(records) - 1]
        previous_input = {
            key: last[key]
            for key in (
                "case_id",
                "frame_index",
                "elapsed_h",
                "source_binding",
                "current",
                "baseline",
            )
            if key in last
        }
    return dict(
        records=records,
        frame_producers=origins,
        source=source,
        prefix_case_ids=[f["case_id"] for f in config["frames"][: len(records)]],
        predecessor_input=previous_input,
        predecessor_index=len(records) - 1 if records else None,
    )


def _lean_without_geometry(final, files, required):
    """A lean frame may omit native_geometry.npz when it wrote no fallback field."""
    if required - set(files) != {"native_geometry.npz"}:
        return False
    result = _bound_json(dict(files["result.json"], path=str(Path(final) / "result.json")))
    lean = result.get("lean_storage")
    static = result.get("static") or {}
    return (isinstance(lean, dict) and isinstance(static.get("storage"), dict)
            and static.get("asset") is None and static["storage"].get("written") == [])


def _verify_lean_frame(result, files):
    """Bind a lean frame's written geometry and its run store by their bytes."""
    from ..storage_profile import frame_storage, store_assets
    lean = frame_storage(result)
    static = result["static"]
    storage = static.get("storage")
    if not isinstance(storage, dict) or storage.get("profile") != "lean":
        raise ValueError("Resume lean frame lacks its static storage record")
    if (static["asset"] is None) != (not storage["written"]):
        raise ValueError("Resume lean frame geometry record differs from its fallback fields")
    if static["asset"] is not None and (
        "native_geometry.npz" not in files
        or files["native_geometry.npz"]["sha256"] != static["asset"]["sha256"]
        or files["native_geometry.npz"]["size_bytes"] != static["asset"]["size_bytes"]
    ):
        raise ValueError("Resume lean frame geometry differs from its committed file")
    if storage["regeneration"] is not None and (
        storage["regeneration"]["store"] != lean["store"]["index"]
    ):
        raise ValueError("Resume lean frame regenerates from another run store")
    for bound in store_assets(lean):
        _verify_file(bound["path"], bound)


def _descriptor(path):
    """Metadata descriptors are small; payload validation streams separately."""
    from .job_queue import descriptor
    return descriptor(path)


def retirement_document(task, attempt, identities, supervisor):
    """Bind original receipts only after the caller proved its child tree absent."""
    attempt = Path(attempt)
    manifest = (
        _descriptor(attempt / "manifest.json")
        if (attempt / "manifest.json").exists()
        else None
    )
    receipts = []
    for path in sorted((attempt / "ops").glob("nas_copy_*.json")):
        publication = read_json(path)
        frame = Path(publication["nas_directory"])
        if _key(frame.parent) != _key(attempt / "frames"):
            raise ValueError("Retirement receipt escapes the owned attempt")
        commit = _bound_json(
            dict(publication["files"]["commit.json"], path=str(frame / "commit.json"))
        )
        registration = (
            attempt / "ops" / ("frame_commit_" + commit["commit_id"] + ".json")
        )
        receipts.append(
            dict(
                frame_id=publication["frame_id"],
                registration=_descriptor(registration),
                publication=_descriptor(path),
            )
        )
    return dict(
        complete=True,
        all_absent=True,
        worker_os_references_closed=True,
        run=task["run"],
        attempt_id=attempt.name,
        attempt_directory=str(attempt),
        identities=identities,
        supervisor=supervisor,
        manifest=manifest,
        frame_receipts=receipts,
        retirement_basis="owned_process_incarnations_absent",
    )


def frame_producer(output, publication, manifest):
    """Expose a newly published frame's genuine original receipt descriptors."""
    final = Path(publication["nas_directory"])
    commit = read_json(final / "commit.json")
    ops = Path(output) / "ops"
    return dict(
        case_id=publication["frame_id"],
        attempt_directory=str(output),
        manifest=manifest,
        commit=_descriptor(final / "commit.json"),
        frame_registration=_descriptor(
            ops / ("frame_commit_" + commit["commit_id"] + ".json")
        ),
        publication=_descriptor(ops / ("nas_copy_" + commit["commit_id"] + ".json")),
        result=_descriptor(final / "result.json"),
    )
