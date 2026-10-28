"""Local original-format queue metadata only; never starts a worker."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest
from darsia.fluidflower import job_launch_state as launches
from darsia.fluidflower import jobs_metadata as model


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False) + "\n").encode())
    return model.descriptor(path)


def bytes_of(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.fixture
def job(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    cases = ["ac52_f1", "ac52_f9"]  # sparse indices are two images, never nine.
    put(
        root / "job.json",
        dict(
            expected_runs=["ac52"],
            commit="a" * 40,
            calibration_sha256="b" * 64,
            precision="float32",
        ),
    )
    config = put(
        root / "config.json", dict(frames=[dict(case_id=case) for case in cases])
    )
    task = dict(
        run="ac52",
        commit="a" * 40,
        calibration_sha256="b" * 64,
        precision="float32",
        pipeline="native-phase-n1",
        job_id="tiny",
        configuration=config,
        backend="cpu",
        threads=2,
        output_directory=str(root / "results" / "ac52"),
        attempt_id="1" * 32,
    )
    put(root / "results" / "ac52" / "task.json", task)
    put(root / "queue" / "ac52.json", task)
    master = dict(
        format="darsia-job-ui-master-v1",
        job=model.descriptor(root / "job.json"),
        machine="Main",
        name="Tiny",
        series_label="Local",
        created_utc="2026-01-01T00:00:00+00:00",
    )
    put(root / "ui" / "master.json", master)
    return root, cases, task


def claim(job):
    root, cases, task = job
    attempt = root / "results" / "ac52" / ".attempts" / task["attempt_id"]
    attempt.mkdir(parents=True, exist_ok=True)
    put(root / "claimed" / "Main" / "ac52.json", task)
    put(
        root / "leases" / ("ac52." + task["attempt_id"]) / "owner.json",
        dict(
            task=task,
            machine="Main",
            token=task["attempt_id"],
            output_path=str(attempt),
            actual_owner=dict(
                machine="Main", pid=123, process_create_time="1790000000.125"
            ),
        ),
    )
    return attempt


def publication(root, case, token="1" * 32):
    final = root / "results" / "ac52" / ".attempts" / token / "final" / case
    bound = put(
        final / "commit.json",
        dict(
            complete=True,
            frame_id=case,
            products=dict(precision="float32", backend="cpu"),
            settings=dict(host="Main"),
        ),
    )
    return dict(
        frame_id=case,
        nas_directory=str(final),
        files={
            "commit.json": {key: value for key, value in bound.items() if key != "path"}
        },
    )


def progress(attempt, sequence, event, seconds):
    return put(
        attempt / f"progress_{sequence:08d}.json",
        dict(
            sequence=sequence,
            run="ac52",
            attempt_id=attempt.name,
            event=event,
            written_unix_ns=int(seconds * 1e9),
        ),
    )


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() + 120


def test_browse_is_readonly_sparse_count_and_no_false_completion(job):
    root, _, _ = job
    put(root / "historical_STOP.json", {"stop": "not this job's terminal result"})
    put(root / "results" / "ac52" / "completed.json", {"complete": True})
    before = bytes_of(root)
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert value["state"] == "Waiting" and not value["complete"]
    assert value["total_frames"] == 2 and value["committed"] == 0
    assert value["eta_seconds"] is None
    assert bytes_of(root) == before


def test_actual_claim_progress_unique_count_and_measured_eta(job):
    root, cases, _ = job
    attempt = claim(job)
    started = NOW - 60
    progress(attempt, 0, dict(state="frame_started", case_id=cases[0]), started)
    record = publication(root, cases[0])
    progress(
        attempt,
        1,
        dict(
            state="frame_completed", complete=True, case_id=cases[0], publication=record
        ),
        NOW - 30,
    )
    progress(
        attempt,
        2,
        dict(
            state="frame_completed", complete=True, case_id=cases[0], publication=record
        ),
        NOW - 20,
    )
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert value["state"] == "Working" and value["committed"] == 1
    assert value["remaining"] == 1 and value["frames_per_minute"] == 0.5
    assert value["eta_seconds"] == 120
    assert value["machines"][0]["frames_per_minute"] == 1
    assert value["machines"][0]["eta_seconds"] == 60
    assert value["workers"][0]["worker"] == "PID 123"
    assert value["runs"][0]["actual_worker"]["process_create_time"] == "1790000000.125"
    assert value["runs"][0]["case_ids"] == [cases[0]]


def test_claim_owner_is_not_numerical_progress(job):
    root, _, _ = job
    claim(job)
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert value["state"] == "Claimed" and value["committed"] == 0
    assert value["eta_seconds"] is None
    assert "liveness not established" in value["workers"][0]["state"]


def test_done_needs_exact_bound_closed_completion_and_every_selected_case(job):
    root, cases, task = job
    records = [publication(root, case) for case in cases]
    completed = dict(
        complete=True,
        worker_references_closed=True,
        run="ac52",
        precision="float32",
        frames=2,
        records=records,
        actual_worker=dict(machine="Main"),
        frame_producers=[],
        backend="cpu",
        commit=task["commit"],
        calibration_sha256=task["calibration_sha256"],
        configuration=task["configuration"],
        task=deepcopy(task),
        attempt_id=task["attempt_id"],
    )
    target = root / "results" / "ac52" / "completed.json"
    bound = put(target, completed)
    put(
        root / "done" / "ac52.json",
        dict(run="ac52", completed=bound, attempt_id=task["attempt_id"]),
    )
    reader = model.JobSnapshotReader()
    before = bytes_of(root)
    value = reader({"root": str(root)}, now=NOW)
    assert value["complete"] and value["committed"] == 2 and value["eta_seconds"] == 0
    assert bytes_of(root) == before
    completed["worker_references_closed"] = False
    bound = put(target, completed)
    put(
        root / "done" / "ac52.json",
        dict(run="ac52", completed=bound, attempt_id=task["attempt_id"]),
    )
    changed = reader({"root": str(root)}, now=NOW)
    assert changed["state"] == "Metadata error" and changed["committed"] == 0
    assert "closed per-run completion" in changed["errors"][0]


def test_tampered_cached_frame_receipt_is_rechecked(job):
    root, cases, _ = job
    attempt = claim(job)
    record = publication(root, cases[0])
    progress(
        attempt,
        0,
        dict(
            state="frame_completed", complete=True, case_id=cases[0], publication=record
        ),
        NOW - 10,
    )
    reader = model.JobSnapshotReader()
    assert reader({"root": str(root)}, now=NOW)["committed"] == 1
    put(Path(record["nas_directory"]) / "commit.json", {"complete": False})
    value = reader({"root": str(root)}, now=NOW)
    assert value["state"] == "Metadata error" and value["committed"] == 0
    assert "SHA/size differs" in value["errors"][0]


def test_float_progress_timestamp_is_rejected_not_used_as_advance(job):
    root, cases, _ = job
    attempt = claim(job)
    path = attempt / "progress_00000000.json"
    put(
        path,
        dict(
            sequence=0,
            run="ac52",
            attempt_id=attempt.name,
            event=dict(state="frame_started", case_id=cases[0]),
            written_unix_ns=1.25,
        ),
    )
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert (
        value["state"] == "Metadata error" and "integer timestamp" in value["errors"][0]
    )


def test_failure_keeps_literal_traceback_and_real_partial_count(job):
    root, cases, task = job
    attempt = claim(job)
    progress(
        attempt,
        0,
        dict(
            state="frame_completed",
            complete=True,
            case_id=cases[0],
            publication=publication(root, cases[0]),
        ),
        NOW - 10,
    )
    text = "Traceback (most recent call last):\n  bad.py, line 2\nValueError: Ω failed verbatim\n"
    put(
        root / "failed" / "ac52.json",
        dict(
            run="ac52",
            attempt_id=task["attempt_id"],
            actual_worker=dict(machine="Main"),
            traceback=text,
            failed_utc="2026-01-01T00:02:00Z",
        ),
    )
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert (
        value["state"] == "Failed"
        and value["committed"] == 1
        and value["eta_seconds"] is None
    )
    assert value["runs"][0]["error"] == text and value["workers"][0]["error"] == text


def test_missing_created_time_gives_no_made_up_rate_or_eta(job):
    root, cases, _ = job
    attempt = claim(job)
    master = model.read_json(root / "ui" / "master.json")
    master.pop("created_utc")
    put(root / "ui" / "master.json", master)
    progress(
        attempt,
        0,
        dict(
            state="frame_completed",
            complete=True,
            case_id=cases[0],
            publication=publication(root, cases[0]),
        ),
        NOW - 1,
    )
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert value["frames_per_minute"] is None and value["eta_seconds"] is None
    assert value["created_utc"] == "Not recorded"


def test_idle_log_is_report_only_and_never_fabricates_working_state(job):
    root, _, _ = job
    log = root / "logs" / "Main.log"
    log.parent.mkdir()
    event = dict(
        kind="completed", run="ac52", worker_slot=0, worker_generation="literal"
    )
    log.write_text("2026-01-01T00:01:00Z " + json.dumps(event) + "\n", encoding="utf-8")
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert value["state"] == "Waiting" and "Idle" in value["workers"][0]["state"]
    assert value["committed"] == 0 and value["logs"][0]["tail"].endswith("\n")


def test_resume_counts_only_bound_original_producer_not_an_unowned_old_output(job):
    root, cases, task = job
    attempt = claim(job)
    old_token = "2" * 32
    record = publication(root, cases[0], old_token)
    old_attempt = root / "results" / "ac52" / ".attempts" / old_token
    receipt = put(old_attempt / "ops" / "nas_copy_fixture.json", record)
    producer = dict(
        case_id=cases[0],
        attempt_directory=str(old_attempt),
        publication=receipt,
        commit=model.descriptor(Path(record["nas_directory"]) / "commit.json"),
    )
    put(
        attempt / "resume.json",
        dict(prefix_case_ids=[cases[0]], frame_producers=[producer]),
    )
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert value["committed"] == 1 and value["state"] == "Claimed"
    assert value["machines"][0]["frames_per_minute"] is None
    producer["commit"] = put(
        root / "foreign" / "commit.json",
        dict(complete=True, frame_id=cases[0], products=dict(precision="float32")),
    )
    put(
        attempt / "resume.json",
        dict(prefix_case_ids=[cases[0]], frame_producers=[producer]),
    )
    invalid = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert invalid["committed"] == 0 and "paths differ" in invalid["errors"][0]


def selection_fixture(tmp_path, *, portable=False):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version=1\nseries_id="unit"\n[runs]\nids=["ac52"]\n', encoding="utf-8"
    )
    if portable:
        with (root/"series.toml").open("a",encoding="utf-8") as stream:
            stream.write('[roots]\nseries="."\n')
    protocol = put(root / "protocol.json", {"fixture": "metadata-only"})
    paths = [str(root / "not-opened-Ω.jpg"), str(root / "not-opened-9.jpg")]
    frames = [
        dict(
            case_id=case,
            elapsed_h=t,
            native_task=dict(
                kind="source",
                arguments=dict(
                    camera_jpeg=dict(path=path, sha256="e" * 64, size_bytes=999)
                ),
            ),
        )
        for case, path, t in zip(["ac52_f1", "ac52_f9"], paths, [0.25, 2.0])
    ]
    if portable:
        # Opaque authored bytes only, never a decoded/historical image.
        for frame in frames:
            bound=frame["native_task"]["arguments"]["camera_jpeg"]
            Path(bound["path"]).write_bytes(b"x"*999)
            bound["sha256"]=hashlib.sha256(b"x"*999).hexdigest()
    configuration = put(root / "scientific.json", {"frames": frames})
    planpath = root / "execution.toml"
    planpath.write_text(
        'format="darsia-series-v1"\nadopted_for_execution=true\njob_id="tiny"\n[job]\nprecision="float32"\n',
        encoding="utf-8",
    )
    plan = model.descriptor(planpath)
    declaration = dict(run="ac52", configuration=configuration)
    resolved = dict(
        format="fluidflower-resolved-series-image-selection-v1",
        series_id="unit",
        run_id="ac52",
        manifest=model.descriptor(root / "series.toml"),
        source_metadata=[protocol],
        selected_frame_count=2,
        selected=[
            dict(input_index=i, frame_index=i, source_path=p, elapsed_h=t)
            for i, p, t in zip([0, 8], paths, [0.25, 2.0])
        ],
    )

    def admit(path, *, expected_sha256):
        assert path == planpath and expected_sha256 == plan["sha256"]
        return dict(runs=[declaration]), [dict(frames=frames)], plan

    return root, plan, frames, resolved, admit


def test_creation_blocks_unbound_or_mismatching_b4_before_existing_producer(
    tmp_path, monkeypatch
):
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        job_filesystem.SharedFolderAdmission(True, "smb", "isolated unit transport", "not a real FS qualification"))
    root, plan, _, resolved, admit = selection_fixture(tmp_path)
    monkeypatch.setattr(model, "require_unc", lambda path: None)
    calls = []
    kwargs = dict(
        display_name="tiny",
        series_label="local",
        series_root=root,
        series_id="unit",
        admitter=admit,
        preparer=lambda *a, **k: calls.append(a),
    )
    with pytest.raises(ValueError, match="blocked"):
        model.create_job_from_plan(
            plan, tmp_path / "job", selection_provider=None, **kwargs
        )
    mismatch = deepcopy(resolved)
    mismatch["selected"][1]["elapsed_h"] = 2.5
    with pytest.raises(ValueError, match="source path/order/acquisition"):
        model.create_job_from_plan(
            plan, tmp_path / "job", selection_provider=lambda runs: [mismatch], **kwargs
        )
    assert not calls and not (tmp_path / "job").exists()


def test_explicit_creation_original_producer_only_and_ui_b4_receipt(
    tmp_path, monkeypatch
):
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        job_filesystem.SharedFolderAdmission(True, "smb", "isolated unit transport", "not a real FS qualification"))
    root, plan, frames, resolved, admit = selection_fixture(tmp_path, portable=True)
    monkeypatch.setattr(model, "require_unc", lambda path: None)
    target = tmp_path / "job"
    calls = []
    original = bytes_of(root)

    def preparer(path, destination, *, expected_sha256, series_manifest):
        calls.append((path, destination, expected_sha256, series_manifest))
        assert series_manifest==model.descriptor(root/"series.toml")
        put(
            destination / "job.json",
            {"expected_runs": ["ac52"], "precision": "float32"},
        )
        return dict(
            complete=True,
            numerical_calls_started=False,
            producer="original-injected-metadata-only",
            series=dict(root_alias="series",relative_path="execution.toml",
                        sha256=plan["sha256"],bytes=plan["size_bytes"],role="configuration"),
        )

    row = model.create_job_from_plan(
        dict(plan, label="GUI only"),
        target,
        display_name="Name",
        series_label="local",
        series_root=root,
        series_id="unit",
        selection_provider=lambda runs: [resolved],
        admitter=admit,
        preparer=preparer,
        machine="Main",
    )
    assert len(calls) == 1 and row["worker_started"] is False
    binding = model.read_json(target / "ui" / "image_selection_binding.json")
    assert (
        binding["execution_entries_unchanged"] and not binding["raw_hashed_by_this_UI"]
    )
    assert [r["case_id"] for r in binding["runs"][0]["cases"]] == [
        f["case_id"] for f in frames
    ]
    assert [r["b4_input_index"] for r in binding["runs"][0]["cases"]] == [0, 8]
    assert model.read_json(target / "ui" / "master.json")[
        "image_selection_binding"
    ] == dict(relative_path="ui/image_selection_binding.json",
              sha256=model.descriptor(target/"ui/image_selection_binding.json")["sha256"],
              bytes=model.descriptor(target/"ui/image_selection_binding.json")["size_bytes"])
    assert bytes_of(root) == original
    assert all(Path(frame["native_task"]["arguments"]["camera_jpeg"]["path"]).read_bytes()==b"x"*999
               for frame in frames)


def test_b4_changes_during_creation_preserve_original_metadata_without_false_master(
    tmp_path, monkeypatch
):
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        job_filesystem.SharedFolderAdmission(True, "smb", "isolated unit transport", "not a real FS qualification"))
    root, plan, _, resolved, admit = selection_fixture(tmp_path, portable=True)
    monkeypatch.setattr(model, "require_unc", lambda path: None)
    target = tmp_path / "job"

    def preparer(path, destination, **kwargs):
        put(destination / "job.json", {"preserve": "original producer output"})
        (root / "series.toml").write_bytes(b"changed manifest")
        return dict(complete=True, numerical_calls_started=False)

    with pytest.raises(ValueError, match="metadata preserved"):
        model.create_job_from_plan(
            plan,
            target,
            display_name="x",
            series_label="local",
            series_root=root,
            series_id="unit",
            selection_provider=lambda runs: [resolved],
            admitter=admit,
            preparer=preparer,
        )
    assert (target / "job.json").exists() and not (
        target / "ui" / "master.json"
    ).exists()


def test_real_b4_resolver_bridge_keeps_protocol_order_without_raw_read(tmp_path):
    root, plan, frames, _, admit = selection_fixture(tmp_path)
    from darsia.fluidflower import series_image_selection as b4

    protocol = model.descriptor(root / "protocol.json")
    metadata = b4.RunImageMetadata(
        "ac52",
        tuple(
            dict(
                source_path=f["native_task"]["arguments"]["camera_jpeg"]["path"],
                elapsed_h=f["elapsed_h"],
                frame_index=i,
                image_id=f["case_id"],
            )
            for i, f in enumerate(frames)
        ),
        0.5,
        1,
        (protocol,),
        (),
        str(root),
    )
    original = (root / "series.toml").read_bytes()
    b4.save_run_image_selection(
        root,
        metadata,
        [row["source_path"] for row in metadata.frames],
        expected_bytes=original,
    )

    def provider(runs):
        assert runs == ["ac52"]
        return [b4.resolve_saved_selection(root, "ac52", metadata)]

    binding = model.validate_current_selection(
        plan,
        series_root=root,
        series_id="unit",
        selection_provider=provider,
        admitter=admit,
    )
    assert [r["case_id"] for r in binding["runs"][0]["cases"]] == ["ac52_f1", "ac52_f9"]
    assert not any(root.glob("*.jpg"))


def test_rig_geometry_metadata_bridge_and_declared_same_bytes_relocation(tmp_path):
    root, plan, frames, resolved, original_admit = selection_fixture(tmp_path)
    physical = root / "physical-only-raw-not-read.jpg"
    logical = frames[0]["native_task"]["arguments"]["camera_jpeg"]
    geometry = put(
        root / "geometry.json",
        dict(camera_jpeg_path=logical["path"], camera_jpeg_sha256=logical["sha256"]),
    )
    frames[0]["native_task"] = dict(kind="rig", geometry=geometry)
    resolved["selected"][0]["source_path"] = str(physical)
    alias = dict(logical=logical, physical=dict(logical, path=str(physical)))

    def admit(*a, **k):
        value, configs, bound = original_admit(*a, **k)
        configs[0]["data_aliases"] = [alias]
        return value, configs, bound

    binding = model.validate_current_selection(
        plan,
        series_root=root,
        series_id="unit",
        selection_provider=lambda runs: [resolved],
        admitter=admit,
    )
    assert binding["runs"][0]["cases"][0]["original_camera_source"]["path"] == str(
        physical
    )
    assert not physical.exists()
    alias["physical"]["size_bytes"] = 17
    with pytest.raises(ValueError, match="byte count"):
        model.validate_current_selection(
            plan,
            series_root=root,
            series_id="unit",
            selection_provider=lambda runs: [resolved],
            admitter=admit,
        )
