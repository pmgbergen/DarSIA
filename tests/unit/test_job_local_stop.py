"""Explicit post-stop observations over real synthetic queue ownership."""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from darsia.fluidflower import job_control, job_local_stop
from darsia.fluidflower.pipeline import job_queue
from .test_setup_failed_build import failed_build, unstarted, setup_job, controls
from .test_setup_job_reprepare import _doc, _retry
from .test_setup_image_preparation_update import completed_geometry, save_independent_choice
from .test_setup_completed_builds import _run


@pytest.fixture
def stopped(failed_build, monkeypatch):
    case = failed_build
    from darsia.fluidflower import store_map
    monkeypatch.setattr(store_map, "default_store_map_path", lambda: Path(case.stores.source["path"]))
    root = case.job_root
    apply = job_queue.read_json(root / "claimed/private-host-cpu/setup-apply.json")
    task = job_queue.read_json(root / "claimed/private-host-cpu/sample.json")
    base = root.parent / root.name
    original = base / "sample/.attempts" / task["attempt_id"] / "retirement.json"
    assert original.is_file()  # The original supervisor identity is retained.
    retired = job_queue.read_json(original)
    request = next((root / "ui/launches").glob("*.request.json"))
    value = job_queue.read_json(request)
    value["requested_utc"] = datetime.fromtimestamp(
        float(retired["supervisor"]["process_create_time"]) - 1, timezone.utc).isoformat()
    _doc(request, value)
    directory = root / value["launch_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    reference = job_control._bindings(root)[0]
    _doc(directory / "launch.json", dict(format="darsia-job-local-launch-v2",
        job=reference(root / "job.json"), machine="private-host", backend="cpu", workers=1,
        entrypoint="python -B -m darsia_service.prestart -- python -m darsia.fluidflower.worker",
        started=True, launcher_pid=None))
    (directory / "console.log").write_text(value["requested_utc"] + ' {"kind":"started"}\n')
    log = root / "logs/private-host-cpu.log"
    log.write_bytes(log.read_bytes().replace(b'"kind": "stopped"', b'"kind": "interrupted"'))
    missing = base / "setup-apply/.attempts" / apply["attempt_id"] / "retirement.json"
    assert not missing.exists()
    for path in (root / ".capacity").glob("*/r-*/released.json"):
        if job_queue.read_json(path.parent / "reserved.json")["run"] == "sample":
            path.unlink()  # Interrupted supervisor did not release its reservation.
    previous = job_control.read_control(root)
    job_control.publish_control(root, "stop", machine="private-host", expected_previous=previous.get("command"))
    return case, missing


def test_gui_local_confirmation_is_separate_and_allows_same_job_update(stopped):
    from darsia.fluidflower.job_launch_state import read_launch_requests
    from darsia.fluidflower.job_lifecycle import require_service_retirement
    case, missing = stopped
    with pytest.raises(ValueError):
        require_service_retirement(case.job_root)
    pin = job_local_stop.confirm_local_stop(case.job_root, machine="private-host", actor="test operator",
                                           stores_path=case.stores.source["path"])
    document = job_queue.read_json(Path(pin["path"]))
    assert document["scientific_completion_claimed"] is False
    assert document["worker_retirement_claimed"] is False
    assert not missing.exists()
    assert all(row["local_stop_confirmation"] == pin for row in read_launch_requests(case.job_root))
    releases = [job_queue.read_json(path) for path in (case.job_root / ".capacity").glob("*/r-*/released.json")]
    assert any(row.get("local_stop", {}).get("confirmation") == pin and row["retirement"] is None for row in releases)
    require_service_retirement(case.job_root)
    result = _retry(case)
    assert result["root"] == str(case.job_root)
    assert (case.job_root / "queue/setup-apply.json").is_file()
    assert not missing.exists()
    assert len(case.registry.jobs(case.series, "synthetic")) == 1


def test_recovery_snapshot_keeps_lock_inode_while_nonce_changes(tmp_path):
    from darsia.fluidflower.setup_job_reprepare import _snapshot
    path = tmp_path / ".capacity/volume/admission.lock"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"first nonce")
    original = _snapshot(tmp_path)
    path.write_bytes(b"next lock acquisition nonce")
    assert _snapshot(tmp_path) == original
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(b"replacement inode")
    replacement.replace(path)
    assert _snapshot(tmp_path) != original


@pytest.mark.parametrize("fault", ["live", "foreign", "output", "missing-supervisor"])
def test_local_confirmation_refuses_unproved_or_scientific_state(stopped, monkeypatch, fault):
    from darsia.fluidflower.pipeline import worker
    case, _ = stopped
    if fault == "live":
        monkeypatch.setattr(worker, "_same_process", lambda _: object())
    elif fault == "foreign":
        pass
    elif fault == "output":
        Path(case.plan["build_plans"][0]["request_template"]["output"]["destination"]).mkdir(parents=True)
    else:
        for path in (case.job_root.parent / case.job_root.name).glob("*/.attempts/*/retirement.json"):
            path.unlink()
    with pytest.raises(ValueError):
        job_local_stop.confirm_local_stop(case.job_root, machine="foreign" if fault == "foreign" else "private-host",
                                          actor="test", stores_path=case.stores.source["path"])
    assert not list((case.job_root / "ui/launches").glob("*.local-stop.json"))


def test_local_confirmation_does_not_survive_new_run_or_changed_identity(stopped):
    case, _ = stopped
    pin = job_local_stop.confirm_local_stop(case.job_root, machine="private-host", actor="test",
                                          stores_path=case.stores.source["path"])
    value = job_queue.read_json(Path(pin["path"]))
    original_identities = list(value["identities"])
    value["identities"] = value["identities"][1:]
    _doc(Path(pin["path"]), value)
    with pytest.raises(ValueError, match="process identities"):
        job_local_stop.verified_local_stop(case.job_root, machine="private-host", stores_path=case.stores.source["path"])
    # Restore the immutable test document, then explicitly change the command.
    value["identities"] = original_identities
    _doc(Path(pin["path"]), value)
    state = job_control.read_control(case.job_root)
    job_control.publish_control(case.job_root, "run", machine="private-host", expected_previous=state["command"])
    with pytest.raises(ValueError, match="Stop/Pause"):
        job_local_stop.verified_local_stop(case.job_root, machine="private-host", stores_path=case.stores.source["path"])


@pytest.fixture
def stopped_images(setup_job, monkeypatch):
    import subprocess
    from copy import deepcopy
    from darsia.fluidflower import store_map
    native = _doc(Path(setup_job.plan["native_parameters"]["path"]),
                  dict(format="private-native-parameter-metadata", source={}))
    setup_job.plan["native_parameters"] = native
    for build in setup_job.plan["build_plans"]:
        build["native_parameters"] = deepcopy(native)
    case = completed_geometry(setup_job, monkeypatch)
    monkeypatch.setattr(store_map, "default_store_map_path", lambda: Path(case.stores.source["path"]))
    case.identity["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save_independent_choice(case)
    _retry(case)
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_MODE", "failed")
    _run(case, monkeypatch)
    root = case.job_root
    task = job_queue.read_json(root / "claimed/private-host-cpu/setup-image-preparation.json")
    output = root.parent / root.name / task["run"] / ".attempts" / task["attempt_id"]
    retirement = job_queue.read_json(output / "retirement.json")
    request = next((root / "ui/launches").glob("*.request.json"))
    launch = job_queue.read_json(request)
    launch["workers"] = 6  # The final single image task uses less than the configured capacity.
    launch["requested_utc"] = datetime.fromtimestamp(
        float(retirement["supervisor"]["process_create_time"]) - 1, timezone.utc).isoformat()
    _doc(request, launch)
    directory = root / launch["launch_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    reference = job_control._bindings(root)[0]
    _doc(directory / "launch.json", dict(format="darsia-job-local-launch-v2",
        job=reference(root / "job.json"), machine="private-host", backend="cpu", workers=6,
        entrypoint="python -B -m darsia_service.prestart -- python -m darsia.fluidflower.worker",
        started=True, launcher_pid=None))
    (directory / "console.log").write_text(launch["requested_utc"] + ' {"kind":"started"}\n')
    log = root / "logs/private-host-cpu.log"
    log.write_bytes(log.read_bytes().replace(b'"kind": "stopped"', b'"kind": "interrupted"'))
    for path in (root / ".capacity").glob("*/r-*/released.json"):
        if job_queue.read_json(path.parent / "reserved.json")["run"] == task["run"]:
            path.unlink()  # The interrupted parent did not release this claim.
    prior = job_control.read_control(root)
    job_control.publish_control(root, "pause", machine="private-host", expected_previous=prior.get("command"))
    return case, task, output


def test_local_confirmation_preserves_completed_geometry_and_retries_only_failed_images(stopped_images, monkeypatch):
    from .test_setup_job_reprepare import _bytes
    from darsia.fluidflower.setup_completed_builds import retained_operations
    from darsia.fluidflower.job_lifecycle import require_service_retirement
    from darsia.fluidflower.jobs_metadata import JobSnapshotReader
    case, task, output = stopped_images
    originals = {build["run_id"]: _bytes(Path(build["request_template"]["output"]["destination"]))
                 for build in case.plan["build_plans"]}
    before = _bytes(output)
    with pytest.raises(ValueError):
        require_service_retirement(case.job_root)
    pin = job_local_stop.confirm_local_stop(case.job_root, machine="private-host", actor="test operator",
                                           stores_path=case.stores.source["path"])
    proof = job_queue.read_json(pin["path"])
    assert proof["scientific_completion_claimed"] is False and proof["worker_retirement_claimed"] is False
    assert proof["claims"][task["run"]]["actual_worker"] in proof["identities"]
    assert proof["supervisors"] and _bytes(output) == before
    require_service_retirement(case.job_root)
    result = _retry(case)
    assert result["image_preparation_added"]
    assert set(retained_operations(case.job_root)) == {"setup-apply", *case.plan["runs"]}
    assert {path.stem for path in (case.job_root / "queue").glob("*.json")} == {"setup-image-preparation"}
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_MODE", "complete")
    _run(case, monkeypatch)
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert snapshot["complete"], snapshot["errors"]
    for build in case.plan["build_plans"]:
        assert _bytes(Path(build["request_template"]["output"]["destination"])) == originals[build["run_id"]]


@pytest.mark.parametrize("fault", ["live", "identity"])
def test_failed_image_local_confirmation_refuses_live_or_changed_identity(stopped_images, monkeypatch, fault):
    from darsia.fluidflower.pipeline import worker
    case, task, _output = stopped_images
    if fault == "live":
        monkeypatch.setattr(worker, "_same_process", lambda _identity: object())
    else:
        owner = case.job_root / "leases" / (task["run"] + "." + task["attempt_id"]) / "owner.json"
        value = job_queue.read_json(owner)
        value["actual_owner"]["pid"] += 1
        _doc(owner, value)
    with pytest.raises(ValueError):
        job_local_stop.confirm_local_stop(case.job_root, machine="private-host", actor="test operator",
                                          stores_path=case.stores.source["path"])
    assert not list((case.job_root / "ui/launches").glob("*.local-stop.json"))
