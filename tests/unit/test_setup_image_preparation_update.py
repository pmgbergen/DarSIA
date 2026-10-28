"""One GUI update adds optical preparation without repeating saved geometry."""
from copy import deepcopy
from pathlib import Path

import pytest
import toml
import tomllib

from darsia.fluidflower import job_control
from darsia.fluidflower.pipeline import job_queue, worker
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from darsia.fluidflower.setup_completed_builds import retained_operations
from .test_series_setup_jobs import setup_job, controls  # noqa: F401
from .test_setup_job_reprepare import _unstarted_case, _retry, _bytes, _doc
from .test_setup_completed_builds import _apply, _run


def completed_geometry(case, monkeypatch):
    source = Path(case.plan["series_manifest"]["path"])
    document = tomllib.loads(source.read_text())
    document["method"] = dict(active="phase_model")
    document["runs"] = dict(ids=list(case.plan["runs"]))
    source.write_text(toml.dumps(document), encoding="utf-8")
    case.plan["series_manifest"] = job_queue.descriptor(source)
    case.plan["metadata_output"] = dict(maximum_output_bytes=20 * 1024**2,
                                       maximum_file_bytes=5 * 1024**2)
    for build in case.plan["build_plans"]:
        build["metadata_output"] = deepcopy(case.plan["metadata_output"])
        build["request_template"]["series_context"]["manifest"] = deepcopy(case.plan["series_manifest"])
    case = _unstarted_case(case, monkeypatch)
    # An immutable original snapshot is already written by real current Setup.
    (case.job_root / "setup-controls/series-manifest.toml").write_bytes(source.read_bytes())
    _apply(case, monkeypatch)
    for build in case.plan["build_plans"]:
        Path(build["request_template"]["output"]["intended_root"]).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DARSIA_TEST_FOLDER_COMPLETED_BUILD_RUNS", ",".join(case.plan["runs"]))
    _run(case, monkeypatch)
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert snapshot["complete"], snapshot["errors"]
    monkeypatch.setattr(worker, "_same_process", lambda identity: None)
    return case


def save_independent_choice(case):
    from darsia.fluidflower.image_normalisation import save_choice
    source = case.series / "series.toml"
    save_choice(case.series, "synthetic", case.plan["runs"], "off", None,
                color_method="off", expected_bytes=source.read_bytes())


def initial_image_job(setup_job, monkeypatch):
    from darsia.fluidflower.image_normalisation import save_choice
    source = Path(setup_job.plan["series_manifest"]["path"])
    document = tomllib.loads(source.read_text())
    document["method"] = dict(active="phase_model")
    document["runs"] = dict(ids=list(setup_job.plan["runs"]))
    source.write_text(toml.dumps(document), encoding="utf-8")
    save_choice(source.parent, "synthetic", setup_job.plan["runs"], "off", None,
                color_method="off", expected_bytes=source.read_bytes())
    setup_job.plan["series_manifest"] = job_queue.descriptor(source)
    setup_job.plan["metadata_output"] = dict(maximum_output_bytes=20 * 1024**2,
                                            maximum_file_bytes=5 * 1024**2)
    for build in setup_job.plan["build_plans"]:
        build["metadata_output"] = deepcopy(setup_job.plan["metadata_output"])
        build["request_template"]["series_context"]["manifest"] = deepcopy(setup_job.plan["series_manifest"])
    return _unstarted_case(setup_job, monkeypatch)


def test_initial_pending_image_job_can_update_before_apply(setup_job, monkeypatch):
    case = initial_image_job(setup_job, monkeypatch)
    result = _retry(case)
    assert result["complete"] and not result.get("image_preparation_added")
    assert {path.stem for path in (case.job_root / "queue").glob("*.json")} == {
        "setup-apply", "setup-image-preparation"}
    assert not retained_operations(case.job_root)


def test_completed_geometry_update_adds_only_image_operation(setup_job, monkeypatch):
    case = completed_geometry(setup_job, monkeypatch)
    original_plan = (case.job_root / "setup-controls/plan.json").read_bytes()
    original_outputs = {build["run_id"]: _bytes(Path(build["request_template"]["output"]["destination"]))
                        for build in case.plan["build_plans"]}
    original_job = job_queue.read_json(case.job_root / "job.json")
    save_independent_choice(case)
    result = _retry(case)
    assert result["image_preparation_added"] and result["paused"]
    assert job_control.read_control(case.job_root)["action"] == "pause"
    job = job_queue.read_json(case.job_root / "job.json")
    assert job["setup_plan"] == original_job["setup_plan"]
    assert job["series_manifest"] == original_job["series_manifest"]
    assert job["expected_runs"] == [*original_job["expected_runs"], "setup-image-preparation"]
    assert (case.job_root / "setup-controls/plan.json").read_bytes() == original_plan
    assert [path.name for path in (case.job_root / "queue").glob("*.json")] == ["setup-image-preparation.json"]
    assert set(retained_operations(case.job_root)) == set(original_job["expected_runs"])
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert not snapshot["complete"] and not snapshot["errors"]
    assert snapshot["completed_operations"] == len(original_job["expected_runs"])
    pending = snapshot["runs"][-1]
    assert pending["run"] == "setup-image-preparation" and pending["operation"] == "image-preparation"
    assert pending["state"] in {"Queued", "Paused"}
    for build in case.plan["build_plans"]:
        assert _bytes(Path(build["request_template"]["output"]["destination"])) == original_outputs[build["run_id"]]
    # A second GUI Update before Start must retain the same completed geometry,
    # rather than trying to find a now-historical local Apply queue again.
    case.identity["commit"] = "5" * 40
    second = _retry(case)
    assert second["image_preparation_added"]
    assert set(retained_operations(case.job_root)) == set(original_job["expected_runs"])
    assert [path.name for path in (case.job_root / "queue").glob("*.json")] == ["setup-image-preparation.json"]
    assert _retry(case)["unchanged"]


@pytest.mark.parametrize("change", ["geometry", "camera", "runs"])
def test_completed_upgrade_refuses_non_optical_series_change(setup_job, monkeypatch, change):
    case = completed_geometry(setup_job, monkeypatch)
    save_independent_choice(case)
    source = case.series / "series.toml"
    document = tomllib.loads(source.read_text())
    document[change] = dict(changed=True)
    source.write_text(toml.dumps(document), encoding="utf-8")
    before = _bytes(case.job_root)
    with pytest.raises(ValueError, match="beyond the lighting choice"):
        _retry(case)
    assert _bytes(case.job_root) == before


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_failed_or_cancelled_images_retry_without_repeating_geometry(setup_job, monkeypatch, state):
    import subprocess
    native = setup_job.plan["native_parameters"]
    native = _doc(Path(native["path"]), dict(format="private-native-parameter-metadata", source={}))
    setup_job.plan["native_parameters"] = native
    for build in setup_job.plan["build_plans"]:
        build["native_parameters"] = deepcopy(native)
    case = completed_geometry(setup_job, monkeypatch)
    case.identity["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    before = {build["run_id"]: _bytes(Path(build["request_template"]["output"]["destination"]))
              for build in case.plan["build_plans"]}
    save_independent_choice(case)
    _retry(case)
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_MODE", state)
    _run(case, monkeypatch)
    claims = list((case.job_root / "claimed").glob("*/setup-image-preparation.json"))
    assert len(claims) == 1
    first_attempt = job_queue.read_json(claims[0])["attempt_id"]
    config = job_queue.read_json(case.job_root / "setup-controls/image-operation.json")
    partial = Path(config["output_root"]) / first_attempt
    partial_bytes = _bytes(partial)
    assert partial_bytes and not (partial / "prepared_image_basis.json").exists()
    if state == "failed":
        # A user may correct the optical choice after a failed attempt, with
        # no need to compute already completed geometry again.
        from darsia.fluidflower.image_normalisation import save_choice
        source = case.series / "series.toml"
        save_choice(case.series, "synthetic", case.plan["runs"], "match_reference_lighting",
                    case.plan["runs"][0], color_method="off", expected_bytes=source.read_bytes())
    retry = _retry(case)
    archived = Path(retry["history"]) / "image-partials" / first_attempt
    assert _bytes(archived) == partial_bytes and not partial.exists()
    assert {path.stem for path in (case.job_root / "queue").glob("*.json")} == {"setup-image-preparation"}
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_MODE", "complete")
    _run(case, monkeypatch)
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert snapshot["complete"], snapshot["errors"]
    assert snapshot["completed_operations"] == len(before) + 2
    final = snapshot["runs"][-1]["result"]
    assert final["image_preparation"]["path"] != str(partial / "prepared_image_basis.json")
    assert final["phase_inference_performed"] is False
    assert final["model_training_performed"] is False
    assert _bytes(archived) == partial_bytes
    for build in case.plan["build_plans"]:
        assert _bytes(Path(build["request_template"]["output"]["destination"])) == before[build["run_id"]]


def test_from_scratch_job_retries_images_using_its_local_geometry_completions(setup_job, monkeypatch):
    import subprocess
    native = _doc(Path(setup_job.plan["native_parameters"]["path"]),
                  dict(format="private-native-parameter-metadata", source={}))
    setup_job.plan["native_parameters"] = native
    for build in setup_job.plan["build_plans"]:
        build["native_parameters"] = deepcopy(native)
    case = initial_image_job(setup_job, monkeypatch)
    case.identity["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    _apply(case, monkeypatch)
    for build in case.plan["build_plans"]:
        Path(build["request_template"]["output"]["intended_root"]).mkdir(parents=True, exist_ok=True)
    _retry(case)  # Use a real source commit for the original image code lease.
    monkeypatch.setenv("DARSIA_TEST_FOLDER_COMPLETED_BUILD_RUNS", ",".join(case.plan["runs"]))
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_MODE", "failed")
    _run(case, monkeypatch)
    before = {build["run_id"]: _bytes(Path(build["request_template"]["output"]["destination"]))
              for build in case.plan["build_plans"]}
    assert not retained_operations(case.job_root)
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert snapshot["failed_runs"] == 1
    assert snapshot["completed_operations"] == len(before) + 1
    monkeypatch.setattr(worker, "_same_process", lambda identity: None)
    result = _retry(case)
    assert result["image_preparation_added"]
    assert set(retained_operations(case.job_root)) == {"setup-apply", *before}
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_MODE", "complete")
    _run(case, monkeypatch)
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert snapshot["complete"], snapshot["errors"]
    for build in case.plan["build_plans"]:
        assert _bytes(Path(build["request_template"]["output"]["destination"])) == before[build["run_id"]]


def image_admission_failure(setup_job, monkeypatch):
    """Real queue rename and empty lease, before the scientific owner exists."""
    import subprocess
    native = _doc(Path(setup_job.plan["native_parameters"]["path"]),
                  dict(format="private-native-parameter-metadata", source={}))
    setup_job.plan["native_parameters"] = native
    for build in setup_job.plan["build_plans"]:
        build["native_parameters"] = deepcopy(native)
    case = initial_image_job(setup_job, monkeypatch)
    case.identity["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    _apply(case, monkeypatch)
    for build in case.plan["build_plans"]:
        Path(build["request_template"]["output"]["intended_root"]).mkdir(parents=True, exist_ok=True)
    _retry(case)
    monkeypatch.setenv("DARSIA_TEST_FOLDER_COMPLETED_BUILD_RUNS", ",".join(case.plan["runs"]))
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_ADMISSION_FAILURE", "1")
    _run(case, monkeypatch)
    monkeypatch.delenv("DARSIA_TEST_FOLDER_IMAGE_ADMISSION_FAILURE")
    monkeypatch.setattr(worker, "_same_process", lambda identity: None)
    claim = next((case.job_root / "claimed").glob("*/setup-image-preparation.json"))
    task = job_queue.read_json(claim)
    lease = case.job_root / "leases" / (task["run"] + "." + task["attempt_id"])
    assert lease.is_dir() and not list(lease.iterdir())
    assert {path.stem for path in (case.job_root / "done").glob("*.json")} == {
        "setup-apply", *case.plan["runs"]}
    assert not list((case.job_root / "failed").iterdir())
    return case, claim, lease


def test_image_admission_retry_preserves_completed_geometry_before_owner(setup_job, monkeypatch):
    case, claim, lease = image_admission_failure(setup_job, monkeypatch)
    before = {build["run_id"]: _bytes(Path(build["request_template"]["output"]["destination"]))
              for build in case.plan["build_plans"]}
    completed = _bytes(case.job_root / "done")
    prior = job_control.read_control(case.job_root)
    job_control.publish_control(case.job_root, "pause", machine="private-host",
                                expected_previous=prior.get("command"))
    result = _retry(case)
    history = Path(result["history"])
    assert _bytes(history / "previous" / "done") == completed
    assert (history / "previous" / claim.relative_to(case.job_root)).exists()
    assert not list((history / "previous" / lease.relative_to(case.job_root)).iterdir())
    assert set(retained_operations(case.job_root)) == {"setup-apply", *case.plan["runs"]}
    assert {path.stem for path in (case.job_root / "queue").glob("*.json")} == {"setup-image-preparation"}
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_MODE", "complete")
    _run(case, monkeypatch)
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert snapshot["complete"], snapshot["errors"]
    assert snapshot["completed_operations"] == len(before) + 2
    for build in case.plan["build_plans"]:
        assert _bytes(Path(build["request_template"]["output"]["destination"])) == before[build["run_id"]]


@pytest.mark.parametrize("mutation", ["claim", "foreign", "lease", "output"])
def test_image_admission_retry_refuses_unverified_work(setup_job, monkeypatch, mutation):
    case, claim, lease = image_admission_failure(setup_job, monkeypatch)
    task = job_queue.read_json(claim)
    if mutation == "claim":
        task["threads"] += 1
        _doc(claim, task)
    elif mutation == "foreign":
        foreign = claim.parent.parent / "another-host-cpu" / claim.name
        foreign.parent.mkdir()
        claim.rename(foreign)
    elif mutation == "lease":
        _doc(lease / "claimed.json", dict(run=task["run"], attempt_id=task["attempt_id"]))
    else:
        output = (Path(case.stores.roots("synthetic")["results"])
                  / case.job_root.name / task["run"] / ".attempts" / task["attempt_id"])
        output.mkdir(parents=True)
    before = _bytes(case.job_root)
    with pytest.raises(ValueError):
        _retry(case)
    assert _bytes(case.job_root) == before
