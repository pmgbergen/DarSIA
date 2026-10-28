"""Same logical Setup retry over real producer/catalogue/publication metadata."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_control, native_jobs_registry, setup_job_reprepare as recovery
from darsia.fluidflower.job_launch_state import begin_launch_request
from darsia.fluidflower.pipeline import job_queue, series_setup_jobs
from darsia.fluidflower.shared_job_folder import SharedJobFolder
from .test_series_setup_jobs import setup_job, _prepare  # noqa: F401
from .test_series_setup_inputs import controls, _doc  # noqa: F401
from .test_folder_cache_worker import _stores


@pytest.fixture
def unstarted(setup_job, monkeypatch):
    return _unstarted_case(setup_job,monkeypatch)


def _unstarted_case(setup_job,monkeypatch):
    case = setup_job
    # Reuse typed synthetic inputs, with genuinely fresh job-owned controls.
    case.plan["control_root"] = str(case.job_root / "setup-controls")
    case.plan["output_scope"] = dict(metadata_roots=[str(case.job_root)], scratch_roots=[])
    for build in case.plan["build_plans"]:
        build["output_scope"] = deepcopy(case.plan["output_scope"])
    case.plan_path = case.job_root / "setup-controls" / "plan.json"
    case.bound_plan = _doc(case.plan_path, case.plan)
    shutil.rmtree(Path(case.export["path"]).parent)
    _prepare(case)
    monkeypatch.setattr(native_jobs_registry, "require_unc", lambda _value: None)
    case.registry = native_jobs_registry.JobsUIRegistry(case.root / "registry")
    case.series = Path(case.plan["series_manifest"]["path"]).parent
    case.registry.register_job(case.series, "synthetic", case.job_root)
    case.stores = _stores(case.root, roots=dict(series=str(case.series), private=str(case.root),
                                              results=str(case.job_root.parent)))
    case.folder = SharedJobFolder(case.job_root.parent)
    case.publication = case.folder.publish(case.job_root, master_machine="private-host")
    case.folder.update_job(case.publication["job_id"], master_machine="private-host", state="paused")
    case.identity = dict(commit="3" * 40,
        environment_lock=dict(path="requirements-worker.lock", sha256="4" * 64))
    return case


def _retry(case, **choices):
    return recovery.reprepare_unstarted_setup(case.registry, case.series, "synthetic", case.job_root,
        machine=choices.pop("machine","private-host"), actor="fixture", stores_path=case.stores.source["path"],
        identity_provider=lambda: case.identity, **choices)


def _bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file() and "history" not in path.relative_to(root).parts}


@pytest.fixture
def normalization_case(setup_job,monkeypatch):
    import toml,tomllib
    source=Path(setup_job.plan["series_manifest"]["path"])
    document=tomllib.loads(source.read_text())
    document["runs"]=dict(ids=list(setup_job.plan["runs"]))
    source.write_text(toml.dumps(document),encoding="utf-8")
    setup_job.plan["series_manifest"]=job_queue.descriptor(source)
    for build in setup_job.plan["build_plans"]:
        build["request_template"]["series_context"]["manifest"]=deepcopy(setup_job.plan["series_manifest"])
    case=_unstarted_case(setup_job,monkeypatch)
    from darsia.fluidflower.project import FluidFlowerProject
    monkeypatch.setattr(FluidFlowerProject,"load",lambda root:SimpleNamespace(root=Path(root)))
    return case


def _save_normalization(case,method="match_reference_lighting"):
    from darsia.fluidflower.image_normalisation import save_choice
    source=Path(case.plan["series_manifest"]["path"])
    return save_choice(case.series,"synthetic",case.plan["runs"],method,
        case.plan["runs"][0] if method!="off" else None,expected_bytes=source.read_bytes())


def _current_plan_fixture(case,*,mutation=None):
    """Keep the actual queue/catalogue producer, substituting only plan discovery."""
    def producer(project,series_root,**options):
        from darsia.fluidflower.setup_plan_refresh import same_plan_authority
        old=deepcopy(case.plan)
        old["series_manifest"]=job_queue.descriptor(Path(series_root)/"series.toml")
        for build in old["build_plans"]:
            build["request_template"]["series_context"]["manifest"]=deepcopy(old["series_manifest"])
        if mutation:mutation(old)
        pin=_doc(Path(options["job_root"])/"setup-controls/plan.json",old)
        (Path(options["job_root"])/"setup-controls/series-manifest.toml").write_bytes((Path(series_root)/"series.toml").read_bytes())
        return series_setup_jobs.prepare_setup_job(pin,options["job_root"],job_id=case.job_root.name,
            commit=options["commit"],environment_lock=options["environment_lock"])
    return producer


def test_same_job_normalization_refresh_after_two_code_generations(normalization_case):
    import json
    from darsia.fluidflower.pipeline.series_setup_jobs import _immutable_bytes
    case=normalization_case
    original=case.job_root.joinpath("job.json").read_bytes()
    first=_retry(case)
    # Represent an actual older-generation receipt written before the explicit
    # replacement snapshot field existed, updating its publication SHA as well.
    history=Path(first["history"])
    receipt_path=history/"receipt.json"
    receipt=json.loads(receipt_path.read_text());receipt.pop("replacement_job")
    receipt_path.write_bytes(_immutable_bytes(receipt))
    revision=case.folder.metadata/"job-revisions"/case.publication["job_id"]/"00000000000000000001.json"
    publication=json.loads(revision.read_text())
    publication["receipt"].update({k:v for k,v in job_queue.descriptor(receipt_path).items() if k!="path"})
    revision.write_bytes(_immutable_bytes(publication))
    case.identity["commit"]="5"*40
    second=_retry(case)
    before=case.job_root.joinpath("job.json").read_bytes()
    before_plan=case.job_root.joinpath("setup-controls/plan.json").read_bytes()
    _save_normalization(case)
    case.identity["commit"]="6"*40
    result=_retry(case,plan_producer=_current_plan_fixture(case))
    new_history=Path(result["history"])
    record=json.loads((new_history/"receipt.json").read_text())
    assert (new_history/"previous/job.json").read_bytes()==before
    assert (new_history/"previous/setup-controls/plan.json").read_bytes()==before_plan
    assert record["plan_refresh"]["format"]=="darsia-setup-normalisation-refresh-v1"
    assert record["replacement_job"]==job_queue.descriptor(new_history/"replacement/job.json")
    assert (history/"previous/job.json").read_bytes()==original
    job=json.loads(case.job_root.joinpath("job.json").read_text())
    assert job["job_id"]==case.job_root.name and job["series"]["sha256"]!=json.loads(before)["series"]["sha256"]
    assert case.folder.jobs()[0]["publication_generation"]==3
    assert case.registry.jobs(case.series,"synthetic")[0]["job"]==job_queue.descriptor(case.job_root/"job.json")
    assert job_control.read_control(case.job_root)["action"]=="pause"
    # The refreshed producer's immutable series snapshot also permits a later
    # GUI choice change without relying on deterministic reconstruction.
    case.plan=json.loads((case.job_root/"setup-controls/plan.json").read_text())
    _save_normalization(case,"off")
    case.identity["commit"]="7"*40
    _retry(case,plan_producer=_current_plan_fixture(case))
    assert case.folder.jobs()[0]["publication_generation"]==4


def test_normalization_refresh_keeps_original_target_spelling_with_gui_display_case(normalization_case):
    case=normalization_case
    _save_normalization(case)
    original_producer=_current_plan_fixture(case)
    observed=[]
    def from_gui(project,series,**options):
        observed.append(options["target_machine"])
        assert options["target_machine"]==case.plan["target_machine"]=="private-host"
        return original_producer(project,series,**options)
    result=_retry(case,machine="Private-Host",plan_producer=from_gui)
    assert result["complete"] and observed==["private-host"]


@pytest.mark.parametrize("mutation",["unrelated-series","baseline","native","producer-failure"])
def test_normalization_refresh_refuses_other_changes_and_restores_originals(normalization_case,mutation):
    case=normalization_case
    _save_normalization(case)
    original=_bytes(case.job_root)
    if mutation=="unrelated-series":
        source=case.series/"series.toml"
        source.write_text(source.read_text()+"\n[unrelated]\nchanged=true\n")
        producer=_current_plan_fixture(case)
    elif mutation=="producer-failure":
        def producer(*args,**kwargs):raise RuntimeError("original planner refused")
    else:
        def changed(plan):
            if mutation=="baseline":plan["build_plans"][0]["request_template"]["baseline"]["sha256"]="f"*64
            else:plan["native_parameters"]["sha256"]="f"*64
        producer=_current_plan_fixture(case,mutation=changed)
    with pytest.raises((ValueError,RuntimeError)):
        _retry(case,plan_producer=producer)
    assert _bytes(case.job_root)==original
    assert case.folder.jobs()[0]["publication_generation"]==0


def test_same_job_reprepare_retains_original_bytes_and_original_science(unstarted):
    case = unstarted
    request = begin_launch_request(case.job_root, machine="private-host", backend="cpu", workers=1, kind="local")
    log = case.job_root / "logs" / "private-host-cpu.log"
    log.write_text(job_queue.utc() + ' {"kind":"startup_failure"}\n', encoding="utf-8")
    original = _bytes(case.job_root)
    publication = Path(case.folder.metadata / "jobs" / (case.publication["job_id"] + ".json"))
    published_bytes = publication.read_bytes()
    admission = case.folder._admission_path(case.publication["job_id"], 0)
    admitted_bytes = admission.read_bytes()
    source = {key: Path(case.plan[key]["path"]).read_bytes()
              for key in ("project_manifest", "adopted_revision", "series_manifest", "native_parameters")}
    result = _retry(case)
    history = Path(result["history"])
    assert result["root"] == str(case.job_root) and result["paused"] is True
    assert all((history / "previous" / relative).read_bytes() == raw
               for relative, raw in original.items() if not relative.startswith("setup-controls/"))
    assert all(Path(case.plan[key]["path"]).read_bytes() == raw for key, raw in source.items())
    assert (case.job_root / "setup-controls/plan.json").read_bytes() == original["setup-controls/plan.json"]
    assert (case.job_root / "setup-controls/apply-operation.json").read_bytes() == original["setup-controls/apply-operation.json"]
    new = job_queue.read_json(case.job_root / "job.json")
    assert new["job_id"] == case.job_root.name and new["commit"] == case.identity["commit"]
    assert job_control.read_control(case.job_root)["action"] == "pause"
    from darsia.fluidflower.job_lifecycle import user_status
    assert user_status(dict(root=str(case.job_root), runs=[]))["state"] == "Pending"
    assert not list((case.job_root / "ui/launches").glob("*.json"))
    assert publication.read_bytes() == published_bytes and admission.read_bytes() == admitted_bytes
    assert (history / "previous-publication/publication.json").read_bytes() == published_bytes
    assert (history / "previous-publication/admission.json").read_bytes() == admitted_bytes
    row = case.folder.jobs()[0]
    assert row["publication_generation"] == 1 and row["state"] == "paused"
    assert case.folder.publish(case.job_root, master_machine="private-host")["job_id"] == case.publication["job_id"]
    assert case.registry.jobs(case.series, "synthetic")[0]["job"] == job_queue.descriptor(case.job_root / "job.json")
    # Simulate an old v1 reader: its untouched publication refuses the new bytes.
    assert job_queue.read_json(publication)["job"]["sha256"] != job_queue.descriptor(case.job_root / "job.json")["sha256"]
    case.folder.update_job(row["job_id"], master_machine="private-host", state="run")
    assert case.folder.jobs()[0]["state"] == "paused"  # Original queue control wins.
    previous = job_control.read_control(case.job_root)["command"]
    from darsia.fluidflower.job_lifecycle import recover_job
    recover_job(case.job_root, machine="private-host", expected_previous=previous,
                stores_path=case.stores.source["path"])
    assert case.folder.jobs()[0]["state"] == "run"
    assert request["source"]["sha256"] in {value["sha256"] for value in
        job_queue.read_json(history / "receipt.json")["previous_files"].values()}


@pytest.mark.parametrize("kind", ["unknown", "active", "claim", "checkpoint", "done", "build", "export", "foreign-master", "reported-work"])
def test_refuses_unknown_or_actual_service_and_any_work(unstarted, kind):
    case = unstarted
    if kind in {"unknown", "active"}:
        begin_launch_request(case.job_root, machine="private-host", backend="cpu", workers=1, kind="local")
        if kind == "active":
            (case.job_root / "logs/private-host-cpu.log").write_text(job_queue.utc() + ' {"kind":"started"}\n')
    elif kind == "claim":
        path = case.job_root / "claimed/private-host-cpu/claim.json"
        path.parent.mkdir()
        path.write_bytes(b"{}")
    elif kind == "checkpoint":
        path = case.job_root / "setup-apply/.attempts/one/checkpoint.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"{}")
    elif kind == "done":
        (case.job_root / "done/setup-apply.json").write_bytes(b"{}")
    elif kind == "build":
        Path(case.plan["build_plans"][0]["request_template"]["output"]["destination"]).mkdir(parents=True)
    elif kind == "export":
        export = Path(case.plan["project_root"]) / "exports/native_setup" / ("a" * 32)
        _doc(export / "apply-complete.json", dict(facies_rebuild=dict(raw_setup_settings=dict(
            rig_path=case.plan["build_plans"][0]["request_template"]["output"]["destination"]))))
    elif kind == "reported-work":
        (case.job_root / "logs/private-host-cpu.log").write_text(job_queue.utc() + ' {"kind":"completed"}\n')
    else:
        master = case.job_root / "ui/master.json"
        value = job_queue.read_json(master)
        value["machine"] = "other-host"
        master.write_text(__import__("json").dumps(value))
    before = _bytes(case.job_root)
    with pytest.raises(ValueError):
        _retry(case)
    assert _bytes(case.job_root) == before


def test_preexisting_shared_geometry_exports_are_preserved_and_not_owned_work(unstarted):
    case = unstarted
    export = Path(case.plan["project_root"]) / "exports/native_setup" / ("a" * 32)
    prior = dict(format="darsia-fluidflower-native-setup-export-v1", schema=1,
        revision_id="a" * 32, run_id="sample", frame="rig-corrected",
        reference_run="sample", current_sha256=case.plan["adopted_revision"]["sha256"],
        created_utc="2020-01-01T00:00:00+00:00", files={})
    _doc(export / "export.json", prior)
    _doc(export / "apply-complete.json", dict(facies_rebuild=dict(raw_setup_settings=dict(
        rig_path=str(case.root / "older-job" / "raw-models" / "sample")))))
    original = _bytes(export)
    result = _retry(case)
    assert result["root"] == str(case.job_root) and result["paused"]
    assert _bytes(export) == original
    assert not Path(case.plan["build_plans"][0]["request_template"]["output"]["destination"]).exists()
    # Reprepare does not certify existing exports. The unchanged original
    # producer still rejects an incompatible export before opening payloads.
    from darsia.fluidflower.corrected_setup import _verify_reusable_export
    with pytest.raises(ValueError, match="different identity"):
        _verify_reusable_export(prior, export, revision="different-revision", run="sample",
                               display="sample", planned={}, current_sha=prior["current_sha256"])
    assert _bytes(export) == original


def test_failed_prestart_empty_result_scaffolding_is_not_scientific_work(unstarted):
    case = unstarted
    # The worker may prepare this empty output root before failing code checks.
    output = case.job_root / "setup-apply"
    (output / ".attempts").mkdir(parents=True)
    result = _retry(case)
    assert result["root"] == str(case.job_root) and result["paused"]
    assert output.is_dir() and not list((output / ".attempts").iterdir())


def test_shared_export_with_explicit_own_job_identity_still_blocks_reprepare(unstarted):
    case = unstarted
    export = Path(case.plan["project_root"]) / "exports/native_setup" / ("a" * 32)
    _doc(export / "export.json", dict(owner=dict(job_id=case.job_root.name)))
    before = _bytes(case.job_root)
    with pytest.raises(ValueError, match="attributed to this job"):
        _retry(case)
    assert _bytes(case.job_root) == before


def test_registered_dependent_refuses_reprepare(unstarted):
    case = unstarted
    other = case.job_root.parent / "dependent"
    other.mkdir()
    job_queue.write_immutable(other / "job.json", dict(pipeline="fixture-consumer", expected_runs=["one"],
        precision="float32", original_setup=str(case.job_root / "setup-controls/plan.json")))
    case.registry.register_job(case.series, "synthetic", other)
    before = _bytes(case.job_root)
    with pytest.raises(ValueError, match="depends on this job"):
        _retry(case)
    assert _bytes(case.job_root) == before


def test_second_unstarted_update_keeps_one_logical_job_and_both_generations(unstarted):
    case = unstarted
    first = _retry(case)
    first_job = job_queue.descriptor(case.job_root / "job.json")
    case.identity = dict(case.identity, commit="5" * 40)
    second = _retry(case)
    assert first["root"] == second["root"]
    assert len(case.registry.jobs(case.series, "synthetic")) == 1
    assert len(list((case.job_root / "history").glob("*/receipt.json"))) == 2
    assert job_queue.read_json(Path(second["history"]) / "receipt.json")["previous_job"] == first_job
    assert case.folder.jobs()[0]["publication_generation"] == 2


def test_publication_revision_rejects_tampered_history(unstarted):
    case = unstarted
    result = _retry(case)
    old = Path(result["history"]) / "previous/job.json"
    old.write_bytes(old.read_bytes() + b" ")
    with pytest.raises(ValueError, match="transition"):
        case.folder.jobs()


def test_source_drift_refuses_without_replacing_metadata(unstarted):
    case = unstarted
    revision = Path(case.plan["adopted_revision"]["path"])
    revision.write_bytes(revision.read_bytes() + b" ")
    before = _bytes(case.job_root)
    with pytest.raises(ValueError, match="changed|SHA|bytes|size"):
        _retry(case)
    assert _bytes(case.job_root) == before


def test_reprepared_job_reaches_actual_spawned_worker_progress(unstarted, monkeypatch):
    from darsia.fluidflower.pipeline import worker
    from darsia.fluidflower.job_lifecycle import recover_job
    from .test_folder_cache_worker import _activate
    from ._folder_worker_fixture import work
    case = unstarted
    _retry(case)
    recover_job(case.job_root, machine="private-host",
        expected_previous=job_control.read_control(case.job_root)["command"],
        stores_path=case.stores.source["path"])
    _activate(case.stores, case.job_root, monkeypatch)
    monkeypatch.setattr(worker, "_work", work)
    monkeypatch.setenv("DARSIA_TEST_FOLDER_PAUSE_CURRENT", "1")
    job = job_queue.read_json(case.job_root / "job.json")
    args = SimpleNamespace(job=case.job_root, machine="private-host", backend="cpu", workers=1,
        once=True, exit_when_idle=True, watchdog=False, watchdog_seconds=30,
        _alias_stores=case.stores, stores=case.stores.source["path"], _cache_terminal_events=[])
    path = case.job_root / "logs/private-host-cpu.log"
    with path.open("a", encoding="utf-8") as log:
        code = worker._parent(args, "private-host-cpu", job, job["commit"], log)
    assert code == 0, path.read_text()
    assert '"kind": "started"' in path.read_text() and "tiny_fixture_compute" in path.read_text()
    assert not list((case.job_root / "failed").glob("*.json")), path.read_text()
    assert not list((case.job_root / "done").glob("*.json"))
    assert list((case.job_root / "leases").glob("*/progress_*.json"))


@pytest.fixture
def failed_before_apply(unstarted, monkeypatch, request):
    """Real child, ownership, failure publication and retired output; tiny fixture science."""
    from darsia.fluidflower.pipeline import worker
    from .test_folder_cache_worker import _activate
    from ._folder_worker_fixture import work
    case = unstarted
    prior = job_control.read_control(case.job_root)
    job_control.publish_control(case.job_root, "run", machine="private-host", expected_previous=prior.get("command"))
    begin_launch_request(case.job_root, machine="private-host", backend="cpu", workers=1, kind="local")
    _activate(case.stores, case.job_root, monkeypatch)
    monkeypatch.setattr(worker, "_work", work)
    monkeypatch.setenv("DARSIA_TEST_FOLDER_FAIL_BEFORE_APPLY", "1")
    monkeypatch.setenv("DARSIA_TEST_FOLDER_APPLY_REFERENCE", case.plan["display_reference_run"])
    monkeypatch.setenv("DARSIA_TEST_FOLDER_FAIL_CHECK_NOTIFICATIONS", "1" if getattr(request, "param", False) else "0")
    job = job_queue.read_json(case.job_root / "job.json")
    args = SimpleNamespace(job=case.job_root, machine="private-host", backend="cpu", workers=1,
        once=True, exit_when_idle=True, watchdog=False, watchdog_seconds=30,
        _alias_stores=case.stores, stores=case.stores.source["path"], _cache_terminal_events=[])
    with (case.job_root / "logs/private-host-cpu.log").open("a", encoding="utf-8") as log:
        worker._parent(args, "private-host-cpu", job, job["commit"], log)
    assert (case.job_root / "failed/setup-apply.json").is_file()
    # The pytest host acted as supervisor; model its later exit for the pure
    # recovery test. Child execution and publication above were not mocked.
    monkeypatch.setattr(worker, "_same_process", lambda _identity: None)
    return case


def test_retired_first_apply_failure_reprepares_same_job_and_preserves_all_bytes(failed_before_apply):
    case = failed_before_apply
    failure = (case.job_root / "failed/setup-apply.json").read_bytes()
    result = _retry(case)
    history = Path(result["history"])
    assert result["paused"] and len(case.registry.jobs(case.series, "synthetic")) == 1
    assert (history / "previous/failed/setup-apply.json").read_bytes() == failure
    receipt = job_queue.read_json(history / "receipt.json")
    archived = Path(receipt["failed_preparation"]["archived_output"])
    assert (archived / "failed.json").read_bytes() == failure
    assert not Path(receipt["failed_preparation"]["original_output"]).exists()
    assert (case.job_root / "queue/setup-apply.json").is_file()
    assert not list((case.job_root / "failed").iterdir())
    # A later source update still works; the old retired attempt is history.
    case.identity = dict(case.identity, commit="5" * 40)
    assert _retry(case)["root"] == result["root"]


def test_retired_initial_apply_can_retry_without_fabricating_a_new_code_version(failed_before_apply):
    case = failed_before_apply
    job = job_queue.read_json(case.job_root / "job.json")
    case.identity = {key: job[key] for key in ("commit", "environment_lock")}
    result = _retry(case)
    assert result["paused"]
    assert job_queue.read_json(case.job_root / "job.json") == job
    assert case.folder.jobs()[0]["state"] == "paused"


def _unapplied_reference_export(case):
    """Opaque outputs for operational preservation; no scientific acceptance."""
    from darsia.fluidflower.corrected_setup import EXPORT_FILES, FORMAT
    directory = case.job_root / "exports/native_setup" / ("a" * 32)
    directory.mkdir(parents=True)
    files = {}
    for filename in EXPORT_FILES:
        path = directory / filename
        path.write_bytes(b"opaque output retained for the original runtime verifier")
        files[filename] = job_queue.descriptor(path)
    _doc(directory / "export.json", dict(format=FORMAT, schema=1, frame="rig-corrected",
        revision_id="a" * 32, run_id=case.plan["display_reference_run"],
        reference_run=case.plan["display_reference_run"], current_sha256=case.plan["adopted_revision"]["sha256"],
        setup_operation=dict(format="darsia-job-corrected-export-scope-v1", job_id=case.job_root.name,
            project_root=case.plan["project_root"], setup_plan=job_queue.descriptor(case.plan_path)), files=files))
    return directory


def test_failed_first_apply_retains_unapplied_export_for_same_job_retry(failed_before_apply):
    case = failed_before_apply
    directory = _unapplied_reference_export(case)
    before = _bytes(directory)
    result = _retry(case)
    assert _bytes(directory) == before
    assert len(case.registry.jobs(case.series, "synthetic")) == 1
    receipt = job_queue.read_json(Path(result["history"]) / "receipt.json")
    assert set(receipt["failed_preparation"]["unapplied_export"]) == set(before)
    assert (case.job_root / "queue/setup-apply.json").is_file()


def _completed_apply_case(case, monkeypatch):
    """Recovery seam; the separate original-CAS tests own completion proof."""
    from darsia.fluidflower import setup_completed_apply
    from darsia.fluidflower.pipeline.series_setup_protocol import APPLY_COMPLETE_FORMAT
    directory = _unapplied_reference_export(case)
    export = job_queue.descriptor(directory / "export.json")
    run = case.plan["display_reference_run"]
    before = directory / "apply-private/00-before.toml"
    before.parent.mkdir()
    before.write_bytes(b"# retained original configuration\n")
    original = job_queue.descriptor(before)
    configuration = Path(case.plan["build_plans"][0]["configuration_paths"][0])
    configuration.parent.mkdir(parents=True, exist_ok=True)
    configuration.write_bytes(b"# completed current configuration\n")
    current = job_queue.descriptor(configuration)
    attempt = _doc(directory / "apply-private/receipt.json", dict(complete=True, run_id=run))
    claim = _doc(directory / "apply-claim.json", dict(format="darsia-corrected-setup-apply-claim-v1",
        attempt_directory=str(before.parent), export=export, run_id=run))
    applied = _doc(directory / "apply-complete.json", dict(format="darsia-corrected-setup-apply-complete-v1",
        complete=True, run_id=run, revision_id="a" * 32, export=export, attempt_receipt=attempt,
        configurations=[dict(before=original, after=current)]))
    rows = [dict(run_id=run, export=export, apply_completion=applied, configuration_files=[current])]
    coordinator = dict(format=APPLY_COMPLETE_FORMAT, complete=True,
        setup_plan=job_queue.descriptor(case.plan_path),
        **{key: deepcopy(case.plan[key]) for key in ("adopted_revision", "project_manifest", "series_manifest")},
        revision_id="a" * 32, current_sha256=case.plan["adopted_revision"]["sha256"], runs=rows,
        Build_run=False, optical_calibration_run=False)
    bound = _doc(case.job_root / "setup-controls/apply-complete.json", coordinator)
    pins = [bound, export, claim, applied, attempt, original, current]

    def completed(config, plan, revision):
        assert plan == case.plan and config["setup_plan"] == job_queue.descriptor(case.plan_path)
        for pin in pins:
            if job_queue.descriptor(Path(pin["path"])) != pin:
                raise ValueError("Original completed Apply metadata changed")
        return dict(coordinator=deepcopy(coordinator), bound=deepcopy(bound), runs=deepcopy(rows),
                    guarded_geometry=[], metadata=deepcopy(pins))

    monkeypatch.setattr(setup_completed_apply, "read_completed_apply", completed)
    request = dict(deepcopy(case.plan["build_plans"][0]["request_template"]), configuration_files=[current])
    request_path = case.job_root / "setup-controls/build" / run / "raw-build-request.json"
    _doc(request_path, request)
    return directory, request_path


def test_code_retry_preserves_complete_apply_and_valid_build_controls(failed_before_apply, monkeypatch):
    case = failed_before_apply
    directory, request = _completed_apply_case(case, monkeypatch)
    before_exports = _bytes(directory)
    before_controls = _bytes(case.job_root / "setup-controls")
    failed = (case.job_root / "failed/setup-apply.json").read_bytes()
    from darsia.fluidflower.corrected_setup import EXPORT_FILES
    read_bytes = Path.read_bytes
    payloads = {directory / name for name in EXPORT_FILES}

    def no_raster_replay(path):
        if path in payloads:
            pytest.fail("Code-only retry reread a completed Apply payload")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", no_raster_replay)
    result = _retry(case)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert _bytes(directory) == before_exports
    assert _bytes(case.job_root / "setup-controls") == before_controls
    receipt = job_queue.read_json(Path(result["history"]) / "receipt.json")
    completed = receipt["failed_preparation"]["completed_apply"]
    assert completed["format"] == "darsia-setup-failed-after-completed-apply-v1"
    assert completed["build_controls"][str(request)] == job_queue.descriptor(request)
    assert (Path(result["history"]) / "previous/failed/setup-apply.json").read_bytes() == failed
    assert (case.job_root / "queue/setup-apply.json").is_file()
    assert len(case.registry.jobs(case.series, "synthetic")) == 1


def _complete_apply_result(case):
    from darsia.fluidflower.setup_failed_start import completed_apply_snapshot, review_failed_setup
    root = case.job_root
    config = job_queue.read_json(root / "setup-controls/apply-operation.json")
    completion = job_queue.read_json(root / "setup-controls/apply-complete.json")
    children = []
    for build in case.plan["build_plans"]:
        run = build["run_id"]
        row = next(row for row in completion["runs"] if row["run_id"] == run)
        directory = root / "setup-controls/build" / run
        request = job_queue.descriptor(directory / "raw-build-request.json")
        operation = dict(format=config["format"], run=run, operation="build",
            **{key: deepcopy(config[key]) for key in ("project_root", "project_manifest", "adopted_revision", "series_manifest", "setup_plan")},
            **{key: deepcopy(build[key]) for key in ("native_parameters", "target_machine", "metadata_output", "output_scope")},
            configuration_files=deepcopy(row["configuration_files"]), raw_build_request=request,
            apply_completion=deepcopy(row["apply_completion"]),
            coordinator_completion=job_queue.descriptor(root / "setup-controls/apply-complete.json"))
        pin = _doc(directory / "operation.json", operation)
        children.append(dict(run_id=run, configuration=pin, raw_build_request=request,
                             target_machine=build["target_machine"]))
    job = job_queue.read_json(root / "job.json")
    review = review_failed_setup(root, job, case.plan, case.stores.source["path"])
    result = dict(format="darsia-series-setup-result-v1", complete=True, pipeline="series-setup",
        run="setup-apply", operation="apply", backend="cpu", precision=job["precision"],
        setup_plan=config["setup_plan"], worker_references_closed=True,
        coordinator_completion=completed_apply_snapshot(root, case.plan)["coordinator"],
        guarded_geometry=[], children=children, Build_run=False, optical_calibration_run=False)
    path = Path(review["original_output"]) / "operation_result.json"
    _doc(path, result)
    return path


def test_code_retry_archives_exact_apply_result_before_queue_publication(failed_before_apply, monkeypatch):
    case = failed_before_apply
    directory, _ = _completed_apply_case(case, monkeypatch)
    path = _complete_apply_result(case)
    before = path.read_bytes()
    exports = _bytes(directory)
    result = _retry(case)
    receipt = job_queue.read_json(Path(result["history"]) / "receipt.json")
    archived = Path(receipt["failed_preparation"]["archived_output"]) / path.name
    assert archived.read_bytes() == before
    assert _bytes(directory) == exports
    assert (case.job_root / "queue/setup-apply.json").is_file()


@pytest.mark.parametrize("fault", ["uncommitted", "child", "geometry", "boolean-type", "extra"])
def test_code_retry_rejects_unproven_apply_result(failed_before_apply, monkeypatch, fault):
    case = failed_before_apply
    _completed_apply_case(case, monkeypatch)
    path = _complete_apply_result(case)
    value = job_queue.read_json(path)
    if fault == "uncommitted":
        value["coordinator_completion"]["sha256"] = "f" * 64
    elif fault == "child":
        value["children"] = []
    elif fault == "geometry":
        value["guarded_geometry"] = [dict(unapproved=True)]
    elif fault == "boolean-type":
        value["complete"] = 1
    else:
        value["new_field"] = True
    _doc(path, value)
    before = (case.job_root / "job.json").read_bytes()
    with pytest.raises(ValueError, match="operation result differs"):
        _retry(case)
    assert (case.job_root / "job.json").read_bytes() == before
    assert path.is_file()


@pytest.mark.parametrize("fault", ["changed-request", "unknown-control", "queued-build", "raw-output", "metadata-drift", "unknown-export"])
def test_complete_apply_retry_does_not_forgive_other_work(failed_before_apply, monkeypatch, fault):
    case = failed_before_apply
    directory, request = _completed_apply_case(case, monkeypatch)
    if fault == "changed-request":
        value = job_queue.read_json(request)
        value["operation_overrides"] = dict(unapproved=True)
        _doc(request, value)
    elif fault == "unknown-control":
        request.with_name("checkpoint.json").write_bytes(b"{}")
    elif fault == "queued-build":
        (case.job_root / "queue/sample.json").write_bytes(b"{}")
    elif fault == "raw-output":
        Path(case.plan["build_plans"][0]["request_template"]["output"]["destination"]).mkdir(parents=True)
    elif fault == "metadata-drift":
        path = directory / "apply-complete.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        (directory / "unknown.npz").write_bytes(b"unapproved work")
    job = (case.job_root / "job.json").read_bytes()
    with pytest.raises(ValueError):
        _retry(case)
    assert (case.job_root / "job.json").read_bytes() == job
    assert (case.job_root / "failed/setup-apply.json").is_file()


@pytest.mark.parametrize("fault", ["apply-claim", "partial", "foreign-job", "changed-size"])
def test_unapplied_export_retry_refuses_unknown_or_applied_work(failed_before_apply, fault):
    case = failed_before_apply
    directory = _unapplied_reference_export(case)
    if fault == "apply-claim":
        (directory / "apply-claim.json").write_bytes(b"{}")
    elif fault == "partial":
        (directory / "native_labels.npz").unlink()
    elif fault == "changed-size":
        (directory / "native_labels.npz").write_bytes(b"changed length")
    else:
        path = directory / "export.json"
        receipt = job_queue.read_json(path)
        receipt["setup_operation"]["job_id"] = "another-job"
        _doc(path, receipt)
    previous_job = (case.job_root / "job.json").read_bytes()
    with pytest.raises(ValueError):
        _retry(case)
    assert (case.job_root / "job.json").read_bytes() == previous_job


@pytest.mark.parametrize("failed_before_apply", [True], indirect=True)
def test_failed_initial_source_notifications_are_preserved_and_retry_same_job(failed_before_apply):
    case = failed_before_apply
    lease = next((case.job_root / "leases").iterdir())
    progress = {path.name: path.read_bytes() for path in lease.glob("progress_*.json")}
    assert len(progress) == 2
    from darsia.fluidflower.setup_failed_start import review_failed_setup
    review = review_failed_setup(case.job_root, job_queue.read_json(case.job_root / "job.json"),
                                case.plan, case.stores.source["path"])
    local = {name: (Path(review["original_output"]) / name).read_bytes() for name in progress}
    result = _retry(case)
    history = Path(result["history"])
    receipt = job_queue.read_json(history / "receipt.json")
    archived = Path(receipt["failed_preparation"]["archived_output"])
    for name, raw in progress.items():
        assert (archived / name).read_bytes() == local[name]
        assert (history / "previous/leases" / lease.name / name).read_bytes() == raw
    assert case.folder.jobs()[0]["state"] == "paused"
    assert len(case.registry.jobs(case.series, "synthetic")) == 1


@pytest.mark.parametrize("fault", ["later-stage", "different-run", "skipped-notification", "extra-field"])
def test_initial_apply_notification_admission_refuses_later_or_untyped_work(tmp_path, fault):
    from darsia.fluidflower.setup_failed_start import _initial_apply_progress
    base = dict(state="setup_progress", operation="apply", step="apply", run="reference")
    _doc(tmp_path / "progress_00000000.json", dict(run="setup-apply", attempt_id="owned", sequence=0, event=base))
    nested = dict(base, message="Original source check", check_progress_sequence=1)
    if fault == "later-stage": nested["step"] = "capture_configuration_sources"
    elif fault == "different-run": nested["run"] = "other"
    elif fault == "skipped-notification": nested["check_progress_sequence"] = 2
    else: nested["completed"] = 1
    _doc(tmp_path / "progress_00000001.json", dict(run="setup-apply", attempt_id="owned", sequence=1, event=nested))
    with pytest.raises(ValueError, match="initial Apply stage"):
        _initial_apply_progress(tmp_path, "owned", "reference")


@pytest.mark.parametrize("fault", ["product", "checkpoint", "alive", "export"])
def test_failed_setup_recovery_refuses_actual_work_or_live_owner(failed_before_apply, monkeypatch, fault):
    case = failed_before_apply
    from darsia.fluidflower.setup_failed_start import review_failed_setup
    review = review_failed_setup(case.job_root, job_queue.read_json(case.job_root / "job.json"),
                                case.plan, case.stores.source["path"])
    if fault in {"product", "checkpoint"}:
        Path(review["original_output"]).joinpath(fault + ".json").write_bytes(b"{}")
    elif fault == "export":
        path = case.job_root / "exports/native_setup/receipt.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"{}")
    else:
        from darsia.fluidflower.pipeline import worker
        monkeypatch.setattr(worker, "_same_process", lambda _identity: object())
    old = (case.job_root / "job.json").read_bytes()
    with pytest.raises(ValueError):
        _retry(case)
    assert (case.job_root / "job.json").read_bytes() == old
    assert (case.job_root / "failed/setup-apply.json").is_file()


def test_clean_checkout_reprepare_passes_actual_startup_gate(unstarted):
    """Run the unchanged source/lock gate on private metadata in a clean checkout."""
    import subprocess
    from darsia.fluidflower.pipeline import worker
    repository = Path(worker.__file__).resolve().parents[4]
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip():
        pytest.skip("The actual startup gate intentionally requires a completely clean checkout")
    case = unstarted
    case.identity = recovery.current_setup_identity()
    with pytest.raises(RuntimeError, match="clean Git worktree at job.json commit"):
        worker.verify_startup(case.job_root, stores_path=case.stores.source["path"], machine_id="private-host")
    _retry(case)
    job, commit = worker.verify_startup(case.job_root, stores_path=case.stores.source["path"], machine_id="private-host")
    assert commit == case.identity["commit"] == job["commit"]


@pytest.mark.parametrize("failure", ["producer", "registration"])
def test_transaction_rolls_back_all_operational_and_publication_bindings(unstarted, monkeypatch, failure):
    case = unstarted
    before = _bytes(case.job_root)
    old_publication = case.folder.jobs()[0]
    old_registry = case.registry.context(case.series, "synthetic")
    def fail(*args, **kwargs):
        raise ValueError("synthetic transaction failure")
    if failure == "registration":
        monkeypatch.setattr(case.registry, "register_reprepared_setup", fail)
        choices = {}
    else:
        def partial(bound, root, **choices):
            series_setup_jobs.prepare_setup_job(bound, root, **choices)
            fail()
        choices = dict(producer=partial)
    with pytest.raises(ValueError, match="synthetic transaction failure"):
        _retry(case, **choices)
    assert _bytes(case.job_root) == before
    assert case.folder.jobs()[0] == old_publication
    assert case.registry.context(case.series, "synthetic") == old_registry
    failures = list((case.job_root / "history").glob("*/failure.json"))
    assert len(failures) == 1 and job_queue.read_json(failures[0])["restored_original"] is True
