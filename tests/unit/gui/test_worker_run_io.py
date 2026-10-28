"""Synthetic worker/parent ownership and callback wiring, no actual process."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import queue

import pytest
from darsia.fluidflower.pipeline import worker,worker_io
from darsia.fluidflower.pipeline.store_runtime import InputHashMismatch
from ..test_storage_n1_saved_products import tiny_config, phase1_data


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json.dumps(value).encode())


def test_fast_input_reject_claim_is_bound_from_its_actual_archive(tmp_path):
    task=dict(format="darsia-alias-task-v1",run="ac52",attempt_id="1"*32)
    owner=dict(machine="main-cpu",token=task["attempt_id"],task=task)
    put(tmp_path/"reclaimed/ac52"/task["attempt_id"]/"lease/owner.json",owner)
    assert worker_io.claimed_task_after_archive(tmp_path,"main-cpu",
        dict(run="ac52",attempt_id=task["attempt_id"]))==task
    with pytest.raises(ValueError,match="worker service"):
        worker_io.claimed_task_after_archive(tmp_path,"second-cpu",
            dict(run="ac52",attempt_id=task["attempt_id"]))


def test_conflicting_current_archive_task_data_is_never_accepted(tmp_path):
    task=dict(format="darsia-alias-task-v1",run="ac52",attempt_id="1"*32)
    put(tmp_path/"claimed/main-cpu/ac52.json",task)
    put(tmp_path/"reclaimed/ac52"/task["attempt_id"]/"task.json",dict(task,precision="wrong"))
    with pytest.raises(ValueError,match="exact immutable"):
        worker_io.claimed_task_after_archive(tmp_path,"main-cpu",
            dict(run="ac52",attempt_id=task["attempt_id"]))


def test_pre_science_hash_rejection_does_not_publish_fail_or_enter_executor(monkeypatch,tmp_path):
    job=dict(format="darsia-alias-job-v1",precision="float32",calibration_sha256="b"*64)
    task=dict(format="darsia-alias-task-v1",run="ac52",attempt_id="1"*32,
              commit="a"*40,calibration_sha256="b"*64,precision="float32")
    claim=SimpleNamespace(task=task,token=task["attempt_id"],output=tmp_path/"not-created")
    calls=[]
    class Shared:
        def __init__(self,*a,**k):pass
        def recover_publications(self):return []
        def reclaim_stale(self):return []
        def claim(self,*a,**k):
            assert callable(k["availability"]) and callable(k["output_factory"])
            assert callable(k["publication_factory"].publish_progress)
            return claim
        def reject_before_science(self,actual,text,owner):
            assert actual is claim and "literal fixture hash mismatch" in text
            calls.append("rejected");return {"run":"ac52"}
        def fail(self,*a):pytest.fail("Input mismatch must not become a scientific FAIL")
    class Runtime:
        def __init__(self,*a):self.publications=SimpleNamespace(publish_progress=lambda *a:None)
        def availability(self,*a):pass
        @contextmanager
        def capacity_claim(self,*a):
            # This original hash-rejection test still reaches no capacity or SCI.
            yield SimpleNamespace(availability=self.availability,bind=lambda claim:None)
        def output_factory(self,*a):pass
        def prepare(self,*a):raise InputHashMismatch("literal fixture hash mismatch")
        def capacity_close(self,*a):pass
        def release(self,*a):calls.append("released")
    monkeypatch.setattr(worker,"JobQueue",Shared)
    monkeypatch.setattr(worker_io,"WorkerRunIO",Runtime)
    monkeypatch.setattr(worker,"_identity",lambda machine:dict(machine=machine))
    # Native registration is an independent dependency of this mocked loop.
    from darsia_cache import adapters
    native_owner_calls = []
    def bind_owner(owner, *, slot=None, generation=None, role=None):
        assert owner == dict(machine="main-cpu")
        assert slot is None and generation is None and role == "slot"
        native_owner_calls.append(dict(owner))
        return dict(TEST_ONLY_NATIVE_OWNER_BOUNDARY=True)
    monkeypatch.setattr(adapters, "bind_actual_worker_owner", bind_owner)
    monkeypatch.setattr(worker,"_job_control_boundary",lambda root:None)
    messages=queue.Queue()
    with pytest.raises(SystemExit) as error:
        worker._run_tasks(str(tmp_path),job,"a"*40,"main-cpu","cpu",messages,
            SimpleNamespace(is_set=lambda:False),once=True,exit_when_idle=False,
            stores=SimpleNamespace(machine_id="Main"))
    assert error.value.code==1 and calls.count("rejected")==1
    assert native_owner_calls == [dict(machine="main-cpu")]
    assert any(event["kind"]=="input_rejected" for event in list(messages.queue))
    assert not claim.output.exists()


def test_publication_router_keeps_original_progress_event(monkeypatch):
    event=dict(state="frame_completed",case_id="ac52_f1",publication={"literal":"source"})
    seen=[]
    class Factory:
        def __call__(self,*args):seen.append(args);return {"candidate":"original"}
        def publish_progress(self,claim,value):
            seen.append((claim,deepcopy(value)));return {"relative_path":"original.json"}
    factory=Factory();runtime=SimpleNamespace(factory=lambda claim:factory)
    router=worker_io._PublicationRouter(runtime);claim=object()
    assert router.publish_progress(claim,event)=={"relative_path":"original.json"}
    assert seen==[(claim,event)] and event["publication"]=={"literal":"source"}


def _n16_alias_inputs(tmp_path, tmp_path_factory, tiny_config, phase1_data, monkeypatch, *, adjacent):
    """Real tiny producer DATA; only filesystem/OS qualification is a unit double."""
    from ..test_storage_n1_saved_products import _genuine_n16_config, _json, COMMIT
    from darsia.fluidflower.pipeline import staged_data, alias_job, series, job_queue
    from darsia.fluidflower import job_filesystem
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    config = _genuine_n16_config(tmp_path, tiny_config, phase1_data, adjacent=adjacent, bottom_strip=True)
    domain = dict(pixel_domain=deepcopy(config["pixel_domain"]), frames={spec["case_id"]:
        {key: deepcopy(spec[key]) for key in ("pixel_domain_geometry", "frozen_response_basis")}
        for spec in config["frames"]})
    sealed = staged_data.seal_phase2_configuration(tmp_path / "sealed_n16_phase2.json",
        config["phase1"]["completion"], config["phase1"]["configuration"],
        expected_phase1_commit=config["phase1"]["code_commit"], fitted_calibration=config["calibration"],
        fitted_anchor=config["anchor"], storage_data=config["storage"], pixel_domain_data=domain,
        reader=DataReader())["configuration"]
    assert json.loads(Path(sealed["path"]).read_bytes()) == config
    selected = _json(tmp_path / "selection_stage_abi.json", dict(complete=True,
        runs=[dict(run=config["run"], ordered_case_ids=[f["case_id"] for f in config["frames"]])]))
    manifest_path = tmp_path / "series.toml"
    manifest_path.write_bytes(b'schema_version=1\nseries_id="Synthetic"\n[roots]\nseries="."\n')
    from darsia.fluidflower.mass.n1_pipeline import asset
    manifest = asset(manifest_path)
    text = ('format="darsia-series-v1"\nadopted_for_execution=true\npipeline="n1"\njob_id="n16-stage-unit"\n'
        '[selection]\npath=' + json.dumps(selected["path"]) + '\nsha256="' + selected["sha256"] +
        '"\nsize_bytes=' + str(selected["size_bytes"]) + '\n[job]\ncommit="' + COMMIT +
        '"\nprecision="float32"\ncalibration_sha256="' + config["calibration"]["sha256"] +
        '"\n[job.environment_lock]\npath="requirements-worker.lock"\nsha256="' + 'c'*64 +
        '"\n[[runs]]\nrun=' + json.dumps(config["run"]) + '\n[runs.configuration]\npath=' +
        json.dumps(sealed["path"]) + '\nsha256="' + sealed["sha256"] + '"\nsize_bytes=' +
        str(sealed["size_bytes"]) + '\n[runs.resources]\nbackend="cpu"\nthreads=1\n')
    execution_path = tmp_path / "execution_stage_abi.toml"
    execution_path.write_text(text, encoding="utf-8")
    execution = asset(execution_path)
    results = tmp_path_factory.mktemp("n16_stage_results")
    shared = tmp_path_factory.mktemp("n16_stage_shared") / "job"
    # No real SMB, Windows token or capacity qualification is claimed by this test.
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path:
        job_filesystem.SharedFolderAdmission(True, "smb", "unit authored transport", "not a production qualification"))
    return SimpleNamespace(config=config, sealed=sealed, execution=execution, manifest=manifest,
        root=tmp_path, results=results, shared=shared, commit=COMMIT, prepare=alias_job.prepare_alias_job)


def _n16_alias_binding(data):
    from darsia.fluidflower.store_map import parse_store_map
    from darsia.fluidflower.job_filesystem import LocalFileAdmission
    from darsia.fluidflower.pipeline import job_queue
    from darsia.fluidflower.pipeline.store_runtime import resolve_catalog, verify_inputs
    from darsia.fluidflower.pipeline.run_io import RunIOBinding
    from darsia.fluidflower.pipeline.publication_runtime import make_publication_factory
    from ..test_phase1_worker_protocol import SyntheticServerClock
    data.prepare(data.execution["path"], data.shared, expected_sha256=data.execution["sha256"],
        series_manifest=data.manifest)
    stores = parse_store_map(('[machine]\nid="Main"\n[stores.Synthetic]\nseries=' +
        json.dumps(str(data.root)) + '\nresults=' + json.dumps(str(data.results)) + '\n').encode())
    queue = job_queue.JobQueue(data.shared, machine_id="Main", server_clock=SyntheticServerClock())
    admission = LocalFileAdmission(True, "unit-local", "unit provider", "not a filesystem qualification", str(data.results))
    def output(task, token):
        return job_queue.LocalOutputBinding("Main", data.results,
            data.results / task["job_id"] / task["run"] / ".attempts" / token, admission)
    job = job_queue.read_json(data.shared / "job.json")
    def no_prefix(*args):
        raise AssertionError("Fresh N16 unit claim requested a predecessor attempt")
    publisher = make_publication_factory(data.shared, job, "Main", producer_origin_for_attempt=no_prefix)
    claim = queue.claim("main-cpu", "cpu", dict(machine="main-cpu", pid=123,
        process_create_time="UNIT_NOT_OS_OWNERSHIP"),
        availability=lambda task: resolve_catalog(task, stores, "Synthetic"), output_factory=output,
        publication_factory=publisher)
    assert claim is not None and claim.task["pipeline"] == "n1" and claim.task["legacy_paths"] is False
    verified = verify_inputs(claim.task, claim.input_rows)
    return claim, verified, RunIOBinding


@pytest.mark.parametrize("adjacent", [False, True])
def test_actual_N16_sealer_alias_RunIO_writer_reader_without_native_phase_replay(
        tmp_path, tmp_path_factory, tiny_config, phase1_data, monkeypatch, adjacent):
    from darsia.fluidflower.pipeline import series
    from darsia.fluidflower.phase import infer
    from darsia.fluidflower.mass import n1_pipeline
    from darsia.fluidflower import storage_n1_saved_products
    from darsia.fluidflower import storage_extensive_io
    def forbidden(*args, **kwargs):
        raise AssertionError("N1 phase two replayed a native or phase producer")
    monkeypatch.setattr(series.SeriesInputs, "native", forbidden)
    monkeypatch.setattr(infer, "infer_labels", forbidden)
    data = _n16_alias_inputs(tmp_path, tmp_path_factory, tiny_config, phase1_data, monkeypatch, adjacent=adjacent)
    before = {p: p.read_bytes() for p in data.root.rglob("*") if p.is_file()}
    claim, verified, binding_type = _n16_alias_binding(data)
    binding = binding_type.replay(claim.task, verified, claim.local_output,
        repository=Path(n1_pipeline.__file__).resolve().parents[4], admitted_commit=data.commit)
    with binding.activate():
        result = n1_pipeline.run_n1(claim.task, claim.output, "cpu", "float32")
    actual = storage_n1_saved_products.read_n1_stored_run(result, resolve_asset=lambda b: Path(b["path"]))
    assert actual["producer_kind"] == "n1-mass-only" and len(actual["frames"]) == 2
    manifest = json.loads(Path(result["manifest"]["path"]).read_bytes())
    for record in result["records"]:
        extensive = storage_extensive_io.read_committed_payload(record, producer_kind="n1-mass-only",
            resolve_asset=lambda b: Path(b["path"]), manifest=manifest, manifest_asset=result["manifest"])
        assert extensive["state"] == "genuine_n15_n16_io"
        assert extensive["U_layer_present"] is adjacent
        assert extensive["physical_binding_verified"] and extensive["producer_identity_verified"]
        assert extensive["production_permission_granted"] is False
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("role", ["mass.pixel_domain.n16.authority", ".source_geometry",
                                  ".frozen_response_result", ".frozen_response_geometry",
                                  ".below_photo_strip", ".phase1.before_receipt"])
def test_N16_omitted_actual_role_stops_RunIO_before_mass_output(
        tmp_path, tmp_path_factory, tiny_config, phase1_data, monkeypatch, role):
    from darsia.fluidflower.mass import n1_pipeline
    from darsia.fluidflower.pipeline.store_runtime import VerifiedInputs
    data = _n16_alias_inputs(tmp_path, tmp_path_factory, tiny_config, phase1_data, monkeypatch, adjacent=False)
    claim, verified, binding_type = _n16_alias_binding(data)
    task = deepcopy(claim.task)
    removed = {use for row in task["input_catalog"]["inputs"] for use in row["uses"] if use.endswith(role)}
    assert removed
    for row in task["input_catalog"]["inputs"]:
        row["uses"] = [use for use in row["uses"] if use not in removed]
    task["input_catalog"]["inputs"] = [row for row in task["input_catalog"]["inputs"] if row["uses"]]
    rows = tuple(type(row)(row.reference, row.path, tuple(use for use in row.uses if use not in removed))
        for row in verified.rows if any(use not in removed for use in row.uses))
    omitted = VerifiedInputs(verified.run, rows, verified.versions, verified.digests)
    with pytest.raises((InputHashMismatch, ValueError)):
        binding_type.replay(task, omitted, claim.local_output,
            repository=Path(n1_pipeline.__file__).resolve().parents[4], admitted_commit=data.commit)
    assert not list(claim.output.iterdir()), "Only the normal empty claim root may exist; mass never started"


def test_N16_new_unlisted_authority_root_refuses_before_job_creation(
        tmp_path, tmp_path_factory, tiny_config, phase1_data, monkeypatch):
    from ..test_storage_n1_saved_products import _json
    from darsia.fluidflower.store_map import StoreMapError
    data = _n16_alias_inputs(tmp_path, tmp_path_factory, tiny_config, phase1_data, monkeypatch, adjacent=False)
    config = deepcopy(data.config)
    old = config["pixel_domain"]["n16"]["authority"]
    foreign = tmp_path_factory.mktemp("n16_unlisted") / "authority.json"
    foreign.write_bytes(Path(old["path"]).read_bytes())
    from darsia.fluidflower.mass.n1_pipeline import asset
    config["pixel_domain"]["n16"]["authority"] = asset(foreign)
    changed = _json(tmp_path / "unlisted_authority_configuration.json", config)
    text = Path(data.execution["path"]).read_text().replace(data.sealed["path"].replace("\\", "\\\\"),
        changed["path"].replace("\\", "\\\\")).replace(data.sealed["sha256"], changed["sha256"]).replace(
        'size_bytes=' + str(data.sealed["size_bytes"]), 'size_bytes=' + str(changed["size_bytes"]))
    execution = tmp_path / "unlisted_execution.toml"
    execution.write_text(text, encoding="utf-8")
    with pytest.raises(StoreMapError, match="No declared root"):
        data.prepare(execution, data.shared, expected_sha256=asset(execution)["sha256"], series_manifest=data.manifest)
    assert not data.shared.exists()


@pytest.mark.parametrize("fault", ["different_calibration_identity", "different_anchor", "omitted_case", "authority_without_decider"])
def test_N16_sealer_never_invents_a_refit_basis_or_repairs_missing_inputs(
        tmp_path, tiny_config, phase1_data, monkeypatch, fault):
    from ..test_storage_n1_saved_products import _genuine_n16_config, _json
    from darsia.fluidflower.pipeline import staged_data
    from darsia.fluidflower.mass import n1_pipeline
    config = _genuine_n16_config(tmp_path, tiny_config, phase1_data, adjacent=False)
    domain = dict(pixel_domain=deepcopy(config["pixel_domain"]), frames={spec["case_id"]:
        {key: deepcopy(spec[key]) for key in ("pixel_domain_geometry", "frozen_response_basis")}
        for spec in config["frames"]})
    calibration, anchor = config["calibration"], deepcopy(config["anchor"])
    if fault == "different_calibration_identity":
        value = json.loads(Path(calibration["path"]).read_bytes())
        value["unit_attempted_refit_identity"] = True
        calibration = _json(tmp_path / "different_fitted_calibration.json", value)
    elif fault == "different_anchor":
        anchor["elapsed_h"] += 1
    elif fault == "omitted_case":
        domain["frames"].pop(config["frames"][-1]["case_id"])
    else:
        authority = json.loads(Path(domain["pixel_domain"]["n16"]["authority"]["path"]).read_bytes())
        authority["decided_by"] = " "  # A decision record must name who decided.
        domain["pixel_domain"]["n16"]["authority"] = _json(tmp_path / "wrong_authority.json", authority)
    def forbidden(*args, **kwargs):
        raise AssertionError("N16 DATA sealer evaluated fitted mass")
    monkeypatch.setattr(n1_pipeline.N1Pipeline, "evaluate", forbidden)
    target = tmp_path / "must_not_seal.json"
    with pytest.raises(ValueError):
        staged_data.seal_phase2_configuration(target, config["phase1"]["completion"],
            config["phase1"]["configuration"], expected_phase1_commit=config["phase1"]["code_commit"],
            fitted_calibration=calibration, fitted_anchor=anchor, storage_data=config["storage"],
            pixel_domain_data=domain, reader=n1_pipeline.DataReader())
    assert not target.exists()


def test_N1_series_stage_is_explicit_and_cannot_relabel_a_native_phase_job(
        tmp_path, tmp_path_factory, tiny_config, phase1_data, monkeypatch):
    from darsia.fluidflower.mass.n1_pipeline import asset
    data = _n16_alias_inputs(tmp_path, tmp_path_factory, tiny_config, phase1_data, monkeypatch, adjacent=False)
    text = Path(data.execution["path"]).read_text().replace('pipeline="n1"', 'pipeline="native-phase-n1"', 1)
    wrong = tmp_path / "wrong_execution_stage.toml"
    wrong.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="native/phase/N1"):
        data.prepare(wrong, data.shared, expected_sha256=asset(wrong)["sha256"], series_manifest=data.manifest)
    assert not data.shared.exists()
