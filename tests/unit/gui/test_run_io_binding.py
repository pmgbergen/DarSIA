"""Synthetic operational I/O only; never invoke a scientific executor."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from darsia.fluidflower.job_filesystem import LocalFileAdmission
from darsia.fluidflower.store_map import parse_store_map
from darsia.fluidflower.pipeline.job_queue import LocalOutputBinding
from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from darsia.fluidflower.pipeline.store_runtime import resolve_catalog,verify_inputs,InputHashMismatch,RunUnavailable
from darsia.fluidflower.pipeline.run_io import RunIOBinding,current_binding,alias_resume_source
from darsia.fluidflower.mass.n1_pipeline import DataReader


def put(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))


def doc(path,value):
    return put(path,json.dumps(value,sort_keys=True).encode())


def fixture(tmp_path, *, declared_alias=None):
    canonical=tmp_path/"old-location";canonical.mkdir()
    def payload(key):return put(canonical/(key+".npy"),b"opaque authored input; never decoded")
    def metadata(key):return doc(canonical/(key+".json"),{})
    full=dict(native_centre=dict(log_linear_rgb=payload("log"),valid_mask=payload("valid")),ring_windows=[])
    frame=dict(case_id="ac52_f1",current=deepcopy(full),baseline=deepcopy(full),
               **{key:payload(key) for key in ("photo_top","labels","classes","area")},
               original_domain=dict(parent_labels_map=put(canonical/"original_domain.png",b"opaque source labels"),
                                    native_shape_hw=[2,2],frame_index=0))
    config=dict(format="darsia-n1-run-v1",run="ac52",frames=[frame],
        calibration=metadata("calibration"),physical_correction=metadata("correction"),boxes={},
        shared=dict(physical_reference=payload("physical"),floor_curve=payload("floor"),
                    depth_points=[],reference_matrix=[],clearwater=dict(rgb=payload("clear"),valid=payload("clear_valid")),
                    optics=dict(metadata=metadata("optics"),color_paths={},signals={},flash=metadata("flash"))))
    if declared_alias is not None:
        config["data_aliases"] = [dict(
            logical=dict(config["calibration"], path=declared_alias),
            physical=dict(config["calibration"]))]
    config_bound=doc(canonical/"run.json",config)
    manifest=put(canonical/"series.toml",b'schema_version=1\nseries_id="Synthetic"\n[roots]\nseries="."\n')
    text='[[runs]]\nrun="ac52"\n[runs.configuration]\npath='+json.dumps(config_bound["path"])+'\nsha256="'+config_bound["sha256"]+'"\nsize_bytes='+str(config_bound["size_bytes"])+'\n'
    execution=put(canonical/"execution.toml",text.encode())
    catalogue=InputCatalog({"series":str(canonical)}, config.get("data_aliases", ()))
    series_ref=catalogue.add(execution,"configuration","run.execution_series")
    manifest_ref=catalogue.add(manifest,"configuration","run.series_manifest")
    inputs=catalogue.collect(config_bound,config);inputs["runtime_support"]="darsia-role-io-runtime-v1"
    task=dict(format="darsia-alias-task-v1",format_version=1,legacy_paths=False,execution_ready=True,
        run="ac52",job_id="unit",configuration=inputs["configuration"],series=series_ref,
        series_manifest=manifest_ref,input_catalog=inputs,attempt_id="1"*32,commit="a"*40,
        calibration_sha256=config["calibration"]["sha256"],backend="cpu",precision="float32",
        output_directory="results/ac52",ordered_case_ids=["ac52_f1"],threads=1)
    host=tmp_path/"host-source";shutil.copytree(canonical,host)
    results=tmp_path/"local-results";results.mkdir()
    stores=parse_store_map(('[machine]\nid="Main"\n[stores.Synthetic]\nseries='+
                            json.dumps(str(host))+'\nresults='+json.dumps(str(results))+'\n').encode())
    rows=resolve_catalog(task,stores,"Synthetic");verified=verify_inputs(task,rows)
    output=results/"unit/ac52/.attempts"/task["attempt_id"]
    admission=LocalFileAdmission(True,"fixture-local","authored provider","not a hardware qualification",str(results))
    local=LocalOutputBinding("Main",results,output,admission)
    binding=RunIOBinding.replay(task,verified,local,repository=tmp_path/"repo",admitted_commit=task["commit"])
    return task,config,config_bound,host,binding


def test_context_reads_exact_original_bytes_at_new_host_without_rewriting_config(tmp_path):
    task,config,bound,host,binding=fixture(tmp_path)
    before=deepcopy(config);raw=Path(bound["path"]).read_bytes()
    with pytest.raises(ValueError,match="UNC"):
        DataReader(unc_only=True).path(config["calibration"])
    with binding.activate():
        reader=DataReader(unc_only=True)
        assert reader.path(config["calibration"])==host/"calibration.json"
        assert reader.json(config["calibration"])=={}
        assert current_binding() is binding
    assert current_binding() is None and config==before
    assert Path(bound["path"]).read_bytes()==raw
    assert binding.configuration==config
    assert task["configuration"]["sha256"]==bound["sha256"]


def test_unlisted_existing_old_input_never_falls_through(tmp_path):
    _task,_config,bound,_host,binding=fixture(tmp_path)
    other=put(Path(bound["path"]).parent/"not-catalogued.npy",b"exists but not admitted")
    with binding.activate():
        with pytest.raises(InputHashMismatch,match="Unlisted"):
            DataReader(unc_only=True).path(other)
        with pytest.raises(InputHashMismatch,match="outside"):
            binding.asset_path(other["path"])


def test_verified_input_changed_after_admission_fails_before_consumption(tmp_path):
    _task,config,_bound,host,binding=fixture(tmp_path)
    (host/"calibration.json").write_bytes(b'{"different":true}')
    with binding.activate(),pytest.raises(InputHashMismatch,match="changed"):
        DataReader().json(config["calibration"])


def test_own_output_allowed_but_foreign_attempt_not_an_implicit_input(tmp_path):
    _task,_config,_bound,_host,binding=fixture(tmp_path)
    output=put(binding.output/"preparation"/"produced.npy",b"original stage bytes")
    foreign=put(binding.output.parent/("2"*32)/"foreign.npy",b"not admitted")
    with binding.activate():
        assert DataReader(unc_only=True).path(output)==Path(output["path"])
        with pytest.raises(InputHashMismatch):
            DataReader(unc_only=True).path(foreign)
    with pytest.raises(RunUnavailable):
        binding.require_output(binding.output.parent/"different")


def test_catalogue_replay_rejects_same_sha_under_wrong_consumer(tmp_path):
    task,_config,_bound,_host,binding=fixture(tmp_path)
    task=deepcopy(task)
    task["input_catalog"]["inputs"][0]["uses"].append("unexpected.consumer")
    with pytest.raises(InputHashMismatch):
        RunIOBinding.replay(task,binding.verified,binding.local_output,
            repository=binding.repository,admitted_commit=task["commit"])


def test_actual_native_code_is_tracked_and_exact_sha_not_arbitrary_repo_data(tmp_path,monkeypatch):
    _task,_config,_bound,_host,binding=fixture(tmp_path)
    relative="src/darsia/fluidflower/native/source.py"
    bound=put(binding.repository/relative,b"# authored source bytes\n")
    calls=[]
    def tracked(args,**kwargs):
        calls.append(args)
        if args[3] == "show":
            assert args[4] == binding.commit + ":" + relative
            return Path(bound["path"]).read_bytes()
        assert args[3] == "ls-tree" and args[-1] == relative
        return args[-1]+"\n"
    monkeypatch.setattr("darsia.fluidflower.pipeline.run_io.subprocess.check_output",tracked)
    binding._admit_code(relative,bound["sha256"],"processor_module_sha256")
    assert calls[0][5]==binding.commit
    assert binding.asset_path(bound["path"])==Path(bound["path"])
    stranger=put(binding.repository/"unlisted.py",b"# not a declared code role")
    with pytest.raises(InputHashMismatch):
        binding.asset_path(stranger["path"])
    with pytest.raises(InputHashMismatch,match="exact execution binding"):
        binding._admit_code(relative,"0"*64,"processor_module_sha256")


def test_RAW_source_roles_use_real_Git_identities_and_recheck_Windows_EOL_bytes(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import run_io
    from darsia.fluidflower.native.io import sha256_file
    _task, _config, _bound, _host, binding = fixture(tmp_path)
    blobs, expected, paths = {}, {}, {}
    for role, relative in {**run_io.RAW_CODE_ROLES, **run_io._PROVENANCE_CODE_ROLES}.items():
        blob = ("# authored " + relative + "\n").encode()
        blobs[relative] = blob
        paths[role] = Path(put(binding.repository / relative, blob.replace(b"\n", b"\r\n"))["path"])
        if role in run_io.RAW_CODE_ROLES:
            expected[role] = hashlib.sha256(blob).hexdigest()
    def git(args, **kwargs):
        if "ls-tree" in args:
            assert args[-1] in blobs
            return args[-1] + "\n"
        assert args[3] == "show"
        commit, _, relative = args[4].partition(":")
        assert commit == binding.commit
        return blobs[relative]
    monkeypatch.setattr(run_io.subprocess, "check_output", git)
    native = dict(source=dict(raw_execution_identity=expected))
    binding._admit_native_code(native)
    with binding.activate():
        for role, path in paths.items():
            if role in expected:
                assert sha256_file(path) == expected[role]
        paths["raw_source"].write_bytes(b"changed executable source")
        with pytest.raises(InputHashMismatch, match="code changed"):
            sha256_file(paths["raw_source"])


def test_RAW_native_execution_roles_must_be_complete_before_consumption(tmp_path):
    from darsia.fluidflower.pipeline import run_io
    _task, _config, _bound, _host, binding = fixture(tmp_path)
    with pytest.raises(InputHashMismatch, match="RAW-only execution identity is incomplete"):
        binding._admit_native_code(dict(source=dict(raw_execution_identity={"raw_source": "a" * 64})))


def test_setup_RAW_code_route_cannot_replace_an_analysis_execution_binding(tmp_path):
    _task, _config, _bound, _host, binding = fixture(tmp_path)
    with pytest.raises(InputHashMismatch, match="original Setup task"):
        binding._admit_native_code(dict(source={}), setup_raw=True)


def queued_fixture(tmp_path):
    task,_config,_bound,_host,binding=fixture(tmp_path)
    root=tmp_path/"shared";run=root/"results/ac52"
    doc(run/"task.json",task)
    claim=SimpleNamespace(task=task,token=task["attempt_id"],output=binding.output,
                          local_output=binding.local_output)
    return root,claim,binding


def test_retry_waits_for_immutable_lineage_instead_of_replaying_fresh(tmp_path):
    root,claim,_binding=queued_fixture(tmp_path)
    retry=deepcopy(claim.task);retry["attempt_id"]="2"*32
    pending=SimpleNamespace(task=retry,token=retry["attempt_id"],output=claim.output.parent/retry["attempt_id"],
                            local_output=claim.local_output)
    with pytest.raises(RunUnavailable,match="visible before"):
        alias_resume_source(SimpleNamespace(root=root),pending)


def test_input_reject_chain_preserves_original_retired_scientific_attempt(tmp_path):
    root,initial,binding=queued_fixture(tmp_path)
    a,b,c="1"*32,"2"*32,"3"*32
    original=initial.task
    old_output=binding.output
    proof=dict(complete=True,all_absent=True,worker_os_references_closed=True,run="ac52",
        attempt_id=a,attempt_directory=str(old_output),machine_id="Main",
        supervisor=dict(machine="main-cpu"),identities=[dict(pid=123,process_create_time="1")])
    retirement=doc(old_output/"retirement.json",proof)
    owner=dict(format="darsia-alias-lease-owner-v1",machine="main-cpu",
        actual_owner=dict(machine="main-cpu"),token=a,task=original,
        output=dict(machine_id="Main",root_alias="results",relative_path="unit/ac52/.attempts/"+a))
    doc(root/"reclaimed/ac52"/a/"lease/owner.json",owner)
    prior=doc(root/"reclaimed/ac52"/a/"reclaimed.json",
        dict(run="ac52",previous_attempt=a,next_attempt=b))
    before=deepcopy(original);before["attempt_id"]=b
    doc(root/"reclaimed/ac52"/b/"task.json",before)
    ref=dict(relative_path="reclaimed/ac52/"+a+"/reclaimed.json",sha256=prior["sha256"],bytes=prior["size_bytes"])
    doc(root/"reclaimed/ac52"/b/"reclaimed.json",
        dict(run="ac52",previous_attempt=b,next_attempt=c,interrupted_claim_before_science=True,
             preserve_prior_resume_lineage=True,prior_resume_lineage=ref))
    retry=deepcopy(original);retry["attempt_id"]=c
    current=SimpleNamespace(task=retry,token=c,output=old_output.parent/c,local_output=binding.local_output)
    source=alias_resume_source(SimpleNamespace(root=root),current)
    assert source==dict(attempt_directory=str(old_output),retirement=retirement)
    # A scientifically identical task from the wrong generation is not proof
    # that this pre-SCI lineage hop ever owned the archived token.
    wrong=deepcopy(before);wrong["attempt_id"]="4"*32
    doc(root/"reclaimed/ac52"/b/"task.json",wrong)
    with pytest.raises(InputHashMismatch,match="Archived task generation"):
        alias_resume_source(SimpleNamespace(root=root),current)


def test_foreign_origin_refused_before_reading_foreign_local_retirement(tmp_path,monkeypatch):
    from darsia.fluidflower.pipeline.run_io import alias_resume_output
    root,claim,binding=queued_fixture(tmp_path)
    owner=dict(format="darsia-alias-lease-owner-v1",machine="second-cpu",actual_owner=dict(machine="second-cpu"),
               output=dict(machine_id="Second",root_alias="results",relative_path="unit/ac52/.attempts/"+"2"*32))
    with pytest.raises(RunUnavailable,match="another machine"):
        alias_resume_output(owner,binding.local_output,claim.task,"2"*32)


def test_reader_reuses_positive_input_hash_but_not_arbitrary_output(tmp_path,monkeypatch):
    _task,config,_bound,host,binding=fixture(tmp_path)
    with binding.activate():
        monkeypatch.setattr("darsia.fluidflower.mass.n1_pipeline.asset",
                            lambda path:pytest.fail("Positive admitted input must not be rehashed"))
        assert DataReader(unc_only=True).path(config["calibration"])==host/"calibration.json"


def candidate_prefix_fixture(tmp_path):
    """Literal metadata/opaque byte products for original prefix guards only."""
    from darsia.fluidflower.pipeline import checkpoint
    task,_config,_bound,_host,binding=fixture(tmp_path)
    old=binding.output.parent/("2"*32)
    case=task["ordered_case_ids"][0]
    spec=dict(case_id=case,frame_index=1)
    config=dict(format="darsia-native-phase-n1-run-v1",frames=[spec])
    calibration=task["calibration_sha256"]
    code=hashlib.sha256(task["commit"].encode("ascii")).hexdigest()
    settings=dict(backend="cpu",precision="float32")
    manifest=doc(old/"manifest.json",dict(format="darsia-series-run-manifest-v1",
        task_identity=checkpoint.run_identity(task,"cpu","float32",calibration),
        configuration=task["configuration"],code=dict(commit=task["commit"]),
        code_sha256=code,settings=settings,frames={case:spec},job_id=task["job_id"]))
    final=old/"frames"/case
    payloads={}
    for leaf in ["products.npz","native_geometry.npz","native_statistics.json.gz"]:
        bound=put(final/leaf,b"opaque fixture bytes, never decoded")
        payloads[leaf]={key:bound[key] for key in ["sha256","size_bytes"]}
    result=doc(final/"result.json",dict(complete=True,run=task["run"],case_id=case,
        frame_index=1,chronological_position=0,backend="cpu",precision="float32",
        working_pixel_dtype="float32",bindings=dict(calibration_sha256=calibration,code_sha256=code)))
    payloads["result.json"]={key:result[key] for key in ["sha256","size_bytes"]}
    commit_id="c"*32
    commit=doc(final/"commit.json",dict(complete=True,frame_id=case,
        manifest_sha256=manifest["sha256"],job_id=task["job_id"],settings=settings,
        products=settings,commit_id=commit_id,files=payloads))
    registration=doc(old/"ops"/("frame_commit_"+commit_id+".json"),
        dict(manifest_sha256=manifest["sha256"],frame_id=case,commit=commit,files=payloads))
    inventory=dict(payloads)
    inventory["commit.json"]={key:commit[key] for key in ["sha256","size_bytes"]}
    publication=doc(old/"ops"/("nas_copy_"+commit_id+".json"),
        dict(frame_id=case,nas_directory=str(final),manifest_sha256=manifest["sha256"],
             files=inventory,direct_NAS_write=False,direct_local_write=True,local_retained=True))
    retirement=doc(old/"retirement.json",dict(complete=True,all_absent=True,
        worker_os_references_closed=True,run=task["run"],attempt_id=old.name,
        attempt_directory=str(old),machine_id="Main",identities=[dict(pid=123,process_create_time="1")],
        manifest=manifest,frame_receipts=[dict(frame_id=case,registration=registration,publication=publication)]))
    prepared=put(old/"preparation"/"original_input.npy",b"original producer literal bytes")
    return task,config,binding,dict(attempt_directory=str(old),retirement=retirement),prepared


def test_original_native_prefix_validator_can_read_only_its_temporary_candidate(tmp_path):
    from darsia.fluidflower.pipeline import checkpoint
    task,config,binding,source,prepared=candidate_prefix_fixture(tmp_path)
    visits=[]
    def original_artifact_validator(spec,result,final,unchanged_config):
        assert binding.resume_roots==set()
        assert unchanged_config is config
        assert DataReader(unc_only=True).path(prepared)==Path(prepared["path"])
        visits.append(spec["case_id"])
    with binding.activate():
        with pytest.raises(InputHashMismatch,match="Unlisted"):
            DataReader(unc_only=True).path(prepared)
        prefix=checkpoint.verify_frame_prefix(task,config,binding.output,"cpu","float32",source,
            frame_validator=original_artifact_validator,calibration_sha256=task["calibration_sha256"])
        assert prefix["prefix_case_ids"]==task["ordered_case_ids"] and visits==task["ordered_case_ids"]
        assert binding._prefix_validation_root is None and binding.resume_roots==set()
        with pytest.raises(InputHashMismatch,match="Unlisted"):
            DataReader(unc_only=True).path(prepared)
        # Permanent access is a separate step, only after full original verify.
        binding.admit_resume_root(prefix["source"])
        assert DataReader(unc_only=True).path(prepared)==Path(prepared["path"])


def test_prefix_validator_exception_removes_temporary_view_and_keeps_SHA_checks(tmp_path):
    from darsia.fluidflower.pipeline import checkpoint
    task,config,binding,source,prepared=candidate_prefix_fixture(tmp_path)
    bad=dict(prepared,sha256="0"*64)
    def rejected_original_validator(*_args):
        DataReader(unc_only=True).path(bad)
    with binding.activate(),pytest.raises(ValueError,match="SHA"):
        checkpoint.verify_frame_prefix(task,config,binding.output,"cpu","float32",source,
            frame_validator=rejected_original_validator,calibration_sha256=task["calibration_sha256"])
    assert binding._prefix_validation_root is None and binding.resume_roots==set()
    # A physically foreign retirement never opens even the temporary path view.
    proof=json.loads(Path(source["retirement"]["path"]).read_bytes())
    proof["machine_id"]="Second"
    foreign=dict(source,retirement=doc(Path(source["retirement"]["path"]),proof))
    with binding.activate(),pytest.raises(RunUnavailable,match="same-machine"):
        with binding.prefix_validation(foreign):
            pytest.fail("No callback may run for an unadmitted machine")



def test_native_payload_hash_is_once_per_run_and_each_reuse_checks_version(tmp_path,monkeypatch):
    from darsia.fluidflower.native.io import sha256_file
    import os
    payload=tmp_path/"host-source"/"log.npy"
    original_open=Path.open
    reads=[]
    def observed_open(path,mode="r",*args,**kwargs):
        if Path(path)==payload and mode=="rb":
            reads.append(str(path))
        return original_open(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,"open",observed_open)
    _task,config,_bound,_host,binding=fixture(tmp_path)
    expected=config["frames"][0]["current"]["native_centre"]["log_linear_rgb"]["sha256"]
    assert len(reads)==1  # the real verify_inputs streaming admission
    with binding.activate():
        for _ in range(4):
            assert sha256_file(payload)==expected
        assert len(reads)==1
        version=payload.stat()
        os.utime(payload,ns=(version.st_atime_ns,version.st_mtime_ns+10000000))
        with pytest.raises(InputHashMismatch,match="changed"):
            sha256_file(payload)
        assert len(reads)==1


def test_native_and_replay_metadata_reuse_hash_but_preserve_expected_sha(tmp_path,monkeypatch):
    from darsia.fluidflower.native.io import read_bound_json
    from darsia.fluidflower.pipeline.run_io import _Replay
    _task,config,_bound,host,binding=fixture(tmp_path)
    path=host/"calibration.json"
    # Admission has actually completed. None of these known-input readers
    # may stream/hash that metadata again; their decode/schema checks remain.
    monkeypatch.setattr("darsia.fluidflower.pipeline.run_io.hashlib.sha256",
                        lambda *args,**kwargs:pytest.fail("Known admitted metadata was rehashed"))
    with binding.activate():
        assert read_bound_json(path,config["calibration"]["sha256"])=={}
        assert read_bound_json(path,config["calibration"]["sha256"])=={}
        replay=_Replay(binding,())
        assert replay.doc(config["calibration"],"calibration","mass.calibration")=={}
        with pytest.raises(ValueError,match="Source artifact changed"):
            read_bound_json(path,"0"*64)
        path.write_bytes(b'{"changed":true}')
        with pytest.raises(InputHashMismatch,match="changed"):
            read_bound_json(path,config["calibration"]["sha256"])
        with pytest.raises(InputHashMismatch,match="changed"):
            replay.doc(config["calibration"],"calibration","mass.calibration")


def test_native_unknown_path_is_refused_without_historical_read(tmp_path,monkeypatch):
    from darsia.fluidflower.native.io import sha256_file,read_bound_json
    _task,_config,bound,_host,binding=fixture(tmp_path)
    unknown=put(Path(bound["path"]).parent/"unlisted.json",b'{}')
    original_open=Path.open
    def no_unknown(path,*args,**kwargs):
        assert Path(path)!=Path(unknown["path"]),"Unknown historical path was read"
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",no_unknown)
    with binding.activate():
        with pytest.raises(InputHashMismatch,match="outside"):
            sha256_file(unknown["path"])
        with pytest.raises(InputHashMismatch,match="outside"):
            read_bound_json(unknown["path"],unknown["sha256"])


def test_native_code_and_own_outputs_keep_original_fresh_hashing(tmp_path,monkeypatch):
    from darsia.fluidflower.native.io import sha256_file,read_bound_json
    _task,_config,_bound,_host,binding=fixture(tmp_path)
    relative="src/darsia/fluidflower/native/source.py"
    code=put(binding.repository/relative,b"# original source bytes\n")
    def original_git(args, **kwargs):
        if args[3] == "show":
            assert args[4] == binding.commit + ":" + relative
            return Path(code["path"]).read_bytes()
        assert args[3] == "ls-tree" and args[-1] == relative
        return args[-1]+"\n"
    monkeypatch.setattr("darsia.fluidflower.pipeline.run_io.subprocess.check_output", original_git)
    binding._admit_code(relative,code["sha256"],"processor_module_sha256")
    output=doc(binding.output/"own.json",{})
    original_open=Path.open
    reads=[]
    def observed_open(path,mode="r",*args,**kwargs):
        if mode=="rb" and Path(path) in (Path(code["path"]),Path(output["path"])):
            reads.append(str(path))
        return original_open(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,"open",observed_open)
    with binding.activate():
        assert sha256_file(code["path"])==code["sha256"]
        assert sha256_file(code["path"])==code["sha256"]
        assert read_bound_json(output["path"],output["sha256"])=={}
        assert read_bound_json(output["path"],output["sha256"])=={}
        assert reads.count(code["path"])==2 and reads.count(output["path"])==2
        Path(output["path"]).write_bytes(b'{"changed":true}')
        with pytest.raises(ValueError,match="Source artifact changed"):
            read_bound_json(output["path"],output["sha256"])
        Path(code["path"]).write_bytes(b"# changed source bytes\n")
        with pytest.raises(InputHashMismatch,match="current code changed"):
            sha256_file(code["path"])
        with pytest.raises(InputHashMismatch,match="current code changed"):
            binding.asset_path(code["path"])


def test_native_unbound_legacy_hash_remains_fresh(tmp_path,monkeypatch):
    from darsia.fluidflower.native.io import sha256_file,read_bound_json
    bound=doc(tmp_path/"legacy.json",{})
    original_open=Path.open
    reads=[]
    def observed_open(path,mode="r",*args,**kwargs):
        if Path(path)==Path(bound["path"]) and mode=="rb":reads.append(str(path))
        return original_open(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,"open",observed_open)
    assert current_binding() is None
    assert sha256_file(bound["path"])==bound["sha256"]
    assert sha256_file(bound["path"])==bound["sha256"]
    assert read_bound_json(bound["path"],bound["sha256"])=={}
    assert read_bound_json(bound["path"],bound["sha256"])=={}
    assert len(reads)==4



def test_native_input_overlap_cannot_shadow_code_own_or_prefix_hash(tmp_path,monkeypatch):
    from dataclasses import replace
    from darsia.fluidflower.native.io import sha256_file
    from darsia.fluidflower.pipeline.run_io import _key
    _task,_config,binding,source,prepared=candidate_prefix_fixture(tmp_path)
    relative="src/darsia/fluidflower/native/source.py"
    code=put(binding.repository/relative,b"# exact current source\n")
    def original_git(args, **kwargs):
        if args[3] == "show":
            assert args[4] == binding.commit + ":" + relative
            return Path(code["path"]).read_bytes()
        assert args[3] == "ls-tree" and args[-1] == relative
        return args[-1]+"\n"
    monkeypatch.setattr("darsia.fluidflower.pipeline.run_io.subprocess.check_output", original_git)
    binding._admit_code(relative,code["sha256"],"processor_module_sha256")
    own=put(binding.output/"own.npy",b"own output bytes")
    input_row=next(row for row in binding.verified.rows if row.path.name=="log.npy")
    # Simulate a future duplicated locator: these paths must never win a
    # verified-input cache lookup over their original code/output authority.
    for bound in (code,own,prepared):
        binding.locations[_key(bound["path"])]=input_row
    monkeypatch.setattr(binding,"_input_digest",lambda row:pytest.fail("Code/output was served from input digest cache"))
    with binding.activate():
        assert sha256_file(code["path"])==code["sha256"]
        assert sha256_file(own["path"])==own["sha256"]
        with binding.prefix_validation(source):
            assert sha256_file(prepared["path"])==prepared["sha256"]
        # An old logical alias pointing into an owned output is refused;
        # it must not cause a raw read of that historical logical path.
        logical=binding.repository/"historical-alias.npy"
        binding.locations[_key(logical)]=replace(input_row,path=Path(own["path"]))
        with pytest.raises(InputHashMismatch,match="overlaps"):
            sha256_file(logical)



def test_native_relative_declared_alias_replays_exact_input_without_cwd_fallback(tmp_path, monkeypatch):
    from darsia.fluidflower.native.io import resolve_asset_path, sha256_file
    logical = r"outputs\historical-fit\ac52_baseline.json"
    _task, config, _bound, host, binding = fixture(tmp_path, declared_alias=logical)
    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    original_open = Path.open
    def no_relative_probe(path, *args, **kwargs):
        assert str(path) != logical, "Relative historical spelling was opened from cwd"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_relative_probe)
    with binding.activate():
        actual = resolve_asset_path(logical, {})
        assert actual == host / "calibration.json"
        assert sha256_file(actual) == config["calibration"]["sha256"]
        with pytest.raises(InputHashMismatch, match="outside"):
            resolve_asset_path(r"other\ac52_baseline.json", {})


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("size_bytes", 999)])
def test_declared_alias_identity_mismatch_is_refused(tmp_path, field, value):
    from darsia.fluidflower.pipeline.run_io import _Replay
    _task, config, _bound, _host, binding = fixture(tmp_path)
    physical = dict(config["calibration"], **{field: value})
    logical = dict(physical, path=r"outputs\exact-but-wrong-identity.json")
    replay = _Replay(binding, [dict(logical=logical, physical=physical)])
    with pytest.raises(InputHashMismatch, match="verified consumed input"):
        replay.bind_declared_aliases()
    with pytest.raises(InputHashMismatch, match="outside"):
        binding.asset_path(logical["path"])


def test_unconsumed_declared_alias_does_not_add_an_input(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline.run_io import _Replay
    _task, _config, _bound, _host, binding = fixture(tmp_path)
    physical = dict(path=str(tmp_path / "not-consumed.npy"), sha256="0" * 64, size_bytes=123)
    logical = dict(physical, path=r"outputs\not-consumed.npy")
    replay = _Replay(binding, [dict(logical=logical, physical=physical)])
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("Unconsumed alias was opened"))
    replay.bind_declared_aliases()
    with pytest.raises(InputHashMismatch, match="outside"):
        binding.asset_path(logical["path"])


def test_declared_alias_cannot_replace_an_existing_verified_locator(tmp_path):
    from darsia.fluidflower.pipeline.run_io import _Replay
    _task, config, _bound, _host, binding = fixture(tmp_path)
    physical = config["physical_correction"]
    logical = dict(physical, path=config["calibration"]["path"])
    replay = _Replay(binding, [dict(logical=logical, physical=physical)])
    with pytest.raises(InputHashMismatch, match="conflicting verified input"):
        replay.bind_declared_aliases()


def test_declared_alias_cannot_shadow_owned_output(tmp_path):
    from darsia.fluidflower.pipeline.run_io import _Replay
    _task, config, _bound, _host, binding = fixture(tmp_path)
    physical = config["calibration"]
    logical = dict(physical, path=str(binding.output / "own.json"))
    replay = _Replay(binding, [dict(logical=logical, physical=physical)])
    with pytest.raises(InputHashMismatch, match="overlaps admitted code or owned output"):
        replay.bind_declared_aliases()



def provenance_source_fixture(tmp_path, monkeypatch, *, changed_display=False):
    """Authored code bytes stand in for exactly two tracked Git blobs."""
    _task, _config, _bound, _host, binding = fixture(tmp_path)
    source_relative = "src/darsia/fluidflower/native/source.py"
    display_relative = "src/darsia/fluidflower/native/display_modes.py"
    source = put(binding.repository / source_relative, b"# exact native processor source\n")
    display_raw = b"# exact named display provenance source\n"
    display = put(binding.repository / display_relative,
                  b"# modified disk source\n" if changed_display else display_raw)
    blobs = {source_relative: Path(source["path"]).read_bytes(), display_relative: display_raw}
    calls = []
    def original_git(args, **kwargs):
        calls.append(list(args))
        if "ls-tree" in args:
            assert args[-1] in blobs
            return args[-1] + "\n"
        assert args[3] == "show"
        commit, separator, relative = args[4].partition(":")
        assert separator and commit == binding.commit and relative in blobs
        return blobs[relative]
    monkeypatch.setattr("darsia.fluidflower.pipeline.run_io.subprocess.check_output", original_git)
    native = dict(source=dict(required_execution_identity_keys=["processor_module_sha256"],
                              execution_identity=dict(processor_module_sha256=source["sha256"])))
    return binding, native, display, calls


def test_named_native_provenance_reader_is_admitted_from_exact_job_commit(tmp_path, monkeypatch):
    from darsia.fluidflower.native.io import sha256_file
    binding, native, display, calls = provenance_source_fixture(tmp_path, monkeypatch)
    binding._admit_native_code(native)
    assert binding.asset_path(display["path"]) == Path(display["path"])
    assert any(args[3] == "show" and args[4].startswith(binding.commit + ":") for args in calls)
    monkeypatch.setattr(binding, "_input_digest", lambda row: pytest.fail("Code used the data digest cache"))
    original_open = Path.open
    reads = []
    def observe(path, mode="r", *args, **kwargs):
        if Path(path) == Path(display["path"]) and mode == "rb":
            reads.append(str(path))
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", observe)
    with binding.activate():
        assert sha256_file(display["path"]) == display["sha256"]
        assert sha256_file(display["path"]) == display["sha256"]
    assert reads == [display["path"], display["path"]]
    stranger = put(binding.repository / "tracked-but-not-an-admitted-reader.py", b"# unrelated code\n")
    with pytest.raises(InputHashMismatch, match="outside"):
        binding.asset_path(stranger["path"])


def test_named_native_provenance_reader_rejects_disk_bytes_not_in_job_commit(tmp_path, monkeypatch):
    binding, native, _display, _calls = provenance_source_fixture(tmp_path, monkeypatch, changed_display=True)
    with pytest.raises(InputHashMismatch, match="exact execution binding"):
        binding._admit_native_code(native)



def test_original_display_contract_self_hash_runs_under_exact_admitted_binding(tmp_path, monkeypatch):
    import importlib.util
    from darsia.fluidflower.native import display_modes as original_display
    _task, _config, _bound, _host, binding = fixture(tmp_path)
    source_relative = "src/darsia/fluidflower/native/source.py"
    display_relative = "src/darsia/fluidflower/native/display_modes.py"
    processor = put(binding.repository / source_relative, b"# authored bound processor\n")
    # Exact original module bytes, not a replacement renderer or inference.
    display_raw = Path(original_display.__file__).read_bytes()
    display = put(binding.repository / display_relative, display_raw)
    def original_git(args, **kwargs):
        if "ls-tree" in args:
            assert args[-1] in (source_relative, display_relative)
            return args[-1] + "\n"
        assert args[3] == "show"
        assert args[4] in (binding.commit + ":" + source_relative, binding.commit + ":" + display_relative)
        return Path(processor["path"]).read_bytes() if args[4].endswith(":" + source_relative) else display_raw
    monkeypatch.setattr("darsia.fluidflower.pipeline.run_io.subprocess.check_output", original_git)
    native = dict(source=dict(required_execution_identity_keys=["processor_module_sha256"],
                              execution_identity=dict(processor_module_sha256=processor["sha256"])))
    binding._admit_native_code(native)
    name = "darsia.fluidflower.native._isolated_original_display_contract"
    spec = importlib.util.spec_from_file_location(name, display["path"])
    module = importlib.util.module_from_spec(spec)
    import sys
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    with binding.activate():
        result = module.display_contract(parameters=dict(contract_metadata=dict(
            contract="authored provenance fixture", display_only_never_features=True)))
    assert result == dict(contract="authored provenance fixture",
                         display_only_never_features=True, renderer_code_sha256=display["sha256"])
    assert Path(display["path"]).read_bytes() == display_raw


def test_input_alias_cannot_shadow_the_named_display_code_role(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline.run_io import _Replay
    binding, native, display, _calls = provenance_source_fixture(tmp_path, monkeypatch)
    binding._admit_native_code(native)
    row = binding.verified.for_use("mass.calibration")
    physical = dict(path=str(row.path), sha256=row.reference["sha256"], size_bytes=row.reference["bytes"])
    logical = dict(physical, path=display["path"])
    replay = _Replay(binding, [dict(logical=logical, physical=physical)])
    with pytest.raises(InputHashMismatch, match="overlaps admitted code or owned output"):
        replay.bind_declared_aliases()
    assert binding.asset_path(display["path"]) == Path(display["path"])
