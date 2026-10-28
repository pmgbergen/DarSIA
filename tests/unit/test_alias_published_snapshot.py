"""Local immutable metadata fixtures; no worker, pixels, NAS or process."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import published_jobs_snapshot as snapshots
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from darsia.fluidflower.pipeline import job_queue as jobs
from darsia.fluidflower.pipeline import publication_runtime
from .test_alias_queue_runtime import fixture as queue_fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + "\r\n").encode()
    path.write_bytes(raw)
    return jobs.descriptor(path)


def reader(path, expected=None):
    actual = jobs.descriptor(path)
    if expected is not None:
        assert actual == expected
    value = jobs.read_json(path)
    assert jobs.descriptor(path) == actual
    return value


def frame(claim, case):
    settings = dict(host="RecordedHardwareHost", backend="cpu", precision="float32", threads=1)
    manifest = write(claim.output / "manifest.json", dict(run="ac52", configuration=claim.task["configuration"],
        code={"commit":claim.task["commit"]}, settings=settings))
    final = claim.output / "frames" / case
    scalar = write(final / "result.json", dict(complete=True, run="ac52", case_id=case, precision="float32", mass=0.125))
    maps = write(final / "products.npz", b"synthetic undecoded product bytes")
    files = {Path(row["path"]).name: {k:v for k,v in row.items() if k!="path"} for row in (scalar,maps)}
    commit_id = ("1" if case.endswith("1") else "2") * 32
    commit = write(final / "commit.json", dict(complete=True, frame_id=case, commit_id=commit_id,
        manifest_sha256=manifest["sha256"], settings=settings,
        products=dict(precision="float32",backend="cpu"), files=files))
    record = dict(frame_id=case, nas_directory=str(final), manifest_sha256=manifest["sha256"],
        files=dict(files,**{"commit.json":{k:v for k,v in commit.items() if k!="path"}}))
    registration = write(claim.output / "ops" / ("frame_commit_"+commit_id+".json"),
        dict(manifest_sha256=manifest["sha256"],frame_id=case,commit=commit,files=files))
    receipt = write(claim.output / "ops" / ("nas_copy_"+commit_id+".json"), record)
    producer = dict(case_id=case,attempt_directory=str(claim.output),manifest=manifest,
        commit=commit,frame_registration=registration,publication=receipt,result=scalar)
    event = dict(state="frame_completed", complete=True, run="ac52", case_id=case,
        frame_index=1 if case.endswith("1") else 2, compute_seconds=12.0, publication=record)
    return record,producer,event


def published_fixture(tmp_path, monkeypatch, *, prefix=False, failure=False):
    queue,clock,callbacks,_events=queue_fixture(tmp_path,monkeypatch)
    job=jobs.read_json(queue.root/"job.json")
    job.update(format_version=1,legacy_paths=False,job_id="synthetic-job",commit="a"*40,
        calibration_sha256="b"*64,precision="float32")
    write(queue.root/"job.json",job)
    if prefix or failure:
        for path in (queue.root/"queue/ac52.json",queue.root/"results/ac52/task.json"):
            task=jobs.read_json(path);task["ordered_case_ids"]=["ac52_f1","ac52_f2"];write(path,task)
    owners={};proofs={};records=[];producers=[];claims=[]
    def origin(attempt):
        return deepcopy(proofs[str(Path(attempt))])
    def make_callbacks():
        factory=publication_runtime.make_publication_factory(queue.root,job,"Main",producer_origin_for_attempt=origin)
        return dict(callbacks,publication_factory=factory)
    actual=dict(machine="main-cpu",pid=123,process_create_time="synthetic-epoch")
    first=queue.claim("Main-cpu","cpu",actual,**make_callbacks());claims.append(first)
    clock.stamp(first.lease/"owner.json")
    record,producer,event=frame(first,"ac52_f1");records.append(record);producers.append(producer)
    queue.progress(first,event);clock.stamp(first.lease/"progress_00000000.json")
    current=first
    if prefix:
        clock.now += 600*1_000_000_000
        queue.reclaim_stale()
        proof=write(first.output/"retirement.json",dict(complete=True,all_absent=True,
            worker_os_references_closed=True,run="ac52",attempt_id=first.token,
            attempt_directory=str(first.output),machine_id="Main"))
        proofs[str(first.output)]=dict(machine_id="Main",proof_descriptor=proof,accessible_root=str(first.output))
        current=queue.claim("Main-cpu","cpu",actual,**make_callbacks());claims.append(current)
        clock.stamp(current.lease/"owner.json")
        record,producer,event=frame(current,"ac52_f2");records.append(record);producers.append(producer)
        queue.progress(current,event)
    result=dict(complete=True,worker_references_closed=True,run="ac52",attempt_id=current.token,
        task=current.task,actual_worker=actual,commit=job["commit"],calibration_sha256=job["calibration_sha256"],
        precision="float32",backend="cpu",frames=len(records),case_count=len(records),
        ordered_case_ids=[record["frame_id"] for record in records],records=records,frame_producers=producers,
        manifest=jobs.descriptor(current.output/"manifest.json"),
        table=write(current.output/"mass_table.csv",b"case,mass\r\nac52_f1,0.125\r\n"))
    if failure:
        queue.fail(current,"Traceback (most recent call last):\nValueError: exact original failure\n",actual)
    else:
        queue.complete(current,result)
    return SimpleNamespace(queue=queue,root=queue.root,job=job,claims=claims,result=result,current=current)


def snapshot(data):
    return JobSnapshotReader()({"root":str(data.root)})


def assert_metadata_error(data):
    value=snapshot(data)
    assert value["state"]=="Metadata error"
    assert value["runs"][0]["committed"]==0
    assert value["complete"] is False
    return value


def test_terminal_original_producers_are_read_only_from_shared_copies(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch)
    original_open=Path.open
    producer=data.current.output
    def forbid_original(path,*args,**kwargs):
        assert not path.is_relative_to(producer),"The UI must never open a producer-local path"
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",forbid_original)
    value=snapshot(data)
    assert value["state"]=="Complete" and value["committed"]==1 and value["total_frames"]==1
    assert value["remaining"]==0 and value["eta_seconds"]==0
    assert value["machines"][0]["machine"]=="Main"
    assert value["machines"][0]["frames_per_minute"] is None


def test_terminal_prefix_uses_genuine_original_owner_and_retirement_bytes(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch,prefix=True)
    value=snapshot(data)
    assert value["state"]=="Complete" and value["committed"]==2
    assert value["runs"][0]["case_ids"]==["ac52_f1","ac52_f2"]
    assert value["machines"][0]["machine"]=="Main"


@pytest.mark.parametrize("field",["manifest","commit","frame_registration","publication","result"])
@pytest.mark.parametrize("changed",["sha256","path"])
def test_forged_original_producer_field_is_not_accepted_by_matching_records(tmp_path,monkeypatch,field,changed):
    data=published_fixture(tmp_path,monkeypatch)
    # Test only the original-producer admission helper: the immutable completion
    # SHA is independently verified by _terminal before this routine is called.
    done=jobs.read_json(data.root/"done/ac52.json")
    index_path=jobs._own_asset(data.root,done["products_index"])
    index=jobs.descriptor(index_path)
    _root,_document,products=snapshots._read_index(index)
    producer=deepcopy(data.result["frame_producers"][0])
    producer[field]=dict(producer[field],**{changed: "0"*64 if changed=="sha256" else "C:/foreign/wrong-attempt/"+field+".json"})
    with pytest.raises(ValueError,match="original|Original|Indexed"):
        snapshots._producer_inventory(index,products,data.result["records"][0],producer,
            data.current.task,data.job,current_attempt=snapshots._pure(data.current.output),
            current_token=data.current.token,owners={data.current.token:"Main"},
            metadata=snapshots._SnapshotMetadata(index,_root,products))


def test_foreign_attempt_path_is_refused_before_any_foreign_local_read(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch)
    done=jobs.read_json(data.root/"done/ac52.json")
    index_path=jobs._own_asset(data.root,done["products_index"])
    index=jobs.descriptor(index_path);_root,_document,products=snapshots._read_index(index)
    producer=deepcopy(data.result["frame_producers"][0])
    producer["attempt_directory"]="C:/foreign/not-owned/synthetic-job/ac52/.attempts/"+"f"*32
    original_open=Path.open
    def forbidden(path,*args,**kwargs):
        assert "foreign" not in str(path)
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",forbidden)
    with pytest.raises(ValueError,match="genuine attempt lineage"):
        snapshots._producer_inventory(index,products,data.result["records"][0],producer,
            data.current.task,data.job,current_attempt=snapshots._pure(data.current.output),
            current_token=data.current.token,owners={data.current.token:"Main"},
            metadata=snapshots._SnapshotMetadata(index,_root,products))


def test_moved_shared_root_and_missing_original_outputs_still_show_exact_completion(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch,prefix=True)
    moved=tmp_path/"other-shared-mount"
    data.root.rename(moved);data.root=moved
    for claim in data.claims:
        claim.output.rename(claim.output.with_name(claim.output.name+"-producer-unavailable"))
    value=snapshot(data)
    assert value["complete"] is True and value["committed"]==2


def test_predecessor_with_changed_science_task_is_not_adopted_as_lineage(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch,prefix=True)
    old=data.claims[0]
    path=data.root/"reclaimed/ac52"/old.token/"task.json"
    task=jobs.read_json(path);task["calibration_sha256"]="f"*64;write(path,task)
    value=assert_metadata_error(data)
    assert "Archived predecessor task differs" in value["runs"][0]["error"]


def test_genuine_failed_publication_keeps_committed_progress_and_literal_error(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch,failure=True)
    value=snapshot(data)
    row=value["runs"][0]
    assert row["state"]=="Failed" and row["committed"]==1 and row["total"]==2
    assert row["case_ids"]==["ac52_f1"] and value["remaining"]==1
    assert row["error"]=="Traceback (most recent call last):\nValueError: exact original failure\n"
    assert value["machines"][0]["machine"]=="Main"
    assert value["eta_seconds"] is None and value["complete"] is False


def test_failure_with_different_canonical_bytes_or_owner_is_metadata_error(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch,failure=True)
    path=data.root/"failed/ac52.json"
    failure=jobs.read_json(path);failure["traceback"]="forged canonical failure";write(path,failure)
    value=assert_metadata_error(data)
    assert "original published failure bytes" in value["runs"][0]["error"]


def test_failed_bundle_after_root_move_never_opens_original_local_attempt(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch,failure=True)
    moved=tmp_path/"other-shared-root";data.root.rename(moved);data.root=moved
    data.current.output.rename(data.current.output.with_name(data.current.token+"-absent"))
    value=snapshot(data)
    assert value["state"]=="Failed" and value["committed"]==1


def test_missing_shared_registration_never_falls_back_to_original_local_file(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch)
    done=jobs.read_json(data.root/"done/ac52.json")
    index_path=jobs._own_asset(data.root,done["products_index"])
    index=jobs.read_json(index_path)
    source=data.result["frame_producers"][0]["frame_registration"]["path"]
    row=next(row for row in index["products"] if row["local_path"]==source)
    copied=index_path.parent/row["metadata_copy"]["relative_path"]
    copied.unlink()  # Synthetic shared copy is absent; original still exists.
    assert Path(source).exists()
    assert_metadata_error(data)


def test_failure_publication_cannot_relabel_its_actual_worker_owner(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch,failure=True)
    # The published owner stays in its in-place lease; publishing/ac52 is its marker file.
    assert (data.root/"publishing/ac52").is_file()
    assert jobs.publication_directory(data.root,"ac52")==data.current.lease
    path=data.current.lease/"owner.json"
    value=jobs.read_json(path);value["actual_owner"]["pid"]=999;write(path,value)
    result=assert_metadata_error(data)
    assert "Original failure task/run/attempt/worker" in result["runs"][0]["error"]


def test_actual_snapshot_reader_open_does_not_write_job_or_control_metadata(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch)
    before={str(path.relative_to(data.root)):path.read_bytes() for path in data.root.rglob("*") if path.is_file()}
    result=snapshot(data)
    after={str(path.relative_to(data.root)):path.read_bytes() for path in data.root.rglob("*") if path.is_file()}
    assert result["complete"] is True and before==after
    assert not (data.root/"ui/commands").exists()


def test_csv_table_is_verified_as_raw_bytes_and_tamper_never_reads_producer(tmp_path,monkeypatch):
    data=published_fixture(tmp_path,monkeypatch)
    done=jobs.read_json(data.root/"done/ac52.json")
    index_path=jobs._own_asset(data.root,done["products_index"])
    index=jobs.read_json(index_path)
    table=next(row for row in index["products"] if row["product_id"]=="mass_table.csv")
    shared=index_path.parent/table["metadata_copy"]["relative_path"]
    original=shared.read_bytes()
    assert original.startswith(b"case,mass\r\n")
    producer=data.current.output
    original_open=Path.open
    def no_original(path,*args,**kwargs):
        assert not path.is_relative_to(producer),"CSV identity must come only from its shared byte copy"
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",no_original)
    assert snapshot(data)["state"]=="Complete"
    shared.write_bytes(original.replace(b"0.125",b"9.999"))
    value=assert_metadata_error(data)
    assert "Metadata" in value["runs"][0]["error"]
